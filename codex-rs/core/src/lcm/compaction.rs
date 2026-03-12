use super::estimate_tokens;
use super::extract_file_ids_from_content;
use super::format_timestamp;
use super::generate_summary_id;
use super::store::LcmStore;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_state::LcmContextItemRecord;
use codex_state::LcmContextItemType;
use codex_state::LcmCreateMessageParams;
use codex_state::LcmCreateSummaryParams;
use codex_state::LcmMessageRole;
use codex_state::LcmReplaceContextRangeParams;
use codex_state::LcmSummaryKind;
use codex_state::LcmSummaryRecord;
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;

const FALLBACK_MAX_CHARS: usize = 512 * 4;
const DEFAULT_LEAF_CHUNK_TOKENS: i64 = 20_000;
const CONDENSED_MIN_INPUT_RATIO: f64 = 0.1;

pub(crate) type LcmSummarizeFuture = Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send>>;
pub(crate) type LcmSummarizeFn =
    dyn Fn(String, bool, Option<LcmSummarizeOptions>) -> LcmSummarizeFuture + Send + Sync;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LcmSummarizeOptions {
    pub previous_summary: Option<String>,
    pub is_condensed: bool,
    pub depth: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactionDecision {
    pub should_compact: bool,
    pub reason: CompactionReason,
    pub current_tokens: i64,
    pub threshold: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionReason {
    Threshold,
    Manual,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CompactionLevel {
    Normal,
    Aggressive,
    Fallback,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactionResult {
    pub action_taken: bool,
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub created_summary_id: Option<String>,
    pub condensed: bool,
    pub level: Option<CompactionLevel>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactionConfig {
    pub context_threshold: f64,
    pub fresh_tail_count: i64,
    pub leaf_min_fanout: i64,
    pub condensed_min_fanout: i64,
    pub condensed_min_fanout_hard: i64,
    pub incremental_max_depth: i64,
    pub leaf_chunk_tokens: Option<i64>,
    pub leaf_target_tokens: i64,
    pub condensed_target_tokens: i64,
    pub max_rounds: i64,
    pub timezone: Option<String>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            context_threshold: 0.75,
            fresh_tail_count: 8,
            leaf_min_fanout: 8,
            condensed_min_fanout: 4,
            condensed_min_fanout_hard: 2,
            incremental_max_depth: 0,
            leaf_chunk_tokens: None,
            leaf_target_tokens: 600,
            condensed_target_tokens: 900,
            max_rounds: 10,
            timezone: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LeafTriggerDecision {
    pub should_compact: bool,
    pub raw_tokens_outside_tail: i64,
    pub threshold: i64,
}

#[derive(Debug, Clone)]
struct PassResult {
    summary_id: String,
    level: CompactionLevel,
}

#[derive(Debug, Clone)]
struct LeafPassResult {
    summary_id: String,
    level: CompactionLevel,
    content: String,
}

#[derive(Debug, Clone)]
struct LeafChunkSelection {
    items: Vec<LcmContextItemRecord>,
    raw_tokens_outside_tail: i64,
    threshold: i64,
}

#[derive(Debug, Clone)]
struct CondensedChunkSelection {
    items: Vec<LcmContextItemRecord>,
    summary_tokens: i64,
}

#[derive(Debug, Clone)]
struct CondensedPhaseCandidate {
    target_depth: i64,
    chunk: CondensedChunkSelection,
}

pub(crate) struct CompactionEngine<'a> {
    store: &'a dyn LcmStore,
    config: CompactionConfig,
}

impl<'a> CompactionEngine<'a> {
    pub(crate) fn new(store: &'a dyn LcmStore, config: CompactionConfig) -> Self {
        Self { store, config }
    }

    pub(crate) async fn evaluate(
        &self,
        thread_id: &str,
        token_budget: i64,
        observed_token_count: Option<i64>,
    ) -> anyhow::Result<CompactionDecision> {
        let stored_tokens = self.store.get_context_token_count(thread_id).await?;
        let live_tokens = observed_token_count.unwrap_or(0).max(0);
        let current_tokens = stored_tokens.max(live_tokens);
        let threshold = self.threshold_for_budget(token_budget);

        if current_tokens > threshold {
            return Ok(CompactionDecision {
                should_compact: true,
                reason: CompactionReason::Threshold,
                current_tokens,
                threshold,
            });
        }

        Ok(CompactionDecision {
            should_compact: false,
            reason: CompactionReason::None,
            current_tokens,
            threshold,
        })
    }

    pub(crate) async fn evaluate_leaf_trigger(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<LeafTriggerDecision> {
        let raw_tokens_outside_tail = self.count_raw_tokens_outside_fresh_tail(thread_id).await?;
        let threshold = self.resolve_leaf_chunk_tokens();
        Ok(LeafTriggerDecision {
            should_compact: raw_tokens_outside_tail >= threshold,
            raw_tokens_outside_tail,
            threshold,
        })
    }

    pub(crate) async fn compact(
        &self,
        thread_id: &str,
        token_budget: i64,
        summarize: &LcmSummarizeFn,
        force: bool,
        hard_trigger: bool,
    ) -> anyhow::Result<CompactionResult> {
        self.compact_full_sweep(thread_id, token_budget, summarize, force, hard_trigger)
            .await
    }

    pub(crate) async fn compact_leaf(
        &self,
        thread_id: &str,
        token_budget: i64,
        summarize: &LcmSummarizeFn,
        force: bool,
        previous_summary_content: Option<String>,
    ) -> anyhow::Result<CompactionResult> {
        let tokens_before = self.store.get_context_token_count(thread_id).await?;
        let threshold = self.threshold_for_budget(token_budget);
        let leaf_trigger = self.evaluate_leaf_trigger(thread_id).await?;

        if !force && tokens_before <= threshold && !leaf_trigger.should_compact {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let leaf_chunk = self.select_oldest_leaf_chunk(thread_id, true).await?;
        if leaf_chunk.items.is_empty() {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let previous_summary_content = previous_summary_content.or(self
            .resolve_prior_leaf_summary_context(thread_id, &leaf_chunk.items)
            .await?);

        let leaf_result = self
            .leaf_pass(
                thread_id,
                &leaf_chunk.items,
                summarize,
                previous_summary_content,
            )
            .await?;
        let tokens_after_leaf = self.store.get_context_token_count(thread_id).await?;
        self.persist_compaction_events(
            thread_id,
            tokens_before,
            tokens_after_leaf,
            tokens_after_leaf,
            Some((&leaf_result.summary_id, leaf_result.level)),
            None,
        )
        .await?;

        let mut tokens_after = tokens_after_leaf;
        let mut condensed = false;
        let mut created_summary_id = Some(leaf_result.summary_id.clone());
        let mut level = Some(leaf_result.level);

        let max_depth = self.resolve_incremental_max_depth();
        let condensed_min_chunk_tokens = self.resolve_condensed_min_chunk_tokens();
        let mut target_depth = 0i64;
        loop {
            if let Some(limit) = max_depth
                && target_depth >= limit
            {
                break;
            }

            let fanout = self.resolve_fanout_for_depth(target_depth, false);
            let chunk = self
                .select_oldest_chunk_at_depth(thread_id, target_depth, None)
                .await?;
            if i64::try_from(chunk.items.len()).unwrap_or(i64::MAX) < fanout
                || chunk.summary_tokens < condensed_min_chunk_tokens
            {
                break;
            }

            let pass_tokens_before = self.store.get_context_token_count(thread_id).await?;
            let condense_result = self
                .condensed_pass(thread_id, &chunk.items, target_depth, summarize)
                .await?;
            let pass_tokens_after = self.store.get_context_token_count(thread_id).await?;
            self.persist_compaction_events(
                thread_id,
                pass_tokens_before,
                pass_tokens_before,
                pass_tokens_after,
                None,
                Some((&condense_result.summary_id, condense_result.level)),
            )
            .await?;

            tokens_after = pass_tokens_after;
            condensed = true;
            created_summary_id = Some(condense_result.summary_id);
            level = Some(condense_result.level);

            if pass_tokens_after >= pass_tokens_before {
                break;
            }
            target_depth += 1;
        }

        Ok(CompactionResult {
            action_taken: true,
            tokens_before,
            tokens_after,
            created_summary_id,
            condensed,
            level,
        })
    }

    pub(crate) async fn compact_full_sweep(
        &self,
        thread_id: &str,
        token_budget: i64,
        summarize: &LcmSummarizeFn,
        force: bool,
        hard_trigger: bool,
    ) -> anyhow::Result<CompactionResult> {
        let tokens_before = self.store.get_context_token_count(thread_id).await?;
        let threshold = self.threshold_for_budget(token_budget);
        let leaf_trigger = self.evaluate_leaf_trigger(thread_id).await?;

        if !force && tokens_before <= threshold && !leaf_trigger.should_compact {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let context_items = self.store.get_context_items(thread_id).await?;
        if context_items.is_empty() {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let protect_fresh_tail = false;
        let mut action_taken = false;
        let mut condensed = false;
        let mut created_summary_id = None;
        let mut level = None;
        let mut previous_summary_content = None;
        let mut previous_tokens = tokens_before;

        loop {
            let leaf_chunk = self
                .select_oldest_leaf_chunk(thread_id, protect_fresh_tail)
                .await?;
            if leaf_chunk.items.is_empty() {
                break;
            }

            let pass_tokens_before = self.store.get_context_token_count(thread_id).await?;
            let leaf_result = self
                .leaf_pass(
                    thread_id,
                    &leaf_chunk.items,
                    summarize,
                    previous_summary_content.clone(),
                )
                .await?;
            let pass_tokens_after = self.store.get_context_token_count(thread_id).await?;
            self.persist_compaction_events(
                thread_id,
                pass_tokens_before,
                pass_tokens_after,
                pass_tokens_after,
                Some((&leaf_result.summary_id, leaf_result.level)),
                None,
            )
            .await?;

            action_taken = true;
            created_summary_id = Some(leaf_result.summary_id.clone());
            level = Some(leaf_result.level);
            previous_summary_content = Some(leaf_result.content);

            if pass_tokens_after >= pass_tokens_before || pass_tokens_after >= previous_tokens {
                break;
            }
            previous_tokens = pass_tokens_after;
        }

        loop {
            let Some(candidate) = self
                .select_shallowest_condensation_candidate(
                    thread_id,
                    hard_trigger,
                    protect_fresh_tail,
                )
                .await?
            else {
                break;
            };

            let pass_tokens_before = self.store.get_context_token_count(thread_id).await?;
            let condense_result = self
                .condensed_pass(
                    thread_id,
                    &candidate.chunk.items,
                    candidate.target_depth,
                    summarize,
                )
                .await?;
            let pass_tokens_after = self.store.get_context_token_count(thread_id).await?;
            self.persist_compaction_events(
                thread_id,
                pass_tokens_before,
                pass_tokens_before,
                pass_tokens_after,
                None,
                Some((&condense_result.summary_id, condense_result.level)),
            )
            .await?;

            action_taken = true;
            condensed = true;
            created_summary_id = Some(condense_result.summary_id);
            level = Some(condense_result.level);

            if pass_tokens_after >= pass_tokens_before || pass_tokens_after >= previous_tokens {
                break;
            }
            previous_tokens = pass_tokens_after;
        }

        Ok(CompactionResult {
            action_taken,
            tokens_before,
            tokens_after: self.store.get_context_token_count(thread_id).await?,
            created_summary_id,
            condensed,
            level,
        })
    }

    pub(crate) async fn compact_until_under(
        &self,
        thread_id: &str,
        token_budget: i64,
        target_tokens: Option<i64>,
        current_tokens: Option<i64>,
        summarize: &LcmSummarizeFn,
    ) -> anyhow::Result<(bool, i64, i64)> {
        let target_tokens = target_tokens.unwrap_or(token_budget).max(1);
        let stored_tokens = self.store.get_context_token_count(thread_id).await?;
        let mut last_tokens = stored_tokens.max(current_tokens.unwrap_or(0));
        if last_tokens < target_tokens {
            return Ok((true, 0, last_tokens));
        }

        for round in 1..=self.config.max_rounds.max(1) {
            let result = self
                .compact(thread_id, token_budget, summarize, true, false)
                .await?;

            if result.tokens_after <= target_tokens {
                return Ok((true, round, result.tokens_after));
            }
            if !result.action_taken || result.tokens_after >= last_tokens {
                return Ok((false, round, result.tokens_after));
            }
            last_tokens = result.tokens_after;
        }

        let final_tokens = self.store.get_context_token_count(thread_id).await?;
        Ok((
            final_tokens <= target_tokens,
            self.config.max_rounds,
            final_tokens,
        ))
    }

    fn threshold_for_budget(&self, token_budget: i64) -> i64 {
        ((self.config.context_threshold * token_budget as f64).floor() as i64).max(0)
    }

    fn resolve_leaf_chunk_tokens(&self) -> i64 {
        self.config
            .leaf_chunk_tokens
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_LEAF_CHUNK_TOKENS)
    }

    fn resolve_fresh_tail_count(&self) -> i64 {
        self.config.fresh_tail_count.max(0)
    }

    fn resolve_fresh_tail_ordinal(&self, context_items: &[LcmContextItemRecord]) -> i64 {
        let fresh_tail_count = self.resolve_fresh_tail_count();
        if fresh_tail_count == 0 {
            return i64::MAX;
        }

        let raw_items = context_items
            .iter()
            .filter(|item| {
                matches!(item.item_type, LcmContextItemType::Message) && item.message_id.is_some()
            })
            .collect::<Vec<_>>();
        if raw_items.is_empty() {
            return i64::MAX;
        }
        let tail_start_idx = raw_items
            .len()
            .saturating_sub(usize::try_from(fresh_tail_count).unwrap_or(usize::MAX));
        raw_items
            .get(tail_start_idx)
            .map(|item| item.ordinal)
            .unwrap_or(i64::MAX)
    }

    async fn get_message_token_count(&self, message_id: i64) -> anyhow::Result<i64> {
        let Some(message) = self.store.get_message_by_id(message_id).await? else {
            return Ok(0);
        };
        if message.token_count > 0 {
            return Ok(message.token_count);
        }
        Ok(estimate_tokens(&message.content))
    }

    async fn count_raw_tokens_outside_fresh_tail(&self, thread_id: &str) -> anyhow::Result<i64> {
        let context_items = self.store.get_context_items(thread_id).await?;
        let fresh_tail_ordinal = self.resolve_fresh_tail_ordinal(&context_items);
        let mut raw_tokens = 0i64;

        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }
            if !matches!(item.item_type, LcmContextItemType::Message) {
                continue;
            }
            let Some(message_id) = item.message_id else {
                continue;
            };
            raw_tokens = raw_tokens.saturating_add(self.get_message_token_count(message_id).await?);
        }

        Ok(raw_tokens)
    }

    async fn select_oldest_leaf_chunk(
        &self,
        thread_id: &str,
        protect_fresh_tail: bool,
    ) -> anyhow::Result<LeafChunkSelection> {
        let context_items = self.store.get_context_items(thread_id).await?;
        let fresh_tail_ordinal = if protect_fresh_tail {
            self.resolve_fresh_tail_ordinal(&context_items)
        } else {
            i64::MAX
        };
        let threshold = self.resolve_leaf_chunk_tokens();

        let mut raw_tokens_outside_tail = 0i64;
        for item in &context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }
            if !matches!(item.item_type, LcmContextItemType::Message) {
                continue;
            }
            let Some(message_id) = item.message_id else {
                continue;
            };
            raw_tokens_outside_tail = raw_tokens_outside_tail
                .saturating_add(self.get_message_token_count(message_id).await?);
        }

        let mut chunk = Vec::new();
        let mut chunk_tokens = 0i64;
        let mut started = false;
        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }
            let is_message =
                matches!(item.item_type, LcmContextItemType::Message) && item.message_id.is_some();
            if !started {
                if !is_message {
                    continue;
                }
                started = true;
            } else if !is_message {
                break;
            }

            let message_id = item.message_id.expect("checked above");
            let message_tokens = self.get_message_token_count(message_id).await?;
            if !chunk.is_empty() && chunk_tokens.saturating_add(message_tokens) > threshold {
                break;
            }
            chunk.push(item);
            chunk_tokens = chunk_tokens.saturating_add(message_tokens);
            if chunk_tokens >= threshold {
                break;
            }
        }

        Ok(LeafChunkSelection {
            items: chunk,
            raw_tokens_outside_tail,
            threshold,
        })
    }

    async fn resolve_prior_leaf_summary_context(
        &self,
        thread_id: &str,
        message_items: &[LcmContextItemRecord],
    ) -> anyhow::Result<Option<String>> {
        if message_items.is_empty() {
            return Ok(None);
        }
        let start_ordinal = message_items
            .iter()
            .map(|item| item.ordinal)
            .min()
            .unwrap_or(0);
        let context_items = self.store.get_context_items(thread_id).await?;
        let prior_summary_items = context_items
            .into_iter()
            .filter(|item| {
                item.ordinal < start_ordinal
                    && matches!(item.item_type, LcmContextItemType::Summary)
                    && item.summary_id.is_some()
            })
            .collect::<Vec<_>>();

        let mut summary_contents = Vec::new();
        for item in prior_summary_items.iter().rev().take(2).rev() {
            let Some(summary_id) = item.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = self.store.get_summary(summary_id).await? else {
                continue;
            };
            let content = summary.content.trim();
            if !content.is_empty() {
                summary_contents.push(content.to_string());
            }
        }

        if summary_contents.is_empty() {
            Ok(None)
        } else {
            Ok(Some(summary_contents.join("\n\n")))
        }
    }

    fn resolve_summary_token_count(&self, summary: &LcmSummaryRecord) -> i64 {
        if summary.token_count > 0 {
            summary.token_count
        } else {
            estimate_tokens(&summary.content)
        }
    }

    fn resolve_leaf_min_fanout(&self) -> i64 {
        self.config.leaf_min_fanout.max(1)
    }

    fn resolve_condensed_min_fanout(&self) -> i64 {
        self.config.condensed_min_fanout.max(1)
    }

    fn resolve_condensed_min_fanout_hard(&self) -> i64 {
        self.config.condensed_min_fanout_hard.max(1)
    }

    fn resolve_incremental_max_depth(&self) -> Option<i64> {
        if self.config.incremental_max_depth < 0 {
            None
        } else {
            Some(self.config.incremental_max_depth)
        }
    }

    fn resolve_fanout_for_depth(&self, target_depth: i64, hard_trigger: bool) -> i64 {
        if hard_trigger {
            return self.resolve_condensed_min_fanout_hard();
        }
        if target_depth == 0 {
            return self.resolve_leaf_min_fanout();
        }
        self.resolve_condensed_min_fanout()
    }

    fn resolve_condensed_min_chunk_tokens(&self) -> i64 {
        let ratio_floor =
            (self.resolve_leaf_chunk_tokens() as f64 * CONDENSED_MIN_INPUT_RATIO).floor() as i64;
        self.config.condensed_target_tokens.max(ratio_floor)
    }

    async fn select_shallowest_condensation_candidate(
        &self,
        thread_id: &str,
        hard_trigger: bool,
        protect_fresh_tail: bool,
    ) -> anyhow::Result<Option<CondensedPhaseCandidate>> {
        let context_items = self.store.get_context_items(thread_id).await?;
        let fresh_tail_ordinal = if protect_fresh_tail {
            self.resolve_fresh_tail_ordinal(&context_items)
        } else {
            i64::MAX
        };
        let min_chunk_tokens = self.resolve_condensed_min_chunk_tokens();
        let depth_levels = self
            .store
            .get_distinct_depths_in_context(thread_id, Some(fresh_tail_ordinal))
            .await?;

        for target_depth in depth_levels {
            let fanout = self.resolve_fanout_for_depth(target_depth, hard_trigger);
            let chunk = self
                .select_oldest_chunk_at_depth(thread_id, target_depth, Some(fresh_tail_ordinal))
                .await?;
            if i64::try_from(chunk.items.len()).unwrap_or(i64::MAX) < fanout {
                continue;
            }
            if chunk.summary_tokens < min_chunk_tokens {
                continue;
            }
            return Ok(Some(CondensedPhaseCandidate {
                target_depth,
                chunk,
            }));
        }

        Ok(None)
    }

    async fn select_oldest_chunk_at_depth(
        &self,
        thread_id: &str,
        target_depth: i64,
        fresh_tail_ordinal_override: Option<i64>,
    ) -> anyhow::Result<CondensedChunkSelection> {
        let context_items = self.store.get_context_items(thread_id).await?;
        let fresh_tail_ordinal = fresh_tail_ordinal_override
            .unwrap_or_else(|| self.resolve_fresh_tail_ordinal(&context_items));
        let chunk_token_budget = self.resolve_leaf_chunk_tokens();

        let mut chunk = Vec::new();
        let mut summary_tokens = 0i64;
        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }
            if !matches!(item.item_type, LcmContextItemType::Summary) || item.summary_id.is_none() {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }

            let summary_id = item.summary_id.clone().expect("checked above");
            let Some(summary) = self.store.get_summary(&summary_id).await? else {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            };
            if summary.depth != target_depth {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }
            let token_count = self.resolve_summary_token_count(&summary);
            if !chunk.is_empty() && summary_tokens.saturating_add(token_count) > chunk_token_budget
            {
                break;
            }
            chunk.push(item);
            summary_tokens = summary_tokens.saturating_add(token_count);
            if summary_tokens >= chunk_token_budget {
                break;
            }
        }

        Ok(CondensedChunkSelection {
            items: chunk,
            summary_tokens,
        })
    }

    async fn resolve_prior_summary_context_at_depth(
        &self,
        thread_id: &str,
        summary_items: &[LcmContextItemRecord],
        target_depth: i64,
    ) -> anyhow::Result<Option<String>> {
        if summary_items.is_empty() {
            return Ok(None);
        }
        let start_ordinal = summary_items
            .iter()
            .map(|item| item.ordinal)
            .min()
            .unwrap_or(0);
        let context_items = self.store.get_context_items(thread_id).await?;
        let prior_summary_items = context_items
            .into_iter()
            .filter(|item| {
                item.ordinal < start_ordinal
                    && matches!(item.item_type, LcmContextItemType::Summary)
                    && item.summary_id.is_some()
            })
            .collect::<Vec<_>>();

        let mut summary_contents = Vec::new();
        for item in prior_summary_items.iter().rev().take(4).rev() {
            let Some(summary_id) = item.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = self.store.get_summary(summary_id).await? else {
                continue;
            };
            if summary.depth != target_depth {
                continue;
            }
            let content = summary.content.trim();
            if !content.is_empty() {
                summary_contents.push(content.to_string());
            }
        }

        if summary_contents.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                summary_contents
                    .into_iter()
                    .rev()
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ))
        }
    }

    async fn summarize_with_escalation(
        &self,
        source_text: &str,
        summarize: &LcmSummarizeFn,
        options: Option<LcmSummarizeOptions>,
    ) -> anyhow::Result<(String, CompactionLevel)> {
        let source_text = source_text.trim();
        if source_text.is_empty() {
            return Ok((
                "[Truncated from 0 tokens]".to_string(),
                CompactionLevel::Fallback,
            ));
        }

        let input_tokens = estimate_tokens(source_text).max(1);
        let mut summary_text = summarize(source_text.to_string(), false, options.clone()).await?;
        let mut level = CompactionLevel::Normal;

        if estimate_tokens(&summary_text) >= input_tokens {
            summary_text = summarize(source_text.to_string(), true, options.clone()).await?;
            level = CompactionLevel::Aggressive;
            if estimate_tokens(&summary_text) >= input_tokens {
                let truncated = if source_text.len() > FALLBACK_MAX_CHARS {
                    source_text[..FALLBACK_MAX_CHARS].to_string()
                } else {
                    source_text.to_string()
                };
                summary_text = format!("{truncated}\n[Truncated from {input_tokens} tokens]");
                level = CompactionLevel::Fallback;
            }
        }

        Ok((summary_text, level))
    }

    async fn leaf_pass(
        &self,
        thread_id: &str,
        message_items: &[LcmContextItemRecord],
        summarize: &LcmSummarizeFn,
        previous_summary_content: Option<String>,
    ) -> anyhow::Result<LeafPassResult> {
        let mut messages = Vec::new();
        for item in message_items {
            let Some(message_id) = item.message_id else {
                continue;
            };
            let Some(message) = self.store.get_message_by_id(message_id).await? else {
                continue;
            };
            messages.push(message);
        }

        let concatenated = messages
            .iter()
            .map(|message| {
                format!(
                    "[{}]\n{}",
                    format_timestamp(message.created_at),
                    message.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let file_ids = dedupe_ordered_ids(
            messages
                .iter()
                .flat_map(|message| extract_file_ids_from_content(&message.content)),
        );
        let (summary_content, level) = self
            .summarize_with_escalation(
                &concatenated,
                summarize,
                Some(LcmSummarizeOptions {
                    previous_summary: previous_summary_content,
                    is_condensed: false,
                    depth: None,
                }),
            )
            .await?;

        let summary_id = generate_summary_id(&summary_content);
        let token_count = estimate_tokens(&summary_content);
        let earliest_at = messages.iter().map(|message| message.created_at).min();
        let latest_at = messages.iter().map(|message| message.created_at).max();
        let source_message_token_count = messages
            .iter()
            .map(|message| message.token_count.max(0))
            .sum::<i64>();
        self.store
            .insert_summary(LcmCreateSummaryParams {
                summary_id: summary_id.clone(),
                thread_id: thread_id.to_string(),
                kind: LcmSummaryKind::Leaf,
                depth: 0,
                content: summary_content.clone(),
                token_count,
                file_ids,
                earliest_at,
                latest_at,
                descendant_count: 0,
                descendant_token_count: 0,
                source_message_token_count,
            })
            .await?;
        let message_ids = messages
            .iter()
            .map(|message| message.message_id)
            .collect::<Vec<_>>();
        self.store
            .link_summary_to_messages(&summary_id, &message_ids)
            .await?;

        let start_ordinal = message_items
            .iter()
            .map(|item| item.ordinal)
            .min()
            .unwrap_or(0);
        let end_ordinal = message_items
            .iter()
            .map(|item| item.ordinal)
            .max()
            .unwrap_or(0);
        self.store
            .replace_context_range_with_summary(LcmReplaceContextRangeParams {
                thread_id: thread_id.to_string(),
                start_ordinal,
                end_ordinal,
                summary_id: summary_id.clone(),
            })
            .await?;

        Ok(LeafPassResult {
            summary_id,
            level,
            content: summary_content,
        })
    }

    async fn condensed_pass(
        &self,
        thread_id: &str,
        summary_items: &[LcmContextItemRecord],
        target_depth: i64,
        summarize: &LcmSummarizeFn,
    ) -> anyhow::Result<PassResult> {
        let mut summaries = Vec::new();
        for item in summary_items {
            let Some(summary_id) = item.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = self.store.get_summary(summary_id).await? else {
                continue;
            };
            summaries.push(summary);
        }

        let concatenated = summaries
            .iter()
            .map(|summary| {
                let earliest_at = summary.earliest_at.unwrap_or(summary.created_at);
                let latest_at = summary.latest_at.unwrap_or(summary.created_at);
                format!(
                    "[{} - {}]\n{}",
                    format_timestamp(earliest_at),
                    format_timestamp(latest_at),
                    summary.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let file_ids = dedupe_ordered_ids(summaries.iter().flat_map(|summary| {
            let mut ids = summary.file_ids.clone();
            ids.extend(extract_file_ids_from_content(&summary.content));
            ids
        }));
        let previous_summary_content = if target_depth == 0 {
            self.resolve_prior_summary_context_at_depth(thread_id, summary_items, target_depth)
                .await?
        } else {
            None
        };
        let (summary_content, level) = self
            .summarize_with_escalation(
                &concatenated,
                summarize,
                Some(LcmSummarizeOptions {
                    previous_summary: previous_summary_content,
                    is_condensed: true,
                    depth: Some(target_depth + 1),
                }),
            )
            .await?;

        let summary_id = generate_summary_id(&summary_content);
        let token_count = estimate_tokens(&summary_content);
        let earliest_at = summaries
            .iter()
            .map(|summary| summary.earliest_at.unwrap_or(summary.created_at))
            .min();
        let latest_at = summaries
            .iter()
            .map(|summary| summary.latest_at.unwrap_or(summary.created_at))
            .max();
        let descendant_count = summaries
            .iter()
            .map(|summary| summary.descendant_count.max(0).saturating_add(1))
            .sum::<i64>();
        let descendant_token_count = summaries
            .iter()
            .map(|summary| {
                summary
                    .token_count
                    .max(0)
                    .saturating_add(summary.descendant_token_count.max(0))
            })
            .sum::<i64>();
        let source_message_token_count = summaries
            .iter()
            .map(|summary| summary.source_message_token_count.max(0))
            .sum::<i64>();

        self.store
            .insert_summary(LcmCreateSummaryParams {
                summary_id: summary_id.clone(),
                thread_id: thread_id.to_string(),
                kind: LcmSummaryKind::Condensed,
                depth: target_depth + 1,
                content: summary_content,
                token_count,
                file_ids,
                earliest_at,
                latest_at,
                descendant_count,
                descendant_token_count,
                source_message_token_count,
            })
            .await?;

        let parent_summary_ids = summaries
            .iter()
            .map(|summary| summary.summary_id.clone())
            .collect::<Vec<_>>();
        self.store
            .link_summary_to_parents(&summary_id, &parent_summary_ids)
            .await?;

        let start_ordinal = summary_items
            .iter()
            .map(|item| item.ordinal)
            .min()
            .unwrap_or(0);
        let end_ordinal = summary_items
            .iter()
            .map(|item| item.ordinal)
            .max()
            .unwrap_or(0);
        self.store
            .replace_context_range_with_summary(LcmReplaceContextRangeParams {
                thread_id: thread_id.to_string(),
                start_ordinal,
                end_ordinal,
                summary_id: summary_id.clone(),
            })
            .await?;

        Ok(PassResult { summary_id, level })
    }

    async fn persist_compaction_events(
        &self,
        thread_id: &str,
        tokens_before: i64,
        tokens_after_leaf: i64,
        tokens_after_final: i64,
        leaf_result: Option<(&str, CompactionLevel)>,
        condense_result: Option<(&str, CompactionLevel)>,
    ) -> anyhow::Result<()> {
        if leaf_result.is_none() && condense_result.is_none() {
            return Ok(());
        }
        let mut created_summary_ids = Vec::new();
        if let Some((summary_id, _)) = leaf_result {
            created_summary_ids.push(summary_id.to_string());
        }
        if let Some((summary_id, _)) = condense_result {
            created_summary_ids.push(summary_id.to_string());
        }
        let condensed_pass_occurred = condense_result.is_some();

        if let Some((summary_id, level)) = leaf_result {
            self.persist_compaction_event(CompactionEventRecord {
                thread_id: thread_id.to_string(),
                pass: "leaf".to_string(),
                level,
                tokens_before,
                tokens_after: tokens_after_leaf,
                created_summary_id: summary_id.to_string(),
                created_summary_ids: created_summary_ids.clone(),
                condensed_pass_occurred,
            })
            .await?;
        }
        if let Some((summary_id, level)) = condense_result {
            self.persist_compaction_event(CompactionEventRecord {
                thread_id: thread_id.to_string(),
                pass: "condensed".to_string(),
                level,
                tokens_before: tokens_after_leaf,
                tokens_after: tokens_after_final,
                created_summary_id: summary_id.to_string(),
                created_summary_ids,
                condensed_pass_occurred,
            })
            .await?;
        }

        Ok(())
    }

    async fn persist_compaction_event(&self, event: CompactionEventRecord) -> anyhow::Result<()> {
        let content = serde_json::to_string(&event)?;
        let next_seq = self
            .store
            .get_max_seq(&event.thread_id)
            .await?
            .saturating_add(1);
        let raw_item = ResponseItem::Message {
            id: None,
            role: "system".to_string(),
            content: vec![ContentItem::InputText {
                text: content.clone(),
            }],
            end_turn: None,
            phase: None,
        };
        let _ = self
            .store
            .create_message(LcmCreateMessageParams {
                thread_id: event.thread_id,
                seq: next_seq,
                role: LcmMessageRole::System,
                content: content.clone(),
                raw_item,
                token_count: estimate_tokens(&content),
            })
            .await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
struct CompactionEventRecord {
    thread_id: String,
    pass: String,
    level: CompactionLevel,
    tokens_before: i64,
    tokens_after: i64,
    created_summary_id: String,
    created_summary_ids: Vec<String>,
    condensed_pass_occurred: bool,
}

fn dedupe_ordered_ids(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut ordered = Vec::new();
    for id in ids {
        if !ordered.contains(&id) {
            ordered.push(id);
        }
    }
    ordered
}

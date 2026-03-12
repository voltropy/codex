mod assembler;
mod compaction;
mod ingest;
mod retrieval;
mod store;
mod summarize;

pub(crate) use assembler::AssembleContextInput;
pub(crate) use assembler::ContextAssembler;
pub(crate) use compaction::CompactionConfig;
pub(crate) use compaction::CompactionEngine;
pub(crate) use compaction::CompactionResult;
pub(crate) use ingest::bootstrap_lcm_history;
pub(crate) use ingest::ingest_lcm_items;
pub(crate) use store::LcmStore;
pub(crate) use store::StateDbLcmStore;
pub(crate) use summarize::build_lcm_summarize_fn;
pub(crate) use retrieval::DescribeItem;
pub(crate) use retrieval::DescribeResult;
pub(crate) use retrieval::GrepResult;
pub(crate) use retrieval::ExpandInput;
pub(crate) use retrieval::ExpandResult;
pub(crate) use retrieval::GrepInput;
pub(crate) use retrieval::GrepMode;
pub(crate) use retrieval::GrepScope;
pub(crate) use retrieval::RetrievalEngine;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::protocol::ContextCompactedEvent;
use crate::protocol::EventMsg;
use chrono::DateTime;
use chrono::Utc;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;
use once_cell::sync::Lazy;
use regex_lite::Regex;
use sha2::Digest;
use sha2::Sha256;
use std::sync::Arc;

#[cfg(test)]
mod tests;

static FILE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bfile_[a-f0-9]{16}\b").expect("valid file id regex"));

pub(crate) fn estimate_tokens(text: &str) -> i64 {
    i64::try_from((text.len().saturating_add(3)) / 4).unwrap_or(i64::MAX)
}

pub(crate) fn response_item_text(item: &ResponseItem) -> String {
    match item {
        ResponseItem::Message { content, .. } => crate::compact::content_items_to_text(content)
            .unwrap_or_else(|| serde_json::to_string(item).unwrap_or_default()),
        _ => serde_json::to_string(item).unwrap_or_default(),
    }
}

pub(crate) fn extract_file_ids_from_content(content: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for candidate in FILE_ID_RE.find_iter(content).map(|value| value.as_str()) {
        let candidate = candidate.to_ascii_lowercase();
        if !ids.contains(&candidate) {
            ids.push(candidate);
        }
    }
    ids
}

pub(crate) fn format_timestamp(value: DateTime<Utc>) -> String {
    value.format("%Y-%m-%d %H:%M UTC").to_string()
}

pub(crate) fn generate_summary_id(content: &str) -> String {
    let timestamp_ms = Utc::now().timestamp_millis().to_string();
    let digest = Sha256::digest(format!("{content}{timestamp_ms}"));
    format!("sum_{:x}", digest)[..20].to_string()
}

pub(crate) async fn ingest_session_items(
    sess: &Session,
    items: &[ResponseItem],
) -> anyhow::Result<()> {
    let Some(state_db) = sess.state_db() else {
        return Ok(());
    };
    let thread_id = sess.conversation_id.to_string();
    let store = StateDbLcmStore::new(state_db);
    ingest_lcm_items(&store, &thread_id, items).await
}

async fn bootstrap_session_history(sess: &Session) -> anyhow::Result<Option<StateDbLcmStore>> {
    let Some(state_db) = sess.state_db() else {
        return Ok(None);
    };
    let store = StateDbLcmStore::new(state_db);
    let thread_id = sess.conversation_id.to_string();
    let history = sess.clone_history().await;
    bootstrap_lcm_history(&store, &thread_id, history.raw_items()).await?;
    Ok(Some(store))
}

pub(crate) async fn run_maintenance(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    force_full_sweep: bool,
) -> CodexResult<bool> {
    let Some(store) = bootstrap_session_history(sess).await.map_err(map_lcm_err)? else {
        return Ok(false);
    };
    let thread_id = sess.conversation_id.to_string();
    let config = CompactionConfig::default();
    let token_budget = turn_context
        .model_info
        .auto_compact_token_limit()
        .or(turn_context.model_context_window())
        .unwrap_or(i64::MAX);
    let existing_context_items = store
        .get_context_items(&thread_id)
        .await
        .map_err(map_lcm_err)?;
    let had_summaries = existing_context_items
        .iter()
        .any(|item| matches!(item.item_type, codex_state::LcmContextItemType::Summary));

    let summarize = build_lcm_summarize_fn(
        Arc::clone(sess),
        Arc::clone(turn_context),
        Some(config.condensed_target_tokens),
    );
    let engine = CompactionEngine::new(&store, config.clone());
    let observed_tokens = Some(sess.get_total_token_usage().await);
    let result = if force_full_sweep {
        engine
            .compact_full_sweep(&thread_id, token_budget, summarize.as_ref(), true, false)
            .await
            .map_err(map_lcm_err)?
    } else {
        let decision = engine
            .evaluate(&thread_id, token_budget, observed_tokens)
            .await
            .map_err(map_lcm_err)?;
        if decision.should_compact {
            engine
                .compact_full_sweep(&thread_id, token_budget, summarize.as_ref(), false, false)
                .await
                .map_err(map_lcm_err)?
        } else {
            let leaf_trigger = engine
                .evaluate_leaf_trigger(&thread_id)
                .await
                .map_err(map_lcm_err)?;
            if !leaf_trigger.should_compact {
                if !had_summaries {
                    return Ok(false);
                }
                CompactionResult {
                    action_taken: false,
                    tokens_before: decision.current_tokens,
                    tokens_after: decision.current_tokens,
                    created_summary_id: None,
                    condensed: false,
                    level: None,
                }
            } else {
                engine
                    .compact_leaf(&thread_id, token_budget, summarize.as_ref(), false, None)
                    .await
                    .map_err(map_lcm_err)?
            }
        }
    };

    if !result.action_taken && !had_summaries {
        return Ok(false);
    }

    let assembler = ContextAssembler::new(&store);
    let assembled = assembler
        .assemble(AssembleContextInput {
            thread_id,
            token_budget,
            fresh_tail_count: usize::try_from(config.fresh_tail_count.max(0)).unwrap_or(0),
        })
        .await
        .map_err(map_lcm_err)?;
    let history = sess.clone_history().await;
    let mut replacement_history = assembled.messages;
    replacement_history.extend(
        history
            .raw_items()
            .iter()
            .filter(|item| matches!(item, ResponseItem::GhostSnapshot { .. }))
            .cloned(),
    );
    sess.replace_history(replacement_history, None).await;
    sess.recompute_token_usage(turn_context).await;
    if result.action_taken {
        sess.send_event(
            turn_context,
            EventMsg::ContextCompacted(ContextCompactedEvent),
        )
        .await;
    }
    Ok(result.action_taken)
}

pub(crate) async fn emit_context_compaction_item(sess: &Session, turn_context: &TurnContext) {
    let item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(turn_context, &item).await;
    sess.emit_turn_item_completed(turn_context, item).await;
}

fn map_lcm_err(err: anyhow::Error) -> CodexErr {
    CodexErr::Fatal(format!("LCM failed: {err:#}"))
}

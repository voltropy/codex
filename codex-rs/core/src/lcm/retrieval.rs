use super::estimate_tokens;
use super::store::LcmStore;
use chrono::DateTime;
use chrono::Utc;
use codex_state::LcmMessageRecord;
use codex_state::LcmSummaryKind;
use regex_lite::Regex;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DescribeResult {
    pub id: String,
    pub item: DescribeItem,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DescribeItem {
    Summary(DescribeSummary),
    File(DescribeFile),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DescribeSummary {
    pub thread_id: String,
    pub kind: LcmSummaryKind,
    pub content: String,
    pub depth: i64,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub file_ids: Vec<String>,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,
    pub message_ids: Vec<i64>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub subtree: Vec<DescribeSummarySubtreeNode>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DescribeSummarySubtreeNode {
    pub summary_id: String,
    pub parent_summary_id: Option<String>,
    pub depth_from_root: i64,
    pub kind: LcmSummaryKind,
    pub depth: i64,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub child_count: i64,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DescribeFile {
    pub thread_id: String,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrepMode {
    Regex,
    FullText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrepScope {
    Messages,
    Summaries,
    Both,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrepInput {
    pub query: String,
    pub mode: GrepMode,
    pub scope: GrepScope,
    pub thread_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrepMessageMatch {
    pub message_id: i64,
    pub thread_id: String,
    pub snippet: String,
    pub created_at: DateTime<Utc>,
    pub rank: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrepSummaryMatch {
    pub summary_id: String,
    pub thread_id: String,
    pub kind: LcmSummaryKind,
    pub snippet: String,
    pub created_at: DateTime<Utc>,
    pub rank: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct GrepResult {
    pub messages: Vec<GrepMessageMatch>,
    pub summaries: Vec<GrepSummaryMatch>,
    pub total_matches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandInput {
    pub summary_id: String,
    pub depth: usize,
    pub include_messages: bool,
    pub token_cap: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandChild {
    pub summary_id: String,
    pub kind: LcmSummaryKind,
    pub content: String,
    pub token_count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandMessage {
    pub message_id: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ExpandResult {
    pub children: Vec<ExpandChild>,
    pub messages: Vec<ExpandMessage>,
    pub estimated_tokens: i64,
    pub truncated: bool,
}

pub(crate) struct RetrievalEngine<'a> {
    store: &'a dyn LcmStore,
}

impl<'a> RetrievalEngine<'a> {
    pub(crate) fn new(store: &'a dyn LcmStore) -> Self {
        Self { store }
    }

    pub(crate) async fn describe(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        if id.starts_with("sum_") {
            return self.describe_summary(id).await;
        }
        if id.starts_with("file_") {
            return self.describe_file(id).await;
        }
        Ok(None)
    }

    pub(crate) async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult> {
        let mut result = GrepResult::default();
        let Some(thread_id) = input.thread_id.as_deref() else {
            return Ok(result);
        };

        if matches!(input.scope, GrepScope::Messages | GrepScope::Both) {
            let messages = self.store.get_messages(thread_id, None, None).await?;
            for message in messages {
                if !within_time_window(message.created_at, input.since, input.before) {
                    continue;
                }
                if !matches_text(&message.content, &input.query, input.mode)? {
                    continue;
                }
                result.messages.push(GrepMessageMatch {
                    message_id: message.message_id,
                    thread_id: message.thread_id,
                    snippet: build_snippet(&message.content),
                    created_at: message.created_at,
                    rank: 0,
                });
            }
            result
                .messages
                .sort_by(|left, right| right.created_at.cmp(&left.created_at));
            if let Some(limit) = input.limit {
                result.messages.truncate(limit);
            }
        }

        if matches!(input.scope, GrepScope::Summaries | GrepScope::Both) {
            let summaries = self.store.get_summaries_by_thread(thread_id).await?;
            for summary in summaries {
                if !within_time_window(summary.created_at, input.since, input.before) {
                    continue;
                }
                if !matches_text(&summary.content, &input.query, input.mode)? {
                    continue;
                }
                result.summaries.push(GrepSummaryMatch {
                    summary_id: summary.summary_id,
                    thread_id: summary.thread_id,
                    kind: summary.kind,
                    snippet: build_snippet(&summary.content),
                    created_at: summary.created_at,
                    rank: 0,
                });
            }
            result
                .summaries
                .sort_by(|left, right| right.created_at.cmp(&left.created_at));
            if let Some(limit) = input.limit {
                result.summaries.truncate(limit);
            }
        }

        result.total_matches = result.messages.len().saturating_add(result.summaries.len());
        Ok(result)
    }

    pub(crate) async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        let mut result = ExpandResult::default();
        self.expand_recursive(
            &input.summary_id,
            input.depth,
            input.include_messages,
            input.token_cap,
            &mut result,
        )
        .await?;
        Ok(result)
    }

    async fn describe_summary(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        let Some(summary) = self.store.get_summary(id).await? else {
            return Ok(None);
        };

        let parents = self.store.get_summary_parents(id).await?;
        let children = self.store.get_summary_children(id).await?;
        let message_ids = self.store.get_summary_messages(id).await?;
        let subtree = self.store.get_summary_subtree(id).await?;

        Ok(Some(DescribeResult {
            id: id.to_string(),
            item: DescribeItem::Summary(DescribeSummary {
                thread_id: summary.thread_id,
                kind: summary.kind,
                content: summary.content,
                depth: summary.depth,
                token_count: summary.token_count,
                descendant_count: summary.descendant_count,
                descendant_token_count: summary.descendant_token_count,
                source_message_token_count: summary.source_message_token_count,
                file_ids: summary.file_ids,
                parent_ids: parents
                    .into_iter()
                    .map(|summary| summary.summary_id)
                    .collect(),
                child_ids: children
                    .into_iter()
                    .map(|summary| summary.summary_id)
                    .collect(),
                message_ids,
                earliest_at: summary.earliest_at,
                latest_at: summary.latest_at,
                subtree: subtree
                    .into_iter()
                    .map(|node| DescribeSummarySubtreeNode {
                        summary_id: node.summary.summary_id,
                        parent_summary_id: node.parent_summary_id,
                        depth_from_root: node.depth_from_root,
                        kind: node.summary.kind,
                        depth: node.summary.depth,
                        token_count: node.summary.token_count,
                        descendant_count: node.summary.descendant_count,
                        descendant_token_count: node.summary.descendant_token_count,
                        source_message_token_count: node.summary.source_message_token_count,
                        earliest_at: node.summary.earliest_at,
                        latest_at: node.summary.latest_at,
                        child_count: node.child_count,
                        path: node.path,
                    })
                    .collect(),
                created_at: summary.created_at,
            }),
        }))
    }

    async fn describe_file(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        let Some(file) = self.store.get_large_file(id).await? else {
            return Ok(None);
        };
        Ok(Some(DescribeResult {
            id: id.to_string(),
            item: DescribeItem::File(DescribeFile {
                thread_id: file.thread_id,
                file_name: file.file_name,
                mime_type: file.mime_type,
                byte_size: file.byte_size,
                storage_uri: file.storage_uri,
                exploration_summary: file.exploration_summary,
                created_at: file.created_at,
            }),
        }))
    }

    async fn expand_recursive(
        &self,
        summary_id: &str,
        depth: usize,
        include_messages: bool,
        token_cap: i64,
        result: &mut ExpandResult,
    ) -> anyhow::Result<()> {
        if depth == 0 || result.truncated {
            return Ok(());
        }

        let Some(summary) = self.store.get_summary(summary_id).await? else {
            return Ok(());
        };

        if matches!(summary.kind, LcmSummaryKind::Condensed) {
            let children = self.store.get_summary_children(summary_id).await?;
            for child in children {
                if result.truncated {
                    break;
                }
                if result.estimated_tokens.saturating_add(child.token_count) > token_cap {
                    result.truncated = true;
                    break;
                }
                result.children.push(ExpandChild {
                    summary_id: child.summary_id.clone(),
                    kind: child.kind,
                    content: child.content.clone(),
                    token_count: child.token_count,
                });
                result.estimated_tokens = result.estimated_tokens.saturating_add(child.token_count);
                if depth > 1 {
                    Box::pin(self.expand_recursive(
                        &child.summary_id,
                        depth - 1,
                        include_messages,
                        token_cap,
                        result,
                    ))
                    .await?;
                }
            }
        } else if include_messages {
            let message_ids = self.store.get_summary_messages(summary_id).await?;
            for message_id in message_ids {
                if result.truncated {
                    break;
                }
                let Some(message) = self.store.get_message_by_id(message_id).await? else {
                    continue;
                };
                let token_count = if message.token_count > 0 {
                    message.token_count
                } else {
                    estimate_tokens(&message.content)
                };
                if result.estimated_tokens.saturating_add(token_count) > token_cap {
                    result.truncated = true;
                    break;
                }
                result
                    .messages
                    .push(message_to_expand_message(message, token_count));
                result.estimated_tokens = result.estimated_tokens.saturating_add(token_count);
            }
        }

        Ok(())
    }
}

fn message_to_expand_message(message: LcmMessageRecord, token_count: i64) -> ExpandMessage {
    ExpandMessage {
        message_id: message.message_id,
        role: format!("{:?}", message.role).to_ascii_lowercase(),
        content: message.content,
        token_count,
    }
}

fn build_snippet(content: &str) -> String {
    content.chars().take(100).collect()
}

fn within_time_window(
    created_at: DateTime<Utc>,
    since: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
) -> bool {
    if let Some(since) = since
        && created_at < since
    {
        return false;
    }
    if let Some(before) = before
        && created_at >= before
    {
        return false;
    }
    true
}

fn matches_text(content: &str, query: &str, mode: GrepMode) -> anyhow::Result<bool> {
    match mode {
        GrepMode::FullText => Ok(content.contains(query)),
        GrepMode::Regex => Ok(Regex::new(query)?.is_match(content)),
    }
}

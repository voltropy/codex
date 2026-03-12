use super::AssembleContextInput;
use super::assembler::ContextAssembler;
use super::compaction::CompactionConfig;
use super::compaction::CompactionEngine;
use super::retrieval::DescribeItem;
use super::retrieval::ExpandInput;
use super::retrieval::GrepInput;
use super::retrieval::GrepMode;
use super::retrieval::GrepScope;
use super::retrieval::RetrievalEngine;
use super::store::LcmStore;
use async_trait::async_trait;
use chrono::Utc;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_state::LcmContextItemRecord;
use codex_state::LcmContextItemType;
use codex_state::LcmCreateLargeFileParams;
use codex_state::LcmCreateMessageParams;
use codex_state::LcmCreateSummaryParams;
use codex_state::LcmLargeFileRecord;
use codex_state::LcmMessageRecord;
use codex_state::LcmMessageRole;
use codex_state::LcmReplaceContextRangeParams;
use codex_state::LcmSummaryKind;
use codex_state::LcmSummaryRecord;
use codex_state::LcmSummarySubtreeNodeRecord;
use pretty_assertions::assert_eq;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

const THREAD_ID: &str = "thread-1";

#[derive(Default)]
struct MockStoreData {
    next_message_id: i64,
    messages: Vec<LcmMessageRecord>,
    summaries: Vec<LcmSummaryRecord>,
    context_items: Vec<LcmContextItemRecord>,
    summary_messages: Vec<(String, i64, i64)>,
    summary_parents: Vec<(String, String, i64)>,
    large_files: Vec<LcmLargeFileRecord>,
}

#[derive(Default)]
struct MockLcmStore {
    data: Mutex<MockStoreData>,
}

impl MockLcmStore {
    fn messages(&self) -> Vec<LcmMessageRecord> {
        self.data.lock().expect("lock").messages.clone()
    }

    fn summaries(&self) -> Vec<LcmSummaryRecord> {
        self.data.lock().expect("lock").summaries.clone()
    }

    fn context_items(&self) -> Vec<LcmContextItemRecord> {
        self.data.lock().expect("lock").context_items.clone()
    }

    fn summary_parents(&self) -> Vec<(String, String, i64)> {
        self.data.lock().expect("lock").summary_parents.clone()
    }
}

#[async_trait]
impl LcmStore for MockLcmStore {
    async fn create_message(
        &self,
        params: LcmCreateMessageParams,
    ) -> anyhow::Result<LcmMessageRecord> {
        let mut data = self.data.lock().expect("lock");
        data.next_message_id += 1;
        let record = LcmMessageRecord {
            message_id: data.next_message_id,
            thread_id: params.thread_id,
            seq: params.seq,
            role: params.role,
            content: params.content,
            raw_item: params.raw_item,
            token_count: params.token_count,
            created_at: Utc::now(),
        };
        data.messages.push(record.clone());
        Ok(record)
    }

    async fn get_message_by_id(&self, message_id: i64) -> anyhow::Result<Option<LcmMessageRecord>> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .messages
            .iter()
            .find(|record| record.message_id == message_id)
            .cloned())
    }

    async fn get_messages(
        &self,
        thread_id: &str,
        after_seq: Option<i64>,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<LcmMessageRecord>> {
        let mut messages = self
            .data
            .lock()
            .expect("lock")
            .messages
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .filter(|record| after_seq.is_none_or(|after_seq| record.seq > after_seq))
            .cloned()
            .collect::<Vec<_>>();
        messages.sort_by_key(|record| record.seq);
        if let Some(limit) = limit {
            messages.truncate(limit);
        }
        Ok(messages)
    }

    async fn get_message_count(&self, thread_id: &str) -> anyhow::Result<usize> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .messages
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .count())
    }

    async fn get_max_seq(&self, thread_id: &str) -> anyhow::Result<i64> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .messages
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .map(|record| record.seq)
            .max()
            .unwrap_or(0))
    }

    async fn insert_summary(
        &self,
        params: LcmCreateSummaryParams,
    ) -> anyhow::Result<LcmSummaryRecord> {
        let record = LcmSummaryRecord {
            summary_id: params.summary_id,
            thread_id: params.thread_id,
            kind: params.kind,
            depth: params.depth,
            content: params.content,
            token_count: params.token_count,
            file_ids: params.file_ids,
            earliest_at: params.earliest_at,
            latest_at: params.latest_at,
            descendant_count: params.descendant_count,
            descendant_token_count: params.descendant_token_count,
            source_message_token_count: params.source_message_token_count,
            created_at: Utc::now(),
        };
        self.data
            .lock()
            .expect("lock")
            .summaries
            .push(record.clone());
        Ok(record)
    }

    async fn get_summary(&self, summary_id: &str) -> anyhow::Result<Option<LcmSummaryRecord>> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .summaries
            .iter()
            .find(|record| record.summary_id == summary_id)
            .cloned())
    }

    async fn get_summaries_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let mut summaries = self
            .data
            .lock()
            .expect("lock")
            .summaries
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .cloned()
            .collect::<Vec<_>>();
        summaries.sort_by_key(|summary| summary.created_at);
        Ok(summaries)
    }

    async fn link_summary_to_messages(
        &self,
        summary_id: &str,
        message_ids: &[i64],
    ) -> anyhow::Result<()> {
        let mut data = self.data.lock().expect("lock");
        for (ordinal, message_id) in message_ids.iter().enumerate() {
            data.summary_messages.push((
                summary_id.to_string(),
                *message_id,
                i64::try_from(ordinal).unwrap_or(0),
            ));
        }
        Ok(())
    }

    async fn link_summary_to_parents(
        &self,
        summary_id: &str,
        parent_summary_ids: &[String],
    ) -> anyhow::Result<()> {
        let mut data = self.data.lock().expect("lock");
        for (ordinal, parent_summary_id) in parent_summary_ids.iter().enumerate() {
            data.summary_parents.push((
                summary_id.to_string(),
                parent_summary_id.clone(),
                i64::try_from(ordinal).unwrap_or(0),
            ));
        }
        Ok(())
    }

    async fn get_summary_messages(&self, summary_id: &str) -> anyhow::Result<Vec<i64>> {
        let mut messages = self
            .data
            .lock()
            .expect("lock")
            .summary_messages
            .iter()
            .filter(|(id, _, _)| id == summary_id)
            .cloned()
            .collect::<Vec<_>>();
        messages.sort_by_key(|(_, _, ordinal)| *ordinal);
        Ok(messages
            .into_iter()
            .map(|(_, message_id, _)| message_id)
            .collect())
    }

    async fn get_summary_parents(&self, summary_id: &str) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let data = self.data.lock().expect("lock");
        let mut parent_ids = data
            .summary_parents
            .iter()
            .filter(|(child_id, _, _)| child_id == summary_id)
            .cloned()
            .collect::<Vec<_>>();
        parent_ids.sort_by_key(|(_, _, ordinal)| *ordinal);
        Ok(parent_ids
            .into_iter()
            .filter_map(|(_, parent_id, _)| {
                data.summaries
                    .iter()
                    .find(|summary| summary.summary_id == parent_id)
                    .cloned()
            })
            .collect())
    }

    async fn get_summary_children(
        &self,
        parent_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let data = self.data.lock().expect("lock");
        let mut child_ids = data
            .summary_parents
            .iter()
            .filter(|(_, parent_id, _)| parent_id == parent_summary_id)
            .cloned()
            .collect::<Vec<_>>();
        child_ids.sort_by_key(|(_, _, ordinal)| *ordinal);
        Ok(child_ids
            .into_iter()
            .filter_map(|(child_id, _, _)| {
                data.summaries
                    .iter()
                    .find(|summary| summary.summary_id == child_id)
                    .cloned()
            })
            .collect())
    }

    async fn get_summary_subtree(
        &self,
        root_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummarySubtreeNodeRecord>> {
        let data = self.data.lock().expect("lock");
        let Some(root) = data
            .summaries
            .iter()
            .find(|summary| summary.summary_id == root_summary_id)
            .cloned()
        else {
            return Ok(Vec::new());
        };

        let mut output = Vec::new();
        let mut queue = VecDeque::from([(root.summary_id.clone(), 0i64, None, String::new())]);
        let mut seen = Vec::new();
        while let Some((summary_id, depth_from_root, parent_summary_id, path)) = queue.pop_front() {
            if seen.contains(&summary_id) {
                continue;
            }
            seen.push(summary_id.clone());
            let Some(summary) = data
                .summaries
                .iter()
                .find(|summary| summary.summary_id == summary_id)
                .cloned()
            else {
                continue;
            };

            let mut children = data
                .summary_parents
                .iter()
                .filter(|(_, parent_id, _)| parent_id == &summary_id)
                .cloned()
                .collect::<Vec<_>>();
            children.sort_by_key(|(_, _, ordinal)| *ordinal);
            output.push(LcmSummarySubtreeNodeRecord {
                summary,
                depth_from_root,
                parent_summary_id,
                path: path.clone(),
                child_count: i64::try_from(children.len()).unwrap_or(0),
            });
            for (child_id, _, ordinal) in children {
                let child_path = if path.is_empty() {
                    format!("{ordinal:04}")
                } else {
                    format!("{path}.{ordinal:04}")
                };
                queue.push_back((
                    child_id,
                    depth_from_root + 1,
                    Some(summary_id.clone()),
                    child_path,
                ));
            }
        }
        Ok(output)
    }

    async fn get_context_items(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmContextItemRecord>> {
        let mut items = self
            .data
            .lock()
            .expect("lock")
            .context_items
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .cloned()
            .collect::<Vec<_>>();
        items.sort_by_key(|item| item.ordinal);
        Ok(items)
    }

    async fn get_distinct_depths_in_context(
        &self,
        thread_id: &str,
        max_ordinal_exclusive: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        let data = self.data.lock().expect("lock");
        let mut depths = Vec::new();
        for item in data
            .context_items
            .iter()
            .filter(|item| item.thread_id == thread_id)
        {
            if !matches!(item.item_type, LcmContextItemType::Summary) {
                continue;
            }
            if let Some(max_ordinal_exclusive) = max_ordinal_exclusive
                && item.ordinal >= max_ordinal_exclusive
            {
                continue;
            }
            let Some(summary_id) = item.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = data
                .summaries
                .iter()
                .find(|summary| summary.summary_id == summary_id)
            else {
                continue;
            };
            if !depths.contains(&summary.depth) {
                depths.push(summary.depth);
            }
        }
        depths.sort_unstable();
        Ok(depths)
    }

    async fn append_context_message(&self, thread_id: &str, message_id: i64) -> anyhow::Result<()> {
        let mut data = self.data.lock().expect("lock");
        let next_ordinal = data
            .context_items
            .iter()
            .filter(|item| item.thread_id == thread_id)
            .map(|item| item.ordinal)
            .max()
            .unwrap_or(-1)
            + 1;
        data.context_items.push(LcmContextItemRecord {
            thread_id: thread_id.to_string(),
            ordinal: next_ordinal,
            item_type: LcmContextItemType::Message,
            message_id: Some(message_id),
            summary_id: None,
            created_at: Utc::now(),
        });
        Ok(())
    }

    async fn append_context_summary(
        &self,
        thread_id: &str,
        summary_id: &str,
    ) -> anyhow::Result<()> {
        let mut data = self.data.lock().expect("lock");
        let next_ordinal = data
            .context_items
            .iter()
            .filter(|item| item.thread_id == thread_id)
            .map(|item| item.ordinal)
            .max()
            .unwrap_or(-1)
            + 1;
        data.context_items.push(LcmContextItemRecord {
            thread_id: thread_id.to_string(),
            ordinal: next_ordinal,
            item_type: LcmContextItemType::Summary,
            message_id: None,
            summary_id: Some(summary_id.to_string()),
            created_at: Utc::now(),
        });
        Ok(())
    }

    async fn replace_context_range_with_summary(
        &self,
        params: LcmReplaceContextRangeParams,
    ) -> anyhow::Result<()> {
        let mut data = self.data.lock().expect("lock");
        data.context_items.retain(|item| {
            item.thread_id != params.thread_id
                || item.ordinal < params.start_ordinal
                || item.ordinal > params.end_ordinal
        });
        data.context_items.push(LcmContextItemRecord {
            thread_id: params.thread_id.clone(),
            ordinal: params.start_ordinal,
            item_type: LcmContextItemType::Summary,
            message_id: None,
            summary_id: Some(params.summary_id),
            created_at: Utc::now(),
        });
        let mut thread_items = data
            .context_items
            .iter()
            .filter(|item| item.thread_id == params.thread_id)
            .cloned()
            .collect::<Vec<_>>();
        thread_items.sort_by_key(|item| item.ordinal);
        for (ordinal, item) in thread_items.iter_mut().enumerate() {
            item.ordinal = i64::try_from(ordinal).unwrap_or(0);
        }
        data.context_items
            .retain(|item| item.thread_id != params.thread_id);
        data.context_items.extend(thread_items);
        Ok(())
    }

    async fn get_context_token_count(&self, thread_id: &str) -> anyhow::Result<i64> {
        let data = self.data.lock().expect("lock");
        let mut total = 0i64;
        for item in data
            .context_items
            .iter()
            .filter(|item| item.thread_id == thread_id)
        {
            match item.item_type {
                LcmContextItemType::Message => {
                    if let Some(message_id) = item.message_id
                        && let Some(message) = data
                            .messages
                            .iter()
                            .find(|message| message.message_id == message_id)
                    {
                        total += message.token_count;
                    }
                }
                LcmContextItemType::Summary => {
                    if let Some(summary_id) = item.summary_id.as_deref()
                        && let Some(summary) = data
                            .summaries
                            .iter()
                            .find(|summary| summary.summary_id == summary_id)
                    {
                        total += summary.token_count;
                    }
                }
            }
        }
        Ok(total)
    }

    async fn insert_large_file(
        &self,
        params: LcmCreateLargeFileParams,
    ) -> anyhow::Result<LcmLargeFileRecord> {
        let record = LcmLargeFileRecord {
            file_id: params.file_id,
            thread_id: params.thread_id,
            file_name: params.file_name,
            mime_type: params.mime_type,
            byte_size: params.byte_size,
            storage_uri: params.storage_uri,
            exploration_summary: params.exploration_summary,
            created_at: Utc::now(),
        };
        self.data
            .lock()
            .expect("lock")
            .large_files
            .push(record.clone());
        Ok(record)
    }

    async fn get_large_file(&self, file_id: &str) -> anyhow::Result<Option<LcmLargeFileRecord>> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .large_files
            .iter()
            .find(|record| record.file_id == file_id)
            .cloned())
    }

    async fn get_large_files_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmLargeFileRecord>> {
        Ok(self
            .data
            .lock()
            .expect("lock")
            .large_files
            .iter()
            .filter(|record| record.thread_id == thread_id)
            .cloned()
            .collect())
    }
}

fn estimate_tokens(text: &str) -> i64 {
    i64::try_from((text.len().saturating_add(3)) / 4).unwrap_or(i64::MAX)
}

fn make_message_item(role: &str, text: impl Into<String>) -> ResponseItem {
    let text = text.into();
    let content = if role == "assistant" {
        vec![ContentItem::OutputText { text }]
    } else {
        vec![ContentItem::InputText { text }]
    };
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content,
        end_turn: None,
        phase: None,
    }
}

fn extract_message_text(item: &ResponseItem) -> String {
    match item {
        ResponseItem::Message { content, .. } => {
            crate::compact::content_items_to_text(content).unwrap_or_default()
        }
        _ => String::new(),
    }
}

async fn ingest_messages(
    store: &MockLcmStore,
    count: usize,
    content_fn: impl Fn(usize) -> String,
    role_fn: impl Fn(usize) -> LcmMessageRole,
    token_count_fn: impl Fn(usize, &str) -> i64,
) -> anyhow::Result<Vec<LcmMessageRecord>> {
    let mut records = Vec::new();
    for index in 0..count {
        let content = content_fn(index);
        let role = role_fn(index);
        let raw_item = make_message_item(
            match role {
                LcmMessageRole::Assistant => "assistant",
                LcmMessageRole::User => "user",
                LcmMessageRole::System => "system",
                LcmMessageRole::Tool => "tool",
            },
            content.clone(),
        );
        let message = store
            .create_message(LcmCreateMessageParams {
                thread_id: THREAD_ID.to_string(),
                seq: i64::try_from(index + 1).unwrap_or(0),
                role,
                content: content.clone(),
                raw_item,
                token_count: token_count_fn(index, &content),
            })
            .await?;
        store
            .append_context_message(THREAD_ID, message.message_id)
            .await?;
        records.push(message);
    }
    Ok(records)
}

fn default_compaction_config() -> CompactionConfig {
    CompactionConfig {
        context_threshold: 0.75,
        fresh_tail_count: 4,
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

#[tokio::test]
async fn ingested_messages_appear_in_assembled_context() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        5,
        |index| format!("Message {index}"),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let assembler = ContextAssembler::new(&store);
    let result = assembler
        .assemble(AssembleContextInput {
            thread_id: THREAD_ID.to_string(),
            token_budget: 100_000,
            fresh_tail_count: 8,
        })
        .await?;

    assert_eq!(result.messages.len(), 5);
    assert_eq!(result.stats.raw_message_count, 5);
    assert_eq!(result.stats.summary_count, 0);
    assert_eq!(result.stats.total_context_items, 5);
    for (index, message) in result.messages.iter().enumerate() {
        assert_eq!(extract_message_text(message), format!("Message {index}"));
    }
    Ok(())
}

#[tokio::test]
async fn assembler_respects_token_budget_by_dropping_oldest_items() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        10,
        |index| format!("M{index} {}", "x".repeat(396)),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let assembler = ContextAssembler::new(&store);
    let result = assembler
        .assemble(AssembleContextInput {
            thread_id: THREAD_ID.to_string(),
            token_budget: 500,
            fresh_tail_count: 4,
        })
        .await?;

    assert!(result.messages.len() < 10);
    assert!(result.messages.len() <= 5);
    let last_four = &result.messages[result.messages.len().saturating_sub(4)..];
    for (offset, message) in last_four.iter().enumerate() {
        assert!(extract_message_text(message).contains(&format!("M{}", 6 + offset)));
    }
    Ok(())
}

#[tokio::test]
async fn assembler_includes_summaries_alongside_messages() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        2,
        |index| format!("Message {index}"),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_test_001".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "This is a leaf summary of earlier conversation.".to_string(),
            token_count: 20,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store
        .append_context_summary(THREAD_ID, "sum_test_001")
        .await?;
    ingest_messages(
        &store,
        2,
        |index| format!("Later message {index}"),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let assembler = ContextAssembler::new(&store);
    let result = assembler
        .assemble(AssembleContextInput {
            thread_id: THREAD_ID.to_string(),
            token_budget: 100_000,
            fresh_tail_count: 8,
        })
        .await?;

    assert_eq!(result.messages.len(), 5);
    assert_eq!(result.stats.raw_message_count, 4);
    assert_eq!(result.stats.summary_count, 1);
    let summary_message = result
        .messages
        .iter()
        .find(|message| extract_message_text(message).contains(r#"<summary id="sum_test_001""#))
        .expect("summary message");
    assert!(extract_message_text(summary_message).contains("This is a leaf summary"));
    Ok(())
}

#[tokio::test]
async fn compaction_creates_leaf_summary_from_oldest_messages() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        10,
        |index| format!("Turn {index}: discussion about topic {index}"),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let engine = CompactionEngine::new(&store, default_compaction_config());
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            10_000,
            &|text, _, _| Box::pin(async move { Ok(format!("Summary: {}", text.len())) }),
            true,
            false,
        )
        .await?;

    assert!(result.action_taken);
    assert!(result.created_summary_id.is_some());
    assert!(
        store
            .summaries()
            .iter()
            .any(|summary| matches!(summary.kind, LcmSummaryKind::Leaf))
    );
    assert!(store.context_items().len() < 10);
    Ok(())
}

#[tokio::test]
async fn compact_leaf_uses_preceding_summary_context_for_soft_leaf_continuity() -> anyhow::Result<()>
{
    let store = MockLcmStore::default();
    let config = CompactionConfig {
        fresh_tail_count: 1,
        ..default_compaction_config()
    };
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_pre_1".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "Prior summary one.".to_string(),
            token_count: 4,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store.append_context_summary(THREAD_ID, "sum_pre_1").await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_pre_2".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "Prior summary two.".to_string(),
            token_count: 4,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store.append_context_summary(THREAD_ID, "sum_pre_2").await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_pre_3".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "Prior summary three.".to_string(),
            token_count: 4,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store.append_context_summary(THREAD_ID, "sum_pre_3").await?;
    ingest_messages(
        &store,
        4,
        |index| format!("Turn {index}: {}", "k".repeat(160)),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, _| 40,
    )
    .await?;

    let engine = CompactionEngine::new(&store, config);
    let previous_summaries = Arc::new(Mutex::new(Vec::new()));
    let result = engine
        .compact_leaf(
            THREAD_ID,
            200,
            &{
                let previous_summaries = Arc::clone(&previous_summaries);
                move |_, _, options| {
                    let previous_summaries = Arc::clone(&previous_summaries);
                    Box::pin(async move {
                        previous_summaries
                            .lock()
                            .expect("lock")
                            .push(options.and_then(|options| options.previous_summary));
                        Ok("Leaf summary with continuity.".to_string())
                    })
                }
            },
            true,
            None,
        )
        .await?;

    assert!(result.action_taken);
    assert_eq!(
        previous_summaries.lock().expect("lock")[0].as_deref(),
        Some("Prior summary two.\n\nPrior summary three.")
    );
    Ok(())
}

#[tokio::test]
async fn compaction_propagates_referenced_file_ids_into_summary_metadata() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    let config = CompactionConfig {
        fresh_tail_count: 16,
        ..default_compaction_config()
    };
    ingest_messages(
        &store,
        20,
        |index| match index {
            1 => "Review [LCM File: file_aaaabbbbccccdddd | spec.md | text/markdown | 1,024 bytes]"
                .to_string(),
            2 => "Also inspect file_1111222233334444 and file_aaaabbbbccccdddd.".to_string(),
            _ => format!("Turn {index}: regular planning text."),
        },
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let engine = CompactionEngine::new(&store, config);
    let _ = engine
        .compact_full_sweep(
            THREAD_ID,
            10_000,
            &|_, _, _| Box::pin(async move { Ok("Condensed file-aware summary.".to_string()) }),
            true,
            false,
        )
        .await?;

    let leaf_summary = store
        .summaries()
        .into_iter()
        .find(|summary| matches!(summary.kind, LcmSummaryKind::Leaf))
        .expect("leaf summary");
    assert_eq!(
        leaf_summary.file_ids,
        vec![
            "file_aaaabbbbccccdddd".to_string(),
            "file_1111222233334444".to_string()
        ]
    );
    Ok(())
}

#[tokio::test]
async fn compaction_emits_durable_messages_for_leaf_and_condensed_passes() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    let config = CompactionConfig {
        leaf_min_fanout: 2,
        leaf_chunk_tokens: Some(100),
        condensed_target_tokens: 10,
        ..default_compaction_config()
    };
    ingest_messages(
        &store,
        8,
        |index| format!("Turn {index}: {}", "c".repeat(200)),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, _| 50,
    )
    .await?;

    let engine = CompactionEngine::new(&store, config);
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            260,
            &|_, _, _| Box::pin(async move { Ok("Compacted summary block.".to_string()) }),
            false,
            false,
        )
        .await?;

    assert!(result.action_taken);
    assert!(result.condensed);
    let event_messages = store
        .messages()
        .into_iter()
        .filter(|message| {
            matches!(message.role, LcmMessageRole::System) && message.content.contains("\"pass\"")
        })
        .collect::<Vec<_>>();
    assert!(event_messages.len() >= 2);
    Ok(())
}

#[tokio::test]
async fn depth_aware_condensation_sets_condensed_depth_to_max_parent_depth_plus_one()
-> anyhow::Result<()> {
    let store = MockLcmStore::default();
    let config = CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: Some(200),
        condensed_target_tokens: 10,
        ..default_compaction_config()
    };
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_depth_parent_a".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Condensed,
            depth: 1,
            content: "Depth one summary A".to_string(),
            token_count: 60,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store
        .append_context_summary(THREAD_ID, "sum_depth_parent_a")
        .await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_depth_parent_b".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Condensed,
            depth: 1,
            content: "Depth one summary B".to_string(),
            token_count: 60,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    store
        .append_context_summary(THREAD_ID, "sum_depth_parent_b")
        .await?;

    let engine = CompactionEngine::new(&store, config);
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            500,
            &|_, _, _| Box::pin(async move { Ok("Depth two merged summary".to_string()) }),
            true,
            false,
        )
        .await?;

    let created_summary = store
        .summaries()
        .into_iter()
        .find(|summary| Some(summary.summary_id.clone()) == result.created_summary_id)
        .expect("created summary");
    assert_eq!(created_summary.depth, 2);
    Ok(())
}

#[tokio::test]
async fn compaction_escalates_to_aggressive_when_normal_does_not_converge() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        8,
        |index| format!("Content {index}: {}", "a".repeat(200)),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let engine = CompactionEngine::new(&store, default_compaction_config());
    let normal_count = Arc::new(Mutex::new(0usize));
    let aggressive_count = Arc::new(Mutex::new(0usize));
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            10_000,
            &{
                let normal_count = Arc::clone(&normal_count);
                let aggressive_count = Arc::clone(&aggressive_count);
                move |text, aggressive, _| {
                    let normal_count = Arc::clone(&normal_count);
                    let aggressive_count = Arc::clone(&aggressive_count);
                    Box::pin(async move {
                        if aggressive {
                            *aggressive_count.lock().expect("lock") += 1;
                            Ok("Aggressively summarized.".to_string())
                        } else {
                            *normal_count.lock().expect("lock") += 1;
                            Ok(format!("{text} (expanded, not summarized)"))
                        }
                    })
                }
            },
            true,
            false,
        )
        .await?;

    assert!(result.action_taken);
    assert_eq!(
        result.level,
        Some(super::compaction::CompactionLevel::Aggressive)
    );
    assert!(*normal_count.lock().expect("lock") >= 1);
    assert!(*aggressive_count.lock().expect("lock") >= 1);
    Ok(())
}

#[tokio::test]
async fn compaction_falls_back_to_truncation_when_aggressive_does_not_converge()
-> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        8,
        |index| format!("Content {index}: {}", "b".repeat(200)),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let engine = CompactionEngine::new(&store, default_compaction_config());
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            10_000,
            &|text, _, _| Box::pin(async move { Ok(format!("{text} (not actually summarized)")) }),
            true,
            false,
        )
        .await?;

    assert_eq!(
        result.level,
        Some(super::compaction::CompactionLevel::Fallback)
    );
    let leaf_summary = store
        .summaries()
        .into_iter()
        .find(|summary| matches!(summary.kind, LcmSummaryKind::Leaf))
        .expect("leaf summary");
    assert!(leaf_summary.content.contains("[Truncated from"));
    Ok(())
}

#[tokio::test]
async fn retrieval_describe_and_expand_returns_lineage_and_messages() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    let messages = ingest_messages(
        &store,
        3,
        |index| format!("Source message {index}"),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_leaf_with_msgs".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "Leaf summary of 3 messages.".to_string(),
            token_count: 10,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;
    let message_ids = messages
        .iter()
        .map(|message| message.message_id)
        .collect::<Vec<_>>();
    store
        .link_summary_to_messages("sum_leaf_with_msgs", &message_ids)
        .await?;

    let retrieval = RetrievalEngine::new(&store);
    let describe = retrieval
        .describe("sum_leaf_with_msgs")
        .await?
        .expect("describe result");
    match describe.item {
        DescribeItem::Summary(summary) => {
            assert_eq!(summary.message_ids, message_ids);
            assert_eq!(summary.parent_ids, Vec::<String>::new());
        }
        DescribeItem::File(_) => panic!("expected summary"),
    }

    let expand = retrieval
        .expand(ExpandInput {
            summary_id: "sum_leaf_with_msgs".to_string(),
            depth: 1,
            include_messages: true,
            token_cap: i64::MAX,
        })
        .await?;
    assert_eq!(expand.messages.len(), 3);
    assert_eq!(expand.messages[0].content, "Source message 0");
    Ok(())
}

#[tokio::test]
async fn grep_searches_messages_and_summaries() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        5,
        |index| {
            if index == 2 {
                "This message mentions the deployment bug".to_string()
            } else {
                format!("Regular message {index}")
            }
        },
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;
    store
        .insert_summary(LcmCreateSummaryParams {
            summary_id: "sum_search_001".to_string(),
            thread_id: THREAD_ID.to_string(),
            kind: LcmSummaryKind::Leaf,
            depth: 0,
            content: "Summary mentioning the deployment bug fix.".to_string(),
            token_count: 15,
            file_ids: Vec::new(),
            earliest_at: None,
            latest_at: None,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 0,
        })
        .await?;

    let retrieval = RetrievalEngine::new(&store);
    let result = retrieval
        .grep(GrepInput {
            query: "deployment".to_string(),
            mode: GrepMode::FullText,
            scope: GrepScope::Both,
            thread_id: Some(THREAD_ID.to_string()),
            since: None,
            before: None,
            limit: None,
        })
        .await?;

    assert!(result.total_matches >= 2);
    assert!(!result.messages.is_empty());
    assert!(!result.summaries.is_empty());
    Ok(())
}

#[tokio::test]
async fn full_round_trip_messages_survive_compaction() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    ingest_messages(
        &store,
        20,
        |index| format!("Discussion turn {index}: topic about integration testing."),
        |index| {
            if index % 2 == 0 {
                LcmMessageRole::User
            } else {
                LcmMessageRole::Assistant
            }
        },
        |_, content| estimate_tokens(content),
    )
    .await?;

    let engine = CompactionEngine::new(&store, default_compaction_config());
    let result = engine
        .compact_full_sweep(
            THREAD_ID,
            10_000,
            &|text, _, _| {
                Box::pin(async move {
                    Ok(format!(
                        "Compacted summary: covered {} chars of discussion.",
                        text.len()
                    ))
                })
            },
            true,
            false,
        )
        .await?;
    assert!(result.action_taken);

    let assembler = ContextAssembler::new(&store);
    let assembled = assembler
        .assemble(AssembleContextInput {
            thread_id: THREAD_ID.to_string(),
            token_budget: 100_000,
            fresh_tail_count: 4,
        })
        .await?;
    assert!(assembled.stats.total_context_items < 20);
    assert!(assembled.stats.summary_count >= 1);

    let retrieval = RetrievalEngine::new(&store);
    let describe = retrieval
        .describe(result.created_summary_id.as_deref().expect("summary id"))
        .await?
        .expect("describe");
    match describe.item {
        DescribeItem::Summary(summary) => {
            assert!(summary.content.contains("Compacted summary"));
        }
        DescribeItem::File(_) => panic!("expected summary"),
    }

    let expand = retrieval
        .expand(ExpandInput {
            summary_id: result.created_summary_id.expect("summary id"),
            depth: 1,
            include_messages: true,
            token_cap: i64::MAX,
        })
        .await?;
    assert!(!expand.messages.is_empty());
    assert!(
        expand
            .messages
            .iter()
            .all(|message| message.content.contains("Discussion turn"))
    );
    Ok(())
}

#[tokio::test]
async fn describe_returns_file_info_for_file_ids() -> anyhow::Result<()> {
    let store = MockLcmStore::default();
    store
        .insert_large_file(LcmCreateLargeFileParams {
            file_id: "file_test_001".to_string(),
            thread_id: THREAD_ID.to_string(),
            file_name: Some("data.csv".to_string()),
            mime_type: Some("text/csv".to_string()),
            byte_size: Some(1024),
            storage_uri: "s3://bucket/data.csv".to_string(),
            exploration_summary: Some("CSV with 100 rows of test data.".to_string()),
        })
        .await?;

    let retrieval = RetrievalEngine::new(&store);
    let describe = retrieval
        .describe("file_test_001")
        .await?
        .expect("describe");
    match describe.item {
        DescribeItem::File(file) => {
            assert_eq!(file.file_name.as_deref(), Some("data.csv"));
            assert_eq!(file.storage_uri, "s3://bucket/data.csv");
        }
        DescribeItem::Summary(_) => panic!("expected file"),
    }
    Ok(())
}

use super::estimate_tokens;
use super::store::LcmStore;
use chrono::SecondsFormat;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_state::LcmContextItemType;
use codex_state::LcmSummaryKind;
use codex_state::LcmSummaryRecord;

#[derive(Debug, Clone)]
pub(crate) struct AssembleContextInput {
    pub thread_id: String,
    pub token_budget: i64,
    pub fresh_tail_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AssembleContextStats {
    pub raw_message_count: usize,
    pub summary_count: usize,
    pub total_context_items: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct AssembleContextResult {
    pub messages: Vec<ResponseItem>,
    pub estimated_tokens: i64,
    pub system_prompt_addition: Option<String>,
    pub stats: AssembleContextStats,
}

#[derive(Debug, Clone)]
struct ResolvedItem {
    ordinal: i64,
    message: ResponseItem,
    tokens: i64,
    is_message: bool,
}

pub(crate) struct ContextAssembler<'a> {
    store: &'a dyn LcmStore,
}

impl<'a> ContextAssembler<'a> {
    pub(crate) fn new(store: &'a dyn LcmStore) -> Self {
        Self { store }
    }

    pub(crate) async fn assemble(
        &self,
        input: AssembleContextInput,
    ) -> anyhow::Result<AssembleContextResult> {
        let context_items = self.store.get_context_items(&input.thread_id).await?;
        if context_items.is_empty() {
            return Ok(AssembleContextResult::default());
        }

        let mut resolved = Vec::with_capacity(context_items.len());
        for item in context_items {
            let Some(resolved_item) = self.resolve_item(item).await? else {
                continue;
            };
            resolved.push(resolved_item);
        }

        let mut stats = AssembleContextStats::default();
        for item in &resolved {
            if item.is_message {
                stats.raw_message_count += 1;
            } else {
                stats.summary_count += 1;
            }
        }
        stats.total_context_items = resolved.len();

        let tail_start = resolved.len().saturating_sub(input.fresh_tail_count);
        let fresh_tail = &resolved[tail_start..];
        let evictable = &resolved[..tail_start];

        let tail_tokens = fresh_tail.iter().map(|item| item.tokens).sum::<i64>();
        let remaining_budget = input.token_budget.saturating_sub(tail_tokens).max(0);

        let mut selected = Vec::new();
        let evictable_total_tokens = evictable.iter().map(|item| item.tokens).sum::<i64>();
        let evictable_tokens = if evictable_total_tokens <= remaining_budget {
            selected.extend(evictable.iter().cloned());
            evictable_total_tokens
        } else {
            let mut kept = Vec::new();
            let mut accum = 0i64;
            for item in evictable.iter().rev() {
                if accum.saturating_add(item.tokens) <= remaining_budget {
                    kept.push(item.clone());
                    accum = accum.saturating_add(item.tokens);
                } else {
                    break;
                }
            }
            kept.reverse();
            selected.extend(kept);
            accum
        };

        selected.extend(fresh_tail.iter().cloned());
        selected.sort_by_key(|item| item.ordinal);

        Ok(AssembleContextResult {
            messages: selected.into_iter().map(|item| item.message).collect(),
            estimated_tokens: evictable_tokens.saturating_add(tail_tokens),
            system_prompt_addition: None,
            stats,
        })
    }

    async fn resolve_item(
        &self,
        item: codex_state::LcmContextItemRecord,
    ) -> anyhow::Result<Option<ResolvedItem>> {
        match item.item_type {
            LcmContextItemType::Message => {
                let Some(message_id) = item.message_id else {
                    return Ok(None);
                };
                let Some(message) = self.store.get_message_by_id(message_id).await? else {
                    return Ok(None);
                };
                Ok(Some(ResolvedItem {
                    ordinal: item.ordinal,
                    message: message.raw_item,
                    tokens: message.token_count.max(0),
                    is_message: true,
                }))
            }
            LcmContextItemType::Summary => {
                let Some(summary_id) = item.summary_id else {
                    return Ok(None);
                };
                let Some(summary) = self.store.get_summary(&summary_id).await? else {
                    return Ok(None);
                };
                let content = format_summary_content(&summary, self.store).await?;
                Ok(Some(ResolvedItem {
                    ordinal: item.ordinal,
                    message: ResponseItem::Message {
                        id: None,
                        role: "user".to_string(),
                        content: vec![ContentItem::InputText {
                            text: content.clone(),
                        }],
                        end_turn: None,
                        phase: None,
                    },
                    tokens: estimate_tokens(&content),
                    is_message: false,
                }))
            }
        }
    }
}

async fn format_summary_content(
    summary: &LcmSummaryRecord,
    store: &dyn LcmStore,
) -> anyhow::Result<String> {
    let mut attributes = vec![
        format!(r#"id="{}""#, summary.summary_id),
        format!(
            r#"kind="{}""#,
            match summary.kind {
                LcmSummaryKind::Leaf => "leaf",
                LcmSummaryKind::Condensed => "condensed",
            }
        ),
        format!(r#"depth="{}""#, summary.depth),
        format!(r#"descendant_count="{}""#, summary.descendant_count),
    ];

    if let Some(earliest_at) = summary.earliest_at {
        attributes.push(format!(
            r#"earliest_at="{}""#,
            earliest_at.to_rfc3339_opts(SecondsFormat::Secs, true)
        ));
    }
    if let Some(latest_at) = summary.latest_at {
        attributes.push(format!(
            r#"latest_at="{}""#,
            latest_at.to_rfc3339_opts(SecondsFormat::Secs, true)
        ));
    }

    let mut lines = vec![format!("<summary {}>", attributes.join(" "))];
    if matches!(summary.kind, LcmSummaryKind::Condensed) {
        let parents = store.get_summary_parents(&summary.summary_id).await?;
        if !parents.is_empty() {
            lines.push("  <parents>".to_string());
            for parent in parents {
                lines.push(format!(r#"    <summary_ref id="{}" />"#, parent.summary_id));
            }
            lines.push("  </parents>".to_string());
        }
    }
    lines.push("  <content>".to_string());
    lines.push(summary.content.clone());
    lines.push("  </content>".to_string());
    lines.push("</summary>".to_string());
    Ok(lines.join("\n"))
}

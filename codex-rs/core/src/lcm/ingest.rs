use super::estimate_tokens;
use super::response_item_text;
use super::store::LcmStore;
use codex_protocol::models::ResponseItem;
use codex_state::LcmCreateMessageParams;
use codex_state::LcmMessageRole;

pub(crate) async fn bootstrap_lcm_history(
    store: &dyn LcmStore,
    thread_id: &str,
    items: &[ResponseItem],
) -> anyhow::Result<()> {
    let message_count = store.get_message_count(thread_id).await?;
    let context_items = store.get_context_items(thread_id).await?;
    if message_count > 0 || !context_items.is_empty() {
        return Ok(());
    }
    ingest_lcm_items(store, thread_id, items).await
}

pub(crate) async fn ingest_lcm_items(
    store: &dyn LcmStore,
    thread_id: &str,
    items: &[ResponseItem],
) -> anyhow::Result<()> {
    let mut next_seq = store.get_max_seq(thread_id).await? + 1;

    for item in items {
        if !should_ingest_into_lcm(item) {
            continue;
        }

        let content = response_item_text(item);
        let record = store
            .create_message(LcmCreateMessageParams {
                thread_id: thread_id.to_string(),
                seq: next_seq,
                role: response_item_role(item),
                content: content.clone(),
                raw_item: item.clone(),
                token_count: estimate_tokens(&content),
            })
            .await?;
        store
            .append_context_message(thread_id, record.message_id)
            .await?;
        next_seq += 1;
    }

    Ok(())
}

pub(crate) fn should_ingest_into_lcm(item: &ResponseItem) -> bool {
    !matches!(
        item,
        ResponseItem::Compaction { .. } | ResponseItem::GhostSnapshot { .. }
    )
}

fn response_item_role(item: &ResponseItem) -> LcmMessageRole {
    match item {
        ResponseItem::Message { role, .. } => match role.as_str() {
            "assistant" => LcmMessageRole::Assistant,
            "tool" => LcmMessageRole::Tool,
            "user" => LcmMessageRole::User,
            "developer" | "system" => LcmMessageRole::System,
            _ => LcmMessageRole::System,
        },
        ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::ToolSearchOutput { .. } => LcmMessageRole::Tool,
        ResponseItem::Reasoning { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::Other => LcmMessageRole::Assistant,
        ResponseItem::Compaction { .. } | ResponseItem::GhostSnapshot { .. } => {
            LcmMessageRole::System
        }
    }
}

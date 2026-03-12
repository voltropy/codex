use super::*;
use crate::model::LcmContextItemRecord;
use crate::model::LcmContextItemType;
use crate::model::LcmCreateLargeFileParams;
use crate::model::LcmCreateMessageParams;
use crate::model::LcmCreateSummaryParams;
use crate::model::LcmLargeFileRecord;
use crate::model::LcmMessageRecord;
use crate::model::LcmMessageRole;
use crate::model::LcmReplaceContextRangeParams;
use crate::model::LcmSummaryKind;
use crate::model::LcmSummaryRecord;
use crate::model::LcmSummarySubtreeNodeRecord;
use chrono::TimeZone;
use sqlx::Row;
use std::collections::HashSet;
use std::collections::VecDeque;

fn epoch_seconds_to_datetime(seconds: i64) -> anyhow::Result<DateTime<Utc>> {
    Utc.timestamp_opt(seconds, 0)
        .single()
        .ok_or_else(|| anyhow::anyhow!("invalid unix timestamp: {seconds}"))
}

fn optional_epoch_seconds_to_datetime(
    seconds: Option<i64>,
) -> anyhow::Result<Option<DateTime<Utc>>> {
    seconds.map(epoch_seconds_to_datetime).transpose()
}

impl StateRuntime {
    pub async fn lcm_create_message(
        &self,
        params: LcmCreateMessageParams,
    ) -> anyhow::Result<LcmMessageRecord> {
        let raw_item_json = serde_json::to_string(&params.raw_item)?;
        let created_at = Utc::now().timestamp();
        let result = sqlx::query(
            r#"
INSERT INTO lcm_messages (
    thread_id,
    seq,
    role,
    content,
    raw_item_json,
    token_count,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(params.thread_id.as_str())
        .bind(params.seq)
        .bind(params.role.as_str())
        .bind(params.content.as_str())
        .bind(raw_item_json)
        .bind(params.token_count)
        .bind(created_at)
        .execute(self.pool.as_ref())
        .await?;

        let message_id = result.last_insert_rowid();
        self.lcm_get_message_by_id(message_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("failed to fetch inserted LCM message {message_id}"))
    }

    pub async fn lcm_get_message_by_id(
        &self,
        message_id: i64,
    ) -> anyhow::Result<Option<LcmMessageRecord>> {
        let row = sqlx::query(
            r#"
SELECT
    message_id,
    thread_id,
    seq,
    role,
    content,
    raw_item_json,
    token_count,
    created_at
FROM lcm_messages
WHERE message_id = ?
            "#,
        )
        .bind(message_id)
        .fetch_optional(self.pool.as_ref())
        .await?;

        row.map(|row| {
            let raw_item_json: String = row.try_get("raw_item_json")?;
            Ok(LcmMessageRecord {
                message_id: row.try_get("message_id")?,
                thread_id: row.try_get("thread_id")?,
                seq: row.try_get("seq")?,
                role: LcmMessageRole::from_db(row.try_get::<&str, _>("role")?)?,
                content: row.try_get("content")?,
                raw_item: serde_json::from_str(&raw_item_json)?,
                token_count: row.try_get("token_count")?,
                created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
            })
        })
        .transpose()
    }

    pub async fn lcm_get_messages(
        &self,
        thread_id: &str,
        after_seq: Option<i64>,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<LcmMessageRecord>> {
        let rows = sqlx::query(
            r#"
SELECT
    message_id,
    thread_id,
    seq,
    role,
    content,
    raw_item_json,
    token_count,
    created_at
FROM lcm_messages
WHERE thread_id = ?
  AND (? IS NULL OR seq > ?)
ORDER BY seq ASC
LIMIT COALESCE(?, -1)
            "#,
        )
        .bind(thread_id)
        .bind(after_seq)
        .bind(after_seq)
        .bind(limit.and_then(|value| i64::try_from(value).ok()))
        .fetch_all(self.pool.as_ref())
        .await?;

        rows.into_iter()
            .map(|row| {
                let raw_item_json: String = row.try_get("raw_item_json")?;
                Ok(LcmMessageRecord {
                    message_id: row.try_get("message_id")?,
                    thread_id: row.try_get("thread_id")?,
                    seq: row.try_get("seq")?,
                    role: LcmMessageRole::from_db(row.try_get::<&str, _>("role")?)?,
                    content: row.try_get("content")?,
                    raw_item: serde_json::from_str(&raw_item_json)?,
                    token_count: row.try_get("token_count")?,
                    created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
                })
            })
            .collect()
    }

    pub async fn lcm_get_message_count(&self, thread_id: &str) -> anyhow::Result<usize> {
        let count = sqlx::query_scalar::<_, i64>(
            r#"
SELECT COUNT(*) FROM lcm_messages WHERE thread_id = ?
            "#,
        )
        .bind(thread_id)
        .fetch_one(self.pool.as_ref())
        .await?;
        usize::try_from(count).map_err(Into::into)
    }

    pub async fn lcm_get_max_seq(&self, thread_id: &str) -> anyhow::Result<i64> {
        let max_seq = sqlx::query_scalar::<_, Option<i64>>(
            r#"
SELECT MAX(seq) FROM lcm_messages WHERE thread_id = ?
            "#,
        )
        .bind(thread_id)
        .fetch_one(self.pool.as_ref())
        .await?;
        Ok(max_seq.unwrap_or(0))
    }

    pub async fn lcm_insert_summary(
        &self,
        params: LcmCreateSummaryParams,
    ) -> anyhow::Result<LcmSummaryRecord> {
        let file_ids_json = serde_json::to_string(&params.file_ids)?;
        let created_at = Utc::now().timestamp();
        sqlx::query(
            r#"
INSERT INTO lcm_summaries (
    summary_id,
    thread_id,
    kind,
    depth,
    content,
    token_count,
    file_ids,
    earliest_at,
    latest_at,
    descendant_count,
    descendant_token_count,
    source_message_token_count,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(params.summary_id.as_str())
        .bind(params.thread_id.as_str())
        .bind(params.kind.as_str())
        .bind(params.depth)
        .bind(params.content.as_str())
        .bind(params.token_count)
        .bind(file_ids_json)
        .bind(params.earliest_at.map(|value| value.timestamp()))
        .bind(params.latest_at.map(|value| value.timestamp()))
        .bind(params.descendant_count)
        .bind(params.descendant_token_count)
        .bind(params.source_message_token_count)
        .bind(created_at)
        .execute(self.pool.as_ref())
        .await?;

        self.lcm_get_summary(params.summary_id.as_str())
            .await?
            .ok_or_else(|| anyhow::anyhow!("failed to fetch inserted LCM summary"))
    }

    pub async fn lcm_get_summary(
        &self,
        summary_id: &str,
    ) -> anyhow::Result<Option<LcmSummaryRecord>> {
        let row = sqlx::query(
            r#"
SELECT
    summary_id,
    thread_id,
    kind,
    depth,
    content,
    token_count,
    file_ids,
    earliest_at,
    latest_at,
    descendant_count,
    descendant_token_count,
    source_message_token_count,
    created_at
FROM lcm_summaries
WHERE summary_id = ?
            "#,
        )
        .bind(summary_id)
        .fetch_optional(self.pool.as_ref())
        .await?;

        row.map(|row| {
            let file_ids_json: String = row.try_get("file_ids")?;
            Ok(LcmSummaryRecord {
                summary_id: row.try_get("summary_id")?,
                thread_id: row.try_get("thread_id")?,
                kind: LcmSummaryKind::from_db(row.try_get::<&str, _>("kind")?)?,
                depth: row.try_get("depth")?,
                content: row.try_get("content")?,
                token_count: row.try_get("token_count")?,
                file_ids: serde_json::from_str(&file_ids_json)?,
                earliest_at: optional_epoch_seconds_to_datetime(row.try_get("earliest_at")?)?,
                latest_at: optional_epoch_seconds_to_datetime(row.try_get("latest_at")?)?,
                descendant_count: row.try_get("descendant_count")?,
                descendant_token_count: row.try_get("descendant_token_count")?,
                source_message_token_count: row.try_get("source_message_token_count")?,
                created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
            })
        })
        .transpose()
    }

    pub async fn lcm_get_summaries_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let summary_ids = sqlx::query_scalar::<_, String>(
            r#"
SELECT summary_id
FROM lcm_summaries
WHERE thread_id = ?
ORDER BY created_at ASC, summary_id ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(self.pool.as_ref())
        .await?;

        let mut summaries = Vec::with_capacity(summary_ids.len());
        for summary_id in summary_ids {
            if let Some(summary) = self.lcm_get_summary(&summary_id).await? {
                summaries.push(summary);
            }
        }
        Ok(summaries)
    }

    pub async fn lcm_link_summary_to_messages(
        &self,
        summary_id: &str,
        message_ids: &[i64],
    ) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        for (ordinal, message_id) in message_ids.iter().enumerate() {
            sqlx::query(
                r#"
INSERT OR IGNORE INTO lcm_summary_messages (summary_id, message_id, ordinal)
VALUES (?, ?, ?)
                "#,
            )
            .bind(summary_id)
            .bind(*message_id)
            .bind(i64::try_from(ordinal)?)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn lcm_link_summary_to_parents(
        &self,
        summary_id: &str,
        parent_summary_ids: &[String],
    ) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        for (ordinal, parent_summary_id) in parent_summary_ids.iter().enumerate() {
            sqlx::query(
                r#"
INSERT OR IGNORE INTO lcm_summary_parents (summary_id, parent_summary_id, ordinal)
VALUES (?, ?, ?)
                "#,
            )
            .bind(summary_id)
            .bind(parent_summary_id.as_str())
            .bind(i64::try_from(ordinal)?)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn lcm_get_summary_messages(&self, summary_id: &str) -> anyhow::Result<Vec<i64>> {
        sqlx::query_scalar::<_, i64>(
            r#"
SELECT message_id
FROM lcm_summary_messages
WHERE summary_id = ?
ORDER BY ordinal ASC
            "#,
        )
        .bind(summary_id)
        .fetch_all(self.pool.as_ref())
        .await
        .map_err(Into::into)
    }

    pub async fn lcm_get_summary_parents(
        &self,
        summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let parent_ids = sqlx::query_scalar::<_, String>(
            r#"
SELECT parent_summary_id
FROM lcm_summary_parents
WHERE summary_id = ?
ORDER BY ordinal ASC
            "#,
        )
        .bind(summary_id)
        .fetch_all(self.pool.as_ref())
        .await?;

        let mut parents = Vec::with_capacity(parent_ids.len());
        for parent_id in parent_ids {
            if let Some(parent) = self.lcm_get_summary(&parent_id).await? {
                parents.push(parent);
            }
        }
        Ok(parents)
    }

    pub async fn lcm_get_summary_children(
        &self,
        parent_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        let child_ids = sqlx::query_scalar::<_, String>(
            r#"
SELECT summary_id
FROM lcm_summary_parents
WHERE parent_summary_id = ?
ORDER BY ordinal ASC
            "#,
        )
        .bind(parent_summary_id)
        .fetch_all(self.pool.as_ref())
        .await?;

        let mut children = Vec::with_capacity(child_ids.len());
        for child_id in child_ids {
            if let Some(child) = self.lcm_get_summary(&child_id).await? {
                children.push(child);
            }
        }
        Ok(children)
    }

    pub async fn lcm_get_summary_subtree(
        &self,
        root_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummarySubtreeNodeRecord>> {
        let Some(root) = self.lcm_get_summary(root_summary_id).await? else {
            return Ok(Vec::new());
        };

        let mut output = Vec::new();
        let mut queue = VecDeque::from([(root.summary_id.clone(), 0i64, None, String::new())]);
        let mut seen = HashSet::new();

        while let Some((summary_id, depth_from_root, parent_summary_id, path)) = queue.pop_front() {
            if !seen.insert(summary_id.clone()) {
                continue;
            }

            let Some(summary) = self.lcm_get_summary(&summary_id).await? else {
                continue;
            };

            let child_rows = sqlx::query(
                r#"
SELECT summary_id, ordinal
FROM lcm_summary_parents
WHERE parent_summary_id = ?
ORDER BY ordinal ASC
                "#,
            )
            .bind(summary_id.as_str())
            .fetch_all(self.pool.as_ref())
            .await?;
            let child_count = i64::try_from(child_rows.len())?;

            output.push(LcmSummarySubtreeNodeRecord {
                summary,
                depth_from_root,
                parent_summary_id: parent_summary_id.clone(),
                path: path.clone(),
                child_count,
            });

            for child_row in child_rows {
                let child_summary_id: String = child_row.try_get("summary_id")?;
                let ordinal: i64 = child_row.try_get("ordinal")?;
                let child_path = if path.is_empty() {
                    format!("{ordinal:04}")
                } else {
                    format!("{path}.{ordinal:04}")
                };
                queue.push_back((
                    child_summary_id,
                    depth_from_root + 1,
                    Some(summary_id.clone()),
                    child_path,
                ));
            }
        }

        Ok(output)
    }

    pub async fn lcm_get_context_items(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmContextItemRecord>> {
        let rows = sqlx::query(
            r#"
SELECT
    thread_id,
    ordinal,
    item_type,
    message_id,
    summary_id,
    created_at
FROM lcm_context_items
WHERE thread_id = ?
ORDER BY ordinal ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(self.pool.as_ref())
        .await?;

        rows.into_iter()
            .map(|row| {
                Ok(LcmContextItemRecord {
                    thread_id: row.try_get("thread_id")?,
                    ordinal: row.try_get("ordinal")?,
                    item_type: LcmContextItemType::from_db(row.try_get::<&str, _>("item_type")?)?,
                    message_id: row.try_get("message_id")?,
                    summary_id: row.try_get("summary_id")?,
                    created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
                })
            })
            .collect()
    }

    pub async fn lcm_get_distinct_depths_in_context(
        &self,
        thread_id: &str,
        max_ordinal_exclusive: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        sqlx::query_scalar::<_, i64>(
            r#"
SELECT DISTINCT summaries.depth
FROM lcm_context_items context_items
JOIN lcm_summaries summaries
  ON summaries.summary_id = context_items.summary_id
WHERE context_items.thread_id = ?
  AND context_items.item_type = 'summary'
  AND (? IS NULL OR context_items.ordinal < ?)
ORDER BY summaries.depth ASC
            "#,
        )
        .bind(thread_id)
        .bind(max_ordinal_exclusive)
        .bind(max_ordinal_exclusive)
        .fetch_all(self.pool.as_ref())
        .await
        .map_err(Into::into)
    }

    pub async fn lcm_append_context_message(
        &self,
        thread_id: &str,
        message_id: i64,
    ) -> anyhow::Result<()> {
        let next_ordinal = sqlx::query_scalar::<_, Option<i64>>(
            r#"
SELECT MAX(ordinal) FROM lcm_context_items WHERE thread_id = ?
            "#,
        )
        .bind(thread_id)
        .fetch_one(self.pool.as_ref())
        .await?
        .unwrap_or(-1)
            + 1;

        sqlx::query(
            r#"
INSERT INTO lcm_context_items (
    thread_id,
    ordinal,
    item_type,
    message_id,
    summary_id,
    created_at
)
VALUES (?, ?, 'message', ?, NULL, ?)
            "#,
        )
        .bind(thread_id)
        .bind(next_ordinal)
        .bind(message_id)
        .bind(Utc::now().timestamp())
        .execute(self.pool.as_ref())
        .await?;
        Ok(())
    }

    pub async fn lcm_append_context_summary(
        &self,
        thread_id: &str,
        summary_id: &str,
    ) -> anyhow::Result<()> {
        let next_ordinal = sqlx::query_scalar::<_, Option<i64>>(
            r#"
SELECT MAX(ordinal) FROM lcm_context_items WHERE thread_id = ?
            "#,
        )
        .bind(thread_id)
        .fetch_one(self.pool.as_ref())
        .await?
        .unwrap_or(-1)
            + 1;

        sqlx::query(
            r#"
INSERT INTO lcm_context_items (
    thread_id,
    ordinal,
    item_type,
    message_id,
    summary_id,
    created_at
)
VALUES (?, ?, 'summary', NULL, ?, ?)
            "#,
        )
        .bind(thread_id)
        .bind(next_ordinal)
        .bind(summary_id)
        .bind(Utc::now().timestamp())
        .execute(self.pool.as_ref())
        .await?;
        Ok(())
    }

    pub async fn lcm_replace_context_range_with_summary(
        &self,
        params: LcmReplaceContextRangeParams,
    ) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
DELETE FROM lcm_context_items
WHERE thread_id = ?
  AND ordinal >= ?
  AND ordinal <= ?
            "#,
        )
        .bind(params.thread_id.as_str())
        .bind(params.start_ordinal)
        .bind(params.end_ordinal)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            r#"
INSERT INTO lcm_context_items (
    thread_id,
    ordinal,
    item_type,
    message_id,
    summary_id,
    created_at
)
VALUES (?, ?, 'summary', NULL, ?, ?)
            "#,
        )
        .bind(params.thread_id.as_str())
        .bind(params.start_ordinal)
        .bind(params.summary_id.as_str())
        .bind(Utc::now().timestamp())
        .execute(&mut *tx)
        .await?;

        let rows = sqlx::query(
            r#"
SELECT rowid
FROM lcm_context_items
WHERE thread_id = ?
ORDER BY ordinal ASC
            "#,
        )
        .bind(params.thread_id.as_str())
        .fetch_all(&mut *tx)
        .await?;

        for (new_ordinal, row) in rows.into_iter().enumerate() {
            let row_id: i64 = row.try_get("rowid")?;
            sqlx::query(
                r#"
UPDATE lcm_context_items
SET ordinal = ?
WHERE rowid = ?
                "#,
            )
            .bind(i64::try_from(new_ordinal)?)
            .bind(row_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn lcm_get_context_token_count(&self, thread_id: &str) -> anyhow::Result<i64> {
        let items = self.lcm_get_context_items(thread_id).await?;
        let mut total = 0i64;

        for item in items {
            match item.item_type {
                LcmContextItemType::Message => {
                    if let Some(message_id) = item.message_id
                        && let Some(message) = self.lcm_get_message_by_id(message_id).await?
                    {
                        total = total.saturating_add(message.token_count.max(0));
                    }
                }
                LcmContextItemType::Summary => {
                    if let Some(summary_id) = item.summary_id
                        && let Some(summary) = self.lcm_get_summary(&summary_id).await?
                    {
                        total = total.saturating_add(summary.token_count.max(0));
                    }
                }
            }
        }

        Ok(total)
    }

    pub async fn lcm_insert_large_file(
        &self,
        params: LcmCreateLargeFileParams,
    ) -> anyhow::Result<LcmLargeFileRecord> {
        let created_at = Utc::now().timestamp();
        sqlx::query(
            r#"
INSERT INTO lcm_large_files (
    file_id,
    thread_id,
    file_name,
    mime_type,
    byte_size,
    storage_uri,
    exploration_summary,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(params.file_id.as_str())
        .bind(params.thread_id.as_str())
        .bind(params.file_name.as_deref())
        .bind(params.mime_type.as_deref())
        .bind(params.byte_size)
        .bind(params.storage_uri.as_str())
        .bind(params.exploration_summary.as_deref())
        .bind(created_at)
        .execute(self.pool.as_ref())
        .await?;

        self.lcm_get_large_file(params.file_id.as_str())
            .await?
            .ok_or_else(|| anyhow::anyhow!("failed to fetch inserted LCM large file"))
    }

    pub async fn lcm_get_large_file(
        &self,
        file_id: &str,
    ) -> anyhow::Result<Option<LcmLargeFileRecord>> {
        let row = sqlx::query(
            r#"
SELECT
    file_id,
    thread_id,
    file_name,
    mime_type,
    byte_size,
    storage_uri,
    exploration_summary,
    created_at
FROM lcm_large_files
WHERE file_id = ?
            "#,
        )
        .bind(file_id)
        .fetch_optional(self.pool.as_ref())
        .await?;

        row.map(|row| {
            Ok(LcmLargeFileRecord {
                file_id: row.try_get("file_id")?,
                thread_id: row.try_get("thread_id")?,
                file_name: row.try_get("file_name")?,
                mime_type: row.try_get("mime_type")?,
                byte_size: row.try_get("byte_size")?,
                storage_uri: row.try_get("storage_uri")?,
                exploration_summary: row.try_get("exploration_summary")?,
                created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
            })
        })
        .transpose()
    }

    pub async fn lcm_get_large_files_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmLargeFileRecord>> {
        let rows = sqlx::query(
            r#"
SELECT
    file_id,
    thread_id,
    file_name,
    mime_type,
    byte_size,
    storage_uri,
    exploration_summary,
    created_at
FROM lcm_large_files
WHERE thread_id = ?
ORDER BY created_at ASC, file_id ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(self.pool.as_ref())
        .await?;

        rows.into_iter()
            .map(|row| {
                Ok(LcmLargeFileRecord {
                    file_id: row.try_get("file_id")?,
                    thread_id: row.try_get("thread_id")?,
                    file_name: row.try_get("file_name")?,
                    mime_type: row.try_get("mime_type")?,
                    byte_size: row.try_get("byte_size")?,
                    storage_uri: row.try_get("storage_uri")?,
                    exploration_summary: row.try_get("exploration_summary")?,
                    created_at: epoch_seconds_to_datetime(row.try_get("created_at")?)?,
                })
            })
            .collect()
    }
}

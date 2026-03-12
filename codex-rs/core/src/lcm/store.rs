use crate::state_db::StateDbHandle;
use async_trait::async_trait;
use codex_state::LcmContextItemRecord;
use codex_state::LcmCreateLargeFileParams;
use codex_state::LcmCreateMessageParams;
use codex_state::LcmCreateSummaryParams;
use codex_state::LcmLargeFileRecord;
use codex_state::LcmMessageRecord;
use codex_state::LcmReplaceContextRangeParams;
use codex_state::LcmSummaryRecord;
use codex_state::LcmSummarySubtreeNodeRecord;

#[async_trait]
pub(crate) trait LcmStore: Send + Sync {
    async fn create_message(
        &self,
        params: LcmCreateMessageParams,
    ) -> anyhow::Result<LcmMessageRecord>;

    async fn get_message_by_id(&self, message_id: i64) -> anyhow::Result<Option<LcmMessageRecord>>;

    async fn get_messages(
        &self,
        thread_id: &str,
        after_seq: Option<i64>,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<LcmMessageRecord>>;

    async fn get_message_count(&self, thread_id: &str) -> anyhow::Result<usize>;

    async fn get_max_seq(&self, thread_id: &str) -> anyhow::Result<i64>;

    async fn insert_summary(
        &self,
        params: LcmCreateSummaryParams,
    ) -> anyhow::Result<LcmSummaryRecord>;

    async fn get_summary(&self, summary_id: &str) -> anyhow::Result<Option<LcmSummaryRecord>>;

    async fn get_summaries_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>>;

    async fn link_summary_to_messages(
        &self,
        summary_id: &str,
        message_ids: &[i64],
    ) -> anyhow::Result<()>;

    async fn link_summary_to_parents(
        &self,
        summary_id: &str,
        parent_summary_ids: &[String],
    ) -> anyhow::Result<()>;

    async fn get_summary_messages(&self, summary_id: &str) -> anyhow::Result<Vec<i64>>;

    async fn get_summary_parents(&self, summary_id: &str) -> anyhow::Result<Vec<LcmSummaryRecord>>;

    async fn get_summary_children(
        &self,
        parent_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>>;

    async fn get_summary_subtree(
        &self,
        root_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummarySubtreeNodeRecord>>;

    async fn get_context_items(&self, thread_id: &str)
    -> anyhow::Result<Vec<LcmContextItemRecord>>;

    async fn get_distinct_depths_in_context(
        &self,
        thread_id: &str,
        max_ordinal_exclusive: Option<i64>,
    ) -> anyhow::Result<Vec<i64>>;

    async fn append_context_message(&self, thread_id: &str, message_id: i64) -> anyhow::Result<()>;

    async fn append_context_summary(&self, thread_id: &str, summary_id: &str)
    -> anyhow::Result<()>;

    async fn replace_context_range_with_summary(
        &self,
        params: LcmReplaceContextRangeParams,
    ) -> anyhow::Result<()>;

    async fn get_context_token_count(&self, thread_id: &str) -> anyhow::Result<i64>;

    async fn insert_large_file(
        &self,
        params: LcmCreateLargeFileParams,
    ) -> anyhow::Result<LcmLargeFileRecord>;

    async fn get_large_file(&self, file_id: &str) -> anyhow::Result<Option<LcmLargeFileRecord>>;

    async fn get_large_files_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmLargeFileRecord>>;
}

#[derive(Clone)]
pub(crate) struct StateDbLcmStore {
    state_db: StateDbHandle,
}

impl StateDbLcmStore {
    pub(crate) fn new(state_db: StateDbHandle) -> Self {
        Self { state_db }
    }
}

#[async_trait]
impl LcmStore for StateDbLcmStore {
    async fn create_message(
        &self,
        params: LcmCreateMessageParams,
    ) -> anyhow::Result<LcmMessageRecord> {
        self.state_db.lcm_create_message(params).await
    }

    async fn get_message_by_id(&self, message_id: i64) -> anyhow::Result<Option<LcmMessageRecord>> {
        self.state_db.lcm_get_message_by_id(message_id).await
    }

    async fn get_messages(
        &self,
        thread_id: &str,
        after_seq: Option<i64>,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<LcmMessageRecord>> {
        self.state_db
            .lcm_get_messages(thread_id, after_seq, limit)
            .await
    }

    async fn get_message_count(&self, thread_id: &str) -> anyhow::Result<usize> {
        self.state_db.lcm_get_message_count(thread_id).await
    }

    async fn get_max_seq(&self, thread_id: &str) -> anyhow::Result<i64> {
        self.state_db.lcm_get_max_seq(thread_id).await
    }

    async fn insert_summary(
        &self,
        params: LcmCreateSummaryParams,
    ) -> anyhow::Result<LcmSummaryRecord> {
        self.state_db.lcm_insert_summary(params).await
    }

    async fn get_summary(&self, summary_id: &str) -> anyhow::Result<Option<LcmSummaryRecord>> {
        self.state_db.lcm_get_summary(summary_id).await
    }

    async fn get_summaries_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        self.state_db.lcm_get_summaries_by_thread(thread_id).await
    }

    async fn link_summary_to_messages(
        &self,
        summary_id: &str,
        message_ids: &[i64],
    ) -> anyhow::Result<()> {
        self.state_db
            .lcm_link_summary_to_messages(summary_id, message_ids)
            .await
    }

    async fn link_summary_to_parents(
        &self,
        summary_id: &str,
        parent_summary_ids: &[String],
    ) -> anyhow::Result<()> {
        self.state_db
            .lcm_link_summary_to_parents(summary_id, parent_summary_ids)
            .await
    }

    async fn get_summary_messages(&self, summary_id: &str) -> anyhow::Result<Vec<i64>> {
        self.state_db.lcm_get_summary_messages(summary_id).await
    }

    async fn get_summary_parents(&self, summary_id: &str) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        self.state_db.lcm_get_summary_parents(summary_id).await
    }

    async fn get_summary_children(
        &self,
        parent_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummaryRecord>> {
        self.state_db
            .lcm_get_summary_children(parent_summary_id)
            .await
    }

    async fn get_summary_subtree(
        &self,
        root_summary_id: &str,
    ) -> anyhow::Result<Vec<LcmSummarySubtreeNodeRecord>> {
        self.state_db.lcm_get_summary_subtree(root_summary_id).await
    }

    async fn get_context_items(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmContextItemRecord>> {
        self.state_db.lcm_get_context_items(thread_id).await
    }

    async fn get_distinct_depths_in_context(
        &self,
        thread_id: &str,
        max_ordinal_exclusive: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        self.state_db
            .lcm_get_distinct_depths_in_context(thread_id, max_ordinal_exclusive)
            .await
    }

    async fn append_context_message(&self, thread_id: &str, message_id: i64) -> anyhow::Result<()> {
        self.state_db
            .lcm_append_context_message(thread_id, message_id)
            .await
    }

    async fn append_context_summary(
        &self,
        thread_id: &str,
        summary_id: &str,
    ) -> anyhow::Result<()> {
        self.state_db
            .lcm_append_context_summary(thread_id, summary_id)
            .await
    }

    async fn replace_context_range_with_summary(
        &self,
        params: LcmReplaceContextRangeParams,
    ) -> anyhow::Result<()> {
        self.state_db
            .lcm_replace_context_range_with_summary(params)
            .await
    }

    async fn get_context_token_count(&self, thread_id: &str) -> anyhow::Result<i64> {
        self.state_db.lcm_get_context_token_count(thread_id).await
    }

    async fn insert_large_file(
        &self,
        params: LcmCreateLargeFileParams,
    ) -> anyhow::Result<LcmLargeFileRecord> {
        self.state_db.lcm_insert_large_file(params).await
    }

    async fn get_large_file(&self, file_id: &str) -> anyhow::Result<Option<LcmLargeFileRecord>> {
        self.state_db.lcm_get_large_file(file_id).await
    }

    async fn get_large_files_by_thread(
        &self,
        thread_id: &str,
    ) -> anyhow::Result<Vec<LcmLargeFileRecord>> {
        self.state_db.lcm_get_large_files_by_thread(thread_id).await
    }
}

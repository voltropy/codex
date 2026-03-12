use chrono::DateTime;
use chrono::Utc;
use codex_protocol::models::ResponseItem;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LcmMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl LcmMessageRole {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    pub(crate) fn from_db(value: &str) -> anyhow::Result<Self> {
        match value {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            other => Err(anyhow::anyhow!("unknown LCM message role: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LcmSummaryKind {
    Leaf,
    Condensed,
}

impl LcmSummaryKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Condensed => "condensed",
        }
    }

    pub(crate) fn from_db(value: &str) -> anyhow::Result<Self> {
        match value {
            "leaf" => Ok(Self::Leaf),
            "condensed" => Ok(Self::Condensed),
            other => Err(anyhow::anyhow!("unknown LCM summary kind: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LcmContextItemType {
    Message,
    Summary,
}

impl LcmContextItemType {
    pub(crate) fn from_db(value: &str) -> anyhow::Result<Self> {
        match value {
            "message" => Ok(Self::Message),
            "summary" => Ok(Self::Summary),
            other => Err(anyhow::anyhow!("unknown LCM context item type: {other}")),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmCreateMessageParams {
    pub thread_id: String,
    pub seq: i64,
    pub role: LcmMessageRole,
    pub content: String,
    pub raw_item: ResponseItem,
    pub token_count: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmMessageRecord {
    pub message_id: i64,
    pub thread_id: String,
    pub seq: i64,
    pub role: LcmMessageRole,
    pub content: String,
    pub raw_item: ResponseItem,
    pub token_count: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmCreateSummaryParams {
    pub summary_id: String,
    pub thread_id: String,
    pub kind: LcmSummaryKind,
    pub depth: i64,
    pub content: String,
    pub token_count: i64,
    pub file_ids: Vec<String>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmSummaryRecord {
    pub summary_id: String,
    pub thread_id: String,
    pub kind: LcmSummaryKind,
    pub depth: i64,
    pub content: String,
    pub token_count: i64,
    pub file_ids: Vec<String>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmSummarySubtreeNodeRecord {
    pub summary: LcmSummaryRecord,
    pub depth_from_root: i64,
    pub parent_summary_id: Option<String>,
    pub path: String,
    pub child_count: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmContextItemRecord {
    pub thread_id: String,
    pub ordinal: i64,
    pub item_type: LcmContextItemType,
    pub message_id: Option<i64>,
    pub summary_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmReplaceContextRangeParams {
    pub thread_id: String,
    pub start_ordinal: i64,
    pub end_ordinal: i64,
    pub summary_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmCreateLargeFileParams {
    pub file_id: String,
    pub thread_id: String,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LcmLargeFileRecord {
    pub file_id: String,
    pub thread_id: String,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
    pub created_at: DateTime<Utc>,
}

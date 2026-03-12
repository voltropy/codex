use async_trait::async_trait;
use chrono::DateTime;
use chrono::Utc;
use codex_state::LcmSummaryKind;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::lcm::{
    DescribeItem, DescribeResult, ExpandInput, ExpandResult, GrepInput, GrepMode, GrepResult, GrepScope,
    RetrievalEngine, StateDbLcmStore, format_timestamp,
};
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LcmGrepMode {
    Regex,
    FullText,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LcmGrepScope {
    Messages,
    Summaries,
    Both,
}

#[derive(Deserialize)]
struct LcmGrepArgs {
    query: String,
    #[serde(default)]
    mode: Option<LcmGrepMode>,
    #[serde(default)]
    scope: Option<LcmGrepScope>,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    before: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct LcmDescribeArgs {
    id: String,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    token_cap: Option<i64>,
}

#[derive(Deserialize)]
struct LcmExpandArgs {
    #[serde(default)]
    summary_ids: Option<Vec<String>>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    max_depth: Option<usize>,
    #[serde(default)]
    token_cap: Option<i64>,
    #[serde(default)]
    include_messages: Option<bool>,
    #[serde(default)]
    thread_id: Option<String>,
}

impl From<LcmGrepMode> for GrepMode {
    fn from(value: LcmGrepMode) -> Self {
        match value {
            LcmGrepMode::Regex => Self::Regex,
            LcmGrepMode::FullText => Self::FullText,
        }
    }
}

impl From<LcmGrepMode> for &str {
    fn from(value: LcmGrepMode) -> Self {
        match value {
            LcmGrepMode::Regex => "regex",
            LcmGrepMode::FullText => "full_text",
        }
    }
}

impl From<LcmGrepScope> for GrepScope {
    fn from(value: LcmGrepScope) -> Self {
        match value {
            LcmGrepScope::Messages => Self::Messages,
            LcmGrepScope::Summaries => Self::Summaries,
            LcmGrepScope::Both => Self::Both,
        }
    }
}

impl From<LcmGrepScope> for &str {
    fn from(value: LcmGrepScope) -> Self {
        match value {
            LcmGrepScope::Messages => "messages",
            LcmGrepScope::Summaries => "summaries",
            LcmGrepScope::Both => "both",
        }
    }
}

fn resolve_thread_id(session_id: &str, requested: Option<String>) -> String {
    requested.unwrap_or_else(|| session_id.to_string())
}

fn parse_iso_timestamp(
    value: &Option<String>,
    label: &str,
) -> Result<Option<DateTime<Utc>>, FunctionCallError> {
    let Some(value) = value.as_deref().map(str::trim) else {
        return Ok(None);
    };

    if value.is_empty() {
        return Ok(None);
    }

    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| Some(timestamp.with_timezone(&Utc)))
        .map_err(|err| {
            FunctionCallError::RespondToModel(format!("failed to parse {label}: {err}"))
        })
}

fn parse_token_cap(value: Option<i64>) -> Result<i64, FunctionCallError> {
    let Some(value) = value else {
        return Ok(i64::MAX);
    };
    if value <= 0 {
        return Err(FunctionCallError::RespondToModel(
            "token_cap must be greater than zero".to_string(),
        ));
    }
    Ok(value)
}

fn format_message_id(message_id: i64) -> String {
    format!("msg_{message_id}")
}

fn format_summary_kind(value: LcmSummaryKind) -> &'static str {
    match value {
        LcmSummaryKind::Leaf => "leaf",
        LcmSummaryKind::Condensed => "condensed",
    }
}

fn validate_id_prefix<'a>(value: &'a str, prefix: &str) -> Result<&'a str, FunctionCallError> {
    if value.starts_with(prefix) {
        Ok(value)
    } else {
        Err(FunctionCallError::RespondToModel(format!(
            "{prefix} id required; got {value}"
        )))
    }
}

pub struct LcmGrepHandler;

#[async_trait]
impl ToolHandler for LcmGrepHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation {
            session,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "lcm_grep handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: LcmGrepArgs = parse_arguments(&arguments)?;
        let query = args.query.trim().to_string();
        if query.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "query must not be empty".to_string(),
            ));
        }

        let Some(state_db) = session.state_db() else {
            return Err(FunctionCallError::RespondToModel(
                "LCM data is unavailable for this session".to_string(),
            ));
        };

        let store = StateDbLcmStore::new(state_db);
        let retrieval = RetrievalEngine::new(&store);
        let mode = args.mode.unwrap_or(LcmGrepMode::Regex);
        let scope = args.scope.unwrap_or(LcmGrepScope::Both);

        let since = parse_iso_timestamp(&args.since, "since")?;
        let before = parse_iso_timestamp(&args.before, "before")?;
        if let (Some(since), Some(before)) = (since, before)
            && since >= before
        {
            return Err(FunctionCallError::RespondToModel(
                "since must be before before".to_string(),
            ));
        }

        let limit = args.limit.filter(|value| *value > 0).unwrap_or(50);
        if limit > 200 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be between 1 and 200".to_string(),
            ));
        }

        let conversation_id = session.conversation_id.to_string();
        let thread_id = resolve_thread_id(&conversation_id, args.thread_id);
        let result: GrepResult = retrieval
            .grep(GrepInput {
                query,
                mode: mode.into(),
                scope: scope.into(),
                thread_id: Some(thread_id.clone()),
                since,
                before,
                limit: Some(limit),
            })
            .await
            .map_err(|err| {
                FunctionCallError::RespondToModel(format!("LCM grep failed: {err}"))
            })?;

        let mut output = vec![
            "## LCM Grep Results".to_string(),
            format!("thread_id={thread_id}"),
            format!("mode={}", <LcmGrepMode as Into<&str>>::into(mode)),
            format!("scope={}", <LcmGrepScope as Into<&str>>::into(scope)),
            format!("total_matches={}", result.total_matches),
        ];

        if !result.messages.is_empty() {
            output.push("\nMessages:".to_string());
            for entry in result.messages {
                output.push(format!(
                    "- {} time={} snippet={}",
                    format_message_id(entry.message_id),
                    format_timestamp(entry.created_at),
                    entry.snippet,
                ));
            }
        }

        if !result.summaries.is_empty() {
            output.push("\nSummaries:".to_string());
            for entry in result.summaries {
                output.push(format!(
                    "- {} kind={} time={} snippet={}",
                    entry.summary_id,
                    format_summary_kind(entry.kind),
                    format_timestamp(entry.created_at),
                    entry.snippet,
                ));
            }
        }

        if output.len() == 5 {
            output.push("No matches found.".to_string());
        }

        let content = output.join("\n");
        Ok(FunctionToolOutput::from_text(content, Some(true)))
    }
}

pub struct LcmDescribeHandler;

#[async_trait]
impl ToolHandler for LcmDescribeHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation {
            session,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "lcm_describe handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: LcmDescribeArgs = parse_arguments(&arguments)?;
        let id = args.id.trim().to_string();
        if id.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "id must not be empty".to_string(),
            ));
        }

        let Some(state_db) = session.state_db() else {
            return Err(FunctionCallError::RespondToModel(
                "LCM data is unavailable for this session".to_string(),
            ));
        };

        let store = StateDbLcmStore::new(state_db);
        let retrieval = RetrievalEngine::new(&store);

        let describe: Option<DescribeResult> = retrieval
            .describe(&id)
            .await
            .map_err(|err| {
                FunctionCallError::RespondToModel(format!("LCM describe failed: {err}"))
            })?;
        let Some(describe) = describe else {
            return Err(FunctionCallError::RespondToModel(format!("not found: {id}")));
        };

        if let Some(thread_id) = &args.thread_id {
            match &describe.item {
                DescribeItem::Summary(summary) if summary.thread_id != *thread_id => {
                    return Err(FunctionCallError::RespondToModel(format!(
                        "not found in requested thread {thread_id}: {id}"
                    )));
                }
                DescribeItem::File(file) if file.thread_id != *thread_id => {
                    return Err(FunctionCallError::RespondToModel(format!(
                        "not found in requested thread {thread_id}: {id}"
                    )));
                }
                _ => {}
            }
        }

                let mut output = Vec::new();
        let token_cap = parse_token_cap(args.token_cap)?;
        match describe.item {
            DescribeItem::Summary(summary) => {
                output.push(format!("LCM_SUMMARY {}", describe.id));
                output.push(format!(
                    "thread={} kind={} depth={} tokens={} descendants={} desc_tokens={} source_tokens={}",
                    summary.thread_id,
                    format_summary_kind(summary.kind),
                    summary.depth,
                    summary.token_count,
                    summary.descendant_count,
                    summary.descendant_token_count,
                    summary.source_message_token_count,
                ));
                output.push(format!(
                    "time_range={}..{}",
                    summary
                        .earliest_at
                        .map(|value| format_timestamp(value))
                        .unwrap_or_else(|| "-".to_string()),
                    summary
                        .latest_at
                        .map(|value| format_timestamp(value))
                        .unwrap_or_else(|| "-".to_string())
                ));
                if !summary.parent_ids.is_empty() {
                    output.push(format!("parents: {}", summary.parent_ids.join(", ")));
                }
                if !summary.child_ids.is_empty() {
                    output.push(format!("children: {}", summary.child_ids.join(", ")));
                }
                if !summary.file_ids.is_empty() {
                    output.push(format!("files: {}", summary.file_ids.join(", ")));
                }
                if !summary.message_ids.is_empty() {
                    output.push(format!(
                        "linked_messages: {}",
                        summary
                            .message_ids
                            .iter()
                            .map(|message_id| format_message_id(*message_id))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                if !summary.subtree.is_empty() {
                    output.push("\nManifest:".to_string());
                    for node in summary.subtree {
                        let summaries_only = node
                            .token_count
                            .saturating_add(node.descendant_token_count)
                            .max(0);
                        let with_messages = summaries_only
                            .saturating_add(node.source_message_token_count)
                            .max(0);
                        output.push(format!(
                            "- {} kind={} token={}/{} depth={} desc_count={} child_count={} path={}",
                            node.summary_id,
                            format_summary_kind(node.kind),
                            summaries_only,
                            with_messages,
                            node.depth,
                            node.descendant_count,
                            node.child_count,
                            node.path,
                        ));
                        if summaries_only > token_cap || with_messages > token_cap {
                            output.push(format!(
                                "  - budget_fit: summaries_only={}, with_messages={}",
                                summaries_only <= token_cap,
                                with_messages <= token_cap
                            ));
                        }
                    }
                }
                output.push("\nContent:".to_string());
                output.push(summary.content);
            }
            DescribeItem::File(file) => {
                validate_id_prefix(&id, "file_")?;
                output.push(format!("LCM_FILE {}", describe.id));
                output.push(format!("thread={}", file.thread_id));
                output.push(format!(
                    "name={} mime={} size={}",
                    file.file_name.unwrap_or_else(|| "(unknown)".to_string()),
                    file.mime_type.unwrap_or_else(|| "(unknown)".to_string()),
                    file.byte_size
                        .map(|value: i64| value.to_string())
                        .unwrap_or_else(|| "-".to_string())
                ));
                output.push(format!("storage_uri={} ", file.storage_uri));
                output.push(format!("created_at={}", format_timestamp(file.created_at)));
                output.push("\nExploration summary:".to_string());
                if let Some(summary) = file.exploration_summary {
                    output.push(summary);
                } else {
                    output.push("(none)".to_string());
                }
            }
        }

        output.push(format!("budget_cap={token_cap}"));

        let content = output.join("\n");
        Ok(FunctionToolOutput::from_text(content, Some(true)))
    }
}

pub struct LcmExpandHandler;

#[async_trait]
impl ToolHandler for LcmExpandHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation {
            session,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "lcm_expand handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: LcmExpandArgs = parse_arguments(&arguments)?;
        let conversation_id = session.conversation_id.to_string();
        let thread_id = resolve_thread_id(&conversation_id, args.thread_id);

        let Some(state_db) = session.state_db() else {
            return Err(FunctionCallError::RespondToModel(
                "LCM data is unavailable for this session".to_string(),
            ));
        };

        let store = StateDbLcmStore::new(state_db);
        let retrieval = RetrievalEngine::new(&store);

        let include_messages = args.include_messages.unwrap_or(false);
        let depth = args.max_depth.unwrap_or(3);
        if depth == 0 {
            return Err(FunctionCallError::RespondToModel(
                "max_depth must be greater than zero".to_string(),
            ));
        }

        let token_cap = parse_token_cap(args.token_cap)?;

        let summary_ids: Vec<String> = if let Some(query) = args.query {
            let query = query.trim().to_string();
            if query.is_empty() {
                return Err(FunctionCallError::RespondToModel(
                    "query must not be empty".to_string(),
                ));
            }
            let grep: GrepResult = retrieval
                .grep(GrepInput {
                    query,
                    mode: GrepMode::FullText,
                    scope: GrepScope::Summaries,
                    thread_id: Some(thread_id.clone()),
                    since: None,
                    before: None,
                    limit: Some(10),
                })
                .await
                .map_err(|err| {
                    FunctionCallError::RespondToModel(format!("LCM expand grep failed: {err}"))
                })?;

            let ids: Vec<String> = grep
                .summaries
                .into_iter()
                .map(|entry| entry.summary_id)
                .collect::<Vec<_>>();
            if ids.is_empty() {
                return Err(FunctionCallError::RespondToModel(
                    "no summaries found for query".to_string(),
                ));
            }
            ids
        } else {
            match args.summary_ids {
                Some(summary_ids) if !summary_ids.is_empty() => summary_ids,
                Some(_) => {
                    return Err(FunctionCallError::RespondToModel(
                        "summary_ids must not be empty".to_string(),
                    ));
                }
                None => {
                    return Err(FunctionCallError::RespondToModel(
                        "either summary_ids or query must be provided".to_string(),
                    ));
                }
            }
        };

        let mut output = vec!["## LCM Expansion".to_string(), format!("thread_id={thread_id}")];

        for summary_id in summary_ids {
            validate_id_prefix(&summary_id, "sum_")?;
            let result: ExpandResult = retrieval
                .expand(ExpandInput {
                    summary_id: summary_id.clone(),
                    depth,
                    include_messages,
                    token_cap,
                })
                .await
                .map_err(|err| {
                    FunctionCallError::RespondToModel(format!("LCM expand failed: {err}"))
                })?;
            output.push(format!(
                "\nExpanded {summary_id}: children={}, messages={}, tokens={}, truncated={}",
                result.children.len(),
                result.messages.len(),
                result.estimated_tokens,
                result.truncated
            ));
            for child in result.children {
                    output.push(format!(
                        "- child {} kind={} tokens={} content={}",
                        child.summary_id,
                        format_summary_kind(child.kind),
                        child.token_count,
                        child.content
                    ));
            }
            if include_messages {
                for message in result.messages {
                    output.push(format!(
                        "- {} ({}): {}",
                        format_message_id(message.message_id),
                        message.role,
                        message.content
                    ));
                }
            }
        }

        let content = output.join("\n");
        Ok(FunctionToolOutput::from_text(content, Some(true)))
    }
}

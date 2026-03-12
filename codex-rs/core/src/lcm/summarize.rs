use super::compaction::LcmSummarizeFn;
use super::estimate_tokens;
use crate::Prompt;
use crate::client_common::ResponseEvent;
use crate::codex::Session;
use crate::codex::TurnContext;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use std::sync::Arc;

const DEFAULT_CONDENSED_TARGET_TOKENS: i64 = 2_000;
const LCM_SUMMARIZER_SYSTEM_PROMPT: &str = "You are a context-compaction summarization engine. Follow user instructions exactly and return plain text summary content only.";

pub(crate) fn build_deterministic_fallback_summary(text: &str, target_tokens: i64) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let max_chars = usize::try_from(target_tokens.max(64)).unwrap_or(usize::MAX / 4) * 4;
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    format!(
        "{}\n[LCM fallback summary; truncated for context management]",
        &trimmed[..max_chars]
    )
}

pub(crate) fn build_lcm_summarize_fn(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    condensed_target_tokens: Option<i64>,
) -> Arc<LcmSummarizeFn> {
    let condensed_target_tokens =
        condensed_target_tokens.unwrap_or(DEFAULT_CONDENSED_TARGET_TOKENS);

    Arc::new(move |text, aggressive, options| {
        let sess = Arc::clone(&sess);
        let turn_context = Arc::clone(&turn_context);
        Box::pin(async move {
            if text.trim().is_empty() {
                return Ok(String::new());
            }

            let options = options.unwrap_or_default();
            let target_tokens = resolve_target_tokens(
                &text,
                aggressive,
                options.is_condensed,
                condensed_target_tokens,
            );
            let prompt_text = if options.is_condensed {
                build_condensed_summary_prompt(
                    &text,
                    target_tokens,
                    options.depth.unwrap_or(1),
                    options.previous_summary.as_deref(),
                )
            } else {
                build_leaf_summary_prompt(
                    &text,
                    aggressive,
                    target_tokens,
                    options.previous_summary.as_deref(),
                )
            };

            let prompt = Prompt {
                input: vec![ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText { text: prompt_text }],
                    end_turn: None,
                    phase: None,
                }],
                tools: Vec::new(),
                parallel_tool_calls: false,
                base_instructions: BaseInstructions {
                    text: LCM_SUMMARIZER_SYSTEM_PROMPT.to_string(),
                },
                personality: None,
                output_schema: None,
            };

            let mut client_session = sess.services.model_client.new_session();
            let mut stream = client_session
                .stream(
                    &prompt,
                    &turn_context.model_info,
                    &turn_context.session_telemetry,
                    turn_context.reasoning_effort,
                    turn_context.reasoning_summary,
                    turn_context.config.service_tier,
                    None,
                )
                .await?;

            let mut result = String::new();
            while let Some(message) = stream.next().await.transpose()? {
                match message {
                    ResponseEvent::OutputTextDelta(delta) => result.push_str(&delta),
                    ResponseEvent::OutputItemDone(item) => {
                        if result.is_empty()
                            && let ResponseItem::Message { content, .. } = item
                            && let Some(text) = crate::compact::content_items_to_text(&content)
                        {
                            result.push_str(&text);
                        }
                    }
                    ResponseEvent::Completed { .. } => break,
                    _ => {}
                }
            }

            let result = result.trim().to_string();
            if result.is_empty() {
                Ok(build_deterministic_fallback_summary(&text, target_tokens))
            } else {
                Ok(result)
            }
        })
    })
}

fn resolve_target_tokens(
    input_text: &str,
    aggressive: bool,
    is_condensed: bool,
    condensed_target_tokens: i64,
) -> i64 {
    if is_condensed {
        return condensed_target_tokens.max(512);
    }

    let input_tokens = estimate_tokens(input_text);
    if aggressive {
        (input_tokens / 5).clamp(96, 640)
    } else {
        ((input_tokens * 35) / 100).clamp(192, 1_200)
    }
}

fn build_leaf_summary_prompt(
    text: &str,
    aggressive: bool,
    target_tokens: i64,
    previous_summary: Option<&str>,
) -> String {
    let policy = if aggressive {
        [
            "Aggressive summary policy:",
            "- Keep only durable facts and current task state.",
            "- Remove examples, repetition, and low-value narrative details.",
            "- Preserve explicit TODOs, blockers, decisions, and constraints.",
        ]
        .join("\n")
    } else {
        [
            "Normal summary policy:",
            "- Preserve key decisions, rationale, constraints, and active tasks.",
            "- Keep essential technical details needed to continue work safely.",
            "- Remove obvious repetition and conversational filler.",
        ]
        .join("\n")
    };
    let previous_summary = previous_summary.unwrap_or("(none)");

    [
        "You summarize a SEGMENT of a Codex conversation for future model turns.",
        "Treat this as incremental memory compaction input, not a full-conversation summary.",
        &policy,
        "Operator instructions: (none)",
        &[
            "Output requirements:",
            "- Plain text only.",
            "- No preamble, headings, or markdown formatting.",
            "- Keep it concise while preserving required details.",
            "- Track file operations (created, modified, deleted, renamed) with file paths and current status.",
            r#"- If no file operations appear, include exactly: "Files: none"."#,
            r#"- End with exactly: "Expand for details about: <comma-separated list of what was dropped or compressed>"."#,
            &format!("- Target length: about {target_tokens} tokens or less."),
        ]
        .join("\n"),
        &format!("<previous_context>\n{previous_summary}\n</previous_context>"),
        &format!("<conversation_segment>\n{text}\n</conversation_segment>"),
    ]
    .join("\n\n")
}

fn build_condensed_summary_prompt(
    text: &str,
    target_tokens: i64,
    depth: i64,
    previous_summary: Option<&str>,
) -> String {
    if depth <= 1 {
        return build_d1_prompt(text, target_tokens, previous_summary);
    }
    if depth == 2 {
        return build_d2_prompt(text, target_tokens);
    }
    build_d3_plus_prompt(text, target_tokens)
}

fn build_d1_prompt(text: &str, target_tokens: i64, previous_summary: Option<&str>) -> String {
    let previous_context_block = if let Some(previous_summary) = previous_summary {
        format!(
            "It already has this preceding summary as context. Do not repeat information\nthat appears there unchanged. Focus on what is new, changed, or resolved:\n\n<previous_context>\n{previous_summary}\n</previous_context>"
        )
    } else {
        "Focus on what matters for continuation:".to_string()
    };

    [
        "You are compacting leaf-level conversation summaries into a single condensed memory node.",
        "You are preparing context for a fresh model instance that will continue this conversation.",
        "Operator instructions: (none)",
        &previous_context_block,
        &[
            "Preserve:",
            "- Decisions made and their rationale when rationale matters going forward.",
            "- Earlier decisions that were superseded, and what replaced them.",
            "- Completed tasks/topics with outcomes.",
            "- In-progress items with current state and what remains.",
            "- Blockers, open questions, and unresolved tensions.",
            "- Specific references (names, paths, URLs, identifiers) needed for continuation.",
            "",
            "Drop low-value detail:",
            "- Context that has not changed from previous_context.",
            "- Intermediate dead ends where the conclusion is already known.",
            "- Transient states that are already resolved.",
            "- Tool-internal mechanics and process scaffolding.",
            "",
            "Use plain text. No mandatory structure.",
            "Include a timeline with timestamps (hour or half-hour) for significant events.",
            "Present information chronologically and mark superseded decisions.",
            r#"- End with exactly: "Expand for details about: <comma-separated list of what was dropped or compressed>"."#,
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        &format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

fn build_d2_prompt(text: &str, target_tokens: i64) -> String {
    [
        "You are condensing multiple session-level summaries into a higher-level memory node.",
        "A future model should understand trajectory, not per-session minutiae.",
        "Operator instructions: (none)",
        &[
            "Preserve:",
            "- Decisions still in effect and their rationale.",
            "- Decisions that evolved: what changed and why.",
            "- Completed work with outcomes.",
            "- Active constraints, limitations, and known issues.",
            "- Current state of in-progress work.",
            "",
            "Drop:",
            "- Session-local operational detail and process mechanics.",
            "- Identifiers that are no longer relevant.",
            "- Intermediate states superseded by later outcomes.",
            "",
            "Use plain text. Brief headers are fine if useful.",
            "Include a timeline with dates and approximate time of day for key milestones.",
            r#"- End with exactly: "Expand for details about: <comma-separated list of what was dropped or compressed>"."#,
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        &format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

fn build_d3_plus_prompt(text: &str, target_tokens: i64) -> String {
    [
        "You are creating a high-level memory node from multiple phase-level summaries.",
        "This may persist for the rest of the conversation. Keep only durable context.",
        "Operator instructions: (none)",
        &[
            "Preserve:",
            "- Key decisions and rationale.",
            "- What was accomplished and current state.",
            "- Active constraints and hard limitations.",
            "- Important relationships between people, systems, or concepts.",
            "- Durable lessons learned.",
            "",
            "Drop:",
            "- Operational and process detail.",
            "- Method details unless the method itself was the decision.",
            "- Specific references unless essential for continuation.",
            "",
            "Use plain text. Be concise.",
            "Include a brief timeline with dates (or date ranges) for major milestones.",
            r#"- End with exactly: "Expand for details about: <comma-separated list of what was dropped or compressed>"."#,
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        &format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

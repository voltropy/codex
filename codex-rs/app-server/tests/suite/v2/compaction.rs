//! Thread compaction request tests.

#![expect(clippy::expect_used)]

use anyhow::Result;
use app_test_support::McpProcess;
use app_test_support::to_response;
use app_test_support::write_mock_responses_config_toml;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadCompactStartParams;
use codex_app_server_protocol::ThreadCompactStartResponse;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;
use tempfile::TempDir;
use tokio::time::timeout;

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const AUTO_COMPACT_LIMIT: i64 = 1_000;
const COMPACT_PROMPT: &str = "Summarize the conversation.";
const INVALID_REQUEST_ERROR_CODE: i64 = -32600;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_compact_start_triggers_compaction_and_returns_empty_response() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;

    let codex_home = TempDir::new()?;
    write_mock_responses_config_toml(
        codex_home.path(),
        &server.uri(),
        &BTreeMap::default(),
        AUTO_COMPACT_LIMIT,
        None,
        "mock_provider",
        COMPACT_PROMPT,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let compact_id = mcp
        .send_thread_compact_start_request(ThreadCompactStartParams {
            thread_id: thread_id.clone(),
        })
        .await?;
    let compact_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(compact_id)),
    )
    .await??;
    let _compact: ThreadCompactStartResponse =
        to_response::<ThreadCompactStartResponse>(compact_resp)?;
    assert_eq!(thread_id.len(), 36);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_compact_start_rejects_invalid_thread_id() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let codex_home = TempDir::new()?;
    write_mock_responses_config_toml(
        codex_home.path(),
        &server.uri(),
        &BTreeMap::default(),
        AUTO_COMPACT_LIMIT,
        None,
        "mock_provider",
        COMPACT_PROMPT,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_compact_start_request(ThreadCompactStartParams {
            thread_id: "not-a-thread-id".to_string(),
        })
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;

    assert_eq!(error.error.code, INVALID_REQUEST_ERROR_CODE);
    assert!(error.error.message.contains("invalid thread id"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_compact_start_rejects_unknown_thread_id() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let codex_home = TempDir::new()?;
    write_mock_responses_config_toml(
        codex_home.path(),
        &server.uri(),
        &BTreeMap::default(),
        AUTO_COMPACT_LIMIT,
        None,
        "mock_provider",
        COMPACT_PROMPT,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_compact_start_request(ThreadCompactStartParams {
            thread_id: "67e55044-10b1-426f-9247-bb680e5fe0c8".to_string(),
        })
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;

    assert_eq!(error.error.code, INVALID_REQUEST_ERROR_CODE);
    assert!(error.error.message.contains("thread not found"));

    Ok(())
}

async fn start_thread(mcp: &mut McpProcess) -> Result<String> {
    let thread_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let thread_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(thread_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(thread_resp)?;
    Ok(thread.id)
}

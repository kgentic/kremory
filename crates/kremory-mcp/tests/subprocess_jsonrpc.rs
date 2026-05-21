//! D.5-e2e cycle-2 — subprocess JSON-RPC round-trip against the real
//! `kremory-mcp-server` binary.
//!
//! ## What this catches that `handler_roundtrip.rs` cannot
//!
//! `handler_roundtrip.rs` calls the rmcp `#[tool]`-annotated handler
//! methods directly with `Parameters(...)`. That bypasses the entire
//! transport stack — JSON-RPC framing, tool registration, capability
//! advertising, and (critically) the `tracing-subscriber` configuration
//! that determines whether logs leak onto stdout and corrupt the protocol
//! stream.
//!
//! This test spawns the actual built `kremory-mcp-server` binary, speaks
//! line-delimited JSON-RPC over its stdin/stdout, and asserts:
//!
//! 1. **stdout is clean JSON** — the very first byte of each line is `{`.
//!    If `FmtSubscriber::with_writer(stderr)` ever drifts back to stdout
//!    (its default), or someone adds a `println!` to the bin, this test
//!    fires with a loud "expected JSON, got <prefix>" message before the
//!    misframing reaches an aidocs MCP host in production.
//! 2. **`initialize` handshake** completes with a parseable result.
//! 3. **`tools/list` registers all four tools** under their canonical
//!    snake_case names — drift catch for tool-name renames.
//! 4. **`tools/call kremory_context_block`** returns a structured payload on
//!    the unbound server (this tool is pure — no graph required).
//! 5. **`tools/call kremory_search` on unbound server returns an error** with
//!    the right shape (graph-not-bound) — pins the unbound contract
//!    end-to-end through MCP framing.
//!
//! ## Why line-delimited JSON?
//!
//! rmcp 1.7's stdio transport uses `read_until(b'\n', ...)` (see
//! `rmcp::transport::async_rw`). Each JSON-RPC message is exactly one
//! newline-terminated line — no Content-Length headers. The test hand-
//! rolls that framing rather than pulling in rmcp's client; using the
//! rmcp client would self-roundtrip and miss the stdout-pollution class
//! of bug this test specifically exists to catch.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

/// Hard cap on every stdio read — a hung server must fail loudly, not
/// hang CI for ten minutes.
const STDIO_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Spawn the real binary. `env!("CARGO_BIN_EXE_kremory-mcp-server")` is
/// auto-defined by cargo for integration tests in the same package, and
/// causes cargo to build the bin before the test runs.
fn spawn_server() -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let path = env!("CARGO_BIN_EXE_kremory-mcp-server");
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // stderr inherits — tracing logs (info-level handshake messages,
        // tool dispatch warnings) appear in `cargo test --nocapture`
        // output without polluting the stdout JSON-RPC stream we read.
        .stderr(Stdio::inherit())
        // RUST_LOG=warn so the spawned binary stays quiet unless something
        // is wrong; warn-level errors still reach stderr for diagnosis.
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("kremory-mcp-server binary must spawn");
    let stdin = child.stdin.take().expect("stdin pipe");
    let stdout = BufReader::new(child.stdout.take().expect("stdout pipe"));
    (child, stdin, stdout)
}

/// Send one JSON-RPC frame (line-delimited JSON, `\n`-terminated).
async fn send(stdin: &mut ChildStdin, msg: &Value) {
    let mut payload = serde_json::to_vec(msg).expect("serialize jsonrpc");
    payload.push(b'\n');
    stdin
        .write_all(&payload)
        .await
        .expect("write request to server stdin");
    stdin.flush().await.expect("flush");
}

/// Read one JSON-RPC line and assert it parses as a JSON object. Panics
/// loudly with the offending bytes if the line is not JSON — that's the
/// stdout-pollution signal.
///
/// Blank lines are skipped (they can appear as framing artifacts after a
/// notification message); the assertion fires only when a non-empty line
/// fails the `{` start-byte check. This narrows the pollution detector
/// to actual content rather than empty framing newlines (review finding
/// MEDIUM, framing edge).
async fn recv(stdout: &mut BufReader<ChildStdout>) -> Value {
    loop {
        let mut line = String::new();
        let n = timeout(STDIO_READ_TIMEOUT, stdout.read_line(&mut line))
            .await
            .expect("server response must arrive within timeout")
            .expect("read from server stdout must succeed");
        assert!(
            n > 0,
            "server closed stdout before responding — likely panic at startup; check stderr"
        );
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            // Framing artifact — keep reading.
            continue;
        }
        assert!(
            trimmed.starts_with('{'),
            "stdout pollution detected: expected JSON-RPC frame starting with '{{', got: {trimmed:?}\n\
             (rmcp stdio framing requires line-delimited JSON. tracing or println! writing to \
             stdout instead of stderr will produce this failure — check \
             FmtSubscriber::with_writer in src/main.rs.)"
        );
        return serde_json::from_str(trimmed).unwrap_or_else(|e| {
            panic!("malformed JSON-RPC frame from server: {e}\nraw: {trimmed:?}")
        });
    }
}

/// Minimal MCP initialize → initialized handshake. Required before any
/// `tools/*` call; rmcp's server-side state machine rejects calls before
/// the handshake completes.
async fn initialize(stdin: &mut ChildStdin, stdout: &mut BufReader<ChildStdout>) {
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "kremory-mcp-e2e-test", "version": "0.0.1" }
        }
    });
    send(stdin, &init).await;
    let resp = recv(stdout).await;
    assert_eq!(resp["id"], json!(1), "initialize response id must echo");
    assert!(
        resp.get("result").is_some(),
        "initialize must return a result, got: {resp}"
    );
    let initialized = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    send(stdin, &initialized).await;
}

/// Graceful subprocess teardown: drop stdin to signal EOF, then await
/// child exit with a timeout so a hung server is caught by the test
/// harness rather than the OS timeout.
async fn shutdown(mut child: Child, stdin: ChildStdin) {
    drop(stdin); // EOF on stdin → rmcp loop returns → process exits.
    match timeout(Duration::from_secs(5), child.wait()).await {
        Ok(Ok(status)) => {
            // Server may exit clean (0) or with a transport-EOF error.
            // Either is acceptable for shutdown — we only flag panics
            // (signal-induced exits surface as != 0 on POSIX).
            assert!(
                status.success() || status.code().is_some(),
                "server exited via signal — likely panic. status: {status:?}"
            );
        }
        Ok(Err(e)) => panic!("waiting for server exit failed: {e}"),
        Err(_) => {
            child.start_kill().ok();
            panic!("server did not exit within 5s of stdin close — hang");
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn initialize_handshake_returns_clean_jsonrpc() {
    let (child, mut stdin, mut stdout) = spawn_server();
    initialize(&mut stdin, &mut stdout).await;
    shutdown(child, stdin).await;
}

#[tokio::test]
async fn tools_list_advertises_all_four_kremory_tools_under_canonical_names() {
    let (child, mut stdin, mut stdout) = spawn_server();
    initialize(&mut stdin, &mut stdout).await;

    let req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    send(&mut stdin, &req).await;
    let resp = recv(&mut stdout).await;
    assert_eq!(resp["id"], json!(2));
    let tools = resp["result"]["tools"]
        .as_array()
        .expect("tools/list must return tools array");
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![
            "kremory_context_block",
            "kremory_ingest_episode",
            "kremory_run_dream_phase",
            "kremory_search",
        ],
        "all four canonical kremory tools must be advertised; got: {names:?}"
    );

    shutdown(child, stdin).await;
}

#[tokio::test]
async fn tools_call_kremory_context_block_works_on_unbound_server() {
    // context_block is pure (template renderer) — the unbound binary
    // serves it without a graph. End-to-end: handshake → tools/call →
    // structured payload reaches the client.
    let (child, mut stdin, mut stdout) = spawn_server();
    initialize(&mut stdin, &mut stdout).await;

    let req = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "kremory_context_block",
            "arguments": {
                "results": [{
                    "entity_id": "ent-1",
                    "entity_name": "Roadmap",
                    "summary": "Q3 priorities locked",
                    "score": 0.9,
                    "source_refs": [{
                        "kind": "meeting",
                        "id": "mtg-1",
                        "occurred_at": "2026-05-19T10:00:00Z"
                    }]
                }],
                "template": "entities"
            }
        }
    });
    send(&mut stdin, &req).await;
    let resp = recv(&mut stdout).await;
    assert_eq!(resp["id"], json!(3), "id echo");
    let result = resp.get("result").unwrap_or_else(|| {
        panic!("tools/call must return result, got error: {resp}")
    });
    let structured = result
        .get("structuredContent")
        .expect("kremory_context_block returns structured content");
    let rendered = structured
        .get("rendered")
        .and_then(|v| v.as_str())
        .expect("rendered field must be a string");
    assert!(
        rendered.contains("Roadmap"),
        "rendered template must mention the entity name: {rendered}"
    );
    assert!(
        rendered.contains("meeting:mtg-1"),
        "rendered template must include the source ref: {rendered}"
    );

    shutdown(child, stdin).await;
}

#[tokio::test]
async fn tools_call_kremory_search_on_unbound_server_returns_graph_not_bound_error() {
    // search requires a bound graph. The default `kremory-mcp-server` binary
    // starts unbound (no `KremoryMcpServer::new`), so this call MUST return
    // a JSON-RPC tool-error pointing the caller at the bound constructor.
    // End-to-end pin of the unbound contract through MCP framing.
    let (child, mut stdin, mut stdout) = spawn_server();
    initialize(&mut stdin, &mut stdout).await;

    let req = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "kremory_search",
            "arguments": {
                "workspace_id": "ws-1",
                "query": "anything"
            }
        }
    });
    send(&mut stdin, &req).await;
    let resp = recv(&mut stdout).await;
    assert_eq!(resp["id"], json!(4));

    // rmcp surfaces tool errors EITHER as a JSON-RPC `error` field (when
    // the handler returns Err(McpError::...)) OR as a result with
    // `isError: true` + an error message in content. The unbound path
    // returns Err — assert that shape, then sanity-check the message
    // names the right contract.
    let err = resp.get("error").unwrap_or_else(|| {
        panic!("unbound search must surface as JSON-RPC error, got: {resp}")
    });
    let message = err["message"]
        .as_str()
        .expect("error message must be a string");
    assert!(
        message.contains("kremory_search") || message.contains("KremoryMcpServer::new"),
        "error must name either the tool or the bound constructor to guide remediation: {message}"
    );

    shutdown(child, stdin).await;
}

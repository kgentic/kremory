//! Subprocess JSON-RPC integration tests.
//!
//! Spawns the real `kremory-mcp-server` binary and speaks the MCP protocol
//! over its stdio via rmcp's own client transport.
//!
//! ## Ollama dependency
//!
//! The binary fails loud at boot if its Ollama endpoint is unreachable (see
//! `src/health.rs`). To exercise `tools/list` WITHOUT a live model, these
//! tests point `KREMORY_MCP_OLLAMA_URL` at a dummy TCP listener bound in-test
//! — the boot reachability check is connectivity-only, so a bare listener
//! satisfies it, and `tools/list` never touches the model or embedder.
//!
//! `tools/call`, by contrast, DOES exercise the embedder (Phase-1 embed) and
//! (for the extract path) the LLM — a dummy listener that speaks no HTTP is
//! not enough. The full tools/call round-trip is therefore `#[ignore]`-gated
//! and needs a real Ollama at `http://localhost:11434` with `gemma4:e4b` +
//! `nomic-embed-text` pulled. Run with:
//!
//! ```bash
//! cargo test -p kremory-mcp --test subprocess_jsonrpc -- --ignored
//! ```

use std::time::Duration;

use anyhow::Result;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::ServiceExt;
use tokio::net::TcpListener;

const EXPECTED_TOOLS: &[&str] = &[
    "kremory_remember",
    "kremory_recall",
    "kremory_dream",
    "kremory_list_mutations",
    "kremory_undo",
];

/// Bind a loopback TCP listener and keep accepting in the background so the
/// server binary's boot reachability check (a single TCP connect) succeeds.
/// Returns the `http://127.0.0.1:<port>` URL plus the accept task handle
/// (held by the caller for the lifetime of the test).
async fn dummy_ollama() -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let url = format!("http://127.0.0.1:{}", addr.port());
    let handle = tokio::spawn(async move {
        loop {
            // Accept and immediately drop — the reachability check only needs
            // the connect to succeed; it sends no bytes we must answer.
            if listener.accept().await.is_err() {
                break;
            }
        }
    });
    Ok((url, handle))
}

/// Unique temp DB path per test run (no uuid dep — nanos + pid is enough for
/// test isolation).
fn temp_db_path(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "kremory-mcp-test-{tag}-{}-{nanos}.db",
        std::process::id()
    ))
}

/// Spawn the built binary as an MCP server over stdio, pointed at `ollama_url`
/// plus a fresh temp DB. Returns the connected client (the initialize
/// handshake is already performed by `().serve(...)`).
async fn spawn_server(
    ollama_url: &str,
    db_path: &std::path::Path,
) -> Result<rmcp::service::RunningService<rmcp::RoleClient, ()>> {
    let bin = env!("CARGO_BIN_EXE_kremory-mcp-server");
    let db = db_path.to_string_lossy().to_string();
    let url = ollama_url.to_string();
    let transport = TokioChildProcess::new(tokio::process::Command::new(bin).configure(|cmd| {
        cmd.env("KREMORY_MCP_DB_PATH", &db)
            .env("KREMORY_MCP_OLLAMA_URL", &url)
            .env("KREMORY_MCP_MODEL_ID", "gemma4:e4b");
    }))?;
    let client = ().serve(transport).await?;
    Ok(client)
}

#[tokio::test]
async fn initialize_and_list_tools_exposes_floor_5_surface() -> Result<()> {
    let (url, accept_task) = dummy_ollama().await?;
    let db = temp_db_path("list");
    let client = spawn_server(&url, &db).await?;

    let tools = client.list_all_tools().await?;
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();

    assert_eq!(
        tools.len(),
        5,
        "server must expose exactly the floor-5 tool surface, got {names:?}"
    );
    for expected in EXPECTED_TOOLS {
        assert!(
            names.contains(expected),
            "tools/list must contain {expected}, got {names:?}"
        );
    }

    // Every tool must advertise a valid (object) input schema.
    for t in &tools {
        let schema = serde_json::to_value(t.input_schema.as_ref())?;
        assert!(
            schema.is_object(),
            "tool {} input_schema must be a JSON object: {schema}",
            t.name
        );
    }

    // The recall tool's schema must surface the format + template enum values
    // (RecallFormat / RecallTemplateWire) so clients can discover them.
    let recall = tools
        .iter()
        .find(|t| t.name == "kremory_recall")
        .expect("kremory_recall present");
    let recall_schema = serde_json::to_string(recall.input_schema.as_ref())?;
    for token in ["structured", "temporal_facts", "edge_summary", "entities"] {
        assert!(
            recall_schema.contains(token),
            "kremory_recall input schema must surface enum value {token:?}: {recall_schema}"
        );
    }

    client.cancel().await?;
    accept_task.abort();
    let _ = std::fs::remove_file(&db);
    Ok(())
}

/// Full `tools/call` round-trip through the real binary. Requires a live
/// Ollama (`gemma4:e4b` + `nomic-embed-text` pulled) because Phase-1 embed +
/// LLM extraction run inside the subprocess — a dummy TCP listener cannot
/// answer those. `#[ignore]` by default; run with `-- --ignored`.
#[tokio::test]
#[ignore = "requires a live Ollama at localhost:11434 with gemma4:e4b + nomic-embed-text"]
async fn tools_call_round_trip_all_three_tools() -> Result<()> {
    use rmcp::model::CallToolRequestParams;

    let url = "http://localhost:11434".to_string();
    let db = temp_db_path("call");
    let client = spawn_server(&url, &db).await?;

    // remember (mode-c pinned path — no LLM extraction, but Phase-1 embed
    // still hits the real embedder).
    let remember_args = serde_json::json!({
        "namespace": "subprocess-ns",
        "content": "Quenby leads the design team",
        "source_kind": "note",
        "source_id": "doc-1",
        "structured_facts": [
            {"subject": "Quenby", "predicate": "leads", "object": "design"}
        ],
        "skip_extraction": true
    });
    let remember = client
        .call_tool(
            CallToolRequestParams::new("kremory_remember")
                .with_arguments(remember_args.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(remember.is_error, Some(false), "remember must not error");

    // recall (structured) → must find the pinned entity.
    let recall_args = serde_json::json!({
        "namespace": "subprocess-ns",
        "query": "Quenby",
        "k": 10,
        "format": "structured"
    });
    let recall = client
        .call_tool(
            CallToolRequestParams::new("kremory_recall")
                .with_arguments(recall_args.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(recall.is_error, Some(false), "recall must not error");
    let recall_body = recall
        .structured_content
        .expect("recall structured payload");
    // NOTE: the deterministic recall>=1 read-path proof lives at the mock tier
    // (handler_roundtrip::recall_returns_pinned_entity_after_enrichment). Over
    // the wire we cannot run the enrichment seam, and a `skip_extraction`
    // pinned entity is not FTS-searchable (its name is populated only by the
    // enrichment/verify stage `skip_extraction` skips), so here we assert only
    // that the recall tool returns a well-formed structured payload.
    let count = recall_body["count"].as_u64().expect("count present");
    let results = recall_body["results"].as_array().expect("results array");
    assert_eq!(
        count as usize,
        results.len(),
        "recall count must equal results len: {recall_body}"
    );

    // dream → returns a summary.
    let dream_args = serde_json::json!({ "namespace": "subprocess-ns" });
    let dream = client
        .call_tool(
            CallToolRequestParams::new("kremory_dream")
                .with_arguments(dream_args.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(dream.is_error, Some(false), "dream must not error");
    let dream_body = dream.structured_content.expect("dream structured payload");
    assert!(
        dream_body["duration_ms"].is_u64(),
        "dream output must carry duration_ms: {dream_body}"
    );

    client.cancel().await?;
    let _ = std::fs::remove_file(&db);
    Ok(())
}

/// Give the boot reachability check time to fail + the process to exit when
/// nothing is listening — proves the fail-loud boot path (no silent degrade).
#[tokio::test]
async fn server_fails_loud_when_ollama_unreachable() -> Result<()> {
    let db = temp_db_path("unreachable");
    // Port 1 on loopback: nothing listening → boot reachability check fails →
    // the process must exit non-zero BEFORE serving, so the client initialize
    // handshake cannot complete.
    let spawn = spawn_server("http://127.0.0.1:1", &db).await;
    match spawn {
        Ok(client) => {
            // If the transport connected, the server must NOT complete a
            // successful initialize — the binary should have exited. Give it a
            // moment, then assert a subsequent request fails.
            tokio::time::sleep(Duration::from_secs(2)).await;
            let listed = client.list_all_tools().await;
            assert!(
                listed.is_err(),
                "server must not answer tools/list when Ollama was unreachable at boot"
            );
            let _ = client.cancel().await;
        }
        Err(_) => {
            // Transport/initialize failed outright — also an acceptable
            // fail-loud shape (the child exited before the handshake).
        }
    }
    let _ = std::fs::remove_file(&db);
    Ok(())
}

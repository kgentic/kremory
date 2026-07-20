//! Live dogfood driver — spawns the real `kremory-mcp-server` binary and
//! drives it end-to-end against a REAL Ollama (`gemma4:e4b`), empirically
//! settling TD-113 (does a mode-(c) pinned `structured_facts` entity become
//! recall-findable, and does `dream()` close the gap without it?).
//!
//! Reuses the same rmcp client-transport spawn pattern as
//! `subprocess_jsonrpc.rs`. Requires:
//! - `target/release/kremory-mcp-server` built
//! - Ollama reachable at `http://localhost:11434` with `gemma4:e4b` +
//!   `nomic-embed-text` pulled
//!
//! Run with:
//! ```bash
//! cargo test -p kremory-mcp --test dogfood_live -- --ignored --nocapture
//! ```
//!
//! Every step is wrapped in a bounded timeout so a hung LLM call reports
//! partial results instead of hanging the suite forever.

use std::time::Duration;

use anyhow::{anyhow, Result};
use rmcp::model::CallToolRequestParams;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::ServiceExt;

const STEP_TIMEOUT: Duration = Duration::from_secs(90);

fn temp_db_path(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "kremory-dogfood-{tag}-{}-{nanos}.db",
        std::process::id()
    ))
}

async fn spawn_release_server(
    db_path: &std::path::Path,
) -> Result<rmcp::service::RunningService<rmcp::RoleClient, ()>> {
    // Use the release binary explicitly per task instructions (not the
    // CARGO_BIN_EXE_ test-profile binary) so we dogfood the exact artifact a
    // consumer would run.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let bin = std::path::Path::new(manifest_dir)
        .join("../../target/release/kremory-mcp-server")
        .canonicalize()
        .map_err(|e| {
            anyhow!(
                "release binary not found — run `cargo build --release -p kremory-mcp` first: {e}"
            )
        })?;
    let db = db_path.to_string_lossy().to_string();
    let transport = TokioChildProcess::new(tokio::process::Command::new(&bin).configure(|cmd| {
        cmd.env("KREMORY_MCP_DB_PATH", &db)
            .env("KREMORY_MCP_OLLAMA_URL", "http://localhost:11434")
            .env("KREMORY_MCP_MODEL_ID", "gemma4:e4b");
    }))?;
    let client = ().serve(transport).await?;
    Ok(client)
}

async fn call(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tool: &str,
    args: serde_json::Value,
) -> Result<rmcp::model::CallToolResult> {
    let fut = client.call_tool(
        CallToolRequestParams::new(tool.to_string())
            .with_arguments(args.as_object().unwrap().clone()),
    );
    match tokio::time::timeout(STEP_TIMEOUT, fut).await {
        Ok(inner) => inner.map_err(|e| anyhow!("{tool} call failed: {e}")),
        Err(_) => Err(anyhow!("{tool} call exceeded {STEP_TIMEOUT:?} timeout")),
    }
}

fn recall_count(result: &rmcp::model::CallToolResult) -> Result<u64> {
    let body = result
        .structured_content
        .clone()
        .ok_or_else(|| anyhow!("recall result missing structured_content: {result:?}"))?;
    body["count"]
        .as_u64()
        .ok_or_else(|| anyhow!("recall structured_content missing count: {body}"))
}

#[tokio::test]
#[ignore = "live dogfood — requires real Ollama at localhost:11434 with gemma4:e4b; run with -- --ignored --nocapture"]
async fn dogfood_live_settles_td_113() -> Result<()> {
    println!("\n=== SCENARIO A: protocol smoke (initialize + tools/list) ===");
    let db_a = temp_db_path("a");
    let client = spawn_release_server(&db_a).await?;

    let tools = client.list_all_tools().await?;
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    println!("tools/list -> {names:?}");
    let scenario_a_pass = names.len() == 5
        && names.contains(&"kremory_remember")
        && names.contains(&"kremory_recall")
        && names.contains(&"kremory_dream")
        && names.contains(&"kremory_list_mutations")
        && names.contains(&"kremory_undo");

    let recall_tool = tools.iter().find(|t| t.name == "kremory_recall");
    let recall_schema_ok = if let Some(t) = recall_tool {
        let schema = serde_json::to_string(t.input_schema.as_ref())?;
        ["structured", "temporal_facts", "edge_summary", "entities"]
            .iter()
            .all(|tok| schema.contains(tok))
    } else {
        false
    };
    println!("scenario_a: 5-tool-surface={scenario_a_pass} recall-schema-enums={recall_schema_ok}");

    // ── SCENARIO B: mode-(a) extraction round-trip (real LLM) ──
    println!("\n=== SCENARIO B: mode-(a) real-LLM extraction round trip ===");
    let remember_b = call(
        &client,
        "kremory_remember",
        serde_json::json!({
            "namespace": "dogfood",
            "content": "Ada Lovelace worked with Charles Babbage on the Analytical Engine in London.",
            "source_kind": "note"
        }),
    )
    .await;
    let remember_b_ok = match &remember_b {
        Ok(r) => {
            println!(
                "remember(dogfood) -> is_error={:?} content={:?}",
                r.is_error, r.structured_content
            );
            r.is_error == Some(false)
        }
        Err(e) => {
            println!("remember(dogfood) FAILED: {e}");
            false
        }
    };

    let mut scenario_b_text_block = String::new();
    let mut scenario_b_recall_count: Option<u64> = None;

    if remember_b_ok {
        let recall_text = call(
            &client,
            "kremory_recall",
            serde_json::json!({ "namespace": "dogfood", "query": "Ada Lovelace", "format": "text" }),
        )
        .await;
        match &recall_text {
            Ok(r) => {
                if let Some(body) = &r.structured_content {
                    scenario_b_text_block = body["block"].as_str().unwrap_or_default().to_string();
                }
                println!("recall(text) block=\n{scenario_b_text_block}");
            }
            Err(e) => println!("recall(text) FAILED: {e}"),
        }

        let recall_structured = call(
            &client,
            "kremory_recall",
            serde_json::json!({ "namespace": "dogfood", "query": "Ada Lovelace", "format": "structured" }),
        )
        .await;
        match &recall_structured {
            Ok(r) => {
                let count = recall_count(r).unwrap_or(0);
                scenario_b_recall_count = Some(count);
                println!(
                    "recall(structured) count={count} body={:?}",
                    r.structured_content
                );
            }
            Err(e) => println!("recall(structured) FAILED: {e}"),
        }
    } else {
        println!("skipping recall checks — remember(dogfood) did not succeed");
    }

    // ── SCENARIO C: THE TD-113 QUESTION (mode-(c)) ──
    println!("\n=== SCENARIO C: mode-(c) pinned fact, TD-113 ===");
    let remember_c = call(
        &client,
        "kremory_remember",
        serde_json::json!({
            "namespace": "modec",
            "content": "placeholder",
            "skip_extraction": true,
            "structured_facts": [
                {"subject": "Grace Hopper", "predicate": "invented", "object": "the compiler"}
            ]
        }),
    )
    .await;
    let remember_c_ok = match &remember_c {
        Ok(r) => {
            println!(
                "remember(modec, skip_extraction) -> is_error={:?} content={:?}",
                r.is_error, r.structured_content
            );
            r.is_error == Some(false)
        }
        Err(e) => {
            println!("remember(modec) FAILED: {e}");
            false
        }
    };

    let mut c1_count: Option<u64> = None;
    if remember_c_ok {
        let recall_c1 = call(
            &client,
            "kremory_recall",
            serde_json::json!({ "namespace": "modec", "query": "Grace Hopper", "format": "structured" }),
        )
        .await;
        match &recall_c1 {
            Ok(r) => {
                let count = recall_count(r).unwrap_or(0);
                c1_count = Some(count);
                println!(
                    "C1 recall(modec, pre-dream) count={count} body={:?}",
                    r.structured_content
                );
            }
            Err(e) => println!("C1 recall FAILED: {e}"),
        }
    }

    println!("\n--- invoking kremory_dream(modec) ---");
    let dream = call(
        &client,
        "kremory_dream",
        serde_json::json!({ "namespace": "modec" }),
    )
    .await;
    let dream_ok = match &dream {
        Ok(r) => {
            println!(
                "dream(modec) -> is_error={:?} content={:?}",
                r.is_error, r.structured_content
            );
            r.is_error == Some(false)
        }
        Err(e) => {
            println!("dream(modec) FAILED: {e}");
            false
        }
    };

    let mut c2_count: Option<u64> = None;
    if dream_ok {
        let recall_c2 = call(
            &client,
            "kremory_recall",
            serde_json::json!({ "namespace": "modec", "query": "Grace Hopper", "format": "structured" }),
        )
        .await;
        match &recall_c2 {
            Ok(r) => {
                let count = recall_count(r).unwrap_or(0);
                c2_count = Some(count);
                println!(
                    "C2 recall(modec, post-dream) count={count} body={:?}",
                    r.structured_content
                );
            }
            Err(e) => println!("C2 recall FAILED: {e}"),
        }
    } else {
        println!("skipping C2 recall — dream(modec) did not succeed");
    }

    // ── SCENARIO D: error path (as_of unsupported) ──
    println!("\n=== SCENARIO D: as_of unsupported error path ===");
    let recall_d = call(
        &client,
        "kremory_recall",
        serde_json::json!({ "namespace": "dogfood", "query": "x", "as_of": "2020-01-01T00:00:00Z" }),
    )
    .await;
    let scenario_d_pass = match &recall_d {
        Ok(r) => {
            let is_err = r.is_error == Some(true);
            println!(
                "recall(as_of) -> is_error={:?} content={:?}",
                r.is_error, r.content
            );
            is_err
        }
        Err(e) => {
            // A JSON-RPC-level error (protocol error, not tool-result error)
            // also satisfies "confirmed to return an error".
            println!("recall(as_of) -> protocol-level error (also acceptable): {e}");
            true
        }
    };

    client.cancel().await?;
    let _ = std::fs::remove_file(&db_a);

    // ── Summary ──
    println!("\n\n---RESULT---");
    let status = if scenario_a_pass && remember_b_ok && remember_c_ok && dream_ok && scenario_d_pass
    {
        "success"
    } else {
        "partial"
    };
    println!("status: {status}");
    println!("scenario_a: 3-tools={scenario_a_pass} recall-schema-enums={recall_schema_ok}");
    println!(
        "scenario_b_remember_ok: {remember_b_ok} scenario_b_recall_count: {:?}",
        scenario_b_recall_count
    );
    println!("scenario_c1_modec_recall_count (pre-dream): {:?}", c1_count);
    println!(
        "scenario_c2_modec_after_dream_count (post-dream): {:?}",
        c2_count
    );
    println!("scenario_d_error_path: {scenario_d_pass}");
    println!("---END---");

    Ok(())
}

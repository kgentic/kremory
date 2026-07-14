//! `kremory-mcp-server` — binary entry point for the kremory MCP server.
//!
//! Stdio transport, 5 tools registered (see `lib.rs`): `kremory_remember`,
//! `kremory_recall`, `kremory_dream`, `kremory_list_mutations`,
//! `kremory_undo`.
//!
//! Env-driven mode-(a) construction — no test-mock injection point in this
//! binary by design (real `Memory` only). Fails loudly (non-zero exit,
//! before serving) when the DB path is missing or Ollama is unreachable.
//!
//! ```bash
//! export KREMORY_MCP_DB_PATH=./agent.db
//! cargo run --release -p kremory-mcp --bin kremory-mcp-server
//! ```
//!
//! ## Env vars
//!
//! - `KREMORY_MCP_DB_PATH` — required. Path to the kremory libSQL database.
//! - `KREMORY_MCP_OLLAMA_URL` — default `http://localhost:11434`.
//! - `KREMORY_MCP_MODEL_ID` — default `gemma4:e4b`.
//! - `KREMORY_MCP_DEBUG=1` — dump raw request/response JSON to stderr.

mod health;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use kremory_mcp::KremoryMcpServer;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const REACHABILITY_TIMEOUT: Duration = Duration::from_secs(5);

#[tokio::main]
async fn main() -> Result<()> {
    // MCP servers traditionally log to stderr because stdout is reserved
    // for the JSON-RPC protocol stream.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = FmtSubscriber::builder()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow!("failed to install tracing subscriber: {e}"))?;

    let db_path = std::env::var("KREMORY_MCP_DB_PATH").map_err(|_| {
        anyhow!(
            "KREMORY_MCP_DB_PATH is required — set it to the path of the kremory \
             libSQL database (e.g. ./agent.db)"
        )
    })?;
    let ollama_url =
        std::env::var("KREMORY_MCP_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model_id = std::env::var("KREMORY_MCP_MODEL_ID").ok();

    tracing::info!(
        db_path = %db_path,
        ollama_url = %ollama_url,
        model_id = model_id.as_deref().unwrap_or("gemma4:e4b (default)"),
        "kremory-mcp-server booting"
    );

    // Fail loud BEFORE constructing Memory — `with_ollama_at_model` does not
    // itself probe reachability (first `.remember()`/`.recall()` call would
    // return a network error instead). See `health.rs` for the rationale.
    health::check_reachable(&ollama_url, REACHABILITY_TIMEOUT)
        .await
        .map_err(|reason| {
            anyhow!(
                "Ollama unreachable at {ollama_url} ({reason}). Start Ollama \
                 (`ollama serve`) or set KREMORY_MCP_OLLAMA_URL to a reachable endpoint."
            )
        })?;

    let mem = kremory::facade::providers::with_ollama_at_model(ollama_url, model_id, &db_path)
        .await
        .with_context(|| format!("failed to open kremory Memory at {db_path}"))?;

    tracing::info!(
        "kremory-mcp-server ready — serving 5 tools (kremory_remember, kremory_recall, \
         kremory_dream, kremory_list_mutations, kremory_undo) on stdio transport"
    );

    let server = KremoryMcpServer::new(Arc::new(mem));
    let (input, output) = stdio();
    let service = server.serve((input, output)).await?;
    service.waiting().await?;
    Ok(())
}

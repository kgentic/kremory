//! `rqlm-mcp-server` — binary entry point for the rqlm MCP server.
//!
//! Stdio transport, 4 tools registered (see `lib.rs`). Per ADR-Phase-D.0:
//! consumers (aidocs, future SDK clients) spawn this binary as a
//! subprocess and speak JSON-RPC over stdin/stdout.
//!
//! ```bash
//! cargo build --release -p rqlm-mcp
//! ./target/release/rqlm-mcp-server  # then send JSON-RPC requests
//! ```

use anyhow::Result;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use rqlm_mcp::RqlmMcpServer;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<()> {
    // MCP servers traditionally log to stderr because stdout is reserved
    // for the JSON-RPC protocol stream.
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = FmtSubscriber::builder()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow::anyhow!("failed to install tracing subscriber: {e}"))?;

    tracing::info!(
        "rqlm-mcp-server starting on stdio transport (unbound — only rqlm_context_block is functional; \
         compose RqlmMcpServer::new(graph, provider) from a host to enable graph-backed tools)"
    );

    let server = RqlmMcpServer::unbound();
    let (input, output) = stdio();
    let service = server.serve((input, output)).await?;
    service.waiting().await?;
    Ok(())
}

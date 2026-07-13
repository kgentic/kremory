//! `kremory-mcp-server` — binary entry point for the kremory MCP server.
//!
//! Stdio transport, 4 tools registered (see `lib.rs`). Per ADR-Phase-D.0:
//! consumers (SDK clients) spawn this binary as a
//! subprocess and speak JSON-RPC over stdin/stdout.
//!
//! ```bash
//! cargo build --release -p kremory-mcp
//! ./target/release/kremory-mcp-server  # then send JSON-RPC requests
//! ```

use anyhow::Result;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use kremory_mcp::KremoryMcpServer;
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
        "kremory-mcp-server starting on stdio transport (unbound — only kremory_context_block is functional; \
         compose KremoryMcpServer::new(graph, provider) from a host to enable graph-backed tools)"
    );

    let server = KremoryMcpServer::unbound();
    let (input, output) = stdio();
    let service = server.serve((input, output)).await?;
    service.waiting().await?;
    Ok(())
}

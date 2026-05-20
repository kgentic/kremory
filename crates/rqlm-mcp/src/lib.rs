//! rqlm-mcp — MCP server wrapping rqlm's 4 public entry points as
//! JSON-RPC tools over rmcp stdio transport.
//!
//! Per ADR-Phase-D.0 §"rqlm-mcp tool surface" + master plan §3 D.4 / D.4b.
//!
//! ## Composition shape (D.4b)
//!
//! The server holds `Option<ServerState>` where `ServerState` carries
//! `Arc<dyn GraphHandle>` + `Arc<dyn ChatProvider>`. Two constructors:
//!
//! - [`RqlmMcpServer::unbound`] — produces an "unbound" server. The
//!   binary entry point (`main.rs`) uses this so it can spawn over
//!   stdio for protocol-level smoke testing; tools that need the graph
//!   return a `GraphNotBound` error. `rqlm_context_block` works
//!   regardless — it's a pure function over already-fetched results.
//! - [`RqlmMcpServer::new`] — bind a real `GraphHandle` + `ChatProvider`.
//!   Downstream consumers (the host application in D.5; aidocs paying SDK) compose
//!   their concrete graph + LLM client here.
//!
//! ## Tools registered (4 — matches rqlm's public surface)
//!
//! - `rqlm_ingest_episode` — delegate to [`rql_memory::ingest_episode`]
//! - `rqlm_run_dream_phase` — delegate to [`rql_memory::run_dream_phase`]
//! - `rqlm_search` — delegate to [`rql_memory::search`]
//! - `rqlm_context_block` — call [`rql_memory::context_block`] directly
//!
//! ## Error mapping
//!
//! - [`conversions::ConversionError`] → MCP `invalid_params` (caller-side
//!   wire shape problem; client can fix).
//! - "graph not bound" → MCP `internal_error` with a stable message; the
//!   binary advertises this state when spawned standalone.
//! - [`rql_memory::RqlmError`] → MCP `internal_error` carrying the
//!   underlying message (Display string). The error path stays open for
//!   diagnostic relay across the JSON-RPC boundary.

pub mod conversions;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::{model::*, tool, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use rql_memory::{
    context_block as rqlm_context_block_fn, ingest_episode as rqlm_ingest_episode_fn,
    run_dream_phase as rqlm_run_dream_phase_fn, search as rqlm_search_fn, ChatProvider,
    GraphHandle, RqlmError,
};

use crate::conversions::ConversionError;

// Re-export the rqlm trait + provider surface so downstream consumers
// (the host application, paying SDK customers) depend on this crate alone when
// composing a bound server.
pub use rql_memory::{ChatProvider as RqlmChatProvider, GraphHandle as RqlmGraphHandle};

// ────────────────────────────────────────────────────────────────────────
// MCP parameter / output types
//
// These duplicate rqlm's types with the added `schemars::JsonSchema`
// derive so rmcp's macros can generate tool input schemas. The
// bidirectional conversions to/from rqlm types live in `conversions.rs`.
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IngestEpisodeParameters {
    /// Workspace identifier — required scoping dimension.
    pub workspace_id: String,
    /// Optional thread / session ID within the workspace.
    pub thread_id: Option<String>,
    /// Raw episode content (transcript, doc chunk, chat message).
    pub content: String,
    /// Source-of-truth reference for this episode.
    pub source_ref_kind: String, // "meeting" | "document" | "chat"
    pub source_ref_id: String,
    pub source_ref_occurred_at: String, // ISO-8601 UTC
    /// Optional caller-supplied facts to pin alongside extraction.
    #[serde(default)]
    pub structured_facts: Vec<StructuredFactInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StructuredFactInput {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_at: Option<String>,
    pub invalid_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IngestEpisodeOutput {
    pub entities_added: usize,
    pub edges_added: usize,
    pub facts_invalidated: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunDreamPhaseParameters {
    pub workspace_id: String,
    pub thread_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunDreamPhaseOutput {
    pub communities_recomputed: usize,
    pub cross_meeting_merges: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchParameters {
    pub workspace_id: String,
    pub thread_id: Option<String>,
    pub query: String,
    pub limit: Option<usize>,
    /// ISO-8601 UTC — when set, returns results valid at that timestamp.
    pub as_of: Option<String>,
    /// Optional filter — "meeting" | "document" | "chat".
    pub source_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResultsOutput {
    pub results: Vec<RetrievedContextOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetrievedContextOutput {
    pub entity_id: String,
    pub entity_name: String,
    pub summary: String,
    pub score: f32,
    pub source_refs: Vec<SourceRefOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceRefOutput {
    pub kind: String,
    pub id: String,
    pub occurred_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextBlockParameters {
    /// Pre-fetched results from a `rqlm_search` call.
    pub results: Vec<RetrievedContextOutput>,
    /// Template strategy — "entities" | "edge_summary" | "temporal_facts".
    pub template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextBlockOutput {
    pub rendered: String,
}

// ────────────────────────────────────────────────────────────────────────
// MCP server
// ────────────────────────────────────────────────────────────────────────

/// rqlm MCP server. Optional state — see crate-level docs for the unbound
/// vs bound distinction.
#[derive(Clone)]
pub struct RqlmMcpServer {
    state: Option<ServerState>,
}

#[derive(Clone)]
struct ServerState {
    graph: Arc<dyn GraphHandle>,
    provider: Arc<dyn ChatProvider>,
}

impl Default for RqlmMcpServer {
    fn default() -> Self {
        Self::unbound()
    }
}

impl std::fmt::Debug for RqlmMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RqlmMcpServer")
            .field("bound", &self.state.is_some())
            .finish()
    }
}

impl RqlmMcpServer {
    /// Construct an unbound server. Tools requiring a graph backend
    /// (`ingest_episode` / `search` / `run_dream_phase`) return a
    /// `GraphNotBound` MCP error. `context_block` works regardless.
    /// Used by the binary entry point so `rqlm-mcp-server` can spawn for
    /// protocol-level smoke testing without a backend.
    pub fn unbound() -> Self {
        Self { state: None }
    }

    /// Construct a bound server. Downstream consumers (the host application, paying
    /// SDK customers) supply their concrete graph + LLM client.
    pub fn new(graph: Arc<dyn GraphHandle>, provider: Arc<dyn ChatProvider>) -> Self {
        Self {
            state: Some(ServerState { graph, provider }),
        }
    }

    /// Returns true when a graph + provider are bound.
    pub fn is_bound(&self) -> bool {
        self.state.is_some()
    }

    fn require_state(&self, tool: &str) -> Result<&ServerState, ErrorData> {
        self.state.as_ref().ok_or_else(|| graph_not_bound(tool))
    }
}

#[tool_router(server_handler)]
impl RqlmMcpServer {
    #[tool(
        name = "rqlm_ingest_episode",
        description = "Ingest one episode (transcript chunk, document, chat message) into the rqlm graph with workspace scoping. Returns counts of entities/edges added + facts invalidated. Requires a bound graph backend; returns a 'graph not bound' error on the unbound binary."
    )]
    pub async fn rqlm_ingest_episode(
        &self,
        Parameters(params): Parameters<IngestEpisodeParameters>,
    ) -> Result<CallToolResult, ErrorData> {
        let state = self.require_state("rqlm_ingest_episode")?;
        let (scope, source_ref, facts, content) = params.into_rqlm().map_err(conv_to_error)?;
        let result = rqlm_ingest_episode_fn(
            state.graph.as_ref(),
            &content,
            source_ref,
            facts,
            state.provider.clone(),
            scope,
        )
        .await
        .map_err(|e| rqlm_to_error("rqlm_ingest_episode", e))?;
        let wire: IngestEpisodeOutput = result.into();
        wire_to_call_result(&wire, "rqlm_ingest_episode")
    }

    #[tool(
        name = "rqlm_run_dream_phase",
        description = "Run the packaged batch consolidation recipe (community recompute, cross-meeting distillation, supersession sweep) over the scoped graph. Consumer-triggered, not a daemon. Requires a bound graph backend."
    )]
    pub async fn rqlm_run_dream_phase(
        &self,
        Parameters(params): Parameters<RunDreamPhaseParameters>,
    ) -> Result<CallToolResult, ErrorData> {
        let state = self.require_state("rqlm_run_dream_phase")?;
        let scope = params.into_rqlm().map_err(conv_to_error)?;
        let result =
            rqlm_run_dream_phase_fn(state.graph.as_ref(), scope, state.provider.clone())
                .await
                .map_err(|e| rqlm_to_error("rqlm_run_dream_phase", e))?;
        let wire: RunDreamPhaseOutput = result.into();
        wire_to_call_result(&wire, "rqlm_run_dream_phase")
    }

    #[tool(
        name = "rqlm_search",
        description = "Query the graph with rqlm's opinionated retrieval defaults (hybrid retrieval + scoping + rerank). Returns retrieved contexts ordered by score. Requires a bound graph backend."
    )]
    pub async fn rqlm_search(
        &self,
        Parameters(params): Parameters<SearchParameters>,
    ) -> Result<CallToolResult, ErrorData> {
        let state = self.require_state("rqlm_search")?;
        let (scope, query, opts) = params.into_rqlm().map_err(conv_to_error)?;
        let results = rqlm_search_fn(state.graph.as_ref(), &query, scope, opts)
            .await
            .map_err(|e| rqlm_to_error("rqlm_search", e))?;
        let wire: SearchResultsOutput = results.into();
        wire_to_call_result(&wire, "rqlm_search")
    }

    #[tool(
        name = "rqlm_context_block",
        description = "Render search results into the final string handed to the LLM, per the requested template (entities | edge_summary | temporal_facts). Pure function — works regardless of graph binding."
    )]
    pub async fn rqlm_context_block(
        &self,
        Parameters(params): Parameters<ContextBlockParameters>,
    ) -> Result<CallToolResult, ErrorData> {
        let (results, template) = params.into_rqlm().map_err(conv_to_error)?;
        let rendered = rqlm_context_block_fn(&results, template);
        let wire = ContextBlockOutput { rendered };
        wire_to_call_result(&wire, "rqlm_context_block")
    }
}

// ────────────────────────────────────────────────────────────────────────
// Error mapping helpers
// ────────────────────────────────────────────────────────────────────────

fn conv_to_error(e: ConversionError) -> ErrorData {
    ErrorData::invalid_params(e.to_string(), None)
}

fn rqlm_to_error(tool: &'static str, e: RqlmError) -> ErrorData {
    ErrorData::internal_error(format!("{tool} failed: {e}"), None)
}

fn graph_not_bound(tool: &str) -> ErrorData {
    ErrorData::internal_error(
        format!(
            "{tool} requires a bound graph backend — this server was constructed with \
             RqlmMcpServer::unbound(). Compose with RqlmMcpServer::new(graph, provider) to enable."
        ),
        None,
    )
}

fn wire_to_call_result<T>(wire: &T, tool: &str) -> Result<CallToolResult, ErrorData>
where
    T: serde::Serialize,
{
    let value = serde_json::to_value(wire).map_err(|e| {
        ErrorData::internal_error(
            format!("{tool}: failed to serialize structured output: {e}"),
            None,
        )
    })?;
    Ok(CallToolResult::structured(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbound_server_constructs() {
        let s = RqlmMcpServer::unbound();
        assert!(!s.is_bound());
        let s = RqlmMcpServer::default();
        assert!(!s.is_bound());
    }

    /// Locks the tool-name contract — these strings must remain stable
    /// for MCP clients (aidocs etc.) wired against them.
    #[test]
    fn tool_name_contract_pins() {
        const EXPECTED_TOOLS: &[&str] = &[
            "rqlm_ingest_episode",
            "rqlm_run_dream_phase",
            "rqlm_search",
            "rqlm_context_block",
        ];
        assert_eq!(EXPECTED_TOOLS.len(), 4);
        for tool in EXPECTED_TOOLS {
            assert!(
                tool.starts_with("rqlm_"),
                "tool name must be rqlm_-prefixed: {tool}"
            );
        }
    }

    #[test]
    fn graph_not_bound_error_carries_tool_name_and_remedy() {
        let err = graph_not_bound("rqlm_search");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("rqlm_search"),
            "error must carry tool name for debug: {msg}"
        );
        assert!(
            msg.contains("RqlmMcpServer::new"),
            "error must point at the bound constructor as remedy: {msg}"
        );
    }
}

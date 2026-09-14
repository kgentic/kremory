//! kremory-mcp — MCP server wrapping kremory's 0.4.0 `Memory` facade as 5
//! JSON-RPC tools over rmcp stdio transport.
//!
//! Per the kremory-mcp-rewrite-0.4.0 spec (+ G3 reversible-mutations gap
//! closure): this crate is the bridge between MCP clients (e.g. Claude Code,
//! aidocs) and the kremory SDK.
//!
//! ## Composition shape
//!
//! [`KremoryMcpServer`] always holds a bound `Arc<kremory::Memory>` — there
//! is no "unbound" state. `main.rs` constructs a real `Memory` from env vars
//! (mode-(a), env-driven) and fails loudly (non-zero exit) if the DB path is
//! missing or Ollama is unreachable at boot, rather than serving a degraded
//! "graph not bound" stub.
//!
//! ## Tools registered (5 — floor-5 tool surface over `Memory`)
//!
//! - `kremory_remember` — ingest (delegates to [`kremory::Memory::remember`])
//! - `kremory_recall` — the ONLY search tool; hybrid keyword + semantic +
//!   graph retrieval (delegates to [`kremory::Memory::recall`])
//! - `kremory_dream` — batch consolidation (delegates to
//!   [`kremory::Memory::dream`])
//! - `kremory_list_mutations` — the SEE half of the reversible-mutations
//!   story: what did `dream()` change? (delegates to
//!   [`kremory::Memory::list_mutations`] / [`kremory::Memory::mutation_history`])
//! - `kremory_undo` — the unified FIX dispatcher: reverse any logged mutation
//!   by `mutation_id` (delegates to [`kremory::Memory::undo`])
//!
//! ## Error mapping
//!
//! - [`conversions::ConversionError`] (bad wire shape, unparseable
//!   RFC 3339 timestamp) → MCP `invalid_params`.
//! - `kremory::MemoryError` (facade error) → MCP `internal_error` carrying
//!   the underlying `Display` string.
//!
//! Both are unified internally as [`handlers::ToolError`] so each handler can
//! record an `outcome` label (`ok` | `invalid_params` | `internal_error`) for
//! the `kremory_mcp.tool.calls` counter before converting to the final
//! `ErrorData` — observability built in at emit, not
//! bolted on after. `handlers::do_remember` / `do_recall` / `do_dream` are
//! the transport-agnostic bodies, shared with the `kremory-http` REST bin
//! (`bin/kremory-http.rs`) — see `handlers.rs` module docs.

pub mod conversions;
pub mod handlers;
pub mod health;
pub mod params;

use std::sync::Arc;
use std::time::Instant;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::{model::*, tool, tool_router};

use kremory::Memory;

use crate::handlers::ToolError;
use crate::params::{
    DreamParams, ListMutationsParams, MutationRecordWire, RecallFormat, RecallParams,
    RememberParams, UndoOutcomeWire, UndoParams,
};

// ────────────────────────────────────────────────────────────────────────
// Internal error unification
// ────────────────────────────────────────────────────────────────────────

/// [`ToolError`] → the final `rmcp::ErrorData` the JSON-RPC boundary expects.
/// The unification itself (`ConversionError` / `kremory::MemoryError` →
/// `ToolError`) lives in `handlers.rs` so it's shared with the REST bin; this
/// conversion is MCP-specific and stays here.
impl From<ToolError> for ErrorData {
    fn from(e: ToolError) -> Self {
        match e {
            ToolError::InvalidParams(msg) => ErrorData::invalid_params(msg, None),
            ToolError::Internal(msg) => ErrorData::internal_error(msg, None),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// MCP server
// ────────────────────────────────────────────────────────────────────────

/// kremory MCP server. Always bound to a real `kremory::Memory` — see
/// crate-level docs.
#[derive(Clone)]
pub struct KremoryMcpServer {
    mem: Arc<Memory>,
}

impl std::fmt::Debug for KremoryMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KremoryMcpServer").finish_non_exhaustive()
    }
}

impl KremoryMcpServer {
    /// Construct a bound server over an already-built `Memory`.
    pub fn new(mem: Arc<Memory>) -> Self {
        Self { mem }
    }
}

/// `KREMORY_MCP_DEBUG=1` — dump raw request/response JSON bodies to stderr
/// via `tracing::debug!` — a runtime-toggle debug switch.
/// Read once per call (cheap env lookup; the switch is not on any hot path).
fn debug_enabled() -> bool {
    std::env::var("KREMORY_MCP_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// `path` label for the `kremory_remember` counter — distinguishes the
/// mode-(c) "pinned" path (`skip_extraction` set, no Phase-2 LLM call at
/// all) from the default "extracted" path (Phase-2 LLM extraction runs).
///
/// The label reflects whether the LLM extraction path RAN, which is decided
/// by `skip_extraction` alone — `do_remember` calls `.skip_extraction()`
/// whenever `resolved.skip_extraction` is set, regardless of whether any
/// `structured_facts` were supplied. Gating the label on
/// `!structured_facts.is_empty()` as well would mislabel a
/// `skip_extraction=true` + no-facts call as "extracted" even though no LLM
/// call happened — a lying o11y counter.
fn remember_path_label(p: &RememberParams) -> &'static str {
    if p.skip_extraction {
        "pinned"
    } else {
        "extracted"
    }
}

fn wire_to_call_result<T: serde::Serialize>(
    wire: &T,
    tool: &str,
) -> Result<CallToolResult, ErrorData> {
    let value = serde_json::to_value(wire).map_err(|e| {
        ErrorData::internal_error(
            format!("{tool}: failed to serialize structured output: {e}"),
            None,
        )
    })?;
    Ok(CallToolResult::structured(value))
}

#[tool_router(server_handler)]
impl KremoryMcpServer {
    #[tool(
        name = "kremory_remember",
        description = "Remember new information into kremory's bi-temporal knowledge graph. Ingests a chat turn, document chunk, or note into a namespace (+ optional thread). Extracts entities/facts via LLM unless skip_extraction is set, in which case only caller-supplied structured_facts are pinned."
    )]
    pub async fn kremory_remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let start = Instant::now();
        let namespace = params.namespace.clone();
        let path = remember_path_label(&params);
        let span = tracing::info_span!(
            "kremory_mcp.tool",
            tool = "kremory_remember",
            namespace = %namespace,
            path,
        );
        let _enter = span.enter();

        if debug_enabled() {
            tracing::debug!(
                target: "kremory_mcp.raw_request",
                tool = "kremory_remember",
                request = %serde_json::to_string(&params).unwrap_or_default(),
            );
        }

        let result = handlers::do_remember(&self.mem, params).await;
        if let Ok(ref wire) = result {
            if debug_enabled() {
                tracing::debug!(
                    target: "kremory_mcp.raw_response",
                    tool = "kremory_remember",
                    response = %serde_json::to_string(&wire).unwrap_or_default(),
                );
            }
        }

        // Serialize BEFORE computing the outcome label so a serialization
        // failure (however unlikely for these output types) is reflected in
        // the `outcome` counter rather than recorded as "ok" while an error
        // is actually returned to the caller — the
        // counter must reflect the FINAL result, not an intermediate one.
        let final_result: Result<CallToolResult, ErrorData> = result
            .map_err(ErrorData::from)
            .and_then(|wire| wire_to_call_result(&wire, "kremory_remember"));
        let outcome = match &final_result {
            Ok(_) => "ok",
            Err(e) if e.code == ErrorCode::INVALID_PARAMS => "invalid_params",
            Err(_) => "internal_error",
        };
        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::debug!(duration_ms, outcome, path, "kremory_remember complete");
        metrics::counter!(
            "kremory_mcp.tool.calls",
            "tool" => "kremory_remember",
            "outcome" => outcome,
            "path" => path,
        )
        .increment(1);

        final_result
    }

    #[tool(
        name = "kremory_recall",
        description = "Search your memory — hybrid keyword + semantic + graph retrieval; find/recall what you know about X. Returns a prompt-ready rendered string by default (format=text), or raw entity-shaped results (format=structured)."
    )]
    pub async fn kremory_recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let start = Instant::now();
        let namespace = params.namespace.clone();
        let format_label = match params.format {
            RecallFormat::Text => "text",
            RecallFormat::Structured => "structured",
        };
        let span = tracing::info_span!(
            "kremory_mcp.tool",
            tool = "kremory_recall",
            namespace = %namespace,
            format = format_label,
        );
        let _enter = span.enter();

        if debug_enabled() {
            tracing::debug!(
                target: "kremory_mcp.raw_request",
                tool = "kremory_recall",
                request = %serde_json::to_string(&params).unwrap_or_default(),
            );
        }

        let result = handlers::do_recall(&self.mem, params).await;
        let outcome = result
            .as_ref()
            .map(|_| "ok")
            .unwrap_or_else(|e| e.outcome_label());
        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::debug!(
            duration_ms,
            outcome,
            format = format_label,
            "kremory_recall complete"
        );
        metrics::counter!(
            "kremory_mcp.tool.calls",
            "tool" => "kremory_recall",
            "outcome" => outcome,
        )
        .increment(1);

        match result {
            Ok(value) => {
                if debug_enabled() {
                    tracing::debug!(
                        target: "kremory_mcp.raw_response",
                        tool = "kremory_recall",
                        response = %value.to_string(),
                    );
                }
                Ok(CallToolResult::structured(value))
            }
            Err(e) => Err(e.into()),
        }
    }

    #[tool(
        name = "kremory_dream",
        description = "Run batch consolidation (the dream phase) over a namespace: community detection, cross-episode merge, supersession sweep, fact archival, type discovery/reclassification. Consumer-triggered, not a daemon."
    )]
    pub async fn kremory_dream(
        &self,
        Parameters(params): Parameters<DreamParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let start = Instant::now();
        let namespace = params.namespace.clone();
        let span = tracing::info_span!(
            "kremory_mcp.tool",
            tool = "kremory_dream",
            namespace = %namespace,
        );
        let _enter = span.enter();

        if debug_enabled() {
            tracing::debug!(
                target: "kremory_mcp.raw_request",
                tool = "kremory_dream",
                request = %serde_json::to_string(&params).unwrap_or_default(),
            );
        }

        let result = handlers::do_dream(&self.mem, params).await;
        if let Ok(ref wire) = result {
            if debug_enabled() {
                tracing::debug!(
                    target: "kremory_mcp.raw_response",
                    tool = "kremory_dream",
                    response = %serde_json::to_string(&wire).unwrap_or_default(),
                );
            }
        }

        // Serialize BEFORE computing the outcome label — see the matching
        // comment in `kremory_remember` above.
        let final_result: Result<CallToolResult, ErrorData> = result
            .map_err(ErrorData::from)
            .and_then(|wire| wire_to_call_result(&wire, "kremory_dream"));
        let outcome = match &final_result {
            Ok(_) => "ok",
            Err(e) if e.code == ErrorCode::INVALID_PARAMS => "invalid_params",
            Err(_) => "internal_error",
        };
        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::debug!(duration_ms, outcome, "kremory_dream complete");
        metrics::counter!(
            "kremory_mcp.tool.calls",
            "tool" => "kremory_dream",
            "outcome" => outcome,
        )
        .increment(1);

        final_result
    }

    #[tool(
        name = "kremory_list_mutations",
        description = "SEE the logged graph mutations in a namespace — the read-only inspect half of the reversible-mutations story. Lists the five LOGGED mutation kinds (entity merges, entity edits, entity deletes, fact deletes, and fact archivals written by dream) newest-first, each carrying a mutation_id you can pass to kremory_undo. Set entity_id to scope to one entity's history (includes already-undone mutations); otherwise lists namespace-wide LIVE (still-reversible) mutations by default. Read-only — never mutates. NOTE: this is still not a complete 'everything dream() changed' view — dream's fact supersession, community assignment and canonical-form writes are not logged and do not appear here; supersession is reversed via its own domain API (unsupersede). Fact archival IS logged and IS undoable from here."
    )]
    pub async fn kremory_list_mutations(
        &self,
        Parameters(params): Parameters<ListMutationsParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let start = Instant::now();
        let namespace = params.namespace.clone();
        let span = tracing::info_span!(
            "kremory_mcp.tool",
            tool = "kremory_list_mutations",
            namespace = %namespace,
        );
        let _enter = span.enter();

        if debug_enabled() {
            tracing::debug!(
                target: "kremory_mcp.raw_request",
                tool = "kremory_list_mutations",
                request = %serde_json::to_string(&params).unwrap_or_default(),
            );
        }

        let result = self.do_list_mutations(params).await;
        if let Ok(ref wire) = result {
            if debug_enabled() {
                tracing::debug!(
                    target: "kremory_mcp.raw_response",
                    tool = "kremory_list_mutations",
                    response = %serde_json::to_string(&wire).unwrap_or_default(),
                );
            }
        }

        // Serialize BEFORE computing the outcome label — see the matching
        // comment in `kremory_remember` above.
        let final_result: Result<CallToolResult, ErrorData> = result
            .map_err(ErrorData::from)
            .and_then(|wire| wire_to_call_result(&wire, "kremory_list_mutations"));
        let outcome = match &final_result {
            Ok(_) => "ok",
            Err(e) if e.code == ErrorCode::INVALID_PARAMS => "invalid_params",
            Err(_) => "internal_error",
        };
        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::debug!(duration_ms, outcome, "kremory_list_mutations complete");
        metrics::counter!(
            "kremory_mcp.tool.calls",
            "tool" => "kremory_list_mutations",
            "outcome" => outcome,
        )
        .increment(1);

        final_result
    }

    #[tool(
        name = "kremory_undo",
        description = "FIX — reverse a mutation dream() (or edit/delete calls) applied, by its mutation_id (get one from kremory_list_mutations). Unified dispatcher: routes to the correct reversal for entity merges (un-merge, restoring the split entity), entity edits (undo rename/retype), entity deletes (restore the entity + its archived facts), and fact deletes (restore the fact). Idempotent — undoing an already-undone mutation is a safe no-op. MUTATES the graph; not read-only."
    )]
    pub async fn kremory_undo(
        &self,
        Parameters(params): Parameters<UndoParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let start = Instant::now();
        let namespace = params.namespace.clone();
        let mutation_id = params.mutation_id;
        let span = tracing::info_span!(
            "kremory_mcp.tool",
            tool = "kremory_undo",
            namespace = %namespace,
            mutation_id,
        );
        let _enter = span.enter();

        if debug_enabled() {
            tracing::debug!(
                target: "kremory_mcp.raw_request",
                tool = "kremory_undo",
                request = %serde_json::to_string(&params).unwrap_or_default(),
            );
        }

        let result = self.do_undo(params).await;
        if let Ok(ref wire) = result {
            if debug_enabled() {
                tracing::debug!(
                    target: "kremory_mcp.raw_response",
                    tool = "kremory_undo",
                    response = %serde_json::to_string(&wire).unwrap_or_default(),
                );
            }
        }

        // Serialize BEFORE computing the outcome label — see the matching
        // comment in `kremory_remember` above.
        let final_result: Result<CallToolResult, ErrorData> = result
            .map_err(ErrorData::from)
            .and_then(|wire| wire_to_call_result(&wire, "kremory_undo"));
        let outcome = match &final_result {
            Ok(_) => "ok",
            Err(e) if e.code == ErrorCode::INVALID_PARAMS => "invalid_params",
            Err(_) => "internal_error",
        };
        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::debug!(duration_ms, outcome, "kremory_undo complete");
        metrics::counter!(
            "kremory_mcp.tool.calls",
            "tool" => "kremory_undo",
            "outcome" => outcome,
        )
        .increment(1);

        final_result
    }
}

// ────────────────────────────────────────────────────────────────────────
// Pure handler bodies for the two tools NOT shared with the REST bin
// (`kremory_list_mutations` / `kremory_undo` have no REST route in Step 2).
// `kremory_remember` / `kremory_recall` / `kremory_dream` delegate straight
// to `handlers::do_remember` / `do_recall` / `do_dream` from the `#[tool]`
// methods above — see `handlers.rs`.
// ────────────────────────────────────────────────────────────────────────

impl KremoryMcpServer {
    async fn do_list_mutations(
        &self,
        params: ListMutationsParams,
    ) -> Result<Vec<MutationRecordWire>, ToolError> {
        let resolved = params.resolve()?;

        let records = if let Some(entity_id) = resolved.entity_id {
            self.mem
                .mutation_history(entity_id)
                .in_namespace(resolved.namespace)
                .await?
        } else {
            let mut req = self
                .mem
                .list_mutations()
                .in_namespace(resolved.namespace)
                .include_undone(resolved.include_undone);
            if let Some(kind) = resolved.kind {
                req = req.kind(kind);
            }
            if let Some(since) = resolved.since {
                req = req.since(since);
            }
            req.await?
        };

        Ok(records.into_iter().map(Into::into).collect())
    }

    async fn do_undo(&self, params: UndoParams) -> Result<UndoOutcomeWire, ToolError> {
        let resolved = params.resolve()?;

        let outcome = self
            .mem
            .undo(resolved.mutation_id)
            .in_namespace(resolved.namespace)
            .execute()
            .await?;

        // `UndoOutcome` is `#[non_exhaustive]` — a future log-dispatchable kind
        // this crate hasn't caught up with yet is a version-skew bug, mapped
        // loudly to `internal_error` (never silently dropped/defaulted).
        UndoOutcomeWire::try_from(outcome).map_err(ToolError::Internal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the tool-name contract — these strings must remain stable for
    /// MCP clients wired against them.
    #[test]
    fn tool_name_contract_pins() {
        const EXPECTED_TOOLS: &[&str] = &[
            "kremory_remember",
            "kremory_recall",
            "kremory_dream",
            "kremory_list_mutations",
            "kremory_undo",
        ];
        assert_eq!(EXPECTED_TOOLS.len(), 5);
        for tool in EXPECTED_TOOLS {
            assert!(
                tool.starts_with("kremory_"),
                "tool name must be kremory_-prefixed: {tool}"
            );
        }
    }

    #[test]
    fn remember_path_label_reflects_skip_extraction() {
        use crate::params::{SourceKindWire, StructuredFactWire};

        let base = RememberParams {
            namespace: "ns".into(),
            thread: None,
            content: "x".into(),
            source_kind: Some(SourceKindWire::Chat),
            source_id: None,
            published_at: None,
            structured_facts: vec![],
            skip_extraction: false,
        };
        assert_eq!(remember_path_label(&base), "extracted");

        let mut skip_no_facts = base.clone();
        skip_no_facts.skip_extraction = true;
        assert_eq!(
            remember_path_label(&skip_no_facts),
            "pinned",
            "skip_extraction alone (no pinned facts) still runs NO LLM call — must label pinned"
        );

        let mut pinned = base.clone();
        pinned.skip_extraction = true;
        pinned.structured_facts = vec![StructuredFactWire {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
            valid_at: None,
            invalid_at: None,
        }];
        assert_eq!(remember_path_label(&pinned), "pinned");
    }

    // `ToolError::outcome_label` is now owned by `handlers.rs` (shared with
    // the REST bin) — see `handlers::tests::tool_error_outcome_labels`.
}

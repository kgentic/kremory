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
//! Both are unified internally as [`ToolError`] so each handler can record
//! an `outcome` label (`ok` | `invalid_params` | `internal_error`) for the
//! `kremory_mcp.tool.calls` counter before converting to the final
//! `ErrorData` (Critical Rule 19 — observability built in at emit, not
//! bolted on after).

pub mod conversions;
pub mod params;

use std::sync::Arc;
use std::time::Instant;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::{model::*, tool, tool_router};

use kremory::Memory;

use crate::conversions::ConversionError;
use crate::params::{
    DreamOutput, DreamParams, ListMutationsParams, MutationRecordWire, RecallFormat, RecallParams,
    RecallStructuredOutput, RecallTextOutput, RememberOutput, RememberParams, UndoOutcomeWire,
    UndoParams,
};

// ────────────────────────────────────────────────────────────────────────
// Internal error unification
// ────────────────────────────────────────────────────────────────────────

/// Unifies [`ConversionError`] and `kremory::MemoryError` so handler bodies
/// can record an observability outcome label BEFORE converting to the final
/// `rmcp::ErrorData` the JSON-RPC boundary expects.
#[derive(Debug)]
enum ToolError {
    InvalidParams(String),
    Internal(String),
}

impl ToolError {
    fn outcome_label(&self) -> &'static str {
        match self {
            ToolError::InvalidParams(_) => "invalid_params",
            ToolError::Internal(_) => "internal_error",
        }
    }
}

impl From<ConversionError> for ToolError {
    fn from(e: ConversionError) -> Self {
        ToolError::InvalidParams(e.to_string())
    }
}

impl From<kremory::MemoryError> for ToolError {
    fn from(e: kremory::MemoryError) -> Self {
        ToolError::Internal(e.to_string())
    }
}

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
/// via `tracing::debug!` (Critical Rule 19 runtime-toggle debug switch).
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
/// call happened — a lying o11y counter (Critical Rule 19 #9).
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

        let result = self.do_remember(params).await;
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
        // is actually returned to the caller (Critical Rule 19 #9 — the
        // counter must reflect the FINAL result, not an intermediate one).
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

        let result = self.do_recall(params).await;
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

        let result = self.do_dream(params).await;
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
        // comment in `kremory_remember` above (Critical Rule 19 #9).
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
        description = "SEE what dream() (or edit/delete calls) changed in a namespace — the read-only inspect half of the reversible-mutations story. Lists logged graph mutations (entity merges, entity edits, entity deletes, fact deletes) newest-first, each carrying a mutation_id you can pass to kremory_undo. Set entity_id to scope to one entity's history (includes already-undone mutations); otherwise lists namespace-wide LIVE (still-reversible) mutations by default. Read-only — never mutates."
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
        // comment in `kremory_remember` above (Critical Rule 19 #9).
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
        // comment in `kremory_remember` above (Critical Rule 19 #9).
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
// Pure handler bodies — facade builder chains. Kept separate from the
// `#[tool]`-annotated dispatch methods above so the observability/error-label
// wiring above doesn't have to be duplicated inside the facade call itself.
// ────────────────────────────────────────────────────────────────────────

impl KremoryMcpServer {
    async fn do_remember(&self, params: RememberParams) -> Result<RememberOutput, ToolError> {
        let resolved = params.resolve()?;

        let mut req = self
            .mem
            .remember(resolved.content)
            .in_namespace(resolved.namespace);
        if let Some(ts) = resolved.published_at {
            req = req.published_at(ts);
        }
        match (resolved.source_kind, resolved.source_id) {
            (Some(kind), id) => {
                let id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                req = req.from_source(id, kind);
            }
            (None, Some(id)) => {
                req = req.from_source(id, kremory::SourceKind::Chat);
            }
            (None, None) => {}
        }
        if !resolved.facts.is_empty() {
            req = req.with_facts(resolved.facts);
        }
        if resolved.skip_extraction {
            req = req.skip_extraction();
        }

        let commit = req.await?;
        Ok(RememberOutput::from(commit))
    }

    async fn do_recall(&self, params: RecallParams) -> Result<serde_json::Value, ToolError> {
        let resolved = params.resolve()?;

        let mut req = self
            .mem
            .recall(resolved.query)
            .in_namespace(resolved.namespace);
        if let Some(k) = resolved.k {
            req = req.k(k);
        }
        if let Some(as_of) = resolved.as_of {
            req = req.as_of(as_of);
        }

        match resolved.format {
            RecallFormat::Text => {
                let block = req.as_template(resolved.template).await?;
                serde_json::to_value(RecallTextOutput { block }).map_err(|e| {
                    ToolError::Internal(format!("failed to serialize recall text output: {e}"))
                })
            }
            RecallFormat::Structured => {
                let results = req.raw().await?;
                let count = results.len();
                let wire = RecallStructuredOutput {
                    results: results.into_iter().map(Into::into).collect(),
                    count,
                };
                serde_json::to_value(wire).map_err(|e| {
                    ToolError::Internal(format!(
                        "failed to serialize recall structured output: {e}"
                    ))
                })
            }
        }
    }

    async fn do_dream(&self, params: DreamParams) -> Result<DreamOutput, ToolError> {
        let resolved = params.resolve()?;

        let mut req = self
            .mem
            .dream()
            .in_namespace(resolved.namespace)
            .await_completion();
        if let Some(id) = resolved.batch_id {
            req = req.for_batch(id);
        }

        let summary = req.await?;
        Ok(DreamOutput::from(summary))
    }

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

    #[test]
    fn tool_error_outcome_labels() {
        assert_eq!(
            ToolError::InvalidParams("x".into()).outcome_label(),
            "invalid_params"
        );
        assert_eq!(
            ToolError::Internal("x".into()).outcome_label(),
            "internal_error"
        );
    }
}

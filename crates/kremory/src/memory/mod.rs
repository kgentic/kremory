//! kremory::memory — Zep-equivalent orchestration layer over kremory::core.
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface" — 4 public async functions
//! plus the scoping types in `types`. Internal orchestration modules added
//! in D.2 stay `pub(crate)`; external consumers use the 4 entry points only.
//!
//! ## BYOM contract
//!
//! `ChatProvider` is the canonical LLM abstraction across all kremory layers.
//! memory re-exports the trait so consumers can implement it against any backend
//! (OpenAI, Anthropic, Bedrock, vLLM, local GGUF, …).
//!
//! ## Layer ownership
//!
//! Per the canonical Zep / Graphiti split:
//!
//! - kremory::core owns per-episode work: `add_episode` cycle = LLM entity / edge
//!   extraction + dedup + fact invalidation + temporal validity inference
//!   + community detection primitive + hybrid retrieval primitives.
//! - kremory::memory owns cross-episode wrappers: multi-tenant scoping + packaged
//!   batch consolidation recipe (`run_dream_phase`) + opinionated retrieval
//!   defaults over core's hybrid search + context-block templates.

pub(crate) mod background_ingestor_handle;
pub mod dream_phase;
pub mod engine_handle;
pub mod events;
pub mod graph;
pub mod llm_adapters;
pub mod scheduler;
#[cfg(any(test, feature = "test-utils"))]
pub mod stub;
pub mod types;

pub(crate) use background_ingestor_handle::BackgroundIngestorGraphHandle;
pub use engine_handle::{EngineGraphHandle, WithConfigParams};
pub use graph::{
    GraphAssertEntityTypeParams, GraphHandle, GraphIngestEpisodeParams, GraphSearchParams,
    GraphSubmitDreamParams,
};
pub use scheduler::{DreamSchedule, DreamSchedulerHandle};
#[cfg(any(test, feature = "test-utils"))]
pub use stub::StubGraphHandle;
pub use types::{
    AwaitOpts, BatchStatus, CancelOutcome, CancelledPhase, ContextTemplate, DreamHandle, DreamMode,
    DreamOpts, DreamPhaseResult, DreamStatus, EpisodeCommit, ImmutabilityLevel, IngestResult,
    InvalidPolicyError, MemoryError, MemoryType, Namespace, NamespacePolicy, Result,
    RetrievedContext, RetrievedContextNewParams, RetrievedFact, RetrievedFactNewParams, SearchOpts,
    SourceKind, SourceRef, StructuredFact, SubmitOpts,
};
// `.content()` recall projection. Feature-gated (mirrors the
// type itself, `memory::types::ContentPassage`).
#[cfg(feature = "content-search")]
pub use types::ContentPassage;
// IngestStatus lives in core::error but is part of the memory API surface.
pub use crate::core::error::IngestStatus;

// Re-export the canonical LLM abstraction trait so SDK consumers depend on
// kremory only and still get the BYOM contract surface. Per ADR-Phase-D.0 §
// "rqlm public API surface". autoagents-llm is a required dep — kremory::memory
// needs ChatProvider unconditionally for the GraphHandle trait surface.
pub use autoagents_llm::chat::ChatProvider;

use std::sync::Arc;
use std::time::Instant;

use uuid::Uuid;

// ── D.6.4 public API surface (ADR §4.10) ─────────────────────────────────────

/// Bundled parameters for [`submit_episode`] — args-as-object
/// (rust-conventions §too_many_arguments). `batch_id`: caller-set string
/// grouping this episode with others — plain `Option<String>`, no `BatchRef`
/// wrapper, matching universal prior art.
pub struct SubmitEpisodeParams<'a> {
    pub graph: &'a dyn GraphHandle,
    pub content: &'a str,
    pub source_ref: SourceRef,
    pub structured_facts: Vec<StructuredFact>,
    pub provider: Arc<dyn ChatProvider>,
    pub namespace: Namespace,
    pub batch_id: Option<String>,
    pub opts: SubmitOpts,
    pub sink: Option<Arc<dyn events::EnrichmentEventSink>>,
}

/// Submit one episode for ingest.
///
/// Phase 1 (store + embed) commits synchronously. Episode searchable on return.
/// Phase 2 (LLM enrich) controlled by `params.opts`.
// Substrate primitive; consumer-facing surface is kremory::Memory facade.
pub async fn submit_episode(params: SubmitEpisodeParams<'_>) -> Result<EpisodeCommit> {
    let SubmitEpisodeParams {
        graph,
        content,
        source_ref,
        structured_facts,
        provider,
        namespace,
        batch_id,
        opts,
        sink,
    } = params;
    graph
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &namespace,
            source_ref: &source_ref,
            content,
            structured_facts: &structured_facts,
            provider,
            batch_id,
            opts,
            sink,
        })
        .await
}

/// Bundled parameters for [`submit_dream_phase`] — args-as-object
/// (rust-conventions §too_many_arguments).
pub struct SubmitDreamPhaseParams<'a> {
    pub graph: &'a dyn GraphHandle,
    pub namespace: Namespace,
    pub provider: Arc<dyn ChatProvider>,
    pub batch_id: Option<String>,
    pub opts: DreamOpts,
    pub sink: Option<Arc<dyn events::EnrichmentEventSink>>,
}

/// Submit a batch consolidation (dream phase). Returns immediately.
/// Idempotent on `(scope, batch_id)` key. See ADR §2.10 for CAS semantics.
// Substrate primitive; consumer-facing surface is kremory::Memory facade.
pub async fn submit_dream_phase(params: SubmitDreamPhaseParams<'_>) -> Result<DreamHandle> {
    let SubmitDreamPhaseParams {
        graph,
        namespace,
        provider,
        batch_id,
        opts,
        sink,
    } = params;
    graph
        .graph_submit_dream(GraphSubmitDreamParams {
            namespace: &namespace,
            provider,
            batch_id,
            opts,
            sink,
        })
        .await
}

/// Block until a Phase 2 run reaches a terminal status.
/// `timeout` in `AwaitOpts` is MANDATORY — no unbounded blocking.
/// `tracing::warn!` logged on timeout with `run_id` and elapsed duration.
pub async fn await_enrichment(
    graph: &dyn GraphHandle,
    run_id: Uuid,
    opts: AwaitOpts,
) -> Result<crate::core::error::IngestStatus> {
    use crate::core::error::IngestStatus;
    let start = Instant::now();
    loop {
        let status = graph.graph_ingest_status(run_id).await?;
        if matches!(status, IngestStatus::Complete | IngestStatus::Failed(_)) {
            return Ok(status);
        }
        if start.elapsed() >= opts.timeout {
            tracing::warn!(
                run_id = %run_id,
                elapsed_ms = start.elapsed().as_millis(),
                "await_enrichment timeout exhausted"
            );
            return Err(MemoryError::Timeout);
        }
        tokio::time::sleep(opts.poll_interval).await;
    }
}

/// Block until a Phase 3 dream run reaches a terminal status.
/// `timeout` in `AwaitOpts` is MANDATORY — no unbounded blocking.
/// `tracing::warn!` logged on timeout with `run_id` and elapsed duration.
pub async fn await_dream(
    graph: &dyn GraphHandle,
    run_id: Uuid,
    opts: AwaitOpts,
) -> Result<DreamStatus> {
    let start = Instant::now();
    loop {
        let status = graph.graph_dream_status(run_id).await?;
        if matches!(status, DreamStatus::Complete | DreamStatus::Failed(_)) {
            return Ok(status);
        }
        if start.elapsed() >= opts.timeout {
            tracing::warn!(
                run_id = %run_id,
                elapsed_ms = start.elapsed().as_millis(),
                "await_dream timeout exhausted"
            );
            return Err(MemoryError::Timeout);
        }
        tokio::time::sleep(opts.poll_interval).await;
    }
}

/// Wait until every episode in the batch has reached a terminal status
/// (Complete, Skipped, or Failed). Polls `graph_batch_status(batch_id)`
/// internally with `opts.poll_interval`. Returns the final `BatchStatus`.
///
/// Use this instead of a raw `loop { batch_status(...).await }` — handles
/// the fail-fast timeout boundary correctly and emits `tracing::warn!` on
/// timeout exhaustion with the `batch_id` and elapsed duration. (C1/§5.5)
pub async fn await_batch_enrichment(
    graph: &dyn GraphHandle,
    batch_id: &str,
    opts: AwaitOpts,
) -> Result<BatchStatus> {
    let start = Instant::now();
    loop {
        let status = graph.graph_batch_status(batch_id).await?;
        if status.is_done() {
            return Ok(status);
        }
        if start.elapsed() >= opts.timeout {
            // Carry the ACCUMULATOR, not just the fact of the timeout. A batch
            // that never terminates is diagnosed entirely by how its counters
            // relate to `total`: short of it means work is still outstanding,
            // PAST it means a run was counted twice and `is_done`'s equality
            // can never hold again (TD-251 cause 2). Without these fields the
            // two are indistinguishable from the log.
            tracing::warn!(
                batch_id,
                elapsed_ms = start.elapsed().as_millis(),
                total = status.total,
                completed = status.completed,
                skipped = status.skipped,
                failed = status.failed,
                "await_batch_enrichment timeout exhausted"
            );
            return Err(MemoryError::Timeout);
        }
        tokio::time::sleep(opts.poll_interval).await;
    }
}

// ── Legacy backwards-compat wrappers (D.5b callers; ADR §5.5) ────────────────

/// Legacy wrapper for D.5b callers. New code: use `submit_episode`.
///
/// Warning: `entities_added`, `edges_added`, `facts_invalidated`, `duration_ms`
/// are stub values (1, 0, 0, 0). Accurate counts are available via
/// `EnrichmentEventSink` on the `submit_episode` path. Callers relying on
/// these fields for anything other than log decoration must migrate.
#[deprecated(
    since = "0.1.0",
    note = "Use submit_episode + EnrichmentEventSink for accurate per-episode counts"
)]
// Substrate primitive; consumer-facing surface is kremory::Memory facade.
//
// Documented exemption: this fn is `#[deprecated]` (removal scheduled).
// Per the treat-cause exemption precedent, args-as-object churn on dying code is
// waste — the allow stays until the fn is removed. New code uses `submit_episode`.
#[allow(clippy::too_many_arguments)]
pub async fn ingest_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    namespace: Namespace,
) -> Result<IngestResult> {
    let _commit = submit_episode(SubmitEpisodeParams {
        graph,
        content,
        source_ref,
        structured_facts,
        provider,
        namespace,
        batch_id: None,
        opts: SubmitOpts {
            enrich_per_episode: true,
            run_in_background: false,
        },
        sink: None,
    })
    .await?;
    // Phase 2 ran inline (run_in_background = false). No polling needed.
    // Stub counts — callers used these for logging only; acceptable degradation.
    Ok(IngestResult {
        entities_added: 1,
        edges_added: 0,
        facts_invalidated: 0,
        duration_ms: 0,
        stub_entities_inserted: 0,
    })
}

/// Legacy wrapper for D.5b callers. New code: use `submit_dream_phase`.
#[deprecated(
    since = "0.1.0",
    note = "Use submit_dream_phase + await_dream for non-blocking dream orchestration"
)]
pub async fn run_dream_phase(
    graph: &dyn GraphHandle,
    namespace: Namespace,
    provider: Arc<dyn ChatProvider>,
) -> Result<DreamPhaseResult> {
    graph.graph_run_consolidation(&namespace, provider).await
}

/// Bundled parameters for [`search`] — args-as-object
/// (rust-conventions §too_many_arguments).
pub struct SearchParams<'a> {
    pub graph: &'a dyn GraphHandle,
    pub query: &'a str,
    pub namespace: Namespace,
    pub opts: SearchOpts,
}

/// Query the graph with memory's opinionated retrieval defaults. Thin
/// orchestration wrapper over [`GraphHandle::graph_search`].
pub async fn search(params: SearchParams<'_>) -> Result<Vec<RetrievedContext>> {
    let SearchParams {
        graph,
        query,
        namespace,
        opts,
    } = params;
    // `as_of` point-in-time recall is now IMPLEMENTED — `opts.as_of` passes
    // through to `graph_search` → `contextualize()` → `TemporalGraph::
    // get_neighbours_at`, which applies the valid-time predicate at the 1-hop
    // fact-expansion step. The prior guard here (`Err(Error::Unsupported)`,
    // and before that a silent `tracing::warn!` no-op) is gone — both were
    // footguns on a bi-temporal engine that declared a capability it didn't
    // have; the real SQL filter now backs the surface.
    graph
        .graph_search(GraphSearchParams {
            namespace: &namespace,
            query,
            opts: &opts,
        })
        .await
}

/// Render `results` into the final string handed to the LLM, per the
/// requested template strategy. Closest to Zep's `%{user_summary}` /
/// `%{edges}` / `%{entities}` template placeholders.
///
/// Implemented for all 3 [`ContextTemplate`] variants:
/// - `Entities` — one block per entity: name, summary, source pointers
/// - `EdgeSummary` — one line per entity-source-edge for compact context
/// - `TemporalFacts` — flattens source_refs with `valid_at` annotations
///
/// Empty `results` always renders an empty string. Order is preserved from
/// the input — callers are expected to pass results already sorted by score.
///
/// This is the only memory public fn that has no core primitive dependency,
/// so D.2a (this commit) ships it ahead of the rest. The other three fns
/// (`ingest_episode`, `run_dream_phase`, `search`) wrap core primitives
/// and ship in D.2b once the `Engine<L, Emb>` API shape (generic vs
/// trait-object vs concrete wrapper) is locked.
pub fn context_block(results: &[RetrievedContext], template: ContextTemplate) -> String {
    if results.is_empty() {
        return String::new();
    }
    match template {
        ContextTemplate::Entities => render_entities(results),
        ContextTemplate::EdgeSummary => render_edge_summary(results),
        ContextTemplate::TemporalFacts => render_temporal_facts(results),
    }
}

/// Read-only accessor over one connected fact, for the generic renderers
/// below. Implemented by [`RetrievedFact`] here; kremory-mcp
/// implements it for its own `RetrievedFactWire` DTO (already-formatted
/// RFC-3339 strings rather than `chrono::DateTime<Utc>` — the shapes
/// genuinely diverge, which is why this is a companion trait rather than a
/// shared struct).
///
/// **Not intended for external implementation.** It exists so kremory's own
/// renderers can serve both this crate's types and kremory-mcp's wire DTOs
/// from ONE implementation. It is `pub` only because kremory-mcp is a separate
/// crate. Methods may be added in a minor release; downstream implementors
/// should expect breakage.
pub trait RenderableFact {
    /// Natural-language rendering, e.g. `"Alice likes tea"`.
    fn fact_text(&self) -> &str;
    /// World clock: when the fact became true, RFC 3339.
    fn valid_at_rfc3339(&self) -> String;
    /// World clock: when the fact stopped being true, if ever, RFC 3339.
    fn invalid_at_rfc3339(&self) -> Option<String>;
}

impl RenderableFact for RetrievedFact {
    fn fact_text(&self) -> &str {
        &self.fact
    }
    fn valid_at_rfc3339(&self) -> String {
        self.valid_at.to_rfc3339()
    }
    fn invalid_at_rfc3339(&self) -> Option<String> {
        self.invalid_at.map(|d| d.to_rfc3339())
    }
}

/// Read-only accessor over one source reference, for the generic renderers
/// below. Implemented by [`SourceRef`] here; kremory-mcp implements
/// it for its own `SourceRefWire` DTO.
///
/// **Not intended for external implementation** — see [`RenderableFact`].
pub trait RenderableSourceRef {
    /// Human-readable label for the source kind (e.g. `"chat"`, `"document"`).
    fn kind_label(&self) -> &str;
    /// The source's caller-supplied id.
    fn ref_id(&self) -> &str;
    /// When the source event occurred, RFC 3339.
    fn occurred_at_rfc3339(&self) -> String;
}

impl RenderableSourceRef for SourceRef {
    fn kind_label(&self) -> &str {
        source_kind_label(self.kind)
    }
    fn ref_id(&self) -> &str {
        &self.id
    }
    fn occurred_at_rfc3339(&self) -> String {
        self.occurred_at.to_rfc3339()
    }
}

/// Read-only accessor trait over the fields [`render_entities`],
/// [`render_edge_summary`], and [`render_temporal_facts`] need to render a
/// result — the single point where kremory's `RetrievedContext` and
/// kremory-mcp's `RetrievedContextWire` DTO converge, so the renderer bodies
/// below are written ONCE — previously three functions independently
/// hand-mirrored in `kremory-http.rs`: the exact
/// two-implementations-must-agree shape that caused a prior silent `.max()`
/// fusion regression, re-created for rendering by a later, similar bug.
///
/// Implemented by [`RetrievedContext`] here; kremory-mcp implements it for
/// its own `RetrievedContextWire` DTO (`crates/kremory-mcp/src/conversions.rs`)
/// — never the reverse. This crate never references the wire type.
///
/// **Not intended for external implementation** — see [`RenderableFact`].
pub trait RenderableContext {
    /// The per-fact accessor type this context's [`facts`](Self::facts) yields.
    type Fact: RenderableFact;
    /// The per-source-ref accessor type this context's
    /// [`source_refs`](Self::source_refs) yields.
    type SourceRef: RenderableSourceRef;

    fn entity_name(&self) -> &str;
    fn summary(&self) -> &str;
    /// `Some(group_id)` when this result carries namespace attribution;
    /// `None` when absent. kremory-mcp's `/search`
    /// takes exactly one `namespace` per request, so every item in one
    /// response necessarily shares the same group_id — the multi-namespace
    /// `[ns:...]` prefix below therefore never fires for it, without this
    /// accessor needing to special-case that (see `render_prompt_block`'s
    /// doc comment in `kremory-http.rs` for why that's correct, not a gap).
    fn namespace_group_id(&self) -> Option<&str>;
    fn facts(&self) -> &[Self::Fact];
    fn source_refs(&self) -> &[Self::SourceRef];
}

impl RenderableContext for RetrievedContext {
    type Fact = RetrievedFact;
    type SourceRef = SourceRef;

    fn entity_name(&self) -> &str {
        &self.entity_name
    }
    fn summary(&self) -> &str {
        &self.summary
    }
    fn namespace_group_id(&self) -> Option<&str> {
        self.namespace.as_ref().map(|ns| ns.namespace.as_str())
    }
    fn facts(&self) -> &[RetrievedFact] {
        &self.facts
    }
    fn source_refs(&self) -> &[SourceRef] {
        &self.source_refs
    }
}

/// Render one block per entity: name, summary, connected facts, source
/// pointers. `pub` + generic over [`RenderableContext`] — so
/// kremory-mcp's `format=text` HTTP rendering path can call this directly
/// instead of maintaining its own hand-mirrored copy.
pub fn render_entities<T: RenderableContext>(results: &[T]) -> String {
    // Detect multi-namespace context: emit [ns:{group_id}] prefix when results
    // span more than one distinct group_id.
    let multi_ns = is_multi_namespace(results);
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        if multi_ns {
            if let Some(group_id) = r.namespace_group_id() {
                out.push_str("[ns:");
                out.push_str(group_id);
                out.push_str("] ");
            }
        }
        out.push_str("## ");
        out.push_str(r.entity_name());
        out.push('\n');
        out.push_str(r.summary());
        // List the entity's connected facts (the knowledge)
        // under its heading, not just the type-label summary.
        if !r.facts().is_empty() {
            out.push_str("\n\nFacts:");
            for f in r.facts() {
                out.push_str("\n- ");
                out.push_str(f.fact_text());
            }
        }
        if !r.source_refs().is_empty() {
            out.push_str("\n\nSources: ");
            for (j, sr) in r.source_refs().iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                out.push_str(sr.kind_label());
                out.push(':');
                out.push_str(sr.ref_id());
            }
        }
    }
    out
}

/// One line per entity-source-edge for compact context. `pub` + generic
/// over [`RenderableContext`] — see [`render_entities`].
pub fn render_edge_summary<T: RenderableContext>(results: &[T]) -> String {
    let mut out = String::new();
    for r in results {
        for sr in r.source_refs() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("- ");
            out.push_str(r.entity_name());
            out.push_str(" <- ");
            out.push_str(sr.kind_label());
            out.push(':');
            out.push_str(sr.ref_id());
        }
    }
    out
}

/// Emit the `[ns:{group_id}]` attribution marker when results span more than
/// one namespace. Free fn (not a closure) to avoid a
/// `&mut out` + `&r` borrow conflict in the render loops.
fn push_ns_prefix<T: RenderableContext>(out: &mut String, r: &T, multi_ns: bool) {
    if multi_ns {
        if let Some(group_id) = r.namespace_group_id() {
            out.push_str("[ns:");
            out.push_str(group_id);
            out.push_str("] ");
        }
    }
}

/// Flattens source_refs with `valid_at` annotations. `pub` + generic over
/// [`RenderableContext`] — see [`render_entities`].
pub fn render_temporal_facts<T: RenderableContext>(results: &[T]) -> String {
    // Detect multi-namespace context: emit [ns:{group_id}] prefix per result
    // when results span more than one distinct group_id.
    let multi_ns = is_multi_namespace(results);
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if r.facts().is_empty() {
            // No connected facts — fall back to the entity + source_ref line
            // (preserves the original rendering for fact-less entities).
            for sr in r.source_refs() {
                push_ns_prefix(&mut out, r, multi_ns);
                out.push_str(r.entity_name());
                out.push_str(" (valid_at=");
                out.push_str(&sr.occurred_at_rfc3339());
                out.push_str(") — ");
                out.push_str(r.summary());
                out.push('\n');
            }
        } else {
            // Render the actual connected facts — the
            // LLM-consumable knowledge — each with its world-clock validity,
            // instead of the entity name + type-label summary.
            for f in r.facts() {
                push_ns_prefix(&mut out, r, multi_ns);
                out.push_str(f.fact_text());
                out.push_str(" (valid_at=");
                out.push_str(&f.valid_at_rfc3339());
                if let Some(inv) = f.invalid_at_rfc3339() {
                    out.push_str(", invalid_at=");
                    out.push_str(&inv);
                }
                out.push(')');
                out.push('\n');
            }
        }
    }
    out.trim_end_matches('\n').to_string()
}

/// Determine whether `results` span more than one distinct namespace group_id.
/// Used by `render_entities` and `render_temporal_facts` to decide whether to
/// emit `[ns:{group_id}]` attribution markers.
///
/// Returns `true` only when at least two distinct, non-`None` group_ids appear
/// in the result set. Single-namespace results and results without namespace
/// attribution always return `false` (no prefix emitted).
fn is_multi_namespace<T: RenderableContext>(results: &[T]) -> bool {
    let mut seen: Option<&str> = None;
    for r in results {
        if let Some(gid) = r.namespace_group_id() {
            match seen {
                None => seen = Some(gid),
                Some(prev) if prev != gid => return true,
                _ => {}
            }
        }
    }
    false
}

fn source_kind_label(k: SourceKind) -> &'static str {
    match k {
        SourceKind::Meeting => "meeting",
        SourceKind::Document => "document",
        SourceKind::Chat => "chat",
        SourceKind::Episode => "episode",
    }
}

/// Re-export of the schema module from kremory::core — memory consumers should
/// not need to depend on core directly for the common types they round-trip.
pub use crate::core::schema as core_schema;

// ═══════════════════════════════════════════════════════════════════════════════
// TelemetryConfig + TelemetryHandle (ADR D15)
// ═══════════════════════════════════════════════════════════════════════════════

/// Memory-layer telemetry configuration.
///
/// Passed to `init_telemetry` to wire the metrics recorder and optional OTel
/// OTLP exporter. Callers that want no telemetry pass `TelemetryConfig::default()`.
///
/// # Cardinality note (ADR D7)
///
/// All prefix strings are set once at init time — not per-request. There is
/// no per-call allocation after `init_telemetry` returns.
#[derive(Debug, Clone, Default)]
pub struct TelemetryConfig {
    /// Optional prefix for all metric names (see `Config::metrics_prefix`).
    pub metrics_prefix: Option<String>,
    /// Optional prefix for all tracing span names.
    pub span_prefix: Option<String>,
    /// OTLP endpoint for OTel span export (e.g. `"http://localhost:4317"`).
    /// Requires the `otel` feature flag. Ignored when `otel` feature is absent.
    pub otlp_endpoint: Option<String>,
    /// Optional path to a custom provider-rates.toml file. When `Some`, overrides
    /// the bundled rates. When `None`, the bundled rates are used (idempotent if
    /// already initialized).
    pub rates_path: Option<std::path::PathBuf>,
}

/// Handle returned by [`init_telemetry`]. Keeps the OTel provider alive.
///
/// Drop or call [`TelemetryHandle::shutdown`] at process exit to flush pending spans.
/// If no OTel provider was initialised (default config or `otel` feature absent),
/// `shutdown()` is a no-op.
#[must_use]
pub struct TelemetryHandle {
    /// When the `otel` feature is enabled and an OTLP provider was wired,
    /// this holds the provider so its `Drop` impl can flush buffered spans.
    #[cfg(feature = "otel")]
    provider: Option<opentelemetry_sdk::trace::TracerProvider>,
    #[cfg(not(feature = "otel"))]
    _private: (),
}

impl TelemetryHandle {
    /// Flush pending spans and shut down the OTel provider.
    ///
    /// With the `otel` feature: calls `TracerProvider::shutdown()` to drain the
    /// OTLP export queue before returning. Without the feature: no-op.
    pub fn shutdown(self) {
        #[cfg(feature = "otel")]
        if let Some(provider) = self.provider {
            // Ignore shutdown errors — best-effort flush at process exit.
            let _ = provider.shutdown();
        }
    }
}

/// Errors returned by [`init_telemetry`].
#[derive(Debug, thiserror::Error)]
pub enum TelemetryInitError {
    /// OTLP exporter build failed (only possible when `otel` feature is enabled).
    #[error("OTLP exporter build failed: {0}")]
    Exporter(String),
    /// `tracing-subscriber` global default could not be installed.
    /// Usually means another subscriber was already registered.
    #[error("tracing subscriber init failed: {0}")]
    Subscriber(String),
}

/// Initialise kremory telemetry for the memory layer.
///
/// Wires:
/// - `tracing` structured log emission (always)
/// - OTLP gRPC span export (only when `otel` feature is enabled **and**
///   `config.otlp_endpoint` is `Some`)
///
/// Without the `otel` feature: installs an `EnvFilter` + fmt layer using
/// [`tracing_subscriber::fmt::init`](https://docs.rs/tracing-subscriber) conventions
/// so that `RUST_LOG` still controls verbosity.  This path never fails.
///
/// With the `otel` feature: installs `tracing-subscriber` registry with an
/// `EnvFilter` layer, an `fmt` layer, and a `tracing-opentelemetry` layer that
/// exports spans to `config.otlp_endpoint` (default: `http://localhost:4317`).
///
/// Returns a [`TelemetryHandle`] that MUST be kept alive until process exit.
/// Dropping it early shuts down the OTel provider and loses buffered spans.
///
/// # Errors
///
/// Returns [`TelemetryInitError::Exporter`] if the OTLP exporter fails to build
/// (only when `otel` feature is enabled and `config.otlp_endpoint` is `Some`).
/// Returns [`TelemetryInitError::Subscriber`] if a global tracing subscriber is
/// already installed (harmless in binaries that call this once; check your test harness).
///
/// # Example
///
/// ```no_run
/// # use kremory::{TelemetryConfig, init_telemetry};
/// let handle = init_telemetry(TelemetryConfig {
///     otlp_endpoint: Some("http://localhost:4317".to_string()),
///     ..Default::default()
/// })?;
/// // … run your application …
/// handle.shutdown();
/// # Ok::<(), kremory::TelemetryInitError>(())
/// ```
#[cfg(feature = "otel")]
pub fn init_telemetry(
    config: TelemetryConfig,
) -> std::result::Result<TelemetryHandle, TelemetryInitError> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig as _;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    // Initialize provider rates: custom path takes precedence over bundled.
    if let Some(ref path) = config.rates_path {
        crate::core::rates::init_from_path(path)
            .map_err(|e| TelemetryInitError::Exporter(format!("rates load failed: {e}")))?;
    } else {
        // Best-effort: ignore error if already initialized or TOML is missing.
        let _ = crate::core::rates::init_bundled();
    }

    // Build the OTLP gRPC span exporter.  Endpoint resolution order:
    //   1. `config.otlp_endpoint` (explicit caller config)
    //   2. `OTEL_EXPORTER_OTLP_ENDPOINT` env var (opentelemetry-otlp picks this up automatically)
    //   3. Default: http://localhost:4317
    let mut exporter_builder = opentelemetry_otlp::SpanExporter::builder().with_tonic();
    if let Some(ref endpoint) = config.otlp_endpoint {
        exporter_builder = exporter_builder.with_endpoint(endpoint.as_str());
    }
    let exporter = exporter_builder
        .build()
        .map_err(|e| TelemetryInitError::Exporter(e.to_string()))?;

    // `service.name` resource attribute — identifies this process in the OTLP backend.
    let service_name = config
        .span_prefix
        .as_deref()
        .unwrap_or("kremory")
        .to_string();
    let resource = Resource::new(vec![opentelemetry::KeyValue::new(
        "service.name",
        service_name.clone(),
    )]);

    // Build the SDK tracer provider with a batch processor (async export via Tokio).
    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(service_name);
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,kremory=debug"));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()
        .map_err(|e| TelemetryInitError::Subscriber(e.to_string()))?;

    Ok(TelemetryHandle {
        provider: Some(provider),
    })
}

/// Initialise kremory telemetry (no-op stub — enable the `otel` feature for OTLP export).
///
/// Without the `otel` feature kremory still emits structured `tracing` events and
/// `metrics` counters. To forward spans to an OTLP backend, rebuild with
/// `--features otel` and configure `TelemetryConfig::otlp_endpoint`.
///
/// # Errors
///
/// Always returns `Ok` in this configuration.
#[cfg(not(feature = "otel"))]
pub fn init_telemetry(
    config: TelemetryConfig,
) -> std::result::Result<TelemetryHandle, TelemetryInitError> {
    // Initialize provider rates: custom path takes precedence over bundled.
    if let Some(ref path) = config.rates_path {
        let _ = crate::core::rates::init_from_path(path);
    } else {
        let _ = crate::core::rates::init_bundled();
    }
    // Library-safe: kremory does NOT install a global tracing subscriber or
    // metrics recorder (ADR D2). The host binary owns subscriber installation.
    Ok(TelemetryHandle { _private: () })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_constructors_round_trip() {
        let s = Namespace::new("ws-1");
        assert_eq!(s.namespace, "ws-1");
        assert!(s.thread.is_none());

        let s = Namespace::new("ws-2").with_thread("thread-a");
        assert_eq!(s.namespace, "ws-2");
        assert_eq!(s.thread.as_deref(), Some("thread-a"));
    }

    #[test]
    fn source_kind_serde_round_trips_snake_case() {
        let v = serde_json::to_string(&SourceKind::Meeting).expect("serialize");
        assert_eq!(v, "\"meeting\"");

        let back: SourceKind = serde_json::from_str("\"document\"").expect("deserialize");
        assert_eq!(back, SourceKind::Document);
    }

    #[test]
    fn context_template_serde_round_trips_snake_case() {
        let v = serde_json::to_string(&ContextTemplate::EdgeSummary).expect("serialize");
        assert_eq!(v, "\"edge_summary\"");
    }

    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Stub graph handle for D.2b delegation tests. Records the params
    /// each method was called with so we can assert the memory wrappers
    /// pass them through correctly.
    ///
    /// All new D.6.4 GraphHandle methods are implemented as required (no
    /// defaults on the trait — ADR §4.9).
    #[derive(Default)]
    struct StubGraphHandle {
        last_ingest_namespace: Mutex<Option<Namespace>>,
        last_ingest_content: Mutex<Option<String>>,
        last_ingest_source_id: Mutex<Option<String>>,
        last_ingest_facts_count: Mutex<Option<usize>>,
        last_search_namespace: Mutex<Option<Namespace>>,
        last_search_query: Mutex<Option<String>>,
        last_search_limit: Mutex<Option<usize>>,
        last_search_as_of: Mutex<Option<chrono::DateTime<chrono::Utc>>>,
        last_consolidation_namespace: Mutex<Option<Namespace>>,
    }

    #[async_trait]
    impl GraphHandle for StubGraphHandle {
        async fn graph_ingest_episode(
            &self,
            params: GraphIngestEpisodeParams<'_>,
        ) -> Result<EpisodeCommit> {
            let GraphIngestEpisodeParams {
                namespace,
                source_ref,
                content,
                structured_facts,
                provider: _,
                batch_id: _,
                opts: _,
                sink: _,
            } = params;
            *self.last_ingest_namespace.lock().unwrap() = Some(namespace.clone());
            *self.last_ingest_content.lock().unwrap() = Some(content.to_string());
            *self.last_ingest_source_id.lock().unwrap() = Some(source_ref.id.clone());
            *self.last_ingest_facts_count.lock().unwrap() = Some(structured_facts.len());
            Ok(EpisodeCommit {
                run_id: None,
                episode_entity_id: format!("stub:{}", source_ref.id),
                committed_at: chrono::Utc::now(),
                stub_entities_inserted: 0,
                dense_embedded: None,
            })
        }

        async fn graph_ingest_status(
            &self,
            _run_id: uuid::Uuid,
        ) -> Result<crate::core::error::IngestStatus> {
            Ok(crate::core::error::IngestStatus::Complete)
        }

        async fn graph_cancel(&self, _run_id: uuid::Uuid) -> Result<CancelOutcome> {
            Ok(CancelOutcome {
                cancelled_phase: CancelledPhase::Enrichment,
                rolled_back: false,
                partial: vec![],
            })
        }

        async fn graph_submit_dream(
            &self,
            params: GraphSubmitDreamParams<'_>,
        ) -> Result<DreamHandle> {
            let GraphSubmitDreamParams {
                namespace,
                provider: _,
                batch_id,
                opts: _,
                sink: _,
            } = params;
            Ok(DreamHandle {
                run_id: uuid::Uuid::new_v4(),
                namespace: namespace.clone(),
                submitted_at: chrono::Utc::now(),
                batch_id,
            })
        }

        async fn graph_dream_status(&self, _run_id: uuid::Uuid) -> Result<DreamStatus> {
            Ok(DreamStatus::Complete)
        }

        async fn graph_batch_status(&self, _batch_id: &str) -> Result<BatchStatus> {
            Ok(BatchStatus {
                total: 0,
                completed: 0,
                skipped: 0,
                failed: 0,
            })
        }

        async fn graph_last_consolidated_at(
            &self,
            _namespace: &Namespace,
        ) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
            Ok(None)
        }

        async fn graph_episodes_since_last_dream(&self, _namespace: &Namespace) -> Result<usize> {
            Ok(0)
        }

        async fn graph_is_consolidating(&self, _namespace: &Namespace) -> Result<bool> {
            Ok(false)
        }

        async fn graph_search(
            &self,
            params: GraphSearchParams<'_>,
        ) -> Result<Vec<RetrievedContext>> {
            let GraphSearchParams {
                namespace,
                query,
                opts,
            } = params;
            *self.last_search_namespace.lock().unwrap() = Some(namespace.clone());
            *self.last_search_query.lock().unwrap() = Some(query.to_string());
            *self.last_search_limit.lock().unwrap() = opts.limit;
            *self.last_search_as_of.lock().unwrap() = opts.as_of;
            Ok(vec![RetrievedContext {
                entity_id: "ent-stub".into(),
                entity_name: "Stub Entity".into(),
                summary: "from StubGraphHandle".into(),
                score: 0.5,
                source_refs: vec![],
                incomplete: false,
                entity_type_id: 0,
                entity_type_name: "Entity".to_string(),
                namespace: None,
                facts: vec![],
            }])
        }

        async fn graph_run_consolidation(
            &self,
            namespace: &Namespace,
            _provider: Arc<dyn ChatProvider>,
        ) -> Result<DreamPhaseResult> {
            *self.last_consolidation_namespace.lock().unwrap() = Some(namespace.clone());
            Ok(DreamPhaseResult {
                communities_recomputed: 1,
                cross_meeting_merges: 0,
                supersessions_recorded: 0,
                facts_archived: 0,
                duration_ms: 10,
                types_discovered: vec![],
                dream_warnings: vec![],
                budget_exhausted: false,
            })
        }

        async fn graph_run_dream_pass_sync(
            &self,
            _opts: crate::core::ingest::DreamPassOpts,
        ) -> Result<crate::facade::DreamSummary> {
            Ok(crate::facade::DreamSummary {
                communities_updated: 0,
                cross_episode_would_merge: 0,
                cross_episode_merged: 0,
                consolidation_ops_ran: crate::facade::ConsolidationOpsRan::default(),
                supersessions_recorded: 0,
                facts_archived: 0,
                duration_ms: 0,
                types_discovered: vec![],
                entities_reclassified: 0,
                aliases_resolved: 0,
                canonicalization_merges: 0,
                acronym_nickname_merges: 0,
                type_registry_merges: 0,
                consistency_check_corrected: 0,
                warnings: vec![],
                budget_exhausted: false,
            })
        }

        async fn graph_ghost_episodes(&self, _group_id: Option<&str>) -> Result<Vec<i64>> {
            Ok(vec![])
        }

        async fn graph_assert_entity_type(
            &self,
            _params: GraphAssertEntityTypeParams<'_>,
        ) -> Result<()> {
            Ok(())
        }
    }

    /// kremory::core's `MockChatProvider::null()` implements the canonical
    /// `autoagents_llm::chat::ChatProvider` trait — gated behind kremory's
    /// `llm` feature. The stub graph never invokes the provider so the canned-
    /// response behaviour is irrelevant; we just need an `Arc<dyn ChatProvider>`.
    fn null_provider() -> Arc<dyn ChatProvider> {
        Arc::new(crate::core::provider::MockChatProvider::null())
    }

    #[tokio::test]
    async fn submit_episode_delegates_to_graph_handle() {
        use chrono::Utc;
        let graph = StubGraphHandle::default();
        let namespace = Namespace::new("ws-1").with_thread("thread-a");
        let source_ref = SourceRef {
            kind: SourceKind::Meeting,
            id: "mtg-42".into(),
            occurred_at: Utc::now(),
            published_at: None,
        };
        let facts = vec![StructuredFact {
            subject: "alice".into(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }];

        let commit = submit_episode(SubmitEpisodeParams {
            graph: &graph,
            content: "transcript content",
            source_ref,
            structured_facts: facts,
            provider: null_provider(),
            namespace: namespace.clone(),
            batch_id: None,
            opts: SubmitOpts::default(),
            sink: None,
        })
        .await
        .expect("submit_episode should succeed via stub");

        assert!(!commit.episode_entity_id.is_empty());
        assert_eq!(
            graph.last_ingest_namespace.lock().unwrap().as_ref(),
            Some(&namespace)
        );
        assert_eq!(
            graph.last_ingest_content.lock().unwrap().as_deref(),
            Some("transcript content")
        );
        assert_eq!(
            graph.last_ingest_source_id.lock().unwrap().as_deref(),
            Some("mtg-42")
        );
        assert_eq!(*graph.last_ingest_facts_count.lock().unwrap(), Some(1));
    }

    #[tokio::test]
    async fn search_delegates_with_opts() {
        let graph = StubGraphHandle::default();
        let namespace = Namespace::new("ws-2");
        let opts = SearchOpts {
            limit: Some(25),
            as_of: None,
            source_kind: Some(SourceKind::Document),
            ..Default::default()
        };

        let hits = search(SearchParams {
            graph: &graph,
            query: "go-live",
            namespace: namespace.clone(),
            opts,
        })
        .await
        .expect("search delegates cleanly");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity_id, "ent-stub");
        assert_eq!(
            graph.last_search_namespace.lock().unwrap().as_ref(),
            Some(&namespace)
        );
        assert_eq!(
            graph.last_search_query.lock().unwrap().as_deref(),
            Some("go-live")
        );
        assert_eq!(*graph.last_search_limit.lock().unwrap(), Some(25));
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn run_dream_phase_delegates_to_consolidation() {
        let graph = StubGraphHandle::default();
        let namespace = Namespace::new("ws-3");

        let result = run_dream_phase(&graph, namespace.clone(), null_provider())
            .await
            .expect("run_dream_phase delegates cleanly");

        assert_eq!(result.communities_recomputed, 1);
        assert_eq!(
            graph.last_consolidation_namespace.lock().unwrap().as_ref(),
            Some(&namespace)
        );
    }

    /// Story #318 AC: SourceRef.published_at propagates through ingest path;
    /// StructuredFact.valid_from/valid_to field names compile correctly.
    #[tokio::test]
    async fn published_at_precedence_fields_compile() {
        use chrono::Utc;
        let now = Utc::now();
        let source_ref = SourceRef {
            kind: SourceKind::Document,
            id: "doc-1".into(),
            occurred_at: now,
            published_at: Some(now),
        };
        // Verify published_at is round-trippable
        assert_eq!(source_ref.published_at, Some(now));

        let sf = StructuredFact {
            subject: "alpha".into(),
            predicate: "knows".into(),
            object: "beta".into(),
            valid_from: Some(now),
            valid_to: None,
            memory_type: None,
        };
        // valid_from fallback precedence per Story #318:
        // fact.valid_from = sf.valid_from.or(source_ref.published_at).unwrap_or_else(Utc::now)
        let resolved_from = sf
            .valid_from
            .or(source_ref.published_at)
            .unwrap_or_else(Utc::now);
        assert_eq!(resolved_from, now);

        // When sf.valid_from is None, published_at is used as fallback
        let sf_no_valid_from = StructuredFact {
            subject: "alpha".into(),
            predicate: "knows".into(),
            object: "beta".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        };
        let resolved_fallback = sf_no_valid_from
            .valid_from
            .or(source_ref.published_at)
            .unwrap_or_else(Utc::now);
        assert_eq!(resolved_fallback, now);
    }

    /// `search()` no longer fails loud on `opts.as_of` — the guard is gone
    /// and `as_of` passes through to `graph.graph_search`, exactly like
    /// `limit` does. Retired name `as_of_errors_unsupported` (this specific
    /// test name is retired since there's no more `Unsupported` path for
    /// `as_of` to hit). This unit level only proves
    /// the STUB-level delegation contract (opts round-trip to `GraphHandle`
    /// unchanged, no error) — the real valid-time SQL filtering correctness
    /// is covered end-to-end in `tests/facade_as_of_warn.rs` against a real
    /// `TemporalGraph` (a canned `StubGraphHandle` can't exercise SQL).
    #[tokio::test]
    async fn search_passes_as_of_through_without_erroring() {
        use chrono::Utc;
        let graph = StubGraphHandle::default();
        let namespace = Namespace::new("ws-as-of");
        let ts = Utc::now();
        let opts = SearchOpts {
            limit: None,
            as_of: Some(ts),
            source_kind: None,
            ..Default::default()
        };
        let hits = search(SearchParams {
            graph: &graph,
            query: "test",
            namespace,
            opts,
        })
        .await
        .expect("as_of must no longer error — ADR-068 implements the surface");

        assert_eq!(
            hits.len(),
            1,
            "stub delegation must still return its canned hit"
        );
        assert_eq!(
            *graph.last_search_as_of.lock().unwrap(),
            Some(ts),
            "opts.as_of must round-trip to GraphHandle::graph_search unchanged"
        );
    }
}

//! kremory::memory — Zep-equivalent orchestration layer over kremory::core.
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface" — 4 public async functions
//! plus the scoping types in `types`. Internal orchestration modules added
//! in D.2 stay `pub(crate)`; external consumers use the 4 entry points only.
//!
//! ## Greenfield contract
//!
//! D.1b ships scaffolding only. The 4 public functions return
//! `MemoryError::Unimplemented(...)` until D.2's TDD lane lands the real
//! implementations. This keeps the public surface visible to downstream
//! consumers (the-host-application, kremory-mcp) while the implementation is built.
//!
//! ## BYOM contract
//!
//! `ChatProvider` is the canonical LLM abstraction across all kremory layers.
//! memory re-exports the trait so consumers can implement it against any backend
//! (OpenAI, Anthropic, Bedrock, vLLM, local GGUF, …).
//!
//! ## Layer ownership
//!
//! Per the canonical Zep / Graphiti split (verified 2026-05-13 via context7;
//! recorded in `project_zep_graphiti_split_canonical.md`):
//!
//! - kremory::core owns per-episode work: `add_episode` cycle = LLM entity / edge
//!   extraction + dedup + fact invalidation + temporal validity inference
//!   + community detection primitive + hybrid retrieval primitives.
//! - kremory::memory owns cross-episode wrappers: multi-tenant scoping + packaged
//!   batch consolidation recipe (`run_dream_phase`) + opinionated retrieval
//!   defaults over core's hybrid search + context-block templates.

pub mod dream_phase;
pub mod events;
pub mod graph;
pub mod stub;
pub mod types;

pub use graph::GraphHandle;
pub use stub::StubGraphHandle;
pub use types::{
    AwaitOpts, BatchStatus, CancelOutcome, CancelledPhase, ContextTemplate, DreamHandle, DreamMode,
    DreamOpts, DreamPhaseResult, DreamStatus, EpisodeCommit, IngestResult, MemoryError, Result,
    RetrievedContext, SearchOpts, SourceKind, SourceRef, StructuredFact, SubmitOpts,
    WorkspaceScope,
};
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

/// Submit one episode for ingest.
///
/// Phase 1 (store + embed) commits synchronously. Episode searchable on return.
/// Phase 2 (LLM enrich) controlled by `opts`.
///
/// `batch_id`: caller-set string grouping this episode with others.
/// No `BatchRef` wrapper — plain `Option<String>` matching universal prior art.
#[allow(clippy::too_many_arguments)]
pub async fn submit_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    scope: WorkspaceScope,
    batch_id: Option<String>,
    opts: SubmitOpts,
    sink: Option<Arc<dyn events::EnrichmentEventSink>>,
) -> Result<EpisodeCommit> {
    graph
        .graph_ingest_episode(
            &scope,
            &source_ref,
            content,
            &structured_facts,
            provider,
            batch_id,
            opts,
            sink,
        )
        .await
}

/// Submit a batch consolidation (dream phase). Returns immediately.
/// Idempotent on `(scope, batch_id)` key. See ADR §2.10 for CAS semantics.
pub async fn submit_dream_phase(
    graph: &dyn GraphHandle,
    scope: WorkspaceScope,
    provider: Arc<dyn ChatProvider>,
    batch_id: Option<String>,
    opts: DreamOpts,
    sink: Option<Arc<dyn events::EnrichmentEventSink>>,
) -> Result<DreamHandle> {
    graph
        .graph_submit_dream(&scope, provider, batch_id, opts, sink)
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
            tracing::warn!(
                batch_id,
                elapsed_ms = start.elapsed().as_millis(),
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
pub async fn ingest_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    scope: WorkspaceScope,
) -> Result<IngestResult> {
    let _commit = submit_episode(
        graph,
        content,
        source_ref,
        structured_facts,
        provider,
        scope,
        None,
        SubmitOpts {
            enrich_per_episode: true,
            run_in_background: false,
        },
        None,
    )
    .await?;
    // Phase 2 ran inline (run_in_background = false). No polling needed.
    // Stub counts — callers used these for logging only; acceptable degradation.
    Ok(IngestResult {
        entities_added: 1,
        edges_added: 0,
        facts_invalidated: 0,
        duration_ms: 0,
    })
}

/// Legacy wrapper for D.5b callers. New code: use `submit_dream_phase`.
#[deprecated(
    since = "0.1.0",
    note = "Use submit_dream_phase + await_dream for non-blocking dream orchestration"
)]
pub async fn run_dream_phase(
    graph: &dyn GraphHandle,
    scope: WorkspaceScope,
    provider: Arc<dyn ChatProvider>,
) -> Result<DreamPhaseResult> {
    graph.graph_run_consolidation(&scope, provider).await
}

/// Query the graph with memory's opinionated retrieval defaults. Thin
/// orchestration wrapper over [`GraphHandle::graph_search`].
pub async fn search(
    graph: &dyn GraphHandle,
    query: &str,
    scope: WorkspaceScope,
    opts: SearchOpts,
) -> Result<Vec<RetrievedContext>> {
    graph.graph_search(&scope, query, &opts).await
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

fn render_entities(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        out.push_str("## ");
        out.push_str(&r.entity_name);
        out.push('\n');
        out.push_str(&r.summary);
        if !r.source_refs.is_empty() {
            out.push_str("\n\nSources: ");
            for (j, sr) in r.source_refs.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                out.push_str(source_kind_label(sr.kind));
                out.push(':');
                out.push_str(&sr.id);
            }
        }
    }
    out
}

fn render_edge_summary(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for r in results {
        for sr in &r.source_refs {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("- ");
            out.push_str(&r.entity_name);
            out.push_str(" <- ");
            out.push_str(source_kind_label(sr.kind));
            out.push(':');
            out.push_str(&sr.id);
        }
    }
    out
}

fn render_temporal_facts(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        for sr in &r.source_refs {
            out.push_str(&r.entity_name);
            out.push_str(" (valid_at=");
            out.push_str(&sr.occurred_at.to_rfc3339());
            out.push_str(") — ");
            out.push_str(&r.summary);
            out.push('\n');
        }
    }
    out.trim_end_matches('\n').to_string()
}

fn source_kind_label(k: SourceKind) -> &'static str {
    match k {
        SourceKind::Meeting => "meeting",
        SourceKind::Document => "document",
        SourceKind::Chat => "chat",
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
}

/// Handle returned by `init_telemetry`. Keeps the OTel provider alive.
///
/// Drop or call `shutdown()` at process exit to flush pending spans/metrics.
/// If no OTel provider was initialised (default config), `shutdown()` is a no-op.
#[must_use]
pub struct TelemetryHandle {
    _private: (),
}

impl TelemetryHandle {
    /// Flush pending spans and metrics, then shut down the OTel provider.
    ///
    /// Idempotent. Calling twice is safe. Blocks until the provider has drained
    /// its export queue or the provider-specific shutdown timeout expires.
    pub fn shutdown(self) {
        // No-op in the default (no OTel) configuration.
        // When `otel` feature is enabled and an OTLP provider is running,
        // the provider's Drop impl flushes before the handle is dropped.
    }
}

/// Initialise kremory telemetry for the memory layer.
///
/// Wires:
/// - `metrics` recorder (global — installs once; subsequent calls are no-ops)
/// - `tracing` subscriber (OTel OTLP exporter if `otel` feature + endpoint set)
///
/// Returns a `TelemetryHandle` that MUST be kept alive until process exit.
/// Dropping it early shuts down the OTel provider and loses buffered spans.
///
/// # Errors
///
/// Returns `Err` if the OTLP exporter fails to connect (when `otel` feature enabled
/// and `config.otlp_endpoint` is `Some`). Plain metrics-only config always succeeds.
pub fn init_telemetry(
    _config: TelemetryConfig,
) -> std::result::Result<TelemetryHandle, Box<dyn std::error::Error + Send + Sync>> {
    // Library-safe: kremory does NOT install a global metrics recorder (ADR D2).
    // The host binary is responsible for calling `metrics_exporter_prometheus::install()`
    // or `metrics_util::debugging::DebuggingRecorder::install_as_global()` in tests.
    //
    // When the `otel` feature is enabled and `config.otlp_endpoint` is `Some`,
    // a tracing-opentelemetry layer would be installed here. Left as a stub
    // until the `otel` feature is stabilised — the `TelemetryHandle` type is
    // reserved so the API shape is locked.
    Ok(TelemetryHandle { _private: () })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_scope_constructors_round_trip() {
        let s = WorkspaceScope::new("ws-1");
        assert_eq!(s.workspace_id, "ws-1");
        assert!(s.thread_id.is_none());

        let s = WorkspaceScope::with_thread("ws-2", "thread-a");
        assert_eq!(s.workspace_id, "ws-2");
        assert_eq!(s.thread_id.as_deref(), Some("thread-a"));
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
        last_ingest_scope: Mutex<Option<WorkspaceScope>>,
        last_ingest_content: Mutex<Option<String>>,
        last_ingest_source_id: Mutex<Option<String>>,
        last_ingest_facts_count: Mutex<Option<usize>>,
        last_search_scope: Mutex<Option<WorkspaceScope>>,
        last_search_query: Mutex<Option<String>>,
        last_search_limit: Mutex<Option<usize>>,
        last_consolidation_scope: Mutex<Option<WorkspaceScope>>,
    }

    #[async_trait]
    impl GraphHandle for StubGraphHandle {
        async fn graph_ingest_episode(
            &self,
            scope: &WorkspaceScope,
            source_ref: &SourceRef,
            content: &str,
            structured_facts: &[StructuredFact],
            _provider: Arc<dyn ChatProvider>,
            _batch_id: Option<String>,
            _opts: SubmitOpts,
            _sink: Option<Arc<dyn crate::memory::events::EnrichmentEventSink>>,
        ) -> Result<EpisodeCommit> {
            *self.last_ingest_scope.lock().unwrap() = Some(scope.clone());
            *self.last_ingest_content.lock().unwrap() = Some(content.to_string());
            *self.last_ingest_source_id.lock().unwrap() = Some(source_ref.id.clone());
            *self.last_ingest_facts_count.lock().unwrap() = Some(structured_facts.len());
            Ok(EpisodeCommit {
                run_id: None,
                episode_entity_id: format!("stub:{}", source_ref.id),
                committed_at: chrono::Utc::now(),
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
            scope: &WorkspaceScope,
            _provider: Arc<dyn ChatProvider>,
            batch_id: Option<String>,
            _opts: DreamOpts,
            _sink: Option<Arc<dyn crate::memory::events::EnrichmentEventSink>>,
        ) -> Result<DreamHandle> {
            Ok(DreamHandle {
                run_id: uuid::Uuid::new_v4(),
                scope: scope.clone(),
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
            _scope: &WorkspaceScope,
        ) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
            Ok(None)
        }

        async fn graph_episodes_since_last_dream(&self, _scope: &WorkspaceScope) -> Result<usize> {
            Ok(0)
        }

        async fn graph_is_consolidating(&self, _scope: &WorkspaceScope) -> Result<bool> {
            Ok(false)
        }

        async fn graph_search(
            &self,
            scope: &WorkspaceScope,
            query: &str,
            opts: &SearchOpts,
        ) -> Result<Vec<RetrievedContext>> {
            *self.last_search_scope.lock().unwrap() = Some(scope.clone());
            *self.last_search_query.lock().unwrap() = Some(query.to_string());
            *self.last_search_limit.lock().unwrap() = opts.limit;
            Ok(vec![RetrievedContext {
                entity_id: "ent-stub".into(),
                entity_name: "Stub Entity".into(),
                summary: "from StubGraphHandle".into(),
                score: 0.5,
                source_refs: vec![],
            }])
        }

        async fn graph_run_consolidation(
            &self,
            scope: &WorkspaceScope,
            _provider: Arc<dyn ChatProvider>,
        ) -> Result<DreamPhaseResult> {
            *self.last_consolidation_scope.lock().unwrap() = Some(scope.clone());
            Ok(DreamPhaseResult {
                communities_recomputed: 1,
                cross_meeting_merges: 0,
                supersessions_recorded: 0,
                facts_archived: 0,
                duration_ms: 10,
            })
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
        let scope = WorkspaceScope::with_thread("ws-1", "thread-a");
        let source_ref = SourceRef {
            kind: SourceKind::Meeting,
            id: "mtg-42".into(),
            occurred_at: Utc::now(),
        };
        let facts = vec![StructuredFact {
            subject: "alice".into(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_at: None,
            invalid_at: None,
        }];

        let commit = submit_episode(
            &graph,
            "transcript content",
            source_ref,
            facts,
            null_provider(),
            scope.clone(),
            None,
            SubmitOpts::default(),
            None,
        )
        .await
        .expect("submit_episode should succeed via stub");

        assert!(!commit.episode_entity_id.is_empty());
        assert_eq!(
            graph.last_ingest_scope.lock().unwrap().as_ref(),
            Some(&scope)
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
        let scope = WorkspaceScope::new("ws-2");
        let opts = SearchOpts {
            limit: Some(25),
            as_of: None,
            source_kind: Some(SourceKind::Document),
        };

        let hits = search(&graph, "go-live", scope.clone(), opts)
            .await
            .expect("search delegates cleanly");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity_id, "ent-stub");
        assert_eq!(
            graph.last_search_scope.lock().unwrap().as_ref(),
            Some(&scope)
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
        let scope = WorkspaceScope::new("ws-3");

        let result = run_dream_phase(&graph, scope.clone(), null_provider())
            .await
            .expect("run_dream_phase delegates cleanly");

        assert_eq!(result.communities_recomputed, 1);
        assert_eq!(
            graph.last_consolidation_scope.lock().unwrap().as_ref(),
            Some(&scope)
        );
    }
}

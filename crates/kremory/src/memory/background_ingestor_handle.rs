//! `BackgroundIngestorGraphHandle` — composing `GraphHandle` impl that routes:
//!   - `graph_ingest_episode(run_in_background=true)` → `BackgroundIngestor`
//!   - `graph_batch_status(batch_id)` → `BackgroundIngestor.batch_tracker`
//!   - everything else → `EngineGraphHandle` pass-through
//!
//! Closes ADR-052 facade gap (v0.2.3 Phase 7 blocking_issue): consumers using
//! `Memory::send_batched()` now receive `on_batch_phase2_complete` via the same
//! sink path that direct `BackgroundIngestor` consumers use.
//!
//! ## Pattern
//!
//! CFSR (Tokio actor bridging) + Composition over inheritance — per
//! `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §2`.
//!
//! ## Routing table (arch spec §2.2)
//!
//! | Method | Route |
//! |---|---|
//! | `graph_ingest_episode(run_in_background=true)` | `BackgroundIngestor` (batched or plain) |
//! | `graph_ingest_episode(run_in_background=false)` | `EngineGraphHandle` pass-through |
//! | `graph_batch_status` | `BackgroundIngestor.batch_tracker` (source of truth) |
//! | all other methods | `EngineGraphHandle` pass-through |
//!
//! ## Drop ordering (arch spec §4)
//!
//! On `Memory` drop: `BackgroundIngestorGraphHandle` drops → `Drop` impl acquires
//! `guard` mutex → takes `IngestGuard` → calls `guard.shutdown()` → sets stop flag
//! → joins worker thread.  The worker fires `on_batch_phase2_complete(outcome="interrupted")`
//! for every open batch before the join completes.  D4 thread-context contract is
//! satisfied: callbacks fire on the background worker OS thread.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::background::{BackgroundIngestor, IngestGuard, IngestRequest, IngestSendError};
use crate::core::error::IngestStatus;
use crate::memory::engine_handle::{namespace_to_group_id, EngineGraphHandle};
use crate::memory::{
    events::EnrichmentEventSink,
    graph::{GraphHandle, GraphIngestEpisodeParams},
    types::{
        BatchStatus, CancelOutcome, DreamHandle, DreamOpts, DreamPhaseResult, DreamStatus,
        EpisodeCommit, MemoryError, Namespace, Result, RetrievedContext, SearchOpts,
    },
    ChatProvider,
};

// ---------------------------------------------------------------------------
// BackgroundIngestorGraphHandle
// ---------------------------------------------------------------------------

/// `GraphHandle` implementation that routes background ingest through the
/// `BackgroundIngestor` OS-thread path (ADR-051) and delegates all
/// non-background methods to `EngineGraphHandle`.
///
/// Rationale: the `EngineGraphHandle` tokio-spawn path does NOT fire
/// `on_batch_phase2_complete` (Quinn MED-3 / v0.2.3 Phase 7 facade gap).
/// `BackgroundIngestor` owns the `BatchTracker` and fires the callback
/// after all episodes in a batch reach Phase 2 terminal state.
///
/// Constructed by `MemoryBuilder::build()` when `.with_sink()` is called.
/// `EngineGraphHandle` remains the default when no sink is configured.
///
/// Per CFSR pattern (research doc §1.5) + arch spec §2.1.
pub(crate) struct BackgroundIngestorGraphHandle {
    /// Background OS-thread ingestor — receives `run_in_background=true` calls.
    /// Owns `BatchTracker` and fires `on_batch_phase2_complete` via configured sink.
    ingestor: Arc<BackgroundIngestor>,
    /// Delegate for all non-background-ingest methods: search, dream, batch_status
    /// (non-background batches), ghost_episodes, entity_type assertions, inline
    /// ingest (`run_in_background=false`).
    engine_handle: Arc<EngineGraphHandle>,
    /// RAII shutdown guard. When `BackgroundIngestorGraphHandle` is dropped,
    /// `Drop` takes the guard and calls `guard.shutdown()`.
    ///
    /// `Arc<Mutex<Option<IngestGuard>>>` is the idiomatic RAII-behind-shared-ref
    /// pattern: `BackgroundIngestorGraphHandle` is stored behind `Arc<dyn GraphHandle>`,
    /// so `&mut self` in Drop is the only mutable access point.
    ///
    /// Per arch spec §2.1 (revised shape) + §4.2 (drop impl shape).
    guard: Arc<Mutex<Option<IngestGuard>>>,
}

impl BackgroundIngestorGraphHandle {
    /// Construct a new handle from an already-started `BackgroundIngestor`.
    ///
    /// `guard` is consumed and stored inside `Arc<Mutex<Option<_>>>` so the
    /// Drop impl can take it out via `guard_opt.take()`.
    pub(crate) fn new(
        ingestor: Arc<BackgroundIngestor>,
        engine_handle: Arc<EngineGraphHandle>,
        guard: IngestGuard,
    ) -> Self {
        Self {
            ingestor,
            engine_handle,
            guard: Arc::new(Mutex::new(Some(guard))),
        }
    }
}

// ---------------------------------------------------------------------------
// Drop impl (arch spec §4.2)
// ---------------------------------------------------------------------------

/// Shutdown the `BackgroundIngestor` worker when the handle drops.
///
/// Acquires the guard mutex → takes the `Option<IngestGuard>` (sets to `None`) →
/// calls `guard.shutdown()` → sets stop flag → joins worker thread.
///
/// If the mutex is poisoned (worker panicked), the worker is already dead;
/// interrupt callbacks will not fire — log the poison case for observability.
///
/// Per arch spec §4.2 + §4.4 (drop-context isolation: the worker thread
/// runs an isolated current-thread tokio runtime and does NOT re-enter the
/// caller's tokio runtime, so `handle.join()` inside `shutdown()` is safe
/// from both sync and async drop contexts).
impl Drop for BackgroundIngestorGraphHandle {
    fn drop(&mut self) {
        match self.guard.lock() {
            Ok(mut guard_opt) => {
                if let Some(guard) = guard_opt.take() {
                    guard.shutdown();
                }
            }
            Err(poisoned) => {
                // Mutex poisoned: worker already dead; no shutdown needed,
                // but log so the condition is visible per CLAUDE.md Rule 19.
                tracing::warn!(
                    target: "kremory.background_ingestor_handle",
                    "guard mutex poisoned at drop — worker already dead; \
                     interrupted batch callbacks will not fire"
                );
                // Attempt to recover the guard and shut down anyway.
                // If the guard was taken before the panic, `take()` returns None.
                if let Some(guard) = poisoned.into_inner().take() {
                    guard.shutdown();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Map IngestSendError → MemoryError
// ---------------------------------------------------------------------------

fn ingest_send_err_to_memory_err(e: IngestSendError) -> MemoryError {
    MemoryError::Other(format!("BackgroundIngestor send failed: {e}"))
}

// ---------------------------------------------------------------------------
// GraphHandle impl
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphHandle for BackgroundIngestorGraphHandle {
    // ── 1. graph_ingest_episode ──────────────────────────────────────────────
    //
    // ROUTING: run_in_background=true → BackgroundIngestor (sink fires via OS-thread path).
    //          run_in_background=false → EngineGraphHandle (inline; no BatchTracker update).
    //
    // This is the primary routing method that closes ADR-052 Gap 1 (Quinn MED-3).

    async fn graph_ingest_episode(
        &self,
        params: GraphIngestEpisodeParams<'_>,
    ) -> Result<EpisodeCommit> {
        let GraphIngestEpisodeParams {
            namespace,
            source_ref,
            content,
            structured_facts,
            provider,
            batch_id,
            opts,
            sink,
        } = params;
        if opts.run_in_background {
            // Background path: route through BackgroundIngestor.
            // The OS-thread worker owns the Engine and the BatchTracker.
            // on_batch_phase2_complete fires via the sink registered at
            // BackgroundIngestor construction time (arch spec §2.2 Routing row 1+2).
            //
            // R-12 mitigation: convert &str + Namespace (borrowed) to owned String
            // before calling enqueue_req. This mirrors engine_handle.rs:192-199.
            let group_id = Some(namespace_to_group_id(namespace));
            let text = content.to_owned();

            let result = if let Some(bid) = batch_id.clone() {
                // Batched background path (BatchTracker update + on_batch_phase2_complete).
                //
                // Quinn final MED-02 fix: must propagate group_id to the batched path or
                // multi-namespace consumers' batched episodes land in the wrong namespace
                // shard.  `send_batched(text, bid)` hardcodes group_id=None (ingestor.rs:230);
                // its own doc-comment directs full-control consumers to `enqueue_req`.
                // Race-safety invariant (arch spec §3.3) is preserved — enqueue_req owns it.
                self.ingestor
                    .enqueue_req(IngestRequest {
                        text,
                        reference_time: None,
                        group_id,
                        content_type: None,
                        batch_id: Some(bid),
                    })
                    .map_err(ingest_send_err_to_memory_err)?
            } else {
                // Un-batched background path (no BatchTracker; no batch callback).
                self.ingestor
                    .send(text, None, group_id, None)
                    .map_err(ingest_send_err_to_memory_err)?
            };

            // Observability: triple-emit per ADR-052 DoD item 5.
            metrics::counter!(
                "kremory.background_ingestor_handle.ingest_total",
                "path" => "background"
            )
            .increment(1);
            tracing::debug!(
                target: "kremory.background_ingestor_handle",
                namespace = %namespace.namespace,
                batch_id = ?batch_id,
                "graph_ingest_episode routed to BackgroundIngestor"
            );

            let _ = result; // send returns () on success

            // Background path: no synchronous run_id or episode_entity_id.
            // EpisodeCommit shape per arch spec §2.3.
            Ok(EpisodeCommit {
                run_id: None,
                episode_entity_id: batch_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
                committed_at: Utc::now(),
                stub_entities_inserted: 0,
            })
        } else {
            // Inline path: delegate to EngineGraphHandle.
            // ROUTING: run_in_background=false → EngineGraphHandle (unchanged semantics).
            metrics::counter!(
                "kremory.background_ingestor_handle.ingest_total",
                "path" => "inline"
            )
            .increment(1);
            tracing::debug!(
                target: "kremory.background_ingestor_handle",
                namespace = %namespace.namespace,
                "graph_ingest_episode routed to EngineGraphHandle (inline)"
            );
            self.engine_handle
                .graph_ingest_episode(GraphIngestEpisodeParams {
                    namespace,
                    source_ref,
                    content,
                    structured_facts,
                    provider,
                    batch_id,
                    opts,
                    sink,
                })
                .await
        }
    }

    // ── 2. graph_ingest_status ───────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — run_id UUIDs are only meaningful on the
    // EngineGraphHandle path. Background path returns run_id=None; callers
    // should not query status for None. Per R-07: not-found → Complete is correct.

    async fn graph_ingest_status(&self, run_id: Uuid) -> Result<IngestStatus> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_ingest_status(run_id).await
    }

    // ── 3. graph_cancel ──────────────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — AbortHandle lives on EngineGraphHandle.ingest_abort.
    // BackgroundIngestor has no per-episode abort (the whole worker drains on shutdown).

    async fn graph_cancel(&self, run_id: Uuid) -> Result<CancelOutcome> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_cancel(run_id).await
    }

    // ── 4. graph_submit_dream ────────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — Dream does not go through BackgroundIngestor.

    async fn graph_submit_dream(
        &self,
        namespace: &Namespace,
        provider: Arc<dyn ChatProvider>,
        batch_id: Option<String>,
        opts: DreamOpts,
        sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<DreamHandle> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_submit_dream(namespace, provider, batch_id, opts, sink)
            .await
    }

    // ── 5. graph_dream_status ────────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — dream_runs DashMap lives on EngineGraphHandle.

    async fn graph_dream_status(&self, run_id: Uuid) -> Result<DreamStatus> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_dream_status(run_id).await
    }

    // ── 6. graph_batch_status ────────────────────────────────────────────────
    //
    // ROUTING: BackgroundIngestor.batch_tracker — source-of-truth per arch spec §2.3.
    // After this fix lands, all batched ingest via Memory::send_batched routes through
    // BackgroundIngestor. The BackgroundIngestor.batch_tracker is the canonical
    // BatchTracker for background-path batches.
    //
    // Translation: BatchProgress.succeeded → BatchStatus.completed (field name mismatch
    // confirmed at batch_tracker.rs:38 vs types.rs:593 — arch spec §2.3 note).

    async fn graph_batch_status(&self, batch_id: &str) -> Result<BatchStatus> {
        // ROUTING: BackgroundIngestor.batch_tracker (source of truth for background path).
        //
        // The lock is scoped in a block so the MutexGuard drops BEFORE any `.await`
        // point — required for Send + 'static futures under #[async_trait].
        // Per arch spec §2.3 + R-12 (no .await held across std::sync::Mutex lock).
        let maybe_status = {
            let tracker = self
                .ingestor
                .batch_tracker
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            tracker.get(batch_id).map(|progress| BatchStatus {
                total: progress.total,
                completed: progress.succeeded, // field name mismatch: BatchProgress.succeeded → BatchStatus.completed
                skipped: progress.skipped,
                failed: progress.failed,
            })
        }; // MutexGuard dropped here — safe to .await below

        match maybe_status {
            Some(status) => Ok(status),
            None => {
                // Check EngineGraphHandle's batch_status as fallback for inline-path
                // batches that predate this routing change. See arch spec §2.3
                // "Correctness note": mixing inline + background batch_ids is unsupported
                // at v0.2.3; fallback prevents spurious errors.
                self.engine_handle.graph_batch_status(batch_id).await
            }
        }
    }

    // ── 7. graph_last_consolidated_at ────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — consolidation state lives on EngineGraphHandle.

    async fn graph_last_consolidated_at(
        &self,
        namespace: &Namespace,
    ) -> Result<Option<DateTime<Utc>>> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_last_consolidated_at(namespace)
            .await
    }

    // ── 8. graph_episodes_since_last_dream ───────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — queries engine.graph().count_episodes_since(...).

    async fn graph_episodes_since_last_dream(&self, namespace: &Namespace) -> Result<usize> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_episodes_since_last_dream(namespace)
            .await
    }

    // ── 9. graph_is_consolidating ────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — dream task state on EngineGraphHandle.

    async fn graph_is_consolidating(&self, namespace: &Namespace) -> Result<bool> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_is_consolidating(namespace).await
    }

    // ── 10. graph_search ─────────────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — read-only; engine handles all search.

    async fn graph_search(
        &self,
        namespace: &Namespace,
        query: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<RetrievedContext>> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_search(namespace, query, opts)
            .await
    }

    // ── 11. graph_run_consolidation ──────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — legacy consolidation (NotImplemented at v0.1.0).

    async fn graph_run_consolidation(
        &self,
        namespace: &Namespace,
        provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_run_consolidation(namespace, provider)
            .await
    }

    // ── 12. graph_run_dream_pass_sync ────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — sync dream, engine-owned.

    async fn graph_run_dream_pass_sync(
        &self,
        opts: crate::core::ingest::DreamPassOpts,
    ) -> Result<crate::facade::DreamSummary> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_run_dream_pass_sync(opts).await
    }

    // ── 13. graph_ghost_episodes ─────────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — database read via engine.

    async fn graph_ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle.graph_ghost_episodes(group_id).await
    }

    // ── 14. graph_assert_entity_type ─────────────────────────────────────────
    //
    // ROUTING: EngineGraphHandle — write to entity table via engine.

    async fn graph_assert_entity_type(
        &self,
        entity_id: &str,
        entity_type_id: u32,
        group_id: Option<&str>,
    ) -> Result<()> {
        // ROUTING: delegate to EngineGraphHandle.
        self.engine_handle
            .graph_assert_entity_type(entity_id, entity_type_id, group_id)
            .await
    }
}

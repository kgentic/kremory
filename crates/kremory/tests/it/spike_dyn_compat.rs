//! Spike A: BackgroundIngestorGraphHandle dyn-compat compile spike
//!
//! Verifies the three load-bearing architectural claims from
//! `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §6`:
//!
//! 1. `BackgroundIngestorGraphHandle` + `#[async_trait]` + `GraphHandle` impl compiles
//! 2. `Arc<BackgroundIngestorGraphHandle>` coerces to `Arc<dyn GraphHandle>`
//! 3. `Drop` impl + `Arc<dyn GraphHandle>` coercion coexist without conflict
//! 4. `BackgroundIngestorGraphHandle: Send + Sync` (required by `GraphHandle: Send + Sync`)
//! 5. `Arc<Mutex<Option<IngestGuard>>>` field is `Send + Sync`
//!
//! IMPORTANT: This is a COMPILE spike — the test bodies contain `unimplemented!()`
//! stubs. No test function in this file is intended to RUN successfully;
//! compile success is the verdict.
//!
//! Per `~/.claude/rules/mechanical-compile-spike-beats-paper-review.md`.

#![allow(dead_code, unused_imports, clippy::unwrap_used)]

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use kremory::core::error::IngestStatus;
use kremory::core::ingest::DreamPassOpts;
use kremory::facade::DreamSummary;
use kremory::memory::{
    events::EnrichmentEventSink,
    graph::{
        GraphAssertEntityTypeParams, GraphHandle, GraphIngestEpisodeParams, GraphSearchParams,
        GraphSubmitDreamParams,
    },
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamOpts, DreamPhaseResult,
        DreamStatus, EpisodeCommit, MemoryError, Namespace, Result, RetrievedContext, SearchOpts,
        SourceRef, StructuredFact, SubmitOpts,
    },
    ChatProvider,
};

// ---------------------------------------------------------------------------
// Stub types that mimic the real substrate shapes without requiring full
// substrate compilation context.
// ---------------------------------------------------------------------------

/// Minimal stub for BackgroundIngestor — only the fields needed by
/// BackgroundIngestorGraphHandle are present.
struct StubBackgroundIngestor;

/// Minimal stub for EngineGraphHandle — only needs to impl GraphHandle.
struct StubEngineGraphHandle;

#[async_trait]
impl GraphHandle for StubEngineGraphHandle {
    async fn graph_ingest_episode(
        &self,
        _params: GraphIngestEpisodeParams<'_>,
    ) -> Result<EpisodeCommit> {
        unimplemented!("spike stub")
    }

    async fn graph_ingest_status(&self, _run_id: Uuid) -> Result<IngestStatus> {
        unimplemented!("spike stub")
    }

    async fn graph_cancel(&self, _run_id: Uuid) -> Result<CancelOutcome> {
        unimplemented!("spike stub")
    }

    async fn graph_submit_dream(&self, _params: GraphSubmitDreamParams<'_>) -> Result<DreamHandle> {
        unimplemented!("spike stub")
    }

    async fn graph_dream_status(&self, _run_id: Uuid) -> Result<DreamStatus> {
        unimplemented!("spike stub")
    }

    async fn graph_batch_status(&self, _batch_id: &str) -> Result<BatchStatus> {
        unimplemented!("spike stub")
    }

    async fn graph_last_consolidated_at(
        &self,
        _namespace: &Namespace,
    ) -> Result<Option<DateTime<Utc>>> {
        unimplemented!("spike stub")
    }

    async fn graph_episodes_since_last_dream(&self, _namespace: &Namespace) -> Result<usize> {
        unimplemented!("spike stub")
    }

    async fn graph_is_consolidating(&self, _namespace: &Namespace) -> Result<bool> {
        unimplemented!("spike stub")
    }

    async fn graph_search(&self, _params: GraphSearchParams<'_>) -> Result<Vec<RetrievedContext>> {
        unimplemented!("spike stub")
    }

    async fn graph_run_consolidation(
        &self,
        _namespace: &Namespace,
        _provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult> {
        unimplemented!("spike stub")
    }

    async fn graph_run_dream_pass_sync(&self, _opts: DreamPassOpts) -> Result<DreamSummary> {
        unimplemented!("spike stub")
    }

    async fn graph_ghost_episodes(&self, _group_id: Option<&str>) -> Result<Vec<i64>> {
        unimplemented!("spike stub")
    }

    async fn graph_assert_entity_type(
        &self,
        _params: GraphAssertEntityTypeParams<'_>,
    ) -> Result<()> {
        unimplemented!("spike stub")
    }
}

/// Minimal stub for IngestGuard — must match the real shape:
/// holds Arc<AtomicBool> + Option<JoinHandle<()>>
///
/// Both fields are Send: AtomicBool: Send, JoinHandle<()>: Send.
/// Therefore StubIngestGuard: Send.
struct StubIngestGuard {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

// Verify IngestGuard-like struct is Send + Sync at compile time.
fn _assert_stub_ingest_guard_send_sync()
where
    StubIngestGuard: Send, // JoinHandle<()>: Send, Arc<AtomicBool>: Send
{
}

// ---------------------------------------------------------------------------
// The struct under test: BackgroundIngestorGraphHandle
// ---------------------------------------------------------------------------

/// Proposed struct per arch spec §2.1 (revised shape with Arc<Mutex<Option<IngestGuard>>>).
struct BackgroundIngestorGraphHandle {
    /// Background OS-thread ingestor (Clone + Send + Sync via Arc<Inner>).
    ingestor: Arc<StubBackgroundIngestor>,
    /// Delegate for non-background methods.
    engine_handle: Arc<StubEngineGraphHandle>,
    /// RAII shutdown guard: Drop takes the guard and calls shutdown().
    /// Arc<Mutex<Option<_>>> is the idiomatic pattern for RAII behind shared refs.
    guard: Arc<Mutex<Option<StubIngestGuard>>>,
}

// ---------------------------------------------------------------------------
// Drop impl (arch spec §4.2) — must coexist with async_trait impl below.
// ---------------------------------------------------------------------------

impl Drop for BackgroundIngestorGraphHandle {
    fn drop(&mut self) {
        if let Ok(mut guard_opt) = self.guard.lock() {
            if let Some(guard) = guard_opt.take() {
                // Calls guard.shutdown() equivalent — sets stop flag, joins thread.
                drop(guard);
            }
        }
        // If mutex is poisoned (worker panicked), worker is dead; callbacks won't fire.
    }
}

// ---------------------------------------------------------------------------
// GraphHandle impl via #[async_trait] — the primary dyn-compat claim.
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphHandle for BackgroundIngestorGraphHandle {
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
            // Route to BackgroundIngestor path (stub: just return a mock commit)
            // In production: self.ingestor.enqueue_req(IngestRequest { ... })
            Ok(EpisodeCommit {
                run_id: None,
                episode_entity_id: batch_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                committed_at: chrono::Utc::now(),
                stub_entities_inserted: 0,
            })
        } else {
            // Delegate to EngineGraphHandle for inline path
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

    async fn graph_ingest_status(&self, run_id: Uuid) -> Result<IngestStatus> {
        self.engine_handle.graph_ingest_status(run_id).await
    }

    async fn graph_cancel(&self, run_id: Uuid) -> Result<CancelOutcome> {
        self.engine_handle.graph_cancel(run_id).await
    }

    async fn graph_submit_dream(&self, params: GraphSubmitDreamParams<'_>) -> Result<DreamHandle> {
        self.engine_handle.graph_submit_dream(params).await
    }

    async fn graph_dream_status(&self, run_id: Uuid) -> Result<DreamStatus> {
        self.engine_handle.graph_dream_status(run_id).await
    }

    async fn graph_batch_status(&self, _batch_id: &str) -> Result<BatchStatus> {
        // In production: read from self.ingestor.batch_tracker
        // Stub: return empty
        Ok(BatchStatus {
            total: 0,
            completed: 0,
            skipped: 0,
            failed: 0,
        })
    }

    async fn graph_last_consolidated_at(
        &self,
        namespace: &Namespace,
    ) -> Result<Option<DateTime<Utc>>> {
        self.engine_handle
            .graph_last_consolidated_at(namespace)
            .await
    }

    async fn graph_episodes_since_last_dream(&self, namespace: &Namespace) -> Result<usize> {
        self.engine_handle
            .graph_episodes_since_last_dream(namespace)
            .await
    }

    async fn graph_is_consolidating(&self, namespace: &Namespace) -> Result<bool> {
        self.engine_handle.graph_is_consolidating(namespace).await
    }

    async fn graph_search(&self, params: GraphSearchParams<'_>) -> Result<Vec<RetrievedContext>> {
        self.engine_handle.graph_search(params).await
    }

    async fn graph_run_consolidation(
        &self,
        namespace: &Namespace,
        provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult> {
        self.engine_handle
            .graph_run_consolidation(namespace, provider)
            .await
    }

    async fn graph_run_dream_pass_sync(&self, opts: DreamPassOpts) -> Result<DreamSummary> {
        self.engine_handle.graph_run_dream_pass_sync(opts).await
    }

    async fn graph_ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        self.engine_handle.graph_ghost_episodes(group_id).await
    }

    async fn graph_assert_entity_type(
        &self,
        params: GraphAssertEntityTypeParams<'_>,
    ) -> Result<()> {
        self.engine_handle.graph_assert_entity_type(params).await
    }
}

// ---------------------------------------------------------------------------
// Compile-only verification functions (never called at runtime)
// ---------------------------------------------------------------------------

/// Claim 1: BackgroundIngestorGraphHandle: Send + Sync
/// Required because GraphHandle: Send + Sync, and Arc<dyn GraphHandle> needs T: Send + Sync.
fn _check_send_sync()
where
    BackgroundIngestorGraphHandle: Send + Sync,
{
    // Fields:
    // - Arc<StubBackgroundIngestor>: Send + Sync iff StubBackgroundIngestor: Send + Sync
    // - Arc<StubEngineGraphHandle>: Send + Sync iff StubEngineGraphHandle: Send + Sync
    // - Arc<Mutex<Option<StubIngestGuard>>>: Send + Sync iff StubIngestGuard: Send
}

/// Claim 2: Arc<BackgroundIngestorGraphHandle> coerces to Arc<dyn GraphHandle>
fn _check_dyn_compat() {
    let handle = BackgroundIngestorGraphHandle {
        ingestor: Arc::new(StubBackgroundIngestor),
        engine_handle: Arc::new(StubEngineGraphHandle),
        guard: Arc::new(Mutex::new(None)),
    };
    // This is the critical coercion — fails at compile time if GraphHandle
    // is not dyn-compatible for this impl, or if Send + Sync bounds are unmet.
    let _: Arc<dyn GraphHandle> = Arc::new(handle);
}

/// Claim 3: Drop impl coexists with Arc<dyn GraphHandle> coercion
/// (compiler verifies both impls on the same struct — no extra test needed
/// beyond the coexistence of _check_dyn_compat and the Drop impl above)
fn _check_drop_and_dyn_coexist() {
    // Construct + immediately drop — invokes Drop impl
    let handle = BackgroundIngestorGraphHandle {
        ingestor: Arc::new(StubBackgroundIngestor),
        engine_handle: Arc::new(StubEngineGraphHandle),
        guard: Arc::new(Mutex::new(None)),
    };
    let arc_dyn: Arc<dyn GraphHandle> = Arc::new(handle);
    drop(arc_dyn); // Drop impl fires on BackgroundIngestorGraphHandle when Arc refcount → 0
}

// ---------------------------------------------------------------------------
// Minimal runtime test — verifies the above doesn't PANIC (not just compile)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn spike_a_background_ingestor_graph_handle_dyn_compat() {
    // Construct
    let handle = BackgroundIngestorGraphHandle {
        ingestor: Arc::new(StubBackgroundIngestor),
        engine_handle: Arc::new(StubEngineGraphHandle),
        guard: Arc::new(Mutex::new(None)),
    };

    // Coerce to Arc<dyn GraphHandle>
    let dyn_handle: Arc<dyn GraphHandle> = Arc::new(handle);

    // Exercise the background path (run_in_background=true)
    use chrono::Utc;
    let ns = Namespace::new("spike-test");
    let source_ref = SourceRef {
        kind: kremory::memory::types::SourceKind::Episode,
        id: "spike-1".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };
    let provider: Arc<dyn ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());

    let result = dyn_handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &source_ref,
            content: "spike test content",
            structured_facts: &[],
            provider,
            batch_id: Some("batch-spike".to_string()),
            opts: SubmitOpts {
                enrich_per_episode: false,
                run_in_background: true,
            },
            sink: None,
        })
        .await;

    assert!(
        result.is_ok(),
        "background path through dyn handle must succeed: {result:?}"
    );
    let commit = result.unwrap();
    assert_eq!(
        commit.run_id, None,
        "background path must return run_id=None"
    );
    assert_eq!(commit.episode_entity_id, "batch-spike");

    // Drop the Arc<dyn GraphHandle> — must invoke Drop impl without panic
    drop(dyn_handle);

    // If we get here: PASS
    // - Arc<dyn GraphHandle> coercion compiled
    // - #[async_trait] impl compiled and executed
    // - Drop impl compiled and fired without panic
    // - Send + Sync bounds satisfied (implied by Arc<dyn GraphHandle> construction)
}

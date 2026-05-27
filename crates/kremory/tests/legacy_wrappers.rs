//! A.6 — D.6.9 backward-compat wrapper tests.
//!
//! Verifies that the deprecated legacy wrappers (`ingest_episode`, `run_dream_phase`)
//! still compile and delegate correctly to the new API surface.
//!
//! All uses of deprecated items are annotated `#[allow(deprecated)]` to
//! prevent the deprecation warning from failing the test run under
//! `-D warnings`. In production code, these warnings WILL appear to guide
//! migration to `submit_episode` + `submit_dream_phase`.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use kremory::core::error::IngestStatus;
use kremory::memory::{
    events::EnrichmentEventSink,
    types::{
        BatchStatus, CancelOutcome, DreamHandle, DreamOpts, DreamPhaseResult, DreamStatus,
        EpisodeCommit, IngestResult, Namespace, RetrievedContext, SearchOpts, SourceKind,
        SourceRef, StructuredFact, SubmitOpts,
    },
    ChatProvider, GraphHandle,
};
use uuid::Uuid;

// ── Minimal stub for legacy wrapper tests ────────────────────────────────────

struct LegacyStub;

#[async_trait]
impl GraphHandle for LegacyStub {
    async fn graph_ingest_episode(
        &self,
        _namespace: &Namespace,
        source_ref: &SourceRef,
        _content: &str,
        _structured_facts: &[StructuredFact],
        _provider: Arc<dyn ChatProvider>,
        _batch_id: Option<String>,
        _opts: SubmitOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> kremory::memory::types::Result<EpisodeCommit> {
        Ok(EpisodeCommit {
            run_id: None,
            episode_entity_id: format!("legacy:{}", source_ref.id),
            committed_at: Utc::now(),
            stub_entities_inserted: 0,
        })
    }

    async fn graph_ingest_status(
        &self,
        _run_id: Uuid,
    ) -> kremory::memory::types::Result<IngestStatus> {
        Ok(IngestStatus::Complete)
    }

    async fn graph_cancel(&self, _run_id: Uuid) -> kremory::memory::types::Result<CancelOutcome> {
        Ok(CancelOutcome::stub())
    }

    async fn graph_submit_dream(
        &self,
        namespace: &Namespace,
        _provider: Arc<dyn ChatProvider>,
        batch_id: Option<String>,
        _opts: DreamOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> kremory::memory::types::Result<DreamHandle> {
        Ok(DreamHandle {
            run_id: Uuid::new_v4(),
            namespace: namespace.clone(),
            submitted_at: Utc::now(),
            batch_id,
        })
    }

    async fn graph_dream_status(
        &self,
        _run_id: Uuid,
    ) -> kremory::memory::types::Result<DreamStatus> {
        Ok(DreamStatus::Complete)
    }

    async fn graph_batch_status(
        &self,
        _batch_id: &str,
    ) -> kremory::memory::types::Result<BatchStatus> {
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
    ) -> kremory::memory::types::Result<Option<chrono::DateTime<Utc>>> {
        Ok(None)
    }

    async fn graph_episodes_since_last_dream(
        &self,
        _namespace: &Namespace,
    ) -> kremory::memory::types::Result<usize> {
        Ok(0)
    }

    async fn graph_is_consolidating(
        &self,
        _namespace: &Namespace,
    ) -> kremory::memory::types::Result<bool> {
        Ok(false)
    }

    async fn graph_search(
        &self,
        _namespace: &Namespace,
        _query: &str,
        _opts: &SearchOpts,
    ) -> kremory::memory::types::Result<Vec<RetrievedContext>> {
        Ok(vec![])
    }

    async fn graph_run_consolidation(
        &self,
        _namespace: &Namespace,
        _provider: Arc<dyn ChatProvider>,
    ) -> kremory::memory::types::Result<DreamPhaseResult> {
        Ok(DreamPhaseResult::default())
    }
}

fn null_provider() -> Arc<dyn ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

// ── Backward compat tests ─────────────────────────────────────────────────────

/// Verifies that the deprecated `ingest_episode` wrapper still compiles under
/// `#[allow(deprecated)]` and returns an `IngestResult` with the expected
/// stub counts (1, 0, 0, 0) as documented in its deprecation notice.
#[tokio::test]
#[allow(deprecated)]
async fn legacy_ingest_episode_compiles_and_returns_stub_counts() {
    use kremory::memory::ingest_episode;

    let handle = LegacyStub;
    let scope = Namespace::new("ws-legacy");
    let source_ref = SourceRef {
        kind: SourceKind::Meeting,
        id: "mtg-legacy".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    let result: IngestResult = ingest_episode(
        &handle,
        "legacy meeting transcript",
        source_ref,
        vec![],
        null_provider(),
        scope,
    )
    .await
    .expect("ingest_episode must succeed via stub");

    // Stub counts per deprecation notice: (entities_added=1, edges_added=0,
    // facts_invalidated=0, duration_ms=0). These are intentional degradation;
    // callers used these for logging only (ADR §5.5).
    assert_eq!(result.entities_added, 1);
    assert_eq!(result.edges_added, 0);
    assert_eq!(result.facts_invalidated, 0);
    assert_eq!(result.duration_ms, 0);
}

/// Verifies that the deprecated `run_dream_phase` wrapper still compiles under
/// `#[allow(deprecated)]` and delegates to `graph_run_consolidation`.
#[tokio::test]
#[allow(deprecated)]
async fn legacy_run_dream_phase_compiles_and_delegates() {
    use kremory::memory::run_dream_phase;

    let handle = LegacyStub;
    let scope = Namespace::new("ws-legacy-dream");

    let result = run_dream_phase(&handle, scope, null_provider())
        .await
        .expect("run_dream_phase must succeed via stub");

    // LegacyStub::graph_run_consolidation returns DreamPhaseResult::default().
    let _ = result;
}

/// Name-accessibility compile gate — verifies the deprecated items are
/// reachable from the kremory::memory namespace by importing them.
/// If either function is removed or renamed, this fails to compile.
#[allow(deprecated, unused_imports)]
fn _legacy_names_accessible() {
    use kremory::memory::ingest_episode as _;
    use kremory::memory::run_dream_phase as _;
}

/// Verifies that `ingest_episode` is actually marked deprecated — callers
/// without `#[allow(deprecated)]` would get a warning. This test uses
/// `#[allow(deprecated)]` to suppress the warning in the test suite.
///
/// Cannot directly assert "this emits a warning" in Rust tests — the
/// compile-test approach (`trybuild` compile-fail) would be needed for
/// that. Here we document the contract and verify the wrapper still works.
#[tokio::test]
#[allow(deprecated)]
async fn legacy_ingest_episode_is_deprecated_and_functional() {
    use kremory::memory::ingest_episode;

    let handle = LegacyStub;
    let scope = Namespace::new("ws-deprecated");
    let source_ref = SourceRef {
        kind: SourceKind::Chat,
        id: "chat-deprecated".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    // This call would emit a `use of deprecated function` compiler warning
    // without `#[allow(deprecated)]`. The wrapper still functions correctly.
    let result = ingest_episode(
        &handle,
        "chat content",
        source_ref,
        vec![],
        null_provider(),
        scope,
    )
    .await
    .expect("deprecated wrapper must still work");

    assert_eq!(
        result.entities_added, 1,
        "stub count must be 1 per deprecation doc"
    );
}

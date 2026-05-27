//! A.3 — D.6.4 compile-time contract pin tests.
//!
//! These tests verify that the four locked public fn signatures from
//! ADR rqlm-async-event-handle-api-design-2026-05-19 §4.10 exist and
//! compile exactly as specified. A signature change breaks these tests,
//! which is the intended enforcement mechanism.

use std::sync::Arc;

use chrono::Utc;
use kremory::memory::{
    await_batch_enrichment, await_dream, await_enrichment, submit_dream_phase, submit_episode,
};
use kremory::memory::{
    events::EnrichmentEventSink,
    types::{
        AwaitOpts, BatchStatus, DreamHandle, DreamOpts, DreamStatus, EpisodeCommit, IngestStatus,
        Namespace, SubmitOpts,
    },
    GraphHandle,
};
use uuid::Uuid;

// ── Minimal stub implementing the extended GraphHandle trait ──────────────────

use async_trait::async_trait;
use kremory::memory::{
    types::{
        CancelOutcome, CancelledPhase, DreamPhaseResult, RetrievedContext, SearchOpts, SourceRef,
        StructuredFact,
    },
    ChatProvider,
};

struct StubHandle;

#[async_trait]
impl GraphHandle for StubHandle {
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
            episode_entity_id: format!("stub:{}", source_ref.id),
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
        Ok(CancelOutcome {
            cancelled_phase: CancelledPhase::Enrichment,
            rolled_back: false,
            partial: vec![],
        })
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

// ── Contract pin tests ────────────────────────────────────────────────────────

/// Verifies submit_episode signature compiles exactly as ADR §4.10 specifies.
#[tokio::test]
async fn submit_episode_signature_compiles() {
    use chrono::Utc;
    use kremory::memory::types::{SourceKind, SourceRef};

    let handle = StubHandle;
    let scope = Namespace::new("ws-a").with_thread("thread-a");
    let source_ref = SourceRef {
        kind: SourceKind::Meeting,
        id: "mtg-1".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };
    let commit = submit_episode(
        &handle,
        "hello world",
        source_ref,
        vec![],
        null_provider(),
        scope,
        Some("batch-1".to_string()),
        SubmitOpts::default(),
        None,
    )
    .await
    .expect("submit_episode must succeed via stub");

    assert!(!commit.episode_entity_id.is_empty());
}

/// Verifies submit_dream_phase signature compiles + returns DreamHandle.
#[tokio::test]
async fn submit_dream_phase_signature_compiles() {
    let handle = StubHandle;
    let scope = Namespace::new("ws-b");

    let dream = submit_dream_phase(
        &handle,
        scope,
        null_provider(),
        Some("batch-2".to_string()),
        DreamOpts::default(),
        None,
    )
    .await
    .expect("submit_dream_phase must succeed via stub");

    assert!(!dream.batch_id.unwrap_or_default().is_empty());
}

/// Verifies await_enrichment compiles and returns IngestStatus::Complete via stub.
#[tokio::test]
async fn await_enrichment_signature_compiles() {
    let handle = StubHandle;
    let run_id = Uuid::new_v4();

    let status = await_enrichment(&handle, run_id, AwaitOpts::default())
        .await
        .expect("await_enrichment must succeed via stub");

    assert_eq!(status, IngestStatus::Complete);
}

/// Verifies await_dream compiles and returns DreamStatus::Complete via stub.
#[tokio::test]
async fn await_dream_signature_compiles() {
    let handle = StubHandle;
    let run_id = Uuid::new_v4();

    let status = await_dream(&handle, run_id, AwaitOpts::default())
        .await
        .expect("await_dream must succeed via stub");

    assert_eq!(status, DreamStatus::Complete);
}

/// Verifies await_batch_enrichment compiles and returns BatchStatus via stub.
#[tokio::test]
async fn await_batch_enrichment_signature_compiles() {
    let handle = StubHandle;

    let status = await_batch_enrichment(&handle, "batch-3", AwaitOpts::default())
        .await
        .expect("await_batch_enrichment must succeed via stub");

    // Stub returns total=0 completed=0 skipped=0 failed=0, is_done() == true vacuously.
    assert!(status.is_done());
}

/// Verifies BatchStatus::is_done semantics.
#[test]
fn batch_status_is_done_semantics() {
    // All zero — vacuously done.
    let s = BatchStatus {
        total: 0,
        completed: 0,
        skipped: 0,
        failed: 0,
    };
    assert!(s.is_done());

    // 3 of 3 terminal.
    let s = BatchStatus {
        total: 3,
        completed: 2,
        skipped: 1,
        failed: 0,
    };
    assert!(s.is_done());

    // 2 of 3 terminal — NOT done.
    let s = BatchStatus {
        total: 3,
        completed: 2,
        skipped: 0,
        failed: 0,
    };
    assert!(!s.is_done());
}

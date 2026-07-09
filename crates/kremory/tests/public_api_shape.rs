#![allow(clippy::unwrap_used, clippy::expect_used)]
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
    SubmitDreamPhaseParams, SubmitEpisodeParams,
};
use kremory::memory::{
    types::{
        AwaitOpts, BatchStatus, DreamHandle, DreamOpts, DreamStatus, EpisodeCommit, IngestStatus,
        Namespace, SubmitOpts,
    },
    GraphHandle, GraphIngestEpisodeParams, GraphSubmitDreamParams,
};
use uuid::Uuid;

// ── Minimal stub implementing the extended GraphHandle trait ──────────────────

use async_trait::async_trait;
use kremory::memory::{
    types::{CancelOutcome, CancelledPhase, DreamPhaseResult, RetrievedContext},
    ChatProvider,
};

struct StubHandle;

#[async_trait]
impl GraphHandle for StubHandle {
    async fn graph_ingest_episode(
        &self,
        params: GraphIngestEpisodeParams<'_>,
    ) -> kremory::memory::types::Result<EpisodeCommit> {
        let GraphIngestEpisodeParams {
            namespace: _,
            source_ref,
            content: _,
            structured_facts: _,
            provider: _,
            batch_id: _,
            opts: _,
            sink: _,
        } = params;
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
        params: GraphSubmitDreamParams<'_>,
    ) -> kremory::memory::types::Result<DreamHandle> {
        let GraphSubmitDreamParams {
            namespace,
            provider: _,
            batch_id,
            opts: _,
            sink: _,
        } = params;
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
        _params: kremory::GraphSearchParams<'_>,
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

    async fn graph_run_dream_pass_sync(
        &self,
        _opts: kremory::DreamPassOpts,
    ) -> kremory::memory::types::Result<kremory::DreamSummary> {
        Ok(kremory::DreamSummary {
            communities_updated: 0,
            cross_episode_merges: 0,
            supersessions_recorded: 0,
            facts_archived: 0,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            duration_ms: 0,
            types_discovered: vec![],
            entities_reclassified: 0,
            warnings: vec![],
            budget_exhausted: false,
        })
    }

    async fn graph_ghost_episodes(
        &self,
        _group_id: Option<&str>,
    ) -> kremory::memory::types::Result<Vec<i64>> {
        Ok(vec![])
    }

    async fn graph_assert_entity_type(
        &self,
        _params: kremory::GraphAssertEntityTypeParams<'_>,
    ) -> kremory::memory::types::Result<()> {
        Ok(())
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
    let commit = submit_episode(SubmitEpisodeParams {
        graph: &handle,
        content: "hello world",
        source_ref,
        structured_facts: vec![],
        provider: null_provider(),
        namespace: scope,
        batch_id: Some("batch-1".to_string()),
        opts: SubmitOpts::default(),
        sink: None,
    })
    .await
    .expect("submit_episode must succeed via stub");

    assert!(!commit.episode_entity_id.is_empty());
}

/// Verifies submit_dream_phase signature compiles + returns DreamHandle.
#[tokio::test]
async fn submit_dream_phase_signature_compiles() {
    let handle = StubHandle;
    let scope = Namespace::new("ws-b");

    let dream = submit_dream_phase(SubmitDreamPhaseParams {
        graph: &handle,
        namespace: scope,
        provider: null_provider(),
        batch_id: Some("batch-2".to_string()),
        opts: DreamOpts::default(),
        sink: None,
    })
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

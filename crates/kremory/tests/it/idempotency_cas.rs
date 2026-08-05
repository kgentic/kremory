#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.3 — D.6.4 idempotency CAS concurrency tests.
//!
//! Verifies that submit_dream_phase with a DashMap-backed StubIdempotentHandle
//! correctly deduplicates 100 concurrent submits with the same
//! (namespace, thread, batch_id) key, returning exactly one unique run_id.
//!
//! This test exercises the CAS pattern documented in ADR §2.10:
//! "DashMap::entry(key).or_insert_with(|| new_run_state)" — the closure only
//! fires if no entry existed. 100 concurrent callers → 1 run started, 99 reuse.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use kremory::memory::{
    submit_dream_phase,
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamOpts, DreamPhaseResult,
        DreamStatus, EpisodeCommit, IngestStatus, Namespace, RetrievedContext,
    },
    ChatProvider, GraphHandle, GraphIngestEpisodeParams, GraphSubmitDreamParams,
    SubmitDreamPhaseParams,
};
use uuid::Uuid;

// ── Idempotent stub using DashMap CAS (mirrors production the host applicationGraphHandle) ─

#[derive(Default)]
struct StubIdempotentHandle {
    /// (namespace, thread_or_empty, batch_id_or_empty) → DreamHandle.
    active_runs: DashMap<(String, String, String), DreamHandle>,
}

#[async_trait]
impl GraphHandle for StubIdempotentHandle {
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
        let key = (
            namespace.namespace.clone(),
            namespace.thread.clone().unwrap_or_default(),
            batch_id.clone().unwrap_or_default(),
        );

        // CAS: DashMap::entry().or_insert_with() is atomic — the closure only
        // fires once even under 100 concurrent callers on the same key.
        let entry = self.active_runs.entry(key).or_insert_with(|| DreamHandle {
            run_id: Uuid::new_v4(),
            namespace: namespace.clone(),
            submitted_at: Utc::now(),
            batch_id,
        });

        Ok(entry.clone())
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
            cross_episode_would_merge: 0,
            cross_episode_merged: 0,
            consolidation_ops_ran: kremory::ConsolidationOpsRan::default(),
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

/// 100 concurrent submit_dream_phase calls with same (namespace, thread,
/// batch_id) key MUST produce exactly 1 unique run_id.
///
/// This test MUST pass 5 consecutive runs (no flakiness from race condition).
/// The CAS pattern `entry().or_insert_with()` is atomic under DashMap's
/// per-key sharding — the closure fires exactly once per unique key.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn submit_dream_phase_idempotency_under_concurrency() {
    let handle = Arc::new(StubIdempotentHandle::default());
    let scope = Namespace::new("ws-concurrent").with_thread("thread-concurrent");
    let batch_id = Some("batch-concurrent-001".to_string());

    let futures: Vec<_> = (0..100)
        .map(|_| {
            let h = Arc::clone(&handle);
            let s = scope.clone();
            let b = batch_id.clone();
            async move {
                submit_dream_phase(SubmitDreamPhaseParams {
                    graph: h.as_ref(),
                    namespace: s,
                    provider: null_provider(),
                    batch_id: b,
                    opts: DreamOpts::default(),
                    sink: None,
                })
                .await
            }
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    let dream_handles: Vec<DreamHandle> = results.into_iter().map(|r| r.unwrap()).collect();

    let unique_run_ids: std::collections::HashSet<Uuid> =
        dream_handles.iter().map(|h| h.run_id).collect();

    assert_eq!(
        unique_run_ids.len(),
        1,
        "100 concurrent submits with same (workspace_id, thread_id, batch_id) key MUST \
         produce exactly 1 unique run_id — CAS deduplication failed (namespace key)"
    );
}

/// Verify that different batch_id values DO produce different run_ids
/// (no spurious deduplication across distinct keys).
#[tokio::test]
async fn submit_dream_phase_different_batch_ids_produce_different_runs() {
    let handle = StubIdempotentHandle::default();
    let scope = Namespace::new("ws-distinct");

    let h1 = submit_dream_phase(SubmitDreamPhaseParams {
        graph: &handle,
        namespace: scope.clone(),
        provider: null_provider(),
        batch_id: Some("batch-A".to_string()),
        opts: DreamOpts::default(),
        sink: None,
    })
    .await
    .expect("first submit");

    let h2 = submit_dream_phase(SubmitDreamPhaseParams {
        graph: &handle,
        namespace: scope.clone(),
        provider: null_provider(),
        batch_id: Some("batch-B".to_string()),
        opts: DreamOpts::default(),
        sink: None,
    })
    .await
    .expect("second submit different batch");

    assert_ne!(
        h1.run_id, h2.run_id,
        "different batch_id keys must produce different run_ids"
    );
}

/// Verify second call with same key returns identical run_id (not a new one).
#[tokio::test]
async fn submit_dream_phase_same_key_returns_existing_handle() {
    let handle = StubIdempotentHandle::default();
    let scope = Namespace::new("ws-reuse").with_thread("thread-reuse");
    let batch_id = Some("batch-reuse".to_string());

    let h1 = submit_dream_phase(SubmitDreamPhaseParams {
        graph: &handle,
        namespace: scope.clone(),
        provider: null_provider(),
        batch_id: batch_id.clone(),
        opts: DreamOpts::default(),
        sink: None,
    })
    .await
    .expect("first submit");

    let h2 = submit_dream_phase(SubmitDreamPhaseParams {
        graph: &handle,
        namespace: scope.clone(),
        provider: null_provider(),
        batch_id,
        opts: DreamOpts::default(),
        sink: None,
    })
    .await
    .expect("second submit same key");

    assert_eq!(
        h1.run_id, h2.run_id,
        "second submit with same key must return existing handle's run_id"
    );
}

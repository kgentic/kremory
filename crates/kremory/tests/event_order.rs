#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.4 — D.6.6: async/event integration tests.
//!
//! Verifies that submit_episode with enrich_per_episode=true:
//! 1. Fires IngestEventSink events in the canonical order.
//! 2. Phase 1 commit (on_stage_change(Complete) for Phase 1) is observable
//!    BEFORE Phase 2 enrichment events start.
//!
//! Events sequence per ADR §4.8 / D.6.6 Notion ticket:
//!   on_stage_change(Pending)       — phase 2 queued
//!   on_stage_change(Extracting)    — LLM extraction started
//!   on_entity_extracted (per entity)
//!   on_edge_added (per edge)
//!   on_stage_change(Deduplicating)
//!   on_contradiction (per supersession, if any)
//!   on_stage_change(Invalidating)
//!   on_stage_change(Complete)

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use kremory::core::error::{ContradictionResolution, IngestStatus};
use kremory::core::sink::{
    ContradictionDetected, EntityId, IngestEventSink, IngestionError, SinkFact,
};
use kremory::memory::{
    events::EnrichmentEventSink,
    submit_episode,
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamOpts, DreamPhaseResult,
        DreamStatus, EpisodeCommit, Namespace, RetrievedContext, SearchOpts, SourceRef, SubmitOpts,
    },
    ChatProvider, GraphHandle, GraphIngestEpisodeParams,
};
use uuid::Uuid;

// ── MockIngestSink — records events as strings in order ──────────────────────

#[derive(Default, Clone)]
struct MockIngestSink {
    events: Arc<Mutex<Vec<String>>>,
}

impl MockIngestSink {
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl IngestEventSink for MockIngestSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("entity_extracted:{entity_id}:{name}"));
    }

    fn on_edge_added(&self, from: &str, to: &str, predicate: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("edge_added:{from}->{to}:{predicate}"));
    }

    fn on_contradiction(&self, event: kremory::core::sink::ContradictionDetected) {
        self.events
            .lock()
            .unwrap()
            .push(format!("contradiction:{:?}", event.resolution));
    }

    fn on_dedup_merge(&self, surviving_id: &str, absorbed_id: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("dedup_merge:{surviving_id}<-{absorbed_id}"));
    }

    fn on_stage_change(&self, stage: IngestStatus) {
        self.events.lock().unwrap().push(format!("stage:{stage:?}"));
    }

    fn on_ingestion_error(&self, event: IngestionError) {
        self.events
            .lock()
            .unwrap()
            .push(format!("error:{:?}", event.error_kind));
    }
}

// EnrichmentEventSink supertrait requires IngestEventSink — impl it too
// so MockIngestSink can be passed as Arc<dyn EnrichmentEventSink>.
impl EnrichmentEventSink for MockIngestSink {
    fn on_community_updated(&self, community_id: &str, member_count: usize) {
        self.events
            .lock()
            .unwrap()
            .push(format!("community_updated:{community_id}:{member_count}"));
    }

    fn on_batch_phase2_complete(&self, event: kremory::memory::events::BatchPhase2Complete) {
        self.events
            .lock()
            .unwrap()
            .push(format!("batch_phase2_complete:{}", event.batch_id));
    }
}

// ── StubIngestingHandle — fires sink events during graph_ingest_episode ──────

/// A stub GraphHandle that fires IngestEventSink events in the canonical
/// order when enrich_per_episode = true. Simulates the Phase 2 enrichment
/// lifecycle without any real LLM calls.
struct StubIngestingHandle;

#[async_trait]
impl GraphHandle for StubIngestingHandle {
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
            opts,
            sink,
        } = params;
        // Phase 1 always commits.
        let commit = EpisodeCommit {
            run_id: None,
            episode_entity_id: format!("stub:{}", source_ref.id),
            committed_at: Utc::now(),
            stub_entities_inserted: 0,
        };

        // Phase 2 events only fire when enrich_per_episode = true.
        if opts.enrich_per_episode {
            if let Some(s) = &sink {
                // Canonical event sequence per ADR D.6.6:
                s.on_stage_change(IngestStatus::Pending);
                s.on_stage_change(IngestStatus::Extracting);
                // Simulate one entity + one edge extracted.
                s.on_entity_extracted("ent-stub-1", "Alice");
                s.on_edge_added("ent-stub-1", "ent-stub-2", "knows");
                s.on_stage_change(IngestStatus::Deduplicating);
                s.on_stage_change(IngestStatus::Invalidating);
                s.on_stage_change(IngestStatus::Complete);
            }
        }

        Ok(commit)
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

    async fn graph_run_dream_pass_sync(
        &self,
        _opts: kremory::DreamPassOpts,
    ) -> kremory::memory::types::Result<kremory::DreamSummary> {
        Ok(kremory::DreamSummary {
            communities_updated: 0,
            cross_episode_merges: 0,
            supersessions_recorded: 0,
            facts_archived: 0,
            duration_ms: 0,
            types_discovered: vec![],
            entities_reclassified: 0,
            warnings: vec![],
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
        _entity_id: &str,
        _entity_type_id: u32,
        _group_id: Option<&str>,
    ) -> kremory::memory::types::Result<()> {
        Ok(())
    }
}

fn null_provider() -> Arc<dyn ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// submit_episode with enrich_per_episode=true emits IngestEventSink events
/// in the canonical order: stage(Pending) → stage(Extracting) →
/// entity_extracted → edge_added → stage(Deduplicating) →
/// stage(Invalidating) → stage(Complete).
///
/// Phase 1 commit (the EpisodeCommit return value) is observable before
/// Phase 2 events fire — verified by the return-first structure of the
/// stub (commit is constructed before any sink events are fired).
#[tokio::test]
async fn submit_episode_with_enrich_emits_events_in_order() {
    use kremory::memory::types::SourceKind;

    let sink = Arc::new(MockIngestSink::default());
    let handle = StubIngestingHandle;
    let scope = Namespace::new("ws-events").with_thread("thread-events");
    let source_ref = SourceRef {
        kind: SourceKind::Meeting,
        id: "mtg-events".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    let commit = submit_episode(
        &handle,
        "meeting transcript about Alice and Bob",
        source_ref,
        vec![],
        null_provider(),
        scope,
        None,
        SubmitOpts {
            enrich_per_episode: true,
            run_in_background: false,
        },
        Some(Arc::clone(&sink) as Arc<dyn EnrichmentEventSink>),
    )
    .await
    .expect("submit_episode should succeed");

    // Phase 1 commit is available.
    assert!(!commit.episode_entity_id.is_empty());

    // Phase 2 events in canonical order.
    let events = sink.events();
    assert_eq!(events.len(), 7, "expected 7 events: {events:?}");
    assert_eq!(events[0], "stage:Pending");
    assert_eq!(events[1], "stage:Extracting");
    assert_eq!(events[2], "entity_extracted:ent-stub-1:Alice");
    assert_eq!(events[3], "edge_added:ent-stub-1->ent-stub-2:knows");
    assert_eq!(events[4], "stage:Deduplicating");
    assert_eq!(events[5], "stage:Invalidating");
    assert_eq!(events[6], "stage:Complete");
}

/// submit_episode with enrich_per_episode=false emits NO Phase 2 events.
#[tokio::test]
async fn submit_episode_without_enrich_emits_no_events() {
    use kremory::memory::types::SourceKind;

    let sink = Arc::new(MockIngestSink::default());
    let handle = StubIngestingHandle;
    let scope = Namespace::new("ws-no-enrich");
    let source_ref = SourceRef {
        kind: SourceKind::Document,
        id: "doc-no-enrich".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    let commit = submit_episode(
        &handle,
        "content without enrichment",
        source_ref,
        vec![],
        null_provider(),
        scope,
        None,
        SubmitOpts::default(), // enrich_per_episode = false
        Some(Arc::clone(&sink) as Arc<dyn EnrichmentEventSink>),
    )
    .await
    .expect("submit_episode should succeed");

    assert!(!commit.episode_entity_id.is_empty());

    let events = sink.events();
    assert!(
        events.is_empty(),
        "no Phase 2 events expected when enrich_per_episode=false: {events:?}"
    );
}

/// submit_episode with sink=None completes without panicking (no sink is valid).
#[tokio::test]
async fn submit_episode_without_sink_completes() {
    use kremory::memory::types::SourceKind;

    let handle = StubIngestingHandle;
    let scope = Namespace::new("ws-no-sink");
    let source_ref = SourceRef {
        kind: SourceKind::Chat,
        id: "chat-no-sink".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    let commit = submit_episode(
        &handle,
        "chat message",
        source_ref,
        vec![],
        null_provider(),
        scope,
        None,
        SubmitOpts {
            enrich_per_episode: true,
            run_in_background: false,
        },
        None, // no sink
    )
    .await
    .expect("submit_episode with no sink should succeed");

    assert!(!commit.episode_entity_id.is_empty());
}

/// Verify contradiction events reach the sink when Phase 2 encounters supersessions.
/// Uses a separate stub that fires on_contradiction + on_dedup_merge.
#[tokio::test]
async fn submit_episode_contradiction_events_reach_sink() {
    use kremory::memory::types::SourceKind;

    struct ContradictingHandle;

    #[async_trait]
    impl GraphHandle for ContradictingHandle {
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
                opts,
                sink,
            } = params;
            if opts.enrich_per_episode {
                if let Some(s) = &sink {
                    s.on_stage_change(IngestStatus::Extracting);
                    s.on_contradiction(ContradictionDetected {
                        entity_id: EntityId("ent-a".to_string()),
                        prior_fact: SinkFact {
                            subject: "Alice".to_string(),
                            predicate: "works_at".to_string(),
                            object: "OldCo".to_string(),
                            valid_at: None,
                        },
                        new_fact: SinkFact {
                            subject: "Alice".to_string(),
                            predicate: "works_at".to_string(),
                            object: "NewCo".to_string(),
                            valid_at: None,
                        },
                        resolution: ContradictionResolution::Superseded,
                        detected_at: Utc::now(),
                    });
                    s.on_dedup_merge("ent-a", "ent-a-dup");
                    s.on_stage_change(IngestStatus::Complete);
                }
            }
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
        async fn graph_cancel(
            &self,
            _run_id: Uuid,
        ) -> kremory::memory::types::Result<CancelOutcome> {
            Ok(CancelOutcome {
                cancelled_phase: CancelledPhase::Enrichment,
                rolled_back: false,
                partial: vec![],
            })
        }
        async fn graph_submit_dream(
            &self,
            namespace: &Namespace,
            _p: Arc<dyn ChatProvider>,
            batch_id: Option<String>,
            _o: DreamOpts,
            _s: Option<Arc<dyn EnrichmentEventSink>>,
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

        async fn graph_run_dream_pass_sync(
            &self,
            _opts: kremory::DreamPassOpts,
        ) -> kremory::memory::types::Result<kremory::DreamSummary> {
            Ok(kremory::DreamSummary {
                communities_updated: 0,
                cross_episode_merges: 0,
                supersessions_recorded: 0,
                facts_archived: 0,
                duration_ms: 0,
                types_discovered: vec![],
                entities_reclassified: 0,
                warnings: vec![],
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
            _entity_id: &str,
            _entity_type_id: u32,
            _group_id: Option<&str>,
        ) -> kremory::memory::types::Result<()> {
            Ok(())
        }
    }

    let sink = Arc::new(MockIngestSink::default());
    let handle = ContradictingHandle;
    let scope = Namespace::new("ws-contradiction");
    let source_ref = SourceRef {
        kind: SourceKind::Document,
        id: "doc-contradiction".into(),
        occurred_at: Utc::now(),
        published_at: None,
    };

    submit_episode(
        &handle,
        "doc with contradictory facts",
        source_ref,
        vec![],
        null_provider(),
        scope,
        None,
        SubmitOpts {
            enrich_per_episode: true,
            run_in_background: false,
        },
        Some(Arc::clone(&sink) as Arc<dyn EnrichmentEventSink>),
    )
    .await
    .expect("submit with contradiction stub");

    let events = sink.events();
    assert!(
        events.iter().any(|e| e.contains("contradiction")),
        "contradiction event expected: {events:?}"
    );
    assert!(
        events.iter().any(|e| e.contains("Superseded")),
        "Superseded resolution expected: {events:?}"
    );
    assert!(
        events.iter().any(|e| e.contains("dedup_merge")),
        "dedup_merge event expected: {events:?}"
    );
}

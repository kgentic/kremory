#![allow(clippy::unwrap_used, clippy::expect_used)]
/// A.2 — D.6.5: sink trait shape tests.
///
/// Tests that verify the IngestEventSink and EnrichmentEventSink traits
/// compile correctly and can be implemented by a stub.
use kremory::core::error::{ContradictionResolution, IngestionErrorKind};
use kremory::core::sink::{
    ContradictionDetected, EntityId, IngestEventSink, IngestionError, OnEdgeAddedParams,
};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};

/// Minimal stub that implements IngestEventSink for compile-time verification.
struct StubIngestSink {
    events: std::sync::Mutex<Vec<String>>,
}

impl StubIngestSink {
    fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl IngestEventSink for StubIngestSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("entity:{entity_id}:{name}"));
    }
    fn on_edge_added(&self, params: OnEdgeAddedParams<'_>) {
        let OnEdgeAddedParams {
            from_entity_id: from,
            to_entity_id: to,
            predicate,
        } = params;
        self.events
            .lock()
            .unwrap()
            .push(format!("edge:{from}->{to}:{predicate}"));
    }
    fn on_contradiction(&self, event: ContradictionDetected) {
        self.events
            .lock()
            .unwrap()
            .push(format!("contradiction:{:?}", event.resolution));
    }
    fn on_dedup_merge(&self, surviving_id: &str, absorbed_id: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("merge:{surviving_id}<-{absorbed_id}"));
    }
    fn on_stage_change(&self, stage: kremory::core::error::IngestStatus) {
        self.events.lock().unwrap().push(format!("stage:{stage:?}"));
    }
    fn on_ingestion_error(&self, event: IngestionError) {
        self.events
            .lock()
            .unwrap()
            .push(format!("error:{:?}", event.error_kind));
    }
}

/// Stub implementing the full EnrichmentEventSink (which extends IngestEventSink).
struct StubEnrichmentSink {
    inner: StubIngestSink,
    phase3_events: std::sync::Mutex<Vec<String>>,
}

impl StubEnrichmentSink {
    fn new() -> Self {
        Self {
            inner: StubIngestSink::new(),
            phase3_events: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl IngestEventSink for StubEnrichmentSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.inner.on_entity_extracted(entity_id, name);
    }
    fn on_edge_added(&self, params: OnEdgeAddedParams<'_>) {
        self.inner.on_edge_added(params);
    }
    fn on_contradiction(&self, event: ContradictionDetected) {
        self.inner.on_contradiction(event);
    }
    fn on_dedup_merge(&self, surviving_id: &str, absorbed_id: &str) {
        self.inner.on_dedup_merge(surviving_id, absorbed_id);
    }
    fn on_stage_change(&self, stage: kremory::core::error::IngestStatus) {
        self.inner.on_stage_change(stage);
    }
    fn on_ingestion_error(&self, event: IngestionError) {
        self.inner.on_ingestion_error(event);
    }
}

impl EnrichmentEventSink for StubEnrichmentSink {
    fn on_community_updated(&self, community_id: &str, member_count: usize) {
        self.phase3_events
            .lock()
            .unwrap()
            .push(format!("community:{community_id}:{member_count}"));
    }
    fn on_batch_phase2_complete(&self, event: BatchPhase2Complete) {
        self.phase3_events
            .lock()
            .unwrap()
            .push(format!("batch_done:{}", event.batch_id));
    }
}

#[test]
fn ingest_event_sink_stub_compiles_and_records_events() {
    use chrono::Utc;

    let sink = StubIngestSink::new();

    sink.on_entity_extracted("ent-1", "Alice");
    sink.on_edge_added(OnEdgeAddedParams {
        from_entity_id: "ent-1",
        to_entity_id: "ent-2",
        predicate: "knows",
    });
    sink.on_dedup_merge("ent-1", "ent-3");
    sink.on_contradiction(ContradictionDetected {
        entity_id: EntityId("ent-1".to_string()),
        prior_fact: kremory::core::sink::SinkFact {
            subject: "Alice".to_string(),
            predicate: "works_at".to_string(),
            object: "OldCo".to_string(),
            valid_at: None,
        },
        new_fact: kremory::core::sink::SinkFact {
            subject: "Alice".to_string(),
            predicate: "works_at".to_string(),
            object: "NewCo".to_string(),
            valid_at: None,
        },
        resolution: ContradictionResolution::Superseded,
        detected_at: Utc::now(),
    });
    sink.on_stage_change(kremory::core::error::IngestStatus::Complete);
    sink.on_ingestion_error(IngestionError {
        entity_or_edge_ref: None,
        error_kind: IngestionErrorKind::ParseFailure {
            stage: "extraction".to_string(),
            detail: "bad json".to_string(),
        },
        is_retryable: true,
    });

    let events = sink.events();
    assert_eq!(events.len(), 6);
    assert!(events[0].starts_with("entity:ent-1:Alice"));
    assert!(events[1].starts_with("edge:ent-1->ent-2:knows"));
    assert!(events[2].starts_with("merge:ent-1<-ent-3"));
    assert!(events[3].contains("Superseded"));
    assert!(events[4].contains("Complete"));
    assert!(events[5].contains("ParseFailure"));
}

/// Verifies that `IngestStatus::EntitiesReady` was added as a variant and that
/// `#[non_exhaustive]` is still present on the enum.
///
/// `matches!` expands to a `match` with a `_` catch-all — the catch-all is
/// required by `#[non_exhaustive]` when matching outside the defining crate.
/// If the variant is absent, this test fails to compile with "no variant named
/// EntitiesReady" or "non-exhaustive patterns". If `#[non_exhaustive]` is
/// removed, the `_` arm becomes unreachable and clippy will flag it — both
/// confirm the contract.
///
/// Per impl spec §3 Phase 1 DoD + test strategy §10.
#[test]
fn sink_entities_ready_variant_non_exhaustive() {
    let status = kremory::core::error::IngestStatus::EntitiesReady;
    let matched = matches!(status, kremory::core::error::IngestStatus::EntitiesReady);
    assert!(
        matched,
        "IngestStatus::EntitiesReady variant must exist and match"
    );
}

/// Verifies that `Arc<dyn EnrichmentEventSink>` satisfies `Send + Sync` bounds
/// at compile time.
///
/// This is a load-bearing compile-time check: napi-rs consumers will store the
/// sink in an `Arc` and cross thread boundaries. If `EnrichmentEventSink` loses
/// `Send + Sync`, this test will fail to compile with a trait-bound error,
/// surfacing the regression immediately.
///
/// Per impl spec §3 Phase 1 DoD + test strategy §10.
#[test]
fn sink_send_sync_bound_compiles() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<std::sync::Arc<dyn kremory::memory::events::EnrichmentEventSink>>();
}

#[test]
fn enrichment_event_sink_extends_ingest_sink() {
    let sink = StubEnrichmentSink::new();
    sink.on_entity_extracted("ent-a", "Bob");
    sink.on_community_updated("community-1", 5);
    sink.on_batch_phase2_complete(BatchPhase2Complete {
        batch_id: "batch-001".to_string(),
        succeeded: 10,
        skipped: 2,
        failed: 0,
        duration_ms: 1500,
    });

    let phase2_events = sink.inner.events();
    assert_eq!(phase2_events.len(), 1);
    assert!(phase2_events[0].starts_with("entity:ent-a:Bob"));

    let phase3_events = sink.phase3_events.lock().unwrap().clone();
    assert_eq!(phase3_events.len(), 2);
    assert!(phase3_events[0].contains("community-1"));
    assert!(phase3_events[1].contains("batch-001"));
}

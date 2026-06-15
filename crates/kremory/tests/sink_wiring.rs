#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase 6 — sink callsite wiring tests: Level 1 (compile-time / pure unit)
//! + Level 2 (in-process, direct function calls with real SQLite).
//!
//! Governing spec: `v0-2-3-impl-spec-2026-06-12.md` §6 Phase 6 DoD.
//! Test strategy: `v0-2-3-sink-wiring-test-strategy-2026-06-12.md` §5.
//!
//! # Level 1 tests
//!
//! - `sink_ingest_status_from_sql_status`     — 4+2 SQL values → enum mapping
//! - `sink_none_does_not_panic`               — `sink: None` throughout, no panic
//! - `sink_entities_ready_maps_to_verified_in_sql_bridge` — "Verified"→EntitiesReady (not Complete)
//! - `sink_batch_progress_race_total_increments_before_enqueue` — BatchProgress unit test
//! - `sink_ingestion_error_carry_correct_error_kind` — IngestionError discriminant
//!
//! # Level 2 tests
//!
//! - `sink_entity_extracted_fires_per_entity` — 3 EntityExtracted events for 3 entities
//! - `sink_edge_added_fires_per_episodic_link` — predicate="mention" on every EdgeAdded
//! - `sink_ingestion_error_fires_on_verify_fail` — IngestionError + StageChange(Failed)
//! - `sink_entities_ready_fires_before_complete` — EntitiesReady precedes Complete
//! - `sink_entity_extracted_emits_counter`    — kremory.sink.entity_extracted_total counter
//! - `sink_stage_transition_emits_counter`    — kremory.sink.stage_transition_total counter
//! - `sink_callback_duration_emits_histogram` — kremory.sink.callback_duration_ms histogram
//! - `sink_community_updated_does_not_fire_in_v023` — sentinel: CommunityUpdated never fires
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS REAL: `stage3_write` SQL, status UPDATEs, sink emit sites in substrate
//! - ALWAYS MOCK: `EntityExtractorDyn` — `MockExtractorReturnsEntities` / `MockExtractorFails`
//! - ALWAYS MOCK: `ChatProvider` for verify_llm — `MockChatProvider` with scripted confirm JSON

mod helpers;

use helpers::recording_sink::{RecordingSink, SinkEvent};

use kremory::core::background::batch_tracker::BatchProgress;
use kremory::core::background::verify_stage::run_verify_stage;
use kremory::core::background::DeferredRequest;
use kremory::core::error::{Error, IngestStatus, IngestionErrorKind};
use kremory::core::intelligence::{
    EntityExtractorDyn, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::core::sink::from_sql_status;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shared DB helpers (mirrors verify_stage_metrics.rs pattern exactly)
// ---------------------------------------------------------------------------

async fn open_graph(tag: &str) -> (Arc<TemporalGraph>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("sink-wiring-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (Arc::new(graph), tmp)
}

async fn seed_entity_types(conn: &libsql::Connection) {
    for (id, name, desc) in [
        (0i64, "Entity", "Catch-all"),
        (1i64, "Person", "A human individual"),
        (2i64, "Organization", "A company or group"),
    ] {
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description) \
             VALUES (?1, 'default', ?2, ?3)",
            libsql::params![id, name.to_string(), desc.to_string()],
        )
        .await
        .expect("seed entity_types");
    }
}

async fn insert_pending_episode(conn: &libsql::Connection, content: &str) -> i64 {
    conn.execute(
        "INSERT INTO episodes (content, timestamp, episode_processing_status) \
         VALUES (?1, datetime('now'), 'Pending')",
        libsql::params![content],
    )
    .await
    .expect("insert episode");

    let mut rows = conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("rowid")
}

// ---------------------------------------------------------------------------
// Mock extractors (mirrors verify_stage_metrics.rs exactly)
// ---------------------------------------------------------------------------

struct MockExtractorReturnsEntities {
    entities: Vec<ExtractedEntity>,
}

impl EntityExtractorDyn for MockExtractorReturnsEntities {
    fn name(&self) -> &'static str {
        "mock-returns-entities"
    }

    fn extract_dyn<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = kremory::core::error::Result<ExtractionResult>>
                + Send
                + 'a,
        >,
    > {
        let entities = self.entities.clone();
        Box::pin(async move {
            Ok(ExtractionResult {
                entities,
                facts: Vec::new(),
            })
        })
    }
}

struct MockExtractorFails;

impl EntityExtractorDyn for MockExtractorFails {
    fn name(&self) -> &'static str {
        "mock-always-fails"
    }

    fn extract_dyn<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = kremory::core::error::Result<ExtractionResult>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            Err(Error::Extraction(
                "mock extractor: simulated extraction failure".to_string(),
            ))
        })
    }
}

/// Build a confirm JSON for N candidates (all `action = "confirm"`).
fn build_confirm_json_for(candidate_count: usize) -> String {
    let decisions: Vec<serde_json::Value> = (0..candidate_count)
        .map(|i| {
            serde_json::json!({
                "entity_id": i,
                "action": "confirm",
                "new_type_id": null
            })
        })
        .collect();
    serde_json::json!({ "decisions": decisions }).to_string()
}

// ---------------------------------------------------------------------------
// Level 1 — pure unit (no I/O)
// ---------------------------------------------------------------------------

/// L1: `from_sql_status` correctly maps all 4 SQL values + 2 unknown/edge cases.
///
/// Critical assertion per test strategy TQ4: `"Verified" → EntitiesReady`, NOT `Complete`.
/// Per arch spec §4.2 doc-comment: Phase 2a writes `'Verified'`; `Complete` has no SQL
/// equivalent.  Phase 6 DoD item 3.
#[test]
fn sink_ingest_status_from_sql_status() {
    // Known values
    assert_eq!(from_sql_status("Pending"), IngestStatus::Pending);
    assert_eq!(from_sql_status("Extracting"), IngestStatus::Extracting);
    // KEY: "Verified" → EntitiesReady (not Complete) per ADR-052 D2/D5.
    assert_eq!(
        from_sql_status("Verified"),
        IngestStatus::EntitiesReady,
        "'Verified' must map to EntitiesReady, NOT Complete \
         (SQL writes 'Verified' at Phase 2a; Complete has no SQL equivalent)"
    );
    assert!(
        matches!(from_sql_status("Failed"), IngestStatus::Failed(_)),
        "'Failed' must map to IngestStatus::Failed(_)"
    );
    // Unknown values fall back to Failed (no panic, no unreachable!())
    assert!(
        matches!(
            from_sql_status("SomeUnknownStatus"),
            IngestStatus::Failed(_)
        ),
        "unknown SQL values must fall back to IngestStatus::Failed(_)"
    );
    // Deduplicating / Invalidating are NOT SQL values — bridge rejects them.
    let dedup_result = from_sql_status("Deduplicating");
    assert!(
        matches!(&dedup_result, IngestStatus::Failed(s) if s.contains("unknown_sql_status")),
        "'Deduplicating' is push-only; bridge must return Failed(unknown_sql_status:…)"
    );
}

/// L1: `"Verified"` maps to `IngestStatus::EntitiesReady` — this is the specific
/// correctness invariant that D2/D5 hinge on.  Repeated here as a single-assertion
/// canary so failures point directly at the bridge mapping.
///
/// Per test strategy §5 Tessa-added row (Level 1).
#[test]
fn sink_entities_ready_maps_to_verified_in_sql_bridge() {
    // This is the load-bearing bridge mapping: SQL 'Verified' means Phase 2a
    // entity write complete, NOT Phase 2b fact extraction complete.
    // EntitiesReady (not Complete) is what wait_for_processing resolves to.
    let status = from_sql_status("Verified");
    assert_eq!(
        status,
        IngestStatus::EntitiesReady,
        "from_sql_status(\"Verified\") MUST return EntitiesReady; \
         returning Complete would be semantically wrong (Complete has no SQL equivalent)"
    );
}

/// L1: `sink: None` throughout `run_verify_stage` does not panic and returns the
/// same result as when a sink is wired.
///
/// Per arch spec §3.1: all existing code paths compile and behave identically
/// when sink = None.  Phase 6 DoD item "preserves no-sink behaviour".
#[tokio::test]
async fn sink_none_does_not_panic() {
    let (graph, _tmp) = open_graph("none-no-panic").await;
    seed_entity_types(&graph.conn).await;
    let episode_id =
        insert_pending_episode(&graph.conn, "three distinct speakers attended the meeting").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let request = DeferredRequest {
        text: "three distinct speakers attended the meeting".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string()],
        batch_id: None,
    };

    // sink = None: must not panic, must return Ok
    let result = run_verify_stage(&request, &extractor, None, &graph, None).await;
    assert!(result.is_ok(), "sink=None must not panic; got {result:?}");
}

/// L1: `BatchProgress` unit test — `total` is registered before enqueue and
/// `is_terminal()` only returns true once all terminal counts match.
///
/// This is the race-safety invariant from arch spec §3.3.
/// Per test strategy §5 Tessa-added row.
#[test]
fn sink_batch_progress_race_total_increments_before_enqueue() {
    let mut progress = BatchProgress::new();
    // After first registration (total=1), succeeded/failed=0 → NOT terminal
    assert!(
        !progress.is_terminal(),
        "batch is not terminal when total=1, succeeded+failed=0"
    );

    // Simulate a second episode being registered before enqueue
    progress.total += 1; // total=2

    // After Phase 2 success of first episode: succeeded=1, failed=0 → NOT terminal (need 2)
    progress.succeeded += 1;
    assert!(
        !progress.is_terminal(),
        "batch should not be terminal when succeeded(1) < total(2)"
    );

    // After Phase 2 success of second episode: succeeded=2, failed=0 → terminal
    progress.succeeded += 1;
    assert!(
        progress.is_terminal(),
        "batch MUST be terminal when succeeded(2) == total(2)"
    );
}

/// L1: `IngestionError.error_kind` discriminant correctness.
///
/// Tests that `IngestionErrorKind::ParseFailure` and `IngestionErrorKind::ProviderError`
/// are distinguishable by their variant at the assertion side.
///
/// Per test strategy §5 Tessa-added row.
#[test]
fn sink_ingestion_error_carry_correct_error_kind() {
    // ParseFailure discriminant
    let parse_err = IngestionErrorKind::ParseFailure {
        stage: "extraction".to_string(),
        detail: "bad json".to_string(),
    };
    assert!(
        matches!(&parse_err, IngestionErrorKind::ParseFailure { .. }),
        "ParseFailure variant must be matchable as ParseFailure"
    );
    assert!(
        !matches!(&parse_err, IngestionErrorKind::ProviderError { .. }),
        "ParseFailure variant must NOT match ProviderError"
    );

    // ProviderError discriminant
    let provider_err = IngestionErrorKind::ProviderError {
        provider_name: "llm".to_string(),
        detail: "timeout".to_string(),
    };
    assert!(
        matches!(&provider_err, IngestionErrorKind::ProviderError { .. }),
        "ProviderError variant must be matchable as ProviderError"
    );
    assert!(
        !matches!(&provider_err, IngestionErrorKind::ParseFailure { .. }),
        "ProviderError variant must NOT match ParseFailure"
    );
}

// ---------------------------------------------------------------------------
// Level 2 — in-process, direct function calls (tokio single-thread + real SQLite)
// ---------------------------------------------------------------------------

/// L2: `run_verify_stage` fires exactly 3 `EntityExtracted` events when the
/// extractor returns 3 entities.
///
/// Asserts the `on_entity_extracted` callsite in `stage3_write` fires inside
/// the entity loop — once per entity.
///
/// Per arch spec §3.1 fire-site 2; test strategy §5 table row.
#[tokio::test]
async fn sink_entity_extracted_fires_per_entity() {
    let (graph, _tmp) = open_graph("entity-extracted").await;
    seed_entity_types(&graph.conn).await;

    // Episode content: plain sentences without Title-Case proper nouns
    // to avoid spurious NER hits (per feedback_treat_cause_not_symptom_load_bearing.md).
    let episode_id = insert_pending_episode(
        &graph.conn,
        "three colleagues participated in the discussion",
    )
    .await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Alice".to_string(),
                properties: serde_json::json!({ "confidence": 0.9 }),
            },
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Bob".to_string(),
                properties: serde_json::json!({ "confidence": 0.85 }),
            },
            ExtractedEntity {
                label: "Organization".to_string(),
                name: "Globex".to_string(),
                properties: serde_json::json!({ "confidence": 0.88 }),
            },
        ],
    };

    let confirm_json = build_confirm_json_for(3);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "three colleagues participated in the discussion".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string(), "Bob".to_string(), "Globex".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    let result = run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await;
    assert!(
        result.is_ok(),
        "run_verify_stage must succeed; got {result:?}"
    );

    let entity_events = sink.entity_events();
    assert_eq!(
        entity_events.len(),
        3,
        "exactly 3 EntityExtracted events must fire for 3 entities; got {entity_events:?}"
    );

    // Verify entity names are recorded (order reflects extraction order)
    let names: Vec<&str> = entity_events.iter().map(|(_, n)| n.as_str()).collect();
    assert!(
        names.contains(&"Alice"),
        "Alice must appear in entity events"
    );
    assert!(names.contains(&"Bob"), "Bob must appear in entity events");
    assert!(
        names.contains(&"Globex"),
        "Globex must appear in entity events"
    );
}

/// L2: `run_verify_stage` fires exactly N `EdgeAdded(predicate="mention")` events
/// for N entity writes in `stage3_write`.
///
/// The predicate `"mention"` is hardcoded in `stage3_write` per arch spec §3.1
/// fire-site 3.
///
/// Per test strategy §5 table row.
#[tokio::test]
async fn sink_edge_added_fires_per_episodic_link() {
    let (graph, _tmp) = open_graph("edge-added").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "two participants reviewed the findings").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Alice".to_string(),
                properties: serde_json::json!({ "confidence": 0.9 }),
            },
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Carol".to_string(),
                properties: serde_json::json!({ "confidence": 0.87 }),
            },
        ],
    };

    let confirm_json = build_confirm_json_for(2);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "two participants reviewed the findings".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string(), "Carol".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    let result = run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await;
    assert!(
        result.is_ok(),
        "run_verify_stage must succeed; got {result:?}"
    );

    let edge_events = sink.edge_events();
    assert_eq!(
        edge_events.len(),
        2,
        "exactly 2 EdgeAdded events must fire for 2 entity writes; got {edge_events:?}"
    );

    // ALL edge events must have predicate = "mention" (hardcoded in stage3_write).
    for (_, _, predicate) in &edge_events {
        assert_eq!(
            predicate, "mention",
            "all Stage 3 episodic edge predicates must be 'mention'; got {predicate:?}"
        );
    }
}

/// L2: when `run_verify_stage` extractor returns `Err`, one `IngestionError` fires
/// AND `StageChange(Failed)` fires.
///
/// Per arch spec §3.1 fire-sites 5b + 15 (extract_fail arm).
/// Per test strategy §5 table row `sink_ingestion_error_fires_on_verify_fail`.
///
/// NOTE: `on_ingestion_error` for this failure arm is wired in `deferred_pipeline.rs`
/// (process_deferred), not inside run_verify_stage itself.  run_verify_stage fires
/// `on_stage_change(Failed)` internally.  process_deferred fires `on_ingestion_error`
/// after run_verify_stage returns Err.  This test calls run_verify_stage directly —
/// so only `StageChange(Failed)` fires here (not IngestionError).
/// For the full chain (including IngestionError), see sink_wiring_integration.rs.
///
/// This test asserts the invariant that `run_verify_stage` correctly fires
/// `StageChange(Failed)` on extraction failure.
#[tokio::test]
async fn sink_ingestion_error_fires_on_verify_fail() {
    let (graph, _tmp) = open_graph("verify-fail").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(&graph.conn, "this will fail during extraction").await;

    let extractor = MockExtractorFails;

    let request = DeferredRequest {
        text: "this will fail during extraction".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: Vec::new(),
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    let result = run_verify_stage(&request, &extractor, None, &graph, Some(sink_ref)).await;
    assert!(
        result.is_err(),
        "MockExtractorFails must produce Err; got Ok"
    );

    let stage_events = sink.stage_events();
    // run_verify_stage fires: Extracting (after status update) then Failed (on extract error)
    let has_extracting = stage_events.contains(&IngestStatus::Extracting);
    let has_failed = stage_events
        .iter()
        .any(|s| matches!(s, IngestStatus::Failed(_)));
    assert!(
        has_extracting,
        "Extracting stage-change must fire before extraction attempt; stages: {stage_events:?}"
    );
    assert!(
        has_failed,
        "Failed stage-change must fire on extraction failure; stages: {stage_events:?}"
    );
    // Entities extracted: none (extractor failed before any writes)
    assert_eq!(
        sink.entity_events().len(),
        0,
        "no EntityExtracted events must fire when extractor fails"
    );
}

/// L2: `StageChange(EntitiesReady)` fires before `StageChange(Complete)` in the
/// event sequence.
///
/// Asserts the phase 2a/2b ordering invariant from ADR-052 D5.
/// Setup: call `run_verify_stage` (→ fires Extracting + EntitiesReady),
/// then manually fire Complete by checking that EntitiesReady appears in snapshot.
///
/// Full Pending→Extracting→EntitiesReady→Complete ordering is tested in
/// sink_wiring_integration.rs `sink_stage_changes_fire_in_order` (L3).
/// This L2 test focuses on the Phase 2a half: verify_stage fires EntitiesReady.
///
/// Per test strategy §5 table row + TQ5 answer.
#[tokio::test]
async fn sink_entities_ready_fires_before_complete() {
    let (graph, _tmp) = open_graph("entities-ready-order").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(&graph.conn, "two team members discussed scope").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Bob".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let confirm_json = build_confirm_json_for(1);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "two team members discussed scope".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Bob".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    // Phase 2a: run_verify_stage fires Extracting → EntitiesReady on success.
    let result = run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await;
    assert!(
        result.is_ok(),
        "run_verify_stage must succeed; got {result:?}"
    );

    let stage_events_2a = sink.stage_events();

    // After Phase 2a: must contain Extracting and EntitiesReady (not Complete yet).
    let has_entities_ready = stage_events_2a.contains(&IngestStatus::EntitiesReady);
    let has_complete = stage_events_2a.contains(&IngestStatus::Complete);

    assert!(
        has_entities_ready,
        "EntitiesReady must fire after run_verify_stage success; stages: {stage_events_2a:?}"
    );
    assert!(
        !has_complete,
        "Complete must NOT fire from run_verify_stage (it fires after ingest_deferred); \
         stages: {stage_events_2a:?}"
    );

    // TQ5: exactly ONE EntitiesReady event fires per episode (not duplicated).
    let entities_ready_count = stage_events_2a
        .iter()
        .filter(|s| **s == IngestStatus::EntitiesReady)
        .count();
    assert_eq!(
        entities_ready_count, 1,
        "EntitiesReady must fire exactly once per run_verify_stage success call; \
         got {entities_ready_count}"
    );

    // Verify ordering: Extracting appears before EntitiesReady in the list.
    let extracting_idx = stage_events_2a
        .iter()
        .position(|s| *s == IngestStatus::Extracting)
        .expect("Extracting must appear in stage events");
    let entities_ready_idx = stage_events_2a
        .iter()
        .position(|s| *s == IngestStatus::EntitiesReady)
        .expect("EntitiesReady must appear in stage events");
    assert!(
        extracting_idx < entities_ready_idx,
        "Extracting (idx {extracting_idx}) must precede EntitiesReady (idx {entities_ready_idx})"
    );
}

/// L2: `kremory.sink.entity_extracted_total` counter increments per entity write.
///
/// Uses `DebuggingRecorder` + `set_default_local_recorder` (thread-local scope)
/// per verify_stage_metrics.rs canonical pattern.
///
/// IMPORTANT: metrics are asserted at Level 2 ONLY (thread-local recorder scope).
/// Level 3 tests use `RecordingSink.snapshot()` for assertions — NOT metrics
/// (per Tessa T2/TQ2 ruling: DebuggingRecorder does not cross OS thread boundary).
///
/// Per test strategy §5 table row.
#[tokio::test]
async fn sink_entity_extracted_emits_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("entity-counter").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(&graph.conn, "two people met at the summit").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Alice".to_string(),
                properties: serde_json::json!({ "confidence": 0.9 }),
            },
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Bob".to_string(),
                properties: serde_json::json!({ "confidence": 0.85 }),
            },
        ],
    };

    let confirm_json = build_confirm_json_for(2);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "two people met at the summit".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string(), "Bob".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    let result = run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await;
    assert!(
        result.is_ok(),
        "run_verify_stage must succeed; got {result:?}"
    );

    let snapshot = snapshotter.snapshot();
    let entity_counter_sum: u64 = snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() == "kremory.sink.entity_extracted_total" {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum();

    // Note: verify_stage.rs emits entity_extracted counter at the call-site wrapper
    // (MED-04 pattern: counter is arm-labelled at call site, not inside stage3_write).
    // The entity_extracted_total counter must be > 0 after 2 entities were written.
    assert!(
        entity_counter_sum > 0,
        "kremory.sink.entity_extracted_total counter must emit ≥1 after entity writes; \
         got sum={entity_counter_sum}"
    );
}

/// L2: `kremory.sink.stage_transition_total` counter increments per stage transition.
///
/// Uses `DebuggingRecorder` + `set_default_local_recorder` (Level 2 only — thread-local).
/// Per test strategy §5 table row.
#[tokio::test]
async fn sink_stage_transition_emits_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("stage-counter").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "the workshop started with introductions").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Carol".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let confirm_json = build_confirm_json_for(1);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "the workshop started with introductions".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Carol".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await
    .expect("run_verify_stage must succeed");

    let snapshot = snapshotter.snapshot();
    let stage_counter_total: u64 = snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() == "kremory.sink.stage_transition_total" {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum();

    assert!(
        stage_counter_total > 0,
        "kremory.sink.stage_transition_total must emit ≥1 after stage transitions; \
         got sum={stage_counter_total}"
    );
}

/// L2: `kremory.sink.callback_duration_ms` histogram records ≥1 value after
/// a stage-change sink call.
///
/// Uses `DebuggingRecorder` (Level 2 — thread-local; NOT usable in L3 multi_thread).
/// Per test strategy §7.1 + TQ2 + §5 table row.
#[tokio::test]
async fn sink_callback_duration_emits_histogram() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("callback-duration").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(&graph.conn, "the session covered key outcomes").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let confirm_json = build_confirm_json_for(1);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "the session covered key outcomes".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await
    .expect("run_verify_stage must succeed");

    let snapshot = snapshotter.snapshot();
    let has_histogram = snapshot
        .into_vec()
        .into_iter()
        .any(|(composite_key, _, _, value)| {
            composite_key.key().name() == "kremory.sink.callback_duration_ms"
                && matches!(value, DebugValue::Histogram(_))
        });

    assert!(
        has_histogram,
        "kremory.sink.callback_duration_ms histogram must emit ≥1 value after \
         run_verify_stage with a sink; no histogram found in snapshot"
    );
}

/// L2 sentinel: `CommunityUpdated` must NOT appear in any `RecordingSink` event list
/// during a standard verify-stage ingest cycle.
///
/// `on_community_updated` is deferred to ADR-050 (dream-pass sprint).  The
/// `RecordingSink` includes a `CommunityUpdated` sentinel arm so this test catches
/// any forward implementation before ADR-050's test harness is ready.
///
/// Per test strategy §12 (Tessa recommendation) + impl spec Phase 6 DoD sentinel.
#[tokio::test]
async fn sink_community_updated_does_not_fire_in_v023() {
    let (graph, _tmp) = open_graph("community-sentinel").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(
        &graph.conn,
        "this episode tests the community updated sentinel",
    )
    .await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Bob".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let confirm_json = build_confirm_json_for(1);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "this episode tests the community updated sentinel".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Bob".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await
    .expect("run_verify_stage must succeed");

    let community_events: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter(|e| *e == SinkEvent::CommunityUpdated)
        .collect();

    assert!(
        community_events.is_empty(),
        "CommunityUpdated MUST NOT fire in v0.2.3 (deferred to ADR-050); \
         got {community_events:?}"
    );
}

/// L2: On extraction failure, `EntitiesReady` must NOT appear in stage events.
///
/// TQ5 failure-path assertion: when `run_verify_stage` returns `Err` (extraction
/// failed), only `Extracting` and `Failed` may appear in stage events.
/// `EntitiesReady` MUST NOT fire because the entity write never completed.
///
/// Per test strategy §9-TQ5 ("Failure-path assertion").
#[tokio::test]
async fn sink_extract_fail_does_not_fire_entities_ready() {
    let (graph, _tmp) = open_graph("extract-fail-no-entities-ready").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(
        &graph.conn,
        "the extraction will fail before any entities write",
    )
    .await;

    let extractor = MockExtractorFails;

    let request = DeferredRequest {
        text: "the extraction will fail before any entities write".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: Vec::new(),
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    let result = run_verify_stage(&request, &extractor, None, &graph, Some(sink_ref)).await;
    assert!(result.is_err(), "MockExtractorFails must produce Err");

    let stage_events = sink.stage_events();

    // EntitiesReady must NOT fire when extraction failed.
    assert!(
        !stage_events.contains(&IngestStatus::EntitiesReady),
        "EntitiesReady must NOT fire when extraction fails; \
         got stage_events: {stage_events:?}"
    );

    // Failed must fire (verifies the error path fires the correct stage event).
    assert!(
        stage_events
            .iter()
            .any(|s| matches!(s, IngestStatus::Failed(_))),
        "Failed must fire when extraction fails; stage_events: {stage_events:?}"
    );
}

/// L2: `run_verify_stage` does NOT fire `StageChange(Pending)`.
///
/// `Pending` is the Phase 1 (process_item) fire-site.  `run_verify_stage` starts
/// at `Extracting` (Phase 2a).  This test asserts the phase boundary is respected:
/// only `Extracting`, `EntitiesReady` (or `Failed`) fire from `run_verify_stage`.
///
/// Per arch spec §3.1 fire-site table: `on_stage_change(Pending)` is in
/// `deferred_pipeline.rs::process_item`, NOT in `verify_stage.rs::run_verify_stage`.
#[tokio::test]
async fn sink_verify_stage_does_not_fire_pending() {
    let (graph, _tmp) = open_graph("verify-stage-no-pending").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "only phase two events should fire here").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let confirm_json = build_confirm_json_for(1);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "only phase two events should fire here".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string()],
        batch_id: None,
    };

    let sink = RecordingSink::new();
    let sink_ref = &sink as &dyn kremory::memory::events::EnrichmentEventSink;

    run_verify_stage(
        &request,
        &extractor,
        Some(&verify_llm),
        &graph,
        Some(sink_ref),
    )
    .await
    .expect("run_verify_stage must succeed");

    let stage_events = sink.stage_events();

    // Pending must NOT fire from run_verify_stage (it fires from process_item only).
    assert!(
        !stage_events.contains(&IngestStatus::Pending),
        "Pending must NOT fire from run_verify_stage; it is a process_item fire-site; \
         stage_events: {stage_events:?}"
    );

    // Extracting must fire (confirming we started from the right state).
    assert!(
        stage_events.contains(&IngestStatus::Extracting),
        "Extracting must fire from run_verify_stage on success; \
         stage_events: {stage_events:?}"
    );
}

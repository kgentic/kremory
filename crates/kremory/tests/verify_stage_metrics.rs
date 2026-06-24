#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase 5 DoD item 1 — metric emission tests for `run_verify_stage`
//!
//! Governing spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md`
//!   Phase 5 DoD #1: "NEW test: `verify_stage_emits_outcome_counter_per_arm`"
//!
//! Validates that `run_verify_stage` emits the canonical
//! `kremory.verify_stage.outcome_total{arm, outcome}` counter on each code path:
//!
//! 1. Path α happy path → `{arm="gliner", outcome="success"}` increments by 1.
//! 2. Path β happy path → `{arm="llm_extract", outcome="success"}` increments by 1.
//! 3. Extractor failure  → `{arm=*, outcome="gliner_fail"}` increments by 1.
//!
//! # Harness
//!
//! Uses `metrics_util::debugging::DebuggingRecorder` + `metrics::set_default_local_recorder`
//! (thread-local recorder scope; each test installs its own recorder so tests are
//! independent). Pattern matches `consistency_check_phase_c.rs::c7_observability` and
//! `ner_extraction.rs` — both use the same `DebuggingRecorder` + `Snapshot::into_vec()` pattern.
//!
//! Counter values are read by iterating `snapshot.into_vec()` (returns
//! `Vec<(CompositeKey, Option<Unit>, Option<SharedString>, DebugValue)>`) and
//! matching on `composite_key.key().name()` + label key-value pairs.
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS REAL: `stage3_write` SQL, status UPDATEs, counter emit sites in `verify_stage.rs`
//! - ALWAYS MOCK: `EntityExtractorDyn` — `MockExtractorReturnsEntities` / `MockExtractorFails`
//! - ALWAYS MOCK: `ChatProvider` for verify_llm — `MockChatProvider` with scripted confirm JSON

use kremory::core::background::verify_stage::{run_verify_stage, RunVerifyStageParams};
use kremory::core::background::DeferredRequest;
use kremory::core::error::Error;
use kremory::core::intelligence::{
    EntityExtractorDyn, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use std::sync::Arc;

// ─── Graph helpers ────────────────────────────────────────────────────────────

async fn open_graph(tag: &str) -> (Arc<TemporalGraph>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("verify-stage-metrics-{tag}.db"));
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

// ─── Mock extractors ──────────────────────────────────────────────────────────

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

// ─── Counter query helper ─────────────────────────────────────────────────────

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

/// Sum `kremory.verify_stage.outcome_total` for a specific arm+outcome label pair.
///
/// Takes a `Snapshot` (consumed). Snapshot::into_vec() returns
/// `Vec<(CompositeKey, Option<Unit>, Option<SharedString>, DebugValue)>`.
/// `CompositeKey::key()` gives the inner `metrics::Key`.
///
/// Per [[observability-first-class]] Rule 19: query per label-dimension —
/// aggregate sum hides per-arm attribution.
fn sum_outcome_counter(
    snapshot: metrics_util::debugging::Snapshot,
    arm: &str,
    outcome: &str,
) -> u64 {
    snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.verify_stage.outcome_total" {
                return None;
            }
            let labels: std::collections::HashMap<&str, &str> =
                key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("arm").copied() == Some(arm)
                && labels.get("outcome").copied() == Some(outcome)
            {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum()
}

/// Sum all `kremory.verify_stage.outcome_total` entries matching only `outcome` label
/// (arm-agnostic; used for the failure test where we don't want to over-specify arm).
fn sum_outcome_counter_any_arm(snapshot: metrics_util::debugging::Snapshot, outcome: &str) -> u64 {
    snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.verify_stage.outcome_total" {
                return None;
            }
            let labels: std::collections::HashMap<&str, &str> =
                key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("outcome").copied() == Some(outcome) {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum()
}

// ─── Test 1: Path α → {arm="gliner", outcome="success"} ─────────────────────

/// Path α happy path emits `kremory.verify_stage.outcome_total{arm="gliner", outcome="success"}`
/// with value exactly 1.
///
/// Spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` Phase 5 DoD #1.
#[tokio::test]
async fn verify_stage_path_alpha_emits_success_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("metrics-alpha").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Alice met Bob at the Acme conference.").await;

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
        text: "Alice met Bob at the Acme conference.".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Alice".to_string(), "Bob".to_string()],
        batch_id: None,
    };

    let result = run_verify_stage(RunVerifyStageParams {
        model: None,
        allowed_entity_types: &[],
        excluded_entity_types: &[],
        request: &request,
        extractor: &extractor,
        verify_llm: Some(&verify_llm),
        graph: &graph,
        sink: None,
    })
    .await;
    assert!(
        result.is_ok(),
        "Path α must return Ok; got: {:?}",
        result.err()
    );

    let count = sum_outcome_counter(snapshotter.snapshot(), "gliner", "success");
    assert_eq!(
        count, 1,
        "kremory.verify_stage.outcome_total{{arm=gliner, outcome=success}} must be exactly 1 \
         after Path α happy path; got: {count}"
    );
}

// ─── Test 2: Path β → {arm="llm_extract", outcome="success"} ────────────────

/// Path β happy path emits `kremory.verify_stage.outcome_total{arm="llm_extract", outcome="success"}`
/// with value exactly 1.
///
/// Spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` Phase 5 DoD #1.
#[tokio::test]
async fn verify_stage_path_beta_emits_success_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("metrics-beta").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Carol joined Globex as head of engineering.").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Carol".to_string(),
                properties: serde_json::json!({ "confidence": 0.92 }),
            },
            ExtractedEntity {
                label: "Organization".to_string(),
                name: "Globex".to_string(),
                properties: serde_json::json!({ "confidence": 0.88 }),
            },
        ],
    };

    let request = DeferredRequest {
        text: "Carol joined Globex as head of engineering.".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: vec!["Carol".to_string(), "Globex".to_string()],
        batch_id: None,
    };

    // Path β: verify_llm = None.
    let result = run_verify_stage(RunVerifyStageParams {
        model: None,
        allowed_entity_types: &[],
        excluded_entity_types: &[],
        request: &request,
        extractor: &extractor,
        verify_llm: None,
        graph: &graph,
        sink: None,
    })
    .await;
    assert!(
        result.is_ok(),
        "Path β must return Ok; got: {:?}",
        result.err()
    );

    let count = sum_outcome_counter(snapshotter.snapshot(), "llm_extract", "success");
    assert_eq!(
        count, 1,
        "kremory.verify_stage.outcome_total{{arm=llm_extract, outcome=success}} must be exactly 1 \
         after Path β happy path; got: {count}"
    );
}

// ─── Test 3: Extractor failure → {arm=*, outcome="gliner_fail"} ───────────────

/// Extractor failure emits `kremory.verify_stage.outcome_total{arm=*, outcome="gliner_fail"}`
/// with total value exactly 1.
///
/// We pass `Some(&llm)` to exercise Path α's arm-selection (arm="gliner" is set
/// before the extractor fires). The failure counter uses `outcome="gliner_fail"` on
/// both Path α and Path β arm choices — querying by outcome-only is arm-agnostic.
///
/// Spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` Phase 5 DoD #1.
#[tokio::test]
async fn verify_stage_extractor_failure_emits_gliner_fail_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph("metrics-fail").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Dan presented the quarterly results.").await;

    let extractor = MockExtractorFails;
    // Pass Some(&llm) to enter Path α branch — arm="gliner" is set before extractor fires.
    let verify_llm = MockChatProvider::null();

    let request = DeferredRequest {
        text: "Dan presented the quarterly results.".to_string(),
        reference_time: None,
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: Vec::new(),
        batch_id: None,
    };

    let result = run_verify_stage(RunVerifyStageParams {
        model: None,
        allowed_entity_types: &[],
        excluded_entity_types: &[],
        request: &request,
        extractor: &extractor,
        verify_llm: Some(&verify_llm),
        graph: &graph,
        sink: None,
    })
    .await;
    assert!(result.is_err(), "extractor failure must return Err; got Ok");

    let gliner_fail_total = sum_outcome_counter_any_arm(snapshotter.snapshot(), "gliner_fail");
    assert_eq!(
        gliner_fail_total, 1,
        "kremory.verify_stage.outcome_total{{outcome=gliner_fail}} must be exactly 1 \
         after extractor failure; got: {gliner_fail_total}"
    );
}

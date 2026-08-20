#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for `verify_stage::run_verify_stage` — Phase 2 DoD.
//!
//! Governing spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` Phase 2.
//! ADR-051: GLiNER-to-background unified hot path.
//!
//! # Coverage (Phase 2 DoD item 8 — three tests)
//!
//! 1. `run_verify_stage_path_alpha_writes_entities_and_transitions_status_to_verified`
//!    — Path α: mock GLiNER extractor → mock verify_llm → entities written → status Verified.
//!
//! 2. `run_verify_stage_path_beta_writes_entities_and_transitions_status_to_verified`
//!    — Path β: mock LLM extractor (verify_llm = None) → entities written → status Verified.
//!
//! 3. `run_verify_stage_failure_transitions_status_to_failed`
//!    — Extractor returns Err → status transitions to Failed, Err propagated.
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS REAL: `stage3_write` SQL, status UPDATEs, episodic_edges, FTS rows
//! - ALWAYS MOCK: `EntityExtractorDyn` — `MockExtractorReturnsEntities` / `MockExtractorFails`
//! - ALWAYS MOCK: `ChatProvider` for verify_llm — `MockChatProvider` with scripted confirm JSON
//! - NOT TESTED HERE: GLiNER model loading (requires `--features ner` + model file)

use std::sync::Arc;

use kremory::core::background::verify_stage::{run_verify_stage, RunVerifyStageParams};
use kremory::core::background::DeferredRequest;
use kremory::core::error::Error;
use kremory::core::intelligence::{
    EntityExtractorDyn, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Open a file-backed `TemporalGraph` in an isolated temp directory.
async fn open_graph(tag: &str) -> (Arc<TemporalGraph>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("verify-stage-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (Arc::new(graph), tmp)
}

/// Seed a minimal entity_type registry row (id=0 catch-all required by the pipeline).
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

/// Insert a minimal episode row and return its id, with status = 'Pending'.
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

/// Read `episode_processing_status` for a given episode id.
async fn read_status(conn: &libsql::Connection, episode_id: i64) -> String {
    let mut rows = conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![episode_id],
        )
        .await
        .expect("status query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<String>(0)
        .expect("status col")
}

/// Count rows in `entities` table.
async fn count_entities(conn: &libsql::Connection) -> i64 {
    let mut rows = conn
        .query("SELECT COUNT(*) FROM entities", ())
        .await
        .expect("count entities");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count")
}

// ─── Mock extractors ──────────────────────────────────────────────────────────

/// Mock extractor that returns a fixed set of entity candidates.
///
/// Used for both Path α (GLiNER-style; entities will be typed by verify_llm)
/// and Path β (LLM-style; entities are treated as already typed via Confirm).
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

/// Mock extractor that always returns an extraction error.
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

// ─── build_confirm_json ────────────────────────────────────────────────────────

/// Build the `verify_batch` JSON response that Confirms all candidates.
///
/// `verify_batch_for_candidates` calls `verify_batch` which expects a JSON
/// array of decision objects: `[{"entity_id": N, "action": "confirm"}, ...]`.
/// We return one Confirm decision for each candidate_idx passed in.
///
/// The mock ChatProvider does substring matching on the last user message;
/// we use an empty key `""` so the fallback empty-string response is returned
/// when no key matches, which results in empty decisions → all Demote defaults.
/// For a real Confirm, we need scripted JSON. We use the `"verify"` key that
/// matches the verify_batch prompt preamble.
fn build_confirm_json_for(candidate_count: usize) -> String {
    let decisions: Vec<serde_json::Value> = (0..candidate_count)
        .map(|i| {
            serde_json::json!({
                "entity_id": i,   // verify_batch uses rowid; -1 for Stage 2 pre-write
                "action": "confirm",
                "new_type_id": null
            })
        })
        .collect();
    serde_json::json!({ "decisions": decisions }).to_string()
}

// ─── Test 1: Path α — GLiNER + verify_llm → Verified ─────────────────────────

/// Path α: extractor returns two entity candidates; verify_llm confirms them;
/// Stage 3 writes them; status transitions Pending → Extracting → Verified.
///
/// Asserts:
/// - `run_verify_stage` returns `Ok(n)` where n > 0
/// - `episodes.episode_processing_status` == 'Verified'
/// - `entities` row count == n
#[tokio::test]
async fn run_verify_stage_path_alpha_writes_entities_and_transitions_status_to_verified() {
    let (graph, _tmp) = open_graph("path-alpha").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Alice met Bob at the Acme conference.").await;

    // Extractor: two candidates (Alice, Bob) — GLiNER-style, type_id=0 (catch-all).
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

    // verify_llm: MockChatProvider scripted to confirm all candidates.
    // verify_batch prompt contains "verify" in the system message; the mock
    // matches on any substring of the last user message. We use a broad match.
    let confirm_json = build_confirm_json_for(2);
    let verify_llm = MockChatProvider::with_response("entity", confirm_json);

    let request = DeferredRequest {
        text: "Alice met Bob at the Acme conference.".to_string(),
        reference_time: None,
        declared_reference_time: None, // TD-187 Gap 1 (2026-08-20): pre-existing fixture, unaffected by the fix
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
    let entities_written = result.unwrap();
    assert!(
        entities_written > 0,
        "Path α must write at least 1 entity; got {entities_written}"
    );

    let status = read_status(&graph.conn, episode_id).await;
    assert_eq!(
        status, "Verified",
        "episode_processing_status must be 'Verified' after Path α success; got: {status:?}"
    );

    let entity_count = count_entities(&graph.conn).await;
    assert_eq!(
        entity_count, entities_written as i64,
        "entities table row count must equal entities_written; \
         table={entity_count}, returned={entities_written}"
    );
}

// ─── Test 2: Path β — LLM extract direct → Verified ──────────────────────────

/// Path β: verify_llm = None; extractor IS the LLM extractor; entities written
/// via Confirm decisions; status transitions Pending → Extracting → Verified.
///
/// Asserts:
/// - `run_verify_stage` returns `Ok(n)` where n > 0
/// - `episodes.episode_processing_status` == 'Verified'
/// - `entities` row count == n
#[tokio::test]
async fn run_verify_stage_path_beta_writes_entities_and_transitions_status_to_verified() {
    let (graph, _tmp) = open_graph("path-beta").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Carol joined Globex as head of engineering.").await;

    // Extractor: returns typed entities (LLM-style, types already known).
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
        declared_reference_time: None, // TD-187 Gap 1 (2026-08-20): pre-existing fixture, unaffected by the fix
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
    let entities_written = result.unwrap();
    assert!(
        entities_written > 0,
        "Path β must write at least 1 entity; got {entities_written}"
    );

    let status = read_status(&graph.conn, episode_id).await;
    assert_eq!(
        status, "Verified",
        "episode_processing_status must be 'Verified' after Path β success; got: {status:?}"
    );

    let entity_count = count_entities(&graph.conn).await;
    assert_eq!(
        entity_count, entities_written as i64,
        "entities table row count must equal entities_written; \
         table={entity_count}, returned={entities_written}"
    );
}

// ─── Test 3: Extractor failure → Failed status ────────────────────────────────

/// Extractor returns `Err` — `run_verify_stage` must:
/// 1. Transition episode status to 'Failed' BEFORE returning `Err`.
/// 2. Return the `Err` to the caller.
///
/// This validates the ADR-051 state machine invariant: the Failed write is
/// load-bearing and must happen even when the function itself errors out.
#[tokio::test]
async fn run_verify_stage_failure_transitions_status_to_failed() {
    let (graph, _tmp) = open_graph("failure-path").await;
    seed_entity_types(&graph.conn).await;

    let episode_id =
        insert_pending_episode(&graph.conn, "Dan presented the quarterly results.").await;

    let extractor = MockExtractorFails;
    // `MockChatProvider::null()` is the never-reached placeholder: the extractor
    // returns Err on the very first call, so `verify_batch_for_candidates` (which
    // would consume the LLM) is unreachable. We pass `Some(&llm)` to exercise
    // Path α's branch decision, but the LLM is never invoked in this scenario.
    let verify_llm = MockChatProvider::null();

    let request = DeferredRequest {
        text: "Dan presented the quarterly results.".to_string(),
        reference_time: None,
        declared_reference_time: None, // TD-187 Gap 1 (2026-08-20): pre-existing fixture, unaffected by the fix
        group_id: None,
        content_type: None,
        episode_id,
        ner_entity_names: Vec::new(),
        batch_id: None,
    };

    // Path α with a failing extractor.
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
        result.is_err(),
        "run_verify_stage must return Err when extractor fails; got Ok"
    );

    let status = read_status(&graph.conn, episode_id).await;
    assert_eq!(
        status, "Failed",
        "episode_processing_status must be 'Failed' after extractor error; got: {status:?}"
    );

    // No entities must have been written (extractor failed before Stage 3).
    let entity_count = count_entities(&graph.conn).await;
    assert_eq!(
        entity_count, 0,
        "no entities must be written when extractor fails; got {entity_count}"
    );
}

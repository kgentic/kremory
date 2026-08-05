#![allow(clippy::unwrap_used, clippy::expect_used)]
//! T2.2 (sprint plan `v0-2-0-phase-b-prep-sprint-plan-2026-06-10`) —
//! C1 confidence gate `<0.7 → downgrade to Confirm` on `verify_batch`.
//!
//! Governing references:
//! - Sprint plan T2.2 DoD: "New unit test `verify_batch_low_confidence_downgrades`"
//! - Basket item #83 — gbrain C1 confidence gate `<0.7 → no_contradiction`
//! - ADR-049 §verify_batch contract — gate enforced structurally at decision time
//! - CLAUDE.md `feedback_load_bearing_invariants_at_emit_not_prompt` — guard
//!   enforced in code, not via prompt instruction
//!
//! ## Invariants under test
//!
//! When `decision.action = "correct"` AND `decision.confidence < 0.7`:
//! - `VerifyBatchOutcome.decisions` entry MUST be `VerifyAction::Confirm`
//!   (NOT `VerifyAction::Correct`).
//! - `counts.confirmed` MUST increment; `counts.corrected` MUST NOT.
//! - The entity row MUST NOT be UPDATEd.
//! - No `dream_pass4_audit` row written.
//!
//! When `decision.confidence >= 0.7` (boundary + above), the correction
//! applies normally.

use std::sync::Arc;

use kremory::core::dream::consistency_check::{
    verify_batch, CandidateRow, VerifyAction, VerifyBatchParams,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;

async fn open_graph(tag: &str) -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("conf-gate-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (graph, tmp)
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

async fn seed_entity(conn: &libsql::Connection, id: &str, type_id: i64) -> i64 {
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id, entity_type_source) \
         VALUES (?1, ?2, '2026-06-10T00:00:00Z', 'default', 'Phase1Ner')",
        libsql::params![id.to_string(), type_id],
    )
    .await
    .expect("seed entity");

    let mut rows = conn
        .query(
            "SELECT rowid FROM entities WHERE id = ?1",
            libsql::params![id.to_string()],
        )
        .await
        .expect("select rowid");
    let row = rows
        .next()
        .await
        .expect("rowid row")
        .expect("rowid present");
    row.get::<i64>(0).expect("rowid i64")
}

fn mock_correct_with_confidence(
    entity_id: i64,
    new_type_id: i64,
    confidence: f64,
) -> MockChatProvider {
    let response = serde_json::json!({
        "decisions": [{
            "entity_id": entity_id,
            "action": "correct",
            "new_type_id": new_type_id,
            "confidence": confidence
        }]
    })
    .to_string();
    let mut map = std::collections::HashMap::new();
    map.insert("entity type".to_string(), response);
    MockChatProvider::new(map)
}

async fn current_type_id(conn: &libsql::Connection, entity_id: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT entity_type_id FROM entities WHERE id = ?1",
            libsql::params![entity_id.to_string()],
        )
        .await
        .expect("select type_id");
    let row = rows
        .next()
        .await
        .expect("type_id row")
        .expect("type_id present");
    row.get::<i64>(0).expect("type_id i64")
}

async fn audit_row_count(conn: &libsql::Connection, entity_rowid: i64) -> i64 {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM dream_pass4_audit WHERE entity_id = ?1",
            libsql::params![entity_rowid],
        )
        .await
        .expect("count audit");
    let row = rows
        .next()
        .await
        .expect("count row")
        .expect("count present");
    row.get::<i64>(0).expect("count i64")
}

fn make_type_map() -> std::collections::HashMap<i64, (String, String)> {
    let mut m = std::collections::HashMap::new();
    m.insert(0i64, ("Entity".to_string(), "Catch-all".to_string()));
    m.insert(
        1i64,
        ("Person".to_string(), "A human individual".to_string()),
    );
    m.insert(
        2i64,
        ("Organization".to_string(), "A company or group".to_string()),
    );
    m
}

/// Core T2.2 invariant: confidence below 0.7 downgrades correction to Confirm.
#[tokio::test]
async fn verify_batch_low_confidence_downgrades() {
    let (graph, _tmp) = open_graph("low_conf").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await;

    // Confidence 0.65 is below the 0.7 threshold — should downgrade.
    let mock_llm = Arc::new(mock_correct_with_confidence(rowid, 2, 0.65));
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];
    let type_map = make_type_map();

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-low-conf-1",
            dry_run: false,
        },
    )
    .await
    .expect("verify_batch low confidence");

    // Decision is downgraded to Confirm.
    assert_eq!(outcome.decisions.len(), 1);
    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Confirm),
        "low-confidence correct must downgrade to Confirm"
    );
    assert_eq!(
        outcome.decisions[0].new_type_id, None,
        "downgraded Confirm must have new_type_id=None"
    );

    // Counts reflect Confirm path, not Correct path.
    assert_eq!(outcome.counts.confirmed, 1, "confirmed count must be 1");
    assert_eq!(outcome.counts.corrected, 0, "corrected count must be 0");

    // Entity table UNCHANGED.
    let current = current_type_id(&graph.conn, "apple").await;
    assert_eq!(
        current, 0,
        "low-confidence correct must NOT UPDATE entity_type_id"
    );

    // No audit row.
    assert_eq!(
        audit_row_count(&graph.conn, rowid).await,
        0,
        "low-confidence correct must NOT write dream_pass4_audit row"
    );
}

/// Regression baseline: confidence at/above 0.7 still applies the correction.
#[tokio::test]
async fn verify_batch_at_threshold_confidence_applies_correction() {
    let (graph, _tmp) = open_graph("at_thresh").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await;

    // 0.7 is the exact threshold — boundary case, must NOT downgrade
    // (gate is strict `<`, so >= 0.7 passes).
    let mock_llm = Arc::new(mock_correct_with_confidence(rowid, 2, 0.7));
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];
    let type_map = make_type_map();

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-at-thresh",
            dry_run: false,
        },
    )
    .await
    .expect("verify_batch at threshold");

    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "confidence at threshold (0.7) must apply correction"
    );
    assert_eq!(outcome.counts.corrected, 1);
    assert_eq!(
        current_type_id(&graph.conn, "apple").await,
        2,
        "boundary confidence must apply correction"
    );
}

/// High-confidence correction applies as expected (sanity check).
#[tokio::test]
async fn verify_batch_high_confidence_applies_correction() {
    let (graph, _tmp) = open_graph("high_conf").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await;

    let mock_llm = Arc::new(mock_correct_with_confidence(rowid, 2, 0.95));
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];
    let type_map = make_type_map();

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-high-conf",
            dry_run: false,
        },
    )
    .await
    .expect("verify_batch high confidence");

    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "high-confidence correct must apply"
    );
    assert_eq!(outcome.counts.corrected, 1);
    assert_eq!(current_type_id(&graph.conn, "apple").await, 2);
    assert_eq!(audit_row_count(&graph.conn, rowid).await, 1);
}

// ── Quinn LOW-01: ordering invariant test (confidence gate before dry_run gate) ──

/// Quinn LOW-01: confidence-gate downgrade fires BEFORE the dry_run gate.
///
/// This pins the structural ordering: a low-confidence correct decision is
/// always downgraded to Confirm, regardless of whether dry_run is true or false.
/// dry_run only affects the actual UPDATE/audit write — but the downgrade
/// happens before either is reached.
///
/// If a future refactor moves the dry_run check above the confidence gate,
/// this test would catch the regression (a low-confidence + dry_run combo
/// would no longer produce Confirm).
#[tokio::test]
async fn verify_batch_low_confidence_dry_run_still_downgrades() {
    let (graph, _tmp) = open_graph("low_conf_dry_run").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await;

    let mock_llm = Arc::new(mock_correct_with_confidence(rowid, 2, 0.5));
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];
    let type_map = make_type_map();

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-low-conf-dry-run",
            // dry_run TRUE — but confidence gate fires first so this is irrelevant.
            dry_run: true,
        },
    )
    .await
    .expect("verify_batch low-confidence + dry_run");

    // Still downgrades to Confirm — ordering preserved.
    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Confirm),
        "low-confidence correct must downgrade to Confirm even when dry_run=true"
    );
    assert_eq!(outcome.counts.confirmed, 1);
    assert_eq!(outcome.counts.corrected, 0);

    // Both gates effectively skip the UPDATE: confidence gate downgrades the
    // decision to Confirm (which doesn't UPDATE anyway), and dry_run would
    // skip Correct path too. Either way, entity type unchanged.
    assert_eq!(current_type_id(&graph.conn, "apple").await, 0);
    assert_eq!(audit_row_count(&graph.conn, rowid).await, 0);
}

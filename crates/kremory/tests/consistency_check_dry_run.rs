#![allow(clippy::unwrap_used, clippy::expect_used)]
//! T1.3 (sprint plan `v0-2-0-phase-b-prep-sprint-plan-2026-06-10`) — dry_run mode on
//! `ConsistencyCheckOpts` + `VerifyBatchParams`.
//!
//! Governing references:
//! - Sprint plan T1.3 DoD: "New unit test `consistency_check_dry_run_preserves_types` passes"
//! - Basket item #194 — gbrain `dryRun: bool` on maintenance operations (STEAL, v0.1.1)
//! - ADR-049 §dream_pass4_audit table contract
//! - CLAUDE.md `feedback_load_bearing_invariants_at_emit_not_prompt` — guard enforced
//!   structurally at the apply_correction call site, not via prompt instruction
//!
//! ## Invariant under test
//!
//! When `VerifyBatchParams.dry_run = true`:
//! - `verify_batch` SHOULD still call the LLM and return a `VerifyBatchOutcome` with
//!   `VerifyAction::Correct` for entities the LLM proposes corrections on.
//! - But `apply_correction` SHOULD NOT UPDATE the `entities` table.
//! - And `write_audit_row` SHOULD NOT INSERT into `dream_pass4_audit`.
//!
//! The decision is returned so callers can preview; the state mutation is skipped.

use std::sync::Arc;

use kremory::core::dream::consistency_check::{
    verify_batch, CandidateRow, VerifyAction, VerifyBatchParams,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;

async fn open_graph(tag: &str) -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("dry-run-{tag}.db"));
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

fn mock_correct_to(entity_id: i64, new_type_id: i64) -> MockChatProvider {
    let response = serde_json::json!({
        "decisions": [{
            "entity_id": entity_id,
            "action": "correct",
            "new_type_id": new_type_id,
            "confidence": 0.95
        }]
    })
    .to_string();
    let mut map = std::collections::HashMap::new();
    // Key on substring present in build_verify_messages system prompt — same pattern as
    // verify_batch_for_candidates.rs helper.
    map.insert("entity type".to_string(), response);
    MockChatProvider::new(map)
}

async fn count_audit_rows(conn: &libsql::Connection, entity_rowid: i64) -> i64 {
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

async fn select_entity_type_id(conn: &libsql::Connection, entity_id: &str) -> i64 {
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

/// Core invariant: dry_run=true returns the would-be correction decision BUT
/// leaves the entities row and dream_pass4_audit table untouched.
#[tokio::test]
async fn consistency_check_dry_run_preserves_types() {
    let (graph, _tmp) = open_graph("preserves_types").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await; // start as catch-all

    let mock_llm = Arc::new(mock_correct_to(rowid, 2)); // LLM wants to correct → Organization
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];

    let mut type_map = std::collections::HashMap::new();
    type_map.insert(0i64, ("Entity".to_string(), "Catch-all".to_string()));
    type_map.insert(
        1i64,
        ("Person".to_string(), "A human individual".to_string()),
    );
    type_map.insert(
        2i64,
        ("Organization".to_string(), "A company or group".to_string()),
    );

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-dry-run-1",
            dry_run: true,
        },
    )
    .await
    .expect("verify_batch dry_run");

    // Assertion 1: decision is RETURNED (caller can preview).
    assert_eq!(outcome.decisions.len(), 1, "one decision returned");
    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "decision shows Correct action (preview); expected VerifyAction::Correct"
    );
    assert_eq!(
        outcome.decisions[0].new_type_id,
        Some(2),
        "decision shows new_type_id=2 (preview)"
    );

    // Assertion 2: entities table UNCHANGED — type_id still 0.
    let current_type = select_entity_type_id(&graph.conn, "apple").await;
    assert_eq!(
        current_type, 0,
        "dry_run must NOT UPDATE entity_type_id; entity still has original type"
    );

    // Assertion 3: dream_pass4_audit has NO new row.
    let audit_count = count_audit_rows(&graph.conn, rowid).await;
    assert_eq!(
        audit_count, 0,
        "dry_run must NOT INSERT into dream_pass4_audit"
    );
}

/// Regression baseline: dry_run=false still applies the correction + writes audit row.
/// Without this companion test, the dry_run guard could degenerate to "always skip"
/// silently.
#[tokio::test]
async fn consistency_check_no_dry_run_applies_corrections() {
    let (graph, _tmp) = open_graph("no_dry_run").await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "apple", 0).await;

    let mock_llm = Arc::new(mock_correct_to(rowid, 2));
    let candidate = CandidateRow {
        rowid,
        name: "apple".to_string(),
        entity_type_id: 0,
        top3_facts: Vec::new(),
        source_episode: Some("Apple Inc. announced earnings.".to_string()),
    };
    let flagged: Vec<&CandidateRow> = vec![&candidate];

    let mut type_map = std::collections::HashMap::new();
    type_map.insert(0i64, ("Entity".to_string(), "Catch-all".to_string()));
    type_map.insert(
        1i64,
        ("Person".to_string(), "A human individual".to_string()),
    );
    type_map.insert(
        2i64,
        ("Organization".to_string(), "A company or group".to_string()),
    );

    let outcome = verify_batch(
        &graph.conn,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm: mock_llm.as_ref(),
            verify_model: "mock",
            run_id: "test-no-dry-run-1",
            dry_run: false,
        },
    )
    .await
    .expect("verify_batch live");

    // Decision returned (same as dry_run).
    assert_eq!(outcome.decisions.len(), 1);
    assert!(matches!(outcome.decisions[0].action, VerifyAction::Correct));

    // entities table UPDATED.
    let current_type = select_entity_type_id(&graph.conn, "apple").await;
    assert_eq!(
        current_type, 2,
        "non-dry_run must apply correction; entity_type_id should be 2 (Organization)"
    );

    // Audit row written.
    let audit_count = count_audit_rows(&graph.conn, rowid).await;
    assert_eq!(audit_count, 1, "non-dry_run must write one audit row");
}

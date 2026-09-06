#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Dream Pass 4 (`consistency_check`) registry-bounds guard on the LLM-emitted
//! `new_type_id`.
//!
//! ## Why this exists
//!
//! `VerifyDecision::new_type_id` (`consistency_check/mod.rs`) is UNTRUSTED LLM
//! output. It flows straight into TWO persistence sites:
//!
//! 1. dream-phase — `verify.rs` → `audit::apply_correction` →
//!    `UPDATE entities SET entity_type_id = ?1, entity_type_source = 'DreamPass4'`
//! 2. Stage 2 pre-write — `VerifyAction::Correct` → `ResolvedDecision::Correct`
//!    → `ingest/pipeline/phase1.rs` `INSERT INTO entities (… entity_type_id …)`
//!
//! `entities.entity_type_id` is `INTEGER NOT NULL DEFAULT 0` with **no foreign
//! key** to `entity_types` (Migration 008, `migrations/defs_b.rs`), so the
//! database accepts any integer. The three guards that already existed —
//! entity-rowid membership, `action=correct` requires `new_type_id`, and the
//! `MIN_VERIFY_CONFIDENCE` downgrade — all check something OTHER than whether
//! the id is a real registry row.
//!
//! This mirrors the guard applied in `dream/reclassify.rs` and the L7 precedent
//! in `core/reclassification.rs`: an out-of-registry id is DISCARDED, never
//! persisted, and never coerced to 0.
//!
//! ## Invariants under test
//!
//! For `action=correct` at high confidence where `new_type_id` is not a live
//! registry row (out-of-range, negative) or is the catch-all sentinel `0`:
//! - the decision MUST NOT be `VerifyAction::Correct`
//! - `counts.corrected` MUST NOT increment
//! - `entities.entity_type_id` MUST be unchanged
//! - no `dream_pass4_audit` row is written
//!
//! And in the other direction (over-blocking is a defect too): an id that IS in
//! the registry must still apply normally.

use std::sync::Arc;

use kremory::core::dream::consistency_check::{
    verify_batch, CandidateRow, VerifyAction, VerifyBatchParams,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;

async fn open_graph(tag: &str) -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("registry-bounds-{tag}.db"));
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

/// Seed an entity already carrying a REAL type (id=1 Person) so that "unchanged"
/// is distinguishable from "demoted to catch-all" — seeding 0 would make the two
/// outcomes indistinguishable.
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

/// High confidence (0.95) throughout — well clear of `MIN_VERIFY_CONFIDENCE`,
/// so a failure can only be attributable to the registry-bounds guard.
fn mock_correct(entity_id: i64, new_type_id: i64) -> MockChatProvider {
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

async fn current_type_source(conn: &libsql::Connection, entity_id: &str) -> Option<String> {
    let mut rows = conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = ?1",
            libsql::params![entity_id.to_string()],
        )
        .await
        .expect("select type_source");
    let row = rows
        .next()
        .await
        .expect("type_source row")
        .expect("type_source present");
    row.get::<Option<String>>(0).expect("type_source col")
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

/// Drive `verify_batch` with one flagged candidate currently typed `Person` (1)
/// and an LLM that "corrects" it to `emitted_type_id`.
async fn run_correct_to(
    tag: &str,
    emitted_type_id: i64,
) -> (
    TemporalGraph,
    tempfile::TempDir,
    i64,
    kremory::core::dream::consistency_check::VerifyBatchOutcome,
) {
    let (graph, tmp) = open_graph(tag).await;
    seed_entity_types(&graph.conn).await;
    let rowid = seed_entity(&graph.conn, "acme", 1).await;

    let mock_llm = Arc::new(mock_correct(rowid, emitted_type_id));
    let candidate = CandidateRow {
        rowid,
        name: "acme".to_string(),
        entity_type_id: 1,
        top3_facts: Vec::new(),
        source_episode: Some("Acme Corp announced earnings.".to_string()),
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
            run_id: &format!("test-registry-bounds-{tag}"),
            dry_run: false,
        },
    )
    .await
    .expect("verify_batch");

    (graph, tmp, rowid, outcome)
}

/// An `entity_type_id` above the registry's max id must never be persisted.
#[tokio::test]
async fn verify_batch_out_of_range_new_type_id_is_not_persisted() {
    let (graph, _tmp, rowid, outcome) = run_correct_to("out_of_range", 9999).await;

    assert!(
        !matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "out-of-registry new_type_id must not yield a Correct decision"
    );
    assert_eq!(
        outcome.decisions[0].new_type_id, None,
        "a rejected correction must not carry the out-of-registry id forward \
         (it would be written by the Stage 2 pre-write path)"
    );
    assert_eq!(outcome.counts.corrected, 0, "corrected count must be 0");

    assert_eq!(
        current_type_id(&graph.conn, "acme").await,
        1,
        "entity_type_id must be untouched — 9999 is not a registry row"
    );
    assert_eq!(
        current_type_source(&graph.conn, "acme").await.as_deref(),
        Some("Phase1Ner"),
        "a rejected correction must not stamp DreamPass4 (that would \
         permanently exclude the entity from load_candidates)"
    );
    assert_eq!(
        audit_row_count(&graph.conn, rowid).await,
        0,
        "no dream_pass4_audit row for a rejected correction"
    );
}

/// `new_type_id` is `i64`, so a NEGATIVE id is representable and must be rejected.
#[tokio::test]
async fn verify_batch_negative_new_type_id_is_not_persisted() {
    let (graph, _tmp, rowid, outcome) = run_correct_to("negative", -1).await;

    assert!(
        !matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "negative new_type_id must not yield a Correct decision"
    );
    assert_eq!(outcome.counts.corrected, 0);
    assert_eq!(
        current_type_id(&graph.conn, "acme").await,
        1,
        "entity_type_id must be untouched — -1 is not a registry row"
    );
    assert_eq!(audit_row_count(&graph.conn, rowid).await, 0);
}

/// `correct → 0` is out of contract: 0 is the catch-all sentinel and demotion
/// has its own action (`Demote`). Applying it through `apply_correction` would
/// additionally stamp `entity_type_source = 'DreamPass4'`, which is excluded by
/// `load_candidates` — a one-way trapdoor out of Pass 4 for that entity.
#[tokio::test]
async fn verify_batch_catch_all_new_type_id_is_not_applied_as_a_correction() {
    let (graph, _tmp, rowid, outcome) = run_correct_to("catch_all", 0).await;

    assert!(
        !matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "correct→0 must not yield a Correct decision (Demote is the demotion action)"
    );
    assert_eq!(outcome.counts.corrected, 0);
    assert_eq!(
        current_type_id(&graph.conn, "acme").await,
        1,
        "correct→0 must not silently demote via the correction path"
    );
    assert_eq!(
        current_type_source(&graph.conn, "acme").await.as_deref(),
        Some("Phase1Ner"),
        "correct→0 must not stamp DreamPass4"
    );
    assert_eq!(audit_row_count(&graph.conn, rowid).await, 0);
}

/// Sensitivity in the OTHER direction: a legitimate in-registry correction must
/// still apply. A guard that blocks real work is a defect, not caution.
#[tokio::test]
async fn verify_batch_in_registry_new_type_id_still_applies() {
    let (graph, _tmp, rowid, outcome) = run_correct_to("in_registry", 2).await;

    assert!(
        matches!(outcome.decisions[0].action, VerifyAction::Correct),
        "an in-registry correction must still apply — the bounds guard must not over-block"
    );
    assert_eq!(outcome.decisions[0].new_type_id, Some(2));
    assert_eq!(outcome.counts.corrected, 1);
    assert_eq!(current_type_id(&graph.conn, "acme").await, 2);
    assert_eq!(
        current_type_source(&graph.conn, "acme").await.as_deref(),
        Some("DreamPass4")
    );
    assert_eq!(audit_row_count(&graph.conn, rowid).await, 1);
}

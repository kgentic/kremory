use super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// ── Plant helpers (direct SQL — full control over temporal columns) ──────────
//
// `facts` carries the composite FK `(subject_id, subject_group_id) REFERENCES
// entities(id, group_id)` (`graph/facts.rs:333-341`), and FK enforcement is ON,
// so the subject entity row MUST exist first + the fact must stamp
// `subject_group_id = group_id`. Value-object facts (`object_id = NULL`) do not
// enforce the object FK.

/// Ensure a minimal `entities` row exists for `(id, group_id)` (idempotent).
/// Uses the real `insert_entity_with_group` API so the composite FK + the
/// `entities_fts` shadow are satisfied correctly (`entities.label` was dropped
/// in Migration 009 — hand-rolled INSERTs against the old shape fail).
async fn ensure_entity(graph: &TemporalGraph, group_id: &str, id: &str) {
    use crate::core::graph::InsertEntityWithGroupParams;
    // INSERT OR IGNORE semantics: a repeat plant of the same (id, group_id) is a
    // benign duplicate — swallow it so multi-fact plants on one subject work.
    let _ = graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(group_id),
        })
        .await;
}

/// Insert a `facts` row with explicit temporal columns. Returns its id.
#[allow(clippy::too_many_arguments)] // test helper — test files are exempt from the arg-count lint
async fn plant_fact(
    graph: &TemporalGraph,
    group_id: &str,
    subject: &str,
    predicate: &str,
    object_value: &str,
    valid_from: DateTime<Utc>,
    valid_to: Option<DateTime<Utc>>,
    expired_at: Option<DateTime<Utc>>,
    invalid_at: Option<DateTime<Utc>>,
    is_dream_generated: i64,
) -> i64 {
    ensure_entity(graph, group_id, subject).await;
    let now = Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
              expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1.0, ?11)",
            libsql::params![
                subject,
                predicate,
                object_value,
                valid_from.to_rfc3339(),
                valid_to.map(|v| v.to_rfc3339()),
                now,
                expired_at.map(|v| v.to_rfc3339()),
                invalid_at.map(|v| v.to_rfc3339()),
                group_id,
                group_id,
                is_dream_generated,
            ],
        )
        .await
        .expect("plant fact");
    let mut rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    rows.next()
        .await
        .expect("row")
        .expect("row present")
        .get::<i64>(0)
        .expect("id")
}

/// Read `(expired_at)` for a fact.
async fn read_expired_at(graph: &TemporalGraph, fact_id: i64) -> Option<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT expired_at FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query expired_at");
    let row = rows.next().await.expect("row").expect("present");
    row.get::<Option<String>>(0).expect("expired_at col")
}

fn budget() -> ConsolidationBudget {
    ConsolidationBudget::new(None, None)
}

// ── Split into supersession_tests/ (TD-243 NEW_FILE_CEILING) ────────────────
//
// Grouped by what each test exercises, not by original position:
//   window_closeout — the 9 DoD-P1.1/P1.2/TD-177 deterministic sweep tests
//   property        — the randomized INV1-INV8 invariant proof + its own
//                     PRNG/snapshot support code (SplitMix64, FactRow,
//                     snapshot_facts, matches_closeout — used ONLY here)
//   rollback        — mid-sweep-failure atomicity + the paired decision-
//                     not-emitted-on-rollback telemetry test
//   llm_nominate    — the P1.3 pure-fn emit-invariant unit test + the
//                     stub-off-lane integration test
// Shared fixtures (ensure_entity, plant_fact, read_expired_at, budget) stay
// here so every child reaches them via `use super::*;`.

#[path = "supersession_tests/window_closeout.rs"]
mod window_closeout;

#[path = "supersession_tests/property.rs"]
mod property;

#[path = "supersession_tests/rollback.rs"]
mod rollback;

#[path = "supersession_tests/llm_nominate.rs"]
mod llm_nominate;

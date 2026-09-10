//! TD-250 — a dream's fact archival must be REACHABLE, so `restore_archived_fact`
//! is callable.
//!
//! `Memory::restore_archived_fact(archived_fact_id)` shipped taking an id that no
//! public read returned: `DreamSummary` reports `facts_archived` as a COUNT, and
//! nothing named WHICH facts. A documented, uncallable reversal is the same defect
//! TD-244 fixed for `supersede`.
//!
//! The fix adds NO public surface. The archive op now writes a `fact_archive` row
//! to `graph_mutation_log`, which `list_mutations` — the read that already existed
//! for exactly this — returns. So this asserts the whole consumer path:
//!
//!   archive -> list_mutations(kind = FactArchive) -> undo(mutation_id) -> fact live
//!
//! DETERMINISTIC, zero-LLM (archival is a temporal SQL sweep; the inspect query is
//! pure SQL), so this sits at the fast tier with NO VCR
//! (`llm-test-pyramid-vcr-seams`). Mirrors `reversible_inspect.rs`'s shape.

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use chrono::{Duration, Utc};

use kremory::core::dream::{archive, list_mutations, undo_fact_archive};
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::schema::TemporalGraph;
use kremory::facade::{MutationFilter, MutationKind};

const GROUP: &str = "archive_reachability";
const GRACE_DAYS: u32 = 90;
const SUBJECT: &str = "caroline vasseur";

async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("archive.db");
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("open graph");
    (graph, dir)
}

async fn insert_entity(graph: &TemporalGraph, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(GROUP),
        })
        .await
        .expect("insert entity");
}

/// Plant a value fact. `expired_at = None` keeps it live (the anchor that stops the
/// ref-count guard refusing to strand the subject).
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn insert_value_fact(
    graph: &TemporalGraph,
    predicate: &str,
    object_value: &str,
    expired_at: Option<String>,
) -> i64 {
    let now = Utc::now();
    let valid_from = (now - Duration::days(400)).to_rfc3339();
    let recorded_at = (now - Duration::days(400)).to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, recorded_at, expired_at, \
              group_id, confidence, subject_group_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1.0, ?7)",
            libsql::params![
                SUBJECT,
                predicate,
                object_value,
                valid_from,
                recorded_at,
                expired_at,
                GROUP
            ],
        )
        .await
        .expect("insert fact");
    let mut rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    rows.next()
        .await
        .expect("row")
        .expect("some row")
        .get::<i64>(0)
        .expect("i64")
}

async fn fact_is_live(graph: &TemporalGraph, fact_id: i64) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT id FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query fact");
    rows.next().await.expect("row").is_some()
}

fn archive_filter(include_undone: bool) -> MutationFilter {
    MutationFilter {
        group_id: Some(GROUP.to_string()),
        kind: Some(MutationKind::FactArchive),
        since: None,
        include_undone,
    }
}

/// The whole consumer path, end to end: archive names what it archived, and that
/// name is enough to put the fact back.
#[tokio::test]
async fn archived_fact_is_nameable_and_restorable_through_the_mutation_log() {
    let (graph, _dir) = open_graph().await;
    insert_entity(&graph, SUBJECT).await;

    // Long-expired candidate + a live anchor, so the ref-count guard (P2.2) does
    // not KEEP the candidate to avoid stranding the subject.
    let expired_at = (Utc::now() - Duration::days(GRACE_DAYS as i64 + 30)).to_rfc3339();
    let candidate = insert_value_fact(&graph, "worked_at", "lysfjord", Some(expired_at)).await;
    insert_value_fact(&graph, "lives_in", "bergen", None).await;

    let report = archive(&graph, GROUP, GRACE_DAYS).await.expect("archive");
    assert_eq!(report.count, 1, "exactly the expired candidate is archived");
    assert!(
        !fact_is_live(&graph, candidate).await,
        "the archived fact must have left `facts`"
    );

    // THE POINT OF THE TEST: the archival is nameable. Before TD-250 this list was
    // empty by construction and the archived id was unobtainable, so
    // `restore_archived_fact` could not be called by anyone.
    let records = list_mutations(&graph, archive_filter(false))
        .await
        .expect("list_mutations");
    assert_eq!(
        records.len(),
        1,
        "the archive op must log exactly one fact_archive mutation, got {records:?}"
    );
    let record = &records[0];
    assert_eq!(record.kind, MutationKind::FactArchive);
    assert!(!record.undone);
    assert_eq!(record.group_id, GROUP);
    assert!(
        record.affected_entities.contains(&SUBJECT.to_string()),
        "the subject must be a locate key: {:?}",
        record.affected_entities
    );
    assert!(
        record.summary.contains(&candidate.to_string())
            && record.summary.contains("worked_at"),
        "the summary must name the fact AND its predicate — 'something about \
         caroline was retired' does not say which relation left: {}",
        record.summary
    );

    // And the name is sufficient to reverse it.
    let outcome = undo_fact_archive(&graph, record.mutation_id)
        .await
        .expect("undo_fact_archive");
    assert_eq!(outcome.restored_fact_id, candidate);
    assert!(
        !outcome.already_live,
        "the fact was archived, so this restore did real work"
    );
    assert!(
        fact_is_live(&graph, candidate).await,
        "the restored fact must be back in `facts`"
    );
}

/// The reversal is recorded, not merely performed: the log row is marked undone,
/// and a second undo is an honest no-op rather than a second restore.
#[tokio::test]
async fn undoing_an_archival_marks_the_log_row_and_is_idempotent() {
    let (graph, _dir) = open_graph().await;
    insert_entity(&graph, SUBJECT).await;
    let expired_at = (Utc::now() - Duration::days(GRACE_DAYS as i64 + 30)).to_rfc3339();
    let candidate = insert_value_fact(&graph, "worked_at", "lysfjord", Some(expired_at)).await;
    insert_value_fact(&graph, "lives_in", "bergen", None).await;
    archive(&graph, GROUP, GRACE_DAYS).await.expect("archive");

    let mutation_id = list_mutations(&graph, archive_filter(false))
        .await
        .expect("list")[0]
        .mutation_id;
    undo_fact_archive(&graph, mutation_id)
        .await
        .expect("first undo");

    // Live-only filter no longer returns it; include_undone does, marked undone.
    assert!(
        list_mutations(&graph, archive_filter(false))
            .await
            .expect("list live")
            .is_empty(),
        "an undone archival must not read as still-live"
    );
    let all = list_mutations(&graph, archive_filter(true))
        .await
        .expect("list all");
    assert_eq!(all.len(), 1);
    assert!(all[0].undone, "the log row must be marked undone_at");

    let second = undo_fact_archive(&graph, mutation_id)
        .await
        .expect("second undo");
    assert_eq!(second.restored_fact_id, candidate);
    assert!(
        second.already_live,
        "a second undo restored nothing — it must say so, not claim a restore"
    );
}

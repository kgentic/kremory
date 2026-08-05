#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Vera H3/H4 crash-window regression guards
//! (`.ai-docs/architecture-review/vera-adr-074-migrate-023-review-2026-07-14.md`).
//!
//! `migrate_023_vector_index_column_type` runs unconditionally against every
//! already-published crates.io consumer's existing, populated database. Its
//! pre-fix crash-recovery logic had two structurally unguarded windows:
//!
//! - **H3** — partial-crash detection was *existence*-based, not *content*-
//!   based: a leftover `X_new_023` scratch table was trusted as "fully
//!   copied" just because it existed, even when a crash left it empty or
//!   partial. Resuming from that state dropped the live table and renamed
//!   the empty shell into place — **silent, total data loss reported as a
//!   SUCCESS message** (no error, no warning surfaced to the operator).
//! - **H4** — the rebuild gate (declared `embedding` column type) is
//!   satisfied the instant `RENAME` completes, but ~10 more `CREATE INDEX` /
//!   `CREATE VIRTUAL TABLE` statements follow it. A crash between rename and
//!   full indexing left the table **permanently under-indexed**: the next
//!   open sees the column already `F32_BLOB` and skips the whole block,
//!   never retrying the missing indexes.
//!
//! Each test constructs the exact on-disk state a crash at that window would
//! leave behind, then re-invokes `migrate_023` via
//! `TemporalGraph::run_migrate_023_for_test` (a test-only wrapper — the real
//! function is `pub(crate)`, called once from `run_migrations`). Assertions
//! are on ROW COUNTS and index existence, not just "returns Ok" — the whole
//! point is that the pre-fix code returned `Ok` while losing data.

use kremory::core::schema::TemporalGraph;

// ─── shared helpers ────────────────────────────────────────────────────────

async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-mig023-crash.db");
    let graph = TemporalGraph::open_with_dim(path.to_str().expect("path utf-8"), 384)
        .await
        .expect("TemporalGraph::open_with_dim must succeed on fresh DB");
    (graph, tmp)
}

async fn row_count(graph: &TemporalGraph, table: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(&format!("SELECT COUNT(*) FROM {table}"), ())
        .await
        .unwrap_or_else(|_| panic!("SELECT COUNT(*) FROM {table}"));
    let row = rows.next().await.expect("iter").expect("row");
    row.get::<i64>(0).expect("count")
}

async fn table_exists(graph: &TemporalGraph, name: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
            libsql::params![name],
        )
        .await
        .expect("sqlite_master query");
    rows.next().await.expect("iter").is_some()
}

async fn index_exists(graph: &TemporalGraph, name: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?1",
            libsql::params![name],
        )
        .await
        .expect("sqlite_master query");
    rows.next().await.expect("iter").is_some()
}

/// Insert a minimal, real entity — satisfies every NOT NULL column on the
/// current (post-migrate_023) `entities` shape.
async fn insert_entity(graph: &TemporalGraph, id: &str, group_id: &str) {
    graph
        .conn
        .execute(
            "INSERT INTO entities (id, group_id, entity_type_id, properties, recorded_at) \
             VALUES (?1, ?2, 0, '{}', datetime('now'))",
            libsql::params![id, group_id],
        )
        .await
        .expect("insert entity");
}

/// Args-as-object per rust-conventions §too_many_arguments (clippy.toml
/// threshold 3).
struct NewTestFact<'a> {
    subject_id: &'a str,
    group_id: &'a str,
    predicate: &'a str,
}

/// Insert a minimal, real fact — satisfies every NOT NULL column on the
/// current (post-migrate_023) `facts` shape.
async fn insert_fact(graph: &TemporalGraph, fact: NewTestFact<'_>) {
    let NewTestFact {
        subject_id,
        group_id,
        predicate,
    } = fact;
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, subject_group_id, predicate, object_value, valid_from, \
              recorded_at, group_id) \
             VALUES (?1, ?2, ?3, 'test-object', datetime('now'), datetime('now'), ?2)",
            libsql::params![subject_id, group_id, predicate],
        )
        .await
        .expect("insert fact");
}

// ─── H3: mid-copy crash must never lose rows ──────────────────────────────

/// Reproduces the exact silent-data-loss window: `facts_bak_023` holds the
/// real pre-migration snapshot (N rows), `facts_new_023` exists but is
/// EMPTY (simulating a crash between `CREATE TABLE facts_new_023` and the
/// `INSERT ... SELECT` copy), and the live `facts` table is untouched (the
/// crash landed before the destructive `DROP TABLE facts` in that same
/// prior run).
///
/// Pre-fix: `partial = table_exists(facts_new_023) = true` was trusted
/// blindly → `DROP TABLE facts` destroys the N real rows → `RENAME
/// facts_new_023 → facts` puts the EMPTY table in their place → zero facts
/// remain, `PRAGMA foreign_key_check` finds nothing to conflict against
/// (both tables now reference nothing), migration returns `Ok` and logs
/// success. Total silent data loss.
///
/// Fixed: `facts_needs_rebuild` fires (facts_partial=true), the content
/// check (`COUNT(facts_new_023)=0` vs `COUNT(facts_bak_023)=N`) detects the
/// mismatch, repopulates `facts_new_023` from `facts_bak_023`, THEN proceeds
/// to drop+rename. Final `facts` row count must equal the original N.
#[tokio::test]
async fn h3_facts_mid_copy_crash_recovers_all_rows_not_silently_lost() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Seed real data: 2 entities + 5 facts referencing them.
    insert_entity(&graph, "alice", "default").await;
    insert_entity(&graph, "bob", "default").await;
    for i in 0..5 {
        insert_fact(
            &graph,
            NewTestFact {
                subject_id: "alice",
                group_id: "default",
                predicate: &format!("likes-{i}"),
            },
        )
        .await;
    }
    let original_fact_count = row_count(&graph, "facts").await;
    assert_eq!(original_fact_count, 5, "sanity: 5 facts seeded");

    // Manually construct the crash state: bak = full snapshot (5 rows),
    // new_023 = empty shell. Live `facts` is left untouched (matches the
    // window where the crash landed before the destructive DROP in a prior
    // run).
    graph
        .conn
        .execute("CREATE TABLE facts_bak_023 AS SELECT * FROM facts", ())
        .await
        .expect("create facts_bak_023 snapshot");
    assert_eq!(
        row_count(&graph, "facts_bak_023").await,
        5,
        "sanity: bak snapshot holds all 5 rows"
    );

    graph
        .conn
        .execute(
            "CREATE TABLE facts_new_023 (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id        TEXT NOT NULL,
                subject_group_id  TEXT NOT NULL DEFAULT 'default',
                predicate         TEXT NOT NULL,
                object_id         TEXT,
                object_group_id   TEXT,
                object_value      TEXT,
                properties        TEXT,
                embedding         F32_BLOB(384),
                valid_from        TEXT NOT NULL,
                valid_to          TEXT,
                recorded_at       TEXT NOT NULL,
                expired_at        TEXT,
                invalid_at        TEXT,
                group_id          TEXT NOT NULL DEFAULT 'default',
                confidence        REAL DEFAULT 1.0,
                source_episode_id INTEGER,
                memory_type       TEXT,
                content_hash      TEXT,
                access_count      INTEGER NOT NULL DEFAULT 0,
                is_dream_generated  INTEGER NOT NULL DEFAULT 0,
                corroboration_inert INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (subject_id, subject_group_id) REFERENCES entities(id, group_id),
                FOREIGN KEY (object_id,  object_group_id)  REFERENCES entities(id, group_id),
                FOREIGN KEY (source_episode_id)            REFERENCES episodes(id)
            )",
            (),
        )
        .await
        .expect("create facts_new_023 empty shell");
    assert_eq!(
        row_count(&graph, "facts_new_023").await,
        0,
        "sanity: facts_new_023 is the crash-window empty shell"
    );

    // Re-invoke the migration against this crash state.
    graph
        .run_migrate_023_for_test()
        .await
        .expect("migrate_023 must succeed and recover the data, not silently lose it");

    // The whole point: `facts` must hold ALL 5 original rows, not 0.
    assert_eq!(
        row_count(&graph, "facts").await,
        5,
        "H3 regression: facts_new_023 being empty-but-existing must NOT cause \
         silent data loss on resume — all 5 original rows must be recovered \
         from facts_bak_023"
    );

    // Scratch tables must be cleaned up (M5) after a successful run.
    assert!(
        !table_exists(&graph, "facts_bak_023").await,
        "facts_bak_023 must be dropped after a successful migration (M5)"
    );
    assert!(
        !table_exists(&graph, "facts_new_023").await,
        "facts_new_023 must no longer exist after rename"
    );

    // FK integrity must hold post-recovery (facts still reference `alice`).
    let mut violations = graph
        .conn
        .query("PRAGMA foreign_key_check", ())
        .await
        .expect("fk check query");
    assert!(
        violations.next().await.expect("iter").is_none(),
        "no dangling FKs after H3 recovery"
    );
}

/// The deepest H3 window: the live `entities` table itself has already been
/// DROPPED (crash landed between `DROP TABLE entities` and the `RENAME` in a
/// prior run), while `entities_new_023` is an empty/partial shell and
/// `entities_bak_023` holds the real pre-migration snapshot.
///
/// This specifically exercises the H3 gate-robustness fix: because `entities`
/// no longer exists, `PRAGMA table_info(entities)` reports no columns, so
/// the column-type check alone can never signal "needs rebuild" again — the
/// gate must also fire on `entities_new_023` merely existing, and `DROP
/// TABLE IF EXISTS entities` must not error on the already-missing table.
#[tokio::test]
async fn h3_entities_dropped_before_rename_crash_recovers_all_rows() {
    let (graph, _tmp) = open_file_backed_graph().await;

    insert_entity(&graph, "carol", "default").await;
    insert_entity(&graph, "dave", "default").await;
    insert_entity(&graph, "erin", "default").await;
    let original_entity_count = row_count(&graph, "entities").await;
    assert_eq!(original_entity_count, 3, "sanity: 3 entities seeded");

    // Snapshot, then construct the empty new_023 shell, then simulate the
    // crash landing AFTER `DROP TABLE entities` succeeded but BEFORE the
    // rename — i.e. `entities` no longer exists at all.
    graph
        .conn
        .execute(
            "CREATE TABLE entities_bak_023 AS SELECT * FROM entities",
            (),
        )
        .await
        .expect("create entities_bak_023 snapshot");
    assert_eq!(row_count(&graph, "entities_bak_023").await, 3);

    graph
        .conn
        .execute(
            "CREATE TABLE entities_new_023 (
                id                      TEXT NOT NULL,
                properties              TEXT,
                embedding               F32_BLOB(384),
                recorded_at             TEXT NOT NULL,
                updated_at              TEXT,
                group_id                TEXT NOT NULL DEFAULT 'default',
                access_count            INTEGER NOT NULL DEFAULT 0,
                entity_type_id          INTEGER NOT NULL DEFAULT 0,
                entity_type_source      TEXT CHECK (entity_type_source IN (
                    'Phase1Ner', 'Phase2Llm', 'DreamPass0',
                    'DreamPass1', 'ConsumerPinned', 'DreamPass4'
                )),
                entity_type_assigned_at TEXT,
                ner_confidence          REAL,
                is_dream_generated      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (id, group_id)
            )",
            (),
        )
        .await
        .expect("create entities_new_023 empty shell");

    // Drop facts first (FK dependency on entities) then drop the live
    // entities table entirely — reproducing "crash after DROP, before
    // RENAME".
    graph
        .conn
        .execute("DELETE FROM facts", ())
        .await
        .expect("clear facts (FK dependency)");

    // Mirror the real migration's own ordering (defs_j.rs): drop the vector
    // index + btree indexes BEFORE dropping the table itself — libsql's
    // DiskANN vector index carries shadow state keyed to the base table and
    // must be torn down first.
    let _ = graph
        .conn
        .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
        .await;
    let _ = graph
        .conn
        .execute("DROP INDEX IF EXISTS idx_entities_group", ())
        .await;
    let _ = graph
        .conn
        .execute("DROP INDEX IF EXISTS idx_entities_type_id", ())
        .await;

    graph
        .conn
        .execute("DROP TABLE entities", ())
        .await
        .expect("drop live entities table (simulates crash-after-drop window)");
    assert!(
        !table_exists(&graph, "entities").await,
        "sanity: entities table genuinely absent"
    );

    graph
        .run_migrate_023_for_test()
        .await
        .expect("migrate_023 must resume cleanly even when `entities` was already dropped");

    assert!(
        table_exists(&graph, "entities").await,
        "entities table must exist again after resume"
    );
    assert_eq!(
        row_count(&graph, "entities").await,
        3,
        "H3 regression: all 3 original entities must be recovered from \
         entities_bak_023 even though the live table was already dropped \
         when the migration resumed"
    );
    assert!(
        !table_exists(&graph, "entities_bak_023").await,
        "entities_bak_023 must be dropped after a successful migration (M5)"
    );
}

// ─── H4: missing indexes must be recreated regardless of the rebuild gate ─

/// Simulates a crash between `RENAME` and the tail `CREATE INDEX` block: the
/// `embedding` columns are already `F32_BLOB` (fully migrated, no leftover
/// scratch tables) but both vector indexes are absent. The column-type gate
/// alone can never re-detect this — index presence must be checked
/// independently.
#[tokio::test]
async fn h4_missing_vector_indexes_are_recreated_without_full_rebuild() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Seed data + real embeddings so a genuine vector index creation over
    // populated F32_BLOB columns is exercised, not just an empty table.
    insert_entity(&graph, "frank", "default").await;
    insert_fact(
        &graph,
        NewTestFact {
            subject_id: "frank",
            group_id: "default",
            predicate: "owns",
        },
    )
    .await;

    let mut embedding = vec![0.0_f32; 384];
    embedding[0] = 1.0;
    graph
        .set_entity_embedding("frank", &embedding)
        .await
        .expect("set entity embedding");

    // Pre-condition: after a normal open, both columns are already
    // F32_BLOB and both vector indexes exist (migrate_023 ran for real
    // during `open_with_dim`).
    assert!(index_exists(&graph, "entities_vec_idx").await);
    assert!(index_exists(&graph, "facts_vec_idx").await);
    assert!(!table_exists(&graph, "entities_new_023").await);
    assert!(!table_exists(&graph, "facts_new_023").await);

    // Simulate the crash: both vector indexes vanish (as they would if the
    // process died mid-index-creation on a prior rebuild run), but nothing
    // else about the tables changes.
    graph
        .conn
        .execute("DROP INDEX entities_vec_idx", ())
        .await
        .expect("drop entities_vec_idx");
    graph
        .conn
        .execute("DROP INDEX facts_vec_idx", ())
        .await
        .expect("drop facts_vec_idx");
    assert!(!index_exists(&graph, "entities_vec_idx").await);
    assert!(!index_exists(&graph, "facts_vec_idx").await);

    graph
        .run_migrate_023_for_test()
        .await
        .expect("migrate_023 must recreate the missing indexes without a full table rebuild");

    assert!(
        index_exists(&graph, "entities_vec_idx").await,
        "H4 regression: entities_vec_idx must be recreated even though \
         entities.embedding was already F32_BLOB (no rebuild-gate signal)"
    );
    assert!(
        index_exists(&graph, "facts_vec_idx").await,
        "H4 regression: facts_vec_idx must be recreated even though \
         facts.embedding was already F32_BLOB (no rebuild-gate signal)"
    );

    // Data must be untouched — this was an index-only repair, not a table
    // rebuild.
    assert_eq!(row_count(&graph, "entities").await, 1);
    assert_eq!(row_count(&graph, "facts").await, 1);

    // Functional check: the recreated index actually serves a query.
    let vec_str = format!(
        "[{}]",
        embedding
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut rows = graph
        .conn
        .query(
            "SELECT e.id FROM vector_top_k('entities_vec_idx', vector(?1), 1) AS v \
             JOIN entities AS e ON e.rowid = v.id",
            libsql::params![vec_str],
        )
        .await
        .expect("vector_top_k query must succeed against the recreated index");
    let row = rows.next().await.expect("iter").expect("one row");
    let id: String = row.get(0).expect("id");
    assert_eq!(id, "frank", "recreated index must be functionally correct");
}

// ─── Idempotency ───────────────────────────────────────────────────────────

/// Running `migrate_023` twice on an already-fully-migrated, already-indexed
/// DB must be a clean no-op both times: `Ok(())`, no error, indexes still
/// present.
#[tokio::test]
async fn migrate_023_is_idempotent_on_fully_migrated_db() {
    let (graph, _tmp) = open_file_backed_graph().await;

    insert_entity(&graph, "gina", "default").await;
    insert_fact(
        &graph,
        NewTestFact {
            subject_id: "gina",
            group_id: "default",
            predicate: "knows",
        },
    )
    .await;

    assert!(index_exists(&graph, "entities_vec_idx").await);
    assert!(index_exists(&graph, "facts_vec_idx").await);
    let entities_before = row_count(&graph, "entities").await;
    let facts_before = row_count(&graph, "facts").await;

    // Second invocation — must be a no-op.
    graph
        .run_migrate_023_for_test()
        .await
        .expect("second migrate_023 invocation on a fully-migrated DB must not error");

    // Third invocation for good measure.
    graph
        .run_migrate_023_for_test()
        .await
        .expect("third migrate_023 invocation must also be a clean no-op");

    assert!(
        index_exists(&graph, "entities_vec_idx").await,
        "entities_vec_idx must still exist after repeated no-op invocations"
    );
    assert!(
        index_exists(&graph, "facts_vec_idx").await,
        "facts_vec_idx must still exist after repeated no-op invocations"
    );
    assert_eq!(
        row_count(&graph, "entities").await,
        entities_before,
        "entities row count must be unchanged by repeated no-op invocations"
    );
    assert_eq!(
        row_count(&graph, "facts").await,
        facts_before,
        "facts row count must be unchanged by repeated no-op invocations"
    );
    assert!(
        !table_exists(&graph, "entities_bak_023").await,
        "no scratch tables should be left behind by no-op invocations"
    );
    assert!(!table_exists(&graph, "facts_bak_023").await);
    assert!(!table_exists(&graph, "entities_new_023").await);
    assert!(!table_exists(&graph, "facts_new_023").await);
}

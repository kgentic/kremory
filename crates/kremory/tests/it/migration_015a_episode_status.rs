#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 015a — `episode_processing_status` column tests.
//!
//! Governing spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md`
//! ADR-051: async extraction gate state machine — Pending → Extracting → Verified | Failed.
//!
//! Test coverage (5 scenarios per Phase 1 DoD):
//!
//! 1. `migration_015a_adds_column_idempotently` — double-apply without error.
//! 2. `migration_015a_backfills_existing_episodes_with_entities_to_verified` — R-02 mitigation.
//! 3. `migration_015a_new_episodes_default_pending` — DEFAULT 'Pending' applies to fresh rows.
//! 4. `migration_015a_index_created` — `idx_episodes_processing_status` exists post-migration.
//! 5. `migration_014c_round_trip` — up → down → up preserves schema + data.

use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Open a file-backed `TemporalGraph` in an isolated temp directory.
/// Returns (graph, _tmp_guard) — drop the guard to clean up the directory.
async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("kremory-mig-015a.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed on fresh file-backed DB");
    (graph, tmp)
}

/// Collect all column names from `PRAGMA table_info('episodes')`.
async fn episodes_column_names(graph: &TemporalGraph) -> Vec<String> {
    let mut rows = graph
        .conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .expect("PRAGMA table_info('episodes') must succeed");

    let mut names = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let name: String = row.get(1).expect("column name at index 1");
        names.push(name);
    }
    names
}

/// Collect all index names on `episodes` from `PRAGMA index_list('episodes')`.
async fn episodes_index_names(graph: &TemporalGraph) -> Vec<String> {
    let mut rows = graph
        .conn
        .query("PRAGMA index_list('episodes')", ())
        .await
        .expect("PRAGMA index_list('episodes') must succeed");

    let mut names = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        // index_list columns: seq(0), name(1), unique(2), origin(3), partial(4)
        let name: String = row.get(1).expect("index name at column 1");
        names.push(name);
    }
    names
}

/// Insert a minimal episode row and return its id.
/// Uses raw SQL so the test does not depend on ingest business logic.
async fn insert_episode(graph: &TemporalGraph, content: &str) -> i64 {
    graph
        .conn
        .execute(
            "INSERT INTO episodes (content, timestamp) VALUES (?1, datetime('now'))",
            libsql::params![content],
        )
        .await
        .expect("episode insert must succeed");

    let mut rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("last_insert_rowid query must succeed");
    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("last_insert_rowid must return a row");
    row.get::<i64>(0).expect("rowid at column 0")
}

/// Insert a minimal entity row and return its id string.
async fn insert_entity(graph: &TemporalGraph, entity_id: &str) -> String {
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities (id, properties, embedding, recorded_at, group_id, access_count)
             VALUES (?1, NULL, NULL, datetime('now'), 'default', 0)",
            libsql::params![entity_id],
        )
        .await
        .expect("entity insert must succeed");
    entity_id.to_string()
}

/// Insert an episodic_edges row linking an episode to an entity.
async fn link_episode_to_entity(graph: &TemporalGraph, episode_id: i64, entity_id: &str) {
    graph
        .conn
        .execute(
            "INSERT INTO episodic_edges (episode_id, entity_id, entity_group_id, role, recorded_at)
             VALUES (?1, ?2, 'default', 'mentioned', datetime('now'))",
            libsql::params![episode_id, entity_id],
        )
        .await
        .expect("episodic_edges insert must succeed");
}

// ─── Test 1: column added idempotently ───────────────────────────────────────

/// Migration 015a must be idempotent: running `run_migrations` twice on the same
/// DB must not error.  The PRAGMA-guard in the migration prevents
/// `ALTER TABLE ADD COLUMN` from firing a second time (which would be a SQLite
/// duplicate-column error).  Column count must also be unchanged.
#[tokio::test]
async fn migration_015a_adds_column_idempotently() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // After first open, column must be present.
    let cols_first = episodes_column_names(&graph).await;
    assert!(
        cols_first.contains(&"episode_processing_status".to_string()),
        "episode_processing_status must be present after first migration run; \
         columns: {cols_first:?}"
    );
    let count_first = cols_first.len();

    // Second run — must not error (idempotency gate must absorb the repeat).
    graph.run_migrations_again_for_test().await.expect(
        "second run_migrations on same DB must be idempotent — \
             duplicate-column error means the PRAGMA guard is missing in migrate_015a",
    );

    // Column count must be unchanged after second run.
    let cols_second = episodes_column_names(&graph).await;
    let count_second = cols_second.len();
    assert_eq!(
        count_first, count_second,
        "episode column count must be unchanged after second migration run: \
         first={count_first}, second={count_second}"
    );

    // Column must still be present.
    assert!(
        cols_second.contains(&"episode_processing_status".to_string()),
        "episode_processing_status must still be present after second run; \
         columns: {cols_second:?}"
    );
}

// ─── Test 2: backfill sets Verified for episodes with extracted entities ──────

/// ADR-051 R-02 mitigation: episodes that already have entries in `episodic_edges`
/// (i.e. at least one entity was extracted from them before the migration ran)
/// must be backfilled to `Verified`.  Episodes without any episodic_edges row
/// must remain `Pending`.
///
/// Approach: open once (all migrations including 015a run; the empty DB has no
/// pre-existing episodes for the backfill to act on), then manually insert an
/// episode + entity + episodic_edge to simulate a pre-extracted episode, then
/// re-open (migrations re-run idempotently) and verify the backfill UPDATE has
/// correctly set the seeded episode's status to `Verified` while a freshly-seeded
/// episode without any episodic_edges row remains `Pending`.
#[tokio::test]
async fn migration_015a_backfills_existing_episodes_with_entities_to_verified() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-mig-015a-backfill.db");
    let path_str = path.to_str().expect("path str");

    // ── Step 1: open and run all migrations EXCEPT 015a manually ─────────────
    //
    // We open the graph normally (all migrations run including 015a), then
    // inspect the backfill result for rows that were seeded via migration 015a's
    // backfill step.
    //
    // Practical approach: open twice. First open seeds schema. We then manually
    // insert an episode + entity + edge to simulate pre-existing extraction.
    // Then run migrations again — the backfill no-ops on the new episode rows
    // and correctly shows the prior backfill already set the status.
    //
    // The canonical test is: open → check status of rows that existed at
    // migration time by seeding a DB without the column, then opening it.
    // Since TemporalGraph::open always runs all migrations, we use a different
    // approach: use the migrate_015a fn directly on a pre-seeded DB.

    // Build a DB that has the base schema (migrations up to 014 but NOT 015a).
    // We do this by opening normally (all migrations run), then downgrading
    // episode_processing_status via migrate_014c, then seeding rows, then
    // re-running migrate_015a to test the backfill.

    let graph = TemporalGraph::open(path_str).await.expect("first open");

    // Seed rows BEFORE the downgrade, while `episodes` is the live canonical table.
    // SQLite FK references in `episodic_edges` DDL are stored by name; after
    // migrate_014c renames `episodes → episodes_bak_014c` and recreates `episodes`,
    // inserting into `episodic_edges` after the rename fails with "no such table:
    // episodes_bak_014c" (the FK lookup still sees the old name in the stored DDL
    // until the connection is re-opened). Seeding before the downgrade avoids this.

    // episode_a has an episodic_edge (entity extracted → should become Verified after re-up).
    let ep_a_id = insert_episode(
        &graph,
        "Quarterly earnings call transcript for Acme Corp Q4 2024",
    )
    .await;
    let entity_id = insert_entity(&graph, "entity-acme-corp-001").await;
    link_episode_to_entity(&graph, ep_a_id, &entity_id).await;

    // episode_b has no episodic_edge (not yet extracted → should remain Pending after re-up).
    let ep_b_id =
        insert_episode(&graph, "Raw meeting notes from the product roadmap session").await;

    // Downgrade to remove the column — simulates rolling back 015a on a DB that
    // already has episodes + edges (the real production scenario for emergency rollback).
    kremory::core::migrations::migrate_014c_downgrade_episode_processing_status(&graph.conn)
        .await
        .expect("014c downgrade must succeed");

    // Confirm column is gone after downgrade.
    let cols_after_down = episodes_column_names(&graph).await;
    assert!(
        !cols_after_down.contains(&"episode_processing_status".to_string()),
        "column must be absent after 014c downgrade; columns: {cols_after_down:?}"
    );

    // Re-run migrate_015a — this is the function under test.
    kremory::core::migrations::migrate_015a_episode_processing_status(&graph.conn)
        .await
        .expect("migrate_015a must succeed on pre-seeded DB");

    // Assert: episode_a must be Verified (backfill matched episodic_edges).
    let mut rows_a = graph
        .conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![ep_a_id],
        )
        .await
        .expect("query episode_a status");
    let row_a = rows_a
        .next()
        .await
        .expect("iteration must not error")
        .expect("episode_a row must exist");
    let status_a: String = row_a.get(0).expect("status at column 0");
    assert_eq!(
        status_a, "Verified",
        "episode with extracted entities must be backfilled to Verified; got: {status_a:?}"
    );

    // Assert: episode_b must remain Pending (no episodic_edges).
    let mut rows_b = graph
        .conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![ep_b_id],
        )
        .await
        .expect("query episode_b status");
    let row_b = rows_b
        .next()
        .await
        .expect("iteration must not error")
        .expect("episode_b row must exist");
    let status_b: String = row_b.get(0).expect("status at column 0");
    assert_eq!(
        status_b, "Pending",
        "episode without extracted entities must remain Pending; got: {status_b:?}"
    );
}

// ─── Test 3: new episodes default to Pending ─────────────────────────────────

/// After migration 015a runs, a freshly inserted episode row must have
/// `episode_processing_status = 'Pending'` via the column DEFAULT.
/// This validates the DEFAULT clause on the ALTER TABLE ADD COLUMN is wired correctly.
#[tokio::test]
async fn migration_015a_new_episodes_default_pending() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Confirm column is present post-migration.
    let cols = episodes_column_names(&graph).await;
    assert!(
        cols.contains(&"episode_processing_status".to_string()),
        "episode_processing_status must be present; columns: {cols:?}"
    );

    // Insert a fresh episode without specifying episode_processing_status.
    let ep_id = insert_episode(
        &graph,
        "Interview transcript: candidate Alice for senior software engineer role",
    )
    .await;

    // Read back the status — must be 'Pending' from DEFAULT.
    let mut rows = graph
        .conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![ep_id],
        )
        .await
        .expect("query episode status");
    let row = rows
        .next()
        .await
        .expect("iteration must not error")
        .expect("row must exist");
    let status: String = row.get(0).expect("status at column 0");
    assert_eq!(
        status, "Pending",
        "freshly inserted episode must have DEFAULT status 'Pending'; got: {status:?}"
    );
}

// ─── Test 4: index created ────────────────────────────────────────────────────

/// `PRAGMA index_list('episodes')` must include `idx_episodes_processing_status`
/// after migration 015a runs.
#[tokio::test]
async fn migration_015a_index_created() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Confirm column present as a pre-condition sanity check.
    let cols = episodes_column_names(&graph).await;
    assert!(
        cols.contains(&"episode_processing_status".to_string()),
        "pre-condition: episode_processing_status must exist; columns: {cols:?}"
    );

    // Check the index list.
    let index_names = episodes_index_names(&graph).await;
    assert!(
        index_names
            .iter()
            .any(|n| n == "idx_episodes_processing_status"),
        "idx_episodes_processing_status must exist in PRAGMA index_list('episodes') \
         after migration 015a; found indexes: {index_names:?}"
    );
}

// ─── Test 5: round-trip (up → down → up) ─────────────────────────────────────

/// Full round-trip: migrate up (015a) → migrate down (014c) → migrate up (015a again).
/// Post-downgrade: column absent, downgrade indexes restored.
/// Post-second-upgrade: column present again, all original rows preserved,
/// `idx_episodes_processing_status` recreated.
///
/// This validates that 014c is a clean inverse of 015a and that the combined
/// path is safe for emergency rollback + re-apply.
#[tokio::test]
async fn migration_014c_round_trip() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-mig-014c-roundtrip.db");
    let path_str = path.to_str().expect("path str");

    // ── Open: all migrations run including 015a ───────────────────────────────
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("initial open must succeed");

    // Seed some episode rows so we can verify data survival through the round-trip.
    let ep1_id = insert_episode(
        &graph,
        "Board meeting minutes from Acme Corp January 2025 strategy session",
    )
    .await;
    let ep2_id = insert_episode(
        &graph,
        "Technical specification review for the new billing microservice",
    )
    .await;

    // Seed entity + edge for ep1 (so it gets backfilled to Verified on re-up).
    let entity_id = insert_entity(&graph, "entity-acme-board-001").await;
    link_episode_to_entity(&graph, ep1_id, &entity_id).await;

    // Assert baseline: column present.
    let cols_up = episodes_column_names(&graph).await;
    assert!(
        cols_up.contains(&"episode_processing_status".to_string()),
        "baseline: column must be present; cols: {cols_up:?}"
    );

    // ── Downgrade (014c) ──────────────────────────────────────────────────────
    kremory::core::migrations::migrate_014c_downgrade_episode_processing_status(&graph.conn)
        .await
        .expect("014c downgrade must succeed");

    // Assert post-downgrade: column absent.
    let cols_down = episodes_column_names(&graph).await;
    assert!(
        !cols_down.contains(&"episode_processing_status".to_string()),
        "after downgrade: column must be absent; cols: {cols_down:?}"
    );

    // Assert post-downgrade: original episode rows are still present.
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM episodes WHERE id IN (?1, ?2)",
            libsql::params![ep1_id, ep2_id],
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("iter").expect("row");
    let count: i64 = row.get(0).expect("count at col 0");
    assert_eq!(
        count, 2,
        "both seeded episodes must survive 014c downgrade; count={count}"
    );

    // Assert post-downgrade: pre-015a indexes restored (idx_episodes_saga must be present).
    let idx_down = episodes_index_names(&graph).await;
    assert!(
        idx_down.iter().any(|n| n == "idx_episodes_saga"),
        "idx_episodes_saga must be recreated after 014c downgrade; indexes: {idx_down:?}"
    );
    // And the 015a index must NOT be present.
    assert!(
        !idx_down
            .iter()
            .any(|n| n == "idx_episodes_processing_status"),
        "idx_episodes_processing_status must be absent after 014c downgrade; indexes: {idx_down:?}"
    );

    // ── Re-upgrade (015a) ─────────────────────────────────────────────────────
    kremory::core::migrations::migrate_015a_episode_processing_status(&graph.conn)
        .await
        .expect("015a re-upgrade must succeed after 014c downgrade");

    // Assert post-re-upgrade: column present.
    let cols_reup = episodes_column_names(&graph).await;
    assert!(
        cols_reup.contains(&"episode_processing_status".to_string()),
        "after re-upgrade: column must be present; cols: {cols_reup:?}"
    );

    // Assert post-re-upgrade: index recreated.
    let idx_reup = episodes_index_names(&graph).await;
    assert!(
        idx_reup
            .iter()
            .any(|n| n == "idx_episodes_processing_status"),
        "idx_episodes_processing_status must be recreated after re-upgrade; \
         indexes: {idx_reup:?}"
    );

    // Assert post-re-upgrade: row data preserved.
    let mut rows2 = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM episodes WHERE id IN (?1, ?2)",
            libsql::params![ep1_id, ep2_id],
        )
        .await
        .expect("count query post re-upgrade");
    let row2 = rows2.next().await.expect("iter").expect("row");
    let count2: i64 = row2.get(0).expect("count at col 0");
    assert_eq!(
        count2, 2,
        "both seeded episodes must survive the full round-trip; count={count2}"
    );

    // Assert post-re-upgrade: ep1 (with episodic_edges) is Verified again.
    let mut status_rows = graph
        .conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![ep1_id],
        )
        .await
        .expect("ep1 status query");
    let status_row = status_rows
        .next()
        .await
        .expect("iter")
        .expect("ep1 must exist");
    let ep1_status: String = status_row.get(0).expect("status col 0");
    assert_eq!(
        ep1_status, "Verified",
        "ep1 (with episodic_edges) must be Verified after re-upgrade backfill; got: {ep1_status:?}"
    );

    // Assert post-re-upgrade: ep2 (no episodic_edges) is Pending.
    let mut status_rows2 = graph
        .conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![ep2_id],
        )
        .await
        .expect("ep2 status query");
    let status_row2 = status_rows2
        .next()
        .await
        .expect("iter")
        .expect("ep2 must exist");
    let ep2_status: String = status_row2.get(0).expect("status col 0");
    assert_eq!(
        ep2_status, "Pending",
        "ep2 (no episodic_edges) must be Pending after re-upgrade; got: {ep2_status:?}"
    );
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 008 schema tests — entity_types registry + entity_type_id column.
//!
//! ## Acceptance criteria
//!
//! AC.a — Clean run: `entity_types` table exists; on a DB that already has entities,
//!        every group_id has id=0 "Entity" row and observed labels are seeded at id≥1.
//! AC.b — Re-run is a no-op (idempotency): calling `run_migrations` twice on the
//!        same DB must not error and must not duplicate or mutate rows.
//! AC.c — Backfill: pre-existing entities whose `label` matches a seeded type have
//!        `entity_type_id` set to the correct non-zero id; entities whose label is
//!        'Entity' stay at id=0.
//! AC.d — `entity_type_id` column exists on `entities` after migration; `label`
//!        column is RETAINED in Phase 1 (DROP label deferred to Phase 2).
//! AC.e — idx_entities_type_id composite index exists on `entities(group_id, entity_type_id)`.
//! AC.f — PRAGMA foreign_key_check returns empty after migration.
//! AC.g — Static source-code gate: migrations.rs contains the `entity_types` DDL,
//!        PRAGMA gate, idx_entities_type_id string, and INSERT OR IGNORE pattern.

use kremory::core::schema::TemporalGraph;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Open a file-backed `TemporalGraph` in an isolated temp directory.
/// Returns the graph + TempDir (caller must hold TempDir alive).
async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("kremory-mig-008.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed on fresh file-backed DB");
    (graph, tmp)
}

/// Open a file-backed DB, insert raw entity rows with specific labels,
/// then close and re-open (triggering migration 008 on pre-populated data).
///
/// This simulates a pre-v0.1.7 database that already has entities with labels.
/// Migration 008 runs at open() time and must backfill entity_type_id from label.
async fn open_graph_with_pre_seeded_entities(
    entities: &[(&str, &str, &str)], // (id, label, group_id)
) -> (TemporalGraph, tempfile::TempDir) {
    use libsql::Builder;

    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("kremory-mig-008-prefill.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");

    // Phase A: create the raw DB schema WITHOUT migration 008 by using
    // a raw libsql builder and the pre-008 DDL shape.
    // This simulates a v0.1.6 database just before the v0.1.7 migration runs.
    {
        let db = Builder::new_local(path_str)
            .build()
            .await
            .expect("raw db build");
        let conn = db.connect().expect("connect");

        // Minimal schema matching the post-007 shape (what migration 008 operates on).
        // NOTE: no entity_type_id column — that is what migration 008 adds.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS app_meta (
                workspace_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL DEFAULT 7,
                rql_schema_version INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO app_meta (workspace_id) VALUES ('ws-test');
             CREATE TABLE IF NOT EXISTS namespaces (
                 group_id TEXT PRIMARY KEY,
                 policy_json TEXT NOT NULL DEFAULT '{}',
                 recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                 schema_version INTEGER NOT NULL DEFAULT 1
             );
             CREATE TABLE IF NOT EXISTS entities (
                 id TEXT NOT NULL,
                 label TEXT NOT NULL,
                 properties TEXT,
                 embedding BLOB,
                 recorded_at TEXT NOT NULL,
                 updated_at TEXT,
                 group_id TEXT NOT NULL DEFAULT 'default',
                 access_count INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (id, group_id)
             );
             CREATE TABLE IF NOT EXISTS episodes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 content TEXT NOT NULL,
                 timestamp TEXT NOT NULL,
                 recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                 source_type TEXT,
                 metadata TEXT,
                 group_id TEXT,
                 saga_id TEXT,
                 sequence_number INTEGER,
                 source_id TEXT,
                 source_uri TEXT
             );
             CREATE TABLE IF NOT EXISTS facts (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 subject_id TEXT NOT NULL,
                 subject_group_id TEXT NOT NULL DEFAULT 'default',
                 predicate TEXT NOT NULL,
                 object_id TEXT,
                 object_group_id TEXT,
                 object_value TEXT,
                 properties TEXT,
                 embedding BLOB,
                 valid_from TEXT NOT NULL,
                 valid_to TEXT,
                 recorded_at TEXT NOT NULL,
                 expired_at TEXT,
                 invalid_at TEXT,
                 group_id TEXT NOT NULL DEFAULT 'default',
                 confidence REAL DEFAULT 1.0,
                 source_episode_id INTEGER,
                 memory_type TEXT,
                 content_hash TEXT,
                 access_count INTEGER NOT NULL DEFAULT 0,
                 FOREIGN KEY (subject_id, subject_group_id) REFERENCES entities(id, group_id),
                 FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
             );
             CREATE TABLE IF NOT EXISTS episodic_edges (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 episode_id INTEGER NOT NULL,
                 entity_id TEXT NOT NULL,
                 entity_group_id TEXT NOT NULL DEFAULT 'default',
                 role TEXT NOT NULL DEFAULT 'mentioned',
                 recorded_at TEXT NOT NULL,
                 FOREIGN KEY (episode_id) REFERENCES episodes(id),
                 FOREIGN KEY (entity_id, entity_group_id) REFERENCES entities(id, group_id)
             );",
        )
        .await
        .expect("pre-migration schema setup must succeed");

        // Insert pre-migration entities (no entity_type_id column yet).
        for (id, label, group_id) in entities {
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, label, recorded_at, group_id) \
                 VALUES (?1, ?2, datetime('now'), ?3)",
                libsql::params![*id, *label, *group_id],
            )
            .await
            .expect("entity insert must succeed");
        }
        // Connection drops here, closing the database.
    }

    // Phase B: open via TemporalGraph which runs all migrations including 008.
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open on pre-seeded DB must succeed");

    (graph, tmp)
}

/// Collect column names from `PRAGMA table_info('<table>')`.
async fn table_columns(graph: &TemporalGraph, table: &str) -> Vec<String> {
    let sql = format!("PRAGMA table_info('{table}')");
    let mut rows = graph
        .conn
        .query(&sql, ())
        .await
        .unwrap_or_else(|_| panic!("PRAGMA table_info('{table}') must succeed"));

    let mut cols = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .expect("PRAGMA table_info row iteration must not error")
    {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ─── AC.a: entity_types table exists ─────────────────────────────────────────

/// Migration 008 must create the `entity_types` table after `TemporalGraph::open`.
#[tokio::test]
async fn migration_008_entity_types_table_exists() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let cols = table_columns(&graph, "entity_types").await;
    assert!(
        !cols.is_empty(),
        "entity_types table must exist after migration 008; PRAGMA table_info returned nothing"
    );

    for required in &["id", "group_id", "name", "description", "created_at", "use_count"] {
        assert!(
            cols.iter().any(|c| c == *required),
            "entity_types must have column `{required}`; found: {cols:?}"
        );
    }
}

// ─── AC.a: id=0 "Entity" seeded per group_id on pre-populated DB ──────────────

/// Opening a DB that already has entities must trigger migration 008 backfill:
/// every group_id must receive id=0 "Entity" catch-all with anti-junk description.
#[tokio::test]
async fn migration_008_seeds_entity_catch_all_per_group() {
    let entities = [
        ("ent1", "Person", "ns-alpha"),
        ("ent2", "Organisation", "ns-alpha"),
        ("ent3", "Location", "ns-beta"),
    ];
    let (graph, _tmp) = open_graph_with_pre_seeded_entities(&entities).await;

    for ns in &["ns-alpha", "ns-beta"] {
        let mut rows = graph
            .conn
            .query(
                "SELECT name, description FROM entity_types WHERE group_id = ?1 AND id = 0",
                libsql::params![*ns],
            )
            .await
            .expect("entity_types query must succeed");

        let row = rows
            .next()
            .await
            .expect("row iteration must not error")
            .unwrap_or_else(|| {
                panic!("id=0 Entity row must exist for group_id='{ns}' after migration 008")
            });

        let name: String = row.get(0).expect("name at col 0");
        let description: String = row.get(1).expect("description at col 1");

        assert_eq!(name, "Entity", "id=0 name must be 'Entity' for {ns}");
        assert!(
            description.contains("Generic catch-all"),
            "id=0 description must contain 'Generic catch-all' for {ns}; got: {description:?}"
        );
        assert!(
            description.contains("DO NOT use for placeholders"),
            "id=0 description must contain 'DO NOT use for placeholders' for {ns}; got: {description:?}"
        );
    }
}

// ─── AC.a: observed labels seeded at id≥1 ────────────────────────────────────

/// Distinct labels other than 'Entity' are seeded as entity_types rows at id≥1,
/// alphabetically ordered within each group_id.
#[tokio::test]
async fn migration_008_seeds_observed_labels_alphabetically() {
    let entities = [
        ("e1", "Location", "test-ns"),
        ("e2", "Person", "test-ns"),
        ("e3", "Person", "test-ns"),   // duplicate — must NOT produce duplicate row
        ("e4", "Entity", "test-ns"),   // must NOT be seeded at id>=1
    ];
    let (graph, _tmp) = open_graph_with_pre_seeded_entities(&entities).await;

    let mut rows = graph
        .conn
        .query(
            "SELECT id, name FROM entity_types WHERE group_id = 'test-ns' ORDER BY id",
            (),
        )
        .await
        .expect("entity_types query must succeed");

    let mut seeded: Vec<(i64, String)> = Vec::new();
    while let Some(row) = rows.next().await.expect("iteration must not error") {
        let id: i64 = row.get(0).expect("id at col 0");
        let name: String = row.get(1).expect("name at col 1");
        seeded.push((id, name));
    }

    // Expected: id=0 "Entity", id=1 "Location", id=2 "Person" (alphabetical).
    assert_eq!(
        seeded.len(),
        3,
        "test-ns must have 3 entity_types rows (Entity + 2 observed labels); got: {seeded:?}"
    );
    assert_eq!(seeded[0], (0, "Entity".to_string()), "id=0 must be 'Entity'");
    assert_eq!(
        seeded[1],
        (1, "Location".to_string()),
        "id=1 must be 'Location' (alphabetically first non-Entity label)"
    );
    assert_eq!(
        seeded[2],
        (2, "Person".to_string()),
        "id=2 must be 'Person' (alphabetically second)"
    );
}

// ─── AC.d: entity_type_id column present; label dropped by Phase 2 ───────────

/// Migration 008 must add `entity_type_id` to `entities`.
/// Phase 2 (Migration 009) drops the `label` column atomically with the Rust caller
/// updates — after both migrations run the label column is ABSENT.
#[tokio::test]
async fn migration_008_entity_type_id_column_present_label_retained() {
    let (graph, _tmp) = open_file_backed_graph().await;
    let cols = table_columns(&graph, "entities").await;

    assert!(
        cols.iter().any(|c| c == "entity_type_id"),
        "entities must have `entity_type_id` column after migrations 008+009; found: {cols:?}"
    );
    // Phase 2 (Migration 009) runs after Migration 008 on open() — label is now dropped.
    assert!(
        !cols.iter().any(|c| c == "label"),
        "entities must NOT have `label` column after Phase 2 (Migration 009) runs; \
         found: {cols:?}"
    );
}

// ─── AC.c: backfill maps labels to correct entity_type_ids ───────────────────

/// Entities whose label matches a seeded entity_types row get the corresponding
/// non-zero entity_type_id. Entities with label='Entity' stay at id=0.
///
/// Phase 2 (Migration 009) drops the `entities.label` column.  This test verifies
/// the backfill invariant via `entity_type_id` + the `entity_types` registry only
/// (no reference to the dropped `label` column on `entities`).
#[tokio::test]
async fn migration_008_backfill_maps_labels_to_type_ids() {
    let entities = [
        ("alice", "Person", "demo"),
        ("acme", "Organisation", "demo"),
        ("unknown_ent", "Entity", "demo"),
    ];
    let (graph, _tmp) = open_graph_with_pre_seeded_entities(&entities).await;

    // Pre-condition: Migration 009 must have dropped entities.label.
    let cols = table_columns(&graph, "entities").await;
    assert!(
        !cols.iter().any(|c| c == "label"),
        "pre-condition: entities.label must be absent (Migration 009 ran); found: {cols:?}"
    );

    // Fetch entity_type_id for each entity using the LEFT JOIN pattern
    // (same shape as graph.rs::row_to_entity).
    let mut rows = graph
        .conn
        .query(
            "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.entity_type_id \
             FROM entities e \
             LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
             WHERE e.group_id = 'demo' ORDER BY e.id",
            (),
        )
        .await
        .expect("entities query must succeed");

    let mut results: Vec<(String, String, i64)> = Vec::new();
    while let Some(row) = rows.next().await.expect("iteration must not error") {
        let id: String = row.get(0).expect("id");
        let label: String = row.get(1).expect("label");
        let type_id: i64 = row.get(2).expect("entity_type_id");
        results.push((id, label, type_id));
    }

    // Get registry mapping for 'demo' group.
    let mut type_rows = graph
        .conn
        .query(
            "SELECT id, name FROM entity_types WHERE group_id = 'demo' AND id > 0",
            (),
        )
        .await
        .expect("entity_types query must succeed");

    let mut type_map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    while let Some(row) = type_rows.next().await.expect("type_rows iteration must not error") {
        let tid: i64 = row.get(0).expect("id");
        let name: String = row.get(1).expect("name");
        type_map.insert(name, tid);
    }

    // Verify each entity has the correct entity_type_id.
    for (id, label, type_id) in &results {
        if label == "Entity" {
            assert_eq!(
                *type_id, 0,
                "entity '{id}' with label='Entity' must have entity_type_id=0; got={type_id}"
            );
        } else {
            let expected_id = type_map.get(label.as_str()).copied().unwrap_or_else(|| {
                panic!(
                    "label '{label}' for entity '{id}' must be in entity_types registry; \
                     registry: {type_map:?}"
                )
            });
            assert_ne!(
                expected_id, 0,
                "non-Entity label '{label}' registry id must be > 0"
            );
            assert_eq!(
                *type_id, expected_id,
                "entity '{id}' with label='{label}' must have entity_type_id={expected_id}; \
                 got={type_id}"
            );
        }
    }

    // Explicit check: entity with label='Entity' must be id=0.
    let entity_row = results
        .iter()
        .find(|(id, _, _)| id == "unknown_ent")
        .expect("unknown_ent must be present in results");
    assert_eq!(
        entity_row.2, 0,
        "entity 'unknown_ent' with label='Entity' must have entity_type_id=0; got={}",
        entity_row.2
    );
}

// ─── AC.b: idempotency (double-apply) ────────────────────────────────────────

/// Running `run_migrations` twice must be a no-op: no errors, no duplicate rows,
/// no column changes.
#[tokio::test]
async fn migration_008_idempotent_double_apply() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let cols_first = table_columns(&graph, "entities").await;
    let count_first = cols_first.len();

    let mut type_count_rows = graph
        .conn
        .query("SELECT COUNT(*) FROM entity_types", ())
        .await
        .expect("entity_types count must succeed");
    let type_count_first: i64 = type_count_rows
        .next()
        .await
        .expect("iteration must not error")
        .expect("must have a count row")
        .get(0)
        .expect("count at col 0");

    // Second run — must not error.
    graph
        .run_migrations_again_for_test()
        .await
        .expect(
            "second run_migrations must be idempotent for migration 008 — \
             duplicate-column or UNIQUE-constraint error means the idempotency gate is missing",
        );

    // Column count unchanged.
    let cols_second = table_columns(&graph, "entities").await;
    assert_eq!(
        count_first,
        cols_second.len(),
        "entities column count must be unchanged after second migration run: \
         first={count_first}, second={}",
        cols_second.len()
    );

    // entity_types row count unchanged.
    let mut type_count_rows2 = graph
        .conn
        .query("SELECT COUNT(*) FROM entity_types", ())
        .await
        .expect("entity_types count must succeed");
    let type_count_second: i64 = type_count_rows2
        .next()
        .await
        .expect("iteration must not error")
        .expect("must have a count row")
        .get(0)
        .expect("count at col 0");

    assert_eq!(
        type_count_first, type_count_second,
        "entity_types row count must be unchanged after second migration run: \
         first={type_count_first}, second={type_count_second}"
    );

    // entity_type_id column must still be present.
    assert!(
        cols_second.iter().any(|c| c == "entity_type_id"),
        "entity_type_id column must still be present after second migration run; \
         columns: {cols_second:?}"
    );
}

// ─── AC.e: idx_entities_type_id index exists ─────────────────────────────────

/// The composite index `idx_entities_type_id` must exist on `entities` after migration.
#[tokio::test]
async fn migration_008_index_exists() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let mut rows = graph
        .conn
        .query("PRAGMA index_list('entities')", ())
        .await
        .expect("PRAGMA index_list('entities') must succeed");

    let mut index_names: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let name: String = row.get(1).expect("index name at column 1");
        index_names.push(name);
    }

    assert!(
        index_names.iter().any(|n| n == "idx_entities_type_id"),
        "idx_entities_type_id must exist after migration 008; found indexes: {index_names:?}"
    );
}

// ─── AC.f: PRAGMA foreign_key_check clean ────────────────────────────────────

/// `PRAGMA foreign_key_check` must return empty after migration 008.
#[tokio::test]
async fn migration_008_foreign_key_check_clean() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let cols = table_columns(&graph, "entities").await;
    assert!(
        cols.iter().any(|c| c == "entity_type_id"),
        "pre-condition: entity_type_id must exist — migration 008 must have run"
    );

    let mut violations = graph
        .conn
        .query("PRAGMA foreign_key_check", ())
        .await
        .expect("PRAGMA foreign_key_check must execute without error");

    let first_violation = violations
        .next()
        .await
        .expect("row iteration must not error");

    assert!(
        first_violation.is_none(),
        "PRAGMA foreign_key_check must return empty result set after migration 008"
    );
}

// ─── AC.g: static source-code gate ───────────────────────────────────────────

/// migrations.rs must statically contain the migration 008 DDL and guard strings.
#[test]
fn migration_008_source_gates_present() {
    use std::path::Path;

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let migrations_src = manifest_dir.join("src/core/migrations.rs");

    let content = std::fs::read_to_string(&migrations_src)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", migrations_src.display()));

    assert!(
        content.contains("CREATE TABLE IF NOT EXISTS entity_types"),
        "migrations.rs must contain `CREATE TABLE IF NOT EXISTS entity_types`; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("PRAGMA table_info('entities')"),
        "migrations.rs must contain `PRAGMA table_info('entities')` as the migration 008 G1 gate; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("idx_entities_type_id"),
        "migrations.rs must contain `idx_entities_type_id` index DDL; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("INSERT OR IGNORE INTO entity_types"),
        "migrations.rs must contain `INSERT OR IGNORE INTO entity_types` for idempotent seeding; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("migrate_008_entity_types"),
        "migrations.rs must define `migrate_008_entity_types` function; \
         found in: {}",
        migrations_src.display()
    );
}

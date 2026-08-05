#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-115 regression guard — permanent (`.ai-docs/tech-debt/tech-debt-register.md`
//! session 2026-07-14).
//!
//! Before `migrate_023_vector_index_column_type`, `entities.embedding` was
//! declared a plain `BLOB` on every migrated DB, which libSQL's DiskANN vector
//! index rejects (`SqliteFailure(1, "vector index: unexpected vector column
//! type: BLOB")`). The `CREATE INDEX` failure was swallowed, so
//! `vector_top_k('entities_vec_idx', ...)` silently found no index and every
//! recall fell back to brute-force. This test asserts the index ACTUALLY
//! ENGAGES on a real file-backed DB — not just that search returns correct
//! results (brute-force is also correct; that's what made the bug invisible).
//!
//! Verified RED before the fix (manually reverting the `migrate_023` call in
//! `schema.rs::run_migrations` reproduces the `libsql_vector_idx` rejection
//! error here) and GREEN after.

use kremory::core::schema::TemporalGraph;

/// Args-as-object per rust-conventions §too_many_arguments (clippy.toml
/// threshold 3).
struct NewTestEntity<'a> {
    id: &'a str,
    group_id: &'a str,
    embedding: &'a [f32],
}

async fn insert_entity_with_group(graph: &TemporalGraph, entity: NewTestEntity<'_>) {
    let NewTestEntity {
        id,
        group_id,
        embedding,
    } = entity;
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO entities (id, group_id, entity_type_id, properties, recorded_at) \
             VALUES (?1, ?2, 0, '{}', ?3)",
            libsql::params![id, group_id, now],
        )
        .await
        .expect("insert entity");
    graph
        .set_entity_embedding(id, embedding)
        .await
        .expect("set embedding");
}

/// (a) `entities_vec_idx` exists in `sqlite_master` after migration (the
/// index name is only ever registered there on a SUCCESSFUL create — a
/// failed create leaves no trace in `sqlite_master`).
///
/// (b) A direct `vector_top_k('entities_vec_idx', ...)` query — the exact
/// mechanism `vector_search_with_index` relies on — returns rows, proving
/// the ANN index path is live (not silently absent, forcing brute-force).
#[tokio::test]
async fn entities_vec_idx_exists_and_vector_top_k_returns_rows() {
    let tmp = std::env::temp_dir().join(format!(
        "td115_regression_{}_{}.db",
        std::process::id(),
        "entities_vec_idx"
    ));
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_str().unwrap().to_string();

    let graph = TemporalGraph::open_with_dim(&path, 384)
        .await
        .expect("open_with_dim must run migrate_023 to completion");

    // (a) index registered in sqlite_master — a failed CREATE INDEX never
    // reaches sqlite_master, so this alone would have failed pre-fix.
    let mut idx_rows = graph
        .conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='entities_vec_idx'",
            (),
        )
        .await
        .expect("query sqlite_master");
    assert!(
        idx_rows.next().await.expect("row").is_some(),
        "entities_vec_idx must exist in sqlite_master after open_with_dim \
         (pre-fix: CREATE INDEX failed silently against a BLOB column, so the \
         index name never registered)"
    );

    // Seed 3 real 384-dim embeddings: "a" and "c" (near-duplicate of "a") vs
    // "b" (orthogonal) — mirrors the load-bearing spike from the migration's
    // design phase.
    let mut a = vec![0.0_f32; 384];
    a[0] = 1.0;
    let mut b = vec![0.0_f32; 384];
    b[1] = 1.0;
    let mut c = vec![0.0_f32; 384];
    c[0] = 0.99;
    c[1] = 0.01;

    insert_entity_with_group(
        &graph,
        NewTestEntity {
            id: "a",
            group_id: "default",
            embedding: &a,
        },
    )
    .await;
    insert_entity_with_group(
        &graph,
        NewTestEntity {
            id: "b",
            group_id: "default",
            embedding: &b,
        },
    )
    .await;
    insert_entity_with_group(
        &graph,
        NewTestEntity {
            id: "c",
            group_id: "default",
            embedding: &c,
        },
    )
    .await;

    // (b) direct vector_top_k against the named index — this is the exact
    // failure surface: pre-fix, this query errors with "unexpected vector
    // column type: BLOB" because the index never created.
    let vec_str = format!(
        "[{}]",
        a.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut rows = graph
        .conn
        .query(
            "SELECT e.id FROM vector_top_k('entities_vec_idx', vector(?1), 2) AS v \
             JOIN entities AS e ON e.rowid = v.id \
             ORDER BY vector_distance_cos(e.embedding, vector(?1)) ASC",
            libsql::params![vec_str],
        )
        .await
        .expect("vector_top_k query must succeed (index must exist and accept the column type)");

    let mut ids = Vec::new();
    while let Some(row) = rows.next().await.expect("row") {
        let id: String = row.get(0).expect("id");
        ids.push(id);
    }

    assert_eq!(
        ids.len(),
        2,
        "expected top-2 nearest neighbours, got {ids:?}"
    );
    assert_eq!(ids[0], "a", "exact match must rank first: {ids:?}");
    assert_eq!(
        ids[1], "c",
        "near-duplicate must rank second (ahead of orthogonal 'b'): {ids:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Same assertion for `facts_vec_idx` — `facts.embedding` was ALREADY
/// `F32_BLOB(dim)` on the normal forward-migration path (migration 006
/// threads `dim` through correctly), but `migrate_023` defensively covers a
/// DB that reached a `BLOB` `facts.embedding` via the emergency downgrade
/// path. Assert the index is present and functional either way.
#[tokio::test]
async fn facts_vec_idx_exists_and_is_functional() {
    let tmp = std::env::temp_dir().join(format!(
        "td115_regression_{}_{}.db",
        std::process::id(),
        "facts_vec_idx"
    ));
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_str().unwrap().to_string();

    let graph = TemporalGraph::open_with_dim(&path, 384)
        .await
        .expect("open_with_dim");

    let mut idx_rows = graph
        .conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='facts_vec_idx'",
            (),
        )
        .await
        .expect("query sqlite_master");
    assert!(
        idx_rows.next().await.expect("row").is_some(),
        "facts_vec_idx must exist in sqlite_master after open_with_dim"
    );

    let mut type_rows = graph
        .conn
        .query("PRAGMA table_info(facts)", ())
        .await
        .expect("pragma table_info");
    let mut embedding_type = None;
    while let Some(row) = type_rows.next().await.expect("row") {
        let name: String = row.get(1).expect("name");
        if name == "embedding" {
            embedding_type = Some(row.get::<String>(2).expect("type"));
        }
    }
    assert_eq!(
        embedding_type.as_deref(),
        Some("F32_BLOB(384)"),
        "facts.embedding must be F32_BLOB(384), not a generic BLOB"
    );

    let _ = std::fs::remove_file(&tmp);
}

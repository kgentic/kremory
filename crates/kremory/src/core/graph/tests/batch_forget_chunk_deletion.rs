use super::super::*;
use crate::core::schema::TemporalGraph;

// === batch_forget — 100-item chunk deletion ===

#[tokio::test]
async fn batch_forget_250_entities_deleted_cleanly() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Insert 250 entities.
    let ids: Vec<String> = (0..250).map(|i| format!("ent-{i:04}")).collect();
    for id in &ids {
        g.insert_entity(InsertEntityParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
    }

    // Verify setup.
    let before = g.list_entities().await.unwrap();
    assert_eq!(before.len(), 250);

    let deleted = g.batch_forget(&ids).await.unwrap();
    assert_eq!(
        deleted.entities, 250,
        "batch_forget must delete all 250 entities"
    );

    let after = g.list_entities().await.unwrap();
    assert!(
        after.is_empty(),
        "no entities must remain after batch_forget"
    );
}

#[tokio::test]
async fn batch_forget_empty_slice_is_noop() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    let deleted = g.batch_forget(&[]).await.unwrap();
    assert_eq!(deleted, crate::core::graph::BatchForgetCounts::default());
}

#[tokio::test]
async fn upsert_entity_with_group_inserts_then_updates() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = format!("{}/test.db", tmp.path().display());
    let g = TemporalGraph::open(&db).await.expect("open");

    // First call inserts (entity_type_id=0 = catch-all).
    g.upsert_entity_with_group(UpsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({"context": "v1"}),
        group_id: None,
    })
    .await
    .expect("first upsert");

    // After Phase 2 (Migration 009), entities.label is gone.
    // Verify via entity_type_id + properties columns.
    let mut rows = g
        .conn
        .query(
            "SELECT entity_type_id, properties FROM entities WHERE id = ?1",
            libsql::params!["alice"],
        )
        .await
        .expect("query");
    let r1 = rows.next().await.expect("row").expect("some");
    let etype1: i64 = r1.get(0).expect("entity_type_id");
    let props1: String = r1.get(1).expect("props");
    assert_eq!(etype1, 0, "first upsert entity_type_id must be 0");
    assert!(props1.contains("v1"));

    // Second call updates in place — same id, no duplicate row.
    // entity_type_id changes from 0 to 1 to verify ON CONFLICT updates it.
    g.upsert_entity_with_group(UpsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 1,
        properties: serde_json::json!({"context": "v2"}),
        group_id: None,
    })
    .await
    .expect("second upsert");

    let mut count_rows = g
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = ?1",
            libsql::params!["alice"],
        )
        .await
        .expect("count");
    let cr = count_rows.next().await.expect("cnt").expect("some");
    let count: i64 = cr.get(0).expect("count");
    assert_eq!(count, 1, "upsert must not create duplicate");

    let mut rows2 = g
        .conn
        .query(
            "SELECT entity_type_id, properties FROM entities WHERE id = ?1",
            libsql::params!["alice"],
        )
        .await
        .expect("query2");
    let r2 = rows2.next().await.expect("row").expect("some");
    let etype2: i64 = r2.get(0).expect("entity_type_id");
    let props2: String = r2.get(1).expect("props");
    assert_eq!(
        etype2, 1,
        "entity_type_id must be updated by ON CONFLICT path"
    );
    assert!(props2.contains("v2"), "properties must be updated");

    // FTS shadow row consistency — exactly 1 fts row after upsert (DELETE+INSERT).
    let mut fts_count = g
        .conn
        .query(
            "SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?1",
            libsql::params!["alice"],
        )
        .await
        .expect("fts count");
    let fc = fts_count.next().await.expect("row").expect("some");
    let fts_n: i64 = fc.get(0).expect("n");
    assert_eq!(fts_n, 1, "FTS shadow must have exactly 1 row after upsert");
}

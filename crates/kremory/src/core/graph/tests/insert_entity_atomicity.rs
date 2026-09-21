use super::super::*;
use crate::core::schema::TemporalGraph;

// === insert_entity atomicity ===
//
// Verifies that if the FTS insert fails (simulated by dropping entities_fts
// before the call), the entities row is NOT committed — i.e., the transaction
// rolls back cleanly and does not leave a partially-indexed entity.

#[tokio::test]
async fn insert_entity_rolls_back_entities_row_when_fts_fails() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Drop entities_fts to force the second INSERT inside insert_entity to fail.
    g.conn
        .execute("DROP TABLE entities_fts", ())
        .await
        .expect("DROP TABLE entities_fts must succeed on fresh in-memory DB");

    let result = g
        .insert_entity(InsertEntityParams {
            id: "test-atomic-1",
            entity_type_id: 0,
            properties: serde_json::json!({"text": "hello"}),
        })
        .await;
    assert!(
        result.is_err(),
        "insert_entity must return Err when FTS table is missing (no partial commit)"
    );

    // The entities row must have been rolled back — zero rows for the attempted ID.
    let mut rows = g
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = 'test-atomic-1'",
            (),
        )
        .await
        .expect("SELECT on entities must succeed even after FTS drop");
    let row = rows.next().await.unwrap().unwrap();
    let count: i64 = row.get(0).unwrap();
    assert_eq!(
        count, 0,
        "entities row must be rolled back when FTS insert fails (HIGH-2)"
    );
}

#[tokio::test]
async fn insert_entity_with_group_rolls_back_entities_row_when_fts_fails() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Drop entities_fts to force the second INSERT to fail.
    g.conn
        .execute("DROP TABLE entities_fts", ())
        .await
        .expect("DROP TABLE entities_fts must succeed on fresh in-memory DB");

    let result = g
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id: "test-atomic-grp-1",
            entity_type_id: 0,
            properties: serde_json::json!({"text": "hello"}),
            group_id: Some("grp-a"),
        })
        .await;
    assert!(
        result.is_err(),
        "insert_entity_with_group must return Err when FTS table is missing"
    );

    let mut rows = g
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = 'test-atomic-grp-1'",
            (),
        )
        .await
        .expect("SELECT on entities must succeed even after FTS drop");
    let row = rows.next().await.unwrap().unwrap();
    let count: i64 = row.get(0).unwrap();
    assert_eq!(
        count, 0,
        "entities row must be rolled back when FTS insert fails (HIGH-2, insert_entity_with_group)"
    );
}


use super::super::*;
use crate::core::schema::TemporalGraph;

// === Entity CRUD ===

#[tokio::test]
async fn test_insert_and_get_entity() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({"role": "engineer"}),
    })
    .await
    .unwrap();
    let entity = g.get_entity("alice").await.unwrap().unwrap();
    assert_eq!(entity.id, "alice");
    // entity_type_id=0 (catch-all) → label resolves to "Entity" via LEFT JOIN COALESCE.
    assert_eq!(entity.entity_type_id, 0);
    assert_eq!(entity.label, "Entity");
    assert_eq!(entity.properties["role"], "engineer");
    assert!(entity.updated_at.is_none());
}

#[tokio::test]
async fn test_get_nonexistent_entity_returns_none() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    let result = g.get_entity("nobody").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_update_entity_properties() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({"role": "engineer"}),
    })
    .await
    .unwrap();
    g.update_entity("alice", serde_json::json!({"role": "manager"}))
        .await
        .unwrap();
    let entity = g.get_entity("alice").await.unwrap().unwrap();
    assert_eq!(entity.properties["role"], "manager");
    assert!(entity.updated_at.is_some());
}

#[tokio::test]
async fn test_list_entities() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    g.insert_entity(InsertEntityParams {
        id: "bob",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    g.insert_entity(InsertEntityParams {
        id: "acme",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let entities = g.list_entities().await.unwrap();
    assert_eq!(entities.len(), 3);
    let mut ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
    ids.sort();
    assert_eq!(ids, vec!["acme", "alice", "bob"]);
}


use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === petgraph export ===

#[tokio::test]
async fn test_to_petgraph_nodes_and_edges() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
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
    g.insert_entity(InsertEntityParams {
        id: "bob",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let t0 = Utc::now() - Duration::hours(1);
    g.insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
        .await
        .unwrap();
    g.insert_fact(FactInsert::new("bob", "works_at", t0).object_id("acme"))
        .await
        .unwrap();

    let pg = g.to_petgraph().await.unwrap();
    assert_eq!(pg.node_count(), 3);
    assert_eq!(pg.edge_count(), 2);
}

#[tokio::test]
async fn test_to_petgraph_excludes_expired() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
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
    let t0 = Utc::now() - Duration::days(2);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
        .await
        .unwrap();
    g.invalidate_fact(id, Utc::now() - Duration::days(1))
        .await
        .unwrap();

    let pg = g.to_petgraph().await.unwrap();
    assert_eq!(pg.node_count(), 2);
    assert_eq!(pg.edge_count(), 0);
}


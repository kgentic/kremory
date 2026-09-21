use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === Graph Traversal ===

#[tokio::test]
async fn test_get_neighbours_one_hop() {
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
    g.insert_fact(FactInsert::new("alice", "manages", t0).object_id("bob"))
        .await
        .unwrap();

    let subgraph = g.get_neighbours("alice", 1).await.unwrap();
    let mut entity_ids: Vec<&str> = subgraph.entities.iter().map(|e| e.id.as_str()).collect();
    entity_ids.sort();
    assert!(entity_ids.contains(&"alice"));
    assert!(entity_ids.contains(&"acme"));
    assert!(entity_ids.contains(&"bob"));
    assert_eq!(subgraph.facts.len(), 2);
}

#[tokio::test]
async fn test_get_neighbours_two_hops() {
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
    let t0 = Utc::now() - Duration::hours(1);
    g.insert_fact(FactInsert::new("alice", "manages", t0).object_id("bob"))
        .await
        .unwrap();
    g.insert_fact(FactInsert::new("bob", "works_at", t0).object_id("acme"))
        .await
        .unwrap();

    let subgraph = g.get_neighbours("alice", 2).await.unwrap();
    let entity_ids: Vec<&str> = subgraph.entities.iter().map(|e| e.id.as_str()).collect();
    assert!(entity_ids.contains(&"alice"));
    assert!(entity_ids.contains(&"bob"));
    assert!(entity_ids.contains(&"acme"));
    assert_eq!(subgraph.facts.len(), 2);
}

#[tokio::test]
async fn test_get_neighbours_excludes_expired_facts() {
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

    let subgraph = g.get_neighbours("alice", 1).await.unwrap();
    // Only alice; acme should not be reached via the expired fact
    assert_eq!(subgraph.facts.len(), 0);
    assert_eq!(subgraph.entities.len(), 1);
    assert_eq!(subgraph.entities[0].id, "alice");
}

#[tokio::test]
async fn test_get_neighbours_zero_hops() {
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
    let t0 = Utc::now() - Duration::hours(1);
    g.insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
        .await
        .unwrap();

    let subgraph = g.get_neighbours("alice", 0).await.unwrap();
    assert_eq!(subgraph.facts.len(), 0);
    assert_eq!(subgraph.entities.len(), 1);
    assert_eq!(subgraph.entities[0].id, "alice");
}


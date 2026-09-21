use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === Temporal Queries ===

#[tokio::test]
async fn test_facts_at_filters_by_valid_from() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let t0 = Utc::now();
    g.insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    // query just before valid_from: should return nothing
    let facts_before = g.facts_at(t0 - Duration::seconds(1)).await.unwrap();
    assert_eq!(facts_before.len(), 0);
    // query after valid_from: should return the fact
    let facts_after = g.facts_at(t0 + Duration::seconds(1)).await.unwrap();
    assert_eq!(facts_after.len(), 1);
    assert_eq!(facts_after[0].predicate, "has_title");
}

#[tokio::test]
async fn test_facts_at_excludes_expired() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let t0 = Utc::now() - Duration::days(2);
    let id = g
        .insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    g.invalidate_fact(id, Utc::now() - Duration::days(1))
        .await
        .unwrap();
    let facts = g.facts_at(Utc::now()).await.unwrap();
    assert_eq!(facts.len(), 0);
}

#[tokio::test]
async fn test_entity_facts_at_filters_by_entity() {
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
    let t0 = Utc::now() - Duration::hours(1);
    g.insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    g.insert_fact(FactInsert::new("bob", "has_title", t0).object_value("Eng"))
        .await
        .unwrap();
    let alice_facts = g.entity_facts_at("alice", Utc::now()).await.unwrap();
    assert_eq!(alice_facts.len(), 1);
    assert_eq!(alice_facts[0].subject_id, "alice");
    assert_eq!(alice_facts[0].object_value.as_deref(), Some("PM"));
}

#[tokio::test]
async fn test_entity_history_includes_expired() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let t0 = Utc::now() - Duration::days(2);
    let id = g
        .insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    g.invalidate_fact(id, Utc::now() - Duration::days(1))
        .await
        .unwrap();
    // expired fact should NOT appear in facts_at
    let live = g.facts_at(Utc::now()).await.unwrap();
    assert_eq!(live.len(), 0);
    // but it SHOULD appear in entity_history
    let history = g.entity_history("alice").await.unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].predicate, "has_title");
    assert!(history[0].expired_at.is_some());
}

#[tokio::test]
async fn test_point_in_time_temporal_evolution() {
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
        id: "newco",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();

    let t0 = Utc::now() - Duration::days(10);
    let t1 = Utc::now() - Duration::days(5);

    // alice works_at acme from t0
    let fact1_id = g
        .insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
        .await
        .unwrap();

    // at t0+1d: acme fact is visible
    let facts_t0 = g.facts_at(t0 + Duration::days(1)).await.unwrap();
    assert_eq!(facts_t0.len(), 1);
    assert_eq!(facts_t0[0].object_id.as_deref(), Some("acme"));

    // administratively retract acme fact at t1, add newco fact from t1
    g.invalidate_fact(fact1_id, t1).await.unwrap();
    g.insert_fact(FactInsert::new("alice", "works_at", t1).object_id("newco"))
        .await
        .unwrap();

    // at t1+1d: only newco fact is visible
    let facts_t1 = g.facts_at(t1 + Duration::days(1)).await.unwrap();
    assert_eq!(facts_t1.len(), 1);
    assert_eq!(facts_t1[0].object_id.as_deref(), Some("newco"));

    // history should show both facts for alice
    let history = g.entity_history("alice").await.unwrap();
    assert_eq!(history.len(), 2);
    let predicates: Vec<&str> = history.iter().map(|f| f.predicate.as_str()).collect();
    assert!(predicates.iter().all(|&p| p == "works_at"));
    let object_ids: Vec<Option<&str>> = history.iter().map(|f| f.object_id.as_deref()).collect();
    assert!(object_ids.contains(&Some("acme")));
    assert!(object_ids.contains(&Some("newco")));
}


use super::super::*;
use crate::core::schema::TemporalGraph;

// === forget_entity — cascade delete atomicity ===

#[tokio::test]
async fn forget_entity_removes_entity_facts_and_edges() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Insert entity + fact + episodic edge referencing it.
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let now = chrono::Utc::now();
    // insert_fact: (subject_id, predicate, object_id, object_value,
    //               valid_from, confidence, source_episode_id, embedding)
    // Use object_value (not object_id) to avoid FK on entities for "bob".
    g.insert_fact(
        FactInsert::new("alice", "knows", now)
            .object_value("bob")
            .confidence(0.9),
    )
    .await
    .unwrap();
    // insert_episodic_edge: (episode_id: i64, entity_id, entity_group_id, role)
    // Must have a valid episode first (FK constraint).
    let ep_id = g
        .insert_episode(InsertEpisodeParams {
            content: "test episode",
            timestamp: now,
            source_type: None,
            metadata: None,
        })
        .await
        .unwrap();
    g.insert_episodic_edge(InsertEpisodicEdgeParams {
        episode_id: ep_id,
        entity_id: "alice",
        entity_group_id: None,
        role: "subject",
    })
    .await
    .unwrap();

    // Verify setup.
    assert!(g.get_entity("alice").await.unwrap().is_some());

    // Forget should return true (entity was present).
    let found = g.forget_entity("alice").await.unwrap();
    assert!(found, "forget_entity must return true when entity existed");

    // Entity gone.
    assert!(g.get_entity("alice").await.unwrap().is_none());

    // Facts with alice as subject should be gone.
    let facts = g.entity_history("alice").await.unwrap();
    assert!(facts.is_empty(), "facts referencing alice must be deleted");

    // Episodic edges gone.
    let edges = g.episodic_edges_for_entity("alice").await.unwrap();
    assert!(edges.is_empty(), "episodic_edges for alice must be deleted");
}

#[tokio::test]
async fn forget_entity_returns_false_for_nonexistent() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    let found = g.forget_entity("ghost").await.unwrap();
    assert!(!found, "forget_entity must return false when entity absent");
}


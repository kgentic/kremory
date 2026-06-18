use super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === Entity CRUD ===

#[tokio::test]
async fn test_insert_and_get_entity() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
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
    g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    let entities = g.list_entities().await.unwrap();
    assert_eq!(entities.len(), 3);
    let mut ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
    ids.sort();
    assert_eq!(ids, vec!["acme", "alice", "bob"]);
}

// === Fact CRUD ===

#[tokio::test]
async fn test_insert_fact_returns_id() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let id = g
        .insert_fact(FactInsert::new("alice", "exists", Utc::now()))
        .await
        .unwrap();
    assert!(id > 0);
}

/// ADR-035 §5 Option A: `try_insert_fact` returns `Ok(Some(id))` on fresh insert.
#[tokio::test]
async fn test_try_insert_fact_returns_some_on_fresh_insert() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let id = g
        .try_insert_fact(FactInsert::new("alice", "exists", Utc::now()))
        .await
        .unwrap();
    assert!(id.is_some(), "fresh insert must return Some(id)");
    assert!(id.unwrap() > 0);
}

/// ADR-035 §5 Option A: `try_insert_fact` returns `Ok(None)` on content_hash collision.
#[tokio::test]
async fn test_try_insert_fact_returns_none_on_duplicate() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t0 = Utc::now();
    let _first = g
        .insert_fact(FactInsert::new("alice", "exists", t0))
        .await
        .unwrap();
    let dup = g
        .try_insert_fact(FactInsert::new("alice", "exists", t0))
        .await
        .unwrap();
    assert!(
        dup.is_none(),
        "duplicate triple must return None (silent dedup)"
    );
}

/// ADR-035 §5 Option A: `try_insert_fact_with_group` returns `Ok(Some(id))` on fresh insert.
#[tokio::test]
async fn test_try_insert_fact_with_group_returns_some_on_fresh_insert() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    // Entities are global (FK on entity id only, not (id, group_id)) — pattern
    // mirrors search.rs:1700 + 1704 successful with_group tests.
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let id = g
        .try_insert_fact_with_group(FactInsert::new("alice", "exists", Utc::now()), Some("g1"))
        .await
        .unwrap();
    assert!(id.is_some(), "fresh insert must return Some(id)");
}

/// ADR-035 finding: `content_hash` does NOT include `group_id` — same triple
/// in different groups still collides. This test pins the invariant that
/// caller's `insert_fact_with_group(group=X)` and engine's `insert_fact(no group)`
/// will dedup against each other (the basis of Path X caller-wins semantics).
#[tokio::test]
async fn test_try_insert_fact_with_group_dedups_cross_variant() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t0 = Utc::now();
    // First: insert via plain insert_fact (no group_id) — simulates LLM Phase 2.
    let _first = g
        .insert_fact(FactInsert::new("alice", "exists", t0))
        .await
        .unwrap();
    // Second: try_insert_fact_with_group (with group_id) — simulates caller pin.
    // Expect None: content_hash collides regardless of group_id.
    let dup = g
        .try_insert_fact_with_group(FactInsert::new("alice", "exists", t0), Some("g1"))
        .await
        .unwrap();
    assert!(
        dup.is_none(),
        "with_group + no_group SAME triple must collide (content_hash is group-agnostic)"
    );
}

#[tokio::test]
async fn test_insert_fact_with_object_id() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", Utc::now()).object_id("acme"))
        .await
        .unwrap();
    assert!(id > 0);
    let facts = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].predicate, "works_at");
    assert_eq!(facts[0].subject_id, "alice");
    assert_eq!(facts[0].object_id.as_deref(), Some("acme"));
}

#[tokio::test]
async fn test_insert_fact_with_object_value() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let id = g
        .insert_fact(FactInsert::new("alice", "has_title", Utc::now()).object_value("PM"))
        .await
        .unwrap();
    assert!(id > 0);
    let facts = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].predicate, "has_title");
    assert_eq!(facts[0].object_value.as_deref(), Some("PM"));
    assert!(facts[0].object_id.is_none());
}

#[tokio::test]
async fn test_invalidate_fact() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t0 = Utc::now() - Duration::days(1);
    let id = g
        .insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    let facts_before = g.facts_at(Utc::now()).await.unwrap();
    assert_eq!(facts_before.len(), 1);
    g.invalidate_fact(id, Utc::now()).await.unwrap();
    let facts_after = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
    assert_eq!(facts_after.len(), 0);
}

// === Temporal Queries ===

#[tokio::test]
async fn test_facts_at_filters_by_valid_from() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("newco", 0, serde_json::json!({}))
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

// === Graph Traversal ===

#[tokio::test]
async fn test_get_neighbours_one_hop() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
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

// === petgraph export ===

#[tokio::test]
async fn test_to_petgraph_nodes_and_edges() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
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
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
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

// === Episode CRUD ===

#[tokio::test]
async fn test_insert_episode() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    let id = g
        .insert_episode(
            "Alice joined the meeting.",
            Utc::now(),
            Some("transcript"),
            Some(serde_json::json!({"speaker": "alice"})),
        )
        .await
        .unwrap();
    assert!(id > 0);
}

// === New E1.S3 Tests ===

#[tokio::test]
async fn test_insert_episodic_edge() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let ep_id = g
        .insert_episode("Alice joined.", Utc::now(), Some("transcript"), None)
        .await
        .unwrap();
    let edge_id = g
        .insert_episodic_edge(ep_id, "alice", None, "mentioned")
        .await
        .unwrap();
    assert!(edge_id > 0);

    let edges = g.episodic_edges_for_entity("alice").await.unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].entity_id, "alice");
    assert_eq!(edges[0].role, "mentioned");
    assert_eq!(edges[0].episode_id, ep_id);
}

#[tokio::test]
async fn test_invalidate_fact_with_reason() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t0 = Utc::now() - Duration::days(2);
    let id = g
        .insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    let expired = Utc::now() - Duration::days(1);
    let invalid = Utc::now() - Duration::hours(12);
    g.invalidate_fact_with_reason(id, expired, invalid)
        .await
        .unwrap();

    // Fact should now be absent from active facts
    let facts = g.facts_at(Utc::now()).await.unwrap();
    assert_eq!(facts.len(), 0);

    // Full history should still show the fact with both timestamps set
    let history = g.entity_history("alice").await.unwrap();
    assert_eq!(history.len(), 1);
    assert!(history[0].expired_at.is_some());
    assert!(history[0].invalid_at.is_some());
}

#[tokio::test]
async fn test_get_facts_by_subject_predicate() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t0 = Utc::now() - Duration::hours(1);
    g.insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
        .await
        .unwrap();
    g.insert_fact(FactInsert::new("alice", "has_title", t0).object_value("Engineer"))
        .await
        .unwrap();

    let works_at_facts = g
        .get_facts_by_subject_predicate("alice", "works_at")
        .await
        .unwrap();
    assert_eq!(works_at_facts.len(), 1);
    assert_eq!(works_at_facts[0].object_id.as_deref(), Some("acme"));

    let title_facts = g
        .get_facts_by_subject_predicate("alice", "has_title")
        .await
        .unwrap();
    assert_eq!(title_facts.len(), 1);
    assert_eq!(title_facts[0].object_value.as_deref(), Some("Engineer"));
}

#[tokio::test]
async fn test_insert_entity_with_group() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity_with_group(
        "alice",
        0,
        serde_json::json!({"name": "Alice"}),
        Some("group-abc"),
    )
    .await
    .unwrap();

    let entity = g.get_entity("alice").await.unwrap().unwrap();
    assert_eq!(entity.id, "alice");
    assert_eq!(entity.group_id.as_deref(), Some("group-abc"));
}

/// Stream 3 A.4.5 contract test — updated for ADR-029b:
/// Lane A indexer call sites pass `group_id=None` (folder_id absent pre-Lane-B).
/// Post-ADR-029b, None maps to 'default' (entities.group_id is NOT NULL).
/// The entity is stored in the default namespace and is visible under
/// unscoped queries or queries scoped to 'default'.
#[tokio::test]
async fn lane_a_indexer_writes_null_group_id_when_folder_id_absent() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    let group_id: Option<&str> = None;
    g.insert_entity_with_group(
        "doc:welcome:chunk_0",
        0,
        serde_json::json!({
            "text": "the host application captures meetings and surfaces insights.",
            "source": "doc:welcome",
            "source_type": "document",
        }),
        group_id,
    )
    .await
    .unwrap();

    let entity = g.get_entity("doc:welcome:chunk_0").await.unwrap().unwrap();
    // ADR-029b: None maps to 'default' — not NULL (NOT NULL constraint enforced).
    assert_eq!(
        entity.group_id.as_deref(),
        Some("default"),
        "Lane A indexer: None group_id must persist as 'default' post-ADR-029b (got {:?})",
        entity.group_id,
    );
}

#[tokio::test]
async fn test_update_entity_group_changes_scope() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Insert entity in group-a
    g.insert_entity_with_group(
        "doc_chunk_0",
        0,
        serde_json::json!({"text": "original content"}),
        Some("group-a"),
    )
    .await
    .unwrap();

    let entity = g.get_entity("doc_chunk_0").await.unwrap().unwrap();
    assert_eq!(entity.group_id.as_deref(), Some("group-a"));

    // Re-scope to group-b with updated properties
    g.update_entity_group(
        "doc_chunk_0",
        Some("group-b"),
        serde_json::json!({"text": "updated content"}),
    )
    .await
    .unwrap();

    let updated = g.get_entity("doc_chunk_0").await.unwrap().unwrap();
    assert_eq!(updated.group_id.as_deref(), Some("group-b"));
    let props: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&updated.properties).unwrap()).unwrap();
    assert_eq!(props["text"], "updated content");
}

#[tokio::test]
async fn test_update_entity_group_to_none() {
    // ADR-029b: entities.group_id is NOT NULL post-migration-004.
    // Passing None to update_entity_group maps to 'default' (not NULL).
    let g = TemporalGraph::open_in_memory().await.unwrap();

    g.insert_entity_with_group("e1", 0, serde_json::json!({}), Some("scoped"))
        .await
        .unwrap();

    g.update_entity_group("e1", None, serde_json::json!({}))
        .await
        .unwrap();

    let entity = g.get_entity("e1").await.unwrap().unwrap();
    // None maps to 'default' — entity is now in the default namespace.
    assert_eq!(
        entity.group_id.as_deref(),
        Some("default"),
        "None group_id must map to 'default' post-ADR-029b"
    );
}

// === HIGH-2 RED→GREEN: insert_entity atomicity ===
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
        .insert_entity("test-atomic-1", 0, serde_json::json!({"text": "hello"}))
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
        .insert_entity_with_group(
            "test-atomic-grp-1",
            0,
            serde_json::json!({"text": "hello"}),
            Some("grp-a"),
        )
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

// === SHA-256 dedup (Story #209) ===

#[tokio::test]
async fn insert_fact_duplicate_returns_error() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t = Utc::now();
    // First insert succeeds
    g.insert_fact(FactInsert::new("alice", "likes", t).object_value("coffee"))
        .await
        .unwrap();
    // Second insert with same triple must return Duplicate error
    let err = g
        .insert_fact(
            FactInsert::new("alice", "likes", t)
                .object_value("coffee")
                .confidence(0.9),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::core::error::Error::Duplicate { .. }),
        "expected Duplicate error, got: {err:?}"
    );
}

#[tokio::test]
async fn insert_fact_different_triples_both_succeed() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t = Utc::now();
    g.insert_fact(FactInsert::new("alice", "likes", t).object_value("coffee"))
        .await
        .unwrap();
    // Different object_value → different hash → succeeds
    g.insert_fact(FactInsert::new("alice", "likes", t).object_value("tea"))
        .await
        .unwrap();
}

#[tokio::test]
async fn fact_content_hash_deterministic() {
    let h1 = fact_content_hash("alice", "likes", None, Some("coffee"));
    let h2 = fact_content_hash("alice", "likes", None, Some("coffee"));
    assert_eq!(h1, h2, "hash must be deterministic");
    let h3 = fact_content_hash("alice", "likes", None, Some("tea"));
    assert_ne!(h1, h3, "different facts must have different hashes");
    // SHA-256 produces 64 hex chars
    assert_eq!(h1.len(), 64);
}

// === SQLite-first vector ordering backfill (Story #214) ===

/// FU.8: facts_missing_embeddings returns (id, subject_id, predicate,
/// object_value, object_id) — the 5-tuple now includes object_id so callers
/// can prefer the entity reference over the literal string.
#[tokio::test]
async fn facts_missing_embeddings_returns_all_null_embedding_facts() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t = Utc::now();
    // Insert a fact with object_value only — simulates SQLite-committed, vector-not-written
    let fact_id = g
        .insert_fact(FactInsert::new("alice", "works_at", t).object_value("ACME"))
        .await
        .unwrap();

    let missing = g.facts_missing_embeddings().await.unwrap();
    assert_eq!(missing.len(), 1, "one fact has no embedding");
    let (id, _sub, _pred, object_value, object_id) = &missing[0];
    assert_eq!(*id, fact_id);
    assert_eq!(object_value.as_deref(), Some("ACME"));
    assert!(
        object_id.is_none(),
        "object_id must be None for literal-value fact"
    );

    // Backfill with a stub embedding
    let embedding: Vec<f32> = vec![0.1_f32; 384];
    g.backfill_fact_embedding(fact_id, &embedding)
        .await
        .unwrap();

    // After backfill, no facts should be missing
    let still_missing = g.facts_missing_embeddings().await.unwrap();
    assert!(
        still_missing.is_empty(),
        "backfill must clear the missing-embedding list"
    );
}

/// FU.8: facts with object_id (entity reference) expose the object_id in the
/// 5-tuple so callers can build the embedding text from the entity name rather
/// than a missing literal.
#[tokio::test]
async fn facts_missing_embeddings_includes_object_id() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();
    g.insert_entity("acme", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t = Utc::now();
    // Insert a fact with object_id (entity reference) — no object_value.
    let fact_id = g
        .insert_fact(FactInsert::new("alice", "works_at", t).object_id("acme"))
        .await
        .unwrap();

    let missing = g.facts_missing_embeddings().await.unwrap();
    let entry = missing
        .iter()
        .find(|(id, _, _, _, _)| *id == fact_id)
        .expect("fact must appear in missing list");
    let (_id, _sub, _pred, object_value, object_id) = entry;
    assert!(
        object_value.is_none(),
        "object_value must be None for entity-ref fact"
    );
    assert_eq!(
        object_id.as_deref(),
        Some("acme"),
        "object_id must be returned so callers can build the embedding text"
    );
}

#[tokio::test]
async fn backfill_fact_embedding_is_idempotent() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity("bob", 0, serde_json::json!({}))
        .await
        .unwrap();
    let t = Utc::now();
    let fact_id = g
        .insert_fact(FactInsert::new("bob", "knows", t).object_value("alice"))
        .await
        .unwrap();
    let embedding: Vec<f32> = vec![0.2_f32; 384];
    // First backfill
    g.backfill_fact_embedding(fact_id, &embedding)
        .await
        .unwrap();
    // Second backfill on same fact must not error
    g.backfill_fact_embedding(fact_id, &embedding)
        .await
        .unwrap();
}

/// FU.1: N concurrent callers racing to insert_fact with the same content must
/// produce exactly 1 Ok(id) and N-1 Err(Duplicate). Only 1 row must exist.
///
/// AC from Story #209 / FU.1: insert_fact_concurrent_dedup_exactly_one_wins.
#[tokio::test]
async fn insert_fact_concurrent_dedup_exactly_one_wins() {
    use std::sync::Arc;
    use tokio::sync::Barrier;

    const N: usize = 4;

    let g = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
    g.insert_entity("alice", 0, serde_json::json!({}))
        .await
        .unwrap();

    let barrier = Arc::new(Barrier::new(N));
    let valid_from = Utc::now();

    let handles: Vec<_> = (0..N)
        .map(|_| {
            let g = Arc::clone(&g);
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                // All tasks reach the barrier before any begins — maximises race window.
                barrier.wait().await;
                g.insert_fact(
                    FactInsert::new("alice", "concurrent_pred", valid_from)
                        .object_value("same_value"),
                )
                .await
            })
        })
        .collect();

    let mut ok_count = 0usize;
    let mut dup_count = 0usize;
    for h in handles {
        match h.await.expect("task panicked") {
            Ok(_) => ok_count += 1,
            Err(crate::core::error::Error::Duplicate { .. }) => dup_count += 1,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    assert_eq!(ok_count, 1, "exactly 1 insert must succeed");
    assert_eq!(
        dup_count,
        N - 1,
        "all other N-1 callers must return Duplicate"
    );

    // Verify exactly 1 row with this content_hash in facts
    let mut rows = g
        .conn
        .query(
            "SELECT COUNT(*) FROM facts WHERE predicate = 'concurrent_pred' AND expired_at IS NULL",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let count: i64 = row.get(0).unwrap();
    assert_eq!(
        count, 1,
        "exactly 1 row must exist in facts after concurrent inserts"
    );
}

// === forget_entity — cascade delete atomicity (Story #216) ===

#[tokio::test]
async fn forget_entity_removes_entity_facts_and_edges() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Insert entity + fact + episodic edge referencing it.
    g.insert_entity("alice", 0, serde_json::json!({}))
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
        .insert_episode("test episode", now, None, None)
        .await
        .unwrap();
    g.insert_episodic_edge(ep_id, "alice", None, "subject")
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

// === batch_forget — 100-item chunk deletion (Story #217) ===

#[tokio::test]
async fn batch_forget_250_entities_deleted_cleanly() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    // Insert 250 entities.
    let ids: Vec<String> = (0..250).map(|i| format!("ent-{i:04}")).collect();
    for id in &ids {
        g.insert_entity(id, 0, serde_json::json!({})).await.unwrap();
    }

    // Verify setup.
    let before = g.list_entities().await.unwrap();
    assert_eq!(before.len(), 250);

    let deleted = g.batch_forget(&ids).await.unwrap();
    assert_eq!(deleted, 250, "batch_forget must delete all 250 entities");

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
    assert_eq!(deleted, 0);
}

#[tokio::test]
async fn upsert_entity_with_group_inserts_then_updates() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = format!("{}/test.db", tmp.path().display());
    let g = TemporalGraph::open(&db).await.expect("open");

    // First call inserts (entity_type_id=0 = catch-all).
    g.upsert_entity_with_group("alice", 0, serde_json::json!({"context": "v1"}), None)
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
    g.upsert_entity_with_group("alice", 1, serde_json::json!({"context": "v2"}), None)
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

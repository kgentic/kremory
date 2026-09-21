use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === New E1.S3 Tests ===

#[tokio::test]
async fn test_insert_episodic_edge() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let ep_id = g
        .insert_episode(InsertEpisodeParams {
            content: "Alice joined.",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .unwrap();
    let edge_id = g
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id: ep_id,
            entity_id: "alice",
            entity_group_id: None,
            role: "mentioned",
        })
        .await
        .unwrap();
    assert!(edge_id > 0);

    let edges = g.episodic_edges_for_entity("alice").await.unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].entity_id, "alice");
    assert_eq!(edges[0].role, "mentioned");
    assert_eq!(edges[0].episode_id, ep_id);
}

/// Migration 017 presence-uniqueness: a second presence edge for the same
/// (episode_id, entity_id, entity_group_id) is suppressed by `INSERT OR IGNORE`
/// — even with a different `role` (role is unread metadata, NOT part of the
/// presence key). First-writer-wins: the surviving row is the first insert, and
/// the suppressed insert returns that existing edge id rather than a new row.
#[tokio::test]
async fn episodic_edge_presence_unique_suppresses_duplicate() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let ep_id = g
        .insert_episode(InsertEpisodeParams {
            content: "Alice joined.",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .unwrap();
    let id1 = g
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id: ep_id,
            entity_id: "alice",
            entity_group_id: None,
            role: "mention",
        })
        .await
        .unwrap();
    // Same (episode, entity), DIFFERENT role → must be suppressed, not a 2nd row.
    let id2 = g
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id: ep_id,
            entity_id: "alice",
            entity_group_id: None,
            role: "object",
        })
        .await
        .unwrap();
    assert_eq!(
        id1, id2,
        "suppressed duplicate must return the existing edge id, not a new one"
    );
    let edges = g.episodic_edges_for_entity("alice").await.unwrap();
    assert_eq!(
        edges.len(),
        1,
        "presence-uniqueness: at most one edge per (episode, entity); got {edges:?}"
    );
    assert_eq!(
        edges[0].role, "mention",
        "first-writer-wins: the original mention edge survives"
    );
}

#[tokio::test]
async fn test_invalidate_fact_with_reason() {
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
    let expired = Utc::now() - Duration::days(1);
    let invalid = Utc::now() - Duration::hours(12);
    g.invalidate_fact_with_reason(InvalidateFactWithReasonParams {
        fact_id: id,
        expired_at: expired,
        invalid_at: invalid,
    })
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
    g.insert_fact(FactInsert::new("alice", "has_title", t0).object_value("Engineer"))
        .await
        .unwrap();

    let works_at_facts = g
        .get_facts_by_subject_predicate(GetFactsBySubjectPredicateParams {
            subject_id: "alice",
            predicate: "works_at",
            group_id: "default",
        })
        .await
        .unwrap();
    assert_eq!(works_at_facts.len(), 1);
    assert_eq!(works_at_facts[0].object_id.as_deref(), Some("acme"));

    let title_facts = g
        .get_facts_by_subject_predicate(GetFactsBySubjectPredicateParams {
            subject_id: "alice",
            predicate: "has_title",
            group_id: "default",
        })
        .await
        .unwrap();
    assert_eq!(title_facts.len(), 1);
    assert_eq!(title_facts[0].object_value.as_deref(), Some("Engineer"));
}

/// TD-254: `get_facts_by_subject_predicate` must be scoped to `group_id` —
/// `subject_id` alone is not unique across namespaces (entities use a
/// composite `(id, group_id)` primary key). Before the fix this query had no
/// `group_id` filter at all, so a same-named subject/predicate in an
/// UNRELATED namespace would leak into the caller's candidate pool.
#[tokio::test]
async fn test_get_facts_by_subject_predicate_is_namespace_scoped() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    for ns in ["tenant_a", "tenant_b"] {
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(ns),
        })
        .await
        .unwrap();
    }
    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "acme",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some("tenant_a"),
    })
    .await
    .unwrap();
    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "globex",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some("tenant_b"),
    })
    .await
    .unwrap();
    let t0 = Utc::now() - Duration::hours(1);
    // Same subject + predicate name in BOTH namespaces — a real-world name
    // collision (e.g. "alice"/"status") across two unrelated tenants.
    // DIFFERENT objects deliberately: `content_hash` (subject+predicate+
    // object, no group_id — a separate, already-known, deliberately-deferred
    // design tension pinned by `test_try_insert_fact_with_group_dedups_cross_variant`'s
    // "Path X caller-wins" semantics) would otherwise reject the second
    // insert as a duplicate of the first, which is NOT what this test is
    // about — this test is scoped to the READ-PATH candidate-pool leak only.
    g.insert_fact_with_group(
        FactInsert::new("alice", "works_at", t0).object_id("acme"),
        Some("tenant_a"),
    )
    .await
    .unwrap();
    g.insert_fact_with_group(
        FactInsert::new("alice", "works_at", t0).object_id("globex"),
        Some("tenant_b"),
    )
    .await
    .unwrap();

    let tenant_a_facts = g
        .get_facts_by_subject_predicate(GetFactsBySubjectPredicateParams {
            subject_id: "alice",
            predicate: "works_at",
            group_id: "tenant_a",
        })
        .await
        .unwrap();
    assert_eq!(
        tenant_a_facts.len(),
        1,
        "must see only tenant_a's own fact, not tenant_b's: {tenant_a_facts:?}"
    );
    assert_eq!(
        tenant_a_facts[0].object_id.as_deref(),
        Some("acme"),
        "must be tenant_a's own object, not tenant_b's leaked in: {tenant_a_facts:?}"
    );
    assert_eq!(
        tenant_a_facts[0].group_id.as_deref(),
        Some("tenant_a"),
        "the returned fact must actually belong to tenant_a"
    );

    let tenant_b_facts = g
        .get_facts_by_subject_predicate(GetFactsBySubjectPredicateParams {
            subject_id: "alice",
            predicate: "works_at",
            group_id: "tenant_b",
        })
        .await
        .unwrap();
    assert_eq!(
        tenant_b_facts.len(),
        1,
        "must see only tenant_b's own fact, not tenant_a's: {tenant_b_facts:?}"
    );
    assert_eq!(
        tenant_b_facts[0].object_id.as_deref(),
        Some("globex"),
        "must be tenant_b's own object, not tenant_a's leaked in: {tenant_b_facts:?}"
    );
}

#[tokio::test]
async fn test_insert_entity_with_group() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({"name": "Alice"}),
        group_id: Some("group-abc"),
    })
    .await
    .unwrap();

    let entity = g.get_entity("alice").await.unwrap().unwrap();
    assert_eq!(entity.id, "alice");
    assert_eq!(entity.group_id.as_deref(), Some("group-abc"));
}

/// Lane A indexer call sites pass `group_id=None` (folder_id absent pre-Lane-B).
/// None maps to 'default' (entities.group_id is NOT NULL). The entity is
/// stored in the default namespace and is visible under unscoped queries or
/// queries scoped to 'default'.
#[tokio::test]
async fn lane_a_indexer_writes_null_group_id_when_folder_id_absent() {
    let g = TemporalGraph::open_in_memory().await.unwrap();

    let group_id: Option<&str> = None;
    g.insert_entity_with_group(InsertEntityWithGroupParams { id: "doc:welcome:chunk_0", entity_type_id: 0, properties: serde_json::json!({ "text": "The welcome document introduces the product.", "source": "doc:welcome", "source_type": "document", }), group_id })
    .await
    .unwrap();

    let entity = g.get_entity("doc:welcome:chunk_0").await.unwrap().unwrap();
    // None maps to 'default' — not NULL (NOT NULL constraint enforced).
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
    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "doc_chunk_0",
        entity_type_id: 0,
        properties: serde_json::json!({"text": "original content"}),
        group_id: Some("group-a"),
    })
    .await
    .unwrap();

    let entity = g.get_entity("doc_chunk_0").await.unwrap().unwrap();
    assert_eq!(entity.group_id.as_deref(), Some("group-a"));

    // Re-scope to group-b with updated properties
    g.update_entity_group(UpdateEntityGroupParams {
        id: "doc_chunk_0",
        group_id: Some("group-b"),
        properties: serde_json::json!({"text": "updated content"}),
    })
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
    // entities.group_id is NOT NULL post-migration-004.
    // Passing None to update_entity_group maps to 'default' (not NULL).
    let g = TemporalGraph::open_in_memory().await.unwrap();

    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "e1",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some("scoped"),
    })
    .await
    .unwrap();

    g.update_entity_group(UpdateEntityGroupParams {
        id: "e1",
        group_id: None,
        properties: serde_json::json!({}),
    })
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


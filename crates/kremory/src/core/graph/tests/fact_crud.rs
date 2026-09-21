use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === Fact CRUD ===

#[tokio::test]
async fn test_insert_fact_returns_id() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let id = g
        .insert_fact(FactInsert::new("alice", "exists", Utc::now()))
        .await
        .unwrap();
    assert!(id > 0);
}

/// `try_insert_fact` returns `Ok(Some(id))` on fresh insert.
#[tokio::test]
async fn test_try_insert_fact_returns_some_on_fresh_insert() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    let id = g
        .try_insert_fact(FactInsert::new("alice", "exists", Utc::now()))
        .await
        .unwrap();
    assert!(id.is_some(), "fresh insert must return Some(id)");
    assert!(id.unwrap() > 0);
}

/// `try_insert_fact` returns `Ok(None)` on content_hash collision.
#[tokio::test]
async fn test_try_insert_fact_returns_none_on_duplicate() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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

/// `try_insert_fact_with_group` returns `Ok(Some(id))` on fresh insert.
#[tokio::test]
async fn test_try_insert_fact_with_group_returns_some_on_fresh_insert() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    // The facts table has a COMPOSITE FK (subject_id, subject_group_id) → entities(id,
    // group_id) (schema.rs:1450). A fact in group "g1" must reference an entity that
    // exists in "g1", so the subject is created in that same namespace.
    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some("g1"),
    })
    .await
    .unwrap();
    let id = g
        .try_insert_fact_with_group(FactInsert::new("alice", "exists", Utc::now()), Some("g1"))
        .await
        .unwrap();
    assert!(id.is_some(), "fresh insert must return Some(id)");
}

/// `content_hash` does NOT include `group_id` — same triple in different
/// groups still collides. This test pins the invariant that caller's
/// `insert_fact_with_group(group=X)` and engine's `insert_fact(no group)` will
/// dedup against each other (the basis of Path X caller-wins semantics).
#[tokio::test]
async fn test_try_insert_fact_with_group_dedups_cross_variant() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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

/// Re-asserting a triple that was superseded/expired must SUCCEED. The dedup
/// pre-check SELECT excludes expired rows (`expired_at IS NULL`) but the
/// UNIQUE index previously covered ALL rows, so a legitimate bi-temporal
/// assert→expire→re-assert collided on the stale expired row's `content_hash`
/// → `UNIQUE constraint failed` → the re-assertion was silently lost (21 such
/// `unique_violation` drops measured on an instrumented corpus run).
/// `migrate_025_fact_dedup_expired_partial` scopes the index to ACTIVE rows,
/// matching the SELECT predicate, so an expired row no longer blocks
/// re-assertion.
#[tokio::test]
async fn test_reassert_expired_fact_succeeds_td133_b2() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();

    // Assert the triple.
    let t0 = Utc::now() - Duration::days(2);
    let id1 = g
        .insert_fact(FactInsert::new("alice", "has_title", t0).object_value("PM"))
        .await
        .unwrap();
    assert!(id1 > 0);

    // Supersede it (bi-temporal expiry).
    g.invalidate_fact(id1, Utc::now() - Duration::days(1))
        .await
        .unwrap();

    // Re-assert the SAME triple — identical content_hash to the now-expired row.
    // Pre-fix this errored `UNIQUE constraint failed` (the global index still
    // held the expired row's hash); post-fix (active-only partial index) it
    // succeeds as a new active fact.
    let id2 = g
        .insert_fact(FactInsert::new("alice", "has_title", Utc::now()).object_value("PM"))
        .await
        .expect("re-asserting a superseded triple must succeed (TD-133 B2)");
    assert!(id2 > 0);
    assert_ne!(
        id1, id2,
        "re-assertion must be a new row, not the expired one"
    );

    // Exactly one ACTIVE fact now (the re-assertion); the expired one excluded.
    let facts = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
    assert_eq!(
        facts.len(),
        1,
        "only the re-asserted active fact should be live"
    );
    assert_eq!(facts[0].predicate, "has_title");
}

#[tokio::test]
async fn test_insert_fact_with_object_id() {
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
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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


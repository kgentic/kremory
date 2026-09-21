use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === get_neighbours_at ===

/// `as_of: None` must be functionally identical to `get_neighbours` (the SQL
/// text is copy-identical by construction — see `get_neighbours_at`'s own doc
/// comment; this pins the OBSERVABLE behaviour: same entities, same facts).
#[tokio::test]
async fn test_get_neighbours_at_none_matches_get_neighbours() {
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

    let via_original = g.get_neighbours("alice", 1).await.unwrap();
    let via_new = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: None,
            max_visited: None,
        })
        .await
        .unwrap();

    assert_eq!(via_original.facts.len(), via_new.facts.len());
    assert_eq!(via_original.entities.len(), via_new.entities.len());
    let mut orig_ids: Vec<&str> = via_original
        .entities
        .iter()
        .map(|e| e.id.as_str())
        .collect();
    let mut new_ids: Vec<&str> = via_new.entities.iter().map(|e| e.id.as_str()).collect();
    orig_ids.sort();
    new_ids.sort();
    assert_eq!(orig_ids, new_ids);
}

/// `as_of(t)` BEFORE a fact's `valid_from` excludes it
/// (and the neighbour reached only via that fact is unreachable).
#[tokio::test]
async fn test_get_neighbours_at_before_window_excludes_fact() {
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
    let valid_from = Utc::now() - Duration::days(5);
    let valid_to = Utc::now() - Duration::days(1);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", valid_from).object_id("acme"))
        .await
        .unwrap();
    g.bound_valid_to(id, valid_to).await.unwrap();

    let as_of = valid_from - Duration::days(1);
    let subgraph = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: Some(as_of),
            max_visited: None,
        })
        .await
        .unwrap();
    assert_eq!(
        subgraph.facts.len(),
        0,
        "fact before valid_from must be excluded"
    );
    // Only the seed itself is reachable — acme is unreachable without the fact.
    assert_eq!(subgraph.entities.len(), 1);
    assert_eq!(subgraph.entities[0].id, "alice");
}

/// `as_of(t)` INSIDE `[valid_from, valid_to)` includes
/// the fact (and reaches the neighbour through it).
#[tokio::test]
async fn test_get_neighbours_at_inside_window_includes_fact() {
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
    let valid_from = Utc::now() - Duration::days(5);
    let valid_to = Utc::now() - Duration::days(1);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", valid_from).object_id("acme"))
        .await
        .unwrap();
    g.bound_valid_to(id, valid_to).await.unwrap();

    let as_of = valid_from + Duration::days(1);
    let subgraph = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: Some(as_of),
            max_visited: None,
        })
        .await
        .unwrap();
    assert_eq!(
        subgraph.facts.len(),
        1,
        "fact inside window must be included"
    );
    let mut entity_ids: Vec<&str> = subgraph.entities.iter().map(|e| e.id.as_str()).collect();
    entity_ids.sort();
    assert_eq!(entity_ids, vec!["acme", "alice"]);
}

/// `as_of(t)` AT `valid_to` excludes the fact: the
/// predicate is `valid_to > ?t` (strictly greater), a half-open window
/// `[valid_from, valid_to)`.
#[tokio::test]
async fn test_get_neighbours_at_at_boundary_valid_to_excludes_fact() {
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
    let valid_from = Utc::now() - Duration::days(5);
    let valid_to = Utc::now() - Duration::days(1);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", valid_from).object_id("acme"))
        .await
        .unwrap();
    g.bound_valid_to(id, valid_to).await.unwrap();

    let subgraph = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: Some(valid_to),
            max_visited: None,
        })
        .await
        .unwrap();
    assert_eq!(
        subgraph.facts.len(),
        0,
        "fact exactly at valid_to must be excluded (half-open window)"
    );
}

/// `as_of(t)` AFTER `valid_to` excludes the fact.
#[tokio::test]
async fn test_get_neighbours_at_after_window_excludes_fact() {
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
    let valid_from = Utc::now() - Duration::days(5);
    let valid_to = Utc::now() - Duration::days(1);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", valid_from).object_id("acme"))
        .await
        .unwrap();
    g.bound_valid_to(id, valid_to).await.unwrap();

    let as_of = Utc::now();
    let subgraph = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: Some(as_of),
            max_visited: None,
        })
        .await
        .unwrap();
    assert_eq!(
        subgraph.facts.len(),
        0,
        "fact after valid_to must be excluded"
    );
}

/// A fact flagged `invalid_at` by the contradiction resolver, but still
/// valid-time-in-window at `t`, MUST still be included: `as_of` filters on
/// `valid_from`/`valid_to` ONLY, deliberately never on `invalid_at`. Stamped
/// via raw SQL against `g.conn`
/// (`pub` specifically for test use, per its own doc comment) because no
/// public primitive sets `invalid_at` without ALSO setting `expired_at`
/// (`invalidate_fact_with_reason` stamps both together) — which would
/// confound this assertion via the pre-existing, unconditional
/// `expired_at IS NULL` clause instead of exercising the `invalid_at`
/// exclusion this test is meant to lock.
#[tokio::test]
async fn test_get_neighbours_at_invalid_at_flagged_fact_still_included() {
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
    let valid_from = Utc::now() - Duration::days(5);
    let id = g
        .insert_fact(FactInsert::new("alice", "works_at", valid_from).object_id("acme"))
        .await
        .unwrap();

    // Stamp invalid_at WITHOUT touching expired_at (test-only raw SQL).
    g.conn
        .execute(
            "UPDATE facts SET invalid_at = ?1 WHERE id = ?2",
            libsql::params![Utc::now().to_rfc3339(), id],
        )
        .await
        .unwrap();

    let as_of = Utc::now();
    let subgraph = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "alice",
            hops: 1,
            as_of: Some(as_of),
            max_visited: None,
        })
        .await
        .unwrap();
    assert_eq!(
        subgraph.facts.len(),
        1,
        "invalid_at-flagged-but-valid-time-in-window fact must still be included (ADR-068 Decision 1)"
    );
}


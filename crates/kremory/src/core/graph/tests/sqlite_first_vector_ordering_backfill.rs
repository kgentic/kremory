use super::super::*;
use crate::core::schema::TemporalGraph;

// === SQLite-first vector ordering backfill ===

/// FU.8: facts_missing_embeddings returns (id, subject_id, predicate,
/// object_value, object_id) — the 5-tuple now includes object_id so callers
/// can prefer the entity reference over the literal string.
#[tokio::test]
async fn facts_missing_embeddings_returns_all_null_embedding_facts() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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
    g.insert_entity(InsertEntityParams {
        id: "bob",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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

/// N concurrent callers racing to insert_fact with the same content must
/// produce exactly 1 Ok(id) and N-1 Err(Duplicate). Only 1 row must exist.
#[tokio::test]
async fn insert_fact_concurrent_dedup_exactly_one_wins() {
    use std::sync::Arc;
    use tokio::sync::Barrier;

    const N: usize = 4;

    let g = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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


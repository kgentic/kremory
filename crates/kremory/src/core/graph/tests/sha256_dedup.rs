use super::super::*;
use crate::core::schema::TemporalGraph;

// === SHA-256 dedup ===

#[tokio::test]
async fn insert_fact_duplicate_returns_error() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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
    g.insert_entity(InsertEntityParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
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
    let h1 = fact_content_hash(FactContentHashParams {
        subject_id: "alice",
        predicate: "likes",
        object_id: None,
        object_value: Some("coffee"),
    });
    let h2 = fact_content_hash(FactContentHashParams {
        subject_id: "alice",
        predicate: "likes",
        object_id: None,
        object_value: Some("coffee"),
    });
    assert_eq!(h1, h2, "hash must be deterministic");
    let h3 = fact_content_hash(FactContentHashParams {
        subject_id: "alice",
        predicate: "likes",
        object_id: None,
        object_value: Some("tea"),
    });
    assert_ne!(h1, h3, "different facts must have different hashes");
    // SHA-256 produces 64 hex chars
    assert_eq!(h1.len(), 64);
}


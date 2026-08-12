#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-211 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-211) —
//! `Memory::backfill_entity_embeddings` public-API-only proofs. Mirrors
//! `td112_reembed_all_entity_embeddings.rs`'s shape (same DB-per-test /
//! embedder-provider helpers) but proves the OPPOSITE-of-full-reembed
//! contract: a NULL-only gap-fill must
//!   1. return exactly the NULL-embedding rows and no others,
//!   2. be idempotent across repeated runs,
//!   3. NEVER touch a row that already carries an embedding, and
//!   4. respect namespace scoping (TD-206) when filling the gap.
//!
//! Entities are seeded directly via `TemporalGraph::insert_entity_with_group`
//! (reachable through `Memory::temporal_graph_for_test`, `test-utils`
//! feature) — same rationale as the TD-112 sibling: keeps each test scoped to
//! `backfill_entity_embeddings` itself.

use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::search::{SearchFilters, VectorSearchEntitiesParams};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td211_backfill_entities_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

/// Deterministic 384-dim embedding derived from `seed` — same shape as the
/// TD-112 sibling's `seeded_embedding`.
fn seeded_embedding(seed: f32) -> Vec<f32> {
    (0..384)
        .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
        .collect()
}

/// Fixed-output embedder (every call returns the same seed-1.0 vector) — the
/// backfill path only needs "some deterministic vector got written", not a
/// seed switch (that's the TD-112 overwrite test's job, not this one's).
struct FixedEmbeddingProvider;

impl kremory::core::provider::EmbeddingProvider for FixedEmbeddingProvider {
    async fn embed<'a>(&'a self, _text: &'a str) -> kremory::CoreResult<Vec<f32>> {
        Ok(seeded_embedding(1.0))
    }
}

// ── Contract 1+2: exact NULL-only set, idempotent across repeated runs ─────

#[tokio::test]
async fn backfill_entity_embeddings_fills_exactly_the_null_rows_and_is_idempotent() {
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(FixedEmbeddingProvider);
    let mem = Memory::open(unique_db("exact_and_idempotent"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    for i in 0..3 {
        let id = format!("gap-entity-{i}");
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id: &id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": format!("Gap Entity {i}") }),
            group_id: None,
        })
        .await
        .expect("insert entity");
    }

    let first = mem
        .backfill_entity_embeddings(2) // batch_size smaller than corpus — forces multiple pages
        .await
        .expect("first backfill_entity_embeddings run must succeed");
    assert_eq!(
        first.embedded, 3,
        "all three NULL-embedding entities must be backfilled"
    );
    assert_eq!(first.failed, 0);

    // Contract 2: a second run over the now-fully-embedded corpus must find
    // NOTHING left to fill — the defining difference from a full re-embed
    // (whose TD-112 sibling test asserts the OPPOSITE: a stable non-zero
    // tally on every run).
    let second = mem
        .backfill_entity_embeddings(2)
        .await
        .expect("second backfill_entity_embeddings run must succeed");
    assert_eq!(
        second.embedded, 0,
        "a second run must find no remaining NULL-embedding entities — this \
         is a gap-fill, not a full re-embed, so the tally must be ZERO once \
         the gap is closed"
    );
    assert_eq!(second.failed, 0);
}

// ── Contract 3: an already-embedded row is never touched ───────────────────

#[tokio::test]
async fn backfill_entity_embeddings_never_touches_an_already_embedded_row() {
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(FixedEmbeddingProvider);
    let mem = Memory::open(unique_db("never_touches_existing"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    // "seeded" already carries an embedding at seed 9.0 — must be left
    // completely alone by the gap-fill.
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "seeded",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Seeded" }),
        group_id: None,
    })
    .await
    .expect("insert entity");
    tg.set_entity_embedding("seeded", &seeded_embedding(9.0))
        .await
        .expect("seed pre-existing embedding");

    // "gap" starts NULL.
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "gap",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Gap" }),
        group_id: None,
    })
    .await
    .expect("insert entity");

    let stats = mem
        .backfill_entity_embeddings(256)
        .await
        .expect("backfill_entity_embeddings must succeed");
    assert_eq!(
        stats.embedded, 1,
        "only the ONE NULL-embedding entity (\"gap\") must be backfilled"
    );
    assert_eq!(stats.failed, 0);

    // Prove "seeded"'s vector is byte-for-byte the seed-9.0 vector, not the
    // FixedEmbeddingProvider's seed-1.0 output — an exact match against
    // seed-9.0 must score ~0 (score = -cosine_distance).
    let hits = tg
        .vector_search_entities(VectorSearchEntitiesParams {
            query_embedding: &seeded_embedding(9.0),
            limit: 10,
            filters: &SearchFilters::new(),
        })
        .await
        .expect("vector search must succeed");
    let seeded_hit = hits
        .iter()
        .find(|h| h.item.id == "seeded")
        .expect("seeded entity must be searchable");
    assert!(
        seeded_hit.score > -0.001,
        "\"seeded\"'s pre-existing embedding must be UNTOUCHED by the \
         gap-fill run — querying with the exact seed-9.0 vector it was \
         seeded with must still score it as an exact match, got score={}",
        seeded_hit.score
    );
}

// ── Contract 4: namespace scoping (TD-206) ──────────────────────────────────

#[tokio::test]
async fn backfill_entity_embeddings_respects_namespace_scoping() {
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(FixedEmbeddingProvider);
    let mem = Memory::open(unique_db("namespace_scoped"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    // Same surface name ("alice") in two namespaces — a legitimate
    // cross-namespace id collision per ADR-029d. Both start NULL.
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Alice A" }),
        group_id: Some("namespace-a"),
    })
    .await
    .expect("insert entity in namespace-a");
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Alice B" }),
        group_id: Some("namespace-b"),
    })
    .await
    .expect("insert entity in namespace-b");

    let stats = mem
        .backfill_entity_embeddings(256)
        .await
        .expect("backfill_entity_embeddings must succeed");
    assert_eq!(
        stats.embedded, 2,
        "both cross-namespace rows sharing id=\"alice\" must be backfilled \
         independently — a bare id-only write would collapse them into one"
    );
    assert_eq!(stats.failed, 0);

    let missing = tg
        .entities_missing_embeddings(100)
        .await
        .expect("entities_missing_embeddings must succeed");
    let missing_group_ids: Vec<&str> = missing.iter().map(|r| r.group_id.as_str()).collect();
    assert!(
        missing.is_empty(),
        "both namespace rows must now have an embedding; still missing \
         group_ids: {missing_group_ids:?}"
    );
}

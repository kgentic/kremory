#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-211 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-211) —
//! `Memory::backfill_fact_embeddings` public-API-only proofs. Gives the
//! pre-existing Story #214 crash-recovery primitives
//! (`TemporalGraph::facts_missing_embeddings`,
//! `TemporalGraph::backfill_fact_embedding`) their FIRST production caller —
//! before this method existed, both were reachable only from their own
//! definitions and from `graph/tests.rs` (register §TD-211).
//!
//! Mirrors `td211_backfill_entity_embeddings.rs`'s four-contract shape:
//!   1. return exactly the NULL-embedding rows and no others,
//!   2. idempotent across repeated runs,
//!   3. NEVER touch a row that already carries an embedding,
//!   4. namespace scoping is inherited for free here (facts are
//!      subject/object-scoped, not independently namespaced), so this file
//!      instead proves the documented judgement call: fact text is built
//!      from the raw subject_id/object_id (entity slugs), not a
//!      `properties.name` lookup — see `backfill_and_store_fact_page`'s doc
//!      comment in `facade/mod.rs`.

use kremory::core::graph::{FactInsert, InsertEntityWithGroupParams};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td211_backfill_facts_{}_{}_{}.db",
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

fn seeded_embedding(seed: f32) -> Vec<f32> {
    (0..384)
        .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
        .collect()
}

struct FixedEmbeddingProvider;

impl kremory::core::provider::EmbeddingProvider for FixedEmbeddingProvider {
    async fn embed<'a>(&'a self, _text: &'a str) -> kremory::CoreResult<Vec<f32>> {
        Ok(seeded_embedding(1.0))
    }
}

/// Records every raw string handed to `.embed()` — same shape as the TD-112
/// sibling's `RecordingEmbeddingProvider`.
struct RecordingEmbeddingProvider {
    dim: usize,
    texts: Mutex<Vec<String>>,
}

impl RecordingEmbeddingProvider {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            texts: Mutex::new(Vec::new()),
        }
    }

    fn texts(&self) -> Vec<String> {
        self.texts.lock().expect("recording mutex poisoned").clone()
    }
}

impl kremory::core::provider::EmbeddingProvider for RecordingEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        self.texts
            .lock()
            .expect("recording mutex poisoned")
            .push(text.to_string());
        let dim = self.dim;
        async move { Ok(vec![1.0_f32; dim]) }
    }
}

// ── Contract 1+2: exact NULL-only set, idempotent across repeated runs ─────

#[tokio::test]
async fn backfill_fact_embeddings_fills_exactly_the_null_rows_and_is_idempotent() {
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(FixedEmbeddingProvider);
    let mem = Memory::open(unique_db("exact_and_idempotent"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Alice" }),
        group_id: None,
    })
    .await
    .expect("insert subject entity");
    for i in 0..3 {
        tg.insert_fact_with_group(
            FactInsert {
                subject_id: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some(&format!("thing-{i}")),
                valid_from: chrono::Utc::now(),
                confidence: 1.0,
                source_episode_id: None,
                embedding: None,
            },
            Some("default"),
        )
        .await
        .expect("insert fact");
    }

    let first = mem
        .backfill_fact_embeddings(2) // batch_size smaller than corpus — forces multiple chunks
        .await
        .expect("first backfill_fact_embeddings run must succeed");
    assert_eq!(
        first.embedded, 3,
        "all three NULL-embedding facts must be backfilled"
    );
    assert_eq!(first.failed, 0);

    let second = mem
        .backfill_fact_embeddings(2)
        .await
        .expect("second backfill_fact_embeddings run must succeed");
    assert_eq!(
        second.embedded, 0,
        "a second run must find no remaining NULL-embedding facts — this is \
         a gap-fill (drives facts_missing_embeddings), not a full re-embed, \
         so the tally must be ZERO once the gap is closed"
    );
    assert_eq!(second.failed, 0);
}

// ── Contract 3: an already-embedded row is never touched ───────────────────

#[tokio::test]
async fn backfill_fact_embeddings_never_touches_an_already_embedded_row() {
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(FixedEmbeddingProvider);
    let mem = Memory::open(unique_db("never_touches_existing"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "bob",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Bob" }),
        group_id: None,
    })
    .await
    .expect("insert subject entity");

    // "seeded" fact already carries an embedding — must be left alone.
    let seeded_fact_id = tg
        .insert_fact_with_group(
            FactInsert {
                subject_id: "bob",
                predicate: "owns",
                object_id: None,
                object_value: Some("a seeded fact"),
                valid_from: chrono::Utc::now(),
                confidence: 1.0,
                source_episode_id: None,
                embedding: None,
            },
            Some("default"),
        )
        .await
        .expect("insert seeded fact");
    tg.set_fact_embedding(seeded_fact_id, &seeded_embedding(9.0))
        .await
        .expect("seed pre-existing embedding");

    // "gap" fact starts NULL.
    let gap_fact_id = tg
        .insert_fact_with_group(
            FactInsert {
                subject_id: "bob",
                predicate: "owns",
                object_id: None,
                object_value: Some("a gap fact"),
                valid_from: chrono::Utc::now(),
                confidence: 1.0,
                source_episode_id: None,
                embedding: None,
            },
            Some("default"),
        )
        .await
        .expect("insert gap fact");

    let stats = mem
        .backfill_fact_embeddings(256)
        .await
        .expect("backfill_fact_embeddings must succeed");
    assert_eq!(
        stats.embedded, 1,
        "only the ONE NULL-embedding fact (\"gap fact\") must be backfilled"
    );
    assert_eq!(stats.failed, 0);

    // Prove the seeded fact's vector is unchanged — exact match against
    // seed-9.0 must score ~0.
    let hits = tg
        .vector_search_facts(kremory::core::search::VectorSearchFactsParams {
            query_embedding: &seeded_embedding(9.0),
            limit: 10,
            filters: &kremory::core::search::SearchFilters::new(),
        })
        .await
        .expect("vector search must succeed");
    let seeded_hit = hits
        .iter()
        .find(|h| h.item.id == seeded_fact_id)
        .expect("seeded fact must be searchable");
    assert!(
        seeded_hit.score > -0.001,
        "the seeded fact's pre-existing embedding must be UNTOUCHED by the \
         gap-fill run — querying with the exact seed-9.0 vector it was \
         seeded with must still score it as an exact match, got score={}",
        seeded_hit.score
    );

    // Sanity: the gap fact id really was the one embedded (not accidentally
    // the seeded one) — a second explicit query on seed-1.0 (the
    // FixedEmbeddingProvider's constant output) must surface the gap fact.
    let hits_seed1 = tg
        .vector_search_facts(kremory::core::search::VectorSearchFactsParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &kremory::core::search::SearchFilters::new(),
        })
        .await
        .expect("vector search must succeed");
    assert!(
        hits_seed1.iter().any(|h| h.item.id == gap_fact_id),
        "the gap fact must be dense-searchable after backfill"
    );
}

// ── Text-construction judgement call: raw entity id/slug, not display name ─

#[tokio::test]
async fn backfill_fact_embeddings_builds_text_from_raw_entity_ids() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(384));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("raw_id_text"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");

    // Subject entity's display name (`properties.name`) DIFFERS from its
    // id/slug — proves the documented judgement call: `backfill_fact_embeddings`
    // uses the raw `subject_id`/`object_id`, NOT a `properties.name` lookup
    // (facts_missing_embeddings does not JOIN entities for display names, and
    // this method must not rewrite that primitive — see facade/mod.rs).
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "carol-the-slug",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Dr. Carol Danvers" }),
        group_id: None,
    })
    .await
    .expect("insert subject entity");
    tg.insert_fact_with_group(
        FactInsert {
            subject_id: "carol-the-slug",
            predicate: "likes",
            object_id: None,
            object_value: Some("reading"),
            valid_from: chrono::Utc::now(),
            confidence: 1.0,
            source_episode_id: None,
            embedding: None,
        },
        Some("default"),
    )
    .await
    .expect("insert fact");

    let stats = mem
        .backfill_fact_embeddings(256)
        .await
        .expect("backfill_fact_embeddings must succeed");
    assert_eq!(stats.embedded, 1);
    assert_eq!(stats.failed, 0);

    let texts = recording.texts();
    assert!(
        texts.iter().any(|t| t == "carol-the-slug likes reading"),
        "backfill_fact_embeddings must build the embed text from the raw \
         subject_id (entity slug), not a properties.name lookup; recorded \
         texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("Dr. Carol Danvers")),
        "the properties.name override must NOT appear — that is the \
         documented judgement call, not a bug; recorded texts: {texts:?}"
    );
}

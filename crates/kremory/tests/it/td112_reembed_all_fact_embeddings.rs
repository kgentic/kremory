#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-112 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-112) —
//! `Memory::reembed_all_fact_embeddings` public-API-only proofs, mirroring
//! `td143_reembed_all_episode_embeddings.rs`'s exact three-test shape (see
//! `td112_reembed_all_entity_embeddings.rs`'s module doc for the full
//! rationale — this file is that same shape, one table over).
//!
//! Facts are seeded directly via `TemporalGraph::insert_entity_with_group` +
//! `insert_fact_with_group` (reachable through `Memory::temporal_graph_for_test`,
//! `test-utils` feature).

use kremory::core::graph::{FactInsert, InsertEntityWithGroupParams};
use kremory::core::search::{SearchFilters, VectorSearchFactsParams};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td112_reembed_facts_{}_{}_{}.db",
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

struct SwitchableEmbeddingProvider {
    use_second_seed: AtomicBool,
}

impl SwitchableEmbeddingProvider {
    fn new() -> Self {
        Self {
            use_second_seed: AtomicBool::new(false),
        }
    }

    fn switch_to_second_seed(&self) {
        self.use_second_seed.store(true, Ordering::SeqCst);
    }
}

impl kremory::core::provider::EmbeddingProvider for SwitchableEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        let seed = if self.use_second_seed.load(Ordering::SeqCst) {
            5.0
        } else {
            1.0
        };
        async move { Ok(seeded_embedding(seed)) }
    }
}

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

/// Seed one entity ("alice") + one literal-object fact
/// ("alice likes reading") directly via the graph layer — no LLM, no facade
/// ingest, so the test stays scoped to `reembed_all_fact_embeddings` alone.
async fn seed_one_fact(tg: &kremory::core::schema::TemporalGraph) -> i64 {
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Alice" }),
        group_id: None,
    })
    .await
    .expect("insert subject entity");
    tg.insert_fact_with_group(
        FactInsert {
            subject_id: "alice",
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
    .expect("insert fact")
}

// ── Overwrite: re-embedding an already-embedded fact moves the vector ──────

#[tokio::test]
async fn reembed_all_overwrites_an_existing_fact_embedding() {
    let embedder = Arc::new(SwitchableEmbeddingProvider::new());
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = embedder.clone();

    let mem = Memory::open(unique_db("overwrite"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .await
        .expect("build memory");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");
    let fact_id = seed_one_fact(tg).await;
    tg.set_fact_embedding(fact_id, &seeded_embedding(1.0))
        .await
        .expect("seed initial embedding");

    let no_filter = SearchFilters::new();
    let hits_before = tg
        .vector_search_facts(VectorSearchFactsParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(
        hits_before.len(),
        1,
        "the one fact must be dense-searchable before re-embed"
    );
    let dist_before_at_seed1 = hits_before[0].score;

    embedder.switch_to_second_seed();
    let stats = mem
        .reembed_all_fact_embeddings(256)
        .await
        .expect("reembed_all_fact_embeddings must succeed");
    assert_eq!(
        stats.embedded, 1,
        "the one existing fact must be re-embedded"
    );
    assert_eq!(stats.failed, 0);

    let hits_after = tg
        .vector_search_facts(VectorSearchFactsParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(hits_after.len(), 1);
    // NOTE: `vector_search_facts`' score is `-distance` ("matching FTS
    // convention" — higher = closer), the OPPOSITE sign convention from
    // `vector_search_episodes` (raw distance, lower = closer). So after the
    // stored vector moves away from the seed-1.0 query, the score must
    // DECREASE, not increase.
    assert!(
        hits_after[0].score < dist_before_at_seed1,
        "re-embedding must overwrite the stored vector — the score against a \
         seed-1.0 query must have DECREASED (score = -distance here) once the \
         stored vector moved to seed-5.0 (before={dist_before_at_seed1}, after={})",
        hits_after[0].score
    );

    let hits_seed5 = tg
        .vector_search_facts(VectorSearchFactsParams {
            query_embedding: &seeded_embedding(5.0),
            limit: 10,
            filters: &no_filter,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(hits_seed5.len(), 1);
    assert!(
        hits_seed5[0].score > hits_after[0].score,
        "a seed-5.0 query must be a MUCH closer (higher-scoring) match than a \
         seed-1.0 query to the freshly-stored seed-5.0 vector \
         (seed5_score={}, seed1_score={})",
        hits_seed5[0].score,
        hits_after[0].score
    );
}

// ── Prefix parity: TD-143 document-prefix applies on the fact re-embed path ─

#[tokio::test]
async fn reembed_all_facts_applies_document_prefix_when_enabled() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(384));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("prefix"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_embed_task_prefix_enabled(true)
        .await
        .expect("build memory with task prefix on");

    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");
    seed_one_fact(tg).await;

    recording.texts.lock().expect("mutex poisoned").clear();

    let stats = mem
        .reembed_all_fact_embeddings(256)
        .await
        .expect("reembed_all_fact_embeddings must succeed");
    assert_eq!(stats.embedded, 1);
    assert_eq!(stats.failed, 0);

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == "search_document: Alice likes reading"),
        "reembed_all_fact_embeddings must document-prefix the resolved \
         'subject predicate object' triple text it hands to the embedder \
         when embed_task_prefix_enabled is on; recorded texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("search_query: ")),
        "a re-embed WRITE call must never receive the query prefix; \
         recorded texts: {texts:?}"
    );
}

// ── Idempotency: running it twice is stable, no failures ────────────────────

#[tokio::test]
async fn reembed_all_facts_is_idempotent_across_two_runs() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(384));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("idempotent"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
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
        .reembed_all_fact_embeddings(2) // batch_size smaller than corpus — forces multiple pages
        .await
        .expect("first reembed_all_fact_embeddings run must succeed");
    assert_eq!(first.embedded, 3, "all three facts must be re-embedded");
    assert_eq!(first.failed, 0);

    let second = mem
        .reembed_all_fact_embeddings(2)
        .await
        .expect("second reembed_all_fact_embeddings run must succeed");
    assert_eq!(
        second.embedded, 3,
        "a second run over the SAME corpus must re-embed the SAME three \
         facts again — this is a full re-embed, not a gap-fill, so the \
         tally must be stable across runs, not zero"
    );
    assert_eq!(second.failed, 0, "no failures on either run");
}

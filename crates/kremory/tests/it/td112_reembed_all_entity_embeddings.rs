#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-112 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-112) —
//! `Memory::reembed_all_entity_embeddings` public-API-only proofs, mirroring
//! `td143_reembed_all_episode_embeddings.rs`'s exact three-test shape:
//! - overwrite: re-embedding an ALREADY-embedded entity moves the stored vector
//!   (proven via `vector_search_entities`, reachable here because — unlike
//!   `vector_search_episodes` — it is `pub`).
//! - prefix parity: with `embed_task_prefix_enabled` on, the text handed to
//!   the embedder on the re-embed path is document-prefixed.
//! - idempotency: running it twice against a stable corpus + embedder leaves
//!   a consistent tally with zero failures.
//!
//! Entities are seeded directly via `TemporalGraph::insert_entity_with_group`
//! (reachable through `Memory::temporal_graph_for_test`, `test-utils` feature)
//! rather than through `Memory::remember(...).with_facts(...)` — this keeps
//! each test scoped to `reembed_all_entity_embeddings` itself, not entangled
//! with the separate `with_facts` pinned-entity embed pathway
//! (`Engine::make_pinned_entity_recallable`).

use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::search::{SearchFilters, VectorSearchEntitiesParams};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td112_reembed_entities_{}_{}_{}.db",
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

/// Deterministic 384-dim embedding derived from `seed` — same shape as
/// `core::search`'s own `make_embedding` test helper / the episode sibling's
/// `seeded_embedding`.
fn seeded_embedding(seed: f32) -> Vec<f32> {
    (0..384)
        .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
        .collect()
}

/// Embedder whose output seed is switchable via an atomic flag — same
/// pattern as the episode sibling's `SwitchableEmbeddingProvider`.
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

/// Records every raw string handed to `.embed()`.
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

// ── Overwrite: re-embedding an already-embedded entity moves the vector ────

#[tokio::test]
async fn reembed_all_overwrites_an_existing_entity_embedding() {
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

    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "alice",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "Alice" }),
        group_id: None,
    })
    .await
    .expect("insert entity");
    // Seed at seed 1.0 — simulates a prior ingest-time embed.
    tg.set_entity_embedding("alice", &seeded_embedding(1.0))
        .await
        .expect("seed initial embedding");

    let no_filter = SearchFilters::new();
    let hits_before = tg
        .vector_search_entities(VectorSearchEntitiesParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(
        hits_before.len(),
        1,
        "the one entity must be dense-searchable before re-embed"
    );
    let dist_before_at_seed1 = hits_before[0].score;

    // Simulate a config/embedder change: re-embed the whole corpus, now
    // producing seed-5.0 vectors.
    embedder.switch_to_second_seed();
    let stats = mem
        .reembed_all_entity_embeddings(256)
        .await
        .expect("reembed_all_entity_embeddings must succeed");
    assert_eq!(
        stats.embedded, 1,
        "the one existing entity must be re-embedded"
    );
    assert_eq!(stats.failed, 0);

    let hits_after = tg
        .vector_search_entities(VectorSearchEntitiesParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(hits_after.len(), 1);
    // NOTE: `vector_search_entities`' score is `-distance` ("matching FTS
    // convention" — higher = closer; see `vector_search_brute_force`'s doc
    // comment), the OPPOSITE sign convention from `vector_search_episodes`
    // (raw distance, lower = closer). So after the stored vector moves away
    // from the seed-1.0 query, the score must DECREASE, not increase.
    assert!(
        hits_after[0].score < dist_before_at_seed1,
        "re-embedding must overwrite the stored vector — the score against a \
         seed-1.0 query must have DECREASED (score = -distance here) once the \
         stored vector moved to seed-5.0 (before={dist_before_at_seed1}, after={})",
        hits_after[0].score
    );

    let hits_seed5 = tg
        .vector_search_entities(VectorSearchEntitiesParams {
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

// ── Prefix parity: TD-143 document-prefix applies on the entity re-embed path ──

#[tokio::test]
async fn reembed_all_entities_applies_document_prefix_when_enabled() {
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
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "prefix-probe",
        entity_type_id: 0,
        properties: serde_json::json!({ "name": "kremory td112 reembed prefix probe" }),
        group_id: None,
    })
    .await
    .expect("insert entity");

    recording.texts.lock().expect("mutex poisoned").clear();

    let stats = mem
        .reembed_all_entity_embeddings(256)
        .await
        .expect("reembed_all_entity_embeddings must succeed");
    assert_eq!(stats.embedded, 1);
    assert_eq!(stats.failed, 0);

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == "search_document: kremory td112 reembed prefix probe"),
        "reembed_all_entity_embeddings must document-prefix the resolved \
         display-name text it hands to the embedder when \
         embed_task_prefix_enabled is on; recorded texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("search_query: ")),
        "a re-embed WRITE call must never receive the query prefix; \
         recorded texts: {texts:?}"
    );
}

// ── Idempotency: running it twice is stable, no failures ────────────────────

#[tokio::test]
async fn reembed_all_entities_is_idempotent_across_two_runs() {
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
    for i in 0..3 {
        let id = format!("idempotent-entity-{i}");
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id: &id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": format!("Idempotent Entity {i}") }),
            group_id: None,
        })
        .await
        .expect("insert entity");
    }

    let first = mem
        .reembed_all_entity_embeddings(2) // batch_size smaller than corpus — forces multiple pages
        .await
        .expect("first reembed_all_entity_embeddings run must succeed");
    assert_eq!(first.embedded, 3, "all three entities must be re-embedded");
    assert_eq!(first.failed, 0);

    let second = mem
        .reembed_all_entity_embeddings(2)
        .await
        .expect("second reembed_all_entity_embeddings run must succeed");
    assert_eq!(
        second.embedded, 3,
        "a second run over the SAME corpus must re-embed the SAME three \
         entities again — this is a full re-embed, not a gap-fill, so the \
         tally must be stable across runs, not zero"
    );
    assert_eq!(second.failed, 0, "no failures on either run");
}

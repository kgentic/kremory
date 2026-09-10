//! `Memory::reembed_all_episode_embeddings` overwrite proof (TD-043 split out of
//! `facade/mod.rs`). Lives
//! in-crate (not `tests/`) because proving the STORED vector actually
//! changed requires `pub(crate)` `TemporalGraph::vector_search_episodes`
//! — unreachable from an external integration test. The prefix-parity and
//! idempotency companion tests use only public API and live in
//! `tests/td143_reembed_all_episode_embeddings.rs`.

use super::*;
use crate::core::provider::{EmbeddingProvider, MockChatProvider};
use crate::core::search::{SearchFilters, VectorSearchEpisodesParams};
use std::sync::atomic::{AtomicBool, Ordering};

fn null_llm() -> Arc<dyn ChatProvider> {
    Arc::new(MockChatProvider::null())
}

/// Deterministic 384-dim embedding derived from `seed` — same shape as
/// `core::search`'s own `make_embedding` test helper. Distinct seeds
/// produce distinguishably-different vectors so `vector_search_episodes`
/// ranking can prove a stored embedding actually moved.
fn seeded_embedding(seed: f32) -> Vec<f32> {
    (0..384)
        .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
        .collect()
}

/// Embedder whose output seed is switchable via an atomic flag — lets a
/// test embed the SAME text twice and get two DIFFERENT vectors,
/// simulating "the embedder/config changed since the corpus was first
/// embedded" without needing a second real model.
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

impl EmbeddingProvider for SwitchableEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> impl std::future::Future<Output = crate::CoreResult<Vec<f32>>> + Send + 'a {
        let seed = if self.use_second_seed.load(Ordering::SeqCst) {
            5.0
        } else {
            1.0
        };
        async move { Ok(seeded_embedding(seed)) }
    }
}

/// Re-embedding an ALREADY-EMBEDDED episode overwrites the stored vector
/// — the exact gap `backfill_episode_embeddings` cannot close (its `WHERE
/// embedding IS NULL` paging only ever fills a gap). Proven via
/// `vector_search_episodes` ranking against two distinguishable query
/// vectors, not by trusting the tally alone.
#[tokio::test]
async fn reembed_all_overwrites_an_existing_embedding() {
    let embedder = Arc::new(SwitchableEmbeddingProvider::new());
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = embedder.clone();

    let mem = Memory::open(":memory:")
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td143-reembed-overwrite"))
        .await
        .expect("build memory with dense episode arm");

    // Ingest → embedded with seed 1.0 (dense arm embeds at ingest time).
    mem.remember("the quick brown fox")
        .skip_extraction()
        .await
        .expect("remember must persist the episode + its initial embedding");

    let tg = mem
        .temporal_graph
        .as_ref()
        .expect("Memory built via the builder/providers path carries a TemporalGraph");
    let no_filter = SearchFilters::new();

    // Sanity: a query embedded at seed 1.0 (matching the corpus) ranks it.
    let hits_before = tg
        .vector_search_episodes(VectorSearchEpisodesParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
            as_of: None,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(
        hits_before.len(),
        1,
        "the one ingested episode must be dense-searchable before re-embed"
    );
    let dist_before_at_seed1 = hits_before[0].score;

    // Simulate a config/embedder change (a flip-the-knob, or a
    // model swap): re-embed the WHOLE corpus, now producing seed-5.0
    // vectors.
    embedder.switch_to_second_seed();
    let stats = mem
        .reembed_all_episode_embeddings(256)
        .await
        .expect("reembed_all_episode_embeddings must succeed");
    assert_eq!(
        stats.embedded, 1,
        "the one existing episode must be re-embedded"
    );
    assert_eq!(stats.failed, 0);

    // The stored vector must have MOVED: a seed-1.0 query is now a worse
    // (larger cosine-distance) match than it was before the re-embed,
    // because the stored vector is no longer the seed-1.0 vector.
    let hits_after = tg
        .vector_search_episodes(VectorSearchEpisodesParams {
            query_embedding: &seeded_embedding(1.0),
            limit: 10,
            filters: &no_filter,
            as_of: None,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(hits_after.len(), 1);
    assert!(
        hits_after[0].score > dist_before_at_seed1,
        "re-embedding must overwrite the stored vector — the distance to \
         a seed-1.0 query must have INCREASED once the stored vector \
         moved to seed-5.0 (before={dist_before_at_seed1}, after={})",
        hits_after[0].score
    );

    // Directly confirm the NEW stored vector is now the BEST match for a
    // seed-5.0 query (distance ~0, since query and stored vector are
    // identical once re-embedded).
    let hits_seed5 = tg
        .vector_search_episodes(VectorSearchEpisodesParams {
            query_embedding: &seeded_embedding(5.0),
            limit: 10,
            filters: &no_filter,
            as_of: None,
        })
        .await
        .expect("vector search must succeed");
    assert_eq!(hits_seed5.len(), 1);
    assert!(
        hits_seed5[0].score < hits_after[0].score,
        "a seed-5.0 query must be a MUCH closer match than a seed-1.0 \
         query to the freshly-stored seed-5.0 vector \
         (seed5_dist={}, seed1_dist={})",
        hits_seed5[0].score,
        hits_after[0].score
    );
}

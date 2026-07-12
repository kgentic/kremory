#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-072 seq1 — foundation-lane smoke test.
//!
//! Spec: `.ai-docs/specs/adr-072-seq1-fts-content-recall-impl-spec-2026-07-09.md`
//! ADR: `.ai-docs/adrs/adr-072-content-rag-and-chunker-seam-2026-07-09.md`
//!
//! Drives the full consumer path: `mem.remember(...).skip_extraction()`
//! (Phase 1 store — the only phase that populates `episodes_fts`, per
//! `core/graph/episodes.rs::index_episode_content`) → `mem.recall(q).content()`
//! (the new BM25-only terminal, `facade/recall.rs::RecallContentRequest`).
//!
//! No LLM call is made (`MockChatProvider::null()` + `skip_extraction()`) —
//! per the impl-spec DoD #8, LLM smoke is N/A for seq1 (no LLM in the loop).

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::memory::ContentPassage;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

async fn make_memory(ns: &str) -> Memory {
    let llm: Arc<dyn kremory::memory::ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .default_namespace(Namespace::new(ns))
        .await
        .expect("Memory must build")
}

/// Ingest 3 distinct episodes, `recall(q).content()`, assert the right
/// `ContentPassage.snippet`/`episode_id` comes back and BM25 score ordering
/// is sane (foundation-lane smoke per the task brief).
#[tokio::test]
async fn content_recall_returns_bm25_ranked_passages_for_matched_episodes() {
    let mem = make_memory("adr072-seq1-smoke").await;

    // Episode A: matches neither query term.
    let commit_a = mem
        .remember("The quarterly roadmap review covers budget allocation for the next fiscal cycle.")
        .skip_extraction()
        .await
        .expect("episode A must commit");

    // Episode B: single mention of each query term — weaker BM25 match.
    let commit_b = mem
        .remember("Team sync notes: a quick mention of Zephyr and a caching improvement.")
        .skip_extraction()
        .await
        .expect("episode B must commit");

    // Episode C: repeated "Zephyr" — higher term frequency, stronger BM25 match.
    let commit_c = mem
        .remember(
            "Zephyr Zephyr Zephyr — the Zephyr caching subsystem now supports invalidation \
             hooks and the Zephyr team shipped it to staging.",
        )
        .skip_extraction()
        .await
        .expect("episode C must commit");

    let episode_id_a: i64 = commit_a
        .episode_entity_id
        .parse()
        .expect("episode A commit must carry a parseable rowid (inline Phase 1 path)");
    let episode_id_b: i64 = commit_b
        .episode_entity_id
        .parse()
        .expect("episode B commit must carry a parseable rowid (inline Phase 1 path)");
    let episode_id_c: i64 = commit_c
        .episode_entity_id
        .parse()
        .expect("episode C commit must carry a parseable rowid (inline Phase 1 path)");

    let passages: Vec<ContentPassage> = mem
        .recall("Zephyr caching")
        .content()
        .await
        .expect("content recall must succeed");

    assert_eq!(
        passages.len(),
        2,
        "only episodes B and C mention BOTH query terms (FTS5 default AND); got: {passages:?}"
    );

    let returned_ids: Vec<i64> = passages.iter().map(|p| p.episode_id).collect();
    assert!(
        !returned_ids.contains(&episode_id_a),
        "episode A (no match) must NOT be returned; got ids: {returned_ids:?}"
    );
    assert!(
        returned_ids.contains(&episode_id_b) && returned_ids.contains(&episode_id_c),
        "episodes B and C must both be returned; got ids: {returned_ids:?}"
    );

    // BM25 ordering sanity: episode C repeats "Zephyr" 5x vs B's 1x, so C's
    // BM25 rank must be strictly more relevant (lower score — kremory's FTS
    // score convention is "lower = more relevant", matching
    // `core::search::SearchHit::score` doc + the existing entity/fact FTS
    // arms' `ORDER BY fts.rank` ascending). C must therefore rank first.
    assert_eq!(
        passages[0].episode_id, episode_id_c,
        "higher term-frequency episode (C) must rank first; got: {passages:?}"
    );
    assert_eq!(passages[1].episode_id, episode_id_b);
    assert!(
        passages[0].score < passages[1].score,
        "BM25 scores must strictly differentiate C (stronger match) from B (weaker); \
         got scores {} (C) vs {} (B)",
        passages[0].score,
        passages[1].score
    );

    // Snippet content sanity — the returned snippet must reflect the actual
    // matched episode text (not a garbled/empty extract).
    for p in &passages {
        let lower = p.snippet.to_lowercase();
        assert!(
            lower.contains("zephyr") && lower.contains("caching"),
            "snippet must contain both matched terms, got: {:?}",
            p.snippet
        );
    }

    // Attribution sanity — source_ref always names the originating episode.
    for p in &passages {
        assert_eq!(p.source_ref.kind, kremory::SourceKind::Episode);
        assert_eq!(p.source_ref.id, p.episode_id.to_string());
    }
}

/// DoD #4 — namespace/`group_id` filtering honored: an episode ingested into
/// a DIFFERENT namespace must never leak into another namespace's content
/// recall.
#[tokio::test]
async fn content_recall_does_not_leak_across_namespaces() {
    let mem = make_memory("adr072-seq1-ns-a").await;

    mem.remember("Namespace A: the Zephyr project kicked off this week.")
        .skip_extraction()
        .await
        .expect("namespace A episode must commit");

    mem.remember("Namespace B content should never surface here.")
        .in_namespace(Namespace::new("adr072-seq1-ns-b"))
        .skip_extraction()
        .await
        .expect("namespace B episode must commit");

    // Default-namespace recall (namespace A) must see only its own episode.
    let passages_a: Vec<ContentPassage> = mem
        .recall("Zephyr")
        .content()
        .await
        .expect("namespace A content recall must succeed");
    assert_eq!(passages_a.len(), 1, "namespace A must see exactly its own episode");

    // Namespace B recall for a namespace-A-only term must return nothing.
    let passages_b: Vec<ContentPassage> = mem
        .recall("Zephyr")
        .in_namespace(Namespace::new("adr072-seq1-ns-b"))
        .content()
        .await
        .expect("namespace B content recall must succeed");
    assert!(
        passages_b.is_empty(),
        "namespace B must not see namespace A's episode; got: {passages_b:?}"
    );
}

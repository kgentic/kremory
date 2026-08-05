#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-066 Increment 1 — content-fusion parity fix, seam-level integration test.
//!
//! Spec: `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md`
//! §3 Increment 1 DoD item 2: "`Memory::recall(query).raw().await` returns
//! content-stream results when `content-search` feature is on (integration
//! test: ingest an episode whose entity-graph has NO matching entity but
//! whose raw text answers the query — assert the content-derived result
//! appears)."
//!
//! Drives the REAL producer (`mem.remember(...).skip_extraction()` — same
//! production path `index_episode_content` uses to populate `episodes_fts`,
//! mirroring `adr072_seq1_content_search.rs`) with NO structured facts, so
//! ZERO entities land in the graph for this episode — the entity-graph
//! `recall()` arm is guaranteed empty for it. The content-search BM25 stream
//! still indexes the raw episode text at Phase 1 store, independent of
//! extraction. This proves the FUSION (both arms feeding one output) — not
//! just the content arm alone, which `adr072_seq1_content_search.rs` already
//! covers via `.content()`.
//!
//! The "no attached `TemporalGraph`" feature-off-shaped degrade path is
//! covered as a facade-internal unit test
//! (`crates/kremory/src/facade/recall.rs::fuse_content_stream_tests`) rather
//! than here — constructing a `Memory` whose `graph_search` never resolves
//! (the only way to reach `temporal_graph: None`) cannot drive a real
//! end-to-end recall without panicking inside the stub first.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
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

/// `.raw()` (the `Vec<RetrievedContext>` terminal `kremory_recall`'s
/// `Structured` format also calls — `kremory-mcp/src/handlers.rs::do_recall`)
/// must surface a content-derived result for a query whose answer exists
/// ONLY in raw episode text, with no matching graph entity.
#[tokio::test]
async fn recall_raw_surfaces_content_derived_result_with_no_matching_entity() {
    let mem = make_memory("td066-inc1-raw").await;

    // No structured_facts, skip_extraction — zero entities land in the
    // graph for this episode, but Phase 1 store still indexes the raw text
    // into `episodes_fts` regardless of extraction (per
    // `adr072_seq1_content_search.rs`'s own doc comment).
    let commit = mem
        .remember("Zephyrine went scuba diving off the coast of Portugal last summer.")
        .skip_extraction()
        .await
        .expect("episode must commit");
    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("commit must carry a parseable rowid (inline Phase 1 path)");

    let fused = mem
        .recall("scuba diving Portugal")
        .in_namespace(Namespace::new("td066-inc1-raw"))
        .raw()
        .await
        .expect("fused recall must succeed");

    assert!(
        fused
            .iter()
            .any(|r| r.entity_id == episode_id.to_string() && r.summary.contains("scuba diving")),
        "content-derived result for the ingested episode must appear in the fused \
         `.raw()` output even though no graph entity matches; got: {fused:?}"
    );

    // Sanity: the entity-graph arm alone genuinely finds nothing for this
    // episode — proves the premise ("entity-graph has NO matching entity")
    // the DoD asks this test to construct, not just that content happens to
    // also be present.
    let entity_only_count = fused
        .iter()
        .filter(|r| r.entity_type_name != "ContentPassage")
        .count();
    assert_eq!(
        entity_only_count, 0,
        "no real graph entity should exist for this skip_extraction episode: {fused:?}"
    );
}

/// `mem.recall(q).await` (the `String` `execute()` terminal) must ALSO fuse
/// in the content stream — the spec requires BOTH `.await` and `.raw().await`
/// to fuse by default, not just `.raw()`.
#[tokio::test]
async fn recall_execute_string_terminal_surfaces_content_derived_text() {
    let mem = make_memory("td066-inc1-execute").await;

    mem.remember("The quarterly roadmap review covers a rare mention of Xylophone Corp.")
        .skip_extraction()
        .await
        .expect("episode must commit");

    let block: String = mem
        .recall("Xylophone Corp roadmap")
        .in_namespace(Namespace::new("td066-inc1-execute"))
        .await
        .expect("recall must succeed");

    assert!(
        block.contains("Xylophone"),
        "the String `recall().await` terminal must surface content-derived text \
         (no matching graph entity exists for this episode); got: {block:?}"
    );
}

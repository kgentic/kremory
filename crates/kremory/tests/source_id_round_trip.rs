//! Regression test: D6 — insert_episode source_id round-trip.
//!
//! Before the v0.1.6 fix, `Memory::remember().from_source(sid, kind).await`
//! → `engine_handle::graph_ingest_episode` → `engine.ingest()` → `insert_episode_with_group`
//! wrote the episode row WITHOUT the `source_id` column (Migration 007 gap).
//! Result: `recall_by_source_id(sid, ns)` always returned empty.
//!
//! This test would FAIL on the pre-fix code and PASSES after the fix.
//! It is the canonical regression prevention test for ADR-032 / spec §3.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, SourceKind};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_source_id_rt_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

async fn make_memory(tag: &str) -> Memory {
    Memory::open(unique_db_path(tag))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open must succeed")
}

// ── Regression test ──────────────────────────────────────────────────────────

/// Hot path round-trip: `remember().from_source(sid, Document)` → `recall_by_source_id(sid)`.
///
/// The fix: `engine_handle::graph_ingest_episode` now forwards `source_ref.id`
/// through `SourceParams` into `insert_episode_with_group`, which writes the
/// `source_id` column. Without the fix this test fails with `episodes.is_empty()`.
#[tokio::test]
async fn remember_from_source_then_recall_by_source_id_returns_episode() {
    let ns = Namespace::new("test-source-id-round-trip");
    let mem = make_memory("round_trip").await;

    let source_slug = "my-doc-slug-001";

    // Ingest via the public hot path (the buggy path pre-fix).
    let commit = mem
        .remember("Episode content for source_id round-trip test.")
        .from_source(source_slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction() // ner feature: test is not about extraction
        .await
        .expect("remember().from_source() must succeed");

    // Confirm the commit returned a valid episode handle.
    assert!(
        !commit.episode_entity_id.is_empty(),
        "EpisodeCommit.episode_entity_id must not be empty"
    );

    // Now recall by source_id — this was always empty before the fix.
    let episodes = mem
        .recall_by_source_id(source_slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");

    assert!(
        !episodes.is_empty(),
        "recall_by_source_id must return at least one episode after remember().from_source() \
         (regression: insert_episode_with_group was not writing source_id before v0.1.6 fix)"
    );

    // Verify the content matches what we ingested.
    assert!(
        episodes
            .iter()
            .any(|ep| ep.content.contains("source_id round-trip")),
        "at least one recalled episode must contain the ingested content"
    );
}

/// Namespace isolation: episodes ingested under NS-A must not appear when
/// recalling by source_id under NS-B, even if both use the same slug.
#[tokio::test]
async fn recall_by_source_id_is_namespace_scoped() {
    let ns_a = Namespace::new("test-source-id-ns-a");
    let ns_b = Namespace::new("test-source-id-ns-b");
    let mem = make_memory("ns_scope").await;

    let slug = "shared-slug";

    mem.remember("Content in namespace A.")
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns_a.clone())
        .skip_extraction() // ner feature: test is not about extraction
        .await
        .expect("ingest into ns-a must succeed");

    // Recall scoped to ns-b must return nothing (no ingest into ns-b).
    let episodes = mem
        .recall_by_source_id(slug, Some(ns_b))
        .await
        .expect("recall_by_source_id must not error");

    assert!(
        episodes.is_empty(),
        "recall scoped to ns-b must return empty when episode was ingested only into ns-a"
    );

    // Recall scoped to ns-a must return the episode.
    let episodes_a = mem
        .recall_by_source_id(slug, Some(ns_a))
        .await
        .expect("recall_by_source_id must not error");

    assert!(
        !episodes_a.is_empty(),
        "recall scoped to ns-a must return the episode ingested under ns-a"
    );
}

//! ADR-070 Fork 5 — facade-level sink-wiring test (§5.5.2, deterministic / default gate).
//!
//! Proves the RESOLVED sink is threaded end-to-end through the REAL producer path:
//! `mem.dream()` → `DreamRequest::execute_blocking` → `run_consolidation` → the
//! orchestrator fires `on_merge_proposed` for each cross_episode merge decision. This
//! closes the coverage gap the lib-level §5.5.2 test leaves open (that one calls
//! `run_consolidation` directly, bypassing the facade's `sink: sink.as_ref()` thread).
//!
//! Deterministic (no real LLM/embeddings):
//! - The null LLM + null embedder are wired only to satisfy `dream()`'s LLM
//!   requirement; every reconciliation pass is turned OFF via `DreamOpts`, so neither
//!   is ever called.
//! - The planted entity pair has NO embeddings, so reconciliation's `canonicalize`
//!   cosine band cannot pre-merge it — it survives to `cross_episode`, which merges it
//!   on label + shared-neighbour structure.
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod helpers;

use std::sync::Arc;

use chrono::Utc;
use helpers::recording_sink::RecordingSink;
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;
use kremory::memory::events::EnrichmentEventSink;
use kremory::{DreamOpts, DynEmbeddingProvider, Memory, Namespace};

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}
fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Plant the canonical exact-label corroborated mergeable pair (mirrors the lib-level
/// `plant_mergeable_pair`): two ids that normalize identically + a shared neighbour
/// `acme` across two distinct episodes → exactly one cross_episode merge.
async fn plant_mergeable_pair(graph: &TemporalGraph, gid: &str) {
    for id in ["John Smith", "john  smith", "acme"] {
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({ "name": id }),
                group_id: Some(gid),
            })
            .await
            .expect("plant entity");
    }
    let e1 = graph
        .insert_episode(InsertEpisodeParams {
            content: "ep1",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("ep1");
    let e2 = graph
        .insert_episode(InsertEpisodeParams {
            content: "ep2",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("ep2");
    for (ep, ent) in [(e1, "John Smith"), (e2, "john  smith")] {
        graph
            .insert_episodic_edge(InsertEpisodicEdgeParams {
                episode_id: ep,
                entity_id: ent,
                entity_group_id: Some(gid),
                role: "mention",
            })
            .await
            .expect("plant episodic edge");
    }
    // Shared neighbour: BOTH ids work at acme (the corroboration signal that lets
    // cross_episode merge them rather than treat them as homonyms).
    for subject in ["John Smith", "john  smith"] {
        let now = Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
                  subject_group_id, object_group_id, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
                libsql::params![subject, "works_at", "acme", now.clone(), now, gid, gid, gid],
            )
            .await
            .expect("plant relational fact");
    }
}

/// Drive the full `mem.dream()` path with cross_episode ON (all reconciliation OFF)
/// and a RecordingSink registered; return the captured `on_merge_proposed` events.
async fn dream_with_sink(dry_run: bool) -> Vec<(String, String, String, bool)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("adr070-sink-wiring");
    let mem = Memory::open(dir.path().join("sink_wiring.db"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open");

    // group_id for a no-thread namespace == the namespace string.
    let gid = "adr070-sink-wiring";
    plant_mergeable_pair(
        mem.temporal_graph_for_test().expect("temporal graph"),
        gid,
    )
    .await;

    let sink = Arc::new(RecordingSink::default());
    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut opts = DreamOpts::default();
    // Every reconciliation pass OFF → the null LLM is never called AND canonicalize
    // can't pre-merge the (embedding-less) pair before cross_episode sees it.
    opts.include_type_discovery = false;
    opts.include_consistency_check = false;
    opts.include_type_registry_collapse = false;
    opts.include_acronym_nickname_recall = false;
    opts.include_type_novelty_llm_verify = false;
    // Only the op under test.
    opts.include_cross_episode_merges = true;
    opts.cross_episode_dry_run = dry_run;

    mem.dream()
        .in_namespace(ns)
        .opts(opts)
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await
        .expect("mem.dream() must succeed");

    sink.merge_proposed_events()
}

/// §5.5.2 mechanical criterion, at the FACADE layer: `on_merge_proposed` fires once
/// per cross_episode merge decision with the correct `dry_run` flag — proving the
/// facade actually threads the resolved sink into `run_consolidation`.
#[tokio::test]
async fn dream_threads_sink_to_cross_episode_on_merge_proposed() {
    // Shadow: decision observed, no fusion, event carries dry_run=true.
    let shadow = dream_with_sink(true).await;
    assert_eq!(
        shadow.len(),
        1,
        "one cross_episode merge decision → one on_merge_proposed (shadow)"
    );
    assert_eq!(shadow[0].0, "adr070-sink-wiring", "group_id threaded through");
    assert_eq!(shadow[0].2, "John Smith", "keeper = lowest id");
    assert_eq!(shadow[0].1, "john  smith", "loser");
    assert!(
        shadow[0].3,
        "dry_run=true propagated through the facade → run_consolidation → event"
    );

    // Applied: same fixture, real fusion, event carries dry_run=false.
    let applied = dream_with_sink(false).await;
    assert_eq!(
        applied.len(),
        1,
        "one cross_episode merge decision → one on_merge_proposed (applied)"
    );
    assert!(
        !applied[0].3,
        "dry_run=false propagated through the facade → run_consolidation → event"
    );
}

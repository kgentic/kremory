#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]
//! ADR-071 §Item 2 — P2 `include_fact_archival` + P4 `include_community_detection`
//! enablement smoke coverage.
//!
//! Spec: `.ai-docs/specs/adr-071-dream-phase-hardening-impl-spec-2026-07-06.md`
//! §"Item 2 — P2 `facts_archival` + P4 `communities_updated` enablement".
//!
//! Closes the exact gap the ADR names: the archive/communities OP's own unit
//! tests (`consolidation_archive_test.rs`, `consolidation_communities_test.rs`)
//! already prove the op body is correct when called DIRECTLY
//! (`archive::archive(...)` / `communities::communities(...)`). What is NOT yet
//! proven is that the op fires correctly when wired through the REAL dispatcher —
//! the full `mem.dream()` facade call — with the corresponding `DreamOpts.include_*`
//! flag flipped on. These three tests drive `mem.dream()` (never the op function
//! directly).
//!
//! No Wilson-LB gate for this item (unlike Item 1's P3 corpus gate) — the
//! reversibility argument is the mechanical proof: archive is an append-only move
//! (recoverable via `facts_archive`), communities is a full-recompute with no
//! accumulated corruption (§Item 2, "No Wilson-LB gate for this item").
//!
//! Deterministic (no real LLM/embeddings needed): `MockChatProvider::null()` is
//! wired only to satisfy `dream()`'s `Category B` LLM requirement (ADR-041) — the
//! archive/communities ops themselves are zero-LLM, pure-structural ops, and the
//! reconciliation passes that DO call the LLM (Pass 0, consistency_check, Site #3,
//! Site #5) are proven non-fatal-on-empty-response by the established
//! `dream_phase2_deterministic_passes.rs` / `dream_phase3_consistency_check.rs`
//! precedent (same `MockChatProvider::null()` + default `DreamOpts` pattern).

use std::sync::Arc;

use chrono::{Duration, Utc};

use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DreamOpts, DynEmbeddingProvider, Memory, Namespace};

const DIM: usize = 64;

/// Mirrors `dream_phase2_deterministic_passes.rs::open_mem` — a `Memory` wired
/// with a null LLM (safe no-op for the always-on reconciliation passes) + a
/// deterministic hash-based embedder, in a fresh temp-dir sqlite file per test.
async fn open_mem(dir: &std::path::Path, ns: &Namespace) -> Memory {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(DIM));
    Memory::open(dir.join("adr071-item2.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(DIM)
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open must succeed")
}

// ─── Archive (P2) smoke-fixture helpers ───────────────────────────────────────
// Mirrors `consolidation_archive_test.rs`'s own local helpers (per-file
// convention — integration test binaries do not share code without a
// `tests/support/` module).

async fn insert_entity(graph: &TemporalGraph, gid: &str, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(gid),
        })
        .await
        .expect("insert entity");
}

async fn insert_value_fact(
    graph: &TemporalGraph,
    gid: &str,
    subject: &str,
    predicate: &str,
    object_value: &str,
    valid_from: &str,
    expired_at: Option<&str>,
    recorded_at: &str,
) -> i64 {
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, recorded_at, \
              expired_at, group_id, subject_group_id, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
            libsql::params![
                subject,
                predicate,
                object_value,
                valid_from,
                recorded_at,
                expired_at,
                gid,
                gid,
            ],
        )
        .await
        .expect("plant fact");
    let mut rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("id")
}

async fn fact_exists(graph: &TemporalGraph, table: &str, fact_id: i64) -> bool {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE id = ?1");
    let mut rows = graph
        .conn
        .query(sql.as_str(), libsql::params![fact_id])
        .await
        .expect("query");
    let count: i64 = rows
        .next()
        .await
        .expect("row")
        .expect("present")
        .get(0)
        .expect("count");
    count > 0
}

// ─── Communities (P4) smoke-fixture helper ────────────────────────────────────
// Mirrors `consolidation_communities_test.rs::plant_two_triangle_bridge` —
// reused verbatim (structured graph, ground truth = 2 communities).

async fn insert_typed_entity(graph: &TemporalGraph, gid: &str, id: &str, type_id: u32) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: type_id,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(gid),
        })
        .await
        .expect("insert entity");
}

async fn new_episode(graph: &TemporalGraph) -> i64 {
    graph
        .insert_episode(InsertEpisodeParams {
            content: "episode content",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("episode")
}

async fn anchor(graph: &TemporalGraph, gid: &str, episode_id: i64, entity: &str) {
    graph
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id,
            entity_id: entity,
            entity_group_id: Some(gid),
            role: "mention",
        })
        .await
        .expect("edge");
}

/// Two dense triangles ({a,b,c} and {d,e,f}), each sharing three episodes, linked
/// by a SINGLE cross-episode bridge edge (c,d co-occur in one episode). Ground
/// truth = 2 communities (mirrors `consolidation_communities_test.rs` exactly).
async fn plant_two_triangle_bridge(graph: &TemporalGraph, gid: &str) {
    for id in ["a", "b", "c", "d", "e", "f"] {
        insert_typed_entity(graph, gid, id, 1).await;
    }
    for _ in 0..3 {
        let ep = new_episode(graph).await;
        for id in ["a", "b", "c"] {
            anchor(graph, gid, ep, id).await;
        }
    }
    for _ in 0..3 {
        let ep = new_episode(graph).await;
        for id in ["d", "e", "f"] {
            anchor(graph, gid, ep, id).await;
        }
    }
    let bridge = new_episode(graph).await;
    anchor(graph, gid, bridge, "c").await;
    anchor(graph, gid, bridge, "d").await;
}

async fn count(graph: &TemporalGraph, sql: &str, gid: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![gid])
        .await
        .expect("count query");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("count")
}

/// `member_hash` column of `community_summaries` for `group_id`, sorted — the
/// persisted form of `substrate::community_member_hash` (crate-private; read via
/// its persisted column instead of calling the fn directly across the test-binary
/// boundary). A SHA-256 over the sorted entity-id membership set (ADR-066 §F-2) —
/// a pure function of membership, order-independent, community_id-independent.
async fn member_hashes(graph: &TemporalGraph, gid: &str) -> Vec<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT member_hash FROM community_summaries WHERE group_id = ?1 \
             ORDER BY member_hash",
            libsql::params![gid],
        )
        .await
        .expect("query community_summaries");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("row") {
        out.push(row.get::<String>(0).expect("member_hash"));
    }
    out
}

// ─── Test 1 — P2 archive fires inside the FULL mem.dream() call ──────────────

#[tokio::test]
async fn p2_archive_enablement_smoke_fires_inside_full_dream_call() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("adr071-item2-archive");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    let now = Utc::now();
    let recorded_at = now.to_rfc3339();
    insert_entity(&graph, &gid, "subject-1").await;

    // Live anchor fact (expired_at NULL) — keeps "subject-1" bound after the
    // candidate is archived, so the P2.2 ref-count "orphans nothing" guard does
    // NOT strand it (would otherwise KEEP the candidate as `sole_binding`).
    insert_value_fact(
        &graph,
        &gid,
        "subject-1",
        "anchor_pred",
        "anchor_val",
        &(now - Duration::days(1)).to_rfc3339(),
        None,
        &recorded_at,
    )
    .await;

    // Candidate: expired 200 days ago — well past the default
    // `archive_grace_days` (90, `DreamOpts::default()`), so `expired_at < cutoff`.
    let candidate_id = insert_value_fact(
        &graph,
        &gid,
        "subject-1",
        "old_pred",
        "old_val",
        &(now - Duration::days(300)).to_rfc3339(),
        Some(&(now - Duration::days(200)).to_rfc3339()),
        &recorded_at,
    )
    .await;

    assert!(
        fact_exists(&graph, "facts", candidate_id).await,
        "candidate fact must be live in `facts` before dream()"
    );

    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut opts = DreamOpts::default();
    opts.include_fact_archival = true;
    let summary = mem
        .dream()
        .in_namespace(ns)
        .with_opts(opts)
        .await
        .expect("mem.dream() with include_fact_archival must succeed");

    assert!(
        summary.facts_archived > 0,
        "DreamSummary.facts_archived must be > 0 when a long-expired, \
         non-stranding fact exists and include_fact_archival is on; got {}",
        summary.facts_archived
    );

    assert!(
        !fact_exists(&graph, "facts", candidate_id).await,
        "archived candidate must be GONE from live `facts` after mem.dream()"
    );
    assert!(
        fact_exists(&graph, "facts_archive", candidate_id).await,
        "archived candidate must be present in `facts_archive` after mem.dream()"
    );
}

// ─── Test 2 — P4 communities fires inside the FULL mem.dream() call ──────────

#[tokio::test]
async fn p4_communities_enablement_smoke_fires_inside_full_dream_call() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("adr071-item2-communities");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    plant_two_triangle_bridge(&graph, &gid).await;

    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut opts = DreamOpts::default();
    opts.include_community_detection = true;
    let summary = mem
        .dream()
        .in_namespace(ns)
        .with_opts(opts)
        .await
        .expect("mem.dream() with include_community_detection must succeed");

    assert!(
        summary.communities_updated > 0,
        "DreamSummary.communities_updated must be > 0 on the first sweep of a \
         structured co-occurrence graph; got {}",
        summary.communities_updated
    );

    let entity_communities_rows = count(
        &graph,
        "SELECT COUNT(*) FROM entity_communities WHERE group_id = ?1",
        &gid,
    )
    .await;
    let community_summaries_rows = count(
        &graph,
        "SELECT COUNT(*) FROM community_summaries WHERE group_id = ?1",
        &gid,
    )
    .await;

    assert!(
        entity_communities_rows > 0,
        "entity_communities rows must exist post-dream() (got {entity_communities_rows})"
    );
    assert!(
        community_summaries_rows > 0,
        "community_summaries rows must exist post-dream() (got {community_summaries_rows})"
    );
}

// ─── Test 3 — P4 deterministic-recompute invariant (Risk #15, P0) ────────────

#[tokio::test]
async fn p4_communities_deterministic_recompute_is_stable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("adr071-item2-communities-determinism");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    plant_two_triangle_bridge(&graph, &gid).await;

    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let opts = || {
        let mut o = DreamOpts::default();
        o.include_community_detection = true;
        o
    };

    let first = mem
        .dream()
        .in_namespace(ns.clone())
        .with_opts(opts())
        .await
        .expect("first mem.dream() call must succeed");
    let hashes_1 = member_hashes(&graph, &gid).await;
    assert!(
        !hashes_1.is_empty(),
        "first sweep must persist at least one community_summaries row"
    );

    let second = mem
        .dream()
        .in_namespace(ns)
        .with_opts(opts())
        .await
        .expect("second mem.dream() call must succeed");
    let hashes_2 = member_hashes(&graph, &gid).await;

    // The "deterministic-recompute invariant" (ADR-071 §Item 2): same input graph
    // (unchanged between calls) -> same partition, twice. `member_hash` is a
    // sorted-membership SHA-256 (ADR-066 §F-2) — content-addressed, independent
    // of the auto-increment `community_id` a full-wipe-then-rebuild may reassign.
    assert_eq!(
        hashes_1, hashes_2,
        "second mem.dream() call on the SAME unchanged graph must persist the \
         IDENTICAL sorted community_member_hash set as the first (deterministic \
         recompute); first={first:?} second={second:?}"
    );

    // Second sweep is idempotent: every membership hash already existed, so the
    // "updated" count should be 0 (DoD-P4.5 idempotency, mirrors
    // `consolidation_communities_test.rs::fixture_unchanged_rerun_zero_updated`).
    assert_eq!(
        second.communities_updated, 0,
        "second sweep on an unchanged graph must report communities_updated == 0 \
         (idempotent); got {}",
        second.communities_updated
    );
}

//! ADR-066 dream CONSOLIDATION P4 — community-detection deterministic-fixture harness.
//!
//! Spec `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md` §3
//! (P4.1–P4.6) + §6 (communities corpus) + ADR-066 §2.1. The communities op is
//! ZERO-LLM and a pure function of the co-occurrence TOPOLOGY, so — unlike site3 and
//! like the P1/P2/P3 harnesses — there is NO VCR cassette. Community structure is not
//! expressible as flat JSONL rows (spec §6 lists "graph fixtures", NOT a `.jsonl`
//! corpus), so each case is an INLINE deterministic graph fixture built from
//! `entities` + `episodic_edges` plants. Mirrors the sibling harnesses:
//!
//! - **per-fixture group isolation** — each fixture plants into its own fresh `group_id`;
//! - **smoke-one-before-batch** — `smoke_one_communities_harness` runs ONE representative
//!   fixture (the two-triangle-bridge) through the full pipeline FIRST;
//! - **o11y cross-check** — the pass's own
//!   `kremory.dream.consolidation.communities_updated_total` counter is snapshotted and
//!   asserted EQUAL to the op's returned `communities_updated` count (DoD-P4.3).
//!
//! Fixture cases (spec §6 communities row):
//!   - `two_triangle_bridge` → 2 communities (structured graph).
//!   - `fully_connected_hairball` → 1 community (HONEST collapse, DoD-P4.6 / RISK-004:
//!     assert count == 1, NOT >1 — the documented caveat).
//!   - `symmetric_tie` → deterministic smallest-label tie-break (exact membership).
//!   - `unchanged_rerun` → `communities_updated == 0` (idempotent, DoD-P4.5).
//!   - `dense_episode_hairball` → RISK-004 quality case: MUST NOT crash + reports 1
//!     community HONESTLY (do NOT assert >1 — documented collapse).
//!   - `td183_fixed_idem` → TD-183 DoD (a) regression: fixed graph, `communities()`
//!     run TWICE, persisted membership asserted BYTE-IDENTICAL. Isolates the
//!     deterministic partition op from `dream()`'s LLM-driven upstream passes
//!     (register `.ai-docs/tech-debt/tech-debt-register.md`, "TD-183 DIAGNOSED").

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use chrono::Utc;

use kremory::core::dream::communities;
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── Plant helpers ─────────────────────────────────────────────────────────────

/// Insert a NON-catch-all entity (`entity_type_id = 1`) so it participates in
/// community structure. `entity_type_id = 0` is the catch-all, excluded (C-INV5).
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

/// The persisted `(entity_id -> community_id)` map for `group_id`.
async fn persisted_membership(graph: &TemporalGraph, gid: &str) -> BTreeMap<String, i64> {
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id, community_id FROM entity_communities \
             WHERE group_id = ?1 ORDER BY entity_id",
            libsql::params![gid],
        )
        .await
        .expect("membership");
    let mut out = BTreeMap::new();
    while let Some(row) = rows.next().await.expect("row") {
        out.insert(
            row.get::<String>(0).expect("entity"),
            row.get::<i64>(1).expect("community"),
        );
    }
    out
}

/// Distinct community count from `community_summaries` for `group_id`.
async fn community_count(graph: &TemporalGraph, gid: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM community_summaries WHERE group_id = ?1",
            libsql::params![gid],
        )
        .await
        .expect("count");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("count")
}

// ─── Fixture builders (deterministic graph topologies, spec §6) ──────────────────

/// Two dense triangles ({a,b,c} and {d,e,f}), each sharing three episodes, linked by a
/// SINGLE cross-episode bridge edge (c,d co-occur in one episode). Ground truth = 2.
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

/// A fully-connected / hairball graph: N entities ALL appearing in ONE big episode →
/// near-complete co-occurrence. HONEST collapse to ONE community (DoD-P4.6).
async fn plant_hairball(graph: &TemporalGraph, gid: &str, ids: &[&str]) {
    for id in ids {
        insert_typed_entity(graph, gid, id, 1).await;
    }
    let big = new_episode(graph).await;
    for id in ids {
        anchor(graph, gid, big, id).await;
    }
}

/// A symmetric path a-b-c (b pulled equally by a and c). Smallest-label tie-break must
/// resolve it to a fixed, reproducible partition.
async fn plant_symmetric_tie(graph: &TemporalGraph, gid: &str) {
    for id in ["a", "b", "c"] {
        insert_typed_entity(graph, gid, id, 1).await;
    }
    let e1 = new_episode(graph).await;
    anchor(graph, gid, e1, "a").await;
    anchor(graph, gid, e1, "b").await;
    let e2 = new_episode(graph).await;
    anchor(graph, gid, e2, "b").await;
    anchor(graph, gid, e2, "c").await;
}

// ─── o11y cross-check helper ─────────────────────────────────────────────────────

/// Sum the `communities_updated_total` counter from a snapshot (DoD-P4.3 cross-check).
fn communities_updated_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(key, _unit, _desc, value)| {
            if key.key().name() != "kremory.dream.consolidation.communities_updated_total" {
                return None;
            }
            match value {
                DebugValue::Counter(n) => Some(n),
                _ => None,
            }
        })
        .sum()
}

// ─── Smoke-one-before-batch (the representative structured fixture) ──────────────

#[tokio::test]
async fn smoke_one_communities_harness() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "smoke_bridge";
    plant_two_triangle_bridge(&graph, gid).await;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let report = communities(&graph, gid).await.expect("communities");
    let count = community_count(&graph, gid).await;
    eprintln!(
        "[smoke-one] fixture=two_triangle_bridge communities={count} updated={} ",
        report.count
    );

    assert_eq!(count, 2, "structured two-triangle-bridge → 2 communities");
    assert_eq!(report.count, 2, "first run: both communities updated");

    // o11y cross-check: the op's counter equals the returned updated count.
    assert_eq!(
        communities_updated_counter(&snapshotter),
        report.count as u64,
        "communities_updated_total counter must equal OpReport.count (DoD-P4.3)"
    );
}

// ─── Fixture: two-triangle-bridge → 2 communities + membership (spec §6) ─────────

#[tokio::test]
async fn fixture_two_triangle_bridge_two_communities() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "fx_bridge";
    plant_two_triangle_bridge(&graph, gid).await;

    let report = communities(&graph, gid).await.expect("communities");
    assert_eq!(community_count(&graph, gid).await, 2, "2 communities");
    assert_eq!(report.count, 2);

    let m = persisted_membership(&graph, gid).await;
    assert_eq!(m["a"], m["b"], "a,b same community");
    assert_eq!(m["b"], m["c"], "b,c same community");
    assert_eq!(m["d"], m["e"], "d,e same community");
    assert_eq!(m["e"], m["f"], "e,f same community");
    assert_ne!(m["a"], m["d"], "the two triangles are DISTINCT communities");
}

// ─── Fixture: fully-connected → 1 HONEST community (DoD-P4.6 caveat) ─────────────

#[tokio::test]
async fn fixture_fully_connected_collapses_to_one_honest_community() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "fx_fullyconnected";
    // 5 entities all in one episode → complete co-occurrence.
    plant_hairball(&graph, gid, &["a", "b", "c", "d", "e"]).await;

    let report = communities(&graph, gid).await.expect("communities");
    // HONEST collapse: assert EXACTLY 1 (NOT >1) — the documented known limitation.
    assert_eq!(
        community_count(&graph, gid).await,
        1,
        "fully-connected graph HONESTLY collapses to ONE community (documented caveat)"
    );
    assert_eq!(report.count, 1);
    let m = persisted_membership(&graph, gid).await;
    let comms: BTreeSet<i64> = m.values().copied().collect();
    assert_eq!(comms.len(), 1, "all members in the single honest community");
}

// ─── Fixture: symmetric-tie → deterministic tie-break (exact membership) ─────────

#[tokio::test]
async fn fixture_symmetric_tie_deterministic_membership() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "fx_tie";
    plant_symmetric_tie(&graph, gid).await;

    communities(&graph, gid).await.expect("run 1");
    let m1 = persisted_membership(&graph, gid).await;
    communities(&graph, gid).await.expect("run 2");
    let m2 = persisted_membership(&graph, gid).await;
    assert_eq!(
        m1, m2,
        "symmetric-tie partition is reproducible (exact membership assertion)"
    );
    assert_eq!(
        m1.len(),
        3,
        "all three nodes assigned exactly one community"
    );
}

// ─── Fixture: unchanged-rerun → communities_updated == 0 (DoD-P4.5) ──────────────

#[tokio::test]
async fn fixture_unchanged_rerun_zero_updated() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "fx_idem";
    plant_two_triangle_bridge(&graph, gid).await;

    let first = communities(&graph, gid).await.expect("first");
    assert!(first.count >= 1, "first run counts new communities");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let second = communities(&graph, gid).await.expect("second");
    assert_eq!(
        second.count, 0,
        "unchanged graph rerun → communities_updated == 0 (idempotent, DoD-P4.5)"
    );
    // o11y cross-check: the second-run counter increments by 0.
    assert_eq!(
        communities_updated_counter(&snapshotter),
        0,
        "communities_updated_total increments by 0 on an unchanged rerun"
    );
}

// ─── Fixture: dense_episode_hairball (RISK-004 quality case) ─────────────────────
// One big episode where MANY entities co-occur → near-complete graph. Assert the op
// does NOT crash + reports 1 community HONESTLY. Do NOT assert >1 — this is the
// documented synchronous-label-propagation collapse (DoD-P4.6 caveat).

#[tokio::test]
async fn fixture_dense_episode_hairball_honest_collapse_no_crash() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "fx_dense_hairball";
    // 12 entities all sharing one big episode (the RISK-004 topology).
    let ids = [
        "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11",
    ];
    plant_hairball(&graph, gid, &ids).await;

    // MUST NOT crash.
    let report = communities(&graph, gid)
        .await
        .expect("communities must not crash on a dense hairball");
    // HONEST report: 1 community (documented collapse, NOT garbage, NOT asserted >1).
    assert_eq!(
        community_count(&graph, gid).await,
        1,
        "dense_episode_hairball HONESTLY collapses to 1 community (documented DoD-P4.6 caveat)"
    );
    assert_eq!(
        report.count, 1,
        "one honest community, first run → updated == 1"
    );
    // Every entity assigned to that one community.
    let m = persisted_membership(&graph, gid).await;
    assert_eq!(m.len(), 12, "all 12 entities assigned");
    let comms: BTreeSet<i64> = m.values().copied().collect();
    assert_eq!(comms.len(), 1, "single honest community");
}

// ─── Batch: run every fixture + o11y cross-check per fixture ──────────────────────

#[tokio::test]
async fn communities_fixture_batch_metrics() {
    // The two-triangle-bridge and the two hairballs pin the structured-vs-collapse
    // behaviour. Each fixture reuses hardcoded slug ids (a..f / a..e / n0..) that would
    // collide across namespaces in ONE db (ADR-029b cross-namespace name-slug guard), so
    // each fixture runs in its OWN in-memory graph — matching how the standalone fixture
    // tests are isolated. The o11y counter is asserted per fixture on its own recorder.

    // Fixture 1: structured → 2.
    let g1 = TemporalGraph::open_in_memory().await.expect("open");
    let rec1 = DebuggingRecorder::new();
    let snap1 = rec1.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&rec1);
        plant_two_triangle_bridge(&g1, "batch_bridge").await;
        let r_bridge = communities(&g1, "batch_bridge").await.expect("bridge");
        assert_eq!(community_count(&g1, "batch_bridge").await, 2);
        assert_eq!(r_bridge.count, 2);
        assert_eq!(
            communities_updated_counter(&snap1),
            r_bridge.count as u64,
            "bridge fixture: counter == OpReport.count (DoD-P4.3)"
        );
    }

    // Fixture 2: fully-connected hairball → 1 (honest collapse).
    let g2 = TemporalGraph::open_in_memory().await.expect("open");
    let rec2 = DebuggingRecorder::new();
    let snap2 = rec2.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&rec2);
        plant_hairball(&g2, "batch_fc", &["a", "b", "c", "d", "e"]).await;
        let r_fc = communities(&g2, "batch_fc").await.expect("fc");
        assert_eq!(community_count(&g2, "batch_fc").await, 1);
        assert_eq!(r_fc.count, 1);
        assert_eq!(
            communities_updated_counter(&snap2),
            r_fc.count as u64,
            "fully-connected fixture: counter == OpReport.count (DoD-P4.3)"
        );
    }

    // Fixture 3: dense_episode_hairball → 1 (honest collapse, no crash).
    let g3 = TemporalGraph::open_in_memory().await.expect("open");
    let rec3 = DebuggingRecorder::new();
    let snap3 = rec3.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&rec3);
        plant_hairball(
            &g3,
            "batch_dense",
            &["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"],
        )
        .await;
        let r_dense = communities(&g3, "batch_dense").await.expect("dense");
        assert_eq!(community_count(&g3, "batch_dense").await, 1);
        assert_eq!(r_dense.count, 1);
        assert_eq!(
            communities_updated_counter(&snap3),
            r_dense.count as u64,
            "dense-hairball fixture: counter == OpReport.count (DoD-P4.3)"
        );
    }
}

// ─── TD-183 regression: whole-graph idempotency at the DETERMINISTIC tier ────────
//
// TD-183 register entry "DIAGNOSED — the flake is INHERITED from dream()'s LLM
// passes, not local to the communities pass" established that `communities()`
// itself contains zero `llm`/`chat`/`model` references and is a pure function of
// co-occurrence topology, but `dream()` chains it after non-deterministic LLM
// passes (reclassify / consistency_check / aliases) that mutate the graph between
// calls — so whole-`dream()` idempotency after ONE call is not a property that
// CAN hold. DoD (a): assert idempotency where it CAN hold — a fixed graph, no
// model, no network, `communities()` run twice, persisted membership asserted
// BYTE-IDENTICAL (a `BTreeMap<String, i64>` structural `assert_eq!` — exact
// key+value equality, the strongest available equality here; strictly stronger
// than comparing `member_hash` digests or the `communities_updated` count alone,
// either of which could theoretically hold under a same-shape-different-content
// coincidence that this does not allow).
//
// This test isolates the deterministic partition op from `dream()`'s LLM-driven
// upstream passes, which the real-LLM idempotency test
// (`dream_full_consolidation_real_llm::idempotency`) structurally cannot do — that
// test's assertion is DELIBERATELY left unchanged (register DoD note: "changing a
// real-LLM test's assertion is a maintainer call, and the safe reading has not
// been excluded — it has only been made unlikely"). If THIS test ever goes RED,
// that is a genuine partition defect (this graph never touches a model), and
// TD-183 reopens with teeth per DoD (c).
#[tokio::test]
async fn td183_communities_idempotent_membership_on_fixed_graph() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let gid = "td183_fixed_idem";
    plant_two_triangle_bridge(&graph, gid).await;

    let first = communities(&graph, gid).await.expect("first run");
    let m1 = persisted_membership(&graph, gid).await;
    let c1 = community_count(&graph, gid).await;

    // Non-vacuity precondition — checked BEFORE the cross-run comparison below. A
    // test that could pass on an empty partition (e.g. a broken fixture yielding
    // zero entities) proves nothing; per
    // [[verify-metric-sensitivity-before-gating-decisions]] an unvalidated
    // instrument is not evidence. Measured baseline for `two_triangle_bridge`: 6
    // entities / 2 communities on the first run.
    assert!(
        !m1.is_empty(),
        "TD-183 non-vacuity: fixture must yield a non-empty partition, got {} members",
        m1.len()
    );
    assert!(
        first.count >= 1,
        "TD-183 non-vacuity: first run must report >=1 updated community, got {}",
        first.count
    );
    eprintln!(
        "[TD-183] fixed-graph baseline: fixture=two_triangle_bridge entities={} communities={} (first run)",
        m1.len(),
        c1
    );

    let second = communities(&graph, gid).await.expect("second run");
    let m2 = persisted_membership(&graph, gid).await;

    assert_eq!(
        m1, m2,
        "TD-183: communities() over a FIXED graph must be idempotent — persisted \
         membership must be BYTE-IDENTICAL across two runs. This graph never \
         touches a model, so a mismatch here is a real partition defect, not \
         model variance (register 2026-08-11 \"TD-183 DIAGNOSED\" entry)."
    );
    assert_eq!(
        second.count, 0,
        "TD-183: second run over an unchanged fixed graph must report 0 updated \
         (no membership changed)"
    );
}

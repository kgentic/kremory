//! Tier-2b DELETE cascade proofs — reversible-graph-mutations arch-spec
//! `.ai-docs/specs/reversible-graph-mutations-arch-spec-2026-07-10.md` §4.4 / §4.5.
//!
//! DETERMINISTIC, zero-LLM (delete/undo are pure SQL over existing FKs), so this whole
//! file sits at the fast tier with NO VCR (`llm-test-pyramid-vcr-seams`). It proves:
//!
//! - `delete_entity_cascade_retracts_facts` — deleting an entity ARCHIVES its facts
//!   (recoverable — NEVER hard-deleted), removes its edges / community / FTS / row, and
//!   retracts the DERIVED community membership of a neighbour whose support drops to
//!   zero — while a neighbour that keeps other support is NOT retracted (precision).
//! - `delete_entity_undo_restores` — undo re-inserts the entity + FTS, un-archives its
//!   facts, re-inserts its edges, and restores memberships; second undo is a no-op.
//! - `delete_fact_retract_and_undo` — delete a fact (archived), retract-on-zero the
//!   endpoint that loses all support, then undo restores both.
//! - `delete_entity_cascade_terminates_on_cyclic_graph` — a graph with `A—fact—B` and
//!   `B—fact—A` cascades in bounded time (the recursive-CTE `UNION` + depth cap
//!   terminate on cycles) and retracts B on zero support.

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use kremory::core::dream::{
    delete_entity, delete_fact, undo_delete_entity, undo_delete_fact, DeleteEntityParams,
};
use kremory::core::graph::{InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams};
use kremory::core::schema::TemporalGraph;

const GROUP: &str = "meeting_42";

async fn insert_bare(graph: &TemporalGraph, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(GROUP),
        })
        .await
        .expect("insert bare entity");
}

async fn plant_fact(graph: &TemporalGraph, subject: &str, predicate: &str, object: &str) -> i64 {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
            libsql::params![subject, predicate, object, now.clone(), now, GROUP, GROUP, GROUP],
        )
        .await
        .expect("plant fact");
    let mut rows = graph
        .conn
        .query(
            "SELECT id FROM facts WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3 \
             ORDER BY id DESC LIMIT 1",
            libsql::params![subject, predicate, object],
        )
        .await
        .expect("query fact id");
    rows.next()
        .await
        .expect("row")
        .expect("fact present")
        .get::<i64>(0)
        .expect("id col")
}

async fn link_episode(graph: &TemporalGraph, entity: &str) -> i64 {
    let ep = graph
        .insert_episode(InsertEpisodeParams {
            content: "meeting transcript segment",
            timestamp: chrono::Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("insert episode");
    graph
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id: ep,
            entity_id: entity,
            entity_group_id: Some(GROUP),
            role: "mention",
        })
        .await
        .expect("insert episodic edge");
    ep
}

async fn set_community(graph: &TemporalGraph, entity: &str, community_id: i64) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO entity_communities (group_id, entity_id, community_id, updated_at) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![GROUP, entity, community_id, now],
        )
        .await
        .expect("set community");
}

async fn count(graph: &TemporalGraph, sql: &str, id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![id])
        .await
        .expect("count query");
    rows.next()
        .await
        .expect("row")
        .expect("count")
        .get::<i64>(0)
        .expect("n")
}

async fn count_i64(graph: &TemporalGraph, sql: &str, id: i64) -> i64 {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![id])
        .await
        .expect("count query");
    rows.next()
        .await
        .expect("row")
        .expect("count")
        .get::<i64>(0)
        .expect("n")
}

async fn entity_exists(graph: &TemporalGraph, id: &str) -> bool {
    count(graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", id).await > 0
}
async fn fts_exists(graph: &TemporalGraph, id: &str) -> bool {
    count(graph, "SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?1", id).await > 0
}
async fn community_exists(graph: &TemporalGraph, id: &str) -> bool {
    count(graph, "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1", id).await > 0
}
async fn fact_live(graph: &TemporalGraph, fact_id: i64) -> bool {
    count_i64(graph, "SELECT COUNT(*) FROM facts WHERE id = ?1", fact_id).await > 0
}
async fn fact_archived(graph: &TemporalGraph, fact_id: i64) -> bool {
    count_i64(graph, "SELECT COUNT(*) FROM facts_archive WHERE id = ?1", fact_id).await > 0
}
async fn edges_for(graph: &TemporalGraph, id: &str) -> i64 {
    count(graph, "SELECT COUNT(*) FROM episodic_edges WHERE entity_id = ?1", id).await
}

// ─── delete_entity — retract-on-zero precision (§4.4) ────────────────────────

#[tokio::test]
async fn delete_entity_cascade_retracts_facts() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    // E links to drop_me (drop_me's ONLY fact) and keep_me. keep_me has its own fact
    // to carol, so keep_me keeps support after E is deleted.
    for id in ["e", "drop_me", "keep_me", "carol"] {
        insert_bare(&graph, id).await;
    }
    let f_drop = plant_fact(&graph, "e", "knows", "drop_me").await;
    let f_keep = plant_fact(&graph, "e", "knows", "keep_me").await;
    plant_fact(&graph, "keep_me", "knows", "carol").await; // keep_me's own support
    link_episode(&graph, "e").await;
    set_community(&graph, "e", 1).await;
    set_community(&graph, "drop_me", 2).await;
    set_community(&graph, "keep_me", 3).await;

    let outcome = delete_entity(
        &graph,
        DeleteEntityParams {
            entity_id: "e".to_string(),
            group_id: GROUP.to_string(),
        },
    )
    .await
    .expect("delete entity");

    // ── honest outcome ──
    assert_eq!(outcome.entity_id, "e");
    assert_eq!(outcome.facts_retracted, 2, "both of E's facts archived");
    assert_eq!(outcome.edges_removed, 1, "E's episodic edge removed");
    assert_eq!(outcome.communities_removed, 1, "E's own community removed");
    assert_eq!(
        outcome.neighbors_retracted, 1,
        "only drop_me loses all support (retract-on-zero precision)"
    );
    assert!(!outcome.already_undone);
    assert!(outcome.mutation_id > 0, "an entity_delete log row was written");

    // ── facts RETRACTED (archived), NOT hard-deleted → still recoverable ──
    assert!(!fact_live(&graph, f_drop).await, "E's fact left the live table");
    assert!(!fact_live(&graph, f_keep).await);
    assert!(fact_archived(&graph, f_drop).await, "E's fact is recoverable in archive");
    assert!(fact_archived(&graph, f_keep).await);

    // ── entity + its derived artifacts gone ──
    assert!(!entity_exists(&graph, "e").await, "entity row deleted");
    assert!(!fts_exists(&graph, "e").await, "FTS shadow deleted");
    assert_eq!(edges_for(&graph, "e").await, 0, "episodic edges deleted");
    assert!(!community_exists(&graph, "e").await, "own community gone");

    // ── retract-on-zero precision: drop_me retracted, keep_me + carol kept ──
    assert!(!community_exists(&graph, "drop_me").await, "drop_me lost all support → retracted");
    assert!(community_exists(&graph, "keep_me").await, "keep_me keeps its own fact → NOT retracted");
    assert!(entity_exists(&graph, "drop_me").await, "base entity NEVER auto-deleted (HippoRAG #17)");
}

// ─── delete_entity → undo restores everything (§4.4) ─────────────────────────

#[tokio::test]
async fn delete_entity_undo_restores() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_bare(&graph, "e").await;
    insert_bare(&graph, "bob").await;
    let f = plant_fact(&graph, "e", "knows", "bob").await;
    link_episode(&graph, "e").await;
    set_community(&graph, "e", 7).await;

    let del = delete_entity(
        &graph,
        DeleteEntityParams {
            entity_id: "e".to_string(),
            group_id: GROUP.to_string(),
        },
    )
    .await
    .expect("delete");
    assert!(!entity_exists(&graph, "e").await);
    assert!(!fact_live(&graph, f).await);

    let undo = undo_delete_entity(&graph, del.mutation_id)
        .await
        .expect("undo delete");
    assert!(!undo.already_undone);
    assert_eq!(undo.facts_retracted, 1, "1 fact un-archived");
    assert_eq!(undo.edges_removed, 1, "1 episodic edge re-inserted");
    assert_eq!(undo.communities_removed, 1, "own community restored");

    // ── fully restored ──
    assert!(entity_exists(&graph, "e").await, "entity back");
    assert!(fts_exists(&graph, "e").await, "FTS back");
    assert!(fact_live(&graph, f).await, "fact back in live table");
    assert!(!fact_archived(&graph, f).await, "fact removed from archive on restore");
    assert_eq!(edges_for(&graph, "e").await, 1, "episodic edge back");
    assert!(community_exists(&graph, "e").await, "community membership back");

    // ── idempotent: second undo is a zero-count no-op ──
    let again = undo_delete_entity(&graph, del.mutation_id).await.expect("second undo");
    assert!(again.already_undone, "second undo is a no-op");
    assert_eq!(again.facts_retracted, 0);
}

// ─── delete_fact → retract-on-zero + undo (§4.5) ─────────────────────────────

#[tokio::test]
async fn delete_fact_retract_and_undo() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    for id in ["alice", "bob", "carol"] {
        insert_bare(&graph, id).await;
    }
    // alice's ONLY fact is f1 (alice—bob); bob also has f2 (bob—carol) → bob keeps support.
    let f1 = plant_fact(&graph, "alice", "knows", "bob").await;
    plant_fact(&graph, "bob", "knows", "carol").await;
    set_community(&graph, "alice", 11).await;
    set_community(&graph, "bob", 12).await;

    let out = delete_fact(&graph, f1).await.expect("delete fact");
    assert_eq!(out.fact_id, f1);
    assert_eq!(
        out.neighbors_retracted, 1,
        "only alice loses all support when f1 is deleted"
    );
    assert!(!out.already_undone);
    assert!(!fact_live(&graph, f1).await, "fact left the live table");
    assert!(fact_archived(&graph, f1).await, "fact recoverable in archive");
    assert!(!community_exists(&graph, "alice").await, "alice retracted-on-zero");
    assert!(community_exists(&graph, "bob").await, "bob keeps its own fact → NOT retracted");

    let undo = undo_delete_fact(&graph, out.mutation_id).await.expect("undo delete fact");
    assert!(!undo.already_undone);
    assert_eq!(undo.neighbors_retracted, 1, "alice community restored");
    assert!(fact_live(&graph, f1).await, "fact restored to live table");
    assert!(!fact_archived(&graph, f1).await, "fact removed from archive on restore");
    assert!(community_exists(&graph, "alice").await, "alice membership restored");

    let again = undo_delete_fact(&graph, out.mutation_id).await.expect("second undo");
    assert!(again.already_undone, "second undo is a no-op");
}

// ─── cascade cycle-safety (§4.1) ─────────────────────────────────────────────

#[tokio::test]
async fn delete_entity_cascade_terminates_on_cyclic_graph() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_bare(&graph, "a").await;
    insert_bare(&graph, "b").await;
    // A cycle: A—fact—B and B—fact—A. The recursive-CTE reachability MUST terminate
    // (UNION set-dedup + MAX_CASCADE_DEPTH). If it looped, this test would hang.
    let f_ab = plant_fact(&graph, "a", "knows", "b").await;
    let f_ba = plant_fact(&graph, "b", "knows", "a").await;
    set_community(&graph, "a", 1).await;
    set_community(&graph, "b", 2).await;

    let out = delete_entity(
        &graph,
        DeleteEntityParams {
            entity_id: "a".to_string(),
            group_id: GROUP.to_string(),
        },
    )
    .await
    .expect("delete on cyclic graph terminates");

    // Both facts referenced A → both archived; B loses all support → retracted.
    assert_eq!(out.facts_retracted, 2, "both cycle facts archived");
    assert_eq!(out.neighbors_retracted, 1, "B retracted-on-zero");
    assert!(!fact_live(&graph, f_ab).await);
    assert!(!fact_live(&graph, f_ba).await);
    assert!(fact_archived(&graph, f_ab).await);
    assert!(fact_archived(&graph, f_ba).await);
    assert!(!entity_exists(&graph, "a").await);
    assert!(!community_exists(&graph, "b").await, "B's derived community retracted");
    assert!(entity_exists(&graph, "b").await, "B base entity kept");
}

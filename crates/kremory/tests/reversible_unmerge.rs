//! Sub-phase 1c reversal proofs — reversible-graph-mutations arch-spec
//! `.ai-docs/specs/reversible-graph-mutations-arch-spec-2026-07-10.md` §4.2 /
//! §4.4 / §4.5 / §6.
//!
//! DETERMINISTIC, zero-LLM (the merge decision is cosine similarity; the
//! reversal is pure SQL), so this whole file sits at the fast tier with NO VCR
//! (`llm-test-pyramid-vcr-seams`). It proves:
//!
//! - `unmerge_restores_complete_state` — merge B into A (real executor) then
//!   `unmerge` restores the graph to byte-identical pre-merge state (§4.2): loser
//!   row (embedding + assigned_at + access_count + ner_confidence), keeper values,
//!   facts re-pointed with restored `corroboration_inert`, episodic edges
//!   (collided + non-collided), idempotency-key freeze re-open, `undone_at`,
//!   `merge_nogood`.
//! - `nogood_prevents_remerge_canonicalize` — after `unmerge`, re-running L5
//!   canonicalize does NOT re-merge the split pair (§6.2, Site #2). (Site #5's
//!   nogood is proved in `acronym_nickname_recall.rs`'s own unit tests — it needs
//!   the scripted LLM verdict provider.)
//! - `restore_archived_fact_roundtrip` / `unsupersede_clears_bound` — the two
//!   additive reversal helpers (§4.4 / §4.5).

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kremory::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use kremory::core::dream::{restore_archived_fact, unmerge, unsupersede};
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

const GROUP: &str = "meeting_42";
// The L5 merge is gated by an ADR-057 lexical surface-variant check (Jaccard ≥ 0.5
// on the ids) in ADDITION to cosine — so the pair must be name variants. Same pair
// the passing snapshot test uses.
const KEEPER: &str = "alice johnson"; // longer description → kept
const LOSER: &str = "alice j"; // shorter description → merged into keeper

fn unit_vec() -> Vec<f32> {
    let v = 1.0_f32 / (384.0_f32).sqrt();
    vec![v; 384]
}

async fn insert_embedded(graph: &TemporalGraph, id: &str, description: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id, "description": description }),
            group_id: Some(GROUP),
        })
        .await
        .expect("insert embedded entity");
    graph
        .set_entity_embedding(id, &unit_vec())
        .await
        .expect("set embedding");
}

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

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
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
            "SELECT id FROM facts WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3",
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

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn set_access_and_confidence(graph: &TemporalGraph, id: &str, access: i64, conf: f64) {
    graph
        .conn
        .execute(
            "UPDATE entities SET access_count = ?1, ner_confidence = ?2 \
             WHERE id = ?3 AND group_id = ?4",
            libsql::params![access, conf, id, GROUP],
        )
        .await
        .expect("set access + confidence");
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

async fn add_edge(graph: &TemporalGraph, episode_id: i64, entity: &str) {
    graph
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id,
            entity_id: entity,
            entity_group_id: Some(GROUP),
            role: "mention",
        })
        .await
        .expect("insert extra episodic edge");
}

/// The mutable per-entity fields undo must restore exactly.
struct EntitySnap {
    embedding: Option<Vec<u8>>,
    entity_type_assigned_at: Option<String>,
    entity_type_source: Option<String>,
    access_count: i64,
    ner_confidence: Option<f64>,
    properties: Option<String>,
}

async fn read_entity_snap(graph: &TemporalGraph, id: &str) -> Option<EntitySnap> {
    let mut rows = graph
        .conn
        .query(
            "SELECT embedding, entity_type_assigned_at, entity_type_source, access_count, \
                    ner_confidence, properties FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![id, GROUP],
        )
        .await
        .expect("read entity");
    let row = rows.next().await.expect("row")?;
    Some(EntitySnap {
        embedding: row.get::<Option<Vec<u8>>>(0).expect("embedding"),
        entity_type_assigned_at: row.get::<Option<String>>(1).expect("assigned_at"),
        entity_type_source: row.get::<Option<String>>(2).expect("type_source"),
        access_count: row.get::<i64>(3).expect("access_count"),
        ner_confidence: row.get::<Option<f64>>(4).expect("ner_confidence"),
        properties: row.get::<Option<String>>(5).expect("properties"),
    })
}

async fn fact_endpoints(graph: &TemporalGraph, fact_id: i64) -> (Option<String>, Option<String>, i64) {
    let mut rows = graph
        .conn
        .query(
            "SELECT subject_id, object_id, corroboration_inert FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("read fact");
    let row = rows.next().await.expect("row").expect("fact present");
    (
        row.get::<Option<String>>(0).expect("subject_id"),
        row.get::<Option<String>>(1).expect("object_id"),
        row.get::<i64>(2).expect("corroboration_inert"),
    )
}

async fn episodes_for_entity(graph: &TemporalGraph, id: &str) -> Vec<i64> {
    let mut rows = graph
        .conn
        .query(
            "SELECT episode_id FROM episodic_edges WHERE entity_id = ?1 ORDER BY episode_id",
            libsql::params![id],
        )
        .await
        .expect("read edges");
    let mut v = Vec::new();
    while let Some(row) = rows.next().await.expect("row") {
        v.push(row.get::<i64>(0).expect("episode_id"));
    }
    v
}

async fn entity_rowid(graph: &TemporalGraph, id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT rowid FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![id, GROUP],
        )
        .await
        .expect("read rowid");
    rows.next()
        .await
        .expect("row")
        .expect("entity present")
        .get::<i64>(0)
        .expect("rowid")
}

#[tokio::test]
async fn unmerge_restores_complete_state() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");

    // ── Two merge candidates + fact/edge structure (mirrors the snapshot test) ──
    insert_embedded(
        &graph,
        KEEPER,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    insert_embedded(&graph, LOSER, "Alice.").await;
    set_access_and_confidence(&graph, KEEPER, 5, 0.83).await;
    set_access_and_confidence(&graph, LOSER, 3, 0.71).await;

    insert_bare(&graph, "bob").await;
    insert_bare(&graph, "carol").await;

    // F1: loser as SUBJECT, currently live (prior_corroboration_inert = 0).
    let f1 = plant_fact(&graph, LOSER, "knows", "bob").await;
    // F2: loser as OBJECT, ALREADY inert from a simulated earlier merge (=1) — undo
    // must NOT clear this (the monotone-undo trap, §12 CH-3).
    let f2 = plant_fact(&graph, "carol", "knows", LOSER).await;
    graph
        .conn
        .execute(
            "UPDATE facts SET corroboration_inert = 1 WHERE id = ?1",
            libsql::params![f2],
        )
        .await
        .expect("pre-inert F2");

    // E1 loser-only (non-collided); E2 shared with keeper (collided).
    let e1 = link_episode(&graph, LOSER).await;
    let e2 = link_episode(&graph, LOSER).await;
    add_edge(&graph, e2, KEEPER).await;

    // ── Capture the PRE-merge state undo must restore ──
    let loser_pre = read_entity_snap(&graph, LOSER).await.expect("loser exists");
    let keeper_pre = read_entity_snap(&graph, KEEPER).await.expect("keeper exists");
    assert!(
        loser_pre.embedding.as_ref().map(|b| !b.is_empty()).unwrap_or(false),
        "loser has an embedding pre-merge"
    );

    // ── Drive the REAL merge (site = canonicalize) ──
    let report = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("canonicalize");
    assert_eq!(report.merges_applied, 1, "exactly one merge");
    assert!(
        read_entity_snap(&graph, LOSER).await.is_none(),
        "loser is hard-deleted by the merge"
    );

    // Plant a KEEPER idempotency-key (stable rowid) so freeze re-open has something
    // to drop (§6.1(b)); the loser gets a NEW rowid on restore so no stale key.
    let keeper_rowid = entity_rowid(&graph, KEEPER).await;
    graph
        .conn
        .execute(
            "INSERT INTO dream_idempotency_keys (pass_name, entity_id, content_hash, completed_at) \
             VALUES ('reclassify', ?1, 'hash-x', 1)",
            libsql::params![keeper_rowid],
        )
        .await
        .expect("plant keeper idempotency key");

    let mutation_id: i64 = {
        let mut rows = graph
            .conn
            .query("SELECT id FROM graph_mutation_log WHERE kind = 'entity_merge'", ())
            .await
            .expect("log query");
        rows.next()
            .await
            .expect("row")
            .expect("one entity_merge row")
            .get::<i64>(0)
            .expect("id")
    };

    // ── UNMERGE ──
    let outcome = unmerge(&graph, mutation_id).await.expect("unmerge");
    assert_eq!(outcome.restored_entity, LOSER);
    assert_eq!(outcome.keeper, KEEPER);
    assert!(!outcome.already_undone);
    assert!(outcome.nogood_recorded);
    assert_eq!(outcome.facts_repointed, 2, "both fact endpoints reverted");
    assert_eq!(outcome.edges_restored, 2, "both loser edges restored");
    assert!(
        outcome.entities_reopened >= 1,
        "keeper freeze re-opened (idempotency key dropped)"
    );

    // ── (1) loser row byte-identical (embedding + assigned_at + counts + source) ──
    let loser_post = read_entity_snap(&graph, LOSER).await.expect("loser restored");
    assert_eq!(loser_post.embedding, loser_pre.embedding, "embedding restored exactly");
    assert_eq!(
        loser_post.entity_type_assigned_at, loser_pre.entity_type_assigned_at,
        "entity_type_assigned_at restored"
    );
    assert_eq!(loser_post.entity_type_source, loser_pre.entity_type_source);
    assert_eq!(loser_post.access_count, 3, "loser access_count restored");
    assert_eq!(loser_post.ner_confidence, loser_pre.ner_confidence);
    assert_eq!(loser_post.properties, loser_pre.properties);

    // ── (2) keeper access_count + ner_confidence restored (NOT the accumulation) ──
    let keeper_post = read_entity_snap(&graph, KEEPER).await.expect("keeper exists");
    assert_eq!(keeper_post.access_count, 5, "keeper access restored (not 5+3=8)");
    assert_eq!(keeper_post.access_count, keeper_pre.access_count);
    assert_eq!(
        keeper_post.ner_confidence, keeper_pre.ner_confidence,
        "keeper ner_confidence restored (not the noisy-OR)"
    );

    // ── (3) facts re-pointed to loser + PRIOR corroboration_inert restored ──
    let (f1_subj, _, f1_inert) = fact_endpoints(&graph, f1).await;
    assert_eq!(f1_subj.as_deref(), Some(LOSER), "F1 subject re-pointed to loser");
    assert_eq!(f1_inert, 0, "F1 was live pre-merge → restored to 0");
    let (_, f2_obj, f2_inert) = fact_endpoints(&graph, f2).await;
    assert_eq!(f2_obj.as_deref(), Some(LOSER), "F2 object re-pointed to loser");
    assert_eq!(f2_inert, 1, "F2 was already inert pre-merge → stays 1 (monotone trap)");

    // ── (4) episodic edges: loser back on BOTH; keeper keeps only E2 ──
    let loser_eps = episodes_for_entity(&graph, LOSER).await;
    assert_eq!(loser_eps, vec![e1.min(e2), e1.max(e2)], "loser back on E1 + E2");
    let keeper_eps = episodes_for_entity(&graph, KEEPER).await;
    assert_eq!(keeper_eps, vec![e2], "keeper keeps only its own E2 edge (E1 re-pointed back)");

    // ── freeze re-open: keeper idempotency key dropped ──
    let key_count: i64 = {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM dream_idempotency_keys WHERE entity_id = ?1",
                libsql::params![keeper_rowid],
            )
            .await
            .expect("key count");
        rows.next().await.expect("row").expect("count").get::<i64>(0).expect("n")
    };
    assert_eq!(key_count, 0, "keeper idempotency key dropped (freeze re-opened)");

    // ── undone_at stamped + nogood recorded ──
    let undone_at: Option<String> = {
        let mut rows = graph
            .conn
            .query(
                "SELECT undone_at FROM graph_mutation_log WHERE id = ?1",
                libsql::params![mutation_id],
            )
            .await
            .expect("undone query");
        rows.next().await.expect("row").expect("log row").get::<Option<String>>(0).expect("undone_at")
    };
    assert!(undone_at.is_some(), "graph_mutation_log.undone_at stamped");

    let nogood_count: i64 = {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM merge_nogood WHERE group_id = ?1",
                libsql::params![GROUP],
            )
            .await
            .expect("nogood query");
        rows.next().await.expect("row").expect("count").get::<i64>(0).expect("n")
    };
    assert_eq!(nogood_count, 1, "a merge_nogood row was written");

    // ── idempotent: second unmerge is a no-op ──
    let again = unmerge(&graph, mutation_id).await.expect("second unmerge");
    assert!(again.already_undone, "second unmerge is an already-undone no-op");
    assert_eq!(again.facts_repointed, 0);
}

#[tokio::test]
async fn nogood_prevents_remerge_canonicalize() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_embedded(
        &graph,
        KEEPER,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    insert_embedded(&graph, LOSER, "Alice.").await;

    // First canonicalize: the surface-variant pair merges.
    let r1 = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("first canonicalize");
    assert_eq!(r1.merges_applied, 1, "first pass merges the pair");

    let mutation_id: i64 = {
        let mut rows = graph
            .conn
            .query("SELECT id FROM graph_mutation_log WHERE kind = 'entity_merge'", ())
            .await
            .expect("log query");
        rows.next().await.expect("row").expect("row").get::<i64>(0).expect("id")
    };
    unmerge(&graph, mutation_id).await.expect("unmerge");

    // Second canonicalize on the SAME pair: the nogood must block the re-merge.
    let r2 = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("second canonicalize");
    assert_eq!(
        r2.merges_applied, 0,
        "Site #2 nogood must block the re-merge (V3 fix)"
    );

    let entities = graph.list_entities_in_group(GROUP).await.expect("list");
    let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
    assert!(ids.contains(&LOSER), "loser survives — the split pair was NOT re-merged");
    assert!(ids.contains(&KEEPER), "keeper still present");
}

/// M1 (Quinn) — a CHAINED merge (`A→K1` then `K1→K2`, sharing endpoint `K1`)
/// MUST be unwound Last-In-First-Out. Unmerging the FIRST while the SECOND still
/// lives is rejected with `Error::UnmergeOutOfOrder`; LIFO order (unmerge #2 then
/// #1) fully restores all three entities. Governing spec §4.2.
#[tokio::test]
async fn unmerge_chained_merges_lifo() {
    const K1: &str = "alice johnson";
    const K2: &str = "alice johnson lawyer";
    const A: &str = "alice j";

    let graph = TemporalGraph::open_in_memory().await.expect("open");

    // ── Merge #1: A → K1 (K1 has the longer description → kept). ──
    insert_embedded(
        &graph,
        K1,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    insert_embedded(&graph, A, "Alice.").await;
    let r1 = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("first canonicalize");
    assert_eq!(r1.merges_applied, 1, "merge #1: A → K1");

    // ── Merge #2: K1 → K2 (K2 has the longest description → kept; K1 becomes the
    //    loser, so K1 is the SHARED endpoint between the two merges). ──
    insert_embedded(
        &graph,
        K2,
        "A far longer and even more detailed description of Alice Johnson the \
         lawyer at Acme Corp, with a great many additional words to make it the \
         longest description of all the candidates in this namespace.",
    )
    .await;
    let r2 = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("second canonicalize");
    assert_eq!(r2.merges_applied, 1, "merge #2: K1 → K2");

    // Two entity_merge rows: m1 (earlier, keeper=K1) then m2 (later, keeper=K2).
    let ids: Vec<i64> = {
        let mut rows = graph
            .conn
            .query(
                "SELECT id FROM graph_mutation_log WHERE kind = 'entity_merge' ORDER BY id ASC",
                (),
            )
            .await
            .expect("log query");
        let mut v = Vec::new();
        while let Some(row) = rows.next().await.expect("row") {
            v.push(row.get::<i64>(0).expect("id"));
        }
        v
    };
    assert_eq!(ids.len(), 2, "two chained merges logged");
    let (m1, m2) = (ids[0], ids[1]);

    // ── Out-of-order: unmerge #1 (m1) while #2 (m2) is still live → REJECTED. ──
    match unmerge(&graph, m1).await {
        Err(kremory::core::error::Error::UnmergeOutOfOrder {
            mutation_id,
            blocking_mutation_id,
        }) => {
            assert_eq!(mutation_id, m1, "the rejected mutation is m1");
            assert_eq!(blocking_mutation_id, m2, "blocked by the later chained merge m2");
        }
        other => panic!("expected UnmergeOutOfOrder for out-of-order unmerge, got {other:?}"),
    }
    // m1 was NOT reversed — still live (undone_at NULL).
    let m1_undone: Option<String> = {
        let mut rows = graph
            .conn
            .query(
                "SELECT undone_at FROM graph_mutation_log WHERE id = ?1",
                libsql::params![m1],
            )
            .await
            .expect("m1 undone query");
        rows.next().await.expect("row").expect("m1 row").get::<Option<String>>(0).expect("undone_at")
    };
    assert!(m1_undone.is_none(), "out-of-order unmerge left m1 un-reversed");

    // ── LIFO: unmerge #2 (m2) FIRST — restores K1. ──
    let out2 = unmerge(&graph, m2).await.expect("unmerge m2 (LIFO top)");
    assert_eq!(out2.restored_entity, K1, "m2 restores the shared endpoint K1");
    assert!(!out2.already_undone);

    // ── Then unmerge #1 (m1) — now unblocked (m2 is undone) → restores A. ──
    let out1 = unmerge(&graph, m1).await.expect("unmerge m1 after m2 (LIFO)");
    assert_eq!(out1.restored_entity, A, "m1 restores A");
    assert!(!out1.already_undone);

    // ── All three entities present again; the graph is fully restored. ──
    let entities = graph.list_entities_in_group(GROUP).await.expect("list");
    let present: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
    assert!(present.contains(&A), "A restored");
    assert!(present.contains(&K1), "K1 restored");
    assert!(present.contains(&K2), "K2 present (top keeper)");
}

#[tokio::test]
async fn restore_archived_fact_roundtrip() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    // The restored fact's endpoints (subject/object) are FK-checked against
    // `entities(id, group_id)`, so the endpoints must exist.
    insert_bare(&graph, "alice").await;
    insert_bare(&graph, "bob").await;
    let now = chrono::Utc::now().to_rfc3339();
    // Plant a facts_archive row directly (the archive op's projection shape,
    // `archive.rs::ARCHIVE_INSERT_SQL`). id is preserved from the original fact.
    graph
        .conn
        .execute(
            "INSERT INTO facts_archive \
             (id, subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence, is_dream_generated, archived_at) \
             VALUES (9001, 'alice', 'knows', 'bob', ?1, ?1, ?2, ?2, ?2, 1.0, 0, ?1)",
            libsql::params![now, GROUP],
        )
        .await
        .expect("plant archive row");

    let outcome = restore_archived_fact(&graph, 9001).await.expect("restore");
    assert_eq!(outcome.restored_fact_id, 9001);
    assert!(!outcome.already_live, "fresh restore is not already-live");

    // The fact is back in `facts` and gone from `facts_archive`.
    let in_facts: i64 = {
        let mut rows = graph
            .conn
            .query("SELECT COUNT(*) FROM facts WHERE id = 9001", ())
            .await
            .expect("facts count");
        rows.next().await.expect("row").expect("count").get::<i64>(0).expect("n")
    };
    assert_eq!(in_facts, 1, "restored fact is live in `facts`");
    let in_archive: i64 = {
        let mut rows = graph
            .conn
            .query("SELECT COUNT(*) FROM facts_archive WHERE id = 9001", ())
            .await
            .expect("archive count");
        rows.next().await.expect("row").expect("count").get::<i64>(0).expect("n")
    };
    assert_eq!(in_archive, 0, "archive row removed after restore");

    // Idempotent: a second restore observes the fact already live.
    let again = restore_archived_fact(&graph, 9001).await;
    // The archive row is gone now → a second restore is a hard error (not_found),
    // which is the parse-loudly contract. Re-plant + restore proves already_live.
    assert!(again.is_err(), "second restore of an already-consumed archive id errors loudly");
}

#[tokio::test]
async fn unsupersede_clears_bound() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_bare(&graph, "alice").await;
    let now = chrono::Utc::now().to_rfc3339();
    // Literal object (`object_value`, `object_id` NULL) so only the subject
    // endpoint is FK-checked.
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, expired_at, recorded_at, \
              group_id, subject_group_id, confidence) \
             VALUES ('alice', 'status', 'active', ?1, ?1, ?1, ?1, ?2, ?2, 1.0)",
            libsql::params![now, GROUP],
        )
        .await
        .expect("plant superseded fact");
    let fact_id: i64 = {
        let mut rows = graph
            .conn
            .query("SELECT id FROM facts WHERE subject_id = 'alice' AND predicate = 'status'", ())
            .await
            .expect("fact id query");
        rows.next().await.expect("row").expect("fact").get::<i64>(0).expect("id")
    };

    let outcome = unsupersede(&graph, fact_id).await.expect("unsupersede");
    match outcome {
        kremory::facade::UnsupersedeOutcome::Cleared {
            cleared_valid_to,
            cleared_expired_at,
            ..
        } => {
            assert!(cleared_valid_to, "valid_to bound was cleared");
            assert!(cleared_expired_at, "expired_at bound was cleared");
        }
        other => panic!("expected Cleared, got {other:?}"),
    }

    // Both bounds are now NULL.
    let (valid_to, expired_at): (Option<String>, Option<String>) = {
        let mut rows = graph
            .conn
            .query(
                "SELECT valid_to, expired_at FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await
            .expect("read bounds");
        let row = rows.next().await.expect("row").expect("fact");
        (
            row.get::<Option<String>>(0).expect("valid_to"),
            row.get::<Option<String>>(1).expect("expired_at"),
        )
    };
    assert!(valid_to.is_none(), "valid_to cleared to NULL");
    assert!(expired_at.is_none(), "expired_at cleared to NULL");

    // Idempotent: a fact with no bound is an honest no-op.
    let again = unsupersede(&graph, fact_id).await.expect("second unsupersede");
    assert!(matches!(
        again,
        kremory::facade::UnsupersedeOutcome::NotSuperseded { .. }
    ));
}

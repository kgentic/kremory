//! Sub-phase 1b V1-completeness proof — reversible-graph-mutations spec
//! `.ai-docs/specs/reversible-graph-mutations-arch-spec-2026-07-10.md` §2.3 /
//! §4 / §8.
//!
//! Drives a REAL entity merge through the public `canonicalize_surface_forms`
//! path (which calls the shared `apply_merge_with_audit` executor, site
//! `canonicalize`), then reads the `graph_mutation_log` row the snapshot wrote
//! and asserts the captured `pre_state` is COMPLETE — the whole loser row (incl.
//! `entity_type_assigned_at` + the embedding BLOB), the keeper's PRE-merge
//! `access_count` / `ner_confidence`, every re-pointed fact with the correct
//! `prior_corroboration_inert`, and the episodic edges with correct `collided`
//! flags. A missed field here = silent data loss on a future `unmerge`.
//!
//! Deterministic + zero-LLM (the merge decision is cosine similarity), so this
//! sits at the fast tier with NO VCR (`llm-test-pyramid-vcr-seams`).

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::Value;

use kremory::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

const GROUP: &str = "meeting_42";
// The L5 merge is gated by an ADR-057 lexical surface-variant check (Jaccard ≥
// 0.5 on the ids) in ADDITION to cosine — so the pair must be name variants, not
// arbitrary ids. `{alice}` ⊂ `{alice, johnson}` → Jaccard 0.5 (the exact pair the
// passing t4 cosine-merge test uses). Longer description → kept.
const KEEPER: &str = "alice johnson"; // longer description → kept by the LightRAG heuristic
const LOSER: &str = "alice j"; // shorter description → merged into the keeper

/// A unit-normalised 384-d vector (matches the default `embedding_dim`). Both
/// merge candidates share it so `vector_distance_cos == 0` → similarity 1.0 >
/// threshold, forcing exactly one merge.
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

/// A bare entity (NO embedding) — excluded from the cosine canonicalization scan
/// so it never becomes an accidental merge candidate; only serves as a fact
/// endpoint.
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

/// Plant a relational fact `subject --predicate--> object` directly via SQL.
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
            libsql::params![
                subject,
                predicate,
                object,
                now.clone(),
                now,
                GROUP,
                GROUP,
                GROUP
            ],
        )
        .await
        .expect("plant fact");
    let mut rows = graph
        .conn
        .query(
            "SELECT id FROM facts WHERE subject_id = ?1 AND predicate = ?2 AND \
             (object_id = ?3 OR (?3 IS NULL AND object_id IS NULL))",
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

#[tokio::test]
async fn merge_snapshot_captures_complete_pre_state() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");

    // ── Two merge candidates (identical embeddings; alice's description longer) ──
    insert_embedded(
        &graph,
        KEEPER,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    insert_embedded(&graph, LOSER, "Alice.").await;
    set_access_and_confidence(&graph, KEEPER, 5, 0.83).await;
    set_access_and_confidence(&graph, LOSER, 3, 0.71).await;

    // ── Fact endpoints (bare entities, excluded from the cosine scan) ──
    insert_bare(&graph, "bob").await;
    insert_bare(&graph, "carol").await;

    // F1: loser as SUBJECT, currently live (prior_corroboration_inert = 0).
    let f1 = plant_fact(&graph, LOSER, "knows", "bob").await;
    // F2: loser as OBJECT, ALREADY inert from a (simulated) earlier merge
    // (prior_corroboration_inert = 1) — undo must NOT clear this (monotone trap).
    let f2 = plant_fact(&graph, "carol", "knows", LOSER).await;
    graph
        .conn
        .execute(
            "UPDATE facts SET corroboration_inert = 1 WHERE id = ?1",
            libsql::params![f2],
        )
        .await
        .expect("pre-inert F2");

    // ── Episodic edges: E1 loser-only (non-collided); E2 shared (collided) ──
    let e1 = link_episode(&graph, LOSER).await; // keeper NOT on E1 → collided=false
    let e2 = link_episode(&graph, LOSER).await;
    add_edge(&graph, e2, KEEPER).await; // keeper ALSO on E2 → collided=true

    // ── Drive the REAL merge (site = canonicalize) ──
    let report = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("canonicalize");
    assert_eq!(report.merges_applied, 1, "exactly one merge expected");

    // ── Read the single graph_mutation_log row ──
    let (kind, log_group, pre_state_json, inputs_json) = {
        let mut rows = graph
            .conn
            .query(
                "SELECT kind, group_id, pre_state, inputs FROM graph_mutation_log",
                (),
            )
            .await
            .expect("query log");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("exactly one log row");
        let out = (
            row.get::<String>(0).expect("kind"),
            row.get::<String>(1).expect("group_id"),
            row.get::<String>(2).expect("pre_state"),
            row.get::<String>(3).expect("inputs"),
        );
        assert!(
            rows.next().await.expect("second").is_none(),
            "exactly one merge → exactly one log row"
        );
        out
    };
    assert_eq!(kind, "entity_merge");
    assert_eq!(log_group, GROUP);

    let pre: Value = serde_json::from_str(&pre_state_json).expect("pre_state is valid JSON");
    let inputs: Value = serde_json::from_str(&inputs_json).expect("inputs is valid JSON");

    // ── (1) loser entity row — complete, incl. assigned_at + embedding BLOB ──
    let loser_row = &pre["loser_entity_row"];
    assert_eq!(loser_row["id"], LOSER);
    assert_eq!(loser_row["group_id"], GROUP);
    assert_eq!(loser_row["access_count"], 3);
    assert!(
        (loser_row["ner_confidence"]
            .as_f64()
            .expect("loser ner_confidence")
            - 0.71)
            .abs()
            < 1e-6
    );
    assert!(
        loser_row["entity_type_assigned_at"].is_string(),
        "entity_type_assigned_at MUST be snapshotted (reclassify sets it) — got {:?}",
        loser_row["entity_type_assigned_at"]
    );
    assert!(
        loser_row["embedding_b64"].is_string() && !loser_row["embedding_b64"].as_str().unwrap().is_empty(),
        "embedding BLOB MUST be captured (base64) — a missed embedding is silent loss on undo; got {:?}",
        loser_row["embedding_b64"]
    );

    // ── (2) keeper PRE-merge values (before accumulate / noisy-OR overwrite) ──
    let keeper_pre = &pre["keeper_pre"];
    assert_eq!(keeper_pre["id"], KEEPER);
    assert_eq!(
        keeper_pre["access_count"], 5,
        "keeper access_count must be the PRE-merge value (not the 5+3 accumulation)"
    );
    assert!((keeper_pre["ner_confidence"].as_f64().expect("keeper ner") - 0.83).abs() < 1e-6);

    // ── (3) re-pointed facts — each id + endpoint + PRIOR inert flag ──
    let facts = pre["repointed_facts"]
        .as_array()
        .expect("repointed_facts array");
    assert_eq!(facts.len(), 2, "both endpoints captured: {facts:?}");
    let subj = facts
        .iter()
        .find(|f| f["fact_id"].as_i64() == Some(f1))
        .expect("F1 (subject endpoint) captured");
    assert_eq!(subj["endpoint"], "subject");
    assert_eq!(
        subj["prior_corroboration_inert"], 0,
        "F1 was live before the merge — prior inert must be 0"
    );
    let obj = facts
        .iter()
        .find(|f| f["fact_id"].as_i64() == Some(f2))
        .expect("F2 (object endpoint) captured");
    assert_eq!(obj["endpoint"], "object");
    assert_eq!(
        obj["prior_corroboration_inert"], 1,
        "F2 was ALREADY inert from an earlier merge — prior inert must be 1 (monotone-undo trap)"
    );

    // ── (4) episodic edges — correct collided flags ──
    let edges = pre["episodic_edges"]
        .as_array()
        .expect("episodic_edges array");
    assert_eq!(edges.len(), 2, "both loser edges captured: {edges:?}");
    let e1_snap = edges
        .iter()
        .find(|e| e["episode_id"].as_i64() == Some(e1))
        .expect("E1 edge captured");
    assert_eq!(
        e1_snap["collided"], false,
        "E1 is loser-only → keeper had no edge → non-collided (undo re-points back)"
    );
    assert_eq!(e1_snap["cols"]["entity_id"], LOSER);
    let e2_snap = edges
        .iter()
        .find(|e| e["episode_id"].as_i64() == Some(e2))
        .expect("E2 edge captured");
    assert_eq!(
        e2_snap["collided"], true,
        "E2 is shared with the keeper → collided (undo re-INSERTs the dropped loser edge)"
    );

    // ── inputs — site + sorted nogood pair ──
    assert_eq!(inputs["site"], "canonicalize");
    assert_eq!(inputs["keeper"], KEEPER);
    assert_eq!(inputs["loser"], LOSER);
    // sort("alice j", "alice johnson"): the shorter prefix sorts first → pair_lo
    // is the loser, pair_hi the keeper (the sorted-pair nogood key, §6.2).
    assert_eq!(inputs["pair_lo"], LOSER);
    assert_eq!(inputs["pair_hi"], KEEPER);
}

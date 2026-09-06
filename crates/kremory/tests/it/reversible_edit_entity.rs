//! Tier-2a EDIT-ENTITY cascade proofs — reversible-graph-mutations arch-spec
//! `.ai-docs/specs/reversible-graph-mutations-arch-spec-2026-07-10.md` §4.3.
//!
//! DETERMINISTIC, zero-LLM (rename/retype are pure SQL; the diarization merge is
//! cosine similarity), so this whole file sits at the fast tier with NO VCR
//! (`llm-test-pyramid-vcr-seams`). It proves:
//!
//! - `edit_entity_rename_propagates_all_fks` — a rename re-points EVERY entity-id
//!   FK (facts subject+object, facts_archive, episodic_edges, entity_communities,
//!   entities PK + FTS) with ZERO dangling old-id references (§4.3 FK-off caveat).
//! - `edit_entity_rename_into_existing_rejected` — renaming INTO an occupied id is
//!   a structured `EntityEditConflict`, never a silent fuse (ADR-059).
//! - `edit_entity_retype` — a retype changes the type + pins `ConsumerPinned` +
//!   invalidates community membership, without touching facts.
//! - `edit_entity_retype_rejects_out_of_range_type_id` /
//!   `edit_entity_retype_rejects_unregistered_gap_type_id` — a CONSUMER-supplied
//!   `new_type_id` that no `entity_types` row explains is a loud
//!   `EntityEditInvalid` naming the id, with NO write and NO provenance row —
//!   never coerced to the catch-all (that is the LLM sites' semantics) and never
//!   written dangling into the FK-less `entities.entity_type_id`.
//! - `edit_entity_retype_accepts_registered_type_id` /
//!   `edit_entity_retype_accepts_catch_all_zero` — the false-positive guards: a
//!   registered id still applies, and id=0 ("Entity") is always admissible, even
//!   on an unseeded namespace, so a mis-typed entity can be demoted.
//! - `edit_entity_undo_restores` — undo reverses a rename (inverse rekey) and a
//!   retype (type + community restore) exactly; second undo is a no-op.
//! - `diarization_merge_unmerge_rename` — the end-to-end driver: merge "Speaker 1"
//!   away, unmerge it, rename it to "Alice", and assert Alice owns Speaker 1's fact.

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use kremory::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use kremory::core::dream::{
    edit_entity, undo_entity_edit, unmerge, EntityEditOp, EntityEditParams,
};
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

const GROUP: &str = "meeting_42";

fn unit_vec() -> Vec<f32> {
    let v = 1.0_f32 / (384.0_f32).sqrt();
    vec![v; 384]
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
    last_fact_id(graph, subject, predicate, object).await
}

async fn last_fact_id(graph: &TemporalGraph, subject: &str, predicate: &str, object: &str) -> i64 {
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

async fn plant_archived_fact(graph: &TemporalGraph, archive_id: i64, subject: &str, object: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts_archive \
             (id, subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence, is_dream_generated, archived_at) \
             VALUES (?1, ?2, 'knows', ?3, ?4, ?4, ?5, ?5, ?5, 1.0, 0, ?4)",
            libsql::params![archive_id, subject, object, now, GROUP],
        )
        .await
        .expect("plant archive row");
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

/// Total references to `id` across EVERY entity-id FK surface. Zero after a rename
/// away from the old id proves the cascade left no dangling reference (§4.3).
async fn dangling_refs(graph: &TemporalGraph, id: &str) -> i64 {
    let facts = count(
        graph,
        "SELECT COUNT(*) FROM facts WHERE subject_id = ?1 OR object_id = ?1",
        id,
    )
    .await;
    let archive = count(
        graph,
        "SELECT COUNT(*) FROM facts_archive WHERE subject_id = ?1 OR object_id = ?1",
        id,
    )
    .await;
    let edges = count(
        graph,
        "SELECT COUNT(*) FROM episodic_edges WHERE entity_id = ?1",
        id,
    )
    .await;
    let comms = count(
        graph,
        "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1",
        id,
    )
    .await;
    let ent = count(graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", id).await;
    let fts = count(
        graph,
        "SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?1",
        id,
    )
    .await;
    facts + archive + edges + comms + ent + fts
}

async fn entity_type(graph: &TemporalGraph, id: &str) -> (i64, Option<String>) {
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_id, entity_type_source FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![id, GROUP],
        )
        .await
        .expect("read type");
    let row = rows.next().await.expect("row").expect("entity present");
    (
        row.get::<i64>(0).expect("type_id"),
        row.get::<Option<String>>(1).expect("type_source"),
    )
}

const OLD: &str = "speaker 1";
const NEW: &str = "alice";

#[tokio::test]
async fn edit_entity_rename_propagates_all_fks() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_bare(&graph, OLD).await;
    insert_bare(&graph, "bob").await;
    insert_bare(&graph, "carol").await;

    let f_subj = plant_fact(&graph, OLD, "knows", "bob").await; // OLD as SUBJECT
    let f_obj = plant_fact(&graph, "carol", "knows", OLD).await; // OLD as OBJECT
    plant_archived_fact(&graph, 7001, OLD, "bob").await; // archived, OLD as subject
    let ep = link_episode(&graph, OLD).await;
    set_community(&graph, OLD, 42).await;

    let outcome = edit_entity(
        &graph,
        EntityEditParams {
            entity_id: OLD.to_string(),
            group_id: GROUP.to_string(),
            op: EntityEditOp::Rename {
                new_id: NEW.to_string(),
            },
        },
    )
    .await
    .expect("rename");

    // ── outcome honesty ──
    assert!(outcome.rekeyed, "rename is a rekey");
    assert!(!outcome.retyped);
    assert_eq!(outcome.entity_id, NEW);
    assert_eq!(outcome.facts_repointed, 2, "both fact endpoints re-pointed");
    assert_eq!(
        outcome.archived_repointed, 1,
        "archived fact subject re-pointed"
    );
    assert_eq!(outcome.edges_repointed, 1, "episodic edge re-pointed");
    assert_eq!(
        outcome.communities_repointed, 1,
        "community membership re-pointed"
    );
    assert!(
        outcome.mutation_id > 0,
        "an entity_edit log row was written"
    );

    // ── EVERY FK now points at NEW; ZERO dangling OLD references ──
    assert_eq!(
        dangling_refs(&graph, OLD).await,
        0,
        "no dangling old-id refs anywhere"
    );

    let (subj, _) = fact_endpoints(&graph, f_subj).await;
    assert_eq!(
        subj.as_deref(),
        Some(NEW),
        "subject fact re-pointed to new id"
    );
    let (_, obj) = fact_endpoints(&graph, f_obj).await;
    assert_eq!(
        obj.as_deref(),
        Some(NEW),
        "object fact re-pointed to new id"
    );

    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM facts_archive WHERE subject_id = ?1",
            NEW
        )
        .await,
        1,
        "archived fact re-pointed (V5)"
    );
    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM episodic_edges WHERE entity_id = ?1",
            NEW
        )
        .await,
        1,
        "episodic edge re-pointed"
    );
    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1",
            NEW
        )
        .await,
        1,
        "community membership re-pointed"
    );
    assert_eq!(
        count(&graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", NEW).await,
        1,
        "entity PK row renamed"
    );
    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?1",
            NEW
        )
        .await,
        1,
        "FTS shadow rebuilt under new id"
    );
    let _ = ep;
}

async fn fact_endpoints(graph: &TemporalGraph, fact_id: i64) -> (Option<String>, Option<String>) {
    let mut rows = graph
        .conn
        .query(
            "SELECT subject_id, object_id FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("read fact");
    let row = rows.next().await.expect("row").expect("fact present");
    (
        row.get::<Option<String>>(0).expect("subject_id"),
        row.get::<Option<String>>(1).expect("object_id"),
    )
}

#[tokio::test]
async fn edit_entity_rename_into_existing_rejected() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_bare(&graph, OLD).await;
    insert_bare(&graph, NEW).await; // target already occupied

    let err = edit_entity(
        &graph,
        EntityEditParams {
            entity_id: OLD.to_string(),
            group_id: GROUP.to_string(),
            op: EntityEditOp::Rename {
                new_id: NEW.to_string(),
            },
        },
    )
    .await
    .expect_err("rename into existing must be rejected");

    match err {
        kremory::core::error::Error::EntityEditConflict {
            from,
            existing,
            group_id,
        } => {
            assert_eq!(from, OLD);
            assert_eq!(existing, NEW);
            assert_eq!(group_id, GROUP);
        }
        other => panic!("expected EntityEditConflict, got {other:?}"),
    }

    // Both entities survive — no silent fuse.
    assert_eq!(
        count(&graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", OLD).await,
        1
    );
    assert_eq!(
        count(&graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", NEW).await,
        1
    );
    // No mutation-log row was written (the conflict short-circuits before the snapshot).
    let logged: i64 = {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM graph_mutation_log WHERE kind = 'entity_edit'",
                (),
            )
            .await
            .expect("log count");
        rows.next()
            .await
            .expect("row")
            .expect("count")
            .get::<i64>(0)
            .expect("n")
    };
    assert_eq!(logged, 0, "no entity_edit row on a rejected rename");
}

#[tokio::test]
async fn edit_entity_retype() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    // Seed the namespace's entity-type registry, as `Engine::ingest_with` does for
    // any namespace a consumer really ingests into. These tests plant entities via
    // the low-level `insert_entity_with_group`, which bypasses ingest — so without
    // this the registry is EMPTY and `retype(3)` was writing an id no `entity_types`
    // row explains (the FK-less-column hole the retype guard now closes).
    seed_default_types(&graph).await;
    insert_bare(&graph, NEW).await;
    insert_bare(&graph, "bob").await;
    let fact = plant_fact(&graph, NEW, "knows", "bob").await;
    set_community(&graph, NEW, 9).await;

    let outcome = edit_entity(
        &graph,
        EntityEditParams {
            entity_id: NEW.to_string(),
            group_id: GROUP.to_string(),
            op: EntityEditOp::Retype { new_type_id: 3 },
        },
    )
    .await
    .expect("retype");

    assert!(outcome.retyped);
    assert!(!outcome.rekeyed);
    assert_eq!(outcome.facts_repointed, 0, "retype does NOT touch facts");
    assert_eq!(
        outcome.communities_repointed, 1,
        "community membership invalidated"
    );

    let (type_id, source) = entity_type(&graph, NEW).await;
    assert_eq!(type_id, 3, "entity_type_id changed");
    assert_eq!(
        source.as_deref(),
        Some("ConsumerPinned"),
        "retype pins ConsumerPinned"
    );
    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1",
            NEW
        )
        .await,
        0,
        "community membership dropped so detection re-places the re-typed entity"
    );
    // The fact is untouched (id unchanged → no rekey).
    let (subj, _) = fact_endpoints(&graph, fact).await;
    assert_eq!(
        subj.as_deref(),
        Some(NEW),
        "fact endpoint unchanged by retype"
    );
}

#[tokio::test]
async fn edit_entity_undo_restores() {
    // ── Scenario A: rename → undo (inverse rekey) ──
    {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_bare(&graph, OLD).await;
        insert_bare(&graph, "bob").await;
        let f = plant_fact(&graph, OLD, "knows", "bob").await;
        plant_archived_fact(&graph, 8001, OLD, "bob").await;
        link_episode(&graph, OLD).await;
        set_community(&graph, OLD, 11).await;

        let out = edit_entity(
            &graph,
            EntityEditParams {
                entity_id: OLD.to_string(),
                group_id: GROUP.to_string(),
                op: EntityEditOp::Rename {
                    new_id: NEW.to_string(),
                },
            },
        )
        .await
        .expect("rename");
        let mid = out.mutation_id;
        assert_eq!(
            dangling_refs(&graph, OLD).await,
            0,
            "old id fully vacated by rename"
        );

        let undo = undo_entity_edit(&graph, mid).await.expect("undo rename");
        assert!(undo.rekeyed);
        assert_eq!(undo.entity_id, OLD, "undo restores the old id");

        // Everything back on OLD; NEW fully vacated.
        assert_eq!(
            dangling_refs(&graph, NEW).await,
            0,
            "new id fully vacated by undo"
        );
        let (subj, _) = fact_endpoints(&graph, f).await;
        assert_eq!(
            subj.as_deref(),
            Some(OLD),
            "fact subject restored to old id"
        );
        assert_eq!(
            count(
                &graph,
                "SELECT COUNT(*) FROM facts_archive WHERE subject_id = ?1",
                OLD
            )
            .await,
            1,
            "archived fact restored to old id"
        );
        assert_eq!(
            count(
                &graph,
                "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1",
                OLD
            )
            .await,
            1,
            "community membership restored to old id"
        );
        assert_eq!(
            count(&graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", OLD).await,
            1
        );

        // undone_at stamped; second undo is a no-op.
        let undone_at: Option<String> = {
            let mut rows = graph
                .conn
                .query(
                    "SELECT undone_at FROM graph_mutation_log WHERE id = ?1",
                    libsql::params![mid],
                )
                .await
                .expect("undone query");
            rows.next()
                .await
                .expect("row")
                .expect("row")
                .get::<Option<String>>(0)
                .expect("col")
        };
        assert!(undone_at.is_some(), "undone_at stamped");
        let again = undo_entity_edit(&graph, mid).await.expect("second undo");
        assert_eq!(
            again.facts_repointed, 0,
            "second undo is a zero-count no-op"
        );
    }

    // ── Scenario B: retype → undo (type + community restore) ──
    {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Registry seeded as real ingest would (see `edit_entity_retype`) — id 4
        // must be a REGISTERED type for the forward retype to be legal.
        seed_default_types(&graph).await;
        insert_bare(&graph, NEW).await;
        set_community(&graph, NEW, 5).await;

        let out = edit_entity(
            &graph,
            EntityEditParams {
                entity_id: NEW.to_string(),
                group_id: GROUP.to_string(),
                op: EntityEditOp::Retype { new_type_id: 4 },
            },
        )
        .await
        .expect("retype");
        assert_eq!(entity_type(&graph, NEW).await.0, 4);

        // Pre-retype source is whatever the insert stamped (Phase1Ner) — undo must
        // restore that EXACT prior value, not blanket-NULL.
        let (pre_type, pre_source) = entity_type(&graph, NEW).await;
        assert_eq!(pre_type, 4, "sanity: retype applied before undo");
        let _ = pre_source;
        undo_entity_edit(&graph, out.mutation_id)
            .await
            .expect("undo retype");
        let (type_id, source) = entity_type(&graph, NEW).await;
        assert_eq!(type_id, 0, "type restored to pre-retype value");
        assert_eq!(
            source.as_deref(),
            Some("Phase1Ner"),
            "entity_type_source restored verbatim"
        );
        assert_eq!(
            count(
                &graph,
                "SELECT COUNT(*) FROM entity_communities WHERE entity_id = ?1",
                NEW
            )
            .await,
            1,
            "community membership restored on retype undo"
        );
    }
}

/// The Tier-2a driver: merge "Speaker 1" away, unmerge it, rename it to "Alice",
/// and assert Alice owns Speaker 1's fact (diarization end-to-end, §9.2). ONE
/// integration path — it does not re-enumerate the unit corpus (no ice-cream cone).
#[tokio::test]
async fn diarization_merge_unmerge_rename() {
    const KEEPER: &str = "speaker 1 lead"; // longer description → kept by canonicalize
    const SPEAKER: &str = "speaker 1"; // the loser; later renamed to alice

    let graph = TemporalGraph::open_in_memory().await.expect("open");
    insert_embedded(
        &graph,
        KEEPER,
        "The lead speaker who opened the meeting and drove the budget discussion at length.",
    )
    .await;
    insert_embedded(&graph, SPEAKER, "A speaker.").await;
    insert_bare(&graph, "bob").await;

    // Speaker 1 asserts a fact.
    let fact = plant_fact(&graph, SPEAKER, "knows", "bob").await;

    // ── Merge: canonicalize fuses "speaker 1" into "speaker 1 lead" ──
    let report = canonicalize_surface_forms(&graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("canonicalize");
    assert_eq!(report.merges_applied, 1, "the speaker pair merged");
    let (subj_after_merge, _) = fact_endpoints(&graph, fact).await;
    assert_eq!(
        subj_after_merge.as_deref(),
        Some(KEEPER),
        "fact re-pointed to keeper by merge"
    );

    // ── Unmerge: Speaker 1 is restored, its fact re-pointed back ──
    let mid: i64 = {
        let mut rows = graph
            .conn
            .query(
                "SELECT id FROM graph_mutation_log WHERE kind = 'entity_merge'",
                (),
            )
            .await
            .expect("log query");
        rows.next()
            .await
            .expect("row")
            .expect("row")
            .get::<i64>(0)
            .expect("id")
    };
    unmerge(&graph, mid).await.expect("unmerge");
    let (subj_after_unmerge, _) = fact_endpoints(&graph, fact).await;
    assert_eq!(
        subj_after_unmerge.as_deref(),
        Some(SPEAKER),
        "fact re-pointed back to Speaker 1"
    );

    // ── Rename: Speaker 1 → Alice; the fact follows ──
    let outcome = edit_entity(
        &graph,
        EntityEditParams {
            entity_id: SPEAKER.to_string(),
            group_id: GROUP.to_string(),
            op: EntityEditOp::Rename {
                new_id: NEW.to_string(),
            },
        },
    )
    .await
    .expect("rename speaker → alice");
    assert!(outcome.rekeyed);
    assert_eq!(
        outcome.facts_repointed, 1,
        "Speaker 1's fact re-keyed to Alice"
    );

    // ── Assert: Alice owns Speaker 1's fact; Speaker 1 is gone; keeper untouched ──
    let (subj_final, _) = fact_endpoints(&graph, fact).await;
    assert_eq!(subj_final.as_deref(), Some(NEW), "Alice owns the fact");
    assert_eq!(
        dangling_refs(&graph, SPEAKER).await,
        0,
        "Speaker 1 fully vacated"
    );
    assert_eq!(
        count(&graph, "SELECT COUNT(*) FROM entities WHERE id = ?1", NEW).await,
        1,
        "Alice exists"
    );
    assert_eq!(
        count(
            &graph,
            "SELECT COUNT(*) FROM entities WHERE id = ?1",
            KEEPER
        )
        .await,
        1,
        "the keeper is untouched by the rename"
    );
}

// ── Retype registry-bounds guard (consumer-supplied `new_type_id`) ───────────
//
// `EntityEditOp::Retype { new_type_id }` arrives from the PUBLIC builder
// `mem.edit_entity(id).retype(n)` and is written straight to
// `entities.entity_type_id`, a column with NO foreign key (Migration 008,
// `defs_b.rs`) — so the database cannot reject a bogus id. Unlike the two LLM
// sites (`dream::reclassify`, `dream::consistency_check::audit`) which coerce an
// unusable id to the catch-all and CONTINUE, a consumer handing us an id no
// registry row explains is a CALLER BUG: it must fail loudly rather than be
// silently coerced (parse-loudly) or written dangling.

/// Seed the standard `DEFAULT_ENTITY_TYPES` vocabulary (ids 0..=9) for `GROUP`,
/// mirroring what `Engine::ingest_with` does lazily (`ingest_with.rs` step 2a)
/// for every namespace a consumer actually ingests into. Tests that plant
/// entities via the low-level `insert_entity_with_group` bypass ingest, so they
/// must seed the registry themselves to be a faithful fixture.
async fn seed_default_types(graph: &TemporalGraph) {
    kremory::core::entity_types::ensure_default_types_seeded(&graph.conn, GROUP)
        .await
        .expect("seed default entity types");
}

/// Count `entity_edit` provenance rows — a rejected edit must write none.
async fn entity_edit_log_rows(graph: &TemporalGraph) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM graph_mutation_log WHERE kind = 'entity_edit'",
            (),
        )
        .await
        .expect("log count");
    rows.next()
        .await
        .expect("row")
        .expect("count")
        .get::<i64>(0)
        .expect("n")
}

async fn retype(
    graph: &TemporalGraph,
    entity_id: &str,
    new_type_id: i64,
) -> kremory::CoreResult<()> {
    edit_entity(
        graph,
        EntityEditParams {
            entity_id: entity_id.to_string(),
            group_id: GROUP.to_string(),
            op: EntityEditOp::Retype { new_type_id },
        },
    )
    .await
    .map(|_| ())
}

/// An id ABOVE the highest registered type is refused with a descriptive error
/// naming the offending id — never silently coerced to the catch-all, and never
/// written dangling into the FK-less `entities.entity_type_id`.
#[tokio::test]
async fn edit_entity_retype_rejects_out_of_range_type_id() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    seed_default_types(&graph).await;
    insert_bare(&graph, NEW).await;

    let err = retype(&graph, NEW, 99)
        .await
        .expect_err("an unregistered, out-of-range type id must be rejected");

    match err {
        kremory::core::error::Error::EntityEditInvalid { detail } => {
            assert!(
                detail.contains("99"),
                "error must NAME the offending id so the caller can fix it; got: {detail}"
            );
            assert!(
                detail.contains(GROUP),
                "error must name the namespace whose registry was consulted; got: {detail}"
            );
        }
        other => panic!("expected EntityEditInvalid, got {other:?}"),
    }

    // The write never happened — no silent fallback to the catch-all, no dangling id.
    let (type_id, source) = entity_type(&graph, NEW).await;
    assert_eq!(type_id, 0, "entity_type_id untouched by a rejected retype");
    assert_ne!(
        source.as_deref(),
        Some("ConsumerPinned"),
        "a rejected retype must not pin the type"
    );
    assert_eq!(
        entity_edit_log_rows(&graph).await,
        0,
        "no entity_edit provenance row on a rejected retype"
    );
}

/// An id WITHIN range but absent from the registry (a gap) is refused too —
/// `<= max_id` is not the invariant; registry membership is.
#[tokio::test]
async fn edit_entity_retype_rejects_unregistered_gap_type_id() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    // Sparse registry: ids 0, 1 and 7 registered — 3 is a GAP below max_id.
    kremory::core::entity_types::upsert_entity_types(
        &graph.conn,
        GROUP,
        &[
            kremory::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            kremory::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A named individual.".to_string(),
            },
            kremory::EntityTypeSpec {
                id: 7,
                name: "Quantity".to_string(),
                description: "A measurement with units.".to_string(),
            },
        ],
    )
    .await
    .expect("seed sparse registry");
    insert_bare(&graph, NEW).await;

    let err = retype(&graph, NEW, 3)
        .await
        .expect_err("an id inside the range but absent from the registry must be rejected");

    match err {
        kremory::core::error::Error::EntityEditInvalid { detail } => {
            assert!(
                detail.contains('3'),
                "error must NAME the offending id; got: {detail}"
            );
        }
        other => panic!("expected EntityEditInvalid, got {other:?}"),
    }

    let (type_id, _) = entity_type(&graph, NEW).await;
    assert_eq!(type_id, 0, "entity_type_id untouched by a rejected retype");
    assert_eq!(
        entity_edit_log_rows(&graph).await,
        0,
        "no entity_edit provenance row on a rejected retype"
    );
}

/// A legitimately registered id still applies — the guard rejects the unknown,
/// not the legal (`over-blocking-is-a-security-failure`: prove sensitivity in
/// BOTH directions, not just on the cases that should be blocked).
#[tokio::test]
async fn edit_entity_retype_accepts_registered_type_id() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    seed_default_types(&graph).await;
    insert_bare(&graph, NEW).await;

    // 3 = "Location" in DEFAULT_ENTITY_TYPES, and 7 is the sparse-registry id
    // used above — both are genuinely registered here.
    retype(&graph, NEW, 3).await.expect("registered id applies");

    let (type_id, source) = entity_type(&graph, NEW).await;
    assert_eq!(type_id, 3, "registered entity_type_id applied");
    assert_eq!(
        source.as_deref(),
        Some("ConsumerPinned"),
        "a legal retype still pins ConsumerPinned"
    );
    assert_eq!(
        entity_edit_log_rows(&graph).await,
        1,
        "a legal retype still writes its provenance row"
    );
}

/// id=0 ("Entity") is the catch-all sentinel — ALWAYS a legal retype target, so
/// a consumer can demote a mis-typed entity back to unclassified. It is admitted
/// unconditionally (matching `EntityTypeRegistry::validate_or_fallback`'s own
/// "id=0 always passes" invariant and the §5.8 guarantee that `ensure_catch_all`
/// puts id=0 in every seeded namespace), so it holds even on an UNSEEDED one.
#[tokio::test]
async fn edit_entity_retype_accepts_catch_all_zero() {
    // Seeded namespace: 3 → 0 is a real, observable demotion.
    {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        seed_default_types(&graph).await;
        insert_bare(&graph, NEW).await;
        retype(&graph, NEW, 3).await.expect("retype to Location");
        assert_eq!(entity_type(&graph, NEW).await.0, 3, "sanity: 3 applied");

        retype(&graph, NEW, 0)
            .await
            .expect("demote to the id=0 catch-all must be legal");
        assert_eq!(
            entity_type(&graph, NEW).await.0,
            0,
            "entity demoted back to the catch-all"
        );
    }

    // UNSEEDED namespace: id=0 bypasses the registry lookup entirely.
    {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_bare(&graph, NEW).await;
        retype(&graph, NEW, 0)
            .await
            .expect("id=0 is admissible without consulting the registry");
        let (type_id, source) = entity_type(&graph, NEW).await;
        assert_eq!(type_id, 0);
        assert_eq!(
            source.as_deref(),
            Some("ConsumerPinned"),
            "the retype ran (it pinned the type), it was not skipped"
        );
    }
}

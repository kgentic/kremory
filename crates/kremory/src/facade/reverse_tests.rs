//! `mem.undo(mutation_id)` dispatches each of the six
//! LOGGED kinds to the correct per-kind undo (same effect as the per-kind
//! method), returns `MutationNotFound` for an unknown id, and returns a loud
//! `UndoUnsupportedKind` for a would-be RESERVED-kind row. Deterministic,
//! zero-LLM — fast tier, no VCR (`llm-test-pyramid-vcr-seams`).

use std::sync::Arc;

use crate::core::canonicalization::{
    canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD,
};
use crate::core::error::Error as CoreError;
use crate::core::graph::InsertEntityWithGroupParams;
use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};

use super::{namespace_to_group_id, MemoryError, Namespace, UndoOutcome};
use crate::facade::Memory;

async fn make_memory() -> Memory {
    let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .await
        .expect("Memory must build")
}

fn ns() -> Namespace {
    Namespace::new("agent")
}

fn unit_vec() -> Vec<f32> {
    let v = 1.0_f32 / (384.0_f32).sqrt();
    vec![v; 384]
}

async fn insert_bare(mem: &Memory, id: &str) {
    let group = namespace_to_group_id(&ns());
    mem.temporal_graph
        .as_ref()
        .unwrap()
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(group.as_str()),
        })
        .await
        .expect("insert bare entity");
}

async fn insert_embedded(mem: &Memory, id: &str, description: &str) {
    let group = namespace_to_group_id(&ns());
    let tg = mem.temporal_graph.as_ref().unwrap();
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id,
        entity_type_id: 0,
        properties: serde_json::json!({ "name": id, "description": description }),
        group_id: Some(group.as_str()),
    })
    .await
    .expect("insert embedded entity");
    tg.set_entity_embedding(id, &unit_vec())
        .await
        .expect("set embedding");
}

/// Drive a real canonicalize merge and return the logged `entity_merge` id.
async fn make_merge(mem: &Memory) -> i64 {
    let group = namespace_to_group_id(&ns());
    insert_embedded(
        mem,
        "alice johnson",
        "A detailed description of Alice Johnson, engineer at Acme.",
    )
    .await;
    insert_embedded(mem, "alice j", "Alice.").await;
    let report = canonicalize_surface_forms(
        mem.temporal_graph.as_ref().unwrap(),
        &group,
        L5_CANONICALIZATION_THRESHOLD,
    )
    .await
    .expect("canonicalize");
    assert_eq!(report.merges_applied, 1, "exactly one merge produced");
    let mut rows = mem
        .temporal_graph
        .as_ref()
        .unwrap()
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
        .expect("one entity_merge row")
        .get::<i64>(0)
        .expect("id")
}

#[tokio::test]
async fn undo_routes_entity_merge_to_unmerge() {
    let mem = make_memory().await;
    let mutation_id = make_merge(&mem).await;

    let outcome = mem.undo(mutation_id).execute().await.expect("undo merge");
    match outcome {
        UndoOutcome::Unmerge(o) => {
            // Same effect as `mem.unmerge(id)`: the loser is restored + a nogood
            // is recorded so the next dream() will not re-merge the pair.
            assert_eq!(o.restored_entity, "alice j", "loser restored");
            assert_eq!(o.keeper, "alice johnson", "keeper named");
            assert!(
                o.nogood_recorded,
                "unmerge records the anti-re-merge nogood"
            );
            assert!(!o.already_undone, "first undo is a real reversal");
        }
        other => panic!("expected Unmerge, got {other:?}"),
    }
}

#[tokio::test]
async fn undo_routes_entity_edit_to_undo_entity_edit() {
    let mem = make_memory().await;
    insert_bare(&mem, "speaker 1").await;
    let edit = mem
        .edit_entity("speaker 1")
        .rename("alice")
        .in_namespace(ns())
        .execute()
        .await
        .expect("rename");
    assert!(edit.rekeyed, "rename rekeys");

    let outcome = mem
        .undo(edit.mutation_id)
        .execute()
        .await
        .expect("undo edit");
    match outcome {
        UndoOutcome::EditEntity(o) => {
            // Same effect as `mem.undo_entity_edit(id)`: the prior id is restored.
            assert_eq!(
                o.entity_id, "speaker 1",
                "rename undo restores the prior id"
            );
            assert!(!o.already_undone, "first undo is a real reversal");
        }
        other => panic!("expected EditEntity, got {other:?}"),
    }
}

#[tokio::test]
async fn undo_routes_entity_delete_to_undo_delete_entity() {
    let mem = make_memory().await;
    insert_bare(&mem, "bob").await;
    let del = mem
        .delete_entity("bob")
        .in_namespace(ns())
        .execute()
        .await
        .expect("delete entity");

    let outcome = mem
        .undo(del.mutation_id)
        .execute()
        .await
        .expect("undo delete");
    match outcome {
        UndoOutcome::DeleteEntity(o) => {
            assert_eq!(o.entity_id, "bob", "delete undo names the restored entity");
            assert!(!o.already_undone, "first undo is a real reversal");
        }
        other => panic!("expected DeleteEntity, got {other:?}"),
    }
}

#[tokio::test]
async fn undo_routes_fact_delete_to_undo_delete_fact() {
    let mem = make_memory().await;
    let group = namespace_to_group_id(&ns());
    insert_bare(&mem, "carol").await;
    insert_bare(&mem, "dave").await;
    let now = chrono::Utc::now().to_rfc3339();
    mem.temporal_graph
        .as_ref()
        .unwrap()
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence) \
             VALUES ('carol', 'knows', 'dave', ?1, ?1, ?2, ?2, ?2, 1.0)",
            libsql::params![now, group.clone()],
        )
        .await
        .expect("plant fact");
    let fact_id: i64 = {
        let mut rows = mem
            .temporal_graph
            .as_ref()
            .unwrap()
            .conn
            .query(
                "SELECT id FROM facts WHERE subject_id = 'carol' AND object_id = 'dave'",
                (),
            )
            .await
            .expect("fact id query");
        rows.next()
            .await
            .expect("row")
            .expect("fact")
            .get::<i64>(0)
            .expect("id")
    };

    let del = mem
        .delete_fact(fact_id)
        .execute()
        .await
        .expect("delete fact");
    let outcome = mem
        .undo(del.mutation_id)
        .execute()
        .await
        .expect("undo delete fact");
    match outcome {
        UndoOutcome::DeleteFact(o) => {
            assert_eq!(
                o.fact_id, fact_id,
                "delete-fact undo names the restored fact"
            );
            assert!(o.fact_restored, "the archived fact was moved back to live");
            assert!(!o.already_undone, "first undo is a real reversal");
        }
        other => panic!("expected DeleteFact, got {other:?}"),
    }
}

#[tokio::test]
async fn undo_unknown_id_is_mutation_not_found() {
    let mem = make_memory().await;
    let err = mem
        .undo(999_999)
        .execute()
        .await
        .expect_err("unknown id must error");
    match err {
        MemoryError::Core(CoreError::MutationNotFound { mutation_id }) => {
            assert_eq!(mutation_id, 999_999);
        }
        other => panic!("expected MutationNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn undo_reserved_kind_is_unsupported_kind() {
    let mem = make_memory().await;
    let group = namespace_to_group_id(&ns());
    let now = chrono::Utc::now().to_rfc3339();
    // Plant a would-be `community_assign` row directly — nothing writes this
    // kind to the log today (the reserved-kind boundary), so `undo` must
    // refuse it loudly rather than dispatch. `fact_supersede` is NOT this
    // example any more — it became a LOGGED, dispatchable kind on 2026-09-14
    // (see `undo_routes_fact_supersede_to_unsupersede` below).
    mem.temporal_graph
        .as_ref()
        .unwrap()
        .conn
        .execute(
            "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
             VALUES ('community_assign', ?1, ?2, '{}', '{}')",
            libsql::params![group, now],
        )
        .await
        .expect("plant reserved-kind row");
    let planted_id: i64 = {
        let mut rows = mem
            .temporal_graph
            .as_ref()
            .unwrap()
            .conn
            .query(
                "SELECT id FROM graph_mutation_log WHERE kind = 'community_assign'",
                (),
            )
            .await
            .expect("planted id query");
        rows.next()
            .await
            .expect("row")
            .expect("planted row")
            .get::<i64>(0)
            .expect("id")
    };

    let err = mem
        .undo(planted_id)
        .execute()
        .await
        .expect_err("reserved kind must error");
    match err {
        MemoryError::Core(CoreError::UndoUnsupportedKind { mutation_id, kind }) => {
            assert_eq!(mutation_id, planted_id);
            assert_eq!(
                kind, "community_assign",
                "the offending kind tag is surfaced"
            );
        }
        other => panic!("expected UndoUnsupportedKind, got {other:?}"),
    }
}

/// AC — `mem.undo(mutation_id)` on a LOGGED `fact_supersede` row dispatches to
/// the same `unsupersede` mechanism the domain-id door uses, and reports it
/// through `UndoOutcome::Unsupersede` — proves the 2026-09-14 fix that made a
/// supersede retraction listable + undoable by `mutation_id` (removing the
/// need for a sixth MCP tool), the same uniform way as every other logged
/// kind.
#[tokio::test]
async fn undo_routes_fact_supersede_to_unsupersede() {
    let mem = make_memory().await;
    let group_ns = ns();
    let group = namespace_to_group_id(&group_ns);
    let now = chrono::Utc::now();
    let valid_from = now - chrono::Duration::days(10);

    let tg = mem.temporal_graph.as_ref().unwrap();
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "entity-undo-supersede-subject",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some(group.as_str()),
    })
    .await
    .expect("seed subject entity");
    let fact_id = tg
        .insert_fact_with_group(
            crate::core::graph::FactInsert::new(
                "entity-undo-supersede-subject",
                "status",
                valid_from,
            )
            .object_value("active"),
            Some(group.as_str()),
        )
        .await
        .expect("seed fact");

    let bound_at = now - chrono::Duration::days(1);
    let outcome = mem
        .supersede(fact_id)
        .in_namespace(group_ns.clone())
        .at(bound_at)
        .execute()
        .await
        .expect("supersede must succeed");
    assert!(matches!(
        outcome,
        crate::facade::SupersedeOutcome::Bounded { retired: 0 }
    ));

    let mutation_id = {
        let records = mem
            .list_mutations()
            .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
            .in_namespace(group_ns.clone())
            .await
            .expect("list_mutations must succeed");
        assert_eq!(
            records.len(),
            1,
            "the supersede must be listable by kind — this is the whole fix"
        );
        records[0].mutation_id
    };

    let undo_outcome = mem
        .undo(mutation_id)
        .execute()
        .await
        .expect("undo of a fact_supersede row must succeed, not UndoUnsupportedKind");
    match undo_outcome {
        UndoOutcome::Unsupersede(
            crate::core::dream::provenance::UnsupersedeOutcome::Cleared {
                fact_id: cleared_id,
                cleared_valid_to,
                ..
            },
        ) => {
            assert_eq!(cleared_id, fact_id);
            assert!(cleared_valid_to, "the bound this test set must be cleared");
        }
        other => panic!("expected UndoOutcome::Unsupersede(Cleared), got {other:?}"),
    }

    let fact = tg
        .get_fact_by_id(fact_id, &group)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist");
    assert_eq!(
        fact.valid_to, None,
        "undo must clear the DB row's valid_to bound"
    );
}

/// Regression for the HIGH-severity gap an adversarial review caught before
/// this shipped: a fact superseded TWICE has a second `fact_supersede` row
/// whose `prior_valid_to` is the FIRST bound, not `NULL`. Undoing the FIRST
/// (older) mutation while the SECOND (newer) one is still live must be
/// REFUSED, not silently clobber the live newer bound — `unsupersede`'s
/// unconditional NULL-both would have destroyed it while reporting success.
#[tokio::test]
async fn undo_fact_supersede_out_of_order_is_rejected() {
    let mem = make_memory().await;
    let group_ns = ns();
    let group = namespace_to_group_id(&group_ns);
    let now = chrono::Utc::now();
    let valid_from = now - chrono::Duration::days(30);

    let tg = mem.temporal_graph.as_ref().unwrap();
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "entity-undo-ooo-subject",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some(group.as_str()),
    })
    .await
    .expect("seed subject entity");
    let fact_id = tg
        .insert_fact_with_group(
            crate::core::graph::FactInsert::new(
                "entity-undo-ooo-subject",
                "status",
                valid_from,
            )
            .object_value("active"),
            Some(group.as_str()),
        )
        .await
        .expect("seed fact");

    // First supersede: None -> t1.
    let t1 = now - chrono::Duration::days(20);
    mem.supersede(fact_id)
        .in_namespace(group_ns.clone())
        .at(t1)
        .execute()
        .await
        .expect("first supersede must succeed");
    let first_mutation_id = {
        let records = mem
            .list_mutations()
            .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
            .in_namespace(group_ns.clone())
            .await
            .expect("list_mutations after first supersede");
        assert_eq!(records.len(), 1);
        records[0].mutation_id
    };

    // Second supersede: t1 -> t2 (a later, narrower bound).
    let t2 = now - chrono::Duration::days(10);
    mem.supersede(fact_id)
        .in_namespace(group_ns.clone())
        .at(t2)
        .execute()
        .await
        .expect("second supersede must succeed");

    // Undoing the FIRST (older) mutation must be refused — the second bound
    // is still live and would otherwise be silently destroyed.
    let err = mem
        .undo(first_mutation_id)
        .execute()
        .await
        .expect_err("undoing an out-of-order supersede must error, not succeed");
    match err {
        MemoryError::Core(CoreError::UndoStale { mutation_id, .. }) => {
            assert_eq!(mutation_id, first_mutation_id);
        }
        other => panic!("expected UndoStale, got {other:?}"),
    }

    // The live (second) bound must be UNTOUCHED by the rejected undo attempt.
    let fact = tg
        .get_fact_by_id(fact_id, &group)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist");
    assert_eq!(
        fact.valid_to.map(|t| t.timestamp()),
        Some(t2.timestamp()),
        "the still-live second bound must survive the rejected out-of-order undo"
    );
}

/// Companion to the out-of-order test: undoing the fact's supersedes
/// newest-first must RESTORE each row's captured prior state, not blind-clear
/// to `None` — proves `undo_fact_supersede` reads `prior_valid_to` back
/// rather than delegating to `unsupersede`'s unconditional NULL-both.
#[tokio::test]
async fn undo_fact_supersede_restores_prior_bound_not_blind_clear() {
    let mem = make_memory().await;
    let group_ns = ns();
    let group = namespace_to_group_id(&group_ns);
    let now = chrono::Utc::now();
    let valid_from = now - chrono::Duration::days(30);

    let tg = mem.temporal_graph.as_ref().unwrap();
    tg.insert_entity_with_group(InsertEntityWithGroupParams {
        id: "entity-undo-lifo-subject",
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some(group.as_str()),
    })
    .await
    .expect("seed subject entity");
    let fact_id = tg
        .insert_fact_with_group(
            crate::core::graph::FactInsert::new(
                "entity-undo-lifo-subject",
                "status",
                valid_from,
            )
            .object_value("active"),
            Some(group.as_str()),
        )
        .await
        .expect("seed fact");

    let t1 = now - chrono::Duration::days(20);
    mem.supersede(fact_id)
        .in_namespace(group_ns.clone())
        .at(t1)
        .execute()
        .await
        .expect("first supersede must succeed");

    let t2 = now - chrono::Duration::days(10);
    mem.supersede(fact_id)
        .in_namespace(group_ns.clone())
        .at(t2)
        .execute()
        .await
        .expect("second supersede must succeed");

    let second_mutation_id = {
        let records = mem
            .list_mutations()
            .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
            .in_namespace(group_ns.clone())
            .await
            .expect("list_mutations after second supersede");
        assert_eq!(records.len(), 2);
        // Newest-first (undone rows excluded by default) — the live row with
        // the LATEST created_at is the second supersede.
        records
            .iter()
            .max_by_key(|r| r.created_at.clone())
            .expect("at least one record")
            .mutation_id
    };

    // Undo the SECOND (latest) mutation — must restore to t1, the captured
    // `prior_valid_to`, NOT to `None`.
    let outcome = mem
        .undo(second_mutation_id)
        .execute()
        .await
        .expect("undo of the latest supersede must succeed");
    match outcome {
        UndoOutcome::Unsupersede(
            crate::core::dream::provenance::UnsupersedeOutcome::Cleared {
                fact_id: cleared_id,
                ..
            },
        ) => assert_eq!(cleared_id, fact_id),
        other => panic!("expected UndoOutcome::Unsupersede(Cleared), got {other:?}"),
    }

    let fact = tg
        .get_fact_by_id(fact_id, &group)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist");
    assert_eq!(
        fact.valid_to.map(|t| t.timestamp()),
        Some(t1.timestamp()),
        "undoing the second supersede must restore the FIRST bound, not wipe to None"
    );
}

#[tokio::test]
async fn undo_wrong_namespace_guard_rejects() {
    let mem = make_memory().await;
    let mutation_id = make_merge(&mem).await;
    // The merge was logged under `ns()`; scoping the undo to a DIFFERENT
    // namespace must refuse loudly rather than reverse it.
    let err = mem
        .undo(mutation_id)
        .in_namespace(Namespace::new("some-other-namespace"))
        .execute()
        .await
        .expect_err("cross-namespace undo must error");
    assert!(
        matches!(err, MemoryError::Core(CoreError::UndoWrongNamespace { .. })),
        "expected UndoWrongNamespace, got {err:?}"
    );
}

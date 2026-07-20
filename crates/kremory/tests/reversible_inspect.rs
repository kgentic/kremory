//! Tier-1 consumer INSPECT-surface proofs — reversible-graph-mutations arch-spec
//! `.ai-docs/specs/reversible-graph-mutations-arch-spec-2026-07-10.md` §3
//! "Inspect surface".
//!
//! DETERMINISTIC, zero-LLM (the merge decision is cosine similarity; the inspect
//! query is pure SQL over `graph_mutation_log`), so this file sits at the fast
//! tier with NO VCR (`llm-test-pyramid-vcr-seams`). It proves the SEE half of the
//! see+fix story:
//!
//! - `mutation_history_lists_merge` — a real canonicalize merge produces a
//!   `MutationRecord` that `mutation_history(loser)` returns with the correct
//!   summary + `undone = false`; after `unmerge`, the same record shows
//!   `undone = true` (the entity is still in its history).
//! - `list_mutations_filters` — the `kind` and `include_undone` filters select
//!   the expected rows.

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing lints.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kremory::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use kremory::core::dream::{list_mutations, mutation_history, unmerge};
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::schema::TemporalGraph;
use kremory::facade::{MutationFilter, MutationKind};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

/// Sum a counter's recorded value across all label sets from a local metrics snapshot.
fn counter_total(
    snapshot: &[(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
) -> u64 {
    snapshot
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == name)
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => Some(*c),
            _ => None,
        })
        .sum()
}

const GROUP: &str = "meeting_42";
// Same surface-variant pair the reversal tests use — L5 canonicalize needs name
// variants (ADR-057 lexical Jaccard ≥ 0.5) in addition to cosine.
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

/// Drive the real canonicalize merge (site = canonicalize) and return the
/// `graph_mutation_log.id` of the resulting `entity_merge` row.
async fn merge_and_log_id(graph: &TemporalGraph) -> i64 {
    insert_embedded(
        graph,
        KEEPER,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    insert_embedded(graph, LOSER, "Alice.").await;

    let report = canonicalize_surface_forms(graph, GROUP, L5_CANONICALIZATION_THRESHOLD)
        .await
        .expect("canonicalize");
    assert_eq!(report.merges_applied, 1, "exactly one merge");

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
        .expect("one entity_merge row")
        .get::<i64>(0)
        .expect("id")
}

#[tokio::test]
async fn mutation_history_lists_merge() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let mutation_id = merge_and_log_id(&graph).await;

    // ── SEE (pre-undo): the merge is in the loser's history, not-undone. ──
    let hist = mutation_history(&graph, LOSER, GROUP)
        .await
        .expect("mutation_history");
    assert_eq!(hist.len(), 1, "exactly one mutation touched the loser");
    let rec = &hist[0];
    assert_eq!(
        rec.mutation_id, mutation_id,
        "record carries the undo mutation_id"
    );
    assert_eq!(rec.kind, MutationKind::EntityMerge);
    assert!(!rec.undone, "merge is live (not yet reversed)");
    assert_eq!(rec.group_id, GROUP);
    // affected_entities = [keeper, loser] — both are locatable by id.
    assert!(
        rec.affected_entities.contains(&KEEPER.to_string()),
        "keeper present"
    );
    assert!(
        rec.affected_entities.contains(&LOSER.to_string()),
        "loser present"
    );
    assert_eq!(
        rec.summary,
        format!("merged '{LOSER}' into '{KEEPER}' (site=canonicalize)"),
        "human-readable summary derived from inputs (no raw JSON)"
    );

    // The KEEPER is also in the pair, so its history includes the same merge.
    let keeper_hist = mutation_history(&graph, KEEPER, GROUP)
        .await
        .expect("keeper history");
    assert_eq!(keeper_hist.len(), 1, "keeper is in the merge pair too");
    assert_eq!(keeper_hist[0].mutation_id, mutation_id);

    // ── FIX then SEE: after unmerge the record shows undone = true. ──
    unmerge(&graph, mutation_id).await.expect("unmerge");
    let hist_after = mutation_history(&graph, LOSER, GROUP)
        .await
        .expect("mutation_history after unmerge");
    assert_eq!(
        hist_after.len(),
        1,
        "the record is still in history after undo"
    );
    assert!(hist_after[0].undone, "record now shows undone = true");
    assert_eq!(hist_after[0].mutation_id, mutation_id);
}

/// F4 (o11y honesty) — one `mutation_history` consumer call must emit EXACTLY one
/// `inspect_query_total` increment. Before the fix `mutation_history` called
/// `list_mutations` internally, firing BOTH `op=list_mutations` AND
/// `op=mutation_history` for a single consumer call (double-count). The shared
/// counter-free `query_mutations` helper collapses it back to one.
#[tokio::test]
async fn mutation_history_emits_single_inspect_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let _mutation_id = merge_and_log_id(&graph).await;

    // Exactly ONE consumer call.
    let _ = mutation_history(&graph, LOSER, GROUP)
        .await
        .expect("mutation_history");

    let snapshot = snapshotter.snapshot().into_vec();
    let n = counter_total(&snapshot, "kremory.graph.inspect_query_total");
    assert_eq!(
        n, 1,
        "one mutation_history() call must emit exactly one inspect_query_total \
         increment (not double-counted via the internal list_mutations query); got {n}"
    );
}

#[tokio::test]
async fn list_mutations_filters() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let mutation_id = merge_and_log_id(&graph).await;

    // Default filter (include_undone = false) → the live merge is listed.
    let live = list_mutations(&graph, MutationFilter::default())
        .await
        .expect("list live");
    assert_eq!(live.len(), 1, "one live mutation");
    assert_eq!(live[0].mutation_id, mutation_id);
    assert_eq!(live[0].kind, MutationKind::EntityMerge);

    // kind filter that MATCHES → returns the row.
    let by_kind = list_mutations(
        &graph,
        MutationFilter {
            kind: Some(MutationKind::EntityMerge),
            ..MutationFilter::default()
        },
    )
    .await
    .expect("list by matching kind");
    assert_eq!(by_kind.len(), 1, "entity_merge kind filter matches");

    // kind filter that does NOT match → empty.
    let by_other_kind = list_mutations(
        &graph,
        MutationFilter {
            kind: Some(MutationKind::FactArchive),
            ..MutationFilter::default()
        },
    )
    .await
    .expect("list by non-matching kind");
    assert!(
        by_other_kind.is_empty(),
        "fact_archive kind filter excludes the merge"
    );

    // group filter that does NOT match → empty.
    let other_group = list_mutations(
        &graph,
        MutationFilter {
            group_id: Some("some_other_group".to_string()),
            ..MutationFilter::default()
        },
    )
    .await
    .expect("list other group");
    assert!(other_group.is_empty(), "group filter scopes the list");

    // After unmerge: default (live-only) filter EXCLUDES it; include_undone SHOWS it.
    unmerge(&graph, mutation_id).await.expect("unmerge");
    let live_after = list_mutations(&graph, MutationFilter::default())
        .await
        .expect("list live after undo");
    assert!(
        live_after.is_empty(),
        "default filter hides the undone mutation"
    );

    let all_after = list_mutations(
        &graph,
        MutationFilter {
            include_undone: true,
            ..MutationFilter::default()
        },
    )
    .await
    .expect("list include_undone after undo");
    assert_eq!(
        all_after.len(),
        1,
        "include_undone surfaces the reversed mutation"
    );
    assert!(all_after[0].undone, "and it is marked undone");
}

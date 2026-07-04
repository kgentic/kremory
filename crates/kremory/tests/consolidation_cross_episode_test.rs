//! ADR-066 dream CONSOLIDATION P3 — cross-episode-merge deterministic-corpus harness.
//!
//! Spec `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md` §3
//! (P3.1–P3.4, incl. P3.1b) + §6 (cross_episode corpus, REVISED) + ADR-066 §2.2. The
//! cross_episode op is ZERO-LLM, so — like the P1 supersession + P2 archive harnesses
//! and UNLIKE site3 — there is NO VCR cassette; each corpus row is a fully-deterministic
//! fixture. Mirrored from `consolidation_archive_test.rs`:
//!
//! - **per-row group isolation** — each row plants into its own fresh `group_id`;
//! - **smoke-one-before-batch** — `smoke_one_cross_episode_harness` runs ONE
//!   representative MERGE row (`xep-001`) through the full pipeline FIRST;
//! - **o11y cross-check** — the pass's own
//!   `kremory.dream.consolidation.cross_episode_merges_total{path}` counter is
//!   snapshotted and asserted EQUAL to the harness-derived merge count (DoD-P3.4).
//!
//! **Hard safety gate (spec §6 / RISK-001):** ZERO false merges on ALL no-merge
//! categories (`same_label_distinct_referent`, `fuzzy_without_corroboration`,
//! `cosine_near_dup_lexically_distinct`, `same_label_same_episode`), asserted EXACTLY
//! (never `>=`). The `same_label_distinct_referent` category is THE homonym-safety
//! category (two distinct real referents sharing a name, no shared structure).

#![cfg(feature = "test-utils")]
// Test binary — CLAUDE.md rule 5 exempts test files from the strict-typing +
// args-as-object lints (plant helpers carry explicit temporal/structural columns).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use chrono::Utc;

use kremory::core::dream::cross_episode;
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── Corpus row ─────────────────────────────────────────────────────────────

/// Parsed corpus row. `entity_a`/`entity_b` are two DISTINCT raw ids (the entity id
/// IS the identity comparand — `entities.label` was dropped by Migration 009). For
/// merge categories they normalize identically (exact) or clear Jaccard ≥0.9 (fuzzy).
/// `episodes_a`/`episodes_b` are the episode ORDINALS each is anchored to (mapped to
/// real episode ids planted per row); a shared ordinal = same episode. `neighbour_a`/
/// `neighbour_b` name the corroborating third entity each links via a fact — EQUAL
/// values create shared structure (merge), DISTINCT values create a homonym (no merge).
struct Row {
    id: String,
    category: String,
    entity_a: String,
    entity_b: String,
    episodes_a: Vec<i64>,
    episodes_b: Vec<i64>,
    neighbour_a: String,
    neighbour_b: String,
    expected_merge: bool,
    /// `"exact"` | `"fuzzy"` | `"none"` — the path the merge (if any) fires on.
    path: String,
}

fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("consolidation_cross_episode_adversarial.jsonl");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read corpus at {path:?}: {e}"));

    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {line_no} failed to parse: {e}"));

        // Parse loudly — required fields carry NO default (a missing field is a corpus
        // authoring bug, surfaced as a panic, not a silent default).
        let str_field = |k: &str| -> String {
            raw[k]
                .as_str()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing str '{k}'"))
                .to_string()
        };
        let bool_field = |k: &str| -> bool {
            raw[k]
                .as_bool()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing bool '{k}'"))
        };
        let i64_vec_field = |k: &str| -> Vec<i64> {
            raw[k]
                .as_array()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing array '{k}'"))
                .iter()
                .map(|v| {
                    v.as_i64().unwrap_or_else(|| {
                        panic!("corpus line {line_no}: '{k}' has non-i64 element")
                    })
                })
                .collect()
        };

        rows.push(Row {
            id: str_field("id"),
            category: str_field("category"),
            entity_a: str_field("entity_a"),
            entity_b: str_field("entity_b"),
            episodes_a: i64_vec_field("episodes_a"),
            episodes_b: i64_vec_field("episodes_b"),
            neighbour_a: str_field("neighbour_a"),
            neighbour_b: str_field("neighbour_b"),
            expected_merge: bool_field("expected_merge"),
            path: str_field("path"),
        });
    }
    assert!(!rows.is_empty(), "corpus must not be empty");
    rows
}

// ─── Per-row isolated plant + run ───────────────────────────────────────────

/// Outcome of running one corpus row through the real cross_episode sweep.
struct RowOutcome {
    row_id: String,
    category: String,
    expected_merge: bool,
    /// Did the sweep actually merge the pair (entity count dropped by 1)?
    merged: bool,
    op_count: usize,
    entities_before: i64,
    entities_after: i64,
    /// Which of the two candidate slugs survived (both survive when no merge).
    a_survives: bool,
    b_survives: bool,
}

async fn insert_entity(graph: &TemporalGraph, gid: &str, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
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

async fn fact_rel(graph: &TemporalGraph, gid: &str, subject: &str, predicate: &str, object: &str) {
    let now = Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
            libsql::params![subject, predicate, object, now.clone(), now, gid, gid, gid],
        )
        .await
        .expect("plant relational fact");
}

async fn entity_ids(graph: &TemporalGraph, gid: &str) -> Vec<String> {
    graph
        .list_entities_in_group(gid)
        .await
        .expect("list")
        .into_iter()
        .map(|e| e.id)
        .collect()
}

async fn entity_count(graph: &TemporalGraph, gid: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = ?1",
            libsql::params![gid],
        )
        .await
        .expect("count entities");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("count")
}

/// Plant one corpus row into a fresh isolated group and run `cross_episode`.
///
/// Each row plants: the two candidate entities, their (distinct or shared) episode
/// anchors, and one corroborating relational fact each to `neighbour_a`/`neighbour_b`.
/// EQUAL neighbours ⇒ shared structure (merge-eligible); DISTINCT neighbours ⇒ homonym
/// (no shared structure). The neighbour entities are planted on demand.
async fn run_one_row(row: &Row) -> RowOutcome {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let gid = format!("xep-corpus-{}", row.id);

    // Plant the two candidate entities (distinct raw ids).
    insert_entity(&graph, &gid, &row.entity_a).await;
    insert_entity(&graph, &gid, &row.entity_b).await;

    // Map episode ordinals → real episode ids. The union of ordinals across both
    // candidates determines how many distinct episodes exist; a shared ordinal maps to
    // the SAME episode id (the `same_label_same_episode` category relies on this).
    let mut ordinals: Vec<i64> = row
        .episodes_a
        .iter()
        .chain(row.episodes_b.iter())
        .copied()
        .collect();
    ordinals.sort_unstable();
    ordinals.dedup();
    let mut episode_of: std::collections::BTreeMap<i64, i64> = std::collections::BTreeMap::new();
    for ord in ordinals {
        episode_of.insert(ord, new_episode(&graph).await);
    }
    for ord in &row.episodes_a {
        anchor(&graph, &gid, episode_of[ord], &row.entity_a).await;
    }
    for ord in &row.episodes_b {
        anchor(&graph, &gid, episode_of[ord], &row.entity_b).await;
    }

    // Corroborating facts: each candidate → its neighbour. When the two neighbour
    // names are EQUAL, a shared third entity exists (structural corroboration);
    // otherwise the structures are disjoint (homonym).
    insert_entity(&graph, &gid, &row.neighbour_a).await;
    if row.neighbour_b != row.neighbour_a {
        insert_entity(&graph, &gid, &row.neighbour_b).await;
    }
    fact_rel(&graph, &gid, &row.entity_a, "works_at", &row.neighbour_a).await;
    fact_rel(&graph, &gid, &row.entity_b, "works_at", &row.neighbour_b).await;

    let entities_before = entity_count(&graph, &gid).await;
    let report = cross_episode(&graph, &gid).await.expect("cross_episode");
    let entities_after = entity_count(&graph, &gid).await;

    let survivors = entity_ids(&graph, &gid).await;
    RowOutcome {
        row_id: row.id.clone(),
        category: row.category.clone(),
        expected_merge: row.expected_merge,
        merged: report.count > 0,
        op_count: report.count,
        entities_before,
        entities_after,
        a_survives: survivors.contains(&row.entity_a),
        b_survives: survivors.contains(&row.entity_b),
    }
}

// ─── o11y cross-check ───────────────────────────────────────────────────────

/// Sum the cross_episode merge counter (both `{path}` labels) across the snapshot.
fn sum_merge_counter(snapshotter: &Snapshotter) -> u64 {
    sum_merge_counter_for_path(snapshotter, None)
}

/// Sum the cross_episode merge counter, optionally filtered to one `{path}` label
/// (`Some("exact")` / `Some("fuzzy")`) — asserts the DoD-P3.4 per-path split.
fn sum_merge_counter_for_path(snapshotter: &Snapshotter, want_path: Option<&str>) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.cross_episode_merges_total" {
                return None;
            }
            if let Some(want) = want_path {
                let matches = key.labels().any(|l| l.key() == "path" && l.value() == want);
                if !matches {
                    return None;
                }
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

// ─── Per-row invariant assertions (shared by smoke + batch) ─────────────────

/// Assert the structural invariants for one row's outcome (spec §6 / P3).
fn assert_row_invariants(o: &RowOutcome) {
    let no_merge_categories = [
        "same_label_distinct_referent",
        "fuzzy_without_corroboration",
        "cosine_near_dup_lexically_distinct",
        "same_label_same_episode",
    ];
    if no_merge_categories.contains(&o.category.as_str()) {
        // HARD safety gate: a no-merge-category pair must NEVER merge (EXACT).
        assert!(
            !o.merged,
            "FALSE MERGE: no-merge-category row {} ({}) merged (op_count={})",
            o.row_id, o.category, o.op_count
        );
        assert_eq!(o.op_count, 0, "no-merge-category op_count must be EXACT 0");
        assert_eq!(
            o.entities_after, o.entities_before,
            "no-merge-category: entity count unchanged"
        );
        assert!(
            o.a_survives && o.b_survives,
            "no-merge-category: BOTH candidate entities must survive (row {})",
            o.row_id
        );
    } else {
        // Merge categories → exactly one merge; the pair collapses to one keeper.
        assert!(o.merged, "row {} ({}) should merge", o.row_id, o.category);
        assert_eq!(o.op_count, 1, "one merge for row {}", o.row_id);
        assert_eq!(
            o.entities_after,
            o.entities_before - 1,
            "entity count drops by exactly 1 (loser merged away) (row {})",
            o.row_id
        );
        // Exactly one candidate survives (the keeper = lowest id).
        assert!(
            o.a_survives ^ o.b_survives,
            "merge category: EXACTLY one candidate survives (row {})",
            o.row_id
        );
    }
    // expected_merge from the corpus must match the observed outcome.
    assert_eq!(
        o.merged, o.expected_merge,
        "row {} ({}): observed merge != corpus expected_merge",
        o.row_id, o.category
    );
}

// ─── Smoke-one-before-batch (hard rule) ─────────────────────────────────────

#[tokio::test]
async fn smoke_one_cross_episode_harness() {
    let corpus = load_corpus();
    let smoke = corpus
        .iter()
        .find(|r| r.id == "xep-001")
        .expect("corpus must contain xep-001");
    assert_eq!(
        smoke.category,
        "verbatim_repeat_distinct_episodes_with_corroboration"
    );

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let outcome = run_one_row(smoke).await;
    eprintln!(
        "[smoke-one] row_id={} category={} expected_merge={} merged={} op_count={} \
         entities {}->{} a_survives={} b_survives={}",
        outcome.row_id,
        outcome.category,
        outcome.expected_merge,
        outcome.merged,
        outcome.op_count,
        outcome.entities_before,
        outcome.entities_after,
        outcome.a_survives,
        outcome.b_survives,
    );

    assert_row_invariants(&outcome);

    // o11y cross-check on this single row.
    let counter = sum_merge_counter(&snapshotter);
    assert_eq!(
        counter, 1,
        "cross_episode merge counter must equal the single merge"
    );
}

// ─── Full batch: safety gate + delta invariants + o11y cross-check ──────────

#[tokio::test]
async fn cross_episode_corpus_batch_metrics() {
    let corpus = load_corpus();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes = Vec::new();
    for row in &corpus {
        outcomes.push(run_one_row(row).await);
    }

    // Per-row structural invariants + hard safety gate.
    let mut false_merges = 0usize;
    let no_merge_categories = [
        "same_label_distinct_referent",
        "fuzzy_without_corroboration",
        "cosine_near_dup_lexically_distinct",
        "same_label_same_episode",
    ];
    let mut homonym_false_merges = 0usize;
    for o in &outcomes {
        assert_row_invariants(o);
        if no_merge_categories.contains(&o.category.as_str()) && o.merged {
            false_merges += 1;
            if o.category == "same_label_distinct_referent" {
                homonym_false_merges += 1;
            }
        }
    }

    // o11y cross-check (DoD-P3.4): counter == sum of op counts == merged rows.
    let counter = sum_merge_counter(&snapshotter);
    let op_count_sum: usize = outcomes.iter().map(|o| o.op_count).sum();
    let merged_rows: usize = outcomes.iter().filter(|o| o.merged).count();

    eprintln!(
        "[batch] rows={} merged={merged_rows} op_count_sum={op_count_sum} \
         counter={counter} false_merges={false_merges} homonym_false_merges={homonym_false_merges}",
        outcomes.len(),
    );

    // Hard gate: ZERO false merges on no-merge categories (EXACT, never `>=`).
    assert_eq!(
        false_merges, 0,
        "hard gate: ZERO false merges on no-merge categories"
    );
    assert_eq!(
        homonym_false_merges, 0,
        "RISK-001 hard gate: ZERO homonym false merges (same_label_distinct_referent)"
    );
    assert_eq!(
        counter as usize, op_count_sum,
        "cross_episode_merges counter ({counter}) must equal summed op counts ({op_count_sum})"
    );
    assert_eq!(
        op_count_sum, merged_rows,
        "summed op counts must equal the number of merged pairs"
    );

    // DoD-P3.4 per-path split: the corpus's expected exact/fuzzy merge rows must match
    // the per-`{path}`-label counter. Each merge row's `path` field names the path its
    // single merge fires on.
    let expected_exact = corpus
        .iter()
        .filter(|r| r.expected_merge && r.path == "exact")
        .count();
    let expected_fuzzy = corpus
        .iter()
        .filter(|r| r.expected_merge && r.path == "fuzzy")
        .count();
    let exact_counter = sum_merge_counter_for_path(&snapshotter, Some("exact")) as usize;
    let fuzzy_counter = sum_merge_counter_for_path(&snapshotter, Some("fuzzy")) as usize;
    assert_eq!(
        exact_counter, expected_exact,
        "exact-path merge counter ({exact_counter}) must equal corpus exact merge rows ({expected_exact})"
    );
    assert_eq!(
        fuzzy_counter, expected_fuzzy,
        "fuzzy-path merge counter ({fuzzy_counter}) must equal corpus fuzzy merge rows ({expected_fuzzy})"
    );

    // Sanity: the corpus DOES exercise BOTH merge paths (not a vacuous all-keep run).
    assert!(merged_rows > 0, "corpus must merge at least one pair");
    assert!(
        expected_exact > 0 && expected_fuzzy > 0,
        "corpus must exercise both paths"
    );
}

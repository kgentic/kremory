//! ADR-066 dream CONSOLIDATION P1 — supersession deterministic-corpus harness.
//!
//! Spec `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md` §3
//! (P1.1–P1.4) + §6 (corpus shape). The deterministic **window-closeout** lane is
//! zero-LLM, so — UNLIKE the site3 harness — there is NO VCR cassette; each corpus
//! row is a fully-deterministic fixture. What IS mirrored from
//! `dream_metrics_harness_site3.rs`:
//!
//! - **per-row group isolation** — each row plants into its own fresh `group_id`
//!   so cross-row facts can never contaminate one row's measurement;
//! - **smoke-one-before-batch** — `smoke_one_supersession_harness` runs ONE
//!   representative row (`sup-001`, `window_already_closed`) through the full
//!   scoring pipeline FIRST and asserts sane wiring before the batch;
//! - **Wilson 95% CI** (`wilson_lower_upper`) over the retirement precision;
//! - **o11y cross-check** — the pass's own
//!   `kremory.dream.consolidation.supersessions_recorded_total{lane=window_closeout}`
//!   counter is snapshotted and asserted EQUAL to the harness-derived retirement
//!   count (DoD-P1.4 — counter == OpReport.count).
//!
//! **Hard safety gate (DoD-P1.2):** ZERO false supersessions on the keep
//! categories (`still_valid_open_ended`, `already_resolved_at_ingest`,
//! `distinct_no_supersede`), asserted EXACTLY (never `>=`). Window-closeout
//! precision is asserted == 1.00 (date-compare is exact).

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use chrono::{Duration, Utc};

use kremory::core::dream::{
    supersession, wilson_lower_upper, ConsolidationBudget, SupersessionParams,
};
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── Corpus row ─────────────────────────────────────────────────────────────

/// Parsed corpus row. `*_days_ago` are offsets from `now` at plant time; a
/// NEGATIVE `valid_to_days_ago` places `valid_to` in the FUTURE (open window,
/// not yet closed). `null` `valid_to_days_ago` = open-ended (`valid_to IS NULL`).
struct Row {
    id: String,
    subject: String,
    predicate: String,
    object_value: String,
    category: String,
    valid_from_days_ago: i64,
    valid_to_days_ago: Option<i64>,
    expired_at_days_ago: Option<i64>,
    invalid_at_days_ago: Option<i64>,
    is_dream_generated: i64,
    expected_supersede: bool,
}

fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("consolidation_supersession_adversarial.jsonl");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read corpus at {path:?}: {e}"));

    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {line_no} failed to parse: {e}"));

        // Parse loudly — required fields carry NO default. A missing field is a
        // corpus authoring bug, surfaced as a panic (not a silent default).
        let str_field = |k: &str| -> String {
            raw[k]
                .as_str()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing str '{k}'"))
                .to_string()
        };
        let i64_field = |k: &str| -> i64 {
            raw[k]
                .as_i64()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing i64 '{k}'"))
        };
        let opt_i64_field = |k: &str| -> Option<i64> {
            match &raw[k] {
                serde_json::Value::Null => None,
                v => Some(
                    v.as_i64()
                        .unwrap_or_else(|| panic!("corpus line {line_no}: '{k}' not i64|null")),
                ),
            }
        };
        let bool_field = |k: &str| -> bool {
            raw[k]
                .as_bool()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing bool '{k}'"))
        };

        rows.push(Row {
            id: str_field("id"),
            subject: str_field("subject"),
            predicate: str_field("predicate"),
            object_value: str_field("object_value"),
            category: str_field("category"),
            valid_from_days_ago: i64_field("valid_from_days_ago"),
            valid_to_days_ago: opt_i64_field("valid_to_days_ago"),
            expired_at_days_ago: opt_i64_field("expired_at_days_ago"),
            invalid_at_days_ago: opt_i64_field("invalid_at_days_ago"),
            is_dream_generated: i64_field("is_dream_generated"),
            expected_supersede: bool_field("expected_supersede"),
        });
    }
    assert!(!rows.is_empty(), "corpus must not be empty");
    rows
}

// ─── Per-row isolated plant + run ───────────────────────────────────────────

/// Outcome of running one corpus row through the real supersession sweep in its
/// own isolated `group_id`.
struct RowOutcome {
    row_id: String,
    category: String,
    expected_supersede: bool,
    /// Did the sweep actually retire this fact (set `expired_at`)?
    superseded: bool,
    /// The op's reported count for this isolated single-fact group.
    op_count: usize,
    /// If the fact had a preset `expired_at`, its value — asserted UNCHANGED for
    /// the `already_resolved_at_ingest` category (DoD-P1.2 keep-untouched EXACT).
    preset_expired_at: Option<String>,
    post_expired_at: Option<String>,
}

/// Plant one row's fact into a fresh isolated group and run `supersession`.
async fn run_one_row(row: &Row) -> RowOutcome {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let gid = format!("sup-corpus-{}", row.id);
    let now = Utc::now();

    // Subject entity (composite FK + FTS shadow) via the real API.
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id: &row.subject,
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(&gid),
        })
        .await
        .expect("plant subject entity");

    let valid_from = (now - Duration::days(row.valid_from_days_ago)).to_rfc3339();
    let valid_to = row
        .valid_to_days_ago
        .map(|d| (now - Duration::days(d)).to_rfc3339());
    let expired_at = row
        .expired_at_days_ago
        .map(|d| (now - Duration::days(d)).to_rfc3339());
    let invalid_at = row
        .invalid_at_days_ago
        .map(|d| (now - Duration::days(d)).to_rfc3339());
    let recorded_at = now.to_rfc3339();

    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
              expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1.0, ?11)",
            libsql::params![
                row.subject.clone(),
                row.predicate.clone(),
                row.object_value.clone(),
                valid_from,
                valid_to,
                recorded_at,
                expired_at.clone(),
                invalid_at,
                gid.clone(),
                gid.clone(),
                row.is_dream_generated,
            ],
        )
        .await
        .expect("plant fact");

    let mut id_rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    let fact_id: i64 = id_rows
        .next()
        .await
        .expect("row")
        .expect("present")
        .get(0)
        .expect("id");
    drop(id_rows);

    let mut budget = ConsolidationBudget::new(None, None);
    let report = supersession(SupersessionParams {
        graph: &graph,
        group_id: &gid,
        budget: &mut budget,
        include_llm_nominate: false,
        model_id: "gemma4:e4b",
    })
    .await
    .expect("supersession");

    let post_expired_at = read_expired_at(&graph, fact_id).await;
    // "superseded" = the sweep newly set expired_at where it was previously NULL,
    // OR the op reported a retirement for this single-fact group. For the
    // preset-expired case, `superseded` must be false AND expired_at unchanged.
    let superseded = report.count > 0;

    RowOutcome {
        row_id: row.id.clone(),
        category: row.category.clone(),
        expected_supersede: row.expected_supersede,
        superseded,
        op_count: report.count,
        preset_expired_at: expired_at,
        post_expired_at,
    }
}

async fn read_expired_at(graph: &TemporalGraph, fact_id: i64) -> Option<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT expired_at FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query expired_at");
    let row = rows.next().await.expect("row").expect("present");
    row.get::<Option<String>>(0).expect("expired_at col")
}

// ─── o11y cross-check ───────────────────────────────────────────────────────

/// Sum the window-closeout supersession counter across the snapshot.
fn sum_window_closeout_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.supersessions_recorded_total" {
                return None;
            }
            let labels: HashMap<&str, &str> = key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("lane").copied() != Some("window_closeout") {
                return None;
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

// ─── Smoke-one-before-batch (hard rule) ─────────────────────────────────────

#[tokio::test]
async fn smoke_one_supersession_harness() {
    let corpus = load_corpus();
    let smoke = corpus
        .iter()
        .find(|r| r.id == "sup-001")
        .expect("corpus must contain sup-001");
    assert_eq!(smoke.category, "window_already_closed");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let outcome = run_one_row(smoke).await;
    eprintln!(
        "[smoke-one] row_id={} category={} expected_supersede={} superseded={} op_count={}",
        outcome.row_id,
        outcome.category,
        outcome.expected_supersede,
        outcome.superseded,
        outcome.op_count
    );

    assert!(
        outcome.superseded,
        "sup-001 (closed window) must be retired"
    );
    assert_eq!(outcome.op_count, 1, "one retirement in the isolated group");

    // o11y cross-check on this single row.
    let counter = sum_window_closeout_counter(&snapshotter);
    assert_eq!(
        counter, 1,
        "window_closeout counter must equal the single retirement"
    );
}

// ─── Full batch: precision + safety gate + o11y cross-check ─────────────────

#[tokio::test]
async fn supersession_corpus_batch_metrics() {
    let corpus = load_corpus();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes = Vec::new();
    for row in &corpus {
        outcomes.push(run_one_row(row).await);
    }

    // ── Retirement precision (window-closeout lane) ─────────────────────────
    // Precision denominator = facts the sweep RETIRED (superseded == true).
    // Precision = of those, how many were genuinely `window_already_closed`
    // (expected_supersede == true). Date-compare is exact → precision must be 1.00.
    let retired: Vec<&RowOutcome> = outcomes.iter().filter(|o| o.superseded).collect();
    let true_positives = retired.iter().filter(|o| o.expected_supersede).count();
    let precision = if retired.is_empty() {
        1.0
    } else {
        true_positives as f64 / retired.len() as f64
    };
    let (wilson_lo, wilson_hi) = wilson_lower_upper(true_positives, retired.len());

    // ── Recall over the retire category ─────────────────────────────────────
    let should_retire: Vec<&RowOutcome> =
        outcomes.iter().filter(|o| o.expected_supersede).collect();
    let retired_of_should = should_retire.iter().filter(|o| o.superseded).count();
    let recall = if should_retire.is_empty() {
        1.0
    } else {
        retired_of_should as f64 / should_retire.len() as f64
    };

    // ── HARD safety gate (DoD-P1.2): ZERO false supersessions on keep cats ──
    let keep_categories = [
        "still_valid_open_ended",
        "already_resolved_at_ingest",
        "distinct_no_supersede",
    ];
    let mut false_supersessions = 0usize;
    for o in &outcomes {
        if keep_categories.contains(&o.category.as_str()) {
            // EXACT assertion, never `>=` — a keep-category fact must never be
            // retired by this sweep.
            assert!(
                !o.superseded,
                "FALSE SUPERSESSION: keep-category row {} ({}) was retired \
                 (op_count={})",
                o.row_id, o.category, o.op_count
            );
            if o.superseded {
                false_supersessions += 1;
            }
            // `already_resolved_at_ingest` with a preset expired_at must stay
            // EXACTLY unchanged (keep-untouched, DoD-P1.2).
            if o.category == "already_resolved_at_ingest" && o.preset_expired_at.is_some() {
                assert_eq!(
                    o.post_expired_at, o.preset_expired_at,
                    "preset expired_at must be UNCHANGED for row {}",
                    o.row_id
                );
            }
        }
    }

    // ── o11y cross-check (DoD-P1.4): counter == sum of op counts ────────────
    let counter = sum_window_closeout_counter(&snapshotter);
    let op_count_sum: usize = outcomes.iter().map(|o| o.op_count).sum();
    assert_eq!(
        counter as usize, op_count_sum,
        "window_closeout counter ({counter}) must equal summed op counts ({op_count_sum})"
    );
    // And each equals the number of retired facts.
    assert_eq!(
        op_count_sum,
        retired.len(),
        "summed op counts must equal the number of retired facts"
    );

    eprintln!(
        "[batch] rows={} retired={} precision={precision:.4} wilson=[{wilson_lo:.4},{wilson_hi:.4}] \
         recall={recall:.4} false_supersessions={false_supersessions} counter={counter}",
        outcomes.len(),
        retired.len(),
    );

    // ── Gate assertions ─────────────────────────────────────────────────────
    assert_eq!(
        false_supersessions, 0,
        "hard gate: ZERO false supersessions on keep categories"
    );
    assert_eq!(
        precision, 1.00,
        "window-closeout precision must be EXACT 1.00 (date-compare)"
    );
    assert_eq!(recall, 1.00, "all closed-window facts must be retired");
    // Wilson interval is reported DESCRIPTIVELY, not gated. Spec §6: "Deterministic
    // ops ... reach precision 1.00 trivially — their gate is the zero-false-positive
    // safety category." The Wilson-95%-lower ≥0.85 floor is the enablement gate for
    // the LLM lanes; on a deterministic date-compare lane the EXACT-1.00-precision +
    // ZERO-false-positive gates above are the load-bearing bar. Gating a wide small-N
    // CI here would be the wrong gate (`feedback_wilson_lower_bound_needs_flagged_n_
    // sizing` — the CI is over the flagged-retired subset N, not the corpus size).
    // Sanity only: the interval must be non-degenerate + bracket the observed 1.00.
    assert!(
        (0.0..=1.0).contains(&wilson_lo)
            && (0.0..=1.0).contains(&wilson_hi)
            && wilson_lo <= wilson_hi,
        "Wilson interval must be a valid [lo,hi] within [0,1]: [{wilson_lo},{wilson_hi}]"
    );
    assert!(
        (wilson_hi - 1.0).abs() < 1e-9,
        "with precision 1.00 the Wilson UPPER bound must be 1.00"
    );
}

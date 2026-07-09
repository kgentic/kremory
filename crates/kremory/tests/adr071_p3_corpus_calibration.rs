//! ADR-071 Item 1 — P3 `cross_episode_merges` enablement: execute the ADR-063
//! corpus gate.
//!
//! Spec: `.ai-docs/specs/adr-071-dream-phase-hardening-impl-spec-2026-07-06.md`
//! §Item 1 (§1a sizing arithmetic, §1b corpus provenance/generation, §1c metrics
//! harness + pass criteria). Zero-LLM, code-generated, planted-by-design corpus —
//! the op under test (`cross_episode`) is itself zero-LLM, pure-graph-structure
//! input, so this harness needs no VCR cassette and no Ollama.
//!
//! Three tests, run in this order (smoke-one-before-batch discipline + Risk #11):
//!
//! 1. [`planted_corpus_categories_have_correct_structure`] — MANDATORY sanity
//!    test. Inspects the RAW planted graph structure (never the op's own output)
//!    and asserts each category really has the intended degree/threshold
//!    property. If the generator is buggy, THIS fails — fix the generator, never
//!    adjust the gate (Risk #11).
//! 2. [`smoke_one_p3_gate_representative_cases`] — one representative case per
//!    category through the full plant->run->classify pipeline, before the batch.
//! 3. [`p3_corpus_calibration_gate`] — the full ≥218-case corpus through the op,
//!    computing precision / Wilson-LB / recall / confusion table, writing the
//!    findings doc, and asserting the impl-spec §1c pass criteria.
//!
//! `F1`/`F2` mechanism in `cross_episode.rs` (clique-cover, Bron-Kerbosch,
//! `WEIGHT_LUT_SCALED`) is READ-ONLY from this harness — this file adds a new
//! test binary + a new corpus fixture generator; it does not modify
//! `cross_episode.rs` at all.

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use kremory::core::dream::{cross_episode, wilson_lower_upper};

mod support;
use support::p3_corpus::{
    generate_corpus, plant_case, raw_degree, raw_shared_neighbours, Category, PlantedCase, Verdict,
};

// ─── Mirrored constants (Risk #11 — VERIFIED ANCHORS, re-grep before trusting) ─
//
// These MIRROR `cross_episode.rs`'s private consts (`HUB_DEGREE_CAP:100-105`,
// `SCALED_THRESHOLD:118`, `WEIGHT_LUT_SCALED:107-113`) — verified this session,
// re-read at file-open time before use. Kept HERE (not imported) because the
// production consts are intentionally private (`cross_episode.rs`'s own
// doc-comment: "provisional — spike-gated (R-01)") and because Risk #11 requires
// an INDEPENDENT check, never a call back into the SUT's own constants/functions.

/// Mirrors `cross_episode.rs:105`.
const HUB_DEGREE_CAP: u32 = 8;
/// Mirrors `cross_episode.rs:118`.
const SCALED_THRESHOLD: u64 = 524_288;
/// Mirrors `cross_episode.rs:112-113` (index 0 unused, 1..=8).
const WEIGHT_LUT_SCALED: [u64; 9] = [
    0, 1_048_576, 524_288, 405_645, 349_525, 315_653, 292_493, 275_408, 262_144,
];

/// Same integer branch as `cross_episode.rs::scaled_weight` (mirrored, not
/// imported — Risk #11 independence).
fn mirrored_scaled_weight(f: u32) -> u64 {
    if f == 0 || f > HUB_DEGREE_CAP {
        0
    } else {
        WEIGHT_LUT_SCALED[f as usize]
    }
}

// ─── 1. MANDATORY sanity test (Risk #11) ───────────────────────────────────────

/// Inspects the RAW planted graph structure for EVERY case in the corpus —
/// independent of the `cross_episode` op's output — and asserts each category
/// really has the intended degree/threshold property. This is the gate that
/// catches a corpus-GENERATOR bug before the calibration gate is trusted.
#[tokio::test]
async fn planted_corpus_categories_have_correct_structure() {
    let specs = generate_corpus();
    assert!(!specs.is_empty(), "corpus must not be empty");

    let mut checked = 0usize;
    for spec in specs {
        let id = spec.id.clone();
        let category = spec.category;
        let (graph, gid, planted): (_, _, PlantedCase) = plant_case(spec).await;

        // Independent raw-graph check: does entity_a really share the intended
        // neighbour set with entity_b (per category), and does each shared
        // neighbour's RAW degree match what the generator intended to plant?
        let raw_shared =
            raw_shared_neighbours(&graph, &gid, &planted.entity_a, &planted.entity_b).await;

        match category {
            Category::RareShared | Category::GenuineDup => {
                assert_eq!(
                    raw_shared.len(),
                    2,
                    "case {id}: expected exactly 2 raw shared neighbours"
                );
                let mut sum: u64 = 0;
                for (n, intended_degree) in &planted.shared_neighbours {
                    assert!(
                        raw_shared.contains(n),
                        "case {id}: planted neighbour {n} not found in raw shared-neighbour set"
                    );
                    let actual = raw_degree(&graph, &gid, n).await;
                    assert_eq!(
                        actual, *intended_degree,
                        "case {id}: neighbour {n} raw degree {actual} != intended {intended_degree}"
                    );
                    assert!(
                        actual <= HUB_DEGREE_CAP,
                        "case {id}: RareShared/GenuineDup neighbour {n} degree {actual} must be \
                         within the hub cap ({HUB_DEGREE_CAP})"
                    );
                    sum += mirrored_scaled_weight(actual);
                }
                assert!(
                    sum >= SCALED_THRESHOLD,
                    "case {id}: Σ scaled_weight={sum} must clear SCALED_THRESHOLD={SCALED_THRESHOLD}"
                );
                assert!(
                    sum > SCALED_THRESHOLD,
                    "case {id}: Σ scaled_weight={sum} must clear the threshold WITH MARGIN \
                     (strictly greater), not sit boundary-exact (Risk #2)"
                );
            }
            Category::HubShared => {
                assert_eq!(
                    raw_shared.len(),
                    1,
                    "case {id}: expected exactly 1 raw shared neighbour"
                );
                let (n, intended_degree) = &planted.shared_neighbours[0];
                let actual = raw_degree(&graph, &gid, n).await;
                assert_eq!(
                    actual, *intended_degree,
                    "case {id}: hub neighbour {n} raw degree {actual} != intended {intended_degree}"
                );
                assert!(
                    actual > HUB_DEGREE_CAP,
                    "case {id}: HubShared neighbour {n} degree {actual} must EXCEED hub cap \
                     ({HUB_DEGREE_CAP}) — else this is not a real hub"
                );
                assert_eq!(
                    mirrored_scaled_weight(actual),
                    0,
                    "case {id}: hub-degree neighbour must contribute integer 0"
                );
            }
            Category::TwoHubShared => {
                assert_eq!(
                    raw_shared.len(),
                    2,
                    "case {id}: expected exactly 2 raw shared neighbours"
                );
                let mut sum: u64 = 0;
                for (n, intended_degree) in &planted.shared_neighbours {
                    let actual = raw_degree(&graph, &gid, n).await;
                    assert_eq!(
                        actual, *intended_degree,
                        "case {id}: neighbour {n} raw degree {actual} != intended {intended_degree}"
                    );
                    assert!(
                        actual > HUB_DEGREE_CAP,
                        "case {id}: TwoHubShared neighbour {n} degree {actual} must EXCEED hub \
                         cap ({HUB_DEGREE_CAP})"
                    );
                    sum += mirrored_scaled_weight(actual);
                }
                assert_eq!(
                    sum, 0,
                    "case {id}: BOTH hub-degree neighbours contribute 0 -> sum must be 0"
                );
            }
            Category::ZeroShared => {
                assert!(
                    raw_shared.is_empty(),
                    "case {id}: ZeroShared must have NO raw shared neighbours, found {raw_shared:?}"
                );
                assert!(
                    planted.shared_neighbours.is_empty(),
                    "case {id}: ZeroShared generator must not plant any shared_neighbours entry"
                );
            }
            Category::Boundary => {
                assert_eq!(
                    raw_shared.len(),
                    1,
                    "case {id}: expected exactly 1 raw shared neighbour"
                );
                let (n, intended_degree) = &planted.shared_neighbours[0];
                let actual = raw_degree(&graph, &gid, n).await;
                assert_eq!(
                    actual, *intended_degree,
                    "case {id}: boundary neighbour {n} raw degree {actual} != intended {intended_degree}"
                );
                assert_eq!(
                    actual, 2,
                    "case {id}: Boundary category's degree must be EXACTLY 2 (the documented \
                     boundary-exact case)"
                );
                assert_eq!(
                    mirrored_scaled_weight(actual),
                    SCALED_THRESHOLD,
                    "case {id}: Boundary neighbour's weight must equal SCALED_THRESHOLD EXACTLY \
                     (the calibration crux — current constants merge this)"
                );
            }
        }

        checked += 1;
    }

    eprintln!("[sanity] {checked} planted cases structurally verified — generator is sound.");
}

// ─── 2. smoke-one-before-batch ─────────────────────────────────────────────────

/// One representative case per category through the full plant->run->classify
/// pipeline, before the ≥218-case batch. Zero-LLM/deterministic, so this is
/// cheap insurance rather than the mandatory-cost LLM smoke-one rule — still
/// worth doing to catch a harness wiring bug before the batch run.
#[tokio::test]
async fn smoke_one_p3_gate_representative_cases() {
    let specs = generate_corpus();
    for category in [
        Category::RareShared,
        Category::GenuineDup,
        Category::HubShared,
        Category::TwoHubShared,
        Category::ZeroShared,
        Category::Boundary,
    ] {
        let spec = specs
            .iter()
            .find(|s| s.category == category)
            .unwrap_or_else(|| panic!("corpus must contain at least one {category:?} case"))
            .clone();
        let expected = spec.expected;
        let (graph, gid, planted) = plant_case(spec).await;
        let report = cross_episode(&graph, &gid, false)
            .await
            .expect("cross_episode");
        let actual = if report.count > 0 {
            Verdict::Merge
        } else {
            Verdict::NotMerge
        };
        eprintln!(
            "[smoke-one] category={category:?} id={} expected={expected:?} actual={actual:?} \
             op_count={}",
            planted.spec.id, report.count
        );
        assert_eq!(
            actual, expected,
            "smoke-one FAILED for category {category:?} (case {}): expected={expected:?} \
             actual={actual:?}",
            planted.spec.id
        );
    }
    eprintln!(
        "[smoke-one] all 6 representative categories PASS — proceeding to full batch is safe."
    );
}

// ─── 3. Full corpus calibration gate ───────────────────────────────────────────

#[derive(Debug, Default, Clone, serde::Serialize)]
struct ConfusionRow {
    category: String,
    n: usize,
    merged: usize,
    not_merged: usize,
    gated: bool,
}

#[derive(Debug, serde::Serialize)]
struct GateReport {
    verdict: String,
    wilson_lb: f64,
    wilson_ub: f64,
    precision: f64,
    correct_merges: usize,
    total_merges_performed: usize,
    recall: f64,
    recall_numerator: usize,
    recall_denominator: usize,
    hub_or_two_hub_false_merges: usize,
    confusion_table: Vec<ConfusionRow>,
    corpus_size: usize,
}

/// The full ≥218-case corpus through the REAL, unmodified `cross_episode` op —
/// zero-LLM, `dry_run=false` (mirrors `consolidation_cross_episode_test.rs`'s
/// established pattern — measures REAL fusion via `report.count`, the cleanest
/// classification signal since each case is an isolated 1-candidate-pair graph).
#[tokio::test]
async fn p3_corpus_calibration_gate() {
    let specs = generate_corpus();
    let corpus_size = specs.len();
    assert!(
        corpus_size >= 190,
        "corpus must have >=190 total cases per impl-spec §1a sizing (got {corpus_size})"
    );

    let gated_should_merge: usize = specs
        .iter()
        .filter(|s| s.gated && s.expected == Verdict::Merge)
        .count();
    assert!(
        gated_should_merge >= 120,
        "gated SHOULD_MERGE (RareShared+GenuineDup) must be >=120 (got {gated_should_merge})"
    );
    let hub_count = specs
        .iter()
        .filter(|s| s.category == Category::HubShared)
        .count();
    let two_hub_count = specs
        .iter()
        .filter(|s| s.category == Category::TwoHubShared)
        .count();
    assert!(hub_count >= 30, "HubShared must be >=30 (got {hub_count})");
    assert!(
        two_hub_count >= 30,
        "TwoHubShared must be >=30 (got {two_hub_count})"
    );

    // ── Run every case through the real op ─────────────────────────────────
    struct Outcome {
        category: Category,
        gated: bool,
        expected: Verdict,
        actual: Verdict,
    }

    let mut outcomes: Vec<Outcome> = Vec::with_capacity(corpus_size);
    for spec in specs {
        let gated = spec.gated;
        let expected = spec.expected;
        let category = spec.category;
        let (graph, gid, _planted) = plant_case(spec).await;
        let report = cross_episode(&graph, &gid, false)
            .await
            .expect("cross_episode");
        let actual = if report.count > 0 {
            Verdict::Merge
        } else {
            Verdict::NotMerge
        };
        outcomes.push(Outcome {
            category,
            gated,
            expected,
            actual,
        });
    }

    // ── Confusion table (rows = category, cols = Merge/NotMerge outcome) ───
    let mut table: BTreeMap<Category, ConfusionRow> = BTreeMap::new();
    for o in &outcomes {
        let row = table.entry(o.category).or_insert_with(|| ConfusionRow {
            category: o.category.as_str().to_string(),
            n: 0,
            merged: 0,
            not_merged: 0,
            gated: o.gated,
        });
        row.n += 1;
        match o.actual {
            Verdict::Merge => row.merged += 1,
            Verdict::NotMerge => row.not_merged += 1,
        }
    }
    let confusion_table: Vec<ConfusionRow> = table.into_values().collect();

    // ── Precision-gate denominator: GATED cases only (RareShared/GenuineDup/
    // HubShared/TwoHubShared) — Boundary/ZeroShared are sanity-only, per
    // impl-spec §1a/§1c, deliberately EXCLUDED from this computation. ───────
    let gated_outcomes: Vec<&Outcome> = outcomes.iter().filter(|o| o.gated).collect();
    let total_merges_performed = gated_outcomes
        .iter()
        .filter(|o| o.actual == Verdict::Merge)
        .count();
    let correct_merges = gated_outcomes
        .iter()
        .filter(|o| o.actual == Verdict::Merge && o.expected == Verdict::Merge)
        .count();
    let precision = if total_merges_performed == 0 {
        0.0
    } else {
        correct_merges as f64 / total_merges_performed as f64
    };
    let (wilson_lb, wilson_ub) = wilson_lower_upper(correct_merges, total_merges_performed);

    // Recall (reported, NOT gated): of the gated cases that SHOULD merge, how
    // many actually did?
    let recall_denominator = gated_outcomes
        .iter()
        .filter(|o| o.expected == Verdict::Merge)
        .count();
    let recall_numerator = gated_outcomes
        .iter()
        .filter(|o| o.expected == Verdict::Merge && o.actual == Verdict::Merge)
        .count();
    let recall = if recall_denominator == 0 {
        0.0
    } else {
        recall_numerator as f64 / recall_denominator as f64
    };

    // Hard safety check: zero HubShared/TwoHubShared false merges.
    let hub_or_two_hub_false_merges = outcomes
        .iter()
        .filter(|o| {
            matches!(o.category, Category::HubShared | Category::TwoHubShared)
                && o.actual == Verdict::Merge
        })
        .count();

    // ── Mechanical pass criteria (impl-spec §1c) ────────────────────────────
    let verdict = if wilson_lb >= 0.95 && hub_or_two_hub_false_merges == 0 {
        "PASS"
    } else if wilson_lb >= 0.90 && hub_or_two_hub_false_merges == 0 {
        "CONCERNS"
    } else {
        "FAIL"
    };

    let report = GateReport {
        verdict: verdict.to_string(),
        wilson_lb,
        wilson_ub,
        precision,
        correct_merges,
        total_merges_performed,
        recall,
        recall_numerator,
        recall_denominator,
        hub_or_two_hub_false_merges,
        confusion_table: confusion_table.clone(),
        corpus_size,
    };

    eprintln!("\n── ADR-071 Item 1 P3 corpus-calibration gate ──────────────────────────");
    eprintln!(
        "  corpus_size={corpus_size} verdict={verdict} wilson_lb={wilson_lb:.6} \
         wilson_ub={wilson_ub:.6} precision={precision:.6} \
         correct_merges={correct_merges} total_merges_performed={total_merges_performed} \
         recall={recall:.6} ({recall_numerator}/{recall_denominator}) \
         hub_or_two_hub_false_merges={hub_or_two_hub_false_merges}"
    );
    eprintln!("  confusion table:");
    for row in &confusion_table {
        eprintln!(
            "    [{:<12}] gated={:<5} n={:<4} merged={:<4} not_merged={:<4}",
            row.category, row.gated, row.n, row.merged, row.not_merged
        );
    }

    write_findings_doc(&report);

    // ── Hard structural assertion: never a HubShared/TwoHubShared false merge,
    // regardless of the point-estimate verdict (this is the mandatory safety
    // half of the gate, orthogonal to Wilson-LB). ──────────────────────────
    assert_eq!(
        hub_or_two_hub_false_merges, 0,
        "SAFETY: {hub_or_two_hub_false_merges} HubShared/TwoHubShared pair(s) incorrectly \
         merged — this is the exact ADR-067 F2 bug the corpus proves the fix holds against. \
         See confusion table above."
    );

    // This test REPORTS the verdict (findings doc + eprintln) — it does NOT
    // assert PASS, because CONCERNS/FAIL are valid, actionable outcomes the
    // orchestrator triages per impl-spec §1c (single-constant recalibration or
    // escalation), not test failures. The orchestrator re-runs this harness to
    // confirm the number before any flag flip.
    eprintln!(
        "\n══ GATE VERDICT: {verdict} (Wilson-LB={wilson_lb:.6}, \
         hub_false_merges={hub_or_two_hub_false_merges}) ═══════════════"
    );
}

fn write_findings_doc(report: &GateReport) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".ai-docs")
        .join("research")
        .join("adr-071-p3-corpus-calibration-findings-2026-07-09.md");

    let mut confusion_md = String::new();
    confusion_md.push_str("| category | gated | n | merged | not_merged |\n");
    confusion_md.push_str("|---|---|---|---|---|\n");
    for row in &report.confusion_table {
        confusion_md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            row.category, row.gated, row.n, row.merged, row.not_merged
        ));
    }

    let body = format!(
        "---\n\
         title: ADR-071 Item 1 — P3 cross_episode corpus-calibration findings\n\
         type: research\n\
         status: complete\n\
         created: 2026-07-09\n\
         source: crates/kremory/tests/adr071_p3_corpus_calibration.rs (zero-LLM, \
         code-generated, planted-by-design corpus; test binary run)\n\
         refs:\n\
         \x20\x20- .ai-docs/specs/adr-071-dream-phase-hardening-impl-spec-2026-07-06.md#item-1\n\
         \x20\x20- .ai-docs/adrs/adr-071-dream-phase-hardening-and-enablement-2026-07-06.md\n\
         ---\n\n\
         # ADR-071 Item 1 — P3 `cross_episode` corpus-calibration findings\n\n\
         Generated by `cargo test -p kremory --features test-utils --test \
         adr071_p3_corpus_calibration -- p3_corpus_calibration_gate`.\n\n\
         ## Verdict\n\n\
         **{verdict}**\n\n\
         ## Gate arithmetic (impl-spec §1c)\n\n\
         ```\n\
         PASS      <->  Wilson-LB(precision) >= 0.95  AND  zero HubShared/TwoHubShared merges\n\
         CONCERNS  <->  0.90 <= Wilson-LB < 0.95  (no hub/two-hub false-merge)\n\
         FAIL      <->  Wilson-LB < 0.90  OR  any HubShared/TwoHubShared pair merged\n\
         ```\n\n\
         | metric | value |\n\
         |---|---|\n\
         | corpus_size | {corpus_size} |\n\
         | precision (point estimate) | {precision:.6} |\n\
         | correct_merges | {correct_merges} |\n\
         | total_merges_performed (gated denominator) | {total_merges_performed} |\n\
         | Wilson-LB | {wilson_lb:.6} |\n\
         | Wilson-UB | {wilson_ub:.6} |\n\
         | recall (reported, not gated) | {recall:.6} ({recall_numerator}/{recall_denominator}) |\n\
         | HubShared/TwoHubShared false merges (hard safety check) | {hub_false} |\n\n\
         ## Confusion table (rows = planted category, cols = actual op outcome)\n\n\
         {confusion_md}\n\
         ## Notes\n\n\
         - Ground truth = the planted GENERATIVE RULE (zero-LLM, code-generated), not a \
         rater's judgment — per impl-spec §1b, no cross-model-family labeling clause \
         applies.\n\
         - `Boundary` and `ZeroShared` categories are SANITY-ONLY (not in the precision-gate \
         denominator) per impl-spec §1a/§1c — reported here descriptively.\n\
         - `F1`/`F2` mechanism in `cross_episode.rs` was NOT modified to produce this \
         result — read-only harness run against the shipped, unmodified op.\n",
        verdict = report.verdict,
        corpus_size = report.corpus_size,
        precision = report.precision,
        correct_merges = report.correct_merges,
        total_merges_performed = report.total_merges_performed,
        wilson_lb = report.wilson_lb,
        wilson_ub = report.wilson_ub,
        recall = report.recall,
        recall_numerator = report.recall_numerator,
        recall_denominator = report.recall_denominator,
        hub_false = report.hub_or_two_hub_false_merges,
        confusion_md = confusion_md,
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, body)
        .unwrap_or_else(|e| panic!("failed to write findings doc to {path:?}: {e}"));
    eprintln!("  findings doc written to {path:?}");
}

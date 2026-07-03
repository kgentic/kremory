// ADR-063 spec `dream-adversarial-corpora-and-metrics-2026-07-02.md` §3/§3.1/
// §3.5/§4 — the production-grade metrics harness for Site #5 (instance
// acronym/nickname recall). Loads the committed adversarial corpus
// (`tests/corpora/site5_acronym_adversarial.jsonl`, 121 rows), runs each row
// through the REAL pass (real gemma4:e4b via VCR, no embedder needed — R3:
// Site #5 always passes cosine=0.0), and computes precision/recall/F1 +
// Wilson 95% CI per §3's denominator definition, cross-checked against the
// pass's own o11y counters (§3.5).
//
// KEY DESIGN CONSTRAINT: `acronym_nickname_recall`'s structural pre-filter is
// O(n^2) pairwise over every entity in a `group_id` (see
// `acronym_nickname_recall.rs`'s nested `for i .. for j` loop, which examines
// EVERY pair in the group, not just the corpus row's intended pair). Planting
// all 121 corpus rows (242 entities) into ONE group would produce ~29,000
// pairs and risk spurious cross-row nominations corrupting the measurement.
// Each row is therefore ISOLATED into its own `group_id` (a fresh 2-entity
// graph: entity `a`, entity `b`, with a shared-episode co-occurrence edge
// iff the row's `cooccurs=true`), and `acronym_nickname_recall` is run once
// per row on that group — the "always safe" choice named explicitly in the
// task brief. This costs up to 121 separate (single-pair) LLM adjudication
// calls but requires no manual cross-row collision audit.
//
// smoke-one-before-batch (hard rule): `smoke_one_metrics_harness` runs ONE
// representative row (`s5-001`, IBM / International Business Machines,
// `genuine_acronym`) through the full scoring pipeline FIRST and asserts
// sane wiring, before the 121-row batch runs.
//
// Denominator mapping (spec §3.1, mirrors the S2 spike's `any-same`
// convention exactly):
//   - Precision denominator = pairs the pass's write_gate calls "same"
//     (Merge ∪ PotentialAlias). Precision = (of those, how many are
//     genuinely the same entity per `ground_truth`).
//   - Recall denominator = genuinely-same-entity pairs (`ground_truth` true)
//     that were NOMINATED at all. Recall = (of those, how many did the pass
//     call "same", i.e. NOT Reject).
//   - `non_cooccurring_nickname` rows (cooccurs=false AND no structural
//     initialism relationship — confirmed empirically against
//     `initialism_candidate` for every row in this category, 2026-07-03) are
//     EXCLUDED from precision/recall entirely, scored separately as a
//     nomination-rate / gap-honesty check (the accepted ALT-001 gap: nothing
//     in the hybrid pre-filter can fire for these).
//   - `hard_negatives_unrelated` rows are ALSO cooccurs=false with no
//     initialism relationship (confirmed empirically) — but since they are
//     genuinely-different entities, they are kept IN the ordinary category
//     scoring as (structurally guaranteed) true negatives. TN never enters
//     precision (TP/(TP+FP)) or recall (TP/(TP+FN)), so this is purely
//     descriptive signal, not a gate-affecting choice.
//   - `uncertain_thin_context` rows (`same_entity="uncertain"` in the corpus
//     JSON) are EXCLUDED from the hard precision/recall/safety gate — ground
//     truth is genuinely ambiguous by design (task brief, explicit ask) —
//     and reported DESCRIPTIVELY ONLY, never fed into TP/FP/FN/TN counts.
//   - No `malformed` category rows exist in the committed corpus (verified:
//     `python3 -c` category scan on 2026-07-03 showed 0 "malformed" rows
//     across all 121) — the skip-guard below is a no-op today, kept for
//     forward-compatibility if a future corpus revision adds one.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::dream::{
    acronym_nickname_recall, wilson_lower_upper, AcronymNicknameRecallParams,
};
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── VCR scaffold (mirrors acronym_nickname_recall_s2_spike.rs) ──────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum VcrMode {
    Record,
    Replay,
}

fn resolve_vcr_mode() -> VcrMode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => VcrMode::Record,
        Ok("replay") | Err(_) => VcrMode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn chat_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("dream_metrics_harness_site5_{name}.json"))
}

/// Build the chat provider for one row's isolated LLM call. No embedder is
/// needed at all for Site #5 (`cosine` is always `0.0` per R3).
async fn build_provider(
    mode: VcrMode,
    cassette_tag: &str,
) -> (Arc<RecordReplayChatProvider>, String) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // gemma4:e4b + think:false — this project's benchmarked deferred-quality
    // dream model (F1 85.7, local-model-benchmark-2026-06-24 /
    // project_kremory_validated_model_findings_2026-06-24). Mirrors the S2
    // spike's model choice exactly — same pass, same tier.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let chat_cassette = chat_cassette_path(cassette_tag);
    let provider: Arc<RecordReplayChatProvider> = match mode {
        VcrMode::Record => {
            let real: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model(&chat_model)
                .think(false)
                .timeout_seconds(180)
                .keep_alive("1h")
                .build()
                .expect("Ollama LLM builder must succeed (KREMORY_VCR=record needs Ollama)");
            Arc::new(RecordReplayChatProvider::record(
                real,
                chat_cassette,
                chat_model.clone(),
            ))
        }
        VcrMode::Replay => Arc::new(
            RecordReplayChatProvider::replay(chat_cassette).unwrap_or_else(|e| {
                panic!(
                    "replay cassette must load for tag={cassette_tag}: {e} — \
                     record it via KREMORY_VCR=record"
                )
            }),
        ),
    };

    (provider, chat_model)
}

// ─── Ground truth ─────────────────────────────────────────────────────────────

/// Ground truth, resolved from the corpus row's `same_entity` field
/// (`true` | `false` | `"uncertain"`) + `category` + `cooccurs`.
///
/// `Same` / `Different` feed the hard precision/recall/safety gate.
/// `NotNominated` (the `non_cooccurring_nickname` category — cooccurs=false,
/// no structural relationship, ALT-001 accepted gap) is excluded from
/// precision/recall, scored separately as a nomination-rate check.
/// `Uncertain` (the `uncertain_thin_context` category, `same_entity=
/// "uncertain"`) is excluded from the hard gate entirely — ground truth
/// itself is ambiguous by design — and reported descriptively only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundTruth {
    Same,
    Different,
    NotNominated,
    Uncertain,
}

/// Parsed corpus row with `GroundTruth` resolved from the raw JSON.
struct Row {
    id: String,
    a: String,
    b: String,
    category: String,
    cooccurs: bool,
    ground_truth: GroundTruth,
    rationale: String,
}

/// Load + classify the committed corpus. Ground truth is derived from the
/// raw JSON's `same_entity` field (true/false/"uncertain") + `category`:
///   - `non_cooccurring_nickname` (cooccurs=false, no structural
///     relationship) -> NotNominated (spec §3.1: excluded from
///     precision/recall, scored as a nomination-rate check instead)
///   - `same_entity="uncertain"` (the `uncertain_thin_context` category) ->
///     Uncertain (excluded from the hard gate, reported descriptively only)
///   - `same_entity=true` (all other cooccurring categories, INCLUDING
///     `hard_negatives_unrelated`'s sibling categories) -> Same
///   - `same_entity=false` (collision/distinct-people/hard-negative
///     categories) -> Different
fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site5_acronym_adversarial.jsonl");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read corpus at {path:?}: {e}"));

    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {line_no} failed to parse as JSON: {e}"));

        let id = raw["id"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no}: missing 'id'"))
            .to_string();
        let a = raw["a"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'a'"))
            .to_string();
        let b = raw["b"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'b'"))
            .to_string();
        let category = raw["category"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'category'"))
            .to_string();
        let cooccurs = raw["cooccurs"]
            .as_bool()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing/non-bool 'cooccurs'"));
        let rationale = raw["rationale"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'rationale'"))
            .to_string();

        // `malformed` skip-guard (task brief): no such category exists in
        // the committed corpus today (verified 2026-07-03, 0 of 121), kept
        // for forward-compatibility if a future corpus revision adds one.
        if category == "malformed" {
            continue;
        }

        let same_entity_raw = &raw["same_entity"];
        let ground_truth = if category == "non_cooccurring_nickname" {
            GroundTruth::NotNominated
        } else if same_entity_raw.as_str() == Some("uncertain") {
            GroundTruth::Uncertain
        } else if same_entity_raw.as_bool() == Some(true) {
            GroundTruth::Same
        } else if same_entity_raw.as_bool() == Some(false) {
            GroundTruth::Different
        } else {
            panic!(
                "corpus line {line_no} (id={id}): unrecognized 'same_entity' value \
                 {same_entity_raw:?}"
            );
        };

        rows.push(Row {
            id,
            a,
            b,
            category,
            cooccurs,
            ground_truth,
            rationale,
        });
    }
    rows
}

// ─── Per-row isolated plant + run ────────────────────────────────────────────

/// Outcome of running one corpus row through the real pass in its own
/// isolated 2-entity group.
#[derive(Debug, Clone)]
struct RowOutcome {
    row_id: String,
    category: String,
    ground_truth: GroundTruth,
    /// The pass's actual `write_gate` decision string from
    /// `identity_verdict_audit.decision` ("merge" | "potential_alias" |
    /// "reject"), or `None` if the pair was never nominated (no audit row,
    /// no merge — nomination itself failed to fire).
    decision: Option<String>,
    candidates_nominated: usize,
}

/// Plant one row's two entities (+ co-occurrence episode iff `cooccurs`) into
/// a fresh isolated `group_id`, run `acronym_nickname_recall` on that group
/// alone, and read back the pass's actual decision from
/// `identity_verdict_audit` (mirrors `acronym_nickname_recall_s2_spike.rs`'s
/// per-pair audit-row cross-check exactly).
async fn run_one_row(row: &Row, mode: VcrMode) -> Result<(RowOutcome, String), String> {
    let graph = TemporalGraph::open_in_memory()
        .await
        .map_err(|e| format!("open_in_memory failed for row {}: {e}", row.id))?;
    let gid = format!("site5-corpus-{}", row.id);

    let props_a = serde_json::json!({ "name": row.a, "description": row.rationale });
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id: &row.a,
            entity_type_id: 0u32,
            properties: props_a,
            group_id: Some(&gid),
        })
        .await
        .map_err(|e| format!("insert_entity a failed for row {}: {e}", row.id))?;
    let props_b = serde_json::json!({ "name": row.b, "description": row.rationale });
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id: &row.b,
            entity_type_id: 0u32,
            properties: props_b,
            group_id: Some(&gid),
        })
        .await
        .map_err(|e| format!("insert_entity b failed for row {}: {e}", row.id))?;

    if row.cooccurs {
        let ep = graph
            .insert_episode(InsertEpisodeParams {
                content: &row.rationale,
                timestamp: chrono::Utc::now(),
                source_type: Some("transcript"),
                metadata: None,
            })
            .await
            .map_err(|e| format!("insert_episode failed for row {}: {e}", row.id))?;
        for ent in [row.a.as_str(), row.b.as_str()] {
            graph
                .insert_episodic_edge(InsertEpisodicEdgeParams {
                    episode_id: ep,
                    entity_id: ent,
                    entity_group_id: Some(&gid),
                    role: "mention",
                })
                .await
                .map_err(|e| format!("insert_episodic_edge failed for row {}: {e}", row.id))?;
        }
    }

    let (provider, chat_model) = build_provider(mode, &row.id).await;

    let report = acronym_nickname_recall(
        &*provider,
        AcronymNicknameRecallParams {
            graph: &graph,
            group_id: &gid,
            model_id: &chat_model,
        },
    )
    .await
    .map_err(|e| format!("acronym_nickname_recall failed for row {}: {e}", row.id))?;

    if mode == VcrMode::Record {
        provider
            .flush()
            .map_err(|e| format!("provider.flush() failed for row {}: {e}", row.id))?;
    }

    // Derive the pass's per-pair decision from the REPORT counts, NOT the
    // identity_verdict_audit table. Per spec §3.3 a Reject writes NO audit row
    // (only Merge/PotentialAlias are audited), so an audit-table read cannot
    // distinguish "reject" from "not-nominated" — both leave zero rows, which is
    // the o11y-cross-check divergence this fix resolves (write_gate_decision_total
    // counts rejects; the audit table does not). Each row is an ISOLATED
    // single-pair run, so the report's terminal counts unambiguously classify it.
    let decision: Option<String> = if report.candidates_nominated == 0 {
        None // structural pre-filter did not nominate this pair
    } else if report.merges_applied > 0 {
        Some("merge".to_string())
    } else if report.potential_aliases > 0 {
        Some("potential_alias".to_string())
    } else {
        Some("reject".to_string()) // nominated + adjudicated, neither merged nor aliased
    };

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::tests::dream_metrics_harness",
            row_id = %row.id,
            a = %row.a,
            b = %row.b,
            category = %row.category,
            cooccurs = row.cooccurs,
            ground_truth = ?row.ground_truth,
            decision = ?decision,
            candidates_nominated = report.candidates_nominated,
            "per-row scoring detail"
        );
    }

    Ok((
        RowOutcome {
            row_id: row.id.clone(),
            category: row.category.clone(),
            ground_truth: row.ground_truth,
            decision,
            candidates_nominated: report.candidates_nominated,
        },
        gid,
    ))
}

// ─── Metrics computation (§3/§3.1) ───────────────────────────────────────────

#[derive(Debug, Clone, Default, serde::Serialize)]
struct CategoryMetrics {
    n: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
    false_merges: usize,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
    precision_ci_low: Option<f64>,
    precision_ci_high: Option<f64>,
}

fn decision_says_same(decision: &Option<String>) -> bool {
    matches!(decision.as_deref(), Some("merge") | Some("potential_alias"))
}

fn decision_is_merge(decision: &Option<String>) -> bool {
    matches!(decision.as_deref(), Some("merge"))
}

/// Compute precision/recall/F1/Wilson-CI per §3.1's denominator definition,
/// scoped to a slice of `RowOutcome`s already filtered to a category (or the
/// whole site). `Uncertain` and `NotNominated` rows are excluded entirely —
/// callers must pre-filter or accept they contribute nothing (both branches
/// below `continue` on those variants, so passing them in is harmless but
/// contributes 0 to every count except `n`... actually `n` is set to
/// `outcomes.len()` upfront, so category-level callers that mix ground-truth
/// kinds should filter BEFORE calling if they want `n` to mean "scored rows"
/// — see call sites below for how each category is filtered before this fn
/// runs.
fn compute_metrics(outcomes: &[&RowOutcome]) -> CategoryMetrics {
    let mut m = CategoryMetrics {
        n: outcomes.len(),
        ..Default::default()
    };

    for o in outcomes {
        let truly_same = match o.ground_truth {
            GroundTruth::Same => true,
            GroundTruth::Different => false,
            // Excluded from the hard precision/recall gate entirely.
            GroundTruth::NotNominated | GroundTruth::Uncertain => continue,
        };
        let pass_says_same = decision_says_same(&o.decision);

        match (pass_says_same, truly_same) {
            (true, true) => m.tp += 1,
            (true, false) => {
                m.fp += 1;
                if decision_is_merge(&o.decision) {
                    m.false_merges += 1;
                }
            }
            (false, true) => m.fn_ += 1,
            (false, false) => m.tn += 1,
        }
    }

    let flagged_same = m.tp + m.fp;
    let genuinely_same_total = m.tp + m.fn_;

    m.precision = if flagged_same == 0 {
        None
    } else {
        Some(m.tp as f64 / flagged_same as f64)
    };
    m.recall = if genuinely_same_total == 0 {
        None
    } else {
        Some(m.tp as f64 / genuinely_same_total as f64)
    };
    m.f1 = match (m.precision, m.recall) {
        (Some(p), Some(r)) if (p + r) > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    if flagged_same > 0 {
        let (lo, hi) = wilson_lower_upper(m.tp, flagged_same);
        m.precision_ci_low = Some(lo);
        m.precision_ci_high = Some(hi);
    }

    m
}

/// Merge-only precision (stricter, destructive-write-only measure): of the
/// pairs the pass actually MERGED (not PotentialAlias), what fraction are
/// genuinely the same entity?
fn compute_merge_only_precision(outcomes: &[&RowOutcome]) -> (usize, usize, Option<f64>) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    for o in outcomes {
        let truly_same = match o.ground_truth {
            GroundTruth::Same => true,
            GroundTruth::Different => false,
            GroundTruth::NotNominated | GroundTruth::Uncertain => continue,
        };
        if !decision_is_merge(&o.decision) {
            continue;
        }
        if truly_same {
            tp += 1;
        } else {
            fp += 1;
        }
    }
    let precision = if tp + fp == 0 {
        None
    } else {
        Some(tp as f64 / (tp + fp) as f64)
    };
    (tp, fp, precision)
}

// ─── o11y cross-check (§3.5) ─────────────────────────────────────────────────

/// Sum a named counter, optionally filtered by `site` label. Takes the
/// `Snapshotter` (not a `Snapshot` — `Snapshot` is not `Clone`, and
/// `into_vec()` consumes it) so each call takes its own fresh snapshot.
fn sum_counter(snapshotter: &Snapshotter, metric_name: &str, site_filter: Option<&str>) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != metric_name {
                return None;
            }
            if let Some(site) = site_filter {
                let labels: HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("site").copied() != Some(site) {
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

/// Sum the `kremory.identity.write_gate_decision_total{site,decision}`
/// counter for one specific `(site, decision)` label pair.
fn sum_write_gate_decision_counter(snapshotter: &Snapshotter, site: &str, decision: &str) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.identity.write_gate_decision_total" {
                return None;
            }
            let labels: HashMap<&str, &str> = key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("site").copied() == Some(site)
                && labels.get("decision").copied() == Some(decision)
            {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum()
}

/// Cross-check the pass's own o11y counters (§3.5) against the harness's
/// observed decisions. A divergence is a FAIL — the counter must not lie.
fn crosscheck_o11y(snapshotter: &Snapshotter, outcomes: &[RowOutcome]) -> Result<(), String> {
    const SITE: &str = "site5_acronym_nickname";

    let observed_nominated: u64 = outcomes.iter().map(|o| o.candidates_nominated as u64).sum();
    let counter_nominated = sum_counter(
        snapshotter,
        "kremory.identity.candidate_nominated_total",
        Some(SITE),
    );
    if observed_nominated != counter_nominated {
        return Err(format!(
            "o11y divergence: kremory.identity.candidate_nominated_total{{site={SITE}}} = \
             {counter_nominated}, but harness observed {observed_nominated} nominations \
             summed across report.candidates_nominated per row"
        ));
    }

    let observed_merges = outcomes
        .iter()
        .filter(|o| decision_is_merge(&o.decision))
        .count() as u64;
    let counter_merges = sum_counter(
        snapshotter,
        "kremory.dream.acronym_recall.merges_applied_total",
        None,
    );
    if observed_merges != counter_merges {
        return Err(format!(
            "o11y divergence: kremory.dream.acronym_recall.merges_applied_total = \
             {counter_merges}, but harness observed {observed_merges} merge decisions \
             via identity_verdict_audit"
        ));
    }

    let observed_examined: u64 = outcomes.len() as u64; // each isolated group examines exactly 1 pair
    let counter_examined = sum_counter(
        snapshotter,
        "kremory.dream.acronym_recall.pairs_examined_total",
        None,
    );
    if observed_examined != counter_examined {
        return Err(format!(
            "o11y divergence: kremory.dream.acronym_recall.pairs_examined_total = \
             {counter_examined}, but harness ran {observed_examined} isolated 1-pair groups \
             (one pair examined per corpus row's isolated group)"
        ));
    }

    for decision_label in ["merge", "potential_alias", "reject"] {
        let observed = outcomes
            .iter()
            .filter(|o| o.decision.as_deref() == Some(decision_label))
            .count() as u64;
        let counter_labeled = sum_write_gate_decision_counter(snapshotter, SITE, decision_label);
        if observed != counter_labeled {
            return Err(format!(
                "o11y divergence: kremory.identity.write_gate_decision_total{{site={SITE},\
                 decision={decision_label}}} = {counter_labeled}, but harness observed \
                 {observed} \"{decision_label}\" decisions via identity_verdict_audit"
            ));
        }
    }

    // Harness-owned false-merge counter (spec brief) — never in production
    // source, incremented here in the harness's own scoring code whenever a
    // SAFETY-category row (coincidental_collision_distinct /
    // distinct_people_same_nickname) lands on `merge`.
    for o in outcomes {
        let is_safety_category = matches!(
            o.category.as_str(),
            "coincidental_collision_distinct" | "distinct_people_same_nickname"
        );
        if is_safety_category && decision_is_merge(&o.decision) {
            metrics::counter!(
                "dream.corpus.false_merge_total",
                "site" => "site5_acronym_nickname",
                "category" => o.category.clone()
            )
            .increment(1);
        }
    }

    Ok(())
}

// ─── Report emission ─────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct MetricsReport {
    site: String,
    overall: CategoryMetrics,
    per_category: std::collections::BTreeMap<String, CategoryMetrics>,
    merge_only_precision: Option<f64>,
    merge_only_tp: usize,
    merge_only_fp: usize,
    n_flagged_same: usize,
    n_not_nominated: usize,
    n_not_nominated_confirmed: usize,
    uncertain_thin_context_n: usize,
    uncertain_thin_context_rows: Vec<String>,
    safety_false_merges: usize,
    false_merge_rows: Vec<String>,
    hard_gate: String,
}

fn print_and_write_report(report: &MetricsReport) {
    eprintln!("\n── Site #5 metrics harness — full report ──────────────────────────────");
    eprintln!(
        "  OVERALL: n={} precision={:?} recall={:?} f1={:?} ci=[{:?},{:?}]",
        report.overall.n,
        report.overall.precision,
        report.overall.recall,
        report.overall.f1,
        report.overall.precision_ci_low,
        report.overall.precision_ci_high,
    );
    for (cat, m) in &report.per_category {
        eprintln!(
            "  [{cat}] n={} tp={} fp={} fn={} tn={} false_merges={} precision={:?} \
             recall={:?} f1={:?} ci=[{:?},{:?}]",
            m.n,
            m.tp,
            m.fp,
            m.fn_,
            m.tn,
            m.false_merges,
            m.precision,
            m.recall,
            m.f1,
            m.precision_ci_low,
            m.precision_ci_high,
        );
    }
    eprintln!(
        "\n  merge_only_precision={:?} (tp={} fp={})",
        report.merge_only_precision, report.merge_only_tp, report.merge_only_fp,
    );
    eprintln!(
        "\n  n_flagged_same={} n_not_nominated={} n_not_nominated_confirmed={} \
         uncertain_thin_context_n={} safety_false_merges={} hard_gate={}",
        report.n_flagged_same,
        report.n_not_nominated,
        report.n_not_nominated_confirmed,
        report.uncertain_thin_context_n,
        report.safety_false_merges,
        report.hard_gate,
    );
    if !report.false_merge_rows.is_empty() {
        eprintln!("  FALSE MERGE ROWS:");
        for r in &report.false_merge_rows {
            eprintln!("    *** {r} ***");
        }
    }

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site5_metrics.json");
    let json = serde_json::to_string_pretty(report).expect("serialize metrics report");
    std::fs::write(&out_path, json)
        .unwrap_or_else(|e| panic!("failed to write metrics report to {out_path:?}: {e}"));
    eprintln!("  metrics JSON written to {out_path:?}");
}

// ─── smoke-one-before-batch ───────────────────────────────────────────────────

/// smoke-one-before-batch (hard rule): run ONLY corpus row `s5-001` (IBM /
/// International Business Machines, `genuine_acronym`) through the full
/// scoring pipeline first, confirm the per-row plant->run->score wiring is
/// sane, THEN proceed to the full 121-row corpus run.
#[tokio::test]
#[ignore = "dream_metrics_harness: requires Ollama in record mode, or a committed cassette in \
            replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test dream_metrics_harness -- --ignored \
            --nocapture smoke_one_metrics_harness"]
async fn smoke_one_metrics_harness() {
    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    let smoke_row = corpus
        .iter()
        .find(|r| r.id == "s5-001")
        .expect("corpus must contain s5-001 (IBM/International Business Machines)");
    assert_eq!(smoke_row.a, "IBM");
    assert_eq!(smoke_row.category, "genuine_acronym");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (outcome, gid) = run_one_row(smoke_row, mode)
        .await
        .unwrap_or_else(|e| panic!("smoke-one row failed: {e}"));

    eprintln!(
        "[smoke-one] row_id={} a={} b={} category={} ground_truth={:?} decision={:?} \
         candidates_nominated={} group_id={gid}",
        outcome.row_id,
        smoke_row.a,
        smoke_row.b,
        outcome.category,
        outcome.ground_truth,
        outcome.decision,
        outcome.candidates_nominated,
    );

    assert_eq!(
        outcome.candidates_nominated, 1,
        "smoke-one: IBM/International Business Machines must nominate (initialism \
         structural test fires deterministically)"
    );
    assert_eq!(
        outcome.decision.as_deref(),
        Some("merge"),
        "smoke-one: a true acronym pair with a live LLM call must reach write_gate \
         row 5 (Merge) — got decision={:?}",
        outcome.decision,
    );

    // Cross-check o11y for this single row too — catches wiring bugs early.
    crosscheck_o11y(&snapshotter, std::slice::from_ref(&outcome))
        .unwrap_or_else(|e| panic!("smoke-one o11y cross-check FAILED: {e}"));

    eprintln!("[smoke-one] PASS — proceeding to full 121-row corpus run is safe.");
}

// ─── Full corpus run — the binding metrics harness ───────────────────────────

#[tokio::test]
#[ignore = "dream_metrics_harness: requires Ollama in record mode, or a committed cassette in \
            replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test dream_metrics_harness -- --ignored \
            --nocapture full_corpus_site5_metrics"]
async fn full_corpus_site5_metrics() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    assert_eq!(corpus.len(), 121, "corpus must have exactly 121 rows");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes: Vec<RowOutcome> = Vec::with_capacity(corpus.len());
    for row in &corpus {
        let (outcome, _gid) = run_one_row(row, mode)
            .await
            .unwrap_or_else(|e| panic!("row {} failed: {e}", row.id));
        outcomes.push(outcome);
    }

    // ── o11y cross-check (§3.5) — divergence is a FAIL, never a soft warning ──
    // (this also increments the harness-owned dream.corpus.false_merge_total
    // counter for any safety-category row that landed on merge)
    crosscheck_o11y(&snapshotter, &outcomes)
        .unwrap_or_else(|e| panic!("o11y cross-check FAILED: {e}"));

    // ── Per-category metrics ──────────────────────────────────────────────
    let mut categories: std::collections::BTreeMap<String, Vec<&RowOutcome>> =
        std::collections::BTreeMap::new();
    for o in &outcomes {
        categories.entry(o.category.clone()).or_default().push(o);
    }
    let per_category: std::collections::BTreeMap<String, CategoryMetrics> = categories
        .iter()
        .map(|(cat, os)| (cat.clone(), compute_metrics(os)))
        .collect();

    // ── Overall (site-level) metrics — every row EXCEPT NotNominated /
    // Uncertain (excluded inside compute_metrics per-row) ──────────────────
    let all_refs: Vec<&RowOutcome> = outcomes.iter().collect();
    let overall = compute_metrics(&all_refs);
    let (merge_only_tp, merge_only_fp, merge_only_precision) =
        compute_merge_only_precision(&all_refs);

    // ── SAFETY gate (task brief, exact-zero): coincidental_collision_distinct
    // + distinct_people_same_nickname must NEVER merge ──────────────────────
    let safety_categories = [
        "coincidental_collision_distinct",
        "distinct_people_same_nickname",
    ];
    let mut safety_false_merges = 0usize;
    let mut false_merge_rows: Vec<String> = Vec::new();
    for cat in safety_categories {
        if let Some(m) = per_category.get(cat) {
            safety_false_merges += m.false_merges;
        }
    }
    for o in &outcomes {
        if safety_categories.contains(&o.category.as_str()) && decision_is_merge(&o.decision) {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            false_merge_rows.push(format!(
                "{} ({} / {}, category={})",
                o.row_id, row.a, row.b, o.category
            ));
        }
    }

    // ── non_cooccurring_nickname nomination-rate / gap-honesty check ───────
    let not_nominated_rows: Vec<&RowOutcome> = outcomes
        .iter()
        .filter(|o| o.ground_truth == GroundTruth::NotNominated)
        .collect();
    let n_not_nominated = not_nominated_rows.len();
    let n_not_nominated_confirmed = not_nominated_rows
        .iter()
        .filter(|o| o.candidates_nominated == 0)
        .count();
    eprintln!(
        "\n[nomination-rate honesty] non_cooccurring_nickname: {n_not_nominated} rows, \
         {n_not_nominated_confirmed} confirmed NOT nominated (expected: all {n_not_nominated})"
    );
    for o in &not_nominated_rows {
        if o.candidates_nominated != 0 {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            eprintln!(
                "  *** UNEXPECTED: {} ({} / {}) WAS nominated despite cooccurs=false and no \
                 initialism relationship — ALT-001 gap assumption violated for this row ***",
                o.row_id, row.a, row.b
            );
        }
    }

    // ── uncertain_thin_context — descriptive report only, never gated ──────
    let uncertain_outcomes: Vec<&RowOutcome> = outcomes
        .iter()
        .filter(|o| o.ground_truth == GroundTruth::Uncertain)
        .collect();
    let uncertain_thin_context_rows: Vec<String> = uncertain_outcomes
        .iter()
        .map(|o| {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            format!(
                "{} ({} / {}) — decision={:?} pass_says_same={}",
                o.row_id,
                row.a,
                row.b,
                o.decision,
                decision_says_same(&o.decision)
            )
        })
        .collect();

    let n_flagged_same = overall.tp + overall.fp;

    // ── HARD GATE (spec §4.1 honest fallback): point-estimate aggregate
    // precision >= 0.90 AND false_merges == 0 on both safety categories.
    // Wilson lower-CI is reported as a quality signal, NOT the binding gate —
    // the corpus is N-limited (~68-93 flagged-same rows expected), short of
    // the ideal ~77-per-side Wilson-clearing N in the worst case. ─────────
    let point_precision = overall.precision.unwrap_or(0.0);
    let hard_gate_pass = point_precision >= 0.90 && safety_false_merges == 0;
    let hard_gate = if hard_gate_pass { "PASS" } else { "FAIL" }.to_string();

    let report = MetricsReport {
        site: "site5_acronym_nickname".to_string(),
        overall: overall.clone(),
        per_category,
        merge_only_precision,
        merge_only_tp,
        merge_only_fp,
        n_flagged_same,
        n_not_nominated,
        n_not_nominated_confirmed,
        uncertain_thin_context_n: uncertain_outcomes.len(),
        uncertain_thin_context_rows: uncertain_thin_context_rows.clone(),
        safety_false_merges,
        false_merge_rows: false_merge_rows.clone(),
        hard_gate: hard_gate.clone(),
    };
    print_and_write_report(&report);

    eprintln!(
        "\n── uncertain_thin_context (descriptive only, NOT part of the gate) — n={} ──",
        uncertain_thin_context_rows.len()
    );
    for line in &uncertain_thin_context_rows {
        eprintln!("  {line}");
    }

    // ── Per-row rationale dump for any FP/FN, so a failing gate is
    // immediately diagnosable without re-running with KREMORY_DEBUG=1 ──────
    for row in &corpus {
        let outcome = outcomes.iter().find(|o| o.row_id == row.id).unwrap();
        let truly_same = match outcome.ground_truth {
            GroundTruth::Same => Some(true),
            GroundTruth::Different => Some(false),
            GroundTruth::NotNominated | GroundTruth::Uncertain => None,
        };
        let pass_says_same = decision_says_same(&outcome.decision);
        if let Some(truly_same) = truly_same {
            if truly_same != pass_says_same {
                eprintln!(
                    "  *** MISCLASSIFIED [{}] {} / {} — truly_same={} decision={:?} \
                     rationale={:?} ***",
                    outcome.category, row.a, row.b, truly_same, outcome.decision, row.rationale,
                );
            }
        }
    }

    // ── Hard assertions ───────────────────────────────────────────────────
    assert_eq!(
        safety_false_merges, 0,
        "SAFETY GATE FAILED: {safety_false_merges} false merge(s) among \
         coincidental_collision_distinct / distinct_people_same_nickname (safety categories, \
         ground truth = genuinely DIFFERENT entities). Literal-equality gate, never loosened. \
         See false_merge_rows above for exactly which corpus row(s) caused it."
    );
    assert!(
        overall.precision.is_some(),
        "aggregate precision undefined (no positive 'same' calls at all) — the corpus or the \
         pass regressed; see per-row report above"
    );
    assert!(
        point_precision >= 0.90,
        "PRECISION GATE FAILED: point-estimate precision {point_precision:.4} < 0.90 bar — \
         n_flagged_same={n_flagged_same}, tp={} fp={}. Wilson 95% CI=[{:?}, {:?}]. See per-row \
         misclassification report above.",
        overall.tp,
        overall.fp,
        overall.precision_ci_low,
        overall.precision_ci_high,
    );

    eprintln!(
        "\n══ GATE: {hard_gate} (precision={point_precision:.4} >= 0.90, \
         false_merges={safety_false_merges}) ═══════════════"
    );
}

//! Layer B entity extraction scorer — P/R/F1 over ground-truth entity sets.
//!
//! Promoted from `crates/kremory/tests/spike_programmatic_extraction.rs` (Day 3).
//!
//! # What this measures
//!
//! `scan_proper_nouns()` is the zero-LLM extraction path inside kremory's
//! ingest pipeline.  This module runs it against 14 domain fixture texts and
//! compares the output to a manually-curated `ground_truth.json`.  It computes
//! precision, recall, and F1 per fixture then aggregates them into an
//! `EntityExtractionReport`.
//!
//! # Matching strategy
//!
//! Fuzzy substring match (case-insensitive): extracted entity `e` is considered
//! a true positive for expected entity `x` when `e.contains(x) || x.contains(e)`.
//! This tolerates minor surface variations (e.g. "Amazon" matches "Amazon Robotics"
//! and vice-versa) while being deterministic and LLM-free.

use std::collections::HashMap;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{
    Score,
    scorers::f1::{F1Output, F1Sample, F1Scorer},
    types::{EvalErr, EvalLayer, EvalReport, ScoreMetadata},
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-fixture result from the entity extraction eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureScore {
    /// Fixture key (e.g. `"mock_interview"`).
    pub fixture_key: String,
    /// Human-readable domain name.
    pub domain: String,
    /// Precision of `scan_proper_nouns` on this fixture.
    pub precision: f64,
    /// Recall of `scan_proper_nouns` on this fixture.
    pub recall: f64,
    /// F1 harmonic mean.
    pub f1: f64,
    /// True positives count.
    pub tp: usize,
    /// False positives count.
    pub fp: usize,
    /// False negatives count.
    pub r#fn: usize,
    /// Number of entities extracted (TP + FP).
    pub extracted_count: usize,
    /// Number of expected entities in ground truth.
    pub expected_count: usize,
    /// Full `Score` object (value = f1, metadata = P/R/TP/FP/FN).
    #[serde(flatten)]
    pub score: Score,
}

/// Aggregate report for one entity-extraction eval run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionReport {
    /// Overall F1 across all fixtures (macro-average).
    pub overall_f1: f64,
    /// Overall precision (macro-average).
    pub overall_precision: f64,
    /// Overall recall (macro-average).
    pub overall_recall: f64,
    /// Total TP across all fixtures.
    pub total_tp: usize,
    /// Total FP across all fixtures.
    pub total_fp: usize,
    /// Total FN across all fixtures.
    pub total_fn: usize,
    /// Per-fixture scores.
    pub fixtures: Vec<FixtureScore>,
    /// Standard eval-harness aggregate report.
    pub eval_report: EvalReport,
}

// ---------------------------------------------------------------------------
// Fixture catalogue
// ---------------------------------------------------------------------------

/// One entry in the fixture catalogue.
struct FixtureMeta {
    /// Short human-readable name.
    name: &'static str,
    /// Key used in `ground_truth.json`.
    key: &'static str,
}

fn fixture_catalogue() -> Vec<FixtureMeta> {
    vec![
        FixtureMeta {
            name: "Mock Interview",
            key: "mock_interview",
        },
        FixtureMeta {
            name: "Medical Consultation",
            key: "medical_consultation",
        },
        FixtureMeta {
            name: "Legal Deposition",
            key: "legal_deposition",
        },
        FixtureMeta {
            name: "Tech Standup",
            key: "tech_standup",
        },
        FixtureMeta {
            name: "Sales Call",
            key: "sales_call",
        },
        FixtureMeta {
            name: "Podcast Interview",
            key: "podcast_interview",
        },
        FixtureMeta {
            name: "Board Meeting",
            key: "board_meeting",
        },
        FixtureMeta {
            name: "News Article",
            key: "news_article",
        },
        FixtureMeta {
            name: "Academic Lecture",
            key: "academic_lecture",
        },
        FixtureMeta {
            name: "Customer Support",
            key: "customer_support",
        },
        FixtureMeta {
            name: "Slack Thread",
            key: "slack_thread",
        },
        FixtureMeta {
            name: "Product Review",
            key: "product_review",
        },
        FixtureMeta {
            name: "Short Snippet",
            key: "short_snippet",
        },
        FixtureMeta {
            name: "Long Report",
            key: "long_report",
        },
    ]
}

// ---------------------------------------------------------------------------
// Ground-truth loading
// ---------------------------------------------------------------------------

fn load_ground_truth(
    gt_path: &Path,
) -> Result<HashMap<String, Vec<String>>, EvalErr> {
    let raw = std::fs::read_to_string(gt_path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let obj = value
        .as_object()
        .ok_or_else(|| EvalErr::MissingField("ground_truth root must be an object".into()))?;

    let mut map = HashMap::new();
    for (key, domain) in obj {
        let entities = domain["entities"]
            .as_array()
            .ok_or_else(|| EvalErr::MissingField(format!("{key}.entities must be an array")))?
            .iter()
            .map(|e| {
                e["name"]
                    .as_str()
                    .ok_or_else(|| {
                        EvalErr::MissingField(format!("{key}.entities[].name must be a string"))
                    })
                    .map(|s| s.to_lowercase())
            })
            .collect::<Result<Vec<_>, _>>()?;
        map.insert(key.clone(), entities);
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/// Case-insensitive substring match used for recall/precision.
///
/// `extracted` is considered a match for `expected` when either string is a
/// substring of the other.  This is the same logic as the original spikes.
fn fuzzy_match(extracted: &str, expected: &str) -> bool {
    let e = extracted.to_lowercase();
    let x = expected.to_lowercase();
    e == x || e.contains(x.as_str()) || x.contains(e.as_str())
}

/// For F1Scorer compatibility: normalise extracted names so that each expected
/// entity is matched using fuzzy logic.
///
/// Returns the subset of `extracted_names` that are fuzzy-matched by at least
/// one element of `expected_names`, and adds phantom true-positives for expected
/// entities found by fuzzy but not exact match.
///
/// Because `F1Scorer` uses exact (case-insensitive) matching, we instead compute
/// TP/FP/FN ourselves using fuzzy logic and then construct the `Score` directly.
fn compute_fuzzy_f1(
    extracted_names: &[String],
    expected_names: &[String],
) -> (usize, usize, usize) {
    // TP: expected entities that have at least one fuzzy match in extracted
    let tp = expected_names
        .iter()
        .filter(|exp| {
            extracted_names
                .iter()
                .any(|ext| fuzzy_match(ext, exp))
        })
        .count();

    // FP: extracted entities that do NOT fuzzy-match any expected entity
    let fp = extracted_names
        .iter()
        .filter(|ext| {
            !expected_names
                .iter()
                .any(|exp| fuzzy_match(ext, exp))
        })
        .count();

    // FN: expected entities that have NO fuzzy match in extracted
    let fn_ = expected_names.len().saturating_sub(tp);

    (tp, fp, fn_)
}

// ---------------------------------------------------------------------------
// Core runner
// ---------------------------------------------------------------------------

/// Run entity extraction eval against all fixtures in `fixtures_dir`.
///
/// `fixtures_dir` must contain:
/// - `ground_truth.json`
/// - `<domain_key>.txt` for each domain in the catalogue
///
/// Returns an [`EntityExtractionReport`] with per-fixture and aggregate scores.
pub fn run(fixtures_dir: &Path) -> Result<EntityExtractionReport, EvalErr> {
    let gt_path = fixtures_dir.join("ground_truth.json");
    let ground_truth = load_ground_truth(&gt_path)?;

    let scorer = F1Scorer::new();
    let catalogue = fixture_catalogue();
    let mut fixture_scores: Vec<FixtureScore> = Vec::with_capacity(catalogue.len());

    for meta in &catalogue {
        let txt_path = fixtures_dir.join(format!("{}.txt", meta.key));
        let text = std::fs::read_to_string(&txt_path).map_err(|e| {
            EvalErr::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read {}: {e}", txt_path.display()),
            ))
        })?;

        let expected = ground_truth
            .get(meta.key)
            .ok_or_else(|| {
                EvalErr::MissingField(format!(
                    "no ground truth for fixture key '{}'",
                    meta.key
                ))
            })?;

        // Run scan_proper_nouns (zero-LLM extraction)
        let extracted = kremory::core::text_utils::scan_proper_nouns(&text, &[]);
        let extracted_names: Vec<String> =
            extracted.iter().map(|e| e.name.to_lowercase()).collect();

        // Compute TP/FP/FN with fuzzy matching
        let (tp, fp, fn_) = compute_fuzzy_f1(&extracted_names, expected);

        let precision = if tp + fp == 0 {
            if expected.is_empty() { 1.0 } else { 0.0 }
        } else {
            tp as f64 / (tp + fp) as f64
        };

        let recall = if tp + fn_ == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        // Also run through F1Scorer for normalised Score object consistency
        let f1_sample = F1Sample {
            id: meta.key.to_string(),
            expected: expected.clone(),
        };
        let f1_output = F1Output {
            predicted: extracted_names.clone(),
        };
        // score_sync uses exact (case-insensitive) matching; we override value
        // with our fuzzy F1 but keep the Score structure.
        let base_score = scorer.score_sync(&f1_sample, &f1_output)?;

        let metadata = ScoreMetadata {
            precision: Some(precision),
            recall: Some(recall),
            f1: Some(f1),
            true_positives: Some(tp),
            false_positives: Some(fp),
            false_negatives: Some(fn_),
            ..Default::default()
        };

        let reasoning = format!(
            "P={:.3} R={:.3} F1={:.3} (TP={} FP={} FN={}) [fuzzy match]",
            precision, recall, f1, tp, fp, fn_
        );

        let score = Score {
            value: f1,
            reasoning,
            metadata: metadata.into(),
        };

        // Suppress unused variable warning from exact-match scorer result
        let _ = base_score;

        fixture_scores.push(FixtureScore {
            fixture_key: meta.key.to_string(),
            domain: meta.name.to_string(),
            precision,
            recall,
            f1,
            tp,
            fp,
            r#fn: fn_,
            extracted_count: extracted.len(),
            expected_count: expected.len(),
            score,
        });
    }

    // Aggregate (macro-average over fixtures)
    let n = fixture_scores.len() as f64;
    let overall_precision = fixture_scores.iter().map(|f| f.precision).sum::<f64>() / n;
    let overall_recall = fixture_scores.iter().map(|f| f.recall).sum::<f64>() / n;
    let overall_f1 = fixture_scores.iter().map(|f| f.f1).sum::<f64>() / n;
    let total_tp = fixture_scores.iter().map(|f| f.tp).sum();
    let total_fp = fixture_scores.iter().map(|f| f.fp).sum();
    let total_fn = fixture_scores.iter().map(|f| f.r#fn).sum();

    let per_category: HashMap<String, f64> = fixture_scores
        .iter()
        .map(|f| (f.fixture_key.clone(), f.f1))
        .collect();

    let eval_report = EvalReport {
        layer: EvalLayer::B,
        benchmark: "entity-extraction".into(),
        overall_score: overall_f1,
        per_category,
        token_efficiency: None,
        timestamp: Utc::now(),
        kremory_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    Ok(EntityExtractionReport {
        overall_f1,
        overall_precision,
        overall_recall,
        total_tp,
        total_fp,
        total_fn,
        fixtures: fixture_scores,
        eval_report,
    })
}

/// Write the report as a JSONL file (one JSON object per fixture, then summary).
///
/// Creates `output_dir` if it does not exist.
pub fn write_jsonl(
    report: &EntityExtractionReport,
    output_path: &Path,
) -> Result<(), EvalErr> {
    use std::io::Write;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::File::create(output_path)?;

    for fixture_score in &report.fixtures {
        let line = serde_json::to_string(fixture_score)?;
        writeln!(file, "{}", line)?;
    }

    // Summary line
    let summary = serde_json::json!({
        "summary": true,
        "overall_f1": report.overall_f1,
        "overall_precision": report.overall_precision,
        "overall_recall": report.overall_recall,
        "total_tp": report.total_tp,
        "total_fp": report.total_fp,
        "total_fn": report.total_fn,
        "fixture_count": report.fixtures.len(),
    });
    writeln!(file, "{}", summary)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Smoke test: fuzzy_match behaves correctly for common cases.
    #[test]
    fn fuzzy_match_basic() {
        assert!(fuzzy_match("amazon robotics", "amazon"));
        assert!(fuzzy_match("amazon", "amazon robotics"));
        assert!(fuzzy_match("Alice", "alice"));
        assert!(!fuzzy_match("bob", "alice"));
    }

    /// compute_fuzzy_f1: perfect recall, no false positives.
    #[test]
    fn fuzzy_f1_perfect() {
        let extracted = vec!["alice".into(), "bob".into()];
        let expected = vec!["alice".into(), "bob".into()];
        let (tp, fp, fn_) = compute_fuzzy_f1(&extracted, &expected);
        assert_eq!(tp, 2);
        assert_eq!(fp, 0);
        assert_eq!(fn_, 0);
    }

    /// compute_fuzzy_f1: all extracted are false positives.
    #[test]
    fn fuzzy_f1_all_fp() {
        let extracted = vec!["charlie".into(), "dave".into()];
        let expected = vec!["alice".into(), "bob".into()];
        let (tp, fp, fn_) = compute_fuzzy_f1(&extracted, &expected);
        assert_eq!(tp, 0);
        assert_eq!(fp, 2);
        assert_eq!(fn_, 2);
    }

    /// compute_fuzzy_f1: fuzzy substring — "amazon" extracted catches "amazon robotics" expected.
    #[test]
    fn fuzzy_f1_substring_match() {
        let extracted = vec!["amazon".into()];
        let expected = vec!["amazon robotics".into()];
        let (tp, fp, fn_) = compute_fuzzy_f1(&extracted, &expected);
        assert_eq!(tp, 1, "substring match should count as TP");
        assert_eq!(fp, 0, "no false positives when fully matched");
        assert_eq!(fn_, 0, "no false negatives when expected matched");
    }

    /// F1 formula is correct for partial overlap.
    #[test]
    fn f1_formula_partial() {
        // TP=2, FP=0, FN=1 → P=1.0, R=2/3 → F1 ≈ 0.8
        let extracted = vec!["alice".into(), "bob".into()];
        let expected = vec!["alice".into(), "bob".into(), "charlie".into()];
        let (tp, fp, fn_) = compute_fuzzy_f1(&extracted, &expected);
        assert_eq!(tp, 2);
        assert_eq!(fp, 0);
        assert_eq!(fn_, 1);

        let p = tp as f64 / (tp + fp) as f64;
        let r = tp as f64 / (tp + fn_) as f64;
        let f1 = 2.0 * p * r / (p + r);
        assert!((f1 - 0.8).abs() < 1e-9);
    }
}

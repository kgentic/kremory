//! F1 scorer — entity-set precision, recall, and F1.
//!
//! Compares two `Vec<String>` entity sets (predicted vs ground-truth) and
//! returns a `Score` with `value` = F1 and `metadata` containing precision,
//! recall, TP/FP/FN counts.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    types::{EvalError, ScoreMetadata},
    Score, TieBreakPolicy,
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// Input to the F1 scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1Sample {
    /// Identifier for this sample.
    pub id: String,
    /// Ground-truth entity set.
    pub expected: Vec<String>,
}

/// Output produced by the solver for this sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1Output {
    /// Predicted entity set.
    pub predicted: Vec<String>,
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// F1 scorer over entity sets.
#[derive(Debug, Clone, Default)]
pub struct F1Scorer {
    /// Normalise entity strings to lowercase before comparing.
    pub case_insensitive: bool,
    /// Trim entity strings before comparing.
    pub trim: bool,
    /// Tie-break policy for threshold-based reporting.
    pub tie_break: TieBreakPolicy,
}

impl F1Scorer {
    /// Create with sensible defaults.
    pub fn new() -> Self {
        Self {
            case_insensitive: true,
            trim: true,
            tie_break: TieBreakPolicy::Pass,
        }
    }

    fn normalise(&self, s: &str) -> String {
        let s = if self.trim { s.trim() } else { s };
        if self.case_insensitive {
            s.to_lowercase()
        } else {
            s.to_owned()
        }
    }

    /// Compute F1 synchronously.
    pub fn score_sync(&self, sample: &F1Sample, output: &F1Output) -> EvalError<Score> {
        let expected: HashSet<String> = sample.expected.iter().map(|s| self.normalise(s)).collect();

        let predicted: HashSet<String> =
            output.predicted.iter().map(|s| self.normalise(s)).collect();

        let tp = expected.intersection(&predicted).count();
        let fp = predicted.difference(&expected).count();
        let fn_ = expected.difference(&predicted).count();

        let precision = if tp + fp == 0 {
            // No predictions → precision is 1.0 only if expected is also empty.
            if expected.is_empty() {
                1.0
            } else {
                0.0
            }
        } else {
            tp as f64 / (tp + fp) as f64
        };

        let recall = if tp + fn_ == 0 {
            // No ground-truth entities → recall is 1.0 (nothing to recall).
            1.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        let reasoning = format!(
            "P={:.3} R={:.3} F1={:.3} (TP={} FP={} FN={})",
            precision, recall, f1, tp, fp, fn_
        );

        let metadata = ScoreMetadata {
            precision: Some(precision),
            recall: Some(recall),
            f1: Some(f1),
            true_positives: Some(tp),
            false_positives: Some(fp),
            false_negatives: Some(fn_),
            ..Default::default()
        };

        Ok(Score {
            value: f1,
            reasoning,
            metadata: metadata.into(),
        })
    }
}

impl crate::Scorer<F1Sample, F1Output> for F1Scorer {
    async fn score(&self, sample: &F1Sample, output: &F1Output) -> EvalError<Score> {
        self.score_sync(sample, output)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn f1_perfect_match() {
        let scorer = F1Scorer::new();
        let sample = F1Sample {
            id: "t1".into(),
            expected: vec!["Alice".into(), "Bob".into()],
        };
        let output = F1Output {
            predicted: vec!["Alice".into(), "Bob".into()],
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert!((score.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn f1_no_overlap() {
        let scorer = F1Scorer::new();
        let sample = F1Sample {
            id: "t2".into(),
            expected: vec!["Alice".into(), "Bob".into()],
        };
        let output = F1Output {
            predicted: vec!["Charlie".into(), "Dave".into()],
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert!((score.value - 0.0).abs() < 1e-9);
    }

    #[test]
    fn f1_partial_overlap() {
        let scorer = F1Scorer::new();
        let sample = F1Sample {
            id: "t3".into(),
            expected: vec!["Alice".into(), "Bob".into(), "Charlie".into()],
        };
        let output = F1Output {
            predicted: vec!["Alice".into(), "Bob".into()],
        };
        // TP=2, FP=0, FN=1 → P=1.0, R=2/3 → F1 = 2*(1.0*0.667)/(1.0+0.667) ≈ 0.8
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert!(score.value > 0.0 && score.value < 1.0);

        let meta: ScoreMetadata = serde_json::from_value(score.metadata).unwrap();
        assert_eq!(meta.true_positives, Some(2));
        assert_eq!(meta.false_positives, Some(0));
        assert_eq!(meta.false_negatives, Some(1));
    }

    #[test]
    fn f1_case_insensitive_normalisation() {
        let scorer = F1Scorer::new(); // case_insensitive=true by default
        let sample = F1Sample {
            id: "t4".into(),
            expected: vec!["Alice".into()],
        };
        let output = F1Output {
            predicted: vec!["alice".into()],
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert!((score.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn f1_empty_both() {
        let scorer = F1Scorer::new();
        let sample = F1Sample {
            id: "t5".into(),
            expected: vec![],
        };
        let output = F1Output { predicted: vec![] };
        let score = scorer.score_sync(&sample, &output).unwrap();
        // Both empty → precision=1.0, recall=1.0, F1=1.0
        assert!((score.value - 1.0).abs() < 1e-9);
    }
}

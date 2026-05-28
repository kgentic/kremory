//! Exact-match scorer — deterministic string comparison.
//!
//! Returns `Score{value: 1.0}` when the output matches the expected string
//! exactly (after optional normalisation), `Score{value: 0.0}` otherwise.

use serde::{Deserialize, Serialize};

use crate::{
    Score, TieBreakPolicy,
    types::{EvalError, ScoreMetadata},
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// Input to the exact-match scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactMatchSample {
    /// Identifier for this sample (for reporting).
    pub id: String,
    /// Expected / ground-truth string.
    pub expected: String,
}

/// Output produced by the solver for this sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactMatchOutput {
    /// The string produced by the solver.
    pub produced: String,
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Exact-match scorer configuration.
#[derive(Debug, Clone, Default)]
pub struct ExactMatchScorer {
    /// Normalise both strings to lowercase before comparing.
    pub case_insensitive: bool,
    /// Trim leading/trailing whitespace before comparing.
    pub trim: bool,
    /// Policy when value is exactly on threshold (not directly used here
    /// since exact-match always returns 0.0 or 1.0, but stored for
    /// composability with threshold-based reporting).
    pub tie_break: TieBreakPolicy,
}

impl ExactMatchScorer {
    /// Create a new scorer with sensible defaults.
    pub fn new() -> Self {
        Self {
            case_insensitive: false,
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

    /// Score a (sample, output) pair synchronously.
    ///
    /// Returns `Score{value: 1.0}` on match, `Score{value: 0.0}` on mismatch.
    pub fn score_sync(
        &self,
        sample: &ExactMatchSample,
        output: &ExactMatchOutput,
    ) -> EvalError<Score> {
        let expected = self.normalise(&sample.expected);
        let produced = self.normalise(&output.produced);
        let matches = expected == produced;

        let reasoning = if matches {
            format!(
                "exact match: produced {:?} == expected {:?}",
                produced, expected
            )
        } else {
            format!(
                "no match: produced {:?} != expected {:?}",
                produced, expected
            )
        };

        let metadata = ScoreMetadata::default();

        Ok(Score {
            value: if matches { 1.0 } else { 0.0 },
            reasoning,
            metadata: metadata.into(),
        })
    }
}

impl crate::Scorer<ExactMatchSample, ExactMatchOutput> for ExactMatchScorer {
    async fn score(
        &self,
        sample: &ExactMatchSample,
        output: &ExactMatchOutput,
    ) -> EvalError<Score> {
        self.score_sync(sample, output)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_pass() {
        let scorer = ExactMatchScorer::new();
        let sample = ExactMatchSample {
            id: "t1".into(),
            expected: "Paris".into(),
        };
        let output = ExactMatchOutput {
            produced: "Paris".into(),
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("exact match"));
    }

    #[test]
    fn exact_match_fail() {
        let scorer = ExactMatchScorer::new();
        let sample = ExactMatchSample {
            id: "t2".into(),
            expected: "Paris".into(),
        };
        let output = ExactMatchOutput {
            produced: "London".into(),
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert_eq!(score.value, 0.0);
        assert!(score.reasoning.contains("no match"));
    }

    #[test]
    fn exact_match_trim_whitespace() {
        let scorer = ExactMatchScorer::new(); // trim=true by default
        let sample = ExactMatchSample {
            id: "t3".into(),
            expected: "Paris".into(),
        };
        let output = ExactMatchOutput {
            produced: "  Paris  ".into(),
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[test]
    fn exact_match_case_sensitive_by_default() {
        let scorer = ExactMatchScorer::new();
        let sample = ExactMatchSample {
            id: "t4".into(),
            expected: "Paris".into(),
        };
        let output = ExactMatchOutput {
            produced: "paris".into(),
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[test]
    fn exact_match_case_insensitive_pass() {
        let scorer = ExactMatchScorer {
            case_insensitive: true,
            ..ExactMatchScorer::new()
        };
        let sample = ExactMatchSample {
            id: "t5".into(),
            expected: "Paris".into(),
        };
        let output = ExactMatchOutput {
            produced: "paris".into(),
        };
        let score = scorer.score_sync(&sample, &output).unwrap();
        assert_eq!(score.value, 1.0);
    }
}

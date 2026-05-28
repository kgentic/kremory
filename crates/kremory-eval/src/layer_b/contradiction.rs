//! G-Eval contradiction detection metric.
//!
//! Measures whether kremory's contradiction resolver flags conflicting facts
//! when the user ingests contradictory information.
//!
//! # Metric: `contradiction_flagged`
//!
//! When a user ingests a fact that directly contradicts an existing fact
//! (e.g., "Alice works at Acme" then "Alice works at Globex"), kremory should
//! detect the contradiction and flag it (via `ContradictionDetected` event,
//! `invalid_at` column, or conflict metadata).
//!
//! The judge receives:
//! - `question`: did kremory detect the contradiction?
//! - `context`: description of the conflicting facts
//! - `answer`: kremory's response / resolution output
//!
//! Score: 1.0 if contradiction was flagged, 0.0 if silently accepted.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    Score, Scorer, TieBreakPolicy,
    judge::Judge,
    types::{EvalError, ScoreMetadata},
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// Input to the contradiction scorer.
///
/// Represents a pair of conflicting facts that should trigger kremory's
/// contradiction resolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionSample {
    /// Stable identifier for this sample.
    pub id: String,
    /// The original fact ingested first.
    pub original_fact: String,
    /// The conflicting fact ingested second.
    pub conflicting_fact: String,
    /// Optional namespace context for the facts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Output produced after kremory processes the contradicting ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionOutput {
    /// Whether kremory's contradiction resolver explicitly flagged the conflict.
    pub contradiction_flagged: bool,
    /// Optional raw output / explanation from kremory (for judge context).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_response: Option<String>,
}

// ---------------------------------------------------------------------------
// ContradictionFlaggedScorer
// ---------------------------------------------------------------------------

/// G-Eval scorer: did kremory flag the contradiction?
///
/// Score 1.0 = contradiction correctly identified and flagged.
/// Score 0.0 = contradiction silently accepted (regression).
///
/// The judge verifies whether the `ContradictionOutput.contradiction_flagged`
/// signal is consistent with a genuine contradiction being present.
pub struct ContradictionFlaggedScorer<J: Judge> {
    judge: Arc<J>,
    /// Tie-break policy for threshold-based reporting.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> ContradictionFlaggedScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<ContradictionSample, ContradictionOutput>
    for ContradictionFlaggedScorer<J>
{
    async fn score(
        &self,
        sample: &ContradictionSample,
        output: &ContradictionOutput,
    ) -> EvalError<Score> {
        // Fast path: if the system already flagged it, ask the judge to verify
        // the facts are genuinely contradictory (guard against false positives).
        let context = format!(
            "Original fact: {}\nConflicting fact: {}",
            sample.original_fact, sample.conflicting_fact
        );
        let question =
            "Are the ORIGINAL FACT and CONFLICTING FACT genuinely contradictory? \
             That is, can both facts be true at the same time? If they cannot both \
             be simultaneously true, then they are contradictory.";

        let system_answer = if output.contradiction_flagged {
            "yes, the contradiction was detected"
        } else {
            "no, the contradiction was not detected"
        };

        let verdict = self
            .judge
            .evaluate(question, &context, system_answer)
            .await?;

        // Judge verdict semantics:
        // - is_correct=true: judge agrees the facts ARE contradictory
        //   → if system flagged it: correct detection → score 1.0
        //   → if system missed it: false negative → score 0.0
        // - is_correct=false: judge says facts are NOT contradictory
        //   → flagging was a false positive → score 0.5 (partial — system worked but judge disagrees)
        //   → not flagging was correct → score 1.0

        let value = if verdict.is_correct {
            // Facts are genuinely contradictory
            if output.contradiction_flagged {
                1.0 // correctly detected
            } else {
                0.0 // missed contradiction
            }
        } else {
            // Facts are not genuinely contradictory
            if output.contradiction_flagged {
                0.5 // false positive (partial — resolver over-triggered)
            } else {
                1.0 // correctly not flagged
            }
        };

        let meta = ScoreMetadata {
            judge_latency_ms: Some(0.0),
            ..Default::default()
        };
        Ok(Score {
            value,
            reasoning: format!(
                "[contradiction_flagged] flagged={}, judge_says_contradictory={}: {}",
                output.contradiction_flagged, verdict.is_correct, verdict.reasoning
            ),
            metadata: meta.into(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{Scorer, judge::MockJudge};

    fn sample_pair() -> ContradictionSample {
        ContradictionSample {
            id: "contra-001".into(),
            original_fact: "Alice works at Acme Corp.".into(),
            conflicting_fact: "Alice works at Globex Inc.".into(),
            namespace: None,
        }
    }

    #[tokio::test]
    async fn correctly_flagged_contradiction() {
        // Judge says: yes, genuinely contradictory. System flagged it. Score = 1.0.
        let scorer = ContradictionFlaggedScorer::new(MockJudge::always_correct());
        let output = ContradictionOutput {
            contradiction_flagged: true,
            system_response: None,
        };
        let score = scorer.score(&sample_pair(), &output).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("[contradiction_flagged]"));
    }

    #[tokio::test]
    async fn missed_contradiction_is_zero() {
        // Judge says: yes, genuinely contradictory. System did NOT flag it. Score = 0.0.
        let scorer = ContradictionFlaggedScorer::new(MockJudge::always_correct());
        let output = ContradictionOutput {
            contradiction_flagged: false,
            system_response: None,
        };
        let score = scorer.score(&sample_pair(), &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn false_positive_is_partial() {
        // Judge says: not actually contradictory. System flagged it anyway. Score = 0.5.
        let scorer = ContradictionFlaggedScorer::new(MockJudge::always_incorrect());
        let output = ContradictionOutput {
            contradiction_flagged: true,
            system_response: None,
        };
        let score = scorer.score(&sample_pair(), &output).await.unwrap();
        assert_eq!(score.value, 0.5);
    }

    #[tokio::test]
    async fn correctly_not_flagged_non_contradiction() {
        // Judge says: not contradictory. System did not flag it. Score = 1.0.
        let scorer = ContradictionFlaggedScorer::new(MockJudge::always_incorrect());
        let output = ContradictionOutput {
            contradiction_flagged: false,
            system_response: None,
        };
        let score = scorer.score(&sample_pair(), &output).await.unwrap();
        assert_eq!(score.value, 1.0);
    }
}

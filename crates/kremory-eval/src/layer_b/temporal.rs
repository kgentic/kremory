//! G-Eval temporal correctness metric.
//!
//! Measures whether kremory returns time-correct results for queries with an
//! explicit `as_of` timestamp. When a user asks "where does Alice work as of
//! 2024-01-15?", kremory must return the fact that was valid on that date —
//! not the current state or a fact from a different window.
//!
//! # Metric: `temporal_correctness`
//!
//! Score 1.0 = answer matches ground truth for the given `as_of` point.
//! Score 0.5 = answer is partially correct (right entity, wrong time window).
//! Score 0.0 = answer is factually wrong for the given timestamp.
//!
//! The judge receives:
//! - `question`: does the answer correctly reflect the state at `as_of`?
//! - `context`: ground truth + all ingested facts with timestamps
//! - `answer`: kremory's response for the `as_of` query

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    judge::Judge,
    types::{EvalError, ScoreMetadata},
    Score, Scorer, TieBreakPolicy,
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// A single fact with temporal validity bounds, used as context in samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedFact {
    /// Human-readable description of the fact.
    pub fact: String,
    /// When this fact became valid (ISO-8601 / RFC-3339).
    pub valid_from: DateTime<Utc>,
    /// When this fact ceased to be valid; `None` = still current.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
}

/// Input to the temporal correctness scorer.
///
/// Encodes a time-sensitive query, all relevant facts with validity windows,
/// and the expected correct answer for the given `as_of` point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSample {
    /// Stable identifier for this sample.
    pub id: String,
    /// The user's query, e.g. "Where did Alice work on 2024-01-15?".
    pub query: String,
    /// The point-in-time at which the query should be evaluated.
    pub as_of: DateTime<Utc>,
    /// All ingested facts with their validity windows (full context for judge).
    pub ingested_facts: Vec<TimestampedFact>,
    /// The expected correct answer for this query at `as_of`.
    pub ground_truth_answer: String,
}

/// Output produced after kremory processes the temporal query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOutput {
    /// The answer returned by kremory for the `as_of` query.
    pub answer: String,
    /// Optionally, which temporal window kremory retrieved from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_window: Option<String>,
}

// ---------------------------------------------------------------------------
// TemporalCorrectnessScorer
// ---------------------------------------------------------------------------

/// G-Eval scorer: does kremory's answer reflect the correct temporal state?
///
/// The judge evaluates whether `output.answer` matches `sample.ground_truth_answer`
/// for the given `as_of` point in time, taking into account all ingested facts
/// and their validity windows.
///
/// Scoring:
/// - `is_correct=true, is_partial=false` → 1.0 (fully correct)
/// - `is_correct=true, is_partial=true`  → 0.5 (partially correct)
/// - `is_correct=false`                  → 0.0 (wrong temporal state)
pub struct TemporalCorrectnessScorer<J: Judge> {
    judge: Arc<J>,
    /// Tie-break policy applied when score == threshold.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> TemporalCorrectnessScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<TemporalSample, TemporalOutput>
    for TemporalCorrectnessScorer<J>
{
    async fn score(&self, sample: &TemporalSample, output: &TemporalOutput) -> EvalError<Score> {
        // Build context: ground truth + all timestamped facts.
        let facts_context = sample
            .ingested_facts
            .iter()
            .map(|f| {
                let to = f
                    .valid_to
                    .map(|t| t.to_rfc3339())
                    .unwrap_or_else(|| "present".to_string());
                format!(
                    "  - {} [valid {} → {}]",
                    f.fact,
                    f.valid_from.to_rfc3339(),
                    to
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let context = format!(
            "Query: {}\n\
             Point-in-time (as_of): {}\n\
             Ingested facts with validity windows:\n{}\n\
             Ground truth answer for as_of: {}",
            sample.query,
            sample.as_of.to_rfc3339(),
            facts_context,
            sample.ground_truth_answer,
        );

        let question = "Does the SYSTEM ANSWER correctly reflect the state of the world \
                        at the given POINT-IN-TIME (as_of)? \
                        The system answer should match the GROUND TRUTH ANSWER. \
                        Answer YES if fully correct, PARTIAL if it retrieves the right \
                        entity but wrong time window or missing precision, NO if wrong.";

        let verdict = self
            .judge
            .evaluate(question, &context, &output.answer)
            .await?;

        // Map judge verdict to score value.
        let value = verdict.to_score_value();

        let window_note = output
            .retrieval_window
            .as_deref()
            .map(|w| format!(", retrieved_window={w}"))
            .unwrap_or_default();

        let meta = ScoreMetadata {
            judge_latency_ms: Some(0.0),
            ..Default::default()
        };
        Ok(Score {
            value,
            reasoning: format!(
                "[temporal_correctness] as_of={}, correct={}, partial={}{}: {}",
                sample.as_of.to_rfc3339(),
                verdict.is_correct,
                verdict.is_partial,
                window_note,
                verdict.reasoning,
            ),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{judge::MockJudge, Scorer};

    fn as_of_date() -> DateTime<Utc> {
        "2024-01-15T00:00:00Z".parse().unwrap()
    }

    fn sample_temporal() -> TemporalSample {
        TemporalSample {
            id: "temporal-001".into(),
            query: "Where did Alice work on 2024-01-15?".into(),
            as_of: as_of_date(),
            ingested_facts: vec![
                TimestampedFact {
                    fact: "Alice works at Acme Corp.".into(),
                    valid_from: "2023-01-01T00:00:00Z".parse().unwrap(),
                    valid_to: Some("2024-06-01T00:00:00Z".parse().unwrap()),
                },
                TimestampedFact {
                    fact: "Alice works at Globex Inc.".into(),
                    valid_from: "2024-06-01T00:00:00Z".parse().unwrap(),
                    valid_to: None,
                },
            ],
            ground_truth_answer: "Alice worked at Acme Corp. on 2024-01-15.".into(),
        }
    }

    #[tokio::test]
    async fn correct_temporal_answer_scores_one() {
        // Judge says: correct, not partial. Score = 1.0.
        let scorer = TemporalCorrectnessScorer::new(MockJudge::always_correct());
        let output = TemporalOutput {
            answer: "Alice worked at Acme Corp.".into(),
            retrieval_window: None,
        };
        let score = scorer.score(&sample_temporal(), &output).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("[temporal_correctness]"));
        assert!(score.reasoning.contains("as_of="));
    }

    #[tokio::test]
    async fn wrong_temporal_answer_scores_zero() {
        // Judge says: not correct. Score = 0.0.
        let scorer = TemporalCorrectnessScorer::new(MockJudge::always_incorrect());
        let output = TemporalOutput {
            answer: "Alice works at Globex Inc.".into(), // wrong time window
            retrieval_window: None,
        };
        let score = scorer.score(&sample_temporal(), &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn partial_temporal_answer_scores_half() {
        // Judge says: correct but partial. Score = 0.5.
        let scorer = TemporalCorrectnessScorer::new(MockJudge::always_partial());
        let output = TemporalOutput {
            answer: "Alice worked at Acme Corp. (approximate date).".into(),
            retrieval_window: Some("2023-2024".into()),
        };
        let score = scorer.score(&sample_temporal(), &output).await.unwrap();
        assert_eq!(score.value, 0.5);
        assert!(score.reasoning.contains("retrieved_window=2023-2024"));
    }

    #[tokio::test]
    async fn reasoning_includes_as_of_timestamp() {
        let scorer = TemporalCorrectnessScorer::new(MockJudge::always_correct());
        let output = TemporalOutput {
            answer: "Alice worked at Acme Corp.".into(),
            retrieval_window: None,
        };
        let score = scorer.score(&sample_temporal(), &output).await.unwrap();
        assert!(score.reasoning.contains("2024-01-15"));
    }

    #[tokio::test]
    async fn non_english_name_sample_scores_correctly() {
        // Diversity check: non-English entity name (王芳).
        let sample = TemporalSample {
            id: "temporal-002".into(),
            query: "Where did 王芳 work in March 2023?".into(),
            as_of: "2023-03-15T00:00:00Z".parse().unwrap(),
            ingested_facts: vec![TimestampedFact {
                fact: "王芳 works at Beijing Tech.".into(),
                valid_from: "2022-01-01T00:00:00Z".parse().unwrap(),
                valid_to: Some("2024-01-01T00:00:00Z".parse().unwrap()),
            }],
            ground_truth_answer: "王芳 worked at Beijing Tech in March 2023.".into(),
        };
        let scorer = TemporalCorrectnessScorer::new(MockJudge::always_correct());
        let output = TemporalOutput {
            answer: "王芳 worked at Beijing Tech.".into(),
            retrieval_window: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 1.0);
    }
}

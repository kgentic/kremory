//! Model-graded fact-presence scorer.
//!
//! Checks whether a specific factual claim (`fact`) is supported by a context
//! passage, using a [`Judge`].  The judge treats `fact` as the "answer" and
//! the `context` as the evidence.  Score: 1.0 if fact is present and supported,
//! 0.0 if absent or contradicted.
//!
//! Unlike [`model_graded_qa`](super::model_graded_qa), there is no question
//! field — the implicit question is "Is this fact supported by the context?".

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    judge::Judge,
    types::{EvalError, ScoreMetadata},
    Score, TieBreakPolicy,
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// Input to the model-graded fact scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGradedFactSample {
    /// Identifier for this sample.
    pub id: String,
    /// The factual claim to verify.
    pub fact: String,
    /// Context passage that may or may not support the fact.
    pub context: String,
}

/// Output produced by the solver for this sample.
///
/// For fact-presence scoring the solver typically returns the memory system's
/// retrieved context or generated answer; the judge checks whether `fact` is
/// supported within `produced`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGradedFactOutput {
    /// Retrieved or generated content to evaluate for fact presence.
    pub produced: String,
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Model-graded fact-presence scorer.
///
/// Calls the judge with:
/// - question = `"Is the following fact present and supported by the context?"`
/// - context  = `sample.context`
/// - answer   = `output.produced`
///
/// The `sample.fact` is embedded into the question so the judge evaluates
/// presence of that specific claim.
pub struct ModelGradedFactScorer<J: Judge> {
    judge: Arc<J>,
    /// Tie-break policy for threshold-based reporting.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> ModelGradedFactScorer<J> {
    /// Create from a judge implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            tie_break: TieBreakPolicy::Pass,
        }
    }

    fn build_question(fact: &str) -> String {
        format!(
            "Is the following factual claim present and supported by the context? Fact: \"{}\"",
            fact
        )
    }
}

impl<J: Judge + Send + Sync> crate::Scorer<ModelGradedFactSample, ModelGradedFactOutput>
    for ModelGradedFactScorer<J>
{
    async fn score(
        &self,
        sample: &ModelGradedFactSample,
        output: &ModelGradedFactOutput,
    ) -> EvalError<Score> {
        let question = Self::build_question(&sample.fact);

        let t0 = Instant::now();
        let verdict = self
            .judge
            .evaluate(&question, &sample.context, &output.produced)
            .await?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Fact presence is binary: partial → 0.5, correct → 1.0, incorrect → 0.0.
        let value = verdict.to_score_value();
        let reasoning = verdict.reasoning.clone();

        let metadata = ScoreMetadata {
            judge_latency_ms: Some(latency_ms),
            ..Default::default()
        };

        Ok(Score {
            value,
            reasoning,
            metadata: metadata.into(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::judge::{JudgeVerdict, MockJudge};
    use crate::Scorer;

    #[tokio::test]
    async fn fact_present_scores_1() {
        let judge = MockJudge::always_correct();
        let scorer = ModelGradedFactScorer::new(judge);

        let sample = ModelGradedFactSample {
            id: "f1".into(),
            fact: "Paris is the capital of France".into(),
            context: "Paris is the capital of France.".into(),
        };
        let output = ModelGradedFactOutput {
            produced: "Paris is the capital of France.".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn fact_absent_scores_0() {
        let judge = MockJudge::always_incorrect();
        let scorer = ModelGradedFactScorer::new(judge);

        let sample = ModelGradedFactSample {
            id: "f2".into(),
            fact: "Berlin is the capital of France".into(),
            context: "Paris is the capital of France.".into(),
        };
        let output = ModelGradedFactOutput {
            produced: "I don't know".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn fact_partial_scores_half() {
        let judge = MockJudge::always_partial();
        let scorer = ModelGradedFactScorer::new(judge);

        let sample = ModelGradedFactSample {
            id: "f3".into(),
            fact: "The event happened in Q3 2024".into(),
            context: "The event happened in Q3 2024.".into(),
        };
        let output = ModelGradedFactOutput {
            produced: "The event happened in 2024".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 0.5);
    }

    #[tokio::test]
    async fn fact_question_includes_fact_text() {
        // Verify build_question embeds the fact string so the judge has context.
        let question = ModelGradedFactScorer::<MockJudge>::build_question("Paris is the capital");
        assert!(question.contains("Paris is the capital"));
        assert!(question.contains("factual claim"));
    }

    #[tokio::test]
    async fn fact_scorer_populates_metadata() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "fact present".into(),
        });
        let scorer = ModelGradedFactScorer::new(judge);

        let sample = ModelGradedFactSample {
            id: "f4".into(),
            fact: "X".into(),
            context: "X is present.".into(),
        };
        let output = ModelGradedFactOutput {
            produced: "X".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        let meta: ScoreMetadata = serde_json::from_value(score.metadata).unwrap();
        assert!(meta.judge_latency_ms.is_some());
    }
}

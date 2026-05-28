//! Model-graded QA scorer.
//!
//! Uses a [`Judge`] to evaluate whether a solver's answer is correct given
//! a question and a context passage.  The judge's verdict maps to a numeric
//! score: 1.0 (correct), 0.5 (partial), 0.0 (incorrect).

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    Score, TieBreakPolicy,
    judge::Judge,
    types::{EvalError, ScoreMetadata},
};

// ---------------------------------------------------------------------------
// Sample / Output types
// ---------------------------------------------------------------------------

/// Input to the model-graded QA scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGradedQaSample {
    /// Identifier for this sample.
    pub id: String,
    /// The question posed to the memory system.
    pub question: String,
    /// Context passage that constitutes the ground truth.
    pub context: String,
}

/// Output produced by the solver for this sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGradedQaOutput {
    /// The answer the memory system produced.
    pub answer: String,
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Model-graded QA scorer.
///
/// Wraps any [`Judge`] implementation. Use [`crate::judge::GemmaJudge`] for
/// production and [`crate::judge::MockJudge`] for unit tests.
pub struct ModelGradedQaScorer<J: Judge> {
    judge: Arc<J>,
    /// Tie-break policy for threshold-based reporting.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> ModelGradedQaScorer<J> {
    /// Create from a boxed judge.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> crate::Scorer<ModelGradedQaSample, ModelGradedQaOutput>
    for ModelGradedQaScorer<J>
{
    async fn score(
        &self,
        sample: &ModelGradedQaSample,
        output: &ModelGradedQaOutput,
    ) -> EvalError<Score> {
        let t0 = Instant::now();
        let verdict = self
            .judge
            .evaluate(&sample.question, &sample.context, &output.answer)
            .await?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

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
mod tests {
    use super::*;
    use crate::Scorer;
    use crate::judge::{JudgeVerdict, MockJudge};

    #[tokio::test]
    async fn model_graded_qa_correct() {
        let judge = MockJudge::always_correct();
        let scorer = ModelGradedQaScorer::new(judge);

        let sample = ModelGradedQaSample {
            id: "t1".into(),
            question: "What is the capital of France?".into(),
            context: "France is a country in Europe. Its capital is Paris.".into(),
        };
        let output = ModelGradedQaOutput {
            answer: "Paris".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("mock"));
    }

    #[tokio::test]
    async fn model_graded_qa_incorrect() {
        let judge = MockJudge::always_incorrect();
        let scorer = ModelGradedQaScorer::new(judge);

        let sample = ModelGradedQaSample {
            id: "t2".into(),
            question: "What is the capital of France?".into(),
            context: "France is a country in Europe. Its capital is Paris.".into(),
        };
        let output = ModelGradedQaOutput {
            answer: "Berlin".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn model_graded_qa_partial() {
        let judge = MockJudge::always_partial();
        let scorer = ModelGradedQaScorer::new(judge);

        let sample = ModelGradedQaSample {
            id: "t3".into(),
            question: "When exactly did the event happen?".into(),
            context: "The event happened in Q3 2024.".into(),
        };
        let output = ModelGradedQaOutput {
            answer: "2024".into(),
        };

        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 0.5);
    }

    #[tokio::test]
    async fn model_graded_qa_metadata_populated() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "test reasoning".into(),
        });
        let scorer = ModelGradedQaScorer::new(judge);

        let sample = ModelGradedQaSample {
            id: "t4".into(),
            question: "Q".into(),
            context: "C".into(),
        };
        let output = ModelGradedQaOutput { answer: "A".into() };

        let score = scorer.score(&sample, &output).await.unwrap();
        // Metadata must have judge_latency_ms populated (mock runs sync so it's ~0ms)
        let meta: ScoreMetadata = serde_json::from_value(score.metadata).unwrap();
        assert!(meta.judge_latency_ms.is_some());
    }
}

//! RAGAS-style diagnostic metrics for kremory recall quality.
//!
//! Six metrics via [`Judge`], all implementing [`Scorer<RagasFixture, RagasOutput>`]:
//!
//! | Metric                            | What it measures                                        |
//! |-----------------------------------|---------------------------------------------------------|
//! | [`FaithfulnessScorer`]            | Recall uses only facts from ingested episodes           |
//! | [`AnswerRelevancyScorer`]         | Returned context addresses the query                    |
//! | [`ContextPrecisionScorer`]        | Retrieved chunks are relevant to the query              |
//! | [`ContextRecallScorer`]           | Retrieval found all required ground-truth context       |
//! | [`ContextEntitiesRecallScorer`]   | Key entities appear in retrieved context                |
//! | [`HallucinationScorer`]           | Recall does not introduce facts absent from ingest      |
//!
//! Default pass threshold: `0.5` (DeepEval convention).

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    Score, Scorer, TieBreakPolicy,
    judge::Judge,
    types::{EvalError, EvalErr, ScoreMetadata},
};

/// Provenance tag — how the fixture was produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Provenance {
    Synthetic,
    RealUserCited,
    CorpusExtracted,
}

/// A single RAGAS evaluation fixture.
///
/// Mirrors the JSON schema defined in `fixtures/ragas/*.json`.
/// Every fixture MUST include a `provenance` field (enforced by deserialisation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagasFixture {
    /// Stable identifier for this fixture.
    pub id: String,
    /// The query posed to the memory system.
    pub query: String,
    /// Facts that were ingested into kremory before recall.
    pub ingested_facts: Vec<String>,
    /// Ground-truth answer the system should produce.
    pub expected_answer: String,
    /// Ground-truth entity names expected in retrieved context.
    pub expected_entities: Vec<String>,
    /// Ground-truth context chunks that should be retrieved.
    pub expected_contexts: Vec<String>,
    /// Diversity tags (non-english-name, temporal, contradiction, abstention, etc.).
    pub diversity_tags: Vec<String>,
    /// How this fixture was produced: synthetic / real-user-cited / corpus-extracted.
    pub provenance: Provenance,
    /// Optional human notes for fixture maintenance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// The recall output produced by kremory for a [`RagasFixture`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagasOutput {
    /// The answer kremory produced (or "I don't know" for abstention).
    pub answer: String,
    /// The context chunks kremory retrieved, in order of relevance.
    pub retrieved_contexts: Vec<String>,
    /// Entities found in the retrieved contexts.
    pub retrieved_entities: Vec<String>,
}

/// Pass threshold used by all RAGAS scorers (DeepEval convention).
pub const RAGAS_PASS_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Faithfulness Scorer
// ---------------------------------------------------------------------------

/// Measures whether the recall answer uses only facts from ingested episodes.
///
/// A high score means the answer is grounded in the ingest.
/// A low score means the answer introduces external or fabricated facts.
pub struct FaithfulnessScorer<J: Judge> {
    judge: Arc<J>,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> FaithfulnessScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            threshold: RAGAS_PASS_THRESHOLD,
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<RagasFixture, RagasOutput> for FaithfulnessScorer<J> {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        let context = sample.ingested_facts.join("\n");
        let question = format!(
            "Does the following ANSWER use only facts from the CONTEXT (ingested episodes)? \
             ANSWER must not introduce information absent from CONTEXT.\nQuery: {}",
            sample.query
        );
        let verdict = self.judge.evaluate(&question, &context, &output.answer).await?;
        let value = verdict.to_score_value();
        let meta = ScoreMetadata {
            judge_latency_ms: Some(0.0),
            ..Default::default()
        };
        Ok(Score {
            value,
            reasoning: format!("[faithfulness] {}", verdict.reasoning),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Answer Relevancy Scorer
// ---------------------------------------------------------------------------

/// Measures whether the retrieved context addresses the original query.
///
/// A high score means the answer is topically relevant to the query.
/// A low score means the context drifted from what was asked.
pub struct AnswerRelevancyScorer<J: Judge> {
    judge: Arc<J>,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> AnswerRelevancyScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            threshold: RAGAS_PASS_THRESHOLD,
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<RagasFixture, RagasOutput> for AnswerRelevancyScorer<J> {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        let context = output.retrieved_contexts.join("\n");
        let question = format!(
            "Does the ANSWER directly address the QUESTION? \
             Is the response relevant to what was asked?\nQuestion: {}",
            sample.query
        );
        let verdict = self.judge.evaluate(&question, &context, &output.answer).await?;
        let value = verdict.to_score_value();
        let meta = ScoreMetadata {
            judge_latency_ms: Some(0.0),
            ..Default::default()
        };
        Ok(Score {
            value,
            reasoning: format!("[answer_relevancy] {}", verdict.reasoning),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Context Precision Scorer
// ---------------------------------------------------------------------------

/// Measures whether retrieved chunks are relevant to the query.
///
/// Precision = (relevant retrieved chunks) / (total retrieved chunks).
/// Each retrieved chunk is evaluated by the judge independently.
pub struct ContextPrecisionScorer<J: Judge> {
    judge: Arc<J>,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> ContextPrecisionScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            threshold: RAGAS_PASS_THRESHOLD,
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<RagasFixture, RagasOutput> for ContextPrecisionScorer<J> {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        if output.retrieved_contexts.is_empty() {
            return Ok(Score::binary(
                0.0,
                "[context_precision] no retrieved contexts",
            ));
        }

        let mut relevant_count = 0usize;
        let total = output.retrieved_contexts.len();

        for chunk in &output.retrieved_contexts {
            let question = format!(
                "Is this CONTEXT chunk relevant to the QUESTION? \
                 Answer yes if the chunk contains information useful for answering the question.\nQuestion: {}",
                sample.query
            );
            let verdict = self.judge.evaluate(&question, chunk, "yes").await?;
            if verdict.is_correct {
                relevant_count += 1;
            }
        }

        let precision = relevant_count as f64 / total as f64;
        let meta = ScoreMetadata {
            precision: Some(precision),
            true_positives: Some(relevant_count),
            ..Default::default()
        };
        Ok(Score {
            value: precision,
            reasoning: format!(
                "[context_precision] {}/{} chunks relevant (precision={:.3})",
                relevant_count, total, precision
            ),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Context Recall Scorer
// ---------------------------------------------------------------------------

/// Measures whether all ground-truth context chunks were retrieved.
///
/// Recall = (expected contexts found in retrieved) / (total expected contexts).
/// Each expected context is checked against the combined retrieved contexts.
pub struct ContextRecallScorer<J: Judge> {
    judge: Arc<J>,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> ContextRecallScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            threshold: RAGAS_PASS_THRESHOLD,
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<RagasFixture, RagasOutput> for ContextRecallScorer<J> {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        if sample.expected_contexts.is_empty() {
            return Ok(Score::binary(1.0, "[context_recall] no expected contexts"));
        }

        let retrieved_combined = output.retrieved_contexts.join("\n");
        let mut found_count = 0usize;
        let total = sample.expected_contexts.len();

        for expected_ctx in &sample.expected_contexts {
            let question = format!(
                "Is the following EXPECTED context covered by the RETRIEVED context?\nExpected: {}",
                expected_ctx
            );
            let verdict = self
                .judge
                .evaluate(&question, &retrieved_combined, expected_ctx)
                .await?;
            if verdict.is_correct {
                found_count += 1;
            }
        }

        let recall = found_count as f64 / total as f64;
        let meta = ScoreMetadata {
            recall: Some(recall),
            true_positives: Some(found_count),
            false_negatives: Some(total - found_count),
            ..Default::default()
        };
        Ok(Score {
            value: recall,
            reasoning: format!(
                "[context_recall] {}/{} expected contexts found (recall={:.3})",
                found_count, total, recall
            ),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Context Entities Recall Scorer
// ---------------------------------------------------------------------------

/// Measures whether key entities expected in context were retrieved.
///
/// Deterministic set-intersection — no judge required.
/// Reuses the Layer B 6.1 F1 approach applied to the recall path.
/// Entity comparison is case-insensitive and trim-normalised.
#[derive(Default)]
pub struct ContextEntitiesRecallScorer {
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl ContextEntitiesRecallScorer {
    /// Create with default tie-break policy.
    pub fn new() -> Self {
        Self::default()
    }

    fn normalise(s: &str) -> String {
        s.trim().to_lowercase()
    }
}

impl Scorer<RagasFixture, RagasOutput> for ContextEntitiesRecallScorer {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        if sample.expected_entities.is_empty() {
            return Ok(Score::binary(
                1.0,
                "[context_entities_recall] no expected entities",
            ));
        }

        let expected: HashSet<String> = sample
            .expected_entities
            .iter()
            .map(|e| Self::normalise(e))
            .collect();

        let retrieved: HashSet<String> = output
            .retrieved_entities
            .iter()
            .map(|e| Self::normalise(e))
            .collect();

        let tp = expected.intersection(&retrieved).count();
        let fn_ = expected.difference(&retrieved).count();
        let recall = tp as f64 / expected.len() as f64;

        let meta = ScoreMetadata {
            recall: Some(recall),
            true_positives: Some(tp),
            false_negatives: Some(fn_),
            ..Default::default()
        };
        Ok(Score {
            value: recall,
            reasoning: format!(
                "[context_entities_recall] {}/{} entities recalled (recall={:.3})",
                tp,
                expected.len(),
                recall
            ),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Hallucination Scorer
// ---------------------------------------------------------------------------

/// Measures whether the recall answer introduces facts NOT present in ingested episodes.
///
/// High score = no hallucination (answer is grounded).
/// Low score = hallucination detected (answer invents facts).
pub struct HallucinationScorer<J: Judge> {
    judge: Arc<J>,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Tie-break policy.
    pub tie_break: TieBreakPolicy,
}

impl<J: Judge> HallucinationScorer<J> {
    /// Create from any [`Judge`] implementation.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
            threshold: RAGAS_PASS_THRESHOLD,
            tie_break: TieBreakPolicy::Pass,
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<RagasFixture, RagasOutput> for HallucinationScorer<J> {
    async fn score(&self, sample: &RagasFixture, output: &RagasOutput) -> EvalError<Score> {
        let context = sample.ingested_facts.join("\n");
        let question = format!(
            "Does the ANSWER contain ONLY facts present in the CONTEXT? \
             If the answer introduces any fact not found in CONTEXT, respond is_correct=false. \
             If the answer abstains (\"I don't know\") when context lacks info, that is correct.\nQuery: {}",
            sample.query
        );
        let verdict = self.judge.evaluate(&question, &context, &output.answer).await?;
        let value = verdict.to_score_value();
        let meta = ScoreMetadata {
            judge_latency_ms: Some(0.0),
            ..Default::default()
        };
        Ok(Score {
            value,
            reasoning: format!("[hallucination] {}", verdict.reasoning),
            metadata: meta.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Aggregate: RagasMetricScores + score_all_metrics
// ---------------------------------------------------------------------------

/// Aggregate RAGAS scores across all 6 metrics for a single fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagasMetricScores {
    pub faithfulness: f64,
    pub answer_relevancy: f64,
    pub context_precision: f64,
    pub context_recall: f64,
    pub context_entities_recall: f64,
    pub hallucination: f64,
}

impl RagasMetricScores {
    /// Mean of all 6 metrics.
    pub fn mean(&self) -> f64 {
        (self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
            + self.context_entities_recall
            + self.hallucination)
            / 6.0
    }

    /// Whether all metrics pass at `threshold` with `policy`.
    pub fn all_pass(&self, threshold: f64, policy: TieBreakPolicy) -> bool {
        let check = |v: f64| match policy {
            TieBreakPolicy::Pass => v >= threshold,
            TieBreakPolicy::Fail => v > threshold,
        };
        check(self.faithfulness)
            && check(self.answer_relevancy)
            && check(self.context_precision)
            && check(self.context_recall)
            && check(self.context_entities_recall)
            && check(self.hallucination)
    }

    /// Compute variance across two sets of scores (for determinism checking).
    pub fn variance_vs(&self, other: &RagasMetricScores) -> f64 {
        let diffs = [
            (self.faithfulness - other.faithfulness).powi(2),
            (self.answer_relevancy - other.answer_relevancy).powi(2),
            (self.context_precision - other.context_precision).powi(2),
            (self.context_recall - other.context_recall).powi(2),
            (self.context_entities_recall - other.context_entities_recall).powi(2),
            (self.hallucination - other.hallucination).powi(2),
        ];
        diffs.iter().sum::<f64>() / diffs.len() as f64
    }
}

/// Run all 6 RAGAS metrics on a single fixture and return aggregate scores.
///
/// `judge` is used for the 5 LLM-graded metrics.
/// [`ContextEntitiesRecallScorer`] is deterministic (no judge needed).
pub async fn score_all_metrics<J: Judge + Send + Sync + Clone>(
    judge: J,
    fixture: &RagasFixture,
    output: &RagasOutput,
) -> EvalError<RagasMetricScores> {
    let faithfulness = FaithfulnessScorer::new(judge.clone())
        .score(fixture, output)
        .await?;
    let answer_relevancy = AnswerRelevancyScorer::new(judge.clone())
        .score(fixture, output)
        .await?;
    let context_precision = ContextPrecisionScorer::new(judge.clone())
        .score(fixture, output)
        .await?;
    let context_recall = ContextRecallScorer::new(judge.clone())
        .score(fixture, output)
        .await?;
    let context_entities_recall = ContextEntitiesRecallScorer::new()
        .score(fixture, output)
        .await?;
    let hallucination = HallucinationScorer::new(judge)
        .score(fixture, output)
        .await?;

    Ok(RagasMetricScores {
        faithfulness: faithfulness.value,
        answer_relevancy: answer_relevancy.value,
        context_precision: context_precision.value,
        context_recall: context_recall.value,
        context_entities_recall: context_entities_recall.value,
        hallucination: hallucination.value,
    })
}

/// Load RAGAS fixtures from a JSON file at `path`.
///
/// File format: `{ "fixtures": [ <RagasFixture>, ... ] }`.
pub fn load_fixtures(path: &std::path::Path) -> Result<Vec<RagasFixture>, EvalErr> {
    let raw = std::fs::read_to_string(path)?;
    let wrapper: FixtureFile = serde_json::from_str(&raw)?;
    Ok(wrapper.fixtures)
}

#[derive(Deserialize)]
struct FixtureFile {
    fixtures: Vec<RagasFixture>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{Scorer, judge::MockJudge};

    fn sample_fixture() -> RagasFixture {
        RagasFixture {
            id: "test-001".into(),
            query: "What does Alice do for work?".into(),
            ingested_facts: vec![
                "Alice is a software engineer at Acme Corp.".into(),
                "Alice joined Acme in 2023.".into(),
            ],
            expected_answer: "Alice is a software engineer at Acme Corp.".into(),
            expected_entities: vec!["Alice".into(), "Acme Corp".into()],
            expected_contexts: vec!["Alice is a software engineer at Acme Corp.".into()],
            diversity_tags: vec![],
            provenance: Provenance::Synthetic,
            notes: None,
        }
    }

    fn sample_output() -> RagasOutput {
        RagasOutput {
            answer: "Alice is a software engineer at Acme Corp.".into(),
            retrieved_contexts: vec!["Alice is a software engineer at Acme Corp.".into()],
            retrieved_entities: vec!["Alice".into(), "Acme Corp".into()],
        }
    }

    #[tokio::test]
    async fn faithfulness_correct() {
        let scorer = FaithfulnessScorer::new(MockJudge::always_correct());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("[faithfulness]"));
    }

    #[tokio::test]
    async fn faithfulness_incorrect() {
        let scorer = FaithfulnessScorer::new(MockJudge::always_incorrect());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn answer_relevancy_correct() {
        let scorer = AnswerRelevancyScorer::new(MockJudge::always_correct());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("[answer_relevancy]"));
    }

    #[tokio::test]
    async fn context_precision_all_relevant() {
        let scorer = ContextPrecisionScorer::new(MockJudge::always_correct());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn context_precision_no_contexts_returns_zero() {
        let scorer = ContextPrecisionScorer::new(MockJudge::always_correct());
        let mut output = sample_output();
        output.retrieved_contexts = vec![];
        let score = scorer.score(&sample_fixture(), &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn context_recall_all_found() {
        let scorer = ContextRecallScorer::new(MockJudge::always_correct());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn context_recall_empty_expected_is_perfect() {
        let scorer = ContextRecallScorer::new(MockJudge::always_correct());
        let mut fixture = sample_fixture();
        fixture.expected_contexts = vec![];
        let score = scorer.score(&fixture, &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn context_entities_recall_perfect() {
        let scorer = ContextEntitiesRecallScorer::new();
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert!((score.value - 1.0).abs() < 1e-9);
        assert!(score.reasoning.contains("[context_entities_recall]"));
    }

    #[tokio::test]
    async fn context_entities_recall_partial() {
        let scorer = ContextEntitiesRecallScorer::new();
        let mut output = sample_output();
        output.retrieved_entities = vec!["Alice".into()];
        let score = scorer.score(&sample_fixture(), &output).await.unwrap();
        assert!((score.value - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn context_entities_recall_case_insensitive() {
        let scorer = ContextEntitiesRecallScorer::new();
        let mut output = sample_output();
        output.retrieved_entities = vec!["alice".into(), "acme corp".into()];
        let score = scorer.score(&sample_fixture(), &output).await.unwrap();
        assert!((score.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn context_entities_recall_empty_expected_is_perfect() {
        let scorer = ContextEntitiesRecallScorer::new();
        let mut fixture = sample_fixture();
        fixture.expected_entities = vec![];
        let score = scorer.score(&fixture, &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn hallucination_clean() {
        let scorer = HallucinationScorer::new(MockJudge::always_correct());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.reasoning.contains("[hallucination]"));
    }

    #[tokio::test]
    async fn hallucination_detected() {
        let scorer = HallucinationScorer::new(MockJudge::always_incorrect());
        let score = scorer.score(&sample_fixture(), &sample_output()).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn score_all_metrics_correct_judge() {
        let scores = score_all_metrics(
            MockJudge::always_correct(),
            &sample_fixture(),
            &sample_output(),
        )
        .await
        .unwrap();
        assert_eq!(scores.faithfulness, 1.0);
        assert_eq!(scores.answer_relevancy, 1.0);
        assert_eq!(scores.context_precision, 1.0);
        assert_eq!(scores.context_recall, 1.0);
        assert_eq!(scores.context_entities_recall, 1.0);
        assert_eq!(scores.hallucination, 1.0);
        assert!((scores.mean() - 1.0).abs() < 1e-9);
        assert!(scores.all_pass(RAGAS_PASS_THRESHOLD, TieBreakPolicy::Pass));
    }

    #[tokio::test]
    async fn score_all_metrics_incorrect_judge() {
        let scores = score_all_metrics(
            MockJudge::always_incorrect(),
            &sample_fixture(),
            &sample_output(),
        )
        .await
        .unwrap();
        // context_entities_recall is deterministic and passes when entities match
        assert_eq!(scores.faithfulness, 0.0);
        assert_eq!(scores.hallucination, 0.0);
        assert!(!scores.all_pass(RAGAS_PASS_THRESHOLD, TieBreakPolicy::Pass));
    }

    #[test]
    fn ragas_metric_scores_mean() {
        let scores = RagasMetricScores {
            faithfulness: 0.8,
            answer_relevancy: 0.6,
            context_precision: 1.0,
            context_recall: 0.4,
            context_entities_recall: 0.9,
            hallucination: 0.7,
        };
        let expected = (0.8 + 0.6 + 1.0 + 0.4 + 0.9 + 0.7) / 6.0;
        assert!((scores.mean() - expected).abs() < 1e-9);
    }

    #[test]
    fn ragas_metric_scores_variance_identical_is_zero() {
        let scores = RagasMetricScores {
            faithfulness: 0.8,
            answer_relevancy: 0.8,
            context_precision: 0.8,
            context_recall: 0.8,
            context_entities_recall: 0.8,
            hallucination: 0.8,
        };
        assert!((scores.variance_vs(&scores.clone()) - 0.0).abs() < 1e-9);
    }
}

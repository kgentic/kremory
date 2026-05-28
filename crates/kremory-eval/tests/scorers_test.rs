//! Unit tests for all kremory-eval scorers (Tessa M7).
#![allow(clippy::unwrap_used, clippy::expect_used)]
//!
//! Each scorer has at least one PASS case and one FAIL case.
//! Model-graded scorers use [`MockJudge`] — no real LLM required.
//!
//! Test count target: ≥ 2 tests per scorer (pass + fail).

use kremory_eval::judge::{JudgeVerdict, MockJudge};
use kremory_eval::scorers::exact_match::{ExactMatchOutput, ExactMatchSample, ExactMatchScorer};
use kremory_eval::scorers::f1::{F1Output, F1Sample, F1Scorer};
use kremory_eval::scorers::model_graded_fact::{
    ModelGradedFactOutput, ModelGradedFactSample, ModelGradedFactScorer,
};
use kremory_eval::scorers::model_graded_qa::{
    ModelGradedQaOutput, ModelGradedQaSample, ModelGradedQaScorer,
};
use kremory_eval::types::ScoreMetadata;
use kremory_eval::{Score, Scorer, TieBreakPolicy};

// ---------------------------------------------------------------------------
// Score helpers
// ---------------------------------------------------------------------------

#[test]
fn score_passes_with_pass_policy_on_boundary() {
    let s = Score::binary(0.5, "boundary");
    assert!(s.passes(0.5, TieBreakPolicy::Pass));
}

#[test]
fn score_fails_with_fail_policy_on_boundary() {
    let s = Score::binary(0.5, "boundary");
    assert!(!s.passes(0.5, TieBreakPolicy::Fail));
}

// ---------------------------------------------------------------------------
// ExactMatchScorer
// ---------------------------------------------------------------------------

#[test]
fn exact_match_pass_case() {
    let scorer = ExactMatchScorer::new();
    let sample = ExactMatchSample {
        id: "em_pass".into(),
        expected: "Paris".into(),
    };
    let output = ExactMatchOutput {
        produced: "Paris".into(),
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    assert_eq!(score.value, 1.0, "exact match should yield 1.0");
}

#[test]
fn exact_match_fail_case() {
    let scorer = ExactMatchScorer::new();
    let sample = ExactMatchSample {
        id: "em_fail".into(),
        expected: "Paris".into(),
    };
    let output = ExactMatchOutput {
        produced: "London".into(),
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    assert_eq!(score.value, 0.0, "mismatch should yield 0.0");
}

#[test]
fn exact_match_reasoning_populated() {
    let scorer = ExactMatchScorer::new();
    let sample = ExactMatchSample {
        id: "em_reason".into(),
        expected: "foo".into(),
    };
    let output = ExactMatchOutput {
        produced: "bar".into(),
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    assert!(!score.reasoning.is_empty(), "reasoning must be populated");
}

// ---------------------------------------------------------------------------
// F1Scorer
// ---------------------------------------------------------------------------

#[test]
fn f1_perfect_match_is_one() {
    let scorer = F1Scorer::new();
    let sample = F1Sample {
        id: "f1_pass".into(),
        expected: vec!["Alice".into(), "Bob".into()],
    };
    let output = F1Output {
        predicted: vec!["Alice".into(), "Bob".into()],
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    assert!((score.value - 1.0).abs() < 1e-9, "perfect match F1 = 1.0");
}

#[test]
fn f1_disjoint_sets_is_zero() {
    let scorer = F1Scorer::new();
    let sample = F1Sample {
        id: "f1_fail".into(),
        expected: vec!["Alice".into(), "Bob".into()],
    };
    let output = F1Output {
        predicted: vec!["Charlie".into(), "Dave".into()],
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    assert!((score.value - 0.0).abs() < 1e-9, "disjoint sets F1 = 0.0");
}

#[test]
fn f1_metadata_has_tp_fp_fn() {
    let scorer = F1Scorer::new();
    let sample = F1Sample {
        id: "f1_meta".into(),
        expected: vec!["A".into(), "B".into(), "C".into()],
    };
    let output = F1Output {
        predicted: vec!["A".into(), "B".into(), "X".into()],
    };
    let score = scorer.score_sync(&sample, &output).unwrap();
    let meta: ScoreMetadata = serde_json::from_value(score.metadata).unwrap();
    // TP=2 (A,B), FP=1 (X), FN=1 (C)
    assert_eq!(meta.true_positives, Some(2));
    assert_eq!(meta.false_positives, Some(1));
    assert_eq!(meta.false_negatives, Some(1));
    assert!(meta.precision.is_some());
    assert!(meta.recall.is_some());
    assert!(meta.f1.is_some());
}

// ---------------------------------------------------------------------------
// ModelGradedQaScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_graded_qa_pass_correct() {
    let scorer = ModelGradedQaScorer::new(MockJudge::always_correct());
    let sample = ModelGradedQaSample {
        id: "qa_pass".into(),
        question: "What is the capital of France?".into(),
        context: "The capital of France is Paris.".into(),
    };
    let output = ModelGradedQaOutput {
        answer: "Paris".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 1.0);
    assert!(!score.reasoning.is_empty());
}

#[tokio::test]
async fn model_graded_qa_fail_incorrect() {
    let scorer = ModelGradedQaScorer::new(MockJudge::always_incorrect());
    let sample = ModelGradedQaSample {
        id: "qa_fail".into(),
        question: "What is the capital of France?".into(),
        context: "The capital of France is Paris.".into(),
    };
    let output = ModelGradedQaOutput {
        answer: "Berlin".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 0.0);
}

#[tokio::test]
async fn model_graded_qa_partial_is_half() {
    let scorer = ModelGradedQaScorer::new(MockJudge::always_partial());
    let sample = ModelGradedQaSample {
        id: "qa_partial".into(),
        question: "When did it happen?".into(),
        context: "It happened in Q3 2024.".into(),
    };
    let output = ModelGradedQaOutput {
        answer: "2024".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 0.5);
}

#[tokio::test]
async fn model_graded_qa_custom_verdict() {
    let verdict = JudgeVerdict {
        is_correct: true,
        is_partial: false,
        reasoning: "custom reasoning text".into(),
    };
    let scorer = ModelGradedQaScorer::new(MockJudge::new(verdict));
    let sample = ModelGradedQaSample {
        id: "qa_custom".into(),
        question: "Q".into(),
        context: "C".into(),
    };
    let output = ModelGradedQaOutput { answer: "A".into() };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 1.0);
    assert_eq!(score.reasoning, "custom reasoning text");
}

// ---------------------------------------------------------------------------
// ModelGradedFactScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_graded_fact_pass_present() {
    let scorer = ModelGradedFactScorer::new(MockJudge::always_correct());
    let sample = ModelGradedFactSample {
        id: "fact_pass".into(),
        fact: "The sky is blue".into(),
        context: "The sky is blue on clear days.".into(),
    };
    let output = ModelGradedFactOutput {
        produced: "The sky is blue.".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 1.0);
    assert!(!score.reasoning.is_empty());
}

#[tokio::test]
async fn model_graded_fact_fail_absent() {
    let scorer = ModelGradedFactScorer::new(MockJudge::always_incorrect());
    let sample = ModelGradedFactSample {
        id: "fact_fail".into(),
        fact: "The sky is green".into(),
        context: "The sky is blue on clear days.".into(),
    };
    let output = ModelGradedFactOutput {
        produced: "I don't know.".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 0.0);
}

#[tokio::test]
async fn model_graded_fact_partial_half_score() {
    let scorer = ModelGradedFactScorer::new(MockJudge::always_partial());
    let sample = ModelGradedFactSample {
        id: "fact_partial".into(),
        fact: "The event happened in Q3 2024".into(),
        context: "The event happened in Q3 2024.".into(),
    };
    let output = ModelGradedFactOutput {
        produced: "The event happened in 2024.".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    assert_eq!(score.value, 0.5);
}

#[tokio::test]
async fn model_graded_fact_metadata_latency_present() {
    let scorer = ModelGradedFactScorer::new(MockJudge::always_correct());
    let sample = ModelGradedFactSample {
        id: "fact_meta".into(),
        fact: "X".into(),
        context: "X is here.".into(),
    };
    let output = ModelGradedFactOutput {
        produced: "X".into(),
    };
    let score = scorer.score(&sample, &output).await.unwrap();
    let meta: ScoreMetadata = serde_json::from_value(score.metadata).unwrap();
    assert!(
        meta.judge_latency_ms.is_some(),
        "judge_latency_ms should be populated"
    );
}

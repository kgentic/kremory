//! LongMemEval Layer A benchmark harness.
//!
//! Implements the `xiaowu0162/longmemeval-cleaned` dataset loader and a Rust
//! port of the canonical `evaluate_qa.py` scorer.
//!
//! # Upstream reference
//!
//! Scorer logic ported from:
//!   `xiaowu0162/LongMemEval@9e0b455` `src/evaluation/evaluate_qa.py`
//!   URL: https://github.com/xiaowu0162/LongMemEval/blob/9e0b455/src/evaluation/evaluate_qa.py
//!
//! Prompt templates are copied **verbatim** from that file to preserve score
//! parity. Any whitespace change = score divergence vs published benchmark.
//!
//! # Judge disclosure
//!
//! v0.1.4 scores use a local Gemma 4 E2B Q4_K_M judge. The canonical scorer
//! (`evaluate_qa.py`) calls GPT-4o (`gpt-4o-2024-08-06`). Scores are
//! comparable-class but not bit-for-bit identical. See
//! `crates/kremory-eval/baselines/v0.1.4-scorer-decision.md` for full disclosure.
//!
//! # O4 reconciliation
//!
//! The LongMemEval paper describes 5 "abilities"; the dataset has 6
//! `question_type` values (`single-session-user`, `single-session-assistant`,
//! `single-session-preference`, `multi-session`, `temporal-reasoning`,
//! `knowledge-update`). "Information Extraction" maps to the three
//! single-session subtypes. Abstention is a cross-cutting view identified by
//! `_abs` suffix on `question_id`, not a 7th category.
//!
//! # HF dataset
//!
//! Three variants available on HF Hub (`xiaowu0162/longmemeval-cleaned`):
//! - `longmemeval_oracle.json` — only evidence sessions (oracle retrieval)
//! - `longmemeval_s_cleaned.json` — ~40 sessions per instance
//! - `longmemeval_m_cleaned.json` — ~500 sessions per instance
//!
//! Default: `longmemeval_oracle.json` (smallest, fastest, tests recall quality
//! independently of retrieval). Use `LongMemEvalConfig::variant` to override.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    judge::Judge,
    types::{EvalErr, EvalError},
    Dataset, Score, Scorer,
};

// ---------------------------------------------------------------------------
// Dataset variant
// ---------------------------------------------------------------------------

/// Which LongMemEval variant to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LongMemEvalVariant {
    /// Oracle retrieval — only evidence sessions present (default).
    #[default]
    Oracle,
    /// Small haystack — ~40 sessions per instance.
    Small,
    /// Medium haystack — ~500 sessions per instance.
    Medium,
}

impl LongMemEvalVariant {
    /// Filename on HuggingFace Hub for this variant.
    pub fn filename(self) -> &'static str {
        match self {
            Self::Oracle => "longmemeval_oracle.json",
            Self::Small => "longmemeval_s_cleaned.json",
            Self::Medium => "longmemeval_m_cleaned.json",
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset JSON schema types
// ---------------------------------------------------------------------------

/// A single turn in a haystack conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// "user" or "assistant"
    pub role: String,
    /// Turn text content.
    pub content: String,
    /// Present on evidence turns (marks the session that contains the answer).
    #[serde(default)]
    pub has_answer: bool,
}

/// One haystack session: a list of conversation turns.
pub type HaystackSession = Vec<ConversationTurn>;

/// Accept heterogeneous JSON types for the `answer` field. Upstream LongMemEval
/// stores text answers as String and counting-question answers (e.g. multi-session
/// "How many ..." questions) as Integer. Both collapse to String here so the
/// downstream judge prompt template ("EXPECTED ANSWER: {answer}") sees a
/// uniform format.
fn deserialize_answer_as_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        other => Err(serde::de::Error::custom(format!(
            "answer must be string, number, or bool; got {other:?}"
        ))),
    }
}

/// One LongMemEval evaluation instance (as stored in the JSON files).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalRecord {
    /// Unique question identifier. Ends in `_abs` for abstention questions.
    pub question_id: String,
    /// One of the 6 question type values (see module-level docs for O4 note).
    pub question_type: String,
    /// The question text posed to the memory system.
    pub question: String,
    /// Ground truth answer (or abstention explanation for `_abs` questions).
    /// Heterogeneous in upstream dataset: text for most, integer for counting
    /// questions ("How many ..."). Normalised to String on deserialise.
    #[serde(deserialize_with = "deserialize_answer_as_string")]
    pub answer: String,
    /// Date the question was asked (YYYY/MM/DD format).
    #[serde(default)]
    pub question_date: String,
    /// Session IDs in the haystack (parallel list to `haystack_sessions`).
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    /// Timestamps for each haystack session (parallel list).
    #[serde(default)]
    pub haystack_dates: Vec<String>,
    /// The actual conversation sessions forming the memory haystack.
    #[serde(default)]
    pub haystack_sessions: Vec<HaystackSession>,
    /// Which session IDs contain the answer evidence.
    #[serde(default)]
    pub answer_session_ids: Vec<String>,
}

impl LongMemEvalRecord {
    /// True if this is an abstention question (no answer in haystack).
    ///
    /// LongMemEval upstream uses Python `'_abs' in question_id` (substring).
    /// We use `ends_with("_abs")` because LongMemEval's published dataset uses `_abs`
    /// exclusively as a trailing suffix on question_id (e.g., `m_q001_abs`). The
    /// stricter suffix check avoids spurious matches on hypothetical IDs like
    /// `abs_q001` or `q_abs_evidence`. Verified against upstream commit
    /// `9e0b455f4ef0e2ab8f2e582289761153549043fc`.
    pub fn is_abstention(&self) -> bool {
        self.question_id.ends_with("_abs")
    }
}

// ---------------------------------------------------------------------------
// Dataset struct + config
// ---------------------------------------------------------------------------

/// Configuration for loading the LongMemEval dataset.
#[derive(Debug, Clone)]
pub struct LongMemEvalConfig {
    /// Which dataset variant to load (default: Oracle).
    pub variant: LongMemEvalVariant,
    /// Optional: limit to first N samples (for fast iteration/CI smoke).
    pub sample_limit: Option<usize>,
    /// HF Hub dataset ID (default: `xiaowu0162/longmemeval-cleaned`).
    pub dataset_id: String,
}

impl Default for LongMemEvalConfig {
    fn default() -> Self {
        Self {
            variant: LongMemEvalVariant::Oracle,
            sample_limit: None,
            dataset_id: "xiaowu0162/longmemeval-cleaned".into(),
        }
    }
}

/// Loaded LongMemEval dataset, ready for iteration.
pub struct LongMemEvalDataset {
    records: Vec<LongMemEvalRecord>,
}

impl LongMemEvalDataset {
    /// Load dataset from a local JSON file path (skips HF download).
    pub fn from_file(path: &std::path::Path, limit: Option<usize>) -> EvalError<Self> {
        let file = std::fs::File::open(path).map_err(EvalErr::Io)?;
        let reader = std::io::BufReader::new(file);
        let all: Vec<LongMemEvalRecord> = serde_json::from_reader(reader).map_err(EvalErr::Json)?;
        let records = match limit {
            Some(n) => all.into_iter().take(n).collect(),
            None => all,
        };
        Ok(Self { records })
    }

    /// Download from HuggingFace Hub (uses local cache; no re-download if
    /// already cached at `~/.cache/huggingface/hub/`).
    pub async fn from_hub(config: &LongMemEvalConfig) -> EvalError<Self> {
        use hf_hub::api::tokio::Api;

        let api = Api::new().map_err(|e| EvalErr::Other(format!("hf-hub init error: {}", e)))?;
        let repo = api.dataset(config.dataset_id.clone());
        let filename = config.variant.filename();

        eprintln!(
            "[longmemeval] Fetching {} from HuggingFace Hub...",
            filename
        );
        eprintln!("[longmemeval] Dataset: {}", config.dataset_id);
        eprintln!("[longmemeval] This will use ~/.cache/huggingface/hub/ for caching.");

        let local_path: std::path::PathBuf = repo
            .get(filename)
            .await
            .map_err(|e| EvalErr::Other(format!("hf-hub download error: {}", e)))?;

        eprintln!("[longmemeval] Cached at: {}", local_path.display());
        Self::from_file(&local_path, config.sample_limit)
    }

    /// Number of records in this dataset.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if the dataset has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Dataset for LongMemEvalDataset {
    type Sample = LongMemEvalSample;

    fn samples(&self) -> impl Iterator<Item = Self::Sample> {
        self.records.iter().cloned().map(LongMemEvalSample::from)
    }
}

// ---------------------------------------------------------------------------
// Sample and Output types for Scorer
// ---------------------------------------------------------------------------

/// A LongMemEval sample — the unit passed to Solver + Scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalSample {
    /// Unique question identifier.
    pub question_id: String,
    /// One of 6 question_type values (see module docs).
    pub question_type: String,
    /// The question text.
    pub question: String,
    /// Ground truth answer (or abstention explanation).
    pub answer: String,
    /// Whether this is an abstention question (derived from `_abs` suffix).
    pub is_abstention: bool,
    /// Haystack sessions to ingest into memory before querying.
    pub haystack_sessions: Vec<HaystackSession>,
    /// Timestamps for each session (parallel to `haystack_sessions`).
    pub haystack_dates: Vec<String>,
    /// Session IDs (parallel to `haystack_sessions`).
    pub haystack_session_ids: Vec<String>,
}

impl From<LongMemEvalRecord> for LongMemEvalSample {
    fn from(r: LongMemEvalRecord) -> Self {
        let is_abstention = r.is_abstention();
        Self {
            question_id: r.question_id,
            question_type: r.question_type,
            question: r.question,
            answer: r.answer,
            is_abstention,
            haystack_sessions: r.haystack_sessions,
            haystack_dates: r.haystack_dates,
            haystack_session_ids: r.haystack_session_ids,
        }
    }
}

/// Solver output for one LongMemEval sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalOutput {
    /// The memory system's response to the question.
    pub response: String,
    /// Number of input tokens used during recall (for token-efficiency metric).
    pub input_tokens_used: Option<u64>,
}

// ---------------------------------------------------------------------------
// Scorer — Rust port of evaluate_qa.py
//
// Upstream: xiaowu0162/LongMemEval@9e0b455 src/evaluation/evaluate_qa.py
// Prompt templates copied verbatim from get_anscheck_prompt().
// Verdict extraction: "yes" in response.lower() (Python line ~85).
// ---------------------------------------------------------------------------

/// Ability category labels (paper's 5 abilities mapped from 6 question_types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbilityCategory {
    InformationExtraction,
    MultiSession,
    KnowledgeUpdate,
    TemporalReasoning,
    Abstention,
}

impl AbilityCategory {
    /// Map `question_type` + `is_abstention` flag to ability category.
    pub fn from_sample(question_type: &str, is_abstention: bool) -> Self {
        if is_abstention {
            return Self::Abstention;
        }
        match question_type {
            "single-session-user" | "single-session-assistant" | "single-session-preference" => {
                Self::InformationExtraction
            }
            "multi-session" => Self::MultiSession,
            "knowledge-update" => Self::KnowledgeUpdate,
            "temporal-reasoning" => Self::TemporalReasoning,
            _ => Self::InformationExtraction,
        }
    }
}

// ---------------------------------------------------------------------------
// Verbatim prompt templates from evaluate_qa.py @ 9e0b455
// ---------------------------------------------------------------------------

/// Template for single-session-user, single-session-assistant, multi-session.
/// Verbatim from evaluate_qa.py get_anscheck_prompt() lines 25-28.
const PROMPT_STANDARD: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.";

/// Template for temporal-reasoning tasks.
/// Verbatim from evaluate_qa.py get_anscheck_prompt() lines 30-33.
const PROMPT_TEMPORAL: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.";

/// Template for knowledge-update tasks.
/// Verbatim from evaluate_qa.py get_anscheck_prompt() lines 35-38.
const PROMPT_KNOWLEDGE_UPDATE: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.";

/// Template for single-session-preference tasks.
/// Verbatim from evaluate_qa.py get_anscheck_prompt() lines 40-43.
const PROMPT_PREFERENCE: &str = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.";

/// Template for abstention questions.
/// Verbatim from evaluate_qa.py get_anscheck_prompt() lines 46-49.
const PROMPT_ABSTENTION: &str = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.";

/// Build the judge prompt for a given task/question/answer/response combination.
///
/// Mirrors `get_anscheck_prompt()` in evaluate_qa.py @ 9e0b455.
fn build_anscheck_prompt(
    question_type: &str,
    question: &str,
    answer: &str,
    response: &str,
    is_abstention: bool,
) -> String {
    let template = if is_abstention {
        PROMPT_ABSTENTION
    } else {
        match question_type {
            "single-session-user" | "single-session-assistant" | "multi-session" => PROMPT_STANDARD,
            "temporal-reasoning" => PROMPT_TEMPORAL,
            "knowledge-update" => PROMPT_KNOWLEDGE_UPDATE,
            "single-session-preference" => PROMPT_PREFERENCE,
            // Fallback to standard for unknown types (forward compat)
            _ => PROMPT_STANDARD,
        }
    };

    template
        .replace("{question}", question)
        .replace("{answer}", answer)
        .replace("{response}", response)
}

// ---------------------------------------------------------------------------
// LongMemEvalScorer
// ---------------------------------------------------------------------------

/// LongMemEval scorer — Rust port of evaluate_qa.py.
///
/// Uses the [`Judge`] trait so tests inject [`MockJudge`] and production
/// uses [`GemmaJudge`].
///
/// Judge is called with:
/// - `question`: the built prompt (full evaluate_qa.py style prompt)
/// - `context`: empty string (LongMemEval uses a single-turn yes/no prompt)
/// - `answer`: empty string
///
/// Then parses `"yes" in verdict.reasoning.to_lowercase()` for binary label,
/// matching the Python `label = 'yes' in eval_response.lower()` logic.
pub struct LongMemEvalScorer<J: Judge> {
    judge: Arc<J>,
}

impl<J: Judge> LongMemEvalScorer<J> {
    /// Construct from any judge.
    pub fn new(judge: J) -> Self {
        Self {
            judge: Arc::new(judge),
        }
    }
}

impl<J: Judge + Send + Sync> Scorer<LongMemEvalSample, LongMemEvalOutput> for LongMemEvalScorer<J> {
    async fn score(
        &self,
        sample: &LongMemEvalSample,
        output: &LongMemEvalOutput,
    ) -> EvalError<Score> {
        let prompt = build_anscheck_prompt(
            &sample.question_type,
            &sample.question,
            &sample.answer,
            &output.response,
            sample.is_abstention,
        );

        // Pass the full evaluate_qa.py-style prompt as the "question" field.
        // Context and answer are empty — the judge sees the complete prompt
        // and its yes/no is extracted from verdict.reasoning.
        let verdict = self.judge.evaluate(&prompt, "", "").await?;

        // Binary label: prefer the structured `is_correct` bool from JudgeVerdict.
        //
        // Upstream evaluate_qa.py uses `'yes' in eval_response.lower()` because
        // its judge returns free-form text. Our judge returns structured JSON
        // (`is_correct: bool`) — using the bool directly is more reliable than
        // substring-matching reasoning text, which fails when the judge model
        // produces positive reasoning without an explicit "yes" prefix
        // (observed with Gemma 4 E2B-IT).
        //
        // Backward compat with MockJudge: all existing tests set
        // `is_correct` to match their "yes"/"no" reasoning strings.
        let judge_response_raw = verdict.reasoning.clone();
        let label = verdict.is_correct;
        let score_value = if label { 1.0_f64 } else { 0.0_f64 };

        let metadata = serde_json::json!({
            "question_type": sample.question_type,
            "question_id": sample.question_id,
            "is_abstention": sample.is_abstention,
            "ability_category": format!("{:?}", AbilityCategory::from_sample(&sample.question_type, sample.is_abstention)),
            "judge_response_raw": judge_response_raw,
            "label": label,
        });

        Ok(Score {
            value: score_value,
            reasoning: format!(
                "LongMemEval judge ({} / abstention={}) → {}",
                sample.question_type,
                sample.is_abstention,
                if label { "CORRECT" } else { "INCORRECT" }
            ),
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Aggregate report
// ---------------------------------------------------------------------------

/// Aggregate evaluation result for one LongMemEval run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalReport {
    /// Mean accuracy across all questions.
    pub overall_accuracy: f64,
    /// Per question_type accuracy (6 values).
    pub per_type_accuracy: HashMap<String, f64>,
    /// Accuracy for abstention questions only.
    pub abstention_accuracy: f64,
    /// Mean input tokens per recall query (O6 token-efficiency formula).
    pub mean_input_tokens_per_recall: Option<f64>,
    /// Number of samples scored.
    pub sample_count: usize,
}

impl LongMemEvalReport {
    /// Build a report from per-sample scores.
    pub fn from_scores(scores: &[(LongMemEvalSample, Score)]) -> Self {
        let mut all_labels: Vec<f64> = Vec::with_capacity(scores.len());
        let mut type_labels: HashMap<String, Vec<f64>> = HashMap::new();
        let mut abstention_labels: Vec<f64> = Vec::new();
        let mut total_tokens: u64 = 0;
        let mut token_query_count: u64 = 0;

        for (sample, score) in scores {
            let label = score.value;
            all_labels.push(label);
            type_labels
                .entry(sample.question_type.clone())
                .or_default()
                .push(label);
            if sample.is_abstention {
                abstention_labels.push(label);
            }
            // Token efficiency: extract from metadata if present
            if let Some(tokens) = score
                .metadata
                .get("input_tokens_used")
                .and_then(|v| v.as_u64())
            {
                total_tokens += tokens;
                token_query_count += 1;
            }
        }

        let overall_accuracy = if all_labels.is_empty() {
            0.0
        } else {
            all_labels.iter().sum::<f64>() / all_labels.len() as f64
        };

        let per_type_accuracy = type_labels
            .into_iter()
            .map(|(k, v)| {
                let acc = if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                };
                (k, acc)
            })
            .collect();

        let abstention_accuracy = if abstention_labels.is_empty() {
            0.0
        } else {
            abstention_labels.iter().sum::<f64>() / abstention_labels.len() as f64
        };

        let mean_input_tokens_per_recall = if token_query_count > 0 {
            Some(total_tokens as f64 / token_query_count as f64)
        } else {
            None
        };

        Self {
            overall_accuracy,
            per_type_accuracy,
            abstention_accuracy,
            mean_input_tokens_per_recall,
            sample_count: scores.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{
        judge::{JudgeVerdict, MockJudge},
        Scorer,
    };

    fn make_sample(question_type: &str, is_abstention: bool) -> LongMemEvalSample {
        let suffix = if is_abstention { "_abs" } else { "" };
        LongMemEvalSample {
            question_id: format!("q001{}", suffix),
            question_type: question_type.into(),
            question: "What did the user say about their coffee preference?".into(),
            answer: "The user prefers oat milk lattes.".into(),
            is_abstention,
            haystack_sessions: vec![],
            haystack_dates: vec![],
            haystack_session_ids: vec![],
        }
    }

    #[tokio::test]
    async fn scorer_returns_1_when_judge_says_correct() {
        // MockJudge::always_correct() returns reasoning="mock: always correct"
        // which contains "correct" but NOT "yes" — so we need a mock that returns "yes"
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "yes, the response contains the correct answer".into(),
        });
        let scorer = LongMemEvalScorer::new(judge);
        let sample = make_sample("single-session-user", false);
        let output = LongMemEvalOutput {
            response: "The user prefers oat milk lattes.".into(),
            input_tokens_used: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn scorer_returns_0_when_judge_says_no() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: "no, the response is incorrect".into(),
        });
        let scorer = LongMemEvalScorer::new(judge);
        let sample = make_sample("single-session-user", false);
        let output = LongMemEvalOutput {
            response: "The user prefers black coffee.".into(),
            input_tokens_used: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    /// Regression: scorer must use `verdict.is_correct` bool, not substring-match
    /// "yes" in reasoning. Observed with Gemma 4 E2B-IT: positive reasoning
    /// ("matching the correct answer") that lacked an explicit "yes" prefix
    /// produced false 0.0 scores before the fix landed in this release.
    #[tokio::test]
    async fn scorer_uses_is_correct_bool_when_reasoning_lacks_yes_prefix() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "the model response matches the correct answer".into(),
        });
        let scorer = LongMemEvalScorer::new(judge);
        let sample = make_sample("single-session-user", false);
        let output = LongMemEvalOutput {
            response: "The user prefers oat milk lattes.".into(),
            input_tokens_used: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(
            score.value, 1.0,
            "scorer must respect verdict.is_correct=true even when reasoning lacks the substring 'yes'"
        );
    }

    /// Regression: mirror of the above for false verdicts. Reasoning that
    /// happens to contain "yes" (e.g. "no, the answer says yes but is wrong")
    /// must NOT score 1.0 just because of the substring.
    #[tokio::test]
    async fn scorer_uses_is_correct_bool_when_reasoning_contains_yes_but_verdict_false() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: "the response says yes but contradicts the ground truth".into(),
        });
        let scorer = LongMemEvalScorer::new(judge);
        let sample = make_sample("single-session-user", false);
        let output = LongMemEvalOutput {
            response: "yes black coffee".into(),
            input_tokens_used: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(
            score.value, 0.0,
            "scorer must respect verdict.is_correct=false even when reasoning contains 'yes'"
        );
    }

    #[tokio::test]
    async fn scorer_abstention_uses_abstention_prompt() {
        let judge = MockJudge::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "yes the model correctly identifies".into(),
        });
        let scorer = LongMemEvalScorer::new(judge);
        let sample = make_sample("multi-session", true);
        let output = LongMemEvalOutput {
            response: "I don't know, the information is not available.".into(),
            input_tokens_used: None,
        };
        let score = scorer.score(&sample, &output).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert!(score.metadata["is_abstention"].as_bool().unwrap());
    }

    #[test]
    fn prompt_contains_question_and_answer() {
        let prompt = build_anscheck_prompt(
            "single-session-user",
            "What is X?",
            "X is Y",
            "Response text",
            false,
        );
        assert!(prompt.contains("What is X?"));
        assert!(prompt.contains("X is Y"));
        assert!(prompt.contains("Response text"));
        assert!(prompt.contains("Answer yes or no only."));
    }

    #[test]
    fn abstention_prompt_uses_different_template() {
        let prompt = build_anscheck_prompt(
            "multi-session",
            "Who is Z?",
            "Z is not mentioned in any session",
            "I don't know",
            true,
        );
        assert!(prompt.contains("unanswerable"));
        assert!(prompt.contains("Does the model correctly identify"));
    }

    #[test]
    fn temporal_prompt_mentions_off_by_one() {
        let prompt = build_anscheck_prompt(
            "temporal-reasoning",
            "How many days?",
            "18 days",
            "19 days",
            false,
        );
        assert!(prompt.contains("off-by-one"));
    }

    #[test]
    fn knowledge_update_prompt_mentions_updated_answer() {
        let prompt = build_anscheck_prompt(
            "knowledge-update",
            "What is user's job?",
            "Software Engineer",
            "Recently became a Software Engineer",
            false,
        );
        assert!(prompt.contains("updated answer"));
    }

    #[test]
    fn preference_prompt_uses_rubric() {
        let prompt = build_anscheck_prompt(
            "single-session-preference",
            "What coffee?",
            "Oat milk, no sugar",
            "Sure, oat milk latte",
            false,
        );
        assert!(prompt.contains("Rubric:"));
        assert!(prompt.contains("personalized"));
    }

    #[test]
    fn ability_category_maps_correctly() {
        assert_eq!(
            AbilityCategory::from_sample("single-session-user", false),
            AbilityCategory::InformationExtraction
        );
        assert_eq!(
            AbilityCategory::from_sample("single-session-assistant", false),
            AbilityCategory::InformationExtraction
        );
        assert_eq!(
            AbilityCategory::from_sample("single-session-preference", false),
            AbilityCategory::InformationExtraction
        );
        assert_eq!(
            AbilityCategory::from_sample("multi-session", false),
            AbilityCategory::MultiSession
        );
        assert_eq!(
            AbilityCategory::from_sample("temporal-reasoning", false),
            AbilityCategory::TemporalReasoning
        );
        assert_eq!(
            AbilityCategory::from_sample("knowledge-update", false),
            AbilityCategory::KnowledgeUpdate
        );
        assert_eq!(
            AbilityCategory::from_sample("multi-session", true),
            AbilityCategory::Abstention
        );
    }

    #[test]
    fn report_aggregates_correctly() {
        let sample_a = make_sample("single-session-user", false);
        let sample_b = make_sample("temporal-reasoning", false);
        let sample_c = make_sample("multi-session", true);

        let scores = vec![
            (
                sample_a,
                Score {
                    value: 1.0,
                    reasoning: "yes".into(),
                    metadata: serde_json::Value::Null,
                },
            ),
            (
                sample_b,
                Score {
                    value: 0.0,
                    reasoning: "no".into(),
                    metadata: serde_json::Value::Null,
                },
            ),
            (
                sample_c,
                Score {
                    value: 1.0,
                    reasoning: "yes".into(),
                    metadata: serde_json::Value::Null,
                },
            ),
        ];

        let report = LongMemEvalReport::from_scores(&scores);
        assert!((report.overall_accuracy - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(report.per_type_accuracy["single-session-user"], 1.0);
        assert_eq!(report.per_type_accuracy["temporal-reasoning"], 0.0);
        assert_eq!(report.abstention_accuracy, 1.0);
        assert_eq!(report.sample_count, 3);
    }

    #[test]
    fn record_is_abstention_checks_suffix() {
        let r = LongMemEvalRecord {
            question_id: "q001_abs".into(),
            question_type: "multi-session".into(),
            question: "q".into(),
            answer: "a".into(),
            question_date: String::new(),
            haystack_session_ids: vec![],
            haystack_dates: vec![],
            haystack_sessions: vec![],
            answer_session_ids: vec![],
        };
        assert!(r.is_abstention());

        let r2 = LongMemEvalRecord {
            question_id: "q002".into(),
            ..r.clone()
        };
        assert!(!r2.is_abstention());
    }

    #[test]
    fn dataset_from_json_bytes() {
        // Minimal in-memory dataset round-trip (no file I/O, no HF download)
        let records = vec![LongMemEvalRecord {
            question_id: "q001".into(),
            question_type: "single-session-user".into(),
            question: "What?".into(),
            answer: "This.".into(),
            question_date: "2024/01/01".into(),
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec!["2024/01/01".into()],
            haystack_sessions: vec![vec![ConversationTurn {
                role: "user".into(),
                content: "This is the content.".into(),
                has_answer: true,
            }]],
            answer_session_ids: vec!["s1".into()],
        }];
        let json = serde_json::to_string(&records).unwrap();
        let tmpfile = std::env::temp_dir().join("longmemeval_test.json");
        std::fs::write(&tmpfile, json).unwrap();
        let ds = LongMemEvalDataset::from_file(&tmpfile, None).unwrap();
        assert_eq!(ds.len(), 1);
        let samples: Vec<_> = ds.samples().collect();
        assert_eq!(samples[0].question_id, "q001");
        assert!(!samples[0].is_abstention);
        std::fs::remove_file(&tmpfile).unwrap();
    }
}

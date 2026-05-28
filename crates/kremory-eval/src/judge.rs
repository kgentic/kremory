//! Gemma 4 E2B judge wrapper via AutoAgents `LlamaCppProvider`.
//!
//! Uses the proven config from P1-T2 fix (commit ba5fadd):
//! - `reasoning_format(Auto)` — routes thinking tokens to separate field
//! - `extra_body(chat_template_kwargs.enable_thinking=true)` — Gemma 4 CoT flag
//! - `max_tokens(1024)` — enough room for CoT before JSON
//! - `temperature(0.0)`, `seed(42)` — deterministic output
//! - Dual content+thinking extraction: try `content` first, fall back to `thinking`
//!
//! # Mock support
//!
//! The [`Judge`] trait allows unit tests to inject a [`MockJudge`] without
//! spinning up the real Gemma model.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::types::{EvalErr, EvalError};

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Structured JSON verdict produced by the judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    /// Whether the answer is factually correct.
    pub is_correct: bool,
    /// Whether the answer is correct but less specific than the context allows.
    pub is_partial: bool,
    /// One-sentence explanation.
    pub reasoning: String,
}

impl JudgeVerdict {
    /// Map verdict to a `[0.0, 1.0]` score.
    ///
    /// - `is_correct && !is_partial` → 1.0
    /// - `is_correct && is_partial`  → 0.5
    /// - `!is_correct`               → 0.0
    pub fn to_score_value(&self) -> f64 {
        if self.is_correct {
            if self.is_partial {
                0.5
            } else {
                1.0
            }
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Judge trait
// ---------------------------------------------------------------------------

/// A judge that evaluates (question, context, answer) triples.
///
/// The trait exists so unit tests can inject a [`MockJudge`] without requiring
/// a real model on disk.
pub trait Judge: Send + Sync {
    /// Evaluate one triple and return a structured verdict.
    fn evaluate(
        &self,
        question: &str,
        context: &str,
        answer: &str,
    ) -> impl std::future::Future<Output = EvalError<JudgeVerdict>> + Send;
}

// ---------------------------------------------------------------------------
// GemmaJudge — real impl backed by LlamaCppProvider
// ---------------------------------------------------------------------------

/// System prompt injected before every judge call.
const JUDGE_PROMPT_TEMPLATE: &str = r#"You are an expert evaluator for agent memory systems. Your task is to judge whether the model's ANSWER correctly addresses the QUESTION based ONLY on the provided CONTEXT.

<question>
{question}
</question>

<context>
{context}
</context>

<answer>
{answer}
</answer>

Evaluate the answer using these criteria:
- is_correct=true: the answer is factually correct and supported by the context
- is_correct=false: the answer is incorrect, contradicts the context, or introduces facts not in the context
- is_partial=true: the answer is correct but less specific than the context allows (e.g., "2024" when context says "Q3 2024")

For abstention cases: if the context does NOT contain information to answer the question, then "I don't know" is a CORRECT answer (is_correct=true).

IMPORTANT: After your reasoning, output the JSON object ON ITS OWN LINE.
The JSON must be valid and contain exactly the keys: is_correct, is_partial, reasoning.
Do not wrap it in markdown code fences. Do not add commentary after the JSON.

Respond ONLY with a JSON object matching this schema:
{
  "is_correct": <bool>,
  "is_partial": <bool>,
  "reasoning": "<one short sentence explaining your verdict>"
}"#;

/// Live judge backed by a local Gemma 4 E2B GGUF via `LlamaCppProvider`.
#[derive(Clone)]
pub struct GemmaJudge {
    model_path: PathBuf,
}

impl GemmaJudge {
    /// Construct from an explicit model path.
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }

    /// Construct from the `KREMORY_EVAL_JUDGE_MODEL_PATH` env var, falling
    /// back to `~/.cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf`.
    pub fn from_env() -> Self {
        let model_path = std::env::var("KREMORY_EVAL_JUDGE_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                PathBuf::from(home).join(".cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf")
            });
        Self { model_path }
    }

    fn build_prompt(question: &str, context: &str, answer: &str) -> String {
        JUDGE_PROMPT_TEMPLATE
            .replace("{question}", question)
            .replace("{context}", context)
            .replace("{answer}", answer)
    }

    fn parse_verdict(raw: &str) -> Result<JudgeVerdict, EvalErr> {
        let start = raw
            .find('{')
            .ok_or_else(|| EvalErr::JudgeParse("no JSON object in output".into()))?;
        let end = raw
            .rfind('}')
            .ok_or_else(|| EvalErr::JudgeParse("no closing brace".into()))?;
        if end <= start {
            return Err(EvalErr::JudgeParse("malformed JSON range".into()));
        }
        let slice = &raw[start..=end];
        serde_json::from_str(slice)
            .map_err(|e| EvalErr::JudgeParse(format!("JSON parse error: {} (slice: {})", e, slice)))
    }
}

impl Judge for GemmaJudge {
    async fn evaluate(
        &self,
        question: &str,
        context: &str,
        answer: &str,
    ) -> EvalError<JudgeVerdict> {
        use autoagents_llamacpp::{
            LlamaCppConfigBuilder, LlamaCppProvider, LlamaCppReasoningFormat,
        };
        use autoagents_llm::chat::ChatRole;
        use autoagents_llm::chat::{ChatMessage, ChatProvider, MessageType};

        let model_path_str = self
            .model_path
            .to_str()
            .ok_or_else(|| EvalErr::Other("model path is not valid UTF-8".into()))?;

        let config = LlamaCppConfigBuilder::new()
            .model_path(model_path_str)
            .max_tokens(1024)
            .temperature(0.0)
            .seed(42)
            .reasoning_format(LlamaCppReasoningFormat::Auto)
            .extra_body(serde_json::json!({
                "chat_template_kwargs": {
                    "enable_thinking": true
                }
            }))
            .build();

        let provider = LlamaCppProvider::from_config(config)
            .await
            .map_err(|e| EvalErr::JudgeInference(format!("failed to load model: {}", e)))?;

        let prompt = Self::build_prompt(question, context, answer);
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: prompt,
        }];

        let response = provider
            .chat_with_tools(&messages, None, None)
            .await
            .map_err(|e| EvalErr::JudgeInference(format!("inference error: {}", e)))?;

        // Try content first; fall back to thinking (Gemma 4 CoT split).
        let raw_content = response.text().unwrap_or_default();
        let raw_thinking = response.thinking().unwrap_or_default();

        if raw_content.contains('{') {
            Self::parse_verdict(&raw_content)
        } else if raw_thinking.contains('{') {
            Self::parse_verdict(&raw_thinking)
        } else {
            Err(EvalErr::JudgeParse(
                "no JSON object found in content or thinking output".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// MockJudge — for unit tests
// ---------------------------------------------------------------------------

/// A deterministic judge for unit tests.
///
/// Returns a fixed [`JudgeVerdict`] regardless of input.  Tests can call
/// [`MockJudge::new`] with the verdict they want to observe.
#[derive(Clone)]
pub struct MockJudge {
    verdict: JudgeVerdict,
}

impl MockJudge {
    /// Construct a mock that always returns `verdict`.
    pub fn new(verdict: JudgeVerdict) -> Self {
        Self { verdict }
    }

    /// Convenience: always returns `is_correct=true, is_partial=false`.
    pub fn always_correct() -> Self {
        Self::new(JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: "mock: always correct".into(),
        })
    }

    /// Convenience: always returns `is_correct=false, is_partial=false`.
    pub fn always_incorrect() -> Self {
        Self::new(JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: "mock: always incorrect".into(),
        })
    }

    /// Convenience: always returns `is_correct=true, is_partial=true`.
    pub fn always_partial() -> Self {
        Self::new(JudgeVerdict {
            is_correct: true,
            is_partial: true,
            reasoning: "mock: always partial".into(),
        })
    }
}

impl Judge for MockJudge {
    async fn evaluate(
        &self,
        _question: &str,
        _context: &str,
        _answer: &str,
    ) -> EvalError<JudgeVerdict> {
        Ok(self.verdict.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_verdict_extracts_json() {
        let raw = r#"Some preamble text.
{"is_correct": true, "is_partial": false, "reasoning": "Correct answer."}"#;
        let v = GemmaJudge::parse_verdict(raw).unwrap();
        assert!(v.is_correct);
        assert!(!v.is_partial);
        assert_eq!(v.reasoning, "Correct answer.");
    }

    #[test]
    fn parse_verdict_returns_error_on_no_json() {
        let raw = "No JSON here at all.";
        let result = GemmaJudge::parse_verdict(raw);
        assert!(result.is_err());
    }

    /// Compile-only test: GemmaJudge must implement Clone (FIX MNT-005).
    /// `score_all_metrics<J: Clone>` fails to compile with GemmaJudge without this.
    /// We verify the bound holds by calling `clone()` on a GemmaJudge value.
    #[test]
    fn gemma_judge_implements_clone() {
        fn assert_clone<T: Clone>(_: &T) {}
        let judge = GemmaJudge::new(PathBuf::from("/tmp/model.gguf"));
        assert_clone(&judge);
        let _cloned = judge.clone();
    }

    #[test]
    fn verdict_to_score_value() {
        let correct = JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: String::new(),
        };
        assert_eq!(correct.to_score_value(), 1.0);

        let partial = JudgeVerdict {
            is_correct: true,
            is_partial: true,
            reasoning: String::new(),
        };
        assert_eq!(partial.to_score_value(), 0.5);

        let incorrect = JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: String::new(),
        };
        assert_eq!(incorrect.to_score_value(), 0.0);
    }
}

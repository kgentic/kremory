//! `MockChatProviderTracking` — mock ChatProvider variants for v0.1.2 observability tests.
// Test helpers: unused struct/enum variants are expected until consumed by integration tests.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]
//!
//! Three behaviors:
//!   - `WithUsage { input, output }` — returns a chat response with usage metadata.
//!   - `AlwaysFail { err }` — always returns an `LLMError`.
//!   - `RespondThenFail { n, input, output }` — returns Ok for the first `n` calls,
//!     then always returns `LLMError::ProviderError`.
//!
//! Used in `tests/llm_integration.rs` under `#[cfg(feature = "llm-integration")]`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use autoagents_llm::chat::Usage;
use autoagents_llm::error::LLMError;
use autoagents_llm::ToolCall;
use kremory::core::provider::{ChatMessage, ChatProvider, ChatResponse};

// ── Response ──────────────────────────────────────────────────────────────────

/// A chat response that carries explicit token usage metadata.
pub struct MockChatResponseWithUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl std::fmt::Debug for MockChatResponseWithUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MockChatResponseWithUsage {{ input: {}, output: {} }}",
            self.input_tokens, self.output_tokens
        )
    }
}

impl std::fmt::Display for MockChatResponseWithUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mock response with usage")
    }
}

impl ChatResponse for MockChatResponseWithUsage {
    fn text(&self) -> Option<String> {
        // Return a minimal valid extraction JSON so the ingest pipeline can parse it.
        Some(
            serde_json::json!({
                "entities": [],
                "relationships": []
            })
            .to_string(),
        )
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }

    fn usage(&self) -> Option<Usage> {
        Some(Usage {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.input_tokens + self.output_tokens,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

// ── Behavior ──────────────────────────────────────────────────────────────────

/// Behavior configuration for `MockChatProviderTracking`.
pub enum MockBehavior {
    /// Always return Ok with the specified token counts.
    WithUsage { input: u32, output: u32 },
    /// Always return the specified error.
    AlwaysFail { err: LLMError },
    /// Return Ok for the first `n` calls, then fail with `ProviderError`.
    RespondThenFail { n: usize, input: u32, output: u32 },
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// Mock ChatProvider with call-count tracking and configurable behavior.
pub struct MockChatProviderTracking {
    behavior: MockBehavior,
    pub call_count: Arc<AtomicUsize>,
}

impl MockChatProviderTracking {
    pub fn new(behavior: MockBehavior) -> Self {
        Self {
            behavior,
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ChatProvider for MockChatProviderTracking {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[autoagents_llm::chat::Tool]>,
        _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);

        match &self.behavior {
            MockBehavior::WithUsage { input, output } => Ok(Box::new(MockChatResponseWithUsage {
                input_tokens: *input,
                output_tokens: *output,
            })),
            MockBehavior::AlwaysFail { err } => Err(err.clone()),
            MockBehavior::RespondThenFail { n, input, output } => {
                if count < *n {
                    Ok(Box::new(MockChatResponseWithUsage {
                        input_tokens: *input,
                        output_tokens: *output,
                    }))
                } else {
                    Err(LLMError::ProviderError(
                        "RespondThenFail threshold reached".into(),
                    ))
                }
            }
        }
    }
}

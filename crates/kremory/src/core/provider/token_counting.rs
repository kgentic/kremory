// ---------------------------------------------------------------------------
// TokenCountingChatProvider — trait-seam token-accounting decorator
// for budget tracking.
//
// COMPILE-SPIKE: this decorator exists because
// `StructuredCallBuilder::call()` (`extraction/structured.rs`) discards
// `ChatResponse.usage()`, so per-call token counts are unreachable without an
// architectural change. The breaking Option A (change the builder's return type)
// is REJECTED. RESOLUTION (this module): wrap the dream-pass provider in a
// decorator that intercepts `chat_with_tools()`, reads `usage()` off the
// response, and accumulates totals into a shared `Arc<Mutex<TokenAccumulator>>`
// the budget-INSERT site reads after the pass.
//
// ## Usage field names — verified against the pinned AutoAgents source
//
// The impl-spec §4 named the fields `usage().tokens_input` / `tokens_output`.
// `autoagents_llm::chat::Usage` (pinned branch checkout `0b9fa9b`) actually
// exposes `prompt_tokens: u32` + `completion_tokens: u32` (+ `total_tokens`).
// This decorator accumulates the REAL fields. Totals are surfaced via the
// crate's existing [`super::TokenUsage`] type, which already uses the matching
// `prompt_tokens` / `completion_tokens` names.
//
// ## Thread-safety
//
// The dream-pass provider Arc is cloned into the Phase-2 background worker
// thread, so the accumulator lives behind a single `Arc<Mutex<_>>` (same shape
// as `RecordReplayChatProvider`'s cassette state). `ChatProvider: Send + Sync`
// is preserved — proved by the `decorator_is_send_sync_as_dyn` test.
// ---------------------------------------------------------------------------

use std::sync::{Arc, Mutex};

use autoagents_llm::chat::{ChatMessage, ChatProvider, StructuredOutputFormat};
use autoagents_llm::error::LLMError;

use super::TokenUsage;

/// Running token totals accumulated across every `chat_with_tools` call routed
/// through a [`TokenCountingChatProvider`].
///
/// `u64` (not the `u32` of `Usage`) so a long-running pass with many calls
/// cannot overflow. `calls_with_usage` counts only responses that actually
/// reported `usage()`; `calls_total` counts every call. A gap between the two
/// means the backend (e.g. a local Ollama build) did not report token usage —
/// surfaced so the budget row is not silently under-counted.
#[derive(Debug, Default, Clone)]
pub struct TokenAccumulator {
    /// Summed `Usage::prompt_tokens` across all calls that reported usage.
    pub prompt_tokens: u64,
    /// Summed `Usage::completion_tokens` across all calls that reported usage.
    pub completion_tokens: u64,
    /// Number of `chat_with_tools` calls routed through the decorator.
    pub calls_total: u64,
    /// Number of those calls whose response reported `usage()` (`Some`).
    pub calls_with_usage: u64,
}

impl TokenAccumulator {
    /// Total tokens (prompt + completion) seen so far.
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    /// Snapshot as the crate's `TokenUsage` (saturating into its `u32` fields).
    ///
    /// Saturation is the safe choice for a budget row: an over-budget run that
    /// genuinely exceeded `u32::MAX` tokens already tripped every cost guard;
    /// clamping the recorded value cannot make that worse.
    pub fn as_token_usage(&self) -> TokenUsage {
        TokenUsage {
            prompt_tokens: u32::try_from(self.prompt_tokens).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(self.completion_tokens).unwrap_or(u32::MAX),
        }
    }
}

/// Trait-seam decorator over an `Arc<dyn ChatProvider>` that accumulates token
/// usage per call into a shared [`TokenAccumulator`].
///
/// Construct with [`TokenCountingChatProvider::new`]; read totals back via
/// [`TokenCountingChatProvider::accumulator`] (a cloned `Arc` handle the budget
/// INSERT site keeps) or [`TokenCountingChatProvider::totals`].
pub struct TokenCountingChatProvider {
    inner: Arc<dyn ChatProvider>,
    accumulator: Arc<Mutex<TokenAccumulator>>,
}

impl std::fmt::Debug for TokenCountingChatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCountingChatProvider")
            .field("totals", &self.totals())
            .finish()
    }
}

impl TokenCountingChatProvider {
    /// Wrap a real provider with a fresh, zeroed accumulator.
    pub fn new(inner: Arc<dyn ChatProvider>) -> Self {
        Self {
            inner,
            accumulator: Arc::new(Mutex::new(TokenAccumulator::default())),
        }
    }

    /// A cloned handle to the shared accumulator. The budget-INSERT site holds
    /// this and reads totals AFTER the pass completes (the decorator Arc itself
    /// may have been moved into the worker thread).
    pub fn accumulator(&self) -> Arc<Mutex<TokenAccumulator>> {
        Arc::clone(&self.accumulator)
    }

    /// Snapshot of the current totals. Poison-safe (recovers the inner value).
    pub fn totals(&self) -> TokenAccumulator {
        self.accumulator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait::async_trait]
impl ChatProvider for TokenCountingChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[autoagents_llm::chat::Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn autoagents_llm::chat::ChatResponse>, LLMError> {
        let response = self
            .inner
            .chat_with_tools(messages, tools, json_schema)
            .await?;
        {
            let mut acc = self
                .accumulator
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            acc.calls_total += 1;
            if let Some(usage) = response.usage() {
                acc.calls_with_usage += 1;
                acc.prompt_tokens += u64::from(usage.prompt_tokens);
                acc.completion_tokens += u64::from(usage.completion_tokens);
            }
        }
        Ok(response)
    }

    // The `ChatProvider::model()` override is removed.
    // Token-counting carries no model knowledge — the model string flows from the
    // builder via `Engine.model` / `ExtractionContext.model`, not via wrapper-chain
    // delegation.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::provider::chat_msg_user;
    use autoagents_llm::chat::{ChatResponse, Usage};

    /// Stub provider returning a fixed response whose `usage()` is configurable
    /// (`None` to simulate a backend that does not report tokens).
    #[derive(Debug)]
    struct StubProvider {
        usage: Option<Usage>,
    }

    #[derive(Debug)]
    struct StubResponse {
        usage: Option<Usage>,
    }

    impl std::fmt::Display for StubResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "stub")
        }
    }

    impl ChatResponse for StubResponse {
        fn text(&self) -> Option<String> {
            Some("stub".to_string())
        }
        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
        fn usage(&self) -> Option<Usage> {
            self.usage.clone()
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for StubProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(StubResponse {
                usage: self.usage.clone(),
            }))
        }
    }

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        }
    }

    #[tokio::test]
    async fn accumulates_usage_across_calls() {
        let stub = Arc::new(StubProvider {
            usage: Some(usage(10, 5)),
        });
        let decorator = TokenCountingChatProvider::new(stub);
        let msgs = vec![chat_msg_user("hi")];
        for _ in 0..3 {
            decorator
                .chat_with_tools(&msgs, None, None)
                .await
                .expect("chat should succeed");
        }
        let totals = decorator.totals();
        assert_eq!(totals.prompt_tokens, 30);
        assert_eq!(totals.completion_tokens, 15);
        assert_eq!(totals.total_tokens(), 45);
        assert_eq!(totals.calls_total, 3);
        assert_eq!(totals.calls_with_usage, 3);
    }

    #[tokio::test]
    async fn counts_calls_without_usage_separately() {
        // A backend that reports no usage still increments calls_total, NOT
        // calls_with_usage — so under-counting is observable, not silent.
        let stub = Arc::new(StubProvider { usage: None });
        let decorator = TokenCountingChatProvider::new(stub);
        let msgs = vec![chat_msg_user("hi")];
        decorator
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("chat should succeed");
        let totals = decorator.totals();
        assert_eq!(totals.prompt_tokens, 0);
        assert_eq!(totals.calls_total, 1);
        assert_eq!(totals.calls_with_usage, 0);
    }

    #[tokio::test]
    async fn accumulator_handle_observes_writes() {
        // The budget-INSERT site holds a cloned handle and reads totals later.
        let stub = Arc::new(StubProvider {
            usage: Some(usage(7, 3)),
        });
        let decorator = TokenCountingChatProvider::new(stub);
        let handle = decorator.accumulator();
        let msgs = vec![chat_msg_user("hi")];
        decorator
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("chat should succeed");
        let snapshot = handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(snapshot.total_tokens(), 10);
        assert_eq!(snapshot.as_token_usage().total(), 10);
    }

    /// Compile-level proof the decorator binds in the `Arc<dyn ChatProvider>`
    /// position (Send + Sync) — the dream-pass worker thread requires it.
    #[tokio::test]
    async fn decorator_is_send_sync_as_dyn() {
        let stub = Arc::new(StubProvider { usage: None });
        let provider: Arc<dyn ChatProvider> = Arc::new(TokenCountingChatProvider::new(stub));
        fn assert_send_sync<T: Send + Sync>(_t: &T) {}
        assert_send_sync(&provider);
    }
}

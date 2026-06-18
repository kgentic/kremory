// ---------------------------------------------------------------------------
// Chat provider implementations.
//
// - `NullChatProvider` / `NullChatResponse` — production no-op for the BYOE
//   no-LLM path.
// - `MockChatProvider` / `MockChatResponse` / `mock_match_response` — test /
//   test-utils gated stub.
// - `ArcChatProvider` — newtype that lets `Arc<dyn ChatProvider>` satisfy the
//   `ChatProvider` trait bound in generic contexts (orphan rule workaround).
//
// Re-exported from `super` (`core::provider`) — callers see the unchanged
// path `crate::core::provider::NullChatProvider` etc.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use super::{ChatMessage, ChatProvider};

#[cfg(any(test, feature = "test-utils"))]
use super::ChatRole;

// ---------------------------------------------------------------------------
// NullChatProvider — production-accessible no-op stub.
//
// Used by `Memory::llm_or_stub()` on the NoLlm BYOE path where `submit_episode`
// requires an `Arc<dyn ChatProvider>` argument but the engine's `_provider`
// parameter is vestigial (the real LLM slot is `Engine.llm: Option<Arc<L>>`).
// Never actually called — if an LLM pipeline step is reached without a wired LLM,
// the engine returns `Error::LlmRequired` from its own `Option::None` check.
// ---------------------------------------------------------------------------

/// No-op `ChatProvider` for the BYOE no-LLM path.
///
/// Always returns an empty string. Only reachable when `Memory` was constructed
/// without `.with_llm(…)` AND the engine reaches a step that calls the provider
/// arg — which does not happen in normal operation because `Engine.llm` guards
/// those steps first.
#[derive(Debug, Clone)]
pub(crate) struct NullChatProvider;

#[async_trait::async_trait]
impl ChatProvider for NullChatProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[autoagents_llm::chat::Tool]>,
        _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<
        Box<dyn autoagents_llm::chat::ChatResponse>,
        autoagents_llm::error::LLMError,
    > {
        // Return an empty-text response. The only path that reaches here is a
        // coding error (calling an LLM-dependent method on a NoLlm Memory that
        // somehow bypassed the LlmRequired guard). Empty response is a safe no-op.
        Ok(Box::new(crate::core::provider::NullChatResponse))
    }
}

/// Response type for `NullChatProvider`.
#[derive(Debug)]
pub(crate) struct NullChatResponse;

impl autoagents_llm::chat::ChatResponse for NullChatResponse {
    fn text(&self) -> Option<String> {
        Some(String::new())
    }
    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
    fn usage(&self) -> Option<autoagents_llm::chat::Usage> {
        None
    }
}

impl std::fmt::Display for NullChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

// MockChatProvider — replaces MockLlmClient + NullLlmClient in tests.
//
// Substring-match: looks for any key in `responses` that appears in the last
// user-role message.  Falls back to "" (empty string) so extractors see an
// empty JSON response and return zero entities / facts — safe no-op.
//
// Gated: test-infra only — not part of the production public API.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-utils"))]
use std::collections::HashMap;

#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Clone)]
pub struct MockChatProvider {
    pub responses: HashMap<String, String>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MockChatProvider {
    /// Create a mock that returns `""` for every prompt (safe no-op).
    pub fn null() -> Self {
        Self {
            responses: HashMap::new(),
        }
    }

    /// Create a mock with explicit substring → response mapping.
    pub fn new(responses: HashMap<String, String>) -> Self {
        Self { responses }
    }

    /// Create a mock with a single key → response pair.
    ///
    /// Convenience constructor for tests that only need one scripted response:
    /// any message containing `key` returns `value`.
    pub fn with_response(key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut responses = HashMap::new();
        responses.insert(key.into(), value.into());
        Self { responses }
    }
}

/// Internal helper: find a matching response for the last user message.
#[cfg(any(test, feature = "test-utils"))]
fn mock_match_response(responses: &HashMap<String, String>, messages: &[ChatMessage]) -> String {
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == ChatRole::User)
        .map(|m| m.content.as_str())
        .unwrap_or("");

    responses
        .iter()
        .find(|(key, _)| last_user.contains(key.as_str()))
        .map(|(_, val)| val.clone())
        .unwrap_or_default()
}

#[cfg(any(test, feature = "test-utils"))]
#[async_trait::async_trait]
impl ChatProvider for MockChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[autoagents_llm::chat::Tool]>,
        _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<
        Box<dyn autoagents_llm::chat::ChatResponse>,
        autoagents_llm::error::LLMError,
    > {
        let text = mock_match_response(&self.responses, messages);
        Ok(Box::new(MockChatResponse { text }))
    }
}

#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug)]
pub struct MockChatResponse {
    pub text: String,
}

#[cfg(any(test, feature = "test-utils"))]
impl std::fmt::Display for MockChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl autoagents_llm::chat::ChatResponse for MockChatResponse {
    fn text(&self) -> Option<String> {
        if self.text.is_empty() {
            None
        } else {
            Some(self.text.clone())
        }
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
}

// ---------------------------------------------------------------------------
// ArcChatProvider — newtype that lets `Arc<dyn ChatProvider>` satisfy the
// `ChatProvider` trait bound required by generic code in the facade layer.
//
// Orphan rule prevents: `impl ChatProvider for Arc<dyn ChatProvider>` (both
// `ChatProvider` and `Arc` are defined outside this crate). The newtype pattern
// is the standard Rust solution.
//
// Used by `EngineGraphHandle` (Phase E.2) which holds `Arc<dyn ChatProvider>`
// and needs to pass it into functions/structs requiring `T: ChatProvider`.
// ---------------------------------------------------------------------------

/// Newtype wrapper that lets an `Arc<dyn ChatProvider>` satisfy the
/// `ChatProvider` trait bound in generic contexts.
///
/// Required because orphan rules forbid `impl ChatProvider for Arc<dyn ChatProvider>`.
/// Construct via `ArcChatProvider::new(arc)` or `From<Arc<dyn ChatProvider>>`.
pub struct ArcChatProvider(pub Arc<dyn ChatProvider + Send + Sync>);

impl ArcChatProvider {
    /// Wrap an `Arc<dyn ChatProvider + Send + Sync>`.
    pub fn new(inner: Arc<dyn ChatProvider + Send + Sync>) -> Self {
        Self(inner)
    }
}

#[async_trait::async_trait]
impl ChatProvider for ArcChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[autoagents_llm::chat::Tool]>,
        json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<
        Box<dyn autoagents_llm::chat::ChatResponse>,
        autoagents_llm::error::LLMError,
    > {
        self.0.chat_with_tools(messages, tools, json_schema).await
    }

    /// TD-013 F1: delegate model() to inner provider so the model string
    /// reaches StructuredCallBuilder.capability_of() at the production
    /// wrapper-chain boundary. Without this delegation, capability_of("")
    /// returns PromptOnly and FormatSchema arm never fires for Ollama models.
    /// See ADR adr-td-013-graph-quality-remediation-2026-06-03.
    fn model(&self) -> &str {
        self.0.model()
    }
}

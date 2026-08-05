#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
//! `ScriptedLlmClient` — shared scripted ChatProvider for integration tests.
//!
//! Promoted from `background_integration.rs` (v0.2.4 ADR-050 Phase 5 DoD).
//! Shared so `adr050_phase5_sink_wiring.rs` and `background_integration.rs`
//! both use the same test double without code duplication.

use std::sync::{Arc, Mutex};

use kremory::core::provider::{
    ChatMessage, ChatProvider, ChatResponse, EmbeddingProvider, LLMError, MockChatResponse,
    MockEmbeddingProvider, StructuredOutputFormat, Tool,
};

// ---------------------------------------------------------------------------
// ScriptedLlmClient
// ---------------------------------------------------------------------------

/// Returns scripted responses in FIFO order.
///
/// When the queue is exhausted, falls back to returning `"[]"` — a safe no-op
/// for both extraction and contradiction prompts.
///
/// Legacy name preserved for compatibility; implements `ChatProvider` per AA
/// adoption (2026-04-12 commit 5e8bddd).
#[derive(Debug, Clone)]
pub struct ScriptedLlmClient {
    queue: Arc<Mutex<Vec<String>>>,
}

impl ScriptedLlmClient {
    pub fn new(responses: Vec<&str>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(
                responses.into_iter().map(str::to_owned).collect(),
            )),
        }
    }

    /// Returns a snapshot of how many scripted responses remain unconsumed.
    ///
    /// Tests use this instead of accessing the private `queue` field directly.
    /// Cross-binary asymmetry: `#[allow(dead_code)]` because only one test binary
    /// calls this.
    #[allow(dead_code)]
    pub fn remaining_count(&self) -> usize {
        self.queue
            .lock()
            .expect("ScriptedLlmClient queue poisoned")
            .len()
    }
}

#[async_trait::async_trait]
impl ChatProvider for ScriptedLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        let text = {
            let mut guard = self.queue.lock().expect("ScriptedLlmClient queue poisoned");
            if guard.is_empty() {
                "[]".to_owned()
            } else {
                guard.remove(0)
            }
        };
        Ok(Box::new(MockChatResponse { text }))
    }
}

// ---------------------------------------------------------------------------
// ScriptedEmbeddingProvider
// ---------------------------------------------------------------------------

/// Thin wrapper delegating to `MockEmbeddingProvider` for deterministic
/// embeddings without requiring the `embeddings` feature.
#[derive(Debug, Clone)]
pub struct ScriptedEmbeddingProvider(MockEmbeddingProvider);

impl ScriptedEmbeddingProvider {
    pub fn new(dim: usize) -> Self {
        Self(MockEmbeddingProvider::new(dim))
    }
}

impl EmbeddingProvider for ScriptedEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::core::error::Result<Vec<f32>>> + Send + 'a {
        self.0.embed(text)
    }
}

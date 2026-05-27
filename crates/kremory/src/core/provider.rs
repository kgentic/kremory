// ---------------------------------------------------------------------------
// AutoAgents re-exports — canonical LLM abstraction for kremory::core.
//
// The BYOM seam is `Arc<dyn ChatProvider>`.  All modules in this crate use
// ChatProvider + Vec<ChatMessage> exclusively; the old LlmClient / LlmRequest
// types are gone.
//
// BYOM invariant (ADR-Phase-D.0, 2026-05-18 §7): kremory exposes ONLY the
// `autoagents-llm` TRAIT surface here. The concrete `LlamaCppProvider`
// (`autoagents-llamacpp` crate) lives in `rust-pipeline/src/rql_llamacpp.rs`
// — the host application binary, not kremory. DoD: `cargo tree -p kremory --edges normal
// | grep autoagents-llamacpp` MUST print 0.
//
// autoagents-llm is a required dep (not optional) — kremory::memory's
// GraphHandle trait always needs ChatProvider.
// ---------------------------------------------------------------------------

pub use autoagents_llm::chat::ChatResponse;
pub use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat, Tool,
};
pub use autoagents_llm::error::LLMError;

// ---------------------------------------------------------------------------
// TokenUsage — kept because IngestionResult.token_usage carries it across
// the entire pipeline.  Not part of the LLM client abstraction per se.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

impl std::ops::Add for TokenUsage {
    type Output = TokenUsage;

    fn add(self, rhs: TokenUsage) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens + rhs.prompt_tokens,
            completion_tokens: self.completion_tokens + rhs.completion_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for ChatMessage
//
// AA's ChatMessage has a builder (ChatMessage::user().content("...").build())
// but no static factory methods for the System role.  These free functions
// give call sites a uniform one-liner API for both roles.
// ---------------------------------------------------------------------------

/// Build a System-role ChatMessage.
pub fn chat_msg_system(content: impl Into<String>) -> ChatMessage {
    ChatMessage {
        role: ChatRole::System,
        content: content.into(),
        message_type: MessageType::Text,
    }
}

/// Build a User-role ChatMessage.
pub fn chat_msg_user(content: impl Into<String>) -> ChatMessage {
    ChatMessage {
        role: ChatRole::User,
        content: content.into(),
        message_type: MessageType::Text,
    }
}

// ---------------------------------------------------------------------------
// build_llm — MOVED to `rust-pipeline/src/rql_llamacpp.rs` per ADR-Phase-D.0
// §7 BYOM invariant. rqlc must not reference `autoagents-llamacpp` in
// production deps. Consumers (rust-pipeline, tauri-app) import via:
//
//   use rust_pipeline::rql_llamacpp::{build_llm, LlamaCppProvider};
//
// In-crate rqlc tests that need a real LlamaCppProvider construct it
// directly via the `autoagents-llamacpp` dev-dependency.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
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
// EmbeddingProvider + null / mock / ONNX impls
// (unchanged from original — embeddings are a separate concern)
// ---------------------------------------------------------------------------

use crate::core::error::Result;
use std::future::Future;
use std::sync::Arc;

pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a;

    /// Token count from the provider's last embed call, if the backend reports usage.
    ///
    /// Per ADR D10: `TokenTrackingEmbedder` calls this after `embed()` to get
    /// server-reported token counts. When the backend doesn't report usage (most
    /// local providers), this returns `None` and the wrapper falls back to a
    /// pre-call approximation.
    ///
    /// Default: `None` (backend does not report token usage).
    fn last_usage_tokens(&self) -> Option<u64> {
        None
    }

    /// Return this provider wrapped in an `Arc<dyn DynEmbeddingProvider>`.
    ///
    /// Provided for facade consumers who need dynamic dispatch. The default
    /// impl wraps `self` in `ArcEmbedder` automatically.
    fn into_dyn(self) -> Arc<dyn DynEmbeddingProvider>
    where
        Self: Sized + 'static,
    {
        Arc::new(ArcEmbedder(Arc::new(self)))
    }
}

/// Dyn-compatible embedding provider. Use `Arc<dyn DynEmbeddingProvider>` when
/// you need type-erased embedding via trait objects (e.g. in the `Memory` facade).
///
/// `EmbeddingProvider` uses `impl Future` in its return type which is not
/// dyn-compatible. `DynEmbeddingProvider` boxes the future, enabling `dyn`.
pub trait DynEmbeddingProvider: Send + Sync {
    /// Embed `text` into a vector, returning a boxed future.
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send + 'a>>;
    /// Token count from the provider's last embed call.
    fn last_usage_tokens_dyn(&self) -> Option<u64>;
}

/// Blanket impl: any `EmbeddingProvider` is also a `DynEmbeddingProvider`.
impl<E: EmbeddingProvider> DynEmbeddingProvider for E {
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(self.embed(text))
    }
    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        self.last_usage_tokens()
    }
}

/// Wrapper that implements `EmbeddingProvider` by delegating to
/// `Arc<dyn DynEmbeddingProvider>`. Used by the facade to turn a
/// `Arc<dyn DynEmbeddingProvider>` back into something generic code can use.
pub struct ArcEmbedder(pub Arc<dyn DynEmbeddingProvider>);

impl EmbeddingProvider for ArcEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        self.0.embed_dyn(text)
    }
    fn last_usage_tokens(&self) -> Option<u64> {
        self.0.last_usage_tokens_dyn()
    }
}

// ---------------------------------------------------------------------------
// NullEmbeddingProvider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NullEmbeddingProvider {
    pub dim: usize,
}

impl EmbeddingProvider for NullEmbeddingProvider {
    fn embed<'a>(&'a self, _text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move { Ok(vec![0.0_f32; dim]) }
    }
}

// ---------------------------------------------------------------------------
// MockEmbeddingProvider
//
// Gated: test-infra only — not part of the production public API.
// ---------------------------------------------------------------------------

/// Returns a deterministic embedding by using a simple FNV-1a hash over the
/// input bytes to seed each dimension.  Same text always → same vector; different
/// texts produce different vectors with overwhelming probability.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    pub dim: usize,
}

#[cfg(any(test, feature = "test-utils"))]
impl MockEmbeddingProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn hash_text(text: &str) -> u64 {
        // FNV-1a 64-bit
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        let base_hash = Self::hash_text(text);

        async move {
            let mut vec = Vec::with_capacity(dim);
            for i in 0..dim {
                const FNV_PRIME: u64 = 1099511628211;
                let h = base_hash
                    .wrapping_mul(FNV_PRIME)
                    .wrapping_add(i as u64)
                    .wrapping_mul(FNV_PRIME);
                // Map to [-1.0, 1.0]
                let val = (h as f32 / u64::MAX as f32) * 2.0 - 1.0;
                vec.push(val);
            }
            Ok(vec)
        }
    }
}

// ---------------------------------------------------------------------------
// OnnxEmbeddingProvider
// ---------------------------------------------------------------------------

/// Real embedding provider backed by all-MiniLM-L6-v2 (ONNX).
/// Produces 384-dimensional L2-normalised sentence embeddings.
/// Requires the `embeddings` feature flag.
#[cfg(feature = "embeddings")]
pub struct OnnxEmbeddingProvider {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "embeddings")]
impl OnnxEmbeddingProvider {
    /// Download the ONNX model + tokenizer from HuggingFace Hub (cached after
    /// first call) and build the inference session.
    pub fn new() -> Result<Self> {
        use anyhow::Context as _;

        let api = hf_hub::api::sync::Api::new().context("failed to init hf-hub API")?;

        let onnx_repo = api.model("optimum/all-MiniLM-L6-v2".to_string());
        let model_path = onnx_repo
            .get("model.onnx")
            .context("failed to download model.onnx from optimum/all-MiniLM-L6-v2")?;

        let tokenizer_repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
        let tokenizer_path = tokenizer_repo
            .get("tokenizer.json")
            .context("failed to download tokenizer.json")?;

        let session = ort::session::Session::builder()
            .context("failed to create ORT session builder")?
            .commit_from_file(&model_path)
            .context("failed to load ONNX model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
        })
    }

    /// Synchronous embed — tokenize, run ONNX inference, mean pool, L2 normalise.
    fn embed_sync(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use anyhow::Context as _;
        use ndarray::Array2;

        const MAX_SEQ_LEN: usize = 128;

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenisation failed: {e}"))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len().min(MAX_SEQ_LEN);

        let input_ids_data: Vec<i64> = ids[..seq_len].iter().map(|&x| x as i64).collect();
        let attention_mask_data: Vec<i64> = attention_mask[..seq_len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let token_type_ids_data: Vec<i64> = type_ids[..seq_len].iter().map(|&x| x as i64).collect();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids_data)
            .context("failed to build input_ids array")?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask_data)
            .context("failed to build attention_mask array")?;
        let token_type_ids_arr = Array2::from_shape_vec((1, seq_len), token_type_ids_data)
            .context("failed to build token_type_ids array")?;

        let input_ids_ref = ort::value::TensorRef::from_array_view(input_ids_arr.view())
            .context("failed to create input_ids tensor")?;
        let attention_mask_ref = ort::value::TensorRef::from_array_view(attention_mask_arr.view())
            .context("failed to create attention_mask tensor")?;
        let token_type_ids_ref = ort::value::TensorRef::from_array_view(token_type_ids_arr.view())
            .context("failed to create token_type_ids tensor")?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids"      => input_ids_ref,
                "attention_mask" => attention_mask_ref,
                "token_type_ids" => token_type_ids_ref
            ])
            .context("ONNX inference failed")?;

        let hidden: ndarray::ArrayViewD<f32> = outputs["last_hidden_state"]
            .try_extract_array()
            .context("failed to extract last_hidden_state")?;

        let shape = hidden.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 3,
            "expected 3-D hidden state, got {:?}",
            shape
        );
        let (_batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);

        // Mean pooling (attention-mask-weighted)
        let mut pooled = vec![0.0_f32; hidden_size];
        let mut mask_sum = 0.0_f32;
        for t in 0..seq {
            let m = attention_mask_arr[[0, t]] as f32;
            mask_sum += m;
            for d in 0..hidden_size {
                pooled[d] += hidden[[0, t, d]] * m;
            }
        }
        if mask_sum > 0.0 {
            for v in &mut pooled {
                *v /= mask_sum;
            }
        }

        // L2 normalise
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingProvider for OnnxEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_sync(text)
            .map_err(crate::core::error::Error::from)
    }
}

// stubs module removed — autoagents-llm is now a required dep (not optional).

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_embedding_returns_zero_vec() {
        let provider = NullEmbeddingProvider { dim: 384 };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime build failed");
        let vec = rt.block_on(provider.embed("test")).expect("embed failed");
        assert_eq!(vec.len(), 384);
        assert!(vec.iter().all(|&v| v == 0.0_f32));
    }

    #[test]
    fn null_embedding_dimension_matches() {
        let provider = NullEmbeddingProvider { dim: 128 };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime build failed");
        let vec = rt
            .block_on(provider.embed("anything"))
            .expect("embed failed");
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn mock_embedding_deterministic() {
        let provider = MockEmbeddingProvider::new(64);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime build failed");
        let a = rt
            .block_on(provider.embed("hello world"))
            .expect("embed failed");
        let b = rt
            .block_on(provider.embed("hello world"))
            .expect("embed failed");
        assert_eq!(a, b);
    }

    #[test]
    fn mock_embedding_different_inputs_differ() {
        let provider = MockEmbeddingProvider::new(64);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime build failed");
        let a = rt
            .block_on(provider.embed("hello world"))
            .expect("embed failed");
        let b = rt
            .block_on(provider.embed("goodbye world"))
            .expect("embed failed");
        assert_ne!(a, b);
    }

    #[test]
    fn token_usage_add() {
        let a = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        let b = TokenUsage {
            prompt_tokens: 20,
            completion_tokens: 15,
        };
        let sum = a + b;
        assert_eq!(sum.prompt_tokens, 30);
        assert_eq!(sum.completion_tokens, 20);
        assert_eq!(sum.total(), 50);
    }

    #[test]
    fn chat_msg_system_has_system_role() {
        let msg = chat_msg_system("you are helpful");
        assert_eq!(msg.role, ChatRole::System);
        assert_eq!(msg.content, "you are helpful");
        assert_eq!(msg.message_type, MessageType::Text);
    }

    #[test]
    fn chat_msg_user_has_user_role() {
        let msg = chat_msg_user("hello");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "hello");
        assert_eq!(msg.message_type, MessageType::Text);
    }

    #[tokio::test]
    async fn mock_provider_null_returns_none() {
        let provider = MockChatProvider::null();
        let msgs = vec![chat_msg_user("extract entities")];
        let resp = provider
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("mock chat_with_tools failed");
        // null provider returns "" which maps to None
        assert!(resp.text().is_none());
    }

    #[tokio::test]
    async fn mock_provider_substring_match() {
        let mut map = HashMap::new();
        map.insert("entities".to_string(), "[{\"name\":\"Alice\"}]".to_string());
        let provider = MockChatProvider::new(map);
        let msgs = vec![chat_msg_user("extract entities from this transcript")];
        let resp = provider
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("mock chat_with_tools failed");
        assert_eq!(resp.text().as_deref(), Some("[{\"name\":\"Alice\"}]"));
    }

    #[tokio::test]
    async fn mock_provider_no_match_returns_none() {
        let map = HashMap::new();
        let provider = MockChatProvider::new(map);
        let msgs = vec![chat_msg_user("summarise the meeting")];
        let resp = provider
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("mock chat_with_tools failed");
        assert!(resp.text().is_none());
    }

    // Real LlamaCppProvider smoke tests live in
    // `rust-pipeline/tests/llamacpp_smoke.rs` per Vera D.1a cycle-1
    // MEDIUM-1 + ADR-Phase-D.0 §7 — keeping `autoagents-llamacpp` out
    // of rql-core's dev-dependencies is required for the strict BYOM
    // invariant (`cargo tree -p rql-core | grep autoagents-llamacpp`
    // must print empty).

    // -----------------------------------------------------------------------
    // Step 1 — AA adoption: mock_provider_basic
    //
    // gemma_provider_builds_v2 and gemma_chat_round_trip_v2 were Red-phase
    // markers that duplicated llm_smoke::* tests (same assertions, different
    // paths).  Removed in Green per spec: "Delete the v2 duplicates.
    // Keep mock_provider_basic once the API supports it."
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod aa_adoption_green {
        use super::*;

        /// Verifies `MockChatProvider::with_response` convenience constructor.
        ///
        /// Constructs a mock with a single key → value pair, sends a message
        /// containing the key, and asserts the response text exactly matches
        /// the fixture value.
        #[tokio::test]
        async fn mock_provider_basic() {
            let provider = MockChatProvider::with_response("arithmetic", "The answer is 4.");
            let msgs = vec![chat_msg_user("arithmetic: what is 2+2?")];
            let resp = provider
                .chat_with_tools(&msgs, None, None)
                .await
                .expect("mock chat_with_tools should not fail");
            assert_eq!(
                resp.text().as_deref(),
                Some("The answer is 4."),
                "mock should return the exact fixture response for a matched key"
            );
        }
    }
}

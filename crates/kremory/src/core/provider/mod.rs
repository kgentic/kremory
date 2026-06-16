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
// — the consumer binary, not the substrate crate. DoD: `cargo tree -p kremory --edges normal
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
// ProviderCaps — per-provider structured-output capability detection.
//
// Used by StructuredCallBuilder to select the appropriate fallback ladder:
// NativeStructuredOutput → FormatSchema → PromptOnly.
//
// Conservative: unknown model strings degrade to PromptOnly.
// NEVER promote an unknown model to NativeStructuredOutput.
// ---------------------------------------------------------------------------

/// Per-provider structured-output capability detection.
///
/// Conservative: unknown model strings degrade to `PromptOnly`
/// (NEVER `NativeStructuredOutput`).
///
/// `pub` (not `pub(crate)`) because kremory-napi exposes capability detection
/// to consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderCaps {
    /// Native grammar-constrained structured output available.
    ///
    /// Includes Anthropic `output_config.format` (GA 2026-01-29) and
    /// OpenAI strict mode (`gpt-4o-2024-08-06+`, `o1`, `o3`, `o4` series).
    NativeStructuredOutput,
    /// JSON format-schema enforcement (Ollama llama.cpp grammar).
    ///
    /// Detected by the `name:tag` colon pattern used in Ollama model references
    /// (e.g. `qwen2.5:14b`, `llama3.2:3b-instruct`).
    FormatSchema,
    /// Prompt-only; no provider-side enforcement.
    ///
    /// Used for Bedrock model strings, unknown proxies, and any model string
    /// that does not match a known-capable pattern.
    PromptOnly,
}

/// Detect structured-output capability from the provider's self-reported model string.
///
/// ## Classification rules (spec §5.2)
///
/// 1. **Anthropic GA models** (`claude-opus-4-*`, `claude-sonnet-4-*`, `claude-haiku-4-5-*`,
///    `claude-mythos-preview`)
///    → `NativeStructuredOutput`
/// 2. **OpenAI strict** (`gpt-4o-2024-08-06+`, `gpt-4.*`, `o1-*`, `o3-*`, `o4-*`)
///    → `NativeStructuredOutput`
/// 3. **Ollama** (model string contains `:` — the `name:tag` pattern; or bare family names
///    `llama`, `qwen`, `phi`, `nuextract`, `mistral` without a dot — Anthropic/OpenAI models
///    never contain `:`)
///    → `FormatSchema`
/// 4. **Everything else** (Bedrock ARNs, unknown proxies, unrecognised strings)
///    → `PromptOnly` (conservative — unknown does NOT get NativeStructuredOutput)
///
/// Note: Bedrock cross-region inference ARNs (`us.`/`eu.`/`ap.` prefixes, e.g.
/// `us.anthropic.claude-opus-4-7-v1:0`) are explicitly guarded in the Bedrock block.
/// They cannot rely on fallthrough because the `:0` version suffix would otherwise
/// be matched by the Ollama `contains(':')` check, incorrectly returning `FormatSchema`.
pub fn capability_of(model: &str) -> ProviderCaps {
    // --- Bedrock / proxy ARN prefixes ----------------------------------------
    // Bedrock does not translate output_config.format; proxy routes these to
    // the upstream model without the native structured-output parameter.
    // Must be checked BEFORE the claude- / gpt- / ollama patterns below.
    //
    // Note: `mistral.` (with dot) catches Bedrock Mistral ARNs like
    // `mistral.mistral-large-2407-v1:0` BEFORE the bare `mistral` Ollama check below.
    //
    // Cross-region inference ARNs (`us.`/`eu.`/`ap.` prefixes, e.g.
    // `us.anthropic.claude-opus-4-7-v1:0`) MUST be explicitly guarded here because
    // they contain a colon in the version suffix (`:0`), which would otherwise be
    // caught by the Ollama `model.contains(':')` check and incorrectly classified
    // as FormatSchema. Bedrock cross-region docs:
    // https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
    if model.starts_with("anthropic.claude-")
        || model.starts_with("amazon.")
        || model.starts_with("meta.")
        || model.starts_with("mistral.")
        || model.starts_with("us.")
        || model.starts_with("eu.")
        || model.starts_with("ap.")
    {
        return ProviderCaps::PromptOnly;
    }

    // --- Anthropic GA models ------------------------------------------------
    // GA families with native output_config.format support (spec §5.2):
    //   claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-5-*, claude-mythos-preview
    if model.starts_with("claude-opus-4-")
        || model.starts_with("claude-sonnet-4-")
        || model.starts_with("claude-haiku-4-5")
        || model == "claude-mythos-preview"
    {
        return ProviderCaps::NativeStructuredOutput;
    }

    // --- OpenAI o-series (o1, o3, o4) ---------------------------------------
    // o1-*, o3-*, o4-* all support strict structured output.
    // Must be checked before the gpt- branch.
    if model.starts_with("o1-") || model.starts_with("o3-") || model.starts_with("o4-") {
        return ProviderCaps::NativeStructuredOutput;
    }

    // --- OpenAI gpt-4. numbered series (4.1, 4.5-preview, etc.) ------------
    // New OpenAI numbered models use the `gpt-4.` prefix (with dot).
    // Distinct from `gpt-4-turbo` (hyphen, no dot) which does not support strict mode.
    if model.starts_with("gpt-4.") {
        return ProviderCaps::NativeStructuredOutput;
    }

    // --- OpenAI gpt-4o strict (date >= 2024-08-06) --------------------------
    // Only gpt-4o-YYYY-MM-DD variants with date >= 2024-08-06 support strict mode.
    // gpt-4o alone (no date suffix) is ambiguous → conservative fall-through.
    // Lexicographic comparison of YYYY-MM-DD strings is correct.
    const GPT4O_STRICT_PREFIX: &str = "gpt-4o-";
    const GPT4O_STRICT_MIN_DATE: &str = "2024-08-06";
    if let Some(rest) = model.strip_prefix(GPT4O_STRICT_PREFIX) {
        // rest is e.g. "2024-08-06", "2024-12-17", "mini", or "preview".
        // Treat as NativeStructuredOutput only when rest is a YYYY-MM-DD date >= threshold.
        let date_candidate = if rest.len() >= 10 { &rest[..10] } else { "" };
        let looks_like_date = date_candidate.len() == 10
            && date_candidate.as_bytes()[4] == b'-'
            && date_candidate.as_bytes()[7] == b'-';
        if looks_like_date && date_candidate >= GPT4O_STRICT_MIN_DATE {
            return ProviderCaps::NativeStructuredOutput;
        }
        // gpt-4o-mini, gpt-4o-preview, gpt-4o-YYYY-MM-DD (< threshold) → fall through
    }

    // --- Ollama (name:tag colon pattern, plus known Ollama model families) ---
    // Anthropic and OpenAI model identifiers never contain ':'.
    // Ollama model references use the 'name:tag' form (qwen2.5:14b, llama3.2:3b).
    // Also catch bare family names commonly served via Ollama without a tag.
    // `mistral` (no dot) matches bare Ollama Mistral instances (mistral-7b, mistral-nemo).
    // `mistral.` (with dot) is caught by the Bedrock block above, so no overlap.
    if model.contains(':')
        || model.starts_with("llama")
        || model.starts_with("mistral")
        || model.starts_with("qwen")
        || model.starts_with("phi")
        || model.starts_with("nuextract")
    {
        return ProviderCaps::FormatSchema;
    }

    // --- Everything else → PromptOnly (conservative) -----------------------
    // Includes: Bedrock cross-region inference ARNs (us./eu./ap. prefixes),
    // unknown proxy routes, and any unrecognised model string.
    ProviderCaps::PromptOnly
}

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
// RecordReplayChatProvider — extracted to `record_replay` submodule.
// Re-exported here to preserve the public path `core::provider::*`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-utils"))]
mod record_replay;

#[cfg(any(test, feature = "test-utils"))]
pub use record_replay::{Cassette, CassetteEntry, RecordReplayChatProvider, VcrMode};

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
// DeterministicEmbeddingProvider — production Anthropic fallback
//
// Uses inline FNV-1a (matching MockEmbeddingProvider pattern; zero new dep).
// NOT gated — available in production builds. Named `Deterministic` (not Mock)
// per F-04 resolution: `MockEmbeddingProvider` remains test-utils gated.
// ---------------------------------------------------------------------------

/// Production-safe deterministic embedding provider.
///
/// Uses FNV-1a hashing to produce a fixed-dimension float vector from any string.
/// Embeddings are deterministic (same input → same output) but NOT semantic
/// (similar inputs produce unrelated vectors). Suitable only for structural recall
/// (exact-match entity lookup) where no embedding model API is available.
///
/// Used by `Memory::with_anthropic` — Anthropic has no embedding API.
///
/// Default `dim` = 384 — matches `NullEmbeddingProvider` and `OnnxEmbeddingProvider`
/// output dimension to preserve vector-column compatibility.
///
/// # Example
///
/// ```rust
/// use kremory::core::provider::DeterministicEmbeddingProvider;
/// let provider = DeterministicEmbeddingProvider::new(384);
/// ```
#[derive(Debug, Clone)]
pub struct DeterministicEmbeddingProvider {
    pub dim: usize,
}

impl DeterministicEmbeddingProvider {
    /// Create a new provider with the given output dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn hash_text(text: &str) -> u64 {
        // FNV-1a 64-bit inline (matches MockEmbeddingProvider pattern; zero new dep)
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

impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        let base_hash = Self::hash_text(text);

        async move {
            // Per-dimension hash: XOR base_hash with a dimension-specific seed
            // BEFORE any multiply, so each element starts from a completely
            // different state. Adding `i` to a large hash product (10^19+) loses
            // precision in f32 conversion — all elements collapse to nearly the
            // same value. XOR-first avoids that collapse.
            //
            // LCG multiplier (Knuth) chosen to spread dimension index across all
            // bits without depending on FNV magnitude.
            const LCG_MUL: u64 = 6364136223846793005;
            let mut vec = Vec::with_capacity(dim);
            for i in 0..dim {
                // Step 1: mix dimension index into a seed that differs by bits,
                //         not magnitude.
                let dim_seed = (i as u64)
                    .wrapping_mul(LCG_MUL)
                    .wrapping_add(1442695040888963407);
                // Step 2: XOR with text hash so same dimension → different text →
                //         different value.
                let h = base_hash ^ dim_seed;
                // Step 3: one more FNV-like avalanche to spread the bits.
                let h = h
                    .wrapping_mul(1099511628211_u64)
                    .wrapping_add(dim_seed.wrapping_mul(2654435761));
                // Map u64 to [-1.0, 1.0]
                let val = (h as f32 / u64::MAX as f32) * 2.0 - 1.0;
                vec.push(val);
            }
            Ok(vec)
        }
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
            // Same per-dimension hash as DeterministicEmbeddingProvider — XOR-first
            // avoids the f32 precision-collapse that occurs when adding small `i` to a
            // large product (~10^19), which made all elements nearly identical.
            const LCG_MUL: u64 = 6364136223846793005;
            let mut vec = Vec::with_capacity(dim);
            for i in 0..dim {
                let dim_seed = (i as u64)
                    .wrapping_mul(LCG_MUL)
                    .wrapping_add(1442695040888963407);
                let h = base_hash ^ dim_seed;
                let h = h
                    .wrapping_mul(1099511628211_u64)
                    .wrapping_add(dim_seed.wrapping_mul(2654435761));
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

// stubs module removed — autoagents-llm is now a required dep (not optional).

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
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
    // ProviderCaps + capability_of — Phase 1 structured-output ladder (§5.2)
    // -----------------------------------------------------------------------

    #[test]
    fn capability_of_anthropic_native() {
        assert_eq!(
            capability_of("claude-opus-4-7"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("claude-sonnet-4-6"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("claude-haiku-4-5-20251001"),
            ProviderCaps::NativeStructuredOutput
        );
    }

    #[test]
    fn capability_of_openai_strict() {
        assert_eq!(
            capability_of("gpt-4o-2024-08-06"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("o1-preview"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("o3-mini"),
            ProviderCaps::NativeStructuredOutput
        );
    }

    #[test]
    fn capability_of_openai_strict_later_dates() {
        // Dates strictly after 2024-08-06 must also be NativeStructuredOutput.
        assert_eq!(
            capability_of("gpt-4o-2024-12-17"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("gpt-4o-2025-01-01"),
            ProviderCaps::NativeStructuredOutput
        );
    }

    #[test]
    fn capability_of_old_openai_not_strict() {
        // Pre-2024-08-06 OpenAI GPT models lack strict mode → PromptOnly.
        assert_eq!(capability_of("gpt-4-turbo"), ProviderCaps::PromptOnly);
        assert_eq!(capability_of("gpt-3.5-turbo"), ProviderCaps::PromptOnly);
        // gpt-4o with an older date → PromptOnly.
        assert_eq!(capability_of("gpt-4o-2024-05-13"), ProviderCaps::PromptOnly);
        // gpt-4o alone (no date) → PromptOnly (ambiguous, conservative).
        assert_eq!(capability_of("gpt-4o"), ProviderCaps::PromptOnly);
    }

    #[test]
    fn capability_of_ollama_format_schema() {
        assert_eq!(capability_of("qwen2.5:14b"), ProviderCaps::FormatSchema);
        assert_eq!(
            capability_of("llama3.2:3b-instruct"),
            ProviderCaps::FormatSchema
        );
        // A hypothetical model name that looks like it could be OpenAI but uses Ollama tag syntax.
        assert_eq!(capability_of("gpt-oss:20b"), ProviderCaps::FormatSchema);
    }

    #[test]
    fn capability_of_unknown_defaults_to_prompt_only() {
        // Bedrock ARNs / proxies / unrecognized strings.
        assert_eq!(
            capability_of("anthropic.claude-3-opus-bedrock"),
            ProviderCaps::PromptOnly
        );
        assert_eq!(capability_of("random-model-xyz"), ProviderCaps::PromptOnly);
        // Bedrock cross-region inference ARNs (us./eu./ap. prefixes) fall through
        // to PromptOnly via the default case — implicit conservative routing.
        assert_eq!(
            capability_of("us.anthropic.claude-opus-4-7-v1:0"),
            ProviderCaps::PromptOnly
        );
        assert_eq!(
            capability_of("eu.meta.llama3-70b-instruct-v1:0"),
            ProviderCaps::PromptOnly
        );
    }

    #[test]
    fn capability_of_mistral_bare_ollama() {
        // Bare Mistral model names served via Ollama → FormatSchema.
        assert_eq!(capability_of("mistral-7b"), ProviderCaps::FormatSchema);
        assert_eq!(capability_of("mistral-nemo"), ProviderCaps::FormatSchema);
        // Bedrock Mistral ARNs (mistral. with dot) must remain PromptOnly.
        assert_eq!(
            capability_of("mistral.mistral-large-2407-v1:0"),
            ProviderCaps::PromptOnly
        );
    }

    #[test]
    fn capability_of_gpt_4_dot_native() {
        // OpenAI gpt-4. numbered series → NativeStructuredOutput.
        assert_eq!(
            capability_of("gpt-4.1"),
            ProviderCaps::NativeStructuredOutput
        );
        assert_eq!(
            capability_of("gpt-4.5-preview"),
            ProviderCaps::NativeStructuredOutput
        );
        // gpt-4-turbo (hyphen, no dot) has no strict mode → PromptOnly.
        assert_eq!(capability_of("gpt-4-turbo"), ProviderCaps::PromptOnly);
    }

    #[test]
    fn capability_of_claude_mythos_preview() {
        assert_eq!(
            capability_of("claude-mythos-preview"),
            ProviderCaps::NativeStructuredOutput
        );
    }

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

    // -----------------------------------------------------------------------
    // RecordReplayChatProvider — v0.2.4 Component 1 (spec §3.4 / §4 / §9 P2).
    // -----------------------------------------------------------------------
    #[cfg(test)]
    mod record_replay {
        use super::*;

        /// A minimal stub `ChatProvider` whose `model()` is configurable and
        /// whose `chat_with_tools` returns a fixed scripted response. No Ollama.
        #[derive(Debug)]
        struct StubProvider {
            model: String,
            response: String,
        }

        #[async_trait::async_trait]
        impl ChatProvider for StubProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[autoagents_llm::chat::Tool]>,
                _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
            ) -> std::result::Result<
                Box<dyn autoagents_llm::chat::ChatResponse>,
                autoagents_llm::error::LLMError,
            > {
                Ok(Box::new(MockChatResponse {
                    text: self.response.clone(),
                }))
            }

            fn model(&self) -> &str {
                &self.model
            }
        }

        fn temp_cassette_path(name: &str) -> std::path::PathBuf {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            p.push(format!("kremory_vcr_{name}_{nanos}.json"));
            p
        }

        /// (a) Record captures a response; (b) Replay returns the recorded
        /// response for a matching fingerprint (round-trip). Spec §9 P2 DoD.
        #[tokio::test]
        async fn record_then_replay_round_trips() {
            let cassette = temp_cassette_path("round_trip");
            let stub = Arc::new(StubProvider {
                model: "gemma4-e2b:latest".to_string(),
                response: r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
            });

            // Record
            let recorder = RecordReplayChatProvider::record(stub.clone(), cassette.clone());
            let msgs = vec![chat_msg_user("Alice met Bob in Boston.")];
            let recorded = recorder
                .chat_with_tools(&msgs, None, None)
                .await
                .expect("record chat should succeed");
            assert_eq!(
                recorded.text().as_deref(),
                Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#)
            );
            recorder.flush().expect("flush should write the cassette");

            // Replay — same request shape ⇒ same fingerprint ⇒ recorded response.
            let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
            let replayed = player
                .chat_with_tools(&msgs, None, None)
                .await
                .expect("replay should hit the recorded fingerprint");
            assert_eq!(
                replayed.text().as_deref(),
                Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#),
                "replay must return the exact recorded response text"
            );

            let _ = std::fs::remove_file(&cassette);
        }

        /// (b) Replay MISS returns a LOUD error naming the unmatched fingerprint.
        /// No silent default. Spec §4.3 / §9 P2 DoD.
        #[tokio::test]
        async fn replay_miss_is_loud_error() {
            let cassette = temp_cassette_path("miss");
            let stub = Arc::new(StubProvider {
                model: "gemma4-e2b:latest".to_string(),
                response: "recorded".to_string(),
            });
            let recorder = RecordReplayChatProvider::record(stub, cassette.clone());
            let recorded_msgs = vec![chat_msg_user("this exact request was recorded")];
            recorder
                .chat_with_tools(&recorded_msgs, None, None)
                .await
                .expect("record should succeed");
            recorder.flush().expect("flush");

            let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
            // A DIFFERENT request ⇒ different fingerprint ⇒ MISS.
            let other_msgs = vec![chat_msg_user("a totally different unrecorded request")];
            let err = player
                .chat_with_tools(&other_msgs, None, None)
                .await
                .expect_err("unrecorded fingerprint must produce a loud error, not a default");
            let msg = err.to_string();
            assert!(
                msg.contains("cassette MISS"),
                "error must name the miss: {msg}"
            );
            assert!(
                msg.contains("fingerprint="),
                "error must name the unmatched fingerprint: {msg}"
            );

            let _ = std::fs::remove_file(&cassette);
        }

        /// (c) `model()` returns the inner/cassette model, NOT the trait default
        /// `""`. Spec §3.4 ASMP-002 / §9 P2 DoD.
        #[tokio::test]
        async fn model_delegation_not_default_empty() {
            let cassette = temp_cassette_path("model");
            let stub = Arc::new(StubProvider {
                model: "gemma4-e2b:latest".to_string(),
                response: "x".to_string(),
            });

            // Record + Passthrough delegate to inner.model().
            let recorder = RecordReplayChatProvider::record(stub.clone(), cassette.clone());
            assert_eq!(recorder.model(), "gemma4-e2b:latest");
            let pass = RecordReplayChatProvider::passthrough(stub.clone());
            assert_eq!(pass.model(), "gemma4-e2b:latest");

            // Replay returns the cassette header model.
            recorder.flush().expect("flush");
            let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
            assert_eq!(
                player.model(),
                "gemma4-e2b:latest",
                "replay model() must read the cassette header, not the \"\" default"
            );
            assert_ne!(player.model(), "", "model() must never collapse to empty");

            let _ = std::fs::remove_file(&cassette);
        }

        /// (LOW-003) Record→replay round-trip through the production FormatSchema
        /// arm: a POPULATED `StructuredOutputFormat` participates in the fingerprint
        /// (§4.2). The other record_replay tests all pass `None` schema, so this is
        /// the only coverage that locks the `Some(schema)` serialization branch.
        /// Asserts (a) the populated-schema request round-trips, and (b) a DIFFERENT
        /// schema yields a cassette MISS — proving the schema is fingerprinted.
        #[tokio::test]
        async fn populated_schema_participates_in_fingerprint() {
            use autoagents_llm::chat::StructuredOutputFormat;

            let cassette = temp_cassette_path("populated_schema");
            let stub = Arc::new(StubProvider {
                model: "gemma4-e2b:latest".to_string(),
                response: r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
            });

            // Mirror how production builds the schema (structured.rs FormatSchema arm).
            let schema_a = StructuredOutputFormat {
                name: "entities".to_string(),
                description: None,
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "entities": { "type": "array" } },
                    "required": ["entities"]
                })),
                strict: Some(false),
            };
            let msgs = vec![chat_msg_user("Alice met Bob in Boston.")];

            // Record with the populated schema.
            let recorder = RecordReplayChatProvider::record(stub.clone(), cassette.clone());
            recorder
                .chat_with_tools(&msgs, None, Some(schema_a.clone()))
                .await
                .expect("record chat with schema should succeed");
            recorder.flush().expect("flush should write the cassette");

            // Replay with the SAME schema ⇒ same fingerprint ⇒ recorded response.
            let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
            let replayed = player
                .chat_with_tools(&msgs, None, Some(schema_a.clone()))
                .await
                .expect("replay with the recorded schema should hit");
            assert_eq!(
                replayed.text().as_deref(),
                Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#),
                "populated-schema replay must return the exact recorded response"
            );

            // Replay with a DIFFERENT schema (same messages/tools/model) ⇒ different
            // fingerprint ⇒ MISS. Proves the schema participates in the fingerprint.
            let schema_b = StructuredOutputFormat {
                name: "entities".to_string(),
                description: None,
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "facts": { "type": "array" } },
                    "required": ["facts"]
                })),
                strict: Some(false),
            };
            let player2 = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
            let err = player2
                .chat_with_tools(&msgs, None, Some(schema_b))
                .await
                .expect_err("a different schema must MISS, proving schema is fingerprinted");
            let msg = err.to_string();
            assert!(
                msg.contains("cassette MISS"),
                "different-schema request must miss: {msg}"
            );

            let _ = std::fs::remove_file(&cassette);
        }

        /// (NEW-201) Compile-level proof the decorator is `Send + Sync`: it must
        /// bind in the `Arc<dyn ChatProvider>` position. A `RefCell`-based impl
        /// would fail this bound.
        #[tokio::test]
        async fn decorator_is_send_sync_as_dyn() {
            let stub = Arc::new(StubProvider {
                model: "gemma4-e2b:latest".to_string(),
                response: "y".to_string(),
            });
            let provider: Arc<dyn ChatProvider> =
                Arc::new(RecordReplayChatProvider::passthrough(stub));
            fn assert_send_sync<T: Send + Sync>(_t: &T) {}
            assert_send_sync(&provider);
            assert_eq!(provider.model(), "gemma4-e2b:latest");
        }
    }
}

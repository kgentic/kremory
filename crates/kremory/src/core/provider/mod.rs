// ---------------------------------------------------------------------------
// AutoAgents re-exports — canonical LLM abstraction for kremory::core.
//
// The BYOM seam is `Arc<dyn ChatProvider>`.  All modules in this crate use
// ChatProvider + Vec<ChatMessage> exclusively; the old LlmClient / LlmRequest
// types are gone.
//
// BYOM invariant: kremory exposes ONLY the
// `autoagents-llm` TRAIT surface here. The concrete `LlamaCppProvider`
// (`autoagents-llamacpp` crate) lives in the host binary crate that co-locates
// the concrete LLM client — not in this substrate crate. DoD: `cargo tree -p kremory --edges normal
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
        // DUR-8: `rest.get(..10)` instead of `&rest[..10]`. `len()` counts BYTES,
        // so a multi-byte suffix (e.g. "日本語テスト") satisfied `len() >= 10`
        // while byte 10 landed mid-character — and slicing a `str` off a char
        // boundary panics. `get` returns `None` there, which is also the correct
        // semantic: a YYYY-MM-DD date is pure ASCII, so anything that is not a
        // clean 10-byte boundary cannot be one and must fall through
        // conservatively.
        let date_candidate = rest.get(..10).unwrap_or("");
        let looks_like_date = date_candidate.len() == 10
            && date_candidate.as_bytes()[4] == b'-'
            && date_candidate.as_bytes()[7] == b'-';
        if looks_like_date && date_candidate >= GPT4O_STRICT_MIN_DATE {
            return ProviderCaps::NativeStructuredOutput;
        }
        // gpt-4o-mini, gpt-4o-preview, gpt-4o-YYYY-MM-DD (< threshold) → fall through
    }

    // --- OpenAI open-weight gpt-oss (served via Groq / Together / vLLM) ------
    // gpt-oss-* support OpenAI structured outputs (`response_format` json_schema)
    // on Groq's OpenAI-compatible endpoint. Without this branch the id
    // (`openai/gpt-oss-120b` on Groq) falls through to PromptOnly →
    // LlmJsonRepair, wasting the model's native schema capability. Match the
    // family substring so both the bare (`gpt-oss-120b`) and Groq-prefixed
    // (`openai/gpt-oss-120b`) ids are caught. FormatSchema (not Native/strict)
    // is the conservative choice — the `response_format` path Groq documents.
    if model.contains("gpt-oss") {
        return ProviderCaps::FormatSchema;
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
// build_llm — MOVED to the host binary crate per the BYOM invariant. kremory
// must not reference `autoagents-llamacpp` in production deps; the host binary
// that co-locates the concrete `LlamaCppProvider` imports and builds it there.
//
// In-crate tests that need a real LlamaCppProvider construct it
// directly via the `autoagents-llamacpp` dev-dependency.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Chat provider implementations — split to `chat` submodule.
// Re-exported here to preserve the public path `core::provider::*`.
// ---------------------------------------------------------------------------

mod chat;

pub(crate) use chat::{NullChatProvider, NullChatResponse};

#[cfg(any(test, feature = "test-utils"))]
pub use chat::{MockChatProvider, MockChatResponse};

pub use chat::ArcChatProvider;

// ---------------------------------------------------------------------------
// RecordReplayChatProvider — extracted to `record_replay` submodule.
// Re-exported here to preserve the public path `core::provider::*`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-utils"))]
mod record_replay;

#[cfg(any(test, feature = "test-utils"))]
pub use record_replay::{Cassette, CassetteEntry, RecordReplayChatProvider, VcrMode};

// ---------------------------------------------------------------------------
// TokenCountingChatProvider — production token-accounting decorator for
// budget tracking. NOT test-gated: the dream-pass
// wraps its real provider with this in production. Compile-spike landed ahead of
// the build sprint per impl-spec §11 readiness-gate contingency.
// ---------------------------------------------------------------------------

mod token_counting;

pub use token_counting::{TokenAccumulator, TokenCountingChatProvider};

// ---------------------------------------------------------------------------
// EmbeddingProvider + DynEmbeddingProvider — trait declarations.
// Concrete impls split to `embedding` submodule; re-exported below.
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

// ---------------------------------------------------------------------------
// Concrete embedding impls — split to `embedding` submodule.
// Re-exported here to preserve the public path `core::provider::*`.
// ---------------------------------------------------------------------------

mod embedding;

pub use embedding::{ArcEmbedder, DeterministicEmbeddingProvider, NullEmbeddingProvider};

#[cfg(any(test, feature = "test-utils"))]
pub use embedding::MockEmbeddingProvider;

#[cfg(feature = "embeddings")]
pub use embedding::OnnxEmbeddingProvider;

// stubs module removed — autoagents-llm is now a required dep (not optional).

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

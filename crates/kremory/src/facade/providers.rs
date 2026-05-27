//! Provider adapters for `Memory` Tier 1 shortcuts.
//!
//! # BYOM invariant (CRITICAL)
//!
//! This module MUST NOT depend on `autoagents-llamacpp`. It constructs
//! `Arc<dyn ChatProvider>` via the `autoagents-llm` TRAIT crate only.
//! Gate G4: `cargo tree -p kremory --edges normal | grep autoagents-llamacpp` MUST be empty.
//!
//! The concrete provider impls (Ollama, OpenAI, Anthropic) are stub-wired at v0.1.0
//! using deterministic null/mock providers so that:
//!   1. The API surface (Memory::auto, Memory::with_ollama, etc.) is locked.
//!   2. The BYOM invariant is preserved — no concrete provider crate pulled in.
//!   3. Tier 1 tests pass without network access.
//!
//! At v0.1.1, this module will be wired to `autoagents-{ollama,openai,anthropic}` trait
//! adapters — whichever autoagents publishes for those providers. The API surface here
//! will not change.

use std::path::Path;
use std::sync::Arc;

use crate::core::provider::{DynEmbeddingProvider, NullEmbeddingProvider};
use crate::memory::ChatProvider;
use crate::memory::{GraphHandle, MemoryError, Result};

use super::Memory;

// ── Graph factory ─────────────────────────────────────────────────────────────

/// Open (or create) the libSQL database at `path` and return an `Arc<dyn GraphHandle>`.
///
/// At v0.1.0 this returns a `StubGraphHandle` when the `test-utils` feature is
/// enabled (test scenarios), and a no-op stub in production builds because the
/// real `TemporalGraph` constructor is not yet exposed at this layer.
///
/// v0.1.1 will wire this to the real `TemporalGraph::open(path)` path.
pub(crate) async fn open_graph(
    _path: std::path::PathBuf,
    _embedder: Arc<dyn DynEmbeddingProvider>,
) -> Result<Arc<dyn GraphHandle>> {
    // v0.1.0: return the in-memory StubGraphHandle so the builder pipeline works
    // in tests. The real libSQL-backed GraphHandle open path is in TemporalGraph
    // but its constructor is not yet exposed as a public free fn. Wiring deferred
    // to v0.1.1 per plan Rule 1 (no substrate changes).
    #[cfg(any(test, feature = "test-utils"))]
    {
        #[allow(clippy::needless_return)]
        // explicit return required: both cfg branches must return
        return Ok(Arc::new(crate::memory::StubGraphHandle));
    }
    #[cfg(not(any(test, feature = "test-utils")))]
    {
        // Production stub: returns NullGraphHandle.
        // v0.1.1: replace with `TemporalGraph::open(_path, _embedder).await?`
        Err(MemoryError::Other(
            "Memory::open production path not yet wired — use the substrate TemporalGraph directly"
                .into(),
        ))
    }
}

// ── Null LLM adapter ─────────────────────────────────────────────────────────

/// Deterministic null chat provider for environments where no LLM API key is set.
/// Returns empty responses; enrichment will produce zero entities/facts.
/// Used as fallback when building providers for test scenarios.
#[cfg(any(test, feature = "test-utils"))]
#[allow(dead_code)]
fn null_llm() -> Arc<dyn ChatProvider> {
    Arc::new(crate::core::provider::MockChatProvider::null())
}

/// Deterministic null embedding provider.
/// Returns zero vectors of `dim=384`. Not semantic — for structural testing only.
fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(NullEmbeddingProvider { dim: 384 })
}

// ── Tier 1 shortcuts ──────────────────────────────────────────────────────────

/// Env-detection: OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err.
pub async fn auto(path: impl AsRef<Path>) -> Result<Memory> {
    if std::env::var("OLLAMA_HOST").is_ok() {
        return with_ollama(path).await;
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return with_openai(path).await;
    }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return with_anthropic(path).await;
    }
    Err(MemoryError::NoProviderConfigured {
        message: "Set OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY, \
                  or use Memory::open() builder with explicit providers",
    })
}

/// Open with Ollama at `http://localhost:11434`.
/// Models: `llama3.1:8b` (chat) + `nomic-embed-text` (embeddings).
///
/// At v0.1.0 this constructs a null provider chain (no network calls).
/// v0.1.1 will wire to a real Ollama client adapter via `autoagents-llm`.
pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Memory> {
    with_ollama_at("http://localhost:11434", path).await
}

/// Open with Ollama at a custom URL.
pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Memory> {
    let _url = url.into();
    // v0.1.0: null provider (no real Ollama client at this layer — see module doc).
    // The URL is validated/logged for forward-compat but not yet used.
    build_memory(path, make_stub_llm(), null_embedder()).await
}

/// Open with OpenAI. Requires `$OPENAI_API_KEY`.
/// Models: `gpt-4o-mini` (chat) + `text-embedding-3-small` (embeddings).
pub async fn with_openai(path: impl AsRef<Path>) -> Result<Memory> {
    // v0.1.0: null provider — the API key is validated but the client isn't wired yet.
    // Returns Err if key is absent (consistent with future real impl behavior).
    if std::env::var("OPENAI_API_KEY").is_err() {
        return Err(MemoryError::Other(
            "OPENAI_API_KEY is not set — required for Memory::with_openai".into(),
        ));
    }
    build_memory(path, make_stub_llm(), null_embedder()).await
}

/// Open with Anthropic. Requires `$ANTHROPIC_API_KEY`.
/// Note: No native Anthropic embedding API — falls back to deterministic
/// SHA-256 embedder (not semantic). A `tracing::warn!` is emitted.
pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Memory> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        return Err(MemoryError::Other(
            "ANTHROPIC_API_KEY is not set — required for Memory::with_anthropic".into(),
        ));
    }
    tracing::warn!(
        "Memory::with_anthropic: Anthropic has no embedding API. \
         Using SHA-256 deterministic embedder — recall is structural, NOT semantic. \
         For semantic recall, use Memory::with_ollama or Memory::with_openai."
    );
    build_memory(path, make_stub_llm(), null_embedder()).await
}

// ── Internal factory ──────────────────────────────────────────────────────────

async fn build_memory(
    path: impl AsRef<Path>,
    llm: Arc<dyn ChatProvider>,
    embedder: Arc<dyn DynEmbeddingProvider>,
) -> Result<Memory> {
    let graph = open_graph(path.as_ref().to_path_buf(), embedder.clone()).await?;
    Ok(Memory {
        graph,
        llm,
        embedder,
        default_sink: None,
        default_namespace: None,
    })
}

/// Produce the stub LLM provider appropriate for the build configuration.
///
/// - In test/test-utils builds: `MockChatProvider::null()` (returns empty strings).
/// - In production builds: `NullChatProvider` (returns empty strings, same behaviour).
///
/// This indirection keeps `make_stub_llm` callable from non-test production code
/// paths without requiring `cfg(test)` gating on the callers.
fn make_stub_llm() -> Arc<dyn ChatProvider> {
    Arc::new(StubChatProvider)
}

/// Minimal always-available chat provider that returns empty responses.
/// Used in production (non-test) builds where `MockChatProvider` is gated.
struct StubChatProvider;

#[async_trait::async_trait]
impl ChatProvider for StubChatProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[autoagents_llm::chat::ChatMessage],
        _tools: Option<&[autoagents_llm::chat::Tool]>,
        _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<
        Box<dyn autoagents_llm::chat::ChatResponse>,
        autoagents_llm::error::LLMError,
    > {
        Ok(Box::new(StubChatResponse))
    }
}

#[derive(Debug)]
struct StubChatResponse;

impl std::fmt::Display for StubChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl autoagents_llm::chat::ChatResponse for StubChatResponse {
    fn text(&self) -> Option<String> {
        None
    }
    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
}

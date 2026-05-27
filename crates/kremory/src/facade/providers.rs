//! Provider adapters for `Memory` Tier 1 shortcuts.
//!
//! # BYOM invariant (CRITICAL)
//!
//! This module MUST NOT depend on `autoagents-llamacpp`. It constructs
//! `Arc<dyn ChatProvider>` via the `autoagents-llm` TRAIT crate only.
//! Gate G4: `cargo tree -p kremory --edges normal | grep autoagents-llamacpp` MUST be empty.
//!
//! The concrete provider impls (Ollama, OpenAI, Anthropic) are from the
//! `autoagents-llm` crate's `backends::` module, gated behind provider feature
//! flags (ollama / openai / anthropic). This is the same crate — not a separate
//! concrete implementation crate — so BYOM is preserved.

use std::path::Path;
use std::sync::Arc;

use autoagents_llm::{
    backends::{anthropic::Anthropic, ollama::Ollama, openai::OpenAI},
    builder::LLMBuilder,
    embedding::{model_provider::EmbeddingBuilder, EmbeddingProvider as AutoEmbeddingProvider},
};

use crate::core::config::PipelineConfig;
use crate::core::ingest::Engine;
use crate::core::provider::{
    ArcChatProvider, ArcEmbedder, DeterministicEmbeddingProvider, DynEmbeddingProvider,
};
use crate::core::schema::TemporalGraph;
use crate::memory::engine_handle::EngineGraphHandle;
use crate::memory::llm_adapters::AutoagentsEmbedderAdapter;
use crate::memory::{ChatProvider, GraphHandle, MemoryError, Result};

use super::Memory;

// ── Graph factory ─────────────────────────────────────────────────────────────

/// Open (or create) the libSQL database at `path` and return an `Arc<dyn GraphHandle>`
/// backed by a real `EngineGraphHandle`.
///
/// Accepts a BYOM `llm` (any `Arc<dyn ChatProvider>`) and `embedder`
/// (any `Arc<dyn DynEmbeddingProvider>`). Both are adapted into the newtype
/// wrappers required by the `Engine` generic bounds.
///
/// `embedding_dim` — when `Some`, overrides the default 384-dim vector index.
/// Pass `Some(768)` for `nomic-embed-text`, `Some(1536)` for OpenAI
/// `text-embedding-3-small`, etc. `None` keeps the 384-dim default (MiniLM).
///
/// Called by:
/// - The Tier 2 builder `IntoFuture` (`Memory::open(...).with_llm(...).with_embedder(...).await`)
/// - All Tier 1 shortcuts (`with_ollama`, `with_openai`, `with_anthropic`)
pub(crate) async fn open_graph(
    path: impl AsRef<Path>,
    llm: Arc<dyn ChatProvider>,
    embedder: Arc<dyn DynEmbeddingProvider>,
    embedding_dim: Option<usize>,
) -> Result<Arc<dyn GraphHandle>> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| MemoryError::Other("path contains non-UTF-8 characters".into()))?;

    let resolved_dim = embedding_dim.unwrap_or(384);
    let graph = Arc::new(
        TemporalGraph::open_with_dim(path_str, resolved_dim)
            .await
            .map_err(MemoryError::Core)?,
    );

    let config = PipelineConfig::builder()
        .embedding_dim(resolved_dim)
        .build()
        .map_err(MemoryError::Core)?;

    let engine = Engine::new(
        graph,
        Arc::new(ArcChatProvider::new(llm)),
        Arc::new(ArcEmbedder(embedder)),
        config,
    );

    Ok(Arc::new(EngineGraphHandle::new(engine)) as Arc<dyn GraphHandle>)
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
///
/// Chat model: `llama3.2` · Embedding model: `nomic-embed-text`.
///
/// Requires a running Ollama server. If no Ollama is available at runtime, the
/// first `.remember()` or `.recall()` call will return a network error.
pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Memory> {
    with_ollama_at("http://localhost:11434", path).await
}

/// Open with Ollama at a custom URL.
///
/// Chat model: `llama3.2` · Embedding model: `nomic-embed-text` (dim=768).
pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Memory> {
    let url: String = url.into();

    let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&url)
        .model("llama3.2")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama chat provider error: {e}")))?;

    let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama embedding provider error: {e}")))?;

    let llm: Arc<dyn ChatProvider> = chat_provider;
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<Ollama>::new(embed_provider),
    ));

    // nomic-embed-text outputs 768-dim vectors.
    build_memory(path, llm, embedder, Some(768)).await
}

/// Open with OpenAI. Requires `$OPENAI_API_KEY`.
///
/// Chat model: `gpt-4o-mini` · Embedding model: `text-embedding-3-small`.
pub async fn with_openai(path: impl AsRef<Path>) -> Result<Memory> {
    let key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| MemoryError::Other("OPENAI_API_KEY is not set".into()))?;

    let chat_provider: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(&key)
        .model("gpt-4o-mini")
        .build()
        .map_err(|e| MemoryError::Other(format!("OpenAI chat provider error: {e}")))?;

    let embed_provider: Arc<OpenAI> = EmbeddingBuilder::<OpenAI>::new()
        .api_key(&key)
        .model("text-embedding-3-small")
        .build()
        .map_err(|e| MemoryError::Other(format!("OpenAI embedding provider error: {e}")))?;

    let llm: Arc<dyn ChatProvider> = chat_provider;
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<OpenAI>::new(embed_provider),
    ));

    // text-embedding-3-small outputs 1536-dim vectors.
    build_memory(path, llm, embedder, Some(1536)).await
}

/// Open with Anthropic. Requires `$ANTHROPIC_API_KEY`.
///
/// Chat model: `claude-3-haiku-20240307`.
///
/// **Note**: Anthropic has no embedding API. Uses `DeterministicEmbeddingProvider`
/// (FNV-1a, dim=384) as a non-semantic stand-in. Bring your own embedder via
/// `Memory::open().with_llm(...).with_embedder(...)` for production search quality.
/// A `tracing::warn!` is emitted at construction time.
pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Memory> {
    let key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| MemoryError::Other("ANTHROPIC_API_KEY is not set".into()))?;

    let chat_provider: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(&key)
        .model("claude-3-haiku-20240307")
        .build()
        .map_err(|e| MemoryError::Other(format!("Anthropic chat provider error: {e}")))?;

    tracing::warn!(
        "Memory::with_anthropic: Anthropic has no embedding API. \
         Using DeterministicEmbeddingProvider (FNV-1a, dim=384) — recall is structural, NOT semantic. \
         For semantic recall, use Memory::with_ollama or Memory::with_openai, \
         or wire a custom embedder via Memory::open().with_embedder(...)."
    );

    let llm: Arc<dyn ChatProvider> = chat_provider;
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    // DeterministicEmbeddingProvider is 384-dim (FNV-1a fallback).
    build_memory(path, llm, embedder, Some(384)).await
}

// ── Internal factory ──────────────────────────────────────────────────────────

async fn build_memory(
    path: impl AsRef<Path>,
    llm: Arc<dyn ChatProvider>,
    embedder: Arc<dyn DynEmbeddingProvider>,
    embedding_dim: Option<usize>,
) -> Result<Memory> {
    let graph = open_graph(path.as_ref(), llm.clone(), embedder.clone(), embedding_dim).await?;
    Ok(Memory {
        graph,
        llm,
        embedder,
        default_sink: None,
        default_namespace: None,
    })
}

// ── EmbedderArc helper ────────────────────────────────────────────────────────
//
// `AutoagentsEmbedderAdapter<P>` takes ownership of `P: AutoEmbeddingProvider`.
// The Tier 1 shortcuts hold `Arc<P>` (from the builder's `build()` return).
// This thin newtype delegates to the inner `Arc<P>` so we don't need to unwrap
// the Arc (which would fail since Arc<P> is not P, and Clone is not guaranteed).

struct EmbedderArc<P> {
    inner: Arc<P>,
}

impl<P> EmbedderArc<P> {
    fn new(arc: Arc<P>) -> Self {
        Self { inner: arc }
    }
}

#[async_trait::async_trait]
impl<P> AutoEmbeddingProvider for EmbedderArc<P>
where
    P: AutoEmbeddingProvider + Send + Sync,
{
    async fn embed(
        &self,
        input: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, autoagents_llm::error::LLMError> {
        self.inner.embed(input).await
    }
}

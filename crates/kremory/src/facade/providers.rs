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

use crate::core::chat_tracking::TokenTrackingChatProvider;
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
    extractor_source: Option<crate::core::extraction::ExtractorSource>,
) -> Result<(Arc<dyn GraphHandle>, Arc<TemporalGraph>)> {
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
    let graph_for_facade = Arc::clone(&graph);

    let config = PipelineConfig::builder()
        .embedding_dim(resolved_dim)
        .build()
        .map_err(MemoryError::Core)?;

    // Use explicit extractor source if pinned via builder; otherwise default
    // to FromEnv (reads KREMORY_EXTRACTOR or falls back to NuExtract).
    let source = extractor_source
        .unwrap_or(crate::core::extraction::ExtractorSource::FromEnv);
    let engine = Engine::with_extractor_source(
        graph,
        Arc::new(ArcChatProvider::new(llm)),
        Arc::new(ArcEmbedder(embedder)),
        config,
        source,
    )
    .map_err(MemoryError::Core)?;

    let handle: Arc<dyn GraphHandle> = Arc::new(EngineGraphHandle::new(engine));
    Ok((handle, graph_for_facade))
}

// ── Tier 1 shortcuts ──────────────────────────────────────────────────────────

/// Env-detection: OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err.
///
/// When `OLLAMA_HOST` is set, its VALUE is used as the Ollama base URL (was
/// previously hardcoded to `http://localhost:11434` regardless). When
/// `OLLAMA_CHAT_MODEL` is also set, its VALUE overrides the default chat
/// model. Previously both were silently dropped, causing env-based config
/// from consumers to be ignored and breaking smoke
/// runs on hosts where `localhost` resolves to `::1` but Ollama binds
/// IPv4-only, or where the default MLX chat model hangs on `/api/chat`.
pub async fn auto(path: impl AsRef<Path>) -> Result<Memory> {
    if let Ok(host) = std::env::var("OLLAMA_HOST") {
        let model = std::env::var("OLLAMA_CHAT_MODEL").ok();
        return with_ollama_at_model(host, model, path).await;
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
/// Chat model: `qwen3.5:9b-mlx` · Embedding model: `nomic-embed-text`.
///
/// Default chat model upgraded from llama3.2 to qwen3.5:9b-mlx per DAG
/// MLX bench (2026-05-30, M4 Max): qwen3.5:9b-mlx delivered 20.1 tok/s
/// (1.83× faster than qwen3:14b GGUF baseline; 3.4× on ReAct loops) with
/// tool-calling acc 1.00 and ReAct acc 1.00. Reference:
/// `~/Documents/Projects/DAG/.ai-docs/research/model-bench-2026-05-30-qwen-mlx-vs-gguf.md`
/// Requires `ollama pull qwen3.5:9b-mlx` (Ollama 0.24+ with MLX backend).
///
/// Requires a running Ollama server. If no Ollama is available at runtime, the
/// first `.remember()` or `.recall()` call will return a network error.
pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Memory> {
    with_ollama_at("http://localhost:11434", path).await
}

/// Open with Ollama at a custom URL and optional custom chat model.
///
/// When `model` is `None`, the default `qwen3.5:9b-mlx` is used. Pass
/// `Some("qwen2.5:14b")` (or any other pulled GGUF model) to override —
/// useful when the default MLX model is unavailable, the host doesn't
/// support MLX, or `/api/chat` hangs on MLX subprocess (Ollama issues
/// #15334 + #15258).
pub async fn with_ollama_at_model(
    url: impl Into<String>,
    model: Option<String>,
    path: impl AsRef<Path>,
) -> Result<Memory> {
    let url: String = url.into();
    let model: String = model.unwrap_or_else(|| "qwen3.5:9b-mlx".to_string());

    let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&url)
        .model(model.clone())
        .timeout_seconds(10)
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama chat provider error: {e}")))?;

    let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama embedding provider error: {e}")))?;

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    // Clone before Box::leak so the warmup call can borrow the model string
    // after TokenTrackingChatProvider has consumed it via into_boxed_str.
    let model_for_warmup = model.clone();
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        "ollama",
        Box::leak(model.into_boxed_str()),
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<Ollama>::new(embed_provider),
    ));

    build_memory_with_model(path, llm, embedder, Some(768), Some(&model_for_warmup)).await
}

/// Open with Ollama at a custom URL.
///
/// Chat model: `qwen3.5:9b-mlx` · Embedding model: `nomic-embed-text` (dim=768).
pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Memory> {
    let url: String = url.into();

    // MLX-backed default per DAG 2026-05-30 bench. 8.9 GB model; fits comfortably
    // in 36GB unified memory; Metal acceleration via Ollama MLX backend.
    let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&url)
        .model("qwen3.5:9b-mlx")
        .timeout_seconds(10)
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama chat provider error: {e}")))?;

    let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama embedding provider error: {e}")))?;

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        "ollama",
        "qwen3.5:9b-mlx",
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<Ollama>::new(embed_provider),
    ));

    // nomic-embed-text outputs 768-dim vectors.
    build_memory_with_model(path, llm, embedder, Some(768), Some("qwen3.5:9b-mlx")).await
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
        .timeout_seconds(10)
        .build()
        .map_err(|e| MemoryError::Other(format!("OpenAI chat provider error: {e}")))?;

    let embed_provider: Arc<OpenAI> = EmbeddingBuilder::<OpenAI>::new()
        .api_key(&key)
        .model("text-embedding-3-small")
        .build()
        .map_err(|e| MemoryError::Other(format!("OpenAI embedding provider error: {e}")))?;

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        "openai",
        "gpt-4o-mini",
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<OpenAI>::new(embed_provider),
    ));

    // text-embedding-3-small outputs 1536-dim vectors.
    build_memory_with_model(path, llm, embedder, Some(1536), Some("gpt-4o-mini")).await
}

/// Open with Anthropic. Requires `$ANTHROPIC_API_KEY`.
///
/// Chat model: `claude-haiku-4-5` (metric label) / `claude-3-haiku-20240307` (API identifier).
///
/// Note: the Anthropic API still uses `claude-3-haiku-20240307` as the model string.
/// The metric label is `claude-haiku-4-5` to match the `monitoring/provider-rates.toml`
/// entry (spec §F-09).
///
/// **Note**: Anthropic has no embedding API. Uses `DeterministicEmbeddingProvider`
/// (FNV-1a, dim=384) as a non-semantic stand-in. Bring your own embedder via
/// `Memory::open().with_llm(...).with_embedder(...)` for production search quality.
/// A `tracing::warn!` is emitted at construction time.
pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Memory> {
    let key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| MemoryError::Other("ANTHROPIC_API_KEY is not set".into()))?;

    // Anthropic API identifier — the model string sent in HTTP requests.
    // NOTE: "claude-3-haiku-20240307" is the API-level name for claude-haiku-4-5.
    // The metric label below uses "claude-haiku-4-5" to match provider-rates.toml.
    let chat_provider: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(&key)
        .model("claude-3-haiku-20240307")
        .timeout_seconds(10)
        .build()
        .map_err(|e| MemoryError::Other(format!("Anthropic chat provider error: {e}")))?;

    tracing::warn!(
        "Memory::with_anthropic: Anthropic has no embedding API. \
         Using DeterministicEmbeddingProvider (FNV-1a, dim=384) — recall is structural, NOT semantic. \
         For semantic recall, use Memory::with_ollama or Memory::with_openai, \
         or wire a custom embedder via Memory::open().with_embedder(...)."
    );

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    // Metric label "claude-haiku-4-5" matches monitoring/provider-rates.toml.
    // API model "claude-3-haiku-20240307" is handled by the LLMBuilder above.
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        "anthropic",
        "claude-haiku-4-5",
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    // DeterministicEmbeddingProvider is 384-dim (FNV-1a fallback).
    // Warmup uses the API model string so the Anthropic NativeSchema arm
    // (24-hour server-side schema cache) is reached at startup.
    build_memory_with_model(
        path,
        llm,
        embedder,
        Some(384),
        Some("claude-3-haiku-20240307"),
    )
    .await
}

// ── Internal factory ──────────────────────────────────────────────────────────

/// Internal factory that accepts an optional model string for warmup.
///
/// `model` is forwarded to [`warm_schema_caches`] so that Tier 1 callers
/// (which know the concrete model string at construction time) reach the
/// provider-native schema-compilation arm (NativeSchema for Anthropic /
/// OpenAI, FormatSchema for Ollama) rather than falling back to connection-pool
/// warm only.
async fn build_memory_with_model(
    path: impl AsRef<Path>,
    llm: Arc<dyn ChatProvider>,
    embedder: Arc<dyn DynEmbeddingProvider>,
    embedding_dim: Option<usize>,
    model: Option<&str>,
) -> Result<Memory> {
    // Initialize bundled provider rates (idempotent — second call is a no-op).
    // Errors are logged but not fatal; cost counters will skip emission with a
    // one-shot warn inside TokenTrackingChatProvider.
    if let Err(e) = crate::core::rates::init_bundled() {
        tracing::warn!(
            error = %e,
            "failed to load bundled provider-rates.toml — cost counters will not be emitted"
        );
    }

    let (graph, temporal_graph) =
        open_graph(path.as_ref(), llm.clone(), embedder.clone(), embedding_dim, None).await?;

    // T6 cycle 2 (ARCH-001 + ARCH-002) — Warm schema caches for Tier 1 paths.
    // Spawned on a background tokio task so Memory construction is not blocked.
    // The concrete model string is forwarded so the provider-native schema arm
    // (NativeSchema / FormatSchema) is reached — not just connection-pool warm.
    // Gated to non-test builds; unit tests must not incur LLM round-trips on
    // every engine construction.  `model` is intentionally unused in the test
    // configuration — consumed here to satisfy the unused-variable lint.
    #[cfg(test)]
    let _ = model;
    #[cfg(not(test))]
    {
        use crate::core::extraction::structured::warm_schema_caches;
        let warmup_llm = Arc::new(ArcChatProvider::new(llm.clone()));
        let model_owned: Option<String> = model.map(str::to_owned);
        tokio::spawn(async move {
            warm_schema_caches(warmup_llm.as_ref(), model_owned.as_deref()).await;
        });
    }

    Ok(Memory {
        graph,
        llm,
        embedder,
        default_sink: None,
        default_namespace: None,
        temporal_graph: Some(temporal_graph),
        episode_content_warn_threshold: Some(10_000),
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

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
use crate::core::config::{PipelineConfig, PipelineConfigBuilder, ResolutionStrategy};
use crate::core::ingest::{
    Engine, EngineNewParams, EngineWithCustomExtractorNoLlmParams, EngineWithExtractorParams,
};
use crate::core::provider::{
    ArcChatProvider, ArcEmbedder, DeterministicEmbeddingProvider, DynEmbeddingProvider,
};
use crate::core::schema::TemporalGraph;
use crate::memory::engine_handle::EngineGraphHandle;
use crate::memory::llm_adapters::AutoagentsEmbedderAdapter;
use crate::memory::{ChatProvider, GraphHandle, MemoryError, Result};

use super::Memory;

// ── Graph factory ─────────────────────────────────────────────────────────────

/// Shared parameters for the three `open_graph*` variants.
///
/// Bundles the path + embedder + sizing knobs that are common to all
/// open-graph call sites, keeping individual function signatures under the
/// `clippy::too_many_arguments` threshold.
pub(crate) struct GraphOpenParams {
    pub path: std::path::PathBuf,
    pub embedder: Arc<dyn DynEmbeddingProvider>,
    pub embedding_dim: Option<usize>,
    pub allowed_entity_types: Vec<String>,
    /// Consumer-supplied model identifier (Option-1, 2026-06-23). Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
}

/// Bundled non-generic parameters for [`open_graph`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). The `path: impl AsRef<Path>`
/// generic stays a lead positional param.
pub(crate) struct OpenGraphParams {
    pub llm: Arc<dyn ChatProvider>,
    pub embedder: Arc<dyn DynEmbeddingProvider>,
    pub embedding_dim: Option<usize>,
    pub allowed_entity_types: Vec<String>,
    /// Consumer-supplied model identifier (Option-1, 2026-06-23). Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
}

/// Bundled non-generic parameters for [`open_engine_handle`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). The `path: impl AsRef<Path>`
/// generic stays a lead positional param.
pub(crate) struct OpenEngineHandleParams {
    pub llm: Arc<dyn ChatProvider>,
    pub embedder: Arc<dyn DynEmbeddingProvider>,
    pub embedding_dim: Option<usize>,
    pub allowed_entity_types: Vec<String>,
    /// Consumer-supplied model identifier (Option-1, 2026-06-23). Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
}

/// Bundled non-generic parameters for [`build_memory_with_model`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments). The
/// `path: impl AsRef<Path>` generic stays a lead positional param.
struct BuildMemoryWithModelParams<'a> {
    llm: Arc<dyn ChatProvider>,
    embedder: Arc<dyn DynEmbeddingProvider>,
    embedding_dim: Option<usize>,
    model: Option<&'a str>,
}

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
    params: OpenGraphParams,
) -> Result<(Arc<dyn GraphHandle>, Arc<TemporalGraph>)> {
    let OpenGraphParams {
        llm,
        embedder,
        embedding_dim,
        allowed_entity_types,
        model,
    } = params;
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

    let mut config_builder =
        resolution_env_overrides(PipelineConfig::builder().embedding_dim(resolved_dim));
    if !allowed_entity_types.is_empty() {
        config_builder = config_builder.allowed_entity_types(allowed_entity_types);
    }
    let config = config_builder.build().map_err(MemoryError::Core)?;

    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::new(ArcChatProvider::new(llm)),
        embedder: Arc::new(ArcEmbedder(embedder)),
        config,
        model,
    });

    let handle: Arc<dyn GraphHandle> = Arc::new(EngineGraphHandle::new(engine));
    Ok((handle, graph_for_facade))
}

/// Open (or create) the libSQL database at `path` and return an `Arc<dyn GraphHandle>`
/// with an **explicit `ExtractorKind`** — used by the `GlinerLlm` and `Custom`+LLM
/// builder paths where the caller has already resolved which extractor to use.
///
/// LLM is still required here (for CascadeResolver + TwoPoolDetector); pass the
/// no-LLM variant via `open_graph_no_llm` when there is no LLM.
pub(crate) async fn open_graph_with_extractor(
    params: GraphOpenParams,
    llm: Arc<dyn ChatProvider>,
    extractor_kind: crate::core::extraction::factory::ExtractorKind<ArcChatProvider>,
) -> Result<(Arc<dyn GraphHandle>, Arc<TemporalGraph>)> {
    let path_str = params
        .path
        .to_str()
        .ok_or_else(|| MemoryError::Other("path contains non-UTF-8 characters".into()))?;

    let resolved_dim = params.embedding_dim.unwrap_or(384);
    let graph = Arc::new(
        TemporalGraph::open_with_dim(path_str, resolved_dim)
            .await
            .map_err(MemoryError::Core)?,
    );
    let graph_for_facade = Arc::clone(&graph);

    let mut config_builder =
        resolution_env_overrides(PipelineConfig::builder().embedding_dim(resolved_dim));
    if !params.allowed_entity_types.is_empty() {
        config_builder = config_builder.allowed_entity_types(params.allowed_entity_types);
    }
    let config = config_builder.build().map_err(MemoryError::Core)?;

    let arc_llm = Arc::new(ArcChatProvider::new(llm));
    let engine = Engine::with_extractor(EngineWithExtractorParams {
        graph,
        llm: arc_llm,
        embedder: Arc::new(ArcEmbedder(params.embedder)),
        config,
        extractor: Arc::new(extractor_kind),
        model: params.model,
    });

    let handle: Arc<dyn GraphHandle> = Arc::new(EngineGraphHandle::new(engine));
    Ok((handle, graph_for_facade))
}

/// Open (or create) the libSQL database at `path` and return an `Arc<dyn GraphHandle>`
/// for the **no-LLM BYOE path** (`Memory::open(…).with_extractor(…).with_embedder(…)`).
///
/// Constructs an `Engine<ArcChatProvider, ArcEmbedder>` using
/// `Engine::with_custom_extractor_no_llm` — the engine's `llm` field is `None`.
/// LLM-dependent pipeline steps (entity resolution via `CascadeResolver`,
/// contradiction detection via `TwoPoolDetector`) return `Error::LlmRequired`
/// if reached.
pub(crate) async fn open_graph_no_llm(
    params: GraphOpenParams,
    custom_extractor: Arc<dyn crate::core::intelligence::EntityExtractorDyn>,
) -> Result<(Arc<dyn GraphHandle>, Arc<TemporalGraph>)> {
    let path_str = params
        .path
        .to_str()
        .ok_or_else(|| MemoryError::Other("path contains non-UTF-8 characters".into()))?;

    let resolved_dim = params.embedding_dim.unwrap_or(384);
    let graph = Arc::new(
        TemporalGraph::open_with_dim(path_str, resolved_dim)
            .await
            .map_err(MemoryError::Core)?,
    );
    let graph_for_facade = Arc::clone(&graph);

    let mut config_builder =
        resolution_env_overrides(PipelineConfig::builder().embedding_dim(resolved_dim));
    if !params.allowed_entity_types.is_empty() {
        config_builder = config_builder.allowed_entity_types(params.allowed_entity_types);
    }
    let config = config_builder.build().map_err(MemoryError::Core)?;

    let extractor_kind = Arc::new(crate::core::extraction::factory::ExtractorKind::Custom(
        custom_extractor,
    ));
    let engine = Engine::with_custom_extractor_no_llm(EngineWithCustomExtractorNoLlmParams {
        graph,
        embedder: Arc::new(ArcEmbedder(params.embedder)),
        config,
        extractor: extractor_kind,
    });

    let handle: Arc<dyn GraphHandle> = Arc::new(EngineGraphHandle::new(engine));
    Ok((handle, graph_for_facade))
}

/// Open (or create) the libSQL database at `path` and return a raw
/// `EngineGraphHandle` (not type-erased) alongside `Arc<TemporalGraph>`.
///
/// Used exclusively by `MemoryBuilder::into_future` when `.with_sink()` is
/// configured — in that case the builder needs `Arc<EngineGraphHandle>` to
/// pass to `BackgroundIngestorGraphHandle::new` (which holds both the
/// `BackgroundIngestor` and the `EngineGraphHandle` delegate).
///
/// Two separate calls to this function open two separate libSQL connections
/// to the same database file.  libSQL WAL mode serialises concurrent writes;
/// safety verified empirically by Spike B + Spike C in
/// `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §6`.
///
/// Per arch spec §3.3 (two-Engine WAL safety note).
pub(crate) async fn open_engine_handle(
    path: impl AsRef<Path>,
    params: OpenEngineHandleParams,
) -> Result<(EngineGraphHandle, Arc<TemporalGraph>)> {
    let OpenEngineHandleParams {
        llm,
        embedder,
        embedding_dim,
        allowed_entity_types,
        model,
    } = params;
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

    let mut config_builder =
        resolution_env_overrides(PipelineConfig::builder().embedding_dim(resolved_dim));
    if !allowed_entity_types.is_empty() {
        config_builder = config_builder.allowed_entity_types(allowed_entity_types);
    }
    let config = config_builder.build().map_err(MemoryError::Core)?;

    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::new(ArcChatProvider::new(llm)),
        embedder: Arc::new(ArcEmbedder(embedder)),
        config,
        model,
    });

    Ok((EngineGraphHandle::new(engine), graph_for_facade))
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
/// Chat model: `gemma4:e4b` (reasoning disabled) · Embedding model: `nomic-embed-text`.
///
/// Default chosen by kremory's own benchmark (2026-06-24, Apple Silicon M4 Max;
/// reproduce via `scripts/model-benchmark/`): `gemma4:e4b` with `think:false`
/// gives the best extraction that fits the inline 30s budget — F1 84% / recall
/// 90% / slowest call ~16s. (With thinking ON the same model is ~44s/call and
/// LOWER quality, F1 75 — kremory extraction is structured-output, not reasoning.)
/// Lighter alternative: `with_ollama_at_model(.., Some("qwen2.5:7b".into()), ..)`
/// — 4.7GB, F1 79, recall 70. Latency figures are M4-Max-only; precision/recall
/// are hardware-independent.
///
/// Requires `ollama pull gemma4:e4b` + `ollama pull nomic-embed-text`, and a
/// running Ollama server (Ollama 0.24+). If no Ollama is available at runtime,
/// the first `.remember()` / `.recall()` call returns a network error.
pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Memory> {
    with_ollama_at("http://localhost:11434", path).await
}

/// ADR-075 P1 (TD-124): apply optional resolution candidate-blocking overrides
/// from env so library DEFAULTS stay pure-P0-safe while benchmarks / advanced
/// consumers can tune throughput. Unset vars leave the builder unchanged.
/// - `KREMORY_RESOLUTION_BLOCK_K` — `usize`, the ANN blocking width.
/// - `KREMORY_RESOLUTION_MIN_COSINE` — `f32` in `[0,1]`, the auto-different floor.
/// - `KREMORY_RESOLUTION_STRATEGY` — `"batched"` (default) | `"pairwise"` (ADR-076):
///   `pairwise` restores the pre-ADR-076 one-LLM-call-per-pair fan-out — used as
///   the A/B baseline + as an instant rollback without a code change.
/// - `KREMORY_RESOLUTION_BATCH_MAX_ENTITIES` — `usize`, the batched-window cap
///   (ADR-076, default 32); raise on a large-context model to shrink call count.
///
/// Reading env here is consistent with this module's existing env-detection
/// (`OLLAMA_HOST` / `OLLAMA_CHAT_MODEL`); the `with_ollama_*` layer is the
/// batteries-included boot surface, not the pure library core.
fn resolution_env_overrides(mut b: PipelineConfigBuilder) -> PipelineConfigBuilder {
    if let Ok(k) = std::env::var("KREMORY_RESOLUTION_BLOCK_K") {
        if let Ok(k) = k.parse::<usize>() {
            b = b.resolution_block_k(k);
        }
    }
    if let Ok(c) = std::env::var("KREMORY_RESOLUTION_MIN_COSINE") {
        if let Ok(c) = c.parse::<f32>() {
            b = b.resolution_min_cosine(c);
        }
    }
    if let Ok(s) = std::env::var("KREMORY_RESOLUTION_STRATEGY") {
        match s.trim().to_ascii_lowercase().as_str() {
            "pairwise" => b = b.resolution_strategy(ResolutionStrategy::Pairwise),
            "batched" => b = b.resolution_strategy(ResolutionStrategy::Batched),
            other => tracing::warn!(
                value = %other,
                "KREMORY_RESOLUTION_STRATEGY must be 'batched' or 'pairwise' — ignoring"
            ),
        }
    }
    if let Ok(n) = std::env::var("KREMORY_RESOLUTION_BATCH_MAX_ENTITIES") {
        if let Ok(n) = n.parse::<usize>() {
            b = b.resolution_batch_max_entities(n);
        }
    }
    if let Ok(n) = std::env::var("KREMORY_EXTRACTION_CONCURRENCY") {
        if let Ok(n) = n.parse::<usize>() {
            b = b.extraction_concurrency(n);
        }
    }
    b
}

/// Open with Ollama at a custom URL and optional custom chat model.
///
/// When `model` is `None`, the default `gemma4:e4b` (reasoning disabled) is used.
/// Pass e.g. `Some("qwen2.5:7b".into())` for a lighter footprint, or any other
/// pulled model. Reasoning is disabled (`think:false`) and `keep_alive` is held
/// at 1h on all models opened this way (see below).
pub async fn with_ollama_at_model(
    url: impl Into<String>,
    model: Option<String>,
    path: impl AsRef<Path>,
) -> Result<Memory> {
    let url: String = url.into();
    let model: String = model.unwrap_or_else(|| "gemma4:e4b".to_string());

    // .think(false): kremory extraction is structured-output, not reasoning.
    // Disabling thinking on capable models (gemma4:e4b, qwen3.5:9b) BOTH raises
    // extraction quality AND keeps per-call latency inside the inline budget
    // (kremory benchmark 2026-06-24, M4 Max: gemma4:e4b 44s/F1 75 thinking-on →
    // 16s/F1 84 think:false). No-op on non-thinking models.
    // .keep_alive("1h"): avoids per-call model-unload thrash on multi-chunk
    // ingest (autoagents-llm defaults keep_alive="0"; TD-024).
    let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&url)
        .model(model.clone())
        .timeout_seconds(120)
        .keep_alive("1h")
        .think(false)
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

    build_memory_with_model(
        path,
        BuildMemoryWithModelParams {
            llm,
            embedder,
            embedding_dim: Some(768),
            model: Some(&model_for_warmup),
        },
    )
    .await
}

/// Open with Ollama at a custom URL.
///
/// Chat model: `gemma4:e4b` (reasoning disabled, `keep_alive=1h`) · Embedding
/// model: `nomic-embed-text` (dim=768). See [`with_ollama`] for the benchmark
/// rationale + the lighter `qwen2.5:7b` alternative.
pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Memory> {
    let url: String = url.into();

    // Default per kremory's own benchmark (2026-06-24, M4 Max; scripts/model-benchmark):
    // gemma4:e4b + think:false is the best extraction model that fits the inline
    // 30s budget (F1 84% / recall 90% / slowest call ~16s). think:false also
    // RAISES quality here (thinking-on: 44s/call, F1 75). keep_alive("1h")
    // avoids per-call unload thrash on multi-chunk ingest (TD-024).
    let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&url)
        .model("gemma4:e4b")
        .timeout_seconds(120)
        .keep_alive("1h")
        .think(false)
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
        "gemma4:e4b",
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<Ollama>::new(embed_provider),
    ));

    // nomic-embed-text outputs 768-dim vectors.
    build_memory_with_model(
        path,
        BuildMemoryWithModelParams {
            llm,
            embedder,
            embedding_dim: Some(768),
            model: Some("gemma4:e4b"),
        },
    )
    .await
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
        .timeout_seconds(120)
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
    build_memory_with_model(
        path,
        BuildMemoryWithModelParams {
            llm,
            embedder,
            embedding_dim: Some(1536),
            model: Some("gpt-4o-mini"),
        },
    )
    .await
}

/// Bundled parameters for [`with_openai_compatible_chat_ollama_embed`] —
/// args-as-object per TD-042.
pub struct OpenAiCompatibleParams<'a> {
    /// Chat endpoint base URL, e.g. `https://api.groq.com/openai/v1` (Groq),
    /// `https://api.together.xyz/v1` (Together), or a local vLLM `…/v1`.
    pub chat_base_url: &'a str,
    /// API key for the chat provider (e.g. the Groq key).
    pub api_key: &'a str,
    /// Chat model id as the provider names it, e.g. `openai/gpt-oss-120b` on Groq.
    pub chat_model: &'a str,
    /// Ollama base URL for the local embedder (`nomic-embed-text`, dim 768).
    pub ollama_url: &'a str,
    /// libSQL database path.
    pub path: &'a str,
}

/// Open with an **OpenAI-API-compatible** chat provider (Groq, Together,
/// Fireworks, vLLM, …) for extraction + a **local Ollama embedder**.
///
/// These providers are chat-only (Groq has no embeddings endpoint), so this
/// pairs a fast/cheap cloud extractor — e.g. `gpt-oss` on Groq — with a local
/// `nomic-embed-text` embedder (dim 768). This is the config that makes
/// benchmarking practical: cloud extraction is latency-optimised (~0.1–0.3s/call
/// vs ~1.5–2s local), and quality matches the gpt-4o-mini extraction the LoCoMo/
/// LongMemEval peers used, while embeddings stay local + free.
pub async fn with_openai_compatible_chat_ollama_embed(
    params: OpenAiCompatibleParams<'_>,
) -> Result<Memory> {
    let OpenAiCompatibleParams {
        chat_base_url,
        api_key,
        chat_model,
        ollama_url,
        path,
    } = params;

    let chat_provider: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .base_url(chat_base_url)
        .api_key(api_key)
        .model(chat_model)
        .timeout_seconds(120)
        .build()
        .map_err(|e| {
            MemoryError::Other(format!("OpenAI-compatible chat provider error: {e}"))
        })?;

    let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(ollama_url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama embedding provider error: {e}")))?;

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        "openai",
        Box::leak(chat_model.to_string().into_boxed_str()),
    ));
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(AutoagentsEmbedderAdapter::new(
        EmbedderArc::<Ollama>::new(embed_provider),
    ));

    build_memory_with_model(
        path,
        BuildMemoryWithModelParams {
            llm,
            embedder,
            embedding_dim: Some(768),
            model: Some(chat_model),
        },
    )
    .await
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
    //
    // Default: claude-haiku-4-5-20251001 (current Claude 4.5 Haiku as of 2026-06-10).
    // Was previously incorrectly defaulted to "claude-3-haiku-20240307" with a comment
    // claiming it was "the API-level name for claude-haiku-4-5" — that was wrong;
    // claude-3-haiku-20240307 is the deprecated March-2024 model and now returns 404.
    // Discovered via OOB cloud-model smoke (crates/kremory/tests/oob_anthropic_smoke.rs)
    // 2026-06-10 during v0.1.2 Phase D resolution work.
    //
    // capability_of() in provider.rs recognizes "claude-haiku-4-5*" prefix and routes
    // to NativeStructuredOutput, so this default is wire-compatible with kremory's
    // structured-output ladder.
    let chat_provider: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(&key)
        .model("claude-haiku-4-5-20251001")
        .timeout_seconds(120)
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
        BuildMemoryWithModelParams {
            llm,
            embedder,
            embedding_dim: Some(384),
            model: Some("claude-3-haiku-20240307"),
        },
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
    params: BuildMemoryWithModelParams<'_>,
) -> Result<Memory> {
    let BuildMemoryWithModelParams {
        llm,
        embedder,
        embedding_dim,
        model,
    } = params;
    // Initialize bundled provider rates (idempotent — second call is a no-op).
    // Errors are logged but not fatal; cost counters will skip emission with a
    // one-shot warn inside TokenTrackingChatProvider.
    if let Err(e) = crate::core::rates::init_bundled() {
        tracing::warn!(
            error = %e,
            "failed to load bundled provider-rates.toml — cost counters will not be emitted"
        );
    }

    let (graph, temporal_graph) = open_graph(
        path.as_ref(),
        OpenGraphParams {
            llm: llm.clone(),
            embedder: embedder.clone(),
            embedding_dim,
            allowed_entity_types: vec![],
            // Tier-1 shortcuts know the concrete model string — thread it so
            // capability detection reaches the provider-native schema arm
            // (Option-1, 2026-06-23).
            model: model.map(str::to_owned),
        },
    )
    .await?;

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
        llm: Some(llm),
        // Tier-1 shortcuts are one-provider convenience constructors; a separate
        // dream model is a two-provider configuration only the `MemoryBuilder`
        // chain exposes. `None` → dream falls back to the main provider,
        // byte-for-byte unchanged behaviour (TD-052b §2.3a).
        dream_llm: None,
        // TD-094: Tier-1 shortcuts know the concrete model string, so thread it
        // to the dream passes for capability detection. `dream_model_id` stays
        // `None` — a dedicated dream model is a two-provider `MemoryBuilder`-only
        // configuration; here dream falls back to this `model_id`.
        model_id: model.map(str::to_owned),
        dream_model_id: None,
        embedder,
        default_sink: None,
        default_namespace: None,
        temporal_graph: Some(temporal_graph),
        episode_content_warn_threshold: Some(10_000),
        dream_scheduler: std::sync::Arc::new(std::sync::Mutex::new(None)),
        // Tier 1 shortcuts default to fire-and-forget (ADR-051 design intent).
        await_extraction: false,
        await_extraction_timeout: std::time::Duration::from_secs(60),
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

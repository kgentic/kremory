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
use crate::core::config::{
    PipelineConfig, PipelineConfigBuilder, PipelineConfigOverrides, ResolutionStrategy,
};
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
    /// Consumer-supplied model identifier. Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
    /// Explicit per-knob `SearchConfig` overrides from
    /// `MemoryBuilder::with_content_stream_weight` /
    /// `with_rrf_k` / `with_episode_dense_enabled`. Applied AFTER
    /// `search_env_overrides` at every construction site — explicit
    /// programmatic config wins over env, which wins over default.
    pub overrides: PipelineConfigOverrides,
}

/// Bundled non-generic parameters for [`open_graph`] — args-as-object to
/// stay under the `clippy::too_many_arguments` threshold. The `path: impl AsRef<Path>`
/// generic stays a lead positional param.
pub(crate) struct OpenGraphParams {
    pub llm: Arc<dyn ChatProvider>,
    pub embedder: Arc<dyn DynEmbeddingProvider>,
    pub embedding_dim: Option<usize>,
    pub allowed_entity_types: Vec<String>,
    /// Consumer-supplied model identifier. Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
    /// See [`GraphOpenParams::search`] doc — same precedence contract.
    pub overrides: PipelineConfigOverrides,
}

/// Bundled non-generic parameters for [`open_engine_handle`] — args-as-object
/// to stay under the `clippy::too_many_arguments` threshold. The `path: impl AsRef<Path>`
/// generic stays a lead positional param.
pub(crate) struct OpenEngineHandleParams {
    pub llm: Arc<dyn ChatProvider>,
    pub embedder: Arc<dyn DynEmbeddingProvider>,
    pub embedding_dim: Option<usize>,
    pub allowed_entity_types: Vec<String>,
    /// Consumer-supplied model identifier. Threaded to
    /// `Engine`; `None` → capability detection falls to `PromptOnly`.
    pub model: Option<String>,
    /// See [`GraphOpenParams::search`] doc — same precedence contract.
    pub overrides: PipelineConfigOverrides,
}

/// Bundled non-generic parameters for [`build_memory_with_model`] —
/// args-as-object to stay under the `clippy::too_many_arguments` threshold. The
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
        overrides,
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

    // Precedence: explicit programmatic `search` overrides are applied
    // LAST, after the env overrides, so `.with_content_stream_weight(...)`
    // etc. win over `KREMORY_CONTENT_WEIGHT` etc. Unset (default) overrides
    // are a no-op — env-only behaviour is unchanged.
    let mut config_builder = overrides.apply(search_env_overrides(resolution_env_overrides(
        PipelineConfig::builder().embedding_dim(resolved_dim),
    )));
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

    // Precedence: explicit programmatic overrides win over env — see
    // `open_graph`'s comment for the full rationale.
    let mut config_builder =
        params
            .overrides
            .apply(search_env_overrides(resolution_env_overrides(
                PipelineConfig::builder().embedding_dim(resolved_dim),
            )));
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

    // Precedence: explicit programmatic overrides win over env — see
    // `open_graph`'s comment for the full rationale.
    let mut config_builder =
        params
            .overrides
            .apply(search_env_overrides(resolution_env_overrides(
                PipelineConfig::builder().embedding_dim(resolved_dim),
            )));
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
/// to the same database file. libSQL WAL mode serialises concurrent writes;
/// this has been verified empirically (two-Engine WAL safety).
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
        overrides,
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

    // Precedence: explicit programmatic overrides win over env — see
    // `open_graph`'s comment for the full rationale.
    let mut config_builder = overrides.apply(search_env_overrides(resolution_env_overrides(
        PipelineConfig::builder().embedding_dim(resolved_dim),
    )));
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
/// Default chosen by kremory's own benchmark (Apple Silicon M4 Max;
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

/// Apply optional resolution candidate-blocking overrides from env so
/// library DEFAULTS stay pure-P0-safe while benchmarks / advanced
/// consumers can tune throughput. Unset vars leave the builder unchanged.
/// - `KREMORY_RESOLUTION_BLOCK_K` — `usize`, the ANN blocking width.
/// - `KREMORY_RESOLUTION_MIN_COSINE` — `f32` in `[0,1]`, the auto-different floor.
/// - `KREMORY_RESOLUTION_STRATEGY` — `"batched"` (default) | `"pairwise"`:
///   `pairwise` restores the one-LLM-call-per-pair fan-out — used as
///   the A/B baseline + as an instant rollback without a code change.
/// - `KREMORY_RESOLUTION_BATCH_MAX_ENTITIES` — `usize`, the batched-window cap
///   (default 32); raise on a large-context model to shrink call count.
///
/// Reading env here is consistent with this module's existing env-detection
/// (`OLLAMA_HOST` / `OLLAMA_CHAT_MODEL`); the `with_ollama_*` layer is the
/// batteries-included boot surface, not the pure library core.
fn resolution_env_overrides(mut b: PipelineConfigBuilder) -> PipelineConfigBuilder {
    // Override contradiction detection, which is default-ON since the
    // coexistence-prompt fix (7/8 set-valued destroyed -> 0/8). Accepts
    // 1/true/yes/on and 0/false/no/off.
    //
    // An unrecognised value WARNS and leaves the DEFAULT in place. The warning
    // must not name a direction: this knob's default has already moved twice in
    // one day, and a message asserting "staying OFF" was wrong within hours of
    // being written — a safety message that lies is worse than none.
    if let Ok(v) = std::env::var("KREMORY_CONTRADICTION_DETECTION") {
        match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => b = b.contradiction_detection_enabled(true),
            "0" | "false" | "no" | "off" => {
                tracing::warn!(
                    "KREMORY_CONTRADICTION_DETECTION disabled — facts will be APPENDED \
                     only; genuine supersession (works_at Acme -> works_at Globex) will \
                     no longer be detected."
                );
                b = b.contradiction_detection_enabled(false);
            }
            other => tracing::warn!(
                value = %other,
                "KREMORY_CONTRADICTION_DETECTION must be a boolean — ignoring it and \
                 keeping the compiled-in default (see PipelineConfig::\
                 contradiction_detection_enabled)"
            ),
        }
    }
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

/// Apply the server-boot
/// search-fusion SWEEP knobs from env to the pipeline config builder, so a
/// weight/`k` sweep costs a **server restart, not a rebuild**. Composed onto the
/// SAME builder chain as [`resolution_env_overrides`] at every `open_graph`
/// construction site, so the resulting `SearchConfig` reaches BOTH live library
/// fusion sweep sites: the entity-graph RRF (`context.rs`, reads
/// `config.search.rrf_k`) and the content-fusion (`search::rrf_fuse_with_content`
/// via the `GraphHandle::search_config()` accessor).
///
/// - `KREMORY_CONTENT_WEIGHT` (f32)  → `SearchConfig::content_stream_weight`
/// - `KREMORY_RRF_K` (usize)         → `SearchConfig::rrf_k`
/// - `KREMORY_PROXIMITY_WEIGHT` (f32) → `SearchConfig::proximity_weight`
/// - `KREMORY_RERANK_CANDIDATE_MAX_CHARS` (usize) → `SearchConfig::rerank_candidate_max_chars` (reranker latency lever 1)
///
/// Absent env → defaults preserved (byte-identical). **Fail-loud**: a
/// malformed value is WARN-logged + ignored — never silently accepted as
/// garbage. Applied values are INFO-logged at the apply site.
/// `KREMORY_ENTITY_STREAM_WEIGHT` is deliberately DEFERRED — its only
/// consumer does not exist yet — no dead config ships now.
fn search_env_overrides(mut b: PipelineConfigBuilder) -> PipelineConfigBuilder {
    if let Ok(raw) = std::env::var("KREMORY_CONTENT_WEIGHT") {
        match raw.trim().parse::<f32>() {
            Ok(v) => {
                tracing::info!(
                    content_stream_weight = v,
                    "KREMORY_CONTENT_WEIGHT override applied to SearchConfig"
                );
                b = b.content_stream_weight(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_CONTENT_WEIGHT is not a valid f32 — ignoring (default 1.0 retained)"
            ),
        }
    }
    if let Ok(raw) = std::env::var("KREMORY_RRF_K") {
        match raw.trim().parse::<usize>() {
            Ok(v) => {
                tracing::info!(rrf_k = v, "KREMORY_RRF_K override applied to SearchConfig");
                b = b.rrf_k(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_RRF_K is not a valid usize — ignoring (default 60 retained)"
            ),
        }
    }
    // Dense-episode A/B knob. Truthy = `1`/`true`/`yes`/`on`
    // (case-insensitive); any other value is treated as OFF and WARN-logged so
    // a typo (`KREMORY_EPISODE_DENSE=ture`) never silently enables/disables the
    // arm. Absent env → default `false` (byte-identical BM25-only). Fail-loud
    // — mirrors the parse-and-warn discipline of the two knobs above.
    if let Ok(raw) = std::env::var("KREMORY_EPISODE_DENSE") {
        let trimmed = raw.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => {
                tracing::info!(
                    episode_dense_enabled = true,
                    "KREMORY_EPISODE_DENSE override applied to SearchConfig (dense episode arm ON)"
                );
                b = b.episode_dense_enabled(true);
            }
            "0" | "false" | "no" | "off" | "" => {
                tracing::info!(
                    episode_dense_enabled = false,
                    "KREMORY_EPISODE_DENSE=off — dense episode arm OFF (BM25-only, default)"
                );
                b = b.episode_dense_enabled(false);
            }
            other => tracing::warn!(
                value = %other,
                "KREMORY_EPISODE_DENSE is not a recognised boolean \
                 (1/true/yes/on | 0/false/no/off) — ignoring (default OFF retained)"
            ),
        }
    }
    // Dense-fact A/B knob. Same truthy/parse/warn discipline
    // as KREMORY_EPISODE_DENSE immediately above — mirrors it exactly, this is
    // a SIBLING knob (its own SearchConfig field), not a re-read of the same one.
    if let Ok(raw) = std::env::var("KREMORY_FACT_DENSE") {
        let trimmed = raw.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => {
                tracing::info!(
                    fact_dense_enabled = true,
                    "KREMORY_FACT_DENSE override applied to SearchConfig (dense fact arm ON)"
                );
                b = b.fact_dense_enabled(true);
            }
            "0" | "false" | "no" | "off" | "" => {
                tracing::info!(
                    fact_dense_enabled = false,
                    "KREMORY_FACT_DENSE=off — dense fact arm OFF (1-hop-only, default)"
                );
                b = b.fact_dense_enabled(false);
            }
            other => tracing::warn!(
                value = %other,
                "KREMORY_FACT_DENSE is not a recognised boolean \
                 (1/true/yes/on | 0/false/no/off) — ignoring (default OFF retained)"
            ),
        }
    }
    // Nomic task-prefix A/B knob. Same truthy/parse/warn discipline as
    // KREMORY_EPISODE_DENSE / KREMORY_FACT_DENSE above. See
    // `core::config::SearchConfig::embed_task_prefix_enabled` for the full
    // correctness note (flipping this on an existing corpus requires a
    // re-embed — the vectors occupy a different task space once prefixed).
    if let Ok(raw) = std::env::var("KREMORY_EMBED_TASK_PREFIX") {
        let trimmed = raw.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => {
                tracing::info!(
                    embed_task_prefix_enabled = true,
                    "KREMORY_EMBED_TASK_PREFIX override applied to SearchConfig \
                     (nomic search_document:/search_query: task prefixing ON)"
                );
                b = b.embed_task_prefix_enabled(true);
            }
            "0" | "false" | "no" | "off" | "" => {
                tracing::info!(
                    embed_task_prefix_enabled = false,
                    "KREMORY_EMBED_TASK_PREFIX=off — nomic task prefixing OFF (bare-text \
                     embedding, default)"
                );
                b = b.embed_task_prefix_enabled(false);
            }
            other => tracing::warn!(
                value = %other,
                "KREMORY_EMBED_TASK_PREFIX is not a recognised boolean \
                 (1/true/yes/on | 0/false/no/off) — ignoring (default OFF retained)"
            ),
        }
    }
    // Axis-C A/B knob. Same parse/fail-loud
    // discipline as KREMORY_CONTENT_WEIGHT above (a float weight, not a
    // boolean). Absent/malformed → default 0.0 (axis OFF) retained.
    if let Ok(raw) = std::env::var("KREMORY_PROXIMITY_WEIGHT") {
        match raw.trim().parse::<f32>() {
            Ok(v) => {
                tracing::info!(
                    proximity_weight = v,
                    "KREMORY_PROXIMITY_WEIGHT override applied to SearchConfig"
                );
                b = b.proximity_weight(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_PROXIMITY_WEIGHT is not a valid f32 — ignoring (default 0.0 retained)"
            ),
        }
    }
    // The temporal-recency axis had working compute
    // and NO way to enable it — no builder method, no env override, and
    // `SearchConfig` derives no `Deserialize`, so no config-file path either.
    // Its `0.0` default was therefore unreachable-by-construction and the axis
    // has never been measured. Same parse/fail-loud discipline as
    // KREMORY_PROXIMITY_WEIGHT directly above; default 0.0 (axis OFF) retained
    // on absent/malformed, so this is byte-identical unless explicitly set.
    if let Ok(raw) = std::env::var("KREMORY_TEMPORAL_WEIGHT") {
        match raw.trim().parse::<f32>() {
            Ok(v) => {
                tracing::info!(
                    temporal_weight = v,
                    "KREMORY_TEMPORAL_WEIGHT override applied to SearchConfig"
                );
                b = b.temporal_weight(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_TEMPORAL_WEIGHT is not a valid f32 — ignoring (default 0.0 retained)"
            ),
        }
    }
    // Axis-C hop bound, sweepable for the SAME reason the weight is. After
    // the first axis-C A/B: at the default
    // `proximity_hop_bound = 2` the per-recall trace showed
    // `seed_count=50 boosted_count=48` — the walk reaches neighbours for ~96%
    // of seeds, so the boost is a near-uniform additive offset and cannot
    // discriminate at ANY weight (measured flat across w=0.05/0.15/0.40). The
    // hop bound, not the weight, is the parameter that controls selectivity —
    // and it was the one knob NOT sweepable without a rebuild: a knob you
    // cannot set is a knob you cannot evaluate, recurring inside axis-C's own
    // tuning surface.
    if let Ok(raw) = std::env::var("KREMORY_PROXIMITY_HOP_BOUND") {
        match raw.trim().parse::<u32>() {
            Ok(v) => {
                tracing::info!(
                    proximity_hop_bound = v,
                    "KREMORY_PROXIMITY_HOP_BOUND override applied to SearchConfig"
                );
                b = b.proximity_hop_bound(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_PROXIMITY_HOP_BOUND is not a valid u32 — ignoring (default retained)"
            ),
        }
    }
    // Reranker latency lever 1 — cap the SUMMARY portion of each rerank
    // candidate's text (`SearchConfig::rerank_candidate_max_chars`). Absent
    // env / `0` → unlimited (default, byte-identical). Same parse/fail-loud
    // discipline as KREMORY_RRF_K above (a usize, not a boolean).
    if let Ok(raw) = std::env::var("KREMORY_RERANK_CANDIDATE_MAX_CHARS") {
        match raw.trim().parse::<usize>() {
            Ok(v) => {
                tracing::info!(
                    rerank_candidate_max_chars = v,
                    "KREMORY_RERANK_CANDIDATE_MAX_CHARS override applied to SearchConfig"
                );
                b = b.rerank_candidate_max_chars(v);
            }
            Err(e) => tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_RERANK_CANDIDATE_MAX_CHARS is not a valid usize — ignoring \
                 (default 0/unlimited retained)"
            ),
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
    // (kremory benchmark, M4 Max: gemma4:e4b 44s/F1 75 thinking-on →
    // 16s/F1 84 think:false). No-op on non-thinking models.
    // .keep_alive("1h"): avoids per-call model-unload thrash on multi-chunk
    // ingest (autoagents-llm defaults keep_alive="0").
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

    // Default per kremory's own benchmark (M4 Max; scripts/model-benchmark):
    // gemma4:e4b + think:false is the best extraction model that fits the inline
    // 30s budget (F1 84% / recall 90% / slowest call ~16s). think:false also
    // RAISES quality here (thinking-on: 44s/call, F1 75). keep_alive("1h")
    // avoids per-call unload thrash on multi-chunk ingest.
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
/// args-as-object to stay under the `clippy::too_many_arguments` threshold.
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
/// Derive the token/cost `provider` label from an OpenAI-compatible chat base
/// URL (looked up against `kremory_core_tokens_total{provider}`). The
/// factory below is generic over ANY OpenAI-compatible endpoint, so the label
/// must reflect the ACTUAL provider — a hardcoded `"openai"` mis-attributes
/// Groq/Together spend and misses their pricing rows in provider-rates.toml.
/// Match the value against the keys in `crates/kremory/monitoring/provider-rates.toml`.
fn openai_compatible_provider_label(base_url: &str) -> &'static str {
    let u = base_url.to_ascii_lowercase();
    if u.contains("groq") {
        "groq"
    } else if u.contains("together") {
        "together"
    } else if u.contains("anthropic") {
        "anthropic"
    } else if u.contains("openai") {
        "openai"
    } else {
        "openai_compatible"
    }
}

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
        .map_err(|e| MemoryError::Other(format!("OpenAI-compatible chat provider error: {e}")))?;

    let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(ollama_url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| MemoryError::Other(format!("Ollama embedding provider error: {e}")))?;

    let arc_llm: Arc<dyn ChatProvider> = chat_provider;
    let llm: Arc<dyn ChatProvider> = Arc::new(TokenTrackingChatProvider::new(
        ArcChatProvider::new(arc_llm),
        // Derive the provider label from the base URL — this factory is GENERIC
        // over any OpenAI-compatible endpoint (Groq / Together / …), so a
        // hardcoded "openai" mis-attributes cost + misses the provider's pricing
        // rows in provider-rates.toml (verified: Groq runs were tagged "openai").
        openai_compatible_provider_label(chat_base_url),
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
/// entry.
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
    // Default: claude-haiku-4-5-20251001 (the current Claude 4.5 Haiku model).
    // Was previously incorrectly defaulted to "claude-3-haiku-20240307" with a comment
    // claiming it was "the API-level name for claude-haiku-4-5" — that was wrong;
    // claude-3-haiku-20240307 is the deprecated March-2024 model and now returns 404.
    // Discovered via OOB cloud-model smoke (crates/kremory/tests/oob_anthropic_smoke.rs).
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
            // capability detection reaches the provider-native schema arm.
            model: model.map(str::to_owned),
            // Tier-1 shortcuts (`with_ollama`, `with_openai`, ...) do not go
            // through `MemoryBuilder`, so there is no programmatic override to
            // thread — env (`KREMORY_CONTENT_WEIGHT` etc.) remains the only
            // tuning path for these convenience constructors.
            overrides: PipelineConfigOverrides::default(),
        },
    )
    .await?;

    // Warm schema caches for Tier 1 paths.
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
        // byte-for-byte unchanged behaviour.
        dream_llm: None,
        // Tier-1 shortcuts know the concrete model string, so thread it
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
        // Tier 1 shortcuts default to fire-and-forget (by design).
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

#[cfg(test)]
mod search_env_override_tests {
    use super::*;

    /// The boot env
    /// override helper applies `KREMORY_CONTENT_WEIGHT`/`KREMORY_RRF_K` to the
    /// resulting `SearchConfig`, preserves defaults when absent, and fail-loud
    /// IGNORES a malformed value (never panics, never silently accepts garbage).
    ///
    /// Sequenced within ONE test (remove → assert default → set → assert applied
    /// → set-garbage → assert default retained → remove) so it is deterministic
    /// regardless of runner — env is process-global, and nextest additionally
    /// isolates each test in its own process (kremory's runner).
    #[test]
    fn search_env_overrides_apply_default_and_failloud() {
        // Absent → byte-identical defaults.
        std::env::remove_var("KREMORY_CONTENT_WEIGHT");
        std::env::remove_var("KREMORY_RRF_K");
        let default_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("default config builds");
        assert_eq!(default_cfg.search.content_stream_weight, 1.0);
        assert_eq!(default_cfg.search.rrf_k, 1);

        // Present + valid → applied.
        std::env::set_var("KREMORY_CONTENT_WEIGHT", "2.0");
        std::env::set_var("KREMORY_RRF_K", "1");
        let over_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("override config builds");
        assert_eq!(
            over_cfg.search.content_stream_weight, 2.0,
            "KREMORY_CONTENT_WEIGHT=2.0 must reach SearchConfig.content_stream_weight"
        );
        assert_eq!(
            over_cfg.search.rrf_k, 1,
            "KREMORY_RRF_K=1 must reach SearchConfig.rrf_k"
        );

        // Malformed → fail-loud ignore (default retained), never panics.
        std::env::set_var("KREMORY_CONTENT_WEIGHT", "not-a-float");
        std::env::set_var("KREMORY_RRF_K", "-5");
        let bad_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("garbage-env config still builds");
        assert_eq!(
            bad_cfg.search.content_stream_weight, 1.0,
            "garbage KREMORY_CONTENT_WEIGHT must be ignored, default 1.0 retained"
        );
        assert_eq!(
            bad_cfg.search.rrf_k, 1,
            "garbage KREMORY_RRF_K must be ignored, default 1 retained"
        );

        std::env::remove_var("KREMORY_CONTENT_WEIGHT");
        std::env::remove_var("KREMORY_RRF_K");
    }

    /// Reranker latency lever 1 — same default/apply/fail-loud sequencing as
    /// `search_env_overrides_apply_default_and_failloud` above, for
    /// `KREMORY_RERANK_CANDIDATE_MAX_CHARS`.
    #[test]
    fn search_env_overrides_rerank_candidate_max_chars() {
        std::env::remove_var("KREMORY_RERANK_CANDIDATE_MAX_CHARS");
        let default_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("default config builds");
        assert_eq!(default_cfg.search.rerank_candidate_max_chars, 0);

        std::env::set_var("KREMORY_RERANK_CANDIDATE_MAX_CHARS", "512");
        let over_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("override config builds");
        assert_eq!(
            over_cfg.search.rerank_candidate_max_chars, 512,
            "KREMORY_RERANK_CANDIDATE_MAX_CHARS=512 must reach \
             SearchConfig.rerank_candidate_max_chars"
        );

        // Malformed → fail-loud ignore (default retained), never panics.
        std::env::set_var("KREMORY_RERANK_CANDIDATE_MAX_CHARS", "not-a-usize");
        let bad_cfg = search_env_overrides(PipelineConfig::builder())
            .build()
            .expect("garbage-env config still builds");
        assert_eq!(
            bad_cfg.search.rerank_candidate_max_chars, 0,
            "garbage KREMORY_RERANK_CANDIDATE_MAX_CHARS must be ignored, default 0 retained"
        );

        std::env::remove_var("KREMORY_RERANK_CANDIDATE_MAX_CHARS");
    }
}

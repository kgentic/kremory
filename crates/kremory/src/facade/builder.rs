//! `MemoryBuilder` — type-state builder for `Memory`.
//!
//! Extracted from `facade/mod.rs` (TD-015 LoC reduction).  All types, impls,
//! and `IntoFuture` implementations are verbatim moves — no logic changes.

use std::future::IntoFuture;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use super::*;

use crate::core::background::{BackgroundIngestor, IngestorConfig};
use crate::core::chat_tracking::TokenTrackingChatProvider;
use crate::core::config::PipelineConfigOverrides;
use crate::core::error::Error as CoreError;
use crate::core::provider::DynEmbeddingProvider;
use crate::memory::{
    events::EnrichmentEventSink, BackgroundIngestorGraphHandle, ChatProvider, GraphHandle,
    MemoryError, Result,
};

// ── MemoryBuilder ─────────────────────────────────────────────────────────────

/// Type-state builder for `Memory`. Compile-time enforced: `.with_llm()` then
/// `.with_embedder()` are both required before `.await`.
///
/// Optional: `.with_event_sink()`, `.default_namespace()`, `.embedding_dim()`.
#[must_use = "MemoryBuilder must be configured with .with_llm() AND .with_embedder() before .await"]
pub struct MemoryBuilder<L, E> {
    path: std::path::PathBuf,
    llm: Option<Arc<dyn ChatProvider>>,
    /// Consumer-supplied model identifier (Option-1, 2026-06-23). Set by Tier-1
    /// shortcuts, `with_llm_tracked` (from the metric label), and the explicit
    /// `.with_model_id(…)` escape hatch. Left `None` by raw `with_llm` →
    /// capability detection falls to `PromptOnly`. Threaded unchanged across
    /// every type-state transition; does NOT change type-state.
    model_id: Option<String>,
    /// Optional dedicated dream-phase LLM (TD-052b). `None` (default) → dream
    /// re-uses the `with_llm` provider. Threaded unchanged across every
    /// type-state transition. Does NOT change type-state.
    dream_llm: Option<Arc<dyn ChatProvider>>,
    /// Optional dedicated dream-phase model id (TD-094). Pairs with `dream_llm`
    /// as `model_id` pairs with `llm`: a dedicated dream provider usually reports
    /// a different model string, and capability detection needs the right one.
    /// `None` (default) → dream falls back to `model_id`. Threaded unchanged
    /// across every type-state transition. Does NOT change type-state.
    dream_model_id: Option<String>,
    embedder: Option<Arc<dyn DynEmbeddingProvider>>,
    default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    default_namespace: Option<Namespace>,
    embedding_dim: Option<usize>,
    /// Optional path to a custom `provider-rates.toml`. When `Some`, overrides
    /// the bundled rates file at build time.
    provider_rates_path: Option<PathBuf>,
    /// Soft warning threshold for episode content length (chars).
    /// Default `Some(10_000)` via `Memory::open` — matches Zep's recommended
    /// chunk size. `None` disables the warning. Never enforced as a hard limit;
    /// observability only.
    episode_content_warn_threshold: Option<usize>,
    /// Custom extractor supplied via `.with_extractor(Arc<impl EntityExtractor>)`.
    /// When `Some`, overrides LLM-derived extraction. Mutually exclusive with
    /// `.with_gliner()` — builder errors at `build()` if both are set.
    custom_extractor: Option<Arc<dyn crate::core::intelligence::EntityExtractorDyn>>,
    /// GLiNER enabled via `.with_gliner()`. Requires the `ner` cargo feature.
    /// When set without `.with_llm()`, builder errors at `build()` because
    /// GLiNER candidate-gen still needs one LLM typing call.
    #[cfg(feature = "ner")]
    use_gliner: bool,
    /// Entity type names the extractor is allowed to emit. Forwarded to
    /// `PipelineConfig::allowed_entity_types`. When empty (the default), the
    /// `GlinerExtractor` rejects all entities — callers that activate the `ner`
    /// feature MUST supply this via [`MemoryBuilder::allowed_entity_types`].
    allowed_entity_types: Vec<String>,
    /// TD-141 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-141): explicit
    /// per-knob `SearchConfig` overrides set via `.with_content_stream_weight()`
    /// / `.with_rrf_k()` / `.with_episode_dense_enabled()` / TD-139's
    /// `.with_fact_dense_enabled()`. All-`None` by
    /// default — env-only behaviour is unchanged for consumers who never call
    /// these setters. Threaded unchanged across every type-state transition;
    /// does NOT change type-state.
    config_overrides: PipelineConfigOverrides,
    /// Seed instruction for the DEFAULT namespace's entity-type registry,
    /// applied at `.await` (build time) via the same catch-all + apply path as
    /// `Memory::register_namespace_with_seed` (spec §5.2.3). Default
    /// `NamespaceSeed::Default`. Builder-only — consumed at build; NOT stored on
    /// `Memory`.
    seed_registry: crate::core::entity_types::NamespaceSeed,
    /// Automatic dream-pass scheduling policy.
    /// Default: `DreamSchedule::Off` (no background task).
    dream_schedule: crate::memory::scheduler::DreamSchedule,
    /// When `true`, `Memory::remember(...).await` blocks until the background
    /// extraction pipeline transitions the episode to `Verified` (or returns
    /// `Err` on `Failed` / timeout). Default: `false`.
    /// Per spec §Phase 4 / ADR-051 D1 peer pattern (Cognee `run_in_background=False`).
    await_extraction: bool,
    /// Timeout applied when `await_extraction = true`.
    /// Default: 60 s (spec §Risk R-12 mitigation).
    await_extraction_timeout: Duration,
    _llm_state: std::marker::PhantomData<L>,
    _emb_state: std::marker::PhantomData<E>,
}

impl<L, E> MemoryBuilder<L, E> {
    /// Set the default event sink for all subsequent operations.
    /// Per-call sinks (via `.with_event_sink()` on request builders) override this.
    ///
    /// # Sink callback contract (G7 — v0.1.6)
    ///
    /// Sink methods are called **sync inline** on the Phase 2 enrichment
    /// pipeline thread. Slow callbacks stall ingest. See the
    /// [`EnrichmentEventSink`] trait rustdoc for the full contract +
    /// recommended consumer pattern (buffer + return immediately + drain
    /// async).
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.default_sink = Some(sink);
        self
    }

    /// Ergonomic alias for [`with_event_sink`](Self::with_event_sink) that
    /// accepts any concrete type implementing [`EnrichmentEventSink`] and wraps
    /// it in `Arc` internally.
    ///
    /// Equivalent to `.with_event_sink(Arc::new(sink))`.  Prefer this form when
    /// the caller does not need to share the `Arc` with other owners.
    ///
    /// # Sink callback contract (ADR-052 D4)
    ///
    /// All callbacks fire **sync-inline** on the background worker OS thread.
    /// Keep callbacks fast (sub-millisecond ideal, sub-100 ms absolute ceiling).
    ///
    /// Refs: ADR-052 Gap 1; impl spec §3 Phase 2 `Memory::with_sink` DoD item.
    #[deprecated(
        since = "0.2.5",
        note = "use `.with_event_sink(Arc::new(sink))` — one stem for sink wiring \
                per F7 / spec memory-builder-dx-hardening-2026-06-20. Removed in a later breaking release."
    )]
    pub fn with_sink(mut self, sink: impl EnrichmentEventSink + Send + Sync + 'static) -> Self {
        self.default_sink = Some(Arc::new(sink));
        self
    }

    /// Override the bundled `provider-rates.toml` with a custom cost-rates file.
    ///
    /// The rates table maps `(provider, model)` to token prices; without a match,
    /// `kremory_core_cost_usd_total` stays zero. Use this when running a model the
    /// bundled table does not price (a self-hosted or newly-released one), or to
    /// pin negotiated rates.
    ///
    /// Load failures are logged and do **not** fail the build — cost counters go
    /// quiet, the rest of the system runs. Pairs with
    /// [`with_token_tracking`](MemoryBuilder::with_token_tracking), which emits the
    /// counters this table prices.
    ///
    /// DOC-1 (V1-CANONICAL §4.2): this setter did not exist. The field, the
    /// threading through every builder state transition, and the
    /// `rates::init_from_path` call were all already implemented — only the setter
    /// was missing, so the capability was unreachable while **six** doc sites
    /// documented it as working (README, crate README, `docs/api.md` ×2,
    /// `docs/observability.md` ×2) and a seventh referenced it from this file's own
    /// rustdoc. Adding the setter is what makes those docs true; deleting them
    /// would have discarded a fully-wired capability one line short of usable.
    /// Same shape as CFG-4's "a knob you cannot set is a knob you cannot evaluate".
    pub fn with_provider_rates_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.provider_rates_path = Some(path.into());
        self
    }

    /// Set the default namespace used by operations that don't specify `.in_namespace()`.
    pub fn default_namespace(mut self, ns: Namespace) -> Self {
        self.default_namespace = Some(ns);
        self
    }

    /// Supply the model identifier kremory should use for capability detection
    /// and metric labels (Option-1, 2026-06-23).
    ///
    /// kremory does **not** read the model back off the provider — the consumer
    /// owns this string. Set it when you wire a provider via the raw
    /// [`with_llm`](Self::with_llm) path and want full provider-native schema
    /// detection (`FormatSchema` for Ollama, `NativeSchema` for OpenAI/Anthropic).
    /// Without it, the raw path falls back to `PromptOnly` extraction and a
    /// `model="unknown"` metric label.
    ///
    /// State-agnostic: callable before or after `.with_llm(…)`. Tier-1 shortcuts
    /// (`with_ollama`/`with_openai`/`with_anthropic`) and
    /// [`with_llm_tracked`](Self::with_llm_tracked) set this for you.
    ///
    /// # Example
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use kremory::Memory;
    /// # async fn ex(
    /// #     my_llm: Arc<dyn kremory::memory::ChatProvider>,
    /// #     my_embedder: Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()> {
    /// let memory = Memory::open("./agent.db")
    ///     .with_llm(my_llm)
    ///     .with_model_id("qwen2.5:7b")
    ///     .with_embedder(my_embedder)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_model_id(mut self, model: impl Into<String>) -> Self {
        self.model_id = Some(model.into());
        self
    }

    /// Set the embedding vector dimensionality.
    ///
    /// **Must match your embedder's output dimension.** A mismatch causes a
    /// SQLite vector index failure on the first `remember()` call:
    /// `vector index(insert): dimensions are different: <actual> != <config>`.
    ///
    /// Common values:
    /// - `384` — MiniLM-L6-v2 (default if not set)
    /// - `768` — `nomic-embed-text` (Ollama), `all-mpnet-base-v2`
    /// - `1536` — OpenAI `text-embedding-3-small`
    /// - `3072` — OpenAI `text-embedding-3-large`
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }

    /// Override the soft warn threshold for episode content length (chars).
    ///
    /// Default: `Some(10_000)` (Zep-compatible). Set `None` to disable the
    /// warning entirely. Never enforced as a hard limit; the threshold only
    /// drives `tracing::warn!` + a `kremory_episode_oversize_total` counter
    /// at ingest time so callers can spot extraction-quality risk early.
    ///
    /// **This threshold does NOT protect the dense/embedding search arm**
    /// (TD-234's sibling, TD-232). Extraction sub-chunks internally
    /// (`ExtractionWindowSplitter`); embedding does not — a single episode
    /// over the embedder's own context window (commonly ~2048 tokens, ≈8-10k
    /// chars for `nomic-embed-text`) silently loses dense-arm coverage on
    /// that episode, staying BM25-only. Kremory does not currently chunk
    /// content for the embedder (a deliberate, separately-decided
    /// architectural boundary — see TD-232 in the tech-debt register for
    /// why). Detect it via `EpisodeCommit::dense_embedded` /
    /// `IngestionResult::dense_embedded` (`Some(false)` = this happened), and
    /// pre-chunk with [`crate::split_for_embedding`] before calling
    /// `remember()` (once per chunk) if it matters for your corpus.
    pub fn episode_content_warn_threshold(mut self, threshold: Option<usize>) -> Self {
        self.episode_content_warn_threshold = threshold;
        self
    }

    /// Provide a custom entity extractor (BYOE — bring your own extractor).
    ///
    /// Accepts any type that implements [`EntityExtractor`]. Wraps it in
    /// `Arc<dyn EntityExtractorDyn>` internally for object-safe dispatch.
    ///
    /// - Mutually exclusive with `.with_gliner()` — builder errors at build time
    ///   if both are set.
    /// - Compatible with or without `.with_llm()`: custom extractor runs regardless.
    ///   If LLM is also wired, it remains available for Category B methods.
    pub fn with_extractor<Ext>(mut self, extractor: Arc<Ext>) -> Self
    where
        Ext: crate::core::intelligence::EntityExtractor + 'static,
    {
        self.custom_extractor =
            Some(extractor as Arc<dyn crate::core::intelligence::EntityExtractorDyn>);
        self
    }

    /// Enable GLiNER-based candidate generation (requires the `ner` cargo feature).
    ///
    /// When combined with `.with_llm()`, the builder selects `ExtractorKind::GlinerLlm`
    /// (GLiNER for candidate spans + one LLM typing call per batch).
    ///
    /// Without `.with_llm()`, the builder errors at build time — GLiNER candidate-gen
    /// still requires one LLM call for entity-type classification.
    ///
    /// Takes no argument (F2): `GlinerConfig` had no public fields, so requiring
    /// it was a do-nothing parameter that forced consumers to construct a useless
    /// value. When real tuning knobs are added later (threshold, model path,
    /// batch size per ADR-039 §A6), a separate `with_gliner_config(GlinerConfig)`
    /// knob will be added non-breakingly.
    #[cfg(feature = "ner")]
    pub fn with_gliner(mut self) -> Self {
        self.use_gliner = true;
        self
    }

    /// Set the entity type names the extractor is allowed to emit.
    ///
    /// Forwarded to [`PipelineConfig::allowed_entity_types`]. Required when
    /// the `ner` feature is active and you want extraction to produce results.
    /// With an empty list (the default) the `GlinerExtractor` will reject all
    /// candidate entities.
    ///
    /// Typically callers pass the names from `DEFAULT_ENTITY_TYPES`:
    ///
    /// ```rust,no_run
    /// use kremory::Memory;
    /// use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
    ///
    /// let names: Vec<String> = DEFAULT_ENTITY_TYPES
    ///     .iter()
    ///     .map(|(_, name, _)| name.to_string())
    ///     .collect();
    /// // Memory::open("./db").allowed_entity_types(names)…
    /// ```
    pub fn allowed_entity_types(mut self, types: Vec<String>) -> Self {
        self.allowed_entity_types = types;
        self
    }

    // ── SearchConfig overrides (TD-141) ─────────────────────────────────────
    //
    // Per-knob setters, not a single `.with_search_config(SearchConfig)` —
    // per `composable-knobs-over-strategy-enum`: the consumer thinks in
    // individual knobs ("I want a higher content weight"), not in a whole
    // replacement config. Each setter is independent; setting one does not
    // affect the other two, and an unset knob still honours its env override
    // (`KREMORY_CONTENT_WEIGHT` / `KREMORY_RRF_K` / `KREMORY_EPISODE_DENSE`)
    // if one is present at `Memory` construction time.
    //
    // Precedence (TD-141 design decision (a)): explicit programmatic config
    // set here ALWAYS wins over the matching env override, which wins over
    // the `SearchConfig` default. See `PipelineConfigOverrides::apply` for the
    // mechanism and `facade::providers::search_env_overrides` for the env
    // layer these compose on top of. Verify the live, in-effect value via
    // `Memory::search_config()`.

    /// Explicit per-stream weight applied to the ADR-072 `content_search` BM25
    /// stream's RRF contribution (`SearchConfig::content_stream_weight`).
    /// Default (unset): `1.0` — equal-weight fusion — unless overridden by
    /// `KREMORY_CONTENT_WEIGHT` at construction time. Calling this setter wins
    /// over BOTH the default and any env override, for this `Memory` only.
    ///
    /// Mirrors [`PipelineConfigBuilder::content_stream_weight`](crate::core::config::PipelineConfigBuilder::content_stream_weight).
    pub fn with_content_stream_weight(mut self, v: f32) -> Self {
        self.config_overrides.content_stream_weight = Some(v);
        self
    }

    /// Explicit RRF fusion constant `k` (Cormack et al. 2009; `SearchConfig::rrf_k`).
    /// Default (unset): `60` unless overridden by `KREMORY_RRF_K` at
    /// construction time. Calling this setter wins over BOTH the default and
    /// any env override, for this `Memory` only.
    ///
    /// Mirrors [`PipelineConfigBuilder::rrf_k`](crate::core::config::PipelineConfigBuilder::rrf_k).
    pub fn with_rrf_k(mut self, v: usize) -> Self {
        self.config_overrides.rrf_k = Some(v);
        self
    }

    /// Explicitly enable/disable the TD-136 dense (embedding) episode
    /// retrieval arm (`SearchConfig::episode_dense_enabled`). Default
    /// (unset): `false` (BM25-only) unless overridden by
    /// `KREMORY_EPISODE_DENSE` at construction time. Calling this setter wins
    /// over BOTH the default and any env override, for this `Memory` only.
    /// Also gates ingest-time episode embedding — enabling this after data
    /// has already been ingested without it means older episodes lack a
    /// dense embedding until re-ingested or dreamed.
    ///
    /// Mirrors [`PipelineConfigBuilder::episode_dense_enabled`](crate::core::config::PipelineConfigBuilder::episode_dense_enabled).
    pub fn with_episode_dense_enabled(mut self, v: bool) -> Self {
        self.config_overrides.episode_dense_enabled = Some(v);
        self
    }

    /// Explicitly enable/disable the TD-139 dense (embedding) fact
    /// retrieval arm (`SearchConfig::fact_dense_enabled`). Default (unset):
    /// `false` (facts reachable only via 1-hop entity expansion) unless
    /// overridden by `KREMORY_FACT_DENSE` at construction time. Calling this
    /// setter wins over BOTH the default and any env override, for this
    /// `Memory` only. Ingest-time fact embedding is UNCHANGED by this knob
    /// (it already runs unconditionally, `core/ingest/pipeline/
    /// ingest_with.rs:~1774`) — this setter only gates whether RECALL reads
    /// `facts.embedding` back.
    ///
    /// Mirrors [`PipelineConfigBuilder::fact_dense_enabled`](crate::core::config::PipelineConfigBuilder::fact_dense_enabled).
    pub fn with_fact_dense_enabled(mut self, v: bool) -> Self {
        self.config_overrides.fact_dense_enabled = Some(v);
        self
    }

    /// Explicitly enable/disable the TD-143 nomic `search_document:` /
    /// `search_query:` task-prefix on every embed call site
    /// (`SearchConfig::embed_task_prefix_enabled`). Default (unset): `false`
    /// (bare-text embedding) unless overridden by `KREMORY_EMBED_TASK_PREFIX`
    /// at construction time. Calling this setter wins over BOTH the default
    /// and any env override, for this `Memory` only.
    ///
    /// ⚠️ Nomic-specific — only meaningful when the wired embedder is
    /// `nomic-embed-text`. ⚠️ Flipping this on an EXISTING corpus makes every
    /// already-stored embedding stale (mixed prefixed/unprefixed vectors in
    /// one index is a silent correctness bug, not a graceful degrade) — see
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled)
    /// for the required re-embed sequence.
    ///
    /// Mirrors [`PipelineConfigBuilder::embed_task_prefix_enabled`](crate::core::config::PipelineConfigBuilder::embed_task_prefix_enabled).
    pub fn with_embed_task_prefix_enabled(mut self, v: bool) -> Self {
        self.config_overrides.embed_task_prefix_enabled = Some(v);
        self
    }

    /// Explicit weight for the ADR-062 / ADR-067 Phase 3 graph-proximity
    /// boost (`SearchConfig::proximity_weight`). Default (unset): `0.0`
    /// (axis OFF — the second bounded-hop graph query never fires) unless
    /// overridden by `KREMORY_PROXIMITY_WEIGHT` at construction time. Calling
    /// this setter wins over BOTH the default and any env override, for this
    /// `Memory` only. `proximity_hop_bound` / `proximity_fan_out_cap` are
    /// config-default-only (2 / 8) — not exposed here, mirroring
    /// `expansion_hop_bound`/`expansion_fan_out_cap`'s own precedent.
    ///
    /// Mirrors [`PipelineConfigBuilder::proximity_weight`](crate::core::config::PipelineConfigBuilder::proximity_weight).
    pub fn with_proximity_weight(mut self, v: f32) -> Self {
        self.config_overrides.proximity_weight = Some(v);
        self
    }

    /// Explicit weight for the ADR-067 temporal-recency axis
    /// (`SearchConfig::temporal_weight`). Default (unset): `0.0` (axis OFF,
    /// byte-identical to pre-TD-157) unless overridden by
    /// `KREMORY_TEMPORAL_WEIGHT` at construction time. Calling this setter wins
    /// over BOTH the default and any env override, for this `Memory` only.
    ///
    /// TD-157 (2026-07-28): until this existed the axis was **unreachable** —
    /// real compute (`core/context.rs`, `core/scoring/temporal.rs`)
    /// permanently multiplied by a `0.0` that no builder method, env override
    /// or config file could change. It has therefore never been measured.
    ///
    /// Mirrors [`PipelineConfigBuilder::temporal_weight`](crate::core::config::PipelineConfigBuilder::temporal_weight).
    pub fn with_temporal_weight(mut self, v: f32) -> Self {
        self.config_overrides.temporal_weight = Some(v);
        self
    }

    /// Explicit cap (in `char`s) on the SUMMARY portion of each rerank
    /// candidate's text (`SearchConfig::rerank_candidate_max_chars`,
    /// reranker latency lever 1). Default (unset): `0` (unlimited — today's
    /// behaviour, byte-identical) unless overridden by
    /// `KREMORY_RERANK_CANDIDATE_MAX_CHARS` at construction time. Calling
    /// this setter wins over BOTH the default and any env override, for this
    /// `Memory` only. `entity_name` is never truncated.
    ///
    /// Mirrors [`PipelineConfigBuilder::rerank_candidate_max_chars`](crate::core::config::PipelineConfigBuilder::rerank_candidate_max_chars).
    pub fn with_rerank_candidate_max_chars(mut self, v: usize) -> Self {
        self.config_overrides.rerank_candidate_max_chars = Some(v);
        self
    }

    /// Explicitly enable/disable ingest-time contradiction detection
    /// (`PipelineConfig::contradiction_detection_enabled`, TD-167 /
    /// ADR-079 rev.2). Default (unset): **`true`** unless overridden by
    /// `KREMORY_CONTRADICTION_DETECTION` at construction time. Calling this
    /// setter wins over BOTH the default and any env override, for this
    /// `Memory` only.
    ///
    /// ⚠️ **This is the only knob on this builder that gates a DESTRUCTIVE
    /// path.** When ON, ingesting a fact that the detector judges to
    /// contradict a stored one SUPERSEDES the stored fact — a soft delete
    /// (`valid_to` closed, reversible via `reversal::unsupersede`), but it
    /// removes the prior fact from live recall. That is correct for a genuine
    /// update (`works_at Acme` → `works_at Globex`) and wrong for a
    /// SET-VALUED predicate, where every member but the last is destroyed
    /// (TD-167 measured 7/8 destroyed pre-fix, 0/8 post-fix; corpus
    /// contradiction rate 38.8% → 11.3%). Two of six real updates are still
    /// missed, so TD-167 remains open.
    ///
    /// Turn it **off** when your corpus is list-heavy (tags, attendees,
    /// preferences, playlist members) and append-only is the safer default;
    /// leave it **on** for profile-shaped data where later statements should
    /// replace earlier ones.
    ///
    /// Added by TD-172: this knob mirrors eight sibling `with_*` setters and
    /// was the only one missing, so opting OUT of a default-ON destructive
    /// path was reachable only through a process-wide env var — which cannot
    /// express two `Memory` instances with different settings.
    ///
    /// Mirrors [`PipelineConfigBuilder::contradiction_detection_enabled`](crate::core::config::PipelineConfigBuilder::contradiction_detection_enabled).
    pub fn with_contradiction_detection_enabled(mut self, v: bool) -> Self {
        self.config_overrides.contradiction_detection_enabled = Some(v);
        self
    }

    /// Per-arm time budget (ms) for a single structured-output extraction
    /// call in the ladder (`PipelineConfig::extraction_arm_budget_ms`).
    /// Default (unset): `30_000` — a production fail-fast tuned for hosted
    /// providers. Slow local models (qwen2.5:14b-class on Apple silicon,
    /// ~43-130s per call at 32k context) need this raised to 180_000-300_000
    /// or every extraction on an ordinary document exhausts the ladder before
    /// it can even try the cheaper fallback arms.
    ///
    /// Added because `Memory::with_ollama()` and this builder previously had
    /// NO reachable path to this knob at all (found via the aidocs-trial fit
    /// check, 2026-08-24/09-02) — despite the field's own doc comment
    /// pointing at "the builder" as if one already existed. This setter
    /// mirrors that doc comment's promise.
    ///
    /// Mirrors [`PipelineConfigBuilder::extraction_arm_budget_ms`](crate::core::config::PipelineConfigBuilder::extraction_arm_budget_ms).
    pub fn extraction_arm_budget_ms(mut self, ms: u64) -> Self {
        self.config_overrides.extraction_arm_budget_ms = Some(ms);
        self
    }

    /// How many preceding episodes of the SAME conversation thread are
    /// replayed into the extraction prompt so references resolve (ADR-080).
    ///
    /// Default (unset): **10**, the value mem0 and Graphiti independently
    /// converged on. `0` disables replay entirely.
    ///
    /// The thread key is `episodes.source_id` — what
    /// [`RememberRequest::from_chat`](crate::RememberRequest::from_chat) sets.
    /// Callers who never tag a source get a fresh uuid per episode, so this is
    /// inert for them whatever the depth.
    ///
    /// This setter exists because the off switch has to be REACHABLE: an A/B of
    /// the replay lever needs a control arm, and a knob you cannot turn off is a
    /// measurement you cannot trust.
    ///
    /// Mirrors [`PipelineConfigBuilder::prior_turn_replay_depth`](crate::core::config::PipelineConfigBuilder::prior_turn_replay_depth).
    pub fn prior_turn_replay_depth(mut self, depth: usize) -> Self {
        self.config_overrides.prior_turn_replay_depth = Some(depth);
        self
    }

    /// Configure automatic dream-pass scheduling.
    ///
    /// Default: [`DreamSchedule::Off`] — no background task is spawned.
    ///
    /// When set to a non-`Off` variant, a background tokio task is spawned
    /// during `.await` (i.e. at `MemoryBuilder::into_future`). The task runs
    /// until [`DreamSchedulerHandle::stop`] is called or the `Memory` is
    /// dropped.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kremory::{Memory, DreamSchedule};
    /// use std::time::Duration;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let llm = todo!(); let emb = todo!();
    /// let mem = Memory::open("./agent.db")
    ///     .with_llm(llm)
    ///     .with_embedder(emb)
    ///     .with_dream_schedule(DreamSchedule::Interval(Duration::from_secs(300)))
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C10 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub fn with_dream_schedule(
        mut self,
        schedule: crate::memory::scheduler::DreamSchedule,
    ) -> Self {
        self.dream_schedule = schedule;
        self
    }

    /// Supply a **distinct** LLM provider for the dream consolidation phase.
    ///
    /// Dream Pass 0 (type discovery) benefits from a deferred-*quality* model
    /// (e.g. `gemma4:e4b`) even when the interactive ingest path uses a fast
    /// model (e.g. `qwen2.5:7b`) wired via [`with_llm`](Self::with_llm).
    /// See TD-052b: the interactive model proposes the placeholder `"..."` in
    /// Pass 0 → silent zero-discovery; the deferred model proposes → accepts →
    /// retypes.
    ///
    /// **Additive + backward-compatible**: when unset, dream falls back to the
    /// [`with_llm`](Self::with_llm) provider — behaviour is unchanged for every
    /// existing consumer. Available in any type-state (mirrors
    /// [`with_dream_schedule`](Self::with_dream_schedule)). BYOM (ADR-002): the
    /// provider is supplied by the consumer; kremory bundles none.
    ///
    /// Composable-knob (not a strategy enum): dream-time selection picks
    /// behaviour from which provider knobs are set — see TD-052b §4 compat
    /// matrix.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # async fn ex(
    /// #     main: Arc<dyn kremory::memory::ChatProvider>,
    /// #     dream: Arc<dyn kremory::memory::ChatProvider>,
    /// #     embedder: Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()> {
    /// let memory = kremory::Memory::open(":memory:")
    ///     .with_llm(main)
    ///     .with_dream_llm(dream)
    ///     .with_embedder(embedder)
    ///     .await?;
    /// # let _ = memory;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dream_llm(mut self, llm: Arc<dyn ChatProvider>) -> Self {
        self.dream_llm = Some(llm);
        self
    }

    /// Set the concrete model id used by the dream phase's LLM passes for
    /// capability detection (TD-094). Pairs with [`with_dream_llm`](Self::with_dream_llm):
    /// when a dedicated dream provider is wired, its model string usually
    /// differs from the interactive model, so the dream passes need their own
    /// id to select the right structured-output strategy.
    ///
    /// Left unset, the dream passes fall back to the main
    /// [`with_model_id`](Self::with_model_id) string (and, absent that, degrade
    /// to `PromptOnly` — exactly as the interactive path does when the model is
    /// unknown). Threaded unchanged across every type-state transition.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # async fn ex(
    /// #     main: Arc<dyn kremory::memory::ChatProvider>,
    /// #     dream: Arc<dyn kremory::memory::ChatProvider>,
    /// #     embedder: Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()> {
    /// let memory = kremory::Memory::open(":memory:")
    ///     .with_llm(main)
    ///     .with_model_id("qwen2.5:7b")
    ///     .with_dream_llm(dream)
    ///     .with_dream_model_id("gemma4:e4b")
    ///     .with_embedder(embedder)
    ///     .await?;
    /// # let _ = memory;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dream_model_id(mut self, model: impl Into<String>) -> Self {
        self.dream_model_id = Some(model.into());
        self
    }

    /// Seed the DEFAULT namespace's entity-type registry at open time
    /// (spec custom-entity-type-registry §5.2.3).
    ///
    /// Applied during `.await` using the same catch-all injection + D9/D9a
    /// apply path as [`Memory::register_namespace_with_seed`]. Only takes effect
    /// when the default namespace has no `entity_types` rows (first open);
    /// subsequent opens are idempotent no-ops (`Default`/`Augment`) or — for a
    /// `Replace` whose taxonomy diverges from existing rows — a build error.
    ///
    /// Default: `NamespaceSeed::Default` (the standard `DEFAULT_ENTITY_TYPES`).
    ///
    /// # Two-knob interaction with `allowed_entity_types` (§5.12)
    ///
    /// Under the `ner` feature, when a `Replace`/`Augment` seed is supplied AND
    /// [`allowed_entity_types`](Self::allowed_entity_types) was NOT explicitly
    /// set, the extractor's allow-filter is DERIVED from the seed's type names
    /// (every seeded name except the id=0 catch-all) and a one-time
    /// `tracing::info!` records the derivation. An explicit `allowed_entity_types`
    /// always wins.
    pub fn with_seed_registry(mut self, seed: crate::core::entity_types::NamespaceSeed) -> Self {
        self.seed_registry = seed;
        self
    }

    /// Opt-in to synchronous-extraction ergonomics: when `true`,
    /// `Memory::remember(...).await` blocks until the ADR-051 background worker
    /// has transitioned the episode to `Verified` (returns `Ok(())`) or
    /// `Failed` / timeout (returns `Err`).
    ///
    /// Default: `false` (fire-and-forget — the ADR-051 design intent).
    ///
    /// Use [`with_await_extraction_timeout`](Self::with_await_extraction_timeout)
    /// to configure the maximum wait duration (default 60 s, see spec §Risk R-12).
    ///
    /// ⚠ **Cost**: enables sync semantics at the expense of the hot-path latency
    /// benefit that ADR-051 provides. Prefer `Memory::wait_for_processing` for
    /// fine-grained per-episode control (spec §Phase 4, Risk R-06).
    ///
    /// Per D1 peer pattern: equivalent to Cognee's `run_in_background=False`.
    pub fn with_await_extraction(mut self, await_extraction: bool) -> Self {
        self.await_extraction = await_extraction;
        self
    }

    /// Configure the maximum time `Memory::remember` will wait when
    /// `with_await_extraction(true)` is set.
    ///
    /// Default: 60 seconds (per spec §Risk R-12 mitigation — prevents
    /// false-timeout on real 30 s LLM extractions).
    ///
    /// Has no effect when `await_extraction` is `false` (the default).
    pub fn with_await_extraction_timeout(mut self, timeout: Duration) -> Self {
        self.await_extraction_timeout = timeout;
        self
    }
}

// ── Token-tracking knob (F3 / ADR adr-memory-builder-tracking-as-knob) ────────

impl<E> MemoryBuilder<WithLlm, E> {
    /// Wrap the configured LLM in token + cost tracking instrumentation.
    ///
    /// Composable-knob form of the (deprecated) `with_llm_tracked`: configure the
    /// provider once via [`with_llm`](MemoryBuilder::with_llm), then add
    /// observability as a separate step — one mental model, one arg shape. Emits
    /// `kremory_core_tokens_total`, `kremory_core_cost_usd_total`, and
    /// `kremory_core_chat_duration_seconds`. The `provider` / `model` labels must
    /// match `monitoring/provider-rates.toml` for cost counters to be non-zero.
    ///
    /// `provider` / `model` are metric-attribution labels (e.g. `"openai"`,
    /// `"gpt-4o-mini"`).
    ///
    /// # Call once
    ///
    /// Wraps the *current* LLM; calling twice nests trackers (double-counts).
    /// Wire `with_llm(...).with_token_tracking(...)` exactly once.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kremory::Memory;
    /// # async fn ex(
    /// #     my_llm: std::sync::Arc<dyn kremory::memory::ChatProvider>,
    /// #     my_embedder: std::sync::Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()> {
    /// let memory = Memory::open("./agent.db")
    ///     .with_llm(my_llm)
    ///     .with_token_tracking("openai", "gpt-4o-mini")
    ///     .with_embedder(my_embedder)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_token_tracking(
        mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        if let Some(llm) = self.llm.take() {
            let wrapped = TokenTrackingChatProvider::new(
                crate::core::provider::ArcChatProvider::new(llm),
                provider,
                model,
            );
            self.llm = Some(Arc::new(wrapped));
        }
        self
    }
}

/// Bundled parameters for [`MemoryBuilder::with_llm_tracked`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). The generic `llm: L`
/// stays a lead positional param; the two label strings are bundled here.
pub struct WithLlmTrackedParams {
    /// Provider label (e.g. `"openai"`, `"ollama"`) for metric/cost attribution.
    pub provider: String,
    /// Model label (e.g. `"gpt-4o-mini"`) for metric/cost attribution.
    pub model: String,
}

impl MemoryBuilder<NoLlm, NoEmb> {
    /// Construct the initial builder state for `Memory::open`.
    ///
    /// `pub(super)` so only `facade/mod.rs` (i.e. `Memory::open`) can call this;
    /// consumers use the type-state chain (`Memory::open(path).with_llm(…)…`).
    pub(super) fn new_open(path: std::path::PathBuf) -> Self {
        Self {
            path,
            llm: None,
            model_id: None,
            dream_llm: None,
            dream_model_id: None,
            embedder: None,
            default_sink: None,
            default_namespace: None,
            embedding_dim: None,
            provider_rates_path: None,
            episode_content_warn_threshold: Some(10_000),
            custom_extractor: None,
            #[cfg(feature = "ner")]
            use_gliner: false,
            allowed_entity_types: vec![],
            config_overrides: PipelineConfigOverrides::default(),
            seed_registry: crate::core::entity_types::NamespaceSeed::Default,
            dream_schedule: crate::memory::scheduler::DreamSchedule::Off,
            await_extraction: false,
            await_extraction_timeout: Duration::from_secs(60),
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }

    /// Configure the LLM provider (required).
    ///
    /// The provider is used as-is, without token or cost instrumentation. For
    /// automatic observability, prefer [`with_llm_tracked`](Self::with_llm_tracked).
    pub fn with_llm(self, llm: Arc<dyn ChatProvider>) -> MemoryBuilder<WithLlm, NoEmb> {
        MemoryBuilder {
            path: self.path,
            llm: Some(llm),
            // Raw `with_llm` sets no model id itself (Option-1) — preserves any
            // prior `.with_model_id(…)`. Unset → capability detection → PromptOnly.
            model_id: self.model_id,
            dream_llm: self.dream_llm,
            dream_model_id: self.dream_model_id,
            embedder: self.embedder,
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            config_overrides: self.config_overrides,
            seed_registry: self.seed_registry,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }

    /// Wrap a user-supplied `ChatProvider` in a [`TokenTrackingChatProvider`],
    /// capturing `(provider, model)` labels for metrics emission.
    ///
    /// Preferred over [`with_llm`](Self::with_llm) when the caller wants automatic
    /// token count, cost, and duration observability via the kremory metrics surface
    /// (`kremory_core_tokens_total`, `kremory_core_cost_usd_total`,
    /// `kremory_core_chat_duration_seconds`).
    ///
    /// The `provider` and `model` labels must match entries in
    /// `monitoring/provider-rates.toml` (or a custom rates file via
    /// `with_provider_rates_path`) for
    /// cost counters to emit non-zero values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kremory::Memory;
    /// # async fn ex<L>(
    /// #     my_openai_client: L,
    /// #     my_embedder: std::sync::Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()>
    /// # where
    /// #     L: kremory::memory::ChatProvider + Send + Sync + 'static,
    /// # {
    /// let memory = Memory::open("./agent.db")
    ///     .with_llm_tracked(
    ///         kremory::WithLlmTrackedParams {
    ///             provider: "openai".into(),
    ///             model: "gpt-4o-mini".into(),
    ///         },
    ///         my_openai_client,
    ///     )
    ///     .with_embedder(my_embedder)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.2.5",
        note = "use `with_llm(llm).with_token_tracking(provider, model)` — composable knob \
                per ADR adr-memory-builder-tracking-as-knob. Removed in a later breaking release."
    )]
    pub fn with_llm_tracked<L: ChatProvider + Send + Sync + 'static>(
        self,
        params: WithLlmTrackedParams,
        llm: L,
    ) -> MemoryBuilder<WithLlm, NoEmb> {
        let WithLlmTrackedParams { provider, model } = params;
        // Capture the tracked model id as kremory-owned data (Option-1) before
        // `model` is moved into the tracking wrapper.
        let model_id = Some(model.clone());
        let tracked = TokenTrackingChatProvider::new(llm, provider, model);
        MemoryBuilder {
            path: self.path,
            llm: Some(Arc::new(tracked) as Arc<dyn ChatProvider>),
            model_id,
            dream_llm: self.dream_llm,
            dream_model_id: self.dream_model_id,
            embedder: self.embedder,
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            config_overrides: self.config_overrides,
            seed_registry: self.seed_registry,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

impl MemoryBuilder<WithLlm, NoEmb> {
    /// Configure the embedding provider (required).
    pub fn with_embedder(
        self,
        emb: Arc<dyn DynEmbeddingProvider>,
    ) -> MemoryBuilder<WithLlm, WithEmb> {
        MemoryBuilder {
            path: self.path,
            llm: self.llm,
            model_id: self.model_id,
            dream_llm: self.dream_llm,
            dream_model_id: self.dream_model_id,
            embedder: Some(emb),
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            config_overrides: self.config_overrides,
            seed_registry: self.seed_registry,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

impl MemoryBuilder<NoLlm, NoEmb> {
    /// Configure the embedding provider for the no-LLM path.
    ///
    /// Use this when you supply a custom extractor via `.with_extractor(…)` but
    /// do not need an LLM provider. The resulting `Memory` supports all Category A
    /// operations; Category B operations (`dream`, `recall_with_disambiguation`,
    /// `detect_contradictions`) return `Error::LlmRequired` at call time.
    pub fn with_embedder(
        self,
        emb: Arc<dyn DynEmbeddingProvider>,
    ) -> MemoryBuilder<NoLlm, WithEmb> {
        MemoryBuilder {
            path: self.path,
            llm: self.llm,
            model_id: self.model_id,
            dream_llm: self.dream_llm,
            dream_model_id: self.dream_model_id,
            embedder: Some(emb),
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            config_overrides: self.config_overrides,
            seed_registry: self.seed_registry,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

/// §5.12 two-knob trap: when a `Replace`/`Augment` seed is supplied under the
/// `ner` feature AND `allowed_entity_types` was not explicitly set, derive the
/// filter from the seed's type names (every seeded name except the id=0
/// catch-all) so the GLiNER allow-filter agrees with the seeded registry. An
/// explicit `allowed_entity_types` always wins. Emits a one-time `tracing::info!`.
///
/// Without the `ner` feature there is no GLiNER allow-filter, so this is a
/// no-op — but it always consumes `&mut self` so the caller's `mut self`
/// binding is used in every cfg (no `#[allow(unused_mut)]` needed).
fn derive_allowed_from_seed_if_unset<L, E>(builder: &mut MemoryBuilder<L, E>) {
    // Without `ner` the param is unused; borrow it so neither `builder` nor the
    // call site's `mut self` triggers an unused warning (no `#[allow]`).
    #[cfg(not(feature = "ner"))]
    let _ = &mut *builder;
    #[cfg(feature = "ner")]
    {
        use crate::core::entity_types::NamespaceSeed;
        if !builder.allowed_entity_types.is_empty() {
            return; // explicit filter wins
        }
        let specs: &[crate::core::entity_types::EntityTypeSpec] = match &builder.seed_registry {
            NamespaceSeed::Replace(s) | NamespaceSeed::Augment(s) => s,
            NamespaceSeed::Default => return,
        };
        let derived: Vec<String> = specs
            .iter()
            .filter(|s| s.id != 0)
            .map(|s| s.name.clone())
            .collect();
        if derived.is_empty() {
            return;
        }
        tracing::info!(
            target: "kremory.facade.builder",
            derived_count = derived.len(),
            "kremory.namespace.allowed_entity_types_derived_from_seed: deriving GLiNER \
             allow-filter from seed type names (§5.12) — allowed_entity_types was unset"
        );
        builder.allowed_entity_types = derived;
    }
}

/// Apply the builder's `seed_registry` to the DEFAULT namespace at build time
/// (spec §5.2.3), inside a `BEGIN IMMEDIATE` txn on `tg`. The default-namespace
/// group_id is `default_namespace`'s key when set, else `"default"` (the
/// facade's implicit default). Errors are surfaced as `MemoryError` — a
/// divergent `Replace` on an already-populated default namespace fails the build
/// loud (D9), consistent with `register_namespace_with_seed`.
async fn apply_builder_seed(
    tg: &crate::core::schema::TemporalGraph,
    default_namespace: Option<&Namespace>,
    seed: &crate::core::entity_types::NamespaceSeed,
) -> Result<()> {
    let group_id = match default_namespace {
        Some(ns) => crate::memory::engine_handle::namespace_to_group_id(ns),
        None => "default".to_string(),
    };
    let guard = tg
        .begin_immediate_if_needed()
        .await
        .map_err(MemoryError::Core)?;
    let result = crate::core::entity_types::apply_namespace_seed(&tg.conn, &group_id, seed).await;
    match &result {
        Ok(_) => guard.commit().await.map_err(MemoryError::Core)?,
        Err(_) => guard.rollback().await.map_err(MemoryError::Core)?,
    }
    result.map(|_| ()).map_err(|e| match e {
        crate::core::entity_types::NamespaceRegistrationError::Store(core) => {
            MemoryError::Core(core)
        }
        other => MemoryError::Other(other.to_string()),
    })
}

impl IntoFuture for MemoryBuilder<WithLlm, WithEmb> {
    type Output = Result<Memory>;
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send>>;

    fn into_future(mut self) -> Self::IntoFuture {
        Box::pin(async move {
            // §5.12 — derive the ner allow-filter from the seed when unset.
            // `derive_allowed_from_seed_if_unset` is a no-op without `ner` but
            // always consumes `&mut self`, so `mut self` is used in every cfg.
            derive_allowed_from_seed_if_unset(&mut self);

            // Initialize provider rates (idempotent). Errors are logged but not
            // fatal — cost counters will skip emission with a one-shot warn.
            if let Some(ref custom_path) = self.provider_rates_path {
                if let Err(e) = crate::core::rates::init_from_path(custom_path.as_path()) {
                    tracing::warn!(
                        error = %e,
                        path  = %custom_path.display(),
                        "failed to load custom provider-rates.toml — cost counters will not be emitted"
                    );
                }
            } else if let Err(e) = crate::core::rates::init_bundled() {
                tracing::warn!(
                    error = %e,
                    "failed to load bundled provider-rates.toml — cost counters will not be emitted"
                );
            }

            let llm = self
                .llm
                .ok_or_else(|| MemoryError::Other("llm missing".into()))?;
            let embedder = self
                .embedder
                .ok_or_else(|| MemoryError::Other("embedder missing".into()))?;

            // Consumer-supplied model id (Option-1) — threaded into every
            // Engine-construction param below. `None` → capability detection
            // → PromptOnly.
            let model_id = self.model_id.clone();

            // TD-141 — captured up-front (mirrors `model_id` above) because it
            // is needed at up to three separate `GraphOpenParams` /
            // `OpenGraphParams` / `OpenEngineHandleParams` construction sites
            // below (the compat-matrix open_graph* branch, then — when
            // `.with_sink()` is configured — the two `open_engine_handle`
            // calls for the `BackgroundIngestorGraphHandle` path).
            let config_overrides = self.config_overrides.clone();

            // ── Compat matrix (ADR-039, 7-row table) ────────────────────────
            // Row 6: .with_extractor conflicts with .with_gliner → Err
            #[cfg(feature = "ner")]
            if self.custom_extractor.is_some() && self.use_gliner {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: ".with_extractor conflicts with .with_gliner — \
                             supply one or the other, not both"
                        .into(),
                }));
            }

            // F1 (ADR adr-memory-builder-gliner-allowlist-fail-loud): GLiNER is a
            // closed-vocabulary model — an empty allowed_entity_types (the builder
            // default) makes GlinerExtractor reject ALL candidates → silent zero
            // extraction. This was the one quiet cell in an otherwise loud-at-build
            // matrix. Fail loud, naming the fix (llm-output-parse-loudly: never
            // silently default-to-nothing).
            #[cfg(feature = "ner")]
            if self.use_gliner && self.allowed_entity_types.is_empty() {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: "GLiNER needs an entity-type allowlist; call \
                             .allowed_entity_types(DEFAULT_ENTITY_TYPES…) — an empty \
                             list rejects all candidates (silent zero extraction)"
                        .into(),
                }));
            }

            // Row 2: LLM only → Llm extractor (default open_graph path)
            // Row 3: LLM + gliner → GlinerLlm extractor
            // Row 5/6 (with LLM): custom extractor wins, LLM still available
            //
            // Clone allowed_entity_types up-front: it may be needed again below
            // when constructing the BackgroundIngestorGraphHandle's two engines
            // (when self.default_sink.is_some()). Moved values cannot be cloned
            // after the move; clone before the compat-matrix branches consume it.
            let allowed_entity_types_for_bg = self.allowed_entity_types.clone();

            let (graph, temporal_graph) = if let Some(custom) = self.custom_extractor {
                // Rows 5/6 with LLM — open with explicit Custom extractor
                providers::open_graph_with_extractor(
                    providers::GraphOpenParams {
                        path: self.path.clone(),
                        embedder: embedder.clone(),
                        embedding_dim: self.embedding_dim,
                        allowed_entity_types: self.allowed_entity_types,
                        model: model_id.clone(),
                        overrides: config_overrides.clone(),
                    },
                    llm.clone(),
                    crate::core::extraction::factory::ExtractorKind::Custom(custom),
                )
                .await?
            } else {
                #[cfg(feature = "ner")]
                if self.use_gliner {
                    // Row 3: LLM + gliner → GlinerLlm
                    use crate::core::provider::ArcChatProvider;
                    let arc_llm = Arc::new(ArcChatProvider::new(llm.clone()));
                    let gliner_ext =
                        crate::core::extraction::hybrid_typer::GlinerLlmExtractor::new(arc_llm)
                            .map_err(|e| {
                                MemoryError::Core(CoreError::BuilderConflict {
                                    detail: format!("GLiNER extractor init failed: {e}"),
                                })
                            })?;
                    providers::open_graph_with_extractor(
                        providers::GraphOpenParams {
                            path: self.path.clone(),
                            embedder: embedder.clone(),
                            embedding_dim: self.embedding_dim,
                            allowed_entity_types: self.allowed_entity_types,
                            model: model_id.clone(),
                            overrides: config_overrides.clone(),
                        },
                        llm.clone(),
                        crate::core::extraction::factory::ExtractorKind::GlinerLlm(Box::new(
                            gliner_ext,
                        )),
                    )
                    .await?
                } else {
                    // Row 2: LLM only → default Llm extractor
                    providers::open_graph(
                        self.path.as_path(),
                        providers::OpenGraphParams {
                            llm: llm.clone(),
                            embedder: embedder.clone(),
                            embedding_dim: self.embedding_dim,
                            allowed_entity_types: self.allowed_entity_types,
                            model: model_id.clone(),
                            overrides: config_overrides.clone(),
                        },
                    )
                    .await?
                }
                #[cfg(not(feature = "ner"))]
                {
                    // Row 2 (no ner feature): default Llm extractor
                    providers::open_graph(
                        self.path.as_path(),
                        providers::OpenGraphParams {
                            llm: llm.clone(),
                            embedder: embedder.clone(),
                            embedding_dim: self.embedding_dim,
                            allowed_entity_types: self.allowed_entity_types,
                            model: model_id.clone(),
                            overrides: config_overrides.clone(),
                        },
                    )
                    .await?
                }
            };

            // T6.2 — Warm schema caches after graph open, before returning.
            // Best-effort: errors are silently ignored inside warm_schema_caches.
            // Spawned on a background tokio task so engine startup is not blocked
            // by the 10 × LLM round-trips.
            // Gated to non-test builds: unit tests use fast-open paths and must
            // not incur LLM round-trips on every engine construction.
            // model is passed as None: MemoryBuilder does not surface the model
            // string at construction time.  Warmup is connection-pool warm only
            // (LlmJsonRepair arm); NativeSchema/FormatSchema arms are not reached.
            // Tier 1 callers (providers::build_memory) pass the concrete model
            // string and reach the provider-native schema-compilation path.
            #[cfg(not(test))]
            {
                use crate::core::extraction::structured::warm_schema_caches;
                use crate::core::provider::ArcChatProvider;
                let warmup_llm = Arc::new(ArcChatProvider::new(llm.clone()));
                tokio::spawn(async move {
                    warm_schema_caches(warmup_llm.as_ref(), None).await;
                });
            }

            // ── BackgroundIngestorGraphHandle (ADR-052 Gap 1 / Quinn MED-3 fix) ─
            //
            // When `.with_sink()` is configured, replace the `Arc<dyn GraphHandle>`
            // produced by `open_graph` with a `BackgroundIngestorGraphHandle` that
            // routes `run_in_background=true` calls through `BackgroundIngestor`.
            //
            // This is the composable-knobs trigger: `.with_sink()` → builder picks
            // `BackgroundIngestorGraphHandle` internally (arch spec §3.2 Option A).
            //
            // Two-Engine shape (arch spec §3.3):
            //   - Engine 1 (consumed by BackgroundIngestor): the engine already
            //     constructed inside `open_graph` above, accessible via `graph`
            //     (which is Arc<dyn GraphHandle> = Arc<EngineGraphHandle>). Since
            //     `open_graph` type-erases the engine handle, we open a SECOND
            //     lightweight `EngineGraphHandle` on the same path for the
            //     non-background delegate role.
            //   - Engine 2 (EngineGraphHandle delegate): opened here via
            //     `providers::open_engine_handle`. Handles search/dream/inline-ingest.
            //   Both engines open their own libSQL WAL connection; write serialisation
            //   is enforced by the background OS thread (ADR-051 invariant).
            //   WAL concurrency safety: empirically verified by Spike B + C in
            //   `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §6`.
            //
            // When no sink is configured: `graph` stays as-is (EngineGraphHandle path,
            // unchanged semantics for all existing callers).
            let graph: Arc<dyn GraphHandle> = if let Some(ref sink) = self.default_sink {
                // Open a second EngineGraphHandle on the same DB path for the
                // non-background delegate (search, dream, inline ingest).
                // Uses `allowed_entity_types_for_bg` (cloned before compat-matrix
                // branches moved `self.allowed_entity_types`).
                let (engine_handle, _tg2) = providers::open_engine_handle(
                    self.path.as_path(),
                    providers::OpenEngineHandleParams {
                        llm: llm.clone(),
                        embedder: embedder.clone(),
                        embedding_dim: self.embedding_dim,
                        allowed_entity_types: allowed_entity_types_for_bg.clone(),
                        model: model_id.clone(),
                        overrides: config_overrides.clone(),
                    },
                )
                .await?;

                // Construct BackgroundIngestor from a THIRD engine connection.
                // open_graph above type-erases the engine into Arc<dyn GraphHandle>
                // so we cannot extract it; open a fresh connection for the
                // BackgroundIngestor's exclusive Engine ownership (ADR-051 invariant).
                //
                // Three-connection shape: BG ingestor Engine + EGH delegate Engine
                // + TemporalGraph (facade). All open their own libSQL WAL connection.
                // WAL serialises concurrent writes safely (Spike B + C, arch spec §6).
                let (bg_engine_handle_for_ingestor, _tg3) = providers::open_engine_handle(
                    self.path.as_path(),
                    providers::OpenEngineHandleParams {
                        llm: llm.clone(),
                        embedder: embedder.clone(),
                        embedding_dim: self.embedding_dim,
                        allowed_entity_types: allowed_entity_types_for_bg,
                        model: model_id.clone(),
                        overrides: config_overrides,
                    },
                )
                .await?;

                // Unwrap the engine from the EngineGraphHandle to pass to BackgroundIngestor.
                // BackgroundIngestor takes exclusive ownership per ADR-051 serialisation
                // invariant (ingestor.rs:62-74).
                //
                // `Arc::try_unwrap` succeeds here because `open_engine_handle` wraps
                // the engine in a fresh `Arc` with exactly one owner. The panic branch
                // surfaces a construction-time invariant violation (not expected in
                // production; surfaces in tests if construction logic changes).
                // Note: `.expect()` requires E: Debug; Engine doesn't impl Debug,
                // so we use `.unwrap_or_else(|_| panic!(...))` instead.
                let bg_engine = Arc::try_unwrap(bg_engine_handle_for_ingestor.engine)
                    .unwrap_or_else(|_| {
                        panic!(
                            "kremory: BackgroundIngestor engine Arc has unexpected extra owners \
                             at build() time — invariant violation in open_engine_handle"
                        )
                    });

                let ingestor_config = IngestorConfig {
                    sink: Some(Arc::clone(sink)),
                    ..IngestorConfig::default()
                };
                let (ingestor, guard) = BackgroundIngestor::new(bg_engine, ingestor_config);

                tracing::debug!(
                    target: "kremory.facade",
                    "MemoryBuilder::build: sink configured — constructing BackgroundIngestorGraphHandle \
                     (ADR-052 Gap 1 fix, arch spec §3.2 Option A)"
                );
                metrics::counter!(
                    "kremory.facade.build_path",
                    "handle" => "BackgroundIngestorGraphHandle"
                )
                .increment(1);

                Arc::new(BackgroundIngestorGraphHandle::new(
                    Arc::new(ingestor),
                    Arc::new(engine_handle),
                    guard,
                ))
            } else {
                // No sink configured: use the existing EngineGraphHandle as before.
                // All existing callers are unaffected (Option A minimal blast radius).
                graph
            };

            // ── Dream scheduler (C10) ────────────────────────────────────────
            // Spawn background scheduler if a non-Off schedule was requested.
            let dream_scheduler_handle = match self.dream_schedule {
                crate::memory::scheduler::DreamSchedule::Off => None,
                schedule => {
                    let handle = crate::memory::scheduler::spawn_scheduler(
                        Arc::clone(&graph),
                        schedule,
                        crate::core::ingest::DreamPassOpts::default,
                    );
                    Some(handle)
                }
            };

            // §5.2.3 — apply the builder seed to the default namespace at open.
            // Skip when `Default` (preserves byte-identical existing behaviour;
            // lazy `ensure_default_types_seeded` still fires on first ingest).
            if !matches!(
                self.seed_registry,
                crate::core::entity_types::NamespaceSeed::Default
            ) {
                apply_builder_seed(
                    &temporal_graph,
                    self.default_namespace.as_ref(),
                    &self.seed_registry,
                )
                .await?;
            }

            Ok(Memory {
                graph,
                llm: Some(llm),
                dream_llm: self.dream_llm,
                model_id: self.model_id,
                dream_model_id: self.dream_model_id,
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
                temporal_graph: Some(temporal_graph),
                episode_content_warn_threshold: self.episode_content_warn_threshold,
                dream_scheduler: std::sync::Arc::new(std::sync::Mutex::new(dream_scheduler_handle)),
                await_extraction: self.await_extraction,
                await_extraction_timeout: self.await_extraction_timeout,
            })
        })
    }
}

impl IntoFuture for MemoryBuilder<NoLlm, WithEmb> {
    type Output = Result<Memory>;
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send>>;

    fn into_future(mut self) -> Self::IntoFuture {
        Box::pin(async move {
            // §5.12 — derive the ner allow-filter from the seed when unset.
            // `derive_allowed_from_seed_if_unset` is a no-op without `ner` but
            // always consumes `&mut self`, so `mut self` is used in every cfg.
            derive_allowed_from_seed_if_unset(&mut self);

            // Row 4: gliner set but no LLM → Err
            #[cfg(feature = "ner")]
            if self.use_gliner && self.custom_extractor.is_none() {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: "GLiNER candidate-gen needs LLM for entity-type classification — \
                             add .with_llm(…) or swap to .with_extractor(…) for a fully \
                             custom extractor that doesn't require LLM"
                        .into(),
                }));
            }

            // Row 6 (conflict): with_extractor + with_gliner → Err
            #[cfg(feature = "ner")]
            if self.custom_extractor.is_some() && self.use_gliner {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: ".with_extractor conflicts with .with_gliner — \
                             supply one or the other, not both"
                        .into(),
                }));
            }

            // Row 0: nothing wired → Err
            let custom = self.custom_extractor.ok_or_else(|| {
                MemoryError::Core(CoreError::BuilderConflict {
                    detail: "no extractor wired — call .with_llm(…) for built-in LLM extraction, \
                             or .with_extractor(Arc<impl EntityExtractor>) to bring your own"
                        .into(),
                })
            })?;

            let embedder = self
                .embedder
                .ok_or_else(|| MemoryError::Other("embedder missing".into()))?;

            // Row 4 (no LLM, custom extractor) — open without LLM.
            // model is carried for struct completeness; the no-LLM path does no
            // capability detection (custom extractor owns extraction).
            let (graph, temporal_graph) = providers::open_graph_no_llm(
                providers::GraphOpenParams {
                    path: self.path.clone(),
                    embedder: embedder.clone(),
                    embedding_dim: self.embedding_dim,
                    allowed_entity_types: self.allowed_entity_types,
                    model: self.model_id.clone(),
                    overrides: self.config_overrides,
                },
                custom,
            )
            .await?;

            // §5.2.3 — apply the builder seed to the default namespace at open.
            if !matches!(
                self.seed_registry,
                crate::core::entity_types::NamespaceSeed::Default
            ) {
                apply_builder_seed(
                    &temporal_graph,
                    self.default_namespace.as_ref(),
                    &self.seed_registry,
                )
                .await?;
            }

            Ok(Memory {
                graph,
                llm: None,
                dream_llm: self.dream_llm,
                model_id: self.model_id,
                dream_model_id: self.dream_model_id,
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
                temporal_graph: Some(temporal_graph),
                episode_content_warn_threshold: self.episode_content_warn_threshold,
                dream_scheduler: std::sync::Arc::new(std::sync::Mutex::new(None)),
                await_extraction: self.await_extraction,
                await_extraction_timeout: self.await_extraction_timeout,
            })
        })
    }
}

// ── §5.12 two-knob derivation tests (T10) ───────────────────────────────────
//
// T10 verifies the `ner`-gated `with_seed_registry` → `allowed_entity_types`
// derivation (spec §5.13). It tests `derive_allowed_from_seed_if_unset`
// directly — the deterministic apply path the builder runs at
// `IntoFuture::into_future` — so it needs no GLiNER model download. The
// derivation only has observable effect under the `ner` feature (without it the
// allow-filter is a no-op), so the whole module is `ner`-gated.
#[cfg(all(test, feature = "ner"))]
mod ner_seed_derivation_tests {
    use super::*;
    use crate::core::entity_types::{EntityTypeSpec, NamespaceSeed};

    fn spec(id: u32, name: &str) -> EntityTypeSpec {
        EntityTypeSpec {
            id,
            name: name.to_string(),
            description: format!("{name} description."),
        }
    }

    fn base_builder() -> MemoryBuilder<NoLlm, NoEmb> {
        MemoryBuilder::new_open(std::path::PathBuf::from(":memory:"))
    }

    /// T10 — `with_seed_registry(Replace(..))` with `allowed_entity_types` unset
    /// derives the allow-filter from the seed's type names (id=0 catch-all
    /// excluded).
    #[test]
    fn t10_replace_seed_derives_allow_filter_when_unset() {
        let mut builder = base_builder().with_seed_registry(NamespaceSeed::Replace(vec![
            spec(11, "Court"),
            spec(12, "Judge"),
        ]));
        assert!(
            builder.allowed_entity_types.is_empty(),
            "precondition: allow-filter unset before derivation"
        );
        derive_allowed_from_seed_if_unset(&mut builder);
        let mut derived = builder.allowed_entity_types.clone();
        derived.sort();
        assert_eq!(
            derived,
            vec!["Court".to_string(), "Judge".to_string()],
            "allow-filter must be derived from seed names, catch-all excluded"
        );
    }

    /// T10 — `Augment(..)` seed likewise derives from its custom names (the
    /// id=0 catch-all is never derived).
    #[test]
    fn t10_augment_seed_derives_custom_names_only() {
        let mut builder = base_builder().with_seed_registry(NamespaceSeed::Augment(vec![
            spec(11, "Court"),
            spec(0, "Entity"),
        ]));
        derive_allowed_from_seed_if_unset(&mut builder);
        assert_eq!(
            builder.allowed_entity_types,
            vec!["Court".to_string()],
            "Augment must derive only the custom names; id=0 catch-all excluded"
        );
    }

    /// T10 — an explicit `allowed_entity_types` ALWAYS wins; the seed-derived
    /// filter must not overwrite it.
    #[test]
    fn t10_explicit_allow_filter_wins_over_seed_derivation() {
        let mut builder = base_builder()
            .allowed_entity_types(vec!["Person".to_string()])
            .with_seed_registry(NamespaceSeed::Replace(vec![spec(11, "Court")]));
        derive_allowed_from_seed_if_unset(&mut builder);
        assert_eq!(
            builder.allowed_entity_types,
            vec!["Person".to_string()],
            "explicit allowed_entity_types must win over seed derivation"
        );
    }

    /// T10 — a `Default` seed derives nothing (status quo; allow-filter stays
    /// unset for the consumer/feature to supply).
    #[test]
    fn t10_default_seed_derives_nothing() {
        let mut builder = base_builder().with_seed_registry(NamespaceSeed::Default);
        derive_allowed_from_seed_if_unset(&mut builder);
        assert!(
            builder.allowed_entity_types.is_empty(),
            "Default seed must not derive an allow-filter"
        );
    }
}

use std::time::Duration;

use super::builder::PipelineConfigBuilder;
use super::scan_configs::{
    EntropyConfig, ExtractionWindowConfig, MinHashConfig, SecretScanConfig, SecretScanMode,
};
use super::search::SearchConfig;

/// Explicit, per-knob overrides for [`SearchConfig`] set programmatically via
/// [`MemoryBuilder`](crate::facade::MemoryBuilder). Each field is `None`
/// unless the consumer called the matching `MemoryBuilder::with_*` setter —
/// `None` means "not set programmatically", NOT "use this field's shipped
/// default".
///
/// # Why a sparse overlay, not `Option<SearchConfig>`
///
/// An earlier design sketched threading `search: Option<SearchConfig>`
/// through the `open_graph*` params. This type deviates from that literal shape:
/// it requires PER-FIELD precedence (explicit programmatic
/// config > env override > default — see [`apply`](Self::apply)), and it
/// requires per-knob setters, not one `with_search_config(SearchConfig)`.
/// A consumer who calls only `.with_content_stream_weight(v)` must NOT
/// silently clobber a `KREMORY_RRF_K` / `KREMORY_EPISODE_DENSE` env override
/// for the other two fields. A monolithic `SearchConfig` cannot express
/// "unset" per field once materialized (every field always holds a concrete
/// value), so wrapping it in `Option` would force an all-or-nothing choice —
/// either every explicit config always wins outright (clobbering env knobs
/// the consumer never touched) or the whole thing is applied before env
/// (defeating the seam's purpose). The sparse `Option<T>`-per-field overlay
/// is the shape that actually satisfies (a) and (d) together.
#[derive(Debug, Clone, Default)]
pub(crate) struct PipelineConfigOverrides {
    /// Explicit override for [`SearchConfig::content_stream_weight`].
    pub content_stream_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::graph_degree_weight`].
    /// `graph_degree_weight` is
    /// the one scoring axis that shipped LIVE (default `0.05`, not a
    /// no-op) but, unlike every sibling axis on this struct, had no builder
    /// setter and no env override — reachable only for *reading* (via
    /// `Memory::search_config()`), never for *writing*. `None` leaves the
    /// compiled-in default of `0.05` unchanged.
    pub graph_degree_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::rrf_k`].
    pub rrf_k: Option<usize>,
    /// Explicit override for [`SearchConfig::episode_dense_enabled`].
    pub episode_dense_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::fact_dense_enabled`].
    pub fact_dense_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::embed_task_prefix_enabled`].
    pub embed_task_prefix_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::proximity_weight`].
    /// `proximity_hop_bound` / `proximity_fan_out_cap` are
    /// deliberately NOT exposed here — config-default-only, mirroring
    /// `expansion_hop_bound`/`expansion_fan_out_cap`'s own precedent (only
    /// the weight is the A/B lever that needs a restart-not-rebuild seam).
    pub proximity_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::temporal_weight`].
    ///
    /// This axis has had working compute behind it all along
    /// (`core/context.rs`, `core/scoring/temporal.rs`) and was **unreachable**:
    /// unlike its sibling `proximity_weight` it had no builder method, no env
    /// override and no config-file path, so its `0.0` default could not be
    /// changed without editing this crate's source. A computed axis multiplied
    /// by an unreachable zero is dead weight, not a default.
    pub temporal_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::rerank_candidate_max_chars`]
    /// (reranker latency lever 1).
    pub rerank_candidate_max_chars: Option<usize>,
    /// Explicit override for [`PipelineConfig::extraction_arm_budget_ms`].
    ///
    /// Added per a fit check against a real consumer: the config
    /// field's own doc comment says to raise this to 180_000-300_000 for slow
    /// local LLMs "via `.extraction_arm_budget_ms(value)` on the builder" —
    /// but that referred to the *internal* `PipelineConfigBuilder`, which no
    /// public constructor accepted. `Memory::with_ollama()` and
    /// `MemoryBuilder` had no reachable path to this knob at all, so a local
    /// Ollama model (qwen2.5:14b-class, ~43-52s per extraction arm) failed
    /// the ladder outright on ordinary documents under the 30s default. This
    /// field closes that gap the same way `rerank_candidate_max_chars` does.
    pub extraction_arm_budget_ms: Option<u64>,
    /// Explicit override for [`PipelineConfig::prior_turn_replay_depth`].
    /// `None` leaves the default of 10.
    pub prior_turn_replay_depth: Option<usize>,
    /// Explicit override for [`PipelineConfig::contradiction_detection_enabled`].
    ///
    /// The first NON-`SearchConfig` member of this overlay, and the reason the
    /// type is named for `PipelineConfig` rather than `SearchConfig`: the
    /// overlay's job is "sparse programmatic overrides applied onto a
    /// [`PipelineConfigBuilder`] AFTER the env layer", which was always wider
    /// than search. Adding a second parallel overrides mechanism for one field
    /// would have been an avoidable redundancy.
    ///
    /// Why it needed a seam at all: this knob defaults to
    /// **ON**, and it gates a DESTRUCTIVE path (supersession soft-deletes
    /// prior facts). Before this seam existed, the only way to opt out was the process-wide
    /// `KREMORY_CONTRADICTION_DETECTION` env var — which cannot express two
    /// `Memory` instances with different settings, and which a library consumer
    /// should not have to reach for. Its nine sibling knobs all have builder
    /// methods for exactly that reason; this one was simply missed.
    pub contradiction_detection_enabled: Option<bool>,
    /// Explicit override for [`PipelineConfig::secret_scan`]'s
    /// [`SecretScanConfig::enabled`] (TD-061). `None` leaves the default `true`.
    pub secret_scan_enabled: Option<bool>,
    /// Explicit override for [`PipelineConfig::secret_scan`]'s
    /// [`SecretScanConfig::mode`] (TD-061). `None` leaves the default
    /// [`SecretScanMode::FlagOnly`].
    pub secret_scan_mode: Option<SecretScanMode>,
}

impl PipelineConfigOverrides {
    /// Apply only the `Some` fields onto `builder`, each WINNING over
    /// whatever env override (`facade::providers::search_env_overrides`) was
    /// already applied earlier in the same chain — the precedence is:
    /// explicit programmatic config > env override > default. Fields left
    /// `None` here pass `builder` through unchanged, so an env override for a
    /// knob the consumer never touched programmatically still applies. An
    /// all-`None` (default-constructed) `PipelineConfigOverrides` is a strict
    /// no-op, so unset-by-default callers get byte-identical behaviour to
    /// env-only / default.
    pub(crate) fn apply(&self, mut builder: PipelineConfigBuilder) -> PipelineConfigBuilder {
        if let Some(v) = self.content_stream_weight {
            builder = builder.content_stream_weight(v);
        }
        if let Some(v) = self.graph_degree_weight {
            builder = builder.graph_degree_weight(v);
        }
        if let Some(v) = self.rrf_k {
            builder = builder.rrf_k(v);
        }
        if let Some(v) = self.episode_dense_enabled {
            builder = builder.episode_dense_enabled(v);
        }
        if let Some(v) = self.fact_dense_enabled {
            builder = builder.fact_dense_enabled(v);
        }
        if let Some(v) = self.embed_task_prefix_enabled {
            builder = builder.embed_task_prefix_enabled(v);
        }
        if let Some(v) = self.proximity_weight {
            builder = builder.proximity_weight(v);
        }
        if let Some(v) = self.temporal_weight {
            builder = builder.temporal_weight(v);
        }
        if let Some(v) = self.rerank_candidate_max_chars {
            builder = builder.rerank_candidate_max_chars(v);
        }
        if let Some(v) = self.contradiction_detection_enabled {
            builder = builder.contradiction_detection_enabled(v);
        }
        if let Some(v) = self.prior_turn_replay_depth {
            builder = builder.prior_turn_replay_depth(v);
        }
        if let Some(v) = self.extraction_arm_budget_ms {
            builder = builder.extraction_arm_budget_ms(v);
        }
        if let Some(v) = self.secret_scan_enabled {
            builder = builder.secret_scan_enabled(v);
        }
        if let Some(v) = self.secret_scan_mode {
            builder = builder.secret_scan_mode(v);
        }
        builder
    }
}

/// Newtype wrapper for the embedding dimension to make API signatures
/// self-documenting and prevent accidental dimension mismatches.
///
/// Default: 384. Rationale: matches the all-MiniLM-L6-v2 output dimension,
/// which is the default bundled embedding model. Changing this requires
/// a full index rebuild.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingDim(pub usize);

impl Default for EmbeddingDim {
    fn default() -> Self {
        Self(384)
    }
}

/// Entity-resolution call-shape strategy.
///
/// Selects how the ambiguous remainder of ingest-time entity resolution (the
/// entities that survive candidate blocking but are NOT resolved by
/// the cheap deterministic tiers — exact-normalize + MinHash) reaches the LLM:
///
/// - `Batched` (default): one structured-output call per window resolves ALL
///   ambiguous entities against a shared candidate pool at once, collapsing
///   the O(ambiguous × candidates) pairwise fan-out to O(windows). See
///   `resolver_batched.rs`.
/// - `Pairwise`: the earlier behaviour — one LLM `ResolutionVerdict` call
///   per (entity, candidate) pair via `CascadeResolver::resolve`. Retained for
///   A/B comparison and as an instant rollback (config flip, no code revert).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionStrategy {
    /// The earlier pairwise `resolve()` fan-out.
    Pairwise,
    /// Batched structured-output resolution (default).
    #[default]
    Batched,
}

/// Top-level pipeline configuration aggregating all sub-configs.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Dimensionality of embedding vectors produced by the model.
    pub embedding_dim: EmbeddingDim,
    /// LLM-extraction-prompt-window splitting parameters (kind-2 chunking).
    pub extraction_window: ExtractionWindowConfig,
    /// MinHash LSH parameters for near-duplicate detection.
    pub minhash: MinHashConfig,
    /// Entropy pre-filter parameters.
    pub entropy: EntropyConfig,
    /// Ingest-boundary secret/token scan parameters (TD-061).
    pub secret_scan: SecretScanConfig,
    /// Hybrid search fusion parameters.
    pub search: SearchConfig,
    /// Entity types the pipeline will extract and index (empty = all types).
    pub allowed_entity_types: Vec<String>,
    /// Relation / edge types the pipeline will resolve (empty = all types).
    pub allowed_edge_types: Vec<String>,
    /// Entity types that are explicitly excluded even if matched by extraction.
    pub excluded_entity_types: Vec<String>,
    /// Default: 300s. Rationale: 5-minute cache TTL balances freshness against
    /// the cost of re-embedding; longer than a typical meeting turn cadence.
    pub cache_ttl: Duration,
    /// Default: 1000. Rationale: caps resident memory for the speculative cache;
    /// at ~1.5 KB per entry this is ~1.5 MB, acceptable on constrained hardware.
    pub cache_max_entries: usize,
    /// Per-arm wall-clock cap for structured-output extraction (ms).
    ///
    /// Default: 30_000 (30s — production fail-fast). The HTTP layer bounds
    /// individual requests (~10s via LLMBuilder); this caps an arm even if
    /// HTTP succeeds-then-hangs.
    ///
    /// Override for slow local LLMs (qwen2.5:14b ~80-130s per call on Apple
    /// M4 Max 36GB with 32k context) by setting 180_000-300_000 via
    /// `.extraction_arm_budget_ms(value)` on the builder. The benchmark
    /// suite sets this to 300_000 to accommodate qwen2.5:14b warm-up latency.
    pub extraction_arm_budget_ms: u64,
    /// How many preceding episodes of the SAME conversation thread
    /// are replayed into the extraction prompt so that references resolve.
    ///
    /// Default **10**, the value mem0 (`mem0/memory/main.py:920`) and Graphiti
    /// (`graphiti_core/graphiti.py:1086`) independently converged on.
    ///
    /// `0` disables the feature entirely — the query is short-circuited before
    /// it touches the database. That off switch is REQUIRED, not decorative:
    /// without it an A/B of this lever has no control arm, and a measurement
    /// you cannot turn off is a measurement you cannot trust.
    ///
    /// The thread key is `episodes.source_id` (what `.from_chat(id)` sets), so
    /// this is inert for callers who never tag a source — they receive a
    /// random uuid per episode and nothing can match it.
    pub prior_turn_replay_depth: usize,
    /// How many per-chunk extraction LLM calls to run
    /// concurrently within a single `ingest_with` call (`futures::buffered`,
    /// order-preserving for determinism). `1` = the pre-change sequential
    /// behaviour. Extraction is read-only (no persisted-graph dependency
    /// between chunks — `known_entities` becomes window-local when >1, the
    /// staleness the dream L5 backstop absorbs), so overlapping the calls
    /// cuts ingest wall-time once resolution is no longer the bottleneck.
    /// Default: 5 (conservative vs provider concurrent-rate limits;
    /// raise via `KREMORY_EXTRACTION_CONCURRENCY` on a higher-limit provider).
    pub extraction_concurrency: usize,
    /// Entity-resolution candidate-blocking width. When a
    /// group has MORE than this many existing entities, ingest resolution
    /// compares each newly-extracted entity only against a bounded candidate
    /// set — exact normalized-name matches UNION the embedding-ANN top-`k`
    /// nearest existing entities — instead of every existing entity. This
    /// collapses the LLM `ResolutionVerdict` fan-out from O(new × existing) to
    /// O(k). Groups with ≤ this many entities keep the exhaustive (earlier)
    /// comparison, so the change is a no-op on small graphs. Dream-phase L5
    /// canonicalization is the completeness backstop for any match blocking
    /// misses. Default: 10.
    pub resolution_block_k: usize,
    /// Auto-different cosine floor for the embedding-ANN
    /// resolution arm. When a blocked ANN candidate's cosine similarity to the
    /// newly-extracted entity is **below** this floor, it is dropped from the
    /// candidate set (treated as `Different`) WITHOUT an LLM `ResolutionVerdict`
    /// call — the model would almost always say "different" for an embedding-far
    /// pair anyway, so the call is pure cost. This can only ever REDUCE merges
    /// (never create a false one), and dream-phase L5 canonicalization is the
    /// completeness backstop. Exact normalized-name matches are unaffected (they
    /// bypass the floor). Range `[0.0, 1.0]`. Default: `0.0` — OFF (every
    /// blocked candidate still reaches `resolve()`); raise
    /// (e.g. `0.5`) to trade a little recall for far fewer resolution LLM calls.
    pub resolution_min_cosine: f32,
    /// Entity-resolution call-shape. `Batched` (default)
    /// collapses the ambiguous-remainder pairwise LLM fan-out into one
    /// structured call per window; `Pairwise` retains the earlier
    /// per-(entity, candidate) `ResolutionVerdict` call for A/B + rollback.
    pub resolution_strategy: ResolutionStrategy,
    /// Maximum number of ambiguous entities packed into a
    /// single batched-resolution window. Mirrors the `resolution_block_k`
    /// pattern — a safe-default overflow-cap knob, not a token-budget
    /// estimator (YAGNI). Default: 32 — conservative
    /// on any 4k-or-larger-context model (32 short name+type lines, plus
    /// pooled candidates and system prompt, totals roughly 2-3k tokens).
    /// Raise on a large-context model to shrink call count further.
    pub resolution_batch_max_entities: usize,
    /// Run LLM contradiction detection during ingest.
    /// **Default: `true`.**
    ///
    /// ## History — this default moved TWICE in one day, and both moves were right
    ///
    /// **OFF (rev.1)** — the mechanism DESTROYED SET-VALUED FACTS. It treated every
    /// predicate as functional (one value per subject), so ingesting a list
    /// superseded all but the last member. Measured on 8 LongMemEval sessions:
    /// 81 of 1,021 facts invalidated, ≥31% provably multi-valued rather than
    /// contradictory — `has_performer: billie eilish / tove lo / lana del rey` all
    /// superseded by `the 1975`; `contain: rolled oats` superseded by `seeds`.
    ///
    /// **ON (rev.2)** — the cause was the PROMPT, not the model: it instructed
    /// *"if the new fact is an UPDATE (same relationship but newer value), return
    /// that index too"*, which describes every item in a list. Replaced by a
    /// coexistence question (*"can both be true at the same time?"*) plus a
    /// temporal-ordering tiebreak. Measured on the production model: **7/8
    /// set-valued destroyed → 0/8**, total errors 7 → 2; a corpus run moved the
    /// contradiction rate 38.8% → 11.3% with no multi-valued predicate destroyed.
    /// Scope also moved MVP → v1, and LongMemEval's `knowledge-update` category
    /// (78 of 500 questions) tests exactly this mechanism.
    ///
    /// ⚠️ **A high contradiction rate is a WARNING, not a success metric.** The
    /// original bug was found only because 43% looked "good". On coherent
    /// conversational data, genuine contradictions are rare.
    ///
    /// ## Turning it off
    ///
    /// `KREMORY_CONTRADICTION_DETECTION=0` (or the builder setter). Off is safe
    /// and loses only auto-correction of stale facts: supersession is a SOFT
    /// delete (`invalid_at` is set, the row stays) and
    /// `core::dream::provenance::reversal::unsupersede` reverses
    /// individual false positives, so nothing written under either default is
    /// unrecoverable.
    ///
    /// **Note on the deeper fix (still open):** derived predicate
    /// cardinality — the approach an earlier version of this comment pointed at —
    /// was investigated and **REFUTED**: 67% of predicates appear exactly once in
    /// a real corpus and only 6% ever show multi-valuedness, so there is nothing
    /// to derive from. The open work is the residual 2-of-6 missed genuine
    /// updates (`works_at`, `current_job_title`), which is an intent problem, not
    /// a cardinality one.
    pub contradiction_detection_enabled: bool,
}

/// The compiled-in default for [`PipelineConfig::contradiction_detection_enabled`],
/// as a `const` so it has exactly ONE definition.
///
/// This value is needed in a second place — the [`GraphHandle`](crate::memory::graph::GraphHandle)
/// trait default, so stub/test handles that carry no `Engine` report what a real
/// one would. Restating `true` there would have created a second source of truth
/// for a flag whose default **has already moved twice in one day** (OFF for a
/// few hours, then back ON), and the copy that drifts is the one reporting a
/// DESTRUCTIVE setting as disabled.
pub(crate) const DEFAULT_CONTRADICTION_DETECTION_ENABLED: bool = true;

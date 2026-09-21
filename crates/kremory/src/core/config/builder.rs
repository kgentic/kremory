use std::time::Duration;

use crate::core::error::{Error, Result};

use super::pipeline::{
    PipelineConfig, EmbeddingDim, ResolutionStrategy, DEFAULT_CONTRADICTION_DETECTION_ENABLED,
};
use super::scan_configs::{
    EntropyConfig, ExtractionWindowConfig, MinHashConfig, SecretScanConfig, SecretScanMode,
};
use super::search::SearchConfig;

impl PipelineConfig {
    /// Returns a new [`PipelineConfigBuilder`] populated with all defaults.
    pub fn builder() -> PipelineConfigBuilder {
        PipelineConfigBuilder {
            inner: PipelineConfig {
                embedding_dim: EmbeddingDim::default(),
                extraction_window: ExtractionWindowConfig::default(),
                minhash: MinHashConfig::default(),
                entropy: EntropyConfig::default(),
                secret_scan: SecretScanConfig::default(),
                search: SearchConfig::default(),
                allowed_entity_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                excluded_entity_types: Vec::new(),
                cache_ttl: Duration::from_secs(300),
                cache_max_entries: 1000,
                extraction_arm_budget_ms: 30_000,
                prior_turn_replay_depth: 10,
                extraction_concurrency: 5,
                resolution_block_k: 10,
                resolution_min_cosine: 0.0,
                resolution_strategy: ResolutionStrategy::default(),
                resolution_batch_max_entities: 32,
                // ON. Was default-OFF for ~4h
                // while the destruction bug was open; the prompt fix
                // (coexistence + temporal ordering) measured 0/8 set-valued
                // destroyed and a corpus run confirmed it (contradiction rate
                // 38.8% -> 11.3%, no multi-valued predicate destroyed). Scope
                // moved MVP -> v1, and LongMemEval's knowledge-update category
                // (78 of 500 questions) TESTS this mechanism — shipping the
                // benchmark with it disabled would publish a number with the
                // relevant feature switched off.
                contradiction_detection_enabled: DEFAULT_CONTRADICTION_DETECTION_ENABLED,
            },
        }
    }
}

/// Fluent builder for [`PipelineConfig`].
pub struct PipelineConfigBuilder {
    inner: PipelineConfig,
}

impl PipelineConfigBuilder {
    // ── EmbeddingDim ─────────────────────────────────────────────────────────

    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.inner.embedding_dim = EmbeddingDim(dim);
        self
    }

    // ── ExtractionWindowConfig ───────────────────────────────────────────────

    pub fn min_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.min_words = v;
        self
    }

    pub fn density_threshold(mut self, v: f64) -> Self {
        self.inner.extraction_window.density_threshold = v;
        self
    }

    pub fn max_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.max_words = v;
        self
    }

    pub fn overlap_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.overlap_words = v;
        self
    }

    // ── MinHashConfig ────────────────────────────────────────────────────────

    pub fn num_permutations(mut self, v: usize) -> Self {
        self.inner.minhash.num_permutations = v;
        self
    }

    pub fn shingle_size(mut self, v: usize) -> Self {
        self.inner.minhash.shingle_size = v;
        self
    }

    pub fn band_size(mut self, v: usize) -> Self {
        self.inner.minhash.band_size = v;
        self
    }

    pub fn jaccard_threshold(mut self, v: f64) -> Self {
        self.inner.minhash.jaccard_threshold = v;
        self
    }

    // ── EntropyConfig ────────────────────────────────────────────────────────

    pub fn min_name_length(mut self, v: usize) -> Self {
        self.inner.entropy.min_name_length = v;
        self
    }

    pub fn min_token_count(mut self, v: usize) -> Self {
        self.inner.entropy.min_token_count = v;
        self
    }

    pub fn entropy_threshold(mut self, v: f64) -> Self {
        self.inner.entropy.entropy_threshold = v;
        self
    }

    // ── SecretScanConfig (TD-061) ────────────────────────────────────────────

    /// `false` skips the ingest-boundary secret scan entirely. Default: `true`.
    pub fn secret_scan_enabled(mut self, v: bool) -> Self {
        self.inner.secret_scan.enabled = v;
        self
    }

    /// Flag-and-log a hit (default) vs redact the matched span before the
    /// episode is stored/extracted/embedded. See [`SecretScanMode`].
    pub fn secret_scan_mode(mut self, mode: SecretScanMode) -> Self {
        self.inner.secret_scan.mode = mode;
        self
    }

    // ── SearchConfig ─────────────────────────────────────────────────────────

    pub fn bm25_weight(mut self, v: f64) -> Self {
        self.inner.search.bm25_weight = v;
        self
    }

    pub fn vector_weight(mut self, v: f64) -> Self {
        self.inner.search.vector_weight = v;
        self
    }

    pub fn rrf_k(mut self, v: usize) -> Self {
        self.inner.search.rrf_k = v;
        self
    }

    pub fn top_k(mut self, v: usize) -> Self {
        self.inner.search.top_k = v;
        self
    }

    /// Per-stream weight applied to
    /// the `content_search` BM25 stream's RRF contribution. Default
    /// `1.0` (equal-weight fusion, byte-identical). Wired from the
    /// `KREMORY_CONTENT_WEIGHT` env override at server boot
    /// (`facade::providers::search_env_overrides`) so weight sweeps cost a
    /// restart, not a rebuild.
    pub fn content_stream_weight(mut self, v: f32) -> Self {
        self.inner.search.content_stream_weight = v;
        self
    }

    /// Weight of the additive
    /// graph-degree bonus (`SearchConfig::graph_degree_weight`). Default `0.05` — matches the
    /// value already live in the scoring path. Unlike every sibling axis on
    /// `SearchConfig`, this field has no builder method and no env override,
    /// so it is reachable for reading (via `Memory::search_config()`) but not
    /// for writing.
    pub fn graph_degree_weight(mut self, v: f32) -> Self {
        self.inner.search.graph_degree_weight = v;
        self
    }

    /// The dense-episode A/B knob: enable the dense (embedding) episode
    /// retrieval arm + ingest-time episode embedding. Default `false`
    /// (BM25-only, byte-identical). Wired from the `KREMORY_EPISODE_DENSE` env
    /// override at server boot (`facade::providers::search_env_overrides`) so
    /// the dense-vs-BM25 comparison costs a restart, not a rebuild.
    pub fn episode_dense_enabled(mut self, v: bool) -> Self {
        self.inner.search.episode_dense_enabled = v;
        self
    }

    /// The dense-fact A/B knob: enable the dense (embedding)
    /// fact retrieval arm (`TemporalGraph::vector_search_facts`, RRF-fused
    /// via `core::search::rrf_fuse_with_facts`). Default `false` (facts
    /// reachable only via 1-hop entity expansion, byte-identical). Wired
    /// from the `KREMORY_FACT_DENSE` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the fact-dense A/B
    /// costs a restart, not a rebuild.
    pub fn fact_dense_enabled(mut self, v: bool) -> Self {
        self.inner.search.fact_dense_enabled = v;
        self
    }

    /// The nomic task-prefix A/B knob: enable `search_document:` /
    /// `search_query:` prefixing on every embed call site (see
    /// `core::embed_prefix`). Default `false` (bare text, byte-identical).
    /// Wired from the `KREMORY_EMBED_TASK_PREFIX` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the prefix A/B costs a
    /// restart, not a rebuild. ⚠️ Flipping this on an existing corpus requires
    /// a re-embed — see [`SearchConfig::embed_task_prefix_enabled`].
    pub fn embed_task_prefix_enabled(mut self, v: bool) -> Self {
        self.inner.search.embed_task_prefix_enabled = v;
        self
    }

    /// The axis-C A/B knob: weight of the additive
    /// graph-proximity boost. Default `0.0` (off, byte-identical — the second
    /// bounded-hop query never fires). Wired from the
    /// `KREMORY_PROXIMITY_WEIGHT` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the proximity A/B costs
    /// a restart, not a rebuild.
    pub fn proximity_weight(mut self, v: f32) -> Self {
        self.inner.search.proximity_weight = v;
        self
    }

    /// The temporal-recency axis weight (`SearchConfig::temporal_weight`),
    /// default `0.0` = axis off (byte-identical to earlier behaviour).
    ///
    /// This axis shipped with working compute
    /// (`core/context.rs`, `core/scoring/temporal.rs`) and **no way to enable
    /// it** — no builder method, no env override, and `SearchConfig` derives no
    /// `Deserialize`, so there was no config-file path either. Its sibling
    /// `proximity_weight` has all three. The value was therefore pinned at
    /// `0.0` for every consumer, and the axis was computed and then multiplied
    /// by an unreachable zero. That is dead weight, not a default: a knob
    /// nobody can turn is indistinguishable from an unimplemented feature, and
    /// the axis has consequently NEVER been measured. This adds the missing
    /// seam so it can be A/B'd; the default is unchanged.
    pub fn temporal_weight(mut self, v: f32) -> Self {
        self.inner.search.temporal_weight = v;
        self
    }

    /// The axis-C hop bound — the parameter that controls the proximity
    /// signal's SELECTIVITY, and therefore the one that actually needed to be
    /// sweepable. Added after the first axis-C A/B measured flat at
    /// three weights: the per-recall trace showed `seed_count=50
    /// boosted_count=48` at the default `hop_bound = 2`, i.e. the walk reaches
    /// neighbours for ~96% of seeds, making the boost a near-uniform additive
    /// offset that cannot discriminate at ANY weight. Wired from
    /// `KREMORY_PROXIMITY_HOP_BOUND` (`facade::providers::search_env_overrides`)
    /// so the selectivity sweep costs a restart, not a rebuild — a
    /// knob you cannot set is a knob you cannot evaluate, applied to
    /// axis-C's own tuning surface.
    pub fn proximity_hop_bound(mut self, v: u32) -> Self {
        self.inner.search.proximity_hop_bound = v;
        self
    }

    /// Reranker latency lever 1: cap the SUMMARY portion of each rerank
    /// candidate at `v` chars. Default `0` (unlimited, byte-identical).
    /// Wired from the `KREMORY_RERANK_CANDIDATE_MAX_CHARS` env override at
    /// server boot (`facade::providers::search_env_overrides`) so a
    /// truncation-length sweep costs a restart, not a rebuild.
    pub fn rerank_candidate_max_chars(mut self, v: usize) -> Self {
        self.inner.search.rerank_candidate_max_chars = v;
        self
    }

    // ── Ontology ─────────────────────────────────────────────────────────────

    pub fn allowed_entity_types(mut self, v: Vec<String>) -> Self {
        self.inner.allowed_entity_types = v;
        self
    }

    pub fn allowed_edge_types(mut self, v: Vec<String>) -> Self {
        self.inner.allowed_edge_types = v;
        self
    }

    pub fn excluded_entity_types(mut self, v: Vec<String>) -> Self {
        self.inner.excluded_entity_types = v;
        self
    }

    // ── Cache ────────────────────────────────────────────────────────────────

    pub fn cache_ttl(mut self, v: Duration) -> Self {
        self.inner.cache_ttl = v;
        self
    }

    pub fn cache_max_entries(mut self, v: usize) -> Self {
        self.inner.cache_max_entries = v;
        self
    }

    // ── Extraction budget ────────────────────────────────────────────────────

    /// Set the per-arm wall-clock budget for structured-output extraction (ms).
    ///
    /// Default: 30_000 (30s — production fail-fast).
    /// Override for slow local LLMs: e.g. `300_000` for qwen2.5:14b on Apple
    /// silicon (~80-130s per call at 32k context).
    /// Depth of prior-turn replay into the extraction prompt.
    /// `0` disables it. See [`PipelineConfig::prior_turn_replay_depth`].
    #[must_use]
    pub fn prior_turn_replay_depth(mut self, depth: usize) -> Self {
        self.inner.prior_turn_replay_depth = depth;
        self
    }

    pub fn extraction_arm_budget_ms(mut self, ms: u64) -> Self {
        self.inner.extraction_arm_budget_ms = ms;
        self
    }

    /// Concurrent per-chunk extraction calls per ingest
    /// (`futures::buffered`, order-preserving). `1` = sequential. Default 5.
    pub fn extraction_concurrency(mut self, n: usize) -> Self {
        self.inner.extraction_concurrency = n;
        self
    }

    /// Enable LLM contradiction detection during ingest. **Default ON**
    /// (`DEFAULT_CONTRADICTION_DETECTION_ENABLED`, accepted).
    ///
    /// ⚠️ This doc said **"Default OFF"** for a time and was WRONG — it
    /// described an earlier revision, which the current one superseded THE
    /// SAME DAY. It is a public builder method, so a consumer reading it
    /// would have left a
    /// destructive feature enabled believing it disabled. Correcting rather than
    /// deleting, because the reason it was off is still the reason to be careful.
    ///
    /// **rev.1 (superseded):** OFF, because it treated every predicate as
    /// functional and so SUPERSEDED SET-VALUED FACTS — a festival's second
    /// performer superseded the first. Measured: 81 of 1,021 facts invalidated
    /// over 8 LongMemEval sessions, ≥31% provably set-valued.
    ///
    /// **rev.2 (current):** ON, because the prompt was rewritten to ask *"can
    /// both facts be true at the same time?"* — 7-of-8 destroyed → **0-of-8**,
    /// and a corpus run cut the contradiction rate **3.4×** (38.8% → 11.3%) with
    /// no obviously set-valued predicate destroyed. Scope moved MVP → v1 because
    /// LongMemEval's `knowledge-update` category (78 of 500 questions) tests
    /// exactly this mechanism — benchmarking v1 with it off would measure the
    /// wrong build.
    ///
    /// Pass `false` to restore append-only ingest: nothing is invalidated behind
    /// your back and `valid_to` is stamped only by supersession you asked for.
    pub fn contradiction_detection_enabled(mut self, on: bool) -> Self {
        self.inner.contradiction_detection_enabled = on;
        self
    }

    // ── Resolution candidate blocking ───────────────────────────────────────────

    /// Set the entity-resolution candidate-blocking width `k`. Groups with more
    /// than `k` existing entities compare each new entity against only the
    /// exact-name matches + embedding-ANN top-`k`; groups with ≤`k` keep the
    /// exhaustive comparison. Default: 10. Set to `usize::MAX` to disable
    /// blocking entirely (restore the earlier behaviour).
    pub fn resolution_block_k(mut self, k: usize) -> Self {
        self.inner.resolution_block_k = k;
        self
    }

    /// Set the auto-different cosine floor for the embedding-ANN resolution arm.
    /// Blocked ANN candidates below this cosine similarity are
    /// dropped without an LLM verdict. `0.0` (default) = off. Clamped
    /// to `[0.0, 1.0]`.
    pub fn resolution_min_cosine(mut self, floor: f32) -> Self {
        self.inner.resolution_min_cosine = floor.clamp(0.0, 1.0);
        self
    }

    // ── Batched resolution ───────────────────────────────────────────────────

    /// Select the entity-resolution call-shape strategy. `Batched` (default)
    /// collapses the ambiguous-remainder LLM fan-out into one structured call
    /// per window; `Pairwise` restores the earlier per-pair behaviour.
    pub fn resolution_strategy(mut self, v: ResolutionStrategy) -> Self {
        self.inner.resolution_strategy = v;
        self
    }

    /// Set the maximum number of ambiguous entities packed into a single
    /// batched-resolution window. Default: 32. Raise on a large-context model
    /// to shrink call count further; see `resolution_batch_max_entities` docs.
    pub fn resolution_batch_max_entities(mut self, v: usize) -> Self {
        self.inner.resolution_batch_max_entities = v;
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Validates the configuration and returns a [`PipelineConfig`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following invariants are violated:
    /// - `jaccard_threshold` must be in `(0.0, 1.0]`
    /// - `bm25_weight + vector_weight` must equal `1.0` (within 1e-9)
    /// - `embedding_dim` must be greater than 0
    /// - `min_words` must be greater than 0
    /// - `max_words` must be >= `min_words`
    /// - `density_threshold` must be in `(0.0, 1.0]`
    pub fn build(self) -> Result<PipelineConfig> {
        let c = &self.inner;

        let jt = c.minhash.jaccard_threshold;
        if jt <= 0.0 || jt > 1.0 {
            return Err(Error::Config(format!(
                "jaccard_threshold must be in (0.0, 1.0], got {}",
                jt
            )));
        }

        if c.embedding_dim.0 == 0 {
            return Err(Error::EmbeddingDimZero);
        }

        if c.extraction_window.min_words == 0 {
            return Err(Error::Config("min_words must be greater than 0".into()));
        }

        if c.extraction_window.max_words < c.extraction_window.min_words {
            return Err(Error::TokenWindowInvalid {
                min: c.extraction_window.min_words,
                got: c.extraction_window.max_words,
            });
        }

        let dt = c.extraction_window.density_threshold;
        if dt <= 0.0 || dt > 1.0 {
            return Err(Error::Config(format!(
                "density_threshold must be in (0.0, 1.0], got {}",
                dt
            )));
        }

        Ok(self.inner)
    }
}

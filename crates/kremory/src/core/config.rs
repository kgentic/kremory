use std::time::Duration;

use crate::core::error::{Error, Result};

/// The content type of a document being ingested into the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    /// Plain unstructured text (e.g. transcripts, notes).
    Text,
    /// A discrete conversational message (e.g. chat turn, email).
    Message,
    /// Structured JSON payload; entity extraction is schema-aware.
    Json,
    /// A standalone document (e.g. markdown file, report, wiki page).
    /// Used by `ingest_document()` — stored as a searchable entity with
    /// full-text embedding in addition to extracted sub-entities.
    Document,
}

/// LLM-extraction-prompt-window parameters.
///
/// **This is kind-2 chunking only** — slices an oversized episode body into prompt-sized
/// windows so the extractor LLM can read it within its context budget. Slices are throwaway
/// and never enter storage/embedding. See `core/extraction_window.rs` module docstring and
/// ADR-Phase-D.0:88 for the kind-1 vs kind-2 distinction.
#[derive(Debug, Clone)]
pub struct ExtractionWindowConfig {
    /// Default: 100 words. Shorter text rarely benefits from splitting; below
    /// this the overhead of extra chunks exceeds the gain.  Graphiti ratio:
    /// min/max ≈ 33%.
    pub min_tokens: usize,

    /// Default: 0.15. Empirically, chunks with >15% of tokens being entity
    /// spans lose inter-entity context when kept whole; splitting at this
    /// threshold keeps entity co-occurrence coherent.
    pub density_threshold: f64,

    /// Default: 300 words (~1500 chars, ~400 BPE tokens).  Sized so 3 chunks
    /// fit in the default 4096-token context with room for system prompt,
    /// query, and generation.  Optimised for latency on the real-time meeting
    /// assistant path.  Users with more memory can increase via
    /// `LLM_CONTEXT_SIZE` + `CHUNK_MAX_TOKENS` env vars (see KGT-69 for UI
    /// presets).  Must stay aligned with `max_chunk_chars` in
    /// `rust-pipeline::PipelineConfig` (1500 chars).
    pub max_tokens: usize,

    /// Number of words from the end of `chunk[i]` to prepend to `chunk[i+1]`.
    /// Default: 50 words.  Graphiti uses 200/3000 (6.7%); ours is 50/300
    /// (16.7%) — slightly higher overlap compensates for smaller chunks.
    pub overlap_tokens: usize,
}

impl ExtractionWindowConfig {
    /// Build from environment variables, falling back to sensible defaults.
    ///
    /// | Env var | Default | Rationale |
    /// |---------|---------|-----------|
    /// | `CHUNK_MAX_TOKENS` | 300 | ~1500 chars, fits 3 chunks in 4096-ctx prompt |
    /// | `CHUNK_MIN_TOKENS` | 100 | Don't chunk short text (Graphiti min/max ≈ 33%) |
    /// | `CHUNK_OVERLAP_TOKENS` | 50 | Context continuity between chunks |
    /// | `CHUNK_DENSITY_THRESHOLD` | 0.15 | Entity-dense regions trigger splitting |
    pub fn from_env() -> Self {
        Self {
            max_tokens: env_usize("CHUNK_MAX_TOKENS", 300),
            min_tokens: env_usize("CHUNK_MIN_TOKENS", 100),
            overlap_tokens: env_usize("CHUNK_OVERLAP_TOKENS", 50),
            density_threshold: env_f64("CHUNK_DENSITY_THRESHOLD", 0.15),
        }
    }
}

impl Default for ExtractionWindowConfig {
    fn default() -> Self {
        Self {
            min_tokens: 100,
            density_threshold: 0.15,
            max_tokens: 300,
            overlap_tokens: 50,
        }
    }
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// MinHash / LSH parameters used for near-duplicate entity detection.
#[derive(Debug, Clone)]
pub struct MinHashConfig {
    /// Default: 32. Rationale: 32 permutations give ~3% Jaccard estimation
    /// error at manageable memory cost (~256 bytes per sketch).
    pub num_permutations: usize,

    /// Default: 3. Rationale: character 3-grams balance sensitivity to small
    /// edits (typos, abbreviations) against noise from very short substrings.
    pub shingle_size: usize,

    /// Default: 4. Rationale: with 32 permutations and bands of 4, we get
    /// 8 bands, yielding a good probability curve around the 0.9 threshold
    /// (P(candidate) ≈ 0.99 at threshold, ~0.01 false-positive rate at 0.5).
    pub band_size: usize,

    /// Default: 0.9. Rationale: entity surface forms that share ≥90% of their
    /// 3-gram shingles are treated as the same entity; below 0.9 too many
    /// distinct entities collapse.
    pub jaccard_threshold: f64,
}

impl Default for MinHashConfig {
    fn default() -> Self {
        Self {
            num_permutations: 32,
            shingle_size: 3,
            band_size: 4,
            jaccard_threshold: 0.9,
        }
    }
}

/// Entropy-based pre-filter that gates whether a token is fed into MinHash.
/// Low-entropy strings (e.g. "Inc.", "Ltd.") are common suffixes that would
/// inflate false-positive collision rates if hashed directly.
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Default: 6. Rationale: entity names shorter than 6 characters are almost
    /// always abbreviations or stop-words; hashing them adds noise without value.
    pub min_name_length: usize,

    /// Default: 2. Rationale: a single-token string is almost never a meaningful
    /// multi-word entity; requiring at least 2 whitespace-delimited tokens
    /// removes most numeric codes and single-letter abbreviations.
    pub min_token_count: usize,

    /// Default: 1.5. Rationale: Shannon entropy of 1.5 bits corresponds roughly
    /// to strings that repeat fewer than 3 distinct characters — effectively
    /// keyboard-mash or padded identifiers that carry no semantic content.
    pub entropy_threshold: f64,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        }
    }
}

/// Hybrid search parameters controlling how BM25 and vector scores are combined.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Default: 0.5. Rationale: equal weighting between BM25 and vector search
    /// is a safe baseline; BM25 handles keyword precision while vector search
    /// handles semantic similarity. Must sum to 1.0 with `vector_weight`.
    pub bm25_weight: f64,

    /// Default: 0.5. Rationale: see `bm25_weight`. Must sum to 1.0 with
    /// `bm25_weight`.
    pub vector_weight: f64,

    /// Default: 60. Rationale: the standard RRF constant from Cormack et al.
    /// (2009); k=60 was shown to be near-optimal across a wide range of
    /// retrieval tasks.
    pub rrf_k: usize,

    /// Default: 10. Rationale: top-10 is the conventional precision@k cut-off
    /// for downstream synthesis; returning more increases context window cost
    /// without commensurate quality gain.
    pub top_k: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 0.5,
            vector_weight: 0.5,
            rrf_k: 60,
            top_k: 10,
        }
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
}

impl PipelineConfig {
    /// Returns a new [`PipelineConfigBuilder`] populated with all defaults.
    pub fn builder() -> PipelineConfigBuilder {
        PipelineConfigBuilder {
            inner: PipelineConfig {
                embedding_dim: EmbeddingDim::default(),
                extraction_window: ExtractionWindowConfig::default(),
                minhash: MinHashConfig::default(),
                entropy: EntropyConfig::default(),
                search: SearchConfig::default(),
                allowed_entity_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                excluded_entity_types: Vec::new(),
                cache_ttl: Duration::from_secs(300),
                cache_max_entries: 1000,
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

    pub fn min_tokens(mut self, v: usize) -> Self {
        self.inner.extraction_window.min_tokens = v;
        self
    }

    pub fn density_threshold(mut self, v: f64) -> Self {
        self.inner.extraction_window.density_threshold = v;
        self
    }

    pub fn max_tokens(mut self, v: usize) -> Self {
        self.inner.extraction_window.max_tokens = v;
        self
    }

    pub fn overlap_tokens(mut self, v: usize) -> Self {
        self.inner.extraction_window.overlap_tokens = v;
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

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Validates the configuration and returns a [`PipelineConfig`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following invariants are violated:
    /// - `jaccard_threshold` must be in `(0.0, 1.0]`
    /// - `bm25_weight + vector_weight` must equal `1.0` (within 1e-9)
    /// - `embedding_dim` must be greater than 0
    /// - `min_tokens` must be greater than 0
    /// - `max_tokens` must be >= `min_tokens`
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

        let weight_sum = c.search.bm25_weight + c.search.vector_weight;
        if (weight_sum - 1.0_f64).abs() > 1e-9 {
            return Err(Error::WeightSumInvalid {
                bm25: c.search.bm25_weight,
                vector: c.search.vector_weight,
            });
        }

        if c.embedding_dim.0 == 0 {
            return Err(Error::EmbeddingDimZero);
        }

        if c.extraction_window.min_tokens == 0 {
            return Err(Error::Config("min_tokens must be greater than 0".into()));
        }

        if c.extraction_window.max_tokens < c.extraction_window.min_tokens {
            return Err(Error::TokenWindowInvalid {
                min: c.extraction_window.min_tokens,
                got: c.extraction_window.max_tokens,
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_builds() {
        let result = PipelineConfig::builder().build();
        assert!(result.is_ok(), "default config should build without error");
    }

    #[test]
    fn test_invalid_jaccard_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(2.0).build();
        assert!(result.is_err(), "jaccard_threshold > 1.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_jaccard_zero_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(0.0).build();
        assert!(result.is_err(), "jaccard_threshold = 0.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_weights_rejected() {
        let result = PipelineConfig::builder()
            .bm25_weight(0.3)
            .vector_weight(0.3)
            .build();
        assert!(result.is_err(), "weights summing to 0.6 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("bm25_weight") || msg.contains("vector_weight"),
            "error should mention weight fields"
        );
    }

    #[test]
    fn test_invalid_embedding_dim_rejected() {
        let result = PipelineConfig::builder().embedding_dim(0).build();
        assert!(result.is_err(), "embedding_dim = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("embedding_dim"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_min_tokens_rejected() {
        let result = PipelineConfig::builder().min_tokens(0).build();
        assert!(result.is_err(), "min_tokens = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("min_tokens"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_max_less_than_min_rejected() {
        let result = PipelineConfig::builder()
            .min_tokens(800)
            .max_tokens(400)
            .build();
        assert!(result.is_err(), "max_tokens < min_tokens must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("max_tokens") || msg.contains("min_tokens"),
            "error should mention token fields"
        );
    }

    #[test]
    fn test_custom_config_builds() {
        let cfg = PipelineConfig::builder()
            .embedding_dim(768)
            .min_tokens(200)
            .max_tokens(600)
            .density_threshold(0.2)
            .num_permutations(64)
            .shingle_size(4)
            .band_size(8)
            .jaccard_threshold(0.85)
            .min_name_length(4)
            .min_token_count(1)
            .entropy_threshold(1.0)
            .bm25_weight(0.7)
            .vector_weight(0.3)
            .rrf_k(30)
            .top_k(5)
            .cache_ttl(Duration::from_secs(60))
            .cache_max_entries(500)
            .build()
            .expect("custom config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(768));
        assert_eq!(cfg.extraction_window.min_tokens, 200);
        assert_eq!(cfg.extraction_window.max_tokens, 600);
        assert!((cfg.extraction_window.density_threshold - 0.2).abs() < 1e-12);
        assert_eq!(cfg.minhash.num_permutations, 64);
        assert_eq!(cfg.minhash.shingle_size, 4);
        assert_eq!(cfg.minhash.band_size, 8);
        assert!((cfg.minhash.jaccard_threshold - 0.85).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 4);
        assert_eq!(cfg.entropy.min_token_count, 1);
        assert!((cfg.entropy.entropy_threshold - 1.0).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.7).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.3).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 30);
        assert_eq!(cfg.search.top_k, 5);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(60));
        assert_eq!(cfg.cache_max_entries, 500);
    }

    #[test]
    fn test_default_values_correct() {
        let cfg = PipelineConfig::builder()
            .build()
            .expect("default config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(384));
        assert_eq!(cfg.extraction_window.min_tokens, 100);
        assert!((cfg.extraction_window.density_threshold - 0.15).abs() < 1e-12);
        assert_eq!(cfg.extraction_window.max_tokens, 300);
        assert_eq!(cfg.minhash.num_permutations, 32);
        assert_eq!(cfg.minhash.shingle_size, 3);
        assert_eq!(cfg.minhash.band_size, 4);
        assert!((cfg.minhash.jaccard_threshold - 0.9).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 6);
        assert_eq!(cfg.entropy.min_token_count, 2);
        assert!((cfg.entropy.entropy_threshold - 1.5).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.5).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.5).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 60);
        assert_eq!(cfg.search.top_k, 10);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(300));
        assert_eq!(cfg.cache_max_entries, 1000);
    }

    #[test]
    fn test_ontology_config() {
        let allowed = vec!["Person".to_string(), "Organization".to_string()];
        let excluded = vec!["StopWord".to_string()];
        let edges = vec!["WORKS_AT".to_string()];

        let cfg = PipelineConfig::builder()
            .allowed_entity_types(allowed.clone())
            .excluded_entity_types(excluded.clone())
            .allowed_edge_types(edges.clone())
            .build()
            .expect("ontology config should build");

        assert_eq!(cfg.allowed_entity_types, allowed);
        assert_eq!(cfg.excluded_entity_types, excluded);
        assert_eq!(cfg.allowed_edge_types, edges);

        // Verify that allowed and excluded are independent — a type can
        // appear in excluded without being in allowed.
        assert!(!cfg.allowed_entity_types.contains(&"StopWord".to_string()));
        assert!(cfg.excluded_entity_types.contains(&"StopWord".to_string()));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Config — core-layer telemetry prefix config (ADR D15)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core-layer telemetry configuration.
///
/// Controls the `metrics_prefix` and `span_prefix` namespace so callers can
/// co-deploy multiple kremory instances without metric label collision (ADR D15).
///
/// Default prefixes match the canonical names in `monitoring/kremory-memory-slos.toml`.
/// Override only when running multiple kremory deployments in the same Prometheus
/// namespace (e.g. staging vs prod scraping into one cluster).
///
/// # Cardinality note (ADR D7)
///
/// Prefixes are `Option<String>` set once at startup — not per-request strings.
/// The prefix is prepended to the base metric name at registration time, not at
/// emit time, so there is no per-call allocation overhead.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Optional prefix prepended to all `metrics::counter!/histogram!/gauge!` names.
    ///
    /// Example: `Some("kremory_prod".to_string())` → `kremory_prod_core_tokens_total`.
    /// `None` (default) uses the canonical `kremory_core_*` namespace.
    pub metrics_prefix: Option<String>,

    /// Optional prefix prepended to all `tracing::info!/warn!/error!` span names.
    ///
    /// Example: `Some("prod".to_string())` → `prod.kremory.embed completed`.
    /// `None` (default) uses the canonical `kremory.*` span namespace.
    pub span_prefix: Option<String>,
}

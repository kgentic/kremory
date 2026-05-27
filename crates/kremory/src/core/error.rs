use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Status of a Phase 2 (per-episode enrichment) run.
///
/// Per ADR D.6.3 — canonical ingest status enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IngestStatus {
    Pending,
    Extracting,
    Deduplicating,
    Invalidating,
    Complete,
    Failed(String),
}

/// Error kind for per-entity/edge ingestion failures during Phase 2 enrichment.
///
/// Per ADR D.6.3 — G2.1 prior-art finding: 4 of 7 surveyed systems make
/// errors first-class. Each variant carries structured context for observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IngestionErrorKind {
    ValidationFailed {
        reason: String,
    },
    ProviderError {
        provider_name: String,
        detail: String,
    },
    RateLimited {
        retry_after: Option<std::time::Duration>,
    },
    ParseFailure {
        stage: String,
        detail: String,
    },
    SchemaViolation {
        field: String,
        expected: String,
    },
}

/// How a contradiction between facts was resolved during Phase 2.
///
/// Per ADR D.6.3 / §2.8 — G2.2 prior-art finding: no surveyed system
/// distinguishes supersession from contradiction as separate event types.
/// `Flagged` variant deferred to v0.y pending consumer UX validation.
/// `#[non_exhaustive]` ensures adding `Flagged` is additive, not breaking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ContradictionResolution {
    /// Prior fact marked `invalid_at = now`; new fact becomes authoritative.
    Superseded,
    /// Prior fact kept; new fact discarded (low confidence).
    Retained,
    /// Facts combined into a richer representation.
    Merged,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("database error: {0}")]
    Database(#[from] libsql::Error),

    #[error("extraction failed: {0}")]
    Extraction(String),

    #[error("entity resolution failed: {0}")]
    Resolution(String),

    #[error("search error: {0}")]
    Search(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),

    // ── Named struct variants (Story #155) ────────────────────────────────────
    /// SQLite INSERT succeeded but `SELECT last_insert_rowid()` returned no row.
    /// `operation` names the specific insert path (e.g. "insert_fact",
    /// "insert_episode_with_group") so callers can correlate with metrics.
    #[error("database invariant violated: no rowid returned after {operation}")]
    InsertReturnedNoRowId { operation: &'static str },

    /// `embedding_dim` was configured as zero — embeddings require at least one
    /// dimension. Distinct from `Config(String)` so callers can match precisely
    /// without substring parsing.
    #[error("embedding_dim must be greater than 0")]
    EmbeddingDimZero,

    /// An LLM call during the multi-stage extraction pipeline failed.
    /// `stage` identifies which pipeline step failed ("entities", "relations",
    /// "triplets", …); `detail` carries the underlying provider error message.
    #[error("extraction stage '{stage}' failed: {detail}")]
    ExtractionStage { stage: String, detail: String },

    /// BM25 and vector search weights no longer need to sum to 1.0; they are
    /// independent RRF multipliers as of v0.1.1. This variant is retained for
    /// backward compatibility with any downstream code that matches on it, but
    /// it is never emitted by the library.
    #[deprecated(
        since = "0.1.1",
        note = "RRF weights are independent multipliers; no sum constraint. Variant retained for backward compat; never emitted."
    )]
    #[error("bm25_weight ({bm25}) + vector_weight ({vector}) must sum to 1.0")]
    WeightSumInvalid { bm25: f64, vector: f64 },

    /// Extraction window `max_tokens` is smaller than `min_tokens`.
    /// `min` is the configured minimum; `got` is the configured maximum that
    /// violates the invariant. Using `got` (the wrong value) rather than `max`
    /// aligns with Rust's conventional `expected`/`got` diagnostic naming.
    #[error("max_tokens must be >= min_tokens ({min}), got {got}")]
    TokenWindowInvalid { got: usize, min: usize },

    // ── Engine lifecycle (Story #5) ───────────────────────────────────────────
    /// Reserved for external callers that need to signal engine-already-init.
    /// `engine_init` itself is idempotent (no-op on second call); this variant
    /// exists for callers that enforce single-init semantics at a higher layer.
    #[error("engine already initialised")]
    EngineAlreadyInitialised,

    /// Reserved; `engine()` panics rather than returning this error (Story #5).
    /// Kept in the enum for backward compatibility with any downstream code that
    /// matches on `Error::EngineNotInitialised`.
    #[error("engine not yet initialised — call engine_init before using the engine")]
    EngineNotInitialised,

    // ── Content-hash dedup (Story #209) ──────────────────────────────────────
    /// An episode or fact with the same content hash already exists in the graph.
    ///
    /// `content_hash` is the SHA-256 hex digest of the deduplicated content so
    /// callers can surface the existing entity without re-parsing.
    #[error("duplicate content detected: hash {content_hash} already present")]
    Duplicate { content_hash: String },

    // ── Pre-mutation validation (Story #150) ──────────────────────────────────
    /// Two episodes within the same ingest batch share an ID, which would
    /// produce a primary-key conflict after the first INSERT. `id` is the
    /// duplicate episode identifier detected during the pre-mutation scan.
    #[error("intra-batch duplicate episode id '{id}'")]
    IntraBatchDuplicate { id: String },
}

pub type Result<T> = std::result::Result<T, Error>;

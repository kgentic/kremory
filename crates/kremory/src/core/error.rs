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
pub enum RqlError {
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
}

pub type Result<T> = std::result::Result<T, RqlError>;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Status of a Phase 2 (per-episode enrichment) run.
///
/// Per ADR D.6.3 + ADR-052 D5 — canonical ingest status enum.
///
/// ## Phase 2 state sequence (successful episode)
///
/// ```text
/// Pending → Extracting → EntitiesReady → (Deduplicating) → (Invalidating) → Complete
/// ```
///
/// `Deduplicating` and `Invalidating` are optional sub-phases that fire only
/// when facts are extracted and contradictions are found/resolved, respectively.
///
/// ## SQL column mapping
///
/// The `episode_processing_status` SQL column is a 4-state subset of this enum.
/// Use `from_sql_status` in `crate::core::sink` (crate-internal helper) to bridge SQL strings to this enum.
/// There is NO inverse bridge; `Complete`, `Deduplicating`, and `Invalidating`
/// have no SQL column equivalent.
///
/// ## `#[non_exhaustive]` — forward-compat contract
///
/// This attribute is load-bearing: match arms MUST include a `_` catch-all so
/// that future variant additions remain additive (semver minor). DO NOT remove it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IngestStatus {
    Pending,
    Extracting,
    /// Phase 2a complete: entities are written and queryable via
    /// `Memory::recall_entities()`. Facts extraction has not yet begun.
    ///
    /// Corresponds to SQL `'Verified'` in `episode_processing_status`.
    /// `Memory::wait_for_processing` resolves when this state is reached
    /// (SQL reaches `'Verified'`).
    ///
    /// Added in v0.2.3 (ADR-052 D5 — Phase 2a/2b event granularity);
    /// re-established by the ADR-052 Gap 1 sink-callsite cause-fix after the
    /// fb85ba8 dual-path consolidation regressed it.
    EntitiesReady,
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

    // ── Namespace policy (ADR-029a, v0.1.4) ───────────────────────────────────
    /// A [`crate::memory::types::NamespacePolicy`] construction or validation
    /// failed. Triggered by
    /// [`crate::memory::types::NamespacePolicy::validate`],
    /// [`crate::memory::types::Namespace::with_policy`], and
    /// [`crate::Memory::register_namespace`].
    #[error(transparent)]
    InvalidPolicy(#[from] crate::memory::types::InvalidPolicyError),

    /// `register_namespace` attempted to overwrite an existing namespace's
    /// policy with a different value. Policies are immutable once stored.
    /// Added v0.1.4 (ADR-029a Decision 7).
    #[error(
        "namespace policy is immutable once set: namespace '{namespace}' \
         has stored policy {stored:?}, attempted to re-register with {attempted:?}"
    )]
    NamespacePolicyImmutable {
        namespace: String,
        stored: crate::memory::types::NamespacePolicy,
        attempted: crate::memory::types::NamespacePolicy,
    },

    // ── ADR-029b AppendOnly enforcement (v0.1.5) ─────────────────────────────
    /// A mutating operation (forget / dream / reassign) was attempted on a
    /// namespace whose policy is `AppendOnly`. Added v0.1.5 (ADR-029b §3.1).
    #[error(
        "namespace '{namespace}' is AppendOnly — operation '{operation}' is not permitted \
         (stored policy: {policy:?})"
    )]
    NamespacePolicyViolation {
        namespace: String,
        operation: String,
        policy: crate::memory::types::NamespacePolicy,
    },

    /// An entity insert was blocked because the same entity name (`name`) is
    /// already registered in a different namespace group (`existing_ns`) and
    /// the `AppendOnly` policy prevents cross-namespace collision. Added v0.1.5
    /// (ADR-029b §3.2 — closes bypass surface #2: swallowed UNIQUE error).
    #[error(
        "cross-namespace collision: entity '{name}' exists in namespace '{existing_ns}' \
         but insert attempted in '{attempted_ns}'"
    )]
    CrossNamespaceCollision {
        name: String,
        existing_ns: String,
        attempted_ns: String,
    },

    /// `as_of_all` returned more facts than the safety ceiling allows. Added
    /// v0.1.5 (ADR-029b §4). `count` is the actual result count; `ceiling` is
    /// the configured limit.
    #[error(
        "as_of_all result overflow: got {count} facts, ceiling is {ceiling}; \
         narrow the query or raise the ceiling explicitly"
    )]
    ContradictionOverflow { count: usize, ceiling: usize },

    // ── ADR-029c multi-namespace recall (v0.1.5) ─────────────────────────────
    /// Both `in_namespace` and `in_namespaces` were set on the same
    /// `RecallRequest`. These selectors are mutually exclusive — use one or
    /// the other. Added v0.1.5 (ADR-029c Decision 6).
    ///
    /// `request` uses `String` (not `&'static str`) so the error message can
    /// carry dynamic context and the variant remains `Send + Sync + 'static`.
    #[error("conflicting namespace selectors: {request}")]
    ConflictingNamespaceSelectors { request: String },

    // ── StructuredCallBuilder (TD-012, Phase 4) ───────────────────────────────
    /// A schema-constrained LLM call received syntactically valid JSON that
    /// failed schema validation. `schema_name` identifies which schema was
    /// violated; `detail` carries the field path / reason.
    #[error("schema violation in '{schema_name}': {detail}")]
    SchemaViolation { schema_name: String, detail: String },

    /// All fallback arms in `StructuredCallBuilder` were exhausted without
    /// producing parseable structured output. `raw_response` carries the last
    /// raw LLM response for diagnostics.
    #[error(
        "structured-output fallback exhausted for schema '{schema_name}'; \
         last response: {raw_response}"
    )]
    FallbackExhausted {
        schema_name: String,
        raw_response: String,
    },

    // ── ExtractorKind factory (ADR-039, v0.2.0) ──────────────────────────────
    /// Extractor construction failed (e.g. GLiNER weight download or
    /// ONNX session init). Only fires when `ExtractorKind::GlinerLlm` was
    /// wired via `.with_gliner()` builder knob and GLiNER failed to initialise.
    ///
    /// Field is `detail` not `source` because thiserror treats `source` as a
    /// chained-error source automatically and requires it to be `std::error::Error`.
    #[error("hybrid extractor init failed: {detail}")]
    ExtractorInit { detail: String },

    /// Consumer requested a feature-gated extractor but kremory was built
    /// without that feature. e.g. `ExtractorKind::GlinerLlm` requires the
    /// `ner` cargo feature.
    #[error("extractor feature disabled: '{feature}' was not compiled in")]
    FeatureDisabled { feature: &'static str },

    /// Builder was given an unrecognised extractor identifier. Builder knobs
    /// (`.with_llm()`, `.with_gliner()`, `.with_extractor()`) are preferred;
    /// this variant fires on programmatic misuse of internal dispatch.
    #[error("unknown extractor source requested: '{requested}'")]
    UnknownExtractor { requested: String },

    // ── E-2 builder knobs (ADR-039, v0.2.0) ─────────────────────────────────
    /// Builder configuration conflict — two mutually exclusive knobs were set,
    /// or a required knob is missing.
    #[error("builder configuration conflict: {detail}")]
    BuilderConflict { detail: String },

    /// Operation requires an LLM provider that was not wired at build time.
    /// Fires at call time on `Memory` instances constructed without `.with_llm()`.
    #[error("operation `{method}` requires an LLM provider — {hint}")]
    LlmRequired {
        method: &'static str,
        hint: &'static str,
    },

    // ── Store integrity (T1.8, v0.2.0 Phase B-prep) ──────────────────────────
    /// `TemporalGraph::open` refused to operate because the on-disk store failed
    /// a critical integrity check: a required table is missing or the schema
    /// version does not match the expected version for this binary.
    ///
    /// `reason` carries the specific diagnostic (e.g. "missing table: episodes",
    /// "schema_version mismatch: expected 13, got 5"). The store is NOT modified;
    /// callers must repair the database before re-opening.
    #[error("store integrity check failed: {reason}")]
    CorruptStore { reason: String },

    // ── ADR-051 async extraction wait API (v0.2.2, Phase 4) ─────────────────
    /// Background extraction pipeline transitioned the episode to `Failed`.
    ///
    /// `episode_id` is the SQLite rowid of the episode whose extraction failed.
    /// Callers can re-ingest or inspect the episode's content to diagnose.
    #[error(
        "background extraction failed for episode {episode_id}: \
         extraction pipeline transitioned status to 'Failed'"
    )]
    ExtractionFailed { episode_id: i64 },

    /// `Memory::wait_for_processing` exceeded the caller-supplied timeout
    /// without the episode reaching a terminal status (`Verified` or `Failed`).
    ///
    /// `episode_id` is the SQLite rowid of the episode that timed out.
    /// `elapsed` is the wall-clock duration that passed before the timeout was
    /// declared. The episode may still complete in the background — this error
    /// only indicates the caller's wait budget was exhausted.
    #[error(
        "wait_for_processing timed out for episode {episode_id} \
         after {elapsed:?} — episode may still complete in background"
    )]
    WaitTimeout {
        episode_id: i64,
        elapsed: std::time::Duration,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

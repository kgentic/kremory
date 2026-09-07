use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Status of a Phase 2 (per-episode enrichment) run.
///
/// Canonical ingest status enum.
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
    /// Added in v0.2.3 (Phase 2a/2b event granularity);
    /// re-established by a sink-callsite cause-fix after the
    /// fb85ba8 dual-path consolidation regressed it.
    EntitiesReady,
    Deduplicating,
    Invalidating,
    Complete,
    Failed(String),
    /// Episode Phase 2 was skipped because a duplicate content hash was detected
    /// at Guard #1 in `run_verify_stage` — the exact entity was already processed
    /// in this pass. Fires via `on_stage_change(SkippedIdempotent)` once per
    /// skipped entity (not once per episode).
    ///
    /// Added in v0.2.4 (crash-safety + idempotency sink events).
    SkippedIdempotent,
    /// Phase 2 extraction was **never requested** — the caller chained
    /// [`RememberRequest::skip_extraction`](crate::facade::remember::RememberRequest::skip_extraction),
    /// so the episode, its embedding and any caller-supplied facts are durably
    /// stored and **nothing further will happen**.
    ///
    /// Corresponds to SQL `'Skipped'` in `episode_processing_status`, and it is a
    /// **terminal** state: `Memory::wait_for_processing` resolves `Ok(())` on it.
    ///
    /// # Not to be confused with [`SkippedIdempotent`](Self::SkippedIdempotent)
    ///
    /// They differ by *reason*, which is why both exist. `SkippedIdempotent`
    /// means extraction *was* requested and the verify stage declined a
    /// duplicate content hash at Guard #1. This variant means extraction was
    /// never on the table.
    ///
    /// # Why the vocabulary needed a new value
    ///
    /// Before this, `skip_extraction` ingests stayed `'Pending'` forever, because
    /// writing `'Verified'` would have been semantically false — nothing was
    /// verified. But `'Pending'` means *work is still coming*, and for these
    /// episodes nothing was ever enqueued. A consumer combining
    /// `skip_extraction()` with `wait_for_processing()` therefore burned its whole
    /// timeout budget and received `WaitTimeout` — **failure reported for an
    /// ingest that succeeded**. The column conflated "not processed yet" with
    /// "will never be processed"; this variant separates them.
    ///
    /// Spelling follows the ecosystem convention for a deliberately-not-run step
    /// (GitHub Actions, Argo Workflows, Tekton all use `skipped`).
    ExtractionSkipped,
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

    // ── Named struct variants ─────────────────────────────────────────────────
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

    /// `TemporalGraph::open`/`open_with_dim` refused to operate because the
    /// caller-supplied `embedding_dim` disagrees with the embedding dimension
    /// this store already contains data for.
    ///
    /// Two independent guards raise this variant (`migrate_024_verify_embedding_dim`,
    /// `core/migrations/defs_k.rs`): (1) the byte length of an existing stored
    /// `embedding` row disagrees with `requested_dim * 4`, or (2) the
    /// `embedding_dim` persisted in `embedding_dim_registry` at first-open
    /// disagrees with `requested_dim`. Reopening with the wrong dim would
    /// otherwise either silently return garbage nearest-neighbour results
    /// (byte reinterpretation across a differently-typed `F32_BLOB` column) or
    /// corrupt data during a column rebuild
    /// (`migrate_023_vector_index_column_type`'s raw `INSERT ... SELECT
    /// embedding` copy). `requested_dim` is the dim this open call supplied;
    /// `detail` names which guard fired and the concrete numbers involved.
    #[error(
        "embedding_dim mismatch: opened with embedding_dim={requested_dim} but {detail} — \
         reopen with the original embedding_dim, or migrate the stored vectors, before \
         changing it"
    )]
    EmbeddingDimMismatch {
        requested_dim: usize,
        detail: String,
    },

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

    /// Extraction window `max_words` is smaller than `min_words`.
    /// `min` is the configured minimum; `got` is the configured maximum that
    /// violates the invariant. Using `got` (the wrong value) rather than `max`
    /// aligns with Rust's conventional `expected`/`got` diagnostic naming.
    #[error("max_words must be >= min_words ({min}), got {got}")]
    TokenWindowInvalid { got: usize, min: usize },

    // ── Engine lifecycle ──────────────────────────────────────────────────────
    /// Reserved for external callers that need to signal engine-already-init.
    /// `engine_init` itself is idempotent (no-op on second call); this variant
    /// exists for callers that enforce single-init semantics at a higher layer.
    #[error("engine already initialised")]
    EngineAlreadyInitialised,

    /// Reserved; `engine()` panics rather than returning this error.
    /// Kept in the enum for backward compatibility with any downstream code that
    /// matches on `Error::EngineNotInitialised`.
    #[error("engine not yet initialised — call engine_init before using the engine")]
    EngineNotInitialised,

    // ── Content-hash dedup ────────────────────────────────────────────────────
    /// An episode or fact with the same content hash already exists in the graph.
    ///
    /// `content_hash` is the SHA-256 hex digest of the deduplicated content so
    /// callers can surface the existing entity without re-parsing.
    #[error("duplicate content detected: hash {content_hash} already present")]
    Duplicate { content_hash: String },

    // ── Self-loop rejection ──────────────────────────────────
    /// A fact insert was rejected because `subject_id == object_id`, which would
    /// create a self-loop edge (entity points to itself). Self-loops are a formal
    /// graph invariant violation: they produce meaningless triples and confuse
    /// graph traversal algorithms.
    ///
    /// `kremory.fact.rejected_total{reason="self_loop"}` is emitted before this
    /// error is returned so quality metrics capture PRE-rejection extraction rate.
    ///
    /// `try_insert_fact` / `try_insert_fact_with_group` swallow this to `Ok(None)`
    /// so a self-loop in a batch does not abort other facts.
    #[error(
        "self-loop fact rejected: predicate '{predicate}' from entity '{subject_id}' to itself"
    )]
    SelfLoop {
        subject_id: String,
        predicate: String,
    },

    // ── Pre-mutation validation ───────────────────────────────────────────────
    /// Two episodes within the same ingest batch share an ID, which would
    /// produce a primary-key conflict after the first INSERT. `id` is the
    /// duplicate episode identifier detected during the pre-mutation scan.
    #[error("intra-batch duplicate episode id '{id}'")]
    IntraBatchDuplicate { id: String },

    // ── Namespace policy (v0.1.4) ───────────────────────────────────
    /// A [`crate::memory::types::NamespacePolicy`] construction or validation
    /// failed. Triggered by
    /// [`crate::memory::types::NamespacePolicy::validate`],
    /// [`crate::memory::types::Namespace::with_policy`], and
    /// [`crate::Memory::register_namespace`].
    #[error(transparent)]
    InvalidPolicy(#[from] crate::memory::types::InvalidPolicyError),

    /// `register_namespace` attempted to overwrite an existing namespace's
    /// policy with a different value. Policies are immutable once stored.
    /// Added v0.1.4.
    #[error(
        "namespace policy is immutable once set: namespace '{namespace}' \
         has stored policy {stored:?}, attempted to re-register with {attempted:?}"
    )]
    NamespacePolicyImmutable {
        namespace: String,
        stored: crate::memory::types::NamespacePolicy,
        attempted: crate::memory::types::NamespacePolicy,
    },

    // ── AppendOnly enforcement (v0.1.5) ─────────────────────────────
    /// A mutating operation (forget / dream / reassign) was attempted on a
    /// namespace whose policy is `AppendOnly`. Added v0.1.5.
    #[error(
        "namespace '{namespace}' is AppendOnly — operation '{operation}' is not permitted \
         (stored policy: {policy:?})"
    )]
    NamespacePolicyViolation {
        namespace: String,
        operation: String,
        policy: crate::memory::types::NamespacePolicy,
    },

    /// **RESERVED — not currently produced.** Entity identity is
    /// per-namespace-open by default: the same name in a different namespace is a
    /// legitimate independent row (emitted as a non-blocking signal, not an error).
    /// This variant is kept for the future opt-in strict-global-identity
    /// `NamespacePolicy` (not yet built) — when a consumer
    /// selects `EntityIdentityScope::StrictGlobal`, `insert_entity_with_group`
    /// will return this. The earlier doc-comment claiming an `AppendOnly` gate was
    /// false (the guard never read policy).
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
    /// v0.1.5. `count` is the actual result count; `ceiling` is
    /// the configured limit.
    #[error(
        "as_of_all result overflow: got {count} facts, ceiling is {ceiling}; \
         narrow the query or raise the ceiling explicitly"
    )]
    ContradictionOverflow { count: usize, ceiling: usize },

    // ── Unsupported feature (DX hardening, F4 / ADR as-of-fail-loud) ──────────
    /// A surfaced API knob was invoked for a capability that is declared but not
    /// yet implemented, so the call would otherwise silently no-op. `feature`
    /// names the capability (e.g. "as_of point-in-time recall"). Fail-loud
    /// replaces silent no-op: callers learn at call time rather than shipping
    /// code that believes the filter applied. Enforced at the consumption point
    /// (e.g. `memory::search`) so it covers every setter path, not one of N.
    #[error("unsupported feature: {feature} is not yet implemented")]
    Unsupported { feature: &'static str },

    // ── Multi-namespace recall (v0.1.5) ─────────────────────────────
    /// Both `in_namespace` and `in_namespaces` were set on the same
    /// `RecallRequest`. These selectors are mutually exclusive — use one or
    /// the other. Added v0.1.5.
    ///
    /// `request` uses `String` (not `&'static str`) so the error message can
    /// carry dynamic context and the variant remains `Send + Sync + 'static`.
    #[error("conflicting namespace selectors: {request}")]
    ConflictingNamespaceSelectors { request: String },

    // ── StructuredCallBuilder ───────────────────────────────
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

    // ── ExtractorKind factory (v0.2.0) ──────────────────────────────
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

    // ── Builder knobs (v0.2.0) ─────────────────────────────────
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

    // ── Async extraction wait API (v0.2.2) ─────────────────
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

    // ── Reversible-graph-mutations LIFO guard ──────
    /// `unmerge` was called out of order on a CHAINED merge. A later, still-live
    /// `entity_merge` (`blocking_mutation_id`) re-used one of this mutation's
    /// endpoints (its keeper or loser), so it re-pointed facts/edges THROUGH the
    /// shared endpoint. Reversing `mutation_id` first would leave a broken fact
    /// chain. Chained merges must be unwound Last-In-First-Out: unmerge
    /// `blocking_mutation_id` (and any later chained merges) FIRST.
    #[error(
        "unmerge out of order: mutation {mutation_id} is chained under still-live \
         merge {blocking_mutation_id} — unmerge later chained merges FIRST (LIFO)"
    )]
    UnmergeOutOfOrder {
        mutation_id: i64,
        blocking_mutation_id: i64,
    },

    /// An undo could not be applied faithfully because the world CHANGED
    /// underneath the snapshot: the entity the reversal must write onto no longer
    /// exists in the form the snapshot assumed. The canonical case is `unmerge`
    /// when the keeper a merge folded the loser into was RENAMED or DELETED
    /// between the merge and the unmerge (a cross-kind chain, e.g.
    /// `merge(A→B) → rename(B→C) → unmerge`): restoring the keeper's overwritten
    /// `access_count` / `ner_confidence` would match zero rows. Rather than report
    /// a FALSE success (an outcome naming a keeper that no longer exists), the undo
    /// fails LOUD and the whole reversal rolls back, so the consumer learns the
    /// mutation is no longer cleanly reversible. `mutation_id` is the row that could
    /// not be reversed; `reason` names the stale precondition.
    #[error("undo stale: mutation {mutation_id} cannot be reversed — {reason}")]
    UndoStale { mutation_id: i64, reason: String },

    // ── Reversible-graph-mutations edit-entity cascade ────────────────────────
    /// `edit_entity(...).rename(new_id)` targeted an id that ALREADY exists in the
    /// namespace. A rekey INTO an existing entity would silently FUSE two distinct
    /// entities (the loser's FKs would land on the existing keeper); the cascade
    /// refuses and returns this structured error so the consumer chooses to `merge`
    /// explicitly instead (structured write-steering — never a silent
    /// auto-merge). `from` is the source id, `existing` the occupied target id.
    #[error(
        "entity edit conflict: cannot rename '{from}' to '{existing}' — an entity \
         with id '{existing}' already exists in namespace '{group_id}'; merge the \
         two explicitly instead of renaming into an occupied id"
    )]
    EntityEditConflict {
        from: String,
        existing: String,
        group_id: String,
    },

    /// The `edit_entity(...)` request itself is malformed — a caller bug, never
    /// retryable. `detail` names which invariant was violated. Two cases:
    ///
    /// - built with neither a `.rename(...)` nor a `.retype(...)` target (or with
    ///   both) — exactly one edit operation is required per call;
    /// - `.retype(type_id)` named a type that is not registered in the target
    ///   namespace's `entity_types` registry. `entities.entity_type_id` has no
    ///   foreign key (Migration 008), so an unregistered id would be written
    ///   dangling and silently resolve to the "Entity" catch-all at read time.
    ///   The cascade refuses instead of coercing, so an explicit mistake is
    ///   visible rather than absorbed; `detail` names the offending id and how to
    ///   register the type. id=0 ("Entity") is always admissible.
    #[error("entity edit request invalid: {detail}")]
    EntityEditInvalid { detail: String },

    /// `edit_entity(...)` / `undo_entity_edit(...)` targeted an entity or a
    /// mutation-log row that does not exist. `detail` carries the specific
    /// diagnostic (the missing entity id or `entity_edit` mutation id).
    #[error("entity edit not found: {detail}")]
    EntityEditNotFound { detail: String },

    // ── Reversible-graph-mutations delete cascade ─────────────────────────────
    /// `delete_entity(...)` / `undo_delete_entity(...)` targeted an entity or an
    /// `entity_delete` mutation-log row that does not exist. Deleting a
    /// non-existent entity (or reversing a delete that was never logged) is a
    /// caller bug, never a silent no-op (parse-loudly). `detail` names which.
    #[error("entity delete not found: {detail}")]
    EntityDeleteNotFound { detail: String },

    /// `delete_fact(...)` / `undo_delete_fact(...)` targeted a fact or a
    /// `fact_delete` mutation-log row that does not exist. `detail` names which.
    #[error("fact delete not found: {detail}")]
    FactDeleteNotFound { detail: String },

    // ── Reversible-graph-mutations unified undo dispatcher ──
    /// `undo(mutation_id)` was called with an id that names no `graph_mutation_log`
    /// row. Reversing a mutation that was never logged is a caller bug, never a
    /// silent no-op (parse-loudly). `mutation_id` is the unknown id.
    #[error("mutation not found: no graph_mutation_log row with id {mutation_id}")]
    MutationNotFound { mutation_id: i64 },

    /// `undo(mutation_id)` matched a `graph_mutation_log` row whose `kind` is not
    /// one of the four currently log-dispatchable, reversible kinds
    /// (`entity_merge` / `entity_edit` / `entity_delete` / `fact_delete`). The
    /// other four [`crate::MutationKind`] variants (`fact_supersede` /
    /// `fact_archive` / `community_assign` / `canonical_form`) are RESERVED — they
    /// are not produced into the log today. `fact_supersede` / `fact_archive` are
    /// reversed via their own domain-id methods (`unsupersede` /
    /// `restore_archived_fact`), not the unified `undo`. `kind` is the offending
    /// `graph_mutation_log.kind` tag.
    #[error(
        "undo unsupported kind: mutation {mutation_id} has kind '{kind}' which the \
         unified undo() dispatcher does not reverse — use the kind-specific method \
         (fact_supersede → unsupersede, fact_archive → restore_archived_fact), or \
         the kind is reserved / not yet produced"
    )]
    UndoUnsupportedKind { mutation_id: i64, kind: String },

    /// `undo(mutation_id).in_namespace(ns)` was scoped to a namespace whose group
    /// differs from the mutation's logged `group_id`. An undo targets the ORIGINAL
    /// mutation's namespace; a mismatch is refused LOUDLY rather than silently
    /// reversing a mutation in a namespace the caller did not intend.
    /// `mutation_group` is where the mutation was logged; `requested_group` is the
    /// scope the caller passed.
    #[error(
        "undo wrong namespace: mutation {mutation_id} was logged in namespace \
         '{mutation_group}' but undo was scoped to '{requested_group}' — omit \
         .in_namespace(...) or pass the mutation's original namespace"
    )]
    UndoWrongNamespace {
        mutation_id: i64,
        mutation_group: String,
        requested_group: String,
    },
}

impl Error {
    /// Classify a fact-insert failure into a bounded `reason` label for the
    /// `*_fact_insert_failed_total` family of counters. Sub-classifies the
    /// coarse `Database` variant by its underlying SQLite constraint text
    /// instead of collapsing every DB failure into one bucket: `FOREIGN KEY`
    /// ⇒ `fk_mismatch` (the composite-FK namespace-mismatch shape), `UNIQUE` ⇒
    /// `unique_violation`, anything else DB-shaped ⇒ `db_error`.
    ///
    /// Shared by the foreground path (`ingest/pipeline/ingest_with.rs`), the
    /// deferred path (`ingest/pipeline/deferred.rs`), and the graph-layer
    /// insert path (`graph/facts.rs`) so all three report the SAME reason
    /// taxonomy instead of each duplicating ad-hoc
    /// `e.to_string().contains(...)` string matching at its own call site.
    pub(crate) fn fact_insert_failure_reason(&self) -> &'static str {
        match self {
            Error::Database(db_err) => {
                let s = db_err.to_string();
                if s.contains("FOREIGN KEY") {
                    "fk_mismatch"
                } else if s.contains("UNIQUE") {
                    "unique_violation"
                } else {
                    "db_error"
                }
            }
            Error::InsertReturnedNoRowId { .. } => "no_rowid",
            _ => "other",
        }
    }

    /// True when this is a SQLite UNIQUE-constraint violation.
    ///
    /// `insert_entity_with_group` issues a bare `INSERT INTO entities`
    /// and returns the libsql error UNMAPPED, so callers could not distinguish
    /// "this entity already exists" (benign — a real row IS present) from a
    /// genuine database failure. Two call sites then guessed OPPOSITELY:
    /// `ingest_with.rs:1114` (stub path) treats a duplicate as fine — its own
    /// comment says so — while the insert-new path did `break 'phases Err(e)`
    /// on ANY error and so killed the whole ingest on a benign duplicate.
    /// Observed in production: 1 of 8 LongMemEval sessions returned HTTP 500
    /// with `UNIQUE constraint failed: entities.id, entities.group_id`.
    ///
    /// This makes the distinction EXPRESSIBLE rather than accidental. It
    /// deliberately does NOT swallow the error inside the insert: a caller that
    /// genuinely cannot tolerate a duplicate must still be able to fail on one.
    ///
    /// Message-text driven, matching the sibling
    /// [`Self::fact_insert_failure_reason`] convention — libsql surfaces
    /// constraint failures as `SqliteFailure(code, message)`.
    pub(crate) fn is_unique_violation(&self) -> bool {
        matches!(self, Error::Database(db_err) if db_err.to_string().contains("UNIQUE"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    // ── B2: `Error::fact_insert_failure_reason` classification ──────────────
    //
    // Pins the shared reason taxonomy the foreground (`ingest_with.rs`),
    // deferred (`deferred.rs`), and graph-layer (`facts.rs`) fact-insert
    // paths all now route through — a real SQLite FK/UNIQUE constraint
    // failure surfaces via `libsql::Error::SqliteFailure(_, message)`, so
    // classification is driven by the message text (matches the sibling
    // deferred-path convention this replaces).

    #[test]
    fn fact_insert_failure_reason_classifies_foreign_key_as_fk_mismatch() {
        let err = Error::Database(libsql::Error::SqliteFailure(
            787,
            "FOREIGN KEY constraint failed".to_string(),
        ));
        assert_eq!(err.fact_insert_failure_reason(), "fk_mismatch");
    }

    #[test]
    fn fact_insert_failure_reason_classifies_unique_as_unique_violation() {
        let err = Error::Database(libsql::Error::SqliteFailure(
            2067,
            "UNIQUE constraint failed: facts.content_hash".to_string(),
        ));
        assert_eq!(err.fact_insert_failure_reason(), "unique_violation");
    }

    #[test]
    fn fact_insert_failure_reason_classifies_generic_database_error_as_db_error() {
        let err = Error::Database(libsql::Error::SqliteFailure(
            1,
            "disk I/O error".to_string(),
        ));
        assert_eq!(err.fact_insert_failure_reason(), "db_error");
    }

    #[test]
    fn fact_insert_failure_reason_classifies_no_rowid_variant() {
        let err = Error::InsertReturnedNoRowId {
            operation: "insert_fact",
        };
        assert_eq!(err.fact_insert_failure_reason(), "no_rowid");
    }

    #[test]
    fn fact_insert_failure_reason_classifies_non_database_variants_as_other() {
        let err = Error::Config("bad config".to_string());
        assert_eq!(err.fact_insert_failure_reason(), "other");
    }
}

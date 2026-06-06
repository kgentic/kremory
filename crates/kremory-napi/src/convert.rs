//! Type-safe conversions between kremory Rust types and napi-rs JS objects.
//!
//! All `#[napi(object)]` structs here generate TypeScript `interface` declarations
//! in `index.d.ts` via the napi-rs derive macro pipeline.

use napi_derive::napi;

use kremory::{DreamSummary, Namespace, RetrievedContext};

// ── Input option structs ──────────────────────────────────────────────────────

/// Options for `Memory.open` — live napi/cdylib build.
///
/// When `with_embedder` is supplied, the Tier-2 BYOM builder path is taken:
/// the env-detected LLM is combined with the JS callback as the embedding
/// provider. When absent, the Tier-1 `Memory::auto` path is used (backward-compat).
///
/// `object_to_js = false`: skip `ToNapiValue` generation so `ThreadsafeFunction`
/// (which is `Send + Sync` but lacks `ToNapiValue`) can be used as a field.
/// `JsFunction` would satisfy both traits but is `!Send` (wraps `*mut napi_env__`),
/// making the async factory future `!Send` and failing `execute_tokio_future`'s
/// Send bound. `ThreadsafeFunction` is `Send + Sync` and correct here.
///
/// `JsOpenOptions` is only ever constructed from JS → Rust, never returned.
/// Skipping `ToNapiValue` is therefore correct and safe.
///
/// This struct is excluded from `cfg(test)` because napi-derive macro-generated
/// `FromNapiValue` code references `ThreadsafeFunction` symbols that are only
/// available in a live napi runtime (cdylib target). The test variant below
/// (`cfg(test)`) omits the `with_embedder` field so unit tests can construct
/// `JsOpenOptions` without a napi runtime.
#[cfg(not(test))]
#[napi(object, object_to_js = false, js_name = "OpenOptions")]
pub struct JsOpenOptions {
    /// Embedding vector dimensionality. Must match the callback's output
    /// dimension when `with_embedder` is set. When `with_embedder` is absent,
    /// this field is ignored (the active env-detected provider's native dim
    /// is used). Defaults to `None`.
    pub embedding_dim: Option<i64>,
    /// Default namespace applied to all operations on this handle when no
    /// per-call namespace is specified.
    pub default_namespace: Option<String>,
    /// Optional BYOM embedder callback (ADR-030 Tier-2).
    ///
    /// Callback signature: `(text: string) => Promise<number[]>`.
    ///
    /// When set, kremory wires this JS function as the embedding provider via
    /// the `MemoryBuilder` Tier-2 path. The LLM is still resolved from env
    /// (`OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`).
    ///
    /// Set `embeddingDim` to the callback's output dimension. A mismatch
    /// between the returned vec length and `embeddingDim` is rejected with a
    /// descriptive error rather than a panic.
    ///
    /// When absent (the default), `Memory.open` behaves identically to
    /// v0.1.6-alpha.0 — `Memory::auto` with env-detected providers.
    #[napi(ts_type = "((text: string) => Promise<number[]>) | undefined | null")]
    pub with_embedder: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            String,
            napi::threadsafe_function::ErrorStrategy::CalleeHandled,
        >,
    >,
    /// Production extractor selection (KH2, kremory v0.1.7).
    ///
    /// - `"auto"` (default if omitted) — read `KREMORY_EXTRACTOR` env var,
    ///   falling back to `nuextract` if unset/unrecognized.
    /// - `"nuextract"` — pin NuExtract regardless of env (TD-013 path).
    /// - `"hybrid"` — pin Hybrid (TD-023: GLiNER + 1 LLM typing call).
    ///   Requires kremory built with `--features ner`. On a non-ner build
    ///   this returns a typed napi error `KremoryError::FeatureDisabled('ner')`.
    ///
    /// See spec `kremory-v017-hybrid-extractor-production-wire-in-spec-2026-06-04` §4.3.
    #[napi(ts_type = "'auto' | 'hybrid' | 'nuextract' | undefined | null")]
    pub extractor: Option<String>,
}

/// Options for `Memory.open` — test build (no napi runtime).
///
/// Omits `with_embedder` — `ThreadsafeFunction` cannot be constructed outside
/// the napi cdylib runtime. Unit tests that exercise the bridge logic use the
/// mock `JsEmbedderBridge` variants in `bridge.rs` directly.
///
/// Implements `FromNapiValue` as a stub so `#[napi]` on `JsMemory::open` compiles
/// in test mode. The impl is never invoked — no napi runtime is present in test
/// binary builds. The test `cfg` path in `JsMemory::open` never reads
/// `opts.with_embedder` (it doesn't exist), so this stub is safe.
#[cfg(test)]
pub struct JsOpenOptions {
    /// Embedding vector dimensionality.
    pub embedding_dim: Option<i64>,
    /// Default namespace.
    pub default_namespace: Option<String>,
    /// Extractor selection — see live struct.
    pub extractor: Option<String>,
}

#[cfg(test)]
impl napi::bindgen_prelude::FromNapiValue for JsOpenOptions {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _nv: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        // Never called in test builds — napi runtime is absent.
        // Stub satisfies the trait bound required by #[napi] on JsMemory::open.
        Ok(JsOpenOptions { embedding_dim: None, default_namespace: None, extractor: None })
    }
}

/// Pre-extracted RDF triple supplied by the caller via
/// `JsRememberOptions.structuredFacts`.
///
/// When present, these triples are pinned directly into the kremory graph at
/// ingest time, BEFORE the Phase 2 LLM extractor runs (ADR-035 Path X /
/// Option A). LLM-extracted duplicates of the same `(subject, predicate,
/// object)` triple are silently swallowed by the substrate's
/// `try_insert_fact` helper — caller wins by virtue of being there first.
///
/// To skip Phase 2 LLM extraction entirely (caller is the sole source of
/// truth for facts), set `JsRememberOptions.skipExtraction = true`.
///
/// # Field defaults
/// - `valid_from` / `valid_to`: ISO-8601 / RFC-3339 UTC strings
///   (e.g. `"2024-03-15T10:00:00Z"`). When absent, the substrate falls back
///   to the source's `published_at`, then to the ingest time.
/// - `memory_type`: optional memory-tier hint (one of: `"decision"`,
///   `"pattern"`, `"preference"`, `"style"`, `"habit"`, `"insight"`,
///   `"observation"`). When absent, the substrate uses its default.
///
/// All required fields (subject, predicate, object) fail loudly with a
/// descriptive `napi::Error` on missing/malformed values per ADR-035 §3
/// (`llm-output-parse-loudly` discipline applied at the binding layer).
#[napi(object, js_name = "StructuredFact")]
#[derive(Debug, Clone)]
pub struct JsStructuredFact {
    /// Subject entity identifier or literal. Required.
    pub subject: String,
    /// Predicate / relationship name. Required.
    pub predicate: String,
    /// Object — entity identifier or literal value. Required.
    pub object: String,
    /// Optional world-time start (ISO-8601 / RFC-3339 UTC string).
    pub valid_from: Option<String>,
    /// Optional world-time end (ISO-8601 / RFC-3339 UTC string).
    pub valid_to: Option<String>,
    /// Optional memory tier hint.
    pub memory_type: Option<String>,
}

impl TryFrom<JsStructuredFact> for kremory::memory::types::StructuredFact {
    type Error = napi::Error;

    fn try_from(js: JsStructuredFact) -> Result<Self, Self::Error> {
        let parse = |label: &str, s: &str| -> Result<chrono::DateTime<chrono::Utc>, napi::Error> {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "JsStructuredFact.{}: invalid RFC-3339 timestamp {:?}: {}",
                        label, s, e
                    ))
                })
        };
        let valid_from = match js.valid_from {
            Some(ref s) => Some(parse("valid_from", s)?),
            None => None,
        };
        let valid_to = match js.valid_to {
            Some(ref s) => Some(parse("valid_to", s)?),
            None => None,
        };
        let memory_type = match js.memory_type.as_deref() {
            None | Some("") => None,
            Some(s) => Some(
                serde_json::from_value::<kremory::MemoryType>(
                    serde_json::Value::String(s.to_string()),
                )
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "JsStructuredFact.memory_type: unknown value {:?} \
                        (expected decision|pattern|preference|style|habit|insight|observation): {}",
                        s, e
                    ))
                })?,
            ),
        };
        Ok(kremory::memory::types::StructuredFact {
            subject: js.subject,
            predicate: js.predicate,
            object: js.object,
            valid_from,
            valid_to,
            memory_type,
        })
    }
}

/// Options for `JsMemory.remember(opts)` — the single ingest surface per
/// ADR-034 (collapses prior `JsMemory.ingest(text, opts)` +
/// `JsMemory.ingest_episode(draft)`).
///
/// All fields except `content` are optional. Backward compat: `structured_facts:
/// undefined` and `skip_extraction: undefined` (or absent) behave identically
/// to pre-v0.1.8 ingest.
#[napi(object, js_name = "RememberOptions")]
pub struct JsRememberOptions {
    /// Episode text content. Required.
    pub content: String,
    /// Stable source identifier (consumer-defined: slug, file path, UUID, etc.).
    /// Stored as `source_id` on the episode row. Required for the post-ingest
    /// `source_uri` and `metadata` writes to take effect.
    pub source_id: Option<String>,
    /// URI of the source document (e.g. file path, URL). Persisted via a
    /// post-ingest `update_source_uri` call when `source_id` is also supplied.
    pub source_uri: Option<String>,
    /// Opaque JSON metadata to attach to this episode. Persisted via a
    /// post-ingest `update_episode_metadata` call when `source_id` is also
    /// supplied.
    pub metadata: Option<serde_json::Value>,
    /// Namespace to scope this episode. Overrides the Memory handle's default.
    pub namespace: Option<String>,
    /// ISO-8601 / RFC-3339 timestamp used as the `published_at` bi-temporal
    /// anchor. When omitted, the substrate uses the ingest time.
    pub reference_time: Option<String>,
    /// Pre-extracted RDF triples (ADR-035 Path X). When supplied, caller
    /// facts are pinned BEFORE Phase 2 LLM extraction; LLM-extracted
    /// duplicates of the same triple are silently swallowed.
    pub structured_facts: Option<Vec<JsStructuredFact>>,
    /// When `true`, Phase 2 LLM extraction is skipped entirely for this
    /// episode (the episode row + embedding + pinned facts are still
    /// persisted). Suitable for bulk-import workloads.
    pub skip_extraction: Option<bool>,
}

/// A single metadata filter key/value pair for `JsRecallOptions.filterMetadata`.
///
/// Maps to `RecallRequest::filter_metadata(key, value)` on the substrate.
/// Multiple entries combine with AND.
#[napi(object, js_name = "MetadataFilter")]
pub struct JsMetadataFilter {
    /// Top-level metadata key to match. Must be non-empty and ≤ 128 chars.
    pub key: String,
    /// Exact value the key must equal.
    pub value: serde_json::Value,
}

/// Options for `Memory.recall`.
#[napi(object, js_name = "RecallOptions")]
pub struct JsRecallOptions {
    /// Maximum number of results to return. Default: 10.
    pub k: Option<i64>,
    /// Namespace to scope this recall. Overrides the Memory handle's default.
    /// Mutually exclusive with `in_namespaces` — setting both rejects at
    /// `.await` time with `ConflictingNamespaceSelectors`.
    pub namespace: Option<String>,
    /// Multi-namespace recall (ADR-029c). Set instead of `namespace` to query
    /// across many namespaces and receive RRF-blended results with per-row
    /// `namespace` attribution on each `JsRetrievedContext`.
    pub in_namespaces: Option<Vec<String>>,
    /// When `true`, sub-query failures emit `warn!` and the failing namespace
    /// is skipped rather than failing the whole call. Default: `false`.
    /// No-op unless `in_namespaces` is set.
    pub best_effort: Option<bool>,
    /// Cap on results fetched per namespace BEFORE cross-namespace RRF blending.
    /// Default: `k` (or `Memory::default_k`). Raises recall diversity at the
    /// cost of extra per-namespace sub-query work. No-op unless `in_namespaces`
    /// is set.
    pub per_namespace_top_k: Option<i64>,
    /// ISO 8601 point-in-time filter. Returns facts that were valid at this
    /// timestamp. e.g. `"2024-03-15T10:00:00Z"`.
    pub as_of: Option<String>,
    /// Post-filter on episode metadata. Each entry maps to a substrate
    /// `RecallRequest::filter_metadata(key, value)` call. Multiple entries
    /// combine with AND. Substrate-generic — a consumer might filter on
    /// document type, conversation id, session id, or any custom key.
    pub filter_metadata: Option<Vec<JsMetadataFilter>>,
}

// ── Output types ──────────────────────────────────────────────────────────────

/// Result of a successful `Memory.remember` call.
#[napi(object, js_name = "IngestResult")]
pub struct JsIngestResult {
    /// Stable entity ID under which the episode is searchable immediately.
    /// This is a graph node UUID, NOT a run ID — see `run_id` below.
    pub episode_entity_id: String,
    /// Background run identifier (Quinn Cycle 2 M-01 fix). Present when the
    /// substrate spawned a background Phase 2 enrichment task; absent when the
    /// inline path was used (no run to poll). Pass this string to
    /// `Memory.statusOf`, `Memory.awaitEnrichment`, or `Memory.cancel` to track
    /// background completion — NOT `episode_entity_id`, which is a different
    /// identifier.
    pub run_id: Option<String>,
    /// ISO 8601 timestamp at which Phase 1 committed.
    pub committed_at: String,
    /// Non-fatal ingest warnings (e.g. content exceeded soft size threshold).
    /// Empty when no warnings were raised.
    pub warnings: Vec<String>,
}

/// A single episode returned by `Memory.getBySourceId`.
#[napi(object, js_name = "Episode")]
pub struct JsEpisode {
    /// Internal database row ID.
    pub id: f64,
    /// The `source_id` this episode was looked up by. Matches the argument
    /// passed to `getBySourceId`. `null` only for episodes ingested before v0.1.6.
    pub source_id: Option<String>,
    /// URI of the source document. Currently always `null` — substrate
    /// `recall_by_source_id` does not return this column. Use `updateUri` to
    /// set and the DB column is persisted; this field is a known gap.
    pub source_uri: Option<String>,
    /// Episode text content.
    pub content: String,
    /// ISO 8601 timestamp of the episode.
    pub timestamp: String,
    /// Source type tag (e.g. "Document", "Chat", "Note"). `null` if not set.
    pub source_type: Option<String>,
    /// Opaque JSON metadata. `null` if not set.
    pub metadata: Option<serde_json::Value>,
    /// SHA-256 content hash for insert-level dedup. `null` for pre-v0.1.6 episodes.
    pub content_hash: Option<String>,
}

/// Summary returned by `Memory.dream`.
///
/// Field names mirror the substrate `DreamSummary` struct exactly.
/// All numeric fields are safe to represent as JS `number` (f64) since they
/// are `usize`/`u64` values far below 2^53 at practical memory scale.
#[napi(object, js_name = "DreamSummary")]
pub struct JsDreamSummary {
    pub communities_updated: f64,
    pub cross_episode_merges: f64,
    pub supersessions_recorded: f64,
    pub facts_archived: f64,
    pub duration_ms: f64,
}

/// Options for `JsMemory.dream`. All fields are optional.
#[napi(object, js_name = "DreamOptions")]
pub struct JsDreamOpts {
    /// Namespace to dream within. `null`/omit for Memory handle's default.
    pub namespace: Option<String>,
}

/// Options for `JsMemory.rememberBatch` — bulk episode ingest per ADR-034.
///
/// Each entry maps to one `RememberBatchBuilder::entry(…).done()` call.
/// `batchId` is threaded through to `RememberBatchBuilder::with_batch_id`.
#[napi(object, js_name = "BatchOptions")]
pub struct JsBatchOptions {
    /// Episodes to ingest. Each entry mirrors `RememberOptions`.
    pub episodes: Vec<JsRememberOptions>,
    /// Optional stable batch ID for idempotent batch tracking and `awaitBatch`.
    pub batch_id: Option<String>,
}

/// Result of `JsMemory.statusOf` / `JsMemory.awaitEnrichment` per ADR-034.
///
/// Maps substrate `IngestStatus` to a flat JS object with a discriminator string
/// so consumers can switch on `result.status` without a Rust enum on the wire.
///
/// `status` is one of: `"pending"` | `"extracting"` | `"deduplicating"` |
/// `"invalidating"` | `"complete"` | `"failed"`.
/// `errorMessage` is set only when `status === "failed"`.
#[napi(object, js_name = "IngestStatusResult")]
pub struct JsIngestStatusResult {
    /// Discriminator string. See struct rustdoc for the full set.
    pub status: String,
    /// Human-readable error detail. Present only when `status === "failed"`.
    pub error_message: Option<String>,
}

/// Result of `JsMemory.awaitDream` per ADR-034.
///
/// Maps substrate `DreamStatus`. `status` is one of:
/// `"pending"` | `"processing"` | `"complete"` | `"failed"`.
#[napi(object, js_name = "DreamStatusResult")]
pub struct JsDreamStatusResult {
    /// Discriminator string. See struct rustdoc.
    pub status: String,
    /// Human-readable error detail. Present only when `status === "failed"`.
    pub error_message: Option<String>,
}

/// Result of `JsMemory.awaitBatch` per ADR-034.
///
/// Maps substrate `BatchStatus`. All counters are safe as JS `number` (f64)
/// since they are usize values at practical memory scale.
#[napi(object, js_name = "BatchStatus")]
pub struct JsBatchStatus {
    /// Total episodes submitted in this batch.
    pub total: f64,
    /// Episodes that completed Phase 2 enrichment successfully.
    pub completed: f64,
    /// Episodes skipped (enrich_per_episode=false).
    pub skipped: f64,
    /// Episodes whose Phase 2 enrichment failed.
    pub failed: f64,
}

/// Result of `JsMemory.cancel` / `JsMemory.cancelDream` per ADR-034.
///
/// Maps substrate `CancelOutcome`.
#[napi(object, js_name = "CancelOutcome")]
pub struct JsCancelOutcome {
    /// Which phase was cancelled: `"enrichment"` | `"consolidation"`.
    pub cancelled_phase: String,
    /// `true` when partial Phase 2 writes were rolled back transactionally.
    /// Always `false` for Phase 3 (partial committed state remains).
    pub rolled_back: bool,
    /// Entity IDs partially written before Phase 3 cancel (committed, not rolled back).
    pub partial: Vec<String>,
}

/// A single retrieved memory context entry from `Memory.recall`.
///
/// Maps directly to `kremory::RetrievedContext`.
#[napi(object, js_name = "RetrievedContext")]
pub struct JsRetrievedContext {
    /// Stable entity identifier in the kremory graph.
    pub entity_id: String,
    /// Human-readable entity name.
    pub entity_name: String,
    /// LLM-generated summary of the entity's known facts.
    pub summary: String,
    /// Relevance score in the range `[0.0, 1.0]`. Higher is more relevant.
    pub score: f64,
    /// Source reference IDs that contributed to this entity. Each entry is an
    /// opaque string key (e.g. session ID, document ID).
    pub source_refs: Vec<String>,
    /// `true` when the entity is a stub placeholder awaiting full extraction.
    pub incomplete: bool,
    /// Integer entity-type id for this entity within its namespace (TD-013).
    ///
    /// Mirrors `entities.entity_type_id`. `0` = "Entity" catch-all sentinel.
    /// Use `entity_type_name` for the human-readable label.
    pub entity_type_id: u32,
    /// Resolved entity type name (TD-013).
    ///
    /// Populated from `COALESCE(entity_types.name, 'Entity')` via the SQL JOIN
    /// already present in all entity SELECT paths. `"Entity"` is the fallback
    /// for the catch-all sentinel (id=0) or any unknown id.
    pub entity_type_name: String,
    /// The namespace this result was retrieved from. Set for both single-namespace
    /// (`namespace`) and multi-namespace (`in_namespaces`) recall. `None` for
    /// raw substrate-level queries that bypass the facade.
    ///
    /// Note: only the namespace string is surfaced. `thread` and `policy` fields
    /// on `kremory::Namespace` are not yet exposed via this binding.
    pub namespace: Option<String>,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a `kremory::RetrievedContext` to the napi-facing `JsRetrievedContext`.
pub fn retrieved_context_to_js(ctx: RetrievedContext) -> JsRetrievedContext {
    let source_refs = ctx
        .source_refs
        .iter()
        .map(|sr| sr.id.clone())
        .collect::<Vec<_>>();

    let namespace = ctx.namespace.map(|ns| ns.namespace);

    JsRetrievedContext {
        entity_id: ctx.entity_id,
        entity_name: ctx.entity_name,
        summary: ctx.summary,
        score: f64::from(ctx.score),
        source_refs,
        incomplete: ctx.incomplete,
        entity_type_id: ctx.entity_type_id,
        entity_type_name: ctx.entity_type_name,
        namespace,
    }
}

/// Resolve an optional namespace from `JsRecallOptions`.
pub fn resolve_recall_namespace(opts: &Option<JsRecallOptions>) -> Option<Namespace> {
    opts.as_ref()
        .and_then(|o| o.namespace.as_deref())
        .map(Namespace::new)
}

/// Resolve the multi-namespace selector from `JsRecallOptions` (ADR-029c).
/// Returns `None` if `in_namespaces` is unset or empty.
pub fn resolve_recall_namespaces(opts: &Option<JsRecallOptions>) -> Option<Vec<Namespace>> {
    opts.as_ref()
        .and_then(|o| o.in_namespaces.as_ref())
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().map(|s| Namespace::new(s.as_str())).collect())
}

/// Convert a substrate `kremory::core::schema::Episode` to `JsEpisode`.
///
/// `source_id` is populated from the call argument (the query parameter used
/// to look up this episode). `source_uri` is always `None` — the substrate
/// `recall_by_source_id` query does not SELECT that column (known gap, tracked
/// in parity-skip.toml).
pub fn episode_to_js(ep: kremory::core::schema::Episode, source_id: &str) -> JsEpisode {
    // id: i64 → f64. Safe: i64 values from SQLite rowid fit in f64 mantissa
    // (2^53 > i64::MAX is false but rowids in practice never exceed 2^53).
    // This is the standard napi-rs pattern for i64 → JS number.
    #[allow(clippy::cast_precision_loss)]
    let id_f64 = ep.id as f64;

    JsEpisode {
        id: id_f64,
        source_id: Some(source_id.to_string()),
        source_uri: None, // substrate recall_by_source_id does not return this column
        content: ep.content,
        timestamp: ep.timestamp.to_rfc3339(),
        source_type: ep.source_type,
        metadata: ep.metadata,
        content_hash: ep.content_hash,
    }
}

/// Convert a substrate `DreamSummary` to `JsDreamSummary`.
///
/// `usize`/`u64` → `f64` casts: safe up to 2^53 (~9 quadrillion).
/// At practical kremory scale these counters will never approach that limit.
#[allow(clippy::cast_precision_loss)]
pub fn dream_summary_to_js(s: DreamSummary) -> JsDreamSummary {
    JsDreamSummary {
        communities_updated: s.communities_updated as f64,
        cross_episode_merges: s.cross_episode_merges as f64,
        supersessions_recorded: s.supersessions_recorded as f64,
        facts_archived: s.facts_archived as f64,
        duration_ms: s.duration_ms as f64,
    }
}

/// Convert a substrate `kremory::IngestStatus` to `JsIngestStatusResult`.
///
/// Intermediate-state variants (`Deduplicating`, `Invalidating`) that were added
/// after ADR-034 are mapped to `"pending"` with a note so consumers are not
/// broken by future substrate additions.
pub fn ingest_status_to_js(s: kremory::IngestStatus) -> JsIngestStatusResult {
    match s {
        kremory::IngestStatus::Pending => JsIngestStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::Extracting => JsIngestStatusResult {
            status: "extracting".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::Deduplicating => JsIngestStatusResult {
            status: "deduplicating".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::Invalidating => JsIngestStatusResult {
            status: "invalidating".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::Complete => JsIngestStatusResult {
            status: "complete".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::Failed(msg) => JsIngestStatusResult {
            status: "failed".to_string(),
            error_message: Some(msg),
        },
        // Non-exhaustive guard: forward-compat for variants added after v0.1.8.
        _ => JsIngestStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
    }
}

/// Convert a substrate `kremory::DreamStatus` to `JsDreamStatusResult`.
pub fn dream_status_to_js(s: kremory::DreamStatus) -> JsDreamStatusResult {
    match s {
        kremory::DreamStatus::Pending => JsDreamStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Processing => JsDreamStatusResult {
            status: "processing".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Complete => JsDreamStatusResult {
            status: "complete".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Failed(msg) => JsDreamStatusResult {
            status: "failed".to_string(),
            error_message: Some(msg),
        },
        // Non-exhaustive guard: forward-compat for variants added after v0.1.8.
        _ => JsDreamStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
    }
}

/// Convert a substrate `kremory::BatchStatus` to `JsBatchStatus`.
///
/// `usize` → `f64` casts: safe up to 2^53 at practical memory scale.
#[allow(clippy::cast_precision_loss)]
pub fn batch_status_to_js(s: kremory::BatchStatus) -> JsBatchStatus {
    JsBatchStatus {
        total: s.total as f64,
        completed: s.completed as f64,
        skipped: s.skipped as f64,
        failed: s.failed as f64,
    }
}

/// Convert a substrate `kremory::CancelOutcome` to `JsCancelOutcome`.
pub fn cancel_outcome_to_js(o: kremory::CancelOutcome) -> JsCancelOutcome {
    let cancelled_phase = match o.cancelled_phase {
        kremory::CancelledPhase::Enrichment => "enrichment".to_string(),
        kremory::CancelledPhase::Consolidation => "consolidation".to_string(),
        // Non-exhaustive guard.
        _ => "enrichment".to_string(),
    };
    JsCancelOutcome {
        cancelled_phase,
        rolled_back: o.rolled_back,
        partial: o.partial,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use kremory::RetrievedContext;

    use super::retrieved_context_to_js;

    // Helper: build a minimal RetrievedContext via ::new() then patch
    // entity_type_id / entity_type_name directly (within-crate access allowed).
    fn make_ctx(entity_type_id: u32, entity_type_name: &str) -> RetrievedContext {
        let mut ctx = RetrievedContext::new(
            "ent-1",
            "Alice",
            "summary text",
            0.9_f32,
            vec![],
        );
        ctx.entity_type_id = entity_type_id;
        ctx.entity_type_name = entity_type_name.to_string();
        ctx
    }

    /// TD-013 Phase 8: entity_type_id is correctly wired as u32 on JsRetrievedContext.
    #[test]
    fn entity_type_id_wired_as_u32() {
        let ctx = make_ctx(1, "Person");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_id, 1u32, "entity_type_id must be u32 id=1");
    }

    /// TD-013 Phase 8: entity_type_name string matches known type for id=1.
    #[test]
    fn entity_type_name_matches_known_type() {
        let ctx = make_ctx(1, "Person");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_name, "Person");
    }

    /// TD-013 Phase 8: id=0 sentinel resolves to "Entity" fallback.
    #[test]
    fn entity_type_id_zero_resolves_to_entity_fallback() {
        let ctx = make_ctx(0, "Entity");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_id, 0u32);
        assert_eq!(js.entity_type_name, "Entity");
    }

    /// Backwards compatibility: existing label field is still present and correctly
    /// populated from entity_name (not entity_type_name).
    #[test]
    fn entity_name_field_unaffected_by_type_fields() {
        let ctx = make_ctx(2, "Organisation");
        let js = retrieved_context_to_js(ctx);
        // entity_name comes from properties["name"] or entity.id in the real recall
        // path; in this unit test it is the value passed to ::new().
        assert_eq!(js.entity_name, "Alice");
        // entity_type_name is additive — does not overwrite entity_name.
        assert_eq!(js.entity_type_name, "Organisation");
    }
}

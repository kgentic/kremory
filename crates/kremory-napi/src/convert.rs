//! Type-safe conversions between kremory Rust types and napi-rs JS objects.
//!
//! All `#[napi(object)]` structs here generate TypeScript `interface` declarations
//! in `index.d.ts` via the napi-rs derive macro pipeline.

use napi_derive::napi;

use kremory::{DreamSummary, Namespace, RetrievedContext};

// ── Input option structs ──────────────────────────────────────────────────────

/// Options for `Memory.open`.
#[napi(object)]
pub struct JsOpenOptions {
    /// Embedding vector dimensionality. Must match the active provider's output
    /// dimension. Defaults to 384 (MiniLM-L6-v2) if omitted.
    pub embedding_dim: Option<i64>,
    /// Default namespace applied to all operations on this handle when no
    /// per-call namespace is specified.
    pub default_namespace: Option<String>,
}

/// Options for `Memory.ingest`.
#[napi(object)]
pub struct JsIngestOptions {
    /// Namespace to scope this ingest. Overrides the Memory handle's default.
    pub namespace: Option<String>,
    /// ISO 8601 date string used as `published_at` bi-temporal anchor. e.g.
    /// `"2024-03-15T10:00:00Z"`.
    pub reference_time: Option<String>,
    /// Content type hint (e.g. `"chat"`, `"document"`). Currently informational;
    /// future versions may route to specialised ingest pipelines.
    pub content_type: Option<String>,
}

/// Draft passed to `JsMemory.ingestEpisode`. All fields except `content` are
/// optional — chat-grain consumers omit source_id/uri; doc-grain consumers supply them.
///
/// # Substrate-purity
///
/// `source_id`/`source_uri` are substrate-generic identifiers. Consumers own
/// the vocabulary — a consumer might pass a slug, path, UUID, conversation id, etc.
#[napi(object)]
pub struct JsEpisodeDraft {
    /// Episode text content. Required.
    pub content: String,
    /// Optional stable source identifier (consumer-defined: slug, file path, UUID, etc.).
    /// Stored as `source_id` on the episode row.
    pub source_id: Option<String>,
    /// Optional URI of the source document (e.g. file path, URL).
    /// Set via a post-ingest `update_source_uri` call.
    pub source_uri: Option<String>,
    /// Opaque JSON metadata to attach to this episode.
    /// Set via a post-ingest `update_episode_metadata` call.
    pub metadata: Option<serde_json::Value>,
    /// Namespace to scope this episode. Overrides the Memory handle's default.
    pub namespace: Option<String>,
}

/// A single metadata filter key/value pair for `JsRecallOptions.filterMetadata`.
///
/// Maps to `RecallRequest::filter_metadata(key, value)` on the substrate.
/// Multiple entries combine with AND.
#[napi(object)]
pub struct JsMetadataFilter {
    /// Top-level metadata key to match. Must be non-empty and ≤ 128 chars.
    pub key: String,
    /// Exact value the key must equal.
    pub value: serde_json::Value,
}

/// Options for `Memory.recall`.
#[napi(object)]
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

/// Result of a successful `Memory.ingest` or `Memory.ingestEpisode` call.
#[napi(object)]
pub struct JsIngestResult {
    /// Stable entity ID under which the episode is searchable immediately.
    pub episode_entity_id: String,
    /// ISO 8601 timestamp at which Phase 1 committed.
    pub committed_at: String,
    /// Non-fatal ingest warnings (e.g. content exceeded soft size threshold).
    /// Empty when no warnings were raised.
    pub warnings: Vec<String>,
}

/// A single episode returned by `Memory.getBySourceId`.
#[napi(object)]
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
#[napi(object)]
pub struct JsDreamSummary {
    pub communities_updated: f64,
    pub cross_episode_merges: f64,
    pub supersessions_recorded: f64,
    pub facts_archived: f64,
    pub duration_ms: f64,
}

/// Options for `JsMemory.dream`. All fields are optional.
#[napi(object)]
pub struct JsDreamOpts {
    /// Namespace to dream within. `null`/omit for Memory handle's default.
    pub namespace: Option<String>,
}

/// A single retrieved memory context entry from `Memory.recall`.
///
/// Maps directly to `kremory::RetrievedContext`.
#[napi(object)]
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
        namespace,
    }
}

/// Resolve an optional namespace from `JsIngestOptions`.
///
/// Returns `None` when no namespace is specified (Memory handle's default is
/// used by kremory internally).
pub fn resolve_ingest_namespace(opts: &Option<JsIngestOptions>) -> Option<Namespace> {
    opts.as_ref()
        .and_then(|o| o.namespace.as_deref())
        .map(Namespace::new)
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

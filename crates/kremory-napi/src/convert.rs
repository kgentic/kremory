//! Type-safe conversions between kremory Rust types and napi-rs JS objects.
//!
//! All `#[napi(object)]` structs here generate TypeScript `interface` declarations
//! in `index.d.ts` via the napi-rs derive macro pipeline.

use napi_derive::napi;

use kremory::{Namespace, RetrievedContext};

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

/// Options for `Memory.recall`.
#[napi(object)]
pub struct JsRecallOptions {
    /// Maximum number of results to return. Default: 10.
    pub k: Option<i64>,
    /// Namespace to scope this recall. Overrides the Memory handle's default.
    pub namespace: Option<String>,
    /// ISO 8601 point-in-time filter. Returns facts that were valid at this
    /// timestamp. e.g. `"2024-03-15T10:00:00Z"`.
    pub as_of: Option<String>,
}

// ── Output types ──────────────────────────────────────────────────────────────

/// Result of a successful `Memory.ingest` call.
#[napi(object)]
pub struct JsIngestResult {
    /// Stable entity ID under which the episode is searchable immediately.
    pub episode_entity_id: String,
    /// ISO 8601 timestamp at which Phase 1 committed.
    pub committed_at: String,
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
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a `kremory::RetrievedContext` to the napi-facing `JsRetrievedContext`.
pub fn retrieved_context_to_js(ctx: RetrievedContext) -> JsRetrievedContext {
    let source_refs = ctx
        .source_refs
        .iter()
        .map(|sr| sr.id.clone())
        .collect::<Vec<_>>();

    JsRetrievedContext {
        entity_id: ctx.entity_id,
        entity_name: ctx.entity_name,
        summary: ctx.summary,
        score: f64::from(ctx.score),
        source_refs,
        incomplete: ctx.incomplete,
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

//! Recall/retrieval napi conversions — `Memory.recall`, retrieved context/fact/source types.
//! Split out of `convert.rs` (TD-243); see `convert/mod.rs` for the domain map.

use napi_derive::napi;

use kremory::{Namespace, RetrievedContext, RetrievedFact, SourceKind, SourceRef};

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
    /// Multi-namespace recall. Set instead of `namespace` to query
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
    /// Rerank the top-`n` fused candidates with a cross-encoder before
    /// returning (substrate `SearchOpts::rerank_k`). Omitted (the
    /// default) = no rerank.
    ///
    /// Mirrors the substrate field's own always-present contract: the option
    /// is ACCEPTED regardless of build features, and is a documented no-op
    /// unless the binding was compiled with the `rerank` feature.
    /// Measurement found `n = 50` is the only depth that changes which items
    /// reach the top-10 — `n = 20` permutes the same set and leaves
    /// recall/hit-rate identical to no rerank at all.
    pub rerank_k: Option<i64>,
}

// ── Output types ──────────────────────────────────────────────────────────────

/// A single connected fact surfaced by recall.
///
/// Maps directly to `kremory::RetrievedFact`. Timestamps are RFC-3339 strings
/// and episode ids are stringified for JS ergonomics.
#[napi(object, js_name = "RetrievedFact")]
pub struct JsRetrievedFact {
    /// Row id of the underlying fact — the HANDLE that makes this fact nameable.
    ///
    /// Without it a caller can read a fact they know is wrong and have no way to
    /// say WHICH one. The facade added this field for exactly that reason
    /// (TD-244: "Without it, `supersede` was documented but UNREACHABLE"), and
    /// this binding dropped it — reintroducing the same unreachability on the
    /// Node surface, where the correction ops take an id the caller could not get.
    ///
    /// `i64` crosses to JS as a BigInt-backed number; kept as the raw row id
    /// rather than a string so it can be passed straight back to a correction
    /// call without a parse step.
    ///
    /// `None` when the item did not come from a fact row: a content passage
    /// retrieved by content-search has no fact id, and inventing one would be
    /// worse than admitting the absence.
    pub fact_id: Option<i64>,
    /// Natural-language rendering, e.g. `"Grace Hopper invented the compiler"`.
    pub fact: String,
    /// Subject entity display name.
    pub subject: String,
    /// Relation / predicate.
    pub predicate: String,
    /// Object — a literal value, or an object-entity's display name.
    pub object: String,
    /// `true` when `object` is an entity (edge), `false` when a literal value.
    pub object_is_entity: bool,
    /// World clock: when the fact became true (RFC-3339).
    pub valid_at: String,
    /// World clock: when the fact stopped being true, if ever (RFC-3339).
    pub invalid_at: Option<String>,
    /// System clock: when the fact was recorded (RFC-3339).
    pub recorded_at: String,
    /// System clock: when the fact row was superseded/expired, if ever (RFC-3339).
    pub expired_at: Option<String>,
    /// Extraction/caller confidence in `[0, 1]`.
    pub confidence: f64,
    /// Source episode id(s) this fact was asserted from (stringified).
    pub source_episode_ids: Vec<String>,
    /// Relevance score inherited from the anchoring entity.
    pub score: f64,
}

/// Convert a `kremory::RetrievedFact` to the napi-facing `JsRetrievedFact`.
pub fn retrieved_fact_to_js(f: RetrievedFact) -> JsRetrievedFact {
    JsRetrievedFact {
        fact_id: f.fact_id,
        fact: f.fact,
        subject: f.subject,
        predicate: f.predicate,
        object: f.object,
        object_is_entity: f.object_is_entity,
        valid_at: f.valid_at.to_rfc3339(),
        invalid_at: f.invalid_at.map(|d| d.to_rfc3339()),
        recorded_at: f.recorded_at.to_rfc3339(),
        expired_at: f.expired_at.map(|d| d.to_rfc3339()),
        confidence: f.confidence,
        source_episode_ids: f
            .source_episode_ids
            .iter()
            .map(|id| id.to_string())
            .collect(),
        score: f64::from(f.score),
    }
}

/// A single source reference contributing to a retrieved entity.
///
/// Maps directly to `kremory::SourceRef`, mirroring the MCP `SourceRefWire`
/// shape so the napi and MCP surfaces are two projections of one data model.
/// Timestamps are RFC-3339 strings.
#[napi(object, js_name = "SourceRef")]
pub struct JsSourceRef {
    /// Source category, e.g. `"meeting"` / `"document"` / `"chat"` / `"episode"`.
    /// Lower-cased string form of `kremory::SourceKind`.
    pub kind: String,
    /// Opaque source identifier (session ID, document ID, episode-edge id, …).
    pub id: String,
    /// RFC-3339 UTC timestamp the source event occurred at.
    pub occurred_at: String,
    /// RFC-3339 UTC publication timestamp of the source document/event, when
    /// known. `null` when the source has no distinct publication time.
    pub published_at: Option<String>,
}

/// Lower-cased wire string for a `SourceKind`. Mirrors the MCP
/// `source_kind_facade_to_wire` mapping. `SourceKind` is `#[non_exhaustive]`
/// upstream, so the catch-all keeps this forward-compatible — it surfaces an
/// unknown future variant loudly (via its `Debug` form) rather than silently
/// mislabeling it.
fn source_kind_to_string(kind: SourceKind) -> String {
    match kind {
        SourceKind::Meeting => "meeting".to_string(),
        SourceKind::Document => "document".to_string(),
        SourceKind::Chat => "chat".to_string(),
        SourceKind::Episode => "episode".to_string(),
        other => {
            tracing::warn!(?other, "unmapped SourceKind variant projected to napi");
            format!("{other:?}").to_lowercase()
        }
    }
}

/// Convert a `kremory::SourceRef` to the napi-facing `JsSourceRef`.
fn source_ref_to_js(sr: SourceRef) -> JsSourceRef {
    JsSourceRef {
        kind: source_kind_to_string(sr.kind),
        id: sr.id,
        occurred_at: sr.occurred_at.to_rfc3339(),
        published_at: sr.published_at.map(|t| t.to_rfc3339()),
    }
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
    /// Source references that contributed to this entity. Each entry
    /// carries `kind`/`id`/`occurred_at`/`published_at` — mirroring the MCP
    /// `SourceRefWire` surface so JS consumers get the same provenance the MCP
    /// wire already exposes (previously flattened to bare `id` strings).
    pub source_refs: Vec<JsSourceRef>,
    /// `true` when the entity is a stub placeholder awaiting full extraction.
    pub incomplete: bool,
    /// Integer entity-type id for this entity within its namespace.
    ///
    /// Mirrors `entities.entity_type_id`. `0` = "Entity" catch-all sentinel.
    /// Use `entity_type_name` for the human-readable label.
    pub entity_type_id: u32,
    /// Resolved entity type name.
    ///
    /// Populated from `COALESCE(entity_types.name, 'Entity')` via the SQL JOIN
    /// already present in all entity SELECT paths. `"Entity"` is the fallback
    /// for the catch-all sentinel (id=0) or any unknown id.
    pub entity_type_name: String,
    /// `true` when this result came from the **content-search** arm (a passage of
    /// episode text) rather than from the entity graph.
    ///
    /// The JS projection of `RetrievedContext::is_content_passage()` — a FIELD
    /// rather than a method, because values crossing the napi boundary are plain
    /// objects. Same information, the modality's own shape.
    ///
    /// **Read this before using `entityId`.** Since `content-search` became a
    /// default, `recall(..).raw()` returns a HETEROGENEOUS list, and a passage's
    /// `entityId` is an **episode id**, not an entity id — passing it to any
    /// entity-scoped call fails with `no entity '<n>' in namespace '<ns>'`.
    /// `entityTypeId` cannot disambiguate either: a passage carries `0`, which is
    /// also the unknown-entity catch-all.
    ///
    /// ```js
    /// const entities = (await mem.recall("Alice").raw())
    ///   .filter(r => !r.isContentPassage);
    /// ```
    ///
    /// The Rust consumer E2E fell into exactly this same trap, and the JS
    /// surface had it too.
    pub is_content_passage: bool,
    /// The namespace this result was retrieved from. Set for both single-namespace
    /// (`namespace`) and multi-namespace (`in_namespaces`) recall. `None` for
    /// raw substrate-level queries that bypass the facade.
    ///
    /// Note: only the namespace string is surfaced. `thread` and `policy` fields
    /// on `kremory::Namespace` are not yet exposed via this binding.
    pub namespace: Option<String>,
    /// Connected facts anchored on this entity — the
    /// LLM-consumable knowledge (natural-language fact strings + structured
    /// triple + both bi-temporal clocks + confidence + provenance). Empty for
    /// entities with no connected facts.
    pub facts: Vec<JsRetrievedFact>,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a `kremory::RetrievedContext` to the napi-facing `JsRetrievedContext`.
pub fn retrieved_context_to_js(ctx: RetrievedContext) -> JsRetrievedContext {
    // FIRST, before any field of `ctx` is moved out below. Derived from the
    // SUBSTRATE predicate rather than by re-testing the string here — two
    // independent implementations of one discriminator is how the JS surface
    // silently drifts from the Rust one.
    let is_content_passage = ctx.is_content_passage();

    let source_refs = ctx
        .source_refs
        .into_iter()
        .map(source_ref_to_js)
        .collect::<Vec<_>>();

    let namespace = ctx.namespace.map(|ns| ns.namespace);
    let facts = ctx.facts.into_iter().map(retrieved_fact_to_js).collect();

    JsRetrievedContext {
        entity_id: ctx.entity_id,
        entity_name: ctx.entity_name,
        summary: ctx.summary,
        score: f64::from(ctx.score),
        source_refs,
        incomplete: ctx.incomplete,
        entity_type_id: ctx.entity_type_id,
        entity_type_name: ctx.entity_type_name,
        is_content_passage,
        namespace,
        facts,
    }
}

/// Resolve an optional namespace from `JsRecallOptions`.
pub fn resolve_recall_namespace(opts: &Option<JsRecallOptions>) -> Option<Namespace> {
    opts.as_ref()
        .and_then(|o| o.namespace.as_deref())
        .map(Namespace::new)
}

/// Resolve the multi-namespace selector from `JsRecallOptions`.
/// Returns `None` if `in_namespaces` is unset or empty.
pub fn resolve_recall_namespaces(opts: &Option<JsRecallOptions>) -> Option<Vec<Namespace>> {
    opts.as_ref()
        .and_then(|o| o.in_namespaces.as_ref())
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().map(|s| Namespace::new(s.as_str())).collect())
}


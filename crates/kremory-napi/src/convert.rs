//! Type-safe conversions between kremory Rust types and napi-rs JS objects.
//!
//! All `#[napi(object)]` structs here generate TypeScript `interface` declarations
//! in `index.d.ts` via the napi-rs derive macro pipeline.

use napi_derive::napi;

use kremory::{DreamSummary, Namespace, RetrievedContext, RetrievedFact, SourceKind, SourceRef};

// ── Input option structs ──────────────────────────────────────────────────────

/// GLiNER configuration passed via `MemoryOpenOptionsJs.gliner` (ADR-039 Part 10).
///
/// # ⚠ Reserved / inert knobs — presence is the ONLY live signal
///
/// The PRESENCE of a `gliner` object is what enables `ExtractorKind::GlinerLlm`;
/// its FIELDS are currently RESERVED and inert on the wire. The substrate
/// `with_gliner()` builder takes no config argument today, so `modelPath` and
/// `threshold` are accepted but NOT threaded through — pass `{}` to enable GLiNER
/// with substrate defaults. To tune the span threshold today, set the
/// `KREMORY_GLINER_THRESHOLD` env var (the substrate reads it, falling back to
/// 0.3). These fields are retained (not removed) so the JS shape is forward-stable
/// for when `GlinerConfig` gains public tuning knobs (ADR-039 §A6); they will
/// become live then.
///
/// Requires kremory-napi built with `--features ner`. Passing this field on a
/// non-ner build causes `Memory.open` to return an error.
#[napi(object, js_name = "GlinerConfig")]
#[derive(Default)]
pub struct GlinerConfigJs {
    /// RESERVED / inert — model file path override. Not threaded to the substrate
    /// today (`with_gliner()` takes no arg); reserved for a future ADR-039 §A6
    /// tuning surface. `null` = use default bundled model.
    pub model_path: Option<String>,
    /// RESERVED / inert — span-detection confidence threshold in `[0.0, 1.0]`. Not
    /// threaded to the substrate today; set `KREMORY_GLINER_THRESHOLD` env instead
    /// (substrate default `0.3`). Reserved for a future ADR-039 §A6 tuning surface.
    pub threshold: Option<f64>,
}

// (F2) `From<GlinerConfigJs> for GlinerConfig` removed — `with_gliner()` no longer
// takes a config arg, so the conversion had no caller (dead code). GlinerConfigJs
// is retained as the JS-side enable-signal (presence of `opts.gliner`). Restore a
// mapping here if/when GlinerConfig gains public tuning knobs (ADR-039 §A6).

/// Options for `Memory.open` — live napi/cdylib build (Shape B, ADR-039 Part 10).
///
/// Replaces the old string-literal `extractor: 'auto'|'hybrid'|'nuextract'` API.
/// Mirrors the Rust `MemoryBuilder` composable knobs:
///   - `{ embedder, llm }`          → `ExtractorKind::IntegerId` (default since ADR-056;
///                                     3-stage integer-ID extractor. Was `Llm`/graphiti.)
///   - `{ embedder, llm, gliner }`  → `ExtractorKind::GlinerLlm`
///   - `{ embedder, extractor }`    → `ExtractorKind::Custom` (NoLlm typestate)
///   - `{ embedder, llm, extractor }` → `ExtractorKind::Custom` (WithLlm typestate)
///   - `{ embedder, gliner, extractor }` → `Err` (conflict)
///   - `{ embedder }` alone          → `Err` (no extractor wired)
///
/// `object_to_js = false`: skip `ToNapiValue` generation so `ThreadsafeFunction`
/// (which is `Send + Sync` but lacks `ToNapiValue`) can be used as a field.
/// `JsOpenOptions` is only ever constructed from JS → Rust, never returned.
///
/// Excluded from `cfg(test)` because napi-derive macro-generated `FromNapiValue`
/// references `ThreadsafeFunction` symbols only available in a live napi runtime.
#[cfg(not(test))]
#[napi(object, object_to_js = false, js_name = "OpenOptions")]
pub struct JsOpenOptions {
    /// Embedding vector dimensionality. Must match the callback's output
    /// dimension when `with_embedder` is set. Ignored when `with_embedder`
    /// is absent.
    pub embedding_dim: Option<i64>,
    /// Default namespace applied to all operations on this handle when no
    /// per-call namespace is specified.
    pub default_namespace: Option<String>,
    /// Optional BYOM embedder callback (ADR-030 Tier-2).
    ///
    /// Callback signature: `(text: string) => Promise<number[]>`.
    ///
    /// When set, kremory wires this JS function as the embedding provider via
    /// the `MemoryBuilder` Tier-2 path.
    #[napi(ts_type = "((text: string) => Promise<number[]>) | undefined | null")]
    pub with_embedder: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            String,
            napi::threadsafe_function::ErrorStrategy::CalleeHandled,
        >,
    >,
    /// GLiNER configuration (ADR-039 Shape B). When set, selects
    /// `ExtractorKind::GlinerLlm` and requires `llm` to also be set.
    /// Requires kremory-napi built with `--features ner`.
    pub gliner: Option<GlinerConfigJs>,
    /// External (BYOE) extractor bridge (ADR-039 Shape B).
    ///
    /// When set, selects `ExtractorKind::Custom`. Mutually exclusive with
    /// `gliner` — setting both returns an error.
    ///
    /// JS shape: `{ extract(text: string, ctx: object): Promise<ExtractionResult>, name(): string }`.
    /// See `ExternalExtractorJs` in bridge.rs for the full interface.
    #[napi(
        ts_type = "{ extract(text: string, ctx: object): Promise<{ entities: Array<{ name: string, label: string }>, facts: Array<{ subject: string, predicate: string, object: string }> }>, name(): string } | undefined | null"
    )]
    pub extractor: Option<crate::bridge::ExternalExtractorHandle>,
}

/// Options for `Memory.open` — test build (no napi runtime).
///
/// Omits `with_embedder` and `extractor` — `ThreadsafeFunction` cannot be
/// constructed outside the napi cdylib runtime. Unit tests that exercise bridge
/// logic use mock variants in `bridge.rs` directly.
///
/// Implements `FromNapiValue` as a stub so `#[napi]` on `JsMemory::open` compiles
/// in test mode. The impl is never invoked — no napi runtime is present in test
/// binary builds.
#[cfg(test)]
pub struct JsOpenOptions {
    /// Embedding vector dimensionality.
    pub embedding_dim: Option<i64>,
    /// Default namespace.
    pub default_namespace: Option<String>,
    /// GLiNER config — test builds use Option<()> placeholder.
    pub gliner: Option<GlinerConfigJs>,
}

#[cfg(test)]
impl napi::bindgen_prelude::FromNapiValue for JsOpenOptions {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _nv: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        // Never called in test builds — napi runtime is absent.
        // Stub satisfies the trait bound required by #[napi] on JsMemory::open.
        Ok(JsOpenOptions {
            embedding_dim: None,
            default_namespace: None,
            gliner: None,
        })
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
                serde_json::from_value::<kremory::MemoryType>(serde_json::Value::String(
                    s.to_string(),
                ))
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
    /// Rerank the top-`n` fused candidates with a cross-encoder before
    /// returning (substrate `SearchOpts::rerank_k` / TD-062). Omitted (the
    /// default) = no rerank.
    ///
    /// Mirrors the substrate field's own always-present contract: the option
    /// is ACCEPTED regardless of build features (Rule 16 surface parity), and
    /// is a documented no-op unless the binding was compiled with the
    /// `rerank` feature. ADR-078 measured `n = 50` as the only depth that
    /// changes which items reach the top-10 — `n = 20` permutes the same set
    /// and leaves recall/hit-rate identical to no rerank at all.
    pub rerank_k: Option<i64>,
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
    /// URI of the source document. Populated from the `source_uri` column on
    /// the `episodes` table (added by Migration 007). `null` when no URI was
    /// set at ingest time.
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

/// A new entity type proposed + accepted by Dream Pass 0 type-discovery.
/// Mirrors substrate `kremory::core::dream::TypeProposal`.
#[napi(object, js_name = "TypeProposal")]
pub struct JsTypeProposal {
    pub name: String,
    pub description: String,
    pub justification: String,
}

/// Per-op ACTUALLY-RAN signal on `DreamSummary` (consumer-API hardening D1b).
///
/// Mirrors substrate `kremory::ConsolidationOpsRan`. With the all-ops-ON dream
/// defaults, a consumer can no longer read an all-zero consolidation count as
/// "the op was off" — a field here is `true` iff the op's `include_*` flag was on
/// AND the op executed. So `communities_updated == 0 && consolidationOpsRan.community
/// == true` reads as "community detection ran and found nothing", NOT "disabled".
#[napi(object, js_name = "ConsolidationOpsRan")]
pub struct JsConsolidationOpsRan {
    /// The community-detection op (`includeCommunityDetection`) executed.
    pub community: bool,
    /// The cross-episode merge op (`crossEpisodeMode != "off"`) executed.
    pub cross_episode: bool,
    /// The fact-archival op (`includeFactArchival`) executed.
    pub archival: bool,
    /// The supersession sweep (`includeSupersessionSweep`) executed.
    pub supersession_sweep: bool,
}

/// Summary returned by `Memory.dream`.
///
/// Field names mirror the substrate `DreamSummary` struct exactly. All numeric
/// fields are safe as JS `number` (f64) — usize/u64 values far below 2^53 at
/// practical memory scale.
///
/// The `crossEpisodeWouldMerge` / `crossEpisodeMerged` split (D5) is the in-band
/// would-merge/did-merge distinction — `wouldMerge` counts every merge DECISION
/// (both shadow + apply branches), `merged` counts only ACTUAL fusions (equals
/// `wouldMerge` in apply mode, `0` in shadow). Use `consolidationOpsRan` (D1b) to
/// tell "op disabled" from "op ran, found nothing" on any all-zero count.
#[napi(object, js_name = "DreamSummary")]
pub struct JsDreamSummary {
    pub communities_updated: f64,
    /// Cross-episode merge DECISIONS this pass ("would-merge", D5). Does NOT imply
    /// entities were fused — see `crossEpisodeMerged`.
    pub cross_episode_would_merge: f64,
    /// Cross-episode merges that ACTUALLY committed this pass (D5). Equals
    /// `crossEpisodeWouldMerge` in apply mode, `0` in shadow (the default).
    pub cross_episode_merged: f64,
    pub supersessions_recorded: f64,
    pub facts_archived: f64,
    /// Per-op ran-signal (D1b) — disambiguates "op disabled" from "op ran, found
    /// nothing" for the all-zero consolidation counts above.
    pub consolidation_ops_ran: JsConsolidationOpsRan,
    /// `true` when a consolidation op was skipped because a per-pass budget
    /// ceiling (token or USD) tripped — distinguishes "nothing to spend" from
    /// "spend was capped" (ADR-071 §Item 4a / TD-060).
    pub budget_exhausted: bool,
    pub duration_ms: f64,
    pub types_discovered: Vec<JsTypeProposal>,
    pub entities_reclassified: f64,
    pub aliases_resolved: f64,
    pub canonicalization_merges: f64,
    pub acronym_nickname_merges: f64,
    pub type_registry_merges: f64,
    pub consistency_check_corrected: f64,
    pub warnings: Vec<String>,
}

/// Options for `JsMemory.dream`. All fields are optional — omit to use the
/// substrate `DreamOpts::default()` (consumer-API hardening D1: all four
/// consolidation ops default ON, made safe by ADR-073 Tier-1 reversibility).
///
/// # DreamOpts field enumeration (D2 / `schemas-enumerate-touching-layers`)
///
/// The substrate `DreamOpts` has 18 pub fields. Each is either EXPOSED here or
/// deliberately OMITTED with a reason:
///
/// **Exposed (consolidation control — the D2 binding-parity surface):**
/// `includeCommunityDetection`, `includeFactArchival`, `includeSupersessionSweep`,
/// `includeSupersessionLlmNominate`, `crossEpisodeMode` (the honest tri-state that
/// maps onto `include_cross_episode_merges` + `cross_episode_dry_run` per O1 —
/// exposed as ONE string, never the two raw bools, to prevent the illegal
/// `apply`+`dry_run` combo), `archiveGraceDays`, `netMutationWarnFloor`,
/// `consolidationBudgetTokens`, `consolidationBudgetUsdMicro`.
///
/// **Omitted (with reason):**
/// - `since`, `maxEpisodesPerRun` — run-scoping filters; the napi `dream()` operates
///   over all un-dreamed episodes. Per-run scoping is reachable via
///   `runDreamPassSync`'s `DreamPassOptions.maxEpisodesPerRun`; a timestamp `since`
///   filter is deferred (v0.2.0).
/// - `includeTypeDiscovery`, `includeConsistencyCheck`, `includeTypeRegistryCollapse`,
///   `includeAcronymNicknameRecall`, `includeTypeNoveltyLlmVerify` — RECONCILIATION
///   pass toggles (a different concern from the consolidation sub-phase this D2
///   surface targets). All default-ON; type-discovery is separately tunable via
///   `runDreamPassSync`'s `DreamPassOptions.includeTypeDiscovery`. Exposing the full
///   reconciliation-pass matrix on `dream()` is deferred (v0.2.0).
/// - `includeEvidenceRetypeBySimilarity` — reason: unvalidated, TD-123
///   quarantine, default-false. NOT exposed on `JsDreamOpts`; the default-false
///   pass-through (substrate `DreamOpts::default()`) is the correct binding
///   behaviour until TD-123 lifts the quarantine.
#[napi(object, js_name = "DreamOptions")]
pub struct JsDreamOpts {
    /// Namespace to dream within. `null`/omit for Memory handle's default.
    pub namespace: Option<String>,
    /// Run the community-detection consolidation op (zero-LLM label propagation).
    /// Default `true`.
    pub include_community_detection: Option<bool>,
    /// Run the fact-archival consolidation op (MOVE long-expired unreferenced facts
    /// into `facts_archive`; reversible via `restoreArchivedFact`). Default `true`.
    pub include_fact_archival: Option<bool>,
    /// Run the supersession sweep (deterministic world-time window close-out;
    /// reversible via `unsupersede`). Default `true`.
    pub include_supersession_sweep: Option<bool>,
    /// Also run supersession's OPT-IN LLM-nominated value-change lane (only
    /// meaningful when `includeSupersessionSweep` is on). Default `false`.
    pub include_supersession_llm_nominate: Option<bool>,
    /// Cross-episode entity-merge control (O1). One of `"off"` | `"shadow"` |
    /// `"apply"` — the honest tri-state that maps onto the substrate's coupled
    /// `include_cross_episode_merges` + `cross_episode_dry_run` bools. Default
    /// (omit) = the substrate default (`"shadow"` — computes decisions, fuses
    /// nothing). An unknown value rejects the `dream()` Promise.
    pub cross_episode_mode: Option<String>,
    /// Grace window (days) before an expired fact is archival-eligible. Default `90`.
    pub archive_grace_days: Option<f64>,
    /// WARN-only floor on the aggregate destructive-mutation count. Default `500`.
    pub net_mutation_warn_floor: Option<f64>,
    /// Per-run token budget ceiling for the consolidation sub-phase (soft
    /// partial-abort). Default `50000`.
    pub consolidation_budget_tokens: Option<f64>,
    /// Per-run USD-micro budget ceiling. INERT today (every op's per-call USD
    /// projection is 0 — the effective budget is token-based); reserved for a
    /// future real cost source (D6). Default: no USD cap.
    pub consolidation_budget_usd_micro: Option<f64>,
}

/// Convert JS `DreamOptions` to substrate `DreamOpts` via FIELD MUTATION seeded
/// from `DreamOpts::default()` (compatible with the substrate's `#[non_exhaustive]`
/// posture — never a struct literal). Template = `js_dream_pass_opts_to_rust`.
///
/// `namespace` is NOT consumed here — it is resolved separately by `dream()`.
/// An unknown `cross_episode_mode` string fails LOUDLY (`llm-output-parse-loudly`
/// extended to consumer input) rather than silently defaulting.
pub fn js_dream_opts_to_rust(js: Option<JsDreamOpts>) -> napi::Result<kremory::DreamOpts> {
    let mut base = kremory::DreamOpts::default();
    let Some(o) = js else {
        return Ok(base);
    };
    if let Some(v) = o.include_community_detection {
        base.include_community_detection = v;
    }
    if let Some(v) = o.include_fact_archival {
        base.include_fact_archival = v;
    }
    if let Some(v) = o.include_supersession_sweep {
        base.include_supersession_sweep = v;
    }
    if let Some(v) = o.include_supersession_llm_nominate {
        base.include_supersession_llm_nominate = v;
    }
    if let Some(v) = o.archive_grace_days {
        base.archive_grace_days = Some(v as u32);
    }
    if let Some(v) = o.net_mutation_warn_floor {
        base.net_mutation_warn_floor = Some(v as usize);
    }
    if let Some(v) = o.consolidation_budget_tokens {
        base.consolidation_budget_tokens = Some(v as u64);
    }
    if let Some(v) = o.consolidation_budget_usd_micro {
        base.consolidation_budget_usd_micro = Some(v as u64);
    }
    // O1: cross_episode as a single mode string, mapped onto the two coupled bools
    // (never exposed raw — prevents the illegal apply+dry_run combination). Parse
    // loudly: an unknown value is a caller error, surfaced not silently defaulted.
    if let Some(mode) = o.cross_episode_mode.as_deref() {
        match mode {
            "off" => base.include_cross_episode_merges = false,
            "shadow" => {
                base.include_cross_episode_merges = true;
                base.cross_episode_dry_run = true;
            }
            "apply" => {
                base.include_cross_episode_merges = true;
                base.cross_episode_dry_run = false;
            }
            other => {
                return Err(napi::Error::from_reason(format!(
                    "JsDreamOpts.cross_episode_mode: unknown value {other:?} \
                     (expected \"off\" | \"shadow\" | \"apply\")"
                )));
            }
        }
    }
    Ok(base)
}

/// Options for `JsMemory.runDreamPassSync` (Phase C DoD C2 / C7).
///
/// All fields optional — omit to use defaults from `DreamPassOpts::default()`.
#[napi(object, js_name = "DreamPassOptions")]
pub struct JsDreamPassOpts {
    /// When `true`, Pass 0 type-discovery runs to find novel entity types.
    /// Requires LLM. Default: `false`.
    pub include_type_discovery: Option<bool>,
    /// Minimum confidence threshold for entity type assignments to be re-examined
    /// during the reclassify pass. Range `[0.0, 1.0]`. Default: `0.5`.
    pub confidence_threshold: Option<f64>,
    /// Cap on the number of ghost episodes processed per run.
    /// `null`/omit = process all available ghost episodes.
    pub max_episodes_per_run: Option<f64>,
    /// Confidence threshold above which `ConsumerPinned` entities are protected
    /// from reclassification. Range `[0.0, 1.0]`. Default: `0.7`.
    pub reclassify_high_conf_threshold: Option<f64>,
}

/// Convert JS `DreamPassOptions` to substrate `DreamPassOpts`.
pub fn js_dream_pass_opts_to_rust(js: Option<JsDreamPassOpts>) -> kremory::DreamPassOpts {
    let base = kremory::DreamPassOpts::default();
    match js {
        None => base,
        Some(o) => kremory::DreamPassOpts {
            include_type_discovery: o
                .include_type_discovery
                .unwrap_or(base.include_type_discovery),
            confidence_threshold: o
                .confidence_threshold
                .map(|v| v as f32)
                .unwrap_or(base.confidence_threshold),
            max_episodes_per_run: o
                .max_episodes_per_run
                .map(|v| Some(v as usize))
                .unwrap_or(base.max_episodes_per_run),
            reclassify_high_conf_threshold: o
                .reclassify_high_conf_threshold
                .map(|v| v as f32)
                .unwrap_or(base.reclassify_high_conf_threshold),
        },
    }
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

/// A single connected fact surfaced by recall (ADR-074 / TD-116).
///
/// Maps directly to `kremory::RetrievedFact`. Timestamps are RFC-3339 strings
/// and episode ids are stringified for JS ergonomics.
#[napi(object, js_name = "RetrievedFact")]
pub struct JsRetrievedFact {
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

/// A single source reference contributing to a retrieved entity (TD-118).
///
/// Maps directly to `kremory::SourceRef`, mirroring the MCP `SourceRefWire`
/// shape so the napi and MCP surfaces are two projections of one data model
/// (`web-app-ui-parity-for-agents`). Timestamps are RFC-3339 strings.
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

/// Convert a `kremory::SourceRef` to the napi-facing `JsSourceRef` (TD-118).
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
    /// Source references that contributed to this entity (TD-118). Each entry
    /// carries `kind`/`id`/`occurred_at`/`published_at` — mirroring the MCP
    /// `SourceRefWire` surface so JS consumers get the same provenance the MCP
    /// wire already exposes (previously flattened to bare `id` strings).
    pub source_refs: Vec<JsSourceRef>,
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
    /// Connected facts anchored on this entity (ADR-074 / TD-116) — the
    /// LLM-consumable knowledge (natural-language fact strings + structured
    /// triple + both bi-temporal clocks + confidence + provenance). Empty for
    /// entities with no connected facts.
    pub facts: Vec<JsRetrievedFact>,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a `kremory::RetrievedContext` to the napi-facing `JsRetrievedContext`.
pub fn retrieved_context_to_js(ctx: RetrievedContext) -> JsRetrievedContext {
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
/// TD-003 Phase G: `source_id` and `source_uri` are now taken directly from the
/// `Episode` struct (columns were always in the DB; G-2 adds them to the Rust type
/// and fixes the SELECT projections). The previous workaround that took `source_id`
/// as a separate `&str` argument and hardcoded `source_uri: None` is removed.
pub fn episode_to_js(ep: kremory::core::schema::Episode) -> JsEpisode {
    // id: i64 → f64. Safe: i64 values from SQLite rowid fit in f64 mantissa
    // (2^53 > i64::MAX is false but rowids in practice never exceed 2^53).
    // This is the standard napi-rs pattern for i64 → JS number.
    let id_f64 = ep.id as f64;

    JsEpisode {
        id: id_f64,
        source_id: ep.source_id,
        source_uri: ep.source_uri,
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
pub fn dream_summary_to_js(s: DreamSummary) -> JsDreamSummary {
    JsDreamSummary {
        communities_updated: s.communities_updated as f64,
        cross_episode_would_merge: s.cross_episode_would_merge as f64,
        cross_episode_merged: s.cross_episode_merged as f64,
        supersessions_recorded: s.supersessions_recorded as f64,
        facts_archived: s.facts_archived as f64,
        consolidation_ops_ran: JsConsolidationOpsRan {
            community: s.consolidation_ops_ran.community,
            cross_episode: s.consolidation_ops_ran.cross_episode,
            archival: s.consolidation_ops_ran.archival,
            supersession_sweep: s.consolidation_ops_ran.supersession_sweep,
        },
        budget_exhausted: s.budget_exhausted,
        duration_ms: s.duration_ms as f64,
        types_discovered: s
            .types_discovered
            .into_iter()
            .map(|t| JsTypeProposal {
                name: t.name,
                description: t.description,
                justification: t.justification,
            })
            .collect(),
        entities_reclassified: s.entities_reclassified as f64,
        aliases_resolved: s.aliases_resolved as f64,
        canonicalization_merges: s.canonicalization_merges as f64,
        acronym_nickname_merges: s.acronym_nickname_merges as f64,
        type_registry_merges: s.type_registry_merges as f64,
        consistency_check_corrected: s.consistency_check_corrected as f64,
        warnings: s.warnings,
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
        // ── DUR-3 wire-layer cause-fix (V1-CANONICAL §4.2, 2026-08-04) ───────
        // These three arms were all falling through the catch-all below and
        // being reported to JS as "pending". Two of them are TERMINAL, so a Node
        // consumer polling `statusOf` would poll forever on an episode that was
        // already finished — DUR-3's exact shape, on the napi surface.
        //
        // `EntitiesReady` is the pre-existing one and the worst: it is SQL
        // 'Verified', the state `Memory::wait_for_processing` resolves `Ok(())`
        // on. Rust callers saw success; JS callers saw "pending".
        //
        // Found by enumerating every layer the new status value touches, not by
        // a gate — `api_parity` walks METHODS, so an enum variant that silently
        // widens the catch-all is invisible to it. `ingest_status_maps_every_known_variant_off_the_catch_all`
        // below is the guard that makes the next one visible.
        kremory::IngestStatus::EntitiesReady => JsIngestStatusResult {
            status: "entities_ready".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::ExtractionSkipped => JsIngestStatusResult {
            status: "skipped".to_string(),
            error_message: None,
        },
        kremory::IngestStatus::SkippedIdempotent => JsIngestStatusResult {
            status: "skipped_idempotent".to_string(),
            error_message: None,
        },
        // Non-exhaustive guard: forward-compat for variants added after v0.1.8.
        // `IngestStatus` is `#[non_exhaustive]`, so this arm cannot be removed —
        // which is exactly why it must not be allowed to accumulate silent
        // members. See the test named above.
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

/// Outcome of `JsMemory.supersede` (ADR-071 §Item 3, consumer-API hardening D4).
///
/// `outcome` is one of `"bounded"` | `"rejected_time_inversion"` | `"not_found"`
/// (mirrors `kremory::SupersedeOutcome` + the `outcome` label on
/// `kremory.dream.consolidation.supersede_request_total` 1:1). The honest rename
/// `Applied` → `Bounded` (D4a): `execute()` only BOUNDS `valid_to`; it does not
/// retire the fact. `retired` carries the inline `.closeNow()` window-closeout
/// count IN-BAND (D4b) — `0` for a plain supersede OR a future-dated bound whose
/// window has not yet closed, so a consumer observes the deferral from the return
/// value, not a doc caveat.
#[napi(object, js_name = "SupersedeOutcome")]
pub struct JsSupersedeOutcome {
    /// `"bounded"` | `"rejected_time_inversion"` | `"not_found"`.
    pub outcome: String,
    /// Facts retired by an inline `.closeNow()` sweep (`0` otherwise).
    /// Namespace-scoped total.
    pub retired: f64,
}

/// Convert a substrate `kremory::SupersedeOutcome` to `JsSupersedeOutcome`.
pub fn supersede_outcome_to_js(o: kremory::SupersedeOutcome) -> JsSupersedeOutcome {
    match o {
        kremory::SupersedeOutcome::Bounded { retired } => JsSupersedeOutcome {
            outcome: "bounded".to_string(),
            retired: retired as f64,
        },
        kremory::SupersedeOutcome::RejectedTimeInversion => JsSupersedeOutcome {
            outcome: "rejected_time_inversion".to_string(),
            retired: 0.0,
        },
        kremory::SupersedeOutcome::NotFound => JsSupersedeOutcome {
            outcome: "not_found".to_string(),
            retired: 0.0,
        },
        // Non-exhaustive guard: a variant added after this build maps to an honest
        // "unknown" marker (never a silent mis-label as "bounded").
        _ => JsSupersedeOutcome {
            outcome: "unknown".to_string(),
            retired: 0.0,
        },
    }
}

// ── Reversible-graph-mutations (ADR-073 Tier-1) outcome + inspect mirrors ──────

/// Outcome of `JsMemory.unmerge` — mirrors `kremory::UnmergeOutcome`. Every count
/// is the ACTUAL number reversed (success-signal honesty, §3.1). All counts as JS
/// `number` (f64): usize values far below 2^53 at practical scale.
#[napi(object, js_name = "UnmergeOutcome")]
pub struct JsUnmergeOutcome {
    /// The restored loser entity id (the entity that had been hard-DELETEd).
    pub restored_entity: String,
    /// The keeper whose overwritten access_count / ner_confidence were restored.
    pub keeper: String,
    /// Facts whose endpoint + corroboration_inert flag were reverted.
    pub facts_repointed: f64,
    /// Episodic edges re-inserted (collided) or re-pointed back (non-collided).
    pub edges_restored: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// A NOGOOD was recorded for the split pair (next `dream()` will not re-merge).
    pub nogood_recorded: bool,
    /// `true` if the mutation was already undone — an idempotent no-op (all counts
    /// zero, `nogoodRecorded = false`), never a re-application.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::UnmergeOutcome` to `JsUnmergeOutcome`.
pub fn unmerge_outcome_to_js(o: kremory::UnmergeOutcome) -> JsUnmergeOutcome {
    JsUnmergeOutcome {
        restored_entity: o.restored_entity,
        keeper: o.keeper,
        facts_repointed: o.facts_repointed as f64,
        edges_restored: o.edges_restored as f64,
        entities_reopened: o.entities_reopened as f64,
        nogood_recorded: o.nogood_recorded,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.restoreArchivedFact` — mirrors
/// `kremory::RestoreArchivedOutcome`.
#[napi(object, js_name = "RestoreArchivedOutcome")]
pub struct JsRestoreArchivedOutcome {
    /// The `facts` row id restored from `facts_archive`.
    pub restored_fact_id: i64,
    /// `true` if the fact was already live — an honest no-op, nothing restored.
    pub already_live: bool,
}

/// Convert a substrate `kremory::RestoreArchivedOutcome` to
/// `JsRestoreArchivedOutcome`.
pub fn restore_archived_outcome_to_js(
    o: kremory::RestoreArchivedOutcome,
) -> JsRestoreArchivedOutcome {
    JsRestoreArchivedOutcome {
        restored_fact_id: o.restored_fact_id,
        already_live: o.already_live,
    }
}

/// Outcome of `JsMemory.unsupersede` — mirrors `kremory::UnsupersedeOutcome`.
///
/// `outcome` is `"cleared"` (a bound was cleared, fact re-opened as
/// currently-true) or `"not_superseded"` (no bound was set — honest no-op). The
/// `clearedValidTo` / `clearedExpiredAt` booleans record WHICH bound(s) were
/// cleared (both `false` for `"not_superseded"`).
#[napi(object, js_name = "UnsupersedeOutcome")]
pub struct JsUnsupersedeOutcome {
    /// `"cleared"` | `"not_superseded"`.
    pub outcome: String,
    /// The fact id this un-supersede targeted.
    pub fact_id: i64,
    /// `true` if the world-time `valid_to` bound was cleared.
    pub cleared_valid_to: bool,
    /// `true` if the system-time `expired_at` bound was cleared.
    pub cleared_expired_at: bool,
}

/// Convert a substrate `kremory::UnsupersedeOutcome` to `JsUnsupersedeOutcome`.
pub fn unsupersede_outcome_to_js(o: kremory::UnsupersedeOutcome) -> JsUnsupersedeOutcome {
    match o {
        kremory::UnsupersedeOutcome::Cleared {
            fact_id,
            cleared_valid_to,
            cleared_expired_at,
        } => JsUnsupersedeOutcome {
            outcome: "cleared".to_string(),
            fact_id,
            cleared_valid_to,
            cleared_expired_at,
        },
        kremory::UnsupersedeOutcome::NotSuperseded { fact_id } => JsUnsupersedeOutcome {
            outcome: "not_superseded".to_string(),
            fact_id,
            cleared_valid_to: false,
            cleared_expired_at: false,
        },
        // Non-exhaustive guard: unknown future variant maps to an honest marker
        // with fact_id = -1 (no real fact id is negative).
        _ => JsUnsupersedeOutcome {
            outcome: "unknown".to_string(),
            fact_id: -1,
            cleared_valid_to: false,
            cleared_expired_at: false,
        },
    }
}

/// Options for `JsMemory.editEntity` — set exactly one of `newId` (rename/rekey)
/// or `typeId` (retype). `namespace` scopes the entity (else the handle default).
#[napi(object, js_name = "EditEntityOptions")]
pub struct JsEditEntityOptions {
    /// Rename/REKEY target id. Mutually exclusive with `typeId`.
    pub new_id: Option<String>,
    /// Retype target `entity_type_id`. Mutually exclusive with `newId`.
    pub type_id: Option<i64>,
    /// Namespace scope. Omit → the Memory handle's `defaultNamespace`.
    pub namespace: Option<String>,
}

/// Outcome of `JsMemory.editEntity` / `JsMemory.undoEntityEdit` — mirrors
/// `kremory::EditEntityOutcome`. Every count is the ACTUAL rows affected
/// (success-signal honesty, §3.1). Counts as JS `number` (f64): usize values far
/// below 2^53 at practical scale.
#[napi(object, js_name = "EditEntityOutcome")]
pub struct JsEditEntityOutcome {
    /// The entity id AFTER the edit (new id for a rename; unchanged for a retype;
    /// the restored prior id for an undo).
    pub entity_id: String,
    /// `true` if this was a rename/rekey (id changed, FKs re-pointed).
    pub rekeyed: bool,
    /// `true` if the entity's type was changed (retype).
    pub retyped: bool,
    /// `facts` rows re-pointed (subject + object). Zero for a retype.
    pub facts_repointed: f64,
    /// `facts_archive` rows re-pointed. Zero for a retype.
    pub archived_repointed: f64,
    /// `episodic_edges` rows re-pointed. Zero for a retype.
    pub edges_repointed: f64,
    /// `entity_communities` rows re-pointed (rename) or invalidated (retype).
    pub communities_repointed: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoEntityEdit` to reverse.
    pub mutation_id: i64,
    /// `true` if `undoEntityEdit` found the mutation already reversed — a zero-count
    /// idempotent no-op. Always `false` for a forward `editEntity`.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::EditEntityOutcome` to `JsEditEntityOutcome`.
pub fn edit_entity_outcome_to_js(o: kremory::EditEntityOutcome) -> JsEditEntityOutcome {
    JsEditEntityOutcome {
        entity_id: o.entity_id,
        rekeyed: o.rekeyed,
        retyped: o.retyped,
        facts_repointed: o.facts_repointed as f64,
        archived_repointed: o.archived_repointed as f64,
        edges_repointed: o.edges_repointed as f64,
        communities_repointed: o.communities_repointed as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.deleteEntity` / `JsMemory.undoDeleteEntity` — mirrors
/// `kremory::DeleteEntityOutcome` (ADR-073 Tier-2b, §4.4). Every count is the ACTUAL
/// rows affected (success-signal honesty). On a forward delete the counts are what was
/// retracted/removed; on an undo they are the inverse (rows restored). Counts as JS
/// `number` (f64): usize values far below 2^53 at practical scale.
#[napi(object, js_name = "DeleteEntityOutcome")]
pub struct JsDeleteEntityOutcome {
    /// The deleted (or, on undo, restored) entity id.
    pub entity_id: String,
    /// Facts archived (forward) or restored from archive (undo) — never hard-deleted.
    pub facts_retracted: f64,
    /// Episodic edges removed (forward) or re-inserted (undo).
    pub edges_removed: f64,
    /// The entity's OWN community memberships removed (forward) or restored (undo).
    pub communities_removed: f64,
    /// Neighbours whose community membership the retract-on-zero cascade retracted
    /// (forward) or restored (undo).
    pub neighbors_retracted: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoDeleteEntity` to reverse.
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::DeleteEntityOutcome` to `JsDeleteEntityOutcome`.
pub fn delete_entity_outcome_to_js(o: kremory::DeleteEntityOutcome) -> JsDeleteEntityOutcome {
    JsDeleteEntityOutcome {
        entity_id: o.entity_id,
        facts_retracted: o.facts_retracted as f64,
        edges_removed: o.edges_removed as f64,
        communities_removed: o.communities_removed as f64,
        neighbors_retracted: o.neighbors_retracted as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.deleteFact` / `JsMemory.undoDeleteFact` — mirrors
/// `kremory::DeleteFactOutcome` (ADR-073 Tier-2b, §4.5).
#[napi(object, js_name = "DeleteFactOutcome")]
pub struct JsDeleteFactOutcome {
    /// The deleted (or, on undo, restored) fact id.
    pub fact_id: i64,
    /// `true` only when an undo actually moved the fact back out of the archive.
    /// `false` on a forward `deleteFact` (archives, restores nothing) and on an undo
    /// that found the fact already live (an honest no-op restore). Distinct from
    /// `already_undone`, which flags the whole mutation as previously reversed.
    pub fact_restored: bool,
    /// Neighbours (endpoint entities) whose community membership the cascade retracted
    /// (forward) or restored (undo).
    pub neighbors_retracted: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoDeleteFact` to reverse.
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::DeleteFactOutcome` to `JsDeleteFactOutcome`.
pub fn delete_fact_outcome_to_js(o: kremory::DeleteFactOutcome) -> JsDeleteFactOutcome {
    JsDeleteFactOutcome {
        fact_id: o.fact_id,
        fact_restored: o.fact_restored,
        neighbors_retracted: o.neighbors_retracted as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.undo` — the unified undo dispatcher (ADR-073 DX R1/R2).
/// Mirrors `kremory::UndoOutcome` as a FLAT JS object carrying the dispatched
/// `kind` PLUS exactly one populated per-kind outcome field (the other three are
/// omitted). A JS consumer switches on `kind`, then reads the matching field:
///
/// ```js
/// const r = await mem.undo(mutationId);
/// switch (r.kind) {
///   case "unmerge":       console.log(r.unmerge.restoredEntity); break;
///   case "edit_entity":   console.log(r.editEntity.entityId);    break;
///   case "delete_entity": console.log(r.deleteEntity.entityId);  break;
///   case "delete_fact":   console.log(r.deleteFact.factId);      break;
/// }
/// ```
#[napi(object, js_name = "UndoOutcome")]
pub struct JsUndoOutcome {
    /// Which per-kind undo ran: `"unmerge"` | `"edit_entity"` | `"delete_entity"`
    /// | `"delete_fact"` (mirrors the `kremory::UndoOutcome` variant).
    pub kind: String,
    /// Populated iff `kind === "unmerge"`.
    pub unmerge: Option<JsUnmergeOutcome>,
    /// Populated iff `kind === "edit_entity"`.
    pub edit_entity: Option<JsEditEntityOutcome>,
    /// Populated iff `kind === "delete_entity"`.
    pub delete_entity: Option<JsDeleteEntityOutcome>,
    /// Populated iff `kind === "delete_fact"`.
    pub delete_fact: Option<JsDeleteFactOutcome>,
}

/// Convert a substrate `kremory::UndoOutcome` to `JsUndoOutcome` (flat object with
/// a `kind` discriminator + the one populated per-kind outcome field).
pub fn undo_outcome_to_js(o: kremory::UndoOutcome) -> JsUndoOutcome {
    let mut js = JsUndoOutcome {
        kind: String::new(),
        unmerge: None,
        edit_entity: None,
        delete_entity: None,
        delete_fact: None,
    };
    match o {
        kremory::UndoOutcome::Unmerge(u) => {
            js.kind = "unmerge".to_string();
            js.unmerge = Some(unmerge_outcome_to_js(u));
        }
        kremory::UndoOutcome::EditEntity(e) => {
            js.kind = "edit_entity".to_string();
            js.edit_entity = Some(edit_entity_outcome_to_js(e));
        }
        kremory::UndoOutcome::DeleteEntity(d) => {
            js.kind = "delete_entity".to_string();
            js.delete_entity = Some(delete_entity_outcome_to_js(d));
        }
        kremory::UndoOutcome::DeleteFact(d) => {
            js.kind = "delete_fact".to_string();
            js.delete_fact = Some(delete_fact_outcome_to_js(d));
        }
        // Non-exhaustive guard: a future log-dispatchable kind maps to an honest
        // "unknown" marker with every outcome field omitted (never a mis-label).
        _ => {
            js.kind = "unknown".to_string();
        }
    }
    js
}

/// A single logged graph mutation from `JsMemory.mutationHistory` /
/// `JsMemory.listMutations` — mirrors `kremory::MutationRecord` (the consumer
/// INSPECT view, §3). Carries the `mutationId` to pass to `unmerge` (or the
/// matching undo method for the kind).
#[napi(object, js_name = "MutationRecord")]
pub struct JsMutationRecord {
    /// The `graph_mutation_log.id` — pass to `unmerge` to reverse the mutation.
    pub mutation_id: i64,
    /// The mutation kind tag, e.g. `"entity_merge"` | `"fact_supersede"` |
    /// `"fact_archive"` | `"entity_edit"` | … (mirrors `kremory::MutationKind`).
    pub kind: String,
    /// RFC3339 timestamp when the mutation was applied.
    pub created_at: String,
    /// `true` once the mutation has been reversed.
    pub undone: bool,
    /// The namespace (group) this mutation scoped.
    pub group_id: String,
    /// Entity ids this mutation touched. For `entity_merge` this is `[keeper, loser]`.
    pub affected_entities: Vec<String>,
    /// A short human/agent-readable summary of what the mutation did.
    pub summary: String,
}

/// Render a `kremory::MutationKind` to its snake_case tag string. Uses serde (the
/// enum's `rename_all = "snake_case"` derive) so the tag matches the substrate's
/// `graph_mutation_log.kind` values 1:1 without reaching for the `pub(crate)`
/// `as_tag` helper. A serialize failure (unreachable for a unit enum) surfaces as
/// `"unknown"` rather than a panic.
fn mutation_kind_to_str(k: kremory::MutationKind) -> String {
    serde_json::to_value(k)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Convert a substrate `kremory::MutationRecord` to `JsMutationRecord`.
pub fn mutation_record_to_js(r: kremory::MutationRecord) -> JsMutationRecord {
    JsMutationRecord {
        mutation_id: r.mutation_id,
        kind: mutation_kind_to_str(r.kind),
        created_at: r.created_at,
        undone: r.undone,
        group_id: r.group_id,
        affected_entities: r.affected_entities,
        summary: r.summary,
    }
}

/// Filter for `JsMemory.listMutations` — mirrors `kremory::MutationFilter`. All
/// fields optional.
#[napi(object, js_name = "MutationFilter")]
pub struct JsMutationFilter {
    /// Restrict to one namespace. Omit → the Memory handle's default, else ALL
    /// namespaces.
    pub namespace: Option<String>,
    /// Restrict to one mutation kind tag (e.g. `"entity_merge"`). An unknown tag
    /// rejects the Promise. Omit → every kind.
    pub kind: Option<String>,
    /// Only mutations at/after this RFC3339 timestamp (`created_at >=`). Omit → no
    /// lower bound. A malformed timestamp rejects the Promise.
    pub since: Option<String>,
    /// Include already-undone mutations. Default `false` (live/still-reversible only).
    pub include_undone: Option<bool>,
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use kremory::RetrievedContext;

    use super::retrieved_context_to_js;

    // Helper: build a minimal RetrievedContext via ::new() then patch
    // entity_type_id / entity_type_name directly (within-crate access allowed).
    fn make_ctx(entity_type_id: u32, entity_type_name: &str) -> RetrievedContext {
        let mut ctx = RetrievedContext::new(kremory::RetrievedContextNewParams {
            entity_id: "ent-1".to_string(),
            entity_name: "Alice".to_string(),
            summary: "summary text".to_string(),
            score: 0.9_f32,
            source_refs: vec![],
        });
        ctx.entity_type_id = entity_type_id;
        ctx.entity_type_name = entity_type_name.to_string();
        ctx
    }

    /// TD-118: source_refs project the full `kremory::SourceRef` shape
    /// (kind/id/occurred_at/published_at), not a flattened bare-id string.
    /// Drives the real producer (`retrieved_context_to_js`) and asserts each
    /// provenance field survives the projection — mirrors the MCP wire.
    #[test]
    fn source_refs_project_full_shape() {
        use chrono::{TimeZone, Utc};

        let occurred = Utc.with_ymd_and_hms(2026, 7, 15, 9, 0, 0).unwrap();
        let published = Utc.with_ymd_and_hms(2026, 7, 14, 8, 30, 0).unwrap();
        let ctx = RetrievedContext::new(kremory::RetrievedContextNewParams {
            entity_id: "ent-1".to_string(),
            entity_name: "Alice".to_string(),
            summary: "s".to_string(),
            score: 0.5_f32,
            source_refs: vec![
                kremory::SourceRef {
                    kind: kremory::SourceKind::Document,
                    id: "doc-42".to_string(),
                    occurred_at: occurred,
                    published_at: Some(published),
                },
                kremory::SourceRef {
                    kind: kremory::SourceKind::Episode,
                    id: "ep-7".to_string(),
                    occurred_at: occurred,
                    published_at: None,
                },
            ],
        });

        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.source_refs.len(), 2, "both source_refs projected");

        let doc = &js.source_refs[0];
        assert_eq!(doc.kind, "document", "kind lower-cased from SourceKind");
        assert_eq!(doc.id, "doc-42");
        assert_eq!(doc.occurred_at, occurred.to_rfc3339());
        assert_eq!(
            doc.published_at.as_deref(),
            Some(published.to_rfc3339()).as_deref(),
            "published_at preserved when set"
        );

        let ep = &js.source_refs[1];
        assert_eq!(ep.kind, "episode");
        assert_eq!(ep.id, "ep-7");
        assert!(
            ep.published_at.is_none(),
            "published_at None survives as null"
        );
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

    // ── ADR-074 review H2: facts must survive the napi wire layer ──────────
    //
    // `make_ctx` above (and `RetrievedContext::new()` generally) always
    // defaults `facts: Vec::new()` — `RetrievedContext` is `#[non_exhaustive]`
    // and `RetrievedContextNewParams` carries no `facts` field, so a non-empty
    // fixture cannot be struct-literalled. This drives the REAL recall path
    // (mode-c pinned fact via `Memory::remember().with_facts().skip_extraction()`
    // — the same mechanism `kremory`'s own
    // `with_facts_integration.rs::td116_recall_returns_connected_facts_under_null_embedder`
    // proves at the facade level) so `retrieved_context_to_js` is exercised
    // against a genuine, non-empty `RetrievedContext.facts`.

    use std::sync::Arc;

    use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
    use autoagents_llm::error::LLMError;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::memory::types::StructuredFact;
    use kremory::{ChatProvider, Memory, Namespace};

    use super::retrieved_fact_to_js;

    /// `Memory::open(...).with_llm(...)` requires a real `Arc<dyn ChatProvider>`
    /// even on the pinned-fact `skip_extraction()` path, which never invokes
    /// it. Errors loudly (not a silent empty response) if that assumption
    /// ever breaks, so a future regression fails this test with a clear cause
    /// instead of a confusing downstream symptom.
    #[derive(Debug, Clone)]
    struct UnreachableChatProvider;

    #[async_trait::async_trait]
    impl ChatProvider for UnreachableChatProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::Generic(
                "UnreachableChatProvider: chat_with_tools must not be called on a \
                 skip_extraction() pinned-fact test path"
                    .to_string(),
            ))
        }
    }

    /// Build an in-memory `Memory` (no live LLM/embedder needed — mirrors
    /// `with_facts_integration.rs::open_with_ns`, kremory-napi's own crate
    /// only having `NullEmbeddingProvider` available outside `test-utils`).
    #[allow(clippy::expect_used)]
    async fn napi_test_memory() -> Memory {
        let llm: Arc<dyn ChatProvider> = Arc::new(UnreachableChatProvider);
        let embedder: Arc<dyn kremory::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .default_namespace(Namespace::new("kremory_napi_h2_tests"))
            .await
            .expect("in-memory Memory must build")
    }

    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    async fn retrieved_context_to_js_round_trips_a_nonempty_pinned_fact() {
        let mem = napi_test_memory().await;

        mem.remember("Ada Lovelace wrote the first algorithm.")
            .with_facts(vec![StructuredFact {
                subject: "Ada Lovelace".to_string(),
                predicate: "wrote".to_string(),
                object: "the first algorithm".to_string(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .from_document("napi-h2-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        let raw = mem
            .recall("Ada Lovelace")
            .raw()
            .await
            .expect("raw recall should succeed");
        let ada = raw
            .into_iter()
            .find(|r| r.entity_name == "Ada Lovelace")
            .expect("Ada Lovelace must be in recall results");
        assert!(
            !ada.facts.is_empty(),
            "H2: recall must surface Ada Lovelace's connected fact before conversion"
        );

        let js = retrieved_context_to_js(ada);
        assert!(
            !js.facts.is_empty(),
            "H2: retrieved_context_to_js must not drop facts crossing the napi wire"
        );

        let fact = js
            .facts
            .iter()
            .find(|f| f.predicate == "wrote")
            .expect("the pinned 'wrote' fact must survive the wire mapping");
        assert_eq!(fact.fact, "Ada Lovelace wrote the first algorithm");
        assert_eq!(fact.subject, "Ada Lovelace");
        assert_eq!(fact.predicate, "wrote");
        assert_eq!(fact.object, "the first algorithm");
        assert!(
            !fact.object_is_entity,
            "literal object → object_is_entity=false"
        );
        assert!(
            !fact.valid_at.is_empty(),
            "valid_at must be a non-empty RFC-3339 string"
        );
        assert!(
            fact.invalid_at.is_none(),
            "an open-ended pinned fact must have invalid_at=None"
        );
        assert!(
            !fact.recorded_at.is_empty(),
            "recorded_at must be a non-empty RFC-3339 string"
        );
        assert!(
            fact.expired_at.is_none(),
            "a fresh pinned fact must have expired_at=None"
        );
        assert_eq!(
            fact.confidence, 1.0,
            "caller-pinned facts default to confidence=1.0"
        );
        assert!(
            !fact.source_episode_ids.is_empty(),
            "source_episode_ids must attribute the fact to its episode"
        );
    }

    /// `retrieved_fact_to_js` (the per-fact half of the conversion) round-trips
    /// every field independently of the containing `RetrievedContext` — the
    /// same real, non-empty fixture as the test above, but asserting the
    /// narrower conversion function directly.
    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    async fn retrieved_fact_to_js_round_trips_every_field() {
        let mem = napi_test_memory().await;

        mem.remember("Grace Hopper invented the compiler.")
            .with_facts(vec![StructuredFact {
                subject: "Grace Hopper".to_string(),
                predicate: "invented".to_string(),
                object: "the compiler".to_string(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .from_document("napi-h2-fact-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        let raw = mem
            .recall("Grace Hopper")
            .raw()
            .await
            .expect("raw recall should succeed");
        let hopper = raw
            .into_iter()
            .find(|r| r.entity_name == "Grace Hopper")
            .expect("Grace Hopper must be in recall results");
        let fact = hopper
            .facts
            .into_iter()
            .find(|f| f.predicate == "invented")
            .expect("the pinned 'invented' fact must be present");

        let js = retrieved_fact_to_js(fact);
        assert_eq!(js.fact, "Grace Hopper invented the compiler");
        assert_eq!(js.subject, "Grace Hopper");
        assert_eq!(js.predicate, "invented");
        assert_eq!(js.object, "the compiler");
        assert!(!js.object_is_entity);
        assert!(!js.valid_at.is_empty());
        assert!(js.invalid_at.is_none());
        assert!(!js.recorded_at.is_empty());
        assert!(js.expired_at.is_none());
        assert_eq!(js.confidence, 1.0);
        assert!(!js.source_episode_ids.is_empty());
        assert!(js.score >= 0.0);
    }

    /// **DUR-3 sibling guard (V1-CANONICAL §4.2).** Every known `IngestStatus`
    /// variant must map to its OWN discriminator — never silently into the
    /// `_ =>` forward-compat arm, which returns `"pending"`.
    ///
    /// This is the defect this test exists for: three variants
    /// (`EntitiesReady`, `SkippedIdempotent`, `ExtractionSkipped`) were falling
    /// through that arm, and two of them are **terminal**. A Node consumer
    /// polling `statusOf` therefore saw `"pending"` on an episode that had
    /// already finished — the same "poll forever on a completed ingest" shape as
    /// DUR-3 itself, but on the wire rather than in the column. `EntitiesReady`
    /// is SQL `'Verified'`, the exact state `Memory::wait_for_processing`
    /// resolves `Ok(())` on, so Rust callers saw success while JS callers saw
    /// pending.
    ///
    /// # Honest limits — read before trusting this
    ///
    /// `IngestStatus` is `#[non_exhaustive]`, so the catch-all arm **cannot** be
    /// removed and this test **cannot** mechanically discover a variant nobody
    /// listed here. It is a hand-maintained roster, not structural enforcement,
    /// and it will not fail on its own the day someone adds a tenth variant.
    /// What it does buy: the moment anyone *does* add a variant and comes here,
    /// the roster and the assertion below state the obligation plainly, and any
    /// regression that re-routes an existing variant into the catch-all fails
    /// loudly. Real enforcement would need the enum to stop being
    /// `#[non_exhaustive]`, which is a breaking change and not v1 scope.
    #[test]
    fn ingest_status_maps_every_known_variant_off_the_catch_all() {
        use crate::convert::ingest_status_to_js;
        use kremory::IngestStatus as S;

        // The roster. Update this when adding a variant — see limits above.
        let cases = vec![
            (S::Pending, "pending"),
            (S::Extracting, "extracting"),
            (S::EntitiesReady, "entities_ready"),
            (S::Deduplicating, "deduplicating"),
            (S::Invalidating, "invalidating"),
            (S::Complete, "complete"),
            (S::Failed("boom".to_string()), "failed"),
            (S::SkippedIdempotent, "skipped_idempotent"),
            (S::ExtractionSkipped, "skipped"),
        ];

        for (variant, expected) in cases {
            let label = format!("{variant:?}");
            let js = ingest_status_to_js(variant);
            assert_eq!(
                js.status, expected,
                "{label} must map to {expected:?}, not {:?}. A value of \
                 \"pending\" here almost always means the variant fell through \
                 the `_ =>` forward-compat arm — which is a silent bug for any \
                 TERMINAL state, because JS consumers poll on \"pending\".",
                js.status
            );
        }

        // Sensitivity in the other direction: the catch-all must still be
        // reachable and must still say "pending", so the assertion above is
        // actually discriminating rather than passing vacuously.
        assert_eq!(
            ingest_status_to_js(S::Pending).status,
            "pending",
            "the catch-all's own value must remain \"pending\" — if this ever \
             changes, the checks above stop distinguishing a real mapping from \
             a fall-through"
        );
    }
}

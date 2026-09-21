//! Core open/ingest napi conversions — `Memory.open`, `.remember`, episode/batch/status types.
//! Split out of `convert.rs` (TD-243); see `convert/mod.rs` for the domain map.

use napi_derive::napi;

/// GLiNER configuration passed via `MemoryOpenOptionsJs.gliner`.
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
/// for when `GlinerConfig` gains public tuning knobs; they will
/// become live then.
///
/// Requires kremory-napi built with `--features ner`. Passing this field on a
/// non-ner build causes `Memory.open` to return an error.
#[napi(object, js_name = "GlinerConfig")]
#[derive(Default)]
pub struct GlinerConfigJs {
    /// RESERVED / inert — model file path override. Not threaded to the substrate
    /// today (`with_gliner()` takes no arg); reserved for a future
    /// tuning surface. `null` = use default bundled model.
    pub model_path: Option<String>,
    /// RESERVED / inert — span-detection confidence threshold in `[0.0, 1.0]`. Not
    /// threaded to the substrate today; set `KREMORY_GLINER_THRESHOLD` env instead
    /// (substrate default `0.3`). Reserved for a future tuning surface.
    pub threshold: Option<f64>,
}

// `From<GlinerConfigJs> for GlinerConfig` removed — `with_gliner()` no longer
// takes a config arg, so the conversion had no caller (dead code). GlinerConfigJs
// is retained as the JS-side enable-signal (presence of `opts.gliner`). Restore a
// mapping here if/when GlinerConfig gains public tuning knobs.

/// Options for `Memory.open` — live napi/cdylib build.
///
/// Replaces the old string-literal `extractor: 'auto'|'hybrid'|'nuextract'` API.
/// Mirrors the Rust `MemoryBuilder` composable knobs:
///   - `{ embedder, llm }`          → `ExtractorKind::IntegerId` (the default;
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
    /// Optional BYOM embedder callback (Tier-2).
    ///
    /// Callback signature: `(err: null, text: string) => Promise<number[]>`.
    ///
    /// ⚠️ **TWO arguments, error-first** — the same napi-rs `CalleeHandled`
    /// convention as `extractor.extract`. `err` is always `null` and **the text
    /// is the SECOND argument**.
    ///
    /// This was declared as a one-argument `(text: string) => …` until
    /// 2026-09-14, and that declaration was wrong in the worst possible way: a
    /// one-argument callback binds its parameter to `null`, most implementations
    /// guard that to `""`, and the result is an ALL-ZERO vector of the correct
    /// length. It passes the dimension check, stores without error, and has zero
    /// cosine similarity with every query — so every entity embeds to nothing and
    /// becomes permanently unrecallable, silently. Every embedder in this repo's
    /// own examples and tests had the one-argument shape and was producing zero
    /// vectors. `validate_embedding` now rejects that outcome rather than
    /// trusting this doc comment to prevent it.
    ///
    /// When set, kremory wires this JS function as the embedding provider via
    /// the `MemoryBuilder` Tier-2 path.
    #[napi(ts_type = "((err: null, text: string) => Promise<number[]>) | undefined | null")]
    pub with_embedder: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            String,
            napi::threadsafe_function::ErrorStrategy::CalleeHandled,
        >,
    >,
    /// GLiNER configuration. When set, selects
    /// `ExtractorKind::GlinerLlm` and requires `llm` to also be set.
    /// Requires kremory-napi built with `--features ner`.
    pub gliner: Option<GlinerConfigJs>,
    /// External (BYOE) extractor bridge.
    ///
    /// When set, selects `ExtractorKind::Custom`. Mutually exclusive with
    /// `gliner` — setting both returns an error.
    ///
    /// JS shape: `{ name: string, extract: (err: null, text: string) => Promise<ExtractionResult> }`.
    /// `name` is a plain string property, NOT a method — and `extract` is
    /// called with the raw napi-rs error-first convention (`err` always
    /// `null`, real text is the second argument). See `ExternalExtractorJs`
    /// / `ExternalExtractorHandle` in bridge.rs for the full interface and
    /// why.
    #[napi(
        ts_type = "{ name: string, extract: (err: null, text: string) => Promise<{ entities: Array<{ name: string, label: string }>, facts: Array<{ subject: string, predicate: string, object: string }> }> } | undefined | null"
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
/// ingest time, BEFORE the Phase 2 LLM extractor runs. LLM-extracted
/// duplicates of the same `(subject, predicate, object)` triple are silently
/// swallowed by the substrate's `try_insert_fact` helper — caller wins by
/// virtue of being there first.
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
/// descriptive `napi::Error` on missing/malformed values rather than
/// silently defaulting.
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

/// Options for `JsMemory.remember(opts)` — the single ingest surface
/// (collapses prior `JsMemory.ingest(text, opts)` +
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
    /// Pre-extracted RDF triples. When supplied, caller
    /// facts are pinned BEFORE Phase 2 LLM extraction; LLM-extracted
    /// duplicates of the same triple are silently swallowed.
    pub structured_facts: Option<Vec<JsStructuredFact>>,
    /// When `true`, Phase 2 LLM extraction is skipped entirely for this
    /// episode (the episode row + embedding + pinned facts are still
    /// persisted). Suitable for bulk-import workloads.
    pub skip_extraction: Option<bool>,
}

/// Result of a successful `Memory.remember` call.
#[napi(object, js_name = "IngestResult")]
pub struct JsIngestResult {
    /// Stable entity ID under which the episode is searchable immediately.
    /// This is a graph node UUID, NOT a run ID — see `run_id` below.
    pub episode_entity_id: String,
    /// Background run identifier. Present when the
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

/// Options for `JsMemory.rememberBatch` — bulk episode ingest.
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

/// Result of `JsMemory.statusOf` / `JsMemory.awaitEnrichment`.
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

/// Result of `JsMemory.awaitBatch`.
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

/// Result of `JsMemory.cancel` / `JsMemory.cancelDream`.
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

/// Convert a substrate `kremory::core::schema::Episode` to `JsEpisode`.
///
/// `source_id` and `source_uri` are now taken directly from the
/// `Episode` struct (these columns were always in the DB; the Rust type and
/// its SELECT projections have since been updated to include them). The
/// previous workaround that took `source_id` as a separate `&str` argument
/// and hardcoded `source_uri: None` is removed.
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

/// Convert a substrate `kremory::IngestStatus` to `JsIngestStatusResult`.
///
/// Intermediate-state variants (`Deduplicating`, `Invalidating`) that were
/// added later are mapped to `"pending"` with a note so consumers are not
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
        // ── Terminal status variants must not fall through to "pending" ──────
        // These three arms were all falling through the catch-all below and
        // being reported to JS as "pending". Two of them are TERMINAL, so a Node
        // consumer polling `statusOf` would poll forever on an episode that was
        // already finished.
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


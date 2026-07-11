//! `kremory-napi` — Node.js binding for the kremory `Memory` facade.
//!
//! Exposes `kremory::Memory` to TypeScript/JavaScript via napi-rs derive macros.
//! Per ADR-030 Decision 2 Form B: single binding, no PyO3, no wasm-bindgen.
//!
//! # Binding surface
//!
//! - `JsMemory` wraps `kremory::Memory`. Async methods delegate to a tokio
//!   multi-thread runtime via napi-rs `async` feature.
//! - Plain data structs (`JsOpenOptions`, `JsRecallOptions`, `JsRememberOptions`,
//!   `JsStructuredFact`, `JsRetrievedContext`, `JsIngestResult`) are `#[napi(object)]` — napi-rs
//!   emits TS `interface` declarations for each.
//!
//! # Error mapping
//!
//! All `kremory::MemoryError` values are converted to `napi::Error::from_reason`
//! so they surface as JS `Error` rejections with a descriptive message.

#![deny(clippy::all)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod bridge;
mod convert;

use napi_derive::napi;

use kremory::{Memory, Namespace, SourceKind};

pub use convert::{
    JsBatchOptions, JsBatchStatus, JsCancelOutcome, JsConsolidationOpsRan, JsDeleteEntityOutcome,
    JsDeleteFactOutcome, JsDreamOpts, JsDreamPassOpts, JsDreamStatusResult, JsDreamSummary,
    JsEditEntityOptions, JsEditEntityOutcome, JsEpisode, JsIngestResult, JsIngestStatusResult,
    JsMetadataFilter, JsMutationFilter, JsMutationRecord, JsOpenOptions, JsRecallOptions,
    JsRememberOptions, JsRestoreArchivedOutcome, JsRetrievedContext, JsStructuredFact,
    JsSupersedeOutcome, JsTypeProposal, JsUnmergeOutcome, JsUnsupersedeOutcome,
};

// ── JsMemory ──────────────────────────────────────────────────────────────────

/// Node.js handle for a kremory `Memory` instance.
///
/// Obtain via `JsMemory.open(path, opts?)`.
/// `close()` should be called at shutdown to future-proof against v0.1.1+ WAL
/// flush semantics.
#[napi(js_name = "Memory")]
pub struct JsMemory {
    inner: Memory,
    /// Handle-level default namespace captured from `JsOpenOptions.defaultNamespace`
    /// at `open` time. Applied to ingest/recall calls that don't pass an explicit
    /// per-call namespace. Set-once, never mutated — safe for concurrent reads.
    default_namespace: Option<Namespace>,
}

#[napi]
impl JsMemory {
    /// Open a kremory Memory at `path`.
    ///
    /// ## Tier-1 path (default, backward-compat)
    ///
    /// When no extractor knobs are set in `opts`, uses env-detected providers
    /// (`OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`) via
    /// `Memory::auto`. Behavior is identical to v0.1.6-alpha.0.
    ///
    /// ## Tier-2 path (BYOM embedder, ADR-030)
    ///
    /// When `opts.withEmbedder` is a callback `(text: string) => Promise<number[]>`,
    /// the env-detected LLM is combined with the JS callback as the embedding
    /// provider via `MemoryBuilder::with_embedder`. Set `opts.embeddingDim` to
    /// the callback's output dimension — a mismatch yields a descriptive error.
    ///
    /// ## BYOE extractor knobs (ADR-039 Shape B)
    ///
    /// Composable knobs that mirror the Rust `MemoryBuilder`:
    ///   - `opts.gliner`              → enables `ExtractorKind::GlinerLlm` (requires `--features ner`)
    ///   - `opts.extractor`           → enables `ExtractorKind::Custom` (BYOE)
    ///   - `opts.gliner + extractor`  → `BuilderConflict` error
    ///
    /// ## `opts.defaultNamespace`
    ///
    /// When set, becomes the handle-level default namespace applied to subsequent
    /// ingest/recall calls that omit per-call namespace.
    #[napi(factory)]
    pub async fn open(path: String, opts: Option<JsOpenOptions>) -> napi::Result<JsMemory> {
        // Live napi/cdylib path: ThreadsafeFunction requires the napi runtime.
        #[cfg(not(test))]
        {
            let default_namespace = opts
                .as_ref()
                .and_then(|o| o.default_namespace.as_deref())
                .map(Namespace::new);
            let expected_dim = opts
                .as_ref()
                .and_then(|o| o.embedding_dim)
                .and_then(|d| usize::try_from(d).ok());

            // Detect which extractor knobs are set.
            let has_gliner = opts.as_ref().is_some_and(|o| o.gliner.is_some());
            let has_extractor = opts.as_ref().is_some_and(|o| o.extractor.is_some());

            // Conflict: gliner + extractor simultaneously is a BuilderConflict.
            if has_gliner && has_extractor {
                return Err(napi::Error::from_reason(
                    "KremoryError::BuilderConflict: \
                     opts.gliner and opts.extractor are mutually exclusive — \
                     set gliner (for GlinerLlm) OR extractor (for Custom), not both",
                ));
            }

            // ner-feature guard for GLiNER.
            #[cfg(not(feature = "ner"))]
            if has_gliner {
                return Err(napi::Error::from_reason(
                    "KremoryError::FeatureDisabled('ner'): \
                     opts.gliner requires kremory-napi built with --features ner",
                ));
            }

            // Extractor knobs require a BYOM embedder (ADR-039 §6 compat matrix).
            // All valid rows that include gliner or extractor also include withEmbedder.
            if (has_gliner || has_extractor)
                && opts.as_ref().is_none_or(|o| o.with_embedder.is_none())
            {
                return Err(napi::Error::from_reason(
                    "KremoryError::BuilderConflict: \
                     opts.gliner / opts.extractor require opts.withEmbedder — \
                     provide a BYOM embedder callback alongside the extractor knob",
                ));
            }

            // Unpack opts, consuming it.
            let (with_embedder_tsfn, gliner_cfg, extractor_handle) = match opts {
                Some(o) => (o.with_embedder, o.gliner, o.extractor),
                None => (None, None, None),
            };

            // Tier-2 BYOM embedder path: wire JS embedder + LLM via MemoryBuilder.
            if let Some(tsfn) = with_embedder_tsfn {
                return open_with_js_embedder(
                    path,
                    tsfn,
                    expected_dim,
                    default_namespace,
                    gliner_cfg,
                    extractor_handle,
                )
                .await;
            }

            // Plain Tier-1: no knobs set (already validated above).
            let mem = Memory::auto(&path)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;
            Ok(JsMemory {
                inner: mem,
                default_namespace,
            })
        }

        // Test path: napi runtime absent — use Memory::auto only.
        // Extractor knobs are tested via MockExtractorBridge directly in
        // extractor_selection_compat_matrix.rs.
        #[cfg(test)]
        {
            let default_namespace = opts
                .as_ref()
                .and_then(|o| o.default_namespace.as_deref())
                .map(Namespace::new);

            let mem = Memory::auto(&path)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;

            Ok(JsMemory {
                inner: mem,
                default_namespace,
            })
        }
    }

    /// Ingest an episode with full provenance + optional pre-extracted facts.
    ///
    /// ADR-034 + ADR-035 v0.1.8 unified ingest surface. Collapses prior
    /// `JsMemory.ingest(text, opts)` + `JsMemory.ingest_episode(draft)`
    /// into a single options-object method that mirrors the substrate's
    /// fluent `mem.remember(content).with_facts(...).skip_extraction()....`
    /// builder chain.
    ///
    /// # Namespace resolution
    /// `opts.namespace` > handle-level default > rejection.
    ///
    /// # Caller-pinned facts (ADR-035)
    /// When `opts.structuredFacts` is supplied, those triples are pinned
    /// into the graph BEFORE Phase 2 LLM extraction runs. LLM-extracted
    /// duplicates are silently swallowed; caller wins via pre-write ordering.
    ///
    /// # Skip Phase 2 extraction
    /// When `opts.skipExtraction == true`, Phase 2 LLM is not invoked for
    /// this episode (episode + embedding + pinned facts are still persisted).
    #[napi]
    pub async fn remember(&self, opts: JsRememberOptions) -> napi::Result<JsIngestResult> {
        // Translate caller's JsStructuredFact (binding layer) → substrate
        // StructuredFact (memory layer). Loud-parse on invalid timestamps
        // per ADR-035 §3.
        let facts: Vec<kremory::memory::types::StructuredFact> = match opts.structured_facts {
            Some(items) => items
                .into_iter()
                .map(kremory::memory::types::StructuredFact::try_from)
                .collect::<Result<_, _>>()?,
            None => Vec::new(),
        };

        let namespace = opts
            .namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        // Build the substrate RememberRequest. Mirror existing ingest_episode
        // semantics: from_source(Document) when source_id is supplied; bare
        // remember(content) otherwise (substrate generates a UUID).
        let mut req = if let Some(ref sid) = opts.source_id {
            self.inner
                .remember(opts.content.clone())
                .from_source(sid.clone(), SourceKind::Document)
        } else {
            self.inner.remember(opts.content.clone())
        };

        if let Some(ns) = namespace {
            req = req.in_namespace(ns);
        }
        if !facts.is_empty() {
            req = req.with_facts(facts);
        }
        if opts.skip_extraction.unwrap_or(false) {
            req = req.skip_extraction();
        }

        // reference_time: parse and propagate via published_at builder when
        // supplied. Loud parse — caller's malformed timestamp surfaces.
        if let Some(ref ts_str) = opts.reference_time {
            let ts = chrono::DateTime::parse_from_rfc3339(ts_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "JsRememberOptions.reference_time: invalid RFC-3339 timestamp {:?}: {}",
                        ts_str, e
                    ))
                })?;
            req = req.published_at(ts);
        }

        let commit = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory remember failed: {e}")))?;

        let mut warnings: Vec<String> = Vec::new();

        // Post-ingest: write source_uri if both source_id + source_uri supplied.
        if let Some(ref uri) = opts.source_uri {
            if let Some(ref sid) = opts.source_id {
                if let Err(e) = self
                    .inner
                    .update_source_uri(sid.clone())
                    .to(uri.clone())
                    .await
                {
                    warnings.push(format!("source_uri update failed: {e}"));
                }
            } else {
                warnings.push("source_uri supplied without source_id — uri not stored".to_string());
            }
        }

        // Post-ingest: write metadata if both source_id + metadata supplied.
        if let Some(meta) = opts.metadata {
            if let Some(ref sid) = opts.source_id {
                if let Err(e) = self
                    .inner
                    .update_episode_metadata(sid.clone())
                    .patch(meta)
                    .await
                {
                    warnings.push(format!("metadata update failed: {e}"));
                }
            } else {
                warnings
                    .push("metadata supplied without source_id — metadata not stored".to_string());
            }
        }

        Ok(JsIngestResult {
            episode_entity_id: commit.episode_entity_id,
            run_id: commit.run_id.map(|u| u.to_string()),
            committed_at: commit.committed_at.to_rfc3339(),
            warnings,
        })
    }

    /// Update an episode's metadata by source_id (B2). v0.1.8 1:1 contract per ADR-034.
    ///
    /// Performs a shallow merge: existing metadata keys are preserved; the
    /// `patch` keys overwrite. Wraps substrate `Memory::update_episode_metadata`.
    /// Returns the count of episode rows updated.
    #[napi]
    pub async fn update_episode_metadata(
        &self,
        source_id: String,
        patch: serde_json::Value,
    ) -> napi::Result<f64> {
        let updated = self
            .inner
            .update_episode_metadata(source_id)
            .patch(patch)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory updateEpisodeMetadata failed: {e}"))
            })?;

        // usize → f64: safe up to 2^53; episode counts never approach that limit.
        Ok(updated as f64)
    }

    /// Update an episode's source URI by source_id (B3). v0.1.8 1:1 contract per ADR-034.
    ///
    /// Wraps substrate `Memory::update_source_uri`. Rejects if no episode matches
    /// `source_id`. Returns the count of rows updated.
    #[napi]
    pub async fn update_source_uri(&self, source_id: String, new_uri: String) -> napi::Result<f64> {
        let updated = self
            .inner
            .update_source_uri(source_id)
            .to(new_uri)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory updateSourceUri failed: {e}"))
            })?;

        // u64 → f64: safe up to 2^53; episode update counts never approach that limit.
        Ok(updated as f64)
    }

    /// Direct slug/source_id lookup — returns episodes matching `source_id` (B5).
    /// v0.1.8 1:1 contract per ADR-034 (renamed from `get_by_source_id`).
    ///
    /// Results are ordered newest-first. When `namespace` is omitted, the
    /// Memory handle's default namespace is used; when the handle has no default,
    /// results span all namespaces.
    ///
    /// # Known gap
    /// The `sourceUri` field on each returned `JsEpisode` is always `null` —
    /// the substrate `recall_by_source_id` query does not select that column.
    /// Use `updateSourceUri` to write and the value is persisted in the DB.
    #[napi]
    pub async fn recall_by_source_id(
        &self,
        source_id: String,
        namespace: Option<String>,
    ) -> napi::Result<Vec<JsEpisode>> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let episodes = self
            .inner
            .recall_by_source_id(source_id.clone(), ns)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory recallBySourceId failed: {e}"))
            })?;

        Ok(episodes.into_iter().map(convert::episode_to_js).collect())
    }

    /// Trigger the dream-phase batch consolidation (B6, consumer-API hardening D2).
    ///
    /// Blocks until the dream completes. Returns a `JsDreamSummary` with per-phase
    /// accounting (incl. the D5 `crossEpisodeWouldMerge`/`crossEpisodeMerged` split
    /// and the D1b `consolidationOpsRan` ran-signal). Wraps `Memory::dream()`,
    /// threading the full `DreamOptions` consolidation-control surface — every
    /// consolidation knob (community / archival / supersession / cross-episode mode /
    /// budgets / grace / warn-floor) is now reachable from JS. An unknown
    /// `opts.crossEpisodeMode` rejects the Promise (loud parse).
    #[napi]
    pub async fn dream(&self, opts: Option<JsDreamOpts>) -> napi::Result<JsDreamSummary> {
        // Resolve namespace BEFORE opts is moved into the conversion (namespace is
        // NOT a DreamOpts field — it routes via `.in_namespace(ns)`).
        let ns = opts
            .as_ref()
            .and_then(|o| o.namespace.as_deref())
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let rust_opts = convert::js_dream_opts_to_rust(opts)?;

        let mut req = self.inner.dream().opts(rust_opts);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }

        let summary = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory dream failed: {e}")))?;

        Ok(convert::dream_summary_to_js(summary))
    }

    /// Run a single dream pass synchronously (Phase C DoD C7 / `v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    ///
    /// Lower-level than `dream()` — calls the engine's `run_dream_pass_sync`
    /// directly with explicit pass options. Concurrent calls serialize via an
    /// internal `Mutex` on `Engine`. Wraps `Memory::run_dream_pass_sync`.
    #[napi]
    pub async fn run_dream_pass_sync(
        &self,
        opts: Option<JsDreamPassOpts>,
    ) -> napi::Result<JsDreamSummary> {
        let rust_opts = convert::js_dream_pass_opts_to_rust(opts);
        let summary = self
            .inner
            .run_dream_pass_sync(rust_opts)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory run_dream_pass_sync failed: {e}"))
            })?;
        Ok(convert::dream_summary_to_js(summary))
    }

    /// Return episode IDs where Phase 1 ingest succeeded but Phase 2 produced
    /// no facts (ghost episodes — Phase C DoD C4 / C7).
    ///
    /// `group_id` restricts the query to one namespace. Omit/`null` to return
    /// ghost episodes across all namespaces. Wraps `Memory::ghost_episodes`.
    #[napi]
    pub async fn ghost_episodes(&self, group_id: Option<String>) -> napi::Result<Vec<i64>> {
        self.inner
            .ghost_episodes(group_id.as_deref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory ghost_episodes failed: {e}")))
    }

    /// Pin an entity as `ConsumerPinned`, protecting it from dream
    /// reclassification (Phase C DoD C5 / C7).
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
    /// Wraps `Memory::assert_entity_type`.
    #[napi]
    pub async fn assert_entity_type(
        &self,
        entity_id: String,
        entity_type_id: u32,
        group_id: Option<String>,
    ) -> napi::Result<()> {
        self.inner
            .assert_entity_type(kremory::GraphAssertEntityTypeParams {
                entity_id: &entity_id,
                entity_type_id,
                group_id: group_id.as_deref(),
            })
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory assert_entity_type failed: {e}"))
            })
    }

    /// Forget (hard-delete) all episodes matching `source_id` in `namespace` (B7).
    ///
    /// Wraps `Memory::forget().by_source_id(source_id).in_namespace(namespace)`.
    /// AppendOnly namespaces reject with `NamespacePolicyViolation`. When `namespace`
    /// is omitted, the Memory handle's default namespace is used. If neither is set
    /// (no per-call namespace AND no default registered on the handle), the call
    /// rejects with a namespace-required error.
    ///
    /// Returns the count of episode rows deleted.
    #[napi]
    pub async fn forget(&self, source_id: String, namespace: Option<String>) -> napi::Result<f64> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "kremory forget failed: namespace required — set per-call or open with defaultNamespace"
                )
            })?;

        let deleted = self
            .inner
            .forget()
            .by_source_id(source_id)
            .in_namespace(ns)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory forget failed: {e}")))?;

        // u64 → f64: safe up to 2^53; delete counts never approach that limit.
        Ok(deleted as f64)
    }

    /// Bound a fact's world-time `valid_to` window explicitly (ADR-071 §Item 3,
    /// TD-070) — the consumer-facing, consumer-EXPLICIT half of the
    /// supersession gap (auto-detected supersession is deferred, TD-P1-AUTO).
    ///
    /// Wraps `Memory::supersede(factId).at(validTo).with_reason(reason?)
    /// .in_namespace(namespace)`. `validTo` is an RFC-3339 timestamp string; a
    /// malformed value rejects the returned Promise. When `namespace` is
    /// omitted, the Memory handle's default namespace is used. If neither is
    /// set, the call rejects with a namespace-required error.
    ///
    /// The dream supersession sweep (`dream({ includeSupersessionSweep: true
    /// })`) later observes the bounded `validTo` and closes the window
    /// (`expiredAt = validTo`, `supersessionsRecorded` increments) —
    /// see `kremory::SupersedeRequest` for the full two-phase mechanism.
    ///
    /// When `closeNow == true` (consumer-API hardening D4b), the deterministic
    /// `window_closeout` sweep runs INLINE right after bounding, retiring
    /// already-past-dated bounds in this one call — mirrors
    /// `SupersedeRequest::close_now()`. Only past-dated bounds retire; a
    /// future-dated bound returns `retired == 0` (deferred to a later dream sweep).
    ///
    /// Returns a `JsSupersedeOutcome` `{ outcome, retired }` where `outcome` is
    /// `"bounded"` | `"rejected_time_inversion"` | `"not_found"` (mirrors the D4a
    /// honest `kremory::SupersedeOutcome::Bounded` rename) and `retired` is the
    /// in-band inline-close count (`0` unless `closeNow` retired past-dated bounds).
    #[napi]
    pub async fn supersede(
        &self,
        fact_id: i64,
        valid_to: String,
        reason: Option<String>,
        namespace: Option<String>,
        close_now: Option<bool>,
    ) -> napi::Result<JsSupersedeOutcome> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "kremory supersede failed: namespace required — set per-call or open with defaultNamespace"
                )
            })?;

        // Loud parse — caller's malformed timestamp surfaces (mirrors the
        // `remember` wrapper's `reference_time` handling above).
        let valid_to_dt = chrono::DateTime::parse_from_rfc3339(&valid_to)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| {
                napi::Error::from_reason(format!(
                    "kremory supersede failed: invalid RFC-3339 validTo {valid_to:?}: {e}"
                ))
            })?;

        let mut req = self
            .inner
            .supersede(fact_id)
            .in_namespace(ns)
            .at(valid_to_dt);
        if let Some(r) = reason {
            req = req.with_reason(r);
        }
        if close_now.unwrap_or(false) {
            req = req.close_now();
        }

        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory supersede failed: {e}")))?;

        Ok(convert::supersede_outcome_to_js(outcome))
    }

    // ── Reversible-graph-mutations (ADR-073 Tier-1) — undo + inspect ──────────

    /// Reverse a prior entity-merge by its `mutationId` (ADR-073 Tier-1, §4.2).
    ///
    /// Fully restores the loser entity, its facts, its episodic edges, and the
    /// keeper's overwritten `access_count` / `ner_confidence`, then records a merge
    /// NOGOOD so the next `dream()` will NOT re-merge the split pair. Idempotent: a
    /// second call returns `alreadyUndone = true`. Wraps `Memory::unmerge`.
    ///
    /// Obtain the `mutationId` from `mutationHistory` / `listMutations`.
    #[napi]
    pub async fn unmerge(&self, mutation_id: i64) -> napi::Result<JsUnmergeOutcome> {
        let outcome = self
            .inner
            .unmerge(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory unmerge failed: {e}")))?;
        Ok(convert::unmerge_outcome_to_js(outcome))
    }

    /// Restore a fact previously moved to `facts_archive` (P2 archival) back into
    /// the live `facts` table (ADR-073 Tier-1, §4.4). Wraps
    /// `Memory::restore_archived_fact`.
    ///
    /// Idempotent: if the fact is already live, returns `alreadyLive = true` and
    /// writes nothing.
    #[napi]
    pub async fn restore_archived_fact(
        &self,
        archived_fact_id: i64,
    ) -> napi::Result<JsRestoreArchivedOutcome> {
        let outcome = self
            .inner
            .restore_archived_fact(archived_fact_id)
            .execute()
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory restoreArchivedFact failed: {e}"))
            })?;
        Ok(convert::restore_archived_outcome_to_js(outcome))
    }

    /// Clear a supersession bound (`valid_to` / `expired_at`) set by `supersede`,
    /// re-opening the fact as currently-true (ADR-073 Tier-1, §4.5). Wraps
    /// `Memory::unsupersede`.
    ///
    /// Idempotent: a fact with no bound set returns `outcome = "not_superseded"`
    /// (an honest no-op).
    #[napi]
    pub async fn unsupersede(&self, fact_id: i64) -> napi::Result<JsUnsupersedeOutcome> {
        let outcome = self
            .inner
            .unsupersede(fact_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory unsupersede failed: {e}")))?;
        Ok(convert::unsupersede_outcome_to_js(outcome))
    }

    /// Edit an entity — retype or rename — with full FK-propagation, provenance,
    /// and reconciler-freeze re-open (ADR-073 Tier-2a, §4.3). Completes the
    /// diarization flow (rename `"Speaker 1"` → `"Alice"` propagating to all its
    /// facts). Wraps `Memory::edit_entity`.
    ///
    /// Set exactly one of `opts.newId` (rename/rekey — rejects an occupied id) or
    /// `opts.typeId` (retype). `opts.namespace` scopes the entity (else the handle
    /// default). Reverse via `undoEntityEdit` with the returned `mutationId`.
    #[napi]
    pub async fn edit_entity(
        &self,
        entity_id: String,
        opts: Option<JsEditEntityOptions>,
    ) -> napi::Result<JsEditEntityOutcome> {
        let opts = opts.unwrap_or(JsEditEntityOptions {
            new_id: None,
            type_id: None,
            namespace: None,
        });
        let ns = opts
            .namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let mut req = self.inner.edit_entity(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }
        if let Some(new_id) = opts.new_id {
            req = req.rename(new_id);
        }
        if let Some(type_id) = opts.type_id {
            req = req.retype(type_id);
        }

        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory editEntity failed: {e}")))?;
        Ok(convert::edit_entity_outcome_to_js(outcome))
    }

    /// Reverse a prior `editEntity` from its provenance snapshot (ADR-073 Tier-2a,
    /// §4.3). Pass the `mutationId` from the `EditEntityOutcome` (or from
    /// `mutationHistory` / `listMutations`). Idempotent. Wraps
    /// `Memory::undo_entity_edit`.
    #[napi]
    pub async fn undo_entity_edit(
        &self,
        mutation_id: i64,
    ) -> napi::Result<JsEditEntityOutcome> {
        let outcome = self
            .inner
            .undo_entity_edit(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory undoEntityEdit failed: {e}")))?;
        Ok(convert::edit_entity_outcome_to_js(outcome))
    }

    /// Delete an entity, reversibly (ADR-073 Tier-2b, §4.4). Archives the entity's
    /// facts (recoverable — never hard-deleted), removes its edges / community
    /// membership / FTS / row, and retracts the DERIVED artifacts of neighbours whose
    /// live-fact support drops to zero. Reverse via `undoDeleteEntity` with the
    /// returned `mutationId`. `namespace` scopes the entity (else the handle default).
    /// Wraps `Memory::delete_entity`.
    #[napi]
    pub async fn delete_entity(
        &self,
        entity_id: String,
        namespace: Option<String>,
    ) -> napi::Result<JsDeleteEntityOutcome> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());
        let mut req = self.inner.delete_entity(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }
        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory deleteEntity failed: {e}")))?;
        Ok(convert::delete_entity_outcome_to_js(outcome))
    }

    /// Delete a single fact, reversibly (ADR-073 Tier-2b, §4.5). The fact is archived
    /// (recoverable via `restoreArchivedFact`); either endpoint whose support drops to
    /// zero has its DERIVED community membership retracted. Reverse via `undoDeleteFact`.
    /// A fact id is global (not namespace-scoped). Wraps `Memory::delete_fact`.
    #[napi]
    pub async fn delete_fact(&self, fact_id: i64) -> napi::Result<JsDeleteFactOutcome> {
        let outcome = self
            .inner
            .delete_fact(fact_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory deleteFact failed: {e}")))?;
        Ok(convert::delete_fact_outcome_to_js(outcome))
    }

    /// Reverse a prior `deleteEntity` from its provenance snapshot (ADR-073 Tier-2b,
    /// §4.4) — re-inserts the entity + FTS, restores its archived facts + episodic
    /// edges, and un-retracts every community membership the cascade retracted.
    /// Idempotent. Wraps `Memory::undo_delete_entity`.
    #[napi]
    pub async fn undo_delete_entity(
        &self,
        mutation_id: i64,
    ) -> napi::Result<JsDeleteEntityOutcome> {
        let outcome = self
            .inner
            .undo_delete_entity(mutation_id)
            .execute()
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory undoDeleteEntity failed: {e}"))
            })?;
        Ok(convert::delete_entity_outcome_to_js(outcome))
    }

    /// Reverse a prior `deleteFact` from its provenance snapshot (ADR-073 Tier-2b,
    /// §4.5) — restores the archived fact + un-retracts any neighbour the cascade
    /// retracted. Idempotent. Wraps `Memory::undo_delete_fact`.
    #[napi]
    pub async fn undo_delete_fact(&self, mutation_id: i64) -> napi::Result<JsDeleteFactOutcome> {
        let outcome = self
            .inner
            .undo_delete_fact(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory undoDeleteFact failed: {e}")))?;
        Ok(convert::delete_fact_outcome_to_js(outcome))
    }

    /// Inspect the mutations that touched one entity, newest-first (ADR-073 Tier-1,
    /// §3 "Inspect surface") — the SEE half of the see+fix story. Each record
    /// carries the `mutationId` to pass to `unmerge`. Includes already-undone
    /// mutations. Wraps `Memory::mutation_history`.
    ///
    /// An entity id is namespace-scoped: pass `namespace` or open the handle with a
    /// `defaultNamespace`, else the call rejects with a namespace-required error.
    #[napi]
    pub async fn mutation_history(
        &self,
        entity_id: String,
        namespace: Option<String>,
    ) -> napi::Result<Vec<JsMutationRecord>> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let mut req = self.inner.mutation_history(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }

        let records = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory mutationHistory failed: {e}")))?;

        Ok(records.into_iter().map(convert::mutation_record_to_js).collect())
    }

    /// List logged graph mutations, newest-first (ADR-073 Tier-1, §3 "Inspect
    /// surface"). Each record carries its `mutationId` to undo. Wraps
    /// `Memory::list_mutations`.
    ///
    /// Filter via `opts`: `namespace` scopes to one namespace (else the
    /// `defaultNamespace`, else ALL namespaces); `kind` restricts the mutation kind
    /// (unknown tag rejects); `since` sets an RFC3339 `created_at` lower bound
    /// (malformed rejects); `includeUndone` adds already-reversed mutations
    /// (default: LIVE / still-reversible only).
    #[napi]
    pub async fn list_mutations(
        &self,
        opts: Option<JsMutationFilter>,
    ) -> napi::Result<Vec<JsMutationRecord>> {
        let mut req = self.inner.list_mutations();

        if let Some(f) = opts {
            let ns = f
                .namespace
                .as_deref()
                .map(Namespace::new)
                .or_else(|| self.default_namespace.clone());
            if let Some(ns) = ns {
                req = req.in_namespace(ns);
            }
            // Parse the kind tag loudly via the substrate enum's snake_case serde —
            // an unknown tag is a caller error, surfaced not silently dropped.
            if let Some(ref kind_str) = f.kind {
                let kind: kremory::MutationKind = serde_json::from_value(
                    serde_json::Value::String(kind_str.clone()),
                )
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "JsMutationFilter.kind: unknown value {kind_str:?}: {e}"
                    ))
                })?;
                req = req.kind(kind);
            }
            if let Some(ref since_str) = f.since {
                let ts = chrono::DateTime::parse_from_rfc3339(since_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        napi::Error::from_reason(format!(
                            "JsMutationFilter.since: invalid RFC-3339 timestamp {since_str:?}: {e}"
                        ))
                    })?;
                req = req.since(ts);
            }
            if f.include_undone.unwrap_or(false) {
                req = req.include_undone(true);
            }
        }

        let records = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory listMutations failed: {e}")))?;

        Ok(records.into_iter().map(convert::mutation_record_to_js).collect())
    }

    /// Search memory for context matching `query`.
    ///
    /// Returns up to `opts.k` (default 10) results ranked by relevance.
    #[napi]
    pub async fn recall(
        &self,
        query: String,
        opts: Option<JsRecallOptions>,
    ) -> napi::Result<Vec<JsRetrievedContext>> {
        let namespaces = convert::resolve_recall_namespaces(&opts);
        // Apply handle-level default ONLY when neither per-call selector is set.
        // If `in_namespaces` is set, we must NOT also inject a default — that
        // would trip `ConflictingNamespaceSelectors`.
        let namespace = convert::resolve_recall_namespace(&opts).or_else(|| {
            if namespaces.is_none() {
                self.default_namespace.clone()
            } else {
                None
            }
        });
        let best_effort = opts.as_ref().and_then(|o| o.best_effort);
        let per_namespace_top_k = opts
            .as_ref()
            .and_then(|o| o.per_namespace_top_k)
            .map(|n| usize::try_from(n).unwrap_or(10));
        let k = opts
            .as_ref()
            .and_then(|o| o.k)
            .map(|k_val| usize::try_from(k_val).unwrap_or(10));
        let as_of = opts
            .as_ref()
            .and_then(|o| o.as_of.as_deref())
            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok());

        let results = {
            let mut builder = self.inner.recall(query);

            // Mutual exclusion is enforced at `.await` time by kremory's
            // `check_selectors`; surface both if caller sets both so the
            // ConflictingNamespaceSelectors error propagates naturally.
            if let Some(ns) = namespace {
                builder = builder.in_namespace(ns);
            }
            if let Some(ref nss) = namespaces {
                builder = builder.in_namespaces(nss);
            }
            if let Some(b) = best_effort {
                builder = builder.best_effort(b);
            }
            if let Some(n) = per_namespace_top_k {
                builder = builder.per_namespace_top_k(n);
            }
            if let Some(k_val) = k {
                builder = builder.k(k_val);
            }
            if let Some(as_of_ts) = as_of {
                builder = builder.as_of(as_of_ts);
            }

            // B9: wire filterMetadata entries. Each entry maps to one
            // RecallRequest::filter_metadata(key, value) call. Validation
            // (key length, JSON-path metachars) is enforced by the substrate
            // at await time, surfacing as an Err from builder.raw().await.
            if let Some(filters) = opts.as_ref().and_then(|o| o.filter_metadata.as_ref()) {
                for f in filters {
                    builder = builder.filter_metadata(&f.key, f.value.clone());
                }
            }

            builder
                .raw()
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory recall failed: {e}")))?
        };

        Ok(results
            .into_iter()
            .map(convert::retrieved_context_to_js)
            .collect())
    }

    /// Bulk-ingest multiple episodes in a single batch (ADR-034 §B8).
    ///
    /// Wraps `Memory::remember_batch()`. Each entry in `opts.episodes` maps to
    /// one `RememberBatchBuilder::entry(content).in_namespace(ns).done()` call.
    /// `opts.batchId` is forwarded to `with_batch_id` for idempotent tracking
    /// via `awaitBatch`.
    ///
    /// Returns a `Vec<IngestResult>` in the same order as the input episodes.
    /// The batch is committed sequentially — the first error aborts the batch.
    #[napi]
    pub async fn remember_batch(&self, opts: JsBatchOptions) -> napi::Result<Vec<JsIngestResult>> {
        let mut builder = self.inner.remember_batch();

        if let Some(ref id) = opts.batch_id {
            builder = builder.with_batch_id(id.clone());
        }

        // Accumulate (content, namespace, source_ref) from each JsRememberOptions.
        // We drive the builder's entry() chain directly.
        for ep_opts in opts.episodes {
            let ns = ep_opts
                .namespace
                .as_deref()
                .map(kremory::Namespace::new)
                .or_else(|| self.default_namespace.clone());

            let facts: Vec<kremory::memory::types::StructuredFact> = match ep_opts.structured_facts
            {
                Some(items) => items
                    .into_iter()
                    .map(kremory::memory::types::StructuredFact::try_from)
                    .collect::<Result<_, _>>()?,
                None => Vec::new(),
            };

            let mut entry = builder.entry(ep_opts.content.clone());

            if let Some(ns) = ns {
                entry = entry.in_namespace(ns);
            }
            if let Some(ref sid) = ep_opts.source_id {
                entry = entry.from_document(sid.clone());
            }
            if let Some(ref ts_str) = ep_opts.reference_time {
                let ts = chrono::DateTime::parse_from_rfc3339(ts_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        napi::Error::from_reason(format!(
                            "BatchOptions.episodes[].reference_time: invalid RFC-3339 {:?}: {}",
                            ts_str, e
                        ))
                    })?;
                entry = entry.published_at(ts);
            }
            if !facts.is_empty() {
                entry = entry.with_facts(facts);
            }

            builder = entry.done();
        }

        let commits = builder
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory rememberBatch failed: {e}")))?;

        Ok(commits
            .into_iter()
            .map(|c| JsIngestResult {
                episode_entity_id: c.episode_entity_id,
                run_id: c.run_id.map(|u| u.to_string()),
                committed_at: c.committed_at.to_rfc3339(),
                warnings: Vec::new(),
            })
            .collect())
    }

    /// Query Phase 2 enrichment status for a background ingest run (ADR-034 §H1).
    ///
    /// Wraps `Memory::status_of`. `commit_id` must be the RFC-4122 UUID string
    /// from `IngestResult.run_id` (NOT `IngestResult.episode_entity_id`, which
    /// is the graph node ID). `run_id` is `Some(...)` when the substrate
    /// spawned a background enrichment task (e.g. via the substrate
    /// `.no_wait()` builder).
    ///
    /// Returns `IngestStatusResult` with `status` one of:
    /// `"pending"` | `"extracting"` | `"deduplicating"` | `"invalidating"` |
    /// `"complete"` | `"failed"`.
    #[napi]
    pub async fn status_of(&self, commit_id: String) -> napi::Result<JsIngestStatusResult> {
        let run_id = uuid::Uuid::parse_str(&commit_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        // Build a minimal EpisodeCommit so the facade's status_of method can
        // extract run_id from it.
        let commit = kremory::EpisodeCommit {
            run_id: Some(run_id),
            episode_entity_id: String::new(),
            committed_at: chrono::Utc::now(),
            stub_entities_inserted: 0,
        };

        let status = self
            .inner
            .status_of(&commit)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory statusOf failed: {e}")))?;

        Ok(convert::ingest_status_to_js(status))
    }

    /// Block until Phase 2 enrichment for `commitId` reaches a terminal status
    /// (ADR-034 §H2).
    ///
    /// Wraps `Memory::await_enrichment`. `commit_id` is the RFC-4122 UUID
    /// string from `IngestResult.run_id` (NOT `IngestResult.episode_entity_id`).
    /// `timeoutMs` is mandatory — pass `300_000` (5 min) for a sensible default.
    /// Returns `IngestStatusResult`.
    #[napi]
    pub async fn await_enrichment(
        &self,
        commit_id: String,
        timeout_ms: i64,
    ) -> napi::Result<JsIngestStatusResult> {
        let run_id = uuid::Uuid::parse_str(&commit_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let commit = kremory::EpisodeCommit {
            run_id: Some(run_id),
            episode_entity_id: String::new(),
            committed_at: chrono::Utc::now(),
            stub_entities_inserted: 0,
        };

        let timeout = std::time::Duration::from_millis(timeout_ms as u64);

        let status = self
            .inner
            .await_enrichment(&commit, timeout)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory awaitEnrichment failed: {e}"))
            })?;

        Ok(convert::ingest_status_to_js(status))
    }

    /// Block until the dream-phase run identified by `handleId` reaches a
    /// terminal status (ADR-034 §H3).
    ///
    /// Wraps `Memory::await_dream`. `timeoutMs` is mandatory.
    /// Returns `DreamStatusResult` with `status` one of:
    /// `"pending"` | `"processing"` | `"complete"` | `"failed"`.
    #[napi]
    pub async fn await_dream(
        &self,
        handle_id: String,
        timeout_ms: i64,
    ) -> napi::Result<JsDreamStatusResult> {
        let run_id = uuid::Uuid::parse_str(&handle_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let handle = kremory::DreamHandle {
            run_id,
            namespace: kremory::Namespace::new(""),
            submitted_at: chrono::Utc::now(),
            batch_id: None,
        };

        let timeout = std::time::Duration::from_millis(timeout_ms as u64);

        let status = self
            .inner
            .await_dream(&handle, timeout)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory awaitDream failed: {e}")))?;

        Ok(convert::dream_status_to_js(status))
    }

    /// Block until all episodes in `batchId` reach a terminal status
    /// (ADR-034 §H4).
    ///
    /// Wraps `Memory::await_batch`. `timeoutMs` is mandatory.
    /// Returns `BatchStatus` with `total`, `completed`, `skipped`, `failed`.
    #[napi]
    pub async fn await_batch(
        &self,
        batch_id: String,
        timeout_ms: i64,
    ) -> napi::Result<JsBatchStatus> {
        let timeout = std::time::Duration::from_millis(timeout_ms as u64);

        let status = self
            .inner
            .await_batch(&batch_id, timeout)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory awaitBatch failed: {e}")))?;

        Ok(convert::batch_status_to_js(status))
    }

    /// Cancel an in-flight Phase 2 or Phase 3 run for `commitId` (ADR-034 §C1).
    ///
    /// Wraps `Memory::cancel`. `commit_id` is the RFC-4122 UUID string from
    /// `IngestResult.run_id` (NOT `IngestResult.episode_entity_id`).
    /// Returns `CancelOutcome`.
    #[napi]
    pub async fn cancel(&self, commit_id: String) -> napi::Result<JsCancelOutcome> {
        let run_id = uuid::Uuid::parse_str(&commit_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let commit = kremory::EpisodeCommit {
            run_id: Some(run_id),
            episode_entity_id: String::new(),
            committed_at: chrono::Utc::now(),
            stub_entities_inserted: 0,
        };

        let outcome = self
            .inner
            .cancel(&commit)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory cancel failed: {e}")))?;

        Ok(convert::cancel_outcome_to_js(outcome))
    }

    /// Cancel a dream-phase run by its handle UUID (ADR-034 §C2).
    ///
    /// Wraps `Memory::cancel_dream`. `handleId` is the RFC-4122 UUID string of
    /// the dream run. Returns `CancelOutcome`.
    #[napi]
    pub async fn cancel_dream(&self, handle_id: String) -> napi::Result<JsCancelOutcome> {
        let run_id = uuid::Uuid::parse_str(&handle_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let handle = kremory::DreamHandle {
            run_id,
            namespace: kremory::Namespace::new(""),
            submitted_at: chrono::Utc::now(),
            batch_id: None,
        };

        let outcome =
            self.inner.cancel_dream(&handle).await.map_err(|e| {
                napi::Error::from_reason(format!("kremory cancelDream failed: {e}"))
            })?;

        Ok(convert::cancel_outcome_to_js(outcome))
    }

    /// Register a namespace and its policy explicitly, ahead of any writes
    /// (ADR-034 §N1, ADR-029a).
    ///
    /// Wraps `Memory::register_namespace`. Idempotent for same policy; returns
    /// error on downgrade attempt. Calling at startup before `remember` is the
    /// recommended pattern per ADR-029a Decision 6.
    #[napi]
    pub async fn register_namespace(&self, namespace: String) -> napi::Result<()> {
        self.inner
            .register_namespace(kremory::Namespace::new(namespace))
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory registerNamespace failed: {e}")))
    }

    /// Monotonically upgrade a namespace's immutability from `Mutable` to
    /// `AppendOnly` (ADR-034 §N2, ADR-029b Decision 5).
    ///
    /// Wraps `Memory::upgrade_namespace_policy`. One-way ratchet: downgrade
    /// attempts return an error. Idempotent if already `AppendOnly`.
    #[napi]
    pub async fn upgrade_namespace_policy(&self, namespace: String) -> napi::Result<()> {
        self.inner
            .upgrade_namespace_policy(kremory::Namespace::new(namespace))
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory upgradeNamespacePolicy failed: {e}"))
            })
    }

    /// Close the memory handle, flushing any pending writes.
    ///
    /// Calling `close()` is a no-op at v0.1.0; WAL flush semantics land in
    /// v0.1.1. Call this at shutdown to future-proof your code.
    #[napi]
    pub async fn close(&self) -> napi::Result<()> {
        self.inner
            .close()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory close failed: {e}")))
    }
}

// ── Tier-2 BYOM / BYOE helpers (live cdylib path only) ───────────────────────
//
// These helpers are excluded from `#[cfg(test)]` because:
// - `ThreadsafeFunction` cannot be constructed outside a napi runtime.
// - `resolve_env_llm` constructs real LLM provider instances that require
//   live network endpoints.
//
// Integration tests cover the extractor-selection matrix via `MockExtractorBridge`
// in `tests/extractor_selection_compat_matrix.rs`.
//
// # Typestate note
//
// `MemoryBuilder::IntoFuture` is only implemented for `<WithLlm, WithEmb>` and
// `<NoLlm, WithEmb>`. Therefore every builder path that ends in `.await` must have
// both an LLM (optional for NoLlm path) AND an embedder set. The `withEmbedder`
// callback is required when extractor knobs are used — the ADR-039 §6 compat matrix
// lists no row where an extractor is set without a BYOM embedder.

/// Open a Memory with a caller-supplied JS embedder callback (ADR-030 Tier-2).
///
/// Also applies any BYOE extractor / GLiNER knobs from the options object.
/// Called when `opts.withEmbedder` is present.
///
/// Builder typestate path: `NoLlm,NoEmb` → `with_llm` → `WithLlm,NoEmb`
///   → `with_embedder` → `WithLlm,WithEmb` → `.await`.
#[cfg(not(test))]
async fn open_with_js_embedder(
    path: String,
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
    expected_dim: Option<usize>,
    default_namespace: Option<Namespace>,
    gliner_cfg: Option<convert::GlinerConfigJs>,
    extractor_handle: Option<bridge::ExternalExtractorHandle>,
) -> napi::Result<JsMemory> {
    use std::sync::Arc;

    // Step 1: env-detect LLM.
    let llm = bridge::resolve_env_llm().await?;

    // Step 2: wrap the JS embedder callback.
    let emb: Arc<dyn kremory::DynEmbeddingProvider> =
        bridge::into_arc(bridge::JsEmbedderBridge::new(tsfn, expected_dim));

    // Step 3: build Memory — LLM + embedder + optional extractor knobs.
    // Builder is now `WithLlm, WithEmb` after with_llm + with_embedder.
    let mut builder = Memory::open(&path).with_llm(llm).with_embedder(emb);

    // Apply BYOE extractor knobs (mutually-exclusive guard already checked in open()).
    if let Some(handle) = extractor_handle {
        builder =
            builder.with_extractor(Arc::new(bridge::ExternalExtractorJs::from_handle(handle)));
    } else if gliner_cfg.is_some() {
        // GLiNER requires ner feature; already checked in open().
        // F2: with_gliner() takes no arg (GlinerConfig had no public fields) — the
        // presence of opts.gliner is the enable signal; its contents are unused.
        #[cfg(feature = "ner")]
        {
            builder = builder.with_gliner();
        }
        #[cfg(not(feature = "ner"))]
        {
            return Err(napi::Error::from_reason(
                "KremoryError::FeatureDisabled('ner'): opts.gliner requires --features ner",
            ));
        }
    }

    let mem = builder
        .await
        .map_err(|e| napi::Error::from_reason(format!("kremory open with embedder failed: {e}")))?;

    Ok(JsMemory {
        inner: mem,
        default_namespace,
    })
}

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

mod convert;
pub mod bridge;

use napi_derive::napi;

use kremory::{Memory, Namespace, SourceKind};

pub use convert::{
    JsBatchOptions, JsBatchStatus, JsCancelOutcome, JsDreamOpts, JsDreamStatusResult,
    JsDreamSummary, JsEpisode, JsIngestResult, JsIngestStatusResult, JsMetadataFilter,
    JsOpenOptions, JsRecallOptions, JsRememberOptions, JsRetrievedContext, JsStructuredFact,
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
    /// When `opts.withEmbedder` is absent or null, uses env-detected providers
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
    /// ## `opts.defaultNamespace`
    ///
    /// When set, becomes the handle-level default namespace applied to subsequent
    /// ingest/recall calls that omit per-call namespace.
    #[napi(factory)]
    pub async fn open(path: String, opts: Option<JsOpenOptions>) -> napi::Result<JsMemory> {
        // KH2: handle opts.extractor BEFORE Tier-1/Tier-2 dispatch.
        // 'hybrid' on a non-`ner` build returns a typed FeatureDisabled error.
        // Valid values set KREMORY_EXTRACTOR env so both Tier-1 (Memory::auto)
        // and Tier-2 (MemoryBuilder) honor the choice via ExtractorSource::FromEnv.
        // Invalid values return a typed napi error.
        if let Some(extractor) = opts.as_ref().and_then(|o| o.extractor.as_deref()) {
            match extractor {
                "auto" => { /* defer to existing env var (or default) */ }
                "nuextract" => std::env::set_var("KREMORY_EXTRACTOR", "nuextract"),
                "hybrid" => {
                    #[cfg(feature = "ner")]
                    std::env::set_var("KREMORY_EXTRACTOR", "hybrid");
                    #[cfg(not(feature = "ner"))]
                    return Err(napi::Error::from_reason(
                        "KremoryError::FeatureDisabled('ner'): \
                         JsOpenOptions.extractor='hybrid' requires kremory-napi \
                         built with --features ner",
                    ));
                }
                other => {
                    return Err(napi::Error::from_reason(format!(
                        "invalid JsOpenOptions.extractor '{other}'; expected 'auto'|'hybrid'|'nuextract'"
                    )));
                }
            }
        }
        // Tier-2 BYOM path: only available in cdylib builds (ThreadsafeFunction
        // requires the napi runtime which is absent in cargo test binary builds).
        // Extract opts fields before consuming opts.with_embedder.
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

            if let Some(tsfn) = opts.and_then(|o| o.with_embedder) {
                // `opts.with_embedder` is already a `ThreadsafeFunction` (napi-rs
                // converts from the JS callback at `FromNapiValue` time). It is
                // `Send + Sync`, so it can be passed directly into the async block
                // without `create_threadsafe_function`.
                return open_with_js_embedder(path, tsfn, expected_dim, default_namespace).await;
            }
            // with_embedder was None — fall through to Tier-1. Re-extract
            // default_namespace from the opts we didn't consume.
            // (opts was None or its with_embedder was None — we already extracted
            // default_namespace above, so reuse it.)
            let mem = Memory::auto(&path)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;
            Ok(JsMemory {
                inner: mem,
                default_namespace,
            })
        }

        // Tier-1 path for test builds (cfg(test) excludes the block above).
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
                warnings.push("metadata supplied without source_id — metadata not stored".to_string());
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
        #[allow(clippy::cast_precision_loss)]
        Ok(updated as f64)
    }

    /// Update an episode's source URI by source_id (B3). v0.1.8 1:1 contract per ADR-034.
    ///
    /// Wraps substrate `Memory::update_source_uri`. Rejects if no episode matches
    /// `source_id`. Returns the count of rows updated.
    #[napi]
    pub async fn update_source_uri(
        &self,
        source_id: String,
        new_uri: String,
    ) -> napi::Result<f64> {
        let updated = self
            .inner
            .update_source_uri(source_id)
            .to(new_uri)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory updateSourceUri failed: {e}")))?;

        // u64 → f64: safe up to 2^53; episode update counts never approach that limit.
        #[allow(clippy::cast_precision_loss)]
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

        Ok(episodes
            .into_iter()
            .map(|ep| convert::episode_to_js(ep, &source_id))
            .collect())
    }

    /// Trigger the dream-phase batch consolidation (B6).
    ///
    /// Blocks until the dream completes. Returns a `JsDreamSummary` with
    /// per-phase accounting. Wraps `Memory::dream()`.
    #[napi]
    pub async fn dream(&self, opts: Option<JsDreamOpts>) -> napi::Result<JsDreamSummary> {
        let ns = opts
            .as_ref()
            .and_then(|o| o.namespace.as_deref())
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let req = if let Some(ns) = ns {
            self.inner.dream().in_namespace(ns)
        } else {
            self.inner.dream()
        };

        let summary = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory dream failed: {e}")))?;

        Ok(convert::dream_summary_to_js(summary))
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
        #[allow(clippy::cast_precision_loss)]
        Ok(deleted as f64)
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
    pub async fn remember_batch(
        &self,
        opts: JsBatchOptions,
    ) -> napi::Result<Vec<JsIngestResult>> {
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

            let facts: Vec<kremory::memory::types::StructuredFact> =
                match ep_opts.structured_facts {
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

        let outcome = self
            .inner
            .cancel_dream(&handle)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory cancelDream failed: {e}")))?;

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
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory registerNamespace failed: {e}"))
            })
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
                napi::Error::from_reason(format!(
                    "kremory upgradeNamespacePolicy failed: {e}"
                ))
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

// ── Tier-2 BYOM helper (live cdylib path only) ────────────────────────────────

/// Open a kremory Memory at `path` with a caller-supplied JS embedder callback.
///
/// Called by `JsMemory::open` when `opts.withEmbedder` is present.
///
/// Steps:
///   1. Env-detect LLM via `bridge::resolve_env_llm`.
///   2. Wrap `tsfn` in `JsEmbedderBridge` with optional `expected_dim` validation.
///   3. Build Memory via `Memory::open(path).with_llm(llm).with_embedder(emb).await`.
///
/// `expected_dim` comes from `opts.embeddingDim`. When present, the bridge will
/// return `Err` if the callback returns a vec of a different length.
///
/// This function is excluded from `#[cfg(test)]` because:
/// - `ThreadsafeFunction` cannot be constructed outside a napi runtime.
/// - `resolve_env_llm` constructs real LLM provider instances that require
///   live network endpoints.
///
/// Cargo tests cover the bridge logic via the mock variant in `bridge.rs`.
#[cfg(not(test))]
async fn open_with_js_embedder(
    path: String,
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
    expected_dim: Option<usize>,
    default_namespace: Option<Namespace>,
) -> napi::Result<JsMemory> {
    use std::sync::Arc;

    // Step 1: env-detect LLM.
    let llm = bridge::resolve_env_llm().await?;

    // Step 2: wrap the JS callback in the bridge, which implements DynEmbeddingProvider.
    let emb: Arc<dyn kremory::DynEmbeddingProvider> =
        bridge::into_arc(bridge::JsEmbedderBridge::new(tsfn, expected_dim));

    // Step 3: build Memory via Tier-2 builder.
    let mem = Memory::open(&path)
        .with_llm(llm)
        .with_embedder(emb)
        .await
        .map_err(|e| napi::Error::from_reason(format!("kremory open with embedder failed: {e}")))?;

    Ok(JsMemory {
        inner: mem,
        default_namespace,
    })
}

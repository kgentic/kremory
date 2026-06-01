//! `kremory-napi` — Node.js binding for the kremory `Memory` facade.
//!
//! Exposes `kremory::Memory` to TypeScript/JavaScript via napi-rs derive macros.
//! Per ADR-030 Decision 2 Form B: single binding, no PyO3, no wasm-bindgen.
//!
//! # Binding surface
//!
//! - `JsMemory` wraps `kremory::Memory`. Async methods delegate to a tokio
//!   multi-thread runtime via napi-rs `async` feature.
//! - Plain data structs (`JsOpenOptions`, `JsRecallOptions`, `JsIngestOptions`,
//!   `JsRetrievedContext`, `JsIngestResult`) are `#[napi(object)]` — napi-rs
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

use napi_derive::napi;

use kremory::{Memory, Namespace, SourceKind};

pub use convert::{
    JsDreamOpts, JsDreamSummary, JsEpisode, JsEpisodeDraft, JsIngestOptions, JsIngestResult,
    JsMetadataFilter, JsOpenOptions, JsRecallOptions, JsRetrievedContext,
};

// ── JsMemory ──────────────────────────────────────────────────────────────────

/// Node.js handle for a kremory `Memory` instance.
///
/// Obtain via `JsMemory.open(path, opts?)`.
/// `close()` should be called at shutdown to future-proof against v0.1.1+ WAL
/// flush semantics.
#[napi]
pub struct JsMemory {
    inner: Memory,
    /// Handle-level default namespace captured from `JsOpenOptions.defaultNamespace`
    /// at `open` time. Applied to ingest/recall calls that don't pass an explicit
    /// per-call namespace. Set-once, never mutated — safe for concurrent reads.
    default_namespace: Option<Namespace>,
}

#[napi]
impl JsMemory {
    /// Open a kremory Memory at `path`, using env-detected providers
    /// (`OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`).
    ///
    /// If `opts.defaultNamespace` is set it becomes the handle-level default
    /// applied to subsequent ingest/recall calls that omit per-call namespace.
    ///
    /// `opts.embeddingDim` is reserved for Tier-2 builder wiring (deferred per
    /// ADR-030 Form B); setting it currently emits a `tracing::warn!` and is
    /// otherwise ignored. The active provider's native dimension is used.
    #[napi(factory)]
    pub async fn open(path: String, opts: Option<JsOpenOptions>) -> napi::Result<JsMemory> {
        // Tier 1: env-auto provider detection.
        let mem = Memory::auto(&path)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;

        let default_namespace = opts
            .as_ref()
            .and_then(|o| o.default_namespace.as_deref())
            .map(Namespace::new);

        if let Some(dim) = opts.as_ref().and_then(|o| o.embedding_dim) {
            tracing::warn!(
                requested_dim = dim,
                "JsOpenOptions.embeddingDim is currently ignored — Tier-2 builder \
                 wiring deferred per ADR-030 Form B. Provider's native dim is used."
            );
        }

        Ok(JsMemory {
            inner: mem,
            default_namespace,
        })
    }

    /// Ingest a text episode into memory.
    ///
    /// Blocks until Phase 2 enrichment completes (default kremory behaviour).
    /// Returns a lightweight `JsIngestResult` with the committed episode ID.
    #[napi]
    pub async fn ingest(
        &self,
        text: String,
        opts: Option<JsIngestOptions>,
    ) -> napi::Result<JsIngestResult> {
        let namespace =
            convert::resolve_ingest_namespace(&opts).or_else(|| self.default_namespace.clone());

        let commit = if let Some(ns) = namespace {
            self.inner
                .remember(text)
                .in_namespace(ns)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory ingest failed: {e}")))?
        } else {
            self.inner
                .remember(text)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory ingest failed: {e}")))?
        };

        Ok(JsIngestResult {
            episode_entity_id: commit.episode_entity_id,
            committed_at: commit.committed_at.to_rfc3339(),
            warnings: vec![],
        })
    }

    /// Ingest an episode with full source provenance (B1).
    ///
    /// Parallel method to `ingest(text, opts)`. Accepts a `JsEpisodeDraft` which
    /// carries optional `source_id`, `source_uri`, `metadata`, and `namespace`.
    ///
    /// When `source_id` is provided, the episode is tagged with that identifier
    /// (queryable via `getBySourceId`). `source_uri` and `metadata` are written
    /// via post-ingest updates (not atomic with the Phase 1 commit — the episode
    /// row exists even if the secondary writes fail).
    ///
    /// # Namespace resolution
    ///
    /// `draft.namespace` > handle-level default > rejection.
    #[napi]
    pub async fn ingest_episode(&self, draft: JsEpisodeDraft) -> napi::Result<JsIngestResult> {
        let namespace = draft
            .namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        // Build the remember request. Use from_source with Document kind when
        // source_id is provided; otherwise let RememberRequest generate a UUID.
        let req = if let Some(ref sid) = draft.source_id {
            self.inner
                .remember(draft.content.clone())
                .from_source(sid.clone(), SourceKind::Document)
        } else {
            self.inner.remember(draft.content.clone())
        };

        let req = if let Some(ns) = namespace {
            req.in_namespace(ns)
        } else {
            req
        };

        let commit = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory ingest_episode failed: {e}")))?;

        let mut warnings: Vec<String> = Vec::new();

        // Post-ingest: set source_uri if provided. Non-fatal — warn on failure.
        if let Some(ref uri) = draft.source_uri {
            if let Some(ref sid) = draft.source_id {
                match self
                    .inner
                    .update_source_uri(sid.clone())
                    .to(uri.clone())
                    .await
                {
                    Ok(_) => {}
                    Err(e) => {
                        warnings.push(format!("source_uri update failed: {e}"));
                    }
                }
            } else {
                warnings.push("source_uri supplied without source_id — uri not stored".to_string());
            }
        }

        // Post-ingest: set metadata if provided. Non-fatal — warn on failure.
        if let Some(meta) = draft.metadata {
            if let Some(ref sid) = draft.source_id {
                match self
                    .inner
                    .update_episode_metadata(sid.clone())
                    .patch(meta)
                    .await
                {
                    Ok(_) => {}
                    Err(e) => {
                        warnings.push(format!("metadata update failed: {e}"));
                    }
                }
            } else {
                warnings
                    .push("metadata supplied without source_id — metadata not stored".to_string());
            }
        }

        Ok(JsIngestResult {
            episode_entity_id: commit.episode_entity_id,
            committed_at: commit.committed_at.to_rfc3339(),
            warnings,
        })
    }

    /// Update an episode's metadata by source_id (B2).
    ///
    /// Performs a shallow merge: existing metadata keys are preserved; the
    /// `patch` keys overwrite. Wraps `Memory::update_episode_metadata`.
    ///
    /// Returns the count of episode rows updated.
    #[napi]
    pub async fn update_metadata(
        &self,
        source_id: String,
        patch: serde_json::Value,
    ) -> napi::Result<f64> {
        let updated = self
            .inner
            .update_episode_metadata(source_id)
            .patch(patch)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory updateMetadata failed: {e}")))?;

        // usize → f64: safe up to 2^53; episode counts never approach that limit.
        #[allow(clippy::cast_precision_loss)]
        Ok(updated as f64)
    }

    /// Update an episode's source URI by source_id (B3).
    ///
    /// Wraps `Memory::update_source_uri`. Rejects if no episode matches
    /// `source_id`. Returns the count of rows updated.
    #[napi]
    pub async fn update_uri(&self, source_id: String, new_uri: String) -> napi::Result<f64> {
        let updated = self
            .inner
            .update_source_uri(source_id)
            .to(new_uri)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory updateUri failed: {e}")))?;

        // u64 → f64: safe up to 2^53; episode update counts never approach that limit.
        #[allow(clippy::cast_precision_loss)]
        Ok(updated as f64)
    }

    /// Direct slug/source_id lookup — returns episodes matching `source_id` (B5).
    ///
    /// Results are ordered newest-first. When `namespace` is omitted, the
    /// Memory handle's default namespace is used; when the handle has no default,
    /// results span all namespaces.
    ///
    /// # Known gap
    ///
    /// The `sourceUri` field on each returned `JsEpisode` is always `null` —
    /// the substrate `recall_by_source_id` query does not select that column.
    /// Use `updateUri` to write and the value is persisted in the DB.
    #[napi]
    pub async fn get_by_source_id(
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
            .map_err(|e| napi::Error::from_reason(format!("kremory getBySourceId failed: {e}")))?;

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

    /// Reindex the memory store (B8 — stub, deferred to v0.1.7).
    ///
    /// Always rejects with a "deferred" error. Reserved for future vector index
    /// rebuild capability. Consumers should not call this in v0.1.6.
    #[napi]
    pub async fn reindex(&self) -> napi::Result<()> {
        Err(napi::Error::from_reason(
            "kremory reindex: not yet implemented — deferred to v0.1.7",
        ))
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

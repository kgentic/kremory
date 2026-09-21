//! Episode ingest: `remember`, metadata/URI patching, batch ingest, and the
//! status/enrichment/cancel tracking surface for in-flight ingest commits.

use napi_derive::napi;
use super::JsMemory;
use crate::convert;
use crate::convert::*;
use kremory::{Namespace, SourceKind};

#[napi]
impl JsMemory {
    /// Ingest an episode with full provenance + optional pre-extracted facts.
    ///
    /// The v0.1.8 unified ingest surface. Collapses prior
    /// `JsMemory.ingest(text, opts)` + `JsMemory.ingest_episode(draft)`
    /// into a single options-object method that mirrors the substrate's
    /// fluent `mem.remember(content).with_facts(...).skip_extraction()....`
    /// builder chain.
    ///
    /// # Namespace resolution
    /// `opts.namespace` > handle-level default > rejection.
    ///
    /// # Caller-pinned facts
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
        // StructuredFact (memory layer). Loud-parse on invalid timestamps.
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

        if let Some(ns) = namespace.clone() {
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

        // TD-253: surface substrate embed failures onto the ONE wire field JS
        // already reads (`IngestResult.warnings`) rather than adding a second
        // one — `commit.embedding_failures` names each fact/entity whose
        // embedding call (or the store that followed it) failed; the row is
        // still persisted and recallable via BM25 + graph traversal, just not
        // via dense search until re-embedded.
        for failure in &commit.embedding_failures {
            warnings.push(format!("embedding failed, not dense-searchable: {failure}"));
        }

        // Post-ingest: write source_uri if both source_id + source_uri supplied.
        //
        // BOTH post-ingest writes MUST carry the same namespace Phase 1
        // wrote into. `self.inner` is built by `Memory::auto()` and therefore
        // holds NO substrate-side `default_namespace` — this wrapper keeps it in
        // `JsMemory::default_namespace` instead. So without threading `namespace`
        // through explicitly, these writes would resolve to "no scope" and span
        // EVERY namespace, silently patching another tenant's rows that happen to
        // share this caller-chosen `source_id` (which carries no uniqueness
        // constraint). Pinned substrate-side by `it::td235_metadata_namespace_scope`.
        if let Some(ref uri) = opts.source_uri {
            if let Some(ref sid) = opts.source_id {
                let mut uri_req = self.inner.update_source_uri(sid.clone());
                if let Some(ns) = namespace.clone() {
                    uri_req = uri_req.in_namespace(ns);
                }
                if let Err(e) = uri_req.to(uri.clone()).await {
                    warnings.push(format!("source_uri update failed: {e}"));
                }
            } else {
                warnings.push("source_uri supplied without source_id — uri not stored".to_string());
            }
        }

        // Post-ingest: write metadata if both source_id + metadata supplied.
        // Namespace-scoped for the same reason as the source_uri write above.
        if let Some(meta) = opts.metadata {
            if let Some(ref sid) = opts.source_id {
                let mut meta_req = self.inner.update_episode_metadata(sid.clone());
                if let Some(ns) = namespace {
                    meta_req = meta_req.in_namespace(ns);
                }
                if let Err(e) = meta_req.patch(meta).await {
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

    /// Update an episode's metadata by source_id. v0.1.8 1:1 contract.
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

    /// Update an episode's source URI by source_id. v0.1.8 1:1 contract.
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

    /// Bulk-ingest multiple episodes in a single batch.
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

    /// Query Phase 2 enrichment status for a background ingest run.
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
            dense_embedded: None,
            embedding_failures: Vec::new(),
        };

        let status = self
            .inner
            .status_of(&commit)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory statusOf failed: {e}")))?;

        Ok(convert::ingest_status_to_js(status))
    }

    /// Block until Phase 2 enrichment for `commitId` reaches a terminal status.
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
            dense_embedded: None,
            embedding_failures: Vec::new(),
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

    /// Block until `episodeId`'s Phase-2 enrichment reaches a terminal
    /// `episodes.episode_processing_status` (`Verified` / `Failed`), or the
    /// timeout elapses.
    ///
    /// Wraps `Memory::wait_for_processing`. `episodeId` is the RAW EPISODE
    /// ROWID — parse it from `IngestResult.episodeEntityId`, NOT `runId` (that
    /// is a separate identifier for `statusOf`/`awaitEnrichment`, which poll a
    /// DIFFERENT in-memory run-tracking table, not this column). `timeoutMs`
    /// is mandatory — pass a sensible default such as `30_000` (30s).
    ///
    /// Node parity pass (2026-09-14): tracked in `parity-skip.toml` since
    /// ADR-051 Phase 4 shipped the Rust side; this closes that gap.
    ///
    /// # Errors
    ///
    /// Rejects if Phase-2 extraction reached `Failed`, or if `timeoutMs`
    /// elapsed before a terminal status was reached.
    #[napi]
    pub async fn wait_for_processing(
        &self,
        episode_id: i64,
        timeout_ms: i64,
    ) -> napi::Result<()> {
        let timeout = std::time::Duration::from_millis(timeout_ms as u64);
        self.inner
            .wait_for_processing(episode_id, timeout)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory waitForProcessing failed: {e}")))
    }

    /// Block until all episodes in `batchId` reach a terminal status.
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

    /// Cancel an in-flight Phase 2 or Phase 3 run for `commitId`.
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
            dense_embedded: None,
            embedding_failures: Vec::new(),
        };

        let outcome = self
            .inner
            .cancel(&commit)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory cancel failed: {e}")))?;

        Ok(convert::cancel_outcome_to_js(outcome))
    }

}

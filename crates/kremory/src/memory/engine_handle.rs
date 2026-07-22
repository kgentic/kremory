//! `EngineGraphHandle` — production `GraphHandle` implementation backed by
//! `Engine<ArcChatProvider, ArcEmbedder>`.
//!
//! ## Architecture
//!
//! This adapter sits between the kremory facade layer and the kremory-core
//! substrate. It implements all 11 `GraphHandle` methods by delegating to
//! `Engine<…>` and `TemporalGraph`, with in-memory DashMap registries for
//! run-tracking state (ingest run IDs, abort handles, dream placeholders).
//!
//! ## Dream paths (v0.1.0)
//!
//! All three dream methods (`graph_submit_dream`, `graph_dream_status`,
//! `graph_run_consolidation`) return `Err(MemoryError::NotImplemented)`
//! uniformly per F-01 LOCKED + ADR-007 §3. Real dream execution ships in
//! v0.1.1. The DashMap registries for dream state are allocated but remain
//! empty at runtime.
//!
//! ## Orphan-rule constraint
//!
//! `Engine<L, Emb>` requires `L: ChatProvider`. `Arc<dyn ChatProvider>` does
//! not satisfy this without an orphan-violating blanket impl. Phase 1 created
//! `ArcChatProvider` — a newtype wrapper around `Arc<dyn ChatProvider + Send +
//! Sync>` — to resolve this. All code here uses `ArcChatProvider` accordingly.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use uuid::Uuid;

use crate::core::config::PipelineConfig;
use crate::core::error::IngestStatus;
use crate::core::ingest::{
    AssertEntityTypeParams, Engine, EngineNewParams, PrePinnedFact, SourceParams,
};
use crate::core::provider::{ArcChatProvider, ArcEmbedder};
use crate::core::schema::TemporalGraph;
use crate::memory::{
    graph::{
        GraphAssertEntityTypeParams, GraphHandle, GraphIngestEpisodeParams, GraphSearchParams,
        GraphSubmitDreamParams,
    },
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamPhaseResult, DreamStatus,
        EpisodeCommit, MemoryError, Namespace, Result, RetrievedContext, RetrievedFact, SourceKind,
        SourceRef,
    },
    ChatProvider,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Map a `Namespace` to the `group_id` string used as the SQL `group_id` column.
///
/// Namespace `{ namespace: "ws-1", thread: Some("t-a") }` → `"ws-1:t-a"`.
/// Namespace `{ namespace: "ws-1", thread: None }` → `"ws-1"`.
///
/// Callers that want to filter by namespace only (ignoring thread) must use the
/// namespace string directly; this helper returns the fully-qualified key used
/// for isolation within a single thread.
///
/// Visibility: `pub(crate)` (Vera cycle-1 MED-4 — promoted in v0.1.4 for the
/// ADR-029a `register_namespace` facade method). External callers MUST NOT
/// depend on the `namespace:thread` string layout — this is substrate detail.
pub(crate) fn namespace_to_group_id(ns: &Namespace) -> String {
    match &ns.thread {
        Some(t) => format!("{}:{}", ns.namespace, t),
        None => ns.namespace.clone(),
    }
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Production `GraphHandle` backed by `Engine<ArcChatProvider, ArcEmbedder>`.
///
/// Constructed by `MemoryBuilder::open_graph` (Phase E.3) and stored inside
/// `Memory` behind an `Arc<dyn GraphHandle>`. The Arc-wrapping happens at the
/// facade boundary — `EngineGraphHandle` itself does not carry an outer `Arc`.
///
/// All DashMap registries are `Arc`-wrapped so a future `Clone` impl (if
/// needed) can be added without deep-copying state.
pub struct EngineGraphHandle {
    /// The core intelligence engine wrapping TemporalGraph.
    pub(crate) engine: Arc<Engine<ArcChatProvider, ArcEmbedder>>,
    /// Maps ingest `run_id` → current `IngestStatus`.
    pub(crate) ingest_runs: Arc<DashMap<Uuid, IngestStatus>>,
    /// Maps ingest `run_id` → `AbortHandle` for the background tokio task.
    pub(crate) ingest_abort: Arc<DashMap<Uuid, tokio::task::AbortHandle>>,
    /// Maps dream `run_id` → `DreamStatus`. Unpopulated at v0.1.0 (dream = NotImplemented).
    pub(crate) dream_runs: Arc<DashMap<Uuid, DreamStatus>>,
    /// Maps dream `run_id` → `AbortHandle`. Unpopulated at v0.1.0.
    pub(crate) dream_abort: Arc<DashMap<Uuid, tokio::task::AbortHandle>>,
    /// Maps `batch_id` string → `BatchStatus` aggregate.
    pub(crate) batch_status: Arc<DashMap<String, BatchStatus>>,
    /// Maps `group_id` string → timestamp of last successful consolidation.
    /// Updated by `graph_run_consolidation` (which returns NotImplemented at v0.1.0
    /// — this registry remains empty at runtime until v0.1.1).
    pub(crate) last_consolidated: Arc<DashMap<String, DateTime<Utc>>>,
}

/// Bundled parameters for [`EngineGraphHandle::with_config`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments).
pub struct WithConfigParams {
    pub graph: Arc<TemporalGraph>,
    pub chat: Arc<dyn ChatProvider + Send + Sync>,
    pub embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider>,
    pub config: PipelineConfig,
}

impl EngineGraphHandle {
    /// Create a new `EngineGraphHandle` from an already-constructed `Engine`.
    ///
    /// All DashMap registries are initialised empty. The engine is wrapped in
    /// an `Arc` so background tokio tasks spawned by `graph_ingest_episode`
    /// can hold a cheap clone of the reference without owning the engine.
    pub fn new(engine: Engine<ArcChatProvider, ArcEmbedder>) -> Self {
        Self {
            engine: Arc::new(engine),
            ingest_runs: Arc::new(DashMap::new()),
            ingest_abort: Arc::new(DashMap::new()),
            dream_runs: Arc::new(DashMap::new()),
            dream_abort: Arc::new(DashMap::new()),
            batch_status: Arc::new(DashMap::new()),
            last_consolidated: Arc::new(DashMap::new()),
        }
    }

    /// Convenience: create with a custom `PipelineConfig` (useful for tests that
    /// need to tune extraction limits without spinning up a full `MemoryBuilder`).
    pub fn with_config(params: WithConfigParams) -> Self {
        let WithConfigParams {
            graph,
            chat,
            embedder,
            config,
        } = params;
        let engine = Engine::new(EngineNewParams {
            graph,
            llm: Arc::new(ArcChatProvider::new(chat)),
            embedder: Arc::new(ArcEmbedder(embedder)),
            config,
            // Test-convenience constructor — no model id (Option-1).
            model: None,
        });
        Self::new(engine)
    }
}

// ---------------------------------------------------------------------------
// GraphHandle impl
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphHandle for EngineGraphHandle {
    // ── search config accessor (recall-improvement-e2e-spec-2026-07-22 §S0-infra) ──
    //
    // Expose the Engine's LIVE `SearchConfig` so the facade recall path
    // (`facade::recall::fuse_content_stream`) reads the configured
    // `content_stream_weight`/`rrf_k` — including the `KREMORY_CONTENT_WEIGHT`
    // / `KREMORY_RRF_K` boot overrides applied at construction
    // (`facade::providers::search_env_overrides`) — instead of the previous
    // `SearchConfig::default()` hardcode.
    fn search_config(&self) -> crate::core::config::SearchConfig {
        self.engine.config.search.clone()
    }

    // ── 1. graph_ingest_episode ──────────────────────────────────────────────

    async fn graph_ingest_episode(
        &self,
        params: GraphIngestEpisodeParams<'_>,
    ) -> Result<EpisodeCommit> {
        let GraphIngestEpisodeParams {
            namespace,
            source_ref,
            content,
            structured_facts,
            provider: _,
            batch_id,
            opts,
            sink,
        } = params;
        let group_id = namespace_to_group_id(namespace);
        let reference_time = Some(source_ref.occurred_at);

        // ── ADR-052 Gap 1: coerce the memory-layer EnrichmentEventSink to the
        // core-layer IngestEventSink supertrait so the unified extraction routine
        // (Engine::ingest → ingest_with) can fire the per-entity / per-edge /
        // stage-transition callbacks. The fb85ba8 consolidation dropped this
        // coercion, leaving `remember().with_event_sink()` a silent no-op on the
        // INLINE path (golden_path_smoke). Both the inline and background sub-paths
        // below set this on `SourceParams.sink`. Trait upcast `Arc<dyn Sub>` →
        // `Arc<dyn Super>` is stable on the project MSRV (Rust 1.86+).
        let core_sink: Option<Arc<dyn crate::core::sink::IngestEventSink>> = sink
            .as_ref()
            .map(|s| Arc::clone(s) as Arc<dyn crate::core::sink::IngestEventSink>);

        // ADR-035 §5 Option A: translate caller's StructuredFact (memory layer)
        // into PrePinnedFact (core layer) so engine.ingest can pin them
        // BEFORE Phase 2 LLM runs. Translation handles None valid_from by
        // falling back to source_ref's published_at, then to occurred_at
        // (caller can override via `published_at()` builder). Object goes
        // to object_value (literal); object_id resolution is the engine's
        // job during Phase 2 if applicable. Confidence defaults to 1.0
        // (StructuredFact does not carry a confidence field). `valid_to`
        // passes through as-is (`None` = open-ended, matching StructuredFact's
        // own doc comment) — ADR-068 boy-scout: this used to be silently
        // dropped (see `PrePinnedFact::valid_to`'s doc comment).
        let pre_pinned_facts: Vec<PrePinnedFact> = structured_facts
            .iter()
            .map(|sf| PrePinnedFact {
                subject: sf.subject.clone(),
                predicate: sf.predicate.clone(),
                object_id: None,
                object_value: Some(sf.object.clone()),
                valid_from: sf
                    .valid_from
                    .or(source_ref.published_at)
                    .unwrap_or(source_ref.occurred_at),
                valid_to: sf.valid_to,
                confidence: 1.0,
            })
            .collect();

        // Phase 1 + optional Phase 2 inline (enrich_per_episode = true, run_in_background = false)
        if opts.enrich_per_episode && opts.run_in_background {
            // Background path: spawn Phase 2, return run_id immediately.
            let run_id = Uuid::new_v4();
            let engine = Arc::clone(&self.engine);
            let ingest_runs = Arc::clone(&self.ingest_runs);
            let ingest_abort = Arc::clone(&self.ingest_abort);
            let batch_status = Arc::clone(&self.batch_status);
            let content_owned = content.to_owned();
            let group_id_owned = group_id.clone();
            let batch_id_owned = batch_id.clone();
            // Clone source provenance for the spawned task (source_ref is borrowed from caller).
            let source_id_owned = source_ref.id.clone();
            // SourceRef does not carry source_uri; consumers set it post-ingest
            // via Memory::update_source_uri() builder (facade/mod.rs G2).
            let source_uri_owned: Option<String> = None;
            let recorded_at_owned = Some(source_ref.occurred_at);

            // Record pending status before spawning so callers can poll immediately.
            ingest_runs.insert(run_id, IngestStatus::Pending);

            // Initialise / update batch counter if batch_id is provided.
            if let Some(ref bid) = batch_id_owned {
                batch_status
                    .entry(bid.clone())
                    .and_modify(|s| s.total += 1)
                    .or_insert(BatchStatus {
                        total: 1,
                        completed: 0,
                        skipped: 0,
                        failed: 0,
                    });
            }

            let pre_pinned_facts_owned = pre_pinned_facts.clone();
            let skip_extraction_owned = !opts.enrich_per_episode;
            // ADR-052 Gap 1: move the coerced core-layer sink into the spawned
            // background task so the deferred Engine::ingest fires the callbacks.
            let core_sink_owned = core_sink.clone();
            let task = tokio::task::spawn(async move {
                ingest_runs.insert(run_id, IngestStatus::Extracting);
                let sp = SourceParams {
                    source_id: Some(source_id_owned),
                    source_uri: source_uri_owned,
                    recorded_at: recorded_at_owned,
                    entity_types_override: None,
                    pre_pinned_facts: pre_pinned_facts_owned,
                    skip_extraction: skip_extraction_owned,
                    sink: core_sink_owned,
                };
                match engine
                    .ingest(crate::core::ingest::IngestParams {
                        text: &content_owned,
                        reference_time,
                        group_id: Some(&group_id_owned),
                        content_type: None,
                        source_params: sp,
                    })
                    .await
                {
                    Ok(_) => {
                        ingest_runs.insert(run_id, IngestStatus::Complete);
                        if let Some(ref bid) = batch_id_owned {
                            batch_status.entry(bid.clone()).and_modify(|s| {
                                s.completed += 1;
                            });
                        }
                    }
                    Err(e) => {
                        ingest_runs.insert(run_id, IngestStatus::Failed(e.to_string()));
                        if let Some(ref bid) = batch_id_owned {
                            batch_status.entry(bid.clone()).and_modify(|s| {
                                s.failed += 1;
                            });
                        }
                    }
                }
                // Remove abort handle once the task has reached a terminal state.
                ingest_abort.remove(&run_id);
            });

            // Capture the abort handle immediately after spawn (before any yield point)
            // and insert into the map. This eliminates the race where a fast-completing
            // task calls `ingest_abort.remove(&run_id)` before this insert fires, which
            // would leave a permanently stale entry in the map.
            let abort_handle = task.abort_handle();
            self.ingest_abort.insert(run_id, abort_handle);

            // Phase 1 episode_entity_id: we use the run_id string as a stable handle.
            // The actual episode row ID is available via graph_ingest_status polling.
            return Ok(EpisodeCommit {
                run_id: Some(run_id),
                episode_entity_id: run_id.to_string(),
                committed_at: Utc::now(),
                stub_entities_inserted: 0,
            });
        }

        // Inline path (run_in_background = false):
        //   - enrich_per_episode = true  → call engine.ingest() (full pipeline)
        //   - enrich_per_episode = false → call engine.ingest() (Phase 1 only via same API;
        //     the engine always does its pipeline — there is no Phase-1-only variant at this
        //     substrate level; skipped enrichment is tracked via batch_status as "skipped").
        //
        // Quinn Phase 4 MED-03 cause-fix (ADR-051 Phase 5) + Phase 5 M-1 doc-correction:
        //
        // On inline ingest error, we CANNOT write `Failed` because the episode_id is
        // unavailable in this scope. Two failure shapes both lose it:
        //   1. Failure BEFORE the episode INSERT (embed fail, early validation) → no
        //      episode row exists; nothing to mark Failed.
        //   2. Failure DURING extraction AFTER the INSERT → the INSERT did happen and
        //      committed a Pending row, but `Engine::ingest`'s Err variant does NOT
        //      carry the partial episode_id back. We've lost the handle.
        //
        // Path asymmetry with the BackgroundIngestor: `verify_stage.rs::run_verify_stage`
        // takes `request.episode_id` as a function parameter (the caller `process_deferred`
        // already holds it post-Phase-1-INSERT), so its Failed-on-error path is intact.
        // The inline path doesn't have that handle.
        //
        // The honest response: emit `kremory.engine_handle.inline_ingest_fail_no_episode_id_total`
        // + `tracing::error!` so the gap is visible per CLAUDE.md Rule 19, then propagate the
        // original error. Episodes that committed a Pending row but failed extraction will
        // stay at Pending forever on this path — accept it as a known limitation, tracked
        // for a future fix that threads `Result<(IngestResult, Option<i64>), Error>` or
        // similar through Engine::ingest's error variant.
        let ingest_result = match self
            .engine
            .ingest(crate::core::ingest::IngestParams {
                text: content,
                reference_time,
                group_id: Some(&group_id),
                content_type: None,
                source_params: SourceParams {
                    source_id: Some(source_ref.id.clone()),
                    source_uri: None,
                    recorded_at: Some(source_ref.occurred_at),
                    entity_types_override: None,
                    pre_pinned_facts,
                    skip_extraction: !opts.enrich_per_episode,
                    // ADR-052 Gap 1: wire the coerced sink onto the INLINE path —
                    // this is the path `Memory::remember(..).with_event_sink(..).await`
                    // (no .no_wait()) drives, exercised by golden_path_smoke. The
                    // background sub-path above returns before this point, so
                    // `core_sink` is still owned here (last use).
                    sink: core_sink,
                },
            })
            .await
        {
            Ok(r) => r,
            Err(e) => {
                // The error may have occurred before the episode row was INSERTed (e.g.
                // embed failure). We do not have access to the episode_id at this level
                // when the INSERT did not fire. Emit a counter for observability so the
                // "no-episode-id" branch is distinguishable from a post-INSERT failure
                // that would carry an episode_id from a different code path.
                metrics::counter!("kremory.engine_handle.inline_ingest_fail_no_episode_id_total")
                    .increment(1);
                tracing::error!(
                    target: "kremory.engine_handle",
                    error = %e,
                    "inline ingest failed; episode_id unavailable at this level — \
                     episode row (if INSERTed) remains at Pending status. \
                     BackgroundIngestor path writes Failed; inline path cannot without episode_id."
                );
                return Err(MemoryError::Core(e));
            }
        };

        // Increment batch counter for inline path.
        if let Some(ref bid) = batch_id {
            if opts.enrich_per_episode {
                batch_status_increment_completed(&self.batch_status, bid);
            } else {
                batch_status_increment_skipped(&self.batch_status, bid);
            }
        }

        // ADR-051 Phase 4: inline ingest completed successfully → mark episode
        // as Verified so `Memory::wait_for_processing` callers do not busy-poll
        // waiting for a background worker that never fires on the inline path.
        //
        // Best-effort: a status write failure is non-fatal for the ingest itself
        // — we log it and continue. The episode data is fully committed; only the
        // status column is affected.
        //
        // Quinn Phase 4 MED-02 cause-fix (ADR-051 Phase 5):
        // Gate the Verified write on `enrich_per_episode`. When skip_extraction=true
        // (enrich_per_episode=false), extraction was intentionally skipped — writing
        // Verified would be semantically wrong ("extraction was verified" when no
        // extraction ran). Leave status at Pending for the skip_extraction path.
        // The episode is durably stored; future background workers that filter for
        // Pending episodes will not attempt to re-extract (the episode was never
        // enqueued to the background worker), which is correct behavior.
        let episode_id_for_status = ingest_result.episode_id;
        if opts.enrich_per_episode {
            if let Err(e) = self
                .engine
                .graph()
                .conn
                .execute(
                    "UPDATE episodes SET episode_processing_status = 'Verified' WHERE id = ?1",
                    libsql::params![episode_id_for_status],
                )
                .await
            {
                tracing::warn!(
                    target: "kremory.engine_handle",
                    episode_id = episode_id_for_status,
                    error = %e,
                    "inline ingest: failed to write Verified status — \
                     wait_for_processing may stall for this episode"
                );
            }
        }

        Ok(EpisodeCommit {
            run_id: None,
            episode_entity_id: ingest_result.episode_id.to_string(),
            committed_at: Utc::now(),
            stub_entities_inserted: ingest_result.stub_entities_inserted,
        })
    }

    // ── 2. graph_ingest_status ───────────────────────────────────────────────

    async fn graph_ingest_status(&self, run_id: Uuid) -> Result<IngestStatus> {
        match self.ingest_runs.get(&run_id) {
            Some(status) => Ok(status.clone()),
            None => Ok(IngestStatus::Complete),
            // Not-found is treated as "completed and evicted" — callers that
            // poll after task completion + eviction get Complete, not an error.
            // This matches Hatchet / Temporal prior art: no registry entry =
            // terminal success (non-failure absence).
        }
    }

    // ── 3. graph_cancel ──────────────────────────────────────────────────────

    async fn graph_cancel(&self, run_id: Uuid) -> Result<CancelOutcome> {
        // Abort background ingest task if still running.
        if let Some((_, handle)) = self.ingest_abort.remove(&run_id) {
            handle.abort();
            self.ingest_runs
                .insert(run_id, IngestStatus::Failed("cancelled by caller".into()));
            return Ok(CancelOutcome {
                cancelled_phase: CancelledPhase::Enrichment,
                rolled_back: false,
                partial: vec![],
            });
        }

        // Check dream runs (no-op at v0.1.0 — registry is always empty).
        if let Some((_, handle)) = self.dream_abort.remove(&run_id) {
            handle.abort();
            self.dream_runs
                .insert(run_id, DreamStatus::Failed("cancelled by caller".into()));
            return Ok(CancelOutcome {
                cancelled_phase: CancelledPhase::Consolidation,
                rolled_back: false,
                partial: vec![],
            });
        }

        // run_id not found — already completed or never existed.
        // Return a no-op success (idempotent cancel).
        Ok(CancelOutcome {
            cancelled_phase: CancelledPhase::Enrichment,
            rolled_back: false,
            partial: vec![],
        })
    }

    // ── 4. graph_submit_dream — NotImplemented (F-01 LOCKED) ────────────────

    async fn graph_submit_dream(&self, _params: GraphSubmitDreamParams<'_>) -> Result<DreamHandle> {
        Err(MemoryError::NotImplemented {
            feature: "dream-submit",
            available_in: "v0.1.1",
            adr_ref: "ADR-007 §3",
        })
    }

    // ── 5. graph_dream_status — NotImplemented (F-01 LOCKED) ────────────────

    async fn graph_dream_status(&self, _run_id: Uuid) -> Result<DreamStatus> {
        Err(MemoryError::NotImplemented {
            feature: "dream-status",
            available_in: "v0.1.1",
            adr_ref: "ADR-007 §3",
        })
    }

    // ── 6. graph_batch_status ────────────────────────────────────────────────

    async fn graph_batch_status(&self, batch_id: &str) -> Result<BatchStatus> {
        match self.batch_status.get(batch_id) {
            Some(status) => Ok(status.clone()),
            None => Ok(BatchStatus {
                total: 0,
                completed: 0,
                skipped: 0,
                failed: 0,
            }),
        }
    }

    // ── 7. graph_last_consolidated_at ────────────────────────────────────────

    async fn graph_last_consolidated_at(
        &self,
        namespace: &Namespace,
    ) -> Result<Option<DateTime<Utc>>> {
        let group_id = namespace_to_group_id(namespace);
        Ok(self.last_consolidated.get(&group_id).map(|v| *v))
    }

    // ── 8. graph_episodes_since_last_dream ───────────────────────────────────

    async fn graph_episodes_since_last_dream(&self, namespace: &Namespace) -> Result<usize> {
        let group_id = namespace_to_group_id(namespace);
        // Use last_consolidated timestamp if present; otherwise count all episodes
        // in this namespace (since UNIX_EPOCH = all time).
        let since = self
            .last_consolidated
            .get(&group_id)
            .map(|v| *v)
            .unwrap_or(DateTime::UNIX_EPOCH);

        self.engine
            .graph()
            .count_episodes_since(&group_id, since)
            .await
            .map_err(MemoryError::Core)
    }

    // ── 9. graph_is_consolidating ────────────────────────────────────────────

    async fn graph_is_consolidating(&self, _namespace: &Namespace) -> Result<bool> {
        // At v0.1.0 dream methods return NotImplemented — no dream tasks are ever
        // spawned, so no namespace can be consolidating. Return false always.
        Ok(false)
    }

    // ── 10. graph_search ─────────────────────────────────────────────────────

    async fn graph_search(&self, params: GraphSearchParams<'_>) -> Result<Vec<RetrievedContext>> {
        let GraphSearchParams {
            namespace,
            query,
            opts,
        } = params;
        let group_id = namespace_to_group_id(namespace);
        let limit = opts.limit;
        let as_of = opts.as_of;

        // ADR-068 NFR — "how often is as_of actually used" adoption counter
        // for a brand-new, previously-erroring surface (observability-first-
        // class: a new capability needs a usage signal from day one, mirrors
        // recall-v2's `intent_classified_total` reasoning for a similarly-new
        // signal).
        if as_of.is_some() {
            metrics::counter!(
                "kremory.recall.as_of_used_total",
                "namespace" => namespace.namespace.clone()
            )
            .increment(1);
        }

        let context = self
            .engine
            .contextualize(crate::core::context::ContextualizeParams {
                query,
                group_id: Some(&group_id),
                limit,
                as_of,
            })
            .await
            .map_err(MemoryError::Core)?;

        // ADR-068: `context.facts` above is ALREADY `as_of`-filtered at its
        // one true source (`TemporalGraph::get_neighbours_at`, called from
        // `contextualize()` for every seed). The per-entity projection below
        // (G5) only re-shapes already-filtered facts into `RetrievedFact` —
        // it does not need (and must not add) a second temporal filter here.
        // A single enforcement site keeps `as_of`'s semantics from drifting
        // between two copies of the same predicate (treat-cause-not-symptom:
        // filter once, at the SQL layer that owns the temporal columns).
        //
        // Rule 19 / ADR-074 review H1: observe the fact→entity ownership
        // projection BEFORE `context.entities` is consumed by the loop below.
        // This is the exact silent projection/filter shape whose prior version
        // dropped facts undetected in production (TD-116) — a same-shaped
        // regression must show up here, not require a re-run of that incident.
        // A fact "drops" when NEITHER its `subject_id` NOR its `object_id`
        // matches any entity in this recall's result set — e.g. the group_id
        // filter below strips a neighbour entity out of `context.entities`
        // while its connecting fact remains in `context.facts` (ADR-074
        // review M3). G5 (gap-register / ADR-074 F1): a fact is now "attached"
        // whenever EITHER endpoint survives into the result set — matching the
        // per-entity projection fix below, which attaches a fact to an entity
        // whether that entity is the fact's subject OR its object. Computed
        // once, globally, so a fact is counted attached at most once (never
        // double-counted across the per-entity loop; Rule 19 anti-pattern #9,
        // "counters that lie").
        let result_entity_ids: std::collections::HashSet<&str> =
            context.entities.iter().map(|e| e.id.as_str()).collect();
        let facts_candidates = context.facts.len();
        let facts_attached = context
            .facts
            .iter()
            .filter(|f| {
                result_entity_ids.contains(f.subject_id.as_str())
                    || f.object_id
                        .as_deref()
                        .is_some_and(|oid| result_entity_ids.contains(oid))
            })
            .count();
        let facts_dropped_ownership = facts_candidates.saturating_sub(facts_attached);
        metrics::counter!("kremory.recall.facts_attached_total").increment(facts_attached as u64);
        metrics::counter!("kremory.recall.facts_dropped_ownership_total")
            .increment(facts_dropped_ownership as u64);
        tracing::debug!(
            facts_candidates,
            facts_attached,
            facts_dropped_ownership,
            "kremory.recall.facts_attached"
        );

        // Map ContextResult (Entity + Fact) → Vec<RetrievedContext>.
        // Each entity becomes one RetrievedContext. source_refs are derived
        // from episodic_edges (Bug A fix: v0.1.1 authoritative path).
        let mut results: Vec<RetrievedContext> = Vec::with_capacity(context.entities.len());
        for entity in context.entities {
            // Score: RRF-derived normalised score from ContextResult.scores.
            // Seed entities have a score in [0.0, 1.0]; 1-hop expansion neighbours
            // that were not in the original seed set default to 0.0 (v0.1.1 policy).
            let score = context.scores.get(&entity.id).copied().unwrap_or(0.0);

            // Bug A fix: query episodic_edges directly — authoritative source for
            // episode attribution. Fact-row IDs (v0.1.0 broken path) are replaced by
            // episodic_edge.episode_id values with SourceKind::Episode.
            let edges = self
                .engine
                .graph
                .episodic_edges_for_entity(&entity.id)
                .await
                .map_err(MemoryError::Core)?;
            let source_refs: Vec<SourceRef> = edges
                .iter()
                .map(|e| SourceRef {
                    kind: SourceKind::Episode,
                    id: e.episode_id.to_string(),
                    occurred_at: e.recorded_at,
                    published_at: None,
                })
                .collect();

            // Bug B: read `properties["context"]` first (verbatim source snippet
            // stored at extract time), fall back to `properties["text"]` (legacy
            // episode-as-entity path), then fall back to entity.label (last resort).
            let summary = entity
                .properties
                .get("context")
                .or_else(|| entity.properties.get("text"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned())
                .unwrap_or_else(|| entity.label.clone());

            // Use properties["name"] if present (original case, e.g. "Alice"),
            // fall back to entity.id (normalized, e.g. "alice").
            // entity.label is the type label ("Person") — NOT the entity name.
            let entity_name = entity
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned())
                .unwrap_or_else(|| entity.id.clone());

            // incomplete: true when this entity is a stub placeholder.
            let incomplete = entity
                .properties
                .get("stub")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // entity.label is the type label resolved via SQL COALESCE(et.name, 'Entity')
            // at query time. entity.entity_type_id is the raw integer id.
            let entity_type_id = entity.entity_type_id;
            let entity_type_name = entity.label.clone();

            // ADR-074 / TD-116 / G5 (gap-register F1): surface the entity's
            // connected facts — BOTH subject-owned and object-owned, deduped by
            // construction (`context.facts.iter()` visits each `Fact` row once;
            // the OR below just decides whether THIS entity's projection keeps
            // it, so a self-referential fact where the entity is both subject
            // and object still appears exactly once) — so recall returns the
            // actual knowledge from every entity's own point of view, not just
            // the subject's. Pre-G5 this filter was `f.subject_id == entity.id`
            // only: an object entity (e.g. "Acme" in "Alice works_at Acme") got
            // an empty `facts` list even though `contextualize`/`get_neighbours`
            // had already collected both the entity and its connecting fact.
            // The facts are already computed by `contextualize` (`context.facts`);
            // here each is projected to the LLM-facing `RetrievedFact` shape
            // (natural-language string + structured triple + BOTH bi-temporal
            // clocks + confidence + provenance).
            let facts: Vec<RetrievedFact> = context
                .facts
                .iter()
                .filter(|f| {
                    f.subject_id == entity.id || f.object_id.as_deref() == Some(entity.id.as_str())
                })
                .map(|f| {
                    let object_is_entity = f.object_id.is_some();
                    // The filter above guarantees at least one side matches;
                    // when the subject side does NOT match, this entity must be
                    // the object side (self-referential facts, where both sides
                    // match, keep the existing subject-perspective rendering).
                    let entity_is_object_only = f.subject_id != entity.id;
                    let (subject, object) = if entity_is_object_only {
                        // This entity IS the fact's object: render its own
                        // (known) display name as `object`; the subject side
                        // falls back to the raw id — the same deferred F2
                        // display-name-resolution limitation noted below for
                        // the literal/object-entity id fallback, just mirrored
                        // onto the subject side for this perspective.
                        (f.subject_id.clone(), entity_name.clone())
                    } else {
                        // Prefer the literal value; fall back to the object entity id.
                        // (F2 endpoint display-name resolution for object entities is a
                        // deferred enhancement — literal objects render cleanly today.)
                        let object = f
                            .object_value
                            .clone()
                            .or_else(|| f.object_id.clone())
                            .unwrap_or_default();
                        (entity_name.clone(), object)
                    };
                    let fact = format!("{subject} {} {object}", f.predicate);
                    RetrievedFact {
                        fact,
                        subject,
                        predicate: f.predicate.clone(),
                        object,
                        object_is_entity,
                        valid_at: f.valid_from,
                        // NOT `f.invalid_at` (schema.rs:126 — the contradiction-resolver's
                        // invalidation timestamp, an explicitly DISTINCT concept from
                        // `valid_to`). The wire-facing "world clock" invalid_at IS
                        // `Fact.valid_to`; a future "fix" to read `f.invalid_at` here
                        // would silently break the supersession signal on every
                        // `RetrievedFact` (ADR-074 review M4).
                        invalid_at: f.valid_to,
                        recorded_at: f.recorded_at,
                        expired_at: f.expired_at,
                        confidence: f.confidence,
                        source_episode_ids: f.source_episode_id.into_iter().collect(),
                        score,
                    }
                })
                .collect();

            results.push(RetrievedContext {
                entity_id: entity.id,
                entity_name,
                summary,
                score,
                source_refs,
                incomplete,
                entity_type_id,
                entity_type_name,
                namespace: Some(namespace.clone()),
                facts,
            });
        }

        Ok(results)
    }

    // ── 11. graph_run_consolidation — NotImplemented (F-01 LOCKED) ──────────

    async fn graph_run_consolidation(
        &self,
        _namespace: &Namespace,
        _provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult> {
        Err(MemoryError::NotImplemented {
            feature: "dream-consolidation-sync",
            available_in: "v0.1.1",
            adr_ref: "ADR-007 §3",
        })
    }

    // ── 12. graph_run_dream_pass_sync (Phase C DoD C1) ───────────────────────

    async fn graph_run_dream_pass_sync(
        &self,
        opts: crate::core::ingest::DreamPassOpts,
    ) -> Result<crate::facade::DreamSummary> {
        let summary = self
            .engine
            .run_dream_pass_sync(opts)
            .await
            .map_err(MemoryError::Core)?;
        Ok(crate::facade::DreamSummary::from(
            crate::core::ingest::DreamPassSummary {
                ghost_episodes_retried: summary.ghost_episodes_retried,
                types_discovered: summary.types_discovered,
                entities_reclassified: summary.entities_reclassified,
                duration_ms: summary.duration_ms,
            },
        ))
    }

    // ── 13. graph_ghost_episodes (Phase C DoD C4) ────────────────────────────

    async fn graph_ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        self.engine
            .ghost_episodes(group_id)
            .await
            .map_err(MemoryError::Core)
    }

    // ── 14. graph_assert_entity_type (Phase C DoD C5) ────────────────────────

    async fn graph_assert_entity_type(
        &self,
        params: GraphAssertEntityTypeParams<'_>,
    ) -> Result<()> {
        let GraphAssertEntityTypeParams {
            entity_id,
            entity_type_id,
            group_id,
        } = params;
        self.engine
            .assert_entity_type(AssertEntityTypeParams {
                entity_id,
                entity_type_id,
                group_id,
            })
            .await
            .map_err(MemoryError::Core)
    }
}

// ---------------------------------------------------------------------------
// Batch status helpers (private)
// ---------------------------------------------------------------------------

fn batch_status_increment_completed(map: &DashMap<String, BatchStatus>, batch_id: &str) {
    map.entry(batch_id.to_owned())
        .and_modify(|s| s.completed += 1)
        .or_insert(BatchStatus {
            total: 1,
            completed: 1,
            skipped: 0,
            failed: 0,
        });
}

fn batch_status_increment_skipped(map: &DashMap<String, BatchStatus>, batch_id: &str) {
    map.entry(batch_id.to_owned())
        .and_modify(|s| s.skipped += 1)
        .or_insert(BatchStatus {
            total: 1,
            completed: 0,
            skipped: 1,
            failed: 0,
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        provider::{MockChatProvider, NullEmbeddingProvider},
        schema::TemporalGraph,
    };
    use crate::memory::types::DreamOpts;

    /// Build a test `EngineGraphHandle` wired to an in-memory libSQL database.
    async fn make_handle() -> EngineGraphHandle {
        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("in-memory graph"),
        );
        let chat: Arc<dyn ChatProvider + Send + Sync> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });

        let config = PipelineConfig::builder()
            .build()
            .expect("default PipelineConfig");
        let engine = Engine::new(EngineNewParams {
            graph,
            llm: Arc::new(ArcChatProvider::new(chat)),
            embedder: Arc::new(ArcEmbedder(embedder)),
            config,
            model: None,
        });
        EngineGraphHandle::new(engine)
    }

    /// F-01 gate: all three dream paths return NotImplemented — NOT panic, NOT Ok.
    #[tokio::test]
    async fn dream_returns_not_implemented() {
        let handle = make_handle().await;
        let ns = Namespace::new("test-ns");
        let provider: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());

        // graph_run_consolidation
        let result = handle
            .graph_run_consolidation(&ns, Arc::clone(&provider))
            .await;
        match result {
            Err(MemoryError::NotImplemented {
                feature,
                available_in,
                adr_ref,
            }) => {
                assert_eq!(available_in, "v0.1.1");
                assert_eq!(adr_ref, "ADR-007 §3");
                assert!(!feature.is_empty(), "feature must be non-empty");
            }
            other => panic!(
                "graph_run_consolidation: expected NotImplemented, got: {:?}",
                other
            ),
        }

        // graph_submit_dream
        let result = handle
            .graph_submit_dream(GraphSubmitDreamParams {
                namespace: &ns,
                provider: Arc::clone(&provider),
                batch_id: None,
                opts: DreamOpts::default(),
                sink: None,
            })
            .await;
        match result {
            Err(MemoryError::NotImplemented {
                available_in,
                adr_ref,
                feature,
            }) => {
                assert_eq!(available_in, "v0.1.1");
                assert_eq!(adr_ref, "ADR-007 §3");
                assert!(!feature.is_empty());
            }
            other => panic!(
                "graph_submit_dream: expected NotImplemented, got: {:?}",
                other
            ),
        }

        // graph_dream_status
        let result = handle.graph_dream_status(Uuid::new_v4()).await;
        match result {
            Err(MemoryError::NotImplemented {
                available_in,
                adr_ref,
                feature,
            }) => {
                assert_eq!(available_in, "v0.1.1");
                assert_eq!(adr_ref, "ADR-007 §3");
                assert!(!feature.is_empty());
            }
            other => panic!(
                "graph_dream_status: expected NotImplemented, got: {:?}",
                other
            ),
        }
    }

    /// Batch status starts empty and accumulates correctly.
    #[tokio::test]
    async fn batch_status_accumulates() {
        let handle = make_handle().await;

        // Nothing stored yet.
        let status = handle
            .graph_batch_status("batch-1")
            .await
            .expect("batch_status ok");
        assert_eq!(status.total, 0);
        assert!(status.is_done(), "empty batch is done");

        // Manually write a value to test the DashMap path.
        handle.batch_status.insert(
            "batch-1".to_owned(),
            BatchStatus {
                total: 3,
                completed: 2,
                skipped: 0,
                failed: 1,
            },
        );
        let status = handle
            .graph_batch_status("batch-1")
            .await
            .expect("batch_status ok after insert");
        assert_eq!(status.total, 3);
        assert!(status.is_done(), "2+0+1 == 3 → done");
    }

    /// `graph_last_consolidated_at` returns None when no consolidation has run.
    #[tokio::test]
    async fn last_consolidated_at_is_none_initially() {
        let handle = make_handle().await;
        let ns = Namespace::new("ws-fresh");
        let result = handle
            .graph_last_consolidated_at(&ns)
            .await
            .expect("last_consolidated_at ok");
        assert!(result.is_none(), "no consolidation has run");
    }

    /// `graph_is_consolidating` always returns false at v0.1.0.
    #[tokio::test]
    async fn is_consolidating_always_false() {
        let handle = make_handle().await;
        let ns = Namespace::new("ws-any");
        let result = handle
            .graph_is_consolidating(&ns)
            .await
            .expect("is_consolidating ok");
        assert!(!result, "v0.1.0: no dream tasks are ever running");
    }

    /// `graph_ingest_status` returns Complete for unknown run IDs (not-found = completed).
    #[tokio::test]
    async fn ingest_status_unknown_id_returns_complete() {
        let handle = make_handle().await;
        let unknown = Uuid::new_v4();
        let status = handle
            .graph_ingest_status(unknown)
            .await
            .expect("ingest_status ok");
        assert_eq!(
            status,
            IngestStatus::Complete,
            "not-found treated as completed/evicted"
        );
    }

    /// `graph_cancel` on an unknown run_id returns a no-op CancelOutcome (idempotent).
    #[tokio::test]
    async fn cancel_unknown_run_id_is_idempotent() {
        let handle = make_handle().await;
        let unknown = Uuid::new_v4();
        let outcome = handle.graph_cancel(unknown).await.expect("cancel ok");
        assert!(!outcome.rolled_back);
        assert!(outcome.partial.is_empty());
    }

    /// Namespace → group_id translation helper.
    #[test]
    fn namespace_to_group_id_with_thread() {
        let ns = Namespace::new("ws-1").with_thread("t-a");
        assert_eq!(namespace_to_group_id(&ns), "ws-1:t-a");
    }

    #[test]
    fn namespace_to_group_id_without_thread() {
        let ns = Namespace::new("ws-1");
        assert_eq!(namespace_to_group_id(&ns), "ws-1");
    }

    /// ADR-074 review H1 (Rule 19): `graph_search`'s fact→entity ownership
    /// projection must be observed — this is the exact silent-drop shape
    /// TD-116 fixed; a regression must show up as a metric delta, not require
    /// another production incident to notice.
    #[tokio::test]
    async fn graph_search_emits_facts_attached_and_dropped_counters() {
        use crate::core::graph::{FactInsert, InsertEntityParams};
        use crate::memory::types::SearchOpts;
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let handle = make_handle().await;
        let now = Utc::now();

        handle
            .engine
            .graph
            .insert_entity(InsertEntityParams {
                id: "alice",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Alice"}),
            })
            .await
            .expect("insert alice");
        handle
            .engine
            .graph
            .insert_entity(InsertEntityParams {
                id: "acme",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Acme"}),
            })
            .await
            .expect("insert acme");
        handle
            .engine
            .graph
            .insert_fact(FactInsert::new("alice", "works_at", now).object_id("acme"))
            .await
            .expect("insert fact");

        let ns = Namespace::new("default");
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        // `set_default_local_recorder` (not `with_local_recorder`) so the guard
        // can be held across the `.await` inside `graph_search` — see the
        // identical pattern + rationale in `core::context::tests`.
        let guard = metrics::set_default_local_recorder(&recorder);
        let results = handle
            .graph_search(GraphSearchParams {
                namespace: &ns,
                query: "Alice",
                opts: &SearchOpts::default(),
            })
            .await
            .expect("graph_search ok");
        drop(guard);

        assert!(!results.is_empty(), "fixture must produce a result");
        assert!(
            results.iter().any(|r| !r.facts.is_empty()),
            "alice's result must carry the works_at fact"
        );

        let sum_counter = |name: &str| -> u64 {
            snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .filter(|(k, _, _, _)| k.key().name() == name)
                .filter_map(|(_, _, _, v)| match v {
                    DebugValue::Counter(c) => Some(c),
                    _ => None,
                })
                .sum()
        };

        let attached = sum_counter("kremory.recall.facts_attached_total");
        let dropped = sum_counter("kremory.recall.facts_dropped_ownership_total");
        assert!(attached >= 1, "alice's fact must be counted attached");
        // Every fact in this fixture has its subject (alice) present in the
        // result set, so none should be dropped by the ownership predicate.
        assert_eq!(dropped, 0, "no facts should be dropped in this fixture");
    }

    /// G5 (gap-register / ADR-074 F1): `graph_search` must attach a fact to
    /// BOTH the subject's AND the object's `RetrievedContext.facts` when the
    /// entity is an *object entity* (`fact.object_id == Some(entity.id)`).
    /// Pre-fix, the per-entity projection at the bottom of `graph_search` only
    /// tested `f.subject_id == entity.id` — the object side of the very same
    /// fact was silently invisible from the object entity's own recall
    /// result, even though `contextualize`/`get_neighbours` correctly
    /// collected the fact and the object entity into the result set.
    #[tokio::test]
    async fn graph_search_attaches_facts_when_entity_is_object() {
        use crate::core::graph::{FactInsert, InsertEntityParams};
        use crate::memory::types::SearchOpts;

        let handle = make_handle().await;
        let now = Utc::now();

        handle
            .engine
            .graph
            .insert_entity(InsertEntityParams {
                id: "alice",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Alice"}),
            })
            .await
            .expect("insert alice");
        handle
            .engine
            .graph
            .insert_entity(InsertEntityParams {
                id: "acme",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Acme"}),
            })
            .await
            .expect("insert acme");
        handle
            .engine
            .graph
            .insert_fact(FactInsert::new("alice", "works_at", now).object_id("acme"))
            .await
            .expect("insert fact");

        let ns = Namespace::new("default");
        // Search on "Acme" so the OBJECT entity is the seed; 1-hop expansion
        // pulls alice (the subject) + the connecting fact into the same
        // ContextResult, per `test_context_result_includes_facts` in
        // `core::context::tests`.
        let results = handle
            .graph_search(GraphSearchParams {
                namespace: &ns,
                query: "Acme",
                opts: &SearchOpts::default(),
            })
            .await
            .expect("graph_search ok");

        assert!(!results.is_empty(), "fixture must produce a result");
        let acme_result = results
            .iter()
            .find(|r| r.entity_id == "acme")
            .expect("acme (the object entity) must appear in the result set");
        assert!(
            !acme_result.facts.is_empty(),
            "acme is the OBJECT of 'alice works_at acme' — the fact must be \
             attached to acme's own RetrievedContext, not just alice's"
        );
        assert!(
            acme_result.facts.iter().any(|f| f.predicate == "works_at"),
            "acme's facts must include the works_at fact connecting it to alice"
        );
    }
}

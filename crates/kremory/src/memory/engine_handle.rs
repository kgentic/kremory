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
use crate::core::ingest::{Engine, PrePinnedFact, SourceParams};
use crate::core::provider::{ArcChatProvider, ArcEmbedder};
use crate::core::schema::TemporalGraph;
use crate::memory::{
    events::EnrichmentEventSink,
    graph::GraphHandle,
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamOpts, DreamPhaseResult,
        DreamStatus, EpisodeCommit, MemoryError, Namespace, Result, RetrievedContext, SearchOpts,
        SourceKind, SourceRef, StructuredFact, SubmitOpts,
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
    pub fn with_config(
        graph: Arc<TemporalGraph>,
        chat: Arc<dyn ChatProvider + Send + Sync>,
        embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider>,
        config: PipelineConfig,
    ) -> Self {
        let engine = Engine::new(
            graph,
            Arc::new(ArcChatProvider::new(chat)),
            Arc::new(ArcEmbedder(embedder)),
            config,
        );
        Self::new(engine)
    }
}

// ---------------------------------------------------------------------------
// GraphHandle impl
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphHandle for EngineGraphHandle {
    // ── 1. graph_ingest_episode ──────────────────────────────────────────────

    async fn graph_ingest_episode(
        &self,
        namespace: &Namespace,
        source_ref: &SourceRef,
        content: &str,
        structured_facts: &[StructuredFact],
        _provider: Arc<dyn ChatProvider>,
        batch_id: Option<String>,
        opts: SubmitOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<EpisodeCommit> {
        let group_id = namespace_to_group_id(namespace);
        let reference_time = Some(source_ref.occurred_at);

        // ADR-035 §5 Option A: translate caller's StructuredFact (memory layer)
        // into PrePinnedFact (core layer) so engine.ingest can pin them
        // BEFORE Phase 2 LLM runs. Translation handles None valid_from by
        // falling back to source_ref's published_at, then to occurred_at
        // (caller can override via `published_at()` builder). Object goes
        // to object_value (literal); object_id resolution is the engine's
        // job during Phase 2 if applicable. Confidence defaults to 1.0
        // (StructuredFact does not carry a confidence field).
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
            let task = tokio::task::spawn(async move {
                ingest_runs.insert(run_id, IngestStatus::Extracting);
                let sp = SourceParams {
                    source_id: Some(source_id_owned),
                    source_uri: source_uri_owned,
                    recorded_at: recorded_at_owned,
                    entity_types_override: None,
                    pre_pinned_facts: pre_pinned_facts_owned,
                    skip_extraction: skip_extraction_owned,
                };
                match engine
                    .ingest(
                        &content_owned,
                        reference_time,
                        Some(&group_id_owned),
                        None,
                        sp,
                    )
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
        let ingest_result = self
            .engine
            .ingest(
                content,
                reference_time,
                Some(&group_id),
                None,
                SourceParams {
                    source_id: Some(source_ref.id.clone()),
                    source_uri: None,
                    recorded_at: Some(source_ref.occurred_at),
                    entity_types_override: None,
                    pre_pinned_facts,
                    skip_extraction: !opts.enrich_per_episode,
                },
            )
            .await
            .map_err(MemoryError::Core)?;

        // Increment batch counter for inline path.
        if let Some(ref bid) = batch_id {
            if opts.enrich_per_episode {
                batch_status_increment_completed(&self.batch_status, bid);
            } else {
                batch_status_increment_skipped(&self.batch_status, bid);
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

    async fn graph_submit_dream(
        &self,
        _namespace: &Namespace,
        _provider: Arc<dyn ChatProvider>,
        _batch_id: Option<String>,
        _opts: DreamOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<DreamHandle> {
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

    async fn graph_search(
        &self,
        namespace: &Namespace,
        query: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<RetrievedContext>> {
        let group_id = namespace_to_group_id(namespace);
        let limit = opts.limit;

        let context = self
            .engine
            .contextualize(query, Some(&group_id), limit)
            .await
            .map_err(MemoryError::Core)?;

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

    async fn graph_ghost_episodes(
        &self,
        group_id: Option<&str>,
    ) -> Result<Vec<i64>> {
        self.engine
            .ghost_episodes(group_id)
            .await
            .map_err(MemoryError::Core)
    }

    // ── 14. graph_assert_entity_type (Phase C DoD C5) ────────────────────────

    async fn graph_assert_entity_type(
        &self,
        entity_id: &str,
        entity_type_id: u32,
        group_id: Option<&str>,
    ) -> Result<()> {
        self.engine
            .assert_entity_type(entity_id, entity_type_id, group_id)
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::{
        provider::{MockChatProvider, NullEmbeddingProvider},
        schema::TemporalGraph,
    };

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
        let engine = Engine::new(
            graph,
            Arc::new(ArcChatProvider::new(chat)),
            Arc::new(ArcEmbedder(embedder)),
            config,
        );
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
            .graph_submit_dream(&ns, Arc::clone(&provider), None, DreamOpts::default(), None)
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
}

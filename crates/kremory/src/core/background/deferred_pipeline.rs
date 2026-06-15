//! Worker loop — spawn_worker, drain logic, per-job processing, error reporting.
//!
//! Sprint plan T2.1 / ADR-049 §Decision 6 — worker loop module.
//! ADR-051: GLiNER-to-background unified hot path (Phase 3 wiring).
//!
//! Contains:
//! - [`process_item`]    — Phase 1: episode INSERT + GLiNER candidates (fast path, no LLM)
//! - [`process_deferred`] — Phase 2: entity write via `run_verify_stage` + fact extraction
//! - [`worker_loop`]     — main OS-thread loop; drives both phases, respects stop flag

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use crate::memory::events::BatchPhase2Complete;

use chrono::Utc;

use crate::core::error::{IngestStatus, IngestionErrorKind};
use crate::core::ingest::{Engine, SourceParams};
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::sink::{IngestEventSink, IngestionError};
use crate::memory::events::EnrichmentEventSink;

use super::{
    batch_tracker::BatchTracker, try_send_error, DeferredRequest, IngestError, IngestErrorKind,
    IngestRequest, RateLimit, TokenBucketState,
};

// ---------------------------------------------------------------------------
// process_item — Phase 1: episode INSERT + GLiNER NER candidates (ADR-051)
// ---------------------------------------------------------------------------

/// Process one ingest request (Phase 1): INSERT episode row + run GLiNER NER.
///
/// ADR-051 Phase 3: replaces the previous `graph.ingest()` (full pipeline) call
/// with `graph.ingest_phase1_ner()` (episode INSERT + NER candidates only, no LLM,
/// no entity writes). Entity writes + fact extraction are deferred to Phase 2
/// (`process_deferred`).
///
/// `sink` is propagated from [`worker_loop`] per ADR-052 Gap 1 (impl spec §3
/// Phase 2).  Phase 3 wires the actual callsites; `sink` is accepted here so the
/// signature is stable before Phase 3 lands.
///
/// Returns `Some(DeferredRequest)` when Phase 2 should be enqueued, or `None`
/// on error (error already forwarded to `error_tx`).
pub(super) async fn process_item<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    req: IngestRequest,
    error_tx: &SyncSender<IngestError>,
    deferred_enabled: bool,
    sink: Option<&dyn EnrichmentEventSink>,
) -> Option<DeferredRequest> {
    // ── Fire-site 1: on_stage_change(Pending) — entry, before NER call ──────────
    // ADR-052 Gap 1 §3.1 row 1 — triple-emit (ADR-2026-05-20 D1).
    // Phase 5: callback_duration_ms wraps sink call (G7 slow-consumer detection).
    let cb_start = std::time::Instant::now();
    if let Some(s) = sink {
        s.on_stage_change(IngestStatus::Pending);
    }
    metrics::counter!(
        "kremory.sink.stage_transition_total",
        "from" => "queued",
        "to" => "Pending"
    )
    .increment(1);
    metrics::histogram!(
        "kremory.sink.callback_duration_ms",
        "callback" => "on_stage_change",
        "stage" => "Pending"
    )
    .record(cb_start.elapsed().as_secs_f64() * 1000.0);
    tracing::info!(
        text_len = req.text.len(),
        "kremory.background.stage_change.pending"
    );

    let start = std::time::Instant::now();

    // ADR-051 §Phase 3: call ingest_phase1_ner() (fast: episode INSERT + NER
    // candidates) instead of ingest() (full pipeline). Entity writes and fact
    // extraction are deferred to process_deferred via run_verify_stage.
    match graph
        .ingest_phase1_ner(&req.text, SourceParams::default())
        .await
    {
        Ok(phase1_result) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.ingested_total").increment(1);
            tracing::info!(
                elapsed_ms,
                episode_id = phase1_result.episode_id,
                candidates = phase1_result.candidates.len(),
                text_len = req.text.len(),
                "kremory.background.phase1_completed"
            );

            if deferred_enabled {
                // Pass candidate names as ner_entity_names so process_deferred
                // can use them for ingest_deferred fact extraction.
                let ner_entity_names: Vec<String> = phase1_result
                    .candidates
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();
                Some(DeferredRequest {
                    text: req.text,
                    reference_time: req.reference_time,
                    group_id: req.group_id,
                    content_type: req.content_type,
                    episode_id: phase1_result.episode_id,
                    ner_entity_names,
                    // Propagate batch_id so worker_loop can do terminal detection
                    // and fire on_batch_phase2_complete (impl spec §6 Phase 4).
                    batch_id: req.batch_id,
                })
            } else {
                None
            }
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.errors_total").increment(1);
            tracing::error!(elapsed_ms, error = %e, "kremory.background.phase1_failed");

            // ── Fire-site 2: on_ingestion_error — Phase 1 NER failure ────────────
            // ADR-052 Gap 1 §3.1 row 2 — triple-emit (ADR-2026-05-20 D1).
            // IngestErrorKind mapping: Llm/Extraction/Resolution/Database/Embedding/Other
            // → IngestionErrorKind::ProviderError (all Phase 1 failures are provider-level;
            // episode_id unknown so entity_or_edge_ref = None).
            let error_detail = e.to_string();
            let ingestion_error_kind = match IngestErrorKind::from(&e) {
                IngestErrorKind::Llm => IngestionErrorKind::ProviderError {
                    provider_name: "llm".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Embedding => IngestionErrorKind::ProviderError {
                    provider_name: "embedding".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Database => IngestionErrorKind::ProviderError {
                    provider_name: "database".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Extraction => IngestionErrorKind::ParseFailure {
                    stage: "phase1_extraction".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Resolution => IngestionErrorKind::ParseFailure {
                    stage: "phase1_resolution".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Other => IngestionErrorKind::ProviderError {
                    provider_name: "unknown".to_string(),
                    detail: error_detail.clone(),
                },
            };
            // HIGH-02 fix: derive error_kind_str from the actual variant so the metric
            // label does not lie when Extraction/Resolution map to ParseFailure.
            let error_kind_str = match &ingestion_error_kind {
                IngestionErrorKind::ParseFailure { .. } => "ParseFailure",
                _ => "ProviderError",
            };
            // Phase 5: callback_duration_ms wraps on_ingestion_error (G7 slow-consumer detection).
            let cb_start = std::time::Instant::now();
            if let Some(s) = sink {
                s.on_ingestion_error(IngestionError {
                    entity_or_edge_ref: None,
                    error_kind: ingestion_error_kind,
                    is_retryable: true,
                });
            }
            metrics::counter!(
                "kremory.sink.ingestion_error_total",
                "error_kind" => error_kind_str,
                "phase" => "phase1"
            )
            .increment(1);
            metrics::histogram!(
                "kremory.sink.callback_duration_ms",
                "callback" => "on_ingestion_error",
                "stage" => "Failed"
            )
            .record(cb_start.elapsed().as_secs_f64() * 1000.0);
            tracing::error!(error = %e, "kremory.background.ingestion_error.phase1");

            let err = IngestError {
                text_preview: req.text.chars().take(256).collect(),
                failed_at: Utc::now(),
                message: e.to_string(),
                kind: IngestErrorKind::from(&e),
                // Phase 1 failure — episode was never committed; id unknown.
                episode_id: 0,
            };
            try_send_error(error_tx, err);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// process_deferred — Phase 2: verify_stage entity write + LLM fact extraction
// ---------------------------------------------------------------------------

/// Process one deferred extraction request (Phase 2).
///
/// ADR-051 Phase 3 wiring:
///   1. Calls `run_verify_stage` for entity extraction + write (GLiNER Path α or
///      LLM Path β depending on whether the engine's LLM is wired).
///   2. Calls `ingest_deferred` for LLM relationship/fact extraction (unchanged
///      from pre-ADR-051; runs after entity write is committed).
///
/// `bucket` is the optional token-bucket rate limiter applied before the LLM
/// fact extraction step, emitting
/// `kremory.ingest.llm_rate_limit_deferred_total{namespace}` per throttle.
///
/// `sink` is propagated from [`worker_loop`] per ADR-052 Gap 1 (impl spec §3
/// Phase 2).  Phase 3 wires the actual callsites inside this function and in
/// `run_verify_stage` / `ingest_deferred`; `sink` is accepted here so the
/// signature is stable before Phase 3 lands.
///
/// Errors are logged via metrics and the error channel but do NOT crash the worker.
/// Outcome of one `process_deferred` call — used by `worker_loop` to update
/// the `BatchProgress` counter without re-acquiring any state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeferredOutcome {
    Succeeded,
    Failed,
}

pub(super) async fn process_deferred<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    req: DeferredRequest,
    error_tx: &SyncSender<IngestError>,
    bucket: &mut Option<TokenBucketState>,
    sink: Option<&dyn EnrichmentEventSink>,
) -> DeferredOutcome {
    let episode_id = req.episode_id;
    let ns = req.group_id.as_deref().unwrap_or("default");

    // ── Step 1: run_verify_stage — entity write (ADR-051) ─────────────────────
    //
    // Path α (ner feature active): extractor = GLiNER singleton, verify_llm = LLM
    // Path β (no ner feature): extractor = engine.extractor (LLM), verify_llm = None
    //
    // engine.extractor implements EntityExtractor → blanket impl gives EntityExtractorDyn.
    // engine.llm is Option<Arc<L>>; deref gives Option<&L> which is &dyn ChatProvider.
    let verify_start = std::time::Instant::now();

    #[cfg(feature = "ner")]
    let verify_result = {
        // Path α: GLiNER singleton for entity candidate extraction;
        // engine's wired LLM (if any) is used by verify_batch for typing.
        // If GLiNER model fails to load, fall back gracefully: emit counter + skip
        // entity write for this episode (fact extraction below still runs).
        match crate::core::ner::ner_singleton() {
            Ok(gliner) => {
                let verify_llm: Option<&dyn crate::core::provider::ChatProvider> = graph
                    .llm
                    .as_ref()
                    .map(|l| l.as_ref() as &dyn crate::core::provider::ChatProvider);
                super::verify_stage::run_verify_stage(&req, gliner, verify_llm, &graph.graph, sink)
                    .await
            }
            Err(e) => {
                metrics::counter!(
                    "kremory.background.ner_singleton_fail_total",
                    "namespace" => ns.to_string()
                )
                .increment(1);
                tracing::error!(
                    episode_id,
                    error = %e,
                    "kremory.background.ner_singleton_load_failed — skipping verify_stage"
                );
                Err(e)
            }
        }
    };

    #[cfg(not(feature = "ner"))]
    let verify_result = {
        // Path β: engine.extractor is an LLM extractor; no separate verify LLM needed.
        let extractor_ref: &dyn crate::core::intelligence::EntityExtractorDyn =
            graph.extractor.as_ref();
        super::verify_stage::run_verify_stage(&req, extractor_ref, None, &graph.graph, sink).await
    };

    let verify_elapsed_ms = verify_start.elapsed().as_secs_f64() * 1000.0;

    match verify_result {
        Ok(entities_written) => {
            metrics::histogram!("rql.background.verify_stage_duration_ms")
                .record(verify_elapsed_ms);
            metrics::counter!("rql.background.verify_stage_ok_total").increment(1);
            tracing::info!(
                verify_elapsed_ms,
                episode_id,
                entities_written,
                "kremory.background.verify_stage completed"
            );
        }
        Err(e) => {
            metrics::histogram!("rql.background.verify_stage_duration_ms")
                .record(verify_elapsed_ms);
            metrics::counter!("rql.background.verify_stage_errors_total").increment(1);
            // Ghost episode: entity write failed. Fact extraction below still
            // runs so any LLM facts can still be committed against the episode
            // row (which was committed in Phase 1).
            // Quinn Phase 3 MED-03: `step` label per ~/.claude/rules/observability-first-class.md
            // — aggregate counter without source attribution would hide which arm produced
            // ghost episodes. Two firing sites: this one (verify_stage) + fact-extraction one below.
            metrics::counter!(
                "rql.ingest.ghost_episode_total",
                "step" => "verify_stage"
            )
            .increment(1);
            tracing::error!(
                verify_elapsed_ms,
                episode_id,
                error = %e,
                "kremory.background.verify_stage failed — ghost episode (fact extraction continues)"
            );
            let err = IngestError {
                text_preview: req.text.chars().take(256).collect(),
                failed_at: Utc::now(),
                message: format!("verify_stage: {e}"),
                kind: IngestErrorKind::from(&e),
                episode_id,
            };
            try_send_error(error_tx, err);
            // ── HIGH-03 fix: on_ingestion_error triple-emit for verify_stage Err ─────
            // ADR-052 §3.1 row 15: on_ingestion_error MUST fire on Err from run_verify_stage.
            // verify_stage already fires on_stage_change(Failed) internally — do NOT
            // duplicate that here. Only on_ingestion_error is missing at this callsite.
            // is_retryable=false: ghost-episode path continues, no retry mechanism.
            let error_detail = e.to_string();
            let ingestion_error_kind = match IngestErrorKind::from(&e) {
                IngestErrorKind::Llm => IngestionErrorKind::ProviderError {
                    provider_name: "llm".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Embedding => IngestionErrorKind::ProviderError {
                    provider_name: "embedding".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Database => IngestionErrorKind::ProviderError {
                    provider_name: "database".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Extraction => IngestionErrorKind::ParseFailure {
                    stage: "verify_stage_extraction".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Resolution => IngestionErrorKind::ParseFailure {
                    stage: "verify_stage_resolution".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Other => IngestionErrorKind::ProviderError {
                    provider_name: "unknown".to_string(),
                    detail: error_detail.clone(),
                },
            };
            let error_kind_str = match &ingestion_error_kind {
                IngestionErrorKind::ParseFailure { .. } => "ParseFailure",
                _ => "ProviderError",
            };
            // Phase 5: callback_duration_ms wraps on_ingestion_error (G7 slow-consumer detection).
            let cb_start = std::time::Instant::now();
            if let Some(s) = sink {
                s.on_ingestion_error(IngestionError {
                    entity_or_edge_ref: None,
                    error_kind: ingestion_error_kind,
                    is_retryable: false,
                });
            }
            metrics::counter!(
                "kremory.sink.ingestion_error_total",
                "error_kind" => error_kind_str,
                "phase" => "phase1_verify_fail"
            )
            .increment(1);
            metrics::histogram!(
                "kremory.sink.callback_duration_ms",
                "callback" => "on_ingestion_error",
                "stage" => "Failed"
            )
            .record(cb_start.elapsed().as_secs_f64() * 1000.0);
            tracing::error!(episode_id, error = %e, "kremory.background.ingestion_error.verify_fail");
            // Fall through to fact extraction — entity write failing does not
            // block fact rows from being committed.
        }
    }

    // ── Step 2: rate-limit gate before LLM fact extraction ────────────────────
    //
    // F2 / F6: apply rate limit before LLM call; emit counter on throttle.
    if let Some(b) = bucket {
        let immediately_available = b.try_consume();
        if !immediately_available {
            let wait = b.wait_duration();
            metrics::counter!(
                "kremory.ingest.llm_rate_limit_deferred_total",
                "namespace" => ns.to_string()
            )
            .increment(1);
            tracing::debug!(
                wait_ms = wait.as_millis(),
                namespace = ns,
                "kremory.background.rate_limit_wait"
            );
            tokio::time::sleep(wait).await;
            // Consume after sleep (bucket has refilled by at least one token).
            b.try_consume();
        }
    }

    // ── Step 3: ingest_deferred — LLM fact/relationship extraction ─────────────
    let fact_start = std::time::Instant::now();
    match graph
        .ingest_deferred(
            &req.text,
            req.reference_time,
            req.group_id.as_deref(),
            req.content_type,
            req.episode_id,
            &req.ner_entity_names,
            // Coerce EnrichmentEventSink (supertrait) → &dyn IngestEventSink for
            // ingest_deferred's inner callbacks (Phase 3c fire-sites).
            // ADR-052 Gap 1 — sink propagation through the deferred pipeline.
            sink.map(|s| s as &dyn IngestEventSink),
        )
        .await
    {
        Ok(facts_extracted) => {
            let elapsed_ms = fact_start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_facts_extracted_total")
                .increment(facts_extracted as u64);
            tracing::info!(
                elapsed_ms,
                facts_extracted,
                "kremory.background.deferred_extraction completed"
            );

            // ── Fire-site 3: on_stage_change(Complete) — ingest_deferred Ok ──────
            // ADR-052 Gap 1 §3.1 row 3 — triple-emit (ADR-2026-05-20 D1).
            // MED-02 fix: "from" must be a bounded stable label, not the function name.
            // Phase 5: callback_duration_ms wraps on_stage_change (G7 slow-consumer detection).
            let cb_start = std::time::Instant::now();
            if let Some(s) = sink {
                s.on_stage_change(IngestStatus::Complete);
            }
            metrics::counter!(
                "kremory.sink.stage_transition_total",
                "from" => "phase2",
                "to" => "Complete"
            )
            .increment(1);
            metrics::histogram!(
                "kremory.sink.callback_duration_ms",
                "callback" => "on_stage_change",
                "stage" => "Complete"
            )
            .record(cb_start.elapsed().as_secs_f64() * 1000.0);
            tracing::info!(episode_id, "kremory.background.stage_change.complete");

            DeferredOutcome::Succeeded
        }
        Err(e) => {
            let elapsed_ms = fact_start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_errors_total").increment(1);
            // Quinn Phase 3 MED-03 sibling: see verify_stage call site comment above.
            metrics::counter!(
                "rql.ingest.ghost_episode_total",
                "step" => "ingest_deferred"
            )
            .increment(1);
            tracing::error!(
                elapsed_ms,
                episode_id = req.episode_id,
                error = %e,
                "kremory.background.deferred_extraction failed — ghost episode"
            );

            // ── Fire-site 4: on_stage_change(Failed) — ingest_deferred Err ───────
            // ── Fire-site 5: on_ingestion_error — ingest_deferred Err ─────────────
            // ADR-052 Gap 1 §3.1 rows 4+5 — triple-emit (ADR-2026-05-20 D1).
            // IngestErrorKind mapping → IngestionErrorKind (D7: no episode_id label).
            let error_detail = e.to_string();
            let ingestion_error_kind = match IngestErrorKind::from(&e) {
                IngestErrorKind::Llm => IngestionErrorKind::ProviderError {
                    provider_name: "llm".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Embedding => IngestionErrorKind::ProviderError {
                    provider_name: "embedding".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Database => IngestionErrorKind::ProviderError {
                    provider_name: "database".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Extraction => IngestionErrorKind::ParseFailure {
                    stage: "deferred_extraction".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Resolution => IngestionErrorKind::ParseFailure {
                    stage: "deferred_resolution".to_string(),
                    detail: error_detail.clone(),
                },
                IngestErrorKind::Other => IngestionErrorKind::ProviderError {
                    provider_name: "unknown".to_string(),
                    detail: error_detail.clone(),
                },
            };
            // HIGH-02 fix: derive error_kind_str from the actual variant so the metric
            // label does not lie when Extraction/Resolution map to ParseFailure.
            let error_kind_str = match &ingestion_error_kind {
                IngestionErrorKind::ParseFailure { .. } => "ParseFailure",
                _ => "ProviderError",
            };
            // MED-02 fix: "from" label must be a bounded state name, not a function name.
            // "phase2" is stable and non-state; normative from/to spec is Phase 7 follow-up.
            // Phase 5: callback_duration_ms wraps both on_stage_change and on_ingestion_error.
            // Two distinct histograms emitted (one per callback type) within the same block.
            let cb_start = std::time::Instant::now();
            if let Some(s) = sink {
                s.on_stage_change(IngestStatus::Failed(error_detail.clone()));
            }
            metrics::counter!(
                "kremory.sink.stage_transition_total",
                "from" => "phase2",
                "to" => "Failed"
            )
            .increment(1);
            metrics::histogram!(
                "kremory.sink.callback_duration_ms",
                "callback" => "on_stage_change",
                "stage" => "Failed"
            )
            .record(cb_start.elapsed().as_secs_f64() * 1000.0);
            tracing::error!(episode_id, error = %e, "kremory.background.stage_change.failed");
            let cb_start = std::time::Instant::now();
            if let Some(s) = sink {
                s.on_ingestion_error(IngestionError {
                    entity_or_edge_ref: None,
                    error_kind: ingestion_error_kind,
                    is_retryable: false,
                });
            }
            metrics::counter!(
                "kremory.sink.ingestion_error_total",
                "error_kind" => error_kind_str,
                "phase" => "phase2"
            )
            .increment(1);
            metrics::histogram!(
                "kremory.sink.callback_duration_ms",
                "callback" => "on_ingestion_error",
                "stage" => "Failed"
            )
            .record(cb_start.elapsed().as_secs_f64() * 1000.0);
            tracing::error!(episode_id, error = %e, "kremory.background.ingestion_error.phase2");

            let err = IngestError {
                text_preview: req.text.chars().take(256).collect(),
                failed_at: Utc::now(),
                message: format!("deferred: {e}"),
                kind: IngestErrorKind::from(&e),
                // Phase 2 failure: episode WAS committed; carry the id so
                // ghost_episodes() can surface it.
                episode_id: req.episode_id,
            };
            try_send_error(error_tx, err);

            DeferredOutcome::Failed
        }
    }
}

// ---------------------------------------------------------------------------
// worker_loop
// ---------------------------------------------------------------------------

// Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
//
// `sink`: Arc owned here; each call to `process_item` / `process_deferred` borrows
// `sink.as_deref()`.  Per ADR-052 Gap 1; impl spec §3 Phase 2.
//
// `batch_tracker`: shared with the caller-side `BackgroundIngestor` handle.
// After each `process_deferred` terminal, the worker increments the appropriate
// counter and, when `is_terminal()`, fires `on_batch_phase2_complete` then
// removes the entry.  Per impl spec §6 Phase 4 DoD item 6.
#[allow(clippy::too_many_arguments)]
pub(super) fn worker_loop<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: Engine<L, Emb>,
    work_rx: Receiver<IngestRequest>,
    error_tx: SyncSender<IngestError>,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    deferred_enabled: bool,
    llm_rate_limit: Option<RateLimit>,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
    batch_tracker: BatchTracker,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            panic!("invariant: tokio runtime build failed for rql-ingestor worker: {e}")
        });

    rt.block_on(async {
        let mut deferred_queue: VecDeque<DeferredRequest> = VecDeque::new();
        // F2: initialise token bucket from config (None = unlimited).
        let mut bucket: Option<TokenBucketState> = llm_rate_limit.map(TokenBucketState::new);

        loop {
            if stop.load(Ordering::Acquire) {
                // Guard was dropped — drain remaining NER items, then exit.
                // Deferred queue is abandoned on forced shutdown (NER takes priority).
                while let Ok(req) = work_rx.try_recv() {
                    queued.fetch_sub(1, Ordering::Relaxed);
                    if let Some(deferred) =
                        process_item(&graph, req, &error_tx, deferred_enabled, sink.as_deref())
                            .await
                    {
                        deferred_queue.push_back(deferred);
                    }
                }
                // ── Stop-flag drain: fire on_batch_phase2_complete(interrupted) ──────
                // ADR-052 Phase 7 — close the silent-hang foot-gun: any batch with
                // outstanding items at stop-flag drain time receives an "interrupted"
                // terminal event so consumers don't wait indefinitely.
                // D7: batch_id in tracing field only; "interrupted" is a bounded label.
                fire_interrupted_batches(&batch_tracker, sink.as_deref());
                break;
            }

            match work_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(req) => {
                    let ner_depth = queued.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                    metrics::gauge!("rql.background.queue_depth").set(ner_depth as f64);
                    if let Some(deferred) =
                        process_item(&graph, req, &error_tx, deferred_enabled, sink.as_deref())
                            .await
                    {
                        deferred_queue.push_back(deferred);
                    }
                    let depth = deferred_queue.len();
                    metrics::gauge!("rql.background.deferred_queue_depth").set(depth as f64);
                    tracing::info!(depth, "kremory.background.deferred_queue updated");
                }
                Err(RecvTimeoutError::Timeout) => {
                    // NER channel is idle — process one deferred item if available,
                    // then loop back to check for new NER work (NER priority).
                    if let Some(deferred) = deferred_queue.pop_front() {
                        let depth = deferred_queue.len();
                        metrics::gauge!("rql.background.deferred_queue_depth").set(depth as f64);
                        tracing::info!(depth, "kremory.background.deferred_queue draining");
                        let batch_id = deferred.batch_id.clone();
                        let outcome = process_deferred(
                            &graph,
                            deferred,
                            &error_tx,
                            &mut bucket,
                            sink.as_deref(),
                        )
                        .await;
                        fire_batch_complete_if_terminal(
                            &batch_tracker,
                            batch_id,
                            outcome,
                            sink.as_deref(),
                        );
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // All senders dropped — drain remaining NER items, then
                    // process the deferred queue, then exit.
                    while let Ok(req) = work_rx.try_recv() {
                        queued.fetch_sub(1, Ordering::Relaxed);
                        if let Some(deferred) =
                            process_item(&graph, req, &error_tx, deferred_enabled, sink.as_deref())
                                .await
                        {
                            deferred_queue.push_back(deferred);
                        }
                    }
                    // Drain deferred queue, but respect the stop flag so
                    // IngestGuard::drop doesn't block on slow LLM calls.
                    while let Some(deferred) = deferred_queue.pop_front() {
                        if stop.load(Ordering::Acquire) {
                            let abandoned = deferred_queue.len() + 1;
                            tracing::warn!(
                                abandoned,
                                "kremory.background.deferred_queue abandoned (stop signal)"
                            );
                            // Quinn LOW-P7-02: fire interrupted events for batches
                            // with outstanding items before breaking (stop fires
                            // mid-Disconnected-drain path).
                            // ADR-052 §3.4 stop-flag drain policy — closes the
                            // narrow race where stop fires after Disconnected arm
                            // begins draining. The top-of-loop check at entry (line ~628)
                            // only covers the steady-state case; this covers the
                            // mid-drain case. Idempotent: fire_interrupted_batches
                            // uses tracker.retain and double-fire is safe (entry
                            // already removed after fire).
                            fire_interrupted_batches(&batch_tracker, sink.as_deref());
                            break;
                        }
                        let depth = deferred_queue.len();
                        metrics::gauge!("rql.background.deferred_queue_depth").set(depth as f64);
                        tracing::info!(
                            depth,
                            "kremory.background.deferred_queue drain-on-disconnect"
                        );
                        let batch_id = deferred.batch_id.clone();
                        let outcome = process_deferred(
                            &graph,
                            deferred,
                            &error_tx,
                            &mut bucket,
                            sink.as_deref(),
                        )
                        .await;
                        fire_batch_complete_if_terminal(
                            &batch_tracker,
                            batch_id,
                            outcome,
                            sink.as_deref(),
                        );
                    }
                    break;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Batch terminal helper — called after every process_deferred completion
// ---------------------------------------------------------------------------

/// Increment the batch counter for `batch_id` (if set) and, when the batch
/// reaches terminal state, fire the triple-emit for `on_batch_phase2_complete`
/// then remove the entry from the tracker.
///
/// ## Triple-emit (ADR-2026-05-20 D1)
///
/// 1. `sink.on_batch_phase2_complete(BatchPhase2Complete { … })`
/// 2. `metrics::counter!("kremory.sink.batch_complete_total", "outcome" => …)`
/// 3. `tracing::info!(batch_id = …, succeeded, skipped, failed, duration_ms, …)`
///
/// ## D7 cardinality discipline
///
/// `batch_id` appears as a **tracing field** only — NEVER as a metric label.
/// `outcome` is a bounded three-value string and IS allowed as a metric label.
///
/// Per impl spec §6 Phase 4 DoD item 6.
fn fire_batch_complete_if_terminal(
    batch_tracker: &BatchTracker,
    batch_id: Option<String>,
    outcome: DeferredOutcome,
    sink: Option<&dyn EnrichmentEventSink>,
) {
    let Some(bid) = batch_id else {
        return; // No batch tracking for this episode.
    };

    // Short critical section: lock → read → mutate → release.
    // No .await is held across the lock (std::sync::Mutex is correct here).
    let terminal_payload = {
        let mut tracker = batch_tracker.lock().unwrap_or_else(|p| p.into_inner());

        let Some(progress) = tracker.get_mut(&bid) else {
            // Entry was already removed (double-fire guard) or never registered.
            return;
        };

        match outcome {
            DeferredOutcome::Succeeded => progress.succeeded += 1,
            DeferredOutcome::Failed => progress.failed += 1,
        }

        if !progress.is_terminal() {
            return; // Batch not yet complete; release lock.
        }

        // Batch is terminal — capture payload before removing the entry.
        let payload = BatchPhase2Complete {
            batch_id: bid.clone(),
            succeeded: progress.succeeded,
            skipped: progress.skipped,
            failed: progress.failed,
            duration_ms: progress.started_at.elapsed().as_millis() as u64,
        };
        tracker.remove(&bid);
        payload
        // Lock released here (tracker guard drops at end of block).
    };

    // ── Fire-site: on_batch_phase2_complete — triple-emit ────────────────────
    // ADR-052 Gap 1 §3.2 + impl spec §6 Phase 4 DoD item 6.
    // D7: batch_id in tracing field; outcome as bounded metric label.
    //
    // NOTE: `skipped` is always 0 at v0.2.3 — no code path increments it.
    // Phase 6 test wiring (per Tessa §5.4) introduces the skip path when
    // `enrich_per_episode = false`; the outcome_str logic already handles
    // the all-skipped case implicitly via `failed == 0` → "success".
    // Per Quinn LOW-2 review finding.
    let outcome_str = if terminal_payload.failed == 0 {
        "success"
    } else if terminal_payload.succeeded == 0 {
        "all_failed"
    } else {
        "partial"
    };
    // Phase 5: callback_duration_ms wraps on_batch_phase2_complete (G7 slow-consumer detection).
    let cb_start = std::time::Instant::now();
    if let Some(s) = sink {
        s.on_batch_phase2_complete(terminal_payload.clone());
    }
    metrics::counter!(
        "kremory.sink.batch_complete_total",
        "outcome" => outcome_str
    )
    .increment(1);
    metrics::histogram!(
        "kremory.sink.callback_duration_ms",
        "callback" => "on_batch_phase2_complete",
        "stage" => "Complete"
    )
    .record(cb_start.elapsed().as_secs_f64() * 1000.0);
    tracing::info!(
        batch_id = %terminal_payload.batch_id,
        succeeded = terminal_payload.succeeded,
        skipped = terminal_payload.skipped,
        failed = terminal_payload.failed,
        duration_ms = terminal_payload.duration_ms,
        "kremory.background.batch_phase2_complete"
    );
}

// ---------------------------------------------------------------------------
// Stop-flag interrupted batch helper
// ---------------------------------------------------------------------------

/// Fire `on_batch_phase2_complete(outcome="interrupted")` for every batch
/// in `batch_tracker` that still has outstanding items (i.e. episodes that
/// never reached Phase 2 terminal state because the stop flag fired first).
///
/// Called from `worker_loop` immediately before `break` in the stop-flag
/// drain path.  Closes the silent-hang foot-gun: consumers blocking on
/// `on_batch_phase2_complete` receive an `"interrupted"` event rather than
/// waiting indefinitely.
///
/// ## Triple-emit (ADR-2026-05-20 D1 / Phase 7 §E)
///
/// 1. `sink.on_batch_phase2_complete(BatchPhase2Complete { … })`
/// 2. `metrics::counter!("kremory.sink.batch_complete_total", "outcome" => "interrupted")`
/// 3. `tracing::warn!(batch_id = …, …, "kremory.background.batch_phase2_interrupted")`
///
/// ## D7 cardinality discipline
///
/// `batch_id` appears as a **tracing field** only — NEVER as a metric label.
/// `outcome = "interrupted"` is a bounded string and IS allowed as a metric label.
fn fire_interrupted_batches(batch_tracker: &BatchTracker, sink: Option<&dyn EnrichmentEventSink>) {
    // Short critical section: drain all entries with outstanding items.
    let interrupted: Vec<BatchPhase2Complete> = {
        let mut tracker = batch_tracker.lock().unwrap_or_else(|p| p.into_inner());
        let mut payloads = Vec::new();
        tracker.retain(|bid, progress| {
            if !progress.is_terminal() {
                // Outstanding items — emit interrupted event.
                payloads.push(BatchPhase2Complete {
                    batch_id: bid.clone(),
                    succeeded: progress.succeeded,
                    skipped: progress.skipped,
                    failed: progress.failed,
                    duration_ms: progress.started_at.elapsed().as_millis() as u64,
                });
                false // remove from tracker
            } else {
                true // already terminal (race: completed just before stop flag) — leave for normal path
            }
        });
        payloads
        // Lock released here.
    };

    for payload in interrupted {
        // ── Fire-site: on_batch_phase2_complete(interrupted) — triple-emit ────
        // ADR-052 Phase 7 §E + arch spec §3.4 stop-flag drain policy.
        // D7: batch_id in tracing field only; "interrupted" is bounded label.
        let cb_start = std::time::Instant::now();
        if let Some(s) = sink {
            s.on_batch_phase2_complete(payload.clone());
        }
        metrics::counter!(
            "kremory.sink.batch_complete_total",
            "outcome" => "interrupted"
        )
        .increment(1);
        metrics::histogram!(
            "kremory.sink.callback_duration_ms",
            "callback" => "on_batch_phase2_complete",
            "stage" => "interrupted"
        )
        .record(cb_start.elapsed().as_secs_f64() * 1000.0);
        tracing::warn!(
            batch_id = %payload.batch_id,
            succeeded = payload.succeeded,
            skipped = payload.skipped,
            failed = payload.failed,
            duration_ms = payload.duration_ms,
            "kremory.background.batch_phase2_interrupted"
        );
    }
}

// Tests for deferred pipeline behaviour live in ingestor.rs (co-located with
// BackgroundIngestor, which is what the tests exercise). This keeps
// deferred_pipeline.rs under the 500 LoC limit per
// feedback_split_files_before_adding_when_over_500_loc.

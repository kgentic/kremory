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
use crate::core::ingest::{Engine, IngestDeferredParams, SourceParams};
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
/// Bundled (non-generic) parameters for [`process_item`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). The generic `graph` receiver
/// stays a lead positional param.
pub(super) struct ProcessItemParams<'a> {
    pub req: IngestRequest,
    pub error_tx: &'a SyncSender<IngestError>,
    pub deferred_enabled: bool,
    pub sink: Option<&'a dyn EnrichmentEventSink>,
}

pub(super) async fn process_item<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    params: ProcessItemParams<'_>,
) -> Option<DeferredRequest> {
    let ProcessItemParams {
        req,
        error_tx,
        deferred_enabled,
        sink,
    } = params;
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
                    // TD-187 Gap 1 (2026-08-20): propagate the caller-declared
                    // anchor across the Phase 1 → Phase 2 handoff so it
                    // reaches `ingest_deferred` below, not just Phase 1.
                    declared_reference_time: req.declared_reference_time,
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

/// Bundled (non-generic) parameters for [`process_deferred`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). The generic `graph`
/// receiver stays a lead positional param.
pub(super) struct ProcessDeferredParams<'a> {
    pub req: DeferredRequest,
    pub error_tx: &'a SyncSender<IngestError>,
    pub bucket: &'a mut Option<TokenBucketState>,
    pub sink: Option<&'a dyn EnrichmentEventSink>,
}

pub(super) async fn process_deferred<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    params: ProcessDeferredParams<'_>,
) -> DeferredOutcome {
    let ProcessDeferredParams {
        req,
        error_tx,
        bucket,
        sink,
    } = params;
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
                super::verify_stage::run_verify_stage(super::verify_stage::RunVerifyStageParams {
                    request: &req,
                    extractor: gliner,
                    verify_llm,
                    graph: &graph.graph,
                    sink,
                    // ADR-051: GLiNER is closed-vocab — forward the configured
                    // entity types so the deferred path doesn't reject on empty.
                    allowed_entity_types: &graph.config.allowed_entity_types,
                    excluded_entity_types: &graph.config.excluded_entity_types,
                    model: graph.model.as_deref(),
                })
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
        super::verify_stage::run_verify_stage(super::verify_stage::RunVerifyStageParams {
            request: &req,
            extractor: extractor_ref,
            verify_llm: None,
            graph: &graph.graph,
            sink,
            // LLM extractor is open-vocab so this is a no-op here, but forward the
            // configured types for parity with the ner arm (ADR-051).
            allowed_entity_types: &graph.config.allowed_entity_types,
            excluded_entity_types: &graph.config.excluded_entity_types,
            model: graph.model.as_deref(),
        })
        .await
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
        .ingest_deferred(IngestDeferredParams {
            text: &req.text,
            reference_time: req.reference_time,
            // TD-187 Gap 1+2 FIXED (2026-08-20): `DeferredRequest` now carries
            // `declared_reference_time`, propagated from `IngestRequest` at
            // the Phase 1 → Phase 2 handoff above. This used to be
            // hardcoded `None` — a real public-API defect (Gap 2): any
            // consumer calling `.with_sink()` got ungrounded extraction on
            // this path even after supplying `SourceRef::published_at`. See
            // the doc comment on `IngestDeferredParams::declared_reference_time`
            // for why this must stay a distinct channel from `reference_time`.
            declared_reference_time: req.declared_reference_time,
            group_id: req.group_id.as_deref(),
            content_type: req.content_type,
            episode_id: req.episode_id,
            ner_entity_names: &req.ner_entity_names,
            // Coerce EnrichmentEventSink (supertrait) → &dyn IngestEventSink for
            // ingest_deferred's inner callbacks (Phase 3c fire-sites).
            // ADR-052 Gap 1 — sink propagation through the deferred pipeline.
            sink: sink.map(|s| s as &dyn IngestEventSink),
        })
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
/// Bundled parameters for [`worker_loop`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Constructed by the caller-side
/// `BackgroundIngestor` spawn path; fields are moved into the worker thread.
pub(super) struct WorkerLoopParams<L: ChatProvider + 'static, Emb: EmbeddingProvider> {
    pub(super) graph: Engine<L, Emb>,
    pub(super) work_rx: Receiver<IngestRequest>,
    pub(super) error_tx: SyncSender<IngestError>,
    pub(super) queued: Arc<AtomicUsize>,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) deferred_enabled: bool,
    pub(super) llm_rate_limit: Option<RateLimit>,
    pub(super) sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(super) batch_tracker: BatchTracker,
}

pub(super) fn worker_loop<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    params: WorkerLoopParams<L, Emb>,
) {
    let WorkerLoopParams {
        graph,
        work_rx,
        error_tx,
        queued,
        stop,
        deferred_enabled,
        llm_rate_limit,
        sink,
        batch_tracker,
    } = params;
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

        // ── ADR-050 Phase 3 — checkpoint resume ───────────────────────────────
        //
        // On boot: SELECT the latest op_checkpoints row for op_name='verify_stage'.
        // A non-null row means the worker previously crashed mid-run. Fire
        // on_worker_resumed (arch spec §3.1.2 fire-site) + triple-emit, then
        // continue. The per-entity idempotency Guard #1 (dream_idempotency_keys)
        // provides the actual skip logic when run_verify_stage is re-entered.
        //
        // op_run_id for the CURRENT run: stable UUID-style id generated once.
        // Checkpoint writes use INSERT OR REPLACE on (op_name, op_run_id), so
        // each new run creates its own row — we only READ the latest row here.
        //
        // D7: episode_id NOT a metric label (high cardinality). Cursor value
        // goes to tracing fields only.
        let run_id = format!(
            "verify_stage_{}",
            Utc::now().timestamp_micros()
        );
        let mut deferred_episodes_processed: u64 = 0;

        {
            let conn = &graph.graph.conn;
            match conn
                .query(
                    "SELECT cursor FROM op_checkpoints \
                     WHERE op_name = 'verify_stage' \
                     ORDER BY updated_at DESC \
                     LIMIT 1",
                    libsql::params![],
                )
                .await
            {
                Ok(mut rows) => {
                    match rows.next().await {
                        Ok(Some(row)) => {
                            let cursor_val: String =
                                row.get(0).unwrap_or_else(|e| {
                                    // TD-046: do not swallow a checkpoint-column read
                                    // failure silently. An unreadable cursor legitimately
                                    // degrades to "no resume point", but the degradation
                                    // must be observable (CLAUDE.md Rule 19).
                                    tracing::warn!(
                                        error = %e,
                                        op_name = "verify_stage",
                                        "kremory.worker_loop: failed to read checkpoint \
                                         cursor column; treating as no-resume"
                                    );
                                    String::new()
                                });
                            if !cursor_val.is_empty() {
                                // Resume path: fire sink event + triple-emit.
                                if let Some(s) = sink.as_deref() {
                                    s.on_worker_resumed(&cursor_val, "verify_stage");
                                }
                                metrics::counter!(
                                    "rql.dream.checkpoint_resume_total",
                                    "op_name" => "verify_stage"
                                )
                                .increment(1);
                                tracing::warn!(
                                    from_cursor = %cursor_val,
                                    op_name = "verify_stage",
                                    "kremory.worker_loop: resuming from crash checkpoint"
                                );
                            }
                        }
                        Ok(None) => {
                            // No checkpoint — fresh start, normal boot.
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "kremory.worker_loop: op_checkpoints row-next failed; continuing without resume"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "kremory.worker_loop: op_checkpoints SELECT failed; continuing without resume"
                    );
                }
            }
        }

        loop {
            if stop.load(Ordering::Acquire) {
                // Guard was dropped — drain remaining NER items, then exit.
                // Deferred queue is abandoned on forced shutdown (NER takes priority).
                while let Ok(req) = work_rx.try_recv() {
                    queued.fetch_sub(1, Ordering::Relaxed);
                    if let Some(deferred) =
                        process_item(
                            &graph,
                            ProcessItemParams {
                                req,
                                error_tx: &error_tx,
                                deferred_enabled,
                                sink: sink.as_deref(),
                            },
                        )
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
                        process_item(
                            &graph,
                            ProcessItemParams {
                                req,
                                error_tx: &error_tx,
                                deferred_enabled,
                                sink: sink.as_deref(),
                            },
                        )
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
                        // Capture episode_id before move into process_deferred.
                        let episode_id_for_checkpoint = deferred.episode_id;
                        let outcome = process_deferred(
                            &graph,
                            ProcessDeferredParams {
                                req: deferred,
                                error_tx: &error_tx,
                                bucket: &mut bucket,
                                sink: sink.as_deref(),
                            },
                        )
                        .await;
                        fire_batch_complete_if_terminal(FireBatchCompleteIfTerminalParams {
                            batch_tracker: &batch_tracker,
                            batch_id,
                            outcome,
                            sink: sink.as_deref(),
                        });
                        // ADR-050 Phase 3: write checkpoint every N=10 deferred episodes.
                        deferred_episodes_processed += 1;
                        if deferred_episodes_processed % CHECKPOINT_INTERVAL == 0 {
                            write_checkpoint(WriteCheckpointParams {
                                conn: &graph.graph.conn,
                                op_name: "verify_stage",
                                run_id: &run_id,
                                episode_id: episode_id_for_checkpoint,
                            })
                            .await;
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // All senders dropped — drain remaining NER items, then
                    // process the deferred queue, then exit.
                    while let Ok(req) = work_rx.try_recv() {
                        queued.fetch_sub(1, Ordering::Relaxed);
                        if let Some(deferred) =
                            process_item(
                            &graph,
                            ProcessItemParams {
                                req,
                                error_tx: &error_tx,
                                deferred_enabled,
                                sink: sink.as_deref(),
                            },
                        )
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
                        // Capture episode_id before move into process_deferred.
                        let episode_id_for_checkpoint = deferred.episode_id;
                        let outcome = process_deferred(
                            &graph,
                            ProcessDeferredParams {
                                req: deferred,
                                error_tx: &error_tx,
                                bucket: &mut bucket,
                                sink: sink.as_deref(),
                            },
                        )
                        .await;
                        fire_batch_complete_if_terminal(FireBatchCompleteIfTerminalParams {
                            batch_tracker: &batch_tracker,
                            batch_id,
                            outcome,
                            sink: sink.as_deref(),
                        });
                        // ADR-050 Phase 3: write checkpoint every N=10 deferred episodes.
                        deferred_episodes_processed += 1;
                        if deferred_episodes_processed % CHECKPOINT_INTERVAL == 0 {
                            write_checkpoint(WriteCheckpointParams {
                                conn: &graph.graph.conn,
                                op_name: "verify_stage",
                                run_id: &run_id,
                                episode_id: episode_id_for_checkpoint,
                            })
                            .await;
                        }
                    }
                    break;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// ADR-050 Phase 3 — checkpoint write helper
// ---------------------------------------------------------------------------

/// Write an `op_checkpoints` row for the verify_stage worker (ADR-050 Phase 3).
///
/// Called every `CHECKPOINT_INTERVAL` deferred-episode completions to record
/// the last-processed `episode_id` as the resume cursor. On crash + restart,
/// `worker_loop` reads this row via `on_worker_resumed` and resumes processing;
/// the per-entity `dream_idempotency_keys` guard (Guard #1) provides the
/// actual skip logic so re-entering an already-processed episode is safe.
///
/// Uses `INSERT OR REPLACE` on `(op_name, op_run_id)` so each run upserts its
/// own row. The SELECT on boot uses `ORDER BY updated_at DESC LIMIT 1` so it
/// finds the most-recently-updated row regardless of `op_run_id`.
///
/// Soft-fail: checkpoint failures are logged + metered but NOT propagated —
/// a failed checkpoint write degrades crash-safety (may re-process on next
/// boot) but does NOT corrupt data (idempotency guard catches re-processing).
/// D7: episode_id in tracing field only, NOT a metric label.
/// Bundled parameters for [`write_checkpoint`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct WriteCheckpointParams<'a> {
    conn: &'a libsql::Connection,
    op_name: &'a str,
    run_id: &'a str,
    episode_id: i64,
}

async fn write_checkpoint(params: WriteCheckpointParams<'_>) {
    let WriteCheckpointParams {
        conn,
        op_name,
        run_id,
        episode_id,
    } = params;
    let cursor = episode_id.to_string();
    let now_epoch = Utc::now().timestamp();
    if let Err(e) = conn
        .execute(
            "INSERT OR REPLACE INTO op_checkpoints \
             (op_name, op_run_id, cursor, updated_at) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![op_name, run_id, cursor.clone(), now_epoch],
        )
        .await
    {
        tracing::warn!(
            error = %e,
            op_name,
            episode_id,
            "kremory.dream.checkpoint_write_fail — crash-safety degraded for this run"
        );
        metrics::counter!(
            "kremory.dream.checkpoint_write_fail_total",
            "op_name" => "verify_stage"
        )
        .increment(1);
    } else {
        tracing::debug!(
            op_name,
            run_id,
            cursor = %cursor,
            "kremory.dream.checkpoint_written"
        );
    }
}

/// Number of deferred episodes to process between checkpoint writes.
/// ADR-050 Phase 3: N=10 provides coarse-grained crash-safety without
/// excessive write amplification.
const CHECKPOINT_INTERVAL: u64 = 10;

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
/// Bundled parameters for [`fire_batch_complete_if_terminal`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments).
struct FireBatchCompleteIfTerminalParams<'a> {
    batch_tracker: &'a BatchTracker,
    batch_id: Option<String>,
    outcome: DeferredOutcome,
    sink: Option<&'a dyn EnrichmentEventSink>,
}

fn fire_batch_complete_if_terminal(params: FireBatchCompleteIfTerminalParams<'_>) {
    let FireBatchCompleteIfTerminalParams {
        batch_tracker,
        batch_id,
        outcome,
        sink,
    } = params;
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

//! Worker loop — spawn_worker, drain logic, per-job processing, error reporting.
//!
//! Sprint plan T2.1 / ADR-049 §Decision 6 — worker loop module.
//!
//! Contains:
//! - [`process_item`]    — Phase 1 NER ingest for one [`IngestRequest`]
//! - [`process_deferred`] — Phase 2 deferred LLM fact extraction for one [`DeferredRequest`]
//! - [`worker_loop`]     — main OS-thread loop; drives both phases, respects stop flag

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;

use crate::core::ingest::{Engine, SourceParams};
use crate::core::provider::{ChatProvider, EmbeddingProvider};

use super::{
    try_send_error, DeferredRequest, IngestError, IngestErrorKind, IngestRequest, RateLimit,
    TokenBucketState,
};

// ---------------------------------------------------------------------------
// process_item — Phase 1 NER ingest
// ---------------------------------------------------------------------------

/// Process one NER ingest request (Phase 1).
///
/// Returns `Some(DeferredRequest)` when Phase 2 should be enqueued, or `None`
/// on error (error already forwarded to `error_tx`).
pub(super) async fn process_item<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    req: IngestRequest,
    error_tx: &SyncSender<IngestError>,
    deferred_enabled: bool,
) -> Option<DeferredRequest> {
    let start = std::time::Instant::now();
    match graph
        .ingest(
            &req.text,
            req.reference_time,
            req.group_id.as_deref(),
            req.content_type.clone(),
            SourceParams::default(),
        )
        .await
    {
        Ok(result) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.ingested_total").increment(1);
            tracing::info!(elapsed_ms, "kremory.background.ingest completed");

            if deferred_enabled {
                let ner_entity_names = result.upserted_entities.clone();
                Some(DeferredRequest {
                    text: req.text,
                    reference_time: req.reference_time,
                    group_id: req.group_id,
                    content_type: req.content_type,
                    episode_id: result.episode_id,
                    ner_entity_names,
                })
            } else {
                None
            }
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.errors_total").increment(1);
            tracing::error!(elapsed_ms, error = %e, "kremory.background.ingest failed");
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
// process_deferred — Phase 2 LLM fact extraction
// ---------------------------------------------------------------------------

/// Process one deferred LLM fact extraction request (Phase 2).
///
/// `bucket` is the optional token-bucket rate limiter.  When present, this
/// function waits until a token is available before dispatching the LLM call,
/// emitting `kremory.ingest.llm_rate_limit_deferred_total{namespace}` per wait.
///
/// Errors are logged via metrics and the error channel but do NOT crash the worker.
pub(super) async fn process_deferred<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: &Engine<L, Emb>,
    req: DeferredRequest,
    error_tx: &SyncSender<IngestError>,
    bucket: &mut Option<TokenBucketState>,
) {
    // F2 / F6: apply rate limit before LLM call; emit counter on throttle.
    if let Some(b) = bucket {
        let immediately_available = b.try_consume();
        if !immediately_available {
            let wait = b.wait_duration();
            let ns = req.group_id.as_deref().unwrap_or("default");
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
    let start = std::time::Instant::now();
    match graph
        .ingest_deferred(
            &req.text,
            req.reference_time,
            req.group_id.as_deref(),
            req.content_type,
            req.episode_id,
            &req.ner_entity_names,
        )
        .await
    {
        Ok(facts_extracted) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_facts_extracted_total")
                .increment(facts_extracted as u64);
            tracing::info!(
                elapsed_ms,
                facts_extracted,
                "kremory.background.deferred_extraction completed"
            );
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_errors_total").increment(1);
            // C6: ghost episode — Phase 1 committed, Phase 2 failed.
            // Counter is observable per Rule 19 §3 (per-source counters).
            metrics::counter!("rql.ingest.ghost_episode_total").increment(1);
            tracing::error!(
                elapsed_ms,
                episode_id = req.episode_id,
                error = %e,
                "kremory.background.deferred_extraction failed — ghost episode"
            );
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
        }
    }
}

// ---------------------------------------------------------------------------
// worker_loop
// ---------------------------------------------------------------------------

// Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
#[allow(clippy::too_many_arguments)]
pub(super) fn worker_loop<L: ChatProvider + 'static, Emb: EmbeddingProvider>(
    graph: Engine<L, Emb>,
    work_rx: Receiver<IngestRequest>,
    error_tx: SyncSender<IngestError>,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    deferred_enabled: bool,
    llm_rate_limit: Option<RateLimit>,
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
                        process_item(&graph, req, &error_tx, deferred_enabled).await
                    {
                        deferred_queue.push_back(deferred);
                    }
                }
                break;
            }

            match work_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(req) => {
                    let ner_depth =
                        queued.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                    metrics::gauge!("rql.background.queue_depth").set(ner_depth as f64);
                    if let Some(deferred) =
                        process_item(&graph, req, &error_tx, deferred_enabled).await
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
                        process_deferred(&graph, deferred, &error_tx, &mut bucket).await;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // All senders dropped — drain remaining NER items, then
                    // process the deferred queue, then exit.
                    while let Ok(req) = work_rx.try_recv() {
                        queued.fetch_sub(1, Ordering::Relaxed);
                        if let Some(deferred) =
                            process_item(&graph, req, &error_tx, deferred_enabled).await
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
                            eprintln!(
                                "[BackgroundIngestor] stop signal — abandoning {abandoned} deferred item(s)"
                            );
                            break;
                        }
                        let depth = deferred_queue.len();
                        metrics::gauge!("rql.background.deferred_queue_depth").set(depth as f64);
                        tracing::info!(
                            depth,
                            "kremory.background.deferred_queue drain-on-disconnect"
                        );
                        process_deferred(&graph, deferred, &error_tx, &mut bucket).await;
                    }
                    break;
                }
            }
        }
    });
}

// Tests for deferred pipeline behaviour live in ingestor.rs (co-located with
// BackgroundIngestor, which is what the tests exercise). This keeps
// deferred_pipeline.rs under the 500 LoC limit per
// feedback_split_files_before_adding_when_over_500_loc.

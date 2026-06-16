//! `EnrichmentEventSink` — Phase 3 event sink extending `IngestEventSink`.
//!
//! Per ADR D.6.5 + ADR rqlm-async-event-handle-api-design-2026-05-19 §4.8:
//! rqlm extends `IngestEventSink` (from kremory::core::sink) with Phase 3
//! (batch consolidation / dream-phase) events.
//!
//! Ordering constraint (C2 per ADR): `IngestEventSink` in kremory::core
//! MUST be committed before this module resolves.

use serde::{Deserialize, Serialize};

use crate::core::sink::IngestEventSink;

/// Completion event for an entire batch's Phase 2 cycle.
///
/// Per ADR §4.8 / G2.3 + G5: explicit `skipped` and `failed` counts.
///
/// - Weaviate surfaces `number_errors` (no skipped bucket).
/// - OpenAI Batch surfaces `request_counts` (no skipped bucket).
/// - Hatchet has `was_skipped` at task level.
///
/// rqlm provides all three for full partial-batch observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPhase2Complete {
    /// The caller-set `batch_id` string identifying this batch.
    pub batch_id: String,
    /// Episodes with `enrich_per_episode = true` that completed successfully.
    pub succeeded: usize,
    /// Episodes with `enrich_per_episode = false` (no enrichment requested — done by definition).
    pub skipped: usize,
    /// Episodes with `enrich_per_episode = true` that failed Phase 2.
    pub failed: usize,
    /// Wall-clock milliseconds from first episode submit to last terminal status.
    pub duration_ms: u64,
}

/// Extension of `IngestEventSink` with Phase 3 (dream-phase) events.
///
/// Per ADR §4.8: consumers wanting both Phase 2 and Phase 3 events implement
/// this single trait. `rqlm::submit_episode` and `rqlm::submit_dream_phase`
/// both accept `Option<Arc<dyn EnrichmentEventSink>>`.
///
/// Supertrait ordering: `IngestEventSink` MUST be declared in kremory::core
/// before this trait can compile (ADR C2 constraint).
///
/// # Sink callback contract (G7 — v0.1.6)
///
/// All `EnrichmentEventSink` methods (including those inherited from
/// `IngestEventSink`) are called **sync inline** on the Phase 2 enrichment
/// pipeline thread. **Slow callbacks stall ingest.**
///
/// Consumers MUST keep callbacks fast — sub-millisecond ideal,
/// sub-100ms absolute ceiling. Typical pattern:
///
/// 1. Append the event to a consumer-owned in-memory buffer
/// 2. Return immediately from the sink method
/// 3. Drain the buffer asynchronously on the consumer's own task / thread
///
/// For consumers using napi-rs `ThreadsafeFunction` callbacks, the TSF's
/// internal queue provides natural backpressure for JS-side delivery, but
/// the *Rust-side enrichment pipeline* is still blocked until `tsf.call()`
/// returns. Configure `max_queue_size` on the TSF if needed.
///
/// # Panic policy (D4 thread-context contract)
///
/// A panic inside a sink callback propagates up the enrichment pipeline and
/// will abort the Phase 2 run for the affected episode. Callbacks SHOULD
/// catch their own panics if they call into FFI or other panic-on-failure
/// code (e.g. wrap in `std::panic::catch_unwind`).
///
/// # Thread-context contract (ADR-052 D4)
///
/// All callbacks inherited from `IngestEventSink` (stage-change, entity
/// extracted, edge added, ingestion error) fire **sync-inline on the
/// background worker OS thread** — NOT the caller's thread or the caller's
/// tokio runtime.  `on_batch_phase2_complete` fires on the same OS thread
/// after the batch's last episode reaches terminal state.
///
/// Consumers using napi-rs MUST use `ThreadsafeFunction::call(value,
/// Mode::NonBlocking)` to cross the thread boundary.  Using blocking mode
/// WILL deadlock because the callback fires from within an `async fn` run
/// via `rt.block_on(async { … })` on the background worker runtime.
///
/// # Inline-path limitation (MED-02)
///
/// **All fire-sites in v0.2.3 are on the background worker path
/// (`BackgroundIngestor`).** The inline ingest path (`run_in_background=false`
/// on the engine handle) bypasses the background worker entirely — a consumer
/// who registers a sink on the inline path receives **zero** sink events.
///
/// Consumers needing sink-driven progress notifications MUST use the
/// background (deferred) path. Inline-path sink wiring is deferred to
/// v0.2.4+ pending a confirmed consumer use case.
///
/// # `IngestStatus::EntitiesReady` vs `Complete` (ADR-052 D2/D5)
///
/// `on_stage_change(EntitiesReady)` fires after Phase 2a (entity writes)
/// completes — i.e., after the SQL column writes `'Verified'`.
/// `on_stage_change(Complete)` fires after Phase 2b (fact/relationship
/// extraction) completes.  `wait_for_processing` resolves at Phase 2a
/// (SQL `'Verified'`); there is no polling API for Phase 2b completion.
/// Use `on_stage_change(Complete)` as the push notification.
///
/// # `on_stage_change(Deduplicating)` — normative fire condition (MED-01)
///
/// `on_stage_change(IngestStatus::Deduplicating)` fires at most ONCE per
/// episode, when contradiction detection begins — specifically when fact
/// extraction returned ≥1 fact AND the contradiction-detection loop is
/// entered.  It does NOT fire when zero facts were extracted (fast-exit
/// path).  Fires regardless of whether contradictions are actually found.
///
/// # `IngestStatus::SkippedIdempotent` — forward-compat (MED-05 / ADR-050)
///
/// The `IngestStatus` enum carries `#[non_exhaustive]`.  ADR-050
/// (crash-safety + idempotency cluster, v0.2.4+) will add a
/// `SkippedIdempotent` variant that fires when a duplicate episode is
/// detected and the pipeline skips Phase 2 for that episode.  Consumers
/// matching on `IngestStatus` MUST include a `_` catch-all arm to remain
/// forward-compatible.
///
/// # `on_batch_phase2_complete` drop-before-complete contract
///
/// If the `BackgroundIngestor` (or its companion `IngestGuard`) is dropped
/// while episodes in a batch are still queued or in Phase 2, the sink
/// receives `on_batch_phase2_complete` with `outcome="interrupted"` for any
/// batch with outstanding items at stop-flag drain time.  This closes the
/// silent-hang foot-gun: consumers can reliably detect an interrupted batch
/// rather than waiting indefinitely.  The `"interrupted"` outcome is a
/// bounded string label; `batch_id` is NEVER used as a metric label
/// (D7 cardinality discipline).
///
/// # Future: substrate-owned backpressure (deferred)
///
/// If sink saturation under burst becomes a real consumer-pain pattern,
/// substrate-owned backpressure (queue + drop policy + ordering guarantees)
/// will be added — G7/G8 deferred per ratified API-gap spec. Until then, sync-inline is the contract.
pub trait EnrichmentEventSink: IngestEventSink {
    /// A community in the graph was recomputed during Phase 3 consolidation.
    fn on_community_updated(&self, community_id: &str, member_count: usize);
    /// All Phase 2 runs for a batch have reached terminal status.
    fn on_batch_phase2_complete(&self, event: BatchPhase2Complete);
}

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
/// # Panic policy
///
/// A panic inside a sink callback propagates up the enrichment pipeline and
/// will abort the Phase 2 run for the affected episode. Callbacks SHOULD
/// catch their own panics if they call into FFI or other panic-on-failure
/// code (e.g. wrap in `std::panic::catch_unwind`).
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

//! `EnrichmentEventSink` — Phase 3 event sink extending `IngestEventSink`.
//!
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

/// A cross-episode merge decision, passed to
/// [`EnrichmentEventSink::on_merge_proposed`]. Args-as-object (mirrors
/// [`BatchPhase2Complete`]) so the callback stays within the
/// project's method-arity budget.
///
/// Borrowed (`&str`) rather than owned: the event fires sync-inline from inside the
/// consolidation clique loop where the ids are already borrowed, so it allocates
/// nothing on the background dream sweep. Per the G7 callback contract, a consumer
/// that needs to retain the data copies it (`.to_string()`) before returning.
#[derive(Debug, Clone, Copy)]
pub struct MergeProposed<'a> {
    /// The namespace the merge decision was made in.
    pub group_id: &'a str,
    /// The entity id that would be (or was) fused into `keeper`.
    pub loser: &'a str,
    /// The surviving entity id.
    pub keeper: &'a str,
    /// `true` = shadowed (decision computed, no entity fused); `false` = applied.
    pub dry_run: bool,
}

/// Extension of `IngestEventSink` with Phase 3 (dream-phase) events.
///
/// Per ADR §4.8: consumers wanting both Phase 2 and Phase 3 events implement
/// this single trait. `rqlm::submit_episode` and `rqlm::submit_dream_phase`
/// both accept `Option<Arc<dyn EnrichmentEventSink>>`.
///
/// Phase 3 events: `on_community_updated`, `on_batch_phase2_complete`,
/// `on_worker_resumed` (crash-resume), and `on_merge_proposed` (a
/// cross-episode merge decision, shadowed or applied). All but the first two carry a
/// default no-op, so consumers implement only the events they care about.
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
/// # Thread-context contract
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
/// # Inline-path coverage
///
/// The inline ingest path (`run_in_background=false`, i.e.
/// `Memory::remember(..).with_event_sink(..)` without `.no_wait()`) **is** sink-wired:
/// `engine_handle.rs` coerces the `EnrichmentEventSink` to the
/// core `IngestEventSink` supertrait and threads it into the shared `engine.ingest()`
/// extraction routine (`sink: core_sink`). A consumer registering a sink on the inline
/// path therefore receives the extraction fire-sites (`on_stage_change`,
/// `on_entity_extracted`, `on_edge_added`, `on_contradiction`, etc.) that the unified
/// routine emits.
///
/// What the inline path does **not** receive: background-worker-lifecycle events that
/// have no inline analogue — `on_worker_resumed` (crash-checkpoint replay) and
/// `on_batch_phase2_complete` (batch terminal detection) fire only from the
/// `BackgroundIngestor` worker loop. Note also the inline error-path asymmetry: on
/// ingest failure the inline path cannot write `Failed` (no `episode_id` handle at that
/// scope — see `engine_handle.rs`), so it emits
/// `kremory.engine_handle.inline_ingest_fail_no_episode_id_total` instead.
///
/// # `IngestStatus::EntitiesReady` vs `Complete`
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
/// # `IngestStatus::SkippedIdempotent`
///
/// The `IngestStatus` enum carries `#[non_exhaustive]`.  `SkippedIdempotent`
/// was added in v0.2.4 and fires via
/// `on_stage_change(SkippedIdempotent)` each time Guard #1 in
/// `run_verify_stage` detects a duplicate content hash — i.e., the entity
/// was already processed in this crash-resume pass and Phase 2 is skipped.
/// Fires once per skipped entity (NOT once per episode).
///
/// Consumers matching on `IngestStatus` MUST retain the `_` catch-all arm
/// (enforced by `#[non_exhaustive]`) to stay forward-compatible with future
/// variants.  Consumers SHOULD deduplicate `on_stage_change(Extracting)`
/// callbacks on `(episode_id, stage)` because crash-resume re-enters
/// `run_verify_stage` for partially-processed episodes.
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

    /// Fires on the background worker OS thread when `worker_loop` boots and
    /// detects a non-null `op_checkpoints` entry — indicating a prior crash.
    ///
    /// `from_cursor` is the serialised cursor value (opaque string; typically
    /// an episode_id serialised to a decimal string). Consumers can use this
    /// to log an audit trail, display a "resumed from checkpoint" banner, or
    /// skip UI reconciliation for already-processed entities (pre-cursor
    /// entries were already signalled before crash). Repeated crashes at the
    /// same cursor produce repeated firings — the event is NOT deduplicated.
    ///
    /// **Default no-op**: This is an operational-observability event, NOT a
    /// correctness event. Existing consumers that don't care about crash-resume
    /// MUST NOT be forced to implement a no-op.
    /// This is an explicit, scoped deviation from v0.2.3's "no default impls"
    /// policy for `IngestEventSink` correctness methods.
    ///
    /// **Forward-compat note**: After v0.2.4,
    /// sink consumers MUST deduplicate `on_stage_change(Extracting)` callbacks
    /// on `(episode_id, stage)` because crash-resume may re-enter
    /// `run_verify_stage` for partially-processed episodes. `on_worker_resumed`
    /// fires once per worker boot on the resume path to signal that redelivery
    /// is about to occur.
    fn on_worker_resumed(&self, _from_cursor: &str, _op_name: &str) {}

    /// A cross-episode merge decision was made during Phase 3 consolidation —
    /// fires whether the decision was shadowed
    /// ([`MergeProposed::dry_run`]` == true`, no entity fused) or applied
    /// (`dry_run == false`, the loser was fused into the keeper). The `dry_run` flag
    /// lets a consumer observing a shadow window tell "would-have-merged" from
    /// "did-merge".
    ///
    /// **Default no-op**, mirroring the `on_worker_resumed` precedent above: existing
    /// consumers implementing only Phase 2/3 correctness events are unaffected (this
    /// is an operational-observability event, not a correctness event).
    ///
    /// Follows the SAME G7 callback contract every other `EnrichmentEventSink` method
    /// documents (see the trait-level "Sink callback contract" above): fired
    /// sync-inline on the dream-phase's executing thread — slow callbacks stall dream.
    /// Consumers buffer + return immediately, exactly as for every other method.
    fn on_merge_proposed(&self, _event: MergeProposed<'_>) {}
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `RecordingSink` — canonical test double for sink-wiring assertions.
//!
//! Design rationale (Tessa §6 T3 + T2):
//!
//! - Uses `Arc<Mutex<Vec<SinkEvent>>>` append ordering — same pattern as
//!   `MockIngestSink` in `event_order.rs` (`Arc<Mutex<Vec<String>>>`).
//! - After `drop(ingestor)` + `guard.shutdown()`, the worker thread has
//!   fully exited so `Mutex<Vec>` reads are temporally non-overlapping with
//!   sink writes.  No race.
//! - `Clone` on `RecordingSink` clones the `Arc` (shared buffer) — both the
//!   test assertion side and the sink-registration side see the same vec.
//!
//! ## Why net-new (not reuse of existing sinks)
//!
//! - `MockIngestSink` (`event_order.rs`): records `Vec<String>` — format-
//!   dependent; cannot assert structural fields like `entity_id` or `predicate`.
//! - `CountingSink` (`facade_event_sink.rs`): counts `entity_extracted` only.
//!   Too coarse for multi-method fire-site validation.
//! - `StubIngestSink` (`sink_trait_shape.rs`): compile-only, no collection.
//!
//! Per test strategy §6 T3 answer.

use std::sync::{Arc, Mutex};

use kremory::core::error::IngestStatus;
use kremory::core::sink::{ContradictionDetected, IngestEventSink, IngestionError};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};

// ---------------------------------------------------------------------------
// SinkEvent
// ---------------------------------------------------------------------------

/// Typed event recorded by [`RecordingSink`].
///
/// Each variant corresponds to one `IngestEventSink` / `EnrichmentEventSink`
/// method.  `CommunityUpdated` is a **sentinel arm** for v0.2.3 — it must
/// NEVER fire because `on_community_updated` is deferred to ADR-050.
/// The sentinel arm lets `sink_community_updated_does_not_fire_in_v023`
/// assert absence structurally rather than string-matching tracing logs.
#[derive(Debug, Clone, PartialEq)]
pub enum SinkEvent {
    /// `on_entity_extracted(entity_id, name)` fired.
    EntityExtracted { entity_id: String, name: String },
    /// `on_edge_added(from, to, predicate)` fired.
    EdgeAdded {
        from: String,
        to: String,
        predicate: String,
    },
    /// `on_stage_change(stage)` fired.
    StageChange(IngestStatus),
    /// `on_ingestion_error(event)` fired.
    IngestionError {
        error_kind_debug: String,
        is_retryable: bool,
    },
    /// `on_contradiction(event)` fired.
    Contradiction { resolution_debug: String },
    /// `on_dedup_merge(surviving, absorbed)` fired.
    DedupMerge { surviving: String, absorbed: String },
    /// `on_batch_phase2_complete(event)` fired.
    BatchComplete {
        batch_id: String,
        succeeded: usize,
        failed: usize,
    },
    /// Sentinel — `on_community_updated` is DEFERRED per ADR-052 / ADR-050.
    ///
    /// If this variant appears in any v0.2.3 test's recorded events it signals
    /// that `on_community_updated` was wired ahead of its planned ADR-050 sprint.
    /// The test `sink_community_updated_does_not_fire_in_v023` asserts absence.
    CommunityUpdated,
}

// ---------------------------------------------------------------------------
// RecordingSink
// ---------------------------------------------------------------------------

/// Thread-safe test double that appends every fired event to a shared vec.
///
/// `Clone`-ing shares the same `Arc<Mutex<Vec<SinkEvent>>>` — both the
/// sink passed to the ingestor and the assertion handle in the test body
/// observe the same buffer.
///
/// ## Send + Sync + 'static
///
/// `Arc<Mutex<Vec<SinkEvent>>>` is `Send + Sync + 'static`, satisfying the
/// `EnrichmentEventSink: Send + Sync` bound required by `IngestorConfig::with_sink`.
#[derive(Clone, Default)]
pub struct RecordingSink {
    pub events: Arc<Mutex<Vec<SinkEvent>>>,
}

// `#[allow(dead_code)]` was previously added defensively here but
// `#[expect(dead_code)]` proved it suppresses NOTHING — the dead_code lint
// does not fire on this impl block in any test binary because at least one
// method is reachable from every binary's call graph.  No suppression needed.
// Cause-fix per CLAUDE.md Rule 8 + Quinn Phase 6 review MED Rule-8.
impl RecordingSink {
    /// Construct a new, empty recording sink.
    ///
    /// `#[allow(dead_code)]` is intentional: `new()` is used by
    /// `sink_wiring.rs`, `sink_wiring_integration.rs`, and
    /// `background_ingestor_handle_routing.rs` but NOT by `llm_integration.rs`
    /// (which pulls in `helpers/mod.rs` for other helpers but doesn't use
    /// RecordingSink).  Cross-binary asymmetry — same rationale as
    /// `entity_events`/`edge_events`/`stage_events` below.  Quinn Phase 6
    /// review MED Rule-8 precedent.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Snapshot the current event list (cloned; does not drain).
    pub fn snapshot(&self) -> Vec<SinkEvent> {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Return only the `StageChange` events, in order.
    ///
    /// `#[allow(dead_code)]` is intentional: this helper is used by
    /// `sink_wiring_integration.rs` stage-order tests but not by
    /// `background_ingestor_handle_routing.rs`.  Cross-binary asymmetry
    /// requires `#[allow]` — same rationale as `entity_events` below.
    /// Quinn Phase 6 review MED Rule-8.
    #[allow(dead_code)]
    pub fn stage_events(&self) -> Vec<IngestStatus> {
        self.snapshot()
            .into_iter()
            .filter_map(|e| {
                if let SinkEvent::StageChange(s) = e {
                    Some(s)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return only the `EntityExtracted` events, in order.
    ///
    /// `#[allow(dead_code)]` is intentional: this helper is used by
    /// `sink_wiring.rs` entity-counting tests but unused in
    /// `sink_wiring_integration.rs`.  `#[expect(dead_code)]` cannot be used
    /// here because the asymmetry across binaries (live in one, dead in the
    /// other) would fire `unfulfilled-lint-expectations` in the binary
    /// where the method IS used.  Per-binary cfg-gating would force a DRY
    /// violation; `#[allow]` is the documented escape hatch for
    /// cross-binary test helpers.  Quinn Phase 6 review MED Rule-8.
    #[allow(dead_code)]
    pub fn entity_events(&self) -> Vec<(String, String)> {
        self.snapshot()
            .into_iter()
            .filter_map(|e| {
                if let SinkEvent::EntityExtracted { entity_id, name } = e {
                    Some((entity_id, name))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return only the `EdgeAdded` events, in order.
    ///
    /// `#[allow(dead_code)]` is intentional: see `entity_events` doc-comment
    /// above for the cross-binary asymmetry rationale.  Quinn Phase 6
    /// review MED Rule-8.
    #[allow(dead_code)]
    pub fn edge_events(&self) -> Vec<(String, String, String)> {
        self.snapshot()
            .into_iter()
            .filter_map(|e| {
                if let SinkEvent::EdgeAdded {
                    from,
                    to,
                    predicate,
                } = e
                {
                    Some((from, to, predicate))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IngestEventSink impl
// ---------------------------------------------------------------------------

impl IngestEventSink for RecordingSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::EntityExtracted {
                entity_id: entity_id.to_string(),
                name: name.to_string(),
            });
    }

    fn on_edge_added(&self, from_entity_id: &str, to_entity_id: &str, predicate: &str) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::EdgeAdded {
                from: from_entity_id.to_string(),
                to: to_entity_id.to_string(),
                predicate: predicate.to_string(),
            });
    }

    fn on_contradiction(&self, event: ContradictionDetected) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::Contradiction {
                resolution_debug: format!("{:?}", event.resolution),
            });
    }

    fn on_dedup_merge(&self, surviving_id: &str, absorbed_id: &str) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::DedupMerge {
                surviving: surviving_id.to_string(),
                absorbed: absorbed_id.to_string(),
            });
    }

    fn on_stage_change(&self, stage: IngestStatus) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::StageChange(stage));
    }

    fn on_ingestion_error(&self, event: IngestionError) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::IngestionError {
                error_kind_debug: format!("{:?}", event.error_kind),
                is_retryable: event.is_retryable,
            });
    }
}

// ---------------------------------------------------------------------------
// EnrichmentEventSink impl
// ---------------------------------------------------------------------------

impl EnrichmentEventSink for RecordingSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {
        // v0.2.3 SENTINEL — this method MUST NOT fire in v0.2.3.
        // on_community_updated is deferred to ADR-050 (dream-pass sprint).
        // Recording the event so `sink_community_updated_does_not_fire_in_v023`
        // can assert its absence structurally.
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::CommunityUpdated);
    }

    fn on_batch_phase2_complete(&self, event: BatchPhase2Complete) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(SinkEvent::BatchComplete {
                batch_id: event.batch_id,
                succeeded: event.succeeded,
                failed: event.failed,
            });
    }
}

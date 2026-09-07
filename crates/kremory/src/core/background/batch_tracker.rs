//! `BatchProgress` — in-memory tracker for `on_batch_phase2_complete` terminal detection.
//!
//! ## Race-safety invariant
//!
//! `total` MUST be incremented (under the mutex) BEFORE the work item is
//! enqueued on the channel.  Only after both the increment AND the enqueue
//! can Phase 2 processing begin, so `succeeded + skipped + failed >= total`
//! can never fire prematurely.
//!
//! ## Mutex choice
//!
//! `std::sync::Mutex` (not `tokio::sync::Mutex`) — the critical sections are
//! O(1) insert / increment operations; no `.await` is ever held across the
//! lock.  This matches the `NamespacePolicyCache` precedent in `schema.rs`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// BatchProgress
// ---------------------------------------------------------------------------

/// Per-batch terminal-detection accumulator.
///
/// One entry per active `batch_id`.  Entries are removed from the map after
/// [`is_terminal`](BatchProgress::is_terminal) returns `true` and
/// `on_batch_phase2_complete` has been fired.
///
/// `pub` fields allow the worker-loop caller to read accumulated counts when
/// building the [`BatchPhase2Complete`](crate::memory::events::BatchPhase2Complete)
/// payload without taking a second lock.
pub struct BatchProgress {
    /// Total episodes registered for this batch (incremented BEFORE enqueue).
    pub total: usize,
    /// Episodes whose Phase 2 completed successfully.
    pub succeeded: usize,
    /// Episodes where Phase 2 was not requested (`enrich_per_episode = false`).
    pub skipped: usize,
    /// Episodes whose Phase 2 failed.
    pub failed: usize,
    /// Monotonic instant captured at batch-entry creation (first episode
    /// enqueue).  Subsequent `and_modify` calls for the same batch do NOT
    /// overwrite this — `started_at` marks when the batch began, not the
    /// last episode added.
    ///
    /// Used to compute `duration_ms` in `BatchPhase2Complete`.
    pub started_at: std::time::Instant,
}

impl BatchProgress {
    /// Construct a new tracker entry for the first episode in a batch.
    ///
    /// `total` starts at 1; caller increments it for each subsequent episode
    /// BEFORE that episode is enqueued (race-safety invariant).
    pub fn new() -> Self {
        Self {
            total: 1,
            succeeded: 0,
            skipped: 0,
            failed: 0,
            started_at: std::time::Instant::now(),
        }
    }

    /// `true` when every registered episode has reached a terminal state.
    ///
    /// Terminal when `succeeded + skipped + failed >= total`.
    /// Because `total` is always incremented BEFORE enqueue the count can
    /// only reach `total` once all work items have been processed.
    pub fn is_terminal(&self) -> bool {
        self.succeeded + self.skipped + self.failed >= self.total
    }
}

impl Default for BatchProgress {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BatchTracker alias
// ---------------------------------------------------------------------------

/// Shared in-memory tracker: `batch_id → BatchProgress`.
///
/// `Arc<Mutex<HashMap<…>>>` keeps `BackgroundIngestor` `Clone` while
/// sharing state between the handle (caller side) and the worker loop.
pub type BatchTracker = Arc<Mutex<HashMap<String, BatchProgress>>>;

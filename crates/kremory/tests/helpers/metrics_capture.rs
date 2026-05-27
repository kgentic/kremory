//! `MetricsCapture` — thin wrapper around `DebuggingRecorder` + `Snapshotter`
//! that makes the pattern used in `b1_observability.rs` re-usable across
//! LLM integration tests.

use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

/// Bundles a `DebuggingRecorder` with its `Snapshotter` so callers can pass
/// the recorder to `metrics::with_local_recorder` and later query it.
pub struct MetricsCapture {
    pub recorder: DebuggingRecorder,
    pub snapshotter: Snapshotter,
}

impl MetricsCapture {
    /// Create a fresh recorder + snapshotter pair.
    pub fn new() -> Self {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        Self {
            recorder,
            snapshotter,
        }
    }

    /// Return the names of all counters visible in the current snapshot.
    pub fn counter_names(&self) -> Vec<String> {
        self.snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(k, _, _, _)| k.key().name().to_string())
            .collect()
    }

    /// Return `true` if `name` appears at least once in the snapshot.
    pub fn has_counter(&self, name: &str) -> bool {
        self.counter_names().iter().any(|n| n == name)
    }
}

impl Default for MetricsCapture {
    fn default() -> Self {
        Self::new()
    }
}

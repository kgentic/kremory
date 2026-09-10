#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-251 cause 2 — a cancel that lands AFTER its task finished must not wedge
//! the batch.
//!
//! `BatchStatus::is_done()` is `completed + skipped + failed == total` — strict
//! equality. Any DOUBLE-count of a single run (the task recording `completed`
//! AND `graph_cancel` recording `failed` for the same `run_id`) pushes the sum
//! one past `total`, which strict equality can never satisfy again. The batch
//! is then permanently non-terminal and `await_batch` burns its whole timeout.
//!
//! The window is small — it opens once the spawned task is past its last
//! `.await` (so `abort()` cannot stop it) but before it has removed its own
//! abort handle — so this is a REPEATED-TRIAL test, not a single shot.
//!
//! Mocking boundary: ALWAYS REAL — `Memory`, engine, batch accounting.
//! ALWAYS MOCK — `ChatProvider` (`MockChatProvider::null`), embeddings
//! (`NullEmbeddingProvider`).

use std::sync::Arc;
use std::time::Duration;

use kremory::{DynEmbeddingProvider, Memory, Namespace};

fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Repeats the example's exact shape: four episodes into one batch, cancel one,
/// then `await_batch`. Every trial must reach a terminal batch.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_never_wedges_its_batch() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    const TRIALS: usize = 40;
    let mut wedged = Vec::new();
    let mut miscounted = Vec::new();

    for trial in 0..TRIALS {
        let dir = tempfile::tempdir().unwrap();
        let ns = Namespace::new("imports");
        let mem = Memory::open(dir.path().join("imports.db"))
            .default_namespace(ns.clone())
            .with_llm(stub_llm())
            .with_embedder(null_embedder())
            .await
            .unwrap();

        let batch = format!("import-run-{trial}");
        let mut commits = Vec::new();
        for i in 1..=4 {
            commits.push(
                mem.send_batched(
                    format!("Import record {i}: Ottoline Vance signed the carriage manifest."),
                    batch.clone(),
                )
                .await
                .unwrap(),
            );
        }

        let _ = mem.cancel(&commits[1]).await.unwrap();

        match mem.await_batch(&batch, Duration::from_secs(2)).await {
            Ok(s) => {
                // SENSITIVITY: the timeout assertion below is NOT enough on its
                // own. `is_done()` uses `>=`, so a double-counted run still
                // reads terminal and the batch never hangs — the accounting is
                // simply wrong and silent. Measured: with the claim protocol in
                // `engine_handle` neutered but `>=` in place, the hang test
                // passes 40/40. This is the assertion that stays sensitive to
                // the ACTUAL defect, so a future regression of the claim cannot
                // hide behind the tolerant comparison.
                let recorded = s.completed + s.skipped + s.failed;
                if recorded != s.total {
                    miscounted.push(format!(
                        "trial {trial}: total={} completed={} skipped={} failed={} (sum {recorded})",
                        s.total, s.completed, s.skipped, s.failed
                    ));
                }
            }
            Err(e) => wedged.push(format!("trial {trial}: {e}")),
        }
    }

    assert!(
        wedged.is_empty(),
        "a cancel must never leave its batch permanently non-terminal — \
         {} of {TRIALS} trials wedged: {wedged:?}",
        wedged.len()
    );
    assert!(
        miscounted.is_empty(),
        "exactly one terminal outcome must be recorded per run — a cancel and \
         its own task must not both account for the same episode: {miscounted:?}"
    );
}

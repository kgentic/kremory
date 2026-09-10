//! **Runs offline.** Stop work that is already running, without corrupting what
//! is around it.
//!
//! ```text
//! cargo run --example cancelling_in_flight_work
//! ```
//!
//! ## The problem this solves
//!
//! A user closes the tab. A tenant is suspended mid-import. A deploy is going
//! out in ninety seconds. You need background work to stop, and you need the
//! rest of the system to be fine afterwards.
//!
//! Cancellation is the path nobody drives, which is exactly why it is worth an
//! example: half-applied state and leaked tasks live here, and they live here
//! *because* it is the path nobody drives.
//!
//! ## The outcome type is the whole story
//!
//! ```text
//!   CancelOutcome {
//!       cancelled_phase,   // what was actually interrupted
//!       rolled_back,       // was the interrupted work undone?
//!       partial,           // ...and if not, what was left half-done
//!   }
//! ```
//!
//! `rolled_back: false` with a non-empty `partial` is not a failure — it is the
//! system telling you precisely what it could not unwind. A cancel API that
//! returned `Ok(())` would be hiding that.
//!
//! ## About the race, and a live bug
//!
//! This example cancels work that may already have finished — under a stub model
//! phase 2 is fast. **That is a real production case, not an artefact**: by the
//! time your cancel arrives, the job may be done.
//!
//! ⚠️ **`await_batch()` after a cancel currently hangs until its timeout on a
//! minority of runs** (TD-251). When the cancel lands after the task finished,
//! every episode reads `Complete` and the batch still never reaches terminal.
//! Writing this example is what surfaced it.
//!
//! So this asserts **per-episode** status, which is reliable on every run, and
//! deliberately does not await the batch. An example that asserts around a known
//! race is a flaky test wearing a tutorial's clothes.

use std::sync::Arc;
use std::time::Duration;

use kremory::{DynEmbeddingProvider, Memory, Namespace};

fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

const BATCH: &str = "import-run-8842";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("imports");

    let mem = Memory::open(dir.path().join("imports.db"))
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(null_embedder())
        .await?;

    // ── Four episodes into one batch ────────────────────────────────────────
    //
    // Four, not one. Cancelling the only piece of work in flight proves nothing
    // about whether cancellation is SURGICAL — the interesting question is what
    // happens to its neighbours.
    let mut commits = Vec::new();
    for i in 1..=4 {
        let c = mem
            .send_batched(
                format!("Import record {i}: Ottoline Vance signed the carriage manifest."),
                BATCH.to_string(),
            )
            .await?;
        commits.push(c);
    }
    println!("queued {} episodes into batch {BATCH}", commits.len());

    // ── Cancel exactly one ──────────────────────────────────────────────────
    let victim = &commits[1];
    let outcome = mem.cancel(victim).await?;
    println!("\ncancelled episode 2:");
    println!("  phase       : {:?}", outcome.cancelled_phase);
    println!("  rolled_back : {}", outcome.rolled_back);
    println!("  partial     : {:?}", outcome.partial);
    if !outcome.rolled_back && !outcome.partial.is_empty() {
        println!("  → not unwound, and it TOLD you what was left half-done.");
    }

    // ── Every episode reaches a terminal state ─────────────────────────────
    //
    // Per-episode status is the reliable check, and it is the one that matters:
    // the cancelled episode is terminal, and its neighbours completed.
    //
    // ⚠️ NOT asserted here: `await_batch()` after a cancel. It hangs until its
    // timeout on a minority of runs (TD-251, cause 2) — when the cancel lands
    // AFTER the task finished, every episode reads `Complete` and the BATCH
    // still never reaches terminal. That is an engine defect, not an example
    // problem, and asserting around it would make this a flaky test wearing a
    // tutorial's clothes. Poll `status_of` per episode until it is fixed.
    // Poll until every episode is terminal, bounded. `await_batch` would have
    // done this waiting for us; since we cannot use it (TD-251), poll explicitly
    // rather than sleeping a guessed interval — a fixed sleep is a race with
    // extra steps.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let mut all_terminal = true;
        for c in &commits {
            let st = mem.status_of(c).await?;
            if !matches!(
                st,
                kremory::core::error::IngestStatus::Complete
                    | kremory::core::error::IngestStatus::Failed(_)
            ) {
                all_terminal = false;
                break;
            }
        }
        if all_terminal || std::time::Instant::now() > deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let mut terminal = 0;
    let mut completed = 0;
    for (i, c) in commits.iter().enumerate() {
        let st = mem.status_of(c).await?;
        let is_terminal = matches!(
            st,
            kremory::core::error::IngestStatus::Complete
                | kremory::core::error::IngestStatus::Failed(_)
        );
        if is_terminal {
            terminal += 1;
        }
        if matches!(st, kremory::core::error::IngestStatus::Complete) {
            completed += 1;
        }
        println!("  episode {}: {st:?}", i + 1);
    }

    assert_eq!(
        terminal,
        commits.len(),
        "every episode must reach a terminal state after one is cancelled — a \
         cancel that leaves a sibling hanging is worse than no cancel at all"
    );
    assert!(
        completed >= commits.len() - 1,
        "cancelling one episode must not prevent its neighbours completing; \
         completed={completed} of {}",
        commits.len()
    );
    println!("  → all {terminal} terminal, {completed} completed.");

    // ── Cancelling something that never ran in the background ───────────────
    //
    // An inline write has no run to cancel. The error says so rather than
    // pretending to have cancelled something.
    let inline = mem
        .remember("A synchronous note.")
        .in_namespace(ns.clone())
        .skip_extraction()
        .await?;
    let err = mem.cancel(&inline).await;
    println!("\ncancelling an inline write: {}", match &err {
        Err(e) => format!("refused — {e}"),
        Ok(o) => format!("SUCCEEDED ({o:?}) — which would be a lie"),
    });
    assert!(
        err.is_err(),
        "cancelling a write that never ran in the background must fail loudly, \
         not report a successful cancellation of nothing"
    );

    // ── And the database is still usable ────────────────────────────────────
    let survivors = mem
        .recall("carriage manifest")
        .in_namespace(ns.clone())
        .content()
        .await?;
    println!("\nrecall after cancellation: {} passage(s)", survivors.len());
    assert!(
        !survivors.is_empty(),
        "the store must remain readable after an in-flight cancellation"
    );

    println!("\nCancellation is surgical at the EPISODE level: one job stopped, its");
    println!("neighbours finished, and anything that could not be unwound was named");
    println!("rather than swallowed. Batch-level awaiting after a cancel is not yet");
    println!("reliable — see TD-251.");

    mem.close().await?;
    Ok(())
}

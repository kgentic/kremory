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
//! ## About the race
//!
//! This example cancels work that may already have finished — under a stub model
//! phase 2 is fast. **That is a real production case, not an artefact**: by the
//! time your cancel arrives, the job may be done.
//!
//! Either way the batch still reaches a terminal state, so `await_batch()` is
//! the right way to wait for it. Exactly one outcome is recorded per episode —
//! whichever of the cancel and the episode's own task gets there first — so the
//! cancelled episode reads `Failed("cancelled by caller")` when the cancel
//! arrived in time and `Complete` when it did not. Both are honest; neither
//! leaves the batch hanging.
//!
//! Writing this example is what surfaced TD-251, where a late cancel and the
//! finishing task BOTH counted the same episode and `await_batch()` then blocked
//! for its whole timeout on 22 of 40 runs. Fixed 2026-09-10.

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

    // ── Wait for the whole batch, cancelled member included ─────────────────
    //
    // Cancelling a member must not stop the batch reaching a terminal state —
    // "a cancel that wedges its own batch is worse than no cancel at all". The
    // counts below are the proof: every episode is accounted for exactly once.
    let status = mem.await_batch(BATCH, Duration::from_secs(10)).await?;
    println!(
        "\nbatch terminal: total={} completed={} skipped={} failed={}",
        status.total, status.completed, status.skipped, status.failed
    );
    assert_eq!(
        status.completed + status.skipped + status.failed,
        status.total,
        "every episode must be accounted for exactly once — a cancel and the \
         episode's own task must not both record an outcome for it"
    );

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
    println!("rather than swallowed. The batch still reaches a terminal state, so");
    println!("await_batch() is safe to use after a cancel.");

    mem.close().await?;
    Ok(())
}

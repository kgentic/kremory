//! **Runs offline.** Accept the write, return to your caller, finish the
//! expensive part in the background.
//!
//! ```text
//! cargo run --example ingest_without_blocking
//! ```
//!
//! ## The problem this solves
//!
//! Ingest has two phases. Phase 1 stores the episode — fast, and the part your
//! user is waiting on. Phase 2 enriches it: extraction, entity resolution,
//! embeddings. Phase 2 is where the seconds go, because it usually calls a
//! model.
//!
//! Doing both inside an HTTP handler means your p99 is a language model's p99.
//! `.no_wait()` commits phase 1 and hands you a receipt for phase 2, so the
//! handler can return.
//!
//! ## The contract, and the part that surprises people
//!
//! `EpisodeCommit::run_id` is the receipt:
//!
//!   - `Some(id)` — phase 2 is running in the background. Track it with
//!     `status_of()`, or block on it later with `await_enrichment()`.
//!   - **`None` — there was no background work to do**, so there is nothing to
//!     track and `status_of()` will tell you so rather than invent a status.
//!
//! You get `None` when nothing needed enriching — most obviously under
//! `.skip_extraction()`, where you supplied the facts yourself and phase 2 has
//! no job. Both cases are shown below, because assuming a `run_id` is always
//! present is the mistake this example exists to prevent.

use std::sync::Arc;
use std::time::Duration;

use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

/// A stand-in chat provider. Real deployments pass a real model here; the
/// asynchronous CONTRACT is identical either way, which is why this example can
/// demonstrate it without one.
fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("inbox");

    let mem = Memory::open(dir.path().join("inbox.db"))
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(null_embedder())
        .await?;

    // ── The handler path: accept and return ─────────────────────────────────
    let started = std::time::Instant::now();
    let commit = mem
        .remember("Marguerite chairs the safety review board at Ardent Rail.")
        .in_namespace(ns.clone())
        .no_wait()
        .await?;
    let handler_time = started.elapsed();

    println!("write accepted in {handler_time:?} — this is what your caller waits for");
    println!("  episode entity : {}", commit.episode_entity_id);
    println!("  run_id         : {:?}", commit.run_id);

    let run_id = commit
        .run_id
        .ok_or_else(|| anyhow::anyhow!("expected a background run for an un-pinned write"))?;
    println!("  → phase 2 is running in the background as {run_id}");

    // ── Later: check on it, or wait for it ──────────────────────────────────
    //
    // `status_of` is the poll. Use it for a progress endpoint, or to decide
    // whether a read is safe yet.
    let status = mem.status_of(&commit).await?;
    println!("\npolled status   : {status:?}");

    // `await_enrichment` is the block. Note the TIMEOUT is required — an
    // unbounded wait on a background job is how a request handler hangs.
    let terminal = mem
        .await_enrichment(&commit, Duration::from_secs(30))
        .await?;
    println!("terminal status : {terminal:?}");

    // Only now is enrichment guaranteed complete. Before this point a read is
    // valid but may not yet see phase-2 output — that is the trade you accepted
    // by not waiting.
    let found = mem
        .recall("Marguerite")
        .in_namespace(ns.clone())
        .content()
        .await?;
    assert!(
        !found.is_empty(),
        "after enrichment reaches a terminal status the episode must be retrievable"
    );
    println!("post-enrichment : {} passage(s) retrievable", found.len());

    // ── The other case: nothing to wait for ─────────────────────────────────
    //
    // With the facts supplied by the caller there is no phase-2 work, so there
    // is no run to track. `run_id` is None and that is CORRECT — it is not a
    // failure, and `status_of` on it would tell you phase 2 was inline rather
    // than making a status up.
    let pinned = mem
        .remember("Ardent Rail operates the northern freight corridor.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "ardent rail".into(),
            predicate: "operates".into(),
            object: "northern freight corridor".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .no_wait()
        .await?;

    println!("\npinned write    : run_id = {:?}", pinned.run_id);
    assert!(
        pinned.run_id.is_none(),
        "a write with nothing to enrich should report no background run, not a \
         run that instantly completes — got {:?}",
        pinned.run_id
    );
    println!("  → nothing ran in the background, so there is nothing to await.");

    println!("\nCheck `run_id` before assuming there is something to wait for.");
    println!("`Some` means poll or await it; `None` means the work is already done.");

    mem.close().await?;
    Ok(())
}

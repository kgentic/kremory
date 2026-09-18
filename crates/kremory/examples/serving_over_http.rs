//! **Runs offline.** The serving shape: one shared `Memory` behind request
//! handlers, with the write taken off the request path.
//!
//! ```text
//! cargo run --example serving_over_http
//! ```
//!
//! ## The problem this solves
//!
//! kremory ships as a **library**. The axum REST server in this repo
//! (`kremory-http`) is `publish = false` and reaches nobody, because HTTP here
//! is a deployment MODE, not a fourth SDK. So if you want a REST surface, you
//! write the handlers — and two questions decide whether that works:
//!
//! 1. **What do the handlers share?** One `Memory`, opened at startup and held
//!    for the life of the process. A handle owns exactly one libsql connection
//!    and there is no pool, so "open one per request" is not a tuning lever —
//!    it is a connection per request. When you need more concurrency the answer
//!    is more handles; `two_handles_one_database.rs` shows that shape.
//!
//! 2. **What does `POST` return?** Not the result. Ingest is LLM-bound and
//!    takes **seconds, not milliseconds**: measured at ~20 s per chunk, 37.5 s
//!    per session end to end on a hosted model and 119 s all-local
//!    (`.ai-docs/SYSTEM-PRIMER.md` §2). A handler that awaited it would hold an
//!    HTTP request open for that long, and no amount of hardware fixes it —
//!    that is a language model's latency, not yours. So `POST` accepts the
//!    write, hands back a receipt, and the caller polls. `.no_wait()` is how.
//!
//! ## The part that surprises people: a 202 means *accepted*, not *stored*
//!
//! It is tempting to tell your callers "we've written it, enrichment follows".
//! That is not what happens. On the `.no_wait()` path the **whole** ingest —
//! the episode INSERT included — is spawned into a background task, and the
//! handler returns before any of it runs (`memory/engine_handle.rs:282-428`).
//!
//! Two consequences your API contract has to carry:
//!
//! - **A `GET` immediately after a `202` can legitimately return nothing.** The
//!   run below prints exactly that. Reads become reliable once the run reaches
//!   a terminal status, which is what the status route is for.
//! - **`EpisodeCommit::episode_entity_id` is the run id on this path**, not an
//!   episode id — it is literally `run_id.to_string()`
//!   (`memory/engine_handle.rs:418`). Handing it to a client as a durable
//!   resource id gives them a job ticket dressed as a record.
//!
//! Neither is a defect; both are the price of not blocking, and both are quiet
//! rather than loud, which is why this example asserts them.
//!
//! ## Why there is no HTTP framework in here
//!
//! Deliberately omitted. Pulling axum (or actix, or hyper) into an example
//! would add a dependency, a port and a shutdown story to a crate that needs
//! none of them, and would bury the decisions above under routing boilerplate.
//! The functions below are plain `async fn`s named for the routes they would
//! serve. Binding them to your framework is the part your framework's own
//! documentation already covers; the part it cannot tell you is what to share
//! and what to return, which is what this file is about.
//!
//! ## What this asserts
//!
//! - `POST` returns a **tracking handle** rather than finished work: `run_id`
//!   is `Some`, which is only possible because the handler did not block.
//! - That handle is a **run** id, not an episode id — asserted, because the
//!   field it arrives in is called `episode_entity_id`.
//! - The read route is reliable **after** the run reaches a terminal status,
//!   and the example shows what it returns before that.
//! - The status route answers for a handle it issued and **refuses to invent
//!   one** for a job it never saw (404, not a fabricated `Pending`).
//! - Concurrent requests share ONE handle and each gets a **distinct** receipt;
//!   nothing reopens the database.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kremory::core::error::IngestStatus;
use kremory::memory::types::EpisodeCommit;
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace};
use uuid::Uuid;

const DEMO_DIM: usize = 16;

/// Deterministic stand-in embedder — see `offline_remember_recall.rs`. Real
/// deployments pass their embedding backend here; nothing about the serving
/// shape changes.
struct DemoEmbedder {
    dim: usize,
}

impl EmbeddingProvider for DemoEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move {
            let mut v = vec![0f32; dim];
            for (i, b) in text.bytes().enumerate() {
                v[i % dim] += f32::from(b) / 255.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            Ok(v)
        }
    }
}

/// A stand-in chat provider, exactly as `ingest_without_blocking.rs` uses. It
/// extracts nothing, so this example needs no model and no network — but the
/// ASYNC CONTRACT it exercises is the real one.
///
/// Note it is a stub LLM rather than a custom `.with_extractor(..)`: background
/// ingest resolves entities through `ingest_with`, which requires an LLM
/// provider to be wired even when a bring-your-own extractor is supplied.
fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

// ── What a framework would call "application state" ─────────────────────────

/// Built once at startup, shared by every request.
///
/// `jobs` is the piece people forget. `Memory::status_of` takes the
/// `EpisodeCommit` the write produced, not a bare id, so a server that hands a
/// `run_id` to a client must keep the commit somewhere to look it up again. A
/// `HashMap` behind a `std::sync::Mutex` is enough here; a deployment that must
/// survive a restart puts this in a table.
struct AppState {
    mem: Memory,
    ns: Namespace,
    jobs: Mutex<HashMap<Uuid, EpisodeCommit>>,
}

/// `202 Accepted` — what `POST /memories` gives the caller instead of the work.
#[derive(Debug)]
struct Accepted {
    /// Named as the field it came from. See the module doc: on this path it is
    /// the run id, which is why this example asserts as much.
    episode_entity_id: String,
    run_id: Uuid,
}

// ── The handlers ────────────────────────────────────────────────────────────

/// `POST /memories` — accept the write, return a receipt.
///
/// The whole point of this function is the `await_enrichment` call it does NOT
/// make. `.no_wait()` spawns the ingest and returns a `run_id` for the caller
/// to poll.
///
/// `run_id: None` means there was nothing to enrich — a real possibility under
/// `.skip_extraction()`, where the caller supplied the facts — so a server has
/// to decide what that means on its route. Here it is an error, because this
/// route promises a pollable handle.
async fn handle_post_memories(state: &AppState, body: &str) -> anyhow::Result<Accepted> {
    let commit = state
        .mem
        .remember(body)
        .in_namespace(state.ns.clone())
        .no_wait()
        .await?;

    let run_id = commit.run_id.ok_or_else(|| {
        anyhow::anyhow!("this route promises a pollable handle; got a commit with no run_id")
    })?;

    // Keep the commit so `GET /memories/{run_id}` can resolve it later.
    state
        .jobs
        .lock()
        .map_err(|_| anyhow::anyhow!("job table poisoned"))?
        .insert(run_id, commit.clone());

    Ok(Accepted {
        episode_entity_id: commit.episode_entity_id,
        run_id,
    })
}

/// `GET /memories/{run_id}` — the progress route.
///
/// `Ok(None)` is a 404. A status endpoint that answers for an id it never
/// issued is worse than one that errors, because the client believes it.
async fn handle_get_ingest_status(
    state: &AppState,
    run_id: Uuid,
) -> anyhow::Result<Option<IngestStatus>> {
    let commit = {
        let jobs = state
            .jobs
            .lock()
            .map_err(|_| anyhow::anyhow!("job table poisoned"))?;
        match jobs.get(&run_id) {
            Some(c) => c.clone(),
            None => return Ok(None),
        }
    };
    Ok(Some(state.mem.status_of(&commit).await?))
}

/// `GET /recall?q=…` — the read route, returning the prompt-ready rendering.
///
/// Reads are the cheap direction: no model call, no background work, safe to
/// serve synchronously. That asymmetry is the design, not an oversight —
/// writes go async because they are slow; reads do not need to.
async fn handle_get_recall(state: &AppState, query: &str) -> anyhow::Result<String> {
    Ok(state
        .mem
        .recall(query)
        .in_namespace(state.ns.clone())
        .await?)
}

/// `GET /recall/passages?q=…` — the structured variant, for callers that want
/// the matching episodes rather than a rendered block of prompt text.
async fn handle_get_passages(state: &AppState, query: &str) -> anyhow::Result<usize> {
    Ok(state
        .mem
        .recall(query)
        .in_namespace(state.ns.clone())
        .content()
        .await?
        .len())
}

/// Not a route. This is the worker side of the contract — a background loop, a
/// webhook, or the client's own polling. The timeout is mandatory: an unbounded
/// wait on a background job is how a request handler hangs.
async fn drain_job(state: &AppState, run_id: Uuid) -> anyhow::Result<IngestStatus> {
    let commit = {
        let jobs = state
            .jobs
            .lock()
            .map_err(|_| anyhow::anyhow!("job table poisoned"))?;
        jobs.get(&run_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no job for {run_id}"))?
    };
    Ok(state
        .mem
        .await_enrichment(&commit, Duration::from_secs(30))
        .await?)
}

// ── "Startup", then a few "requests" ────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("service");

    let mem = Memory::open(dir.path().join("service.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .await?;

    let state = Arc::new(AppState {
        mem,
        ns: ns.clone(),
        jobs: Mutex::new(HashMap::new()),
    });
    println!("service up — ONE Memory handle, shared by every handler");

    // ── POST /memories ──────────────────────────────────────────────────────
    let started = std::time::Instant::now();
    let accepted = handle_post_memories(
        &state,
        "Marguerite chairs the safety review board at Ardent Rail.",
    )
    .await?;
    let handler_time = started.elapsed();

    println!("\nPOST /memories -> 202 Accepted in {handler_time:?}");
    println!("  run_id            : {}", accepted.run_id);
    println!("  episode_entity_id : {}", accepted.episode_entity_id);
    println!("  → the receipt is what your caller waits for, NOT the extraction");

    assert_eq!(
        accepted.episode_entity_id,
        accepted.run_id.to_string(),
        "on the .no_wait() path `episode_entity_id` IS the run id \
         (memory/engine_handle.rs:418). If this ever stops holding, servers \
         that published it as a resource id need to hear about it; got {:?}",
        accepted.episode_entity_id
    );

    // ── GET immediately after the 202 ───────────────────────────────────────
    //
    // Printed, not asserted, because it is a race by construction — and that is
    // the point. The ingest is a spawned task; whether it has reached the
    // episode INSERT by now is not something your API can promise.
    let straight_away = handle_get_passages(&state, "Ardent Rail").await?;
    println!("\nGET /recall/passages?q=Ardent+Rail (immediately) -> {straight_away} passage(s)");
    println!("  → a 202 means ACCEPTED, not STORED. Do not promise read-back yet.");

    // ── GET /memories/{run_id} — the progress route ─────────────────────────
    let status = handle_get_ingest_status(&state, accepted.run_id).await?;
    println!("GET /memories/{} -> {status:?}", accepted.run_id);
    assert!(
        status.is_some(),
        "the status route must resolve a handle it issued itself"
    );

    let unknown = Uuid::new_v4();
    let missing = handle_get_ingest_status(&state, unknown).await?;
    println!("GET /memories/{unknown} -> {missing:?}  (404)");
    assert!(
        missing.is_none(),
        "a status route must 404 on an id it never issued rather than invent a \
         status; got {missing:?}"
    );

    // ── Concurrent requests over ONE handle ─────────────────────────────────
    //
    // Three "requests" in flight at once, each cloning the Arc. No handler
    // opens a database; there is one connection underneath all of them, which
    // is exactly why the write had to leave the request path.
    let bodies = [
        "The northern corridor reopened on Tuesday.",
        "Ardent Rail runs freight overnight.",
        "The safety review board meets monthly.",
    ];
    let mut tasks = Vec::new();
    for body in bodies {
        let state = Arc::clone(&state);
        tasks.push(tokio::spawn(async move {
            handle_post_memories(&state, body).await
        }));
    }
    let mut run_ids = Vec::new();
    for t in tasks {
        run_ids.push(t.await??.run_id);
    }
    println!("\n3 concurrent POSTs -> {} receipts", run_ids.len());
    let issued = run_ids.len();
    run_ids.sort();
    run_ids.dedup();
    assert_eq!(
        run_ids.len(),
        issued,
        "receipts must be distinct — a shared run_id would make the status \
         route answer about the wrong job"
    );
    assert_eq!(issued, 3, "every concurrent request gets its own receipt");

    // ── The worker drains the jobs ──────────────────────────────────────────
    let terminal = drain_job(&state, accepted.run_id).await?;
    println!("worker awaited {} -> {terminal:?}", accepted.run_id);
    assert!(
        matches!(terminal, IngestStatus::Complete),
        "the stub pipeline must reach Complete; a Failed here means the \
         offline wiring broke, not that your handler did. got {terminal:?}"
    );
    for id in &run_ids {
        drain_job(&state, *id).await?;
    }

    // ── Now the read route is reliable ──────────────────────────────────────
    let passages = handle_get_passages(&state, "Ardent Rail").await?;
    println!("\nGET /recall/passages?q=Ardent+Rail (after terminal) -> {passages} passage(s)");
    assert!(
        passages > 0,
        "once a run is terminal its episode MUST be retrievable — if this \
         fails, the receipt does not mean what the status route says it means"
    );

    let rendered = handle_get_recall(&state, "Ardent Rail").await?;
    println!("\nGET /recall?q=Ardent+Rail\n{rendered}");
    assert!(
        !rendered.is_empty(),
        "the read route must return prompt-ready text once the run is terminal"
    );

    println!("Share one handle. Return a receipt, not the result. Poll the receipt.");
    println!("The HTTP framework is the easy part, and is deliberately not here.");

    state.mem.close().await?;
    Ok(())
}

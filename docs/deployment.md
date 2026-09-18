# Deploying kremory

kremory is an **embedded library**, not a server. There is no daemon to run and no network hop
between your process and your memory. That is the whole point — and it is also why deployment
questions that a database would answer for you become yours to answer.

This page states the three constraints that actually shape a deployment. Each one is verified
against the code, with the file and line, because a deployment doc that is optimistic is worse
than no deployment doc at all.

> **HTTP is a deployment MODE, not a fourth SDK.** kremory publishes as a library. The repo also
> contains `kremory-http`, a full axum REST server, but it is marked `publish = false` and ships
> to nobody. If you want HTTP, you write the handler — see
> [`examples/serving_over_http.rs`](../crates/kremory/examples/serving_over_http.rs). This mirrors
> the ruling already taken for MCP: an agent/transport surface, not a separate SDK with its own
> parity obligations.

---

## 1. One connection per handle. Reads *and* writes. There is no pool.

A `Memory` handle owns exactly one libsql connection:

```rust
// crates/kremory/src/core/schema.rs:411
pub conn: libsql::Connection,
```

There is no connection pool anywhere in the crate — no `r2d2`, no `deadpool`, no `bb8`. Every
read and every write on a handle goes down that single connection.

**What this means for you.** Concurrency is bounded by what one connection can do. A recall issued
while an ingest holds a write transaction waits. If you are serving many concurrent readers, the
lever is **more handles**, not a bigger pool — and handles are cheap to open on the same file (see
[`examples/two_handles_one_database.rs`](../crates/kremory/examples/two_handles_one_database.rs)).

**What this does *not* mean.** It is not a correctness problem. Serialisation is deliberate:
`BEGIN IMMEDIATE` on one connection is what keeps concurrent writes from interleaving.

## 2. The write lock is **per handle**, not per process — and it is narrower than it sounds.

Writes are serialised by:

```rust
// crates/kremory/src/core/schema.rs:415
write_lock: Arc<AsyncMutex<()>>,
```

That is a `tokio::sync::Mutex`, and the architecture note is explicit that it is
*"acquired first — before any sub-lock — on every write path"*
(`crates/kremory/src/core/mod.rs:22`).

**Read that carefully — it is narrower than it looks.** That mutex is **per handle**, not per
process: `TemporalGraph::open_with_dim` mints a fresh one every time
(`crates/kremory/src/core/schema.rs:460`, and again at `:478`). So it excludes *tasks sharing one
handle*. **Two handles in the same process each have their own**, and between separate OS processes
it contributes nothing at all.

Arbitration beyond one handle is therefore SQLite/libsql's: file locking, separate WAL index
mappings, separate page caches, and `PRAGMA busy_timeout = 5000` (`core/schema.rs:456`).

**Status, corrected 2026-09-18: cross-process write-visibility is now TESTED and it works.**
`crates/kremory/tests/it/cross_process_one_database.rs` spawns two real OS processes over one
database file and asserts a write in one becomes visible in the other. It ships with a permanent
negative control — a second test pointing the reader at a *different* file and asserting it finds
nothing — so the passing case cannot be a false positive.

What that does and does not license:

- ✅ **Exercised**: many handles, many tasks, one process.
- ✅ **Exercised**: a write in process A becoming visible to process B over one file.
- ✅ **Fine**: multiple processes reading while one writes — ordinary SQLite WAL behaviour.
- ⚠️ **Still unverified**: *sustained concurrent* writing from multiple processes under load.
  Visibility is proven; contention behaviour at volume is not. `busy_timeout` is 5s, so a writer
  held off longer than that surfaces as an error rather than a wait.

If your deployment needs heavy multi-process writes, still prefer one writer (a single ingest
worker) with the others reading. That sidesteps contention rather than betting on it.

## 3. Ingest takes **seconds**, not milliseconds. Writes must be asynchronous.

`remember()` runs extraction synchronously by default: a Phase-1 NER pass, then a **three-stage
LLM pipeline** (entities → relations → triplets), then resolution and dedup
(`crates/kremory/src/core/ingest/pipeline/ingest_with.rs`).

Measured, not estimated — from `.ai-docs/SYSTEM-PRIMER.md` §2:

| path | measured ingest |
|---|---|
| Groq (`openai/gpt-oss-120b`) | **37.5 s per session** (~9.3 chunks, 47 LLM calls, ~117k tokens) |
| all-local Ollama fallback | **119 s per session** — 3.2× slower |

**A synchronous POST handler would hold an HTTP request open for that long.** That is not a tuning
problem you can fix with a bigger box; it is LLM latency and it is inherent.

So a serving deployment must take writes off the request path:

```rust
let commit = memory
    .remember(body)
    .in_namespace(ns.clone())
    .no_wait()
    .await?;                       // terminal is `.await` — `execute()` is NOT public here
let run_id = commit.run_id;        // Option — `None` means there was nothing to enrich
// return 202 Accepted + run_id immediately; poll it on another route
```

> ⚠️ **`.no_wait()` does NOT commit phase 1 before returning, despite what several
> doc-comments in this crate still say.** The background path returns before the ingest
> task resolves — `memory/engine_handle.rs:420-422` says so in its own words:
> *"Background-spawn path: returns before the task (and any embed attempt inside it)
> resolves"*. Measured over 5 runs, a read issued immediately after the 202 returned
> **0 passages twice and 1 passage three times**. That is a real race, not a warm-up
> artefact.
>
> **Do not tell a client its write is durable or searchable when you return the 202.**
> Return the `run_id` and let them poll. Known-false claims still in the tree, all
> outside this page's scope and recorded in the 2026-09-18 handoff:
> `examples/ingest_without_blocking.rs:16`, `memory/types.rs:849-851`, `memory/types.rs:855`.
>
> Also: on the `.no_wait()` path `EpisodeCommit::episode_entity_id` is **the run id**
> (`engine_handle.rs:418` returns `run_id.to_string()`). A server that publishes it as a
> durable resource identifier is handing clients a job ticket.

[`examples/serving_over_http.rs`](../crates/kremory/examples/serving_over_http.rs) shows this
shape end to end, and [`examples/ingest_without_blocking.rs`](../crates/kremory/examples/ingest_without_blocking.rs)
covers `.no_wait()` on its own.

> **Cost is not incidental either.** That same measurement puts ingest at **47 LLM calls and a 48×
> token amplification** per session. If you are sizing a deployment, size the extraction bill
> first — it dominates everything else kremory does.

---

## Choosing a shape

| shape | when | watch out for |
|---|---|---|
| **One process, many handles** | the default; API server, worker threads, request handlers | bounded by one connection per handle — open more handles, not a pool |
| **One writer + N readers** | you need multiple processes | the multi-process shape that avoids contention entirely; write-visibility across processes is tested |
| **Many writers, many processes** | heavy concurrent ingest | visibility is **tested**; *sustained contention* is **not**. `write_lock` is per handle and helps you not at all here — `busy_timeout` is 5s, past which a blocked writer errors rather than waits. |

## What this page deliberately does not tell you

- **Authentication and authorisation.** kremory has none. It is a library; the trust boundary is
  your process. Anything you expose over HTTP, you secure.
- **Horizontal scaling across machines.** One database file, one filesystem. Distributed
  deployment is not a solved problem here and pretending otherwise would be the optimism this
  page exists to avoid.
- **Whether `kremory-http` will be published.** It is `publish = false` today and needs an auth
  story plus a settled search-surface question before that could change.

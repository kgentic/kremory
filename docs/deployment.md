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

## 2. The write lock is an **in-process** mutex. Cross-process is UNTESTED.

Writes are serialised by:

```rust
// crates/kremory/src/core/schema.rs:415
write_lock: Arc<AsyncMutex<()>>,
```

That is a `tokio::sync::Mutex`, and the architecture note is explicit that it is
*"acquired first — before any sub-lock — on every write path"*
(`crates/kremory/src/core/mod.rs:22`).

**Read that carefully.** An in-process mutex excludes *tasks inside one process*. It provides
**no exclusion whatsoever between separate OS processes.** Two processes writing to one database
file are arbitrated by SQLite/libsql file locking alone, with kremory's own serialisation
contributing nothing.

**Honest status: this is untested, not "supported" and not "broken".** The repo has no test that
spans processes, so nobody here can tell you whether sustained concurrent cross-process writing is
safe under load. Treat it as unverified territory:

- ✅ **Safe and exercised**: many handles, many tasks, **one process**.
- ⚠️ **Unverified**: multiple processes writing the same file concurrently.
- ✅ **Fine**: multiple processes *reading* while one writes is ordinary SQLite WAL behaviour.

If your deployment needs multi-process writes, design one writer (a single ingest worker) and let
the others read. That sidesteps the question rather than betting on it.

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
let handle = memory.remember(text).no_wait().execute().await?;
// return 202 Accepted + handle id immediately; poll or await elsewhere
```

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
| **One writer + N readers** | you need multiple processes | the only multi-process shape that avoids §2's unverified territory |
| **Many writers, many processes** | — | **unverified.** No test covers it. Do not assume the in-process `write_lock` helps you; it does not. |

## What this page deliberately does not tell you

- **Authentication and authorisation.** kremory has none. It is a library; the trust boundary is
  your process. Anything you expose over HTTP, you secure.
- **Horizontal scaling across machines.** One database file, one filesystem. Distributed
  deployment is not a solved problem here and pretending otherwise would be the optimism this
  page exists to avoid.
- **Whether `kremory-http` will be published.** It is `publish = false` today and needs an auth
  story plus a settled search-surface question before that could change.

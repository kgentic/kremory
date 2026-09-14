# Python binding spike — 2026-09-14 · **GO**

Three spikes, pre-registered falsification conditions written **before** any code ran, judged
against those conditions afterwards. The artefacts here are the ones worth keeping; the 20 GB of
build scratch is not.

## Verdict: GO

| spike | pre-registered question | result |
|---|---|---|
| **A — callback bridge** | can a Python object satisfy `DynEmbeddingProvider` without `unsafe`, driven from a real asyncio loop? | **PASS** |
| **B — wheels** | does `libsql-ffi`'s native chain survive cross-compilation to manylinux? | **PASS, exceeded** |
| **C — async ergonomics** | async-only, sync-only, or both? | **BOTH, on one runtime and one handle** |

## Spike A — the bridge is a solved problem, and smaller than anyone assumed

`bridge.rs` (from `spike-a/src/lib.rs`) carries `#![forbid(unsafe_code)]` at line 4 **and compiles**
— so "no unsafe" is mechanical here, not an argument. `grep -n unsafe` returns three hits: the
attribute and two comments.

Measured, re-run by the judge rather than taken from the report:

```
driver.py 3 ok  ->  A_EXIT=0,  0.30s / 0.14s / 0.13s
py_calls=7  rust_calls=7        # Python counter == Rust AtomicUsize: the callback really ran
bridge_is_dyn_object: True
```

The embedder in `driver.py` is `async def embed(text)` — the **async** case the condition targeted,
not an easier synchronous substitution.

The only compile error hit was `E0521 borrowed data escapes outside of method` — a lifetime escape
from tokio's `'static` bound, **not** the `E0277 future cannot be sent between threads safely` that
would have been the no-go. The fix was three lines of owning the captures (`Py::clone_ref` under an
attached GIL, `Arc::clone`), both pre-registered as PASS shapes.

**Error propagation works.** A Python `ValueError` in the callback arrives as a Rust error with type
and message intact; a 19-dim return against `dim=16` is rejected before any write.

### The one rule that makes it work: `py.detach` around every `block_on`

Without it, an embedder invoked on a tokio worker **deadlocks**. Native `sample` caught the worker
at `PyGILState_Ensure -> take_gil` for **1665 of 1668 samples** while main sat parked in
`block_on`. The trap is that the naive spelling looks correct.

### One forbidden cell, and `py.detach` does NOT rescue it

**A blocking Python API called with an ASYNC Python embedder hangs forever.** An independent asyncio
heartbeat froze at 5 ticks with the embedder called 0 times — the loop's own thread is parked inside
Rust. The awaitable API is the only shape that can drive an async callback. This must be a typed
error at the boundary, not a hang.

## Spike B — the feared risk was aimed at the wrong dependency

**All four wheels built, installed, imported, and real-file smoke-tested on one M4 Max.** None is
impossible locally.

| # | wheel | how | time |
|---|---|---|---|
| 1 | `cp39-abi3-macosx_11_0_arm64` | native | 2m09s |
| 2 | `cp39-abi3-macosx_10_12_x86_64` | cross, stock Apple toolchain | 1m42s |
| 3 | `cp39-abi3-manylinux_2_17_aarch64` | native inside stock `ghcr.io/pyo3/maturin` | 1m51s |
| 4 | `cp39-abi3-manylinux_2_17_x86_64` | cross via `--zig` (`uv pip install ziglang`) | 2m29s |

`abi3-py39` collapses the Python-version axis entirely: **four wheels, not four × N interpreters.**
Cumulative cargo time 8m11s — a scripted release step, not a project.

> ⚠️ **`libsql-ffi` was the wrong thing to fear, and that error is worth carrying forward.**
> Its `build.rs` runs bindgen **only** under `LIBSQL_DEV`; otherwise it copies prebuilt bindings and
> compiles `sqlite3.c` with plain `cc`. **libclang is never needed.** The real cmake+cc dependency is
> `aws-lc-sys` arriving through a required kremory dep — and it cross-compiles fine.
> Record this in the SDK build docs or the next person re-derives the wrong central unknown, exactly
> as this brief did.

## Spike C — ship both spellings; sync is primary

`pyo3_async_runtimes::tokio::get_runtime()` returns **one** `&'static tokio::Runtime` that drives
both `future_into_py` (async) and `py.detach(|| rt.block_on(..))` (blocking). The same handle used
both ways in one process returned identical results — `mixed async+sync on one handle: (206, 206)`.

**There is no fork in the object model.** Sync and async are two spellings over one runtime and one
handle, so shipping both is additive rather than a parallel type hierarchy.

**Sync is primary** because it needs zero event-loop plumbing (17-line example), and it works from
inside a running `asyncio.run(main())` — the LangChain and notebook case. LangChain derives its
async methods from the sync ones via `run_in_executor`, so sync-first gives async away free.

Shape C (a dedicated loop thread) is **dropped**: 4 lines of user plumbing, buys nothing.

## Size: 3–5 weeks for a shippable v1

The bridge — the assumed risk — is **~90 lines of production Rust** inside a 346-line spike file
that also carries a do-nothing extractor, sync+async dispatch, dim validation, a pure-Rust control
and a deliberately-broken probe.

The real work is surface area: **~54 awaitable public entry points** (12 `IntoFuture` await-terminals
+ 42 `pub async fn`) across ~6 core types. Hand-writing both spellings doubles that to ~108, which is
the largest line-count driver — **design the generating macro before the fortieth method, not after.**

## Traps to carry into the build

1. **Workspace profile hazard.** The kremory workspace root sets `[profile.release] panic = "abort"`.
   A Python extension module must **unwind** into PyO3, not abort the interpreter. The spike crates
   lived outside the workspace and were unaffected; a `crates/kremory-py` inherits it. **Decide the
   crate's home before writing code** — cheap now, expensive later. See `spike-b-Cargo.toml`, which
   overrides it deliberately.
2. **Nested-runtime panic.** `Cannot start a runtime from within a runtime` reaches Python as an
   opaque `RustPanic` when a blocking wrapper is called from a tokio worker — in practice from inside
   an async callback the binding itself handed out. Mechanical fix: a `Handle::try_current()` guard
   converting it to a typed Python error.
3. **`maturin sdist` has NEVER been run.** Every spike consumes kremory as a *path* dependency, which
   cannot produce a publishable sdist. The crates.io-version-dep path is entirely unexercised and
   must be proven before any PyPI publish.
4. **Pin interpreters by ABSOLUTE PATH in release automation.** `uv venv --python 3.14` silently
   resolved to the **free-threaded** interpreter and produced an install-time ABI mismatch.
5. **abi3 cannot serve free-threaded CPython** — measured: the cp39-abi3 wheel is rejected on 3.14t
   with *"requires a GIL-enabled interpreter"*. Defer those wheels and record why; adoption is
   currently negligible. This is a packaging decision and must not block the API shape.

## Inherited core behaviour — a TD against kremory, not the binding

`remember` returns `Ok` when the embedder **errors**, persisting rows with NULL embeddings and
deferring the failure to `recall`. Proved language-independent with a pure-Rust `AlwaysErrEmbedder`
control that behaves identically. Whatever is decided, the Python SDK inherits it.

This is the same family as the Node zero-vector bug fixed in `130d174e`: an embedding failure that
does not stop the write leaves a record that is stored, reported successful, and unrecallable.

## Method note

All three spike agents shared one scratch directory and at least two hit real cross-contamination —
one `cargo check` read back another spike's compile output, another blocked on a cargo file lock.
**Give parallel build spikes separate target directories.** It did not invalidate these results
(each was re-verified), but it easily could have.

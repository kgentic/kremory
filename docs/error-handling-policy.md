# Error Handling Policy

**Document**: `docs/error-handling-policy.md`  
**Status**: Active  
**Story**: #156 (Concern B)  
**References**: Story #9 site classification (.ship/sessions/2026-05-26-story-9/swarm-memory/story-9-red.md)

---

## Purpose

This document defines when a fallible operation in kremory source code returns
`Err(Error::...)` versus when it calls `panic!("invariant: ...")`. It is the
written guidance behind the clippy lint enforcement added in Story #9
(`unwrap_used = "deny"`, `expect_used = "deny"`) and the G6 acceptance gate.

---

## User-input failures

**Use `Result<_, Error>` for anything reachable from the public API by feeding
bad inputs.**

A user-input failure is any condition that depends on caller-supplied data or
external state, not on the internal consistency of the program. The criterion is:

> "Could a well-written, well-intentioned caller trigger this by passing different
> arguments or operating in a different environment?"

If yes → it is a user-input failure → return `Err(Error::...)`.

Concrete examples of user-input failures:

- Malformed configuration (field out of range, required field empty)
- Missing or unreadable files passed as arguments
- Network errors from LLM/embedding providers
- LLM response that cannot be parsed as expected JSON
- Embedding dimension configured as zero
- Token window `max_tokens` smaller than `min_tokens`
- Database errors from libsql (I/O failure, constraint violation)
- Caller-supplied entity type list that is empty when the extractor requires it

All of these have named variants on `core::error::Error` (or `memory::MemoryError`)
so callers can match precisely without parsing strings.

---

## Contract violations

**Use `panic!("invariant: ...")` for post-validation state that cannot fail by
construction.**

A contract violation is a condition that can only occur if:

1. The program's own internal logic is broken (a bug), OR
2. An unrecoverable OS/runtime failure has already occurred

The criterion is:

> "Is this a state the program itself guaranteed could not happen, given that
> prior checks and initialization succeeded?"

If yes → it is a contract violation → `panic!("invariant: ...")`.

**The panic message MUST start with `invariant:` for greppability.** This
convention lets `grep -rn "invariant:"` find every contract-violation site in
the codebase.

Concrete examples of contract violations:

- `OnceLock::get()` after `OnceLock::set()` in the same thread — by construction
  the `get()` is `Some` (a `None` here means the initialization logic is broken)
- `chars().next()` on a string that was guarded by `len() > 1` two lines above —
  the guard makes the first character's existence a structural invariant
- `Mutex::lock().unwrap()` on a mutex that, if poisoned, signals another thread
  panicked mid-write, leaving shared state corrupted and unrecoverable
- OS thread spawn failure after the program has already allocated resources
  that require the thread to release them — cannot proceed without the thread
- Tokio runtime build failure inside the worker thread — no runtime = no async,
  unrecoverable by definition

---

## Test code exemption

**`#[cfg(test)]` modules and files in `tests/` use `.unwrap()` freely.**

The `unwrap_used = "deny"` clippy lint applies to production code only.
Test code is exempt because:

- Test failures produce readable error messages via test harness output
- `.unwrap()` panics in tests are caught and reported per-test, not
  crashing the whole process
- Forcing `expect()` or `?` in tests adds noise without adding safety

The `[lints]` section in `Cargo.toml` applies the deny lint only to the
library target. Test targets inherit `warnings = "deny"` but not `unwrap_used`
nor `expect_used`.

---

## Decision rubric

At each candidate site, run this three-question checklist:

1. **Can a correct caller trigger this?**  
   If a caller following the documented API contract could cause this condition
   by varying their inputs or environment → **return `Err(...)`**.

2. **Does this represent broken internal state?**  
   If this condition can only occur when the program's own logic has violated
   a previously-established invariant → **use `panic!("invariant: ...")`**.

3. **Is this an unrecoverable OS/runtime failure?**  
   If the process cannot meaningfully continue (no thread, no runtime, corrupted
   mutex) → **use `panic!("invariant: ...")`**.

If none of the three applies cleanly, default to `Err(...)`. Only use `panic!`
when you can state the invariant precisely in the message.

---

## Examples

### Example 1 — User-input (configuration validation)

**Do — return `Err` for bad caller input:**

```rust
fn validate_config(config: &PipelineConfig) -> Result<()> {
    if config.embedding_dim == 0 {
        return Err(Error::EmbeddingDimZero);
    }
    if config.bm25_weight + config.vector_weight != 1.0 {
        return Err(Error::WeightSumInvalid {
            bm25: config.bm25_weight,
            vector: config.vector_weight,
        });
    }
    Ok(())
}
```

**Don't — panic on configuration errors the caller can fix:**

```rust
fn validate_config(config: &PipelineConfig) {
    // WRONG: caller can pass embedding_dim = 0; this is user-input, not a bug
    assert!(config.embedding_dim > 0, "embedding_dim must be > 0");
}
```

---

### Example 2 — Contract violation (OnceLock post-init)

**Do — panic with `invariant:` message for a structurally impossible `None`:**

```rust
if GLINER.get().is_none() {
    let g = GlinerExtractor::new().map_err(Error::from)?;
    let _ = GLINER.set(g);
}
// The set() above guarantees get() is Some. A None here is a logic bug.
let extractor = GLINER.get().unwrap_or_else(|| {
    panic!("invariant: GLINER OnceLock empty immediately after set")
});
```

**Don't — use `.expect()` with a non-invariant message, or return `Err` for
a structural guarantee:**

```rust
// WRONG: expect() without "invariant:" prefix; clippy denies expect_used anyway
let extractor = GLINER.get().expect("just initialised");

// WRONG: returning Err implies the caller can recover, but they cannot —
// this is a logic bug in our code, not bad caller input
let extractor = GLINER.get().ok_or(Error::Other("OnceLock empty".into()))?;
```

---

## Quick reference

| Situation | Action |
|-----------|--------|
| Bad argument from caller | `Err(Error::Config(...))` or named variant |
| Missing file / IO error | `Err(Error::Database(...))` or `Err(Error::Other(...))` |
| LLM/provider error | `Err(Error::Llm(...))` or `Err(Error::Embedding(...))` |
| Parse failure on LLM output | `Err(Error::Parse(...))` or `Err(Error::ExtractionStage {...})` |
| OnceLock empty after set | `panic!("invariant: ...")` |
| Mutex poisoned | `panic!("invariant: ... mutex poisoned")` |
| OS thread spawn fails | `panic!("invariant: ...")` |
| Tokio runtime build fails | `panic!("invariant: ...")` |
| `chars().next()` after `len() > 1` | `panic!("invariant: ...")` |
| Test code (any context) | `.unwrap()` freely |

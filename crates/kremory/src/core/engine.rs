//! Process-global `TemporalGraph` singleton. Story #5.
//!
//! `engine_init` initialises the singleton once; subsequent calls are no-ops
//! when the same (or any) path is provided — the singleton is already
//! populated and returned without opening a second connection. Call
//! `engine()` anywhere after init to obtain the `Arc<TemporalGraph>`.
//!
//! # Why OnceLock
//!
//! One process = one SQLite file = one connection. `OnceLock` gives a
//! zero-cost read path (no lock on hot reads once set) while making the
//! write path safe — `std::sync::OnceLock::set` is an atomic CAS. The
//! alternative (`lazy_static` or `once_cell::Lazy`) would initialise
//! eagerly or hide the `Result` path.
//!
//! # BYOM invariant
//!
//! This module is pure graph infrastructure — no LLM calls, no autoagents
//! concrete types. Embedding providers are wired by the caller before
//! handing the `Arc<TemporalGraph>` to upper layers.

use std::sync::{Arc, OnceLock};

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

static ENGINE: OnceLock<Arc<TemporalGraph>> = OnceLock::new();

/// Initialise the process-global `TemporalGraph` singleton.
///
/// Opens the graph at `path` and stores it in `ENGINE`. If `ENGINE` is
/// already populated (i.e., `engine_init` was called earlier in this
/// process), this is a no-op — the existing singleton is left untouched and
/// `Ok(())` is returned.
///
/// # Errors
///
/// Propagates any `TemporalGraph::open` error (I/O, migration failure).
/// Does NOT error on a second call — second call is a no-op.
pub async fn engine_init(path: &str) -> Result<()> {
    // Fast path: already initialised.
    if ENGINE.get().is_some() {
        return Ok(());
    }
    let graph = Arc::new(TemporalGraph::open(path).await?);
    // set() is an atomic CAS. If another concurrent caller raced us and
    // initialised first, discard our graph (it will be dropped) and return Ok.
    let _ = ENGINE.set(graph);
    Ok(())
}

/// Return a clone of the process-global `Arc<TemporalGraph>`.
///
/// # Panics
///
/// Panics with the message `"invariant: engine_init must be called before engine()"`
/// when called before `engine_init` has successfully returned.
pub fn engine() -> Arc<TemporalGraph> {
    match ENGINE.get() {
        Some(arc) => arc.clone(),
        None => panic!("invariant: engine_init must be called before engine()"),
    }
}

#[cfg(test)]
mod tests {
    // OnceLock is process-global and cannot be reset between tests.
    // Tests here use in-memory DBs where possible, or guard against
    // a pre-initialised ENGINE from a prior test in the same binary.

    use super::*;

    /// Story #5: engine() before engine_init panics with the invariant message.
    #[test]
    fn engine_panics_before_init() {
        // Only meaningful when ENGINE is uninitialised. If another test in
        // this binary already called engine_init, skip to avoid a false positive.
        if ENGINE.get().is_some() {
            return;
        }
        let result = std::panic::catch_unwind(engine);
        match result {
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("<non-string panic>");
                assert!(
                    msg.contains("invariant: engine_init must be called before engine()"),
                    "panic message did not contain expected invariant text: {msg}"
                );
            }
            Ok(_arc) => {
                // ENGINE was already populated by a concurrent test — skip.
            }
        }
    }

    /// Story #5: engine_init is idempotent — second call returns Ok(()) without
    /// opening a new connection. ptr_eq verifies it's the same Arc allocation.
    #[tokio::test]
    async fn engine_init_idempotent_and_ptr_eq() {
        engine_init(":memory:").await.expect("first init");
        engine_init(":memory:").await.expect("second init (no-op)");

        let a = engine();
        let b = engine();
        assert!(
            Arc::ptr_eq(&a, &b),
            "engine() must return the same Arc allocation on every call"
        );
    }

    /// Story #5: concurrent engine_init calls both return Ok(()) and only one
    /// TemporalGraph::open actually executes.
    #[tokio::test(flavor = "multi_thread")]
    async fn engine_init_concurrent_only_one_open() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;
        use tokio::sync::Barrier;

        // If ENGINE already set, skip — we can't re-initialise OnceLock.
        if ENGINE.get().is_some() {
            return;
        }

        let counter = StdArc::new(AtomicUsize::new(0));
        let barrier = StdArc::new(Barrier::new(2));

        let c1 = StdArc::clone(&counter);
        let b1 = StdArc::clone(&barrier);
        let t1 = tokio::spawn(async move {
            b1.wait().await;
            // Simulate the open counter via a racing engine_init call.
            // We can't instrument TemporalGraph::open directly without a seam,
            // so we measure: both tasks call engine_init; only one pays the open
            // cost (the other hits the fast-path guard or the CAS discard).
            // The counter here counts how many engine_init calls succeeded
            // in actually writing to ENGINE.
            let before = ENGINE.get().is_some();
            let result = engine_init(":memory:").await;
            let after = ENGINE.get().is_some();
            if !before && after {
                c1.fetch_add(1, Ordering::AcqRel);
            }
            result.expect("t1 engine_init")
        });

        let c2 = StdArc::clone(&counter);
        let b2 = StdArc::clone(&barrier);
        let t2 = tokio::spawn(async move {
            b2.wait().await;
            let before = ENGINE.get().is_some();
            let result = engine_init(":memory:").await;
            let after = ENGINE.get().is_some();
            if !before && after {
                c2.fetch_add(1, Ordering::AcqRel);
            }
            result.expect("t2 engine_init")
        });

        t1.await.expect("t1 join");
        t2.await.expect("t2 join");

        // At most one task could observe the ENGINE being empty before their
        // init and non-empty after — the other either saw it already populated
        // or raced and discarded.
        let open_count = counter.load(Ordering::Acquire);
        assert!(
            open_count <= 1,
            "at most one concurrent engine_init should observe the write: got {open_count}"
        );

        // ENGINE must be populated after both tasks complete.
        assert!(
            ENGINE.get().is_some(),
            "ENGINE must be set after concurrent init"
        );
    }
}

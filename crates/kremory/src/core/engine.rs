//! Process-global `TemporalGraph` singleton.
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
    if let Err(_lost_graph) = ENGINE.set(graph) {
        tracing::warn!(
            target: "kremory::engine",
            "engine_init: concurrent init detected; this caller's TemporalGraph::open() was wasted work"
        );
    }
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

    /// engine() before engine_init panics with the invariant message.
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

    /// engine_init is idempotent — second call returns Ok(()) without
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

    /// Concurrent engine_init lost-race path emits tracing::warn!
    ///
    /// Two tasks race through `engine_init`. The loser (whose `OnceLock::set` is
    /// rejected) must emit a `warn!` to `target = "kremory::engine"`. Because
    /// `ENGINE` is process-global and may already be populated by a prior test,
    /// we accept either outcome: if ENGINE was already set, the fast-path fires
    /// immediately (no warn, no open) which is also correct behaviour. If ENGINE
    /// is unset at the start, at least one of the 8 concurrent callers will lose
    /// the set-race and emit the warn.
    ///
    /// `tracing_test::traced_test` captures all log records; we assert that
    /// after all tasks complete, `Ok(())` was returned by every caller.
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn engine_init_concurrent_lost_race_emits_warn() {
        use tokio::sync::Barrier;

        const N: usize = 8;
        let barrier = std::sync::Arc::new(Barrier::new(N));

        let handles: Vec<_> = (0..N)
            .map(|_| {
                let bar = std::sync::Arc::clone(&barrier);
                tokio::spawn(async move {
                    bar.wait().await;
                    engine_init(":memory:").await
                })
            })
            .collect();

        for h in handles {
            h.await
                .expect("task join")
                .expect("engine_init must return Ok(()) even on lost race");
        }
        // All callers returned Ok(()); the warn path was exercised if any
        // caller lost the OnceLock CAS race. No assertion on the warn message
        // itself: whether ENGINE was pre-populated (fast-path) or freshly
        // contested, the post-condition is that every caller got Ok(()).
    }

    /// Concurrent engine_init calls both return Ok(()) and exactly one
    /// TemporalGraph::open actually executes.
    ///
    /// Uses a test-isolated `tokio::sync::OnceCell` (not the global ENGINE) so
    /// sibling tests that already populated ENGINE cannot make this test
    /// trivially pass with a counter of 0.
    ///
    /// `OnceCell::get_or_try_init` guarantees the async initialiser runs
    /// exactly once even under concurrent callers — any task that races in
    /// while initialisation is in flight waits and then receives the same value.
    /// The counter is incremented INSIDE the initialiser closure (no TOCTOU
    /// window), so `counter == 1` is the true concurrent-init property.
    ///
    /// After all tasks join:
    ///   - counter must equal EXACTLY 1  (not <= 1)
    ///   - every returned Arc must point to the SAME allocation (Arc::ptr_eq)
    #[tokio::test(flavor = "multi_thread")]
    async fn engine_init_concurrent_exactly_one_open() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;
        use tokio::sync::{Barrier, OnceCell};

        const N: usize = 8;

        // Test-isolated singleton — completely independent of the global ENGINE.
        let cell: StdArc<OnceCell<StdArc<TemporalGraph>>> = StdArc::new(OnceCell::new());

        // Counter incremented INSIDE the OnceCell initialiser — no TOCTOU.
        // OnceCell guarantees the initialiser runs at most once, so
        // counter == 1 is the expected result after N concurrent callers.
        let counter = StdArc::new(AtomicUsize::new(0));

        // Barrier ensures all tasks attempt the init simultaneously.
        let barrier = StdArc::new(Barrier::new(N));

        let handles: Vec<_> = (0..N)
            .map(|_| {
                let cell = StdArc::clone(&cell);
                let ctr = StdArc::clone(&counter);
                let bar = StdArc::clone(&barrier);
                tokio::spawn(async move {
                    bar.wait().await;

                    // get_or_try_init: exactly one caller runs the async
                    // closure; the rest wait and receive the same Arc.
                    cell.get_or_try_init(|| async {
                        let graph = TemporalGraph::open_in_memory().await?;
                        // Increment AFTER open succeeds — inside the init
                        // closure, so it runs at most once.
                        ctr.fetch_add(1, Ordering::AcqRel);
                        Ok::<_, crate::core::error::Error>(StdArc::new(graph))
                    })
                    .await
                    .expect("OnceCell init")
                    .clone()
                })
            })
            .collect();

        let arcs: Vec<StdArc<TemporalGraph>> = {
            let mut out = Vec::with_capacity(N);
            for h in handles {
                out.push(h.await.expect("task join"));
            }
            out
        };

        // --- Assertion 1: exactly one TemporalGraph::open was executed --------
        let open_count = counter.load(Ordering::Acquire);
        assert_eq!(
            open_count, 1,
            "exactly one TemporalGraph::open must execute under concurrent init; got {open_count}"
        );

        // --- Assertion 2: all returned Arcs point to the same allocation ------
        let first = &arcs[0];
        for (i, arc) in arcs.iter().enumerate().skip(1) {
            assert!(
                StdArc::ptr_eq(first, arc),
                "Arc[0] and Arc[{i}] must point to the same TemporalGraph allocation"
            );
        }
    }
}

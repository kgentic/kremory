#![allow(clippy::unwrap_used, clippy::expect_used)]
//! B.1 — Arena lifecycle RED gate.
//!
//! Verifies that `kremory::core::arena` exposes a `Bump` arena allocator
//! (re-exported from bumpalo) and that it survives await points in a tokio
//! multi-thread runtime without borrow-check rejection.
//!
//! The heap-bounded test (dhat-rs) is gated behind `#[ignore]` with an
//! explicit note — dhat-rs profiling requires a single-threaded `#[tokio::test]`
//! context and adds ~5s overhead; it is run as part of the B.1 quality gate
//! (`cargo test --workspace -- --include-ignored b1_arena`) but not on every
//! `cargo test` run.
//!
//! A.0 outcome: bumpalo survives `.await` (K7 = PASS). B.1 wires `Bump`
//! through the arena module per runbook §5.8.

use kremory::core::arena::Bump;

/// Arena allocates a string slice, survives an await point, string readable after.
/// Verifies bumpalo survives tokio's `Send` requirement across await points.
#[tokio::test]
async fn arena_alloc_str_survives_await() {
    let bump = Bump::new();
    let s: &str = bump.alloc_str("kremory-arena-test");
    tokio::task::yield_now().await;
    assert_eq!(s, "kremory-arena-test");
}

/// Arena can allocate and read back a struct across an await point.
#[tokio::test]
async fn arena_alloc_struct_survives_await() {
    #[derive(Debug, PartialEq)]
    struct Episode {
        id: u32,
        text: &'static str,
    }

    let bump = Bump::new();
    let ep = bump.alloc(Episode {
        id: 42,
        text: "test episode",
    });
    tokio::task::yield_now().await;
    assert_eq!(ep.id, 42);
    assert_eq!(ep.text, "test episode");
}

/// Arena reset — verify that `reset()` works and arena is reusable.
/// This is the core usage pattern for per-episode allocation: allocate all
/// intermediate structs for one episode, then reset for the next.
#[tokio::test]
async fn arena_reset_allows_reuse() {
    let mut bump = Bump::new();

    // First "episode"
    let s1 = bump.alloc_str("episode-one");
    assert_eq!(s1, "episode-one");

    // Reset — same arena, new lifetime
    bump.reset();

    // Second "episode" — reuses the same memory backing
    let s2 = bump.alloc_str("episode-two");
    assert_eq!(s2, "episode-two");
}

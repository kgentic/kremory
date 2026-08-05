#![allow(clippy::unwrap_used, clippy::expect_used)]
/// A.0 — K7 bumpalo + tokio borrow-checker prototype gate.
///
/// Verifies that a `bumpalo::Bump` arena allocator survives an `.await`
/// point in a tokio async context. This is the K7 confidence gate from
/// `rqlcm-mvp-v1-architecture-2026-05-20.md §11`.
///
/// Outcome:
/// - PASS → K7 promoted to HIGH; B.1 ships arena with bumpalo.
/// - FAIL (borrow-check reject) → switch to typed-arena OR skip P4.
///
/// Note: this test is intentionally kept as a regression check after
/// the gate passes (not removed post-gate), per runbook §5.1.
#[tokio::test]
async fn bumpalo_survives_await() {
    let bump = bumpalo::Bump::new();
    let s: &str = bump.alloc_str("hello");
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    assert_eq!(s, "hello");
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8a — ForgetRequest shape and stub tests.
//!
//! `ForgetRequest` must use `.execute()` (not `.await`) and validates namespace
//! before touching the graph. The v0.1.0 stub returns `Ok(0)`.

use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::Arc;

// Unique-per-test DB paths: tests run concurrently by default and
// previously all shared `/tmp/test.db`, causing intermittent
// `SqliteFailure(5, "database is locked")`. Each test now owns its file.
fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_facade_forget_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

/// `forget().execute()` on an empty namespace erases nothing — and says so
/// per table, rather than through a single count that also reads `0` when a
/// real erasure pinned every shared entity (TD-247).
#[tokio::test]
async fn forget_stub_returns_zero() {
    let db = unique_db_path("stub_returns_zero");
    let mem = Memory::open(&db)
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("tests"))
        .await
        .expect("builder should succeed");
    let deleted = mem.forget().execute().await.expect("forget should succeed");
    assert!(
        deleted.is_empty(),
        "an empty namespace erases nothing; got {deleted:?}"
    );
}

/// `.in_namespace()` override: no default on Memory but explicit on request → ok.
#[tokio::test]
async fn forget_in_namespace_override_succeeds() {
    let db = unique_db_path("in_namespace_override");
    let mem = Memory::open(&db)
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder should succeed");

    let deleted = mem
        .forget()
        .in_namespace(Namespace::new("explicit-ns"))
        .execute()
        .await
        .expect("should succeed with in_namespace override");
    assert!(deleted.is_empty(), "nothing to erase; got {deleted:?}");
}

/// Default namespace on Memory flows into `forget()`.
#[tokio::test]
async fn forget_uses_memory_default_namespace() {
    let db = unique_db_path("uses_default_namespace");
    let mem = Memory::open(&db)
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("default-ns"))
        .await
        .expect("builder should succeed");

    // No `.in_namespace()` call — relies on Memory.default_namespace.
    let deleted = mem.forget().execute().await.expect("should succeed");
    assert!(deleted.is_empty(), "nothing to erase; got {deleted:?}");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

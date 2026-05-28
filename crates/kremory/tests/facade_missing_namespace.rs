#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8a — MissingNamespace error tests.
//!
//! When neither `.in_namespace()` nor `default_namespace` is set, every
//! operation that requires a namespace must return `MemoryError::MissingNamespace`.

use kremory::{DynEmbeddingProvider, Memory, MemoryError};
use std::sync::Arc;

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_facade_missing_ns_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

async fn open_no_ns() -> Memory {
    Memory::open(unique_db_path("open_no_ns"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder should succeed")
}

/// `forget()` without namespace → MissingNamespace.
#[tokio::test]
async fn forget_without_namespace_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let err = mem.forget().execute().await.expect_err("should fail");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `forget().in_namespace()` with explicit override → no error.
#[tokio::test]
async fn forget_with_explicit_namespace_succeeds() {
    use kremory::Namespace;
    let mem = open_no_ns().await;
    let count = mem
        .forget()
        .in_namespace(Namespace::new("override"))
        .execute()
        .await
        .expect("should succeed with explicit namespace");
    assert_eq!(count, 0, "stub returns 0 deleted");
}

/// `remember()` without namespace → MissingNamespace.
///
/// The namespace check fires before any graph call.
#[tokio::test]
async fn remember_without_namespace_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let err = mem.remember("test content").await.expect_err("should fail");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `recall()` without namespace → MissingNamespace.
#[tokio::test]
async fn recall_without_namespace_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let err = mem
        .recall("what does user prefer?")
        .await
        .expect_err("should fail");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `recall().raw()` without namespace → MissingNamespace.
#[tokio::test]
async fn recall_raw_without_namespace_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let err = mem.recall("query").raw().await.expect_err("should fail");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

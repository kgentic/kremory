#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8a — ForgetRequest shape and stub tests.
//!
//! `ForgetRequest` must use `.execute()` (not `.await`) and validates namespace
//! before touching the graph. The v0.1.0 stub returns `Ok(0)`.

use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::Arc;

async fn open_with_ns() -> Memory {
    Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("tests"))
        .await
        .expect("builder should succeed")
}

/// Stub `forget().execute()` returns `Ok(0)` when namespace is present.
#[tokio::test]
async fn forget_stub_returns_zero() {
    let mem = open_with_ns().await;
    let deleted = mem.forget().execute().await.expect("forget should succeed");
    assert_eq!(deleted, 0, "v0.1.0 stub always returns 0 deleted");
}

/// `.in_namespace()` override: no default on Memory but explicit on request → ok.
#[tokio::test]
async fn forget_in_namespace_override_succeeds() {
    let mem = Memory::open("/tmp/test.db")
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
    assert_eq!(deleted, 0);
}

/// Default namespace on Memory flows into `forget()`.
#[tokio::test]
async fn forget_uses_memory_default_namespace() {
    let mem = Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("default-ns"))
        .await
        .expect("builder should succeed");

    // No `.in_namespace()` call — relies on Memory.default_namespace.
    let deleted = mem.forget().execute().await.expect("should succeed");
    assert_eq!(deleted, 0);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

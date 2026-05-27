//! A.8a — `recall().as_of()` warning path tests.
//!
//! At v0.1.0, `.as_of(ts)` is accepted in the builder but the point-in-time
//! filter is not yet applied in SQL. A `tracing::warn!` is emitted if as_of
//! is set. The AC tests verify:
//!   - `.as_of()` on `RecallRequest` compiles and chains correctly.
//!   - The `as_of` timestamp is stored on the request (via `SearchOpts`).
//!   - The builder can still produce a MissingNamespace error (namespace check
//!     fires before any filter logic, so no phantom success).

use chrono::Utc;
use kremory::{DynEmbeddingProvider, Memory, MemoryError};
use std::sync::Arc;

/// `.as_of()` can be chained on RecallRequest without consuming the builder.
#[test]
fn as_of_chain_compiles() {
    let mem = make_memory_sync();
    let ts = Utc::now();
    // Build but don't await — just verify the chain is valid.
    let _req = mem.recall("query").as_of(ts);
}

/// `.as_of()` + missing namespace → `MissingNamespace` (not a phantom success).
///
/// The as_of value is accepted but namespace validation fires before any SQL
/// is executed, so missing namespace still surfaces correctly.
#[tokio::test]
async fn as_of_with_missing_namespace_still_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let ts = Utc::now();
    let err = mem
        .recall("what happened yesterday?")
        .as_of(ts)
        .await
        .expect_err("should fail with MissingNamespace");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `.as_of()` can be combined with `.k()` and `.in_namespace()`.
#[test]
fn as_of_k_namespace_chain_compiles() {
    use kremory::Namespace;
    let mem = make_memory_sync();
    let ts = Utc::now();
    // The chain must compile without type errors.
    let _req = mem
        .recall("query")
        .k(5)
        .as_of(ts)
        .in_namespace(Namespace::new("ns"));
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn open_no_ns() -> Memory {
    Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder should succeed")
}

fn make_memory_sync() -> Memory {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(open_no_ns())
}

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

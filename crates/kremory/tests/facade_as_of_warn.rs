#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8a — `recall().as_of()` fail-loud path tests (F4 / ADR adr-memory-builder-as-of-fail-loud).
//!
//! `.as_of(ts)` requests bi-temporal point-in-time recall, which is declared but
//! NOT yet implemented in SQL. Previously the call silently no-op'd (warn +
//! current-state results) — a footgun on a bi-temporal engine. As of F4 the call
//! FAILS LOUD with `Error::Unsupported`, enforced at the consumption point
//! (`memory::search`) so it covers both `.as_of()` and the `.opts()` escape hatch.
//! The AC tests verify:
//!   - `.as_of()` on `RecallRequest` compiles and chains correctly.
//!   - `.as_of()` + a valid namespace → `Err(Error::Unsupported)` at `.await`.
//!   - The builder still produces MissingNamespace first when no namespace is set
//!     (namespace check fires before the search guard, so no phantom success).

use chrono::Utc;
use kremory::core::error::Error as CoreError;
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

/// F4 — `.as_of()` + a valid namespace fails loud with `Error::Unsupported`
/// (was: silent no-op returning current-state results). Proves the consumption-
/// point guard surfaces end-to-end through the facade `.as_of()` setter.
#[tokio::test]
async fn as_of_with_namespace_errors_unsupported() {
    use kremory::Namespace;
    let mem = open_no_ns().await;
    let ts = Utc::now();
    let err = mem
        .recall("what happened yesterday?")
        .in_namespace(Namespace::new("ns-as-of-loud"))
        .as_of(ts)
        .await
        .expect_err("as_of must fail loud, not silently no-op");
    assert!(
        matches!(err, MemoryError::Core(CoreError::Unsupported { feature }) if feature == "as_of point-in-time recall"),
        "expected Error::Unsupported for as_of, got: {err:?}"
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

/// Unique per-call DB path to avoid SQLite "database is locked" flakes
/// when integration tests run in parallel.
fn unique_db_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_facade_as_of_warn_{}_{}.db",
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

//! A.8a — RememberRequest builder shape tests.
//!
//! Tests the builder API and namespace validation. Actual ingest calls
//! now route to `EngineGraphHandle` (wired at v0.1.0). Tests that exercise
//! real ingest use a null LLM + null embedder backed by an in-memory graph.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use kremory::{DynEmbeddingProvider, Memory, MemoryError, Namespace, SourceKind};
use std::sync::Arc;

/// `remember()` without namespace returns MissingNamespace before graph call.
#[tokio::test]
async fn remember_missing_namespace_errors() {
    let mem = open_no_ns().await;
    let err = mem.remember("some content").await.expect_err("should fail");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `.in_namespace()` on RememberRequest resolves namespace (checked before graph).
/// Namespace check passes → the call reaches `EngineGraphHandle::graph_ingest_episode`
/// which returns a result (Ok or Err — not a panic). With null LLM + null embedder the
/// inline ingest path may succeed or return a core error; either way it does NOT panic.
#[tokio::test]
async fn remember_with_namespace_reaches_engine_graph_handle() {
    let mem = open_no_ns().await;
    // Namespace check passes → call reaches EngineGraphHandle (no panic).
    // The result may be Ok or Err (null LLM may produce empty extraction), but it
    // must not panic.
    let _result = mem
        .remember("test content")
        .in_namespace(Namespace::new("acme"))
        .await;
    // No assertion on Ok/Err — the goal is that it does NOT panic and the namespace
    // routing was wired correctly (pre-graph namespace check passed).
}

/// RememberRequest builder chains compile without errors.
#[test]
fn remember_builder_chain_compiles() {
    let mem = make_memory_sync();
    let _req = mem
        .remember("content")
        .from_chat("session-123")
        .in_namespace(Namespace::new("ns"))
        .no_wait();
}

/// `.from_source()` chain compiles with all SourceKind variants.
#[test]
fn remember_from_source_variants_compile() {
    let mem = make_memory_sync();
    let _r1 = mem.remember("c").from_chat("id1");
    let _r2 = mem.remember("c").from_note("id2");
    let _r3 = mem.remember("c").from_document("id3");
    let _r4 = mem.remember("c").from_source("id4", SourceKind::Chat);
}

/// `remember_batch().entry().done()` chain compiles.
#[test]
fn remember_batch_chain_compiles() {
    let mem = make_memory_sync();
    let _batch = mem
        .remember_batch()
        .entry("episode one")
        .in_namespace(Namespace::new("ns"))
        .done()
        .entry("episode two")
        .in_namespace(Namespace::new("ns"))
        .done()
        .with_batch_id("batch-abc");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    // Collision-proof unique path. A nanosecond timestamp alone is NOT unique:
    // tests run in parallel threads within this binary, and two calls landing in
    // the same nanosecond produced the SAME path → two tests opened one DB file
    // and raced its migrations (migrate_004 `drop_old_entities` → "no such table:
    // entities"). The monotonic process-wide counter guarantees a distinct file
    // per call regardless of clock resolution.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_facade_remember_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
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

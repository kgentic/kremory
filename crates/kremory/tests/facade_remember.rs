//! A.8a — RememberRequest builder shape tests.
//!
//! Tests the builder API and namespace validation. Actual ingest calls
//! would hit StubGraphHandle's `unimplemented!`, so tests that need to
//! invoke `submit_episode` are deferred to integration tests with a real graph.

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
/// The graph call would panic (StubGraphHandle) so we test only namespace resolution
/// by verifying a *different* error fires (the unimplemented! panic is a separate path).
/// Namespace check passes → we hit the graph stub → panic caught by `#[should_panic]`.
#[tokio::test]
#[should_panic(expected = "StubGraphHandle::graph_ingest_episode")]
async fn remember_with_namespace_reaches_graph_stub() {
    let mem = open_no_ns().await;
    // This should pass namespace check but then panic in StubGraphHandle.
    let _ = mem
        .remember("test content")
        .in_namespace(Namespace::new("acme"))
        .await;
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

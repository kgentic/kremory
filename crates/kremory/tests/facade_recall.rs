//! A.8a — RecallRequest builder shape tests.
//!
//! Tests validate the builder API surface (compile-time + runtime) and
//! `RecallTemplate` conversion. Tests that would invoke the graph
//! (`.await` with a namespace) are deferred to integration tests that
//! use a real GraphHandle.

use kremory::{DynEmbeddingProvider, Memory, MemoryError, Namespace, RecallTemplate};
use std::sync::Arc;

/// `RecallTemplate::parse_str` round-trips for all three variants.
#[test]
fn recall_template_parse_str_roundtrip() {
    for slug in &["entities", "edge_summary", "temporal_facts"] {
        let t = RecallTemplate::parse_str(slug)
            .unwrap_or_else(|| panic!("parse_str should recognise slug: {slug}"));
        assert_eq!(t.as_str(), *slug, "as_str should return the original slug");
    }
}

/// Unknown slug returns `None`.
#[test]
fn recall_template_parse_str_unknown_returns_none() {
    assert!(RecallTemplate::parse_str("unknown_slug").is_none());
}

/// `RecallTemplate::TemporalFacts` converts to `ContextTemplate::TemporalFacts`.
#[test]
fn recall_template_converts_to_context_template() {
    use kremory::ContextTemplate;
    let ct: ContextTemplate = RecallTemplate::TemporalFacts.into();
    assert!(matches!(ct, ContextTemplate::TemporalFacts));

    let ct2: ContextTemplate = RecallTemplate::Entities.into();
    assert!(matches!(ct2, ContextTemplate::Entities));

    let ct3: ContextTemplate = RecallTemplate::EdgeSummary.into();
    assert!(matches!(ct3, ContextTemplate::EdgeSummary));
}

/// `recall().raw()` returns a `RecallRawRequest` — verified by MissingNamespace
/// firing (namespace check precedes graph call).
#[tokio::test]
async fn recall_raw_missing_namespace_is_checked_first() {
    let mem = open_no_ns().await;
    let err = mem.recall("query").raw().await.expect_err("should fail");
    assert!(matches!(err, MemoryError::MissingNamespace { .. }));
}

/// `recall().as_template(Entities)` still validates namespace before graph.
#[tokio::test]
async fn recall_with_template_missing_namespace_errors() {
    let mem = open_no_ns().await;
    let err = mem
        .recall("query")
        .as_template(RecallTemplate::Entities)
        .await
        .expect_err("should fail");
    assert!(matches!(err, MemoryError::MissingNamespace { .. }));
}

/// `recall().k(5)` chains without consuming the builder.
#[test]
fn recall_k_chain_compiles() {
    let mem = kremory_test_utils::make_memory_no_ns();
    // Just verify the chain compiles and builds the request (don't .await).
    let _req = mem.recall("query").k(5).in_namespace(Namespace::new("ns"));
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn open_no_ns() -> Memory {
    Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder should succeed")
}

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Thin helper module (replaces a common/ file for these tests only).
mod kremory_test_utils {
    use super::*;

    pub fn make_memory_no_ns() -> Memory {
        // Synchronously build a memory with no namespace — only usable for
        // compile/shape tests that don't .await the request.
        // This spawns a current_thread runtime inline.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(async {
                Memory::open("/tmp/test.db")
                    .with_llm(Arc::new(kremory::core::provider::MockChatProvider::null()))
                    .with_embedder(Arc::new(kremory::core::provider::NullEmbeddingProvider {
                        dim: 384,
                    }))
                    .await
                    .expect("builder should succeed")
            })
    }
}

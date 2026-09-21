use super::*;

use axum::body::{to_bytes, Body};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::{ChatProvider, DynEmbeddingProvider, Memory};
use kremory_mcp::handlers;
use kremory_mcp::params::RememberParams;

// ─── in-process HTTP round-trip over a mock-provider Memory ──────────

async fn mock_memory() -> Arc<Memory> {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    let mem = Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .await
        .expect("in-memory Memory must build");
    Arc::new(mem)
}

/// Pin a mode-(c) fact directly through the shared handler so the entity
/// is recall-findable with mock providers (the pin path stamps the FTS name +
/// embedding at PIN time — no live LLM / enrichment seam needed, same as
/// `handler_roundtrip.rs::recall_structured_surfaces_pinned_fact...`).
async fn pin_fact(mem: &Memory, namespace: &str, subject: &str) {
    let params = RememberParams {
        namespace: namespace.to_string(),
        thread: None,
        content: format!("{subject} wrote the first algorithm"),
        source_kind: Some(kremory_mcp::params::SourceKindWire::Note),
        source_id: Some("doc-1".into()),
        published_at: None,
        structured_facts: vec![kremory_mcp::params::StructuredFactWire {
            subject: subject.to_string(),
            predicate: "wrote".into(),
            object: "the first algorithm".into(),
            valid_at: None,
            invalid_at: None,
        }],
        skip_extraction: true,
    };
    handlers::do_remember(mem, params)
        .await
        .expect("pin must succeed");
}

async fn body_json(body: Body) -> serde_json::Value {
    let bytes = to_bytes(body, 1 << 20).await.expect("read body");
    serde_json::from_slice(&bytes).expect("body is JSON")
}

#[cfg(feature = "content-search")]
mod fact_dense;
#[cfg(feature = "content-search")]
mod fusion;
mod http_roundtrip;
mod parse_rerank_k;
#[cfg(feature = "content-search")]
mod search_modes;
mod wire;

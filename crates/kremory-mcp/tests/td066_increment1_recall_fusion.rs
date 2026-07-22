#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-066 Increment 1 — content-fusion parity fix, MCP integration test.
//!
//! Spec: `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md`
//! §3 Increment 1 DoD item 3: "`kremory_recall` MCP tool returns the same
//! fused result shape (integration test through `handlers::do_recall`, not
//! just a unit test of the fusion fn in isolation)."
//!
//! `handlers::do_recall`'s `RecallFormat::Structured` branch calls
//! `req.raw().await` (`kremory-mcp/src/handlers.rs`) — since Increment 1 wires
//! fusion into `RecallRawRequest::into_future`, this closes the parity gap the
//! spec's §1.2 root-cause section names verbatim:
//! `handlers.rs:133-134` — "the MCP `kremory_recall` tool has no mode concept
//! and never calls this fn [`do_recall_content`]". Structured recall now
//! reaches the fused surface via the SAME core fn the REST layer's
//! `hybrid_mode_results` used to own alone, without `kremory_recall` needing
//! a mode concept at all.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::{ChatProvider, DynEmbeddingProvider, Memory};
use kremory_mcp::handlers;
use kremory_mcp::params::{RecallFormat, RecallParams, RecallStructuredOutput, RememberParams};

async fn mock_memory() -> Memory {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .await
        .expect("in-memory Memory must build")
}

/// `handlers::do_recall` with `format: Structured` must surface a
/// content-derived result for a query whose answer exists only in raw
/// episode text (no structured facts, no LLM extraction — mirrors the
/// core-level seam test's fixture) — proving the MCP tool surface, not just
/// `core::search::rrf_fuse_with_content` in isolation.
#[tokio::test]
async fn do_recall_structured_surfaces_content_derived_result() {
    let mem = mock_memory().await;
    let ns = "td066-mcp-inc1";

    handlers::do_remember(
        &mem,
        RememberParams {
            namespace: ns.to_string(),
            thread: None,
            content: "Zephyrine went scuba diving off the coast of Portugal last summer."
                .to_string(),
            source_kind: None,
            source_id: None,
            published_at: None,
            structured_facts: Vec::new(),
            skip_extraction: true,
        },
    )
    .await
    .expect("do_remember must succeed");

    let value = handlers::do_recall(
        &mem,
        RecallParams {
            namespace: ns.to_string(),
            thread: None,
            query: "scuba diving Portugal".to_string(),
            k: None,
            as_of: None,
            format: RecallFormat::Structured,
            template: Default::default(),
            rerank_k: None,
        },
    )
    .await
    .expect("do_recall must succeed");

    let wire: RecallStructuredOutput =
        serde_json::from_value(value).expect("structured output must deserialize");

    assert!(
        wire.results
            .iter()
            .any(|r| r.summary.contains("scuba diving")),
        "kremory_recall's Structured format must surface the content-derived result via \
         the SAME fused .raw() path Memory::recall() uses; got: {:?}",
        wire.results
    );
}

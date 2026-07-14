//! In-process handler round-trip tests for the floor-3 tool surface.
//!
//! Each test constructs a real `kremory::Memory` on an in-memory libSQL db
//! wired with `MockChatProvider::null()` + `NullEmbeddingProvider` (no live
//! model needed), builds a `KremoryMcpServer` over it, and calls the tool
//! handlers directly with `Parameters(...)`.
//!
//! ## On the DOD-001 recall proof + the enrichment seam
//!
//! The spec's DOD-001 asks that a `remember` with pinned `structured_facts` +
//! `skip_extraction=true` be provable through a `recall` that returns ≥1
//! result. Investigation against the substrate (2026-07-14) showed this is
//! NOT achievable with mock providers *alone*: kremory's recall entity arm
//! surfaces an entity only via (a) the entity FTS index or (b) the entity
//! vector — and BOTH are populated by the LLM enrichment / verify stage,
//! which `skip_extraction` deliberately skips (and which `MockChatProvider`
//! cannot perform anyway). A `skip_extraction` pinned entity lands with an
//! EMPTY FTS label and no embedding, so recall-by-name returns 0. kremory's
//! own `with_facts_integration.rs` documents the same reality — it verifies
//! pins via a substrate COUNTER, not via recall.
//!
//! So the DOD-001 read-path proof here [`recall_returns_pinned_entity_after_
//! enrichment`] writes via the `kremory_remember` tool, then simulates the
//! ONE thing the mock LLM can't do — populate the FTS-indexed entity name the
//! verify stage would write — using kremory's sanctioned
//! `temporal_graph_for_test` seam, and then proves the `kremory_recall` tool
//! surfaces that entity end-to-end. The enrichment step is a clearly-scoped
//! test seam, not production behaviour.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::{ChatProvider, DynEmbeddingProvider, Memory};
use kremory_mcp::params::{
    DreamParams, RecallFormat, RecallParams, RecallTemplateWire, RememberParams, SourceKindWire,
    StructuredFactWire,
};
use kremory_mcp::KremoryMcpServer;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::ErrorCode;

/// Build an in-memory Memory (mock LLM + null embedder). Returned as an `Arc`
/// so a test can both hand it to the server AND keep a handle to drive the
/// `temporal_graph_for_test` enrichment seam.
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

async fn mock_server() -> KremoryMcpServer {
    KremoryMcpServer::new(mock_memory().await)
}

/// A mode-(c) remember: pin a structured fact + skip LLM extraction. No live
/// model needed — the pinned fact + its subject entity are written directly.
fn pinned_remember_params(namespace: &str, subject: &str) -> RememberParams {
    RememberParams {
        namespace: namespace.to_string(),
        thread: None,
        content: format!("{subject} leads the design team"),
        source_kind: Some(SourceKindWire::Note),
        source_id: Some("doc-1".into()),
        published_at: None,
        structured_facts: vec![StructuredFactWire {
            subject: subject.to_string(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_at: None,
            invalid_at: None,
        }],
        skip_extraction: true,
    }
}

/// Simulate the LLM enrichment / verify stage for a single entity: write the
/// entity NAME into the FTS-indexed `properties` column (both the `entities`
/// row and its `entities_fts` shadow row), which is what makes an entity
/// findable by name via recall. `MockChatProvider` cannot do this; a live
/// model does it as part of extraction. Uses kremory's `test-utils`
/// `temporal_graph_for_test` seam.
async fn simulate_enrichment(mem: &Memory, entity_id: &str) {
    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder path exposes a TemporalGraph");
    let props = format!("{{\"name\":\"{entity_id}\",\"stub\":false}}");
    tg.conn
        .execute(
            "UPDATE entities SET properties = ?2 WHERE id = ?1",
            (entity_id.to_string(), props.clone()),
        )
        .await
        .expect("seed entity properties");
    tg.conn
        .execute(
            "UPDATE entities_fts SET properties = ?2 WHERE entity_id = ?1",
            (entity_id.to_string(), props),
        )
        .await
        .expect("seed entities_fts properties");
}

// ─── remember (mode-c pinned path) ───────────────────────────────────────

#[tokio::test]
async fn remember_pinned_fact_returns_commit_output() {
    let server = mock_server().await;
    let result = server
        .kremory_remember(Parameters(pinned_remember_params("ns-remember", "alice")))
        .await
        .expect("mode-c remember must succeed");

    assert_eq!(result.is_error, Some(false));
    let structured = result
        .structured_content
        .expect("remember returns structured payload");
    assert!(
        structured["episode_entity_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "episode_entity_id must be a non-empty string: {structured}"
    );
    assert!(
        structured["committed_at"].as_str().is_some(),
        "committed_at must be an ISO-8601 string: {structured}"
    );
}

// ─── recall shape / bridge proof (mock tier, no enrichment) ──────────────

/// Without enrichment a mock-tier recall may legitimately return 0 results
/// (see module docs). This test proves the BRIDGE: the `kremory_recall` tool
/// returns a well-formed structured payload (`count` + `results` array, with
/// `count == results.len()`) and never errors — regardless of hit count.
#[tokio::test]
async fn recall_structured_returns_wellformed_payload() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-shape", "alice")))
        .await
        .expect("remember must succeed");

    let recall = RecallParams {
        namespace: "ns-shape".into(),
        thread: None,
        query: "alice".into(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall must not error");

    let structured = result
        .structured_content
        .expect("structured recall returns structured payload");
    let count = structured["count"].as_u64().expect("count present");
    let results = structured["results"].as_array().expect("results array");
    assert_eq!(
        count as usize,
        results.len(),
        "count field must equal results array length: {structured}"
    );
}

// ─── DOD-001: mode-c remember → (simulated enrichment) → recall ≥ 1 ───────

#[tokio::test]
async fn recall_returns_pinned_entity_after_enrichment() {
    let mem = mock_memory().await;
    let server = KremoryMcpServer::new(mem.clone());

    // Distinctive token so the entity FTS arm can find it unambiguously.
    let subject = "Zephyrine";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-dod001", subject)))
        .await
        .expect("mode-c remember must succeed");

    // The pinned subject entity is created with id == subject slug. Simulate
    // the enrichment the mock LLM cannot perform (populate the FTS-indexed
    // name), then prove the recall tool surfaces it.
    simulate_enrichment(&mem, subject).await;

    let recall = RecallParams {
        namespace: "ns-dod001".into(),
        thread: None,
        query: subject.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall must succeed");

    let structured = result
        .structured_content
        .expect("structured recall returns structured payload");
    let count = structured["count"].as_u64().expect("count present");
    let results = structured["results"].as_array().expect("results array");
    assert!(
        count >= 1,
        "DOD-001: recall after remember + enrichment must return >=1 result, got count={count}: {structured}"
    );
    let names: Vec<&str> = results
        .iter()
        .filter_map(|r| r["entity_name"].as_str())
        .collect();
    assert!(
        names.iter().any(|n| n.contains(subject)),
        "recall results must contain the remembered entity {subject:?}, got names={names:?}"
    );
}

// ─── recall (text format) ─────────────────────────────────────────────────

#[tokio::test]
async fn recall_text_renders_entity_after_enrichment() {
    let mem = mock_memory().await;
    let server = KremoryMcpServer::new(mem.clone());

    let subject = "Bartholomew";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-text", subject)))
        .await
        .expect("remember must succeed");
    simulate_enrichment(&mem, subject).await;

    let recall = RecallParams {
        namespace: "ns-text".into(),
        thread: None,
        query: subject.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Text,
        // Entities template renders name + summary regardless of source_refs
        // (temporal_facts needs episodic edges the pinned entity may lack).
        template: RecallTemplateWire::Entities,
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall text must succeed");

    let structured = result
        .structured_content
        .expect("text recall returns structured payload with a `block` field");
    let block = structured["block"].as_str().expect("block is a string");
    assert!(
        block.contains(subject),
        "entities template block must mention the remembered entity: {block:?}"
    );
}

// ─── dream ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn dream_returns_summary_outcome() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-dream", "Cornelius")))
        .await
        .expect("remember must succeed");

    let dream = DreamParams {
        namespace: "ns-dream".into(),
        thread: None,
        batch_id: None,
    };
    let result = server
        .kremory_dream(Parameters(dream))
        .await
        .expect("dream must succeed with the mock LLM");

    assert_eq!(result.is_error, Some(false));
    let structured = result
        .structured_content
        .expect("dream returns structured payload");
    // Honest-zero fields are fine (mock LLM); the contract is that the real
    // DreamSummary fields are present + typed, not that they are non-zero.
    assert!(
        structured["duration_ms"].is_u64(),
        "dream output must carry duration_ms: {structured}"
    );
    assert!(
        structured["consolidation_ops_ran"].is_object(),
        "dream output must carry the consolidation_ops_ran object: {structured}"
    );
    assert!(
        structured["warnings"].is_array(),
        "dream output must carry a warnings array: {structured}"
    );
}

// ─── concurrency (ASMP-002) ───────────────────────────────────────────────

/// Two overlapping recalls dispatched onto SEPARATE `tokio::spawn` tasks
/// (not `tokio::join!`, which only interleaves both futures cooperatively
/// within a single task and never proves cross-thread simultaneous access).
/// Spawning each recall as its own task on the multi-thread runtime lets the
/// scheduler run them on different OS threads, genuinely exercising
/// concurrent access to the shared `Arc<Memory>` — the real ASMP-002
/// concern (libSQL under concurrent dispatch). Proves `Memory` is
/// `Send + Sync` enough for rmcp's concurrent-handler dispatch (stdio does
/// not serialize calls). If `Memory` were not `Sync` this would not compile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_overlapping_recalls_run_concurrently() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params(
            "ns-concurrent",
            "Delphine",
        )))
        .await
        .expect("remember must succeed");

    let recall = |query: &str| RecallParams {
        namespace: "ns-concurrent".into(),
        thread: None,
        query: query.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };

    let server_a = server.clone();
    let recall_a = recall("Delphine");
    let handle_a = tokio::spawn(async move { server_a.kremory_recall(Parameters(recall_a)).await });

    let server_b = server.clone();
    let recall_b = recall("design");
    let handle_b = tokio::spawn(async move { server_b.kremory_recall(Parameters(recall_b)).await });

    let a = handle_a
        .await
        .expect("first concurrent recall task must not panic");
    let b = handle_b
        .await
        .expect("second concurrent recall task must not panic");
    assert!(a.is_ok(), "first concurrent recall must succeed: {a:?}");
    assert!(b.is_ok(), "second concurrent recall must succeed: {b:?}");
}

// ─── error mapping ────────────────────────────────────────────────────────

#[tokio::test]
async fn remember_rejects_empty_namespace_with_invalid_params() {
    let server = mock_server().await;
    let mut params = pinned_remember_params("x", "alice");
    params.namespace = "".into();
    let err = server
        .kremory_remember(Parameters(params))
        .await
        .expect_err("empty namespace must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.to_lowercase().contains("namespace"),
        "invalid_params must name the namespace problem: {}",
        err.message
    );
}

#[tokio::test]
async fn remember_rejects_malformed_published_at_with_invalid_params() {
    let server = mock_server().await;
    let mut params = pinned_remember_params("ns-bad-ts", "alice");
    params.published_at = Some("yesterday at noon".into());
    let err = server
        .kremory_remember(Parameters(params))
        .await
        .expect_err("malformed published_at must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("published_at"),
        "invalid_params must name the offending field: {}",
        err.message
    );
}

#[tokio::test]
async fn recall_rejects_empty_namespace_with_invalid_params() {
    let server = mock_server().await;
    let recall = RecallParams {
        namespace: "".into(),
        thread: None,
        query: "x".into(),
        k: None,
        as_of: None,
        format: RecallFormat::Text,
        template: RecallTemplateWire::default(),
    };
    let err = server
        .kremory_recall(Parameters(recall))
        .await
        .expect_err("empty namespace must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(err.message.to_lowercase().contains("namespace"));
}

/// `as_of` point-in-time recall is declared-but-unimplemented in the
/// substrate (`kremory::memory::mod.rs` — `opts.as_of.is_some()` returns
/// `Err(Error::Unsupported { feature: "as_of point-in-time recall" })`
/// rather than silently ignoring the filter). `params.rs` documents this as
/// a deliberate fail-loud contract; this test locks that exact behaviour
/// through the `kremory_recall` tool: setting `as_of` must map to MCP
/// `internal_error`, never succeed and never silently drop the filter.
#[tokio::test]
async fn recall_as_of_fails_loud_with_internal_error() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-as-of", "alice")))
        .await
        .expect("remember must succeed");

    let recall = RecallParams {
        namespace: "ns-as-of".into(),
        thread: None,
        query: "alice".into(),
        k: None,
        as_of: Some("2026-01-01T00:00:00Z".into()),
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let err = server
        .kremory_recall(Parameters(recall))
        .await
        .expect_err("as_of must fail loud, never silently succeed");
    assert_eq!(
        err.code,
        ErrorCode::INTERNAL_ERROR,
        "as_of must map to internal_error (Unsupported), not invalid_params: {err:?}"
    );
    assert!(
        err.message.to_lowercase().contains("as_of"),
        "internal_error must name the unsupported as_of feature: {}",
        err.message
    );
}

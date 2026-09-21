use super::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use kremory::core::provider::MockEmbeddingProvider;
use std::collections::HashMap;
use tower::ServiceExt as _;

// ─── Dense fact arm end-to-end (real recall path) ──
//
// `pin_fact`'s caller-supplied `structured_facts` + `skip_extraction`
// path does NOT populate `facts.embedding` — only the LLM-extraction
// insert paths do (`ingest_with.rs:~1774`, `deferred.rs:~411`). So this
// section drives REAL LLM-extraction ingest (a staged `MockChatProvider`,
// mirrors `crates/kremory/tests/facade_fact_persistence_mock.rs::
// staged_mock`) — the ONLY way to get a real, embedded fact through the
// real `remember()` → deferred Phase 2 → `ingest_with` path, then queries
// it back through the real `/search` REST route. Per the lesson that
// (`instrument-real-data-flow-before-hypothesizing` §3), a hand-fed pure
// function cannot prove the arm is actually WIRED — only a real
// end-to-end run can.

/// Staged mock for `IntegerIdLlmExtractor`'s 3 prompt stages (substring-
/// keyed on prompt BOILERPLATE, content-agnostic — same keys
/// `facade_fact_persistence_mock.rs::staged_mock` uses), producing ONE
/// fact triple: `Priya relocated_to Berlin`. The embedded fact text
/// (`ingest_with.rs`'s `format!("{subject} {predicate} {object}")`) is
/// therefore exactly [`FACT_DENSE_QUERY`] — querying with that identical
/// string gives `MockEmbeddingProvider` (hash-based: same text -> same
/// vector) a PERFECT cosine match, deterministically surfacing this fact
/// as the dense arm's top hit without a real semantic model.
#[cfg(feature = "content-search")]
fn fact_dense_staged_mock() -> MockChatProvider {
    let mut map = HashMap::new();
    map.insert(
        "Each entity must appear exactly once".to_string(),
        r#"{"entities":[{"name":"Priya","entity_type_id":1},{"name":"Berlin","entity_type_id":2}]}"#
            .to_string(),
    );
    map.insert(
        "Output a JSON array of relationship name strings.".to_string(),
        r#"["relocated_to"]"#.to_string(),
    );
    map.insert(
        "Output a concise JSON array of objects with".to_string(),
        r#"[{"subject":"Priya","predicate":"relocated_to","object":"Berlin","is_entity_ref":true,"confidence":0.95}]"#
            .to_string(),
    );
    map.insert(
        "Are these two entities".to_string(),
        "\"different\"".to_string(),
    );
    map.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );
    MockChatProvider::new(map)
}

#[cfg(feature = "content-search")]
const FACT_DENSE_QUERY: &str = "Priya relocated_to Berlin";
#[cfg(feature = "content-search")]
const FACT_DENSE_NS: &str = "ns-fact-dense";

/// Builds a `Memory` via the staged-mock LLM extraction path, `remember`s
/// one episode, waits for Phase 2 to land the embedded fact, and returns
/// `(mem, episode_id)`.
#[cfg(feature = "content-search")]
async fn build_fact_dense_mem(fact_dense_enabled: bool) -> (Arc<Memory>, i64) {
    let llm: Arc<dyn ChatProvider> = Arc::new(fact_dense_staged_mock());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));
    let mem = kremory::Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .with_fact_dense_enabled(fact_dense_enabled)
        .await
        .expect("Memory::open with fact-dense knob");

    let commit = handlers::do_remember(
        &mem,
        RememberParams {
            namespace: FACT_DENSE_NS.to_string(),
            thread: None,
            content: "Priya relocated to Berlin last year.".to_string(),
            source_kind: Some(kremory_mcp::params::SourceKindWire::Chat),
            source_id: Some("mock-session".into()),
            published_at: None,
            structured_facts: Vec::new(),
            skip_extraction: false,
        },
    )
    .await
    .expect("do_remember with real LLM-extraction path");

    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("episode_entity_id must parse to an i64 rowid");
    mem.wait_for_processing(episode_id, std::time::Duration::from_secs(30))
        .await
        .expect("wait_for_processing must complete (Phase 2 lands the embedded fact)");

    (Arc::new(mem), episode_id)
}

/// With `fact_dense_enabled = true`, the fact
/// surfaces through the REAL `/search?mode=recall` route as
/// `kind: "fact"` with the CORRECT `source_episode_id` — driven through
/// the real `Memory` → `fuse_content_stream` → `rrf_fuse_with_facts` →
/// `handlers::do_recall` → `recall_mode_results` chain, not a hand-fed
/// pure function.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_fact_dense_arm_on_surfaces_kind_fact_with_source_episode_id() {
    let (mem, episode_id) = build_fact_dense_mem(true).await;
    let router = build_router(AppState { mem, rrf_k: 60 });

    let search = Request::builder()
        .method("GET")
        .uri(format!(
            "/search?q={q}&namespace={FACT_DENSE_NS}&k=10&mode=recall",
            q = FACT_DENSE_QUERY.replace(' ', "%20")
        ))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");

    let fact_hit = results
        .iter()
        .find(|r| r["kind"].as_str() == Some("fact"))
        .unwrap_or_else(|| {
            panic!("fact_dense_enabled=true must surface a kind:\"fact\" result: {json}")
        });
    assert_eq!(
        fact_hit["source_episode_id"].as_i64(),
        Some(episode_id),
        "kind:\"fact\" result must carry the CORRECT source_episode_id: {fact_hit}"
    );
}

/// The gate half: with `fact_dense_enabled = false`
/// (the default), the SAME fact — embedded identically, same query — is
/// NEVER surfaced as `kind: "fact"`. Proves the knob gates the arm rather
/// than the arm always firing regardless of config.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_fact_dense_arm_off_never_surfaces_kind_fact() {
    let (mem, _episode_id) = build_fact_dense_mem(false).await;
    let router = build_router(AppState { mem, rrf_k: 60 });

    let search = Request::builder()
        .method("GET")
        .uri(format!(
            "/search?q={q}&namespace={FACT_DENSE_NS}&k=10&mode=recall",
            q = FACT_DENSE_QUERY.replace(' ', "%20")
        ))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(
        !results.iter().any(|r| r["kind"].as_str() == Some("fact")),
        "fact_dense_enabled=false (default) must NEVER surface a kind:\"fact\" result: {json}"
    );
}

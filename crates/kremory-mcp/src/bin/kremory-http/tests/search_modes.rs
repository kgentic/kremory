use super::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use kremory::core::provider::MockEmbeddingProvider;
use tower::ServiceExt as _;

// ─── mode=content / mode=hybrid (benchmark-completion-roadmap W0.1) ──

/// `GET /search?mode=content` reaches the BM25 FTS5 stream
/// (`handlers::do_recall_content`) rather than the entity/fact path —
/// asserts a passage carrying the pinned subject's snippet comes back
/// through the SAME `{id, content, score}` wire contract `mode=recall`
/// uses.
/// `GET /search?format=text` must return kremory's OWN prompt-ready
/// rendering, and `format=structured` (the default) must stay byte-identical.
///
/// This endpoint hardcoded `RecallFormat::Structured` and discarded
/// `template`, so a REST caller could not reach the rendering the MCP tool
/// surface has always DEFAULTED to (`Text` + `TemporalFacts`). Because the
/// LoCoMo harness drives this endpoint, every published number measured the
/// rendering MCP agents do not get — worth ~+4pt answerability, +10.8pt on
/// temporal, on identical retrieval.
///
/// Drives the real HTTP path so it cannot pass while the parameter is
/// silently ignored — the failure mode being fixed.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_format_text_returns_the_prompt_ready_rendering() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http-format-text";
    pin_fact(&mem, ns, "Zephyrine").await;

    let text = router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!(
                    "/search?q=Zephyrine&namespace={ns}&k=10&format=text&template=temporal_facts"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(text.status(), StatusCode::OK);
    let json = body_json(text.into_body()).await;
    assert!(
        json.get("results").is_none(),
        "format=text must NOT return the structured results array — that would \
         mean the parameter was accepted and ignored: {json}"
    );
    let rendered = serde_json::to_string(&json).unwrap();
    assert!(
        rendered.contains("Zephyrine"),
        "format=text must carry the recalled subject in kremory's rendering: {json}"
    );

    // The default is unchanged: omitting `format` still yields the rows.
    let structured = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json = body_json(structured.into_body()).await;
    assert!(
        json["results"].as_array().is_some(),
        "omitting ?format= must stay byte-identical (structured rows): {json}"
    );
}

/// Regression guard: on the FUSED recall path, a content-derived
/// item must be reported as `kind: episode`, not `entity`.
///
/// `recall_mode_results` matched only `entity_type_name == "Fact"` and
/// defaulted everything else to `Entity`, while `core::search` tags a fused
/// content passage `"ContentPassage"` (`search.rs:1995`) and
/// `SearchResultKindWire::Episode` is documented as exactly that case. The
/// mapping was simply missing, so a consumer was told VERBATIM EPISODE TEXT
/// is a derived entity summary. Measured on conv0 before the fix: 3980 of
/// 3980 items across the top-20 came back `entity`, zero `episode`.
///
/// This drives the real HTTP path end-to-end rather than calling the mapper
/// with a hand-built `RetrievedContext` — a pure-function test would assert
/// the arithmetic of a mapping while remaining blind to whether the fused
/// path reaches it at all.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_recall_mode_labels_content_items_as_episode() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http-kind-fix";

    pin_fact(&mem, ns, "Zephyrine").await;

    // `mode=recall` (NOT `mode=content`) — the fused path the library's own
    // `recall()` uses and the one every benchmark measures.
    let search = Request::builder()
        .method("GET")
        .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10&mode=recall"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "expected fused results: {json}");

    let kinds: Vec<&str> = results.iter().filter_map(|r| r["kind"].as_str()).collect();
    assert_eq!(
        kinds.len(),
        results.len(),
        "every result must carry a `kind`: {json}"
    );
    assert!(
        kinds.contains(&"episode"),
        "the fused recall path must label content-derived items `episode`, not \
         collapse everything to `entity` — a consumer cannot otherwise tell \
         verbatim source text from a derived summary. got kinds={kinds:?}: {json}"
    );
}

#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_mode_content_returns_bm25_passage() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http-content";

    pin_fact(&mem, ns, "Zephyrine").await;

    let search = Request::builder()
        .method("GET")
        .uri(format!(
            "/search?q=Zephyrine&namespace={ns}&k=10&mode=content"
        ))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(
        results.iter().any(|r| {
            r["content"]
                .as_str()
                .is_some_and(|c| c.contains("Zephyrine"))
        }),
        "mode=content results[].content must carry the pinned subject's BM25 snippet: {json}"
    );
    for r in results {
        assert!(
            r["id"].as_str().is_some(),
            "result.id must be a string: {r}"
        );
        assert!(
            r["score"].as_f64().is_some(),
            "result.score must be numeric: {r}"
        );
        // `mode=content` items are
        // EPISODE-kind (a BM25-matched passage), never Fact.
        assert_eq!(
            r["kind"].as_str(),
            Some("episode"),
            "mode=content result.kind must be \"episode\": {r}"
        );
        assert!(
            r["source_episode_id"].is_null(),
            "mode=content result.source_episode_id must be null (episode's own id is \
             already `id`; only Fact items carry source_episode_id): {r}"
        );
    }
}

/// Real-substrate regression for the content-search AND->OR fallback
/// ladder (`TemporalGraph::content_search`, `core/search.rs`).
///
/// The sibling test above (`http_search_mode_content_returns_bm25_passage`)
/// queries with a SINGLE word (`q=Zephyrine`) — AND-joining one token is
/// indistinguishable from OR-joining one token, so that test cannot
/// detect a regression in the AND->OR ladder (it was GREEN even when the
/// underlying substrate silently returned empty for every multi-word,
/// natural-language query — the exact failure the LoCoMo benchmark
/// harness hit: 100% empty content-mode recalls on real questions).
///
/// This test drives the REAL ingest pipeline (`pin_fact` -> `do_remember`,
/// same production path `insert_episode_with_group` uses — not a
/// hand-built fixture) then queries with a multi-word sentence whose
/// tokens ("Did"/"invent"/"anything"/"remarkable") are ABSENT from the
/// pinned content except "Zephyrine" — an AND-only match is impossible by
/// construction, so a non-empty result here can only have come from the
/// OR-fallback rung. Without the fallback (pre-fix `content_search`),
/// this asserts and fails.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_mode_content_natural_language_query_uses_or_fallback() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http-content-nl";

    // Pinned content: "Zephyrine wrote the first algorithm" (see `pin_fact`).
    pin_fact(&mem, ns, "Zephyrine").await;

    let question = "Did Zephyrine invent anything remarkable";
    let search = Request::builder()
        .method("GET")
        .uri(format!(
            "/search?q={question}&namespace={ns}&k=10&mode=content",
            question = question.replace(' ', "%20")
        ))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "mode=content must rescue a multi-word natural-language query via the \
         AND->OR fallback ladder — an AND-only match is impossible here (query \
         tokens 'invent'/'anything'/'remarkable' are absent from the pinned \
         content). Empty here reproduces the LoCoMo-benchmark content-search bug: {json}"
    );
    assert!(
        results.iter().any(|r| {
            r["content"]
                .as_str()
                .is_some_and(|c| c.contains("Zephyrine"))
        }),
        "OR-fallback result must still carry the pinned subject's snippet: {json}"
    );
}
/// `GET /search` with no `?mode=` now defaults to `mode=hybrid` (RRF fusion
/// of the entity/fact recall stream + the BM25 content stream), changed from
/// the pre-W0.1 `recall` default per an earlier LoCoMo diagnostic run
/// (entity-graph `recall` judged 40.2% vs 71.4% hybrid). Guard: the default
/// arm reaches the fused surface and still finds a pinned entity.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_default_mode_is_hybrid() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http-default-mode";

    pin_fact(&mem, ns, "Ada").await;

    let search = Request::builder()
        .method("GET")
        .uri(format!("/search?q=Ada&namespace={ns}&k=10"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(
        results
            .iter()
            .any(|r| r["content"].as_str().is_some_and(|c| c.contains("Ada"))),
        "omitted ?mode= must default to hybrid and still find the pinned entity: {json}"
    );
}
/// `format=text` must retrieve the SAME item set, in the SAME
/// relative ORDER, `format=structured` does for the same `mode` —
/// proven via a scenario engineered so `mode=recall`'s single-pass
/// internal fusion (`.raw()`) and `mode=hybrid`'s union with a SEPARATE
/// `mode=content` BM25 pass disagree on ranking:
///
/// - a weak entity match (subject == query text, but episode content is
///   unrelated — findable only via the entity/fact arm)
/// - 20 keyword-irrelevant decoys (noise the entity/fact + internal
///   content fusion has to rank against)
/// - ONE genuinely relevant content-only passage
///
/// Empirically, `mode=recall` ranks the
/// relevant passage LAST (rank 3 of 3, out-competed by the decoy +
/// entity within its single fusion pass); `mode=hybrid` ranks it FIRST
/// (its RRF contribution is SUMMED across both the recall arm AND the
/// separate content arm, per `rrf_merge`'s dedup-by-id accumulation).
///
/// Before the fix, `format=text` read `query.mode` ONLY to gate
/// `mode=content` (422) — for `mode=recall` and `mode=hybrid` alike it
/// always computed via a single `handlers::do_recall(format: Text)`
/// call, i.e. `mode=recall`'s underlying fetch. So
/// `format=text&mode=hybrid` silently rendered the relevant passage
/// LAST, disagreeing with what `format=structured&mode=hybrid` (and any
/// other consumer of the documented hybrid ranking) actually retrieves.
#[cfg(feature = "content-search")]
#[tokio::test]
async fn http_search_format_text_hybrid_matches_structured_hybrid_ranking_td196() {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));
    let mem = Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .await
        .expect("in-memory Memory must build");
    let mem = Arc::new(mem);
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-td196-hybrid-ranking";

    let entity_params = RememberParams {
        namespace: ns.to_string(),
        thread: None,
        content: "This document discusses corporate wellness policy trends.".to_string(),
        source_kind: Some(kremory_mcp::params::SourceKindWire::Note),
        source_id: Some("entity-only".into()),
        published_at: None,
        structured_facts: vec![kremory_mcp::params::StructuredFactWire {
            subject: "Zephyrine".to_string(),
            predicate: "is_a".to_string(),
            object: "notable subject".to_string(),
            valid_at: None,
            invalid_at: None,
        }],
        skip_extraction: true,
    };
    handlers::do_remember(&mem, entity_params)
        .await
        .expect("entity remember must succeed");

    for i in 0..20 {
        let params = RememberParams {
            namespace: ns.to_string(),
            thread: None,
            content: format!("Quarterly report section {i} covers regional sales figures."),
            source_kind: Some(kremory_mcp::params::SourceKindWire::Note),
            source_id: Some(format!("noise-{i}")),
            published_at: None,
            structured_facts: Vec::new(),
            skip_extraction: true,
        };
        handlers::do_remember(&mem, params)
            .await
            .expect("noise remember must succeed");
    }

    let relevant_params = RememberParams {
        namespace: ns.to_string(),
        thread: None,
        content: "Zephyrine mentioned the wobbling turnstile incident yesterday.".to_string(),
        source_kind: Some(kremory_mcp::params::SourceKindWire::Note),
        source_id: Some("relevant-content".into()),
        published_at: None,
        structured_facts: Vec::new(),
        skip_extraction: true,
    };
    handlers::do_remember(&mem, relevant_params)
        .await
        .expect("relevant remember must succeed");

    // Reference: format=structured&mode=hybrid ranks the relevant
    // passage FIRST (summed RRF contribution across both streams).
    let structured = router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/search?q=Zephyrine&namespace={ns}&k=3&mode=hybrid"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(structured.status(), StatusCode::OK);
    let structured_json = body_json(structured.into_body()).await;
    let results = structured_json["results"]
        .as_array()
        .expect("results array");
    assert!(
        results[0]["content"]
            .as_str()
            .is_some_and(|c| c.contains("Zephyrine mentioned the wobbling turnstile")),
        "structured&hybrid must rank the relevant passage FIRST: {structured_json}"
    );

    // format=text&mode=hybrid must retrieve the SAME item set in the
    // SAME relative order: the relevant passage must render BEFORE the
    // irrelevant decoy, not after.
    let text = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!(
                    "/search?q=Zephyrine&namespace={ns}&k=3&format=text&mode=hybrid&template=entities"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(text.status(), StatusCode::OK);
    let text_json = body_json(text.into_body()).await;
    let block = text_json["block"]
        .as_str()
        .expect("format=text must return a block string");
    let relevant_pos =
        block.find("Zephyrine mentioned the wobbling turnstile incident yesterday.");
    let decoy_pos = block.find("Quarterly report section");
    match (relevant_pos, decoy_pos) {
        (Some(r), Some(d)) => assert!(
            r < d,
            "format=text&mode=hybrid must render the SAME item set in the SAME \
             order format=structured&mode=hybrid retrieves — the \
             relevant passage must appear BEFORE the irrelevant decoy, not \
             after (i.e. format=text must not silently fall back to \
             mode=recall's ordering). Got block:\n{block}"
        ),
        _ => panic!(
            "format=text&mode=hybrid must contain BOTH the relevant passage and \
             the decoy (same item set as structured&hybrid): {block}"
        ),
    }
}

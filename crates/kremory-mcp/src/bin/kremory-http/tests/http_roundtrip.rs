use super::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt as _;

#[tokio::test]
async fn http_roundtrip_memories_search_delete_consolidation() {
    let mem = mock_memory().await;
    let router = build_router(AppState {
        mem: mem.clone(),
        rrf_k: 60,
    });
    let ns = "ns-http";

    // POST /memories → 201 + non-empty {id}. (Extraction path + mock LLM
    // means this specific episode won't itself be recall-findable, but the
    // route contract — 201 + an id — is what's asserted here.)
    let post = Request::builder()
        .method("POST")
        .uri("/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "content": "Ada Lovelace wrote the first algorithm.",
                "namespace": ns,
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(post).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let json = body_json(resp.into_body()).await;
    assert!(
        json["id"].as_str().is_some_and(|s| !s.is_empty()),
        "POST /memories must return a non-empty id: {json}"
    );

    // Seed a recall-findable pinned fact so GET /search has a deterministic
    // hit with mock providers, then assert the flattened `content` carries
    // the fact text (the benchmark substring path, end-to-end through the
    // route + adapter).
    pin_fact(&mem, ns, "Zephyrine").await;
    // `mode=recall` is explicit here (not the default `hybrid`): under a
    // default-features build (no `content-search`), B1 fail-loud makes
    // `hybrid`/`content` a hard 422, so the servable roundtrip mode is
    // `recall`. `recall` returns the pinned entity/fact regardless of the
    // `content-search` feature, keeping this route-contract test
    // config-agnostic. The 422 fail-loud contract is asserted separately in
    // `hybrid_without_content_search_feature_hard_fails`.
    let search = Request::builder()
        .method("GET")
        .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10&mode=recall"))
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(search).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let results = json["results"].as_array().expect("results array");
    assert!(
        results.iter().any(|r| {
            r["content"]
                .as_str()
                .is_some_and(|c| c.contains("Zephyrine"))
        }),
        "GET /search results[].content must contain the pinned subject: {json}"
    );
    // Each result carries the id/content/score contract shape.
    for r in results {
        assert!(
            r["id"].as_str().is_some(),
            "result.id must be a string: {r}"
        );
        assert!(
            r["content"].as_str().is_some(),
            "result.content must be a string: {r}"
        );
        assert!(
            r["score"].as_f64().is_some(),
            "result.score must be numeric: {r}"
        );
        // `mode=recall` items carry no per-fact episode provenance
        // (the connected facts' own `source_episode_ids` are discarded by
        // `flatten_result_content`'s join — see the doc comment on
        // `SearchResultWire::source_episode_id`).
        //
        // ⚠️ AMENDED. This previously asserted
        // `kind == "entity"` for EVERY `mode=recall` item — which **encoded
        // the bug as the contract**. In a `content-search` build the fused
        // recall path also returns content passages, and one of them
        // (`"Episode #2: Zephyrine wrote the first algorithm"`) was being
        // reported as an `entity`. The mapper simply never matched
        // `"ContentPassage"`. The correct invariant is not "everything is an
        // entity" — it is "nothing on this path is a FACT unless the
        // dense-fact arm is on", which the sibling tests
        // `http_search_fact_dense_arm_{off,on}_*` already pin.
        assert!(
            matches!(r["kind"].as_str(), Some("entity") | Some("episode")),
            "mode=recall result.kind must be entity or episode (never fact with \
             the dense-fact arm off): {r}"
        );
        assert!(
            r["source_episode_id"].is_null(),
            "mode=recall result.source_episode_id must be null (only Fact items carry it): {r}"
        );
    }

    // POST /consolidation/{cycle} WITHOUT ?namespace= → 422.
    let no_ns = Request::builder()
        .method("POST")
        .uri("/consolidation/creative")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(no_ns).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "missing namespace must be a loud 422, not a silent no-op default"
    );
    let json = body_json(resp.into_body()).await;
    assert!(
        json["error"]
            .as_str()
            .is_some_and(|e| e.to_lowercase().contains("namespace")),
        "422 body must name the missing namespace param: {json}"
    );

    // POST /consolidation/{cycle}?namespace=ns → 200.
    let with_ns = Request::builder()
        .method("POST")
        .uri(format!("/consolidation/creative?namespace={ns}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(with_ns).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // DELETE /namespaces/{ns} → 200 + the full per-table outcome (TD-247).
    // `ns` had a real pinned fact written into it earlier in this test
    // (the `Zephyrine` fixture the GET /search assertions above check
    // for), so this is a REAL erasure, not an empty no-op — the
    // discriminating case the old `{"deleted": <entities>}` shape could
    // not represent: entity-count alone can legitimately read `0` on a
    // real erasure (shared-entity preservation), so the strong assertion
    // here is `is_empty` being `false`, not any single field being
    // nonzero.
    let del = Request::builder()
        .method("DELETE")
        .uri(format!("/namespaces/{ns}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(del).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    for field in ["entities", "facts", "episodes", "edges"] {
        assert!(
            json[field].is_u64(),
            "DELETE must return a numeric {field} count: {json}"
        );
    }
    assert_eq!(
        json["is_empty"].as_bool(),
        Some(false),
        "DELETE erased a real pinned fact — is_empty must be false, not just \
         some individual count field: {json}"
    );
}

#[tokio::test]
async fn health_returns_200() {
    let mem = mock_memory().await;
    let router = build_router(AppState { mem, rrf_k: 60 });
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// `GET /health`
/// must report the server's ACTIVE scoring config + build-feature flags so
/// the bench harness (`provenance.build_provenance`) stamps a FAITHFUL
/// provenance record instead of re-reading the harness process's env (which
/// can silently differ from the server's — the config-mismatch that produced
/// a bogus benchmark number).
///
/// The load-bearing assertion: a value set via the `KREMORY_CONTENT_WEIGHT`
/// / `KREMORY_RRF_K` boot override appears in `/health`'s `scoring` block —
/// proving the handler reads the LIVE `SearchConfig` (via
/// `Memory::search_config`, which `open_graph` populates from these env
/// overrides at construction) rather than a hardcoded default. nextest runs
/// each test in its own process, so this env mutation is isolated (same
/// pattern as `providers::search_env_overrides_apply_*`).
#[tokio::test]
async fn health_reports_active_scoring_config_and_features() {
    std::env::set_var("KREMORY_CONTENT_WEIGHT", "2.5");
    std::env::set_var("KREMORY_RRF_K", "42");
    // `open_graph` (reached through the builder in `mock_memory`) applies the
    // overrides to the Engine's `SearchConfig` at construction.
    let mem = mock_memory().await;
    std::env::remove_var("KREMORY_CONTENT_WEIGHT");
    std::env::remove_var("KREMORY_RRF_K");

    let router = build_router(AppState { mem, rrf_k: 42 });
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;

    // Build-feature booleans reflect the compiled build (cfg!-evaluated).
    assert_eq!(
        json["content_search"].as_bool(),
        Some(cfg!(feature = "content-search")),
        "/health content_search must reflect the compiled feature: {json}"
    );
    assert_eq!(
        json["rerank"].as_bool(),
        Some(cfg!(feature = "rerank")),
        "/health rerank must reflect the compiled feature: {json}"
    );
    assert_eq!(
        json["prometheus"].as_bool(),
        Some(cfg!(feature = "prometheus")),
        "/health prometheus must reflect the compiled feature: {json}"
    );

    // Scoring block reflects the LIVE (env-overridden) SearchConfig — NOT a
    // hardcoded default. This is the faithfulness guarantee this test exists to prove.
    let scoring = &json["scoring"];
    assert_eq!(
        scoring["content_stream_weight"].as_f64(),
        Some(2.5),
        "content_stream_weight must be the KREMORY_CONTENT_WEIGHT override (2.5), \
         proving /health reads the live SearchConfig: {json}"
    );
    assert_eq!(
        scoring["rrf_k"].as_u64(),
        Some(42),
        "rrf_k must be the KREMORY_RRF_K override (42): {json}"
    );
    // The remaining post-RRF axes must be present + numeric (defaults here).
    assert!(
        scoring["graph_degree_weight"].as_f64().is_some(),
        "graph_degree_weight must be present + numeric: {json}"
    );
    assert!(
        scoring["temporal_weight"].as_f64().is_some(),
        "temporal_weight must be present + numeric: {json}"
    );
}

/// B1 fail-loud invariant: on a build WITHOUT `content-search`,
/// an explicit `mode=hybrid`/`mode=content` request MUST hard-fail (422),
/// never silently degrade to recall-only. Silent degradation here served a
/// ~40%-surface answer under the DEFAULT `hybrid` mode and produced the
/// LoCoMo 13.9% garbage baseline. Only compiled/relevant when the feature is
/// OFF (with it ON, hybrid/content are servable and return 200 — covered by
/// `http_search_mode_content_returns_bm25_passage`).
#[cfg(not(feature = "content-search"))]
#[tokio::test]
async fn hybrid_without_content_search_feature_hard_fails() {
    let mem = mock_memory().await;
    let router = build_router(AppState { mem, rrf_k: 60 });
    let ns = "ns-http-faildude";
    for mode in ["hybrid", "content"] {
        let req = Request::builder()
            .method("GET")
            .uri(format!("/search?q=anything&namespace={ns}&k=10&mode={mode}"))
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "mode={mode} without the content-search feature must hard-fail \
             422 (B1 fail-loud), not silently degrade to recall"
        );
        let json = body_json(resp.into_body()).await;
        assert!(
            json["error"]
                .as_str()
                .is_some_and(|e| e.to_lowercase().contains("content-search")),
            "422 body must name the missing content-search feature: {json}"
        );
    }
}

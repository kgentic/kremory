// ─── Real-LLM discovery-OUTCOME test ──────────────────────────────────────────
//
// Closes the gaps the existing `#[ignore]`d smoke (`tests/phase_d_pass_0.rs`)
// leaves open: that smoke (a) can pass VACUOUSLY (if stochastic extraction
// produced zero catch-alls, discovery early-returns and every assertion still
// passes), and (b) never asserts that evidence was RETYPED. This test drives a
// REAL model through the discovery path with a DETERMINISTIC trigger:
//
// - The catch-all bucket is seeded directly (5 drugs) — discovery cannot run
//   vacuously; the trigger is asserted to exist before the call.
// - Drugs are a type genuinely ABSENT from the 10 defaults, so a competent
//   model's proposal is NOT (correctly) rejected by anti-redundancy.
// - `embedder = None` skips the gate and takes `retype_evidence_all`, making the
//   RETYPE COUNT deterministic. The only stochastic element is "did the real
//   model propose >=1 valid type for an unambiguous drug cluster" — which is
//   exactly the discovery-quality signal this test exists to surface (and is
//   reliable for gemma4-e2b on a clear cluster).
//
// `#[ignore]` + `feature = "llm-integration"`: needs live Ollama. Run with:
//   OLLAMA_CHAT_MODEL=gemma4:e4b cargo test -p kremory \
//     --features llm-integration --lib discover_types_real_llm -- --ignored --nocapture
//
// MODEL TIER (load-bearing): defaults to `gemma4:e4b` — the
// DEFERRED-phase QUALITY model (90%, Phase 2 per tests/llm_integration.rs:1-25),
// NOT the interactive `gemma4-e2b`. Discovery is a background/quality task. The
// smoke run that built this test showed `gemma4-e2b` proposing a placeholder
// name `"..."` → rejected (`ellipsis_placeholder`) → ZERO types discovered,
// while `gemma4:e4b` proposes "Over-the-Counter Pain Reliever" → accepted → all
// evidence retyped. A consumer wiring only the fast interactive model for dreams
// gets silent zero-discovery.
//
// Complements the
// deterministic `td050_full_workflow_tests` (scripted proposal) by proving the
// REAL model end of the chain.

use super::*;
use crate::core::entity_types::ensure_default_types_seeded;
use crate::core::schema::TemporalGraph;
use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use std::sync::Arc;

async fn count_catch_alls(conn: &libsql::Connection, group_id: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = ?1 AND entity_type_id = 0",
            libsql::params![group_id],
        )
        .await
        .expect("count catch-alls");
    rows.next()
        .await
        .expect("row")
        .expect("count row")
        .get::<i64>(0)
        .expect("count col")
}

async fn type_id_by_name(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id, name],
        )
        .await
        .expect("select type id");
    rows.next()
        .await
        .expect("row")
        .expect("discovered-type row must exist")
        .get::<i64>(0)
        .expect("id col")
}

#[tokio::test]
#[ignore]
async fn discover_types_real_llm_proposes_accepts_and_retypes() {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "med")
        .await
        .expect("seed defaults 0..=9");

    // Deterministic catch-all trigger: 5 drugs (a type ABSENT from the 10
    // defaults). entity_type_id = 0 = catch-all.
    let now = Utc::now().to_rfc3339();
    let drugs = [
        "aspirin",
        "ibuprofen",
        "paracetamol",
        "metformin",
        "atorvastatin",
    ];
    for d in drugs {
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES (?1, 0, ?2, ?3)",
            libsql::params![d.to_string(), now.clone(), "med".to_string()],
        )
        .await
        .expect("insert catch-all entity");
    }

    // Non-vacuous guarantee: the discovery trigger MUST exist.
    assert_eq!(
        count_catch_alls(&conn, "med").await,
        5,
        "5 catch-all entities must exist before discovery — guards against a vacuous pass"
    );

    let base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Deferred-phase QUALITY model (Phase 2, 90% per tests/llm_integration.rs:1-25).
    // gemma4-e2b (interactive) is too weak for discovery — see module doc.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&chat_model)
        .timeout_seconds(120)
        .keep_alive("1h")
        .build()
        .expect("Ollama LLM builder must succeed");

    // embedder = None → anti-redundancy gate skipped + retype_evidence_all
    // (deterministic retype count). Discovery itself is fully real.
    let result = discover_types(
        &*llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "med",
            embedder: None,
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must not error with a live model");

    // Surface what the real model actually discovered (operator observability).
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.dream.discover_types",
            model = %chat_model,
            proposed = ?result.types_proposed.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            accepted = ?result.types_accepted.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            rejected = ?result.types_rejected.iter().map(|(t, r)| format!("{}:{r}", t.name)).collect::<Vec<_>>(),
            entities_retyped = result.entities_retyped,
            "td050-real-llm discover_types result"
        );
    }

    // Gap: the real model actually produced a usable proposal (not vacuous,
    // not scripted). Reliable for an unambiguous drug cluster; if this flaps,
    // that IS the discovery-quality signal this test surfaces.
    assert!(
        !result.types_proposed.is_empty(),
        "real model must propose >=1 type for an unambiguous Drug cluster"
    );
    assert!(
        !result.types_accepted.is_empty(),
        "the proposal must survive the shape validator and be accepted; rejected={:?}",
        result.types_rejected
    );

    // Gap: evidence retyped (deterministic in degraded mode → all 5).
    assert_eq!(
        result.entities_retyped, 5,
        "degraded-mode accept retypes ALL catch-all evidence"
    );

    // Consistency: every accepted type persisted above the seeded range.
    for t in &result.types_accepted {
        let id = type_id_by_name(&conn, "med", &t.name).await;
        assert!(
            id > 9,
            "discovered type '{}' must allocate id>9 (above seeded 0..=9), got {id}",
            t.name
        );
    }

    // Retype provenance: every drug entity now non-catch-all with DreamPass0.
    let mut rows = conn
        .query(
            "SELECT entity_type_id, entity_type_source FROM entities WHERE group_id = 'med'",
            (),
        )
        .await
        .expect("query retyped entities");
    let mut n = 0usize;
    while let Some(r) = rows.next().await.expect("row") {
        let tid: i64 = r.get(0).expect("entity_type_id");
        let src: String = r.get(1).expect("entity_type_source");
        assert!(
            tid > 9,
            "every drug entity must be retyped above the seeded range, got {tid}"
        );
        assert_eq!(src, "DreamPass0", "retype provenance must be 'DreamPass0'");
        n += 1;
    }
    assert_eq!(n, 5, "all 5 drug entities retyped — none left at id=0");
}

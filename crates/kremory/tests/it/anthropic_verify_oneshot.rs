#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Isolated Pass 4 verify smoke — diagnose Phase D empty-decisions bug.
//!
//! Skips Phase 1 NER ingest entirely. Manually seeds 1 entity into an in-memory
//! TemporalGraph + invokes run_consistency_check with anthropic claude-haiku-4-5.
//! The `eprintln!("[CONSISTENCY_CHECK_RAW] ...")` injected in consistency_check.rs
//! line 310 will dump the raw LLM response verbatim.
//!
//! Run: `export ANTHROPIC_API_KEY=...; cargo test --features llm-integration
//!   --test it anthropic_verify_oneshot:: -- --ignored anthropic_verify_oneshot --nocapture`
//!
//! Tells us:
//! 1. Did anthropic actually return a response, or did the call fail?
//! 2. Did it return JSON with decisions, or empty `{}`, or something else?
//! 3. If empty: prompt+schema bug. If valid: parser bug.

#[cfg(feature = "llm-integration")]
#[tokio::test]
#[ignore]
async fn anthropic_verify_oneshot_dumps_raw_response() {
    use std::sync::Arc;

    use autoagents_llm::backends::anthropic::Anthropic;
    use autoagents_llm::builder::LLMBuilder;
    use kremory::core::dream::consistency_check::{
        run_consistency_check, ConsistencyCheckOpts, RunConsistencyCheckParams,
    };
    use kremory::core::provider::{ArcChatProvider, DeterministicEmbeddingProvider};
    use kremory::core::schema::TemporalGraph;
    use kremory::DynEmbeddingProvider;

    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");

    // ── Build anthropic LLM (same shape kremory-eval/consistency_check_sweep uses) ──
    let llm_inner: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(&key)
        .model("claude-haiku-4-5-20251001")
        .timeout_seconds(60)
        .build()
        .expect("Anthropic builder must succeed");
    let llm: Arc<dyn autoagents_llm::chat::ChatProvider + Send + Sync> = llm_inner;
    let llm_wrapped = ArcChatProvider::new(llm);

    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    // ── In-memory graph + manually seed ONE deliberately wrong-typed entity ──
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory must succeed");

    // Migration 010 already seeds default entity_types vocabulary. Look up the
    // Location + Organisation IDs (deliberately mistyped Apple as Location).
    let mut rows = graph
        .conn
        .query(
            "SELECT id, name FROM entity_types WHERE group_id = 'default' AND name IN ('Location', 'Organisation', 'Org')",
            (),
        )
        .await
        .expect("query entity_types must succeed");
    let mut location_id: i64 = -1;
    while let Some(row) = rows.next().await.expect("row iter") {
        let id: i64 = row.get(0).expect("id");
        let name: String = row.get(1).expect("name");
        eprintln!("[ONESHOT] entity_type seeded: id={id} name={name}");
        if name == "Location" {
            location_id = id;
        }
    }
    drop(rows);
    assert!(location_id > 0, "Location type must be in default seed");

    // Seed Apple-as-Location with high confidence + facts pointing to Org-shaped predicates
    graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, properties, recorded_at, group_id, entity_type_id, entity_type_source, ner_confidence) \
             VALUES ('apple-mistyped', '{}', datetime('now'), 'default', ?1, 'Phase1Ner', 0.9)",
            libsql::params![location_id],
        )
        .await
        .expect("seed Apple-as-Location must succeed");

    // Skip facts seeding — not essential for LLM-verify diagnostic; the eprintln
    // in consistency_check.rs:310 dumps raw LLM response regardless of facts.

    // ── Invoke run_consistency_check — eprintln in consistency_check.rs:310 dumps raw ──
    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: 0.0, // force flag everything regardless of cosine
        max_candidates_per_run: Some(10),
        verify_model_override: Some("claude-haiku-4-5-20251001".to_string()),
        dry_run: false,
    };

    eprintln!("\n========================================");
    eprintln!("[ONESHOT] Invoking run_consistency_check with anthropic + 1 entity...");
    eprintln!("========================================\n");

    let summary = run_consistency_check(
        &graph.conn,
        RunConsistencyCheckParams {
            embedder: &*embedder,
            llm: &llm_wrapped,
            opts,
        },
    )
    .await
    .expect("run_consistency_check must succeed");

    eprintln!("\n========================================");
    eprintln!("[ONESHOT] Summary: {summary:?}");
    eprintln!("========================================\n");

    // Diagnostic assertion — at minimum the verify call should have been invoked
    assert!(summary.scanned >= 1, "must scan ≥1 entity");
}

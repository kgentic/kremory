#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Real-LLM validation of the reconciled dream pass chain — 5-lane fixture.
//!
//! Governing spec: `.ai-docs/specs/dream-phase-reconciliation-v2-2026-06-30.md`
//! (§D3 canonical 5-pass ordering). Phase 6 DoD E2E.
//!
//! Design note: this plants FIVE disjoint fixture lanes — one per §D3 pass
//! (type_discovery, aliases, reclassify, consistency_check, canonicalize) —
//! directly via the graph, then makes ONE real `mem.dream()` call. Every
//! LLM-driven lane asserts `>= 1` (never `== N` — real-LLM output is
//! stochastic); the two deterministic lanes (aliases, canonicalize) assert
//! exact post-conditions, mirroring `tests/dream_phase2_deterministic_passes.rs`.
//!
//! Run: `cargo test -p kremory --features llm-integration,test-utils --test dream_e2e_real_llm -- --ignored --nocapture`
//! Requires: Ollama at localhost:11434 with `gemma4:e4b` + `nomic-embed-text`.
//!
//! Tier: currently tier-3 (`llm-integration` + `#[ignore]`, live-Ollama only).
//! Promotion to the project's tier-2 (`--features llm-smoke`, offline via a
//! KREMORY_VCR cassette like `golden_path_smoke.rs`) is tracked as TD-093 — do
//! NOT naively re-gate on `llm-smoke` without recording a cassette first (it
//! would fail wherever Ollama is absent).
//!
//! ⚠️ CURRENT STATUS (2026-07-01): RED on the 3 LLM lanes (type_discovery,
//! reclassify, consistency_check) — they run with an EMPTY model string and
//! produce zero output. Root cause = TD-094 (the "Option-1 2026-06-23" refactor
//! left dream model-threading half-built: discover_types + reclassify hardcode
//! `model_str = String::new()`, consistency_check gets `verify_model_override:
//! None`). The 2 DETERMINISTIC lanes (aliases, canonicalize) PASS. This test is
//! the REGRESSION GUARD for TD-094 and goes GREEN when it is fixed. It is
//! `#[ignore]` so it does not affect the default gate. Do NOT loosen the
//! assertions to force a pass — the red is the whole point.

use std::sync::Arc;

use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

#[cfg(feature = "llm-integration")]
mod helpers;

/// Unit-normalised embedding with all components equal → any two are cosine ≈ 1.0.
fn unit_vec(dim: usize) -> Vec<f32> {
    let v = 1.0_f32 / (dim as f32).sqrt();
    vec![v; dim]
}

/// Plant a catch-all (`entity_type_id = 0`, `Phase1Ner`) entity whose `id` is a
/// semantically clear name — Lane A (type_discovery) / Lane C / Lane E
/// (deterministic passes) candidate. Mirrors `tests/phase_e_reclassify.rs::insert_entity`.
// Test helper: clippy.toml Rule-5 exempt (test helpers may carry a documented
// too_many_arguments allow, per feedback_no_clippy_allow_in_src_args_as_object;
// TD-042 args-as-object targets `src/` production fns only).
#[allow(clippy::too_many_arguments)]
async fn plant_entity_row(
    graph: &TemporalGraph,
    id: &str,
    group_id: &str,
    entity_type_id: i64,
    entity_type_source: &str,
    ner_confidence: f64,
) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, group_id, entity_type_id, entity_type_source, ner_confidence, \
              recorded_at, updated_at, entity_type_assigned_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?6)",
            libsql::params![
                id.to_string(),
                group_id.to_string(),
                entity_type_id,
                entity_type_source.to_string(),
                ner_confidence,
                now
            ],
        )
        .await
        .expect("plant entity row");
}

/// Look up an `entity_types.id` by name within `group_id`. Runtime query — never
/// hardcode registry ids, they are seeded per-group by `ensure_default_types_seeded`.
async fn entity_type_id_by_name(graph: &TemporalGraph, group_id: &str, name: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id.to_string(), name.to_string()],
        )
        .await
        .expect("query entity_types by name");
    let row = rows
        .next()
        .await
        .expect("row read")
        .unwrap_or_else(|| panic!("entity_types row for name={name} in group={group_id} must exist"));
    row.get::<i64>(0).expect("id at index 0")
}

/// Insert a custom (wrong-on-purpose) entity type into the registry, returning its id.
async fn insert_custom_entity_type(
    graph: &TemporalGraph,
    group_id: &str,
    name: &str,
    description: &str,
) -> i64 {
    let now = chrono::Utc::now().to_rfc3339();
    // Pick an id well above the default vocabulary range so it can never collide.
    let mut rows = graph
        .conn
        .query(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query next entity_type id");
    let row = rows.next().await.expect("row read").expect("row present");
    let next_id: i64 = row.get(0).expect("next id at index 0");
    let new_id = next_id.max(900);
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                new_id,
                group_id.to_string(),
                name.to_string(),
                description.to_string(),
                now
            ],
        )
        .await
        .expect("insert custom entity_type");
    new_id
}

/// Map of entity id → (entity_type_id, entity_type_source) for the group.
async fn entity_fields(graph: &TemporalGraph, group_id: &str) -> Vec<(String, i64, String)> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, entity_type_id, entity_type_source FROM entities WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query entity fields");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration") {
        let id: String = row.get(0).expect("id");
        let type_id: i64 = row.get(1).expect("entity_type_id");
        let source: String = row.get(2).expect("entity_type_source");
        out.push((id, type_id, source));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real LLM — run explicitly with --features llm-integration,test-utils --ignored"]
#[cfg(feature = "llm-integration")]
async fn dream_e2e_real_llm_five_pass_chain() {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use autoagents_llm::embedding::EmbeddingBuilder;
    use chrono::Utc;
    use kremory::core::disambiguation::{
        insert_potential_alias_fact, AliasProvenance, InsertPotentialAliasFactParams,
    };
    use kremory::core::graph::FactInsert;
    use metrics_util::debugging::DebuggingRecorder;

    use helpers::ollama_adapter::OllamaEmbedderAdapter;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Dream is a QUALITY pass — use gemma4:e4b (deferred-quality default). Overridable.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&chat_model)
        .timeout_seconds(180)
        .keep_alive("1h")
        .build()
        .expect("Ollama LLM builder must succeed");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder must succeed");
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("dream-e2e-5pass");
    let mem = Memory::open(dir.path().join("dream_e2e.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_embedder(emb.clone())
        .embedding_dim(768)
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open must succeed");

    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    // This fixture plants entities directly via raw SQL (never calls mem.remember()),
    // so the lazy `ensure_default_types_seeded` (normally triggered on first ingest,
    // see `core/ingest/pipeline/phase1.rs`) never fires. Seed explicitly so Lane B's
    // runtime type-id lookups (Organisation/Person) resolve.
    kremory::core::entity_types::ensure_default_types_seeded(&graph.conn, &gid)
        .await
        .expect("ensure_default_types_seeded for test group");

    // ── Lane A — type_discovery (LLM): 3 catch-all entities, no embeddings needed ──
    for name in ["aspirin", "ibuprofen", "paracetamol"] {
        plant_entity_row(&graph, name, &gid, 0, "Phase1Ner", 0.9).await;
    }

    // ── Lane B — reclassify (LLM), de-collided from Pass 0's type_id=0 scope ──
    // Plant with a WRONG non-zero type id + LOW confidence so these hit reclassify's
    // low_confidence arm (structurally disjoint from Pass 0's catch_all_cascade arm,
    // which only touches type_id=0).
    let wrong_org_id = entity_type_id_by_name(&graph, &gid, "Organisation").await;
    for name in ["Albert Einstein", "Marie Curie"] {
        plant_entity_row(&graph, name, &gid, wrong_org_id, "Phase1Ner", 0.2).await;
    }

    // ── Lane C — aliases (deterministic): identical unit-vector embeddings ──
    plant_entity_row(&graph, "acme corporation", &gid, 0, "Phase1Ner", 0.9).await;
    plant_entity_row(&graph, "acme corp", &gid, 0, "Phase1Ner", 0.9).await;
    let alias_vec = unit_vec(768);
    graph
        .set_entity_embedding("acme corporation", &alias_vec)
        .await
        .expect("set acme corporation embedding");
    graph
        .set_entity_embedding("acme corp", &alias_vec)
        .await
        .expect("set acme corp embedding");
    insert_potential_alias_fact(InsertPotentialAliasFactParams {
        graph: &graph,
        new_entity_id: "acme corp",
        existing_id: "acme corporation",
        similarity: 0.99,
        provenance: AliasProvenance {
            source_episode_id: None,
            group_id: Some(&gid),
        },
    })
    .await
    .expect("plant potential_alias fact");

    // ── Lane D — consistency_check (LLM): entity typed to a deliberately-absurd
    // custom entity_type; a real LLM verify pass should correct it back toward Person.
    let wrong_plumbing_id = insert_custom_entity_type(
        &graph,
        &gid,
        "Industrial Plumbing Fitting",
        "A metal coupling or valve used to join high-pressure water or gas pipes in \
         industrial plumbing systems.",
    )
    .await;
    let correct_person_id = entity_type_id_by_name(&graph, &gid, "Person").await;
    plant_entity_row(
        &graph,
        "Leonardo da Vinci",
        &gid,
        wrong_plumbing_id,
        "Phase1Ner",
        0.9,
    )
    .await;
    let davinci_embedding = emb
        .embed_dyn("Leonardo da Vinci painted the Mona Lisa and designed flying machines")
        .await
        .expect("embed Leonardo da Vinci text");
    graph
        .set_entity_embedding("Leonardo da Vinci", &davinci_embedding)
        .await
        .expect("set Leonardo da Vinci embedding");
    let now = Utc::now();
    for (predicate, object_value) in [
        ("painted", "the Mona Lisa"),
        ("designed", "flying machines"),
        ("born_in", "Vinci, Italy"),
    ] {
        graph
            .insert_fact_with_group(
                FactInsert::new("Leonardo da Vinci", predicate, now).object_value(object_value),
                Some(&gid),
            )
            .await
            .expect("plant Leonardo da Vinci fact");
    }

    // ── Lane E — canonicalize (deterministic): near-dup entities, Jaccard >= 0.5 ──
    plant_entity_row(&graph, "wolfgang amadeus mozart", &gid, 0, "Phase1Ner", 0.9).await;
    plant_entity_row(&graph, "wolfgang mozart", &gid, 0, "Phase1Ner", 0.9).await;
    let canon_vec = unit_vec(768);
    graph
        .set_entity_embedding("wolfgang amadeus mozart", &canon_vec)
        .await
        .expect("set wolfgang amadeus mozart embedding");
    graph
        .set_entity_embedding("wolfgang mozart", &canon_vec)
        .await
        .expect("set wolfgang mozart embedding");

    let before = entity_fields(&graph, &gid).await;
    eprintln!("[dream-e2e-5pass] before dream (id, type_id, source): {before:?}");

    // Run the FULL 5-pass chain via ONE real mem.dream() call.
    let summary = mem
        .dream()
        .await
        .expect("mem.dream() must succeed end-to-end with a real LLM");

    let after = entity_fields(&graph, &gid).await;
    eprintln!(
        "[dream-e2e-5pass] types_discovered={} aliases_resolved={} entities_reclassified={} \
         consistency_check_corrected={} canonicalization_merges={} warnings={:?}",
        summary.types_discovered.len(),
        summary.aliases_resolved,
        summary.entities_reclassified,
        summary.consistency_check_corrected,
        summary.canonicalization_merges,
        summary.warnings,
    );
    eprintln!("[dream-e2e-5pass] after dream (id, type_id, source): {after:?}");

    // (1) Whole §D3 chain executed — every pass counter fired via mem.dream().
    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    for expected in [
        "kremory.dream.passes_continued_past_reclassify_total", // Phase 1 restructure
        "kremory.dream.aliases_resolved_total",                 // aliases
        "kremory.dream.canonicalization_merges_total",          // canonicalize
        "kremory.dream.consistency_check.scanned_total",        // consistency_check (core)
        "kremory.dream.consistency_check_corrected_total",      // consistency_check (facade)
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected dream counter `{expected}` absent after real-LLM mem.dream(); \
             the pass did not run. Counters seen: {names:?}",
        );
    }

    // ── Lane A assertion — type_discovery proposed + accepted >= 1 new type ──
    assert!(
        summary.types_discovered.len() >= 1,
        "Lane A (type_discovery): summary.types_discovered must be >= 1 given 3 \
         catch-all entities (aspirin/ibuprofen/paracetamol); got {}",
        summary.types_discovered.len()
    );

    // ── Lane B assertion — reclassify retyped >= 1 low-confidence entity ──
    assert!(
        summary.entities_reclassified >= 1,
        "Lane B (reclassify): summary.entities_reclassified must be >= 1; before={before:?} \
         after={after:?}",
    );
    let einstein_after = after.iter().find(|(id, _, _)| id == "Albert Einstein");
    let curie_after = after.iter().find(|(id, _, _)| id == "Marie Curie");
    let einstein_retyped = einstein_after.is_some_and(|(_, t, _)| *t != wrong_org_id);
    let curie_retyped = curie_after.is_some_and(|(_, t, _)| *t != wrong_org_id);
    assert!(
        einstein_retyped || curie_retyped,
        "Lane B (reclassify): at least one of Einstein/Curie must have entity_type_id \
         changed off the wrong-seed Organisation id ({wrong_org_id}); \
         einstein={einstein_after:?} curie={curie_after:?}",
    );

    // ── Lane C assertion — aliases resolved the planted potential_alias fact ──
    assert!(
        summary.aliases_resolved >= 1,
        "Lane C (aliases): summary.aliases_resolved must be >= 1; got {}",
        summary.aliases_resolved
    );
    let pending = graph
        .get_alias_facts_in_group(&gid)
        .await
        .expect("get_alias_facts_in_group");
    assert!(
        pending.is_empty(),
        "Lane C (aliases): potential_alias fact for acme corp/acme corporation must be \
         invalidated by mem.dream(); {} still pending",
        pending.len(),
    );

    // ── Lane D assertion — consistency_check corrected the absurd plumbing type ──
    assert!(
        summary.consistency_check_corrected >= 1,
        "Lane D (consistency_check): summary.consistency_check_corrected must be >= 1; got {}",
        summary.consistency_check_corrected
    );
    let davinci_after = after
        .iter()
        .find(|(id, _, _)| id == "Leonardo da Vinci")
        .unwrap_or_else(|| panic!("Leonardo da Vinci must still exist post-dream; after={after:?}"));
    assert_eq!(
        davinci_after.1, correct_person_id,
        "Lane D (consistency_check): Leonardo da Vinci entity_type_id must be corrected \
         to Person ({correct_person_id}); got {} (wrong-seed plumbing id was {wrong_plumbing_id})",
        davinci_after.1,
    );
    assert_eq!(
        davinci_after.2, "DreamPass4",
        "Lane D (consistency_check): Leonardo da Vinci entity_type_source must be stamped \
         'DreamPass4' after correction; got '{}'",
        davinci_after.2,
    );

    // ── Lane E assertion — canonicalize merged the near-dup Mozart pair ──
    assert!(
        summary.canonicalization_merges >= 1,
        "Lane E (canonicalize): summary.canonicalization_merges must be >= 1; got {}",
        summary.canonicalization_merges
    );
    let ids_after: Vec<&String> = after.iter().map(|(id, _, _)| id).collect();
    assert!(
        ids_after.iter().any(|i| *i == "wolfgang amadeus mozart"),
        "Lane E (canonicalize): keeper 'wolfgang amadeus mozart' must survive; present: {ids_after:?}",
    );
    assert!(
        !ids_after.iter().any(|i| *i == "wolfgang mozart"),
        "Lane E (canonicalize): loser 'wolfgang mozart' must be merged away; present: {ids_after:?}",
    );

    // (Whole-chain) No pass may hard-fail — each non-fatal failure pushes a "…failed…" warning.
    let failures: Vec<&String> = summary
        .warnings
        .iter()
        .filter(|w| w.contains("failed"))
        .collect();
    assert!(
        failures.is_empty(),
        "no dream pass may fail in the real-LLM E2E; pass failures: {failures:?}",
    );
}

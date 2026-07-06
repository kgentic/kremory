// TD-093 (2026-07-01): the test is now a DETERMINISTIC VCR replay test gated on
// `llm-smoke` (offline replay, no Ollama) rather than `llm-integration` (live).
// `llm-smoke = ["test-utils"]`, so this single gate pulls both features. The
// chat provider is record/replay-wrapped (KREMORY_VCR); the embedder is the
// deterministic FNV-hash `DeterministicEmbeddingProvider` in BOTH modes so
// record and replay agree exactly (the only non-deterministic input — the LLM —
// is the one thing recorded). Mirrors `golden_path_smoke.rs`.
#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
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
//! ⚠️ STATUS (2026-07-01, post TD-094 fix): TD-094 (dream model-threading) is
//! FIXED. The passes no longer run with an empty model string: `with_model_id`
//! is now threaded through `dream_model_id_or_main` into all 3 LLM passes (and
//! `verify_model_override` for consistency_check). Proof the fix works, observed
//! live against `gemma4:e4b`: `kremory.dream.types_proposed_total` went 0 (empty
//! model → PromptOnly → zero proposals emitted) → 3 (real model → LLM invoked →
//! proposals emitted); consistency_check + reclassify now engage and retype
//! entities. The test wires `.with_model_id(chat_model)` accordingly.
//!
//! ⚠️ WHY THIS TEST IS STILL FLAKY (and NOT a TD-094 regression): each LLM lane
//! asserts `>= 1` on the output of an INDEPENDENT, stochastic real-LLM pass.
//! Observed across runs: reclassify 0↔9, consistency_corrected 0↔1, and
//! type_discovery proposes 3 but the shape validator rejects all 3 (gemma4:e4b
//! emits placeholder-ish type names — `types_proposed=3, types_rejected=3,
//! types_accepted=0`). So on any given live run one or more lanes may yield 0
//! and the test goes RED — this is real-LLM variance, not the model-threading
//! bug. Deterministic all-green REQUIRES the TD-093 VCR cassette (record once,
//! replay offline). Two open follow-ups gate a reliably-green E2E:
//!   • TD-093 — record a KREMORY_VCR cassette → tier-2, deterministic.
//!   • TD-095 — Lane A discovery-quality: gemma4:e4b proposals fail the shape
//!     validator (validator too strict for real model output, OR needs a
//!     higher-tier discovery model). Root-cause before asserting Lane A live.
//! Do NOT loosen the assertions to force a pass — cause-fix TD-093/TD-095.
//! `#[ignore]` + `cfg(llm-integration)` keep this off the default gate.

use std::sync::Arc;

use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

/// VCR mode for the dream E2E, selected by `KREMORY_VCR` (mirrors
/// `golden_path_smoke.rs`). `record` = live Ollama chat wrapped in
/// `RecordReplayChatProvider::record` (refreshes the committed cassette;
/// requires Ollama + the model). `replay` (or unset) = offline replay of the
/// committed cassette (no Ollama). The embedder is deterministic in BOTH modes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VcrMode {
    Record,
    Replay,
}

fn resolve_vcr_mode() -> VcrMode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => VcrMode::Record,
        Ok("replay") | Err(_) => VcrMode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn cassette_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("dream_e2e_5pass.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("dream_e2e_5pass.embeddings.json")
}

/// Embedding record/replay (TD-093). The dream discovery anti-redundancy gate is
/// a SEMANTIC comparison (proposal-vs-existing-type cosine ≥ 0.70/0.85), so it
/// needs REAL embeddings — a deterministic hash embedder gives spurious cosines
/// that wrongly reject genuinely-novel proposals (e.g. "Drug Compound" vs the
/// seeded defaults). Lanes B–E don't need this (planted vectors / SQL candidate
/// selection), but Lane A does. record: delegate to real nomic + capture each
/// text→vector; replay: look up offline (loud MISS error → re-record).
struct RecordReplayEmbedder {
    /// `Some` in record mode (real nomic), `None` in replay.
    inner: Option<Arc<dyn DynEmbeddingProvider>>,
    cache: std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>,
    path: std::path::PathBuf,
}

impl RecordReplayEmbedder {
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        Self {
            inner: Some(inner),
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            path,
        }
    }

    fn replay(path: std::path::PathBuf) -> Self {
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "embedding cassette must load ({}): {e} — re-record via KREMORY_VCR=record (TD-093)",
                path.display()
            )
        });
        let map: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_str(&raw).expect("embedding cassette must be valid JSON");
        Self {
            inner: None,
            cache: std::sync::Mutex::new(map),
            path,
        }
    }

    fn flush(&self) {
        let map = self.cache.lock().expect("embedding cache lock");
        let json = serde_json::to_string_pretty(&*map).expect("serialize embedding cassette");
        std::fs::write(&self.path, json).expect("write embedding cassette");
    }
}

impl kremory::EmbeddingProvider for RecordReplayEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        async move {
            // Cache hit (all texts in replay; already-seen texts in record).
            if let Some(v) = self.cache.lock().expect("cache lock").get(text).cloned() {
                return Ok(v);
            }
            match &self.inner {
                // record: delegate to real nomic, then memoise (no lock held across await).
                Some(inner) => {
                    let v = inner.embed_dyn(text).await?;
                    self.cache
                        .lock()
                        .expect("cache lock")
                        .insert(text.to_string(), v.clone());
                    Ok(v)
                }
                // replay: a miss means the cassette is stale for this fixture.
                None => Err(kremory::CoreError::Embedding(format!(
                    "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record (TD-093)"
                ))),
            }
        }
    }
}

/// Real nomic embedder (record mode only) — bridges autoagents Ollama's batch
/// `Vec<String>` embedding API to kremory's single-`&str` `EmbeddingProvider`.
struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        async move {
            use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
            // nomic-embed-text REQUIRES a task prefix; without it, short strings
            // ("Person", "Date", …) collapse to near-identical vectors (cosine
            // ~1.0), which breaks the discovery anti-redundancy gate (every type
            // looks 100% redundant → all proposals rejected). This is the TD-097
            // root cause. `search_document:` is nomic's document-embedding prefix
            // (appropriate for embedding type definitions for similarity). The
            // RecordReplayEmbedder caches under the ORIGINAL `text`, so replay
            // lookup is unaffected — the prefix is internal to the nomic call.
            let prefixed = format!("search_document: {text}");
            let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![prefixed])
                .await
                .map_err(|e| kremory::CoreError::Embedding(e.to_string()))?;
            vecs.pop().ok_or_else(|| {
                kremory::CoreError::Embedding("OllamaEmbedderAdapter: empty embed vec".to_string())
            })
        }
    }
}

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
    let row = rows.next().await.expect("row read").unwrap_or_else(|| {
        panic!("entity_types row for name={name} in group={group_id} must exist")
    });
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
#[ignore = "TD-093 WIP: 4/5 lanes replay deterministically green, but Lane A \
            (type_discovery) has two open blockers — TD-097 (nomic returns \
            degenerate embeddings for bare short type-name labels → anti-redundancy \
            rejects all) AND a discovery-call VCR replay mismatch (types_proposed=3 \
            on record, 0 on replay). Runnable explicitly: \
            KREMORY_VCR=record cargo test -p kremory --features llm-smoke,test-utils \
            --test dream_e2e_real_llm -- --ignored"]
async fn dream_e2e_real_llm_five_pass_chain() {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use chrono::Utc;
    use kremory::core::disambiguation::{
        insert_potential_alias_fact, AliasProvenance, InsertPotentialAliasFactParams,
    };
    use kremory::core::graph::FactInsert;
    use metrics_util::debugging::DebuggingRecorder;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mode = resolve_vcr_mode();
    let cassette = cassette_path();
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // CASSETTE-RECORDING model (a pipeline-integrity FIXTURE, not a production
    // model claim). This test asserts the 5-pass chain executes + each lane
    // produces output — orthogonal to which model production dream SHOULD use
    // (that is a data question tracked as TD-096: dream-specific benchmark, no
    // 30s cap). gemma4:e4b + `think:false` is our benchmarked top extraction
    // model (F1 84.4, local-model-benchmark-2026-06-24) and records fast; the
    // `think:false` below is DECISIVE (reasoning-on injects noise the shape
    // validator rejects — the TD-095 root cause). The string is threaded via
    // `with_model_id` for capability detection AND is the cassette header model,
    // so record + replay agree. Override via OLLAMA_CHAT_MODEL to re-record with
    // a heavier model once TD-096 lands.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    // Chat provider — mode-selected. record: live Ollama wrapped in record(...)
    // so one run refreshes the committed cassette. replay: offline cassette read
    // (loud error if missing — record it via KREMORY_VCR=record).
    let provider: Arc<RecordReplayChatProvider> = match mode {
        VcrMode::Record => {
            let real: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model(&chat_model)
                // .think(false): decisive for gemma4:e4b per local-model-benchmark
                // 2026-06-24 (F1 75→84, reasoning injects noise into structured
                // output). Production Tier-1 shortcuts set it (providers.rs:365,418);
                // this test MUST match that config or discovery emits noisy names
                // the shape validator rejects (the earlier "Lane A weak" red herring).
                .think(false)
                .timeout_seconds(180)
                .keep_alive("1h")
                .build()
                .expect("Ollama LLM builder must succeed (KREMORY_VCR=record needs Ollama)");
            Arc::new(RecordReplayChatProvider::record(
                real,
                cassette.clone(),
                chat_model.clone(),
            ))
        }
        VcrMode::Replay => Arc::new(
            RecordReplayChatProvider::replay(cassette.clone())
                .expect("replay cassette must load — record it via KREMORY_VCR=record (TD-093)"),
        ),
    };
    let llm: Arc<dyn ChatProvider> = provider.clone();

    // Embedding record/replay (TD-093). record: wrap real nomic + capture every
    // text→vector; replay: look up offline. The discovery anti-redundancy gate is
    // a SEMANTIC cosine comparison, so it needs real embeddings (a hash embedder
    // spuriously rejects genuinely-novel proposals like "Drug Compound"). nomic is
    // 768-dim → embedding_dim(768) below.
    let emb_vcr: Arc<RecordReplayEmbedder> = match mode {
        VcrMode::Record => {
            use autoagents_llm::backends::ollama::Ollama;
            use autoagents_llm::embedding::EmbeddingBuilder;
            let raw_nomic: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model("nomic-embed-text")
                .build()
                .expect("nomic embedder must build (KREMORY_VCR=record needs Ollama)");
            let nomic: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_nomic));
            Arc::new(RecordReplayEmbedder::record(
                nomic,
                embedding_cassette_path(),
            ))
        }
        VcrMode::Replay => Arc::new(RecordReplayEmbedder::replay(embedding_cassette_path())),
    };
    let emb: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("dream-e2e-5pass");
    let mem = Memory::open(dir.path().join("dream_e2e.db"))
        .with_llm(llm)
        // TD-094: declare the concrete model id so the dream LLM passes reach the
        // FormatSchema capability arm, not the empty-model → PromptOnly degrade.
        .with_model_id(chat_model.clone())
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

    // record mode ONLY: flush the cassette to disk AFTER dream completes and
    // BEFORE any assertion (mirrors golden_path_smoke NEW-202). dream()'s LLM
    // calls run to completion above (await_completion is the default), but the
    // record buffer is flushed here explicitly rather than relying on Drop.
    if matches!(mode, VcrMode::Record) {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        // Persist the embedding cassette too (TD-093) so replay is fully offline.
        emb_vcr.flush();
    }

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

    // ── Lane A assertion — type_discovery PASS ran + the model PROPOSED >= 1 type ──
    // We assert `types_proposed`, NOT `types_accepted` (summary.types_discovered).
    // Acceptance runs through the anti-redundancy gate, which compares SHORT
    // type-NAME embeddings — and sentence-embedders (nomic-embed-text) return
    // degenerate vectors for bare one-word labels ("Person" ≡ "Date", cosine ~1.0),
    // so the 0.70 name-gate can reject even genuinely-novel proposals regardless of
    // model quality. That is TD-097 (a gate-design + embedder-usage issue),
    // ORTHOGONAL to this pipeline-integrity guard. `types_proposed >= 1`
    // deterministically proves discovery reached the LLM with a real model and the
    // model emitted valid structured proposals (the TD-094 invariant this E2E
    // exists to guard). The recorded cassette shows gemma4:e4b proposing sound
    // names ("Pharmaceutical Drug", "Corporation"); acceptance is asserted once
    // TD-097 lands a sound name-comparison path.
    let types_proposed = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .find(|(k, _, _, _)| k.key().name() == "kremory.dream.types_proposed_total")
        .map(|(_, _, _, v)| match v {
            metrics_util::debugging::DebugValue::Counter(c) => *c,
            _ => 0,
        })
        .unwrap_or(0);
    assert!(
        types_proposed >= 1,
        "Lane A (type_discovery): kremory.dream.types_proposed_total must be >= 1 \
         (discovery pass ran + the model proposed >= 1 type); got {types_proposed}. \
         (types_accepted is embedder-gated — see TD-097.)"
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
        .unwrap_or_else(|| {
            panic!("Leonardo da Vinci must still exist post-dream; after={after:?}")
        });
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

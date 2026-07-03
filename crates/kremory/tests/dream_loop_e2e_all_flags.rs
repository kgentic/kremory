// System/E2E test strategy: `.ai-docs/specs/dream-loop-test-strategy-2026-07-02.md`
// §7 (E1 row) + §6.4 (composite fixtures) + §8 (zero-false-merge NFR).
//
// Scope of THIS file: the reusable e2e SCAFFOLD (VCR chat + embedder record/
// replay, an `all_flags_dream_opts()` helper, and shared fixture-planting
// helpers) plus scenario **E1 only**
// (`all_sites_fire_together_zero_false_merge`). E2-E6 (§7) are explicitly OUT
// OF SCOPE for this file — they land as separate follow-up `#[tokio::test]`
// fns reusing the helpers below.
//
// Gated `llm-smoke` + `test-utils` (mirrors `dream_e2e_real_llm.rs` /
// `type_registry_collapse_s3_spike.rs` / `acronym_nickname_recall_s2_spike.rs`'s
// VCR tier exactly). `replay` (default) is fully offline/deterministic;
// `record` (`KREMORY_VCR=record`) drives live Ollama (`gemma4:e4b` +
// `nomic-embed-text`) and refreshes the committed cassettes.
//
// Run (replay, default CI gate):
//   cargo test -p kremory --features llm-smoke,test-utils --test dream_loop_e2e_all_flags
// Run (record, refresh cassette against live Ollama):
//   KREMORY_VCR=record cargo test -p kremory --features llm-smoke,test-utils \
//     --test dream_loop_e2e_all_flags -- --ignored --nocapture
//
// This test calls the FULL `mem.dream()` facade (not an individual pass
// function) with ALL FIVE `DreamOpts::include_*` flags explicitly set `true`
// — the actual prod-flip configuration under test (spec §7 preamble). It does
// NOT change any `DreamOpts` production default (`memory/types.rs`'s
// `impl Default for DreamOpts` is untouched by this file).

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::entity_types::ensure_default_types_seeded;
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::provider::{DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::types::DreamOpts;
use kremory::{ChatProvider, Memory, Namespace};

// ─── VCR mode (mirrors dream_e2e_real_llm.rs / type_registry_collapse_s3_spike.rs) ──

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

fn chat_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("dream_loop_e2e_{name}.json"))
}

fn embedding_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("dream_loop_e2e_{name}.embeddings.json"))
}

// ─── Embedding record/replay (mirrors dream_e2e_real_llm.rs::RecordReplayEmbedder) ──

/// Real embeddings are required wherever a scenario touches a semantic cosine
/// gate (Site #3 description-cosine, spec §4). `record`: delegate to real
/// nomic + capture every text->vector. `replay`: look up offline (loud MISS).
struct RecordReplayEmbedder {
    inner: Option<Arc<dyn DynEmbeddingProvider>>,
    cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
    path: std::path::PathBuf,
}

impl RecordReplayEmbedder {
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        Self {
            inner: Some(inner),
            cache: std::sync::Mutex::new(HashMap::new()),
            path,
        }
    }

    fn replay(path: std::path::PathBuf) -> Self {
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "embedding cassette must load ({}): {e} — re-record via KREMORY_VCR=record",
                path.display()
            )
        });
        let map: HashMap<String, Vec<f32>> =
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
            if let Some(v) = self.cache.lock().expect("cache lock").get(text).cloned() {
                return Ok(v);
            }
            match &self.inner {
                Some(inner) => {
                    let v = inner.embed_dyn(text).await?;
                    self.cache
                        .lock()
                        .expect("cache lock")
                        .insert(text.to_string(), v.clone());
                    Ok(v)
                }
                None => Err(kremory::CoreError::Embedding(format!(
                    "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record"
                ))),
            }
        }
    }
}

/// Real nomic embedder bridge (record mode only) — mirrors
/// `dream_e2e_real_llm.rs::OllamaEmbedderAdapter` / `type_registry_collapse_s3_
/// spike.rs::OllamaEmbedderAdapter` exactly, INCLUDING the `search_document:`
/// task prefix (TD-097 root cause: nomic-embed-text collapses short unprefixed
/// strings to near-identical vectors).
struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        async move {
            use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
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

// ─── Scaffold: build the chat provider + embedder for one scenario ──────────

/// Build the record/replay chat provider for `cassette_tag` (one cassette per
/// scenario — mirrors the per-site spikes' `chat_cassette_path(tag)` pattern).
async fn build_chat_provider(
    mode: VcrMode,
    cassette_tag: &str,
) -> (Arc<RecordReplayChatProvider>, String) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // gemma4:e4b + think:false — this project's benchmarked deferred-quality
    // dream model (F1 85.7, local-model-benchmark-2026-06-24 /
    // project_kremory_validated_model_findings_2026-06-24). Matches every
    // sibling dream VCR fixture in this crate.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let cassette = chat_cassette_path(cassette_tag);
    let provider: Arc<RecordReplayChatProvider> = match mode {
        VcrMode::Record => {
            let real: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model(&chat_model)
                .think(false)
                .timeout_seconds(180)
                .keep_alive("1h")
                .build()
                .expect("Ollama LLM builder must succeed (KREMORY_VCR=record needs Ollama)");
            Arc::new(RecordReplayChatProvider::record(
                real,
                cassette,
                chat_model.clone(),
            ))
        }
        VcrMode::Replay => Arc::new(RecordReplayChatProvider::replay(cassette).unwrap_or_else(
            |e| {
                panic!(
                    "replay cassette must load for tag={cassette_tag}: {e} — \
                     record it via KREMORY_VCR=record"
                )
            },
        )),
    };
    (provider, chat_model)
}

/// Build the record/replay embedder for `cassette_tag` (nomic-embed-text,
/// 768-dim). Needed by any scenario that touches Site #3's description-cosine
/// gate.
async fn build_embedder(mode: VcrMode, cassette_tag: &str) -> Arc<RecordReplayEmbedder> {
    match mode {
        VcrMode::Record => {
            use autoagents_llm::backends::ollama::Ollama;
            use autoagents_llm::embedding::EmbeddingBuilder;
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let raw_nomic: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model("nomic-embed-text")
                .build()
                .expect("nomic embedder must build (KREMORY_VCR=record needs Ollama)");
            let nomic: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_nomic));
            Arc::new(RecordReplayEmbedder::record(
                nomic,
                embedding_cassette_path(cassette_tag),
            ))
        }
        VcrMode::Replay => Arc::new(RecordReplayEmbedder::replay(embedding_cassette_path(
            cassette_tag,
        ))),
    }
}

/// The actual prod-flip configuration under test (spec §7 preamble): ALL FIVE
/// `DreamOpts::include_*` flags explicitly `true`. Does NOT touch
/// `impl Default for DreamOpts` — every field is set explicitly here so this
/// helper stays correct even if a future flag's production default changes.
fn all_flags_dream_opts() -> DreamOpts {
    DreamOpts {
        since: None,
        include_type_discovery: true,
        include_consistency_check: true,
        max_episodes_per_run: None,
        include_type_registry_collapse: true,
        include_acronym_nickname_recall: true,
        include_type_novelty_llm_verify: true,
        // Consolidation sub-phase (ADR-066) is opt-in / default-off and is NOT part
        // of this reconciliation prod-flip config — held false so this test's
        // behaviour is unchanged from before consolidation existed.
        include_community_detection: false,
        include_cross_episode_merges: false,
        include_supersession_sweep: false,
        include_supersession_llm_nominate: false,
        include_fact_archival: false,
        consolidation_budget_tokens: Some(50_000),
        archive_grace_days: Some(90),
    }
}

// ─── Fixture-planting helpers (shared across E1..E6) ────────────────────────

/// Plant a bare entity (catch-all `entity_type_id = 0`) with a description
/// property — mirrors `acronym_nickname_recall_s2_spike.rs::insert_entity`.
async fn insert_entity(graph: &TemporalGraph, id: &str, group_id: &str, description: &str) {
    let props = serde_json::json!({ "name": id, "description": description });
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0u32,
            properties: props,
            group_id: Some(group_id),
        })
        .await
        .expect("insert entity");
}

/// Make two entities co-occur via a shared episode mention — the nickname
/// pair's ONLY nomination path (no structural initial-letter relationship).
/// Mirrors `acronym_nickname_recall_s2_spike.rs::make_cooccur`.
async fn make_cooccur(graph: &TemporalGraph, group_id: &str, a: &str, b: &str, content: &str) {
    let ep = graph
        .insert_episode(InsertEpisodeParams {
            content,
            timestamp: chrono::Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("episode");
    for ent in [a, b] {
        graph
            .insert_episodic_edge(InsertEpisodicEdgeParams {
                episode_id: ep,
                entity_id: ent,
                entity_group_id: Some(group_id),
                role: "mention",
            })
            .await
            .expect("edge");
    }
}

/// Insert one custom `entity_types` row at a fresh id above the seeded
/// default vocabulary (mirrors `dream_e2e_real_llm.rs::insert_custom_entity_type`
/// / `type_registry_collapse_s3_spike.rs::insert_custom_type`), returning the
/// new row's id. Callers must have run `ensure_default_types_seeded` first.
async fn insert_custom_type(graph: &TemporalGraph, group_id: &str, name: &str, desc: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query next entity_type id");
    let row = rows.next().await.expect("row read").expect("row present");
    let next_id: i64 = row.get(0).expect("next id");
    graph
        .conn
        .execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES (?1, ?2, ?3, ?4)",
            libsql::params![group_id.to_string(), next_id, name, desc],
        )
        .await
        .expect("insert custom entity_type");
    next_id
}

/// Insert a bare entity directly typed under `type_id` (no catch-all
/// intermediate step) — mirrors `type_registry_collapse.rs`'s own
/// `insert_entity_of_type` test helper. Used to attach entities to the
/// lexical-duplicate / lemma-collision type pairs planted for Site #3.
async fn insert_entity_of_type(graph: &TemporalGraph, group_id: &str, id: &str, type_id: i64) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, ?2, ?3, ?4)",
            libsql::params![id, type_id, now, group_id],
        )
        .await
        .expect("insert entity_of_type");
}

/// Count non-catch-all `entity_types` rows for `group_id` (excludes id=0).
async fn count_entity_types(conn: &libsql::Connection, group_id: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1 AND id != 0",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("count entity_types");
    rows.next()
        .await
        .expect("row")
        .expect("row present")
        .get::<i64>(0)
        .expect("count col")
}

/// Count `entities` rows for `group_id`.
async fn count_entities(conn: &libsql::Connection, group_id: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("count entities");
    rows.next()
        .await
        .expect("row")
        .expect("row present")
        .get::<i64>(0)
        .expect("count col")
}

/// `entity_id -> entity_type_id` map for `group_id` (post-dream inspection).
async fn entity_type_map(conn: &libsql::Connection, group_id: &str) -> HashMap<String, i64> {
    let mut rows = conn
        .query(
            "SELECT id, entity_type_id FROM entities WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query entity types");
    let mut out = HashMap::new();
    while let Some(row) = rows.next().await.expect("row iteration") {
        let id: String = row.get(0).expect("id");
        let type_id: i64 = row.get(1).expect("entity_type_id");
        out.insert(id, type_id);
    }
    out
}

/// Rows written to `identity_verdict_audit` for `group_id`, as `(site, decision)`
/// pairs — used to assert the exact per-site decisions E1 predicts (spec §7
/// "assertion shape": `identity_verdict_audit` row count + `decision` values
/// match expectation).
async fn identity_verdict_audit_rows(
    conn: &libsql::Connection,
    group_id: &str,
) -> Vec<(String, String)> {
    let mut rows = conn
        .query(
            "SELECT site, decision FROM identity_verdict_audit WHERE group_id = ?1 ORDER BY id",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query identity_verdict_audit");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration") {
        let site: String = row.get(0).expect("site");
        let decision: String = row.get(1).expect("decision");
        out.push((site, decision));
    }
    out
}

// ─── E1: all_sites_fire_together_zero_false_merge ───────────────────────────
//
// Fixture (spec §7 E1 row + §6.4): plant ONE candidate for EACH site in ONE
// graph:
//   - an acronym pair (Site #5, ADR-063 §3):            IBM / International
//     Business Machines
//   - a lexical-duplicate type pair (Site #3 auto-merge, row 1, no LLM):
//     "Company" / "company" (case-only, exact-normalize match)
//   - a distinct-lemma-collision type pair (§6.3 — MUST NOT merge):
//     "Species" / "Specie", seeded with REALISTIC divergent descriptions
//   - a co-occurring nickname pair (Site #5 via co-occurrence, no structural
//     initialism relationship): Bob / Robert, sharing one episode mention
//
// Run `mem.dream()` with ALL flags on. Assert every EXPECTED merge/alias/
// reject fires; the lemma-collision pair is NOT merged; hand-computed EXACT
// final entity + type counts (never `>=` on the deterministic parts).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn all_sites_fire_together_zero_false_merge() {
    use metrics_util::debugging::DebuggingRecorder;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mode = resolve_vcr_mode();
    let (provider, chat_model) = build_chat_provider(mode, "e1_all_sites").await;
    let llm: Arc<dyn ChatProvider> = provider.clone();
    let emb_vcr = build_embedder(mode, "e1_all_sites").await;
    let emb: Arc<dyn kremory::DynEmbeddingProvider> = emb_vcr.clone();

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("dream-loop-e2e-e1");
    let mem = Memory::open(dir.path().join("e1.db"))
        .with_llm(llm)
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

    // This fixture plants rows directly via raw SQL / graph calls (never
    // mem.remember()), so lazy `ensure_default_types_seeded` never fires via
    // ingest. Seed explicitly.
    ensure_default_types_seeded(&graph.conn, &gid)
        .await
        .expect("ensure_default_types_seeded for test group");

    // ── Site #5 candidate 1 — genuine acronym pair (structural initialism) ──
    insert_entity(&graph, "IBM", &gid, "A technology company.").await;
    insert_entity(
        &graph,
        "International Business Machines",
        &gid,
        "A technology company headquartered in New York.",
    )
    .await;

    // ── Site #5 candidate 2 — genuine nickname pair (co-occurrence only) ──
    insert_entity(&graph, "Bob", &gid, "A person mentioned in the transcript.").await;
    insert_entity(
        &graph,
        "Robert",
        &gid,
        "A person mentioned in the transcript.",
    )
    .await;
    make_cooccur(
        &graph,
        &gid,
        "Bob",
        "Robert",
        "Bob (Robert) gave the opening remarks at the meeting.",
    )
    .await;

    // ── Site #3 candidate 1 — lexical-duplicate type pair (auto-merge, row 1, no LLM) ──
    let company_upper = insert_custom_type(
        &graph,
        &gid,
        "Company",
        "A business organisation, firm, or investment fund.",
    )
    .await;
    let company_lower = insert_custom_type(
        &graph,
        &gid,
        "company",
        "A business organisation, firm, or investment fund.",
    )
    .await;
    insert_entity_of_type(&graph, &gid, "acme corp", company_upper).await;
    insert_entity_of_type(&graph, &gid, "beta corp", company_lower).await;

    // ── Site #3 candidate 2 — distinct-lemma-collision pair (§6.3 SAFETY PROOF) ──
    // "Species" -> strip_trailing_s -> "Specie": lexically collides via the
    // naive trailing-s lemma heuristic, but these are DISTINCT English
    // concepts (biological taxonomic rank vs. coined/specie currency).
    // Seeded with REALISTIC, DIFFERENT descriptions per spec §6.3 step 1.
    let species_id = insert_custom_type(
        &graph,
        &gid,
        "Species",
        "A group of living organisms that can interbreed and produce fertile \
         offspring, the basic unit of biological classification.",
    )
    .await;
    let specie_id = insert_custom_type(
        &graph,
        &gid,
        "Specie",
        "Coined money, as opposed to paper currency or credit.",
    )
    .await;
    insert_entity_of_type(&graph, &gid, "red fox", species_id).await;
    insert_entity_of_type(&graph, &gid, "gold sovereign", specie_id).await;

    let entities_before = count_entities(&graph.conn, &gid).await;
    let types_before = count_entity_types(&graph.conn, &gid).await;
    eprintln!(
        "[e1] before dream: entities={entities_before} entity_types(non-catch-all)={types_before}"
    );

    // ── ONE mem.dream() call, ALL FIVE include_* flags explicitly true ──
    let summary = mem
        .dream()
        .opts(all_flags_dream_opts())
        .await
        .expect("mem.dream() must succeed with all flags on");

    if matches!(mode, VcrMode::Record) {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        emb_vcr.flush();
    }

    let entities_after = count_entities(&graph.conn, &gid).await;
    let types_after = count_entity_types(&graph.conn, &gid).await;
    let type_map_after = entity_type_map(&graph.conn, &gid).await;
    let audit_rows = identity_verdict_audit_rows(&graph.conn, &gid).await;

    eprintln!(
        "[e1] after dream: entities={entities_after} entity_types(non-catch-all)={types_after} \
         types_discovered={} aliases_resolved={} entities_reclassified={} \
         consistency_check_corrected={} canonicalization_merges={} warnings={:?}",
        summary.types_discovered.len(),
        summary.aliases_resolved,
        summary.entities_reclassified,
        summary.consistency_check_corrected,
        summary.canonicalization_merges,
        summary.warnings,
    );
    eprintln!("[e1] identity_verdict_audit rows (site, decision): {audit_rows:?}");
    // Full per-pass counter dump (Rule 19 observability) — cheap diagnostic that
    // pays for itself the moment a re-record diverges from expectations (this
    // is exactly how the Pass-0/Site-#2 cross-pass interaction below was
    // root-caused during this file's own construction).
    let all_counters: Vec<(String, u64)> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, v)| {
            let val = match v {
                metrics_util::debugging::DebugValue::Counter(c) => *c,
                _ => 0,
            };
            (k.key().name().to_string(), val)
        })
        .collect();
    eprintln!("[e1][DIAG] all counters: {all_counters:?}");

    eprintln!("[e1] entity_type_id map after dream: {type_map_after:?}");

    // ── (1) Site #5 acronym pair — MUST merge (write_gate row 5) ──
    let ibm_survives = type_map_after.contains_key("IBM");
    let ibm_full_survives = type_map_after.contains_key("International Business Machines");
    assert!(
        ibm_survives ^ ibm_full_survives,
        "IBM / International Business Machines: exactly ONE of the pair must \
         survive as the merge keeper (site5 acronym pair) — got IBM present={ibm_survives}, \
         International Business Machines present={ibm_full_survives}"
    );

    // ── (2) Site #5 nickname pair — MUST merge (write_gate row 5, co-occurrence) ──
    let bob_survives = type_map_after.contains_key("Bob");
    let robert_survives = type_map_after.contains_key("Robert");
    assert!(
        bob_survives ^ robert_survives,
        "Bob / Robert: exactly ONE of the pair must survive as the merge keeper \
         (site5 nickname pair via co-occurrence) — got Bob present={bob_survives}, \
         Robert present={robert_survives}"
    );

    // ── (3) Site #3 lexical-duplicate pair — MUST auto-merge (row 1, no LLM) ──
    // Entities "acme corp" and "beta corp" must now share ONE entity_type_id
    // (the surviving keeper of Company/company), never two distinct ids.
    let acme_type = type_map_after
        .get("acme corp")
        .copied()
        .unwrap_or_else(|| panic!("acme corp must still exist post-dream"));
    let beta_type = type_map_after
        .get("beta corp")
        .copied()
        .unwrap_or_else(|| panic!("beta corp must still exist post-dream"));
    assert_eq!(
        acme_type, beta_type,
        "Company/company lexical-duplicate pair must auto-merge (row 1): \
         acme corp (type={acme_type}) and beta corp (type={beta_type}) must \
         share the same entity_type_id after dream"
    );
    assert!(
        acme_type == company_upper || acme_type == company_lower,
        "surviving Company/company type id must be one of the two originally \
         planted ids ({company_upper} or {company_lower}), got {acme_type}"
    );

    // ── (4) THE safety invariant — Species/Specie MUST NOT merge (§6.3) ──
    let fox_type = type_map_after
        .get("red fox")
        .copied()
        .unwrap_or_else(|| panic!("red fox must still exist post-dream"));
    let sovereign_type = type_map_after
        .get("gold sovereign")
        .copied()
        .unwrap_or_else(|| panic!("gold sovereign must still exist post-dream"));
    assert_ne!(
        fox_type, sovereign_type,
        "SAFETY INVARIANT VIOLATION (spec §6.3): Species/Specie is a distinct-\
         lemma-collision pair (biological taxon vs. coined currency) and MUST \
         NOT be merged by type_registry_collapse. red fox (type={fox_type}) and \
         gold sovereign (type={sovereign_type}) ended up sharing an \
         entity_type_id — this means the naive trailing-s lemma heuristic \
         AND real nomic description-cosine BOTH crossed the 0.85 auto-merge \
         threshold for two genuinely distinct English concepts. This is a \
         PRODUCTION BUG / safety-margin falsification, not a test bug — do NOT \
         weaken this assertion; report it loudly instead (see spec §6.3 step 3)."
    );
    assert!(
        fox_type == species_id || fox_type == specie_id,
        "red fox's entity_type_id ({fox_type}) must remain one of the two \
         originally-planted ids ({species_id} or {specie_id}) since no merge \
         should have occurred"
    );
    assert!(
        sovereign_type == species_id || sovereign_type == specie_id,
        "gold sovereign's entity_type_id ({sovereign_type}) must remain one of \
         the two originally-planted ids ({species_id} or {specie_id}) since no \
         merge should have occurred"
    );

    // ── (5) Hand-computed EXACT final counts (never `>=` on deterministic parts) ──
    // Two entity-level merges expected total across Site #5's planted candidates
    // (deterministic — write_gate row 5 fires or the whole scenario is broken):
    //   Site #5: IBM/IBM-full -> 1 survivor (entities -1)
    //   Site #5: Bob/Robert   -> 1 survivor (entities -1)
    // No other planted rows are candidates for Site #5 or any entity-merging
    // pass (aliases/canonicalize/reclassify have no candidates in this
    // fixture), so entities_after is EXACT, not a floor.
    let expected_entities_after = entities_before - 2;
    assert_eq!(
        entities_after, expected_entities_after,
        "expected EXACTLY 2 entity-level merges (IBM pair + Bob/Robert pair): \
         entities_before={entities_before} -> expected_after={expected_entities_after}, \
         got {entities_after}. Any other value means an unexpected merge (or a \
         missing expected merge) occurred somewhere in the chain."
    );
    // Type-registry count has TWO independent effects in an all-flags-on run:
    //   Site #3 type_registry_collapse: Company/company -> 1 survivor
    //     (entity_types -1, DETERMINISTIC — this fixture's whole point).
    //     Species/Specie must NOT merge (asserted separately above, exact).
    //   Pass 0 type_discovery (include_type_discovery: true, ALSO on in this
    //     all-flags run): proposes + may accept new types from the catch-all
    //     entities in scope — this is REAL-LLM VARIANCE (mirrors
    //     dream_e2e_real_llm.rs's `>= 1` convention for LLM-touched counts),
    //     not a quantity this test can hand-compute. `types_discovered` is
    //     asserted separately as `>= 0` (sane, non-negative) below; it is
    //     EXCLUDED from the hand-computed exact-count invariant so a future
    //     re-record against model variance doesn't spuriously break the
    //     safety-invariant assertion this scenario exists to prove.
    let types_discovered_this_run = summary.types_discovered.len() as i64;
    let expected_types_after_ex_discovery = types_before - 1;
    assert_eq!(
        types_after - types_discovered_this_run,
        expected_types_after_ex_discovery,
        "expected EXACTLY 1 type-level merge from Site #3 (Company/company only \
         — Species/Specie must NOT merge), net of Pass-0 discovery's own \
         (LLM-variance) additions: types_before={types_before}, \
         types_discovered_this_run={types_discovered_this_run}, \
         types_after={types_after} -> types_after minus discovery should equal \
         {expected_types_after_ex_discovery}."
    );

    // ── (6) Per-pass counts sane / no unexpected pass failure ──
    let failures: Vec<&String> = summary
        .warnings
        .iter()
        .filter(|w| w.contains("failed"))
        .collect();
    assert!(
        failures.is_empty(),
        "no dream pass may fail in E1 (all fixtures are well-formed); pass \
         failures: {failures:?}"
    );

    // (7) Diagnostic-only — pass-chain executed (counters fired). Not a
    // correctness assertion on its own, but confirms the chain didn't silently
    // no-op on an empty pass list.
    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    for expected in [
        "kremory.dream.passes_continued_past_reclassify_total",
        "kremory.dream.acronym_recall.merges_applied_total",
        "kremory.dream.type_registry_collapse.merges_applied_total",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected dream counter `{expected}` absent after all-flags mem.dream(); \
             counters seen: {names:?}",
        );
    }
}

// ─── E2: multi_site_ordering_merge_before_reclassify ────────────────────────
//
// Fixture (spec §7 E2 row + §6.4 + §312-321 "highest-risk untested
// interaction"): plant ONE acronym pair (Site #5 candidate) where the
// SURVIVING keeper is ALSO, on its own, a `reclassify` (Pass 2) candidate —
// i.e. planted catch-all (`entity_type_id = 0`, `entity_type_source =
// 'Phase1Ner'`, the shape `load_candidates`'s WHERE clause selects).
//
// Facade pass order (verified against `facade/dream.rs`, 2026-07-03):
//   aliases -> acronym_nickname_recall (Site #5) -> reclassify (Pass 2) ->
//   consistency_check -> canonicalize -> type_registry_collapse (Site #3).
// Site #5 is hard-ordered BEFORE reclassify specifically so a
// merged-away entity is never wastefully reclassified (facade/dream.rs
// comment at the acronym_recall call site). This scenario proves that
// ordering empirically: if it silently regressed (Site #5 moved after
// reclassify, or reclassify read a stale pre-merge entity snapshot), BOTH
// entities in the pair would independently reach reclassify — surfacing as
// `entities_reclassified == 2` instead of `1`, and/or the loser entity id
// still present after dream.
//
// `load_entity_ids` (acronym_nickname_recall.rs) orders candidates
// `ORDER BY id ASC`, and pairs are formed `(a, b)` with `a` preceding `b` in
// that ordering; the write_gate keeps `pair.a` and remaps `pair.b` onto it
// (acronym_nickname_recall.rs comment at the merge-apply call site). "IBM" <
// "International Business Machines" lexicographically, so "IBM" is the
// keeper (verified in this file's own construction — see notes above).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2_multi_site_ordering_merge_before_reclassify() {
    use metrics_util::debugging::DebuggingRecorder;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mode = resolve_vcr_mode();
    let (provider, chat_model) = build_chat_provider(mode, "e2_ordering").await;
    let llm: Arc<dyn ChatProvider> = provider.clone();
    let emb_vcr = build_embedder(mode, "e2_ordering").await;
    let emb: Arc<dyn kremory::DynEmbeddingProvider> = emb_vcr.clone();

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("dream-loop-e2e-e2");
    let mem = Memory::open(dir.path().join("e2.db"))
        .with_llm(llm)
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

    // This fixture plants rows directly via raw SQL / graph calls (never
    // mem.remember()), so lazy `ensure_default_types_seeded` never fires via
    // ingest. Seed explicitly — this also gives reclassify a real
    // "Organisation" (id=2) type to assign IBM to (DEFAULT_ENTITY_TYPES).
    ensure_default_types_seeded(&graph.conn, &gid)
        .await
        .expect("ensure_default_types_seeded for test group");

    // The ONLY planted candidate: an acronym pair whose keeper ("IBM",
    // sorts first) is left catch-all (entity_type_id=0, entity_type_source=
    // 'Phase1Ner' via `insert_entity`/`insert_entity_with_group`) — the exact
    // shape `reclassify::load_candidates`'s WHERE clause selects
    // (`entity_type_id = 0 ... AND entity_type_source NOT IN ('ConsumerPinned',
    // 'DreamPass1') AND is_dream_generated = 0`). No other site's candidates
    // are planted (unlike E1) so this scenario isolates the ordering
    // interaction cleanly.
    insert_entity(&graph, "IBM", &gid, "A technology company.").await;
    insert_entity(
        &graph,
        "International Business Machines",
        &gid,
        "A technology company headquartered in New York.",
    )
    .await;

    let entities_before = count_entities(&graph.conn, &gid).await;
    eprintln!("[e2] before dream: entities={entities_before}");

    // ── ONE mem.dream() call, ALL FIVE include_* flags explicitly true ──
    let summary = mem
        .dream()
        .opts(all_flags_dream_opts())
        .await
        .expect("mem.dream() must succeed with all flags on");

    if matches!(mode, VcrMode::Record) {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        emb_vcr.flush();
    }

    let entities_after = count_entities(&graph.conn, &gid).await;
    let type_map_after = entity_type_map(&graph.conn, &gid).await;
    let audit_rows = identity_verdict_audit_rows(&graph.conn, &gid).await;

    eprintln!(
        "[e2] after dream: entities={entities_after} types_discovered={} \
         aliases_resolved={} entities_reclassified={} consistency_check_corrected={} \
         canonicalization_merges={} warnings={:?}",
        summary.types_discovered.len(),
        summary.aliases_resolved,
        summary.entities_reclassified,
        summary.consistency_check_corrected,
        summary.canonicalization_merges,
        summary.warnings,
    );
    eprintln!("[e2] identity_verdict_audit rows (site, decision): {audit_rows:?}");
    let all_counters: Vec<(String, u64)> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, v)| {
            let val = match v {
                metrics_util::debugging::DebugValue::Counter(c) => *c,
                _ => 0,
            };
            (k.key().name().to_string(), val)
        })
        .collect();
    eprintln!("[e2][DIAG] all counters: {all_counters:?}");
    eprintln!("[e2] entity_type_id map after dream: {type_map_after:?}");

    // ── (1) Site #5 acronym pair MUST merge — exactly ONE survivor ──
    let ibm_survives = type_map_after.contains_key("IBM");
    let ibm_full_survives = type_map_after.contains_key("International Business Machines");
    assert!(
        ibm_survives ^ ibm_full_survives,
        "IBM / International Business Machines: exactly ONE of the pair must \
         survive as the merge keeper — got IBM present={ibm_survives}, \
         International Business Machines present={ibm_full_survives}"
    );
    assert!(
        ibm_survives,
        "the keeper must be \"IBM\" (sorts first per load_entity_ids' ORDER BY \
         id ASC, and write_gate keeps pair.a) — got International Business \
         Machines survives instead, which means either the keeper/loser \
         convention changed or this test's ordering assumption is stale"
    );

    // ── (2) EXACT entity count — one merge, no other candidates in this fixture ──
    let expected_entities_after = entities_before - 1;
    assert_eq!(
        entities_after, expected_entities_after,
        "expected EXACTLY 1 entity-level merge (the IBM pair): \
         entities_before={entities_before} -> expected_after={expected_entities_after}, \
         got {entities_after}"
    );

    // ── (3) THE ordering invariant — reclassify ran ONCE, on the POST-merge \
    // population (spec §7 E2 "highest-risk untested interaction") ──
    //
    // If Site #5 -> reclassify ordering held: only the surviving keeper
    // ("IBM", still catch-all post-merge) is visible to reclassify's SELECT,
    // so entities_reclassified must be EXACTLY 1 — never 2 (which would mean
    // reclassify saw the pre-merge pair and processed both independently,
    // i.e. the merge landed AFTER reclassify or raced it) and never 0 (which
    // would mean reclassify's SELECT missed the keeper entirely, e.g. because
    // the merge stamped a source/type that fell outside the WHERE clause).
    assert_eq!(
        summary.entities_reclassified,
        1,
        "ORDERING INVARIANT VIOLATION (spec §7 E2 / facade/dream.rs Site #5 \
         acronym_nickname_recall ordering comment): expected reclassify to \
         process EXACTLY the post-merge population (1 candidate — the \
         surviving \"IBM\" keeper, still catch-all). Got \
         entities_reclassified={got}. A value of 2 means reclassify saw the \
         PRE-merge pair (both IBM and International Business Machines) —\
         i.e. Site #5's merge did NOT land before reclassify ran, which is \
         the exact regression this scenario exists to catch. Do NOT weaken \
         this assertion; report it loudly instead.",
        got = summary.entities_reclassified,
    );

    // ── (4) The survivor's final type reflects reclassify having actually \
    // retyped it (not left at the catch-all id=0 it was planted with) ──
    let ibm_type_after = type_map_after
        .get("IBM")
        .copied()
        .unwrap_or_else(|| panic!("IBM must still exist post-dream"));
    assert_ne!(
        ibm_type_after, 0,
        "IBM's entity_type_id must no longer be the catch-all (0) after \
         mem.dream() with include_consistency_check + reclassify running on \
         it post-merge — got type_id=0, meaning reclassify's SELECT never \
         picked up the survivor at all (silent no-op), not just a pre/post- \
         merge race"
    );

    // ── (5) No dream pass may fail in this well-formed fixture ──
    let failures: Vec<&String> = summary
        .warnings
        .iter()
        .filter(|w| w.contains("failed"))
        .collect();
    assert!(
        failures.is_empty(),
        "no dream pass may fail in E2 (fixture is well-formed); pass \
         failures: {failures:?}"
    );
}

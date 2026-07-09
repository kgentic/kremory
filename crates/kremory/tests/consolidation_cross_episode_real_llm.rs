//! ADR-066 dream CONSOLIDATION P3 — cross-episode merge L3 "isolated real-LLM" test.
//!
//! # Pyramid tier
//! This is the tier ABOVE the deterministic corpus + property harness
//! (`consolidation_cross_episode_test.rs`, hand-planted fixtures + a 300-iteration
//! seeded property test, zero LLM — plus the L1 unit/L2 module tests inside
//! `cross_episode.rs`) and BELOW the full dream e2e (`whole_project_e2e.rs`, real
//! ingest → `mem.dream()` runs ALL 5 reconciliation passes + consolidation). It
//! proves the cross-episode merge op behaves correctly when run IN ISOLATION over a
//! graph produced by REAL-model ingest (`mem.remember()` → phase-1 embed+NER →
//! phase-2 LLM relationships), catching real-data-shape bugs the hand-planted corpus
//! cannot. Only the ONE op under test runs against that real graph — nothing else
//! from the dream phase fires — so a failure localises to cross_episode, not to a
//! reconciliation pass or the P1/P2 supersession/archive lanes upstream. Mirrors
//! `consolidation_supersession_real_llm.rs` + `consolidation_archive_real_llm.rs`.
//!
//! # KEY FINDING (verified 2026-07-03 — the reason the SAFETY arm is primary)
//! cross_episode fires ONLY when the SAME entity (same normalized name-slug, OR
//! fuzzy Jaccard ≥ 0.9 on token-shingles) recurs across ≥ 2 DISTINCT episodes AND
//! the two candidates share STRUCTURE (a common third neighbour, or an identical
//! (predicate, object) assertion — the RISK-001 homonym guard, `cross_episode.rs`
//! §P3.1b). Real ingest of DISTINCT, non-overlapping domains produces entities whose
//! name-slugs do NOT recur across the episodes (an "Acme Corporation" episode and an
//! "Amazon River" episode share no entity), so on real-ingest data the op finds NO
//! candidate pair → merges EXACTLY 0. That 0-merge SAFETY property IS the load-bearing
//! arm: it proves the op does not spuriously fuse the distinct real entities that
//! distinct-domain ingest actually emits.
//!
//! The op's TRIGGER (a genuine cross-episode recurrence WITH shared structure) and
//! its critical HOMONYM-SAFETY case (same label across episodes with NO shared
//! structure — two distinct referents sharing a name) are NOT reliably reachable from
//! a fresh distinct-domain ingest: a real model may or may not emit the exact slug +
//! neighbour structure on demand, and coupling replay stability to nondeterministic
//! extraction would make the cassette brittle. So — exactly as the P1/P2 L3 siblings
//! plant their bounded-validity / long-expired facts via the real graph API onto the
//! real-ingested graph — this test PLANTS the trigger + homonym entity/episode/edge/
//! fact scenarios via the real `insert_entity_with_group` / `insert_episode` /
//! `insert_episodic_edge` / facts-table APIs AFTER the real ingest. That is still
//! "isolated real-LLM": the substrate under test operates over a graph that CONTAINS
//! real-model-ingested entities + facts, and the specific structural scenarios are
//! explicit (deterministic) plants layered on top.
//!
//! # Honesty ledger — which arm is real-ingest vs API-planted
//!   * ARM 1 (SAFETY, PRIMARY): **REAL-INGEST.** cross_episode over the graph as
//!     distinct-domain ingest left it → 0 merges. Nothing fused that shouldn't be.
//!   * ARM 2 (TRIGGER — MERGE): **API-PLANTED on top of the real graph.** A
//!     same-normalized entity pair anchored to 2 distinct episodes + a shared
//!     neighbour fact → exactly 1 merge (loser gone, facts/edges remapped to keeper).
//!   * ARM 3 (HOMONYM SAFETY — the RISK-001 real-data proof): **API-PLANTED on top of
//!     the real graph.** A same-label entity pair across 2 distinct episodes but with
//!     DISJOINT neighbours (distinct referents) → 0 merges (both survive).
//!   * ARM 4 (IDEMPOTENCY): a second full sweep merges 0 more.
//!
//! Arms 2 + 3 coexist on ONE graph. cross_episode sweeps the WHOLE group, so both the
//! trigger pair (shared structure → merges) and the homonym pair (disjoint structure →
//! survives) are present simultaneously in the TRIGGER sweep: the op must merge the
//! trigger pair AND leave the homonym pair alone in the SAME pass. That is the
//! sharpest possible statement of the RISK-001 property — the guard discriminates
//! merge-worthy recurrence from homonym coincidence within a single real sweep.
//!
//! # Two run modes (mirrors the P1/P2 L3 siblings), by `KREMORY_VCR`:
//!   * `KREMORY_VCR=record` → LIVE: real gemma4:e4b chat (`.think(false)`) wrapped in
//!     `RecordReplayChatProvider::record(...)` AND real nomic-embed-text wrapped in
//!     `RecordReplayEmbedder::record(...)`; one run refreshes BOTH committed cassettes.
//!     `provider.flush()` + `emb_vcr.flush()` are MANDATORY after the last background
//!     ingest write and before assertions.
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: fully deterministic, NO Ollama. BOTH
//!     cassettes are replayed; a missing entry is a LOUD error.
//!
//! Note: cross_episode itself is ZERO-LLM (deterministic label + structural SQL). The
//! VCR machinery here is solely for the real-model INGEST that builds the graph; the
//! op sweep consumes no cassette entries.
//!
//! # Determinism note
//! The two ingest episodes are DISTINCT, non-overlapping domains on purpose
//! (whole_project_e2e §28): overlapping facts across episodes trigger an ingest-time
//! contradiction-detection LLM call whose retrieved-context ordering is not yet
//! VCR-deterministic. Distinct domains avoid that call, so multi-episode replay is
//! byte-stable. The planted scenarios use dedicated `xe-*`-prefixed name-slugs that
//! cannot collide with any real-ingest entity slug.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
// Test files use expect/unwrap/panic as intentional assertion mechanisms
// (project-wide test convention — see golden_path_smoke.rs / whole_project_e2e.rs).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;

use kremory::core::dream::cross_episode;
use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

use support::test_log::init_test_log;

// ── Mode selection (mirrors whole_project_e2e / supersession + archive L3) ────

enum Mode {
    Live,
    Replay,
}

fn resolve_mode() -> Mode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => Mode::Live,
        Ok("replay") | Err(_) => Mode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("consolidation_cross_episode_real_llm.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("consolidation_cross_episode_real_llm.embeddings.json")
}

// ── Embedding record/replay (TD-093), copied from supersession + archive L3 ────
// Recall + phase-1 embed are SEMANTIC (cosine); a null/hash embedder gives
// spurious results, so replay must reuse the REAL nomic vectors captured at record
// time. record: delegate to real nomic + memoise each text→vector; replay: look
// up offline (loud MISS → re-record).
struct RecordReplayEmbedder {
    /// `Some` in record mode (real nomic), `None` in replay.
    inner: Option<Arc<dyn DynEmbeddingProvider>>,
    cache: Mutex<std::collections::HashMap<String, Vec<f32>>>,
    path: std::path::PathBuf,
}

impl RecordReplayEmbedder {
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        Self {
            inner: Some(inner),
            cache: Mutex::new(std::collections::HashMap::new()),
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
            cache: Mutex::new(map),
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
    async fn embed<'a>(&'a self, text: &'a str) -> kremory::CoreResult<Vec<f32>> {
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
                "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record (TD-093)"
            ))),
        }
    }
}

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_chat_model() -> String {
    // Project's benchmarked deferred-quality chat model with reasoning disabled
    // (facade default + metrics harness). Same model as whole_project_e2e.
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string())
}

fn real_ollama_chat() -> Arc<dyn ChatProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    // `.think(false)`: kremory extraction is structured-output, not reasoning.
    // `.keep_alive("1h")`: avoid keep_alive thrash silently dropping precision
    // (TD-024). `.timeout_seconds(180)` matches the harness.
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model(ollama_chat_model())
        .think(false)
        .keep_alive("1h")
        .timeout_seconds(180)
        .build()
        .expect("real Ollama chat provider must build (KREMORY_VCR=record requires Ollama)");
    llm as Arc<dyn ChatProvider>
}

fn real_ollama_embedder() -> Arc<dyn DynEmbeddingProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::embedding::EmbeddingBuilder;

    let raw: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model("nomic-embed-text")
        .build()
        .expect("real Ollama embedder must build (KREMORY_VCR=record requires Ollama)");
    Arc::new(OllamaEmbedderAdapter(raw))
}

struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed(&self, text: &str) -> kremory::CoreResult<Vec<f32>> {
        use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
        // nomic-embed-text REQUIRES a task prefix (TD-097). RecordReplayEmbedder
        // caches under the ORIGINAL `text`, so replay lookup is unaffected.
        let prefixed = format!("search_document: {text}");
        let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![prefixed])
            .await
            .map_err(|e| kremory::CoreError::Embedding(e.to_string()))?;
        vecs.pop().ok_or_else(|| {
            kremory::CoreError::Embedding(
                "OllamaEmbedderAdapter: embed returned empty vec".to_string(),
            )
        })
    }
}

// ── o11y cross-check ──────────────────────────────────────────────────────────

/// Sum the cross_episode merge counter across the snapshot (both `{path=exact|fuzzy}`
/// labels), mirroring `consolidation_cross_episode_test.rs::sum_merge_counter`. The
/// o11y cross-check asserts this SUM equals the op's reported merge count each arm.
fn sum_merge_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.cross_episode_merges_total" {
                return None;
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

// ── Direct-SQL helpers (targeted SELECTs, not the API under test) ─────────────

/// Does an entity with `id` still exist in the LIVE `entities` table for the group?
async fn entity_exists(graph: &TemporalGraph, group_id: &str, id: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT 1 FROM entities WHERE group_id = ?1 AND id = ?2",
            libsql::params![group_id, id],
        )
        .await
        .expect("entity exists query");
    rows.next().await.expect("iter").is_some()
}

/// Count facts still referencing `entity` as subject OR object in `group_id`.
/// A merged-away loser must have ZERO remaining references (remapped to keeper).
async fn fact_refs_to(graph: &TemporalGraph, group_id: &str, entity: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts \
             WHERE group_id = ?1 AND (subject_id = ?2 OR object_id = ?2)",
            libsql::params![group_id, entity],
        )
        .await
        .expect("fact refs query");
    scalar(&mut rows).await
}

/// Count episodic edges still referencing `entity` in `group_id`. A merged-away loser
/// must have ZERO remaining edges (remapped to keeper).
async fn edge_refs_to(graph: &TemporalGraph, group_id: &str, entity: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM episodic_edges \
             WHERE entity_id = ?1 AND (entity_group_id = ?2 OR entity_group_id IS NULL)",
            libsql::params![entity, group_id],
        )
        .await
        .expect("edge refs query");
    scalar(&mut rows).await
}

/// Count entities in the group whose id starts with `prefix` (used to assert the
/// real-ingest entity population is untouched by the planted-scenario sweeps).
async fn count_entities_not_prefixed(graph: &TemporalGraph, group_id: &str, prefix: &str) -> i64 {
    let pattern = format!("{prefix}%");
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = ?1 AND id NOT LIKE ?2",
            libsql::params![group_id, pattern],
        )
        .await
        .expect("non-prefixed entity count query");
    scalar(&mut rows).await
}

async fn scalar(rows: &mut libsql::Rows) -> i64 {
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

// ── Plant helpers (real graph API — the same surface the L2 corpus harness uses) ──

async fn plant_entity(graph: &TemporalGraph, group_id: &str, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(group_id),
        })
        .await
        .unwrap_or_else(|e| panic!("plant entity {id:?}: {e:?}"));
}

async fn plant_episode(graph: &TemporalGraph) -> i64 {
    graph
        .insert_episode(InsertEpisodeParams {
            content: "planted episode content",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("plant episode")
}

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption (mirrors L2 corpus `anchor`)
async fn anchor(graph: &TemporalGraph, group_id: &str, episode_id: i64, entity: &str) {
    graph
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id,
            entity_id: entity,
            entity_group_id: Some(group_id),
            role: "mention",
        })
        .await
        .expect("plant episodic edge");
}

/// Plant a relational fact `subject --predicate--> object` in `group_id` (mirrors
/// the L2 corpus harness `fact_rel`). is_dream_generated defaults to 0.
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption (mirrors L2 corpus `fact_rel`)
async fn plant_fact_rel(
    graph: &TemporalGraph,
    group_id: &str,
    subject: &str,
    predicate: &str,
    object: &str,
) {
    let now = Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
              subject_group_id, object_group_id, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
            libsql::params![
                subject,
                predicate,
                object,
                now.clone(),
                now,
                group_id,
                group_id,
                group_id
            ],
        )
        .await
        .expect("plant relational fact");
}

// ── Ingest corpus (distinct, non-overlapping domains — no contradiction call) ──
const EPISODES: &[(&str, &str)] = &[
    (
        "xe-e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "xe-e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
];

/// Run ONLY the cross_episode op over `group_id` and return its reported count.
/// Nothing else from the dream phase runs — this is the isolation the L3 tier
/// provides. cross_episode is 2-arg (graph, group_id) — no params struct (TD-042).
async fn run_cross_episode(graph: &TemporalGraph, group_id: &str) -> usize {
    // dry_run=false: the real-LLM corpus harness measures REAL cross-episode fusion.
    let report = cross_episode(graph, group_id, false)
        .await
        .expect("cross_episode op must succeed");
    report.count
}

// ── The isolated real-LLM test ────────────────────────────────────────────────

#[tokio::test]
#[ignore = "consolidation_cross_episode_real_llm: requires Ollama in record mode, or the committed \
            cassette in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test consolidation_cross_episode_real_llm -- \
            --ignored --nocapture"]
async fn cross_episode_isolated_over_real_ingest() {
    let _log = init_test_log("consolidation_cross_episode_real_llm");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) VCR-backed chat + embedder so replay faithfully reproduces the recorded
    //    real-model ingest offline (real nomic vectors → phase-1 embed works
    //    deterministically). Mirrors whole_project_e2e / P1 / P2 L3.
    let (provider, emb_vcr): (Arc<RecordReplayChatProvider>, Arc<RecordReplayEmbedder>) = match mode
    {
        Mode::Live => {
            let rec = Arc::new(RecordReplayChatProvider::record(
                real_ollama_chat(),
                cassette.clone(),
                ollama_chat_model(),
            ));
            let emb = Arc::new(RecordReplayEmbedder::record(
                real_ollama_embedder(),
                embedding_cassette_path(),
            ));
            (rec, emb)
        }
        Mode::Replay => {
            let rep = Arc::new(
                RecordReplayChatProvider::replay(cassette.clone())
                    .expect("replay cassette must load (record it via KREMORY_VCR=record)"),
            );
            let emb = Arc::new(RecordReplayEmbedder::replay(embedding_cassette_path()));
            (rep, emb)
        }
    };

    let llm: Arc<dyn ChatProvider> = provider.clone();
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();
    let embedding_dim: usize = 768;

    // 2) Build Memory through the public facade.
    let dir = tempfile::tempdir().expect("tempdir");
    let namespace = Namespace::new("cross-episode-real-llm");
    let mem = Memory::open(dir.path().join("cross_episode_real_llm.db"))
        .with_llm(llm)
        .with_embedder(embedder)
        .embedding_dim(embedding_dim)
        .default_namespace(namespace.clone())
        .await
        .expect("Memory::open must succeed");

    // 3) Real-model ingest of the distinct-domain episodes through the FULL public
    //    pipeline (phase1 embed+NER + phase2 LLM relationships).
    for (session, text) in EPISODES {
        let commit = mem
            .remember(*text)
            .from_chat(*session)
            .await
            .unwrap_or_else(|e| panic!("remember({session}) must succeed: {e:?}"));

        let episode_id: i64 = commit
            .episode_entity_id
            .parse()
            .expect("episode_entity_id must be a parseable i64 rowid");

        mem.wait_for_processing(episode_id, Duration::from_secs(90))
            .await
            .unwrap_or_else(|e| {
                panic!("wait_for_processing({session}) must reach Verified: {e:?}")
            });
    }

    // record mode: flush BOTH cassettes AFTER the last background ingest write and
    // BEFORE assertions (NEW-202).
    if matches!(mode, Mode::Live) {
        provider
            .flush()
            .expect("provider.flush() must succeed in record mode");
        emb_vcr.flush();
    }

    // The exact group `mem.dream()`/cross_episode operates on for this namespace.
    let group_id = mem.group_id_for_test(&namespace);
    let tg: Arc<TemporalGraph> = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // Real ingest must have produced SOME entities (distinct-domain → distinct slugs).
    // These are the entities the SAFETY arm proves cross_episode does NOT fuse.
    let real_entities_before = count_entities_not_prefixed(&tg, &group_id, "xe-").await;
    eprintln!(
        "[cross-episode-real-llm] group_id={group_id} real_ingest_entities={real_entities_before}"
    );
    assert!(
        real_entities_before >= 2,
        "real-model ingest must produce >= 2 entities across the two distinct-domain episodes; \
         got {real_entities_before}. If < 2, phase-1 NER / phase-2 extraction silently produced \
         no entities — a real ingest regression, not a cross_episode property."
    );

    // ── ARM 1 (PRIMARY — SAFETY, REAL-INGEST): cross_episode over the distinct-domain
    //    real graph merges EXACTLY 0. The "Acme Corporation" entities and the
    //    "Amazon River" entities share NO name-slug across episodes, so no candidate
    //    pair is admitted → nothing fused that shouldn't be. ───────────────────────
    let recorder1 = DebuggingRecorder::new();
    let snapshotter1 = recorder1.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&recorder1);

        let merged = run_cross_episode(&tg, &group_id).await;

        assert_eq!(
            merged, 0,
            "SAFETY VIOLATION: cross_episode merged {merged} entity pair(s) over a \
             real-model-ingested graph of DISTINCT-domain entities that share no recurring \
             name-slug across episodes. The op must not fuse distinct real entities."
        );
        let real_entities_after = count_entities_not_prefixed(&tg, &group_id, "xe-").await;
        assert_eq!(
            real_entities_after, real_entities_before,
            "SAFETY VIOLATION: the real-ingest entity population changed \
             ({real_entities_before} → {real_entities_after}) across a sweep that should have \
             merged nothing"
        );

        // o11y cross-check: the counter must AGREE with the op — 0 both sides.
        let counter = sum_merge_counter(&snapshotter1);
        assert_eq!(
            counter, 0,
            "cross_episode_merges counter ({counter}) must equal the op's merge count (0) on the \
             SAFETY arm"
        );
    }

    // ── Plant the TRIGGER pair (ARM 2) + the HOMONYM pair (ARM 3) onto the SAME real
    //    graph via the real graph API. Both pairs coexist so the TRIGGER sweep must
    //    merge the trigger pair AND leave the homonym pair alone in ONE pass. All
    //    planted ids are `xe-`-prefixed → cannot collide with real-ingest slugs. ────
    //
    // TRIGGER pair: two DISTINCT raw ids that NORMALIZE identically (case + whitespace)
    // — the exact-path candidate pair. Anchored to 2 DISTINCT episodes + a SHARED
    // neighbour fact (both --works_at--> the same org) → corroborated recurrence.
    // Both raw ids share the `xe-` prefix so the real-vs-planted count filter cleanly
    // excludes them; `cross_episode.rs` normalizes the id (case-fold + whitespace
    // collapse), so `"xe-Trigger-Co"` and `"xe-trigger-co "` both normalize to
    // `"xe-trigger-co"` → an exact-path candidate pair.
    let trig_a = "xe-Trigger-Co"; // keeper: uppercase 'T' 0x54 < lowercase 't' 0x74.
    let trig_b = "xe-trigger-co "; // loser (trailing space normalizes away).
    let trig_neighbour = "xe-shared-org";
    plant_entity(&tg, &group_id, trig_a).await;
    plant_entity(&tg, &group_id, trig_b).await;
    plant_entity(&tg, &group_id, trig_neighbour).await;

    // HOMONYM pair: same normalized label across 2 episodes, but DISJOINT neighbours
    // (two distinct referents sharing a name — the RISK-001 category). No shared third
    // entity, no identical (predicate, object) assertion → must NOT merge. Both raw
    // ids share the `xe-` prefix + normalize identically (case + trailing space).
    let hom_a = "xe-Homonym-Person"; // keeper.
    let hom_b = "xe-homonym-person "; // would-be loser IF the guard failed.
    let hom_neighbour_a = "xe-homonym-lawfirm";
    let hom_neighbour_b = "xe-homonym-olympics";
    plant_entity(&tg, &group_id, hom_a).await;
    plant_entity(&tg, &group_id, hom_b).await;
    plant_entity(&tg, &group_id, hom_neighbour_a).await;
    plant_entity(&tg, &group_id, hom_neighbour_b).await;

    // Two distinct episodes (real-slug-free) for the cross-episode span on both pairs.
    let ep1 = plant_episode(&tg).await;
    let ep2 = plant_episode(&tg).await;

    // TRIGGER anchors + SHARED-neighbour facts (corroborated) → the pair MERGES.
    anchor(&tg, &group_id, ep1, trig_a).await;
    anchor(&tg, &group_id, ep2, trig_b).await;
    plant_fact_rel(&tg, &group_id, trig_a, "works_at", trig_neighbour).await;
    plant_fact_rel(&tg, &group_id, trig_b, "works_at", trig_neighbour).await;

    // HOMONYM anchors + DISJOINT facts (no shared structure) → the pair SURVIVES.
    anchor(&tg, &group_id, ep1, hom_a).await;
    anchor(&tg, &group_id, ep2, hom_b).await;
    plant_fact_rel(&tg, &group_id, hom_a, "works_at", hom_neighbour_a).await;
    plant_fact_rel(&tg, &group_id, hom_b, "competed_in", hom_neighbour_b).await;

    // Sanity: all six planted principals exist pre-sweep.
    for id in [trig_a, trig_b, hom_a, hom_b] {
        assert!(
            entity_exists(&tg, &group_id, id).await,
            "planted principal {id:?} must exist before the trigger sweep"
        );
    }

    // ── ARM 2 (TRIGGER — MERGE) + ARM 3 (HOMONYM SAFETY) in ONE sweep. ────────────
    let recorder2 = DebuggingRecorder::new();
    let snapshotter2 = recorder2.snapshotter();
    let merged_trigger = {
        let _guard = metrics::set_default_local_recorder(&recorder2);
        run_cross_episode(&tg, &group_id).await
    };

    // EXACTLY 1 merge: the trigger pair fuses; the homonym pair does not; the real
    // entities do not. A count != 1 means either the trigger failed to fire OR the
    // homonym guard false-merged OR a real entity was fused.
    assert_eq!(
        merged_trigger, 1,
        "trigger+homonym sweep: cross_episode must merge EXACTLY the ONE corroborated trigger \
         pair (the homonym pair + all real entities must be left alone); op reported \
         {merged_trigger}"
    );

    // ARM 2 detail: keeper survives, loser GONE, loser's facts + edges remapped away.
    assert!(
        entity_exists(&tg, &group_id, trig_a).await,
        "trigger arm: keeper {trig_a:?} (lowest id) must survive the merge"
    );
    assert!(
        !entity_exists(&tg, &group_id, trig_b).await,
        "trigger arm: loser {trig_b:?} must be gone after the merge"
    );
    assert_eq!(
        fact_refs_to(&tg, &group_id, trig_b).await,
        0,
        "trigger arm: no fact may still reference the merged-away loser {trig_b:?} (remapped to keeper)"
    );
    assert_eq!(
        edge_refs_to(&tg, &group_id, trig_b).await,
        0,
        "trigger arm: no episodic edge may still reference the merged-away loser {trig_b:?}"
    );
    // The shared neighbour is untouched.
    assert!(
        entity_exists(&tg, &group_id, trig_neighbour).await,
        "trigger arm: the shared neighbour {trig_neighbour:?} must be untouched"
    );

    // ARM 3 detail (the critical RISK-001 real-data proof): BOTH homonym referents
    // SURVIVE — same label alone with NO shared structure must NOT merge.
    let homonym_a_survives = entity_exists(&tg, &group_id, hom_a).await;
    let homonym_b_survives = entity_exists(&tg, &group_id, hom_b).await;
    let homonym_no_merge = homonym_a_survives && homonym_b_survives;
    assert!(
        homonym_no_merge,
        "HOMONYM SAFETY VIOLATION (RISK-001): the same-label homonym pair \
         ({hom_a:?} / {hom_b:?}) with DISJOINT neighbours must NOT merge — both must survive. \
         a_survives={homonym_a_survives} b_survives={homonym_b_survives}"
    );

    // The real-ingest entities are UNTOUCHED by this mixed sweep.
    let real_entities_after_trigger = count_entities_not_prefixed(&tg, &group_id, "xe-").await;
    assert_eq!(
        real_entities_after_trigger, real_entities_before,
        "trigger+homonym sweep: the {real_entities_before} real-ingest entities must be UNTOUCHED; \
         count changed to {real_entities_after_trigger}"
    );

    // o11y cross-check on the trigger arm: counter == op count == 1.
    let counter2 = sum_merge_counter(&snapshotter2);
    assert_eq!(
        counter2, 1,
        "cross_episode_merges counter ({counter2}) must equal the trigger-arm merge count (1)"
    );

    // ── ARM 4 (IDEMPOTENCY): a second sweep merges 0 more. The trigger loser is gone
    //    (its cluster is now one entity), the homonym pair is still a non-merge, and
    //    the real entities are still distinct → no eligible pair remains. ───────────
    let recorder3 = DebuggingRecorder::new();
    let snapshotter3 = recorder3.snapshotter();
    let merged_second = {
        let _guard = metrics::set_default_local_recorder(&recorder3);
        run_cross_episode(&tg, &group_id).await
    };
    assert_eq!(
        merged_second, 0,
        "idempotency: a second cross_episode sweep must merge 0 more (trigger loser already gone, \
         homonym pair still guarded, real entities still distinct); got {merged_second}"
    );
    // Homonym pair STILL both present after the idempotent sweep.
    assert!(
        entity_exists(&tg, &group_id, hom_a).await && entity_exists(&tg, &group_id, hom_b).await,
        "idempotency: the homonym pair must STILL both survive the second sweep"
    );
    let counter3 = sum_merge_counter(&snapshotter3);
    assert_eq!(
        counter3, 0,
        "idempotency: cross_episode_merges counter must be 0 on the second sweep; got {counter3}"
    );

    eprintln!(
        "[cross-episode-real-llm] PASS — SAFETY(real distinct entities)=0 merged, \
         TRIGGER(planted corroborated recurrence)=1 merged, HOMONYM(planted same-label disjoint)=0 \
         merged (both survive), idempotent second sweep=0."
    );

    drop(dir);
}

//! ADR-066 dream CONSOLIDATION P4 — community-detection L3 "isolated real-LLM" test.
//!
//! # Pyramid tier
//! This is the tier ABOVE the deterministic fixture corpus + 300-iteration property
//! harness (the L1 unit / L2 module tests inside `communities.rs`) and BELOW the full
//! dream e2e (`whole_project_e2e.rs`, real ingest → `mem.dream()`). It proves the
//! community-detection op behaves correctly when run IN ISOLATION over a graph produced
//! by REAL-model ingest (`mem.remember()` → phase-1 embed+NER → phase-2 LLM
//! relationships), catching real-data-shape bugs the hand-planted fixtures cannot. Only
//! the ONE op under test runs against that real graph — nothing else from the dream
//! phase fires — so a failure localises to `communities`, not to a reconciliation pass
//! or the P1/P2/P3 supersession/archive/cross_episode lanes. Mirrors
//! `consolidation_supersession_real_llm.rs` / `consolidation_archive_real_llm.rs` /
//! `consolidation_cross_episode_real_llm.rs`.
//!
//! # KEY FINDING — why distinct-domain ingest yields DISTINCT communities
//! The communities op is ZERO-LLM (the VCR here is SOLELY for the real-model ingest that
//! builds the entity/episode graph; the op sweep consumes no cassette entries). It builds
//! a `petgraph::UnGraph` of entity CO-OCCURRENCE (two entities that anchor to the SAME
//! episode) and runs deterministic weighted label propagation. Real ingest of MULTIPLE
//! DISTINCT, non-overlapping domains (the `whole_project_e2e` shape: a company, a river,
//! a scientist, a recipe) produces a graph that is a SET OF DISJOINT per-episode cliques:
//! the "Acme Corporation" entities co-occur only inside the company episode, the "Amazon
//! River" entities only inside the river episode, and the two cliques share NO entity
//! (distinct domains → name-slugs never recur across episodes,
//! `consolidation_cross_episode_real_llm.rs:22-27`). Label propagation on disjoint cliques
//! yields ONE community PER clique → MULTIPLE communities. This matches the P4 modularity
//! go/no-go spike's REALISTIC topology D (mirrors this exact fixture): Q≈0.71,
//! communities == #distinct-domains, deterministic
//! (`.ai-docs/research/p4-community-modularity-spike-2026-07-03.md`).
//!
//! # HAIRBALL CAVEAT (documented known limitation, NOT a bug — DoD-P4.6 / RISK-004)
//! The multi-community result depends on the graph being MULTI-episode + multi-domain
//! (disjoint cliques). A namespace dominated by ONE large episode in which most entities
//! co-occur is a near-complete "hairball": synchronous label propagation HONESTLY
//! collapses it to a SINGLE community (Q≈0.0, spike topology B) — the correct answer for a
//! genuine blob, not garbage. This L3 test deliberately ingests FOUR distinct domains so
//! the real graph is the multi-clique regime, not the hairball; the hairball collapse is
//! proved by the L2 `dense_episode_hairball_collapses_to_one_honest_community` fixture.
//!
//! # Honesty ledger — which arm is real-ingest vs API-planted
//!   * ARM 1 (STRUCTURE, PRIMARY): **REAL-INGEST.** `communities` over the graph as
//!     4-distinct-domain ingest left it → MULTIPLE communities (> 1), every real entity
//!     in exactly one community, persisted to `entity_communities` + `community_summaries`,
//!     `communities_updated > 0` on the first sweep.
//!   * ARM 2 (DETERMINISM): the op is re-run on the SAME unchanged real graph → the
//!     persisted partition (entity → community map) is BYTE-IDENTICAL to the first sweep.
//!   * ARM 3 (IDEMPOTENCY): that same second sweep reports `communities_updated == 0` and
//!     every persisted `member_hash` is unchanged (DoD-P4.5).
//!
//! No API planting is needed here: real 4-domain ingest reliably produces the disjoint-
//! clique structure (the spike's topology D is grounded in this exact fixture — it does
//! NOT depend on the model emitting one specific slug, only on each distinct-domain
//! episode naming ≥1 entity, which the whole_project_e2e floor already asserts). The test
//! asserts a MULTI-community floor (> 1), not the exact count, because the number of real
//! entities per domain is nondeterministic. If a future model regression made real ingest
//! yield fewer than 2 distinct communities, this test would FAIL LOUDLY (structure floor)
//! rather than silently pass — the planted-arm fallback the P3 sibling uses is not wired
//! because the primary real-ingest arm is reliable here; it would be added (honestly
//! labelled) only if the floor proved flaky.
//!
//! # Two run modes (mirrors the P1/P2/P3 L3 siblings), by `KREMORY_VCR`:
//!   * `KREMORY_VCR=record` → LIVE: real gemma4:e4b chat (`.think(false)`) wrapped in
//!     `RecordReplayChatProvider::record(...)` AND real nomic-embed-text wrapped in
//!     `RecordReplayEmbedder::record(...)`; one run refreshes BOTH committed cassettes.
//!     `provider.flush()` + `emb_vcr.flush()` are MANDATORY after the last background
//!     ingest write and before assertions.
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: fully deterministic, NO Ollama. BOTH
//!     cassettes are replayed; a missing entry is a LOUD error.
//!
//! # Determinism note
//! The four ingest episodes are DISTINCT, non-overlapping domains on purpose
//! (whole_project_e2e §286): overlapping facts across episodes trigger an ingest-time
//! contradiction-detection LLM call whose retrieved-context ordering is not yet
//! VCR-deterministic. Distinct domains avoid that call, so multi-episode replay is
//! byte-stable.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
// Test files use expect/unwrap/panic as intentional assertion mechanisms
// (project-wide test convention — see golden_path_smoke.rs / whole_project_e2e.rs).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kremory::core::dream::communities;
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

use support::test_log::init_test_log;

// ── Mode selection (mirrors whole_project_e2e / supersession + archive + cross_episode L3) ──

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
        .join("consolidation_communities_real_llm.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("consolidation_communities_real_llm.embeddings.json")
}

// ── Embedding record/replay (TD-093), copied from the P1/P2/P3 L3 siblings ─────
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

/// Sum the `communities_updated` counter across the snapshot, mirroring the
/// `communities.rs` op counter (`emit_updated_counter`). The o11y cross-check asserts
/// this SUM equals the op's reported `communities_updated` count each arm.
fn sum_updated_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.communities_updated_total" {
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

/// The persisted `(entity_id -> community_id)` map for `group_id` (from
/// `entity_communities`). ARM 2 (determinism) compares two of these for byte-equality.
async fn persisted_membership(graph: &TemporalGraph, group_id: &str) -> BTreeMap<String, i64> {
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id, community_id FROM entity_communities \
             WHERE group_id = ?1 ORDER BY entity_id",
            libsql::params![group_id],
        )
        .await
        .expect("membership query");
    let mut out = BTreeMap::new();
    while let Some(row) = rows.next().await.expect("row") {
        out.insert(
            row.get::<String>(0).expect("entity_id"),
            row.get::<i64>(1).expect("community_id"),
        );
    }
    out
}

/// Distinct community count from `community_summaries` for `group_id`.
async fn persisted_community_count(graph: &TemporalGraph, group_id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM community_summaries WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await
        .expect("community count query");
    scalar(&mut rows).await
}

/// The set of persisted `member_hash` values for `group_id`. ARM 3 (idempotency)
/// asserts these are UNCHANGED across the second sweep.
async fn persisted_member_hashes(
    graph: &TemporalGraph,
    group_id: &str,
) -> std::collections::BTreeSet<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT member_hash FROM community_summaries WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await
        .expect("member_hash query");
    let mut out = std::collections::BTreeSet::new();
    while let Some(row) = rows.next().await.expect("row") {
        out.insert(row.get::<String>(0).expect("member_hash"));
    }
    out
}

/// Count NON-catch-all entities (`entity_type_id != 0`) in the group — the population
/// that participates in community structure (C-INV5). Asserts real ingest produced a
/// clusterable node set.
async fn count_typed_entities(graph: &TemporalGraph, group_id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = ?1 AND entity_type_id != 0",
            libsql::params![group_id],
        )
        .await
        .expect("typed entity count query");
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

// ── Ingest corpus (FOUR distinct, non-overlapping domains — no contradiction call,
//    disjoint per-episode cliques → the multi-community regime, spike topology D) ──
const EPISODES: &[(&str, &str)] = &[
    (
        "comm-e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "comm-e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
    (
        "comm-e2e-scientist",
        "Marie Curie, a physicist born in Warsaw, was awarded Nobel Prizes in both Physics \
         and Chemistry for her research on radioactivity.",
    ),
    (
        "comm-e2e-recipe",
        "The banana bread recipe calls for two cups of flour, a teaspoon of baking soda, \
         and three ripe bananas mashed into the batter.",
    ),
];

/// Run ONLY the communities op over `group_id` and return its reported
/// `communities_updated` count. Nothing else from the dream phase runs — this is the
/// isolation the L3 tier provides. `communities` is 2-arg (graph, group_id) — no params
/// struct (TD-042).
async fn run_communities(graph: &TemporalGraph, group_id: &str) -> usize {
    let report = communities(graph, group_id)
        .await
        .expect("communities op must succeed");
    report.count
}

// ── The isolated real-LLM test ────────────────────────────────────────────────

#[tokio::test]
#[ignore = "consolidation_communities_real_llm: requires Ollama in record mode, or the committed \
            cassette in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test consolidation_communities_real_llm -- \
            --ignored --nocapture"]
async fn communities_isolated_over_real_ingest() {
    let _log = init_test_log("consolidation_communities_real_llm");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) VCR-backed chat + embedder so replay faithfully reproduces the recorded
    //    real-model ingest offline (real nomic vectors → phase-1 embed works
    //    deterministically). Mirrors whole_project_e2e / P1 / P2 / P3 L3.
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
    let namespace = Namespace::new("communities-real-llm");
    let mem = Memory::open(dir.path().join("communities_real_llm.db"))
        .with_llm(llm)
        .with_embedder(embedder)
        .embedding_dim(embedding_dim)
        .default_namespace(namespace.clone())
        .await
        .expect("Memory::open must succeed");

    // 3) Real-model ingest of the FOUR distinct-domain episodes through the FULL public
    //    pipeline (phase1 embed+NER + phase2 LLM relationships). Distinct domains →
    //    disjoint per-episode co-occurrence cliques → the multi-community regime.
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

    // The exact group `mem.dream()`/communities operates on for this namespace.
    let group_id = mem.group_id_for_test(&namespace);
    let tg: Arc<TemporalGraph> = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // Real ingest must have produced a clusterable node set. Each distinct-domain episode
    // names ≥1 typed entity; a floor of 2 typed entities across the four domains catches a
    // phase-1 NER / phase-2 extraction regression that would leave the graph un-clusterable.
    let typed_before = count_typed_entities(&tg, &group_id).await;
    eprintln!(
        "[communities-real-llm] group_id={group_id} real_ingest_typed_entities={typed_before}"
    );
    assert!(
        typed_before >= 2,
        "real-model ingest must produce >= 2 typed (non-catch-all) entities across the four \
         distinct-domain episodes; got {typed_before}. If < 2, phase-1 NER / phase-2 extraction \
         silently produced no typed entities — a real ingest regression, not a communities property."
    );

    // ── ARM 1 (PRIMARY — STRUCTURE, REAL-INGEST): communities over the 4-distinct-domain
    //    real graph forms MULTIPLE communities (> 1). The distinct domains' entities are
    //    disjoint per-episode cliques (an "Acme Corporation" episode and an "Amazon River"
    //    episode share no entity) so label propagation surfaces one community per domain
    //    clique. Every real entity lands in exactly one community; the partition persists;
    //    communities_updated > 0 on the first sweep. ─────────────────────────────────────
    let recorder1 = DebuggingRecorder::new();
    let snapshotter1 = recorder1.snapshotter();
    let (updated_first, membership_first) = {
        let _guard = metrics::set_default_local_recorder(&recorder1);

        let updated = run_communities(&tg, &group_id).await;
        let membership = persisted_membership(&tg, &group_id).await;
        (updated, membership)
    };

    let community_count = persisted_community_count(&tg, &group_id).await;
    eprintln!(
        "[communities-real-llm] ARM1 STRUCTURE — communities_updated={updated_first} \
         persisted_communities={community_count} members_assigned={}",
        membership_first.len()
    );

    // MULTI-community floor (not an exact count — real per-domain entity counts are
    // nondeterministic). Distinct domains ⇒ disjoint cliques ⇒ > 1 community (spike D).
    assert!(
        community_count > 1,
        "STRUCTURE: communities over a real 4-distinct-domain ingest must form MORE THAN ONE \
         community (disjoint per-domain cliques, spike topology D Q≈0.71); got \
         {community_count}. A count of 1 would mean the real graph collapsed to a hairball — \
         which the distinct-domain fixture is specifically constructed to avoid."
    );

    // First sweep: every persisted community is new → communities_updated == community_count.
    assert_eq!(
        updated_first as i64, community_count,
        "STRUCTURE: on the first sweep every community is new, so communities_updated \
         ({updated_first}) must equal the persisted community count ({community_count})"
    );
    assert!(
        updated_first > 1,
        "STRUCTURE: communities_updated ({updated_first}) must be > 1 on the first sweep of the \
         multi-domain real graph"
    );

    // Every real entity is assigned to EXACTLY one community. `entity_communities` PK is
    // (group_id, entity_id), so a present key IS "exactly one"; assert the assigned count
    // equals the typed-entity population (no entity dropped, no catch-all leaked in).
    assert_eq!(
        membership_first.len() as i64,
        typed_before,
        "STRUCTURE: every one of the {typed_before} typed real entities must be assigned to \
         exactly one community; got {} assigned. A mismatch means an entity was dropped from \
         the partition OR a catch-all (type 0) leaked into it.",
        membership_first.len()
    );

    // o11y cross-check on ARM 1: the counter must AGREE with the op's report.
    let counter1 = sum_updated_counter(&snapshotter1);
    assert_eq!(
        counter1, updated_first as u64,
        "communities_updated counter ({counter1}) must equal the op's reported count \
         ({updated_first}) on the STRUCTURE arm"
    );

    // Snapshot the first-sweep member_hashes for the ARM 3 idempotency comparison.
    let hashes_first = persisted_member_hashes(&tg, &group_id).await;

    // ── ARM 2 (DETERMINISM) + ARM 3 (IDEMPOTENCY): re-run the op on the SAME unchanged
    //    real graph. The op full-recomputes (clear + re-persist) each run, so an identical
    //    persisted partition proves label propagation is a pure function of the topology
    //    (ARM 2); communities_updated == 0 + unchanged member_hashes prove idempotency
    //    (ARM 3, DoD-P4.5). ───────────────────────────────────────────────────────────
    let recorder2 = DebuggingRecorder::new();
    let snapshotter2 = recorder2.snapshotter();
    let (updated_second, membership_second) = {
        let _guard = metrics::set_default_local_recorder(&recorder2);

        let updated = run_communities(&tg, &group_id).await;
        let membership = persisted_membership(&tg, &group_id).await;
        (updated, membership)
    };

    // ARM 2: the second-sweep partition is BYTE-IDENTICAL to the first (deterministic).
    assert_eq!(
        membership_first, membership_second,
        "DETERMINISM: re-running communities on the unchanged real graph must yield the \
         IDENTICAL entity→community partition. A difference means label propagation is \
         non-deterministic on real-ingest data."
    );

    // ARM 3: unchanged graph → every member_hash matches → 0 updated (idempotent).
    assert_eq!(
        updated_second, 0,
        "IDEMPOTENCY (DoD-P4.5): a second communities sweep over the unchanged real graph must \
         report communities_updated == 0 (every membership set already persisted); got \
         {updated_second}"
    );
    let hashes_second = persisted_member_hashes(&tg, &group_id).await;
    assert_eq!(
        hashes_first, hashes_second,
        "IDEMPOTENCY: the persisted member_hashes must be UNCHANGED across the idempotent \
         second sweep"
    );
    // The community count is stable across the idempotent sweep.
    assert_eq!(
        persisted_community_count(&tg, &group_id).await,
        community_count,
        "IDEMPOTENCY: the persisted community count must be unchanged ({community_count}) after \
         the second sweep"
    );

    // o11y cross-check on ARM 3: counter == op count == 0.
    let counter2 = sum_updated_counter(&snapshotter2);
    assert_eq!(
        counter2, 0,
        "IDEMPOTENCY: communities_updated counter ({counter2}) must equal the second-sweep \
         count (0)"
    );

    eprintln!(
        "[communities-real-llm] PASS — STRUCTURE(real 4-distinct-domain ingest)={community_count} \
         communities formed ({updated_first} updated first sweep, {} entities each in exactly one \
         community), DETERMINISM(identical partition on rerun)=true, IDEMPOTENCY(second sweep \
         updated)=0.",
        membership_first.len()
    );

    drop(dir);
}

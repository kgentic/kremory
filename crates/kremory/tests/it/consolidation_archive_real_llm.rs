//! ADR-066 dream CONSOLIDATION P2 — archive L3 "isolated real-LLM" test.
//!
//! # Pyramid tier
//! This is the tier ABOVE the deterministic corpus harness
//! (`consolidation_archive_test.rs`, hand-planted fixtures, zero LLM — plus the L1
//! unit/L2 module tests inside `archive.rs`) and BELOW the full dream e2e
//! (`whole_project_e2e.rs`, real ingest → `mem.dream()` runs ALL 5 reconciliation
//! passes + consolidation). It proves the archive op behaves correctly when run IN
//! ISOLATION over a graph produced by REAL-model ingest (`mem.remember()` →
//! phase-1 embed+NER → phase-2 LLM relationships), catching real-data-shape bugs
//! the hand-planted corpus cannot. Only the ONE op under test runs against that
//! real graph — nothing else from the dream phase fires — so a failure localises to
//! archive, not to a reconciliation pass or the P1 supersession lane upstream.
//! Mirrors `consolidation_supersession_real_llm.rs` (the P1 L3 sibling).
//!
//! # KEY FINDING (verified 2026-07-03 — the reason the SAFETY arm is primary)
//! The archive op targets facts that are EXPIRED and past the grace window:
//! `expired_at IS NOT NULL AND expired_at < now - grace_days`
//! (`archive.rs::archive`). But the public `remember()` / LLM-extraction path emits
//! facts with `expired_at = NULL` — a fact is only expired once supersession /
//! invalidation retires it (P1) or a caller sets it explicitly. Real-model ingest
//! therefore produces OPEN-ENDED, NON-EXPIRED facts, which the archive candidate
//! SELECT excludes by its first predicate clause (`expired_at IS NOT NULL`).
//!
//! Therefore archive's real-world trigger is NOT reachable from a fresh ingest
//! alone: it depends on supersession/invalidation having run first (P1) — or the
//! caller setting expiry. This test documents that dependency and covers both
//! ends of it:
//!   1. **PRIMARY — SAFETY property.** Over the real-model-ingested graph (all facts
//!      open-ended, `expired_at NULL`), archive must move EXACTLY 0 facts — nothing
//!      wrongly archived. This is the load-bearing arm: it proves the op does not
//!      false-archive on the fact shape production actually emits. `facts` unchanged,
//!      `facts_archive` empty.
//!   2. **SECONDARY — trigger path.** Plant ONE fact whose `expired_at` is 100 days
//!      ago (well past the 90-day grace) via the real facts-table SQL onto the SAME
//!      real-ingested graph, re-run archive, and assert it moves EXACTLY that one
//!      fact: the live `facts` row is gone, a `facts_archive` row is present with
//!      `archived_at` stamped, its `facts_fts` shadow row is gone (RISK-003), and no
//!      real open-ended fact is touched.
//!   3. **IDEMPOTENCY.** A second archive run over the same graph moves 0 more (the
//!      planted fact is gone from `facts`, so the candidate SELECT self-excludes it —
//!      P2.4).
//!
//! # Two run modes (mirrors consolidation_supersession_real_llm), by `KREMORY_VCR`:
//!   * `KREMORY_VCR=record` → LIVE: real gemma4:e4b chat (`.think(false)`) wrapped
//!     in `RecordReplayChatProvider::record(...)` AND real nomic-embed-text wrapped
//!     in `RecordReplayEmbedder::record(...)`; one run refreshes BOTH committed
//!     cassettes. `provider.flush()` + `emb_vcr.flush()` are MANDATORY after the
//!     last background write and before assertions.
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: fully deterministic, NO Ollama.
//!     BOTH cassettes are replayed; a missing entry is a LOUD error.
//!
//! # Determinism note
//! The two ingest episodes are DISTINCT, non-overlapping domains on purpose
//! (whole_project_e2e §28): overlapping facts across episodes trigger an
//! ingest-time contradiction-detection LLM call whose retrieved-context ordering
//! is not yet VCR-deterministic. Distinct domains avoid that call, so multi-episode
//! replay is byte-stable.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
// Test files use expect/unwrap/panic as intentional assertion mechanisms
// (project-wide test convention — see golden_path_smoke.rs / whole_project_e2e.rs).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};

use kremory::core::dream::archive;
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

use crate::support::test_log::init_test_log;

// The default archival grace window the trigger arm plants past.
const ARCHIVE_GRACE_DAYS: u32 = 90;

// ── Mode selection (mirrors whole_project_e2e / supersession L3) ──────────────

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
        .join("consolidation_archive_real_llm.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("consolidation_archive_real_llm.embeddings.json")
}

// ── Embedding record/replay (TD-093), copied from supersession L3 ──────────────
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
    crate::helpers::chat_model::chat_model_or("gemma4:e4b")
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

/// Sum the archive counter across the snapshot. Unlike the P1 window-closeout
/// counter, `kremory.dream.consolidation.facts_archived_total` carries NO `lane`
/// label (`archive.rs::emit_archived_counter`), so no label filter is applied.
fn sum_archived_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.facts_archived_total" {
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

/// Count facts still in the LIVE `facts` table for the group.
async fn count_live_facts(graph: &TemporalGraph, group_id: &str) -> i64 {
    count_scalar(
        graph,
        "SELECT COUNT(*) FROM facts WHERE group_id = ?1",
        group_id,
    )
    .await
}

/// Count facts whose `expired_at` is set in the live table (used to assert real
/// ingest produced zero pre-expired facts).
async fn count_expired_facts(graph: &TemporalGraph, group_id: &str) -> i64 {
    count_scalar(
        graph,
        "SELECT COUNT(*) FROM facts WHERE group_id = ?1 AND expired_at IS NOT NULL",
        group_id,
    )
    .await
}

/// Count rows moved into `facts_archive` for the group.
async fn count_archived_rows(graph: &TemporalGraph, group_id: &str) -> i64 {
    count_scalar(
        graph,
        "SELECT COUNT(*) FROM facts_archive WHERE group_id = ?1",
        group_id,
    )
    .await
}

async fn count_scalar(graph: &TemporalGraph, sql: &str, group_id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![group_id])
        .await
        .expect("count query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

async fn fact_row_exists(graph: &TemporalGraph, fact_id: i64) -> bool {
    row_exists(graph, "SELECT 1 FROM facts WHERE id = ?1", fact_id).await
}

async fn archive_row_exists(graph: &TemporalGraph, fact_id: i64) -> bool {
    row_exists(graph, "SELECT 1 FROM facts_archive WHERE id = ?1", fact_id).await
}

async fn fts_shadow_count_for(graph: &TemporalGraph, fact_id: i64) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts_fts WHERE fact_id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query facts_fts");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count")
}

async fn read_archived_at(graph: &TemporalGraph, fact_id: i64) -> Option<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT archived_at FROM facts_archive WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query archived_at");
    let row = rows.next().await.expect("row").expect("present");
    row.get::<Option<String>>(0).expect("archived_at col")
}

async fn row_exists(graph: &TemporalGraph, sql: &str, id: i64) -> bool {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![id])
        .await
        .expect("exists query");
    rows.next().await.expect("row").is_some()
}

// ── Ingest corpus (distinct, non-overlapping domains — no contradiction call) ──
const EPISODES: &[(&str, &str)] = &[
    (
        "arc-e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "arc-e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
];

/// Run ONLY the archive op over `group_id` and return its reported count. Nothing
/// else from the dream phase runs — this is the isolation the L3 tier provides.
async fn run_archive(graph: &TemporalGraph, group_id: &str) -> usize {
    let report = archive(graph, group_id, ARCHIVE_GRACE_DAYS)
        .await
        .expect("archive op must succeed");
    report.count
}

// ── The isolated real-LLM test ────────────────────────────────────────────────

#[tokio::test]
#[ignore = "consolidation_archive_real_llm: requires Ollama in record mode, or the committed \
            cassette in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test it consolidation_archive_real_llm:: -- \
            --ignored --nocapture"]
async fn archive_isolated_over_real_ingest() {
    let _log = init_test_log("consolidation_archive_real_llm");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) VCR-backed chat + embedder so replay faithfully reproduces the recorded
    //    real-model ingest offline (real nomic vectors → phase-1 embed works
    //    deterministically). Mirrors whole_project_e2e / supersession L3.
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
    let namespace = Namespace::new("archive-real-llm");
    let mem = Memory::open(dir.path().join("archive_real_llm.db"))
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

    // The exact group `mem.dream()`/archive operates on for this namespace.
    let group_id = mem.group_id_for_test(&namespace);
    let tg: Arc<TemporalGraph> = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // Real ingest must have produced SOME facts, and (the load-bearing finding)
    // NONE of them may already be expired — extraction sets `expired_at = NULL`.
    // If a future temporal/supersession change lands and real ingest starts
    // emitting expired facts, THIS assertion breaks LOUDLY and the SAFETY arm below
    // must be re-derived — that is intended.
    let live_before = count_live_facts(&tg, &group_id).await;
    let expired_before = count_expired_facts(&tg, &group_id).await;
    eprintln!(
        "[archive-real-llm] group_id={group_id} live_facts={live_before} \
         expired_facts_before={expired_before}"
    );
    assert!(
        live_before >= 1,
        "real-model ingest must produce >= 1 fact; got {live_before}. If 0, phase-2 \
         relationship extraction silently produced no facts — a real ingest regression, not an \
         archive property."
    );
    assert_eq!(
        expired_before, 0,
        "no fact should already be expired straight out of ingest (expired_at NULL by \
         construction); got {expired_before}. archive's real-world trigger depends on \
         supersession/invalidation (P1) running first — see module doc KEY FINDING."
    );

    // ── ARM 1 (PRIMARY — SAFETY): archive over real fresh facts moves EXACTLY 0.
    //    `facts` unchanged, `facts_archive` empty. Nothing wrongly archived. ──────
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&recorder);

        let archived = run_archive(&tg, &group_id).await;

        // SAFETY FINDING (see module doc KEY FINDING): the archive candidate SELECT
        // requires `expired_at IS NOT NULL`, but real ingest emits `expired_at = NULL`
        // for every extracted fact. So over a REAL-model-ingested graph the sweep MUST
        // touch nothing. archive's real-world trigger depends on supersession /
        // invalidation (P1) having retired a fact first — until then real ingest cannot
        // itself produce an archive candidate, and this SAFETY arm is the primary
        // property this L3 test guards (0 false-archive on the fact shape production
        // actually emits).
        assert_eq!(
            archived, 0,
            "SAFETY VIOLATION: archive moved {archived} fact(s) over a real-model-ingested graph \
             whose facts are ALL non-expired (expired_at = NULL). The archive lane must touch \
             nothing when no fact is expired past the grace window."
        );
        let live_after_safety = count_live_facts(&tg, &group_id).await;
        assert_eq!(
            live_after_safety, live_before,
            "SAFETY VIOLATION: live facts changed ({live_before} → {live_after_safety}) across a \
             real-ingest archive sweep that should have moved nothing"
        );
        let archived_rows_safety = count_archived_rows(&tg, &group_id).await;
        assert_eq!(
            archived_rows_safety, 0,
            "SAFETY VIOLATION: {archived_rows_safety} row(s) landed in facts_archive after the \
             real-ingest archive sweep, but none should have been archived"
        );

        // o11y cross-check: the counter must AGREE with the op — 0 both sides.
        let counter = sum_archived_counter(&snapshotter);
        assert_eq!(
            counter, 0,
            "facts_archived counter ({counter}) must equal the op's archive count (0) on the \
             SAFETY arm"
        );
    }

    // ── ARM 2 (SECONDARY — trigger path): plant ONE fact whose expired_at is 100
    //    days ago (past the 90-day grace) onto the SAME real graph via the real
    //    facts-table SQL, re-run archive, assert it archives EXACTLY that one fact
    //    and nothing else. ─────────────────────────────────────────────────────
    let now = Utc::now();
    let planted_valid_from = (now - ChronoDuration::days(200)).to_rfc3339();
    // expired_at 100 days ago → past the 90-day grace by a clear margin.
    let planted_expired_at = (now - ChronoDuration::days(100)).to_rfc3339();
    let planted_recorded_at = now.to_rfc3339();

    // A dedicated deterministic subject entity for the composite FK — coupling to a
    // nondeterministically-extracted entity would break replay stability.
    let planted_subject = "arc-planted-subject";
    tg.insert_entity_with_group(kremory::core::graph::InsertEntityWithGroupParams {
        id: planted_subject,
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some(&group_id),
    })
    .await
    .expect("plant subject entity for the trigger-arm fact");

    // A LIVE anchor fact on the SAME subject so archiving the expired fact does NOT
    // strand the subject (ref-count guard P2.2 / RISK-007 would otherwise KEEP it).
    // The live anchor stays in `facts` throughout and must never be archived.
    tg.conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
              expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, NULL, NULL, ?6, ?7, 1.0, 0)",
            libsql::params![
                planted_subject,
                "is_currently",
                "active",
                planted_valid_from.clone(),
                planted_recorded_at.clone(),
                group_id.clone(),
                group_id.clone(),
            ],
        )
        .await
        .expect("plant live anchor fact");

    // The long-expired fact to be archived. is_dream_generated = 0. Its facts_fts
    // shadow row is inserted too so FTS-shadow deletion (RISK-003) is observable.
    tg.conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
              expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, NULL, ?7, ?8, 1.0, 0)",
            libsql::params![
                planted_subject,
                "held_role",
                "interim CEO",
                planted_valid_from,
                planted_recorded_at,
                planted_expired_at.clone(),
                group_id.clone(),
                group_id.clone(),
            ],
        )
        .await
        .expect("plant long-expired fact");

    let mut id_rows = tg
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    let planted_fact_id: i64 = id_rows
        .next()
        .await
        .expect("row")
        .expect("present")
        .get(0)
        .expect("id");
    drop(id_rows);

    // FTS shadow row for the planted expired fact (mirrors real ingest,
    // `facts.rs:370-377`) so its deletion during archive is observable.
    tg.conn
        .execute(
            "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
            libsql::params![planted_fact_id, "held_role", "interim CEO"],
        )
        .await
        .expect("plant fts shadow for the expired fact");
    assert_eq!(
        fts_shadow_count_for(&tg, planted_fact_id).await,
        1,
        "planted fact's fts shadow row must exist before the sweep"
    );

    // Two facts were planted (live anchor + expired). Live count is now
    // live_before + 2; nothing archived yet.
    let live_after_plant = count_live_facts(&tg, &group_id).await;
    assert_eq!(
        live_after_plant,
        live_before + 2,
        "two facts planted (live anchor + expired); live count must be live_before + 2"
    );

    let recorder2 = DebuggingRecorder::new();
    let snapshotter2 = recorder2.snapshotter();
    let archived_trigger = {
        let _guard = metrics::set_default_local_recorder(&recorder2);
        run_archive(&tg, &group_id).await
    };

    // The op must archive EXACTLY the one planted expired fact.
    assert_eq!(
        archived_trigger, 1,
        "trigger arm: archive must move EXACTLY the one planted long-expired fact; op reported \
         {archived_trigger}"
    );

    // The planted expired fact is GONE from live `facts`.
    assert!(
        !fact_row_exists(&tg, planted_fact_id).await,
        "trigger arm: the archived fact must be gone from live `facts`"
    );
    // …and PRESENT in facts_archive with archived_at stamped.
    assert!(
        archive_row_exists(&tg, planted_fact_id).await,
        "trigger arm: the archived fact must be present in facts_archive"
    );
    let archived_at = read_archived_at(&tg, planted_fact_id).await;
    assert!(
        archived_at
            .as_deref()
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "trigger arm: archived_at must be stamped on the archive row; got {archived_at:?}"
    );
    // Its facts_fts shadow row is GONE (RISK-003).
    let fts_shadow_gone = fts_shadow_count_for(&tg, planted_fact_id).await == 0;
    assert!(
        fts_shadow_gone,
        "trigger arm: the archived fact's facts_fts shadow row must be removed (RISK-003)"
    );

    // Exactly ONE row in facts_archive now (the planted one); the real open-ended
    // facts + the live anchor are untouched — live count dropped by exactly 1.
    assert_eq!(
        count_archived_rows(&tg, &group_id).await,
        1,
        "trigger arm: exactly ONE row must be in facts_archive after planting + sweeping"
    );
    let live_after_trigger = count_live_facts(&tg, &group_id).await;
    assert_eq!(
        live_after_trigger,
        live_before + 1,
        "trigger arm: live count must drop by exactly 1 (the expired fact archived); the \
         {live_before} real fact(s) + the live anchor remain. Got {live_after_trigger}"
    );

    // o11y cross-check on the trigger arm: counter == op count == 1.
    let counter2 = sum_archived_counter(&snapshotter2);
    assert_eq!(
        counter2, 1,
        "facts_archived counter ({counter2}) must equal the trigger-arm archive count (1)"
    );

    // ── ARM 3 (IDEMPOTENCY, P2.4): a second sweep archives 0 more (the moved fact
    //    is gone from `facts`, so the candidate SELECT self-excludes it). ─────────
    let recorder3 = DebuggingRecorder::new();
    let snapshotter3 = recorder3.snapshotter();
    let archived_second = {
        let _guard = metrics::set_default_local_recorder(&recorder3);
        run_archive(&tg, &group_id).await
    };
    assert_eq!(
        archived_second, 0,
        "idempotency: a second archive sweep must move 0 more (already-archived fact is gone \
         from `facts`); got {archived_second}"
    );
    assert_eq!(
        count_archived_rows(&tg, &group_id).await,
        1,
        "idempotency: no double-archive — facts_archive still holds exactly 1 row"
    );
    let counter3 = sum_archived_counter(&snapshotter3);
    assert_eq!(
        counter3, 0,
        "idempotency: facts_archived counter must be 0 on the second sweep; got {counter3}"
    );

    eprintln!(
        "[archive-real-llm] PASS — SAFETY(real fresh facts)=0 archived, \
         TRIGGER(planted long-expired)=1 archived (fts shadow gone), idempotent second sweep=0."
    );

    drop(dir);
}

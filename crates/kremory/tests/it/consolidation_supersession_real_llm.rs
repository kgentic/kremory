//! ADR-066 dream CONSOLIDATION P1 — supersession L3 "isolated real-LLM" test.
//!
//! # Pyramid tier
//! This is the tier ABOVE the deterministic corpus harness
//! (`consolidation_supersession_test.rs`, hand-planted fixtures, zero LLM) and
//! BELOW the full dream e2e (`whole_project_e2e.rs`, real ingest → `mem.dream()`
//! runs ALL 5 reconciliation passes + consolidation). It proves the supersession
//! window-closeout op behaves correctly when run IN ISOLATION over a graph
//! produced by REAL-model ingest (`mem.remember()` → phase-1 embed+NER → phase-2
//! LLM relationships), catching real-data-shape bugs the hand-planted corpus
//! cannot. Only the ONE op under test runs against that real graph — nothing else
//! from the dream phase fires — so a failure localises to supersession, not to a
//! reconciliation pass upstream.
//!
//! # KEY FINDING (verified 2026-07-03 — the reason the SAFETY arm is primary)
//! The deterministic window-closeout lane retires facts where
//! `valid_to IS NOT NULL AND valid_to < now AND expired_at IS NULL AND
//!  invalid_at IS NULL AND is_dream_generated = 0` (`supersession.rs::window_closeout`).
//! But the public `remember()` / LLM-extraction path does NOT derive bounded
//! validity: `intelligence.rs::make_fact` sets `valid_to: None` for every
//! extracted fact (verified — it is a production helper, not a `#[cfg(test)]`
//! builder). So real-model ingest produces OPEN-ENDED facts (`valid_to IS NULL`),
//! which the window-closeout SELECT excludes by its first predicate clause.
//!
//! Therefore this test has TWO arms:
//!   1. **PRIMARY — SAFETY property.** Over the real-model-ingested graph
//!      (all facts open-ended), supersession must retire EXACTLY 0 facts —
//!      nothing wrongly retired. This is the load-bearing arm: it proves the op
//!      does not false-retire on the fact shape production actually emits.
//!   2. **SECONDARY — trigger path.** Plant ONE fact with `valid_to` in the past
//!      (+ `expired_at` NULL) via the real graph fact-insert API onto the SAME
//!      real-ingested graph, re-run supersession, and assert it retires EXACTLY
//!      that one fact (op count == 1, its `expired_at` now set to `valid_to`) and
//!      touches no other fact. This proves the trigger path fires end-to-end on a
//!      real graph, not only on a synthetic fixture.
//!
//! A tech-debt item tracks temporal-extraction populating `valid_to` from the
//! episode text (so real ingest can itself produce closed-window facts); until
//! then the window-closeout lane's trigger is only reachable via the planted-fact
//! arm here / the deterministic corpus. See the TD note at the SAFETY assertion.
//!
//! # Two run modes (mirrors whole_project_e2e §17), by `KREMORY_VCR`:
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

use kremory::core::dream::{supersession, ConsolidationBudget, SupersessionParams};
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

use crate::support::test_log::init_test_log;

// ── Mode selection (mirrors whole_project_e2e) ────────────────────────────────

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
        .join("consolidation_supersession_real_llm.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("consolidation_supersession_real_llm.embeddings.json")
}

// ── Embedding record/replay (TD-093), copied from whole_project_e2e.rs ─────────
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

/// Sum the window-closeout supersession counter across the snapshot (mirrors
/// `consolidation_supersession_test.rs::sum_window_closeout_counter`).
fn sum_window_closeout_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.supersessions_recorded_total" {
                return None;
            }
            let labels: std::collections::HashMap<&str, &str> =
                key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("lane").copied() != Some("window_closeout") {
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

async fn count_open_ended_facts(graph: &TemporalGraph, group_id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts WHERE group_id = ?1 AND valid_to IS NULL",
            libsql::params![group_id],
        )
        .await
        .expect("open-ended fact count query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

/// Count facts in the group whose `expired_at` is set (i.e. retired). The
/// SAFETY arm asserts this stays 0 across the real ingest sweep; the trigger arm
/// asserts it becomes exactly 1.
async fn count_retired_facts(graph: &TemporalGraph, group_id: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts WHERE group_id = ?1 AND expired_at IS NOT NULL",
            libsql::params![group_id],
        )
        .await
        .expect("retired fact count query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

async fn read_expired_at(graph: &TemporalGraph, fact_id: i64) -> Option<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT expired_at FROM facts WHERE id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query expired_at");
    let row = rows.next().await.expect("row").expect("present");
    row.get::<Option<String>>(0).expect("expired_at col")
}

// ── Ingest corpus (distinct, non-overlapping domains — no contradiction call) ──
const EPISODES: &[(&str, &str)] = &[
    (
        "sup-e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "sup-e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
];

/// Run ONLY the supersession op over `group_id` and return its reported count.
/// Nothing else from the dream phase runs — this is the isolation the L3 tier
/// provides. `include_llm_nominate: false` (the LLM value-change lane is stub-off
/// this phase and irrelevant to the deterministic window-closeout property here).
async fn run_supersession(graph: &TemporalGraph, group_id: &str) -> usize {
    let mut budget = ConsolidationBudget::new(None, None);
    let report = supersession(SupersessionParams {
        graph,
        group_id,
        budget: &mut budget,
        include_llm_nominate: false,
        model_id: &ollama_chat_model(),
    })
    .await
    .expect("supersession op must succeed");
    report.count
}

// ── The isolated real-LLM test ────────────────────────────────────────────────

#[tokio::test]
#[ignore = "consolidation_supersession_real_llm: requires Ollama in record mode, or the committed \
            cassette in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test it consolidation_supersession_real_llm:: -- \
            --ignored --nocapture"]
async fn supersession_isolated_over_real_ingest() {
    let _log = init_test_log("consolidation_supersession_real_llm");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) VCR-backed chat + embedder so replay faithfully reproduces the recorded
    //    real-model ingest offline (real nomic vectors → phase-1 embed works
    //    deterministically). Mirrors whole_project_e2e.
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
    let namespace = Namespace::new("supersession-real-llm");
    let mem = Memory::open(dir.path().join("supersession_real_llm.db"))
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

    // The exact group `mem.dream()`/supersession operates on for this namespace.
    let group_id = mem.group_id_for_test(&namespace);
    let tg: Arc<TemporalGraph> = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // Real ingest must have produced SOME facts, and (the load-bearing finding)
    // they must all be OPEN-ENDED — extraction sets `valid_to = None`
    // (intelligence.rs::make_fact). If a future temporal-extraction TD lands and
    // real ingest starts emitting bounded `valid_to`, THIS assertion breaks LOUDLY
    // and the SAFETY arm below must be re-derived — that is intended.
    let open_ended = count_open_ended_facts(&tg, &group_id).await;
    let retired_before = count_retired_facts(&tg, &group_id).await;
    eprintln!(
        "[supersession-real-llm] group_id={group_id} open_ended_facts={open_ended} \
         retired_facts_before={retired_before}"
    );
    assert!(
        open_ended >= 1,
        "real-model ingest must produce >= 1 fact; got {open_ended} open-ended (extraction \
         emits facts with valid_to = None). If 0, phase-2 relationship extraction silently \
         produced no facts — a real ingest regression, not a supersession property."
    );
    assert_eq!(
        retired_before, 0,
        "no fact should already be retired straight out of ingest (expired_at NULL by \
         construction); got {retired_before}"
    );

    // ── ARM 1 (PRIMARY — SAFETY): supersession over real open-ended facts retires
    //    EXACTLY 0. Nothing wrongly retired. ─────────────────────────────────────
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    {
        let _guard = metrics::set_default_local_recorder(&recorder);

        let retired = run_supersession(&tg, &group_id).await;

        // SAFETY FINDING (see module doc + KEY FINDING): the window-closeout lane
        // retires facts with `valid_to < now`, but the public remember()/extraction
        // path emits `valid_to = None` (intelligence.rs::make_fact — production
        // helper, verified 2026-07-03). So over a REAL-model-ingested graph the
        // sweep MUST touch nothing. A TD tracks temporal-extraction populating
        // valid_to from episode text — until then real ingest cannot itself produce
        // a closed window, and this SAFETY arm is the primary property this L3 test
        // guards (0 false-retire on the fact shape production actually emits).
        assert_eq!(
            retired, 0,
            "SAFETY VIOLATION: supersession retired {retired} fact(s) over a real-model-ingested \
             graph whose facts are ALL open-ended (valid_to = None). The window-closeout lane must \
             touch nothing when no window has closed."
        );
        let retired_after_safety = count_retired_facts(&tg, &group_id).await;
        assert_eq!(
            retired_after_safety, 0,
            "SAFETY VIOLATION: {retired_after_safety} fact(s) have expired_at set after the \
             real-ingest supersession sweep, but none should have been retired"
        );

        // o11y cross-check: the counter must AGREE with the op — 0 both sides.
        let counter = sum_window_closeout_counter(&snapshotter);
        assert_eq!(
            counter, 0,
            "window_closeout counter ({counter}) must equal the op's retirement count (0) on the \
             SAFETY arm"
        );
    }

    // ── ARM 2 (SECONDARY — trigger path): plant ONE bounded-validity fact
    //    (valid_to in the PAST, expired_at NULL) onto the SAME real graph via the
    //    real fact-insert SQL, re-run supersession, assert it retires EXACTLY that
    //    one fact and nothing else. ────────────────────────────────────────────
    let now = Utc::now();
    let planted_valid_from = (now - ChronoDuration::days(30)).to_rfc3339();
    // valid_to 10 days ago → the window is demonstrably closed.
    let planted_valid_to = (now - ChronoDuration::days(10)).to_rfc3339();
    let planted_recorded_at = now.to_rfc3339();

    // Subject entity must exist for the composite FK; use one that real ingest is
    // guaranteed to have produced would couple to nondeterministic extraction, so
    // plant a dedicated deterministic subject via the real graph API instead.
    let planted_subject = "sup-planted-subject";
    tg.insert_entity_with_group(kremory::core::graph::InsertEntityWithGroupParams {
        id: planted_subject,
        entity_type_id: 0,
        properties: serde_json::json!({}),
        group_id: Some(&group_id),
    })
    .await
    .expect("plant subject entity for the trigger-arm fact");

    // Insert the bounded-validity fact via the real facts-table SQL (mirrors
    // consolidation_supersession_test.rs::run_one_row's plant). is_dream_generated
    // = 0 so the F-4 anti-loop guard does not exclude it.
    tg.conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
              expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, ?7, ?8, 1.0, 0)",
            libsql::params![
                planted_subject,
                "held_role",
                "interim CEO",
                planted_valid_from,
                planted_valid_to.clone(),
                planted_recorded_at,
                group_id.clone(),
                group_id.clone(),
            ],
        )
        .await
        .expect("plant bounded-validity fact");

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

    let recorder2 = DebuggingRecorder::new();
    let snapshotter2 = recorder2.snapshotter();
    let retired_trigger = {
        let _guard = metrics::set_default_local_recorder(&recorder2);
        run_supersession(&tg, &group_id).await
    };

    // The op must retire EXACTLY the one planted fact.
    assert_eq!(
        retired_trigger, 1,
        "trigger arm: supersession must retire EXACTLY the one planted closed-window fact; \
         op reported {retired_trigger}"
    );

    // The planted fact's expired_at is now set to its valid_to (the demonstrated
    // close-out time — NOT `now`).
    let planted_expired_at = read_expired_at(&tg, planted_fact_id).await;
    assert_eq!(
        planted_expired_at.as_deref(),
        Some(planted_valid_to.as_str()),
        "trigger arm: the planted fact's expired_at must be set to its valid_to \
         ({planted_valid_to}); got {planted_expired_at:?}"
    );

    // No OTHER fact was touched — exactly ONE retired fact exists in the group now
    // (the planted one). The real open-ended facts must remain open-ended.
    let retired_after_trigger = count_retired_facts(&tg, &group_id).await;
    assert_eq!(
        retired_after_trigger, 1,
        "trigger arm: exactly ONE fact must be retired after planting + sweeping (the planted \
         one); got {retired_after_trigger} — a real open-ended fact was wrongly retired"
    );
    let open_ended_after = count_open_ended_facts(&tg, &group_id).await;
    assert_eq!(
        open_ended_after, open_ended,
        "trigger arm: the {open_ended} real open-ended fact(s) must be UNTOUCHED by the sweep; \
         open-ended count changed to {open_ended_after}"
    );

    // o11y cross-check on the trigger arm: counter == op count == 1.
    let counter2 = sum_window_closeout_counter(&snapshotter2);
    assert_eq!(
        counter2, 1,
        "window_closeout counter ({counter2}) must equal the trigger-arm retirement count (1)"
    );

    // Idempotency: a second sweep retires 0 more (the planted fact now has
    // expired_at set, so the SELECT excludes it).
    let recorder3 = DebuggingRecorder::new();
    let snapshotter3 = recorder3.snapshotter();
    let retired_second = {
        let _guard = metrics::set_default_local_recorder(&recorder3);
        run_supersession(&tg, &group_id).await
    };
    assert_eq!(
        retired_second, 0,
        "idempotency: a second supersession sweep must retire 0 more (already-retired fact is \
         excluded by the expired_at IS NULL predicate); got {retired_second}"
    );
    let counter3 = sum_window_closeout_counter(&snapshotter3);
    assert_eq!(
        counter3, 0,
        "idempotency: window_closeout counter must be 0 on the second sweep; got {counter3}"
    );

    eprintln!(
        "[supersession-real-llm] PASS — SAFETY(real open-ended)=0 retired, \
         TRIGGER(planted closed window)=1 retired, idempotent second sweep=0."
    );

    drop(dir);
}

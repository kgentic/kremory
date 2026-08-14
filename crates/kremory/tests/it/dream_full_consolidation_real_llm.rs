//! ADR-066 dream CONSOLIDATION — full `mem.dream()` system-level "all ops ON"
//! real-LLM test.
//!
//! # Pyramid tier
//! This is the DEPTH/integration handful ABOVE the per-op "isolated real-LLM"
//! tier (`consolidation_supersession_real_llm.rs`, `consolidation_archive_real_llm.rs`,
//! `consolidation_cross_episode_real_llm.rs`, `consolidation_communities_real_llm.rs`
//! — each proves ONE op in isolation) and mirrors `dream_e2e_real_llm.rs`'s role for
//! the reconciliation chain: it drives the PUBLIC `mem.dream()` entry point with
//! EVERY `DreamOpts.include_*` consolidation flag flipped ON at once, over a graph
//! built by real-model ingest, and asserts the whole system composes correctly —
//! order of ops (supersession → archive → cross_episode → communities, ADR-066 §2.6),
//! the RISK-001 homonym guard holding end-to-end, full-run idempotency, and the
//! bi-temporal re-assertion lifecycle (an explicit product question: "what happens
//! if a fact expires and a LATER episode re-asserts it?"). Per-op correctness is
//! NOT re-proven here — that's the isolated tier's job; this file does NOT
//! re-enumerate per-op variants.
//!
//! # Two run modes (mirrors the P1-P4 L3 siblings + `dream_e2e_real_llm.rs`), by
//! `KREMORY_VCR`:
//!   * `KREMORY_VCR=record` → LIVE: real gemma4:e4b chat (`.think(false)`) wrapped in
//!     `RecordReplayChatProvider::record(...)` AND real nomic-embed-text wrapped in
//!     `RecordReplayEmbedder::record(...)`; one run refreshes the cassette pair for
//!     the case being (re-)recorded. `provider.flush()` + `emb_vcr.flush()` are
//!     MANDATORY after the last background write and before assertions.
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: fully deterministic, NO Ollama. Both
//!     cassettes are replayed; a missing entry is a LOUD error.
//!
//! Each case gets its OWN cassette pair (own Memory, own namespace, own tempdir) so
//! the three cases are independently re-recordable and replay-order-independent.
//! (A fourth case — the bi-temporal re-assertion lifecycle — was relocated to a
//! DETERMINISTIC fast-tier test: it exercises deterministic substrate behaviour that
//! does not depend on model output, so a stochastic real-LLM VCR test is the wrong
//! tier for it per the llm-test-pyramid rule; testing it correctly also needs a
//! substrate dedup-vs-expiry investigation. Original fn saved at
//! .context/reassertion-lifecycle-original-real-llm-fn.rs.txt for the rebuild.)
//!
//! # TD-094 note
//! `mem.dream()` (unlike the per-op isolated tests, which call the op function
//! directly) requires `.with_model_id(...)` threaded on the builder — otherwise the
//! LLM-driven reconciliation lanes silently degrade to the empty-model → PromptOnly
//! arm (TD-094, fixed 2026-07-01). All three cases below wire it.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
// Test files use expect/unwrap/panic as intentional assertion mechanisms
// (project-wide test convention — see golden_path_smoke.rs / whole_project_e2e.rs).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;

use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::schema::TemporalGraph;
use kremory::memory::types::DreamOpts;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use crate::support::test_log::init_test_log;

// ── Mode selection (mirrors the P1-P4 L3 siblings) ─────────────────────────────

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

fn cassette_path(case: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("dream_full_consolidation_real_llm.{case}.json"))
}

fn embedding_cassette_path(case: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!(
            "dream_full_consolidation_real_llm.{case}.embeddings.json"
        ))
}

// ── Embedding record/replay (TD-093), copied from the P1-P4 L3 siblings ────────
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
    /// Record mode SEEDS from the existing cassette so successive recordings
    /// ACCUMULATE rather than overwrite (TD-221, 2026-08-14).
    ///
    /// # Why this is not merely a convenience
    /// Two dream passes (`consistency_check`, `type_registry_collapse`) embed
    /// text built from GRAPH STATE — e.g. `"acme corporation | Jane Smith | | "`.
    /// That state is produced by the passes that ran before them, and the
    /// record-mode and replay-mode chains do not land in byte-identical states
    /// (dream is non-deterministic across processes — TD-183; `facts_archived`
    /// was 3 under record and 1 under replay on 2026-08-14). So a cassette
    /// captured on the RECORD path is missing exactly the texts the REPLAY path
    /// asks for, and the two passes died on a MISS **on the very run that had
    /// just recorded them** — the cassette could never converge, because each
    /// `flush()` discarded the other path's texts.
    ///
    /// Seeding makes the cassette hold the UNION over recording runs, so one
    /// record → replay cycle closes the gap and it stays closed.
    ///
    /// Trade-off, stated rather than hidden: entries for texts no longer
    /// requested are never evicted, so the file only grows. That is the right
    /// side to err on — a stale entry is inert, whereas a missing one kills a
    /// pass and (before TD-221's `warnings.is_empty()` assertion) did so
    /// silently.
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        let seed: std::collections::HashMap<String, Vec<f32>> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        eprintln!(
            "[vcr] embedding cassette seeded with {} existing entr(ies) from {}",
            seed.len(),
            path.display()
        );
        Self {
            inner: Some(inner),
            cache: Mutex::new(seed),
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
    // (facade default + metrics harness). Same model as the P1-P4 L3 siblings +
    // dream_e2e_real_llm.
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

/// Build a VCR-backed `(chat, embedder)` pair for `case`, matching the mode. Each
/// case owns its own cassette pair so cases are independently re-recordable.
async fn build_vcr_pair(
    mode: &Mode,
    case: &str,
) -> (Arc<RecordReplayChatProvider>, Arc<RecordReplayEmbedder>) {
    match mode {
        Mode::Live => {
            let rec = Arc::new(RecordReplayChatProvider::record(
                real_ollama_chat(),
                cassette_path(case),
                ollama_chat_model(),
            ));
            let emb = Arc::new(RecordReplayEmbedder::record(
                real_ollama_embedder(),
                embedding_cassette_path(case),
            ));
            (rec, emb)
        }
        Mode::Replay => {
            let rep = Arc::new(
                RecordReplayChatProvider::replay(cassette_path(case)).unwrap_or_else(|e| {
                    panic!(
                        "replay cassette for case {case:?} must load ({e}) — record it via \
                         KREMORY_VCR=record"
                    )
                }),
            );
            let emb = Arc::new(RecordReplayEmbedder::replay(embedding_cassette_path(case)));
            (rep, emb)
        }
    }
}

/// `DreamOpts` with EVERY consolidation op flipped ON (the surface this file
/// exists to prove). Reconciliation-pass flags are left at their validated
/// defaults (`DreamOpts::default()`); only the four ADR-066 CONSOLIDATION
/// `include_*` flags + `archive_grace_days` are overridden.
///
/// `archive_grace_days: Some(0)` — the default 90-day grace window cannot be
/// exercised by a test corpus that plants facts "expired 10 days ago" (still
/// inside a 90-day grace); 0 makes any already-expired fact immediately
/// archival-eligible so the archive op has honest work to do within a single
/// dream call, matching the isolated `consolidation_archive_real_llm.rs` trigger
/// arm's planting convention.
fn all_consolidation_on() -> DreamOpts {
    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut o = DreamOpts::default();
    o.include_community_detection = true;
    o.include_cross_episode_merges = true;
    // dry_run=false → exercise REAL fusion (this real-LLM harness asserts merges +
    // idempotency, which require the destructive write to actually commit, not shadow).
    o.cross_episode_dry_run = false;
    o.include_supersession_sweep = true;
    o.include_supersession_llm_nominate = false;
    o.include_fact_archival = true;
    o.archive_grace_days = Some(0);
    o
}

// ── Direct-SQL helpers (targeted SELECTs, not the API under test) ─────────────

async fn scalar(rows: &mut libsql::Rows) -> i64 {
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

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

/// Every graph mutation recorded this run, as `kind @ created_at :: inputs`
/// (TD-221). Attached to the vanished-entity assertion so a failure says WHICH
/// pass consumed the entity, not merely that one disappeared — `inputs` carries
/// the merge site and the loser/keeper ids.
async fn mutation_log(graph: &TemporalGraph, group_id: &str) -> Vec<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT kind, created_at, inputs FROM graph_mutation_log \
             WHERE group_id = ?1 ORDER BY id",
            libsql::params![group_id],
        )
        .await
        .expect("graph_mutation_log query");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("mutation log row iteration") {
        let kind: String = row.get(0).expect("kind");
        let created_at: String = row.get(1).expect("created_at");
        let inputs: Option<String> = row.get(2).expect("inputs");
        // Truncated: `pre_state` snapshots can be tens of KB and the identifying
        // fields (site, loser, keeper) are at the front.
        let inputs = inputs.unwrap_or_else(|| "<null>".to_string());
        let inputs: String = inputs.chars().take(400).collect();
        out.push(format!("  {kind} @ {created_at} :: {inputs}"));
    }
    out
}

/// The typed entity IDS, sorted (TD-221). The count alone cannot say WHICH
/// entity a consolidation pass consumed, and "one fewer entity" is the
/// difference between a correct merge of two surface forms and a destroyed
/// distinct entity. Same lesson as TD-219's archive assertion: name it.
/// Renders as `id(type_id)` — TD-222 needs the TYPE, not just the name. The
/// type veto only fires when two entities are POSITIVELY KNOWN to be different
/// kinds of thing, so "did it fire?" is unanswerable without seeing what types
/// the extractor actually assigned. Printing the id alone sent one debugging
/// round-trip to the logs and back for information the assertion already had in
/// hand.
async fn typed_entity_ids(graph: &TemporalGraph, group_id: &str) -> Vec<String> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, entity_type_id FROM entities \
             WHERE group_id = ?1 AND entity_type_id != 0 ORDER BY id",
            libsql::params![group_id],
        )
        .await
        .expect("typed entity id query");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("typed entity row iteration") {
        let id: String = row.get(0).expect("id");
        let type_id: u32 = row.get(1).expect("entity_type_id");
        out.push(format!("{id}({type_id})"));
    }
    out
}

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

// ── Plant helpers (real graph API, mirrors consolidation_cross_episode_real_llm) ──

/// One archived fact, with its endpoints kept SEPARATE so an invariant can be
/// asserted over them (TD-219) rather than over a rendered string.
struct ArchivedFact {
    subject: String,
    object: String,
    rendered: String,
}

impl ArchivedFact {
    /// A self-loop (`subject == object`) carries no information — it is what a
    /// merge leaves behind when it collapses the two endpoints of a relation
    /// onto one entity. Retiring these is CLEANUP. Retiring a fact between two
    /// DISTINCT entities on this fixture would not be.
    fn is_self_loop(&self) -> bool {
        self.subject == self.object
    }
}

/// Every row currently in `facts_archive` for `group_id`, with endpoints split
/// out and a readable `id: subject --predicate--> object` rendering (TD-219).
///
/// Exists so a failing archive assertion NAMES what was retired. The previous
/// `facts_archived == 0` assertion could only ever report a number, which meant
/// answering "was that legitimate?" required reproducing a stochastic real-LLM
/// run and hoping it archived the same things — and it did not (2 on one run,
/// 0 on the next).
///
/// Projects `archived_at` rather than `COUNT(*)`: `facts_archive` mirrors the
/// vector-indexed `facts` table, where a bare `COUNT(*)` returns 0 under this
/// driver even when rows exist (the libsql vector-index trap, SYSTEM-PRIMER
/// gotcha #1).
async fn archived_fact_triples(graph: &TemporalGraph, group_id: &str) -> Vec<ArchivedFact> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, subject_id, predicate, object_id, object_value, expired_at, archived_at \
             FROM facts_archive WHERE group_id = ?1 ORDER BY id",
            libsql::params![group_id],
        )
        .await
        .expect("facts_archive query must succeed");

    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("facts_archive row iteration") {
        let id: i64 = row.get(0).expect("id");
        let subject: String = row.get(1).expect("subject_id");
        let predicate: String = row.get(2).expect("predicate");
        let object_id: Option<String> = row.get(3).expect("object_id");
        let object_value: Option<String> = row.get(4).expect("object_value");
        let expired_at: Option<String> = row.get(5).expect("expired_at");
        let archived_at: Option<String> = row.get(6).expect("archived_at");
        let object = object_id
            .or(object_value)
            .unwrap_or_else(|| "<none>".to_string());
        let rendered = format!(
            "  fact {id}: {subject} --{predicate}--> {object}  (expired_at={}, archived_at={})",
            expired_at.unwrap_or_else(|| "<null>".to_string()),
            archived_at.unwrap_or_else(|| "<null>".to_string()),
        );
        out.push(ArchivedFact {
            subject,
            object,
            rendered,
        });
    }
    out
}

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

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption (mirrors L3 siblings)
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

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption (mirrors L3 siblings)
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

// ── Ingest corpus (distinct, non-overlapping domains — avoids the ingest-time
//    contradiction-detection LLM call, per the P1-P4 L3 siblings' determinism
//    note; multi-domain also gives `communities` a disjoint-clique graph to
//    partition and `cross_episode`/`archival` real entities to leave untouched) ──
const MULTI_DOMAIN_EPISODES: &[(&str, &str)] = &[
    (
        "fc-e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "fc-e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
    (
        "fc-e2e-scientist",
        "Marie Curie, a physicist born in Warsaw, was awarded Nobel Prizes in both Physics \
         and Chemistry for her research on radioactivity.",
    ),
];

/// TD-181: a FIXED document anchor, so these cassettes can replay at all.
///
/// `core/contradiction.rs:143` renders each existing fact's `valid_from` into
/// the `ContradictionVerdict` prompt as RFC-3339. For LLM-EXTRACTED facts that
/// `valid_from` is the ingest `reference_time`, so with a wall-clock anchor the
/// prompt text — and therefore the VCR fingerprint — differed on every run. The
/// call could never replay: each re-record appended one more entry the next run
/// would not match (observed 14 -> 15 entries, still missing, with the missing
/// fingerprint changing each time).
///
/// Pinning `published_at` fixes the prompt at its SOURCE, which keeps the
/// fingerprint honest. Normalising timestamps out of the fingerprint instead
/// would make cassettes match when prompts genuinely differ — re-creating the
/// exact blindness TD-180 was about.
///
/// NOTE: an earlier attempt at this failed because `published_at` was consulted
/// only for CALLER-SUPPLIED structured facts, never on the extraction path
/// (`memory/engine_handle.rs:190` used `occurred_at` unconditionally). That is
/// fixed in the same change; without it this call is a silent no-op.
///
/// Distinct per episode (base + index minutes) so ingest ORDER is still
/// modelled; one shared instant would flatten it.
const FIXED_ANCHOR_RFC3339: &str = "2026-01-01T00:00:00Z";

fn fixed_anchor(index: i64) -> chrono::DateTime<Utc> {
    chrono::DateTime::parse_from_rfc3339(FIXED_ANCHOR_RFC3339)
        .expect("FIXED_ANCHOR_RFC3339 is a valid RFC-3339 literal")
        .with_timezone(&Utc)
        + chrono::Duration::minutes(index)
}

async fn ingest_multi_domain(mem: &Memory) {
    for (index, (session, text)) in MULTI_DOMAIN_EPISODES.iter().enumerate() {
        let commit = mem
            .remember(*text)
            .published_at(fixed_anchor(index as i64))
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
}

/// Build a fresh `Memory` wired to VCR chat + embedder for `case`, with
/// `.with_model_id(...)` threaded (TD-094 — required for `mem.dream()`'s LLM
/// reconciliation lanes to engage instead of silently PromptOnly-degrading).
/// Returns the `Memory`, its VCR provider/embedder (for the record-mode flush),
/// its namespace, and the tempdir (kept alive by the caller).
async fn build_memory(
    mode: &Mode,
    case: &str,
) -> (
    Memory,
    Arc<RecordReplayChatProvider>,
    Arc<RecordReplayEmbedder>,
    Namespace,
    tempfile::TempDir,
) {
    let (provider, emb_vcr) = build_vcr_pair(mode, case).await;
    let llm: Arc<dyn ChatProvider> = provider.clone();
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();

    let dir = tempfile::tempdir().expect("tempdir");
    let namespace = Namespace::new(format!("full-consolidation-{case}"));
    let mem = Memory::open(dir.path().join(format!("{case}.db")))
        .with_llm(llm)
        .with_model_id(ollama_chat_model())
        .with_embedder(embedder)
        .embedding_dim(768)
        .default_namespace(namespace.clone())
        .await
        .expect("Memory::open must succeed");

    (mem, provider, emb_vcr, namespace, dir)
}

fn flush_if_live(mode: &Mode, provider: &RecordReplayChatProvider, emb_vcr: &RecordReplayEmbedder) {
    if matches!(mode, Mode::Live) {
        provider
            .flush()
            .expect("provider.flush() must succeed in record mode");
        emb_vcr.flush();
    }
}

// ── Case 1: happy_path ──────────────────────────────────────────────────────────

/// Multi-domain real ingest → ONE `mem.dream()` call with ALL consolidation flags
/// ON. Asserts the graph improved where expected (communities formed) and nothing
/// panicked, and that the same call replays byte-for-byte deterministically.
#[tokio::test]
#[ignore = "dream_full_consolidation_real_llm::happy_path: requires Ollama in record mode, or the \
            committed cassette in replay mode. Run explicitly: \
            KREMORY_VCR=record cargo test -p kremory --features test-utils,llm-smoke \
            --test it dream_full_consolidation_real_llm:: -- --ignored --nocapture happy_path"]
async fn happy_path() {
    let _log = init_test_log("dream_full_consolidation_real_llm_happy_path");
    let mode = resolve_mode();
    let case = "happy_path";

    let (mem, provider, emb_vcr, namespace, _dir) = build_memory(&mode, case).await;
    ingest_multi_domain(&mem).await;
    flush_if_live(&mode, &provider, &emb_vcr);

    let group_id = mem.group_id_for_test(&namespace);
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    let typed_before = count_typed_entities(&tg, &group_id).await;
    let ids_before = typed_entity_ids(&tg, &group_id).await;
    eprintln!(
        "[full-consolidation-happy] group_id={group_id} typed_entities={typed_before} ids={ids_before:?}"
    );
    assert!(
        typed_before >= 2,
        "real-model ingest must produce >= 2 typed entities across the three distinct-domain \
         episodes; got {typed_before}"
    );

    // ONE mem.dream() call with EVERY consolidation flag ON — the system-level
    // surface this file exists to prove.
    let summary = mem
        .dream()
        .with_opts(all_consolidation_on())
        .await
        .expect("mem.dream() with all consolidation ops ON must succeed end-to-end");

    eprintln!(
        "[full-consolidation-happy] communities_updated={} cross_episode_merges={} \
         supersessions_recorded={} facts_archived={} warnings={:?}",
        summary.communities_updated,
        summary.cross_episode_would_merge,
        summary.supersessions_recorded,
        summary.facts_archived,
        summary.warnings,
    );

    // TD-221 — a pass that FAILED TO RUN must fail this test. MODE-SCOPED, and
    // the scoping is the finding, not a concession.
    //
    // Until 2026-08-14 this test passed while printing two failures in
    // `summary.warnings` — `consistency_check` and `type_registry_collapse`
    // both died on an embedding cassette MISS — so its "every flag ON, the
    // whole system composes" claim was actually being made about six of eight
    // passes, and nothing distinguished "ran and found nothing" from "never
    // ran".
    //
    // **In LIVE mode the bar is absolute and it is MET**: measured 2026-08-14,
    // `warnings=[]` with all eight passes executing.
    //
    // **In REPLAY it is structurally unreachable, and this was verified rather
    // than assumed.** Those two passes embed text derived from GRAPH STATE
    // (e.g. `"acme corporation | Jane Smith |  | "`), and the record and replay
    // chains do not land in byte-identical states — dream is non-deterministic
    // across processes (TD-183; `facts_archived` measured 3 / 1 / 0 on three
    // consecutive runs of this same case). The replay path therefore requests a
    // text the RECORD path never generates, so **no number of re-recordings can
    // capture it** — confirmed empirically: three record→replay cycles, with
    // record mode seeding from the existing cassette so entries accumulate, and
    // replay still missed on a text record never asks for.
    //
    // So: in replay, tolerate ONLY a cassette MISS, and fail on any other
    // warning — which is what keeps this from degrading back into "collect
    // warnings and pass regardless".
    let non_cassette_warnings: Vec<&String> = summary
        .warnings
        .iter()
        .filter(|w| !w.contains("cassette MISS"))
        .collect();
    match mode {
        Mode::Live => assert!(
            summary.warnings.is_empty(),
            "happy_path (LIVE): every dream pass must RUN. {} pass(es) failed and were \
             swallowed as warnings, so this test's all-flags-on composition claim does not \
             cover them:\n  {}",
            summary.warnings.len(),
            summary.warnings.join("\n  "),
        ),
        Mode::Replay => assert!(
            non_cassette_warnings.is_empty(),
            "happy_path (REPLAY): a pass failed for a reason OTHER than a cassette miss — a \
             cassette gap is a known replay limitation (TD-221), anything else is a real \
             failure:\n  {}",
            non_cassette_warnings
                .iter()
                .map(|w| w.as_str())
                .collect::<Vec<_>>()
                .join("\n  "),
        ),
    }

    // Communities: multi-domain disjoint-clique real ingest reliably forms > 1
    // community (see consolidation_communities_real_llm.rs KEY FINDING). This is
    // the "graph improved" floor for this case.
    let community_count = persisted_community_count(&tg, &group_id).await;
    assert!(
        community_count > 1,
        "happy_path: communities over a 3-distinct-domain real ingest must form MORE THAN ONE \
         community; got {community_count}"
    );
    assert!(
        summary.communities_updated > 0,
        "happy_path: DreamSummary.communities_updated must be > 0 on the first sweep; got {}",
        summary.communities_updated
    );

    // Cross-episode merges + supersessions + archival: real-model ingest of
    // DISTINCT domains produces no recurring name-slugs and only open-ended facts
    // (per consolidation_cross_episode_real_llm.rs / consolidation_supersession_real_llm.rs
    // KEY FINDINGs) — so these are HONEST ZEROS here, not failures. Asserted
    // explicitly (not just omitted) so a future false-merge/false-archive
    // regression fails loudly.
    assert_eq!(
        summary.cross_episode_would_merge, 0,
        "happy_path: cross_episode_merges must be the HONEST ZERO (distinct-domain real ingest \
         has no recurring name-slug across episodes); got {}",
        summary.cross_episode_would_merge
    );
    assert_eq!(
        summary.supersessions_recorded, 0,
        "happy_path: supersessions_recorded must be the HONEST ZERO (real-model extraction emits \
         open-ended facts, valid_to = None); got {}",
        summary.supersessions_recorded
    );
    // TD-219 — an INVARIANT over WHAT was archived, not a count of HOW MANY.
    //
    // This assertion was `facts_archived == 0`. It reported **2** earlier on
    // 2026-08-13, **0** during that day's re-record, and **3** on the
    // 2026-08-14 re-record — so the number varies run to run and any cassette
    // freezes whichever sample it caught. A count has a second defect on top of
    // that: `== 3` passes when the WRONG three are archived, which is exactly
    // the TD-167 set-valued destruction this tripwire exists to catch (this
    // fixture contains *"flows over six thousand kilometres through Brazil and
    // Peru"* — two facts true SIMULTANEOUSLY).
    //
    // THE INVARIANT: every archived fact must be a SELF-LOOP.
    //
    // Mechanism, so the next reader need not re-derive it. Archive never
    // decides what to EXPIRE — it only relocates rows that are already
    // `expired_at IS NOT NULL AND expired_at < now - grace_days`
    // (`consolidation/archive.rs:93`), and `all_consolidation_on()` sets
    // `archive_grace_days = Some(0)` (production default 90), so anything
    // expired during THIS run is instantly eligible. `supersessions_recorded`
    // is asserted 0 above, so the expiry came from a MERGE — and a merge that
    // collapses the two endpoints of a relation onto one entity leaves
    // `x --rel--> x` behind. A self-referential relation carries no
    // information; retiring it is cleanup.
    //
    // Retiring a fact between two DISTINCT entities on a fixture of three
    // deliberately disjoint domains would be a different animal entirely, and
    // that is what this now catches — at any count, including the counts a
    // frozen `== N` would have waved through.
    //
    // OBSERVED (2026-08-14 live re-record, printed by this very helper):
    //   fact 7:  marie curie --research_area--> marie curie
    //   fact 9:  marie curie --research_area--> marie curie
    //   fact 10: marie curie --award_for-->     marie curie
    // — all self-loops, confirming the mechanism above against real output
    // rather than inference.
    let archived_rows = archived_fact_triples(&tg, &group_id).await;
    let cross_entity: Vec<&ArchivedFact> = archived_rows
        .iter()
        .filter(|f| !f.is_self_loop())
        .collect();
    assert!(
        cross_entity.is_empty(),
        "happy_path: archival retired {} fact(s) BETWEEN DISTINCT ENTITIES on a fixture of three \
         disjoint domains. Self-loop cleanup after a merge is expected; this is not — check each \
         against TD-167 (set-valued facts that are true SIMULTANEOUSLY must NOT retire one \
         another):\n{}\n(all {} archived fact(s) this run:\n{})",
        cross_entity.len(),
        cross_entity
            .iter()
            .map(|f| f.rendered.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        archived_rows.len(),
        archived_rows
            .iter()
            .map(|f| f.rendered.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    );
    // Cross-check the reported count against the table it claims to describe —
    // a summary field that disagrees with the durable state is its own bug, and
    // asserting only one of them cannot see it.
    assert_eq!(
        summary.facts_archived,
        archived_rows.len(),
        "happy_path: DreamSummary.facts_archived ({}) disagrees with the facts_archive table \
         ({} rows) — the counter is lying about a persisted mutation",
        summary.facts_archived,
        archived_rows.len(),
    );

    // No entity was spuriously dropped.
    let typed_after = count_typed_entities(&tg, &group_id).await;
    let ids_after = typed_entity_ids(&tg, &group_id).await;
    let vanished: Vec<&String> = ids_before
        .iter()
        .filter(|id| !ids_after.contains(id))
        .collect();
    if typed_after < typed_before {
        let log = mutation_log(&tg, &group_id).await;
        panic!(
            "happy_path: typed entity count SHRANK from consolidation; before={typed_before} \
             after={typed_after}. Vanished: {vanished:?}\n  before={ids_before:?}\n  \
             after={ids_after:?}\nmutation log ({} rows):\n{}",
            log.len(),
            log.join("\n"),
        );
    }

    // ── Determinism: re-run dream on the SAME already-consolidated graph and
    //    confirm the persisted community partition is unchanged (the deterministic
    //    label-propagation op re-converges to the identical result). ────────────
    let summary2 = mem
        .dream()
        .with_opts(all_consolidation_on())
        .await
        .expect("second mem.dream() call must also succeed");
    let community_count2 = persisted_community_count(&tg, &group_id).await;
    assert_eq!(
        community_count2, community_count,
        "happy_path determinism: community count must be unchanged across a second dream() call \
         on the same (already-consolidated) graph; before={community_count} after={community_count2}"
    );
    assert_eq!(
        summary2.communities_updated, 0,
        "happy_path determinism: the second dream() must change ZERO community memberships \
         (idempotent re-convergence); got {}",
        summary2.communities_updated
    );
    eprintln!(
        "[full-consolidation-happy] PASS — communities={community_count} (stable across rerun: \
         {community_count2}), second-run summary communities_updated={}",
        summary2.communities_updated
    );
}

// ── Case 2: homonym_trap_no_wrong_merge ─────────────────────────────────────────

/// Plant two DISTINCT same-normalized-label entities sharing only a hub (no
/// corroborating structure — the RISK-001 category) on top of a real-ingested
/// graph, run `mem.dream()` with ALL consolidation ops ON, and assert the two
/// homonym entities are STILL DISTINCT after the full dispatcher chain (not just
/// after cross_episode in isolation). This is the system-level analogue of
/// `consolidation_cross_episode_real_llm.rs`'s ARM 3 — proving no OTHER
/// consolidation op (nor their interaction/ordering) accidentally fuses the pair.
#[tokio::test]
#[ignore = "dream_full_consolidation_real_llm::homonym_trap_no_wrong_merge: requires Ollama in \
            record mode, or the committed cassette in replay mode. Run explicitly: \
            KREMORY_VCR=record cargo test -p kremory --features test-utils,llm-smoke \
            --test it dream_full_consolidation_real_llm:: -- --ignored --nocapture \
            homonym_trap_no_wrong_merge"]
async fn homonym_trap_no_wrong_merge() {
    let _log = init_test_log("dream_full_consolidation_real_llm_homonym_trap");
    let mode = resolve_mode();
    let case = "homonym_trap";

    let (mem, provider, emb_vcr, namespace, _dir) = build_memory(&mode, case).await;
    ingest_multi_domain(&mem).await;

    let group_id = mem.group_id_for_test(&namespace);
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // HOMONYM pair: same normalized label across 2 episodes, but DISJOINT
    // neighbours (two distinct referents sharing a name). No shared third entity,
    // no identical (predicate, object) assertion → must NOT merge. `hp-`-prefixed
    // so it can never collide with a real-ingest slug.
    let hom_a = "hp-Homonym-Person"; // keeper if the guard held.
    let hom_b = "hp-homonym-person "; // would-be loser IF the guard failed.
    let hom_neighbour_a = "hp-homonym-lawfirm";
    let hom_neighbour_b = "hp-homonym-olympics";
    plant_entity(&tg, &group_id, hom_a).await;
    plant_entity(&tg, &group_id, hom_b).await;
    plant_entity(&tg, &group_id, hom_neighbour_a).await;
    plant_entity(&tg, &group_id, hom_neighbour_b).await;

    let ep1 = plant_episode(&tg).await;
    let ep2 = plant_episode(&tg).await;
    anchor(&tg, &group_id, ep1, hom_a).await;
    anchor(&tg, &group_id, ep2, hom_b).await;
    plant_fact_rel(&tg, &group_id, hom_a, "works_at", hom_neighbour_a).await;
    plant_fact_rel(&tg, &group_id, hom_b, "competed_in", hom_neighbour_b).await;

    for id in [hom_a, hom_b] {
        assert!(
            entity_exists(&tg, &group_id, id).await,
            "planted homonym principal {id:?} must exist before dream()"
        );
    }
    let real_entities_before = count_entities_not_prefixed(&tg, &group_id, "hp-").await;

    flush_if_live(&mode, &provider, &emb_vcr);

    // ONE full mem.dream() call, ALL consolidation ops ON — the homonym pair rides
    // through supersession → archive → cross_episode → communities in one pass.
    let summary =
        mem.dream().with_opts(all_consolidation_on()).await.expect(
            "mem.dream() with all consolidation ops ON must succeed over the homonym fixture",
        );

    eprintln!(
        "[full-consolidation-homonym] cross_episode_merges={} communities_updated={}",
        summary.cross_episode_would_merge, summary.communities_updated
    );

    // The homonym pair must BOTH survive — the system-level RISK-001 proof.
    let homonym_a_survives = entity_exists(&tg, &group_id, hom_a).await;
    let homonym_b_survives = entity_exists(&tg, &group_id, hom_b).await;
    assert!(
        homonym_a_survives && homonym_b_survives,
        "HOMONYM SAFETY VIOLATION (RISK-001, system-level): the same-label homonym pair must \
         BOTH survive a full mem.dream() call with every consolidation op ON; \
         a_survives={homonym_a_survives} b_survives={homonym_b_survives}"
    );

    // No corroborating structure anywhere in this fixture → cross_episode's ONLY
    // candidate pair is the homonym pair, which the guard must reject → the
    // system-level merge count is the HONEST ZERO.
    assert_eq!(
        summary.cross_episode_would_merge, 0,
        "HOMONYM SAFETY VIOLATION: cross_episode_merges must be 0 (the only candidate pair in \
         this fixture is the guarded homonym pair); got {}",
        summary.cross_episode_would_merge
    );

    // The real-ingest entity population is untouched by the planted homonym scenario.
    let real_entities_after = count_entities_not_prefixed(&tg, &group_id, "hp-").await;
    assert_eq!(
        real_entities_after, real_entities_before,
        "the real-ingest entity population must be UNTOUCHED by the homonym-trap dream() call; \
         before={real_entities_before} after={real_entities_after}"
    );

    eprintln!("[full-consolidation-homonym] PASS — homonym pair survived, 0 false merges");
}

// ── Case 3: idempotency ──────────────────────────────────────────────────────────

/// Run `mem.dream()` (all consolidation ops ON) TWICE on the same post-ingest
/// graph. The second run must be a no-op at the system level: 0 new merges, 0
/// community-partition churn, 0 new supersessions/archivals — full-system
/// convergence, not just per-op idempotency (already proven by the isolated
/// tier's ARM 2/ARM 3 pairs).
#[tokio::test]
#[ignore = "dream_full_consolidation_real_llm::idempotency: requires Ollama in record mode, or the \
            committed cassette in replay mode. Run explicitly: \
            KREMORY_VCR=record cargo test -p kremory --features test-utils,llm-smoke \
            --test it dream_full_consolidation_real_llm:: -- --ignored --nocapture idempotency"]
async fn idempotency() {
    let _log = init_test_log("dream_full_consolidation_real_llm_idempotency");
    let mode = resolve_mode();
    let case = "idempotency";

    let (mem, provider, emb_vcr, namespace, _dir) = build_memory(&mode, case).await;
    ingest_multi_domain(&mem).await;
    flush_if_live(&mode, &provider, &emb_vcr);

    let group_id = mem.group_id_for_test(&namespace);
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path")
        .clone();

    // First dream() call settles the graph (reconciliation + consolidation).
    let summary1 = mem
        .dream()
        .with_opts(all_consolidation_on())
        .await
        .expect("first mem.dream() call must succeed");
    let community_count1 = persisted_community_count(&tg, &group_id).await;
    let typed1 = count_typed_entities(&tg, &group_id).await;
    eprintln!(
        "[full-consolidation-idempotency] run1 communities={community_count1} typed={typed1} \
         communities_updated={} cross_episode_merges={} supersessions_recorded={} \
         facts_archived={}",
        summary1.communities_updated,
        summary1.cross_episode_would_merge,
        summary1.supersessions_recorded,
        summary1.facts_archived,
    );

    // Second dream() call on the SAME (now-settled) graph — no new episodes, no
    // planted scenarios, nothing changed in between.
    let summary2 = mem
        .dream()
        .with_opts(all_consolidation_on())
        .await
        .expect("second mem.dream() call must succeed");
    let community_count2 = persisted_community_count(&tg, &group_id).await;
    let typed2 = count_typed_entities(&tg, &group_id).await;
    eprintln!(
        "[full-consolidation-idempotency] run2 communities={community_count2} typed={typed2} \
         communities_updated={} cross_episode_merges={} supersessions_recorded={} \
         facts_archived={}",
        summary2.communities_updated,
        summary2.cross_episode_would_merge,
        summary2.supersessions_recorded,
        summary2.facts_archived,
    );

    // System-level convergence: the CONSOLIDATION ops (zero-LLM, deterministic
    // over the graph shape) must report ZERO new work on the second run — the
    // graph didn't change between calls, so there is nothing left to consolidate.
    assert_eq!(
        summary2.cross_episode_would_merge, 0,
        "idempotency: cross_episode_merges must be 0 on the second dream() call (no new \
         candidate pairs on an unchanged graph); got {}",
        summary2.cross_episode_would_merge
    );
    assert_eq!(
        summary2.supersessions_recorded, 0,
        "idempotency: supersessions_recorded must be 0 on the second dream() call; got {}",
        summary2.supersessions_recorded
    );
    assert_eq!(
        summary2.facts_archived, 0,
        "idempotency: facts_archived must be 0 on the second dream() call; got {}",
        summary2.facts_archived
    );
    // communities_updated is a CHANGE-count — the number of communities whose sorted-member
    // hash changed vs the prior persisted partition (per communities.rs DoD-P4.5), NOT the
    // total recomputed count. On an idempotent rerun the deterministic label-prop re-converges
    // to the IDENTICAL partition, so ZERO memberships change. This is the strongest
    // convergence proof (mirrors the isolated communities test's ARM 3 `updated_second == 0`)
    // and catches membership CHURN that a same-count check would miss.
    assert_eq!(
        summary2.communities_updated, 0,
        "idempotency: communities_updated must be 0 on the second dream() call (partition \
         re-converged, no membership changed); got {}",
        summary2.communities_updated
    );
    assert_eq!(
        community_count2, community_count1,
        "idempotency: community count must ALSO be STABLE across two dream() calls on an \
         unchanged graph; run1={community_count1} run2={community_count2}"
    );
    assert_eq!(
        typed2, typed1,
        "idempotency: typed entity count must be STABLE across two dream() calls on an unchanged \
         graph; run1={typed1} run2={typed2}"
    );

    eprintln!("[full-consolidation-idempotency] PASS — system converged, second run is a no-op");
}

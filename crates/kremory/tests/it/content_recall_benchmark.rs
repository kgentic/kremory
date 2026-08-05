#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-072 seq1 — content-RAG PRECISION/RECALL quality benchmark.
//!
//! Sibling of `tests/retrieval_benchmark.rs` (entity/fact recall@k) and a
//! measurement-grade complement to `tests/adr072_seq1_content_search.rs` (the
//! 3-episode ranking-order smoke). Where the smoke only proves BM25 ordering is
//! sane, this benchmark measures **precision@k and recall@k** of
//! `mem.recall(q).content()` (BM25/FTS5 over raw `episodes.content`) against a
//! labeled ground-truth corpus, then gates on hard thresholds that FAIL if
//! content-recall quality regresses.
//!
//! Deterministic: BM25 is lexical — no LLM, no embedder, no network. Ingest uses
//! `MockChatProvider::null()` + `.skip_extraction()` (the Phase-1 store is the
//! only phase that populates `episodes_fts`), exactly like the smoke test.
//!
//! ## Corpus design (honest, not rigged)
//!
//! 15 short episodes across 5 topics (Zephyr rocket, Acme revenue, Alice's diet,
//! Postgres migration, marathon training) plus deliberate distractors:
//!   - `distractor_zephyr_cache` shares the term "Zephyr" with the rocket topic
//!     but is a different Zephyr (a caching product) → single-term `Zephyr`
//!     query precision must drop below 1.0.
//!   - `distractor_crm_migration` shares "migration" with the Postgres topic but
//!     is a CRM migration → single-term `migration` query precision drops.
//!   - `distractor_roadmap` shares "quarterly" with the revenue topic → the AND
//!     query `Acme revenue` must NOT return it (FTS5 default-AND precision).
//!
//! Two unrelated filler episodes (weather, recipe) act as pure noise that no
//! query should ever surface.
//!
//! `content_search`'s first rung AND-joins tokens (FTS5's default operator —
//! same shape as `sanitise_fts5_query`, tokens quoted and space-joined), so
//! multi-term queries require ALL terms present. Every query below is a
//! short keyword phrase and matches on this AND rung, so these gates measure
//! AND-precision exactly as before. `content_search` also has an OR-fallback
//! rung for natural-language sentence queries (fires only when AND returns
//! zero hits) — see `content_recall_or_fallback_rescues_natural_language_question`
//! below for that arm's own coverage; it never fires for this file's
//! keyword-shaped `query_set()`.

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::memory::ContentPassage;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

/// Retrieval depth for all precision@k / recall@k measurements.
const K: usize = 5;

async fn make_memory(ns: &str) -> Memory {
    let llm: Arc<dyn kremory::memory::ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .default_namespace(Namespace::new(ns))
        .await
        .expect("Memory must build")
}

/// (label, episode-text) — the labeled corpus, ingested into the default
/// namespace. Labels are the ground-truth keys used by the query set below.
const CORPUS: &[(&str, &str)] = &[
    // ── Topic: Zephyr rocket engine ──────────────────────────────────────────
    (
        "rocket_hotfire",
        "The Zephyr rocket engine completed its first hotfire test at the desert launch pad. \
         The turbopump ran hot, so the engineers inspected the turbopump seals and cleared the \
         turbopump for reflight.",
    ),
    (
        "rocket_telemetry",
        "Zephyr rocket telemetry showed the turbopump reached full thrust during the sustained \
         orbital burn.",
    ),
    // ── Topic: Acme quarterly revenue ────────────────────────────────────────
    (
        "revenue_report",
        "Acme Corporation reported quarterly revenue of forty million dollars, beating the \
         analyst estimates for the period.",
    ),
    (
        "revenue_growth",
        "The Acme finance chief attributed the revenue growth to strong enterprise subscription \
         renewals.",
    ),
    // ── Topic: Alice's dietary preferences ───────────────────────────────────
    (
        "diet_vegetarian",
        "Alice follows a strict vegetarian diet and avoids dairy because of a lactose \
         intolerance.",
    ),
    (
        "diet_oatmilk",
        "Alice said she switched to oat milk since dairy tends to upset her stomach.",
    ),
    // ── Topic: Postgres migration ────────────────────────────────────────────
    (
        "pg_repartition",
        "The Postgres migration repartitioned the analytics tables to improve slow query \
         latency.",
    ),
    (
        "pg_maintenance",
        "Engineers ran the Postgres migration during the overnight maintenance window to avoid \
         downtime.",
    ),
    // ── Topic: marathon training ─────────────────────────────────────────────
    (
        "marathon_tempo",
        "Marathon training this week emphasised tempo runs and long hill repeats to build \
         endurance.",
    ),
    (
        "marathon_taper",
        "The marathon taper reduces weekly mileage in the fortnight before race day.",
    ),
    // ── Distractors (share ONE term with a topic, not truly relevant) ────────
    (
        "distractor_roadmap",
        "The quarterly roadmap review covered budget allocation for the next fiscal planning \
         cycle.",
    ),
    (
        "distractor_zephyr_cache",
        "The Zephyr caching layer stores rendered dashboard fragments to reduce page load time.",
    ),
    (
        "distractor_crm_migration",
        "The team documented the CRM data migration but deferred the underlying database rework.",
    ),
    // ── Pure noise (no query should ever surface these) ──────────────────────
    (
        "noise_weather",
        "Heavy rainfall delayed the outdoor concert scheduled for the weekend.",
    ),
    (
        "noise_recipe",
        "The bakery published a new sourdough recipe featuring rye flour and a longer proof.",
    ),
];

/// (query, relevant-labels) — ground truth. `k = K` for every query.
///
/// Precision is a REAL measurement here: `Zephyr` and `migration` are
/// single-term queries whose distractor shares the term, so BM25 correctly
/// returns 3 hits for 2 relevant → precision 2/3. The AND queries
/// (`Acme revenue`, `Postgres migration`, `Alice dairy`) exclude their
/// distractor and hit precision 1.0.
fn query_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // Single-term, distractor shares the term → precision < 1.0.
        ("Zephyr", vec!["rocket_hotfire", "rocket_telemetry"]),
        // Distinctive term unique to the rocket topic → precision 1.0.
        ("turbopump", vec!["rocket_hotfire", "rocket_telemetry"]),
        // AND query excludes the "quarterly" roadmap distractor.
        ("Acme revenue", vec!["revenue_report", "revenue_growth"]),
        // AND query — both Alice diet episodes carry "dairy".
        ("Alice dairy", vec!["diet_vegetarian", "diet_oatmilk"]),
        // AND query excludes the CRM-migration distractor (lacks "postgres").
        (
            "Postgres migration",
            vec!["pg_repartition", "pg_maintenance"],
        ),
        // Single-term, CRM distractor also matches "migration" → precision < 1.0.
        ("migration", vec!["pg_repartition", "pg_maintenance"]),
        // Clean topic term.
        ("marathon", vec!["marathon_tempo", "marathon_taper"]),
    ]
}

/// Ingest the full corpus into the memory's default namespace, returning a
/// `label -> episode_id` map. Uses the same Phase-1 store path as the smoke
/// test (`.skip_extraction()` — the only phase that populates `episodes_fts`).
async fn ingest_corpus(mem: &Memory) -> HashMap<&'static str, i64> {
    let mut ids = HashMap::with_capacity(CORPUS.len());
    for (label, text) in CORPUS {
        let commit = mem
            .remember(*text)
            .skip_extraction()
            .await
            .unwrap_or_else(|e| panic!("episode {label} must commit: {e}"));
        let id: i64 = commit
            .episode_entity_id
            .parse()
            .unwrap_or_else(|_| panic!("episode {label} must carry a parseable rowid"));
        ids.insert(*label, id);
    }
    ids
}

// ─── Test 1: precision@k / recall@k benchmark with hard gates ────────────────

#[tokio::test]
async fn content_recall_precision_recall_meets_thresholds() {
    let mem = make_memory("content-bench-main").await;
    let ids = ingest_corpus(&mem).await;
    // Reverse map for readable diagnostics (episode_id -> label).
    let label_of: HashMap<i64, &'static str> = ids.iter().map(|(l, i)| (*i, *l)).collect();

    let queries = query_set();

    eprintln!("\n{:-<78}", "");
    eprintln!("  CONTENT-RAG PRECISION/RECALL BENCHMARK (BM25 over episodes.content, k={K})");
    eprintln!("{:-<78}", "");
    eprintln!(
        "  {:<22} | {:>7} | {:>4} | {:>4} | {:>11} | {:>8}",
        "Query", "Ret", "Rel", "Hit", "Precision@k", "Recall@k"
    );
    eprintln!(
        "  {:-<22}-+-{:-<7}-+-{:-<4}-+-{:-<4}-+-{:-<11}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    let mut precision_sum = 0.0_f64;
    let mut recall_sum = 0.0_f64;

    for (query, relevant_labels) in &queries {
        let relevant_ids: Vec<i64> = relevant_labels
            .iter()
            .map(|l| *ids.get(*l).unwrap_or_else(|| panic!("unknown label {l}")))
            .collect();

        let passages: Vec<ContentPassage> = mem
            .recall(*query)
            .k(K)
            .content()
            .await
            .unwrap_or_else(|e| panic!("content recall for {query:?} must succeed: {e}"));

        let returned_ids: Vec<i64> = passages.iter().map(|p| p.episode_id).collect();
        let hits = returned_ids
            .iter()
            .filter(|id| relevant_ids.contains(id))
            .count();

        let precision = if returned_ids.is_empty() {
            0.0
        } else {
            hits as f64 / returned_ids.len() as f64
        };
        let recall = if relevant_ids.is_empty() {
            1.0
        } else {
            hits as f64 / relevant_ids.len() as f64
        };

        precision_sum += precision;
        recall_sum += recall;

        let returned_labels: Vec<&str> = returned_ids
            .iter()
            .map(|id| *label_of.get(id).unwrap_or(&"?"))
            .collect();

        eprintln!(
            "  {:<22} | {:>7} | {:>4} | {:>4} | {:>10.1}% | {:>7.1}%   -> {:?}",
            query,
            returned_ids.len(),
            relevant_ids.len(),
            hits,
            precision * 100.0,
            recall * 100.0,
            returned_labels,
        );
    }

    let n = queries.len() as f64;
    let mean_precision = precision_sum / n;
    let mean_recall = recall_sum / n;

    eprintln!(
        "  {:-<22}-+-{:-<7}-+-{:-<4}-+-{:-<4}-+-{:-<11}-+-{:-<8}",
        "", "", "", "", "", ""
    );
    eprintln!(
        "  {:<22} | {:>7} | {:>4} | {:>4} | {:>10.1}% | {:>7.1}%",
        "MEAN",
        "",
        "",
        "",
        mean_precision * 100.0,
        mean_recall * 100.0,
    );
    eprintln!("{:-<78}\n", "");

    // ── Hard gates ────────────────────────────────────────────────────────────
    //
    // Achieved on this corpus (2026-07-12): mean precision@5 = 90.5%, mean
    // recall@5 = 100.0%. Two single-term queries (`Zephyr`, `migration`) each
    // land precision 2/3 by design (their distractor shares the term); the five
    // remaining queries hit precision 1.0.
    //
    // Gates are set just below the achieved values so a REAL regression fails
    // (e.g. BM25 surfacing an unrelated/noise episode, or the AND semantics
    // breaking so distractors leak into the AND queries), while leaving small
    // headroom for benign corpus edits:
    //   - recall@5 ≥ 0.90 (achieved 1.00, headroom 0.10)
    //   - precision@5 ≥ 0.80 (achieved 0.905, headroom ~0.10)
    assert!(
        mean_recall >= 0.90,
        "mean content recall@{K} {:.1}% is below the 90% gate — BM25 is dropping \
         genuinely-relevant episodes",
        mean_recall * 100.0,
    );
    assert!(
        mean_precision >= 0.80,
        "mean content precision@{K} {:.1}% is below the 80% gate — BM25 is surfacing \
         irrelevant episodes (distractor/noise leakage)",
        mean_precision * 100.0,
    );
}

// ─── Test 2: near-miss precision + BM25 term-frequency ranking ───────────────

#[tokio::test]
async fn content_recall_near_miss_precision_and_tf_ranking() {
    let mem = make_memory("content-bench-nearmiss").await;
    let ids = ingest_corpus(&mem).await;

    // Near-miss precision: single-term `Zephyr` returns the two rocket episodes
    // AND the caching-product distractor — precision is a genuine 2/3, NOT 1.0.
    let zephyr: Vec<ContentPassage> = mem
        .recall("Zephyr")
        .k(K)
        .content()
        .await
        .expect("Zephyr recall must succeed");
    let zephyr_ids: Vec<i64> = zephyr.iter().map(|p| p.episode_id).collect();
    assert!(
        zephyr_ids.contains(ids.get("distractor_zephyr_cache").unwrap()),
        "the Zephyr caching distractor MUST be returned for a bare `Zephyr` query \
         (proves precision is really being measured, not rigged): got {zephyr_ids:?}"
    );
    assert_eq!(
        zephyr_ids.len(),
        3,
        "`Zephyr` must return exactly the 2 rocket episodes + 1 caching distractor"
    );

    // BM25 term-frequency ranking: `rocket_hotfire` mentions "turbopump" 3×,
    // `rocket_telemetry` mentions it once. The higher-TF episode must rank first
    // (lower score = more relevant, per kremory's FTS convention).
    let tp: Vec<ContentPassage> = mem
        .recall("turbopump")
        .k(K)
        .content()
        .await
        .expect("turbopump recall must succeed");
    assert_eq!(
        tp.len(),
        2,
        "only the two rocket episodes mention turbopump"
    );
    assert_eq!(
        tp[0].episode_id,
        *ids.get("rocket_hotfire").unwrap(),
        "higher term-frequency episode (turbopump ×3) must rank first; got {:?}",
        tp.iter().map(|p| p.episode_id).collect::<Vec<_>>(),
    );
    assert!(
        tp[0].score < tp[1].score,
        "BM25 scores must strictly differentiate the strong match from the weak one; \
         got {} (strong) vs {} (weak)",
        tp[0].score,
        tp[1].score,
    );

    // AND semantics keep the CRM-migration distractor OUT of `Postgres migration`
    // but a bare `migration` query surfaces it — precision drop is by design.
    let and_hits: Vec<ContentPassage> = mem
        .recall("Postgres migration")
        .k(K)
        .content()
        .await
        .expect("AND recall must succeed");
    let and_ids: Vec<i64> = and_hits.iter().map(|p| p.episode_id).collect();
    assert!(
        !and_ids.contains(ids.get("distractor_crm_migration").unwrap()),
        "FTS5 default-AND must exclude the CRM distractor from `Postgres migration`: {and_ids:?}"
    );
    assert_eq!(
        and_ids.len(),
        2,
        "only the two Postgres episodes match both terms"
    );

    let bare_migration: Vec<ContentPassage> = mem
        .recall("migration")
        .k(K)
        .content()
        .await
        .expect("bare migration recall must succeed");
    assert!(
        bare_migration
            .iter()
            .any(|p| p.episode_id == *ids.get("distractor_crm_migration").unwrap()),
        "bare `migration` query MUST surface the CRM distractor (real precision measurement)"
    );
}

// ─── Test 3: namespace isolation — no cross-namespace leakage ─────────────────

#[tokio::test]
async fn content_recall_namespace_isolation_holds() {
    let mem = make_memory("content-bench-ns-a").await;
    // Full corpus in the default namespace (ns-a).
    let ids_a = ingest_corpus(&mem).await;
    let a_ids: std::collections::HashSet<i64> = ids_a.values().copied().collect();

    // Distinct episodes in ns-b that share query terms with ns-a's corpus, so a
    // leak would be detectable.
    let ns_b = Namespace::new("content-bench-ns-b");
    let mut b_ids = std::collections::HashSet::new();
    for text in [
        "The Zephyr orbital cluster deployment slipped to the following quarter.",
        "Acme revenue in the northern region grew after the marathon sponsorship deal.",
    ] {
        let commit = mem
            .remember(text)
            .in_namespace(ns_b.clone())
            .skip_extraction()
            .await
            .expect("ns-b episode must commit");
        b_ids.insert(
            commit
                .episode_entity_id
                .parse::<i64>()
                .expect("ns-b rowid parseable"),
        );
    }

    // Every ns-a query result must be an ns-a episode — never an ns-b one.
    for (query, _) in query_set() {
        let passages: Vec<ContentPassage> = mem
            .recall(query)
            .k(K)
            .content()
            .await
            .expect("ns-a recall must succeed");
        for p in &passages {
            assert!(
                a_ids.contains(&p.episode_id),
                "ns-a query {query:?} returned episode {} which is not an ns-a episode",
                p.episode_id
            );
            assert!(
                !b_ids.contains(&p.episode_id),
                "ns-a query {query:?} LEAKED ns-b episode {}",
                p.episode_id
            );
        }
    }

    // Conversely, an ns-b recall for a shared term must return ONLY ns-b rows.
    let b_passages: Vec<ContentPassage> = mem
        .recall("Zephyr")
        .in_namespace(ns_b.clone())
        .k(K)
        .content()
        .await
        .expect("ns-b recall must succeed");
    assert!(
        !b_passages.is_empty(),
        "ns-b must surface its own Zephyr episode"
    );
    for p in &b_passages {
        assert!(
            b_ids.contains(&p.episode_id),
            "ns-b query returned {} which is not an ns-b episode (cross-ns leak)",
            p.episode_id
        );
    }
}

// ─── Test 4: empty / no-match queries return no false positives ───────────────

#[tokio::test]
async fn content_recall_empty_and_no_match_return_empty() {
    let mem = make_memory("content-bench-empty").await;
    let _ids = ingest_corpus(&mem).await;

    // A query with no lexical overlap with any episode must return nothing —
    // BM25 never invents a match.
    let no_match: Vec<ContentPassage> = mem
        .recall("quokka helicopter zeppelin")
        .k(K)
        .content()
        .await
        .expect("no-match recall must succeed");
    assert!(
        no_match.is_empty(),
        "a query with zero corpus overlap must return no passages; got {no_match:?}"
    );

    // An all-punctuation query sanitises to an empty FTS5 query → empty result,
    // not an error (per `content_search` empty-query short-circuit).
    let empty: Vec<ContentPassage> = mem
        .recall("!!! ??? ...")
        .k(K)
        .content()
        .await
        .expect("empty-sanitised query must succeed (not error)");
    assert!(
        empty.is_empty(),
        "an all-punctuation query must return no passages; got {empty:?}"
    );
}

// ─── Test 5: AND→OR fallback ladder — natural-language questions ─────────────

/// Every query in `query_set()` above is a short keyword phrase (1-2 terms)
/// and always matches on the FIRST rung of `content_search`'s ladder
/// (AND-joined tokens) — this benchmark's precision/recall gates never
/// exercise the OR-fallback rung.
///
/// Real consumers of `.content()` (the LoCoMo/longmemeval benchmark harness,
/// chat-style callers) query with full natural-language SENTENCES, not
/// keyword phrases. This is the substrate bug the AND→OR ladder fixes:
/// AND-joining every token in a sentence — including stopwords like
/// "did"/"the"/"about"/"during" — requires ALL of them to appear in a single
/// short conversational passage, which is realistically impossible, so the
/// content-search arm silently returned empty for the vast majority of
/// natural-language queries prior to this fix.
///
/// The query below includes "discover" and "regarding" — neither word
/// appears anywhere in `CORPUS` — so an AND-only match is impossible BY
/// CONSTRUCTION (proves the eventual non-empty result came from the
/// OR-fallback rung, not a coincidental AND hit).
#[tokio::test]
async fn content_recall_or_fallback_rescues_natural_language_question() {
    let mem = make_memory("content-bench-or-fallback").await;
    let ids = ingest_corpus(&mem).await;

    let question =
        "What did the engineers discover regarding the turbopump during the hotfire test?";

    let passages: Vec<ContentPassage> = mem
        .recall(question)
        .k(K)
        .content()
        .await
        .expect("natural-language recall must succeed");

    assert!(
        !passages.is_empty(),
        "AND->OR fallback must rescue a natural-language question whose AND-join is \
         impossible to satisfy (contains 'discover'/'regarding', absent from every \
         episode) — got empty for {question:?}. Without the OR-fallback rung this \
         is exactly the LoCoMo-benchmark content-search failure (100% empty recalls)."
    );
    let returned_ids: Vec<i64> = passages.iter().map(|p| p.episode_id).collect();
    assert!(
        returned_ids.contains(ids.get("rocket_hotfire").unwrap()),
        "OR-fallback must surface `rocket_hotfire` (shares turbopump/hotfire/test/engineers \
         terms with the question): got {returned_ids:?}"
    );
}

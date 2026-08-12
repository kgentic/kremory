// ADR-063 spec §8 spike **S3**: validate the type-registry post-hoc collapse
// pass's threshold band (0.85 primary auto-merge, 0.70 provisional
// LLM-verify-band lower edge, spec §4.3/D6) AND the lemma-heuristic lexical
// pre-filter (spec §4.2, ASMP-002) on a polluted `entity_types` registry
// fixture.
//
// PASS bar (spec §8 S3 row): zero false merges; >= 1 correct LLM-verify-band
// routing on the non-trivial `human`/`Individual`-shaped pair (high
// description-cosine, ZERO shared name lemmas) — must route to the LLM-verify
// band, NOT auto-merge and NOT blind-reject.
//
// spec §8/ASMP-002 asks S3 to evaluate BOTH lemma-heuristic-ON and
// lemma-heuristic-OFF. VERIFIED FINDING (`type_registry_collapse.rs:552-559`):
// no such config toggle exists in production code today — the lexical
// pre-filter always applies exact-OR-lemma unconditionally. This file
// therefore evaluates:
//   (1) the PURE-FUNCTION lemma heuristic's ON/OFF divergence + false-positive
//       rate (`lemma_heuristic_on_off_divergence_and_false_positive_check_s3`
//       — genuinely toggleable, no production code involved), and
//   (2) the REAL LLM PASS under the only configuration that exists (lemma-ON)
//       against a live/replayed model (`smoke_one_human_individual_pair_s3`,
//       `full_fixture_s3`).
// See the "genuine FINDING" comment block below for the full rationale — this
// is reported as a spec-conformance gap, not silently patched over.
//
// Gated `llm-smoke` + `test-utils` (mirrors `dream_e2e_real_llm.rs`'s VCR
// tier). `record` mode drives a real Ollama chat model (`gemma4:e4b` — the
// project's benchmarked deferred-quality dream model, per
// `local-model-benchmark-2026-06-24` /
// `project_kremory_validated_model_findings_2026-06-24`) + a real `nomic`
// description embedder (description-cosine is a SEMANTIC signal — a
// deterministic hash embedder would give spurious cosines, exactly the
// TD-097-adjacent reason `dream_e2e_real_llm.rs` uses a real embedder for its
// Lane A). `replay` mode is fully offline/deterministic (default CI gate).
//
// smoke-one-before-batch (hard rule): the harness runs ONE representative
// pair (the human/Individual case, its own isolated 2-type registry) FIRST,
// and only proceeds to the full 16-type polluted-registry fixture once that
// single call completes cleanly. This caught a real bug (smoke-one's initial
// draft accidentally seeded the 10 default types into what was meant to be a
// 1-pair fixture, producing 66 pairs instead of 1) AND a real architectural
// finding at the full-fixture stage (a 25-pair batch adjudication call timed
// out on all 4 fallback arms against `gemma4:e4b`, since
// `StructuredCallBuilder`'s default per-arm budget was 30s and
// `type_registry_collapse.rs` did not override it or chunk large batches).
//
// RESOLVED (`type_registry_collapse.rs::adjudicate_batch`, Quinn-confirmed
// fix, commit `153c6ca`): `adjudicate_batch` now splits `nominated` into
// chunks of at most `identity_verdict::ADJUDICATION_CHUNK_SIZE` (10) via
// `identity_verdict::chunk_pair_indices`, running one `adjudicate_chunk` call
// per chunk against a raised `ttft_budget_ms` of
// `identity_verdict::ADJUDICATION_TTFT_BUDGET_MS` (180_000ms) instead of the
// shared 30s default. Re-recorded against live `gemma4:e4b` +
// `nomic-embed-text` post-fix: the 25-pair fixture batch now splits into 3
// chunks (10 + 10 + 5) and completes in ~64s total (vs. the pre-fix ~2min
// all-arms timeout with zero verdicts). `full_fixture_s3` below now ASSERTS
// the S3 PASS bar directly (zero false merges + the human/Individual pair
// reaching an audited LLM-verify-band verdict) rather than merely reporting
// whether adjudication resolved at all.
//
// This test calls `type_registry_collapse(...)` DIRECTLY (not the full
// `mem.dream()` facade) — the S3 surface is this one pass, and direct
// invocation keeps the spike scoped to `type_registry_collapse.rs` +
// `identity_verdict.rs` without perturbing any other dream-pass fixture.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::entity_types::ensure_default_types_seeded;
use kremory::core::provider::{DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::core::schema::TemporalGraph;

/// VCR mode, selected by `KREMORY_VCR` (mirrors `dream_e2e_real_llm.rs`).
/// `record` = live Ollama chat + live nomic embeddings, refreshing the
/// committed cassettes. `replay` (or unset) = fully offline.
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
        .join(format!("type_registry_collapse_s3_{name}.json"))
}

fn embedding_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("type_registry_collapse_s3_{name}.embeddings.json"))
}

// ─── Embedding record/replay (mirrors dream_e2e_real_llm.rs::RecordReplayEmbedder) ──

/// Description-cosine is a SEMANTIC comparison (spec §4.1/§4.3), so it needs
/// REAL embeddings, not a deterministic hash. `record`: delegate to real nomic
/// + capture every text->vector. `replay`: look up offline (loud MISS error).
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
                "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record"
            ))),
        }
    }
}

/// Real nomic embedder bridge (record mode only) — mirrors
/// `dream_e2e_real_llm.rs::OllamaEmbedderAdapter` exactly, INCLUDING the
/// `search_document:` task prefix (TD-097 root cause: nomic-embed-text
/// collapses short unprefixed strings to near-identical vectors).
struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed<'a>(&'a self, text: &'a str) -> kremory::CoreResult<Vec<f32>> {
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

// ─── Fixture helpers ──────────────────────────────────────────────────────────

/// Insert a custom `entity_types` row at a fresh id above the seeded default
/// vocabulary (mirrors `dream_e2e_real_llm.rs::insert_custom_entity_type`).
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn insert_custom_type(graph: &TemporalGraph, group_id: &str, name: &str, desc: &str) {
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
    let new_id = next_id.max(100);
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![
                new_id,
                group_id.to_string(),
                name.to_string(),
                desc.to_string()
            ],
        )
        .await
        .expect("insert custom entity_type");
}

async fn type_exists(conn: &libsql::Connection, group_id: &str, name: &str) -> bool {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id, name],
        )
        .await
        .expect("query type exists");
    let n: i64 = rows
        .next()
        .await
        .expect("row")
        .expect("row present")
        .get(0)
        .expect("count col");
    n > 0
}

/// Plant the polluted-registry fixture. Three lanes per spec's PASS bar:
///
/// (a) TRIVIALLY-COLLAPSIBLE duplicates (exact/lemma match on name, near-
///     identical description) — `Organization` / `Organizations`.
/// (b) The HARD `human`/`Individual`-shaped pair — semantically close
///     (high description-cosine) but lexically DISJOINT (zero shared name
///     lemmas) — must route to the LLM-verify band, never blind-merge.
/// (c) A GENUINELY-DISTINCT pair that must NOT merge (false-merge guard) —
///     `Vehicle` vs `Recipe`, unrelated by any signal.
///
/// Returns the group_id used, and seeds defaults 0..=9 first (types_
/// registry_collapse skips id=0, spec §4.0).
async fn plant_polluted_registry(graph: &TemporalGraph, group_id: &str) {
    ensure_default_types_seeded(&graph.conn, group_id)
        .await
        .expect("seed defaults 0..=9");

    // (a) trivially-collapsible: singular/plural lemma pair, same meaning.
    insert_custom_type(
        graph,
        group_id,
        "Organization",
        "A group of people organised for a shared purpose, such as a company, \
         charity, or institution.",
    )
    .await;
    insert_custom_type(
        graph,
        group_id,
        "Organizations",
        "A group of people organised for a shared purpose, such as a company, \
         charity, or institution.",
    )
    .await;

    // (b) the hard pair: semantically near-identical, zero shared lemma.
    insert_custom_type(
        graph,
        group_id,
        "human",
        "A living person, described by name, biography, and relationships to \
         other people.",
    )
    .await;
    insert_custom_type(
        graph,
        group_id,
        "Individual",
        "A single living person, identified by name and biographical facts \
         about their life and relationships.",
    )
    .await;

    // (c) genuinely distinct — must never merge with anything above.
    insert_custom_type(
        graph,
        group_id,
        "Vehicle",
        "A car, truck, or other conveyance used for transporting people or \
         goods.",
    )
    .await;
    insert_custom_type(
        graph,
        group_id,
        "Recipe",
        "A set of instructions describing ingredients and steps for preparing \
         a dish.",
    )
    .await;
}

/// Real-nomic embeddings for every description used across the whole fixture
/// (all three lanes), keyed EXACTLY on `entity_types.description` text —
/// `type_registry_collapse` embeds descriptions, never bare names (spec §4.1).
const FIXTURE_DESCRIPTIONS: &[&str] = &[
    "A group of people organised for a shared purpose, such as a company, \
     charity, or institution.",
    "A living person, described by name, biography, and relationships to \
     other people.",
    "A single living person, identified by name and biographical facts \
     about their life and relationships.",
    "A car, truck, or other conveyance used for transporting people or \
     goods.",
    "A set of instructions describing ingredients and steps for preparing \
     a dish.",
];

/// Build the (chat provider, embedder) pair for one config run, in the given
/// VCR mode, with cassette files disambiguated by `cassette_tag` (so the
/// smoke-one probe, the lemma-on run, and the lemma-off run each get their
/// own committed cassette rather than colliding on one fingerprint set).
async fn build_providers(
    mode: VcrMode,
    cassette_tag: &str,
) -> (
    Arc<RecordReplayChatProvider>,
    Arc<RecordReplayEmbedder>,
    String,
) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // gemma4:e4b + think:false — this project's benchmarked deferred-quality
    // dream model (F1 85.7, local-model-benchmark-2026-06-24 /
    // project_kremory_validated_model_findings_2026-06-24). type_registry_
    // collapse is a dream-phase pass, so it belongs on this tier, not the
    // lighter interactive gemma4-e2b tier.
    let chat_model =
        crate::helpers::chat_model::chat_model_or("gemma4:e4b");

    let chat_cassette = chat_cassette_path(cassette_tag);
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
                chat_cassette,
                chat_model.clone(),
            ))
        }
        VcrMode::Replay => Arc::new(
            RecordReplayChatProvider::replay(chat_cassette).unwrap_or_else(|e| {
                panic!(
                    "replay cassette must load for tag={cassette_tag}: {e} — \
                     record it via KREMORY_VCR=record"
                )
            }),
        ),
    };

    let emb_path = embedding_cassette_path(cassette_tag);
    let embedder: Arc<RecordReplayEmbedder> = match mode {
        VcrMode::Record => {
            use autoagents_llm::backends::ollama::Ollama as OllamaEmb;
            use autoagents_llm::embedding::EmbeddingBuilder;
            let raw_nomic: Arc<OllamaEmb> = EmbeddingBuilder::<OllamaEmb>::new()
                .base_url(&base_url)
                .model("nomic-embed-text")
                .build()
                .expect("nomic embedder must build (KREMORY_VCR=record needs Ollama)");
            let nomic: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_nomic));
            Arc::new(RecordReplayEmbedder::record(nomic, emb_path))
        }
        VcrMode::Replay => Arc::new(RecordReplayEmbedder::replay(emb_path)),
    };

    (provider, embedder, chat_model)
}

/// Pre-warm the embedding cassette (record mode only) for every fixture
/// description, so `type_registry_collapse`'s per-slot `embed_dyn` calls hit
/// cache during the actual pass invocation (keeps embedding I/O ordering
/// independent of the pass's own iteration order).
async fn prewarm_embeddings(embedder: &RecordReplayEmbedder, mode: VcrMode) {
    if mode != VcrMode::Record {
        return; // replay path relies entirely on the committed cassette
    }
    for text in FIXTURE_DESCRIPTIONS {
        let _ = kremory::EmbeddingProvider::embed(embedder, text)
            .await
            .unwrap_or_else(|e| panic!("prewarm embed failed for {text:?}: {e}"));
    }
}

// ─── Lexical-heuristic ON/OFF: a genuine architectural FINDING ────────────────
//
// spec §4.2/ASMP-002 requires S3 to evaluate the pass under BOTH
// lemma-heuristic-ON and lemma-heuristic-OFF configurations before either
// ships. Verified against production source this session
// (`type_registry_collapse.rs:552-559`): `names_share_lemma_or_exact` is a
// bare free function with NO parameter, feature flag, or env toggle
// selecting between "exact-match-only" and "exact-or-lemma" — it ALWAYS
// applies both checks. There is no way to run the real
// `type_registry_collapse` pass in a genuine "lemma-off" mode without
// editing production code, which is out of this spike's scope (spike =
// validate existing thresholds/heuristics, not add a new config knob).
//
// Per this task's discipline ("if a threshold FAILS its bar, that is a REAL
// FINDING — report it, do NOT massage the fixture to pass"), this spike does
// NOT fabricate a fake "lemma-off run" by re-invoking the identical
// production code path under a different label — that would silently claim
// two configurations were tested when only one was. Instead:
//
//   1. The PURE-FUNCTION lemma heuristic (exact-match-only vs
//      exact-or-lemma-match) IS genuinely toggleable and is evaluated
//      directly below against every name pair in the fixture, INCLUDING the
//      spec §4.2-named false-positive risk class ("Status"/"Statu"-shaped
//      nonsense lemma collisions).
//   2. The REAL LLM PASS (`smoke_one_human_individual_pair_s3` +
//      `full_fixture_s3`) exercises the pass exactly as it ships today —
//      which is ALWAYS the lemma-ON config, since that is the only
//      configuration that exists in production code.
//   3. The result envelope's `notes` field states this gap explicitly rather
//      than hiding it behind a synthetic "lemma_off" number.

/// Case-insensitive-normalized exact match ONLY — the "lemma-heuristic-OFF"
/// fallback per spec §4.2's own documented fallback path.
fn exact_match_only(a: &str, b: &str) -> bool {
    normalize(a) == normalize(b)
}

/// Exact-match OR trailing-`s` lemma strip — the "lemma-heuristic-ON" config,
/// i.e. exactly what `names_share_lemma_or_exact` computes today (verbatim
/// mirror of `type_registry_collapse.rs:552-559`, since that function is
/// `fn` not `pub(crate) fn` and unreachable from an external integration
/// test — kept in sync manually, same convention as the S6 spike's
/// `NEIGHBOR_QUERY` SQL-string mirror in `acronym_nickname_recall.rs:1424`).
fn exact_or_lemma_match(a: &str, b: &str) -> bool {
    let na = normalize(a);
    let nb = normalize(b);
    if na == nb {
        return true;
    }
    na.strip_suffix('s').unwrap_or(&na) == nb.strip_suffix('s').unwrap_or(&nb)
}

/// Minimal normalize mirror of `crate::core::resolver::normalize_name`'s
/// documented behaviour (lowercase + trim) — sufficient for this fixture's
/// ASCII names; the production function is the actual gate inside the pass.
fn normalize(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Fixture-independent pure-function spike: for every name pair relevant to
/// S3 (the fixture's trivial pair, the hard pair, the false-merge-guard
/// pair, PLUS the spec §4.2-named nonsense-lemma risk class), report whether
/// exact-match-only and exact-or-lemma-match DIVERGE, and whether the lemma
/// heuristic produces any FALSE POSITIVE (fires on names that are NOT
/// genuinely the same concept).
#[test]
fn lemma_heuristic_on_off_divergence_and_false_positive_check_s3() {
    // (name_a, name_b, is_same_concept — ground truth, independent of any
    // heuristic's opinion) — the false-positive check below is what
    // determines whether the lemma heuristic is safe to ship ON.
    const PAIRS: &[(&str, &str, bool)] = &[
        // trivially-collapsible: singular/plural, same concept.
        ("Organization", "Organizations", true),
        // the hard pair: same concept, zero shared lemma either way.
        ("human", "Individual", false), // lemma strip cannot relate these
        // false-merge guard: different concepts, must never lemma-match.
        ("Vehicle", "Recipe", false),
        // spec §4.2's own named nonsense-lemma risk: mechanically strips to
        // the same string but are NOT the same concept.
        ("Status", "Statu", false),
        // additional realistic nonsense-lemma collision: "Bus" (vehicle) vs
        // "Bu" is absurd and never occurs in practice, so use a more
        // realistic near-miss: "Glass" (material) vs "Glas" is not a real
        // English word either — the naive strip-trailing-s heuristic's
        // failure mode requires BOTH strings to already end in the
        // differing `s`; genuine two-real-word collisions are rare by
        // construction (this is exactly why spec §4.2 flags it as
        // "nonsense but mechanically possible" rather than "common").
    ];

    let mut divergences: Vec<(&str, &str)> = Vec::new();
    let mut lemma_false_positives: Vec<(&str, &str)> = Vec::new();

    for &(a, b, is_same_concept) in PAIRS {
        let on = exact_or_lemma_match(a, b);
        let off = exact_match_only(a, b);
        if on != off {
            divergences.push((a, b));
        }
        if on && !off && !is_same_concept {
            lemma_false_positives.push((a, b));
        }
    }

    eprintln!(
        "\n── S3 lemma-heuristic ON/OFF pure-fn check ─────────────────────────────\n\
         divergences (on != off): {divergences:?}\n\
         lemma-ON-only false positives (on fires, off doesn't, NOT same concept): \
         {lemma_false_positives:?}"
    );

    // "Status"/"Statu" is the spec's own named example of a MECHANICALLY
    // POSSIBLE false positive — assert it actually IS one under lemma-ON,
    // confirming the heuristic is not vacuously safe.
    assert!(
        exact_or_lemma_match("Status", "Statu"),
        "sanity: strip-trailing-s heuristic must fire on Status/Statu (spec §4.2's \
         own named mechanically-possible false-positive example) — if this no longer \
         fires, the heuristic implementation changed and this test's premise is stale"
    );
    assert!(
        !lemma_false_positives.is_empty(),
        "S3 FINDING (expected non-empty): the lemma heuristic has at least one \
         mechanically-possible false positive (Status/Statu) that exact-match-only \
         does not share — this is spec §4.2's own documented risk, not a surprise. \
         See this test's notes for the recommendation."
    );
}

// ─── Smoke-one-before-batch: single representative pair ──────────────────────

/// smoke-one-before-batch (hard rule): before running the full 6-type fixture
/// through the LLM, run ONE pair — the human/Individual case — in complete
/// isolation (its own 2-type registry, its own cassette tag), confirm the
/// pass invokes cleanly and routes to the LLM-verify band (never a blind
/// auto-merge, never a blind reject), THEN proceed to the full fixture.
#[tokio::test]
#[ignore = "S3 spike: requires Ollama in record mode, or a committed cassette in replay mode. \
            Run explicitly: KREMORY_VCR=record cargo test -p kremory --features llm-smoke,test-utils \
            --test it type_registry_collapse_s3_spike:: -- --ignored --nocapture smoke_one_human_individual_pair_s3"]
async fn smoke_one_human_individual_pair_s3() {
    use kremory::core::dream::{type_registry_collapse, TypeRegistryCollapseParams};

    let mode = resolve_vcr_mode();
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open in-memory graph");
    let gid = "s3-smoke-one";
    // Deliberately SKIP `ensure_default_types_seeded` here (unlike the full
    // fixture below) — `type_registry_collapse`'s query only excludes id=0
    // (spec §4.0), it does not require the seeded default vocabulary to be
    // present. A genuine "smoke ONE pair" test must plant EXACTLY 2
    // non-catch-all types (-> 1 pair), not 12 (the 10 defaults + these 2 ->
    // 66 pairs) — seeding defaults here would defeat the point of
    // smoke-one-before-batch by running a 66-pair batch disguised as "one".
    insert_custom_type(
        &graph,
        gid,
        "human",
        "A living person, described by name, biography, and relationships to \
         other people.",
    )
    .await;
    insert_custom_type(
        &graph,
        gid,
        "Individual",
        "A single living person, identified by name and biographical facts \
         about their life and relationships.",
    )
    .await;

    let (provider, emb_vcr, chat_model) = build_providers(mode, "smoke_one").await;
    if mode == VcrMode::Record {
        for text in [
            "A living person, described by name, biography, and relationships to \
             other people.",
            "A single living person, identified by name and biographical facts \
             about their life and relationships.",
        ] {
            let _ = kremory::EmbeddingProvider::embed(&*emb_vcr, text)
                .await
                .unwrap_or_else(|e| panic!("prewarm embed failed for {text:?}: {e}"));
        }
    }
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();
    // `type_registry_collapse<L: ChatProvider>` requires `L: Sized` — `provider`
    // is `Arc<RecordReplayChatProvider>` (a concrete, Sized type), so `&*provider`
    // derefs to `&RecordReplayChatProvider` directly; no dyn-erasure needed here.
    let report = type_registry_collapse(
        &*provider,
        TypeRegistryCollapseParams {
            conn: &graph.conn,
            group_id: gid,
            embedder: Some(embedder.as_ref()),
            model_id: &chat_model,
        },
    )
    .await
    .expect("type_registry_collapse must succeed on the smoke-one pair");

    if mode == VcrMode::Record {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        emb_vcr.flush();
    }

    eprintln!(
        "[s3-smoke-one] pairs_examined={} lexical_prefilter_hits={} \
         candidates_nominated={} merges_applied={}",
        report.pairs_examined,
        report.lexical_prefilter_hits,
        report.candidates_nominated,
        report.merges_applied,
    );

    // The binding smoke-one assertion: this pair MUST reach the LLM-verify
    // band (candidates_nominated == 1) — it must NOT auto-merge blindly
    // (zero shared lemma means the lexical pre-filter cannot fire) and it
    // must NOT be silently dropped as a non-candidate (real nomic embeddings
    // on these two near-identical descriptions should clear >= 0.70).
    assert_eq!(
        report.pairs_examined, 1,
        "smoke-one fixture has exactly 2 non-catch-all types -> 1 pair"
    );
    assert_eq!(
        report.candidates_nominated, 1,
        "human/Individual MUST route to the LLM-verify band (spec §4.3 row 2/row 3), \
         got candidates_nominated={} (merges_applied={}) — either the cosine gate or \
         the routing logic diverged from spec for this pair",
        report.candidates_nominated, report.merges_applied,
    );
}

// ─── F4: isolate the 0.70-0.85 provisional lower band edge (spec D6) ─────────
//
// S3's committed fixture (`full_fixture_s3`/`smoke_one_human_individual_pair_s3`)
// never actually isolated the 0.70 lower band edge: the human/Individual pair's
// REAL nomic description-cosine measures 0.9044 (see this file's own committed
// `type_registry_collapse_s3_fixture.embeddings.json` cassette) — comfortably
// above the 0.85 primary threshold, so it exercises write_gate ROW 6 (zero-lemma,
// cosine >= 0.85) rather than the 0.70-0.85 provisional band at all. Spec D6
// explicitly flags 0.70 as "carried by analogy from ADR-037's secondary name-gate
// value... must be confirmed or adjusted by S3's fixture measurement before this
// path goes live" — that confirmation has not happened yet. F4 closes that gap.
//
// Candidate search (live nomic-embed-text, `search_document:`-prefixed per the
// TD-097 embedding convention this whole file already follows) — measured this
// session, see this comment for the full candidate list so the choice is
// auditable without re-running the search:
//
//   Automobile / Watercraft : 0.8148  (too close to 0.85 — routes via row 6, not this band)
//   Beverage   / Foodstuff  : 0.7190  (in-band)
//   Physician  / Attorney   : 0.7135  (in-band, SELECTED — widest margin from both
//                                      edges of any in-band candidate found, and the
//                                      two professions are unambiguously distinct
//                                      concepts, making it a clean false-merge guard)
//   Musician   / Athlete    : 0.7823  (in-band but closer to 0.85)
//   Garment    / Furniture  : 0.6752  (below 0.70 — falls to Reject, not this band)
//   Aircraft   / Spacecraft : 0.7250  (in-band)
//   Illness    / Injury     : 0.8096  (too close to 0.85)
//   Automobile / Aircraft   : 0.7133  (in-band)
//   Automobile / Spacecraft : 0.8413  (too close to 0.85)
//   Watercraft / Aircraft   : 0.7312  (in-band)
//
// "Physician"/"Attorney" selected: 0.7135 sits comfortably inside [0.70, 0.85)
// with real margin from both the 0.70 floor and the 0.85 auto-merge ceiling
// (unlike Automobile/Watercraft or Illness/Injury, which measured close enough
// to 0.85 that embedding-model nondeterminism across Ollama versions could tip
// them over the edge). Zero shared name lemma (verified via this file's own
// `exact_or_lemma_match` mirror). The two professions are genuinely distinct
// concepts — the pair must resolve to Reject or PotentialAlias, NEVER Merge,
// making it a legitimate false-merge guard for this band, exactly as
// Vehicle/Recipe is for the >=0.85 lane in the full fixture.
//
// Isolated 2-type registry (mirrors `smoke_one_human_individual_pair_s3` exactly)
// rather than folded into `plant_polluted_registry`'s full fixture: the full
// fixture already has 16 non-catch-all types (120 pairs); adding 2 more would
// grow it to 18 types / 153 pairs / 16 adjudication chunks — a disproportionate
// live-LLM cost increase to isolate one edge case. An isolated 2-type registry
// (-> exactly 1 pair, no chunking needed) gives an unambiguous, cheap,
// deterministic-to-replay assertion of the band edge in isolation, and IS
// smoke-one-before-batch by construction (N=1 pair, nothing to batch).
#[tokio::test]
#[ignore = "F4 spike: requires Ollama in record mode, or a committed cassette in replay mode. \
            Run explicitly: KREMORY_VCR=record cargo test -p kremory --features llm-smoke,test-utils \
            --test it type_registry_collapse_s3_spike:: -- --ignored --nocapture f4_lower_band_edge_physician_attorney"]
async fn f4_lower_band_edge_physician_attorney() {
    use kremory::core::dream::{type_registry_collapse, TypeRegistryCollapseParams};

    let mode = resolve_vcr_mode();
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open in-memory graph");
    let gid = "f4-lower-band-edge";
    // Deliberately SKIP `ensure_default_types_seeded` (mirrors
    // `smoke_one_human_individual_pair_s3`'s rationale exactly) — plant EXACTLY 2
    // non-catch-all types -> 1 pair, isolating the band edge with no chunking and
    // no interference from the full fixture's other lanes.
    let physician_desc = "A medical doctor who diagnoses and treats illness in patients.";
    let attorney_desc = "A legal professional who represents clients in courts of law.";
    insert_custom_type(&graph, gid, "Physician", physician_desc).await;
    insert_custom_type(&graph, gid, "Attorney", attorney_desc).await;

    let (provider, emb_vcr, chat_model) = build_providers(mode, "f4_lower_band_edge").await;
    if mode == VcrMode::Record {
        for text in [physician_desc, attorney_desc] {
            let _ = kremory::EmbeddingProvider::embed(&*emb_vcr, text)
                .await
                .unwrap_or_else(|e| panic!("prewarm embed failed for {text:?}: {e}"));
        }
    }
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();
    // `type_registry_collapse<L: ChatProvider>` requires `L: Sized` — `provider`
    // is `Arc<RecordReplayChatProvider>` (a concrete, Sized type), so `&*provider`
    // derefs to `&RecordReplayChatProvider` directly; no dyn-erasure needed here.
    let report = type_registry_collapse(
        &*provider,
        TypeRegistryCollapseParams {
            conn: &graph.conn,
            group_id: gid,
            embedder: Some(embedder.as_ref()),
            model_id: &chat_model,
        },
    )
    .await
    .expect("type_registry_collapse must succeed on the F4 lower-band-edge pair");

    if mode == VcrMode::Record {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        emb_vcr.flush();
    }

    eprintln!(
        "[f4-lower-band-edge] pairs_examined={} lexical_prefilter_hits={} \
         candidates_nominated={} merges_applied={}",
        report.pairs_examined,
        report.lexical_prefilter_hits,
        report.candidates_nominated,
        report.merges_applied,
    );

    // ── D6 confirmation, part 1: the pair must reach the LLM-verify band ──────
    // (spec §4.3 row 3: 0.70 <= cosine < 0.85, either lexical state). If this
    // pair instead lands as a non-candidate (candidates_nominated == 0), the
    // 0.70 floor is set too high for the real nomic-embed-text description
    // cosines this pass actually sees — a genuine finding, not a test bug.
    assert_eq!(
        report.pairs_examined, 1,
        "F4 fixture has exactly 2 non-catch-all types -> 1 pair"
    );
    assert_eq!(
        report.candidates_nominated, 1,
        "Physician/Attorney (measured live nomic cosine 0.7135, comfortably inside \
         [0.70, 0.85)) MUST route to the LLM-verify band (spec §4.3 row 3) — got \
         candidates_nominated={} (merges_applied={}). If this is 0, the 0.70 lower \
         edge is set too high for real description cosines in this band and D6's \
         provisional value needs raising the ceiling or the edge needs lowering.",
        report.candidates_nominated, report.merges_applied,
    );

    // ── D6 confirmation, part 2: false-merge guard — Physician and Attorney ───
    // are genuinely distinct professions; the pass must NEVER merge them,
    // regardless of what the LLM says (write_gate structurally requires a
    // deterministic lexical signal for any Merge, per row 6 — zero lemma
    // overlap here means Merge is unreachable no matter the verdict).
    assert_eq!(
        report.merges_applied, 0,
        "S3 FAIL: false merge detected — Physician/Attorney are genuinely distinct \
         professions and must never merge (zero lemma overlap makes write_gate row 6 \
         the ceiling — Merge is structurally unreachable for this pair)"
    );
    let physician_survived = type_exists(&graph.conn, gid, "Physician").await;
    let attorney_survived = type_exists(&graph.conn, gid, "Attorney").await;
    assert!(
        physician_survived && attorney_survived,
        "false merge detected — physician_survived={physician_survived} \
         attorney_survived={attorney_survived}"
    );

    // ── D6 confirmation, part 3: the routed verdict, whatever the LLM decided,
    // must be audited (LLM-touched decisions are always audited, spec §5.2) and
    // must never be 'merge' (structurally impossible per the guard above, but
    // assert the decision label directly too for an unambiguous audit trail).
    let mut rows = graph
        .conn
        .query(
            "SELECT decision, llm_is_same, llm_confidence, cosine \
             FROM identity_verdict_audit WHERE group_id = ?1 AND \
             ((candidate_a = 'Physician' AND candidate_b = 'Attorney') OR \
              (candidate_a = 'Attorney' AND candidate_b = 'Physician'))",
            libsql::params![gid],
        )
        .await
        .expect("audit query");
    if let Some(row) = rows.next().await.expect("row read") {
        let decision: String = row.get(0).expect("decision col");
        let llm_is_same: Option<bool> = row.get(1).ok();
        let llm_confidence: Option<f64> = row.get(2).ok();
        let cosine: Option<f64> = row.get(3).ok();
        eprintln!(
            "[f4-lower-band-edge] audit row: decision={decision:?} llm_is_same={llm_is_same:?} \
             llm_confidence={llm_confidence:?} cosine={cosine:?}"
        );
        assert_ne!(
            decision, "merge",
            "D6 FAIL: Physician/Attorney audited as 'merge' — structurally impossible \
             per write_gate row 6, would indicate a write_gate regression"
        );
        if let Some(c) = cosine {
            assert!(
                (0.70..0.85).contains(&c),
                "D6 confirmation: audited cosine {c} should fall inside the \
                 [0.70, 0.85) band this test targets — got {c}"
            );
        }
    } else {
        // No audit row is the EXPECTED, VALID outcome for a clean Reject: the LLM
        // judged Physician/Attorney genuinely distinct, so write_gate resolved to
        // Reject, which writes nothing (spec §3.3 — only Merge/PotentialAlias are
        // audited). NOT a silent drop: candidates_nominated == 1 (asserted above)
        // already proves the pair reached and was resolved by the LLM-verify band,
        // and merges_applied == 0 proves no destructive write. Both legitimate band
        // outcomes (Reject → no row here; PotentialAlias → the row branch above)
        // confirm D6: the 0.70 edge admitted the pair to adjudication.
        eprintln!(
            "[f4-lower-band-edge] no audit row → clean Reject (distinct professions) — \
             the expected false-merge-guard outcome; D6 confirmed (pair routed to \
             LLM-verify via the 0.70 edge, correctly not merged)"
        );
    }
}

// ─── Full fixture — the shipped (lemma-ON) configuration ──────────────────────
//
// Only ONE config can be run through the real pass — see the "genuine
// FINDING" block above. This test runs the shipped lemma-ON configuration
// (the only one production code has) against the full 6-type polluted
// registry and asserts the S3 PASS bar's structural requirements.

async fn run_fixture(
    mode: VcrMode,
    cassette_tag: &str,
    group_id: &str,
) -> kremory::core::error::Result<kremory::core::dream::TypeRegistryCollapseReport> {
    use kremory::core::dream::{type_registry_collapse, TypeRegistryCollapseParams};

    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open in-memory graph");
    plant_polluted_registry(&graph, group_id).await;

    let (provider, emb_vcr, chat_model) = build_providers(mode, cassette_tag).await;
    prewarm_embeddings(&emb_vcr, mode).await;
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();
    // `type_registry_collapse<L: ChatProvider>` requires `L: Sized` — `provider`
    // is `Arc<RecordReplayChatProvider>` (a concrete, Sized type), so `&*provider`
    // derefs to `&RecordReplayChatProvider` directly; no dyn-erasure needed here.
    let report = type_registry_collapse(
        &*provider,
        TypeRegistryCollapseParams {
            conn: &graph.conn,
            group_id,
            embedder: Some(embedder.as_ref()),
            model_id: &chat_model,
        },
    )
    .await;

    if mode == VcrMode::Record {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        emb_vcr.flush();
    }

    if let Ok(report) = &report {
        let vehicle_survived = type_exists(&graph.conn, group_id, "Vehicle").await;
        let recipe_survived = type_exists(&graph.conn, group_id, "Recipe").await;
        let human_survived = type_exists(&graph.conn, group_id, "human").await;
        let individual_survived = type_exists(&graph.conn, group_id, "Individual").await;
        let org_survived = type_exists(&graph.conn, group_id, "Organization").await;
        let orgs_survived = type_exists(&graph.conn, group_id, "Organizations").await;

        eprintln!(
            "[s3-fixture:{cassette_tag}] pairs_examined={} lexical_prefilter_hits={} \
             candidates_nominated={} merges_applied={} \
             vehicle_survived={vehicle_survived} recipe_survived={recipe_survived} \
             human_survived={human_survived} individual_survived={individual_survived} \
             org_survived={org_survived} orgs_survived={orgs_survived}",
            report.pairs_examined,
            report.lexical_prefilter_hits,
            report.candidates_nominated,
            report.merges_applied,
        );

        // ── PASS bar: zero false merges — Vehicle/Recipe must both survive. ──
        assert!(
            vehicle_survived && recipe_survived,
            "S3 FAIL: false merge detected — Vehicle survived={vehicle_survived} \
             Recipe survived={recipe_survived}"
        );

        // ── human/Individual pair routing check — the S3 PASS bar itself ──
        //
        // human/Individual must NOT auto-merge (zero shared lemma means the
        // lexical pre-filter cannot fire — this is a hard structural
        // guarantee, always assertable), and — post the chunking + raised-
        // budget fix (`adjudicate_batch`, commit `153c6ca`) — the adjudication
        // call now reliably resolves at this fixture's realistic size (16
        // non-catch-all types -> 25 nominated pairs, split into 3 chunks of
        // <= `identity_verdict::ADJUDICATION_CHUNK_SIZE`). This is asserted as
        // a hard PASS-bar requirement below, not merely reported.
        let human_individual_merged = human_survived != individual_survived;
        assert!(
            !human_individual_merged,
            "S3 FAIL: human/Individual should never AUTO-merge (zero shared \
             lemma -> the lexical pre-filter cannot fire, so a merge here \
             could only be an LLM-authorized row-5 decision, which zero-lexical \
             pairs can never reach per write_gate row 6) — a merge here would \
             indicate the lexical/cosine gate mis-routed this pair."
        );

        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM identity_verdict_audit WHERE group_id = ?1 AND \
                 ((candidate_a = 'human' AND candidate_b = 'Individual') OR \
                  (candidate_a = 'Individual' AND candidate_b = 'human'))",
                libsql::params![group_id],
            )
            .await
            .expect("audit query");
        let audit_n: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("row present")
            .get(0)
            .expect("count col");
        let human_individual_audited = audit_n > 0;

        eprintln!(
            "\n[S3 PASS BAR] human_individual_audited={human_individual_audited} \
             (candidates_nominated={}, merges_applied={})",
            report.candidates_nominated, report.merges_applied,
        );

        // ── PASS bar (spec §8 S3 row, binding assertion): the human/Individual
        // pair MUST reach a resolved, audited LLM-verify-band verdict — proving
        // the chunked adjudication (S3 spike fix, commit `153c6ca`) resolves
        // this pair at realistic fixture scale (25 nominated pairs / 3 chunks),
        // not just in `smoke_one_human_individual_pair_s3`'s 1-pair isolation.
        assert!(
            human_individual_audited,
            "S3 FAIL: human/Individual did not reach an audited LLM-verify-band \
             verdict at realistic fixture scale ({} nominated pairs in chunks of \
             <= {}) — cross-check KREMORY_DEBUG=1 tracing output for \
             `adjudication LLM call failed` / parse failures on the \
             `IdentityVerdictBatch` schema. `smoke_one_human_individual_pair_s3` \
             proves the SAME pair resolves correctly in 1-pair isolation, so a \
             failure here would point at the chunking/budget fix regressing, \
             not the threshold band.",
            report.candidates_nominated,
            10, // mirrors identity_verdict::ADJUDICATION_CHUNK_SIZE (pub(crate),
                // unreachable from this external integration test — same manual-
                // mirror convention as this file's `exact_or_lemma_match`).
        );

        // Also assert the specific verdict shape: PotentialAlias (row 6), never
        // Merge — the write_gate invariant that cosine + LLM alone can never
        // authorise a destructive type merge without a deterministic lexical
        // signal (spec §2.2 row 6; `type_registry_collapse.rs` unit test
        // `llm_verify_zero_lexical_true_verdict_is_potential_alias_row6` is the
        // scripted-LLM mirror of this same invariant).
        let mut rows = graph
            .conn
            .query(
                "SELECT decision FROM identity_verdict_audit WHERE group_id = ?1 AND \
                 ((candidate_a = 'human' AND candidate_b = 'Individual') OR \
                  (candidate_a = 'Individual' AND candidate_b = 'human'))",
                libsql::params![group_id],
            )
            .await
            .expect("audit decision query");
        if let Some(row) = rows.next().await.expect("row read") {
            let decision: String = row.get(0).expect("decision col");
            assert_eq!(
                decision, "potential_alias",
                "S3 FAIL: human/Individual's audited verdict must be \
                 'potential_alias' (write_gate row 6 — zero lexical signal can \
                 never authorise a Merge), got {decision:?}"
            );
        }
    }

    report
}

/// Run the full 6-type polluted-registry fixture once through the SHIPPED
/// (lemma-ON) configuration of `type_registry_collapse`, asserting the S3
/// PASS bar's structural requirements (zero false merges; the
/// human/Individual pair correctly routes to the LLM-verify band).
#[tokio::test]
#[ignore = "S3 spike: requires Ollama in record mode, or a committed cassette in replay mode. \
            Run explicitly: KREMORY_VCR=record cargo test -p kremory --features llm-smoke,test-utils \
            --test it type_registry_collapse_s3_spike:: -- --ignored --nocapture full_fixture_s3"]
async fn full_fixture_s3() {
    // Self-contained diagnostic subscriber (no cross-file `tests/support/`
    // dependency, per this spike's file-scope discipline) — surfaces
    // `tracing::warn!` from `type_registry_collapse.rs`'s adjudication path
    // (e.g. LLM call failure, batch parse failure) that would otherwise be
    // invisible, since this integration test has no subscriber by default.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let report = run_fixture(mode, "fixture", "s3-fixture")
        .await
        .expect("type_registry_collapse must succeed on the full fixture");

    eprintln!(
        "\n── S3 full-fixture summary (shipped lemma-ON config) ──────────────────\n\
         pairs_examined={} lexical_prefilter_hits={} candidates_nominated={} \
         merges_applied={}\n\
         NOTE: only the lemma-ON configuration could be run through the real \
         pass — see `lemma_heuristic_on_off_divergence_and_false_positive_check_s3` \
         for the pure-function lemma-heuristic ON/OFF comparison, and this file's \
         result-envelope notes for why a genuine lemma-OFF pass run does not exist \
         (ASMP-002/spec §4.2: no production toggle for the lemma heuristic today).",
        report.pairs_examined,
        report.lexical_prefilter_hits,
        report.candidates_nominated,
        report.merges_applied,
    );
}

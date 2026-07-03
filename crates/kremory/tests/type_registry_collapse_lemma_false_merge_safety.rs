// ADR-063 test-strategy spec §6.3: the LEMMA FALSE-MERGE SAFETY CORPUS — the
// single highest-consequence untested path in the dream-loop enablement gate
// (`.ai-docs/specs/dream-loop-test-strategy-2026-07-02.md`, finding F3).
//
// Mechanism under test (`type_registry_collapse.rs::names_share_lemma_or_exact`,
// lines 564-571): a naive `strip_trailing_s` singular/plural heuristic. Correct
// for `Organization`/`Organizations` (same concept); WRONG for pairs that
// mechanically strip to the same lemma but are NOT the same real-world
// concept — `Species`/`Specie`, `Physics`/`Physic`, `Customs`/`Custom`,
// `Arms`/`Arm`. `write_gate` row 1 (type_registry_collapse.rs:235) auto-merges
// WITHOUT an LLM call and WITHOUT an audit row (spec §5.2) whenever
// `cosine >= 0.85 AND lexical_compatible` — so if any of these distinct-concept
// pairs' REAL description-cosine also clears 0.85, the pass would silently,
// irreversibly, un-auditedly merge two unrelated concepts.
//
// This is a SAFETY-MARGIN proof, not a routing-correctness check: the unit
// test (`type_registry_collapse.rs::names_share_lemma_or_exact_adversarial_corpus`)
// already proves the LEXICAL signal fires on every one of these pairs — that
// half of row 1's condition is a known, mechanically-confirmed false positive
// per F3. The only thing standing between these pairs and a silent merge is
// whether the SECOND half of the condition (real nomic-embed-text cosine)
// also clears 0.85. This file measures that empirically against a real
// embedder — a hash/deterministic embedder would give spurious cosines and
// invalidate the whole test (spec §4, mocking boundary).
//
// Gated `llm-smoke` + `test-utils`, mirrors `type_registry_collapse_s3_spike.rs`'s
// VCR tier exactly (same cassette-tag/record/replay convention, same
// `RecordReplayEmbedder`/`OllamaEmbedderAdapter` shape). `replay` mode is fully
// offline/deterministic (default gate, spec §9). `record` mode drives a real
// Ollama nomic-embed-text call (and, defensively, a real gemma4:e4b chat call
// only in the unexpected event that a pair routes to the LLM-verify band —
// see spec §6.3's cassette note: "record defensively... covering both the
// expected zero-LLM-call path and a defensive LLM-call path").
//
// PRIMARY assertion (the observable safety outcome): `report.merges_applied
// == 0` for every distinct-concept pair, run in isolation (one pair per
// `type_registry_collapse` call, so a false merge on one pair can never be
// masked by `merges_applied` counting an unrelated legitimate merge in the
// same call).
//
// DESIGNED-TO-FAIL-LOUDLY assertion (F3 resolution): for every pair, if the
// lexical signal fires (verified true for all 6 pairs in the Tier 1 unit
// test) AND the real cosine also clears 0.85, that is EXACTLY the row-1
// auto-merge path firing on a false pair — the test names the falsified
// safety margin explicitly in its failure message rather than just failing
// the primary assertion silently.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::provider::{DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::core::schema::TemporalGraph;

/// VCR mode, selected by `KREMORY_VCR` (mirrors `type_registry_collapse_s3_spike.rs`
/// / `dream_e2e_real_llm.rs`). `record` = live Ollama nomic embeddings (+ a
/// defensive live gemma4:e4b chat call, only if a pair unexpectedly routes to
/// the LLM-verify band). `replay` (or unset) = fully offline.
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

fn chat_cassette_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("site3_lemma_false_merge_safety.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("site3_lemma_false_merge_safety.embeddings.json")
}

// ─── Embedding record/replay (verbatim-shape mirror of
// type_registry_collapse_s3_spike.rs::RecordReplayEmbedder) ──────────────────

/// Description-cosine is a SEMANTIC comparison (spec §4.1/§4.3, §6.3), so it
/// needs REAL embeddings, not a deterministic hash. `record`: delegate to
/// real nomic + capture every text->vector. `replay`: look up offline (loud
/// MISS error, never a silent fallback per this project's parse/observability
/// discipline).
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

/// Real nomic embedder bridge (record mode only) — verbatim mirror of
/// `type_registry_collapse_s3_spike.rs::OllamaEmbedderAdapter`, INCLUDING the
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

/// Build the (chat provider, embedder, model_id) triple for the given VCR
/// mode. A single shared cassette pair covers this whole file (one pair per
/// `type_registry_collapse` call, all against the same two cassettes) — the
/// spec's own cassette note explicitly allows the chat cassette to stay empty
/// on the expected (safety-margin-holds) path, since row 1 auto-merge never
/// invokes the LLM.
async fn build_providers(
    mode: VcrMode,
) -> (
    Arc<RecordReplayChatProvider>,
    Arc<RecordReplayEmbedder>,
    String,
) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // gemma4:e4b — this project's benchmarked deferred-quality dream model
    // (project_kremory_validated_model_findings_2026-06-24). Defensive-only
    // here: expected to never actually be called (row 1 auto-merge has no LLM
    // step), but built anyway so a genuine safety-margin FALSIFICATION (a
    // pair routing to the LLM-verify band instead of auto-merge) doesn't
    // crash on a missing provider.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let chat_cassette = chat_cassette_path();
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
                    "replay cassette must load: {e} — record it via KREMORY_VCR=record \
                     (expected to be a near-empty cassette if the safety margin holds, \
                     since row-1 auto-merge never calls the LLM)"
                )
            }),
        ),
    };

    let emb_path = embedding_cassette_path();
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

/// One distinct-concept collision pair under test: two type names whose
/// naive trailing-s lemma strip collides (verified true for all of these in
/// the Tier 1 unit test, `type_registry_collapse.rs::
/// names_share_lemma_or_exact_adversarial_corpus`) but which name genuinely
/// DIFFERENT real-world concepts, each seeded with a REALISTIC description a
/// Pass-0 discovery LLM would plausibly generate for that concept (spec
/// §6.3 step 1 — "not artificially divergent placeholder text").
struct CollisionPair {
    name_a: &'static str,
    desc_a: &'static str,
    name_b: &'static str,
    desc_b: &'static str,
}

const COLLISION_PAIRS: &[CollisionPair] = &[
    CollisionPair {
        name_a: "Species",
        desc_a: "A group of living organisms that can interbreed and produce \
                 fertile offspring, the basic unit of biological classification.",
        name_b: "Specie",
        desc_b: "Coined money, as opposed to paper currency or credit.",
    },
    CollisionPair {
        name_a: "Physics",
        desc_a: "The natural science that studies matter, motion, energy, and \
                 the fundamental forces governing the physical universe.",
        name_b: "Physic",
        desc_b: "An archaic term for a medicine, remedy, or purgative given \
                 to treat illness.",
    },
    CollisionPair {
        name_a: "Customs",
        desc_a: "A government agency responsible for regulating and taxing \
                 the import and export of goods across a country's borders.",
        name_b: "Custom",
        desc_b: "A traditional or habitual practice followed by a particular \
                 group, community, or society.",
    },
    CollisionPair {
        name_a: "Arms",
        desc_a: "Weapons and military equipment collectively, such as \
                 firearms, ammunition, and armaments used in warfare.",
        name_b: "Arm",
        desc_b: "The upper limb of a human body, extending from the shoulder \
                 to the hand.",
    },
];

/// All descriptions used across the whole corpus, keyed exactly as
/// `type_registry_collapse` embeds them (descriptions, never bare names —
/// spec §4.1) — used to pre-warm the embedding cassette in record mode.
fn all_descriptions() -> Vec<&'static str> {
    COLLISION_PAIRS
        .iter()
        .flat_map(|p| [p.desc_a, p.desc_b])
        .collect()
}

/// Plant exactly ONE collision pair into a fresh in-memory graph and run
/// `type_registry_collapse` in isolation — one pair per call, so a false
/// merge on this pair can never be masked by an unrelated legitimate merge
/// counted in the same `merges_applied` total (smoke-one-shaped isolation,
/// mirrors `type_registry_collapse_s3_spike.rs::smoke_one_human_individual_pair_s3`'s
/// "exactly 2 types -> 1 pair" discipline).
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn run_pair_in_isolation(
    pair: &CollisionPair,
    chat: &RecordReplayChatProvider,
    embedder: &RecordReplayEmbedder,
    model_id: &str,
) -> (kremory::core::dream::TypeRegistryCollapseReport, f32) {
    use kremory::core::dream::{type_registry_collapse, TypeRegistryCollapseParams};

    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open in-memory graph");
    let gid = format!("lemma-safety-{}", pair.name_a.to_lowercase());

    // Deliberately skip `ensure_default_types_seeded` — as in the S3 spike's
    // `smoke_one_human_individual_pair_s3`, this pass only excludes id=0
    // (spec §4.0); a genuine ONE-PAIR isolation must plant exactly 2 types.
    graph
        .conn
        .execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES \
             (?1, 11, ?2, ?3)",
            libsql::params![gid.clone(), pair.name_a, pair.desc_a],
        )
        .await
        .expect("insert type A");
    graph
        .conn
        .execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES \
             (?1, 12, ?2, ?3)",
            libsql::params![gid.clone(), pair.name_b, pair.desc_b],
        )
        .await
        .expect("insert type B");

    // Measure the real cosine independently (diagnostic, spec §6.3 step 3)
    // so the failure message can name the exact falsified margin even if
    // `type_registry_collapse`'s internal routing masked it somehow.
    let emb_a = kremory::EmbeddingProvider::embed(embedder, pair.desc_a)
        .await
        .unwrap_or_else(|e| panic!("embed failed for {:?}: {e}", pair.desc_a));
    let emb_b = kremory::EmbeddingProvider::embed(embedder, pair.desc_b)
        .await
        .unwrap_or_else(|e| panic!("embed failed for {:?}: {e}", pair.desc_b));
    let cosine = cosine_similarity(&emb_a, &emb_b);

    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = Arc::new(PassthroughEmbedder {
        pairs: vec![
            (pair.desc_a.to_string(), emb_a),
            (pair.desc_b.to_string(), emb_b),
        ],
    });

    let report = type_registry_collapse(
        chat,
        TypeRegistryCollapseParams {
            conn: &graph.conn,
            group_id: &gid,
            embedder: Some(embedder_dyn.as_ref()),
            model_id,
        },
    )
    .await
    .unwrap_or_else(|e| {
        panic!(
            "type_registry_collapse must succeed on pair {}/{}: {e}",
            pair.name_a, pair.name_b
        )
    });

    (report, cosine)
}

/// Thin `DynEmbeddingProvider` adapter that serves pre-fetched vectors by
/// exact text match (mirrors `type_registry_collapse.rs::tests::
/// MockEmbeddingProvider`'s lookup shape) — lets `run_pair_in_isolation`
/// reuse the SAME already-fetched-through-VCR vectors for the pass's own
/// per-slot `embed_dyn` calls, rather than fetching each description twice
/// (once for the diagnostic cosine, once inside the pass) against a shared
/// mutable cassette cache.
struct PassthroughEmbedder {
    pairs: Vec<(String, Vec<f32>)>,
}

impl DynEmbeddingProvider for PassthroughEmbedder {
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a>,
    > {
        let found = self
            .pairs
            .iter()
            .find(|(k, _)| k == text)
            .map(|(_, v)| v.clone());
        Box::pin(async move {
            found.ok_or_else(|| {
                kremory::CoreError::Embedding(format!(
                    "PassthroughEmbedder MISS for {text:?} — pair not pre-fetched"
                ))
            })
        })
    }
    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

/// THE safety-invariant proof (spec §6.3): run every distinct-concept
/// collision pair through the real `type_registry_collapse` pass, in
/// isolation, against a real nomic embedder, and assert (a) nothing was
/// silently merged, and (b) name the measured cosine for each pair so the
/// safety margin is documented, not just asserted blind.
#[tokio::test]
#[ignore = "lemma false-merge safety corpus: requires Ollama in record mode, or a \
            committed cassette in replay mode. Run explicitly: KREMORY_VCR=record \
            cargo test -p kremory --features llm-smoke,test-utils \
            --test type_registry_collapse_lemma_false_merge_safety -- --ignored \
            --nocapture lemma_false_merge_safety_corpus"]
async fn lemma_false_merge_safety_corpus() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let (provider, embedder, chat_model) = build_providers(mode).await;

    if mode == VcrMode::Record {
        for text in all_descriptions() {
            let _ = kremory::EmbeddingProvider::embed(&*embedder, text)
                .await
                .unwrap_or_else(|e| panic!("prewarm embed failed for {text:?}: {e}"));
        }
    }

    let mut results: Vec<(&'static str, &'static str, f32, usize)> = Vec::new();
    let mut falsified: Vec<String> = Vec::new();

    for pair in COLLISION_PAIRS {
        let (report, cosine) = run_pair_in_isolation(pair, &provider, &embedder, &chat_model).await;

        eprintln!(
            "[lemma-safety] {}/{} cosine={cosine:.4} merges_applied={} \
             pairs_examined={} candidates_nominated={}",
            pair.name_a,
            pair.name_b,
            report.merges_applied,
            report.pairs_examined,
            report.candidates_nominated,
        );

        results.push((pair.name_a, pair.name_b, cosine, report.merges_applied));

        // DESIGNED-TO-FAIL-LOUDLY (spec §6.3 step 3): the lexical signal is
        // KNOWN to fire on every one of these pairs (Tier 1 unit test proved
        // this). If the real cosine ALSO clears 0.85, row 1's auto-merge
        // condition is fully satisfied on a genuinely-distinct-concept pair —
        // name that falsification explicitly.
        if cosine >= crate::type_registry_collapse_primary_cosine() {
            falsified.push(format!(
                "{}/{} cosine={cosine:.4} >= 0.85 threshold — lexical signal is KNOWN \
                 to fire on this pair (Tier 1 unit test), so BOTH halves of write_gate \
                 row 1's auto-merge condition are satisfied. The safety margin \
                 (lemma-naive-collision x real-world-low-cosine) is FALSIFIED for \
                 this pair.",
                pair.name_a, pair.name_b
            ));
        }

        // PRIMARY assertion, per-pair (isolated call — cannot be masked by
        // an unrelated legitimate merge in the same report): nothing was
        // silently merged.
        assert_eq!(
            report.merges_applied, 0,
            "SAFETY VIOLATION: type_registry_collapse silently auto-merged the \
             distinct-concept pair {}/{} (cosine={cosine:.4}) with NO LLM \
             involvement and NO audit row (spec §5.2) — the lemma false-merge \
             safety margin has been falsified. This margin is the basis on which \
             F3 was resolved (lemma-on-only, no `exact_only` toggle) and \
             `include_type_registry_collapse` was enabled by default (2026-07-03) \
             — a falsification here means that enablement must be revisited.",
            pair.name_a, pair.name_b,
        );
    }

    eprintln!(
        "\n── lemma false-merge safety corpus summary ─────────────────────────────\n\
         {results:#?}\n\
         falsified margins: {falsified:?}"
    );

    assert!(
        falsified.is_empty(),
        "F3 safety margin FALSIFIED for {} pair(s): {falsified:#?} — the lemma-naive \
         collision heuristic combined with real embedding cosine is NOT safe as \
         currently shipped; this margin is what justified enabling \
         `include_type_registry_collapse` by default (2026-07-03) with lemma-on-only \
         and no `exact_only` toggle, so a falsification here requires reverting that \
         default or changing the lemma heuristic.",
        falsified.len(),
    );

    if mode == VcrMode::Record {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
        embedder.flush();
    }
}

// Small crate-local helper re-exposing the pass's own primary threshold
// constant, avoiding a second hardcoded 0.85 magic number in this file (the
// production constant is `pub(crate)`, unreachable from this external
// integration test — same manual-constant convention as
// `type_registry_collapse_s3_spike.rs`'s other mirrored constants).
fn type_registry_collapse_primary_cosine() -> f32 {
    0.85
}

/// Verbatim mirror of `anti_redundancy::cosine` (`pub(crate)`, unreachable
/// from this external integration test — same manual-mirror convention this
/// file already follows for the chat model id / lexical heuristic). Assumes
/// unit-normalized inputs are NOT guaranteed (unlike the production
/// `TypeSlot` embeddings, which nomic returns pre-normalized in practice) —
/// computes the full cosine formula rather than a bare dot product, so this
/// diagnostic measurement is correct regardless of embedder normalization.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

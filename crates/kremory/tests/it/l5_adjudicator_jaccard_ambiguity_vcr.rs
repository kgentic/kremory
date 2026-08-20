// TD-225 DoD (c) — THE ADVERSARIAL VCR FIXTURE FOR BOTH SIDES OF THE
// JACCARD-0.500 AMBIGUITY.
//
// ## Why this file exists
//
// TD-225 fixed L5 canonicalize by demoting `names_lexically_compatible` from
// DECIDER to NOMINATOR and routing survivors through the shared ADR-063
// `write_gate`. The measured recovery (54.1 → 64.1 nDCG@10, 2026-08-20) proved
// the REJECT half against a real model: all 6 audit rejects were hypernym
// collapses, `pottery class` → `pottery` among them.
//
// It proved NOTHING about the MERGE half. LoCoMo conv0 contains no abbreviated
// person-name pair, so every one of the 7 nominations in that run was a
// hypernym. The register states the gap plainly: *"The only evidence that a real
// model preserves abbreviated person names is the fast-tier unit tests plus one
// scripted mock."*
//
// That evidence is insufficient, and the reason is structural rather than a
// matter of degree. `adjudicate.rs`'s two unit tests
// (`hypernym_collapse_is_rejected_when_llm_says_not_same`,
// `abbreviated_person_name_still_merges_when_llm_says_same`) hand `decide()` a
// SCRIPTED verdict. They prove the gate ROUTES a `false` to Reject and a
// confident `true` to Merge — real coverage, and not the question here. The
// open question is whether a real model, given the real prompt, actually
// RETURNS `false` for a hypernym and `true` for an abbreviation. A scripted
// verdict cannot answer that: the author writes both the input and the
// expectation, so the test passes whatever the model would really do.
//
// This is the LLM-seam tier of the test pyramid — real model, recorded once,
// replayed deterministically, over an ADVERSARIAL corpus. The seam carries
// BREADTH (both sides of the ambiguity, several pairs each); the benchmark
// carries depth.
//
// ## Why this is the top residual risk, and not a nice-to-have
//
// The two classes are LEXICALLY INDISTINGUISHABLE. Both sit at token-Jaccard
// EXACTLY 0.500, and it is worth spelling out why, because the arithmetic is
// not obvious from the names alone — `names_lexically_compatible`
// (`disambiguation/lexical.rs:99-106`) DROPS tokens shorter than 2 characters
// before comparing:
//
// | pair | significant tokens | ∩ / ∪ | Jaccard |
// |---|---|---|---|
// | `pottery class` / `pottery`     | {pottery, class} vs {pottery}   | 1/2 | 0.500 |
// | `lgbtq community` / `community` | {lgbtq, community} vs {community} | 1/2 | 0.500 |
// | `alice j` / `alice johnson`     | {alice} vs {alice, johnson}     | 1/2 | 0.500 |
// | `ria patel` / `ria`             | {ria, patel} vs {ria}           | 1/2 | 0.500 |
//
// The `j` in `alice j` is discarded as insignificant, which is what lands that
// pair on 0.500 rather than 0.333. **No threshold separates these rows** — that
// is precisely why raising `L4_LEXICAL_JACCARD_MIN` 0.5 → 0.6 was measured
// (−0.4 nDCG, nearly a full recovery) and then REVERTED: it also destroyed the
// abbreviation case, dropping corpus recall 0.568 → 0.263. A fourth per-pair
// deterministic discriminator is banned by `20e1f4e3`.
//
// So the ONLY thing standing between `alice johnson` and destruction is the
// model's semantic judgement — and a regression that lost it would be
// INVISIBLE to the LoCoMo benchmark, which never exercises the class. This
// file is the regression guard the benchmark cannot be.
//
// ## Design: why synthetic embeddings, and why that is not cheating
//
// The pass under test has two stages: NOMINATE (cosine > 0.8 AND lexically
// compatible) then ADJUDICATE (LLM + `write_gate`). Only the second is under
// test here, so the first is pinned rather than measured: each pair is seeded
// with near-identical synthetic vectors (cos ≈ 0.9988) that clear the
// threshold by construction.
//
// This is deliberate and CONSERVATIVE — it guarantees every pair actually
// reaches the adjudicator. Real embeddings would add an uncontrolled second
// variable: a pair falling below 0.8 would never be nominated, `merges_applied`
// would read 0, and the REJECT assertions would pass for entirely the wrong
// reason. That is the vacuity trap this project has been bitten by before
// (TD-224: a graph-integrity harness that passed because it was handed an EMPTY
// graph).
//
// Contrast `type_registry_collapse_lemma_false_merge_safety.rs`, which uses
// REAL nomic embeddings — correctly, because there the embedding IS the safety
// margin under test. Here it is a precondition, so it is pinned.
//
// Everything downstream of nomination is real: real descriptions, the real
// adjudication prompt, a real model (recorded), the real `write_gate`, the real
// destructive merge, the real audit table.
//
// ## Non-vacuity, enforced rather than hoped for
//
// `write_gate` is FAIL-CLOSED at this site by design (divergence 1:
// `merge_threshold = f32::INFINITY` makes row 1 unreachable). A missing
// cassette entry, a parse failure, or a timed-out call therefore yields
// `Reject` — which would make every REJECT assertion in this file pass while
// the adjudicator was never consulted at all. All-reject is INERT, not correct.
//
// Three guards make that unrepresentable, and the first two are asserted
// BEFORE the outcome is checked:
//
//   1. an `identity_verdict_audit` row must exist for the pair. THIS IS THE
//      LOAD-BEARING ONE, verified by falsification rather than by reading:
//      blanking every recorded response in the cassette trips exactly this
//      assert. A verdict-less `Reject` writes NO audit row at all
//      (`apply_adjudicated_merges`: "Reject — audit row when a verdict exists,
//      nothing otherwise"), so a failed call, a MISS or a parse failure shows
//      up here as ZERO rows.
//   2. its `llm_is_same` must be NON-NULL. Defence-in-depth, and — as shipped
//      today — UNREACHABLE, for the reason in (1): the only way to get a row
//      at this site is to have had a verdict. It is kept deliberately, one
//      line, because L5 already writes `Reject` rows MORE widely than Site #3
//      does, so a future widening to verdict-less rejects would create exactly
//      the NULL-column row this catches — and would otherwise silently convert
//      guard 1 from a tripwire into a no-op.
//
//      An earlier version of this comment called (2) the load-bearing guard.
//      That was asserted, never tested, and it was wrong.
//
//   3. the corpus itself spans BOTH decisions. An adjudicator stuck on `false`
//      fails the merge cases; one stuck on `true` fails the reject cases.
//      Neither a rubber stamp nor an inert pass can be green here.
//
// ## Running it
//
//   record (needs Ollama):
//     KREMORY_VCR=record cargo nextest run -p kremory \
//       --features llm-smoke,test-utils -E 'test(/l5_adjudicator_jaccard/)' \
//       --run-ignored all
//   replay (offline, deterministic — the committed default):
//     cargo nextest run -p kremory --features llm-smoke,test-utils \
//       -E 'test(/l5_adjudicator_jaccard/)' --run-ignored all

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use kremory::core::canonicalization::{
    canonicalize_surface_forms_with_embedder, CanonicalizeSurfaceFormsParams, L5Adjudicator,
    L5_CANONICALIZATION_THRESHOLD,
};
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::provider::{ArcChatProvider, RecordReplayChatProvider};
use kremory::core::schema::TemporalGraph;

/// Match `TemporalGraph::open_in_memory()`'s default width.
const DIM: usize = 384;

/// Rotation between the two seeded vectors, in radians. `cos(0.05) ≈ 0.99875`,
/// comfortably above [`L5_CANONICALIZATION_THRESHOLD`] (0.8, strict `>`), so
/// every pair in the corpus is nominated by construction. Asserted rather than
/// assumed — see `synthetic_pair_clears_the_nomination_threshold`.
const PAIR_ROTATION_RAD: f32 = 0.05;

// ─── VCR plumbing (mirrors `type_registry_collapse_lemma_false_merge_safety.rs`) ──

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
        .join("l5_adjudicator_jaccard_ambiguity.json")
}

/// Build the (chat provider, model id) pair for `mode`. No embedder here — this
/// file pins the nominator with synthetic vectors (see the header), so nothing
/// in it ever calls a real embedding model.
fn build_provider(mode: VcrMode) -> (Arc<RecordReplayChatProvider>, String) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    // gemma4:e4b — this project's benchmarked deferred-quality dream model
    // (`project_kremory_validated_model_findings_2026-06-24`), and the model
    // the measured TD-225 recovery run actually used. Recording against a
    // different model would not test the shipped configuration.
    let chat_model = crate::helpers::chat_model::chat_model_or("gemma4:e4b");
    let cassette = chat_cassette_path();

    let provider: Arc<RecordReplayChatProvider> = match mode {
        VcrMode::Record => {
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
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
                    "replay cassette must load: {e} — re-record via KREMORY_VCR=record. \
                     This file's whole purpose is that a real model's verdicts are pinned; \
                     there is deliberately no live fallback and no mock arm."
                )
            },
        )),
    };

    (provider, chat_model)
}

// ─── The adversarial corpus ───────────────────────────────────────────────────

/// What the adjudicator is expected to decide for a pair — and, crucially, what
/// the graph must look like afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    /// A CATEGORY and an INSTANCE of it. Merging destroys the distinguishing
    /// token and, at benchmark scale, cost 9.9 nDCG@10.
    Reject,
    /// An abbreviated and a full form of the SAME person. Refusing to merge
    /// leaves the graph fragmented — the failure mode the reverted threshold
    /// fix would have caused (corpus recall 0.568 → 0.263).
    Merge,
}

/// One adversarial pair. `keeper` is the name that must SURVIVE: L5 selects the
/// keeper by longer description, so the descriptions below are written to make
/// the keeper deterministic, and the test asserts on identity rather than on a
/// bare count (a merge in the wrong direction would satisfy `merges_applied ==
/// 1` while destroying the full name — see TD-226).
struct AmbiguousPair {
    label: &'static str,
    expected: Expected,
    keeper: &'static str,
    keeper_description: &'static str,
    loser: &'static str,
    loser_description: &'static str,
}

/// Descriptions are written the way a real extraction LLM would phrase them
/// (`type_registry_collapse_lemma_false_merge_safety.rs` calls this out
/// explicitly: "not artificially divergent placeholder text"). Divergent
/// placeholders would make the discrimination trivially easy and the test
/// would stop measuring anything.
const CORPUS: &[AmbiguousPair] = &[
    // ── REJECT side: category vs instance. Both drawn from the 10 real
    // destructive merges measured on 2026-08-19; 8 of those 10 sat at Jaccard
    // exactly 0.500. ────────────────────────────────────────────────────────
    AmbiguousPair {
        label: "hypernym/pottery",
        expected: Expected::Reject,
        keeper: "pottery",
        keeper_description: "The craft of shaping and firing clay into vessels and \
                             objects; a subject Melanie has been interested in for years.",
        loser: "pottery class",
        loser_description: "A weekly evening course Melanie signed up for in June, \
                            held at the community centre.",
    },
    AmbiguousPair {
        label: "hypernym/community",
        expected: Expected::Reject,
        keeper: "community",
        keeper_description: "A group of people living in the same place, or sharing a \
                             common characteristic, interest or identity; the general \
                             social unit Caroline refers to when talking about local \
                             organising.",
        loser: "lgbtq community",
        loser_description: "The specific group of lesbian, gay, bisexual, transgender \
                            and queer people that Caroline organises events for.",
    },
    // ── MERGE side: abbreviated vs full person name. THE UNTESTED HALF —
    // structurally identical to the rows above, decidable only semantically. ──
    AmbiguousPair {
        label: "abbreviation/alice",
        expected: Expected::Merge,
        keeper: "alice johnson",
        keeper_description: "A software engineer who joined the team in March and \
                             leads the platform migration project.",
        loser: "alice j",
        loser_description: "Engineer on the platform migration.",
    },
    AmbiguousPair {
        label: "abbreviation/ria",
        expected: Expected::Merge,
        keeper: "ria patel",
        keeper_description: "Melanie's cousin, who lives in Bristol and recently \
                             started training for a marathon.",
        loser: "ria",
        loser_description: "Melanie's cousin in Bristol.",
    },
];

// ─── Graph seeding ────────────────────────────────────────────────────────────

/// Unit vector with 1.0 at `axis`.
fn axis_unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[axis] = 1.0;
    v
}

/// `[cos θ, sin θ, 0, …]` — unit-norm by construction, cosine with
/// `axis_unit(0)` is exactly `cos θ`.
fn rotated_unit(theta_rad: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[0] = theta_rad.cos();
    v[1] = theta_rad.sin();
    v
}

/// Args-as-object per TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
/// `graph` stays a lead positional param, mirroring the production convention in
/// `canonicalization::CanonicalizeSurfaceFormsParams`.
struct SeedEntity<'a> {
    id: &'a str,
    group_id: &'a str,
    description: &'a str,
    embedding: &'a [f32],
}

async fn seed_entity(graph: &TemporalGraph, params: SeedEntity<'_>) {
    let SeedEntity {
        id,
        group_id,
        description,
        embedding,
    } = params;
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0u32,
            properties: serde_json::json!({ "name": id, "description": description }),
            group_id: Some(group_id),
        })
        .await
        .unwrap_or_else(|e| panic!("insert entity {id:?}: {e}"));
    graph
        .set_entity_embedding(id, embedding)
        .await
        .unwrap_or_else(|e| panic!("set embedding for {id:?}: {e}"));
}

/// What the audit table recorded for one adjudicated pair.
#[derive(Debug)]
struct AuditRow {
    decision: String,
    /// `None` means the `llm_is_same` column is NULL — i.e. NO verdict was
    /// returned or parsed. See the header: this is the guard that separates a
    /// genuine `false` from a silent fail-closed.
    llm_is_same: Option<bool>,
    llm_confidence: Option<f64>,
    llm_reasoning: Option<String>,
}

async fn read_audit_rows(graph: &TemporalGraph, group_id: &str) -> Vec<AuditRow> {
    let mut rows = graph
        .conn
        .query(
            "SELECT decision, llm_is_same, llm_confidence, llm_reasoning \
             FROM identity_verdict_audit \
             WHERE group_id = ?1 AND site = 'l5_canonicalize' ORDER BY id",
            libsql::params![group_id],
        )
        .await
        .expect("query identity_verdict_audit");

    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("audit row") {
        out.push(AuditRow {
            decision: row.get::<String>(0).expect("decision column"),
            llm_is_same: row.get::<Option<bool>>(1).expect("llm_is_same column"),
            llm_confidence: row.get::<Option<f64>>(2).expect("llm_confidence column"),
            llm_reasoning: row.get::<Option<String>>(3).expect("llm_reasoning column"),
        });
    }
    out
}

/// Does an entity still exist in the group? The observable outcome of a
/// destructive merge is that the loser is GONE.
async fn entity_exists(graph: &TemporalGraph, id: &str, group_id: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT 1 FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![id, group_id],
        )
        .await
        .expect("query entities");
    rows.next().await.expect("entity row").is_some()
}

/// Run ONE pair through the real adjudicated L5 path, in its own graph.
///
/// Isolation is load-bearing, not tidiness: with several pairs in one graph a
/// wrong decision on one could be masked by a correct one on another in the
/// same `merges_applied` total, and — worse — they would be batched into a
/// single LLM call, so the pairs would become context for each other. The
/// `l5-context-probe` (2026-08-20) measured that giving this adjudicator more
/// context makes it WORSE (100% → 36% on hypernyms), so one pair per call is
/// also the configuration the shipped benchmark actually exercised.
async fn adjudicate_one_pair(
    pair: &AmbiguousPair,
    provider: &Arc<RecordReplayChatProvider>,
    model_id: &str,
) -> (usize, Vec<AuditRow>, bool, bool) {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open in-memory graph");
    let group_id = format!("l5-ambiguity-{}", pair.label.replace('/', "-"));

    seed_entity(
        &graph,
        SeedEntity {
            id: pair.keeper,
            group_id: &group_id,
            description: pair.keeper_description,
            embedding: &axis_unit(0),
        },
    )
    .await;
    seed_entity(
        &graph,
        SeedEntity {
            id: pair.loser,
            group_id: &group_id,
            description: pair.loser_description,
            embedding: &rotated_unit(PAIR_ROTATION_RAD),
        },
    )
    .await;

    let llm = ArcChatProvider::new(provider.clone() as Arc<_>);
    let report = canonicalize_surface_forms_with_embedder(
        &graph,
        CanonicalizeSurfaceFormsParams {
            group_id: &group_id,
            threshold: L5_CANONICALIZATION_THRESHOLD,
            embedder: None,
            adjudicator: Some(L5Adjudicator {
                llm: &llm,
                model_id,
            }),
        },
    )
    .await
    .unwrap_or_else(|e| panic!("canonicalize must succeed for {}: {e}", pair.label));

    let audit = read_audit_rows(&graph, &group_id).await;
    let keeper_alive = entity_exists(&graph, pair.keeper, &group_id).await;
    let loser_alive = entity_exists(&graph, pair.loser, &group_id).await;

    (report.merges_applied, audit, keeper_alive, loser_alive)
}

// ─── Tier 1: the pinned nominator is itself pinned ────────────────────────────

/// Guards the guard. If `PAIR_ROTATION_RAD` ever drifted above the L5 threshold
/// the corpus would stop being nominated, every pair would report
/// `merges_applied == 0`, and the two REJECT cases would pass vacuously.
///
/// Deterministic and instant — no Ollama, no cassette, not `#[ignore]`d. It
/// still sits behind this file's `llm-smoke` cfg, so it does NOT run in the
/// standard `content-search,test-utils` gate. That is the correct placement
/// rather than a compromise: the only thing it can catch is drift in THIS
/// file's corpus, and the standard gate never compiles this file, so there is
/// nothing here for it to guard.
#[test]
fn synthetic_pair_clears_the_nomination_threshold() {
    let cosine = PAIR_ROTATION_RAD.cos();
    assert!(
        cosine > L5_CANONICALIZATION_THRESHOLD,
        "seeded pair cosine {cosine} must exceed L5's threshold {L5_CANONICALIZATION_THRESHOLD} \
         (STRICT >), or no pair in this file is ever nominated and the REJECT assertions \
         pass vacuously"
    );
}

/// The corpus's `keeper`/`loser` field names are a CLAIM about which entity
/// survives, and L5 decides that by longer description — not by field name. A
/// first recording caught this file getting it backwards on `hypernym/community`
/// (the log said `keeper_id=lgbtq community` while the struct said `community`).
/// The test still passed, because a REJECT pair merges in neither direction, so
/// nothing would ever have flagged the lie.
///
/// Pinning it here rather than fixing the one row makes the whole class
/// unrepresentable, and matters most for the MERGE rows, where the direction is
/// asserted downstream: a mis-stated keeper there would silently invert what
/// `keeper_alive` means (TD-226).
#[test]
fn corpus_keeper_is_the_one_l5_will_actually_choose() {
    for pair in CORPUS {
        assert!(
            pair.keeper_description.len() > pair.loser_description.len(),
            "{}: `keeper` is the entity L5 will SELECT, and it selects by longer \
             description — but `{}` ({} chars) is not longer than `{}` ({} chars), so \
             this row's field names describe a merge direction that will not happen",
            pair.label,
            pair.keeper,
            pair.keeper_description.len(),
            pair.loser,
            pair.loser_description.len(),
        );
    }
}

// ─── Tier 2: the LLM seam ─────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "L5 adjudicator adversarial corpus: needs Ollama in record mode, or the \
            committed cassette in replay mode. Run: cargo nextest run -p kremory \
            --features llm-smoke,test-utils -E 'test(/l5_adjudicator_jaccard/)' \
            --run-ignored all"]
async fn jaccard_500_ambiguity_is_decided_semantically_not_lexically() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let (provider, model_id) = build_provider(mode);

    let mut failures: Vec<String> = Vec::new();
    let mut observed: Vec<(&'static str, Expected, String, Option<bool>)> = Vec::new();

    for pair in CORPUS {
        let (merges, audit, keeper_alive, loser_alive) =
            adjudicate_one_pair(pair, &provider, &model_id).await;

        // ── NON-VACUITY GUARD 1: the adjudicator ran at all. ────────────────
        //
        // The two causes of zero rows are OPPOSITES and the message must not
        // pick one: `merges` disambiguates them, so it is quoted. This was
        // found by falsifying the test rather than by reading it — an earlier
        // message named only the nomination cause, which is the WRONG
        // diagnosis for the regression this file exists to catch (reverting to
        // `adjudicator: None` nominates and merges, it just never adjudicates),
        // and it would have pointed the next reader at the cosine while a
        // destructive merge sat in the same output.
        assert_eq!(
            audit.len(),
            1,
            "{}: expected exactly ONE identity_verdict_audit row (one nominated pair, \
             one decision), got {}. merges_applied={merges}, loser_alive={loser_alive}.\n\
             • merges=0 ⇒ EITHER the pair was never nominated (seeded cosine or the \
             lexical gate changed) OR it was nominated and the LLM call failed / \
             MISSed the cassette / failed to parse, so write_gate fell through to its \
             fail-closed Reject and wrote no row. Both make every REJECT assertion \
             here vacuous. Check the `kremory::canonicalization::adjudicate` warn log \
             to tell them apart.\n\
             • merges>0 ⇒ the pair WAS merged with NO adjudication — the \
             pre-TD-225 deterministic path is back (an `adjudicator: None` call site, \
             or a facade that stopped supplying one). That IS the −9.9 nDCG defect, \
             live.",
            pair.label,
            audit.len()
        );
        let row = &audit[0];

        // ── NON-VACUITY GUARD 2 — defence-in-depth, and UNREACHABLE as
        // shipped. A verdict-less decision writes no audit row at all, so
        // guard 1 above catches that case first (measured: blanking every
        // cassette response trips guard 1, never this). Kept because L5 writes
        // `Reject` rows more widely than Site #3 does, so a future widening to
        // verdict-less rejects would produce exactly this NULL-column row —
        // and would silently turn guard 1 into a no-op if nothing were
        // watching here. ────────────────────────────────────────────────────
        assert!(
            row.llm_is_same.is_some(),
            "{}: an identity_verdict_audit row exists but llm_is_same is NULL — a \
             decision was recorded WITHOUT a model verdict, i.e. the fail-closed \
             default is being audited as though it were an adjudication. decision={:?}. \
             If the audit-write policy was widened deliberately, guard 1 above no \
             longer detects a silent adjudicator and this file needs rethinking, not \
             relaxing.",
            pair.label,
            row.decision
        );

        eprintln!(
            "[l5-ambiguity] {:<22} expected={:?} decision={:<16} llm_is_same={:?} \
             conf={:?} merges={merges} keeper_alive={keeper_alive} loser_alive={loser_alive}\n\
             {:>27}reasoning: {}",
            pair.label,
            pair.expected,
            row.decision,
            row.llm_is_same,
            row.llm_confidence,
            "",
            row.llm_reasoning.as_deref().unwrap_or("<none>"),
        );
        observed.push((pair.label, pair.expected, row.decision.clone(), row.llm_is_same));

        // ── The outcome assertions, on the GRAPH rather than on a counter. ──
        match pair.expected {
            Expected::Reject => {
                if merges != 0 || !loser_alive {
                    failures.push(format!(
                        "{}: DESTRUCTIVE MERGE OF A CATEGORY INTO AN INSTANCE. \
                         `{}` was merged into `{}` (merges={merges}, loser_alive={loser_alive}). \
                         This is the exact class of merge that cost 9.9 nDCG@10 on \
                         2026-08-19 — 8 of those 10 merges sat at Jaccard 0.500, as this \
                         pair does. model said is_same={:?} conf={:?}: {}",
                        pair.label,
                        pair.loser,
                        pair.keeper,
                        row.llm_is_same,
                        row.llm_confidence,
                        row.llm_reasoning.as_deref().unwrap_or("<none>"),
                    ));
                }
            }
            Expected::Merge => {
                if merges != 1 {
                    failures.push(format!(
                        "{}: the abbreviated form `{}` was NOT merged into `{}` \
                         (merges={merges}, decision={:?}). This is the half LoCoMo conv0 \
                         cannot see: raising L4_LEXICAL_JACCARD_MIN to 0.6 produced exactly \
                         this failure and dropped corpus recall 0.568 → 0.263. \
                         model said is_same={:?} conf={:?}: {}",
                        pair.label,
                        pair.loser,
                        pair.keeper,
                        row.decision,
                        row.llm_is_same,
                        row.llm_confidence,
                        row.llm_reasoning.as_deref().unwrap_or("<none>"),
                    ));
                }
                // Direction matters independently of the count (TD-226): a
                // merge that keeps the ABBREVIATION and destroys the full name
                // still reports `merges_applied == 1`.
                if !keeper_alive {
                    failures.push(format!(
                        "{}: merge ran in the WRONG DIRECTION — the full name `{}` was \
                         destroyed and the abbreviated `{}` survived. Keeper selection is \
                         by longer description; see TD-226.",
                        pair.label, pair.keeper, pair.loser,
                    ));
                }
            }
        }
    }

    // Flush BEFORE the verdict assertions, never after. The cassette is an
    // artefact OF the run, not a reward for passing it: if a pair is decided
    // wrongly the recording is exactly what you need in order to read the
    // model's reasoning, and a flush placed after the asserts would throw it
    // away at the only moment it mattered.
    if mode == VcrMode::Record {
        provider
            .flush()
            .expect("provider.flush() must succeed in KREMORY_VCR=record mode");
    }

    // ── NON-VACUITY GUARD 3: the corpus spans both decisions, so neither an
    // always-`false` adjudicator (inert) nor an always-`true` one (rubber
    // stamp) can be green. Asserted explicitly so the property is checked
    // rather than merely implied by the corpus contents. ────────────────────
    let saw_true = observed.iter().any(|(_, _, _, s)| *s == Some(true));
    let saw_false = observed.iter().any(|(_, _, _, s)| *s == Some(false));
    assert!(
        saw_true && saw_false,
        "the adjudicator returned the SAME verdict for every pair in a corpus built to \
         span both (saw_true={saw_true}, saw_false={saw_false}). All-false is an INERT \
         adjudicator, all-true is a rubber stamp; both are failures even where the \
         per-pair outcomes happen to line up. observed={observed:#?}"
    );

    assert!(
        failures.is_empty(),
        "\n{} of {} adversarial pairs decided WRONGLY by the real adjudicator:\n\n{}\n\n\
         Both classes sit at token-Jaccard EXACTLY 0.500, so no lexical threshold can \
         separate them and a fourth per-pair deterministic discriminator is banned \
         (20e1f4e3). A failure here means the SEMANTIC decision has regressed — via the \
         model, the prompt, or the gate — and TD-225's fix no longer holds.\n",
        failures.len(),
        CORPUS.len(),
        failures.join("\n\n"),
    );
}

//! v0.1.8 ADR-035 — `with_facts` + `skip_extraction` integration tests.
//!
//! Exercises the Path X end-to-end mechanism: caller pre-extracted triples
//! pinned at engine layer via `SourceParams.pre_pinned_facts`, with the
//! `try_insert_fact` helper silently swallowing LLM Phase 2 duplicates so
//! caller wins via pre-write ordering. Also covers the `skip_extraction()`
//! opt-out path that bails after caller-pin without invoking the extractor.
//!
//! Tests use `MockChatProvider::null()` + `NullEmbeddingProvider` — the
//! mechanism under test is the substrate's storage + dedup behavior, not the
//! LLM extractor itself (LLM behavior is covered by `llm_integration.rs`).
//!
//! Per [[treat-cause-not-symptom]] + [[observability-first-class]] cardinal
//! failure mode #9: counter assertions go through labeled
//! `kremory.with_facts.*` metrics so post-recovery successes are
//! distinguishable from first-attempt successes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use chrono::{Duration, Utc};
use kremory::memory::types::StructuredFact;
use kremory::{DynEmbeddingProvider, Memory, Namespace, SearchOpts};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_with_facts_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

async fn open_with_ns(tag: &str) -> Memory {
    Memory::open(unique_db_path(tag))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("with_facts_tests"))
        .await
        .expect("builder should succeed")
}

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Sum a labeled counter across all label-value variants matching `metric_name`.
/// Used per [[observability-first-class]] cardinal failure mode #9 — labeled
/// counters can have multiple variants (e.g. `axis=caller_vs_llm`,
/// `axis=intra_caller_set`); a single `.find()` would under-report.
///
/// Consumes the snapshot (Snapshot::into_vec takes self). Callers must
/// take a fresh snapshot via `snapshotter.snapshot()` per query.
fn find_counter_labeled(snapshot: metrics_util::debugging::Snapshot, name: &str) -> u64 {
    snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(key, _unit, _desc, value)| {
            if key.key().name() == name {
                if let metrics_util::debugging::DebugValue::Counter(v) = value {
                    Some(v)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .sum()
}

/// Vocabulary-neutral three-triple fixture per substrate-purity boundary
/// (Quinn L-02: substrate tests must not leak consumer vocabulary like
/// "frontmatter" / "adr-XXX" into shared fixtures).
fn three_fact_fixture() -> Vec<StructuredFact> {
    let t0 = Utc::now() - Duration::hours(1);
    vec![
        StructuredFact {
            subject: "subject-a".to_string(),
            predicate: "relates-to".to_string(),
            object: "object-x".to_string(),
            valid_from: Some(t0),
            valid_to: None,
            memory_type: None,
        },
        StructuredFact {
            subject: "subject-a".to_string(),
            predicate: "has-property".to_string(),
            object: "literal-value-1".to_string(),
            valid_from: Some(t0),
            valid_to: None,
            memory_type: None,
        },
        StructuredFact {
            subject: "subject-a".to_string(),
            predicate: "depends-on".to_string(),
            object: "subject-b".to_string(),
            valid_from: Some(t0),
            valid_to: None,
            memory_type: None,
        },
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// ADR-035 §1 / acceptance criterion 1 — Caller frontmatter triples are pinned at
/// engine layer + visible to recall after ingest. Uses NullLLM so no Phase 2 work
/// interferes with the assertion.
#[test]
fn with_facts_pins_facts_and_recall_returns_them() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("pin_and_recall").await;
            let facts = three_fact_fixture();
            let n_facts = facts.len();

            let commit = mem
                .remember("ADR-035 body — with_facts semantics and skip_extraction.")
                .with_facts(facts)
                .from_document("adr-035-doc")
                .skip_extraction() // ner feature: test is not about extraction
                .await
                .expect("remember should succeed");

            assert!(
                !commit.episode_entity_id.is_empty(),
                "episode_entity_id must be set"
            );

            let results = mem
                .recall("extends")
                .opts(SearchOpts {
                    limit: Some(10),
                    ..Default::default()
                })
                .await
                .expect("recall should succeed");

            // Recall over a null-embedder + FTS path: the 3 pinned facts should
            // be visible in the result set as fact-anchored retrievals.
            // Some retrieval paths may surface 0 entities with null embedder;
            // the key invariant is that recall does NOT error AND the substrate
            // counter shows 3 caller-pinned facts.
            assert!(
                results.len() <= 50,
                "recall returned implausibly large set: {}",
                results.len()
            );

            let pinned =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.pinned_total");
            assert_eq!(
                pinned, n_facts as u64,
                "all {} caller-supplied facts must have been pinned; counter={}",
                n_facts, pinned
            );
        });
    });
}

/// ADR-035 §2 / acceptance criterion — `skip_extraction()` opt-out wired:
/// engine.ingest bails after pin-loop, never invoking Phase 2 LLM.
/// Verified via `kremory.skip_extraction.invoked_total` counter.
#[test]
fn skip_extraction_suppresses_phase2_via_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("skip_extraction").await;

            let _commit = mem
                .remember("Body that would normally be extracted by Phase 2 LLM.")
                .with_facts(three_fact_fixture())
                .skip_extraction()
                .from_document("skip-test-doc")
                .await
                .expect("remember with skip_extraction should succeed");

            let skipped = find_counter_labeled(
                snapshotter.snapshot(),
                "kremory.skip_extraction.invoked_total",
            );
            assert_eq!(
                skipped, 1,
                "skip_extraction.invoked_total must increment exactly once per call; got {}",
                skipped
            );

            // Sanity: pinned counter still incremented (caller facts written).
            let pinned =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.pinned_total");
            assert_eq!(
                pinned,
                three_fact_fixture().len() as u64,
                "skip_extraction still pins caller facts; got pinned={}",
                pinned
            );
        });
    });
}

/// Regression guard for the lying-counter fix (2026-07-21, memory
/// `project_kremory_insert_fact_error_counts_benign_dedup`): a benign
/// `Error::Duplicate` (content-hash idempotency — the same fact re-pinned) is
/// swallowed by `try_insert_fact_with_group` into `with_facts.deduped_total`
/// and MUST NOT increment `kremory.db.insert_fact_error_total`. Before the fix
/// the inner `insert_fact_with_group` fired the error counter for EVERY
/// duplicate (it matched `with_facts_deduped_total` 1:1 on the LoCoMo Groq run),
/// mis-framing benign dedups as insert failures and making the fail-loud
/// `insert_fact_error_total == 0` gate unsatisfiable. [[observability-first-class]] #9.
#[test]
fn duplicate_fact_does_not_increment_insert_fact_error_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("dup_fact_counter").await;
            // Bind ONCE + clone so both pins carry byte-identical facts (same
            // valid_from) → identical content_hash → the second pin is a true
            // duplicate regardless of what the hash covers.
            let facts = three_fact_fixture();

            mem.remember("first pin of the fixture facts.")
                .with_facts(facts.clone())
                .skip_extraction()
                .from_document("dup-doc-1")
                .await
                .expect("first remember succeeds");

            mem.remember("second pin of the same fixture facts.")
                .with_facts(facts.clone())
                .skip_extraction()
                .from_document("dup-doc-2")
                .await
                .expect("second remember succeeds despite duplicate facts");

            let deduped =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.deduped_total");
            let insert_errors =
                find_counter_labeled(snapshotter.snapshot(), "kremory.db.insert_fact_error_total");

            assert!(
                deduped >= 1,
                "re-pinning identical facts must register benign dedup(s); got deduped={deduped}"
            );
            assert_eq!(
                insert_errors, 0,
                "benign Error::Duplicate must NOT increment insert_fact_error_total \
                 (lying-counter fix); got insert_errors={insert_errors}"
            );
        });
    });
}

/// ADR-035 acceptance criterion — `with_facts(vec![])` MUST be behaviorally
/// identical to not calling `with_facts` at all (backward compatibility).
#[test]
fn with_facts_empty_vec_equivalent_to_no_facts() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("empty_vec_compat").await;

            // No `.skip_extraction()` here on purpose: this test asserts that
            // `with_facts(vec![])` is behaviourally identical to NOT calling
            // `with_facts` at all — and "not calling with_facts at all" does not
            // skip extraction. Calling `.skip_extraction()` AND asserting
            // `skipped == 0` (as a prior revision did) is self-contradictory.
            // Extraction here is a no-op: the harness wires `MockChatProvider::null()`
            // (returns "" → zero entities), so the non-skip path is safe + covers
            // the genuine backward-compat invariant. (Rule 8 test-input fix 2026-06-16.)
            let commit = mem
                .remember("plain content, no caller facts")
                .with_facts(vec![])
                .from_chat("empty-fixture")
                .await
                .expect("empty-vec remember should succeed");

            assert!(!commit.episode_entity_id.is_empty());

            let pinned =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.pinned_total");
            assert_eq!(
                pinned, 0,
                "empty with_facts vec must NOT increment pinned_total; got {}",
                pinned
            );
            let skipped = find_counter_labeled(
                snapshotter.snapshot(),
                "kremory.skip_extraction.invoked_total",
            );
            assert_eq!(
                skipped, 0,
                "no skip_extraction call must NOT increment counter; got {}",
                skipped
            );
        });
    });
}

/// ADR-035 acceptance — Quinn M-01: `with_facts + skip_extraction + recall`
/// round-trip MUST complete cleanly. Combines both opt-ins to verify they
/// compose: caller facts MUST be pinned (counter increments) AND extraction
/// MUST be suppressed (counter increments) AND recall MUST execute without
/// error. Recall result-set under null embedder is not asserted (vector path
/// is null + FTS rank shape varies by SQLite version per the existing test
/// `with_facts_pins_facts_and_recall_returns_them` precedent).
#[test]
fn with_facts_skip_extraction_combined_roundtrip() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("skip_recall_roundtrip").await;
            let facts = three_fact_fixture();
            let n_facts = facts.len();

            let commit = mem
                .remember("Episode body — should NOT be extracted because skip_extraction is set.")
                .with_facts(facts)
                .skip_extraction()
                .from_document("skip-recall-doc")
                .await
                .expect("remember should succeed");

            assert!(!commit.episode_entity_id.is_empty());

            // Round-trip invariant 1: caller facts pinned.
            let pinned =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.pinned_total");
            assert_eq!(
                pinned, n_facts as u64,
                "caller facts MUST be pinned even when skip_extraction is set; got pinned={}",
                pinned
            );

            // Round-trip invariant 2: skip_extraction was invoked.
            let skipped = find_counter_labeled(
                snapshotter.snapshot(),
                "kremory.skip_extraction.invoked_total",
            );
            assert_eq!(
                skipped, 1,
                "skip_extraction MUST increment its counter; got skipped={}",
                skipped
            );

            // Round-trip invariant 3: Phase 2 LLM-extraction-pipeline counters MUST
            // NOT fire when skip_extraction is set (the early-return path bails
            // BEFORE chunk-split + extractor invocation).
            let extraction_attempts = find_counter_labeled(
                snapshotter.snapshot(),
                "rql.extraction.structured_call_attempt",
            );
            assert_eq!(
                extraction_attempts, 0,
                "Phase 2 MUST NOT run when skip_extraction is set; got attempts={}",
                extraction_attempts
            );

            // Round-trip invariant 4: recall must complete cleanly (no panic, no
            // error) over the now-populated graph state.
            let _ = mem
                .recall("literal-value-1")
                .opts(SearchOpts {
                    limit: Some(20),
                    ..Default::default()
                })
                .await
                .expect("recall after with_facts+skip_extraction must not error");
        });
    });
}

/// Builder chain compiles + types check for the new `.with_facts().skip_extraction()`
/// composition. Compile-time pin only (no runtime assertion beyond no-panic).
#[test]
fn with_facts_skip_extraction_chain_compiles() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");
    rt.block_on(async {
        let mem = open_with_ns("chain_compiles").await;
        let _commit = mem
            .remember("c")
            .with_facts(three_fact_fixture())
            .skip_extraction()
            .from_document("chain-test")
            .await
            .expect("chain should succeed");
    });
}

/// TD-113 DOD-001 — `remember(skip_extraction) + recall(<subject name>)` MUST
/// return ≥ 1. This is the regression guard for the fix that stamps the pinned
/// entity's literal name into the FTS `properties` channel at write time.
///
/// Runs under `NullEmbeddingProvider` (zero-vector embeddings) ON PURPOSE: it
/// proves the FTS channel ALONE restores recall-findability — no embedder, no
/// second LLM (spec §3 F1). Before the fix, the pinned subject entity carried a
/// bare `{"stub": false}` properties blob (no name token) + NULL embedding, so
/// both recall seed arms missed and `recall` returned 0 — the empirically
/// confirmed gap (dogfood 2026-07-14; `with_facts_pins_facts_and_recall_returns_them`
/// conceded it by asserting only "recall does not error").
#[test]
fn td113_pinned_subject_is_recall_findable_by_name_under_null_embedder() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    rt.block_on(async {
        let mem = open_with_ns("td113_recall_findable").await;

        // Single pinned triple — subject is an entity, object is a literal
        // (StructuredFact→PrePinnedFact always sets object_id=None).
        let facts = vec![StructuredFact {
            subject: "Grace Hopper".to_string(),
            predicate: "invented".to_string(),
            object: "the compiler".to_string(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }];

        mem.remember("Grace Hopper invented the compiler.")
            .with_facts(facts)
            .from_document("td113-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        // `recall().await` renders the context block to a String (default
        // `TemporalFacts` template). Pre-fix this was empty: the pinned entity
        // had (a) no name token in FTS `properties`, (b) NULL embedding, and
        // (c) no episodic edge → so even a lucky FTS hit rendered to "" because
        // the default template emits output only per source_ref. Post-fix all
        // three channels are stamped at write time (no second LLM).
        let rendered = mem
            .recall("Grace Hopper")
            .opts(SearchOpts {
                limit: Some(10),
                ..Default::default()
            })
            .await
            .expect("recall should succeed");

        assert!(
            !rendered.trim().is_empty(),
            "TD-113: a caller-pinned subject entity MUST be recall-findable by \
             its literal name (no second LLM); got empty recall"
        );
        assert!(
            rendered.contains("Grace Hopper"),
            "expected the pinned 'Grace Hopper' entity in the recall block, got: {rendered:?}"
        );
    });
}

/// TD-116 / ADR-074 — recall surfaces the connected FACTS, not just entity names.
///
/// Regression guard for the recall-drops-facts gap the MCP dogfood exposed:
/// after a mode-(c) pin, `recall` must return the entity WITH its connected fact
/// (natural-language `fact` string + structured triple + temporal validity), and
/// the default rendered template must contain the fact sentence. Runs under
/// `NullEmbeddingProvider` — the facts come from the graph (contextualize), not
/// the embedder, so no LLM/embedder is needed.
#[test]
fn td116_recall_returns_connected_facts_under_null_embedder() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    rt.block_on(async {
        let mem = open_with_ns("td116_recall_facts").await;

        mem.remember("Grace Hopper invented the compiler.")
            .with_facts(vec![StructuredFact {
                subject: "Grace Hopper".to_string(),
                predicate: "invented".to_string(),
                object: "the compiler".to_string(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .from_document("td116-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        // Structured: the entity result carries the connected fact.
        let raw = mem
            .recall("Grace Hopper")
            .raw()
            .await
            .expect("raw recall should succeed");
        let hopper = raw
            .iter()
            .find(|r| r.entity_name == "Grace Hopper")
            .expect("Grace Hopper must be in recall results");
        assert!(
            !hopper.facts.is_empty(),
            "TD-116: recall MUST surface the entity's connected facts; got none"
        );
        let f = hopper
            .facts
            .iter()
            .find(|f| f.predicate == "invented")
            .expect("the pinned 'invented' fact must be present");
        assert_eq!(f.fact, "Grace Hopper invented the compiler");
        assert_eq!(f.subject, "Grace Hopper");
        assert_eq!(f.object, "the compiler");
        assert!(
            !f.object_is_entity,
            "literal object → object_is_entity=false"
        );

        // Rendered (default TemporalFacts template): the fact sentence appears.
        let rendered = mem
            .recall("Grace Hopper")
            .await
            .expect("rendered recall should succeed");
        assert!(
            rendered.contains("Grace Hopper invented the compiler"),
            "default template must render the fact sentence, got: {rendered:?}"
        );
    });
}

/// MED-1 (Quinn P1 correctness) — a `with_facts` pin whose caller-asserted
/// `valid_to` predates the (defaulted) `valid_from` is a time-inversion: the
/// resulting `[valid_from, valid_to)` window is empty, so once `bound_valid_to`
/// writes it the fact is permanently invisible to every `as_of(t)` query. The
/// pin path must REJECT such a pin (mirroring `facade/supersede.rs`'s
/// `rejected_time_inversion` guard) — emitting a named counter + warn and NOT
/// persisting an inverted-window fact.
///
/// Trigger mirrors the real bug: `valid_from: None` (defaults to ingest `now()`
/// in `engine_handle.rs`) + `valid_to` 10 days in the past → guaranteed
/// inversion regardless of test wall-clock.
///
/// Surgical: a valid (non-inverted) pin in the SAME batch must still succeed —
/// the guard rejects ONLY the inverted pin, never the whole batch.
///
/// RED before the guard: the inverted fact is inserted + bound with an inverted
/// window (`pinned_total == 2`, `valid_to_bound_total == 1`,
/// `pin_rejected_total` absent). GREEN after: `pinned_total == 1` (good pin
/// only), `pin_rejected_total == 1`, `valid_to_bound_total == 0`.
#[test]
fn with_facts_rejects_time_inversion_and_does_not_persist_inverted_window() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = open_with_ns("time_inversion_reject").await;

            let inverted_valid_to = Utc::now() - Duration::days(10);
            let facts = vec![
                // Valid pin — open-ended window, must survive.
                StructuredFact {
                    subject: "subject-good".to_string(),
                    predicate: "relates-to".to_string(),
                    object: "object-x".to_string(),
                    valid_from: Some(Utc::now() - Duration::hours(1)),
                    valid_to: None,
                    memory_type: None,
                },
                // Inverted pin — valid_from defaults to ingest now(), which is
                // ~10 days AFTER this valid_to → empty window. Must be rejected.
                StructuredFact {
                    subject: "subject-bad".to_string(),
                    predicate: "relates-to".to_string(),
                    object: "object-y".to_string(),
                    valid_from: None,
                    valid_to: Some(inverted_valid_to),
                    memory_type: None,
                },
            ];

            mem.remember("time-inversion guard body")
                .with_facts(facts)
                .from_document("med1-doc")
                .skip_extraction()
                .await
                .expect("remember should succeed (rejection is in-band, not Err)");

            let rejected = find_counter_labeled(
                snapshotter.snapshot(),
                "kremory.with_facts.pin_rejected_total",
            );
            assert_eq!(
                rejected, 1,
                "the inverted pin must fire the named time-inversion rejection counter; got {rejected}"
            );

            let pinned =
                find_counter_labeled(snapshotter.snapshot(), "kremory.with_facts.pinned_total");
            assert_eq!(
                pinned, 1,
                "only the valid pin may persist — the inverted-window fact must NOT be written; pinned_total={pinned}"
            );

            let bound = find_counter_labeled(
                snapshotter.snapshot(),
                "kremory.with_facts.valid_to_bound_total",
            );
            assert_eq!(
                bound, 0,
                "no valid_to bind may occur — the inverted pin is rejected BEFORE insert, the good pin has valid_to=None; valid_to_bound_total={bound}"
            );
        });
    });
}

// ── TD-197: reserved meta-edge predicates must never reach recall() ────────
//
// TD-197 shape-2 detector (see `scripts/audit-reserved-predicates.py`): the
// ORIGINAL version of this test pinned exactly one literal predicate string
// (`RESERVED_PREDICATE_POTENTIAL_ALIAS`). That is a single-instance
// regression pin, not a class guard — a second `RESERVED_PREDICATE_*` const
// could be added to `disambiguation::RESERVED_PREDICATES` tomorrow and this
// test would keep passing while the new value leaked, because nothing in the
// test itself was coupled to the reserved-value SET. This version iterates
// `disambiguation::RESERVED_PREDICATES` — the exact slice
// `is_reserved_predicate()` consults — so a future reserved predicate is
// covered automatically, the moment it is added to that slice, with zero
// test-file changes required.
//
// The Python detector (`scripts/audit-reserved-predicates.py`) proves the
// SOURCE-LEVEL sync between `RESERVED_PREDICATE_*` consts and the
// `RESERVED_PREDICATES` slice they must appear in. This test proves the
// RUNTIME consequence: for every predicate that slice contains, ingest it as
// a real fact and prove it never reaches either consumer-facing recall
// surface. The two are complementary, not duplicative: the Python detector
// would not catch a `RESERVED_PREDICATES` slice that is complete but whose
// entries a *read path* fails to filter (i.e. `is_reserved_predicate()`
// itself, or a call site, regresses) — only a real end-to-end recall proves
// that.
#[test]
fn td197_recall_never_surfaces_reserved_predicate() {
    // Non-vacuity (CLAUDE.md "Non-vacuity is mandatory" / TD-197 shape-2 spec):
    // a loop over an empty set passes trivially and proves nothing. If this
    // ever fires, `RESERVED_PREDICATES` itself is broken — see
    // `disambiguation::mod.rs`'s
    // `reserved_predicates_slice_contains_every_reserved_predicate_const`
    // unit test, which pins the const-vs-slice pairing directly.
    assert!(
        !kremory::core::disambiguation::RESERVED_PREDICATES.is_empty(),
        "RESERVED_PREDICATES must not be empty — a vacuous loop here would prove \
         nothing about reserved-predicate leakage"
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    for &predicate in kremory::core::disambiguation::RESERVED_PREDICATES {
        rt.block_on(assert_reserved_predicate_never_surfaces(predicate));
    }
}

/// Pins `predicate` directly via `with_facts` (bypassing the L4
/// disambiguation flow entirely) since the read-side filter matches on the
/// predicate STRING alone, regardless of how the fact was inserted — mirrors
/// the exact leaked line measured in production: `"adoption potential_alias
/// adoption agencies (valid_at=...)"` (RECALL-LEDGER §4.19 / tech-debt-
/// register TD-197). Then proves `predicate` reaches neither consumer-facing
/// recall surface: the structured `.facts` on `recall().raw()`, nor the
/// rendered (LLM-facing) recall string.
async fn assert_reserved_predicate_never_surfaces(predicate: &str) {
    let mem = open_with_ns(&format!("td197_reserved_{predicate}")).await;

    mem.remember("Adoption is discussed by adoption agencies.")
        .with_facts(vec![StructuredFact {
            subject: "adoption".to_string(),
            predicate: predicate.to_string(),
            object: "adoption agencies".to_string(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .from_document("td197-doc")
        .skip_extraction()
        .await
        .expect("remember(skip_extraction) should succeed");

    // Structured: the reserved-predicate fact must NOT be in `.facts`.
    let raw = mem
        .recall("adoption")
        .raw()
        .await
        .expect("raw recall should succeed");
    // Non-vacuity: an empty result set would make "no reserved predicate
    // found" trivially true for the wrong reason (nothing was checked).
    assert!(
        !raw.is_empty(),
        "recall() must return results to prove absence means something \
         (predicate={predicate:?}); got 0 hits"
    );
    let hit = raw
        .iter()
        .find(|r| r.entity_name == "adoption")
        .unwrap_or_else(|| {
            panic!("adoption entity must be in recall results (predicate={predicate:?})")
        });
    assert!(
        hit.facts
            .iter()
            .all(|f| !kremory::core::disambiguation::is_reserved_predicate(&f.predicate)),
        "TD-197 [shape-2]: reserved predicate {predicate:?} must never reach \
         recall() structured output; got facts: {:?}",
        hit.facts
    );

    // Rendered (default TemporalFacts template): must not leak the
    // predicate text into the LLM-facing prompt string either.
    let rendered = mem
        .recall("adoption")
        .await
        .expect("rendered recall should succeed");
    assert!(
        !rendered.contains(predicate),
        "TD-197 [shape-2]: reserved predicate {predicate:?} must never reach \
         rendered recall() output, got: {rendered:?}"
    );
}

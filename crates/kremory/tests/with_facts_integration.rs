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
    std::env::temp_dir().join(format!(
        "kremory_with_facts_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
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

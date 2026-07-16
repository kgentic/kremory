#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-068 — `recall().as_of()` point-in-time (valid-time) correctness tests.
//!
//! Rewritten a second time (not deleted) per ADR-068 Decision 3 / the
//! companion spec's own precedent for this exact file: `as_of_emits_v010_warn`
//! (silent no-op) → `as_of_errors_unsupported` (F4 fail-loud guard) → now this
//! POSITIVE correctness suite, because `as_of` is implemented end-to-end
//! (`memory::search` → `contextualize()` → `TemporalGraph::get_neighbours_at`)
//! and there is no more `Unsupported` path for it to hit.
//!
//! Semantics under test (ADR-068 Decision 1 — valid-time, not transaction-time):
//!   - `as_of(t)` BEFORE a fact's `valid_from` → fact EXCLUDED.
//!   - `as_of(t)` INSIDE `[valid_from, valid_to)` → fact INCLUDED.
//!   - `as_of(t)` AT/AFTER `valid_to` → fact EXCLUDED.
//!   - `as_of=None` (the default/no-op) → unaffected, present-day results
//!     (regression pin — this is what every other recall test already
//!     exercises implicitly; pinned here explicitly for this file's own
//!     narrative completeness).
//!
//! `invalid_at`-flagged-but-still-valid-time-in-window inclusion (the other
//! half of Decision 1) is locked at the lower `TemporalGraph::get_neighbours_at`
//! unit level (`core/graph/tests.rs`) where the `invalid_at` column can be
//! stamped directly without also touching `expired_at` — no public facade API
//! writes `invalid_at` in isolation (`invalidate_fact_with_reason` always
//! stamps both columns together, which would confound the assertion via the
//! pre-existing, unconditional `expired_at IS NULL` clause).
//!
//! Runs under `NullEmbeddingProvider` + `MockChatProvider::null()` +
//! `with_facts`/`skip_extraction` (ADR-035 Path X) — deterministic, no LLM —
//! the mechanism under test is the substrate's temporal SQL filter, not
//! extraction (mirrors `with_facts_integration.rs`'s own precedent).

use chrono::{Duration, Utc};
use kremory::memory::types::StructuredFact;
use kremory::{DynEmbeddingProvider, Memory, MemoryError, Namespace};
use std::sync::Arc;

/// `.as_of()` can be chained on RecallRequest without consuming the builder.
#[test]
fn as_of_chain_compiles() {
    let mem = make_memory_sync();
    let ts = Utc::now();
    // Build but don't await — just verify the chain is valid.
    let _req = mem.recall("query").as_of(ts);
}

/// `.as_of()` + missing namespace → `MissingNamespace` (not a phantom success).
///
/// The as_of value is accepted but namespace validation fires before any SQL
/// is executed, so missing namespace still surfaces correctly. Still valid
/// post-ADR-068: guard ordering (namespace check before search) is unrelated
/// to the as_of guard that was removed.
#[tokio::test]
async fn as_of_with_missing_namespace_still_errors_missing_namespace() {
    let mem = open_no_ns().await;
    let ts = Utc::now();
    let err = mem
        .recall("what happened yesterday?")
        .as_of(ts)
        .await
        .expect_err("should fail with MissingNamespace");
    assert!(
        matches!(err, MemoryError::MissingNamespace { .. }),
        "expected MissingNamespace, got: {err:?}"
    );
}

/// `.as_of()` can be combined with `.k()` and `.in_namespace()`.
#[test]
fn as_of_k_namespace_chain_compiles() {
    let mem = make_memory_sync();
    let ts = Utc::now();
    // The chain must compile without type errors.
    let _req = mem
        .recall("query")
        .k(5)
        .as_of(ts)
        .in_namespace(Namespace::new("ns"));
}

/// ADR-068 Decision 1 — `as_of(t)` BEFORE a fact's `valid_from` excludes it.
/// The seed entity is still found (entity search is temporal-agnostic —
/// Decision 4 point 1) but its connected fact must NOT appear.
#[tokio::test]
async fn as_of_before_valid_from_excludes_fact() {
    let mem = open_with_ns("before_window").await;
    let valid_from = Utc::now() - Duration::days(10);
    let valid_to = Utc::now() - Duration::days(3);

    mem.remember("Ada Lovelace wrote the first algorithm.")
        .with_facts(vec![StructuredFact {
            subject: "Ada Lovelace".to_string(),
            predicate: "wrote".to_string(),
            object: "the first algorithm".to_string(),
            valid_from: Some(valid_from),
            valid_to: Some(valid_to),
            memory_type: None,
        }])
        .from_document("as-of-before-doc")
        .skip_extraction()
        .await
        .expect("remember(skip_extraction) should succeed");

    let t_before = valid_from - Duration::days(1);
    let raw = mem
        .recall("Ada Lovelace")
        .as_of(t_before)
        .raw()
        .await
        .expect("recall should succeed");

    let ada = raw.iter().find(|r| r.entity_name == "Ada Lovelace");
    if let Some(ada) = ada {
        assert!(
            ada.facts.is_empty(),
            "as_of before valid_from must exclude the fact; got: {:?}",
            ada.facts
        );
    }
    // Entity absence entirely is also an acceptable (stricter) outcome of the
    // same invariant — the fact is excluded either way.
}

/// ADR-068 Decision 1 — `as_of(t)` INSIDE `[valid_from, valid_to)` includes
/// the fact.
#[tokio::test]
async fn as_of_inside_window_includes_fact() {
    let mem = open_with_ns("inside_window").await;
    let valid_from = Utc::now() - Duration::days(10);
    let valid_to = Utc::now() - Duration::days(3);

    mem.remember("Ada Lovelace wrote the first algorithm.")
        .with_facts(vec![StructuredFact {
            subject: "Ada Lovelace".to_string(),
            predicate: "wrote".to_string(),
            object: "the first algorithm".to_string(),
            valid_from: Some(valid_from),
            valid_to: Some(valid_to),
            memory_type: None,
        }])
        .from_document("as-of-inside-doc")
        .skip_extraction()
        .await
        .expect("remember(skip_extraction) should succeed");

    let t_inside = valid_from + Duration::days(2);
    let raw = mem
        .recall("Ada Lovelace")
        .as_of(t_inside)
        .raw()
        .await
        .expect("recall should succeed");

    let ada = raw
        .iter()
        .find(|r| r.entity_name == "Ada Lovelace")
        .expect("Ada Lovelace must be found — entity search is as_of-agnostic");
    assert!(
        !ada.facts.is_empty(),
        "as_of inside [valid_from, valid_to) must include the fact; got none"
    );
    let f = ada
        .facts
        .iter()
        .find(|f| f.predicate == "wrote")
        .expect("the pinned 'wrote' fact must be present");
    assert_eq!(f.fact, "Ada Lovelace wrote the first algorithm");
}

/// ADR-068 Decision 1 — `as_of(t)` AT/AFTER `valid_to` excludes the fact.
#[tokio::test]
async fn as_of_at_or_after_valid_to_excludes_fact() {
    let mem = open_with_ns("after_window").await;
    let valid_from = Utc::now() - Duration::days(10);
    let valid_to = Utc::now() - Duration::days(3);

    mem.remember("Ada Lovelace wrote the first algorithm.")
        .with_facts(vec![StructuredFact {
            subject: "Ada Lovelace".to_string(),
            predicate: "wrote".to_string(),
            object: "the first algorithm".to_string(),
            valid_from: Some(valid_from),
            valid_to: Some(valid_to),
            memory_type: None,
        }])
        .from_document("as-of-after-doc")
        .skip_extraction()
        .await
        .expect("remember(skip_extraction) should succeed");

    // Exactly at valid_to: predicate is `valid_to > ?t`, strictly-greater, so
    // `t == valid_to` must exclude (half-open window, boundary-exclusive).
    let raw_at_boundary = mem
        .recall("Ada Lovelace")
        .as_of(valid_to)
        .raw()
        .await
        .expect("recall should succeed");
    if let Some(ada) = raw_at_boundary
        .iter()
        .find(|r| r.entity_name == "Ada Lovelace")
    {
        assert!(
            ada.facts.is_empty(),
            "as_of AT valid_to must exclude the fact (half-open window); got: {:?}",
            ada.facts
        );
    }

    // Well after valid_to.
    let raw_after = mem
        .recall("Ada Lovelace")
        .as_of(Utc::now())
        .raw()
        .await
        .expect("recall should succeed");
    if let Some(ada) = raw_after.iter().find(|r| r.entity_name == "Ada Lovelace") {
        assert!(
            ada.facts.is_empty(),
            "as_of well after valid_to must exclude the fact; got: {:?}",
            ada.facts
        );
    }
}

/// Regression pin — `as_of=None` (the default, no `.as_of()` call) is
/// unaffected by ADR-068 and still returns present-day facts, matching every
/// pre-existing recall test's implicit assumption (e.g. `td116_recall_
/// returns_connected_facts_under_null_embedder` in `with_facts_integration.rs`).
#[tokio::test]
async fn as_of_none_is_unaffected_regression_pin() {
    let mem = open_with_ns("as_of_none_regression").await;
    let valid_from = Utc::now() - Duration::days(10);

    mem.remember("Ada Lovelace wrote the first algorithm.")
        .with_facts(vec![StructuredFact {
            subject: "Ada Lovelace".to_string(),
            predicate: "wrote".to_string(),
            object: "the first algorithm".to_string(),
            valid_from: Some(valid_from),
            valid_to: None,
            memory_type: None,
        }])
        .from_document("as-of-none-doc")
        .skip_extraction()
        .await
        .expect("remember(skip_extraction) should succeed");

    // No .as_of() call at all — SearchOpts.as_of defaults to None.
    let raw = mem
        .recall("Ada Lovelace")
        .raw()
        .await
        .expect("recall should succeed");
    let ada = raw
        .iter()
        .find(|r| r.entity_name == "Ada Lovelace")
        .expect("Ada Lovelace must be found");
    assert!(
        !ada.facts.is_empty(),
        "as_of=None must be unaffected — fact must still be present"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Unique per-call DB path to avoid SQLite "database is locked" flakes
/// when integration tests run in parallel.
fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_facade_as_of_warn_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

async fn open_no_ns() -> Memory {
    Memory::open(unique_db_path("open_no_ns"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder should succeed")
}

/// Like `open_no_ns` but with a default namespace, so `remember`/`recall`
/// calls don't need an explicit `.in_namespace(...)` on every call — mirrors
/// `with_facts_integration.rs`'s own `open_with_ns` helper.
async fn open_with_ns(tag: &str) -> Memory {
    Memory::open(unique_db_path(tag))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("as_of_correctness_tests"))
        .await
        .expect("builder should succeed")
}

fn make_memory_sync() -> Memory {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(open_no_ns())
}

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

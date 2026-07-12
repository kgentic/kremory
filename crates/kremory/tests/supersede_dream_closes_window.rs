#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-071 §Item 3 (TD-070) — LOAD-BEARING e2e: `mem.supersede(fact_id).at(...)`
//! bounds `valid_to`, then a REAL `mem.dream()` call with
//! `DreamOpts.include_supersession_sweep = true` closes the window
//! (`expired_at = valid_to`) via the pre-existing deterministic `window_closeout`
//! lane (`core/dream/consolidation/supersession.rs`, unchanged this item).
//!
//! This is the FIRST real producer for the previously-honest-zero
//! `DreamSummary.supersessions_recorded` field when driven end-to-end through the
//! public consumer surface (`SupersedeRequest` -> `mem.dream()`), proving the
//! two-phase chain documented in `facade/supersede.rs`:
//!
//! 1. `mem.supersede(fact_id).at(valid_to).execute()` — PRODUCER, bounds
//!    `facts.valid_to` (world-time) via `bound_valid_to` (Amendment C — NOT
//!    `invalidate_fact`).
//! 2. `mem.dream().with_opts(DreamOpts { include_supersession_sweep: true, .. })` —
//!    CONSUMER, `window_closeout` observes `valid_to IS NOT NULL AND valid_to <
//!    now AND expired_at IS NULL AND invalid_at IS NULL AND is_dream_generated =
//!    0`, sets `expired_at = valid_to`.
//!
//! If this test fails, the mechanism itself is wrong — per CLAUDE.md Rule 8
//! (treat-cause-not-symptom) the fix is in the producer/consumer wiring, never
//! loosening this assertion.
//!
//! Deterministic (no real LLM/embeddings needed): `MockChatProvider::null()` is
//! wired only to satisfy `dream()`'s `Category B` LLM requirement (ADR-041) — the
//! supersession op itself is zero-LLM, pure date-compare. Mirrors the established
//! `adr071_item2_p2_p4_enablement.rs` / `dream_phase2_deterministic_passes.rs`
//! precedent (same `MockChatProvider::null()` + explicit narrow `DreamOpts`
//! pattern, so the always-on reconciliation passes run as safe no-ops on this
//! single-fact fixture).

use std::sync::Arc;

use chrono::{Duration, Utc};

use kremory::core::graph::{FactInsert, InsertEntityWithGroupParams};
use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::memory::ChatProvider;
use kremory::{DreamOpts, DynEmbeddingProvider, Memory, Namespace, SupersedeOutcome};

const DIM: usize = 64;

/// Mirrors `adr071_item2_p2_p4_enablement.rs::open_mem` — a `Memory` wired with a
/// null LLM (safe no-op for the always-on reconciliation passes) + a
/// deterministic hash-based embedder, in a fresh temp-dir sqlite file per test.
async fn open_mem(dir: &std::path::Path, ns: &Namespace) -> Memory {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(DIM));
    Memory::open(dir.join("supersede-dream-e2e.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(DIM)
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open must succeed")
}

#[tokio::test]
async fn supersede_then_dream_closes_window() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("adr071-item3-supersede-dream-e2e");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    let now = Utc::now();

    // Seed a CONSUMER fact (is_dream_generated defaults to 0 — migration 016
    // `NOT NULL DEFAULT 0`), valid_to NULL, expired_at NULL, invalid_at NULL, a
    // valid_from safely in the past.
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id: "supersede-e2e-subject",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(&gid),
        })
        .await
        .expect("seed subject entity");

    let valid_from = now - Duration::days(30);
    let fact_id = graph
        .insert_fact_with_group(
            FactInsert::new("supersede-e2e-subject", "status", valid_from)
                .object_value("active"),
            Some(&gid),
        )
        .await
        .expect("seed fact");

    // Sanity: fact is live pre-supersede (expired_at NULL, valid_to NULL).
    let pre = graph
        .get_fact_by_id(fact_id, &gid)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist pre-supersede");
    assert!(pre.valid_to.is_none(), "valid_to must be NULL before supersede");
    assert!(pre.expired_at.is_none(), "expired_at must be NULL before supersede");

    // ── Phase 1: PRODUCER — mem.supersede bounds valid_to (world-time, in the
    // past relative to `now` so window_closeout's `valid_to < now` predicate
    // matches it on the very next dream() call). ────────────────────────────
    let bound_at = now - Duration::days(1);
    let outcome = mem
        .supersede(fact_id)
        .in_namespace(ns.clone())
        .at(bound_at)
        .with_reason("adr071-item3 e2e: superseded by newer status")
        .execute()
        .await
        .expect("mem.supersede(...).execute() must succeed");
    assert_eq!(
        outcome,
        // D4: execute() bounds only — retirement is deferred to the dream sweep below.
        SupersedeOutcome::Bounded { retired: 0 },
        "supersede must bound (bound_at is after valid_from and fact exists)"
    );

    let mid = graph
        .get_fact_by_id(fact_id, &gid)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist post-supersede");
    assert_eq!(
        mid.valid_to.map(|t| t.timestamp()),
        Some(bound_at.timestamp()),
        "valid_to must be bounded after supersede"
    );
    assert!(
        mid.expired_at.is_none(),
        "expired_at must STILL be NULL after supersede alone — bound_valid_to \
         (Amendment C) writes valid_to, not expired_at; the system-time close is \
         the dream sweep's job, not supersede's"
    );

    // ── Phase 2: CONSUMER — a REAL mem.dream() call with
    // include_supersession_sweep: true closes the window. ───────────────────
    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut opts = DreamOpts::default();
    opts.include_type_discovery = false;
    opts.include_consistency_check = false;
    opts.include_type_registry_collapse = false;
    opts.include_acronym_nickname_recall = false;
    opts.include_type_novelty_llm_verify = false;
    opts.include_community_detection = false;
    opts.include_cross_episode_merges = false;
    opts.include_supersession_sweep = true;
    let summary = mem
        .dream()
        .in_namespace(ns)
        .with_opts(opts)
        .await
        .expect("mem.dream() with include_supersession_sweep must succeed");

    assert!(
        summary.supersessions_recorded > 0,
        "DreamSummary.supersessions_recorded must be > 0 — this is the load-bearing \
         proof that the supersede -> dream two-phase chain actually fires end-to-end; \
         got {}",
        summary.supersessions_recorded
    );

    let post = graph
        .get_fact_by_id(fact_id, &gid)
        .await
        .expect("get_fact_by_id must succeed")
        .expect("fact must exist post-dream");
    assert!(
        post.expired_at.is_some(),
        "expired_at must be SET after dream()'s supersession sweep closes the window \
         (window_closeout sets expired_at = valid_to)"
    );
    assert_eq!(
        post.expired_at.map(|t| t.timestamp()),
        Some(bound_at.timestamp()),
        "expired_at must equal the bounded valid_to (the demonstrated close-out time, \
         not `now`)"
    );
}

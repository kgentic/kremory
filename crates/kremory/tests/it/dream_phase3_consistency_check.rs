#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase 3 — consistency_check pass dispatched by the facade orchestrator.
//!
//! Governing spec: `.ai-docs/specs/dream-phase-reconciliation-v2-2026-06-30.md`
//! §D3 (ordering: … reclassify → consistency_check → canonicalize), ADR-047.
//! Phase 3 of `.ai-docs/plans/dream-phase-reconciliation-readiness-gate-2026-06-30.md`.
//!
//! ## Scope
//!
//! This proves `mem.dream()` INVOKES `run_consistency_check` (the orchestration
//! wiring). The correction LOGIC (embed-prefilter, LLM verify, type correction,
//! `DreamPass4` audit stamp) is already covered deterministically and with a real
//! LLM by `tests/consistency_check_phase_c.rs` (C8/C9/C10). The full real-LLM
//! 5-pass end-to-end (planted wrong-type entity corrected via `mem.dream()`) is
//! the Phase 6 DoD (`dream_e2e_*`), not duplicated here.
//!
//! Deterministic: `run_consistency_check` emits `…consistency_check.scanned_total`
//! on invocation (before any LLM call), so dispatch is provable with a mock LLM.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use metrics_util::debugging::DebuggingRecorder;

#[tokio::test]
async fn dream_dispatches_consistency_check() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dir = tempfile::tempdir().expect("tempdir");
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(64));
    let mem = Memory::open(dir.path().join("phase3-cc.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(64)
        .default_namespace(Namespace::new("phase3-cc"))
        .await
        .expect("Memory::open must succeed");

    // Empty graph → consistency_check loads zero candidates and returns early,
    // but it still emits scanned_total on invocation (ADR-047). No LLM call is
    // made (no flagged candidates), so the null mock LLM is never exercised.
    mem.dream().execute().await.expect("dream must succeed");

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();

    assert!(
        names
            .iter()
            .any(|n| n == "kremory.dream.consistency_check.scanned_total"),
        "mem.dream() must invoke run_consistency_check between reclassify and \
         canonicalize (§D3); scanned_total counter absent. Counters: {names:?}",
    );
    assert!(
        names
            .iter()
            .any(|n| n == "kremory.dream.consistency_check_corrected_total"),
        "facade must emit consistency_check_corrected_total after the pass runs. \
         Counters: {names:?}",
    );
}

/// The per-pass opt-out (`DreamOpts.include_consistency_check = false`) must skip
/// the LLM-cost pass entirely — proving the cost lever added for the readiness
/// gate's "wire consistency_check + mode gating" DoD actually gates (mirrors
/// `include_type_discovery`; the coarser DreamMode::Light is SCOPE-002/Phase 5).
#[tokio::test]
async fn dream_skips_consistency_check_when_opted_out() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dir = tempfile::tempdir().expect("tempdir");
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(64));
    let mem = Memory::open(dir.path().join("phase3-cc-off.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(64)
        .default_namespace(Namespace::new("phase3-cc-off"))
        .await
        .expect("Memory::open must succeed");

    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut opts = kremory::DreamOpts::default();
    opts.include_consistency_check = false;
    mem.dream()
        .with_opts(opts)
        .execute()
        .await
        .expect("dream must succeed with consistency_check opted out");

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();

    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("kremory.dream.consistency_check")),
        "consistency_check must NOT run when include_consistency_check=false; \
         counters seen: {names:?}",
    );
}

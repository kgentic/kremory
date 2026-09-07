#![allow(clippy::unwrap_used, clippy::expect_used)]
//! SCOPE-001 — dream-phase orchestrator control-flow restructure.
//!
//! Governing spec: `.ai-docs/specs/dream-phase-reconciliation-v2-2026-06-30.md`
//! (§SCOPE-001, §D1 one-canonical-orchestrator, §D3 canonical 5-pass ordering),
//! Phase 1 of `.ai-docs/plans/dream-phase-reconciliation-readiness-gate-2026-06-30.md`.
//!
//! ## What this guards
//!
//! The reclassify pass in `facade/dream.rs::execute_blocking` previously did
//! `return Ok(summary)` on success, making every pass ordered AFTER reclassify
//! (consistency_check, canonicalize per §D3) structurally unreachable — the root
//! cause of "live `mem.dream()` runs only 2 of the 5 designed passes" (TD-089).
//!
//! Phase 1 restructures the early-return into a fall-through accumulator so the
//! summary is built once at the end of the pass chain. This test is the
//! mechanical regression guard (readiness-gate R-03 stop condition: "Phase 1
//! restructure test absent → block Phase 2").
//!
//! ## Method
//!
//! Drive `mem.dream()` with a mock LLM on an empty graph. Reclassify finds no
//! candidates → returns `Ok` (the success arm — the exact path that used to
//! early-return). Assert the post-reclassify boundary counter fired, proving
//! control flowed PAST reclassify instead of returning inside its success arm.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use metrics_util::debugging::DebuggingRecorder;

const POST_RECLASSIFY_COUNTER: &str = "kremory.dream.passes_continued_past_reclassify_total";

#[tokio::test]
async fn dream_control_flow_continues_past_reclassify_on_success() {
    // Install a local metrics recorder so the boundary counter is observable.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dir = tempfile::tempdir().expect("tempdir");
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(64));

    let mem = Memory::open(dir.path().join("scope001.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(64)
        .default_namespace(Namespace::new("scope001"))
        .await
        .expect("Memory::open must succeed");

    // Empty graph → reclassify finds no candidates → Ok (the success arm that
    // used to early-return). The pass chain must run to completion regardless.
    let summary = mem
        .dream()
        .execute()
        .await
        .expect("dream must succeed on an empty graph");

    // The single end-of-chain summary build folds the accumulated per-pass
    // counts. On an empty graph reclassify does no work, so this deterministically
    // confirms the accumulator→summary fold ran on the fall-through path.
    // (duration_ms is intentionally NOT asserted > 0 — an empty-graph mock dream
    // can complete in <1ms → 0ms as u128→u64; a non-zero assert would be flaky.)
    assert_eq!(
        summary.entities_reclassified, 0,
        "empty graph → zero entities reclassified; confirms the end-path fold ran",
    );

    let snapshot = snapshotter.snapshot().into_vec();
    let continued = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == POST_RECLASSIFY_COUNTER);

    assert!(
        continued,
        "post-reclassify boundary counter `{POST_RECLASSIFY_COUNTER}` must fire — \
         control must flow PAST the reclassify pass, not early-return inside its \
         success arm (SCOPE-001 / TD-089 regression guard). Counters seen: {:?}",
        snapshot
            .iter()
            .map(|(k, _, _, _)| k.key().name().to_string())
            .collect::<Vec<_>>()
    );
}

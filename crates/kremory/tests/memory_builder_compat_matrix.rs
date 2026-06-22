#![allow(clippy::unwrap_used, clippy::expect_used)]
//! E-2 — MemoryBuilder compat matrix tests (ADR-039, 7-row table).
//!
//! Covers every row of the matrix that is testable without a real LLM or ner feature:
//!
//! | Row | with_llm | with_extractor | with_gliner | Expected result          |
//! |-----|----------|----------------|-------------|--------------------------|
//! | 0   | ✗        | ✗              | ✗           | Err(BuilderConflict)     |
//! | 1   | ✓        | ✗              | ✗           | Ok(Memory) — Llm arm     |
//! | 4   | ✗        | ✓              | ✗           | Ok(Memory) — Custom arm  |
//! | 5   | ✓        | ✓              | ✗           | Ok(Memory) — Custom arm  |
//!
//! Row 2 (LLM + gliner) and Row 3 (gliner without LLM → Err) are covered
//! only when the `ner` feature is active (see feature-gated block below).
//! Row 6 (extractor + gliner conflict) is also ner-gated.

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::Arc;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn unique_db(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_compat_matrix_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

// ── Minimal BYOE extractor for testing ───────────────────────────────────────

/// Zero-entity extractor — satisfies `EntityExtractor` without real LLM.
struct NullExtractor;

impl EntityExtractor for NullExtractor {
    fn name(&self) -> &'static str {
        "null-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![],
            facts: vec![],
        })
    }
}

// ── Row 0: nothing wired → Err(BuilderConflict) ──────────────────────────────

#[tokio::test]
async fn row0_no_llm_no_extractor_returns_builder_conflict() {
    let result = Memory::open(unique_db("row0"))
        .with_embedder(null_embedder())
        .await;

    let Err(err) = result else {
        panic!("row 0 must fail");
    };
    let detail = err.to_string();
    assert!(
        detail.contains("no extractor wired") || detail.contains("BuilderConflict"),
        "expected BuilderConflict about missing extractor, got: {detail}"
    );
}

// ── Row 1: LLM only → Ok(Memory) using Llm arm ───────────────────────────────

#[tokio::test]
async fn row1_llm_only_builds_memory() {
    let mem = Memory::open(unique_db("row1"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("row 1: LLM only should succeed");

    // Category B guard: llm is wired so dream should not return LlmRequired.
    // We don't call dream — just verify the Memory was constructed.
    drop(mem);
}

// ── Row 4: custom extractor, no LLM → Ok(Memory) ─────────────────────────────

#[tokio::test]
async fn row4_custom_extractor_no_llm_builds_memory() {
    let ext = Arc::new(NullExtractor);

    let mem = Memory::open(unique_db("row4"))
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .await
        .expect("row 4: custom extractor without LLM should succeed");

    drop(mem);
}

// ── Row 5: LLM + custom extractor → Ok(Memory), custom wins ──────────────────

#[tokio::test]
async fn row5_llm_and_custom_extractor_builds_memory() {
    let ext = Arc::new(NullExtractor);

    let mem = Memory::open(unique_db("row5"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .await
        .expect("row 5: LLM + custom extractor should succeed");

    drop(mem);
}

// ── Row 6 (ner-gated): extractor + gliner → Err(BuilderConflict) ──────────────

#[cfg(feature = "ner")]
#[tokio::test]
async fn row6_extractor_and_gliner_conflict_returns_err() {
    let ext = Arc::new(NullExtractor);

    let result = Memory::open(unique_db("row6"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .with_gliner()
        .await;

    let Err(err) = result else {
        panic!("row 6 must fail — extractor + gliner conflict");
    };
    let detail = err.to_string();
    assert!(
        detail.contains("conflicts") || detail.contains("BuilderConflict"),
        "expected BuilderConflict about extractor/gliner conflict, got: {detail}"
    );
}

// ── Row 3 (ner-gated): gliner without LLM → Err(BuilderConflict) ─────────────

#[cfg(feature = "ner")]
#[tokio::test]
async fn row3_gliner_without_llm_returns_err() {
    let result = Memory::open(unique_db("row3"))
        .with_embedder(null_embedder())
        .with_gliner()
        .await;

    let Err(err) = result else {
        panic!("row 3: gliner without LLM must fail");
    };
    let detail = err.to_string();
    assert!(
        detail.contains("GLiNER") || detail.contains("LLM") || detail.contains("BuilderConflict"),
        "expected BuilderConflict about GLiNER needing LLM, got: {detail}"
    );
}

// ── F1 (ner-gated): gliner + LLM but EMPTY allowed_entity_types → Err ─────────
//
// ADR adr-memory-builder-gliner-allowlist-fail-loud. GLiNER is closed-vocabulary;
// an empty allowlist (the builder default) silently rejects ALL candidates. The
// build() guard fires BEFORE GlinerLlmExtractor::new, so this test does not load
// the ~650MB model.

#[cfg(feature = "ner")]
#[tokio::test]
async fn gliner_empty_allowlist_errs_at_build() {
    let result = Memory::open(unique_db("gliner_empty_allowlist"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_gliner()
        // deliberately NO .allowed_entity_types(...) — the default empty list
        .await;

    let Err(err) = result else {
        panic!("gliner + empty allowlist must fail loud (silent zero extraction otherwise)");
    };
    let detail = err.to_string();
    assert!(
        detail.contains("allowlist")
            || detail.contains("rejects all")
            || detail.contains("BuilderConflict"),
        "expected BuilderConflict about empty entity-type allowlist, got: {detail}"
    );
}

// ── TD-052b — `with_dream_llm` public surface (spec §8 T2, §4 row 2 / row 3) ──
//
// Governing spec: `.ai-docs/specs/td-052b-dream-llm-slot-spec-2026-06-22.md`.
// These cover the PUBLIC builder surface (method shape + build success). The
// dream-time provider SELECTION (which slot is invoked) is covered by in-crate
// unit tests on `dream_llm_or_main` (facade/mod.rs) where the `pub(crate)`
// accessor + the role counter are reachable.

/// T2 — `with_dream_llm` exists, returns the same type-state, and builds a
/// `Memory` on the WithLlm path (compat matrix row 2: main + dream set).
#[tokio::test]
async fn td052b_with_dream_llm_builds_on_with_llm_path() {
    let main: Arc<dyn kremory::memory::ChatProvider> = null_llm();
    let dream: Arc<dyn kremory::memory::ChatProvider> = null_llm();

    let mem = Memory::open(unique_db("td052b_row2"))
        .with_llm(main)
        .with_dream_llm(dream)
        .with_embedder(null_embedder())
        .await
        .expect("row 2: with_llm + with_dream_llm should build");

    drop(mem);
}

/// T6 row 3 — `with_dream_llm` is callable on the NoLlm builder (it lives on
/// `impl<L,E>`), so a custom-extractor build can carry a dedicated dream model
/// with no main LLM. The build MUST succeed (§4.1: purely additive, no new
/// build-time conflict).
#[tokio::test]
async fn td052b_row3_no_main_llm_custom_extractor_plus_dream_llm_builds() {
    let ext = Arc::new(NullExtractor);
    let dream: Arc<dyn kremory::memory::ChatProvider> = null_llm();

    let mem = Memory::open(unique_db("td052b_row3"))
        .with_extractor(ext)
        .with_dream_llm(dream)
        .with_embedder(null_embedder())
        .await
        .expect("row 3: custom extractor + dream_llm (no main LLM) should build");

    drop(mem);
}

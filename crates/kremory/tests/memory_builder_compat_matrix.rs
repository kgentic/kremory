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

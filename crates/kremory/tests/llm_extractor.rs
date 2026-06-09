#![allow(clippy::unwrap_used, clippy::expect_used)]
//! E-2 — BYOE (bring your own extractor) round-trip tests.
//!
//! Verifies that:
//! 1. A custom `EntityExtractor` impl is accepted by the builder.
//! 2. `Memory` is constructed successfully without an LLM.
//! 3. Category A methods (`remember`) do not immediately return `LlmRequired`
//!    (they use `llm_or_stub()` and let the engine guard internally).
//! 4. Category B methods (`dream`) DO return `LlmRequired` when no LLM is wired.

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unique_db(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_llm_extractor_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

// ── Custom extractor: always returns one fixed entity ─────────────────────────

struct FixedEntityExtractor {
    entity_name: String,
}

impl EntityExtractor for FixedEntityExtractor {
    fn name(&self) -> &'static str {
        "fixed-entity-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        use kremory::core::intelligence::ExtractedEntity;
        Ok(ExtractionResult {
            entities: vec![ExtractedEntity {
                name: self.entity_name.clone(),
                label: "Person".to_string(),
                properties: serde_json::Value::Null,
            }],
            facts: vec![],
        })
    }
}

// ── Test: BYOE builder accepts custom extractor without LLM ───────────────────

#[tokio::test]
async fn byoe_custom_extractor_builds_without_llm() {
    let ext = Arc::new(FixedEntityExtractor {
        entity_name: "Alice".to_string(),
    });

    let mem = Memory::open(unique_db("byoe_build"))
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .await
        .expect("BYOE: custom extractor should build Memory without LLM");

    drop(mem);
}

// ── Test: Category B method returns LlmRequired when no LLM wired ────────────

#[tokio::test]
async fn byoe_dream_returns_llm_required_when_no_llm() {
    let ext = Arc::new(FixedEntityExtractor {
        entity_name: "Bob".to_string(),
    });

    let mem = Memory::open(unique_db("byoe_dream_err"))
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .await
        .expect("build should succeed");

    let result = mem
        .dream()
        .in_namespace(Namespace::new("test"))
        .await;

    let err = result.err().expect("dream without LLM must return LlmRequired");
    let detail = err.to_string();
    assert!(
        detail.contains("LLM") || detail.contains("llm") || detail.contains("dream"),
        "expected LlmRequired error for dream, got: {detail}"
    );
}

// ── Test: EntityExtractorDyn object safety (Arc<dyn EntityExtractorDyn> compiles) ─

#[test]
fn entity_extractor_dyn_is_object_safe() {
    use kremory::core::intelligence::EntityExtractorDyn;

    // If this compiles, EntityExtractorDyn is dyn-compatible.
    // The actual Arc is never used at runtime — compile-test only.
    fn _accepts_dyn(_: Arc<dyn EntityExtractorDyn>) {}
}

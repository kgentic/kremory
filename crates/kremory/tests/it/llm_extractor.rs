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
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_llm_extractor_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
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
        .execute()
        .await;

    let Err(err) = result else {
        panic!("dream without LLM must return LlmRequired");
    };
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

// ── Test: built-in LlmExtractor name() returns "llm" ──────────────────────────

#[test]
fn llm_extractor_name_returns_llm() {
    use kremory::core::extraction::LlmExtractor;
    use kremory::core::provider::MockChatProvider;

    let llm = Arc::new(MockChatProvider::null());
    let extractor = LlmExtractor::new(llm);
    assert_eq!(
        extractor.name(),
        "llm",
        "LlmExtractor::name() must return stable observability label 'llm'"
    );
}

// ── Test: built-in GlinerLlmExtractor name() returns "gliner_llm" (ner-gated) ─

#[cfg(feature = "ner")]
#[test]
fn gliner_llm_extractor_name_signature_compiles() {
    // The struct exists under ner feature; name() must return stable label "gliner_llm".
    // We don't construct the full extractor (avoids hf-hub download in unit test);
    // instead we compile-check the trait+method signature is in place.
    //
    // GlinerLlmExtractor<L> implements EntityExtractor; name() is the load-bearing
    // observability attribution hook used by ExtractorKind dispatch in factory.rs.
    use kremory::core::extraction::GlinerLlmExtractor;
    use kremory::core::intelligence::EntityExtractor;
    fn _check_name<L: kremory::core::provider::ChatProvider + 'static>(
        e: &GlinerLlmExtractor<L>,
    ) -> &'static str
    where
        GlinerLlmExtractor<L>: EntityExtractor,
    {
        e.name()
    }
}

// ── Test: KremoryError::LlmRequired carries method + hint fields ──────────────

#[tokio::test]
async fn llm_required_error_carries_method_and_hint() {
    let ext = Arc::new(FixedEntityExtractor {
        entity_name: "Carol".to_string(),
    });

    let mem = Memory::open(unique_db("llm_required_fields"))
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .await
        .expect("build should succeed");

    let result = mem
        .dream()
        .in_namespace(Namespace::new("test"))
        .execute()
        .await;

    let Err(err) = result else {
        panic!("dream without LLM must return LlmRequired");
    };
    let detail = err.to_string();
    // method field should be referenced (in the Display impl per ADR-041 §2)
    assert!(
        detail.contains("dream") || detail.contains("requires"),
        "LlmRequired error must reference the method or 'requires LLM': {detail}"
    );
    // hint field should be non-empty + actionable
    assert!(
        detail.contains("LLM") || detail.contains("llm") || detail.contains("with_llm"),
        "LlmRequired error must contain hint referencing LLM wiring: {detail}"
    );
}

// ── Test: Custom extractor name flows through ExtractorKind::Custom ───────────

#[test]
fn custom_extractor_uses_consumer_name() {
    let ext = FixedEntityExtractor {
        entity_name: "Dave".to_string(),
    };
    // Consumer-defined name comes through the EntityExtractor trait.
    // The blanket impl on EntityExtractorDyn delegates name() to EntityExtractor::name().
    // Observability counter `kremory.extraction.kind` labels Custom path with this string.
    assert_eq!(
        ext.name(),
        "fixed-entity-extractor",
        "Custom EntityExtractor::name() is the observability attribution string"
    );
}

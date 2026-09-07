//! `ExtractorKind` — enum dispatch for the built-in extractors + custom escape hatch.
//!
//! The consumer-facing API is the composable knobs on `MemoryBuilder` (`.with_llm()`,
//! `.with_gliner()`, `.with_extractor()`).  `ExtractorKind` is the internal dispatch
//! enum that `build()` selects based on which knobs were set.
//!
//! # Variants
//!
//! | Variant      | When selected                                       |
//! |--------------|-----------------------------------------------------|
//! | `Llm`        | `with_llm` set, `with_gliner` not set               |
//! | `GlinerLlm`  | `with_llm` set AND `with_gliner` set (`ner` feature)|
//! | `Custom`     | `with_extractor` set (overrides built-ins)          |
//!
//! Enum dispatch was chosen for the built-in variants because the trait method
//! returns `impl Future` (RPITIT, not dyn-compatible on stable Rust).  The
//! `Custom` variant carries `Arc<dyn EntityExtractorDyn>` (the object-safe twin
//! of `EntityExtractor`) so consumers can bring their own implementation.

use std::sync::Arc;

use metrics::counter;

use crate::core::error::Result;
use crate::core::intelligence::{
    EntityExtractor, EntityExtractorDyn, ExtractionContext, ExtractionResult,
};
use crate::core::provider::ChatProvider;

use super::default_extractor::IntegerIdLlmExtractor;
use super::graphiti::LlmExtractor;
#[cfg(feature = "ner")]
use super::hybrid_typer::GlinerLlmExtractor;

/// Internal dispatch enum — built-in extractors + consumer escape hatch.
///
/// `pub(crate)` — consumers access this only via `ExtractorKind::Custom` through
/// the `MemoryBuilder::with_extractor` knob.
pub(crate) enum ExtractorKind<L: ChatProvider> {
    /// Integer-ID 3-stage extraction (entities → relation names → triplets).
    /// **Production default** (wired in `Engine::new`). On real LLMs this extracts
    /// materially more relationship facts than the `Llm` (graphiti) variant — 16 vs 3
    /// on the `mock_interview` corpus, 3 vs 0 on short prose (qwen2.5:14b, measured
    /// empirically). This reverses an earlier default that used `Llm` for
    /// `Engine::new`.
    IntegerId(IntegerIdLlmExtractor<L>),

    /// Pure-LLM 2-stage extraction (Graphiti-quality prompts, entities → triplets).
    /// Retained as a selectable variant; NOT the default (under-extracts facts vs
    /// `IntegerId` on real LLMs — see above). No builder path constructs it yet, so
    /// `#[allow(dead_code)]` matches the sibling `GlinerLlm`/`Custom` convention for
    /// intentionally-retained-but-unwired variants.
    #[allow(dead_code)]
    Llm(LlmExtractor<L>),

    /// GLiNER span-discovery + ONE LLM typing call.
    /// Requires the `ner` cargo feature.
    /// Construction wired in E-2 via `MemoryBuilder::with_gliner` + `with_llm` knobs.
    #[cfg(feature = "ner")]
    #[allow(dead_code)]
    GlinerLlm(Box<GlinerLlmExtractor<L>>),

    /// Consumer-provided extractor via `MemoryBuilder::with_extractor`.
    /// Uses the object-safe `EntityExtractorDyn` trait so any `EntityExtractor`
    /// implementation can be passed as `Arc<dyn EntityExtractorDyn>`.
    /// Constructed in E-2 via `MemoryBuilder::with_extractor` knob.
    #[allow(dead_code)]
    Custom(Arc<dyn EntityExtractorDyn>),
}

impl<L: ChatProvider + 'static> EntityExtractor for ExtractorKind<L> {
    fn name(&self) -> &'static str {
        match self {
            Self::IntegerId(e) => EntityExtractor::name(e),
            Self::Llm(e) => EntityExtractor::name(e),
            #[cfg(feature = "ner")]
            Self::GlinerLlm(e) => EntityExtractor::name(e.as_ref()),
            Self::Custom(e) => EntityExtractorDyn::name(e.as_ref()),
        }
    }

    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        // Emit per-kind dispatch counter BEFORE the call so failures are attributed.
        counter!("kremory.extraction.kind", "kind" => EntityExtractor::name(self)).increment(1);

        match self {
            Self::IntegerId(e) => e.extract(text, ctx).await,
            Self::Llm(e) => e.extract(text, ctx).await,
            #[cfg(feature = "ner")]
            Self::GlinerLlm(e) => e.extract(text, ctx).await,
            Self::Custom(e) => e.extract_dyn(text, ctx).await,
        }
    }
}

/// `Arc<ExtractorKind<L>>` also implements `EntityExtractor` by forwarding to
/// the inner value.  This avoids callers (test callsites using
/// `ingest_with(&extractor)`) having to write `&*extractor` manually.
impl<L: ChatProvider + 'static> EntityExtractor for Arc<ExtractorKind<L>> {
    fn name(&self) -> &'static str {
        EntityExtractor::name(self.as_ref())
    }

    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        self.as_ref().extract(text, ctx).await
    }
}

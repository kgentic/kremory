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

use super::graphiti::LlmExtractor;
#[cfg(feature = "ner")]
use super::hybrid_typer::GlinerLlmExtractor;

/// Internal dispatch enum — built-in extractors + consumer escape hatch.
///
/// `pub(crate)` — consumers access this only via `ExtractorKind::Custom` through
/// the `MemoryBuilder::with_extractor` knob.
pub(crate) enum ExtractorKind<L: ChatProvider> {
    /// Pure-LLM 3-stage extraction (Graphiti-quality prompts).
    Llm(LlmExtractor<L>),

    /// GLiNER span-discovery + ONE LLM typing call (TD-023).
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

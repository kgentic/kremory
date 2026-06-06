//! Production extractor factory — runtime selection between
//! [`NuExtractExtractor`] (default) and [`HybridGlinerLlmExtractor`]
//! (TD-023, behind the `ner` cargo feature).
//!
//! See `.ai-docs/specs/kremory-v017-hybrid-extractor-production-wire-in-spec-2026-06-04.md`
//! and ADR-029 for the architectural reasoning. Key points:
//!
//! - Enum dispatch (not `Box<dyn>`) because `EntityExtractor::extract` returns
//!   `impl Future` (RPITIT, not dyn-compatible).
//! - Constructed once per `Engine` and shared via `Arc` (GLiNER weights cost
//!   ~650MB to load — per-call construction would thrash).
//! - 3-layer config: builder explicit pin > `KREMORY_EXTRACTOR` env var > NuExtract default.
//! - Explicit pin fails LOUDLY on construct error; env-var path WARNS and
//!   falls back to NuExtract (per KD4).

use std::sync::Arc;

use metrics::counter;

use super::NuExtractExtractor;
#[cfg(feature = "ner")]
use crate::core::error::Error;
use crate::core::error::Result;
use crate::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use crate::core::provider::ChatProvider;

/// Consumer-facing choice of extractor source. Passed to the
/// [`MemoryBuilder`](crate::facade) → [`Engine::new`](crate::core::ingest::Engine)
/// chain at construction time.
///
/// Default: [`ExtractorSource::FromEnv`] — preserves bit-for-bit pre-v0.1.7
/// behaviour when no env var is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractorSource {
    /// Read `KREMORY_EXTRACTOR` env var; fall back to NuExtract.
    /// Unrecognized values WARN and default to NuExtract.
    #[default]
    FromEnv,

    /// Explicit pin to NuExtract regardless of env.
    NuExtract,

    /// Explicit pin to Hybrid. Construction returns Err if GLiNER weights
    /// fail to load (no silent fallback in explicit-pin mode per KD4).
    ///
    /// Requires the `ner` cargo feature; the variant is `cfg`-gated so a
    /// consumer of a non-ner build cannot construct it.
    #[cfg(feature = "ner")]
    Hybrid,
}

/// Production extractor enum — dispatches via match arm to the concrete
/// extractor implementation. Implements [`EntityExtractor`] for the trait
/// surface used by all 8 ingest callsites.
///
/// Enum dispatch was chosen over `Box<dyn EntityExtractor>` because the
/// trait method returns `impl Future` (RPITIT), which is not dyn-compatible
/// on stable Rust. See ADR-029.
pub enum ProductionExtractor<L: ChatProvider> {
    NuExtract(NuExtractExtractor<L>),
    /// Boxed because `HybridGlinerLlmExtractor` carries ~1.2KB of model handles
    /// and would otherwise force `NuExtract` callers to pay the same stack
    /// footprint per enum value.
    #[cfg(feature = "ner")]
    Hybrid(Box<super::hybrid_typer::HybridGlinerLlmExtractor<L>>),
}

/// Private resolved choice — `ExtractorSource::FromEnv` resolves to one of
/// these at construction. Kept private so the public `ExtractorSource` surface
/// only exposes intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedSource {
    NuExtract,
    #[cfg(feature = "ner")]
    Hybrid,
}

impl ResolvedSource {
    fn label(self) -> &'static str {
        match self {
            Self::NuExtract => "nuextract",
            #[cfg(feature = "ner")]
            Self::Hybrid => "hybrid",
        }
    }
}

impl<L: ChatProvider + 'static> ProductionExtractor<L> {
    /// Build the production extractor from a configured source.
    ///
    /// Precedence enforcement: explicit `Hybrid` / `NuExtract` variants bypass
    /// env entirely. Only `FromEnv` consults `KREMORY_EXTRACTOR`. This is a
    /// belt-and-braces invariant — a future maintainer might wrongly add
    /// env-reading inside the `Hybrid` arm thinking it was missed.
    /// Infallible NuExtract constructor. Use when the caller knows it wants the
    /// default extractor (no env consultation, no GLiNER weights) and needs a
    /// `Self` directly without a `Result` to unwrap. The single existing caller
    /// is the test-convenience `EngineGraphHandle::with_config`; production code
    /// should normally use [`Self::from_source`] with [`ExtractorSource::FromEnv`].
    pub fn nuextract_only(llm: Arc<L>) -> Self {
        counter!("kremory.extraction.factory_choice",
            "extractor" => "nuextract")
            .increment(1);
        tracing::debug!(
            target: "kremory.extraction.factory",
            "using NuExtractExtractor (infallible constructor)"
        );
        Self::NuExtract(NuExtractExtractor::new(llm))
    }

    pub fn from_source(source: ExtractorSource, llm: Arc<L>) -> Result<Self> {
        let chosen = match source {
            ExtractorSource::FromEnv => Self::resolve_env(),
            ExtractorSource::NuExtract => ResolvedSource::NuExtract,
            #[cfg(feature = "ner")]
            ExtractorSource::Hybrid => ResolvedSource::Hybrid,
        };

        // Emit BEFORE construction so failures are attributable.
        counter!("kremory.extraction.factory_choice",
            "extractor" => chosen.label())
            .increment(1);

        match chosen {
            ResolvedSource::NuExtract => {
                tracing::debug!(
                    target: "kremory.extraction.factory",
                    "using NuExtractExtractor (default)"
                );
                Ok(Self::NuExtract(NuExtractExtractor::new(llm)))
            }
            #[cfg(feature = "ner")]
            ResolvedSource::Hybrid => {
                tracing::info!(
                    target: "kremory.extraction.factory",
                    "using HybridGlinerLlmExtractor (TD-023)"
                );
                let hybrid = super::hybrid_typer::HybridGlinerLlmExtractor::new(llm)
                    .map_err(|e| Error::ExtractorInit {
                        detail: format!("{e}"),
                    })?;
                Ok(Self::Hybrid(Box::new(hybrid)))
            }
        }
    }

    /// Resolve the `KREMORY_EXTRACTOR` env var to a `ResolvedSource`.
    ///
    /// - unset / empty / "nuextract" → NuExtract
    /// - "hybrid" → Hybrid (cfg gate: only if `ner` feature on; otherwise
    ///   emits a fallback metric and defaults to NuExtract)
    /// - anything else → WARN log + fallback metric + NuExtract
    ///
    /// The fallback metric `factory_choice_fallback{reason, attempted}` is
    /// the observability signal operators dashboard on to spot misconfig
    /// like `KREMORY_EXTRACTOR=hybid` (typo).
    fn resolve_env() -> ResolvedSource {
        let raw = std::env::var("KREMORY_EXTRACTOR").unwrap_or_default();
        let normalized = raw.trim().to_lowercase();
        match normalized.as_str() {
            "" | "nuextract" => ResolvedSource::NuExtract,
            #[cfg(feature = "ner")]
            "hybrid" => ResolvedSource::Hybrid,
            #[cfg(not(feature = "ner"))]
            "hybrid" => {
                tracing::warn!(
                    target: "kremory.extraction.factory",
                    "KREMORY_EXTRACTOR=hybrid requested but kremory was built without `ner` feature; falling back to NuExtract"
                );
                counter!("kremory.extraction.factory_choice_fallback",
                    "reason" => "feature_disabled",
                    "attempted" => "hybrid")
                    .increment(1);
                ResolvedSource::NuExtract
            }
            other => {
                tracing::warn!(
                    target: "kremory.extraction.factory",
                    requested = %other,
                    "unrecognized KREMORY_EXTRACTOR value; defaulting to nuextract"
                );
                counter!("kremory.extraction.factory_choice_fallback",
                    "reason" => "unrecognized_env_value",
                    "attempted" => other.to_string())
                    .increment(1);
                ResolvedSource::NuExtract
            }
        }
    }
}

impl<L: ChatProvider + 'static> EntityExtractor for ProductionExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        match self {
            Self::NuExtract(e) => e.extract(text, ctx).await,
            #[cfg(feature = "ner")]
            Self::Hybrid(e) => e.extract(text, ctx).await,
        }
    }
}

/// `Arc<ProductionExtractor<L>>` also implements `EntityExtractor` by
/// forwarding to the inner value. This avoids requiring callers (especially
/// `ingest_with(&extractor)` callsites in tests) to write `&*extractor`
/// manually. Forwarding is zero-cost — the call is dispatched on the inner
/// match arm exactly as if Arc weren't there.
impl<L: ChatProvider + 'static> EntityExtractor for Arc<ProductionExtractor<L>> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        self.as_ref().extract(text, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_source_label() {
        assert_eq!(ResolvedSource::NuExtract.label(), "nuextract");
        #[cfg(feature = "ner")]
        assert_eq!(ResolvedSource::Hybrid.label(), "hybrid");
    }

    // Env-var tests use a serial mutex so parallel tests don't race on
    // KREMORY_EXTRACTOR. Tests are conservative: each test sets, asserts,
    // unsets explicitly.

    fn with_env<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var(key).ok();
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        let result = f();
        match prior {
            Some(p) => std::env::set_var(key, p),
            None => std::env::remove_var(key),
        }
        result
    }

    #[test]
    fn resolve_env_unset_defaults_nuextract() {
        let resolved = with_env("KREMORY_EXTRACTOR", None, || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn resolve_env_empty_defaults_nuextract() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some(""), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn resolve_env_nuextract_explicit() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("nuextract"), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn resolve_env_garbage_warns_and_defaults() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("hybid"), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn resolve_env_case_insensitive() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("NuExtract"), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn resolve_env_whitespace_trimmed() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("  nuextract  "), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[cfg(feature = "ner")]
    #[test]
    fn resolve_env_hybrid_ner_on() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("hybrid"), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::Hybrid);
    }

    #[cfg(not(feature = "ner"))]
    #[test]
    fn resolve_env_hybrid_ner_off_falls_back() {
        let resolved = with_env("KREMORY_EXTRACTOR", Some("hybrid"), || {
            ProductionExtractor::<crate::core::provider::MockChatProvider>::resolve_env()
        });
        assert_eq!(resolved, ResolvedSource::NuExtract);
    }

    #[test]
    fn extractor_source_default_is_from_env() {
        assert_eq!(ExtractorSource::default(), ExtractorSource::FromEnv);
    }
}

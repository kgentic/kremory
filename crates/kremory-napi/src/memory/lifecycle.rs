//! `open()` / `close()` — construction and teardown of a `JsMemory` handle,
//! plus the Tier-2 BYOM/BYOE `open_with_js_embedder` helper used by `open()`'s
//! live cdylib path (not `#[napi]` itself — a private helper `open` calls into).

use napi_derive::napi;
use super::JsMemory;
// Gated to match their ONLY consumers. Every `bridge::` and `convert::`
// reference in this file sits inside the `#[cfg(not(test))]` helper block
// below (deliberately excluded from test builds -- see the note above it).
// Ungated, these imports have no consumer when `cfg(test)` is on, and the
// crate's `-D unused-imports` turns that into a hard error in the lib-test
// target. An import must be conditional on exactly what its users are.
#[cfg(not(test))]
use crate::bridge;
#[cfg(not(test))]
use crate::convert;
use crate::convert::*;
use kremory::{Memory, Namespace};

#[napi]
impl JsMemory {
    /// Open a kremory Memory at `path`.
    ///
    /// ## Tier-1 path (default, backward-compat)
    ///
    /// When no extractor knobs are set in `opts`, uses env-detected providers
    /// (`OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`) via
    /// `Memory::auto`. Behavior is identical to v0.1.6-alpha.0.
    ///
    /// ## Tier-2 path (BYOM embedder)
    ///
    /// When `opts.withEmbedder` is a callback `(text: string) => Promise<number[]>`,
    /// the env-detected LLM is combined with the JS callback as the embedding
    /// provider via `MemoryBuilder::with_embedder`. Set `opts.embeddingDim` to
    /// the callback's output dimension — a mismatch yields a descriptive error.
    ///
    /// ## BYOE extractor knobs
    ///
    /// Composable knobs that mirror the Rust `MemoryBuilder`:
    ///   - `opts.gliner`              → enables `ExtractorKind::GlinerLlm` (requires `--features ner`)
    ///   - `opts.extractor`           → enables `ExtractorKind::Custom` (BYOE)
    ///   - `opts.gliner + extractor`  → `BuilderConflict` error
    ///
    /// ## `opts.defaultNamespace`
    ///
    /// When set, becomes the handle-level default namespace applied to subsequent
    /// ingest/recall calls that omit per-call namespace.
    #[napi(factory)]
    pub async fn open(path: String, opts: Option<JsOpenOptions>) -> napi::Result<JsMemory> {
        // Live napi/cdylib path: ThreadsafeFunction requires the napi runtime.
        #[cfg(not(test))]
        {
            let default_namespace = opts
                .as_ref()
                .and_then(|o| o.default_namespace.as_deref())
                .map(Namespace::new);
            let expected_dim = opts
                .as_ref()
                .and_then(|o| o.embedding_dim)
                .and_then(|d| usize::try_from(d).ok());

            // Detect which extractor knobs are set.
            let has_gliner = opts.as_ref().is_some_and(|o| o.gliner.is_some());
            let has_extractor = opts.as_ref().is_some_and(|o| o.extractor.is_some());

            // Conflict: gliner + extractor simultaneously is a BuilderConflict.
            if has_gliner && has_extractor {
                return Err(napi::Error::from_reason(
                    "KremoryError::BuilderConflict: \
                     opts.gliner and opts.extractor are mutually exclusive — \
                     set gliner (for GlinerLlm) OR extractor (for Custom), not both",
                ));
            }

            // ner-feature guard for GLiNER.
            #[cfg(not(feature = "ner"))]
            if has_gliner {
                return Err(napi::Error::from_reason(
                    "KremoryError::FeatureDisabled('ner'): \
                     opts.gliner requires kremory-napi built with --features ner",
                ));
            }

            // Extractor knobs require a BYOM embedder.
            // All valid rows that include gliner or extractor also include withEmbedder.
            if (has_gliner || has_extractor)
                && opts.as_ref().is_none_or(|o| o.with_embedder.is_none())
            {
                return Err(napi::Error::from_reason(
                    "KremoryError::BuilderConflict: \
                     opts.gliner / opts.extractor require opts.withEmbedder — \
                     provide a BYOM embedder callback alongside the extractor knob",
                ));
            }

            // Unpack opts, consuming it.
            let (with_embedder_tsfn, gliner_cfg, extractor_handle) = match opts {
                Some(o) => (o.with_embedder, o.gliner, o.extractor),
                None => (None, None, None),
            };

            // Tier-2 BYOM embedder path: wire JS embedder + LLM via MemoryBuilder.
            if let Some(tsfn) = with_embedder_tsfn {
                return open_with_js_embedder(
                    path,
                    tsfn,
                    expected_dim,
                    default_namespace,
                    gliner_cfg,
                    extractor_handle,
                )
                .await;
            }

            // Plain Tier-1: no knobs set (already validated above).
            let mem = Memory::auto(&path)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;
            Ok(JsMemory {
                inner: mem,
                default_namespace,
            })
        }

        // Test path: napi runtime absent — use Memory::auto only.
        // Extractor knobs are tested via MockExtractorBridge directly in
        // extractor_selection_compat_matrix.rs.
        #[cfg(test)]
        {
            let default_namespace = opts
                .as_ref()
                .and_then(|o| o.default_namespace.as_deref())
                .map(Namespace::new);

            let mem = Memory::auto(&path)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;

            Ok(JsMemory {
                inner: mem,
                default_namespace,
            })
        }
    }

    /// Close the memory handle, flushing any pending writes.
    ///
    /// Calling `close()` is a no-op at v0.1.0; WAL flush semantics land in
    /// v0.1.1. Call this at shutdown to future-proof your code.
    #[napi]
    pub async fn close(&self) -> napi::Result<()> {
        self.inner
            .close()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory close failed: {e}")))
    }
}

// ── Tier-2 BYOM / BYOE helpers (live cdylib path only) ───────────────────────
//
// These helpers are excluded from `#[cfg(test)]` because:
// - `ThreadsafeFunction` cannot be constructed outside a napi runtime.
// - `resolve_env_llm` constructs real LLM provider instances that require
//   live network endpoints.
//
// Integration tests cover the extractor-selection matrix via `MockExtractorBridge`
// in `tests/extractor_selection_compat_matrix.rs`.
//
// # Typestate note
//
// `MemoryBuilder::IntoFuture` is only implemented for `<WithLlm, WithEmbedder>` and
// `<NoLlm, WithEmbedder>`. Therefore every builder path that ends in `.await` must have
// both an LLM (optional for NoLlm path) AND an embedder set. The `withEmbedder`
// callback is required when extractor knobs are used — the compat matrix
// lists no row where an extractor is set without a BYOM embedder.

/// Open a Memory with a caller-supplied JS embedder callback (Tier-2).
///
/// Also applies any BYOE extractor / GLiNER knobs from the options object.
/// Called when `opts.withEmbedder` is present.
///
/// Builder typestate path: `NoLlm,NoEmbedder` → `with_llm` → `WithLlm,NoEmbedder`
///   → `with_embedder` → `WithLlm,WithEmbedder` → `.await`.
#[cfg(not(test))]
async fn open_with_js_embedder(
    path: String,
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
    expected_dim: Option<usize>,
    default_namespace: Option<Namespace>,
    gliner_cfg: Option<convert::GlinerConfigJs>,
    extractor_handle: Option<bridge::ExternalExtractorHandle>,
) -> napi::Result<JsMemory> {
    use std::sync::Arc;

    // Wrap the JS embedder callback (needed on both typestate paths below).
    let emb: Arc<dyn kremory::DynEmbeddingProvider> =
        bridge::into_arc(bridge::JsEmbedderBridge::new(tsfn, expected_dim));

    // `embeddingDim` must reach the SCHEMA, not only the bridge's length check.
    //
    // It used to do only the latter: `expected_dim` was handed to
    // `JsEmbedderBridge` to validate the callback's output length, and never to
    // `MemoryBuilder::embedding_dim`. So the vector columns and indexes were
    // created at the 384 default whatever the caller asked for, a caller with a
    // 16-dim embedder wrote 16-dim vectors into a 384-dim column, the insert was
    // rejected, and `make_pinned_entity_recallable`'s best-effort branch swallowed
    // it. `remember()` returned `warnings: []`, the JS callback ran cleanly, and
    // every pinned entity was silently invisible to every vector arm.
    //
    // Measured before the fix, same scenario both SDKs: Rust `.embedding_dim(16)`
    // stored a 64-byte embedding and recalled 2 facts; Node `embeddingDim: 16`
    // stored NULL and recalled 0.
    let base = match expected_dim {
        Some(dim) => Memory::open(&path).embedding_dim(dim),
        None => Memory::open(&path),
    };

    // A BYOE extractor is exactly the documented "NoLlm
    // typestate" case (`{ embedder, extractor }` → `ExtractorKind::Custom`,
    // per this fn's own doc comment above). Env-detecting an LLM here was
    // unconditional — forcing `MemoryBuilder<WithLlm, _>` regardless of
    // `extractor_handle` — which required OLLAMA_HOST/OPENAI_API_KEY/
    // ANTHROPIC_API_KEY for a path whose entire point is not needing one.
    //
    // FIRST DRAFT of this fix unconditionally skipped `resolve_env_llm()`
    // whenever an extractor was supplied — WRONG, caught by re-running this
    // fn's own doc-cited example (`07-byoe-custom-extractor.mjs`) WITH
    // OLLAMA_HOST set: it regressed the substrate's own compat-matrix Row 5
    // (`memory_builder_compat_matrix.rs::row5_llm_and_custom_extractor_builds_memory`
    // — "LLM + custom extractor → Ok(Memory), custom wins") by NEVER wiring
    // an available LLM once an extractor was present, even though the LLM
    // stays needed for entity-RESOLUTION (`ingest_with`'s `CascadeResolver`)
    // and Category B ops, independent of which extractor produced the
    // entities. Corrected: check env-var PRESENCE directly (not swallow
    // every `resolve_env_llm()` error) so a genuine misconfiguration (e.g.
    // `OLLAMA_HOST` set to a malformed URL) still surfaces loudly instead of
    // silently downgrading to the NoLlm path.
    if let Some(handle) = extractor_handle {
        let extractor = Arc::new(bridge::ExternalExtractorJs::from_handle(handle));

        let has_env_llm = std::env::var("OLLAMA_HOST").is_ok()
            || std::env::var("OPENAI_API_KEY").is_ok()
            || std::env::var("ANTHROPIC_API_KEY").is_ok();

        let build_result = if has_env_llm {
            let llm = bridge::resolve_env_llm().await?;
            base.with_llm(llm)
                .with_embedder(emb)
                .with_extractor(extractor)
                .await
        } else {
            base.with_embedder(emb).with_extractor(extractor).await
        };

        let mem = build_result.map_err(|e| {
            napi::Error::from_reason(format!("kremory open with embedder failed: {e}"))
        })?;
        return Ok(JsMemory {
            inner: mem,
            default_namespace,
        });
    }

    // No BYOE extractor: GLiNER (still LLM-dependent for entity-type
    // classification, per `.with_gliner()`'s own doc comment) or a plain
    // BYOM-embedder-only open — both need a real LLM.
    let llm = bridge::resolve_env_llm().await?;
    let builder = base.with_llm(llm).with_embedder(emb);

    // Shadowed (not `mut`-reassigned): under a build without the `ner`
    // feature, the `gliner_cfg.is_some()` arm always diverges (`return
    // Err(..)`), so `builder` would otherwise never be reassigned and `mut`
    // would trip `-D unused-mut` (real regression hit while building this
    // fix — `napi build` denies warnings). GLiNER requires ner feature;
    // already checked in open(). F2: with_gliner() takes no arg (GlinerConfig
    // had no public fields) — the presence of opts.gliner is the enable
    // signal; its contents are unused.
    let builder = if gliner_cfg.is_some() {
        #[cfg(feature = "ner")]
        {
            builder.with_gliner()
        }
        #[cfg(not(feature = "ner"))]
        {
            return Err(napi::Error::from_reason(
                "KremoryError::FeatureDisabled('ner'): opts.gliner requires --features ner",
            ));
        }
    } else {
        builder
    };

    let mem = builder
        .await
        .map_err(|e| napi::Error::from_reason(format!("kremory open with embedder failed: {e}")))?;

    Ok(JsMemory {
        inner: mem,
        default_namespace,
    })
}

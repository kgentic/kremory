//! `JsEmbedderBridge` — adapts a JS callback into `kremory::DynEmbeddingProvider`.
//!
//! # Design
//!
//! aidocs (and other Node.js consumers) own their own embedders (e.g. a 256-dim
//! in-process model). To wire a JS-side embedder into kremory's vector index, the
//! consumer passes a callback `(text: string) => Promise<number[]>` via
//! `JsOpenOptions.withEmbedder` at `Memory.open` time. This module wraps that
//! callback in a struct that implements `kremory::DynEmbeddingProvider`, fulfilling
//! the Tier-2 BYOM contract from ADR-030.
//!
//! # API shape
//!
//! The single factory `JsMemory::open(path, opts)` accepts `opts.withEmbedder` as
//! an optional callback. When present, the builder path is taken:
//!   1. Env-detect LLM (OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err).
//!   2. Wrap the JS callback in `JsEmbedderBridge`.
//!   3. Call `Memory::open(path).with_llm(llm).with_embedder(bridge).await`.
//!
//! When absent, `Memory::auto(&path).await` is used (backward-compat path).
//!
//! # Threading model
//!
//! The live napi path stores the JS callback as a
//! `napi::threadsafe_function::ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>`.
//! `ThreadsafeFunction` is `Send + Sync` (napi-rs guarantee; enforced via its own
//! `unsafe impl Send/Sync`). This means `JsEmbedderBridge` is safe to hold in an
//! `Arc` and call from tokio tasks.
//!
//! `call_async::<Vec<f64>>` suspends the calling tokio task, posts the call to the
//! Node.js event loop, and resolves when the JS Promise settles. f64 → f32 cast
//! happens at this boundary (JS Number is always f64; kremory uses f32).
//!
//! # Why `#[cfg(test)]` vs live are separate struct variants
//!
//! `cargo test` builds an ordinary binary that does NOT link napi symbols — the
//! napi crate requires the cdylib ABI to communicate with the Node.js runtime.
//! `ThreadsafeFunction` in particular calls `napi_create_threadsafe_function` at
//! construction which is an undefined symbol in a standard test binary. This is a
//! LEGITIMATE constraint, not a band-aid: the cause is that napi-rs is tightly
//! coupled to the Node.js runtime ABI. The cause-fix is to make the behavioural
//! logic (dim validation, error propagation) testable via a mock variant that
//! shares the same `DynEmbeddingProvider` impl trait but does not call into napi.
//! The live napi path is thin (just the `ThreadsafeFunction::call_async` dispatch)
//! and is covered by JS smoke tests in `__test__/smoke-embedder.test.mjs`.
//!
//! # Precision note (f64 → f32)
//!
//! Embedding values are typically in [-1, 1]. f32 has ~7 decimal digits of
//! precision. The maximum representable error is ~1.2e-7 per element, well
//! within the noise floor of cosine-similarity ranking.
//!
//! # Dim validation
//!
//! When `expected_dim` is `Some(n)`, the bridge validates that the returned
//! embedding length equals `n`. Mismatch → `kremory::CoreError` with a descriptive
//! message naming both expected and actual dimensions. Never panics.
//!
//! # Token counting
//!
//! JS callbacks have no token-count API. `last_usage_tokens_dyn` always returns
//! `None`. Consumers that need token attribution should implement a wrapper.
//!
//! # LLM env-detection
//!
//! `resolve_env_llm` replicates the detection logic from `kremory::facade::providers::auto`
//! but returns only the `Arc<dyn ChatProvider>` (not the full Memory), enabling
//! the caller to wire a custom embedder via `MemoryBuilder::with_embedder`.
//! `autoagents-llm` is added as a direct dep for this purpose — the same crate
//! that kremory uses internally, redirected by the workspace `[patch.crates-io]`
//! to the kgentic fork. No kremory-internal type is leaked at this boundary.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::anyhow;
use kremory::{CoreError, DynEmbeddingProvider};

#[cfg(not(test))]
use kremory::ChatProvider;

// ── ExternalExtractorHandle + ExternalExtractorJs ─────────────────────────────
//
// BYOE (Bring-Your-Own-Extractor) bridge: adapts a JS callback pair into
// `kremory::core::intelligence::EntityExtractor`.
//
// # JS interface
//
// The consumer passes an object with:
//   - `name: string`               — short identifier for metrics / logging
//   - `extract: (text: string) => Promise<ExtractionResult>` — extraction fn
//
// `ExtractionContext` is not forwarded to JS: its fields are `&'a [T]` slices
// with lifetimes that cannot cross FFI. The simplified contract (text only) is
// sufficient for v0.2.0 BYOE (ADR-039 §6). Full context forwarding is deferred
// to a future release when a serialised ContextSnapshot type is designed.
//
// # Threading model
//
// The live path stores the extract function as a
// `napi::threadsafe_function::ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>`.
// `name` is stored as a `Box<str>` and leaked once to produce `&'static str`
// for the `EntityExtractor::name()` contract. This is a single small allocation
// per handle — acceptable for the open-time construction.
//
// # `#[cfg(not(test))]` / `#[cfg(test)]` split
//
// Same reason as `JsEmbedderBridge` — napi ABI symbols are absent in test
// binaries. The mock variant (`MockExtractorBridge`) is used by the
// `extractor_selection_compat_matrix` integration tests.

/// Live (napi/cdylib) BYOE extractor handle received from JS at `Memory.open` time.
///
/// JS consumers pass `{ name: string, extract: (text: string) => Promise<{entities,facts}> }`.
/// `object_to_js = false` suppresses `ToNapiValue` generation — `ExternalExtractorHandle`
/// is only ever constructed from JS → Rust, never returned to JS.
#[cfg(not(test))]
#[napi_derive::napi(object, object_to_js = false, js_name = "ExternalExtractor")]
pub struct ExternalExtractorHandle {
    /// Short identifier for this extractor — used in metrics labels and log output.
    pub name: String,
    /// The JS extraction callback.
    ///
    /// Receives `text: string`. Returns a `Promise` resolving to:
    /// `{ entities: Array<{name: string, label: string}>, facts: Array<{subject: string, predicate: string, object: string}> }`
    #[napi(
        ts_type = "(text: string) => Promise<{ entities: Array<{ name: string, label: string }>, facts: Array<{ subject: string, predicate: string, object: string }> }>"
    )]
    pub extract: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
}

/// Live BYOE extractor: wraps a JS callback + leaked name into an `EntityExtractor`.
///
/// Constructed via `ExternalExtractorHandle::into_bridge()` (see `open_with_extractor`
/// in `lib.rs`). Holds a `ThreadsafeFunction` that posts text to the Node.js event
/// loop and resolves when the JS Promise settles.
#[cfg(not(test))]
pub struct ExternalExtractorJs {
    /// Leaked once at construction — satisfies `EntityExtractor::name() -> &'static str`.
    name: &'static str,
    /// The JS extract callback as a thread-safe napi handle.
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
}

#[cfg(not(test))]
impl ExternalExtractorJs {
    /// Construct from an `ExternalExtractorHandle`, leaking the name string once.
    pub fn from_handle(handle: ExternalExtractorHandle) -> Self {
        // Leak once per handle — small String, acceptable cost at open() time.
        let name: &'static str = Box::leak(handle.name.into_boxed_str());
        Self {
            name,
            tsfn: handle.extract,
        }
    }
}

/// JS extraction response shape — must match the TS interface declared in
/// `ExternalExtractorHandle.extract` ts_type annotation.
#[cfg(not(test))]
#[napi_derive::napi(object)]
pub struct JsExtractedEntity {
    pub name: String,
    pub label: String,
}

#[cfg(not(test))]
#[napi_derive::napi(object)]
pub struct JsExtractedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[cfg(not(test))]
#[napi_derive::napi(object)]
pub struct JsExtractionResult {
    pub entities: Vec<JsExtractedEntity>,
    pub facts: Vec<JsExtractedFact>,
}

#[cfg(not(test))]
impl kremory::core::intelligence::EntityExtractor for ExternalExtractorJs {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extract<'a>(
        &'a self,
        text: &'a str,
        _ctx: &'a kremory::core::intelligence::ExtractionContext<'a>,
    ) -> impl std::future::Future<
        Output = kremory::core::error::Result<kremory::core::intelligence::ExtractionResult>,
    > + Send
           + 'a {
        let text_owned = text.to_owned();
        let tsfn = self.tsfn.clone();

        async move {
            let js_result: JsExtractionResult = tsfn
                .call_async(Ok(text_owned))
                .await
                .map_err(|e| kremory::CoreError::Other(anyhow!("extractor callback error: {e}")))?;

            let entities = js_result
                .entities
                .into_iter()
                .map(|e| kremory::core::intelligence::ExtractedEntity {
                    name: e.name,
                    label: e.label,
                    properties: serde_json::Value::Null,
                })
                .collect();

            let facts = js_result
                .facts
                .into_iter()
                .map(|f| kremory::core::intelligence::ExtractedFact {
                    subject: f.subject,
                    predicate: f.predicate,
                    object: f.object,
                    is_entity_ref: false,
                    confidence: 1.0,
                })
                .collect();

            Ok(kremory::core::intelligence::ExtractionResult { entities, facts })
        }
    }
}

// ── Mock BYOE extractor (test path) ──────────────────────────────────────────

/// Test-only BYOE extractor mock.
///
/// Does not require a live Node.js runtime. Three variants cover the main
/// test dimensions: happy path (fixed result), error path, and empty result.
#[cfg(test)]
pub struct MockExtractorBridge {
    kind: MockExtractorKind,
}

#[cfg(test)]
pub enum MockExtractorKind {
    /// Returns a fixed `ExtractionResult` on every call.
    Fixed(kremory::core::intelligence::ExtractionResult),
    /// Every call returns a descriptive error.
    Error { message: String },
    /// Returns an empty `ExtractionResult` (no entities, no facts).
    Empty,
}

#[cfg(test)]
impl MockExtractorBridge {
    /// Fixed-result mock — returns `result` on every call.
    pub fn new_fixed(result: kremory::core::intelligence::ExtractionResult) -> Self {
        Self {
            kind: MockExtractorKind::Fixed(result),
        }
    }

    /// Error mock — every call returns an error containing `message`.
    pub fn new_error(message: impl Into<String>) -> Self {
        Self {
            kind: MockExtractorKind::Error {
                message: message.into(),
            },
        }
    }

    /// Empty mock — returns no entities and no facts.
    pub fn new_empty() -> Self {
        Self {
            kind: MockExtractorKind::Empty,
        }
    }
}

#[cfg(test)]
impl kremory::core::intelligence::EntityExtractor for MockExtractorBridge {
    fn name(&self) -> &'static str {
        "mock-extractor"
    }

    fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a kremory::core::intelligence::ExtractionContext<'a>,
    ) -> impl std::future::Future<
        Output = kremory::core::error::Result<kremory::core::intelligence::ExtractionResult>,
    > + Send
           + 'a {
        let result: kremory::core::error::Result<kremory::core::intelligence::ExtractionResult> =
            match &self.kind {
                MockExtractorKind::Fixed(r) => Ok(r.clone()),
                MockExtractorKind::Error { message } => Err(kremory::CoreError::Other(anyhow!(
                    "extractor error: {message}"
                ))),
                MockExtractorKind::Empty => Ok(kremory::core::intelligence::ExtractionResult {
                    entities: vec![],
                    facts: vec![],
                }),
            };
        std::future::ready(result)
    }
}

// ── Live (napi/cdylib) path ───────────────────────────────────────────────────

/// JS-callback-backed embedding provider for the live napi cdylib build.
///
/// This struct is only instantiated when building the .node binary. In `cargo test`
/// runs (which cannot link napi symbols — see module doc), the mock variant below
/// is used instead.
#[cfg(not(test))]
pub struct JsEmbedderBridge {
    /// The JS callback held as a thread-safe napi handle.
    /// `ThreadsafeFunction` is `Send + Sync` by napi-rs construction.
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        String,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
    /// Expected embedding dimensionality from `JsOpenOptions.embeddingDim`.
    /// When `Some(n)`, the bridge validates the returned vec length against n.
    expected_dim: Option<usize>,
}

#[cfg(not(test))]
impl JsEmbedderBridge {
    /// Construct from a napi `ThreadsafeFunction` and an optional expected dim.
    pub fn new(
        tsfn: napi::threadsafe_function::ThreadsafeFunction<
            String,
            napi::threadsafe_function::ErrorStrategy::CalleeHandled,
        >,
        expected_dim: Option<usize>,
    ) -> Self {
        Self { tsfn, expected_dim }
    }
}

#[cfg(not(test))]
impl DynEmbeddingProvider for JsEmbedderBridge {
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CoreError>> + Send + 'a>> {
        let text_owned = text.to_owned();
        let expected_dim = self.expected_dim;

        // ThreadsafeFunction::clone() is safe — increments a ref-count inside
        // the napi handle (napi-2.16.x threadsafe_function.rs:244-257).
        // We clone here so the future can cross thread boundaries.
        let tsfn = self.tsfn.clone();

        Box::pin(async move {
            // call_async suspends this tokio task and resumes when the JS Promise
            // resolves. Vec<f64> maps to JS `number[]` (JS Number is always f64).
            let f64_vec: Vec<f64> = tsfn
                .call_async(Ok(text_owned))
                .await
                .map_err(|e| CoreError::Other(anyhow!("embedder callback error: {e}")))?;

            // f64 → f32 cast: see precision note in module doc.
            let f32_vec: Vec<f32> = f64_vec.iter().map(|&v| v as f32).collect();

            // Dim validation after conversion so the error message can cite both dims.
            validate_dim(&f32_vec, expected_dim)?;

            Ok(f32_vec)
        })
    }

    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        // JS embedder callbacks have no token-count API.
        None
    }
}

// ── Mock (test) path ──────────────────────────────────────────────────────────

/// Test-only variant of `JsEmbedderBridge`.
///
/// Does not require a live Node.js runtime. Four constructors cover the main
/// test dimensions: happy path, error path, dim-check, and call counting.
#[cfg(test)]
pub struct JsEmbedderBridge {
    kind: MockKind,
}

#[cfg(test)]
enum MockKind {
    /// Returns a fixed vec on every call. Optional dim check.
    Fixed {
        vec: Vec<f32>,
        expected_dim: Option<usize>,
    },
    /// Every call returns a descriptive error.
    Error { message: String },
    /// Returns vec and increments a shared counter. Dim is validated.
    Counting {
        vec: Vec<f32>,
        expected_dim: usize,
        counter: Arc<std::sync::atomic::AtomicUsize>,
    },
}

#[cfg(test)]
impl JsEmbedderBridge {
    /// Happy-path mock: returns `vec` on every call. No dim validation.
    /// For dim-validated mocks, use `new_mock_with_dim_check`.
    pub fn new_mock(vec: Vec<f32>) -> Self {
        Self {
            kind: MockKind::Fixed {
                vec,
                expected_dim: None,
            },
        }
    }

    /// Error mock: every call returns a descriptive error containing `message`.
    pub fn new_error_mock(message: impl Into<String>) -> Self {
        Self {
            kind: MockKind::Error {
                message: message.into(),
            },
        }
    }

    /// Dim-check mock: returns `vec` but validates length against `expected_dim`.
    /// When lengths differ, embed_dyn returns a descriptive CoreError.
    pub fn new_mock_with_dim_check(vec: Vec<f32>, expected_dim: usize) -> Self {
        Self {
            kind: MockKind::Fixed {
                vec,
                expected_dim: Some(expected_dim),
            },
        }
    }

    /// Counting mock: returns `vec` and increments `counter` on each embed_dyn call.
    pub fn new_counting_mock(
        vec: Vec<f32>,
        expected_dim: usize,
        counter: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            kind: MockKind::Counting {
                vec,
                expected_dim,
                counter,
            },
        }
    }
}

#[cfg(test)]
impl DynEmbeddingProvider for JsEmbedderBridge {
    fn embed_dyn<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CoreError>> + Send + 'a>> {
        let result: Result<Vec<f32>, CoreError> = match &self.kind {
            MockKind::Fixed { vec, expected_dim } => match validate_dim(vec, *expected_dim) {
                Err(e) => Err(e),
                Ok(()) => Ok(vec.clone()),
            },
            MockKind::Error { message } => Err(CoreError::Other(anyhow!(
                "embedder callback error: {message}"
            ))),
            MockKind::Counting {
                vec,
                expected_dim,
                counter,
            } => {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                match validate_dim(vec, Some(*expected_dim)) {
                    Err(e) => Err(e),
                    Ok(()) => Ok(vec.clone()),
                }
            }
        };
        Box::pin(std::future::ready(result))
    }

    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

// ── Shared validation helper ──────────────────────────────────────────────────

/// Validate that `vec.len() == expected_dim` when `expected_dim` is `Some`.
///
/// Returns `Ok(())` when no expected dim is set or lengths match.
/// Returns `Err(CoreError)` with a descriptive message on mismatch.
/// Never panics.
fn validate_dim(vec: &[f32], expected_dim: Option<usize>) -> Result<(), CoreError> {
    if let Some(expected) = expected_dim {
        if vec.len() != expected {
            return Err(CoreError::Other(anyhow!(
                "embedder dimension mismatch: expected {expected}-dim embedding \
                 but callback returned {}-dim. Ensure your embedder model output \
                 matches the `embeddingDim` passed to Memory.open.",
                vec.len()
            )));
        }
    }
    Ok(())
}

// ── Arc constructor helper ────────────────────────────────────────────────────

/// Wrap any `DynEmbeddingProvider` implementor in `Arc<dyn DynEmbeddingProvider>`
/// for use with `MemoryBuilder::with_embedder`.
///
/// The primary caller is `lib.rs` (wrapping `JsEmbedderBridge`). Integration
/// tests pass custom mock implementations to verify the `Arc` wrapping is correct.
pub fn into_arc<T: DynEmbeddingProvider + 'static>(provider: T) -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(provider)
}

// ── LLM env-detection helper (live path only) ─────────────────────────────────

/// Env-detect an LLM provider for the BYOM bridge Tier-2 path.
///
/// Replicates the detection logic from `kremory::facade::providers::auto` but
/// returns only `Arc<dyn ChatProvider>` — not a full `Memory`. This allows the
/// caller to wire a custom embedder via `MemoryBuilder::with_embedder`.
///
/// Detection order: `OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → Err.
///
/// `autoagents-llm` is used directly here because:
///   - `kremory::facade::providers` does not expose an LLM-only detection function.
///   - The workspace `[patch.crates-io]` redirects `autoagents-llm 0.3.x` to
///     the kgentic fork (feat/chat-provider-model-accessor), matching kremory's
///     own dependency. This guarantees version consistency.
///
/// This function is excluded from `#[cfg(test)]` because cargo test builds cannot
/// link napi, and the LLM construction path also requires live providers. Tests
/// for the bridge logic use mock variants that bypass this function.
#[cfg(not(test))]
pub(crate) async fn resolve_env_llm() -> napi::Result<Arc<dyn ChatProvider>> {
    use autoagents_llm::{
        backends::{anthropic::Anthropic, ollama::Ollama, openai::OpenAI},
        builder::LLMBuilder,
    };

    if let Ok(host) = std::env::var("OLLAMA_HOST") {
        // Default: gemma4:e4b — ALIGNED with the Rust `Memory::with_ollama` default
        // (facade/mod.rs) and the validated-best default (F1 85.7 @ 11.6s per
        // `project_kremory_validated_model_findings_2026-06-24`). This closes the
        // N4 binding-parity gap where the napi default (`qwen2.5:14b`) differed from
        // Rust's, giving the SAME library two default brains depending on binding.
        // Override via OLLAMA_CHAT_MODEL env var. Avoid `-mlx` variants pending
        // upstream autoagents-llm structured-output patches.
        let model =
            std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

        // Default 30s timeout — Apple Silicon MLX/14B cold-load or model-swap
        // routinely exceeds 10s (kremory facade convention). Override via
        // OLLAMA_TIMEOUT_SECS env var for slower hardware or contention.
        let timeout_secs: u64 = std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        // Default keep_alive 1h — autoagents-llm defaults to "0" (immediate
        // unload after each call), which causes model swap thrash on
        // GPU-constrained hosts when multiple consumers share Ollama.
        // Override via OLLAMA_KEEP_ALIVE env var.
        let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "1h".to_string());

        let chat: Arc<Ollama> = LLMBuilder::<Ollama>::new()
            .base_url(&host)
            .model(model)
            .timeout_seconds(timeout_secs)
            .keep_alive(keep_alive)
            .build()
            .map_err(|e| napi::Error::from_reason(format!("Ollama LLM build failed: {e}")))?;

        return Ok(chat as Arc<dyn ChatProvider>);
    }

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let chat: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
            .api_key(&key)
            .model("gpt-4o-mini")
            .build()
            .map_err(|e| napi::Error::from_reason(format!("OpenAI LLM build failed: {e}")))?;

        return Ok(chat as Arc<dyn ChatProvider>);
    }

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let chat: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
            .api_key(&key)
            .model("claude-3-haiku-20240307")
            .build()
            .map_err(|e| napi::Error::from_reason(format!("Anthropic LLM build failed: {e}")))?;

        return Ok(chat as Arc<dyn ChatProvider>);
    }

    Err(napi::Error::from_reason(
        "kremory open with embedder: no LLM provider configured — \
         set OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY",
    ))
}

// ── Unit tests (10 behavioural dimensions) ────────────────────────────────────
//
// These tests use the `#[cfg(test)]` mock variant of `JsEmbedderBridge` defined
// above. They run inside the library's own test binary (via `cargo test --lib`),
// NOT as a separate integration test binary. This is the correct pattern for
// napi-rs crates — integration test binaries cannot link `_napi_*` symbols
// because the napi ABI is only available when loaded by a Node.js runtime.
//
// What is covered here:
//   D1: happy-path embed_dyn dispatch
//   D2: error propagation from callback
//   D3: dimension mismatch → descriptive error, no panic
//   D4: correct dimension → no error
//   D5: call counting confirms impl is invoked
//   D6: lifecycle (construct → Arc → drop) without panic
//   D7: concurrent calls on same Arc (Send + Sync)
//   D8: last_usage_tokens_dyn is always None (JS has no token counter API)
//   D9: multiple sequential calls return consistent results
//   D10: into_arc produces correct Arc<dyn DynEmbeddingProvider> trait object
//
// What requires JS smoke tests (`__test__/smoke-embedder.test.mjs`):
//   - Live ThreadsafeFunction dispatch (real JS callback → real embedding vec)
//   - embeddingDim mismatch propagation at the JS layer
//   - Backward-compat Memory.open(path) without withEmbedder

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    #![allow(clippy::unwrap_used)]

    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use kremory::DynEmbeddingProvider;

    use super::{into_arc, JsEmbedderBridge};

    // ── D1: Happy-path dispatch ───────────────────────────────────────────────

    #[tokio::test]
    async fn bridge_dispatches_text_and_receives_vec() {
        let expected_dim = 256usize;
        let fixed: Vec<f32> = (0..expected_dim)
            .map(|i| i as f32 / expected_dim as f32)
            .collect();

        let arc = into_arc(JsEmbedderBridge::new_mock(fixed.clone()));

        let result = arc.embed_dyn("hello world").await;
        let vec = result.expect("embed_dyn must succeed on happy path");

        assert_eq!(
            vec.len(),
            expected_dim,
            "returned vec length must equal expected_dim"
        );
        for (i, (&got, &exp)) in vec.iter().zip(fixed.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element[{i}]: got {got}, expected {exp}, diff {}",
                (got - exp).abs()
            );
        }
    }

    // ── D2: Error propagation ─────────────────────────────────────────────────

    #[tokio::test]
    async fn bridge_propagates_callback_error() {
        let arc = into_arc(JsEmbedderBridge::new_error_mock(
            "embed failed: model unavailable",
        ));

        let result = arc.embed_dyn("any text").await;
        assert!(
            result.is_err(),
            "bridge must propagate callback error as Err"
        );

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("embed") || msg.contains("model unavailable") || msg.contains("embedder"),
            "error message must reference the failure reason: got '{msg}'"
        );
    }

    // ── D3: Dim validation — mismatch ─────────────────────────────────────────

    #[tokio::test]
    async fn bridge_validates_embedding_dim_mismatch() {
        let expected_dim = 256usize;
        let wrong_dim = 128usize;
        let wrong_vec: Vec<f32> = vec![0.5f32; wrong_dim];

        // Bridge configured to expect 256-dim but mock returns 128-dim vec.
        let arc = into_arc(JsEmbedderBridge::new_mock_with_dim_check(
            wrong_vec,
            expected_dim,
        ));

        let result = arc.embed_dyn("mismatch text").await;
        assert!(result.is_err(), "dim mismatch must produce Err, not panic");

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("dimension") || msg.contains("dim") || msg.contains("length"),
            "error must describe the dimension mismatch: got '{msg}'"
        );
    }

    // ── D4: Dim validation — correct dim ──────────────────────────────────────

    #[tokio::test]
    async fn bridge_accepts_correct_dim() {
        let dim = 384usize;
        let vec: Vec<f32> = vec![0.1f32; dim];
        let arc = into_arc(JsEmbedderBridge::new_mock_with_dim_check(vec, dim));

        let result = arc.embed_dyn("matching dim text").await;
        assert!(
            result.is_ok(),
            "correct-dim vec must not trigger a dim error"
        );
        assert_eq!(result.unwrap().len(), dim);
    }

    // ── D5: Call counting ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn bridge_call_is_counted() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let dim = 384usize;
        let fixed: Vec<f32> = vec![0.1f32; dim];
        let arc = into_arc(JsEmbedderBridge::new_counting_mock(
            fixed,
            dim,
            Arc::clone(&call_count),
        ));

        let _result = arc.embed_dyn("test text for counting").await;

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "bridge must have been called exactly once"
        );
    }

    // ── D6: Lifecycle — no panic on drop ──────────────────────────────────────

    #[test]
    fn bridge_drop_does_not_panic() {
        let arc: Arc<dyn DynEmbeddingProvider> =
            into_arc(JsEmbedderBridge::new_mock(vec![0.0f32; 16]));
        drop(arc);
        // Reaching here without panic is the assertion.
    }

    // ── D7: Concurrent calls (Send + Sync) ────────────────────────────────────

    #[tokio::test]
    async fn bridge_concurrent_calls_both_succeed() {
        let dim = 64usize;
        let fixed: Vec<f32> = vec![0.42f32; dim];
        let bridge: Arc<dyn DynEmbeddingProvider> = Arc::new(JsEmbedderBridge::new_mock(fixed));
        let bridge2 = Arc::clone(&bridge);

        let h1 = tokio::spawn(async move { bridge.embed_dyn("first concurrent text").await });
        let h2 = tokio::spawn(async move { bridge2.embed_dyn("second concurrent text").await });

        let (r1, r2) = tokio::join!(h1, h2);
        let r1 = r1.expect("tokio task 1 must not panic");
        let r2 = r2.expect("tokio task 2 must not panic");

        assert!(r1.is_ok(), "concurrent call 1 must succeed: {:?}", r1.err());
        assert!(r2.is_ok(), "concurrent call 2 must succeed: {:?}", r2.err());
        assert_eq!(r1.unwrap().len(), dim, "call 1 must return correct dim");
        assert_eq!(r2.unwrap().len(), dim, "call 2 must return correct dim");
    }

    // ── D8: Token counter is None ─────────────────────────────────────────────

    #[test]
    fn bridge_last_usage_tokens_is_none() {
        let arc: Arc<dyn DynEmbeddingProvider> =
            into_arc(JsEmbedderBridge::new_mock(vec![0.0f32; 32]));
        assert!(
            arc.last_usage_tokens_dyn().is_none(),
            "JS bridge has no token counter; last_usage_tokens_dyn must return None"
        );
    }

    // ── D9: Multiple sequential calls return consistent results ───────────────

    #[tokio::test]
    async fn bridge_multiple_calls_return_consistent_results() {
        let dim = 128usize;
        let fixed: Vec<f32> = (0..dim).map(|i| i as f32 / 1000.0).collect();
        let arc: Arc<dyn DynEmbeddingProvider> =
            into_arc(JsEmbedderBridge::new_mock(fixed.clone()));

        let r1 = arc.embed_dyn("first call").await;
        let r2 = arc.embed_dyn("second call").await;

        let v1 = r1.expect("first embed_dyn call must succeed");
        let v2 = r2.expect("second embed_dyn call must succeed");

        assert_eq!(v1.len(), dim, "first call must return correct dim");
        assert_eq!(v2.len(), dim, "second call must return correct dim");
        assert_eq!(
            v1, v2,
            "both calls on same mock bridge must return identical vecs"
        );
    }

    // ── D10: into_arc produces correct trait object ───────────────────────────

    #[test]
    fn into_arc_produces_dyn_embedding_provider() {
        let arc: Arc<dyn DynEmbeddingProvider> =
            into_arc(JsEmbedderBridge::new_mock(vec![1.0f32; 64]));
        // Callable without downcasting — confirms correct trait object wiring.
        assert!(arc.last_usage_tokens_dyn().is_none());
    }
}

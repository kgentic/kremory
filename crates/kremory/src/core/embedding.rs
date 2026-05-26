//! Embedding provider wrappers — observability instrumentation layer.
//!
//! ## `TokenTrackingEmbedder` (ADR D10)
//!
//! Wraps any `EmbeddingProvider` and emits the canonical dual-emit pair after
//! every embed call:
//!
//! - `metrics::counter!("kremory_core_tokens_total", ...)` — Prometheus-scraped
//!   cumulative token count per provider/model/operation/direction
//! - `tracing::info!` span with OTel GenAI SemConv fields (`gen_ai.*`) per ADR D8
//!
//! This prevents the langchain-rust failure mode: `TokenUsage` captured in result
//! structs but never emitted. The wrapper enforces emission at the boundary.
//!
//! ## Snapshot recorder test
//!
//! `tests/b1_observability.rs::token_tracking_embedder_emits_tokens_total_counter`
//! verifies emission using `DebuggingRecorder + metrics::with_local_recorder` —
//! no global recorder required (library-safe per ADR D2).

use std::future::Future;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::provider::EmbeddingProvider;

/// Wraps any `EmbeddingProvider` with dual-emit observability per ADR D10.
///
/// Emits after every embed call:
/// - `kremory_core_tokens_total{provider, model, operation="embed", direction="input"}`
/// - `kremory_core_request_duration_seconds{provider, model, operation="embed"}`
/// - `tracing::info!` with OTel GenAI SemConv fields
///
/// The `provider` and `model` labels are bounded strings set at construction time —
/// cardinality discipline per ADR D7 (no free strings, no UUIDs as labels).
pub struct TokenTrackingEmbedder<E: EmbeddingProvider> {
    inner: E,
    /// `provider` label — bounded enum-like string (e.g. "openai", "voyage", "local").
    /// Set at construction; not computed per-call (cardinality safe per ADR D7).
    provider: &'static str,
    /// `model` label — bounded string from `monitoring/provider-rates.toml` allow-list.
    model: &'static str,
}

impl<E: EmbeddingProvider> TokenTrackingEmbedder<E> {
    /// Wrap `inner` with token tracking.
    ///
    /// `provider`: bounded label string (e.g. `"openai"`, `"voyage"`, `"local"`).
    /// `model`: bounded label string from the provider-rates allow-list.
    pub fn new(inner: E, provider: &'static str, model: &'static str) -> Self {
        Self {
            inner,
            provider,
            model,
        }
    }
}

impl<E: EmbeddingProvider> EmbeddingProvider for TokenTrackingEmbedder<E> {
    // RPIT return required to match EmbeddingProvider trait signature (which uses impl Future).
    // The lint fires because the body is a single async block, but the trait shape forces RPIT here.
    #[allow(clippy::manual_async_fn)]
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        async move {
            let start = Instant::now();

            // Pre-call token approximation: split by whitespace as a rough proxy.
            // When the backend reports usage via `last_usage_tokens()`, that value
            // takes precedence after the call.
            let pre_token_approx = text.split_whitespace().count() as u64;

            let result = self.inner.embed(text).await?;

            let elapsed_secs = start.elapsed().as_secs_f64();

            // Post-call: prefer server-reported token count; fall back to approximation.
            let tokens_input = self.inner.last_usage_tokens().unwrap_or(pre_token_approx);

            // Dual-emit: metrics counter + tracing span (ADR D1 + D8)
            metrics::counter!(
                "kremory_core_tokens_total",
                "provider" => self.provider,
                "model"    => self.model,
                "operation" => "embed",
                "direction" => "input",
            )
            .increment(tokens_input);

            metrics::histogram!(
                "kremory_core_request_duration_seconds",
                "provider"  => self.provider,
                "model"     => self.model,
                "operation" => "embed",
            )
            .record(elapsed_secs);

            tracing::info!(
                // OTel GenAI SemConv fields per ADR D8
                "gen_ai.system" = self.provider,
                "gen_ai.request.model" = self.model,
                "gen_ai.usage.input_tokens" = tokens_input,
                operation = "embed",
                elapsed_secs = elapsed_secs,
                "kremory.embed completed"
            );

            Ok(result)
        }
    }

    /// Delegate `last_usage_tokens` to the inner provider so nested wrappers
    /// (e.g. `TokenTrackingEmbedder<TokenTrackingEmbedder<E>>`) compose correctly.
    fn last_usage_tokens(&self) -> Option<u64> {
        self.inner.last_usage_tokens()
    }
}

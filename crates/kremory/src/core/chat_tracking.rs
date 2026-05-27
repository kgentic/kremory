//! Chat observability wrapper — `TokenTrackingChatProvider<L>` (ADR D10 §Gap A).
//!
//! ## What it does
//!
//! Wraps any [`ChatProvider`] implementation and emits after every call:
//!
//! - `kremory_core_tokens_total{operation="chat", provider, model, direction="input"|"output"}`
//!   — cumulative token count (increments by 0 when provider returns no usage).
//! - `kremory_core_cost_usd_total{operation="chat", provider, model}` — cumulative
//!   micro-USD cost derived from `PROVIDER_RATES` (skip when rate table not
//!   initialized or model is unlisted).
//! - `kremory_core_chat_duration_seconds{provider, model, outcome, [error_type]}`
//!   — histogram per call; `outcome` is bounded to `"success"` or `"error"`.
//!
//! ## Cardinality discipline (ADR D7)
//!
//! `provider` and `model` are set at construction time — never derived from
//! response content. `error_type` is bounded to the 11 known `LLMError`
//! variants in `autoagents-llm 0.3.7` via `error_type_label`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kremory::core::chat_tracking::TokenTrackingChatProvider;
//! let tracked = TokenTrackingChatProvider::new(ollama_client, "ollama", "llama3");
//! ```

use std::sync::OnceLock;
use std::time::Instant;

use async_trait::async_trait;
use dashmap::DashSet;

use crate::core::provider::{
    ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool,
};
use crate::core::rates::PROVIDER_RATES;
use autoagents_llm::error::LLMError;

/// Wraps any `ChatProvider` with dual-emit observability per ADR D10.
///
/// Generic parameter `L` must be `Send + Sync` so the wrapper itself is
/// `Send + Sync` and satisfies the `#[async_trait]` constraint.
pub struct TokenTrackingChatProvider<L: ChatProvider + Send + Sync> {
    inner: L,
    /// Bounded label: set at construction, never computed per-call.
    provider: String,
    /// Bounded label: set at construction, never computed per-call.
    model: String,
}

impl<L: ChatProvider + Send + Sync> TokenTrackingChatProvider<L> {
    /// Wrap `inner` with token + cost + duration tracking.
    ///
    /// `provider` and `model` are used verbatim as metric labels — they must
    /// match entries in `monitoring/provider-rates.toml` for cost emission to
    /// produce non-zero values.
    pub fn new(inner: L, provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner,
            provider: provider.into(),
            model: model.into(),
        }
    }
}

// ── one-shot warning registries ────────────────────────────────────────────

/// Emits a usage-absent warning at most once per `(provider, model)` pair.
/// `DashSet::insert` returns `true` when the element was newly inserted (i.e.
/// first time we see this pair), so we warn on first occurrence.
static USAGE_NONE_WARNED: OnceLock<DashSet<(String, String)>> = OnceLock::new();

/// Emits an unknown-rate warning at most once per `(provider, model, direction)`.
static UNKNOWN_RATE_WARNED: OnceLock<DashSet<(String, String, String)>> = OnceLock::new();

/// Returns `true` on the first call for this `(provider, model)` pair.
fn first_usage_warning(provider: &str, model: &str) -> bool {
    let set = USAGE_NONE_WARNED.get_or_init(DashSet::new);
    set.insert((provider.to_string(), model.to_string()))
}

/// Returns `true` on the first call for this `(provider, model, direction)`.
fn first_rate_warning(provider: &str, model: &str, direction: &str) -> bool {
    let set = UNKNOWN_RATE_WARNED.get_or_init(DashSet::new);
    set.insert((
        provider.to_string(),
        model.to_string(),
        direction.to_string(),
    ))
}

// ── bounded error type label (Gap D) ───────────────────────────────────────

/// Map an [`LLMError`] variant to a short, cardinality-safe `error_type` label
/// string.
///
/// **Verified against `autoagents-llm 0.3.7` `src/error.rs`** (path:
/// `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/autoagents-llm-0.3.7/src/error.rs`).
/// All 11 variants are enumerated; the default arm is unreachable in practice
/// unless a future version of the dependency adds a new variant.
fn error_type_label(err: &LLMError) -> &'static str {
    use LLMError::{
        AuthError, Generic, GuardrailBlocked, GuardrailExecutionFailed, HttpError, InvalidRequest,
        JsonError, NoToolSupport, ProviderError, ResponseFormatError, ToolConfigError,
    };
    match err {
        HttpError(_) => "http_error",
        AuthError(_) => "auth_error",
        InvalidRequest(_) => "invalid_request",
        ProviderError(_) => "provider_error",
        ResponseFormatError { .. } => "response_format_error",
        Generic(_) => "generic_error",
        JsonError(_) => "parse_error",
        ToolConfigError(_) => "tool_config_error",
        NoToolSupport(_) => "no_tool_support",
        GuardrailBlocked { .. } => "guardrail_blocked",
        GuardrailExecutionFailed { .. } => "guardrail_failed",
    }
}

// ── ChatProvider impl ───────────────────────────────────────────────────────

#[async_trait]
impl<L: ChatProvider + Send + Sync> ChatProvider for TokenTrackingChatProvider<L> {
    /// Delegates to `inner.chat_with_tools` and emits metrics after the call.
    ///
    /// Token counters are emitted with `direction="input"` and `direction="output"`.
    /// Cost counters are emitted in micro-USD (×1_000_000) for integer precision.
    /// Duration histogram is emitted with `outcome="success"` or `outcome="error"`.
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let start = Instant::now();
        let result = self
            .inner
            .chat_with_tools(messages, tools, json_schema)
            .await;
        let duration_seconds = start.elapsed().as_secs_f64();

        match &result {
            Ok(response) => {
                let (input_tokens, output_tokens) = match response.usage() {
                    Some(u) => (u64::from(u.prompt_tokens), u64::from(u.completion_tokens)),
                    None => {
                        if first_usage_warning(&self.provider, &self.model) {
                            tracing::warn!(
                                provider = %self.provider,
                                model    = %self.model,
                                "chat usage not returned — token counters will be 0 for this call"
                            );
                        }
                        (0, 0)
                    }
                };

                // Token counters — dual-emit: input + output directions.
                metrics::counter!(
                    "kremory_core_tokens_total",
                    "operation" => "chat",
                    "provider"  => self.provider.clone(),
                    "model"     => self.model.clone(),
                    "direction" => "input",
                )
                .increment(input_tokens);

                metrics::counter!(
                    "kremory_core_tokens_total",
                    "operation" => "chat",
                    "provider"  => self.provider.clone(),
                    "model"     => self.model.clone(),
                    "direction" => "output",
                )
                .increment(output_tokens);

                // Cost emission — only when PROVIDER_RATES is initialized.
                if let Some(rates) = PROVIDER_RATES.get() {
                    let input_rate = rates.lookup_rate(&self.provider, &self.model, Some("input"));
                    let output_rate =
                        rates.lookup_rate(&self.provider, &self.model, Some("output"));

                    if input_rate.is_none()
                        && first_rate_warning(&self.provider, &self.model, "input")
                    {
                        tracing::warn!(
                            provider  = %self.provider,
                            model     = %self.model,
                            direction = "input",
                            "no chat rate found in provider-rates.toml — cost counter will skip this call"
                        );
                    }

                    let input_cost = input_rate.unwrap_or(0.0) * (input_tokens as f64) / 1000.0;
                    let output_cost = output_rate.unwrap_or(0.0) * (output_tokens as f64) / 1000.0;
                    let total_cost = input_cost + output_cost;

                    // Emit cost only when non-zero to avoid polluting zero-rate records.
                    if total_cost > 0.0 {
                        // Store as micro-USD integer for counter precision.
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let micro_usd = (total_cost * 1_000_000.0) as u64;
                        metrics::counter!(
                            "kremory_core_cost_usd_total",
                            "operation" => "chat",
                            "provider"  => self.provider.clone(),
                            "model"     => self.model.clone(),
                        )
                        .increment(micro_usd);
                    }
                }

                // Duration histogram — success path.
                metrics::histogram!(
                    "kremory_core_chat_duration_seconds",
                    "provider" => self.provider.clone(),
                    "model"    => self.model.clone(),
                    "outcome"  => "success",
                )
                .record(duration_seconds);
            }

            Err(err) => {
                let label = error_type_label(err);

                // Duration histogram — error path (includes error_type label).
                metrics::histogram!(
                    "kremory_core_chat_duration_seconds",
                    "provider"   => self.provider.clone(),
                    "model"      => self.model.clone(),
                    "outcome"    => "error",
                    "error_type" => label,
                )
                .record(duration_seconds);
            }
        }

        result
    }
}

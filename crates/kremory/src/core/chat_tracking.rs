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

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::core::provider::{ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType};
    use autoagents_llm::chat::Usage;
    use autoagents_llm::error::LLMError;
    use metrics_util::debugging::DebuggingRecorder;

    // ── Local mock for chat tracking tests ──────────────────────────────────

    /// Behavior enum for `TrackingMockProvider`.
    enum MockBehavior {
        WithUsage { input: u32, output: u32 },
        AlwaysFail { kind: u8 }, // u8 index into error_type_label test cases
    }

    /// Minimal mock ChatProvider used only inside this test module.
    struct TrackingMockProvider {
        behavior: MockBehavior,
    }

    struct TrackingMockResponse {
        input_tokens: u32,
        output_tokens: u32,
    }

    impl std::fmt::Debug for TrackingMockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "TrackingMockResponse {{ input: {}, output: {} }}",
                self.input_tokens, self.output_tokens
            )
        }
    }

    impl std::fmt::Display for TrackingMockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock response")
        }
    }

    impl ChatResponse for TrackingMockResponse {
        fn text(&self) -> Option<String> {
            Some("mock".to_string())
        }

        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }

        fn usage(&self) -> Option<Usage> {
            Some(Usage {
                prompt_tokens: self.input_tokens,
                completion_tokens: self.output_tokens,
                total_tokens: self.input_tokens + self.output_tokens,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            })
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for TrackingMockProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            match &self.behavior {
                MockBehavior::WithUsage { input, output } => Ok(Box::new(TrackingMockResponse {
                    input_tokens: *input,
                    output_tokens: *output,
                })),
                MockBehavior::AlwaysFail { kind } => Err(make_err(*kind)),
            }
        }
    }

    /// Construct a test `LLMError` by index for the 11 variants.
    fn make_err(kind: u8) -> LLMError {
        match kind {
            0 => LLMError::HttpError("http".into()),
            1 => LLMError::AuthError("auth".into()),
            2 => LLMError::InvalidRequest("inv".into()),
            3 => LLMError::ProviderError("prov".into()),
            4 => LLMError::ResponseFormatError {
                message: "fmt".into(),
                raw_response: "raw".into(),
            },
            5 => LLMError::Generic("gen".into()),
            6 => LLMError::JsonError("json".into()),
            7 => LLMError::ToolConfigError("tool".into()),
            8 => LLMError::NoToolSupport("nts".into()),
            9 => LLMError::GuardrailBlocked {
                phase: autoagents_llm::error::GuardrailPhase::Input,
                guard: "g".into(),
                rule_id: "r".into(),
                category: "c".into(),
                severity: "s".into(),
                message: "m".into(),
            },
            _ => LLMError::GuardrailExecutionFailed {
                guard: "g".into(),
                message: "m".into(),
            },
        }
    }

    /// Expected label for each error index (mirrors `error_type_label`).
    fn expected_label(kind: u8) -> &'static str {
        match kind {
            0 => "http_error",
            1 => "auth_error",
            2 => "invalid_request",
            3 => "provider_error",
            4 => "response_format_error",
            5 => "generic_error",
            6 => "parse_error",
            7 => "tool_config_error",
            8 => "no_tool_support",
            9 => "guardrail_blocked",
            _ => "guardrail_failed",
        }
    }

    // ── G_v012_1: token_tracking_chat_emits_metrics ──────────────────────────

    /// G_v012_1 — Wrap mock with WithUsage(100, 50); assert tokens_total(input)=100,
    /// tokens_total(output)=50, and duration histogram is recorded.
    #[test]
    fn token_tracking_chat_emits_metrics() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let inner = TrackingMockProvider {
            behavior: MockBehavior::WithUsage {
                input: 100,
                output: 50,
            },
        };
        let tracked = TokenTrackingChatProvider::new(inner, "test-provider", "test-model");

        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
            message_type: MessageType::Text,
        }];

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds");
            rt.block_on(async {
                let _resp = tracked
                    .chat_with_tools(&msgs, None, None)
                    .await
                    .expect("chat must succeed");
            });
        });

        let snapshot = snapshotter.snapshot().into_vec();

        // Collect counter and histogram names
        let counter_names: Vec<String> = snapshot
            .iter()
            .map(|(k, _, _, _)| k.key().name().to_string())
            .collect();

        assert!(
            counter_names
                .iter()
                .any(|n| n == "kremory_core_tokens_total"),
            "kremory_core_tokens_total must be emitted; got: {counter_names:?}"
        );
        assert!(
            counter_names
                .iter()
                .any(|n| n == "kremory_core_chat_duration_seconds"),
            "kremory_core_chat_duration_seconds must be emitted; got: {counter_names:?}"
        );

        // Verify token counts — find input and output counters and check their values.
        for (key, _unit, _desc, value) in &snapshot {
            if key.key().name() == "kremory_core_tokens_total" {
                let labels: std::collections::HashMap<&str, &str> =
                    key.key().labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("direction") == Some(&"input") {
                    if let metrics_util::debugging::DebugValue::Counter(n) = value {
                        assert_eq!(*n, 100, "input token count must be 100; got {n}");
                    }
                }
                if labels.get("direction") == Some(&"output") {
                    if let metrics_util::debugging::DebugValue::Counter(n) = value {
                        assert_eq!(*n, 50, "output token count must be 50; got {n}");
                    }
                }
            }
        }
    }

    // ── G_v012_2: chat_error_emits_bounded_error_type ────────────────────────

    /// G_v012_2 — Wrap mock(AlwaysFail). Assert duration histogram is emitted
    /// with outcome="error". Loop all 11 LLMError variants and verify each maps
    /// to the expected bounded label per §17.1 table.
    #[test]
    fn chat_error_emits_bounded_error_type() {
        // Validate the error_type_label fn covers all 11 variants.
        let cases: &[(u8, &'static str)] = &[
            (0, "http_error"),
            (1, "auth_error"),
            (2, "invalid_request"),
            (3, "provider_error"),
            (4, "response_format_error"),
            (5, "generic_error"),
            (6, "parse_error"),
            (7, "tool_config_error"),
            (8, "no_tool_support"),
            (9, "guardrail_blocked"),
            (10, "guardrail_failed"),
        ];

        for &(kind_index, expected) in cases {
            let err = make_err(kind_index);
            let label = error_type_label(&err);
            assert_eq!(
                label,
                expected_label(kind_index),
                "error_type_label({kind_index}) must map to '{}'",
                expected
            );
        }

        // Also verify that the histogram is emitted on error path.
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let inner = TrackingMockProvider {
            behavior: MockBehavior::AlwaysFail { kind: 0 }, // HttpError
        };
        let tracked = TokenTrackingChatProvider::new(inner, "test-provider", "test-model");

        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "fail".into(),
            message_type: MessageType::Text,
        }];

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds");
            rt.block_on(async {
                let result = tracked.chat_with_tools(&msgs, None, None).await;
                assert!(result.is_err(), "mock must return error");
            });
        });

        let snapshot = snapshotter.snapshot().into_vec();
        let histogram_names: Vec<String> = snapshot
            .iter()
            .map(|(k, _, _, _)| k.key().name().to_string())
            .collect();

        assert!(
            histogram_names
                .iter()
                .any(|n| n == "kremory_core_chat_duration_seconds"),
            "kremory_core_chat_duration_seconds histogram must be emitted on error path; got: {histogram_names:?}"
        );

        // Verify outcome="error" label is present on the duration histogram.
        let has_error_outcome = snapshot.iter().any(|(key, _, _, _)| {
            key.key().name() == "kremory_core_chat_duration_seconds"
                && key
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "error")
        });
        assert!(
            has_error_outcome,
            "duration histogram must carry outcome=error label on failure path"
        );
    }

    // ── G_v012_3: chat_unknown_provider_emits_zero_cost_with_warn ────────────

    /// G_v012_3 — Wrap mock with usage; provider="unregistered" model="unknown".
    /// Assert cost_total is NOT incremented (PROVIDER_RATES not initialized for this key).
    #[test]
    fn chat_unknown_provider_emits_zero_cost_with_warn() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        // PROVIDER_RATES may or may not be initialized from another test.
        // The key "unregistered"/"unknown" is guaranteed absent from the table.
        let inner = TrackingMockProvider {
            behavior: MockBehavior::WithUsage {
                input: 500,
                output: 500,
            },
        };
        let tracked = TokenTrackingChatProvider::new(inner, "unregistered", "unknown");

        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "test".into(),
            message_type: MessageType::Text,
        }];

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds");
            rt.block_on(async {
                let _resp = tracked
                    .chat_with_tools(&msgs, None, None)
                    .await
                    .expect("chat must succeed");
            });
        });

        let snapshot = snapshotter.snapshot().into_vec();

        // cost_total must NOT appear when the provider is unknown — no rate entry.
        let cost_emitted = snapshot.iter().any(|(key, _, _, _)| {
            key.key().name() == "kremory_core_cost_usd_total"
                && key
                    .key()
                    .labels()
                    .any(|l| l.key() == "provider" && l.value() == "unregistered")
        });
        assert!(
            !cost_emitted,
            "kremory_core_cost_usd_total must NOT be emitted for an unlisted provider/model"
        );
    }

    // ── G_v012_6: token_tracking_chat_cost_correct ───────────────────────────

    /// G_v012_6 — Wrap mock(200, 300) with provider="openai" model="gpt-4o-mini".
    /// Initialize PROVIDER_RATES from bundled. Assert cost increment equals
    /// (200 × 0.00015 + 300 × 0.0006) / 1000 × 1_000_000 micro-USD (±1 tolerance).
    #[test]
    fn token_tracking_chat_cost_correct() {
        // Initialize bundled rates (idempotent).
        crate::core::rates::init_bundled().expect("rates must load");

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let inner = TrackingMockProvider {
            behavior: MockBehavior::WithUsage {
                input: 200,
                output: 300,
            },
        };
        let tracked = TokenTrackingChatProvider::new(inner, "openai", "gpt-4o-mini");

        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "cost-test".into(),
            message_type: MessageType::Text,
        }];

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds");
            rt.block_on(async {
                let _resp = tracked
                    .chat_with_tools(&msgs, None, None)
                    .await
                    .expect("chat must succeed");
            });
        });

        // Expected cost: (200 * 0.00015 + 300 * 0.0006) / 1000 * 1_000_000 micro-USD
        // = (0.03 + 0.18) / 1000 * 1_000_000 = 0.21 * 1000 = 210 micro-USD
        let expected_micro_usd: u64 =
            ((200_f64 * 0.00015 + 300_f64 * 0.0006) / 1000.0 * 1_000_000.0) as u64;

        let snapshot = snapshotter.snapshot().into_vec();

        let cost_counter = snapshot.iter().find(|(key, _, _, _)| {
            key.key().name() == "kremory_core_cost_usd_total"
                && key
                    .key()
                    .labels()
                    .any(|l| l.key() == "provider" && l.value() == "openai")
        });

        let actual = match cost_counter {
            Some((_, _, _, metrics_util::debugging::DebugValue::Counter(n))) => *n,
            Some(_) => panic!("kremory_core_cost_usd_total is not a counter"),
            None => panic!(
                "kremory_core_cost_usd_total was not emitted for openai/gpt-4o-mini; expected {expected_micro_usd} micro-USD"
            ),
        };

        let delta = if actual >= expected_micro_usd {
            actual - expected_micro_usd
        } else {
            expected_micro_usd - actual
        };
        assert!(
            delta <= 1,
            "cost counter {actual} micro-USD differs from expected {expected_micro_usd} by more than 1"
        );
    }
}

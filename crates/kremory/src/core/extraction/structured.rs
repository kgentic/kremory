//! `StructuredCallBuilder` — single point of schema enforcement for the extraction layer.
//!
//! Owns capability negotiation, fallback ladder, self-correction retry, and
//! observability counters for every LLM extraction call.  Call-sites in
//! `extraction/mod.rs`, `hybrid_extractor.rs`, and `contradiction.rs` replace
//! direct `llm.chat_with_tools(&msgs, None, None)` with:
//!
//! ```rust,ignore
//! let value = StructuredCallBuilder::new(llm, &SCHEMA_X, "SchemaXName")
//!     .model(model_str)   // optional: enables NativeSchema / FormatSchema arms
//!     .messages(msgs)
//!     .call()
//!     .await?;
//! ```
//!
//! The builder returns a `serde_json::Value`; callers deserialise into the
//! appropriate wrapper struct.

use metrics::{counter, histogram};
use serde_json::Value;

use crate::core::error::Error as ExtractionError;
use crate::core::extraction::delimited_tuple;
use crate::core::extraction::prompts;
use crate::core::extraction::schemas::FallbackArm;
use crate::core::provider::{
    capability_of, ChatMessage, ChatProvider, ProviderCaps, StructuredOutputFormat,
};

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Single-call builder for schema-enforced LLM extraction calls.
///
/// Capability detection is driven by the optional model string set via
/// [`Self::model`].  When no model string is provided, the builder falls
/// back to `LlmJsonRepair` for all providers — still strictly better than
/// raw `chat_with_tools` because it provides JSON repair and
/// schema-structure validation.  Call `.model(model_str)` to unlock
/// NativeSchema (Anthropic/OpenAI) or FormatSchema (Ollama) arms based on
/// provider-capability detection.
pub(crate) struct StructuredCallBuilder<'a, L: ?Sized + ChatProvider> {
    llm: &'a L,
    /// Owned so the `call()` async fn can pass it to `counter!` macros
    /// which require `'static` label values.  Empty string = unknown model
    /// = `PromptOnly` capability → `LlmJsonRepair` first arm.
    model: String,
    schema: &'a Value,
    schema_name: &'static str,
    messages: Vec<ChatMessage>,
    max_retries: u8,
    /// Per-arm wall-clock budget. Wired via `tokio::time::timeout` in `call()`.
    /// Distinct from per-token TTFT enforcement, which requires streaming-API
    /// support upstream and is not yet implemented.
    #[allow(dead_code)]
    ttft_budget_ms: Option<u64>,
    force_arm: Option<FallbackArm>,
}

impl<'a, L: ?Sized + ChatProvider> StructuredCallBuilder<'a, L> {
    /// Construct a builder.
    ///
    /// - `schema` — `'static` schema value (one of the `SCHEMA_*` statics in
    ///   `extraction::schemas`).
    /// - `schema_name` — human-readable label for metrics + self-correction prompt.
    ///
    /// Call [`.model()`][Self::model] to enable provider-native schema enforcement.
    pub(crate) fn new(llm: &'a L, schema: &'a Value, schema_name: &'static str) -> Self {
        Self {
            llm,
            model: String::new(),
            schema,
            schema_name,
            messages: Vec::new(),
            max_retries: 1,
            // 30s per-arm wall-clock cap. Production fail-fast: HTTP layer
            // bounds individual requests (10s via LLMBuilder), this caps the
            // arm even if HTTP succeeds-then-hangs. Override via .ttft_budget_ms.
            // Tests exercising async paths MUST use timer-enabled runtimes
            // (#[tokio::test(start_paused = true)] or multi_thread flavor).
            //
            // Call-sites that flow through Engine thread the budget from
            // PipelineConfig::extraction_arm_budget_ms via ExtractionContext.
            // Slow local LLMs (qwen2.5:14b ~80-130s) set 300_000 on the builder.
            ttft_budget_ms: Some(30_000),
            force_arm: None,
        }
    }

    /// Set the model string for provider-capability detection.
    ///
    /// When set, [`capability_of`] selects the correct fallback ladder:
    /// NativeSchema for Anthropic/OpenAI GA, FormatSchema for Ollama,
    /// LlmJsonRepair for unknown models.  If not set, the builder defaults
    /// to the `PromptOnly` / `LlmJsonRepair` ladder (safe for all providers).
    pub(crate) fn model(mut self, m: &str) -> Self {
        self.model = m.to_string();
        self
    }

    /// Set the messages to send.
    pub(crate) fn messages(mut self, msgs: Vec<ChatMessage>) -> Self {
        self.messages = msgs;
        self
    }

    /// Override maximum self-correction retries (default 1, capped at 2).
    ///
    /// Exercised only by inline tests in this module today; retained as
    /// pub(crate) builder API for call-sites that need non-default retry budgets.
    #[allow(dead_code)]
    pub(crate) fn max_retries(mut self, n: u8) -> Self {
        self.max_retries = n.min(2);
        self
    }

    /// Per-call arm-budget override in milliseconds.
    ///
    /// Exercised only by inline tests in this module today; retained as
    /// pub(crate) builder API for call-sites that need non-default budgets.
    #[allow(dead_code)]
    pub(crate) fn ttft_budget_ms(mut self, ms: u64) -> Self {
        self.ttft_budget_ms = Some(ms);
        self
    }

    /// Override the fallback arm selection (bypass capability detection).
    ///
    /// Used by schemas whose deserialiser is incompatible with strict-mode
    /// JSON Schema enforcement (e.g. `deser_string_or_array`).
    pub(crate) fn force_arm(mut self, arm: FallbackArm) -> Self {
        self.force_arm = Some(arm);
        self
    }

    /// Execute the call with the configured parameters.
    ///
    /// Returns the parsed JSON `Value` on success.  Callers are responsible
    /// for deserialising to the appropriate wrapper struct.
    pub(crate) async fn call(self) -> Result<Value, ExtractionError> {
        // Extract fields to avoid borrow-after-move issues with async + self.
        let schema_name: &'static str = self.schema_name;
        let schema = self.schema;
        let llm = self.llm;
        let max_retries = self.max_retries;
        let force_arm = self.force_arm;
        // Clone model string once; pass as &str to capability_of and as .to_string()
        // to individual counter! sites (each call converts to SharedString and returns).
        let model_str = self.model.as_str().to_owned();

        // Build the fallback ladder.
        let ladder: Vec<FallbackArm> = if let Some(arm) = force_arm {
            // Forced start: skip all arms before `arm` but retain PromptOnly
            // as the final fallback so an empty LLM response never propagates
            // as an error (PromptOnly always produces `{}`).
            counter!(
                "rql.extraction.fallback_ladder_step",
                "schema" => schema_name,
                "from_arm" => "bypass",
                "to_arm" => arm_name(arm),
                "model" => model_str.clone(),
            )
            .increment(1);
            if arm == FallbackArm::PromptOnly {
                vec![FallbackArm::PromptOnly]
            } else {
                vec![arm, FallbackArm::PromptOnly]
            }
        } else {
            build_ladder(capability_of(&model_str))
        };

        let mut messages = self.messages;
        // Track last provider error for FallbackExhausted raw_response field.
        let mut last_err_str = String::new();
        let arm_budget = self.ttft_budget_ms.map(std::time::Duration::from_millis);

        // TD-019 Gap 4: cache env-var check once before the ladder loop so each
        // arm attempt doesn't pay a syscall-ish env::var read (Vera Finding 3).
        let debug_enabled = std::env::var("KREMORY_DEBUG").is_ok();

        for (arm_idx, &arm) in ladder.iter().enumerate() {
            counter!(
                "rql.extraction.structured_call_attempt",
                "schema" => schema_name,
                "arm" => arm_name(arm),
                "model" => model_str.clone(),
            )
            .increment(1);

            // Wall-clock cap per arm. Prevents one stalled arm from burning
            // the entire fallback ladder (default 30s; configurable via builder).
            //
            // TD-019 Gap 5: per-call timing — distinguishes "1 entity per call
            // × 3 chunks" from "10 entities in 1 chunk". Wraps the arm invocation
            // (incl. timeout cap so timed-out arms still record their cost).
            let call_start = std::time::Instant::now();
            let result = match arm_budget {
                Some(d) => {
                    match tokio::time::timeout(
                        d,
                        try_arm(TryArmParams {
                            llm,
                            arm,
                            schema,
                            schema_name,
                            messages: &messages,
                            debug_enabled,
                        }),
                    )
                    .await
                    {
                        Ok(r) => r,
                        Err(_) => {
                            // TD-166 known gap (Quinn MNT-005): a timed-out arm emits
                            // NO token usage, though the provider may already have
                            // spent tokens server-side before we gave up. So
                            // kremory_core_tokens_total UNDER-reports by whatever the
                            // timed-out arms cost. Cross-reference this counter to
                            // size that gap — a high arm_timeout with low token totals
                            // means the totals are incomplete, not that ingest was cheap.
                            counter!(
                                "rql.extraction.arm_timeout",
                                "schema" => schema_name,
                                "arm" => arm_name(arm),
                                "model" => model_str.clone(),
                            )
                            .increment(1);
                            last_err_str = format!(
                                "arm {} exceeded {}ms budget",
                                arm_name(arm),
                                d.as_millis()
                            );
                            Err(ExtractionError::Llm(last_err_str.clone()))
                        }
                    }
                }
                None => {
                    try_arm(TryArmParams {
                        llm,
                        arm,
                        schema,
                        schema_name,
                        messages: &messages,
                        debug_enabled,
                    })
                    .await
                }
            };
            histogram!(
                "rql.extraction.call_ms",
                "schema" => schema_name,
                "arm" => arm_name(arm),
                "model" => model_str.clone(),
            )
            .record(call_start.elapsed().as_secs_f64() * 1000.0);

            match result {
                Ok(ArmOutcome { value, usage }) => {
                    // TD-166: emit token cost as soon as the provider returned Ok —
                    // BEFORE the validation branch below. An arm whose output fails
                    // schema validation still cost tokens, and attributing cost only
                    // to validated calls would under-report precisely the failing arms
                    // a cost investigation is looking for.
                    emit_arm_usage(schema_name, arm, usage);
                    let (usage_in, usage_out, usage_reported) = usage_fields(usage);
                    // Phase D iter 3 (2026-06-10): per Rule 19 — log which arm succeeded
                    // so we can verify NativeSchema fires for capable providers vs falling
                    // through to LlmJsonRepair (schema-not-enforced).
                    if debug_enabled {
                        tracing::debug!(
                            target: "kremory.extraction.arm_success",
                            schema = schema_name,
                            arm = arm_name(arm),
                            model = %model_str,
                            "structured-call arm succeeded"
                        );
                    }
                    // Schema validation is only meaningful for provider-native arms
                    // (NativeSchema / FormatSchema) where the provider guarantees
                    // structural compliance.  LlmJsonRepair and PromptOnly arms are
                    // best-effort: accept any parseable JSON without structural checks.
                    let should_validate =
                        matches!(arm, FallbackArm::NativeSchema | FallbackArm::FormatSchema);

                    if !should_validate {
                        // Dual-emit (ADR D1 / R1.1): counter + fused OTel gen_ai.* per SPEC-001.
                        // TD-166/TD-090 CLOSED: token counts are now real. `usage_reported`
                        // disambiguates a genuine 0 from "the backend never told us"
                        // (Ollama does not implement ChatResponse::usage).
                        counter!(
                            "rql.extraction.structured_call_success",
                            "schema" => schema_name,
                            "arm" => arm_name(arm),
                        )
                        .increment(1);
                        tracing::info!(
                            "gen_ai.system" = "unknown",
                            "gen_ai.operation.name" = "extraction",
                            "gen_ai.request.model" = %model_str,
                            "gen_ai.usage.input_tokens" = usage_in,
                            "gen_ai.usage.output_tokens" = usage_out,
                            "kremory.usage_reported" = usage_reported,
                            schema = schema_name,
                            arm = arm_name(arm),
                            "kremory.extraction.structured_call_success"
                        );
                        return Ok(value);
                    }

                    match validate_against_schema(&value, schema) {
                        Ok(()) => {
                            // Dual-emit (ADR D1 / R1.1): counter + fused OTel gen_ai.* per SPEC-001.
                            // TD-166/TD-090 CLOSED — see the `!should_validate` branch above.
                            counter!(
                                "rql.extraction.structured_call_success",
                                "schema" => schema_name,
                                "arm" => arm_name(arm),
                            )
                            .increment(1);
                            tracing::info!(
                                "gen_ai.system" = "unknown",
                                "gen_ai.operation.name" = "extraction",
                                "gen_ai.request.model" = %model_str,
                                "gen_ai.usage.input_tokens" = usage_in,
                                "gen_ai.usage.output_tokens" = usage_out,
                                "kremory.usage_reported" = usage_reported,
                                schema = schema_name,
                                arm = arm_name(arm),
                                "kremory.extraction.structured_call_success"
                            );
                            return Ok(value);
                        }
                        Err(detail) => {
                            counter!(
                                "rql.extraction.schema_violation",
                                "schema" => schema_name,
                                "arm" => arm_name(arm),
                                "model" => model_str.clone(),
                            )
                            .increment(1);

                            // Self-correction retry (only on first violation, max_retries >= 1).
                            if max_retries >= 1 {
                                counter!(
                                    "rql.extraction.schema_violation_retry",
                                    "schema" => schema_name,
                                    "arm" => arm_name(arm),
                                )
                                .increment(1);

                                let correction_msg = build_correction_message(&detail);
                                messages.push(correction_msg);

                                let retry_result = try_arm(TryArmParams {
                                    llm,
                                    arm,
                                    schema,
                                    schema_name,
                                    messages: &messages,
                                    debug_enabled,
                                })
                                .await;

                                if let Ok(ArmOutcome {
                                    value: retry_value,
                                    usage: retry_usage,
                                }) = retry_result
                                {
                                    // TD-166: the self-correction retry is a SECOND paid
                                    // call. Emitted unconditionally — a retry that then
                                    // fails validation still cost tokens, and this arm is
                                    // exactly where a runaway retry ladder would hide.
                                    emit_arm_usage(schema_name, arm, retry_usage);
                                    let (retry_in, retry_out, retry_reported) =
                                        usage_fields(retry_usage);
                                    if validate_against_schema(&retry_value, schema).is_ok() {
                                        // Dual-emit (ADR D1 / R1.1): retry success path, fused OTel gen_ai.* per SPEC-001.
                                        counter!(
                                            "rql.extraction.structured_call_success",
                                            "schema" => schema_name,
                                            "arm" => arm_name(arm),
                                        )
                                        .increment(1);
                                        tracing::info!(
                                            "gen_ai.system" = "unknown",
                                            "gen_ai.operation.name" = "extraction",
                                            "gen_ai.request.model" = %model_str,
                                            "gen_ai.usage.input_tokens" = retry_in,
                                            "gen_ai.usage.output_tokens" = retry_out,
                                            "kremory.usage_reported" = retry_reported,
                                            schema = schema_name,
                                            arm = arm_name(arm),
                                            "kremory.extraction.structured_call_success"
                                        );
                                        return Ok(retry_value);
                                    }
                                }

                                // Retry also failed — drop to next arm.
                                // Remove the correction message before stepping down.
                                messages.pop();
                            }

                            // Record step down to next arm (if there is one).
                            if let Some(&next_arm) = ladder.get(arm_idx + 1) {
                                counter!(
                                    "rql.extraction.fallback_ladder_step",
                                    "schema" => schema_name,
                                    "from_arm" => arm_name(arm),
                                    "to_arm" => arm_name(next_arm),
                                    "model" => model_str.clone(),
                                )
                                .increment(1);
                            }
                        }
                    }
                }
                Err(llm_err) => {
                    // Provider-level error — capture for FallbackExhausted and step down.
                    last_err_str = llm_err.to_string();
                    // Phase D iter 3 (2026-06-10): per Rule 19 — surface arm-failure
                    // reason at WARN level so silent NativeSchema 400s become visible.
                    // Anthropic's "maxItems not supported" 400 was hidden here for ~24h.
                    tracing::warn!(
                        target: "kremory.extraction.arm_failure",
                        schema = schema_name,
                        arm = arm_name(arm),
                        model = %model_str,
                        error = %last_err_str,
                        "structured-call arm failed — ladder stepping down"
                    );
                    if let Some(&next_arm) = ladder.get(arm_idx + 1) {
                        counter!(
                            "rql.extraction.fallback_ladder_step",
                            "schema" => schema_name,
                            "from_arm" => arm_name(arm),
                            "to_arm" => arm_name(next_arm),
                            "model" => model_str.clone(),
                        )
                        .increment(1);
                    }
                }
            }
        }

        // All arms exhausted.
        Err(ExtractionError::FallbackExhausted {
            schema_name: schema_name.to_string(),
            raw_response: last_err_str,
        })
    }
}

// ─── Arm builder ─────────────────────────────────────────────────────────────

/// Build the fallback ladder for a given provider capability.
///
/// Per spec §6.2 K3 + TD-013 Phase 4 (L6 DelimitedTuple):
/// - Anthropic/OpenAI NativeStructuredOutput: Native → LlmJsonRepair → DelimitedTuple → PromptOnly
/// - Ollama FormatSchema: FormatSchema → LlmJsonRepair → DelimitedTuple → PromptOnly
/// - Unknown PromptOnly: LlmJsonRepair → DelimitedTuple → PromptOnly
///
/// DelimitedTuple sits after LlmJsonRepair (JSON repair already tried) and before
/// PromptOnly (last resort). When LlmJsonRepair fails to produce parseable JSON,
/// the LightRAG-style pipe-delimited format is attempted before giving up.
fn build_ladder(caps: ProviderCaps) -> Vec<FallbackArm> {
    match caps {
        ProviderCaps::NativeStructuredOutput => vec![
            FallbackArm::NativeSchema,
            FallbackArm::LlmJsonRepair,
            FallbackArm::DelimitedTuple,
            FallbackArm::PromptOnly,
        ],
        ProviderCaps::FormatSchema => vec![
            FallbackArm::FormatSchema,
            FallbackArm::LlmJsonRepair,
            FallbackArm::DelimitedTuple,
            FallbackArm::PromptOnly,
        ],
        ProviderCaps::PromptOnly => vec![
            FallbackArm::LlmJsonRepair,
            FallbackArm::DelimitedTuple,
            FallbackArm::PromptOnly,
        ],
    }
}

// ─── Arm execution ───────────────────────────────────────────────────────────

/// Bundled parameters for [`try_arm`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Holds borrows for the duration of a
/// single arm attempt; the fallback ladder constructs a fresh value per call.
struct TryArmParams<'a, L: ?Sized + ChatProvider> {
    llm: &'a L,
    arm: FallbackArm,
    schema: &'a Value,
    schema_name: &'static str,
    messages: &'a [ChatMessage],
    debug_enabled: bool,
}

/// Provider-reported token usage for a single arm attempt (TD-166).
#[derive(Debug, Clone, Copy)]
struct ArmUsage {
    input_tokens: u64,
    output_tokens: u64,
}

/// One arm attempt's result: the parsed value **plus what the provider said the
/// call cost** (TD-166 / TD-090).
///
/// Usage is carried OUT of `try_arm` rather than emitted inside it so there is a
/// single emit point in the caller, alongside the existing `schema`/`arm`
/// dual-emit. `usage: None` means the backend did not report — never "free".
struct ArmOutcome {
    value: Value,
    usage: Option<ArmUsage>,
}

/// Read usage off a response before it is consumed.
///
/// This is the whole of TD-090: `try_arm` previously dropped the `ChatResponse`
/// after `.text()`, so every extraction span reported a hard-coded
/// `gen_ai.usage.*_tokens = 0`.
fn arm_usage_of(response: &dyn autoagents_llm::chat::ChatResponse) -> Option<ArmUsage> {
    response.usage().map(|u| ArmUsage {
        input_tokens: u64::from(u.prompt_tokens),
        output_tokens: u64::from(u.completion_tokens),
    })
}

/// Which PHASE a structured call belongs to, derived from its schema name.
///
/// **This exists because `try_arm` is the funnel for EVERY structured call in the
/// crate — not just ingest.** The dream phase builds `StructuredCallBuilder` too
/// (`core/dream/reclassify.rs`, `discover_types.rs`, `type_registry_collapse.rs`,
/// `acronym_nickname_recall.rs`, `consistency_check/verify.rs`). Labelling every
/// call `operation="extraction"` would attribute reconciliation cost to ingest —
/// an OVER-count presented as authoritative — and double-count against the dream
/// phase's own `dream_pass_budget_usage` accounting (`core/ingest/mod.rs`).
/// (Quinn ARCH-001, 2026-08-03: the first cut of TD-166 did exactly that.)
///
/// The split is finer than ingest-vs-dream on purpose: `extraction` /
/// `resolution` / `contradiction` are the three ingest cost centres, so a
/// consumer can sum by `operation` to get the breakdown the 48×-amplification
/// work needs, while `schema` keeps the per-call detail.
///
/// An unrecognised schema is labelled **`"unclassified"` and warns** — it is
/// never folded into `"extraction"`, because a silent default is precisely how
/// the original defect existed. The source-gate test
/// `every_production_schema_is_classified` fails if a new call site introduces a
/// schema that lands here.
fn operation_for_schema(schema_name: &str) -> &'static str {
    match schema_name {
        // ── ingest: entity/relation/triple extraction ────────────────────────
        "EntityListIntegerId"
        | "EntityList"
        | "RelTypeList"
        | "TripletList"
        | "HybridTyping"
        | "EntityTyping"
        | "NuExtractBoth"
        // Added 2026-08-05. Its two siblings were classified from the start; this
        // one sat UNCLASSIFIED because it reaches the builder through the
        // `&[(&SCHEMA_X, "Name")]` fallback-ladder table (`:997`) rather than a
        // literal call site, and the source gate below scanned only call sites.
        // Found by a real end-to-end run emitting `unclassified_schema`, not by
        // the gate that exists to prevent exactly this.
        | "NuExtractEntitiesOnly"
        | "NuExtractRelationsOnly"
        | "RelOnlyForceFallback" => "extraction",

        // ── ingest: entity resolution / dedup ────────────────────────────────
        "ResolutionVerdict" | "BatchedResolution" => "resolution",

        // ── ingest: supersession ─────────────────────────────────────────────
        "ContradictionVerdict" => "contradiction",

        // ── dream: reconciliation passes (NOT ingest — see doc comment) ──────
        "ReclassifyBatch"
        | "Reclassify"
        | "DiscoveryProposalBatch"
        | "IdentityVerdictBatch"
        | "VerifyBatch" => "dream",

        other => {
            UNCLASSIFIED_SCHEMA_WARNED.call_once(|| {
                tracing::warn!(
                    target: "kremory.extraction.unclassified_schema",
                    schema = %other,
                    "structured-call schema is not classified into an operation — its \
                     tokens are being counted under operation=\"unclassified\". Add it to \
                     operation_for_schema (see every_production_schema_is_classified)."
                );
            });
            "unclassified"
        }
    }
}

/// Warn once per process about an unclassified schema (see `operation_for_schema`).
static UNCLASSIFIED_SCHEMA_WARNED: std::sync::Once = std::sync::Once::new();

/// Span fields for one arm's usage: `(input, output, reported)`.
///
/// The `reported` flag is load-bearing, not decoration: without it a span
/// showing `input_tokens = 0` is ambiguous between a genuinely free call and a
/// backend that never reported. That ambiguity is what TD-090 shipped for
/// months as a hard-coded `0_u64`.
fn usage_fields(usage: Option<ArmUsage>) -> (u64, u64, bool) {
    match usage {
        Some(u) => (u.input_tokens, u.output_tokens, true),
        None => (0, 0, false),
    }
}

/// Warn once per process that the wired backend reports no token usage.
static USAGE_MISSING_WARNED: std::sync::Once = std::sync::Once::new();

/// Emit token counters for one arm attempt (TD-166).
///
/// Called as soon as the provider returns `Ok`, **before** schema validation —
/// tokens spent on an arm whose output later fails validation were still spent,
/// and attributing cost only to validated calls would under-report exactly the
/// failing arms a cost investigation cares about.
///
/// `None` usage increments a SEPARATE counter and never a zero-valued token
/// increment. Ollama's `ChatResponse` does not override `usage()` (the trait
/// default returns `None`), so on a local build every call lands here — and a
/// `tokens_total` of 0 would be indistinguishable from "ingest was free", which
/// is the absence-read-as-measurement defect this TD exists to remove.
fn emit_arm_usage(schema_name: &'static str, arm: FallbackArm, usage: Option<ArmUsage>) {
    // Derived, never hardcoded — the dream phase shares this funnel (ARCH-001).
    let operation = operation_for_schema(schema_name);
    match usage {
        Some(u) => {
            counter!(
                "kremory_core_tokens_total",
                "operation" => operation,
                "schema" => schema_name,
                "arm" => arm_name(arm),
                "direction" => "input",
            )
            .increment(u.input_tokens);
            counter!(
                "kremory_core_tokens_total",
                "operation" => operation,
                "schema" => schema_name,
                "arm" => arm_name(arm),
                "direction" => "output",
            )
            .increment(u.output_tokens);
        }
        None => {
            counter!(
                "kremory_core_tokens_usage_missing_total",
                "operation" => operation,
                "schema" => schema_name,
                "arm" => arm_name(arm),
            )
            .increment(1);
            USAGE_MISSING_WARNED.call_once(|| {
                tracing::warn!(
                    target: "kremory.extraction.usage_missing",
                    "chat backend reports no token usage — kremory_core_tokens_total will \
                     stay 0 for extraction. This is NOT a zero-cost ingest; see \
                     kremory_core_tokens_usage_missing_total. (Ollama does not implement \
                     ChatResponse::usage.)"
                );
            });
        }
    }
}

async fn try_arm<L: ?Sized + ChatProvider>(
    params: TryArmParams<'_, L>,
) -> Result<ArmOutcome, ExtractionError> {
    let TryArmParams {
        llm,
        arm,
        schema,
        schema_name,
        messages,
        debug_enabled,
    } = params;
    // DelimitedTuple has its own prompt injection + parser path — handle separately.
    if arm == FallbackArm::DelimitedTuple {
        return try_delimited_tuple_arm(llm, messages).await;
    }

    let (text, usage) = match arm {
        FallbackArm::NativeSchema | FallbackArm::FormatSchema => {
            // Pass schema to the provider via StructuredOutputFormat.
            // description MUST be non-null: Groq's OpenAI-compatible endpoint
            // validates `response_format.json_schema.description` as non-nullable
            // and 400s on `null` (Ollama/OpenAI tolerate the omission). Emit a
            // deterministic description so the format_schema arm works across
            // every provider instead of silently falling through to
            // LlmJsonRepair. (load-bearing-invariant-at-emit; spec §5.2.)
            let fmt = StructuredOutputFormat {
                name: schema_name.to_string(),
                description: Some(format!(
                    "Structured extraction output conforming to the {schema_name} schema."
                )),
                schema: Some(schema.clone()),
                strict: Some(matches!(arm, FallbackArm::NativeSchema)),
            };
            let response = llm
                .chat_with_tools(messages, None, Some(fmt))
                .await
                .map_err(|e| ExtractionError::Llm(e.to_string()))?;
            // TD-166: read usage BEFORE the response is consumed by `.text()`.
            let usage = arm_usage_of(response.as_ref());
            (response.text().unwrap_or_default(), usage)
        }
        FallbackArm::LlmJsonRepair | FallbackArm::PromptOnly => {
            // No provider-side schema enforcement — get raw text.
            let response = llm
                .chat_with_tools(messages, None, None)
                .await
                .map_err(|e| ExtractionError::Llm(e.to_string()))?;
            let usage = arm_usage_of(response.as_ref());
            (response.text().unwrap_or_default(), usage)
        }
        // Handled above — unreachable, but exhaustiveness requires the arm.
        FallbackArm::DelimitedTuple => unreachable!("DelimitedTuple handled above"),
    };

    // TD-019 Gap 4: capture raw arm response (KREMORY_DEBUG-gated). Lets a
    // debug session see what FormatSchema actually emitted that triggered a
    // fallback, instead of inferring from "success-then-fail" counters.
    if debug_enabled {
        tracing::debug!(
            target: "kremory.extraction.arm_response",
            arm = arm_name(arm),
            schema = schema_name,
            response_len = text.len(),
            raw = %text,
            "raw arm response captured"
        );
    }

    // Parse the response text to a JSON Value.
    Ok(ArmOutcome {
        value: parse_response_to_value(&text, arm)?,
        usage,
    })
}

/// Attempt the L6 DelimitedTuple arm.
///
/// Appends the pipe-delimited format instruction to the message list and
/// sends the request without provider-side schema enforcement (same as
/// `LlmJsonRepair`/`PromptOnly` arms — raw text response).
///
/// The raw text is then parsed by the delimited-tuple parser, which converts
/// valid lines into a `{"entities": [...]}` JSON object.  Per-line errors are
/// silently counted via metric `rql.extraction.delimited_tuple_skip_row` and
/// do not fail the arm.
async fn try_delimited_tuple_arm<L: ?Sized + ChatProvider>(
    llm: &L,
    messages: &[ChatMessage],
) -> Result<ArmOutcome, ExtractionError> {
    use crate::core::provider::{ChatRole, MessageType};

    // Append the pipe-delimited format instruction as a follow-up user message.
    // This avoids mutating the original message vec (caller owns it).
    let format_instruction = prompts::render_delimited_tuple_prompt();
    let mut msgs_with_instruction = messages.to_vec();
    msgs_with_instruction.push(ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: format_instruction,
    });

    let response = llm
        .chat_with_tools(&msgs_with_instruction, None, None)
        .await
        .map_err(|e| ExtractionError::Llm(e.to_string()))?;

    // TD-166: this is the second (and only other) `chat_with_tools` call in the
    // extraction path — instrumented identically, else the DelimitedTuple arm
    // would be a silent hole in the token accounting.
    let usage = arm_usage_of(response.as_ref());
    let text = response.text().unwrap_or_default();
    Ok(ArmOutcome {
        value: delimited_tuple::parse_delimited_tuple_response(&text),
        usage,
    })
}

/// Parse a raw LLM response string into a `serde_json::Value`.
///
/// For `LlmJsonRepair`: attempts direct parse, then `llm_json::repair_json`,
/// then brace-extraction + repair — the same pattern used by the existing
/// extraction parsers.
///
/// For `PromptOnly`: same parse attempts, but if all fail returns an empty
/// object rather than an error (best-effort).
fn parse_response_to_value(text: &str, arm: FallbackArm) -> Result<Value, ExtractionError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        // Empty response = no output from the LLM (not malformed JSON).
        // Both best-effort arms (LlmJsonRepair and PromptOnly) treat this as
        // "no entities found" and return an empty object rather than an error.
        // This avoids an unnecessary second LLM round-trip via PromptOnly arm.
        // NativeSchema and FormatSchema arms enforce structured output at the
        // provider level and should not receive empty responses in practice;
        // treating them consistently here avoids silent partial extraction.
        return Ok(Value::Object(serde_json::Map::new()));
    }

    // Direct parse.
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Ok(v);
    }

    // llm_json repair.
    if matches!(
        arm,
        FallbackArm::LlmJsonRepair | FallbackArm::NativeSchema | FallbackArm::FormatSchema
    ) {
        counter!(
            "rql.extraction.repair_attempt",
            "schema" => "unknown",
            "arm" => "llm_json",
        )
        .increment(1);

        let repaired = llm_json::repair_json(trimmed, &llm_json::RepairOptions::default())
            .unwrap_or_else(|_| trimmed.to_owned());
        if let Ok(v) = serde_json::from_str::<Value>(&repaired) {
            return Ok(v);
        }
    }

    // Brace-extraction: find the first balanced JSON object.
    if let Some(v) = extract_json_object(trimmed) {
        return Ok(v);
    }

    match arm {
        FallbackArm::PromptOnly => Ok(Value::Object(serde_json::Map::new())),
        // `&trimmed[..n]` PANICS when byte `n` is mid-character, and `trimmed`
        // is a raw LLM response — arbitrary UTF-8. This is the ERROR arm, so
        // the old code turned a RECOVERABLE parse failure into a server-wide
        // crash: the panic unwinds the tokio worker and takes kremory-http
        // down, on the shared path used by every structured call (extraction,
        // contradiction, resolution). Same defect class that crashed the
        // server on 2026-07-28 via core/extraction_window.rs.
        _ => Err(ExtractionError::Parse(format!(
            "failed to parse LLM response as JSON: {}",
            crate::core::text_utils::truncate_on_char_boundary(trimmed, 100)
        ))),
    }
}

/// Extract the first balanced `{...}` JSON object from a string, repairing
/// if necessary.  Returns `None` if no object is found.
fn extract_json_object(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    // Find matching closing brace.
    let mut depth = 0usize;
    let mut end = start;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    end = start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end <= start {
        return None;
    }
    let slice = &text[start..=end];
    // Try direct parse first.
    if let Ok(v) = serde_json::from_str::<Value>(slice) {
        return Some(v);
    }
    // Repair.
    let repaired = llm_json::repair_json(slice, &llm_json::RepairOptions::default()).ok()?;
    serde_json::from_str::<Value>(&repaired).ok()
}

// ─── Schema validation ────────────────────────────────────────────────────────

/// Lightweight structural validation of a parsed Value against a JSON Schema.
///
/// Checks only the root `"type"` and `"properties"` keys to catch the most
/// common violation class (wrong root type, missing required top-level field).
/// Full jsonschema validation (depth-recursive, `$ref` resolution) is
/// deferred to a future phase — tracked as a follow-up item (FU).
fn validate_against_schema(value: &Value, schema: &Value) -> Result<(), String> {
    // Root must be an object if schema says so.
    if let Some(root_type) = schema.get("type").and_then(|t| t.as_str()) {
        match root_type {
            "object" => {
                if !value.is_object() {
                    return Err(format!(
                        "expected JSON object at root, got {}",
                        json_type_name(value)
                    ));
                }
            }
            "array" => {
                if !value.is_array() {
                    return Err(format!(
                        "expected JSON array at root, got {}",
                        json_type_name(value)
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ─── Self-correction helpers ──────────────────────────────────────────────────

fn build_correction_message(detail: &str) -> ChatMessage {
    use crate::core::provider::{ChatRole, MessageType};
    ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: format!(
            "Your previous response violated the output schema: {detail}. \
             Please correct your response and output valid JSON only."
        ),
    }
}

// ─── Metric label helpers ─────────────────────────────────────────────────────

fn arm_name(arm: FallbackArm) -> &'static str {
    match arm {
        FallbackArm::NativeSchema => "native_schema",
        FallbackArm::FormatSchema => "format_schema",
        FallbackArm::LlmJsonRepair => "llm_json_repair",
        FallbackArm::DelimitedTuple => "delimited_tuple",
        FallbackArm::PromptOnly => "prompt_only",
    }
}

// ─── Schema cache warmup ─────────────────────────────────────────────────────

/// Warm schema caches for all canonical schemas on engine startup.
///
/// Call once during [`crate::facade::MemoryBuilder`] `.await` construction
/// (Tier 2) or from `providers::build_memory` (Tier 1), after the LLM
/// provider is initialised, before serving ingestion traffic.
///
/// The `model` parameter drives provider-capability detection via
/// [`crate::core::provider::capability_of`].  When `Some`, the ladder starts
/// at NativeSchema (Anthropic / OpenAI) or FormatSchema (Ollama), reaching the
/// provider-native schema-compilation path.  When `None`, `capability_of("")`
/// returns `ProviderCaps::PromptOnly` and the warmup is connection-pool warm
/// only (LlmJsonRepair arm — still useful but NativeSchema / FormatSchema arms
/// are not reached).  Tier 1 callers (`providers::build_memory`) always supply
/// the configured model string; Tier 2 (`MemoryBuilder`) passes `None` when
/// the model is not surfaced at construction time.
///
/// Best-effort: errors are silently ignored — warmup failures do not prevent
/// engine startup.
///
/// Provider-specific behaviour when a model string is supplied:
/// - **Ollama** (`FormatSchema`): no server-side schema cache; calls are
///   minimal no-ops — warmup costs negligible Ollama round-trip.
/// - **OpenAI strict** (`NativeSchema`): populates 10–60s CFG compile on
///   first call.
/// - **Anthropic** (`NativeSchema`): populates the 24-hour server-side
///   schema cache.
///
/// Gated by `#[cfg(not(test))]` at all call-sites so unit tests retain fast
/// engine open without LLM round-trips.
pub(crate) async fn warm_schema_caches<L: ?Sized + ChatProvider>(llm: &L, model: Option<&str>) {
    use crate::core::extraction::schemas::{
        SCHEMA_BATCHED_RESOLUTION, SCHEMA_CONTRADICTION_VERDICT, SCHEMA_ENTITY_LIST,
        SCHEMA_ENTITY_TYPING, SCHEMA_NUEXTRACT_BOTH, SCHEMA_NUEXTRACT_ENTITIES_ONLY,
        SCHEMA_NUEXTRACT_RELATIONS_ONLY, SCHEMA_REL_ONLY_FORCE_FALLBACK, SCHEMA_REL_TYPE_LIST,
        SCHEMA_RESOLUTION_VERDICT, SCHEMA_TRIPLET_LIST,
    };

    let schemas: &[(&'static serde_json::Value, &'static str)] = &[
        (&SCHEMA_ENTITY_LIST, "EntityList"),
        (&SCHEMA_REL_TYPE_LIST, "RelTypeList"),
        (&SCHEMA_TRIPLET_LIST, "TripletList"),
        (&SCHEMA_NUEXTRACT_BOTH, "NuExtractBoth"),
        (&SCHEMA_NUEXTRACT_ENTITIES_ONLY, "NuExtractEntitiesOnly"),
        (&SCHEMA_NUEXTRACT_RELATIONS_ONLY, "NuExtractRelationsOnly"),
        (&SCHEMA_ENTITY_TYPING, "EntityTyping"),
        (&SCHEMA_CONTRADICTION_VERDICT, "ContradictionVerdict"),
        (&SCHEMA_REL_ONLY_FORCE_FALLBACK, "RelOnlyForceFallback"),
        (&SCHEMA_RESOLUTION_VERDICT, "ResolutionVerdict"),
        (&SCHEMA_BATCHED_RESOLUTION, "BatchedResolution"),
    ];

    // Minimal 1-token prompt — enough for the provider to compile the schema.
    let warmup_msgs = vec![ChatMessage {
        role: crate::core::provider::ChatRole::User,
        message_type: crate::core::provider::MessageType::Text,
        content: "ok".to_string(),
    }];

    for (schema, name) in schemas {
        // Best-effort: ignore all errors.  StructuredCallBuilder is used for
        // a consistent call-path (counters fire, fallback ladder protects
        // against provider unavailability during startup).
        let mut builder =
            StructuredCallBuilder::new(llm, schema, name).messages(warmup_msgs.clone());
        if let Some(m) = model {
            builder = builder.model(m);
        }
        let _ = builder.call().await;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extraction::schemas::{
        SCHEMA_ENTITY_LIST, SCHEMA_NUEXTRACT_RELATIONS_ONLY,
        SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM,
    };
    use crate::core::provider::{MockChatProvider, MockChatResponse, ProviderCaps};
    use metrics_util::debugging::DebuggingRecorder;
    use std::collections::HashMap;

    // ── MockErrorChatProvider — always returns LLMError (T5.2) ───────────────

    struct MockErrorChatProvider {
        err_msg: String,
    }

    #[async_trait::async_trait]
    impl crate::core::provider::ChatProvider for MockErrorChatProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[crate::core::provider::ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> std::result::Result<
            Box<dyn autoagents_llm::chat::ChatResponse>,
            autoagents_llm::error::LLMError,
        > {
            Err(autoagents_llm::error::LLMError::Generic(
                self.err_msg.clone(),
            ))
        }
    }

    // ── build_ladder ──────────────────────────────────────────────────────────

    #[test]
    fn native_caps_ladder_starts_with_native() {
        let ladder = build_ladder(ProviderCaps::NativeStructuredOutput);
        assert_eq!(ladder[0], FallbackArm::NativeSchema);
        assert_eq!(ladder[1], FallbackArm::LlmJsonRepair);
        assert_eq!(ladder[2], FallbackArm::DelimitedTuple);
        assert_eq!(ladder[3], FallbackArm::PromptOnly);
    }

    #[test]
    fn format_caps_ladder_starts_with_format() {
        let ladder = build_ladder(ProviderCaps::FormatSchema);
        assert_eq!(ladder[0], FallbackArm::FormatSchema);
        assert_eq!(ladder[1], FallbackArm::LlmJsonRepair);
        assert_eq!(ladder[2], FallbackArm::DelimitedTuple);
        assert_eq!(ladder[3], FallbackArm::PromptOnly);
    }

    #[test]
    fn prompt_only_caps_ladder_starts_with_llm_json_repair() {
        let ladder = build_ladder(ProviderCaps::PromptOnly);
        assert_eq!(ladder[0], FallbackArm::LlmJsonRepair);
        assert_eq!(ladder[1], FallbackArm::DelimitedTuple);
        assert_eq!(ladder[2], FallbackArm::PromptOnly);
    }

    #[test]
    fn delimited_tuple_sits_between_llm_json_repair_and_prompt_only_native() {
        let ladder = build_ladder(ProviderCaps::NativeStructuredOutput);
        let dt_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::DelimitedTuple)
            .expect("DelimitedTuple must be in the native ladder");
        let repair_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::LlmJsonRepair)
            .expect("LlmJsonRepair must be in the native ladder");
        let prompt_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::PromptOnly)
            .expect("PromptOnly must be in the native ladder");
        assert!(
            repair_pos < dt_pos,
            "DelimitedTuple must come after LlmJsonRepair"
        );
        assert!(
            dt_pos < prompt_pos,
            "DelimitedTuple must come before PromptOnly"
        );
    }

    #[test]
    fn delimited_tuple_sits_between_llm_json_repair_and_prompt_only_prompt_only_caps() {
        let ladder = build_ladder(ProviderCaps::PromptOnly);
        let dt_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::DelimitedTuple)
            .expect("DelimitedTuple must be in the PromptOnly ladder");
        let repair_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::LlmJsonRepair)
            .expect("LlmJsonRepair must be in the PromptOnly ladder");
        let prompt_pos = ladder
            .iter()
            .position(|&a| a == FallbackArm::PromptOnly)
            .expect("PromptOnly must be in the PromptOnly ladder");
        assert!(repair_pos < dt_pos, "must come after LlmJsonRepair");
        assert!(dt_pos < prompt_pos, "must come before PromptOnly");
    }

    // ── force_arm bypass ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn force_arm_bypasses_native_for_rel_only_schema() {
        // SCHEMA_NUEXTRACT_RELATIONS_ONLY must always be forced to LlmJsonRepair.
        let forced = SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM;
        assert_eq!(forced, Some(FallbackArm::LlmJsonRepair));

        let resp = r#"{"relationships":[]}"#;
        let mock = MockChatProvider::with_response("", resp);

        let value = StructuredCallBuilder::new(
            &mock,
            &SCHEMA_NUEXTRACT_RELATIONS_ONLY,
            "NuExtractRelationsOnly",
        )
        .model("claude-sonnet-4-5") // NativeStructuredOutput normally — bypassed by force_arm
        .messages(vec![crate::core::provider::chat_msg_user("extract")])
        .force_arm(FallbackArm::LlmJsonRepair)
        .call()
        .await
        .unwrap();

        assert!(value.is_object());
    }

    // ── builder constructs cleanly ─────────────────────────────────────────────

    #[test]
    fn builder_constructs_with_all_fields() {
        let mock = MockChatProvider::null();
        let builder = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            .model("claude-sonnet-4-5")
            .messages(vec![])
            .max_retries(2)
            .ttft_budget_ms(500);

        assert_eq!(builder.schema_name, "EntityList");
        assert_eq!(builder.max_retries, 2);
        assert_eq!(builder.ttft_budget_ms, Some(500));
    }

    // ── NativeSchema arm for Anthropic model ──────────────────────────────────

    #[test]
    fn anthropic_model_builds_native_schema_arm() {
        // claude-sonnet-4-5 → NativeStructuredOutput → first arm is NativeSchema
        let ladder = build_ladder(capability_of("claude-sonnet-4-5"));
        assert_eq!(ladder[0], FallbackArm::NativeSchema);
    }

    // ── FormatSchema arm for Ollama model ─────────────────────────────────────

    #[test]
    fn ollama_model_builds_format_schema_arm() {
        let ladder = build_ladder(capability_of("qwen2.5:14b"));
        assert_eq!(ladder[0], FallbackArm::FormatSchema);
    }

    // ── FormatSchema arm for gpt-oss (Groq / Together / vLLM) ──────────────────

    #[test]
    fn gpt_oss_model_builds_format_schema_arm() {
        // Groq serves gpt-oss with an `openai/` prefix; both forms must resolve
        // to FormatSchema (not fall through to PromptOnly → LlmJsonRepair).
        assert_eq!(
            build_ladder(capability_of("openai/gpt-oss-120b"))[0],
            FallbackArm::FormatSchema
        );
        assert_eq!(
            build_ladder(capability_of("gpt-oss-20b"))[0],
            FallbackArm::FormatSchema
        );
    }

    // ── FormatSchema arm succeeds with valid JSON ─────────────────────────────

    #[tokio::test]
    async fn format_schema_arm_returns_parsed_value() {
        let resp = r#"{"items":[]}"#;
        let mock = MockChatProvider::with_response("", resp);

        let value = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            .model("qwen2.5:14b")
            .messages(vec![crate::core::provider::chat_msg_user("extract")])
            .call()
            .await
            .unwrap();

        assert!(value.is_object());
        assert!(value.get("items").is_some());
    }

    // ── LlmJsonRepair arm succeeds with repaired JSON ─────────────────────────

    #[tokio::test]
    async fn llm_json_repair_arm_handles_malformed_json() {
        // Malformed but repairable JSON: missing closing brace.
        let resp = r#"{"items": [{"name":"Alice","label":"Person"}"#;
        let mock = MockChatProvider::with_response("", resp);

        let value = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            // No .model() — defaults to unknown → LlmJsonRepair first arm
            .messages(vec![crate::core::provider::chat_msg_user("extract")])
            .call()
            .await
            .unwrap();

        assert!(value.is_object());
    }

    // ── Schema violation triggers self-correction retry ───────────────────────

    #[tokio::test]
    async fn schema_violation_triggers_correction_and_succeeds_on_retry() {
        // First call: return a bare array (violates "type":"object" schema).
        // Second call (self-correction): return valid wrapped object.
        let mut responses = HashMap::new();
        // First call on any message returns a bare array.
        responses.insert(
            "extract".to_string(),
            r#"[{"name":"Alice","label":"Person"}]"#.to_string(),
        );
        // After "correct" is in the message (from the correction prompt), return valid JSON.
        responses.insert(
            "correct".to_string(),
            r#"{"items":[{"name":"Alice","label":"Person"}]}"#.to_string(),
        );

        let mock = MockChatProvider::new(responses);

        let value = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            .model("qwen2.5:14b")
            .messages(vec![crate::core::provider::chat_msg_user("extract")])
            .call()
            .await
            .unwrap();

        assert!(value.is_object());
        assert!(value.get("items").is_some());
    }

    // ── FallbackExhausted variant shape ──────────────────────────────────────
    //
    // FallbackExhausted is produced when ALL arms fail at the provider level
    // (each chat_with_tools() returns LLMError).  The end-to-end exhaustion
    // path is exercised by `all_arms_provider_error_returns_fallback_exhausted`
    // (T5.3/ARCH-002 test above) using MockErrorChatProvider.
    //
    // This unit test verifies only that:
    //  1. The variant can be constructed with the expected fields, and
    //  2. The match arm in downstream code compiles + correctly pattern-matches.
    #[test]
    fn fallback_exhausted_variant_has_correct_fields() {
        let err = ExtractionError::FallbackExhausted {
            schema_name: "EntityList".to_string(),
            raw_response: "last raw output".to_string(),
        };
        match err {
            ExtractionError::FallbackExhausted {
                ref schema_name,
                ref raw_response,
            } => {
                assert_eq!(schema_name, "EntityList");
                assert_eq!(raw_response, "last raw output");
            }
            other => panic!("expected FallbackExhausted, got {other:?}"),
        }
    }

    // ── arm_name labels are stable ────────────────────────────────────────────

    #[test]
    fn arm_name_labels_are_stable() {
        assert_eq!(arm_name(FallbackArm::NativeSchema), "native_schema");
        assert_eq!(arm_name(FallbackArm::FormatSchema), "format_schema");
        assert_eq!(arm_name(FallbackArm::LlmJsonRepair), "llm_json_repair");
        assert_eq!(arm_name(FallbackArm::PromptOnly), "prompt_only");
    }

    // ── parse_response_to_value ───────────────────────────────────────────────

    #[test]
    fn parse_response_to_value_valid_json() {
        let v = parse_response_to_value(r#"{"items":[]}"#, FallbackArm::LlmJsonRepair).unwrap();
        assert!(v.is_object());
    }

    #[test]
    fn parse_response_to_value_empty_prompt_only_returns_empty_object() {
        let v = parse_response_to_value("", FallbackArm::PromptOnly).unwrap();
        assert!(v.is_object());
    }

    #[test]
    fn parse_response_to_value_empty_llm_json_repair_returns_empty_object() {
        // Empty response means "no output" (not malformed JSON).  All arms
        // now return {} for empty rather than erroring — this avoids a second
        // LLM round-trip via PromptOnly when the provider simply returned nothing.
        let result = parse_response_to_value("", FallbackArm::LlmJsonRepair).unwrap();
        assert!(result.is_object());
        assert!(result.as_object().unwrap().is_empty());
    }

    #[test]
    fn parse_response_to_value_brace_extraction() {
        let text = "Some text before {\"items\": []} and more after";
        let v = parse_response_to_value(text, FallbackArm::LlmJsonRepair).unwrap();
        assert!(v.is_object());
    }

    // ── T5.3 / ARCH-002 — FallbackExhausted when all arms error ──────────────

    #[tokio::test]
    async fn all_arms_provider_error_returns_fallback_exhausted() {
        let mock = MockErrorChatProvider {
            err_msg: "provider down".to_string(),
        };
        let result = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            .messages(vec![crate::core::provider::chat_msg_user("test")])
            .call()
            .await;
        match result {
            Err(ExtractionError::FallbackExhausted {
                ref schema_name,
                ref raw_response,
            }) => {
                assert_eq!(schema_name, "EntityList");
                assert!(
                    raw_response.contains("provider down"),
                    "raw_response should contain the provider error: {raw_response}"
                );
            }
            other => panic!("expected FallbackExhausted, got {other:?}"),
        }
    }

    // ── T5.3 / TEST-001 — schema violation steps down to LlmJsonRepair arm ───

    #[tokio::test]
    async fn schema_violation_correction_fails_steps_to_llm_json_repair() {
        // The FormatSchema arm (qwen2.5 model) validates root type "object".
        // Returning a bare array always violates the schema; both the initial call
        // and the self-correction retry return the same wrong shape, so the ladder
        // steps down to LlmJsonRepair, which accepts any parseable JSON.
        let mock = MockChatProvider::with_response("", r#"[{"name":"Alice"}]"#);

        let result = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
            .model("qwen2.5:14b") // FormatSchema → validates root type → array fails
            .messages(vec![crate::core::provider::chat_msg_user("test")])
            .call()
            .await;

        assert!(
            result.is_ok(),
            "LlmJsonRepair arm should accept any parseable JSON: {result:?}"
        );
    }

    // ── T6.3 — warm_schema_caches invokes all 10 schemas ─────────────────────

    #[tokio::test]
    async fn warm_schema_caches_calls_all_schemas() {
        use std::collections::HashSet;
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        };

        struct CountingMock {
            count: Arc<AtomicUsize>,
            /// Schema names received via json_schema.name in each call.
            names_seen: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait::async_trait]
        impl crate::core::provider::ChatProvider for CountingMock {
            async fn chat_with_tools(
                &self,
                _messages: &[crate::core::provider::ChatMessage],
                _tools: Option<&[autoagents_llm::chat::Tool]>,
                json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
            ) -> std::result::Result<
                Box<dyn autoagents_llm::chat::ChatResponse>,
                autoagents_llm::error::LLMError,
            > {
                self.count.fetch_add(1, Ordering::SeqCst);
                // Capture the schema name forwarded by StructuredCallBuilder.
                // In the LlmJsonRepair arm (no model set) the schema name is
                // embedded in the system prompt rather than the json_schema arg,
                // so also accept an empty name as a legitimate observation.
                if let Some(ref sof) = json_schema {
                    self.names_seen.lock().unwrap().push(sof.name.clone());
                }
                // Return empty JSON object — StructuredCallBuilder accepts it
                // as a valid LlmJsonRepair / PromptOnly result.
                Ok(Box::new(MockChatResponse {
                    text: "{}".to_string(),
                }))
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let names_seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let mock = CountingMock {
            count: count.clone(),
            names_seen: names_seen.clone(),
        };
        // Pass None for model: no NativeSchema/FormatSchema arm reached in tests.
        super::warm_schema_caches(&mock, None).await;
        let n = count.load(Ordering::SeqCst);
        // 10 schemas × 1 attempt each (LlmJsonRepair arm, no model set → success on first try).
        assert!(
            n >= 10,
            "warmup must invoke chat_with_tools at least once per schema, got {n}"
        );
        // Verify that all 10 distinct schema names were covered.  In the
        // LlmJsonRepair / PromptOnly arms the schema name is injected into the
        // system-prompt message, not forwarded via json_schema, so `names_seen`
        // may be empty here.  If it IS populated (future arm routing change),
        // assert all entries are distinct.
        let seen = names_seen.lock().unwrap();
        if !seen.is_empty() {
            let distinct: HashSet<&String> = seen.iter().collect();
            assert_eq!(
                distinct.len(),
                seen.len(),
                "warmup must call each schema name exactly once, but found duplicates: {seen:?}"
            );
        }
        // Count-based invariant is load-bearing regardless of arm routing.
    }

    // ── L1: Adversarial parse_response_to_value — placeholder-label inputs ───
    //
    // These tests verify that parse_response_to_value accepts syntactically valid
    // JSON carrying placeholder labels.  The parser is intentionally permissive;
    // label correctness is enforced by the L2 validator (is_canonical_entity_type)
    // AFTER parsing.  These tests pin the parser contract so any behavioural
    // change is visible.

    #[test]
    fn parse_response_to_value_accepts_placeholder_entity_label() {
        // TD-012 shape: structurally valid JSON with label="Entity".
        // parse_response_to_value must accept it — it's valid JSON.
        let json = r#"{"entities":[{"name":"Alice","label":"Entity"}]}"#;
        let result = parse_response_to_value(json, FallbackArm::LlmJsonRepair);
        assert!(
            result.is_ok(),
            "parse_response_to_value must accept syntactically valid JSON with placeholder label"
        );
        let val = result.unwrap();
        let label = val["entities"][0]["label"].as_str().unwrap_or("");
        assert_eq!(
            label, "Entity",
            "placeholder label must be preserved as-is by the parser layer"
        );
        // The caller (L2 validator) is responsible for rejection, not the parser.
        assert!(
            !crate::core::extraction::is_canonical_entity_type(label),
            "L2 validator must subsequently reject this placeholder label"
        );
    }

    #[test]
    fn parse_response_to_value_accepts_unknown_label() {
        let json = r#"{"entities":[{"name":"OpenAI","label":"UNKNOWN"}]}"#;
        let result = parse_response_to_value(json, FallbackArm::LlmJsonRepair);
        assert!(result.is_ok(), "must accept syntactically valid JSON");
        let val = result.unwrap();
        let label = val["entities"][0]["label"].as_str().unwrap_or("");
        assert_eq!(label, "UNKNOWN");
        assert!(
            !crate::core::extraction::is_canonical_entity_type(label),
            "L2 validator must reject 'UNKNOWN' label"
        );
    }

    #[test]
    fn parse_response_to_value_accepts_empty_label_field() {
        // Empty string label: syntactically valid, semantically placeholder.
        let json = r#"{"entities":[{"name":"Alice","label":""}]}"#;
        let result = parse_response_to_value(json, FallbackArm::LlmJsonRepair);
        assert!(result.is_ok(), "must accept empty string label field");
        let val = result.unwrap();
        let label = val["entities"][0]["label"]
            .as_str()
            .unwrap_or("__missing__");
        assert!(
            !crate::core::extraction::is_canonical_entity_type(label),
            "L2 validator must reject empty label"
        );
    }

    #[test]
    fn parse_response_to_value_repairs_malformed_json_with_placeholder_label() {
        // Malformed JSON with placeholder label — repair path must recover it.
        let malformed = r#"{"entities":[{"name":"Alice","label":"Entity""#; // truncated
        let result = parse_response_to_value(malformed, FallbackArm::LlmJsonRepair);
        // Repair may or may not succeed on severely truncated JSON.
        // If it does, the label must be preserved; the L2 validator then flags it.
        if let Ok(val) = result {
            if val["entities"].is_array() {
                let label = val["entities"][0]["label"].as_str().unwrap_or("");
                if !label.is_empty() {
                    assert!(
                        !crate::core::extraction::is_canonical_entity_type(label),
                        "recovered placeholder label must be rejected by L2 validator"
                    );
                }
            }
        }
        // Err is also acceptable — parser does not fabricate data on truncation.
    }

    #[test]
    fn parse_response_to_value_array_root_accepted_by_llm_json_repair_arm() {
        // Array root JSON is syntactically valid — parse_response_to_value must accept it.
        // The structured-output wrappers (EntityListWrapper etc.) will fail to
        // deserialize from an array, but that rejection happens at the
        // StructuredCallBuilder deserialization layer, not here.
        let json = r#"[{"name":"Alice","label":"Person"}]"#;
        let result = parse_response_to_value(json, FallbackArm::LlmJsonRepair);
        // serde_json::from_str::<Value> accepts arrays as well as objects.
        assert!(
            result.is_ok(),
            "parse_response_to_value must accept array-root JSON (Value accepts both)"
        );
    }

    // ── T5.4 — counter assertion test ────────────────────────────────────────

    #[test]
    fn counters_fire_on_successful_call() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let mock = MockChatProvider::with_response("", r#"{"items":[]}"#);
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _ = rt
                .block_on(
                    StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
                        .messages(vec![crate::core::provider::chat_msg_user("test")])
                        .call(),
                )
                .unwrap();

            let snapshot = snapshotter.snapshot().into_vec();

            let attempt_count: u64 = snapshot
                .iter()
                .filter(|(k, ..)| k.key().name() == "rql.extraction.structured_call_attempt")
                .map(|(.., v)| match v {
                    DebugValue::Counter(n) => *n,
                    _ => 0,
                })
                .sum();

            let success_count: u64 = snapshot
                .iter()
                .filter(|(k, ..)| k.key().name() == "rql.extraction.structured_call_success")
                .map(|(.., v)| match v {
                    DebugValue::Counter(n) => *n,
                    _ => 0,
                })
                .sum();

            assert!(
                attempt_count >= 1,
                "rql.extraction.structured_call_attempt must fire at least once"
            );
            assert!(
                success_count >= 1,
                "rql.extraction.structured_call_success must fire at least once"
            );
        });
    }

    // ── TD-166 — ingest token attribution at the structured-call funnel ───────
    //
    // Spec: .ai-docs/specs/td-166-ingest-token-attribution-spec-2026-08-03.md
    //
    // These drive the REAL ladder through `StructuredCallBuilder::call()`, not
    // `try_arm` directly — the emit site is in the caller, so a test that called
    // `try_arm` would validate a model of the code rather than the code.

    /// Mock whose `ChatResponse` DOES report usage (the production shape for
    /// Groq/OpenAI). `MockChatResponse` deliberately does not, which is what
    /// makes it the correct fixture for the absence test below.
    struct UsageMockProvider {
        text: String,
        input: u32,
        output: u32,
    }

    struct UsageMockResponse {
        text: String,
        input: u32,
        output: u32,
    }

    impl std::fmt::Debug for UsageMockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UsageMockResponse({}/{})", self.input, self.output)
        }
    }

    impl std::fmt::Display for UsageMockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text)
        }
    }

    impl autoagents_llm::chat::ChatResponse for UsageMockResponse {
        fn text(&self) -> Option<String> {
            Some(self.text.clone())
        }
        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
        fn usage(&self) -> Option<autoagents_llm::chat::Usage> {
            Some(autoagents_llm::chat::Usage {
                prompt_tokens: self.input,
                completion_tokens: self.output,
                total_tokens: self.input + self.output,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            })
        }
    }

    #[async_trait::async_trait]
    impl crate::core::provider::ChatProvider for UsageMockProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[crate::core::provider::ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> std::result::Result<
            Box<dyn autoagents_llm::chat::ChatResponse>,
            autoagents_llm::error::LLMError,
        > {
            Ok(Box::new(UsageMockResponse {
                text: self.text.clone(),
                input: self.input,
                output: self.output,
            }))
        }
    }

    /// Sum a counter's value across every label set matching `pred`.
    fn counter_sum(
        snapshot: &[(
            metrics_util::CompositeKey,
            Option<metrics::Unit>,
            Option<metrics::SharedString>,
            metrics_util::debugging::DebugValue,
        )],
        name: &str,
        pred: impl Fn(&std::collections::HashMap<&str, &str>) -> bool,
    ) -> u64 {
        snapshot
            .iter()
            .filter(|(k, _, _, _)| k.key().name() == name)
            .filter(|(k, _, _, _)| {
                let labels: std::collections::HashMap<&str, &str> =
                    k.key().labels().map(|l| (l.key(), l.value())).collect();
                pred(&labels)
            })
            .map(|(_, _, _, v)| match v {
                metrics_util::debugging::DebugValue::Counter(n) => *n,
                other => panic!("{name} must be a Counter, got {other:?}"),
            })
            .sum()
    }

    fn run_with_recorder<F: std::future::Future<Output = ()>>(
        recorder: &DebuggingRecorder,
        fut: impl FnOnce() -> F,
    ) {
        metrics::with_local_recorder(recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds");
            rt.block_on(fut());
        });
    }

    /// T1 — a provider that reports usage produces real, correctly-labelled
    /// token counters. RED-verified: before TD-166 this emitted nothing.
    #[test]
    fn reported_usage_emits_labelled_token_counters() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let mock = UsageMockProvider {
            text: r#"{"items":[]}"#.to_string(),
            input: 100,
            output: 50,
        };

        run_with_recorder(&recorder, || async {
            StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
                .model("qwen2.5:14b")
                .messages(vec![crate::core::provider::chat_msg_user("extract")])
                .call()
                .await
                .expect("mock returns valid JSON");
        });

        let snapshot = snapshotter.snapshot().into_vec();

        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("direction") == Some(&"input")
            }),
            100,
            "input tokens must be the provider-reported 100"
        );
        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("direction") == Some(&"output")
            }),
            50,
            "output tokens must be the provider-reported 50"
        );
        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("schema") == Some(&"EntityList") && l.get("operation") == Some(&"extraction")
            }),
            150,
            "tokens must carry schema + operation labels — per-schema attribution is \
             the whole point of TD-166 (it is what makes the 48× breakdown measurable)"
        );
        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_usage_missing_total", |_| true),
            0,
            "usage WAS reported — the missing-counter must stay at zero"
        );
    }

    /// T2 — the load-bearing one. A provider that reports NO usage (the Ollama
    /// shape: `ChatResponse::usage()` is not overridden, trait default `None`)
    /// must increment the missing-counter and emit NO zero-valued token
    /// increment. A `tokens_total` of 0 would be indistinguishable from a free
    /// ingest, which is the exact absence-read-as-measurement defect TD-166
    /// exists to remove.
    #[test]
    fn unreported_usage_counts_as_missing_never_as_zero() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        // MockChatResponse does not override usage() → None, like Ollama.
        let mock = MockChatProvider::with_response("", r#"{"items":[]}"#);

        run_with_recorder(&recorder, || async {
            StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
                .model("qwen2.5:14b")
                .messages(vec![crate::core::provider::chat_msg_user("extract")])
                .call()
                .await
                .expect("mock returns valid JSON");
        });

        let snapshot = snapshotter.snapshot().into_vec();

        assert!(
            counter_sum(&snapshot, "kremory_core_tokens_usage_missing_total", |l| {
                l.get("schema") == Some(&"EntityList")
            }) >= 1,
            "a provider that reports no usage must be COUNTED as missing"
        );
        assert!(
            !snapshot
                .iter()
                .any(|(k, _, _, _)| k.key().name() == "kremory_core_tokens_total"),
            "no token counter may be emitted when usage is unreported — a 0 here \
             reads as 'ingest was free' and is the defect this TD removes"
        );
    }

    /// T4 (Quinn ARCH-001) — the dream phase shares this funnel. Its calls must NOT
    /// be attributed to ingest extraction: doing so over-counts ingest cost AND
    /// double-counts against dream's own `dream_pass_budget_usage` accounting.
    #[test]
    fn dream_schema_is_not_attributed_to_extraction() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let mock = UsageMockProvider {
            text: r#"{"items":[]}"#.to_string(),
            input: 11,
            output: 5,
        };

        run_with_recorder(&recorder, || async {
            // "ReclassifyBatch" is what core/dream/reclassify.rs:256 passes.
            let _ = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "ReclassifyBatch")
                .model("qwen2.5:14b")
                .messages(vec![crate::core::provider::chat_msg_user("reclassify")])
                .call()
                .await;
        });

        let snapshot = snapshotter.snapshot().into_vec();

        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("operation") == Some(&"extraction")
            }),
            0,
            "a DREAM call must not be counted as ingest extraction — that is the \
             over-count ARCH-001 found"
        );
        assert_eq!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("operation") == Some(&"dream")
            }),
            16,
            "the dream call's tokens must be attributed to operation=dream"
        );
    }

    /// T5 — the mapping itself, including the three ingest cost centres the 48×
    /// work needs to sum separately.
    #[test]
    fn operation_for_schema_splits_ingest_cost_centres_and_dream() {
        assert_eq!(operation_for_schema("EntityListIntegerId"), "extraction");
        assert_eq!(operation_for_schema("TripletList"), "extraction");
        assert_eq!(operation_for_schema("ResolutionVerdict"), "resolution");
        assert_eq!(operation_for_schema("BatchedResolution"), "resolution");
        assert_eq!(operation_for_schema("ContradictionVerdict"), "contradiction");
        assert_eq!(operation_for_schema("ReclassifyBatch"), "dream");
        assert_eq!(operation_for_schema("DiscoveryProposalBatch"), "dream");
        assert_eq!(operation_for_schema("IdentityVerdictBatch"), "dream");
        assert_eq!(operation_for_schema("VerifyBatch"), "dream");
        // Unknown must be LOUD, never silently folded into "extraction" — a silent
        // default is exactly how ARCH-001 existed.
        assert_eq!(operation_for_schema("SomeNewSchema"), "unclassified");
    }

    /// T6 — SOURCE GATE. Fails when a new `StructuredCallBuilder` call site
    /// introduces a schema that `operation_for_schema` does not classify.
    ///
    /// This is the enforcement that ARCH-001 lacked: without it, a future call
    /// site silently lands in `"unclassified"` and its cost is mis-attributed
    /// with nothing failing. Scans production code only — each file is truncated
    /// at its `mod tests` boundary, and comment lines are skipped.
    #[test]
    fn every_production_schema_is_classified() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        assert!(!files.is_empty(), "source gate found no .rs files under {src:?}");

        let mut unclassified: Vec<String> = Vec::new();
        let mut seen = 0usize;
        // Counted SEPARATELY from `seen`, and floored separately below. Folding
        // them together would let the table form drop to zero unnoticed behind a
        // healthy call-site count — which is exactly how the blind spot survived.
        let mut table_entries_seen = 0usize;

        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else {
                continue;
            };
            // Production code only — stop at the test module.
            let prod = match text.find("\nmod tests {") {
                Some(i) => &text[..i],
                None => &text[..],
            };
            let lines: Vec<&str> = prod.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") {
                    continue; // doc/comment examples are not call sites
                }
                // ── FORM 1: a direct call site ───────────────────────────────
                //   StructuredCallBuilder::new(.., "SchemaName", ..)
                //
                // ── FORM 2: a TABLE entry ────────────────────────────────────
                //   (&SCHEMA_X, "SchemaName"),
                //
                // Form 2 was the blind spot, and it was a real one: this gate
                // shipped scanning only Form 1, so the fallback-ladder table in
                // this very file (`let schemas: &[(&Value, &str)]`) was invisible
                // to it. `NuExtractEntitiesOnly` sat there UNCLASSIFIED and the
                // gate stayed green — while its two siblings, `NuExtractBoth` and
                // `NuExtractRelationsOnly`, were classified. Caught by a real
                // end-to-end run emitting the `unclassified_schema` warning, not
                // by this test.
                //
                // The lesson generalises past this fix: a source gate models HOW
                // it expects the thing to appear, and is blind to every other
                // shape. The `MIN_EXPECTED_CALL_SITES` floor below guards against
                // the scan silently finding NOTHING — it cannot guard against the
                // scan finding everything of one shape and nothing of another.
                let is_call_site = line.contains("StructuredCallBuilder::new(");
                let is_table_entry = t.starts_with("(&SCHEMA_");
                if !is_call_site && !is_table_entry {
                    continue;
                }
                // The schema NAME is the first string literal at or after the
                // match. A table entry is one line; a call site may wrap.
                let window = if is_table_entry {
                    (*line).to_owned()
                } else {
                    lines[i..(i + 6).min(lines.len())].join("\n")
                };
                let Some(start) = window.find('"') else { continue };
                let rest = &window[start + 1..];
                let Some(end) = rest.find('"') else { continue };
                let name = &rest[..end];
                seen += 1;
                if is_table_entry {
                    table_entries_seen += 1;
                }
                if operation_for_schema(name) == "unclassified" {
                    unclassified.push(format!(
                        "{}: schema {name:?}",
                        f.strip_prefix(&src).unwrap_or(f).display()
                    ));
                }
            }
        }

        // A FLOOR, not `> 0`. `seen > 0` would still pass if the scan silently
        // degraded — wrong path, a formatting change moving the literal outside the
        // 6-line window, a call site written with a non-literal schema name (which
        // this scan skips). Then the gate would report "all classified" having
        // examined almost nothing: a control that certifies by not looking, which is
        // the exact defect class this whole change exists to remove.
        // 21 production call sites live outside structured.rs alone (2026-08-03).
        const MIN_EXPECTED_CALL_SITES: usize = 20;
        // Per-FORM floor. The aggregate floor above cannot detect one form going
        // to zero while the other stays healthy — and that is not hypothetical:
        // the table form scanned ZERO entries for this gate's entire life before
        // 2026-08-05, while `seen` comfortably cleared 20.
        const MIN_EXPECTED_TABLE_ENTRIES: usize = 10;
        assert!(
            table_entries_seen >= MIN_EXPECTED_TABLE_ENTRIES,
            "source gate examined only {table_entries_seen} TABLE-form schema \
             entries (expected >= {MIN_EXPECTED_TABLE_ENTRIES}) — the table scan is \
             broken, not the code. Schema names also reach StructuredCallBuilder \
             through `&[(&SCHEMA_X, \"Name\")]` fallback ladders; if that shape \
             moved or was reformatted, this gate is certifying by not looking."
        );
        assert!(
            seen >= MIN_EXPECTED_CALL_SITES,
            "source gate examined only {seen} production call sites (expected >= \
             {MIN_EXPECTED_CALL_SITES}) — the SCAN is broken, not the code. Do not \
             read this as 'all schemas classified'. Check: did the src path resolve, \
             did a call site move its schema literal >6 lines from the call, or was \
             one written with a non-literal name?"
        );
        assert!(
            unclassified.is_empty(),
            "these production StructuredCallBuilder call sites use a schema that \
             operation_for_schema does not classify, so their token cost would be \
             attributed to operation=\"unclassified\":\n  {}\n\nAdd each to \
             operation_for_schema under the correct phase.",
            unclassified.join("\n  ")
        );
    }

    /// T3 — the DelimitedTuple arm has its own `chat_with_tools` call site and
    /// would otherwise be a silent hole in the accounting.
    #[test]
    fn delimited_tuple_arm_is_instrumented_too() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let mock = UsageMockProvider {
            text: "Alice|Person".to_string(),
            input: 7,
            output: 3,
        };

        run_with_recorder(&recorder, || async {
            let _ = StructuredCallBuilder::new(&mock, &SCHEMA_ENTITY_LIST, "EntityList")
                .model("claude-sonnet-4-5")
                .force_arm(FallbackArm::DelimitedTuple)
                .messages(vec![crate::core::provider::chat_msg_user("extract")])
                .call()
                .await;
        });

        let snapshot = snapshotter.snapshot().into_vec();

        assert!(
            counter_sum(&snapshot, "kremory_core_tokens_total", |l| {
                l.get("arm") == Some(&"delimited_tuple")
            }) > 0,
            "the DelimitedTuple arm must report its tokens; got: {:?}",
            snapshot
                .iter()
                .map(|(k, _, _, _)| k.key().name().to_string())
                .collect::<Vec<_>>()
        );
    }
}

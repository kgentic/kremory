//! Dream Pass 4 — consistency_check (ADR-047).
//!
//! Hybrid embed-prefilter + LLM-verify for detecting high-confidence wrong entity
//! types.  Scope: entities with `entity_type_source NOT IN ('ConsumerPinned','DreamPass4')`
//! and `entity_type_id != 0` (catch-all excluded).
//!
//! ## Observability (CLAUDE.md Rule 19)
//! - `kremory.dream.consistency_check.scanned_total`
//! - `kremory.dream.consistency_check.flagged_total`
//! - `kremory.dream.consistency_check.cap_overflow_total{drop_count}`
//! - `kremory.dream.consistency_check.verify_confirmed_total{source=explicit|downgrade_low_conf}`
//! - `kremory.dream.consistency_check.verify_corrected_total{from_type,to_type}`
//! - `kremory.dream.consistency_check.verify_uncertain_total`
//! - `kremory.dream.consistency_check.verify_low_confidence_downgrade_total{from_type,proposed_to_type}`
//! - `kremory.dream.consistency_check.verify_correction_rejected_total{reason=registry_rejected|catch_all_target}`
//! - `kremory.dream.consistency_check.dry_run_skipped_total{path=dream_phase|stage2_pre_write}`
//! - `kremory.dream.consistency_check.embed_cosine_histogram`
//! - `kremory.dream.consistency_check.llm_call_latency_ms_histogram`
//! - `kremory.dream.consistency_check.verify_model_used{model_name,provider}`

mod audit;
mod verify;

use std::time::Instant;

use metrics::{counter, histogram};
use serde::Deserialize;
use uuid::Uuid;

use crate::core::{error::Result, provider::ChatProvider};

// ─── Confidence gate (T2.2 — basket #83 gbrain C1 pattern) ────────────────────

/// Minimum LLM-reported confidence required to accept a `correct` decision.
///
/// Per sprint plan T2.2 + basket item #83 (gbrain C1 confidence gate): any
/// correction with confidence strictly below this threshold is downgraded to
/// `Confirm` (keep the current type). The gbrain semantics of "downgrade to
/// no_contradiction" map cleanly to kremory's `Confirm` — the LLM signalled it
/// is not sure the correction is right, so the safest action is to keep what's
/// already there rather than risk a wrong-direction correction.
///
/// Threshold value 0.7 chosen per the gbrain pattern. Future ADR may revisit
/// (e.g. per-model calibration, multi-tier gates).
pub(crate) const MIN_VERIFY_CONFIDENCE: f32 = 0.7;

// ─── Public opts / summary ────────────────────────────────────────────────────

/// Tuning knobs for [`run_consistency_check`].
#[derive(Debug, Clone)]
pub struct ConsistencyCheckOpts {
    /// Cosine threshold; flag when `cos < τ` (strict less-than). Default: 0.6.
    pub embed_prefilter_threshold: f32,
    /// Hard cap on candidates sent to LLM per run (RISK-003). Default: `Some(50)`.
    pub max_candidates_per_run: Option<usize>,
    /// Override LLM model string. `None` = use `llm.model()`.
    pub verify_model_override: Option<String>,
    /// Preview mode — when `true`, the verify pass runs through the LLM call and
    /// returns the would-be decisions in `VerifyBatchOutcome` but SKIPS the
    /// downstream `apply_correction` UPDATE on `entities` and SKIPS the
    /// `dream_pass4_audit` row insert. Default: `false`.
    ///
    /// Per sprint plan T1.3 (basket item #194 — gbrain `dryRun: bool` on
    /// maintenance operations). Operators + tests use this to preview Pass 4
    /// corrections before committing irreversible entity-type rewrites.
    pub dry_run: bool,
}

impl Default for ConsistencyCheckOpts {
    fn default() -> Self {
        Self {
            embed_prefilter_threshold: 0.6,
            max_candidates_per_run: Some(50),
            verify_model_override: None,
            dry_run: false,
        }
    }
}

/// Summary returned by [`run_consistency_check`].
#[derive(Debug, Default)]
pub struct ConsistencyCheckSummary {
    pub scanned: usize,
    pub flagged: usize,
    pub confirmed: usize,
    pub corrected: usize,
    pub uncertain: usize,
    /// Entities dropped by the cap guard (RISK-003).
    pub cap_overflow_dropped: usize,
    /// p50 LLM call latency in ms.
    pub latency_ms_p50: u64,
    /// p95 LLM call latency in ms.
    pub latency_ms_p95: u64,
}

// ─── LLM output schema ────────────────────────────────────────────────────────
//
// Required fields have NO `#[serde(default)]` per [[llm-output-parse-loudly]].
// `action=correct` structurally REQUIRES `new_type_id` — enforced via custom
// Deserialize (Vera SCOPE-001 fold from ADR-047 amendments).

/// A single verify decision from the LLM.
/// `action=correct` requires `new_type_id` (enforced at deserialize time).
#[derive(Debug)]
pub struct VerifyDecision {
    pub entity_id: i64,
    pub action: String,
    pub new_type_id: Option<i64>,
    pub confidence: f32,
}

impl<'de> Deserialize<'de> for VerifyDecision {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            entity_id: i64,
            action: String,
            new_type_id: Option<i64>,
            confidence: f32,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.action == "correct" && raw.new_type_id.is_none() {
            return Err(serde::de::Error::custom(
                "action='correct' requires new_type_id to be present",
            ));
        }
        Ok(Self {
            entity_id: raw.entity_id,
            action: raw.action,
            new_type_id: raw.new_type_id,
            confidence: raw.confidence,
        })
    }
}

/// Batch wrapper; `#[serde(default)]` on Vec is acceptable (empty = no decisions).
#[derive(Debug, Deserialize)]
pub(crate) struct VerifyBatch {
    #[serde(default)]
    pub(crate) decisions: Vec<serde_json::Value>,
}

/// Hand-crafted JSON Schema for `VerifyBatch` compatible with Anthropic's
/// `NativeSchema` arm.
///
/// Anthropic structured-output constraints (verified empirically 2026-06-10 via
/// `curl -d '{...,"maxItems":1}' https://api.anthropic.com/v1/messages` →
/// `{"error":{"type":"invalid_request_error","message":"output_config.format.schema:
/// For 'array' type, property 'maxItems' is not supported"}}`):
///
/// - REQUIRED: `additionalProperties: false` on every `object` node. Without
///   this, Anthropic returns 400 → ladder falls to LlmJsonRepair which does
///   NOT enforce schema → LLM free to emit wrong field names. (Phase D iter 1.)
/// - FORBIDDEN: `minItems` and `maxItems` on `array` types. (Phase D iter 3 finding —
///   my earlier addition of these BROKE NativeSchema for ~24h until curl-tested.)
/// - FORBIDDEN: `minimum`, `maximum` on `number` types.
/// - The `decision_count` parameter is retained in the signature for caller
///   compatibility but is UNUSED — array-length enforcement must come from
///   the prompt + caller-side validation, not the schema (Anthropic limitation).
///
/// Per [[load-bearing-invariants-at-emit-not-prompt]] the load-bearing
/// invariant (one-decision-per-entity) is structurally enforced by:
/// 1. `additionalProperties: false` rejects extra fields → enforces field-name correctness
/// 2. `required: ["entity_id", "action", "confidence"]` rejects missing required fields
/// 3. `enum` on `action` rejects values outside {confirm, correct, uncertain}
/// 4. Prompt instruction "MUST output exactly one decision per entity"
/// 5. Caller-side post-call check that `summary.scanned == flagged.len()`
#[doc(hidden)]
pub fn verify_batch_schema(_decision_count: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["entity_id", "action", "confidence"],
                    "properties": {
                        "entity_id": { "type": "integer" },
                        "action": {
                            "type": "string",
                            "enum": ["confirm", "correct", "uncertain"]
                        },
                        "new_type_id": { "type": ["integer", "null"] },
                        "confidence": { "type": "number" }
                    }
                }
            }
        }
    })
}

// ─── Pure functions ───────────────────────────────────────────────────────────

/// Deterministic embed input: `name | f0 | f1 | f2` (padded, top-3 only).
pub fn embed_input_formatter(name: &str, facts: &[String]) -> String {
    let mut parts = vec![name.to_string()];
    for i in 0..3 {
        parts.push(facts.get(i).cloned().unwrap_or_default());
    }
    parts.join(" | ")
}

/// `true` when entity should be flagged: `cos < τ` (exclusive boundary, ADR-047 §D1).
pub fn embed_prefilter_gate(tau: f32, cos: f32) -> bool {
    cos < tau
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ─── Candidate row ────────────────────────────────────────────────────────────

/// Internal candidate row for verify batch operations.
///
/// `pub` so the `pub(crate)` functions `verify_batch` and `build_verify_messages`
/// can use it in their signatures without triggering `private_interfaces`. Also
/// exposed via `test-utils` feature for integration-test GAP-002 DoD checks.
/// Not part of the public semver API — treat as sealed.
#[doc(hidden)]
pub struct CandidateRow {
    pub rowid: i64,
    pub name: String,
    pub entity_type_id: i64,
    pub top3_facts: Vec<String>,
    /// Source-episode text (the most recent episode that mentioned this entity).
    /// Phase D iter 4 (2026-06-10) addition per ADR-047 amendment.
    pub source_episode: Option<String>,
}

// ─── Core verify batch ────────────────────────────────────────────────────────

/// Counts from one `verify_batch` invocation. `#[doc(hidden)]` — internal API.
#[doc(hidden)]
pub struct VerifyBatchCounts {
    pub confirmed: usize,
    pub corrected: usize,
    pub uncertain: usize,
}

/// The action taken for a single candidate in a `verify_batch` call.
/// `#[doc(hidden)]` — internal API, not part of the public semver contract.
#[doc(hidden)]
pub enum VerifyAction {
    /// LLM confirmed the current type is correct.
    Confirm,
    /// LLM corrected the type — `new_type_id` holds the replacement.
    Correct,
    /// LLM was uncertain / candidate missing from response — demote to catch-all.
    ///
    /// Per C6 spec §10.4: missing LLM decisions → Demote (not Confirm).
    /// This matches DK3 ratified direction (Option α: strict safety default).
    Demote,
}

/// Per-entity decision returned by `verify_batch` alongside the counts.
/// Carries enough information to derive `ResolvedDecision` without a DB
/// round-trip — critical for the Stage 2 pre-write flow where entities are
/// NOT yet in the DB when `verify_batch_for_candidates` is called.
/// `#[doc(hidden)]` — internal API.
#[doc(hidden)]
pub struct VerifyBatchDecision {
    /// Index into the `params.flagged` slice (NOT a DB rowid).
    pub candidate_idx: usize,
    /// The action the LLM decided (or Demote for missing/parse-error responses).
    pub action: VerifyAction,
    /// Populated only when `action == Correct`.
    pub new_type_id: Option<i64>,
}

/// Combined outcome of `verify_batch`: aggregate counts + per-entity decisions.
///
/// `decisions` has exactly `params.flagged.len()` entries — one per input
/// candidate. Missing LLM responses default to `VerifyAction::Demote` per
/// C6 spec §10.4.
/// `#[doc(hidden)]` — internal API.
#[doc(hidden)]
pub struct VerifyBatchOutcome {
    pub counts: VerifyBatchCounts,
    pub decisions: Vec<VerifyBatchDecision>,
}

/// Bundled parameters for [`verify_batch`] (keeps arg count ≤ 5 per clippy).
/// `#[doc(hidden)]` — internal API.
#[doc(hidden)]
pub struct VerifyBatchParams<'a> {
    /// Candidate rows to verify (caller owns the subset selection).
    pub flagged: &'a [&'a CandidateRow],
    /// Entity type id → (name, description), loaded by caller.
    pub type_map: &'a std::collections::HashMap<i64, (String, String)>,
    /// Chat provider for the structured verify call.
    pub llm: &'a dyn ChatProvider,
    /// Model string passed to `StructuredCallBuilder`.
    pub verify_model: &'a str,
    /// UUID string for the audit trail (`dream_pass4_audit.run_id`).
    pub run_id: &'a str,
    /// Preview mode — skip the entity-type UPDATE and `dream_pass4_audit` insert.
    /// Wired from [`ConsistencyCheckOpts::dry_run`]. Per sprint plan T1.3.
    pub dry_run: bool,
}

/// Run the LLM verify call on `params.flagged` candidates and apply corrections to the DB.
///
/// Promoted to `pub` (with `#[doc(hidden)]`) per C6 spec §5.2 — `verify_stage.rs`
/// MUST invoke this directly rather than routing through `run_consistency_check`
/// to avoid spurious embed-prefilter exclusions at Stage 2 scope.
///
/// ## Return contract (C6 spec §10.4 — SCOPE-001)
///
/// Returns `VerifyBatchOutcome` containing both aggregate `counts` and a
/// `decisions` vec with exactly `params.flagged.len()` entries — one per input
/// candidate, ordered by input index.
///
/// Missing LLM decisions (M of N returned) default to `VerifyAction::Demote`
/// per spec §10.4 DK3 ratified direction (Option α: strict safety default).
/// A `correct` whose `new_type_id` is not a live `entity_types` row (or is the
/// catch-all `0`) Demotes identically — see `verify.rs`'s registry-bounds guard.
/// This ensures `verify_batch_for_candidates` can derive `ResolvedDecision`
/// from the outcome without a DB round-trip — critical for the Stage 2
/// pre-write flow (entities NOT yet in DB when this function returns).
///
/// ## Why `unreachable!` for action=correct without new_type_id (SCOPE-001)
///
/// The custom `VerifyDecision` `Deserialize` impl rejects `action=correct`
/// entries that lack `new_type_id` at parse time (returns an Err). Any
/// `VerifyDecision` that reaches the match arm where `action == "correct"` is
/// therefore guaranteed to have `new_type_id.is_some()`. The `unreachable!`
/// is not a fallback — it is a structural invariant: if the deserializer lets
/// through a correct-without-new_type_id, it is a bug in the deserializer,
/// not a recoverable runtime case. Panicking loudly is correct here.
///
/// ## Why WARN not DEBUG for malformed decisions
///
/// Malformed decisions indicate the LLM emitted a structurally invalid
/// response. This is observable signal for prompt/schema tuning — DEBUG would
/// hide it by default in production logs. WARN surfaces it without being
/// ERROR-level (single malformed entry does not fail the batch). This log
/// level was escalated from DEBUG during Phase D iter 3 debugging when
/// silent skips made precision regressions invisible.
#[doc(hidden)]
pub async fn verify_batch(
    db: &libsql::Connection,
    params: VerifyBatchParams<'_>,
) -> Result<VerifyBatchOutcome> {
    verify::verify_batch(db, params).await
}

// ─── Main entry point ─────────────────────────────────────────────────────────

// `RunConsistencyCheckParams` is defined in `audit.rs` (re-exported below) to
// keep this module under the TD-C 500-LoC guard (ADR-050 §5).

/// Run Dream Pass 4 consistency check (ADR-047).
pub async fn run_consistency_check(
    db: &libsql::Connection,
    params: RunConsistencyCheckParams<'_>,
) -> Result<ConsistencyCheckSummary> {
    let RunConsistencyCheckParams {
        embedder,
        llm,
        opts,
    } = params;
    let mut summary = ConsistencyCheckSummary::default();
    let tau = opts.embed_prefilter_threshold;
    let run_id = Uuid::new_v4().to_string();
    // Option-1 (2026-06-23): no longer read `llm.model()`. The model is the
    // consumer-supplied `verify_model_override` (set by the dream entry point
    // from the Engine model, or by the consumer directly). Empty → `PromptOnly`.
    let verify_model = opts.verify_model_override.clone().unwrap_or_default();

    let type_map = audit::load_type_registry(db).await?;
    let candidates = audit::load_candidates(db).await?;
    summary.scanned = candidates.len();
    counter!("kremory.dream.consistency_check.scanned_total").increment(candidates.len() as u64);

    if candidates.is_empty() {
        return Ok(summary);
    }

    // Embed-prefilter
    let mut flagged: Vec<&CandidateRow> = Vec::new();
    for candidate in &candidates {
        let embed_input = embed_input_formatter(&candidate.name, &candidate.top3_facts);
        let entity_vec = embedder.embed_dyn(&embed_input).await?;
        let cos = if let Some((_name, type_desc)) = type_map.get(&candidate.entity_type_id) {
            let type_vec = embedder.embed_dyn(type_desc).await?;
            cosine_similarity(&entity_vec, &type_vec)
        } else {
            0.0
        };
        histogram!("kremory.dream.consistency_check.embed_cosine_histogram").record(cos as f64);
        if embed_prefilter_gate(tau, cos) {
            flagged.push(candidate);
        }
    }

    // Cap guard (RISK-003)
    if let Some(cap) = opts.max_candidates_per_run {
        if flagged.len() > cap {
            let dropped = flagged.len() - cap;
            flagged.truncate(cap);
            summary.cap_overflow_dropped = dropped;
            counter!(
                "kremory.dream.consistency_check.cap_overflow_total",
                "drop_count" => dropped.to_string()
            )
            .increment(1);
        }
    }

    summary.flagged = flagged.len();
    counter!("kremory.dream.consistency_check.flagged_total").increment(flagged.len() as u64);

    if flagged.is_empty() {
        return Ok(summary);
    }

    // Derive provider label from model string for correct attribution
    // (prev hardcoded "ollama" was wrong for Anthropic verify provider — Rule 19).
    let provider_label = if verify_model.starts_with("claude-") {
        "anthropic"
    } else if verify_model.starts_with("gpt-")
        || verify_model.starts_with("o1-")
        || verify_model.starts_with("o3-")
    {
        "openai"
    } else {
        "ollama"
    };
    counter!(
        "kremory.dream.consistency_check.verify_model_used",
        "model_name" => verify_model.clone(),
        "provider" => provider_label
    )
    .increment(1);

    let call_start = Instant::now();
    let outcome = verify::verify_batch(
        db,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm,
            verify_model: &verify_model,
            run_id: &run_id,
            dry_run: opts.dry_run,
        },
    )
    .await?;
    let elapsed_ms = call_start.elapsed().as_millis() as u64;
    summary.latency_ms_p50 = elapsed_ms;
    summary.latency_ms_p95 = elapsed_ms;

    summary.confirmed = outcome.counts.confirmed;
    summary.corrected = outcome.counts.corrected;
    summary.uncertain = outcome.counts.uncertain;

    Ok(summary)
}

// ─── verify_batch_for_candidates (GAP-003) — lives in verify.rs ─────────────

pub use audit::RunConsistencyCheckParams;
pub use verify::{
    verify_batch_for_candidates, VerifyBatchForCandidatesOpts, VerifyBatchForCandidatesParams,
    VerifyBatchForCandidatesResult,
};

/// Build the LLM messages for a verify batch (ADR-047 §3).
///
/// The user message MUST include the full type registry (id → name) so the LLM
/// can assign valid `new_type_id` values when action=correct.  Without this menu
/// the LLM has no grounding for integer IDs and will produce corrections that
/// don't match any row in `entity_types`, yielding zero precision lift even when
/// the action decisions are semantically correct.
#[doc(hidden)]
pub fn build_verify_messages(
    candidates: &[&CandidateRow],
    type_map: &std::collections::HashMap<i64, (String, String)>,
) -> Vec<crate::core::provider::ChatMessage> {
    audit::build_verify_messages(candidates, type_map)
}

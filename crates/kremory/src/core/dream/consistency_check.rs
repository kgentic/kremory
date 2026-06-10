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
//! - `kremory.dream.consistency_check.verify_confirmed_total`
//! - `kremory.dream.consistency_check.verify_corrected_total{from_type,to_type}`
//! - `kremory.dream.consistency_check.verify_uncertain_total`
//! - `kremory.dream.consistency_check.embed_cosine_histogram`
//! - `kremory.dream.consistency_check.llm_call_latency_ms_histogram`
//! - `kremory.dream.consistency_check.verify_model_used{model_name,provider}`

use std::time::Instant;

use chrono::Utc;
use metrics::{counter, histogram};
use serde::Deserialize;
use uuid::Uuid;

use crate::core::{
    error::{Error, Result},
    provider::{chat_msg_system, chat_msg_user, ChatProvider, DynEmbeddingProvider},
};

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
}

impl Default for ConsistencyCheckOpts {
    fn default() -> Self {
        Self {
            embed_prefilter_threshold: 0.6,
            max_candidates_per_run: Some(50),
            verify_model_override: None,
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
struct VerifyBatch {
    #[serde(default)]
    decisions: Vec<serde_json::Value>,
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    let flagged = params.flagged;
    let type_map = params.type_map;
    let llm = params.llm;
    let verify_model = params.verify_model;
    let run_id = params.run_id;
    let mut counts = VerifyBatchCounts {
        confirmed: 0,
        corrected: 0,
        uncertain: 0,
    };

    // Pre-allocate decisions vec with one Demote per candidate.
    // Per C6 spec §10.4: missing decisions → Demote (not Confirm).
    // Entries are overwritten below when the LLM provides a decision for that rowid.
    let mut decisions: Vec<VerifyBatchDecision> = flagged
        .iter()
        .enumerate()
        .map(|(idx, _)| VerifyBatchDecision {
            candidate_idx: idx,
            action: VerifyAction::Demote,
            new_type_id: None,
        })
        .collect();

    if flagged.is_empty() {
        return Ok(VerifyBatchOutcome { counts, decisions });
    }

    let messages = build_verify_messages(flagged, type_map);
    let call_start = Instant::now();
    let schema = verify_batch_schema(flagged.len());
    let raw_value = crate::core::extraction::structured::StructuredCallBuilder::new(
        llm,
        &schema,
        "VerifyBatch",
    )
    .model(verify_model)
    .messages(messages)
    .call()
    .await;
    let elapsed_ms = call_start.elapsed().as_millis() as u64;
    histogram!("kremory.dream.consistency_check.llm_call_latency_ms_histogram")
        .record(elapsed_ms as f64);

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.dream.consistency_check.raw_response",
            verify_model = %verify_model,
            response = ?raw_value,
            "verify_batch raw response"
        );
    }

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::dream::consistency_check",
                error = %e,
                "verify_batch LLM call failed — all candidates default to Demote"
            );
            // All decisions remain Demote (pre-allocated above). Count as uncertain.
            counts.uncertain = flagged.len();
            return Ok(VerifyBatchOutcome { counts, decisions });
        }
    };

    // Parse decisions
    let batch: VerifyBatch = {
        match serde_json::from_value(raw_value.clone()) {
            Ok(b) => b,
            Err(_) => {
                let repaired = if raw_value.is_array() {
                    serde_json::json!({ "decisions": raw_value })
                } else {
                    raw_value.clone()
                };
                match serde_json::from_value::<VerifyBatch>(repaired) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!(target: "kremory::dream::consistency_check",
                            error = %e, "verify_batch failed to parse LLM batch — all candidates default to Demote");
                        counts.uncertain = flagged.len();
                        return Ok(VerifyBatchOutcome { counts, decisions });
                    }
                }
            }
        }
    };

    // Build a rowid → candidate_idx map so we can overwrite the pre-allocated Demote entries.
    let rowid_to_idx: std::collections::HashMap<i64, usize> =
        flagged.iter().enumerate().map(|(idx, c)| (c.rowid, idx)).collect();
    let now = Utc::now().to_rfc3339();

    for raw_decision in &batch.decisions {
        let decision: VerifyDecision = match serde_json::from_value(raw_decision.clone()) {
            Ok(d) => d,
            Err(e) => {
                // See fn-level doc: WARN not DEBUG — malformed entries are tuning signal.
                tracing::warn!(target: "kremory::dream::consistency_check",
                    error = %e, decision = %raw_decision, "skipping malformed decision");
                continue;
            }
        };

        let Some(&candidate_idx) = rowid_to_idx.get(&decision.entity_id) else {
            tracing::debug!(target: "kremory::dream::consistency_check",
                entity_id = decision.entity_id, "decision for unknown rowid — skipped");
            continue;
        };
        let candidate = flagged[candidate_idx];

        match decision.action.as_str() {
            "confirm" => {
                decisions[candidate_idx] = VerifyBatchDecision {
                    candidate_idx,
                    action: VerifyAction::Confirm,
                    new_type_id: None,
                };
                counts.confirmed += 1;
                counter!("kremory.dream.consistency_check.verify_confirmed_total").increment(1);
            }
            "correct" => {
                // See fn-level doc: unreachable! enforces SCOPE-001 structural invariant.
                let new_type_id = decision.new_type_id.unwrap_or_else(|| {
                    unreachable!("action=correct without new_type_id should be rejected at parse (SCOPE-001)")
                });
                // For dream-phase flow (entities already in DB), apply the correction now.
                // For Stage 2 pre-write flow (entities NOT in DB), rowid == -1 so
                // apply_correction UPDATE hits 0 rows — no harm done. The per-entity
                // decision is what matters; write_verified_entities uses new_type_id directly.
                if candidate.rowid > 0 {
                    apply_correction(db, candidate, new_type_id, &now).await?;
                    let audit = AuditRowParams {
                        entity_rowid: candidate.rowid,
                        pre_type_id: candidate.entity_type_id,
                        post_type_id: new_type_id,
                        verify_confidence: decision.confidence,
                        verify_model,
                        run_id,
                    };
                    write_audit_row(db, audit).await?;
                }
                decisions[candidate_idx] = VerifyBatchDecision {
                    candidate_idx,
                    action: VerifyAction::Correct,
                    new_type_id: Some(new_type_id),
                };
                counts.corrected += 1;
                counter!(
                    "kremory.dream.consistency_check.verify_corrected_total",
                    "from_type" => candidate.entity_type_id.to_string(),
                    "to_type" => new_type_id.to_string()
                )
                .increment(1);
            }
            "uncertain" => {
                // LLM said uncertain — keep the pre-allocated Demote for this entry,
                // since uncertain means we cannot confirm the NER type is correct.
                // counts.uncertain tracks the LLM-signalled uncertainty explicitly.
                decisions[candidate_idx] = VerifyBatchDecision {
                    candidate_idx,
                    action: VerifyAction::Demote,
                    new_type_id: None,
                };
                counts.uncertain += 1;
                counter!("kremory.dream.consistency_check.verify_uncertain_total").increment(1);
            }
            other => {
                tracing::debug!(target: "kremory::dream::consistency_check",
                    action = %other, "unknown action — skipped (treated as Demote)");
            }
        }
    }

    Ok(VerifyBatchOutcome { counts, decisions })
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Run Dream Pass 4 consistency check (ADR-047).
pub async fn run_consistency_check(
    db: &libsql::Connection,
    embedder: &dyn DynEmbeddingProvider,
    llm: &dyn ChatProvider,
    opts: ConsistencyCheckOpts,
) -> Result<ConsistencyCheckSummary> {
    let mut summary = ConsistencyCheckSummary::default();
    let tau = opts.embed_prefilter_threshold;
    let run_id = Uuid::new_v4().to_string();
    let verify_model = opts
        .verify_model_override
        .clone()
        .unwrap_or_else(|| llm.model().to_string());

    let type_map = load_type_registry(db).await?;
    let candidates = load_candidates(db).await?;
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
    let outcome = verify_batch(
        db,
        VerifyBatchParams {
            flagged: &flagged,
            type_map: &type_map,
            llm,
            verify_model: &verify_model,
            run_id: &run_id,
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

// ─── verify_batch_for_candidates (GAP-003) ───────────────────────────────────

/// Tuning knobs for [`verify_batch_for_candidates`].
///
/// Mirrors the relevant subset of [`ConsistencyCheckOpts`] without the
/// embed-prefilter fields (which are irrelevant when candidates are supplied
/// directly by the caller).
#[derive(Debug, Clone, Default)]
pub struct VerifyBatchForCandidatesOpts {
    /// Hard cap on the number of candidates sent to LLM. `None` = no cap (uses
    /// all provided candidates). Default: `None`.
    pub max_candidates: Option<usize>,
    /// Override LLM model string. `None` = use `llm.model()`.
    pub verify_model_override: Option<String>,
}

/// Result of [`verify_batch_for_candidates`].
#[derive(Debug)]
pub struct VerifyBatchForCandidatesResult {
    /// One decision per input candidate (same length as `candidates` slice).
    pub decisions: Vec<crate::core::ingest::ResolvedDecision>,
}

/// Run the LLM verify call on the provided `candidates` directly.
///
/// Per C6 spec §5.2 — Stage 2 MUST invoke `verify_batch` directly to avoid
/// spurious embed-prefilter exclusions. This function is the public entry-point
/// for that path: it accepts candidates in hand (from Phase 1 NER) without any
/// DB round-trip to load or filter the candidate set.
///
/// # Invariant
///
/// Returns exactly `candidates.len()` decisions — one per input candidate. If
/// the LLM returns fewer decisions, missing candidates default to `Demote`
/// (catch-all, entity_type_id = 0) per C6 spec §10.4 DK3 ratified direction.
///
/// # DB usage
///
/// `db` is used ONLY for:
/// - Looking up entity rowids (needed for the `verify_batch` LLM prompt format;
///   entities NOT in DB get rowid = -1 sentinel, still included in the prompt)
/// - Writing `dream_pass4_audit` rows for `correct` decisions on entities that
///   ARE already in the DB (dream-phase compat; no-op for rowid = -1 entries)
///
/// The function does NOT perform a `SELECT * FROM entities` to expand/replace
/// the input candidate set. The `candidates` slice is the authoritative input.
/// Decisions are derived DIRECTLY from `VerifyBatchOutcome.decisions` — NOT
/// from a post-verify DB re-query — so this function is correct for both:
/// - Stage 2 ingest-time flow (entities not yet in DB, rowid = -1)
/// - Dream-phase flow (entities already in DB, rowid > 0)
///
/// ## Observability (CLAUDE.md Rule 19)
/// - `kremory.verify_batch_for_candidates.invoked_total`
/// - `kremory.verify_batch_for_candidates.decisions_total{variant}`
pub async fn verify_batch_for_candidates(
    db: &libsql::Connection,
    candidates: &[crate::core::ingest::EntityCandidate],
    source_episode_text: &str,
    llm: &dyn ChatProvider,
    opts: VerifyBatchForCandidatesOpts,
) -> Result<VerifyBatchForCandidatesResult> {
    use crate::core::ingest::ResolvedDecision;
    use crate::core::resolver::normalize_name;

    counter!("kremory.verify_batch_for_candidates.invoked_total").increment(1);

    if candidates.is_empty() {
        return Ok(VerifyBatchForCandidatesResult {
            decisions: Vec::new(),
        });
    }

    let verify_model = opts
        .verify_model_override
        .clone()
        .unwrap_or_else(|| llm.model().to_string());

    // Cap candidates if requested.
    let effective_candidates: &[crate::core::ingest::EntityCandidate] =
        if let Some(cap) = opts.max_candidates {
            &candidates[..cap.min(candidates.len())]
        } else {
            candidates
        };

    // Load entity type registry for the verify prompt.
    let type_map = load_type_registry(db).await?;

    // Build CandidateRow values from the provided EntityCandidate slice.
    // Look up each entity's rowid by name so the LLM prompt uses integer ids
    // (consistent with how run_consistency_check works).
    // For entities not yet in DB (Stage 2 ingest-time flow), rowid = -1 sentinel.
    // verify_batch will include them in the LLM prompt; rowid = -1 entries skip
    // apply_correction and write_audit_row (no DB rows to update yet).
    let mut candidate_rows: Vec<CandidateRow> = Vec::with_capacity(effective_candidates.len());
    for ec in effective_candidates {
        let entity_id = normalize_name(&ec.name);
        let rowid: i64 = {
            let mut rows = db
                .query(
                    "SELECT rowid FROM entities WHERE id = ?1 LIMIT 1",
                    libsql::params![entity_id.clone()],
                )
                .await
                .map_err(|e| Error::Other(anyhow::anyhow!("rowid lookup '{}': {e}", entity_id)))?;
            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| Error::Other(anyhow::anyhow!("rowid row '{}': {e}", entity_id)))?
            {
                row.get(0)
                    .map_err(|e| Error::Other(anyhow::anyhow!("rowid col '{}': {e}", entity_id)))?
            } else {
                // Entity not yet in DB — Stage 2 ingest-time flow.
                // rowid = -1 is a sentinel; verify_batch skips DB writes for these.
                -1i64
            }
        };
        let top3_facts = if rowid > 0 {
            load_top3_facts(db, &entity_id).await.unwrap_or_default()
        } else {
            Vec::new()
        };
        candidate_rows.push(CandidateRow {
            rowid,
            name: ec.name.clone(),
            entity_type_id: ec.entity_type_id_raw,
            top3_facts,
            source_episode: Some(source_episode_text.to_string()),
        });
    }

    // Build &[&CandidateRow] for verify_batch.
    let flagged_refs: Vec<&CandidateRow> = candidate_rows.iter().collect();

    let run_id = uuid::Uuid::new_v4().to_string();
    let outcome = verify_batch(
        db,
        VerifyBatchParams {
            flagged: &flagged_refs,
            type_map: &type_map,
            llm,
            verify_model: &verify_model,
            run_id: &run_id,
        },
    )
    .await?;

    // Derive ResolvedDecision directly from outcome.decisions — NO DB re-query.
    //
    // This is the ARCH-001 fix: the previous implementation re-queried
    // `SELECT entity_type_id FROM entities WHERE id=?` after verify_batch returned,
    // which always returned None for Stage 2 (entities not yet written) and fell
    // through to Confirm for every candidate — rendering the verify gate a no-op.
    //
    // outcome.decisions has exactly flagged_refs.len() entries (pre-allocated with
    // Demote, overwritten for candidates the LLM returned decisions for).
    // Per C6 spec §10.4: missing entries stay Demote (strict safety default).
    let mut decisions: Vec<ResolvedDecision> = Vec::with_capacity(effective_candidates.len());
    for vbd in &outcome.decisions {
        let resolved = match vbd.action {
            VerifyAction::Confirm => {
                counter!(
                    "kremory.verify_batch_for_candidates.decisions_total",
                    "variant" => "confirm"
                )
                .increment(1);
                ResolvedDecision::Confirm {
                    candidate_idx: vbd.candidate_idx,
                }
            }
            VerifyAction::Correct => {
                let new_type_id = vbd.new_type_id.unwrap_or_else(|| {
                    unreachable!("VerifyAction::Correct must always carry new_type_id (SCOPE-001)")
                });
                counter!(
                    "kremory.verify_batch_for_candidates.decisions_total",
                    "variant" => "correct"
                )
                .increment(1);
                ResolvedDecision::Correct {
                    candidate_idx: vbd.candidate_idx,
                    new_type_id,
                }
            }
            VerifyAction::Demote => {
                counter!(
                    "kremory.verify_batch_for_candidates.decisions_total",
                    "variant" => "demote"
                )
                .increment(1);
                ResolvedDecision::Demote {
                    candidate_idx: vbd.candidate_idx,
                }
            }
        };
        decisions.push(resolved);
    }

    Ok(VerifyBatchForCandidatesResult { decisions })
}

// ─── DB helpers ───────────────────────────────────────────────────────────────

/// Load entity type registry: id → (name, description).
///
/// Both name and description are needed:
/// - description for embed-prefilter cosine comparison
/// - name for the LLM verify-prompt type menu (so LLM knows which ID = which type)
async fn load_type_registry(
    db: &libsql::Connection,
) -> Result<std::collections::HashMap<i64, (String, String)>> {
    let mut rows = db
        .query("SELECT id, name, description FROM entity_types", ())
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_type_registry: {e}")))?;
    let mut map = std::collections::HashMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_type_registry row: {e}")))?
    {
        let id: i64 = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_id: {e}")))?;
        let name: String = row
            .get(1)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_name: {e}")))?;
        let desc: String = row
            .get(2)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_desc: {e}")))?;
        map.insert(id, (name, desc));
    }
    Ok(map)
}

async fn load_candidates(db: &libsql::Connection) -> Result<Vec<CandidateRow>> {
    let mut rows = db
        .query(
            "SELECT rowid, id, entity_type_id FROM entities \
             WHERE entity_type_id != 0 \
             AND entity_type_source NOT IN ('ConsumerPinned', 'DreamPass4')",
            (),
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_candidates: {e}")))?;
    let mut candidates = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_candidates row: {e}")))?
    {
        let rowid: i64 = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("rowid: {e}")))?;
        let name: String = row
            .get(1)
            .map_err(|e| Error::Other(anyhow::anyhow!("entity.id: {e}")))?;
        let type_id: i64 = row
            .get(2)
            .map_err(|e| Error::Other(anyhow::anyhow!("entity_type_id: {e}")))?;
        let top3_facts = load_top3_facts(db, &name).await?;
        let source_episode = load_source_episode(db, &name).await?;
        candidates.push(CandidateRow {
            rowid,
            name,
            entity_type_id: type_id,
            top3_facts,
            source_episode,
        });
    }
    Ok(candidates)
}

async fn load_top3_facts(db: &libsql::Connection, entity_id: &str) -> Result<Vec<String>> {
    let mut rows = db
        .query(
            "SELECT object_value FROM facts \
             WHERE subject_id = ?1 AND object_value IS NOT NULL \
             AND expired_at IS NULL ORDER BY recorded_at DESC LIMIT 3",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_top3_facts: {e}")))?;
    let mut facts = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_top3_facts row: {e}")))?
    {
        let val: String = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("fact: {e}")))?;
        facts.push(val);
    }
    Ok(facts)
}

/// Load the source-episode text for the most recent episode that mentioned
/// this entity. Returns `None` if no episodic edge exists.
///
/// Phase D iter 4 (2026-06-10) addition per ADR-047 amendment. The verify call
/// needs source-text context to disambiguate polyseme entities ("Apple emailed
/// me" vs "Apple is a fruit"). Without it the LLM operates on name + thin facts
/// alone and produces ~50% precision on polysemes per RISK-001 iter 3 evidence.
async fn load_source_episode(
    db: &libsql::Connection,
    entity_id: &str,
) -> Result<Option<String>> {
    let mut rows = db
        .query(
            "SELECT e.content FROM episodes e \
             INNER JOIN episodic_edges ee ON ee.episode_id = e.id \
             WHERE ee.entity_id = ?1 \
             ORDER BY e.recorded_at DESC LIMIT 1",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_source_episode: {e}")))?;
    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_source_episode row: {e}")))?
    {
        let content: String = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("episode.content: {e}")))?;
        Ok(Some(content))
    } else {
        Ok(None)
    }
}

async fn apply_correction(
    db: &libsql::Connection,
    candidate: &CandidateRow,
    new_type_id: i64,
    now: &str,
) -> Result<()> {
    db.execute(
        "UPDATE entities SET entity_type_id = ?1, entity_type_source = 'DreamPass4', \
         entity_type_assigned_at = ?2, updated_at = ?2 WHERE rowid = ?3",
        libsql::params![new_type_id, now.to_string(), candidate.rowid],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "apply_correction rowid={}: {e}",
            candidate.rowid
        ))
    })?;
    Ok(())
}

/// Parameters for a single `dream_pass4_audit` row (IRREV-001).
/// Bundled to keep `write_audit_row` under the clippy 5-arg limit.
struct AuditRowParams<'a> {
    entity_rowid: i64,
    pre_type_id: i64,
    post_type_id: i64,
    verify_confidence: f32,
    verify_model: &'a str,
    run_id: &'a str,
}

async fn write_audit_row(db: &libsql::Connection, p: AuditRowParams<'_>) -> Result<()> {
    db.execute(
        "INSERT INTO dream_pass4_audit \
         (entity_id, pre_type_id, post_type_id, verify_confidence, verify_model, run_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        libsql::params![
            p.entity_rowid,
            p.pre_type_id,
            p.post_type_id,
            p.verify_confidence as f64,
            p.verify_model.to_string(),
            p.run_id.to_string()
        ],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "write_audit_row entity_id={}: {e}",
            p.entity_rowid
        ))
    })?;
    Ok(())
}

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
    let system = "You are an entity-type verification assistant. \
        For EACH entity provided below you MUST output exactly one decision. \
        The `decisions` array length MUST equal the number of entities listed. \
        \
        Each decision MUST use these EXACT JSON field names (do NOT use synonyms): \
        - `entity_id` (integer) — copy verbatim from the entity provided \
        - `action` (string) — MUST be one of EXACTLY \"confirm\", \"correct\", or \"uncertain\" \
        - `new_type_id` (integer, REQUIRED when action=correct) — id from the registry below \
        - `confidence` (number 0.0-1.0) — your confidence in this decision \
        \
        Do NOT use the field name `decision` (use `action`). Do NOT use the field name \
        `reason` (use `confidence` as a number). Do NOT include extra fields like \
        `current_type_id`, `name`, or `reason` — they will cause schema rejection. \
        \
        Action semantics: confirm (current type is correct), correct (current type is \
        wrong — you MUST provide new_type_id from the registry), uncertain (insufficient \
        info to decide — prefer this over skipping). \
        \
        Respond with a JSON object matching the VerifyBatch schema exactly.";

    // Build the type registry menu so the LLM knows which ID corresponds to which type.
    let mut registry_lines: Vec<String> = type_map
        .iter()
        .filter(|(&id, _)| id != 0) // exclude catch-all
        .map(|(&id, (name, _desc))| format!("  id={id} name=\"{name}\""))
        .collect();
    registry_lines.sort(); // deterministic order
    let registry_block = registry_lines.join("\n");

    let entity_lines: Vec<String> = candidates
        .iter()
        .map(|c| {
            let (type_name, type_desc) = type_map
                .get(&c.entity_type_id)
                .map(|(n, d)| (n.as_str(), d.as_str()))
                .unwrap_or(("unknown", "unknown"));
            // Truncate source episode to avoid bloat (8000 chars ~= 2000 tokens
            // per entity; cap covers most natural-prose paragraphs).
            let source_excerpt: String = c
                .source_episode
                .as_deref()
                .map(|s| {
                    if s.len() > 8000 {
                        format!("{}…[truncated]", &s[..8000])
                    } else {
                        s.to_string()
                    }
                })
                .unwrap_or_else(|| "[no source episode available]".to_string());
            // Per ADR-047 amendment + arXiv:2605.29168 ontology-grounded post-extraction
            // correction precedent — source-episode text gives the LLM disambiguating
            // context for polysemes ("Apple emailed me" vs "Apple is a fruit").
            format!(
                "entity_id={} name=\"{}\" current_type_id={} current_type_name=\"{}\" \
                 type_description=\"{}\" facts=\"{}\" source_episode=\"\"\"{}\"\"\"",
                c.rowid,
                c.name,
                c.entity_type_id,
                type_name,
                type_desc,
                c.top3_facts.join("; "),
                source_excerpt
            )
        })
        .collect();

    vec![
        chat_msg_system(system),
        chat_msg_user(format!(
            "Available entity types:\n{registry_block}\n\nVerify these entity type assignments:\n{}",
            entity_lines.join("\n")
        )),
    ]
}

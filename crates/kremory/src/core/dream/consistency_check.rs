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

/// Hand-crafted JSON Schema for `VerifyBatch` that is compatible with Anthropic's
/// `NativeSchema` arm (which requires `additionalProperties: false` on every
/// `object` node and rejects `minimum`/`maximum` on `number` types).
///
/// `schemars::schema_for!(VerifyBatch)` does NOT add `additionalProperties: false`
/// by default, causing the Anthropic API to return a 400 error, which makes the
/// fallback ladder step down to PromptOnly → `{}` → empty decisions list.
/// This static schema is the cause-fix (ADR-047 §2, CLAUDE.md Rule 8).
fn verify_batch_schema() -> serde_json::Value {
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
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ─── Candidate row ────────────────────────────────────────────────────────────

struct CandidateRow {
    rowid: i64,
    name: String,
    entity_type_id: i64,
    top3_facts: Vec<String>,
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

    // LLM-verify call
    let messages = build_verify_messages(&flagged, &type_map);
    let call_start = Instant::now();
    // Use the hand-crafted Anthropic-compatible schema (see verify_batch_schema()).
    // schemars::schema_for!(VerifyBatch) does NOT add `additionalProperties: false`
    // which Anthropic NativeSchema arm requires — causing silent 400 → fallback to
    // PromptOnly → {} → empty decisions (root cause of Phase D FAIL, ADR-047 §2).
    let schema = verify_batch_schema();
    let raw_value =
        crate::core::extraction::structured::StructuredCallBuilder::new(llm, &schema, "VerifyBatch")
            .model(&verify_model)
            .messages(messages)
            .call()
            .await;
    let elapsed_ms = call_start.elapsed().as_millis() as u64;
    histogram!("kremory.dream.consistency_check.llm_call_latency_ms_histogram")
        .record(elapsed_ms as f64);
    summary.latency_ms_p50 = elapsed_ms;
    summary.latency_ms_p95 = elapsed_ms;
    // Derive provider label from model string for correct attribution
    // (prev hardcoded "ollama" was wrong for Anthropic verify provider — Rule 19).
    let provider_label = if verify_model.starts_with("claude-") {
        "anthropic"
    } else if verify_model.starts_with("gpt-") || verify_model.starts_with("o1-") || verify_model.starts_with("o3-") {
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

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::dream::consistency_check",
                error = %e,
                "LLM call failed — returning partial summary"
            );
            return Ok(summary);
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
                            error = %e, "failed to parse LLM batch");
                        return Ok(summary);
                    }
                }
            }
        }
    };

    let candidate_by_rowid: std::collections::HashMap<i64, &CandidateRow> =
        flagged.iter().map(|c| (c.rowid, *c)).collect();
    let now = Utc::now().to_rfc3339();

    for raw_decision in &batch.decisions {
        let decision: VerifyDecision = match serde_json::from_value(raw_decision.clone()) {
            Ok(d) => d,
            Err(e) => {
                tracing::debug!(target: "kremory::dream::consistency_check",
                    error = %e, decision = %raw_decision, "skipping malformed decision");
                continue;
            }
        };

        let Some(candidate) = candidate_by_rowid.get(&decision.entity_id) else {
            tracing::debug!(target: "kremory::dream::consistency_check",
                entity_id = decision.entity_id, "decision for unknown rowid — skipped");
            continue;
        };

        match decision.action.as_str() {
            "confirm" => {
                summary.confirmed += 1;
                counter!("kremory.dream.consistency_check.verify_confirmed_total").increment(1);
            }
            "correct" => {
                // SCOPE-001: action=correct structurally requires new_type_id — enforced by
                // the custom Deserialize on VerifyDecision (missing new_type_id is a parse
                // error, not a runtime case). unreachable! documents the invariant without
                // triggering the clippy::expect_used lint.
                let new_type_id = decision.new_type_id.unwrap_or_else(|| {
                    unreachable!("action=correct without new_type_id should be rejected at parse")
                });
                apply_correction(db, candidate, new_type_id, &now).await?;
                let audit = AuditRowParams {
                    entity_rowid: candidate.rowid,
                    pre_type_id: candidate.entity_type_id,
                    post_type_id: new_type_id,
                    verify_confidence: decision.confidence,
                    verify_model: &verify_model,
                    run_id: &run_id,
                };
                write_audit_row(db, audit).await?;
                summary.corrected += 1;
                counter!(
                    "kremory.dream.consistency_check.verify_corrected_total",
                    "from_type" => candidate.entity_type_id.to_string(),
                    "to_type" => new_type_id.to_string()
                )
                .increment(1);
            }
            "uncertain" => {
                summary.uncertain += 1;
                counter!("kremory.dream.consistency_check.verify_uncertain_total").increment(1);
            }
            other => {
                tracing::debug!(target: "kremory::dream::consistency_check",
                    action = %other, "unknown action — skipped");
            }
        }
    }

    Ok(summary)
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
    while let Some(row) = rows.next().await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_type_registry row: {e}")))?
    {
        let id: i64 = row.get(0).map_err(|e| Error::Other(anyhow::anyhow!("type_id: {e}")))?;
        let name: String = row.get(1).map_err(|e| Error::Other(anyhow::anyhow!("type_name: {e}")))?;
        let desc: String = row.get(2).map_err(|e| Error::Other(anyhow::anyhow!("type_desc: {e}")))?;
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
    while let Some(row) = rows.next().await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_candidates row: {e}")))?
    {
        let rowid: i64 = row.get(0).map_err(|e| Error::Other(anyhow::anyhow!("rowid: {e}")))?;
        let name: String = row.get(1).map_err(|e| Error::Other(anyhow::anyhow!("entity.id: {e}")))?;
        let type_id: i64 = row.get(2).map_err(|e| Error::Other(anyhow::anyhow!("entity_type_id: {e}")))?;
        let top3_facts = load_top3_facts(db, &name).await?;
        candidates.push(CandidateRow { rowid, name, entity_type_id: type_id, top3_facts });
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
    while let Some(row) = rows.next().await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_top3_facts row: {e}")))?
    {
        let val: String = row.get(0).map_err(|e| Error::Other(anyhow::anyhow!("fact: {e}")))?;
        facts.push(val);
    }
    Ok(facts)
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
    .map_err(|e| Error::Other(anyhow::anyhow!("apply_correction rowid={}: {e}", candidate.rowid)))?;
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
            p.entity_rowid, p.pre_type_id, p.post_type_id,
            p.verify_confidence as f64, p.verify_model.to_string(), p.run_id.to_string()
        ],
    )
    .await
    .map_err(|e| Error::Other(anyhow::anyhow!("write_audit_row entity_id={}: {e}", p.entity_rowid)))?;
    Ok(())
}

/// Build the LLM messages for a verify batch (ADR-047 §3).
///
/// The user message MUST include the full type registry (id → name) so the LLM
/// can assign valid `new_type_id` values when action=correct.  Without this menu
/// the LLM has no grounding for integer IDs and will produce corrections that
/// don't match any row in `entity_types`, yielding zero precision lift even when
/// the action decisions are semantically correct.
fn build_verify_messages(
    candidates: &[&CandidateRow],
    type_map: &std::collections::HashMap<i64, (String, String)>,
) -> Vec<crate::core::provider::ChatMessage> {
    let system = "You are an entity-type verification assistant. \
        Given entities and their assigned types, decide for each: confirm (type correct), \
        correct (type wrong — provide new_type_id from the registry below), or uncertain \
        (insufficient info). Respond with a JSON object matching the VerifyBatch schema.";

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
            format!(
                "entity_id={} name=\"{}\" current_type_id={} current_type_name=\"{}\" \
                 type_description=\"{}\" facts=\"{}\"",
                c.rowid, c.name, c.entity_type_id, type_name, type_desc,
                c.top3_facts.join("; ")
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

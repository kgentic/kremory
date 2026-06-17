//! Pass 2 reclassify primitive (ADR-046 / ADR-037 §5).
//!
//! ## What this does
//!
//! Queries entities matching two arms (ADR-046 Amendment 2026-06-09 Option E):
//!
//! - **catch_all_cascade** — `entity_type_id = 0` (Entity catch-all, never classified)
//! - **low_confidence**    — `entity_type_source = 'Phase1Ner' AND ner_confidence < threshold`
//!
//! `ConsumerPinned` and `DreamPass1` entities are excluded STRUCTURALLY in the WHERE
//! clause — never touched by prompt engineering.  This satisfies DoD E1 /
//! `[[load-bearing-invariants-at-emit-not-prompt]]`.
//!
//! For each batch the LLM returns a `ReclassifyBatch` (entity_id → entity_type_id
//! mapping).  Results are validated and written back with confidence-aware source-tier
//! stamping (DoD E3):
//!
//! - `confidence >= 0.7` → `entity_type_source = 'DreamPass1'` (high-confidence; re-re-type
//!   protection is structural because DreamPass1 is excluded from next cycle's SELECT).
//! - `confidence < 0.7` (or absent) → UPDATE `entity_type_id` only; existing source-tier
//!   preserved.
//!
//! ## Observability (DoD E9)
//!
//! - `kremory.dream.reclassify_call_duration_ms` histogram
//! - `kremory.dream.reclassify_call_outcome_total{outcome=ok|parse_repair|parse_fail|llm_err}`
//! - `kremory.dream.reclassify_entities_skipped_total{reason=consumer_pinned|dreampass1_excluded}`
//! - `kremory.dream.reclassify_source_tier_written_total{tier=DreamPass1|preserved}`
//! - `kremory.dream.entities_reclassified_total{trigger=catch_all_cascade|low_confidence}`

use std::time::Instant;

use chrono::Utc;
use metrics::{counter, histogram};
use schemars::schema_for;

use crate::core::{
    error::Result,
    extraction::structured::StructuredCallBuilder,
    provider::{chat_msg_system, chat_msg_user, ChatProvider},
    schema::TemporalGraph,
};

// ─── Result type ─────────────────────────────────────────────────────────────

/// Result of a single [`reclassify`] invocation.
#[derive(Debug, Default)]
pub struct ReclassifyResult {
    /// Total entities successfully reclassified (summed across both arms).
    pub entities_reclassified: usize,
    /// Warnings for operator attention.
    pub warnings: Vec<String>,
}

// ─── LLM output structs ───────────────────────────────────────────────────────
//
// Required fields have NO `#[serde(default)]` per [[llm-output-parse-loudly]].
// Missing required field = parse error → fallback ladder retries.
// `confidence` is the sole exception: absent = semantically valid (LLM may omit),
// modelled as `Option<f32>` with `#[serde(default)]`.

/// A single entity reclassification decision from the LLM.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub(crate) struct ReclassifyDecision {
    /// Entity id (== normalised name). Required — no default.
    pub(crate) entity_id: String,
    /// New integer entity type id.  Required — no default.
    pub(crate) entity_type_id: u32,
    /// Confidence score [0.0, 1.0].
    /// `#[serde(default)]` is intentional: absent = None (LLM may omit confidence).
    #[serde(default)]
    pub(crate) confidence: Option<f32>,
}

/// Batch of reclassification decisions.
///
/// `decisions` uses `#[serde(default)]` — empty array is semantically identical
/// to an absent field (no reclassifications this cycle).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub(crate) struct ReclassifyBatch {
    #[serde(default)]
    pub(crate) decisions: Vec<ReclassifyDecision>,
}

// ─── Candidate entity ─────────────────────────────────────────────────────────

/// An entity candidate fetched from the two-arm SELECT.
#[derive(Debug)]
struct Candidate {
    id: String,
    entity_type_id: u32,
    ner_confidence: Option<f64>,
    entity_type_source: String,
    /// The SELECT arm that surfaced this entity (used for per-trigger counter).
    trigger: CandidateTrigger,
}

#[derive(Debug, Clone, Copy)]
enum CandidateTrigger {
    CatchAllCascade,
    LowConfidence,
}

impl CandidateTrigger {
    fn as_str(self) -> &'static str {
        match self {
            CandidateTrigger::CatchAllCascade => "catch_all_cascade",
            CandidateTrigger::LowConfidence => "low_confidence",
        }
    }
}

// ─── Schema helper ────────────────────────────────────────────────────────────

/// Build the JSON Schema value for `ReclassifyBatch`.
fn reclassify_schema() -> std::result::Result<serde_json::Value, serde_json::Error> {
    let schema = schema_for!(ReclassifyBatch);
    serde_json::to_value(schema)
}

// ─── Main primitive ───────────────────────────────────────────────────────────

/// Default batch size for the two-arm SELECT.
#[doc(hidden)]
pub const MAX_RECLASSIFY_BATCH: usize = 20;

/// Tuning knobs for a single [`reclassify`] invocation.
///
/// Grouping these three values prevents the function from exceeding clippy's
/// `too_many_arguments` limit (5) while keeping all call sites readable.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct ReclassifyOpts {
    /// Low-confidence arm threshold (entities with `ner_confidence < this` qualify).
    /// Default: `0.5`.
    pub confidence_threshold: f32,
    /// Source-tier stamp threshold.  Decisions with `confidence >= this` get
    /// `entity_type_source = 'DreamPass1'`; below this the existing source is preserved.
    /// Default: `0.7`.
    pub high_conf_threshold: f32,
    /// Maximum number of candidates to process per invocation.
    /// Default: [`MAX_RECLASSIFY_BATCH`].
    pub max_batch_size: usize,
}

impl Default for ReclassifyOpts {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            high_conf_threshold: 0.7,
            max_batch_size: MAX_RECLASSIFY_BATCH,
        }
    }
}

/// Reclassify entities matching the two-arm SELECT for `group_id`.
///
/// Called by `mem.dream()` after Pass 0 (type discovery) and before Pass 3
/// (canonicalise). Satisfies DoD E7 pass ordering.
///
/// Parameters:
/// - `conn` — open SQLite connection
/// - `group_id` — the namespace (equals `namespace_to_group_id(&ns)`)
/// - `llm` — the chat provider
/// - `opts` — tuning knobs (thresholds + batch size)
#[doc(hidden)]
pub async fn reclassify<L: ChatProvider>(
    conn: &libsql::Connection,
    group_id: &str,
    llm: &L,
    opts: ReclassifyOpts,
) -> Result<ReclassifyResult> {
    let confidence_threshold = opts.confidence_threshold;
    let high_conf_threshold = opts.high_conf_threshold;
    let max_batch_size = opts.max_batch_size;
    let mut result = ReclassifyResult::default();

    // ── Step 1: Emit skipped-entity counters (proof-of-exclusion for E9) ──────
    //
    // ConsumerPinned and DreamPass1 are excluded STRUCTURALLY at SQL WHERE level
    // (E1). These counters are informational — they prove exclusion fired as
    // expected. Counted via separate query to avoid polluting the candidate SELECT.

    emit_exclusion_counters(conn, group_id, confidence_threshold).await?;

    // ── Step 2: Two-arm SELECT ────────────────────────────────────────────────

    let candidates = load_candidates(conn, group_id, confidence_threshold, max_batch_size).await?;
    if candidates.is_empty() {
        tracing::debug!(
            target: "kremory::dream::reclassify",
            group_id = %group_id,
            "reclassify: no candidates — skipping"
        );
        return Ok(result);
    }

    tracing::debug!(
        target: "kremory::dream::reclassify",
        group_id = %group_id,
        candidate_count = candidates.len(),
        "reclassify: found candidates"
    );

    // ── Step 3: Load entity type registry for prompt building ─────────────────

    let registry =
        crate::core::entity_types::EntityTypeRegistry::load_for_group(conn, group_id).await?;
    let model_str = llm.model().to_string();

    // ── Step 4: Build JSON schema ─────────────────────────────────────────────

    let schema =
        reclassify_schema().map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!(e)))?;

    // ── Step 5: Build LLM prompt ──────────────────────────────────────────────

    let messages = build_reclassify_messages(&candidates, &registry);

    // ── Step 6: LLM call with observability ──────────────────────────────────

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(llm, &schema, "ReclassifyBatch")
        .model(&model_str)
        .messages(messages)
        .call()
        .await;
    let elapsed_ms = call_start.elapsed().as_millis() as f64;

    histogram!("kremory.dream.reclassify_call_duration_ms").record(elapsed_ms);

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            counter!(
                "kremory.dream.reclassify_call_outcome_total",
                "outcome" => "llm_err"
            )
            .increment(1);
            tracing::warn!(
                target: "kremory::dream::reclassify",
                error = %e,
                group_id = %group_id,
                "reclassify: LLM call failed — returning empty result"
            );
            return Ok(result);
        }
    };

    // ── Step 7: Deserialise with per-arm parse outcome counters (E9) ──────────
    //
    // Direct-parse success vs repair-path success counted separately per
    // [[llm-output-parse-loudly]] / cardinal failure mode #9.

    let batch: ReclassifyBatch = match serde_json::from_value(raw_value.clone()) {
        Ok(b) => {
            counter!(
                "kremory.dream.reclassify_call_outcome_total",
                "outcome" => "ok"
            )
            .increment(1);
            b
        }
        Err(_) => {
            // Repair: wrap bare array in {"decisions": ...}
            let repaired = if raw_value.is_array() {
                serde_json::json!({ "decisions": raw_value })
            } else {
                raw_value.clone()
            };
            match serde_json::from_value::<ReclassifyBatch>(repaired) {
                Ok(b) => {
                    counter!(
                        "kremory.dream.reclassify_call_outcome_total",
                        "outcome" => "parse_repair"
                    )
                    .increment(1);
                    b
                }
                Err(e2) => {
                    counter!(
                        "kremory.dream.reclassify_call_outcome_total",
                        "outcome" => "parse_fail"
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::dream::reclassify",
                        error = %e2,
                        group_id = %group_id,
                        "reclassify: failed to parse LLM batch — returning empty result"
                    );
                    return Ok(result);
                }
            }
        }
    };

    // ── Step 8: Apply decisions with confidence-aware source-tier stamping ─────

    // Build candidate lookup by id for fast match
    let candidate_map: std::collections::HashMap<&str, &Candidate> =
        candidates.iter().map(|c| (c.id.as_str(), c)).collect();

    let now = Utc::now().to_rfc3339();

    for decision in &batch.decisions {
        // Validate: entity_id must be in the candidate set (guards against hallucinated ids)
        let Some(candidate) = candidate_map.get(decision.entity_id.as_str()) else {
            tracing::debug!(
                target: "kremory::dream::reclassify",
                entity_id = %decision.entity_id,
                "reclassify: decision for unknown entity_id — skipped"
            );
            result.warnings.push(format!(
                "reclassify: decision for unknown entity_id='{}' — skipped",
                decision.entity_id
            ));
            continue;
        };

        // Validate: entity_type_id must be non-zero (avoid re-creating catch-all assignments)
        if decision.entity_type_id == 0 {
            tracing::debug!(
                target: "kremory::dream::reclassify",
                entity_id = %decision.entity_id,
                "reclassify: entity_type_id=0 in decision — skipped (catch-all is not a reclassification)"
            );
            continue;
        }

        let conf = decision.confidence.unwrap_or(0.0);
        let high_conf = conf >= high_conf_threshold;

        if high_conf {
            // High-confidence path: stamp DreamPass1 + update type_id (E3, E5).
            // DreamPass1 entities are structurally excluded from next cycle's SELECT (E5).
            apply_reclassify_high_conf(
                conn,
                group_id,
                &decision.entity_id,
                decision.entity_type_id,
                &now,
            )
            .await?;
            counter!(
                "kremory.dream.reclassify_source_tier_written_total",
                "tier" => "DreamPass1"
            )
            .increment(1);
        } else {
            // Low-confidence path: update type_id only; preserve existing source tier (E3).
            apply_reclassify_low_conf(
                conn,
                group_id,
                &decision.entity_id,
                decision.entity_type_id,
                &now,
            )
            .await?;
            counter!(
                "kremory.dream.reclassify_source_tier_written_total",
                "tier" => "preserved"
            )
            .increment(1);
        }

        // Per-trigger counter (E2)
        counter!(
            "kremory.dream.entities_reclassified_total",
            "trigger" => candidate.trigger.as_str()
        )
        .increment(1);

        result.entities_reclassified += 1;

        tracing::debug!(
            target: "kremory::dream::reclassify",
            entity_id = %decision.entity_id,
            old_type_id = candidate.entity_type_id,
            new_type_id = decision.entity_type_id,
            confidence = conf,
            high_conf = high_conf,
            trigger = candidate.trigger.as_str(),
            "reclassify: entity reclassified"
        );
    }

    if result.entities_reclassified > 0 {
        tracing::info!(
            target: "kremory::dream::reclassify",
            group_id = %group_id,
            entities_reclassified = result.entities_reclassified,
            "reclassify: pass complete"
        );
    }

    Ok(result)
}

// ─── SQL helpers ──────────────────────────────────────────────────────────────

/// Load two-arm candidates, excluding ConsumerPinned and DreamPass1 STRUCTURALLY
/// in the WHERE clause (DoD E1 / [[load-bearing-invariants-at-emit-not-prompt]]).
///
/// Both exclusions are SQL-level, not post-fetch filter — the invariant is enforced
/// at emit (the SELECT), not in application logic.
async fn load_candidates(
    conn: &libsql::Connection,
    group_id: &str,
    confidence_threshold: f32,
    max_batch_size: usize,
) -> Result<Vec<Candidate>> {
    let threshold_f64 = confidence_threshold as f64;
    let limit = max_batch_size as i64;

    // ADR-046 Amendment 2026-06-09 Option E — 2-arm SELECT.
    // Drift arm is explicitly NOT included (deferred per ADR-046 §8).
    //
    // ConsumerPinned and DreamPass1 are excluded via `entity_type_source NOT IN (...)`.
    // This is structural (SQL WHERE) — not a prompt instruction. E1 satisfied.
    //
    // ADR-050 Phase 3 — Guard anti-loop: `AND is_dream_generated = 0` excludes
    // entities written by Pass 4 / verify_stage (dream-generated). Without this,
    // reclassify would re-process dream-generated entities on the next dream pass,
    // creating a runaway loop (R-10 stop condition). Structural at SQL WHERE, not
    // application logic — load-bearing-invariants-at-emit-not-prompt.
    let mut rows = conn
        .query(
            "SELECT id, entity_type_id, ner_confidence, entity_type_source \
             FROM entities \
             WHERE ( \
                 entity_type_id = 0 \
                 OR (entity_type_source = 'Phase1Ner' AND ner_confidence < ?1) \
             ) \
             AND entity_type_source NOT IN ('ConsumerPinned', 'DreamPass1') \
             AND is_dream_generated = 0 \
             AND group_id = ?2 \
             LIMIT ?3",
            libsql::params![threshold_f64, group_id.to_string(), limit],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify: load_candidates query failed: {e}"
            ))
        })?;

    let mut candidates = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "reclassify: load_candidates row read failed: {e}"
        ))
    })? {
        let id: String = row.get(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify: id column read failed: {e}"
            ))
        })?;
        let type_id: i64 = row.get(1).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify: entity_type_id column read failed: {e}"
            ))
        })?;
        // ner_confidence is REAL NULL-able in schema
        let ner_conf: Option<f64> = row.get(2).ok();
        let source: String = row.get(3).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify: entity_type_source column read failed: {e}"
            ))
        })?;

        // Determine which arm surfaced this entity for per-trigger counter (E2)
        let trigger = if type_id == 0 {
            CandidateTrigger::CatchAllCascade
        } else {
            CandidateTrigger::LowConfidence
        };

        candidates.push(Candidate {
            id,
            entity_type_id: type_id as u32,
            ner_confidence: ner_conf,
            entity_type_source: source,
            trigger,
        });
    }
    Ok(candidates)
}

/// Count entities excluded by structural WHERE clause and emit per-reason counters
/// (DoD E9 — proof-of-exclusion observability).
async fn emit_exclusion_counters(
    conn: &libsql::Connection,
    group_id: &str,
    _confidence_threshold: f32,
) -> Result<()> {
    // ConsumerPinned exclusion count — structural exclusion proof (E9)
    // Note: entity_type_source='ConsumerPinned' means that column can never also
    // be 'Phase1Ner', so the inner arm check is vacuously false for them.
    // We count all ConsumerPinned entities that *would* have been in scope if not
    // excluded — i.e. entity_type_id = 0 (the only arm that fires for ConsumerPinned).
    let consumer_pinned_count = count_rows_with_one_param(
        conn,
        "SELECT COUNT(*) FROM entities \
         WHERE entity_type_source = 'ConsumerPinned' \
         AND entity_type_id = 0 \
         AND group_id = ?1",
        group_id,
    )
    .await?;

    if consumer_pinned_count > 0 {
        counter!(
            "kremory.dream.reclassify_entities_skipped_total",
            "reason" => "consumer_pinned"
        )
        .increment(consumer_pinned_count);
    }

    // DreamPass1 exclusion count — re-re-type protection proof (E5, E9)
    let dreampass1_count = count_rows_with_one_param(
        conn,
        "SELECT COUNT(*) FROM entities \
         WHERE entity_type_source = 'DreamPass1' \
         AND group_id = ?1",
        group_id,
    )
    .await?;

    if dreampass1_count > 0 {
        counter!(
            "kremory.dream.reclassify_entities_skipped_total",
            "reason" => "dreampass1_excluded"
        )
        .increment(dreampass1_count);
    }

    Ok(())
}

/// Generic COUNT(*) helper — runs a query returning a single i64 count.
async fn count_rows_with_one_param(
    conn: &libsql::Connection,
    sql: &str,
    param: &str,
) -> Result<u64> {
    let mut rows = conn
        .query(sql, libsql::params![param.to_string()])
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify: count_rows query failed: {e}"
            ))
        })?;
    let row = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!("reclassify: count_rows row failed: {e}"))
    })?;
    let count = match row {
        Some(r) => r.get::<i64>(0).unwrap_or(0).max(0) as u64,
        None => 0,
    };
    Ok(count)
}

/// High-confidence UPDATE: set new type_id + stamp `entity_type_source = 'DreamPass1'`
/// + update `entity_type_assigned_at`.
///
/// Re-re-type protection is structural: DreamPass1 is excluded from the next
/// cycle's SELECT at SQL WHERE level (E5).
async fn apply_reclassify_high_conf(
    conn: &libsql::Connection,
    group_id: &str,
    entity_id: &str,
    new_type_id: u32,
    now: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE entities \
         SET entity_type_id = ?1, \
             entity_type_source = 'DreamPass1', \
             entity_type_assigned_at = ?2, \
             updated_at = ?2 \
         WHERE id = ?3 AND group_id = ?4",
        libsql::params![
            new_type_id as i64,
            now.to_string(),
            entity_id.to_string(),
            group_id.to_string(),
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "reclassify: high_conf UPDATE failed for id='{}': {e}",
            entity_id
        ))
    })?;
    Ok(())
}

/// Low-confidence UPDATE: set new type_id only; `entity_type_source` preserved (E3).
/// `entity_id` is preserved across UPDATE per ADR-046 §3 (E4).
async fn apply_reclassify_low_conf(
    conn: &libsql::Connection,
    group_id: &str,
    entity_id: &str,
    new_type_id: u32,
    now: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE entities \
         SET entity_type_id = ?1, \
             updated_at = ?2 \
         WHERE id = ?3 AND group_id = ?4",
        libsql::params![
            new_type_id as i64,
            now.to_string(),
            entity_id.to_string(),
            group_id.to_string(),
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "reclassify: low_conf UPDATE failed for id='{}': {e}",
            entity_id
        ))
    })?;
    Ok(())
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

fn build_reclassify_messages(
    candidates: &[Candidate],
    registry: &crate::core::entity_types::EntityTypeRegistry,
) -> Vec<crate::core::provider::ChatMessage> {
    // Build entity type menu (id → name)
    let type_menu = registry
        .specs()
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| format!("- id={}: {} — {}", s.id, s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n");

    let type_menu = if type_menu.is_empty() {
        "(no entity types defined in this namespace yet)".to_string()
    } else {
        type_menu
    };

    // Build candidate list
    let candidate_list = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let conf_str = c
                .ner_confidence
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "n/a".to_string());
            format!(
                "{}. entity_id=\"{}\" (current_type_id={}, source={}, ner_confidence={})",
                i + 1,
                c.id,
                c.entity_type_id,
                c.entity_type_source,
                conf_str
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let system = "\
You are a knowledge-graph type curator. Your task is to reclassify entities that \
currently have an incorrect or missing entity type assignment.\n\
\n\
You will receive:\n\
1. A list of available entity types with their integer IDs and descriptions.\n\
2. A list of entities that need reclassification.\n\
\n\
Rules:\n\
- For each entity, assign the most appropriate entity_type_id from the available types.\n\
- Set confidence to a value between 0.0 and 1.0 reflecting how certain you are.\n\
- If no type fits well, omit the entity from the decisions array — do NOT assign id=0.\n\
- Respond ONLY with the JSON structure — no extra commentary."
        .to_string();

    let user = format!(
        "Available entity types:\n{type_menu}\n\n\
Entities to reclassify:\n{candidate_list}\n\n\
Return a JSON object with a `decisions` array. Each element must have: \
`entity_id` (string, exact match from the list above), \
`entity_type_id` (integer, from the available types), \
`confidence` (float 0.0-1.0)."
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

// ─── All-groups helper (Engine path) ─────────────────────────────────────────

/// Run `reclassify` for every distinct `group_id` present in the `entities` table.
///
/// Used by `Engine::run_dream_pass_sync` which has no namespace scope — it
/// processes all namespaces in the graph.  The facade path (`mem.dream()`) calls
/// `reclassify` directly with a single `group_id`.
#[doc(hidden)]
pub async fn reclassify_all_groups<L: ChatProvider>(
    graph: &TemporalGraph,
    llm: &L,
    opts: ReclassifyOpts,
) -> Result<ReclassifyResult> {
    let conn = &graph.conn;

    // Collect distinct group_ids first to avoid borrow conflicts on conn.
    let mut rows = conn
        .query(
            "SELECT DISTINCT group_id FROM entities ORDER BY group_id",
            libsql::params![],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify_all_groups: DISTINCT group_id query failed: {e}"
            ))
        })?;

    let mut group_ids: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "reclassify_all_groups: row read failed: {e}"
        ))
    })? {
        let gid: String = row.get(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "reclassify_all_groups: group_id column read failed: {e}"
            ))
        })?;
        group_ids.push(gid);
    }

    let mut total = ReclassifyResult::default();
    for gid in &group_ids {
        match reclassify(conn, gid, llm, opts).await {
            Ok(r) => {
                total.entities_reclassified += r.entities_reclassified;
                total.warnings.extend(r.warnings);
            }
            Err(e) => {
                // Per-group failure is non-fatal — surface as warning and continue.
                let msg = format!("reclassify_all_groups: group_id={gid} failed: {e}");
                tracing::warn!(target: "kremory::dream::reclassify", "{msg}");
                total.warnings.push(msg);
            }
        }
    }

    Ok(total)
}

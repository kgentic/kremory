//! LLM verify-batch primitive for Dream Pass 4 (ADR-047).
//!
//! Contains the `verify_batch` async function that calls the LLM structured
//! endpoint, parses decisions, applies corrections, and writes audit rows.
//! Extracted from the `consistency_check` monolith as part of TD-C split
//! (ADR-050 §5).

use std::time::Instant;

use chrono::Utc;
use metrics::{counter, histogram};

use crate::core::error::Result;

use super::{
    audit::{apply_correction, write_audit_row, AuditRowParams},
    verify_batch_schema, VerifyAction, VerifyBatch, VerifyBatchCounts, VerifyBatchDecision,
    VerifyBatchOutcome, VerifyBatchParams, VerifyDecision, MIN_VERIFY_CONFIDENCE,
};

/// Run the LLM verify call on `params.flagged` candidates and apply corrections to the DB.
///
/// See the public wrapper in `mod.rs` for full documentation.
pub(super) async fn verify_batch(
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

    let messages = super::audit::build_verify_messages(flagged, type_map);
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
    let rowid_to_idx: std::collections::HashMap<i64, usize> = flagged
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.rowid, idx))
        .collect();
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
                // Quinn MED-01 fix: label confirm-counter by source so downstream observers
                // can distinguish explicit LLM agreement from confidence-gate downgrades.
                counter!(
                    "kremory.dream.consistency_check.verify_confirmed_total",
                    "source" => "explicit"
                )
                .increment(1);
            }
            "correct" => {
                // See fn-level doc: unreachable! enforces SCOPE-001 structural invariant.
                let new_type_id = decision.new_type_id.unwrap_or_else(|| {
                    unreachable!("action=correct without new_type_id should be rejected at parse (SCOPE-001)")
                });

                // T2.2 (sprint plan + basket #83 gbrain C1 gate): downgrade
                // low-confidence corrections to Confirm. The gbrain pattern
                // says "downgrade to no_contradiction" — kremory's `Confirm`
                // is the structural equivalent (keep current type, no change).
                // Without this gate, every confidence value the LLM emits is
                // treated as load-bearing, including the 0.50-0.65 band where
                // local models hedge under polysemous-hard cases (per
                // `c6-verify-model-local-ladder-benchmark-2026-06-10.md`
                // sub-finding #2).
                if decision.confidence < MIN_VERIFY_CONFIDENCE {
                    counter!(
                        "kremory.dream.consistency_check.verify_low_confidence_downgrade_total",
                        "from_type" => candidate.entity_type_id.to_string(),
                        "proposed_to_type" => new_type_id.to_string()
                    )
                    .increment(1);
                    tracing::debug!(
                        target: "kremory::dream::consistency_check",
                        rowid = candidate.rowid,
                        confidence = decision.confidence,
                        threshold = MIN_VERIFY_CONFIDENCE,
                        from_type = candidate.entity_type_id,
                        proposed_to_type = new_type_id,
                        "low-confidence correct → downgraded to Confirm (keep current type)"
                    );
                    decisions[candidate_idx] = VerifyBatchDecision {
                        candidate_idx,
                        action: VerifyAction::Confirm,
                        new_type_id: None,
                    };
                    counts.confirmed += 1;
                    // Quinn MED-01 fix: label confirm-counter by source. This downgrade
                    // path is observably distinct from the explicit "confirm" LLM emit.
                    counter!(
                        "kremory.dream.consistency_check.verify_confirmed_total",
                        "source" => "downgrade_low_conf"
                    )
                    .increment(1);
                    continue;
                }

                // For dream-phase flow (entities already in DB), apply the correction now.
                // For Stage 2 pre-write flow (entities NOT in DB), rowid == -1 so
                // apply_correction UPDATE hits 0 rows — no harm done. The per-entity
                // decision is what matters; write_verified_entities uses new_type_id directly.
                //
                // Dry-run mode (sprint plan T1.3 — basket #194): skip BOTH the
                // entity-type UPDATE and the audit-row INSERT. The decision is
                // still returned in VerifyBatchOutcome so the caller can preview
                // what would have changed.
                if candidate.rowid > 0 && !params.dry_run {
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
                } else if params.dry_run {
                    // Quinn LOW-02 fix: emit dry_run counter for BOTH dream-phase
                    // (rowid > 0) and Stage 2 pre-write (rowid <= 0) paths. Without
                    // the Stage 2 branch, callers running verify_batch_for_candidates
                    // with dry_run=true would have no observability signal that the
                    // dry-run guard fired. Labels on the counter distinguish paths.
                    counter!(
                        "kremory.dream.consistency_check.dry_run_skipped_total",
                        "path" => if candidate.rowid > 0 { "dream_phase" } else { "stage2_pre_write" }
                    )
                    .increment(1);
                    tracing::debug!(
                        target: "kremory::dream::consistency_check",
                        rowid = candidate.rowid,
                        from_type = candidate.entity_type_id,
                        to_type = new_type_id,
                        "dry_run — skipped apply_correction + audit row"
                    );
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

// ─── verify_batch_for_candidates (GAP-003) ───────────────────────────────────

/// Tuning knobs for [`verify_batch_for_candidates`].
///
/// Mirrors the relevant subset of [`super::ConsistencyCheckOpts`] without the
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
    llm: &dyn crate::core::provider::ChatProvider,
    opts: VerifyBatchForCandidatesOpts,
) -> crate::core::error::Result<VerifyBatchForCandidatesResult> {
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
    let type_map = super::audit::load_type_registry(db).await?;

    // Build CandidateRow values from the provided EntityCandidate slice.
    // Look up each entity's rowid by name so the LLM prompt uses integer ids
    // (consistent with how run_consistency_check works).
    // For entities not yet in DB (Stage 2 ingest-time flow), rowid = -1 sentinel.
    // verify_batch will include them in the LLM prompt; rowid = -1 entries skip
    // apply_correction and write_audit_row (no DB rows to update yet).
    let mut candidate_rows: Vec<super::CandidateRow> =
        Vec::with_capacity(effective_candidates.len());
    for ec in effective_candidates {
        let entity_id = normalize_name(&ec.name);
        let rowid: i64 = {
            let mut rows = db
                .query(
                    "SELECT rowid FROM entities WHERE id = ?1 LIMIT 1",
                    libsql::params![entity_id.clone()],
                )
                .await
                .map_err(|e| {
                    crate::core::error::Error::Other(anyhow::anyhow!(
                        "rowid lookup '{}': {e}",
                        entity_id
                    ))
                })?;
            if let Some(row) = rows.next().await.map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!("rowid row '{}': {e}", entity_id))
            })? {
                row.get(0).map_err(|e| {
                    crate::core::error::Error::Other(anyhow::anyhow!(
                        "rowid col '{}': {e}",
                        entity_id
                    ))
                })?
            } else {
                // Entity not yet in DB — Stage 2 ingest-time flow.
                // rowid = -1 is a sentinel; verify_batch skips DB writes for these.
                -1i64
            }
        };
        let top3_facts = if rowid > 0 {
            super::audit::load_top3_facts(db, &entity_id)
                .await
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        candidate_rows.push(super::CandidateRow {
            rowid,
            name: ec.name.clone(),
            entity_type_id: ec.entity_type_id_raw,
            top3_facts,
            source_episode: Some(source_episode_text.to_string()),
        });
    }

    // Build &[&CandidateRow] for verify_batch.
    let flagged_refs: Vec<&super::CandidateRow> = candidate_rows.iter().collect();

    let run_id = uuid::Uuid::new_v4().to_string();
    let outcome = verify_batch(
        db,
        VerifyBatchParams {
            flagged: &flagged_refs,
            type_map: &type_map,
            llm,
            verify_model: &verify_model,
            run_id: &run_id,
            // Stage 2 pre-write path: dry_run is irrelevant here because entities
            // aren't in the DB yet (rowid == -1), so apply_correction is already
            // a no-op. Always false to match the historical behavior.
            dry_run: false,
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

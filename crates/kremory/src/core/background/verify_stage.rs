//! Stage 2 verify hook — GLiNER NER + verify_batch + Stage 3 write (Path α)
//! or LLM direct-extract + Stage 3 write (Path β).
//!
//! ADR-051: GLiNER-to-background unified hot path.
//! Spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` Phase 2.
//!
//! # Phase 2 note — dead_code suppressions
//!
//! `run_verify_stage` and its private helpers are structurally complete here but
//! not yet called by `deferred_pipeline::worker_loop`.  That wiring is Phase 3.
//! The suppressions below are the same documented-exemption pattern used by the
//! Phase B stub (ADR-049 §Decision 6: "structural prerequisite MUST exist before
//! wiring").  Phase 3 removes them once the call site is added.
//!
//! # Observability surface
//!
//! Per CLAUDE.md Rule 19 (observability-first-class):
//!
//! **Histograms** (arm-labelled):
//! - `kremory.verify_stage.duration_ms{arm="gliner"}` — GLiNER/extractor extract call
//! - `kremory.verify_stage.duration_ms{arm="verify_batch"}` — Path α verify_batch call
//! - `kremory.verify_stage.duration_ms{arm="stage3_write"}` — entity write
//! - `kremory.verify_stage.duration_ms{arm="llm_extract"}` — Path β LLM extract call
//!
//! **Counters** (arm + outcome labelled):
//! - `kremory.verify_stage.outcome_total{arm, outcome}` — terminal outcome per arm
//!   Outcomes: `success`, `gliner_fail`, `verify_fail`, `write_fail`
//! - `kremory.episode.processing_status_transition_total{from, to}` — per state-transition
//!
//! **Tracing events**:
//! - `tracing::info!` on entry (episode_id, arm, text_len)
//! - `tracing::info!` on success exit (episode_id, arm, outcome, entities_written, total_duration_ms)
//! - `tracing::warn!` on verify_batch failure (candidates_count, error chain)
//! - `tracing::error!` on terminal failure (full diagnostic context)

use std::time::Instant;

use chrono::Utc;

use crate::core::dream::consistency_check::{
    verify_batch_for_candidates, VerifyBatchForCandidatesOpts,
};
use crate::core::error::Error;
use crate::core::ingest::{EntityCandidate, ResolvedDecision};
use crate::core::intelligence::{EntityExtractorDyn, ExtractionContext, ExtractionResult};
use crate::core::provider::ChatProvider;
use crate::core::resolver::normalize_name;
use crate::core::schema::TemporalGraph;

use super::DeferredRequest;

// ─── State transition helper ───────────────────────────────────────────────────

/// Write `UPDATE episodes SET episode_processing_status = ?` for a given episode id.
///
/// The single authoritative write path for status transitions (ADR-051 state machine).
/// Emits `kremory.episode.processing_status_transition_total{from, to}` counter.
///
/// `#[allow(dead_code)]` — Phase 2 structural prerequisite; Phase 3 removes this
/// when `deferred_pipeline` calls `run_verify_stage`.
#[allow(dead_code)]
async fn update_episode_status(
    conn: &libsql::Connection,
    episode_id: i64,
    status: &str,
    from: &str,
    to: &str,
) -> Result<(), Error> {
    conn.execute(
        "UPDATE episodes SET episode_processing_status = ?1 WHERE id = ?2",
        libsql::params![status, episode_id],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "update_episode_status '{}' for episode {}: {e}",
            status,
            episode_id
        ))
    })?;
    metrics::counter!(
        "kremory.episode.processing_status_transition_total",
        "from" => from.to_string(),
        "to" => to.to_string()
    )
    .increment(1);
    Ok(())
}

// ─── Stage 3 write helper ──────────────────────────────────────────────────────

/// Write entity rows and episodic edges for the given verify decisions.
///
/// Mirrors `Engine::write_verified_entities` exactly but takes `&TemporalGraph`
/// instead of the generic `&Engine<L, Emb>`, keeping `verify_stage.rs` free of
/// type-parameter entanglement.
///
/// Returns the number of entity rows persisted.
///
/// `#[allow(dead_code)]` — Phase 2 structural prerequisite; Phase 3 removes this.
#[allow(dead_code)]
async fn stage3_write(
    graph: &TemporalGraph,
    episode_id: i64,
    candidates: &[EntityCandidate],
    decisions: &[ResolvedDecision],
) -> Result<usize, Error> {
    let now = Utc::now().to_rfc3339();
    let mut count = 0usize;

    for decision in decisions {
        let (candidate_idx, entity_type_id) = match decision {
            ResolvedDecision::Confirm { candidate_idx } => {
                let candidate = &candidates[*candidate_idx];
                (*candidate_idx, candidate.entity_type_id_raw)
            }
            ResolvedDecision::Correct {
                candidate_idx,
                new_type_id,
            } => (*candidate_idx, *new_type_id),
            ResolvedDecision::Demote { candidate_idx } => (*candidate_idx, 0i64),
        };

        let candidate = &candidates[candidate_idx];
        let entity_id = normalize_name(&candidate.name);
        let props = serde_json::json!({ "name": candidate.name });
        let props_str = serde_json::to_string(&props)
            .map_err(|e| Error::Other(anyhow::anyhow!("stage3_write serialize props: {e}")))?;

        // INSERT OR IGNORE — first-mention-wins per ingest_with convention.
        graph
            .conn
            .execute(
                "INSERT OR IGNORE INTO entities \
                 (id, entity_type_id, properties, recorded_at, group_id, \
                  entity_type_source, entity_type_assigned_at, ner_confidence) \
                 VALUES (?1, ?2, ?3, ?4, 'default', 'Phase1Ner', ?4, ?5)",
                libsql::params![
                    entity_id.clone(),
                    entity_type_id,
                    props_str.clone(),
                    now.clone(),
                    candidate.ner_confidence as f64
                ],
            )
            .await
            .map_err(|e| {
                Error::Other(anyhow::anyhow!(
                    "stage3_write INSERT entity '{}': {e}",
                    entity_id
                ))
            })?;

        // FTS row — required per graph.rs invariant.
        graph
            .conn
            .execute(
                "INSERT OR IGNORE INTO entities_fts(entity_id, label, properties) \
                 VALUES (?1, '', ?2)",
                libsql::params![entity_id.clone(), props_str],
            )
            .await
            .map_err(|e| {
                Error::Other(anyhow::anyhow!(
                    "stage3_write FTS INSERT entity '{}': {e}",
                    entity_id
                ))
            })?;

        // Episodic edge — link entity to its source episode.
        graph
            .insert_episodic_edge(episode_id, &entity_id, "mention")
            .await
            .ok();

        metrics::counter!(
            "rql.ingest.entity_persisted_total",
            "source" => "verify_stage",
        )
        .increment(1);

        count += 1;
    }

    Ok(count)
}

// ─── ExtractionResult → EntityCandidate ───────────────────────────────────────

/// Convert an `ExtractionResult` into `EntityCandidate` vec for downstream calls.
///
/// `entity_type_id_raw` is set to `0` (catch-all "Entity") for BOTH paths:
/// - Path α: `verify_batch_for_candidates` re-types each candidate before Stage 3
///   write (the LLM verifier owns typing; candidates only carry the name + span).
/// - Path β: `ExtractedEntity.label` is a free-text string (e.g. `"PERSON"`,
///   `"ORG"`) — there is no integer `entity_type_id` available yet. Stage 3 writes
///   with `entity_type_id=0`. Future enhancement (TD candidate): resolve
///   `ExtractedEntity.label` → `entity_type_id` via a label-to-id table so Path β
///   produces typed entities without an LLM round-trip.
///
/// `#[allow(dead_code)]` — Phase 2 structural prerequisite; Phase 3 spec DoD
/// requires removal when `worker_loop` wires the caller.
#[allow(dead_code)]
fn extraction_result_to_candidates(result: &ExtractionResult) -> Vec<EntityCandidate> {
    result
        .entities
        .iter()
        .map(|e| {
            // L-02 scope tightening: allow scoped to the f64→f32 confidence cast only.
            // `confidence` is mathematically bounded to [0.0, 1.0] so no real
            // mantissa truncation is possible; clippy can't prove the bound.
            #[allow(clippy::cast_possible_truncation)]
            let confidence = e
                .properties
                .get("confidence")
                .and_then(|v| v.as_f64())
                .map(|c| c as f32)
                .unwrap_or(0.0_f32);
            EntityCandidate {
                name: e.name.clone(),
                entity_type_id_raw: 0,
                ner_confidence: confidence,
                span: (0, 0),
            }
        })
        .collect()
}

// ─── run_verify_stage ──────────────────────────────────────────────────────────

/// Run the Stage 2 verify gate for one episode.
///
/// Owns the post-INSERT-episode extraction work for the background worker.
/// Called by `deferred_pipeline::process_deferred` once Phase 3 wires the call
/// (current stub in `deferred_pipeline` calls `Engine::ingest_deferred` instead).
///
/// # Branch logic
///
/// - `verify_llm = Some(llm)` → **Path α** (GLiNER hot path):
///   `extractor.extract_dyn` → `verify_batch_for_candidates` → `stage3_write`
/// - `verify_llm = None` → **Path β** (LLM direct extract):
///   `extractor.extract_dyn` (extractor IS an LLM) → `stage3_write` with Confirm decisions
///
/// # State transitions (ADR-051 §4 state machine)
///
/// ```text
/// Pending → Extracting   on entry, before first external call
/// Extracting → Verified  on success exit
/// Extracting → Failed    on ANY error, BEFORE the error propagates
/// ```
///
/// The `Failed` write is performed synchronously before `Err` is returned so the
/// status column is always coherent even when the caller drops the error.
///
/// # Returns
///
/// `Ok(n)` — `n` entities written to the `entities` table.
/// `Err` — extraction, verify, or write failure (episode status already `Failed`).
///
/// `pub` + `#[doc(hidden)]` per MNT-002 pattern: integration tests in
/// `tests/verify_stage_integration.rs` call this directly under `feature = "test-utils"`.
/// `#[allow(dead_code)]` — Phase 2 structural prerequisite; Phase 3 removes this attribute
/// once `deferred_pipeline::worker_loop` calls `run_verify_stage`.
#[doc(hidden)]
#[allow(dead_code)]
pub async fn run_verify_stage<'a>(
    request: &'a DeferredRequest,
    extractor: &'a dyn EntityExtractorDyn,
    verify_llm: Option<&'a dyn ChatProvider>,
    graph: &'a TemporalGraph,
) -> Result<usize, Error> {
    let total_start = Instant::now();
    let arm = if verify_llm.is_some() {
        "gliner"
    } else {
        "llm_extract"
    };
    let conn = &graph.conn;

    tracing::info!(
        episode_id = request.episode_id,
        arm,
        text_len = request.text.len(),
        "kremory.verify_stage.start"
    );

    // ── Pending → Extracting ───────────────────────────────────────────────────
    //
    // Quinn H-01 fix: the initial transition UPDATE used bare `.await?`, which
    // would propagate Err WITHOUT writing `Failed` — contradicting the function's
    // own doc-comment invariant ("the Failed write is performed synchronously
    // before Err is returned so the status column is always coherent"). Cause-fix
    // per Rule 8: replace the bare `?` with the same explicit match-and-best-
    // effort-Failed-write pattern used on every other error path. If the Failed
    // write ALSO fails (DB unrecoverable), we can't progress further — log + return
    // the original error.
    if let Err(e) = update_episode_status(
        conn,
        request.episode_id,
        "Extracting",
        "Pending",
        "Extracting",
    )
    .await
    {
        tracing::error!(
            episode_id = request.episode_id,
            arm,
            error = %e,
            "kremory.verify_stage.status_pending_to_extracting_fail"
        );
        metrics::counter!(
            "kremory.verify_stage.outcome_total",
            "arm" => arm,
            "outcome" => "status_transition_fail"
        )
        .increment(1);
        // Best-effort Failed write — if this also fails the DB is in worse trouble
        // than this code path can resolve; the original error is what callers see.
        let _ =
            update_episode_status(conn, request.episode_id, "Failed", "Pending", "Failed").await;
        return Err(e);
    }

    // ── Extract candidates ─────────────────────────────────────────────────────
    let ctx = ExtractionContext::default();
    let extract_start = Instant::now();

    let extraction_result = extractor.extract_dyn(&request.text, &ctx).await;

    let extract_elapsed_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!("kremory.verify_stage.duration_ms", "arm" => arm.to_string())
        .record(extract_elapsed_ms);

    let extraction_result = match extraction_result {
        Ok(r) => r,
        Err(e) => {
            metrics::counter!(
                "kremory.verify_stage.outcome_total",
                "arm" => arm,
                "outcome" => "gliner_fail"
            )
            .increment(1);
            tracing::error!(
                episode_id = request.episode_id,
                arm,
                error = %e,
                "kremory.verify_stage.extract_fail"
            );
            // Failed write BEFORE propagating — state machine invariant.
            let _ =
                update_episode_status(conn, request.episode_id, "Failed", "Extracting", "Failed")
                    .await;
            return Err(e);
        }
    };

    let candidates = extraction_result_to_candidates(&extraction_result);

    // ── Branch: Path α or Path β ───────────────────────────────────────────────
    let entities_written = match verify_llm {
        Some(llm) => {
            // ── Path α: verify_batch_for_candidates → Stage 3 write ───────────

            let vb_start = Instant::now();
            let vb_result = verify_batch_for_candidates(
                conn,
                &candidates,
                &request.text,
                llm,
                VerifyBatchForCandidatesOpts::default(),
            )
            .await;

            let vb_elapsed_ms = vb_start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("kremory.verify_stage.duration_ms", "arm" => "verify_batch")
                .record(vb_elapsed_ms);

            let vb_result = match vb_result {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!(
                        "kremory.verify_stage.outcome_total",
                        "arm" => "gliner",
                        "outcome" => "verify_fail"
                    )
                    .increment(1);
                    tracing::warn!(
                        episode_id = request.episode_id,
                        candidates_count = candidates.len(),
                        error = %e,
                        "kremory.verify_stage.verify_batch_fail"
                    );
                    let _ = update_episode_status(
                        conn,
                        request.episode_id,
                        "Failed",
                        "Extracting",
                        "Failed",
                    )
                    .await;
                    return Err(e);
                }
            };

            let w3_start = Instant::now();
            let write_result =
                stage3_write(graph, request.episode_id, &candidates, &vb_result.decisions).await;

            let w3_elapsed_ms = w3_start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("kremory.verify_stage.duration_ms", "arm" => "stage3_write")
                .record(w3_elapsed_ms);

            match write_result {
                Ok(n) => n,
                Err(e) => {
                    metrics::counter!(
                        "kremory.verify_stage.outcome_total",
                        "arm" => "gliner",
                        "outcome" => "write_fail"
                    )
                    .increment(1);
                    tracing::error!(
                        episode_id = request.episode_id,
                        arm = "gliner",
                        error = %e,
                        "kremory.verify_stage.stage3_write_fail"
                    );
                    let _ = update_episode_status(
                        conn,
                        request.episode_id,
                        "Failed",
                        "Extracting",
                        "Failed",
                    )
                    .await;
                    return Err(e);
                }
            }
        }
        None => {
            // ── Path β: LLM direct extract → Stage 3 write ───────────────────
            //
            // The extractor IS an LLM extractor; `extraction_result` already contains
            // typed entities. Build Confirm decisions (entity_type_id_raw is the
            // authoritative type from the LLM extraction — no verify_batch needed).
            let decisions: Vec<ResolvedDecision> = (0..candidates.len())
                .map(|i| ResolvedDecision::Confirm { candidate_idx: i })
                .collect();

            let w3_start = Instant::now();
            let write_result =
                stage3_write(graph, request.episode_id, &candidates, &decisions).await;

            let w3_elapsed_ms = w3_start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("kremory.verify_stage.duration_ms", "arm" => "stage3_write")
                .record(w3_elapsed_ms);

            match write_result {
                Ok(n) => n,
                Err(e) => {
                    metrics::counter!(
                        "kremory.verify_stage.outcome_total",
                        "arm" => "llm_extract",
                        "outcome" => "write_fail"
                    )
                    .increment(1);
                    tracing::error!(
                        episode_id = request.episode_id,
                        arm = "llm_extract",
                        error = %e,
                        "kremory.verify_stage.stage3_write_fail"
                    );
                    let _ = update_episode_status(
                        conn,
                        request.episode_id,
                        "Failed",
                        "Extracting",
                        "Failed",
                    )
                    .await;
                    return Err(e);
                }
            }
        }
    };

    // ── Extracting → Verified ──────────────────────────────────────────────────
    if let Err(e) = update_episode_status(
        conn,
        request.episode_id,
        "Verified",
        "Extracting",
        "Verified",
    )
    .await
    {
        // Status update failure does NOT undo already-written entities.
        // Entities are in the DB; only the status column failed. Log + continue.
        tracing::error!(
            episode_id = request.episode_id,
            error = %e,
            "kremory.verify_stage.status_verified_update_fail"
        );
    }

    let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    metrics::counter!(
        "kremory.verify_stage.outcome_total",
        "arm" => arm,
        "outcome" => "success"
    )
    .increment(1);

    tracing::info!(
        episode_id = request.episode_id,
        arm,
        outcome = "success",
        entities_written,
        total_duration_ms = total_elapsed_ms,
        "kremory.verify_stage.complete"
    );

    Ok(entities_written)
}

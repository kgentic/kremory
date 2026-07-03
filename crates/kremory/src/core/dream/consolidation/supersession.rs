//! CONSOLIDATION op — supersession sweep (ADR-066 §2.3, spec P1).
//!
//! Two lanes:
//! - **deterministic world-time window close-out** (default, this op's payload):
//!   retire facts whose world-time window has demonstrably ended but were never
//!   marked expired — `valid_to IS NOT NULL AND valid_to < now AND expired_at IS
//!   NULL AND invalid_at IS NULL AND is_dream_generated = 0`. Set
//!   `expired_at = valid_to` via `invalidate_fact`. Pure date-compare, ZERO LLM.
//!   Orthogonal to ingest's same-object dedup (`contradiction.rs:285-294`) — that
//!   keys on same-object duplicates + overlap, NOT on a closed `valid_to`.
//! - **optional LLM-nominated value-change** (default-off via
//!   `include_llm_nominate`): different-object value-changes where the LLM only
//!   NOMINATES the pairing and a deterministic emit-invariant REJECTS any
//!   nomination with `newer.valid_from <= older.valid_from` (an LLM can never
//!   invert the arrow of time). **STUB-OFF this phase** — the lane is plumbed but
//!   returns 0 with a documented no-op (P1 ships the deterministic lane fully).
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P1.1–P1.4) + §6 (corpus) + ADR-066 §2.3 (the RESHAPED window-closeout lane).

use chrono::{DateTime, Utc};
use metrics::counter;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::{ConsolidationBudget, OpReport};

/// Bundled params for [`supersession`] — args-as-object per TD-042
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic corpus harness (`tests/consolidation_supersession_test.rs`)
/// can construct it under `feature = "test-utils"` (external test binaries cannot
/// import `pub(crate)` items — E0365). NOT part of the stable public API.
#[doc(hidden)]
pub struct SupersessionParams<'a> {
    pub graph: &'a TemporalGraph,
    pub group_id: &'a str,
    /// Shared soft budget (the optional LLM lane advances it via `record`).
    pub budget: &'a mut ConsolidationBudget,
    /// Select the optional LLM-nominated value-change lane (P1.3). Default-off.
    pub include_llm_nominate: bool,
    /// Resolved dream model id, threaded for the LLM lane (TD-094 style).
    pub model_id: &'a str,
}

/// One window-closeout candidate: a fact whose `valid_to` window has already
/// closed. `expired_at` is set to `valid_to` (the demonstrated close-out time).
#[derive(Debug, Clone, PartialEq, Eq)]
struct WindowCloseoutCandidate {
    fact_id: i64,
    /// The RFC3339 `valid_to` string — becomes the fact's `expired_at`.
    valid_to: String,
}

/// Emit-invariant for the LLM value-change lane (P1.3): a nomination is VALID
/// only when the newer fact's `valid_from` is strictly AFTER the older fact's.
/// An LLM can never invert the arrow of time; a nomination that would supersede a
/// newer fact with an older one is REJECTED structurally (returns `false`).
///
/// This is a pure function (unit-tested) so the emit guard is enforced at the
/// emission boundary, not via a prompt instruction
/// (`load-bearing-invariants-at-emit-not-prompt`).
#[cfg_attr(not(test), allow(dead_code))] // wired into the LLM lane (stub-off P1); unit-tested now.
fn llm_nomination_is_time_ordered(
    older_valid_from: DateTime<Utc>,
    newer_valid_from: DateTime<Utc>,
) -> bool {
    newer_valid_from > older_valid_from
}

/// Run the supersession sweep over `group_id`.
///
/// P1 ships the **deterministic window-closeout lane** fully. The optional
/// LLM-nominated value-change lane is plumbed but STUB-OFF (returns 0 with a
/// documented no-op) — it is default-off (no functional-predicate registry) and
/// its emit-invariant is unit-tested via [`llm_nomination_is_time_ordered`].
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic corpus harness can call it under `feature = "test-utils"`.
/// NOT part of the stable public API.
#[doc(hidden)]
pub async fn supersession(params: SupersessionParams<'_>) -> Result<OpReport> {
    let SupersessionParams {
        graph,
        group_id,
        budget: _budget,
        include_llm_nominate,
        model_id: _model_id,
    } = params;

    let mut report = OpReport::default();

    // ── Lane 1: deterministic world-time window close-out (P1.1) ─────────────────
    let window_closed = window_closeout(graph, group_id).await?;
    report.count += window_closed;

    // ── Lane 2: optional LLM-nominated value-change (P1.3) — STUB-OFF this phase ──
    if include_llm_nominate {
        // The lane is plumbed (flag threaded, emit-invariant `llm_nomination_is_
        // time_ordered` implemented + unit-tested) but its nominator is NOT built
        // this phase. It is default-off (no functional-predicate registry makes
        // auto-superseding different-object facts unsafe without LLM nomination).
        // Documented no-op rather than a silent skip so the surface is observable.
        counter!("kremory.dream.consolidation.supersession_llm_nominate_stub_off_total",)
            .increment(1);
        report.warnings.push(
            "supersession LLM-nominate lane requested but is stub-off this phase (P1 \
             ships the deterministic window-closeout lane only)"
                .to_string(),
        );
    }

    Ok(report)
}

/// Deterministic window close-out (P1.1): retire facts whose world-time window
/// (`valid_to`) has already ended but which were never marked expired.
///
/// SELECT predicate (P1.1, P1.2 — the double-handle guard is the `expired_at IS
/// NULL AND invalid_at IS NULL` exclusion; the anti-loop guard is
/// `is_dream_generated = 0`):
///
/// ```sql
/// valid_to IS NOT NULL AND valid_to < now
///   AND expired_at IS NULL AND invalid_at IS NULL
///   AND is_dream_generated = 0
///   AND group_id = ?
/// ```
///
/// For each match, set `expired_at = valid_to` via `invalidate_fact`. Returns the
/// count of retirements THIS sweep applied (never ingest-handled facts — those
/// are excluded by construction). Idempotent: a retired fact now has `expired_at`
/// set, so the next run's SELECT excludes it (second run → 0).
///
/// The whole sweep runs inside ONE `BEGIN IMMEDIATE` so the SELECT + the batch of
/// `expired_at` writes are atomic (no partial retirement on a mid-sweep failure).
async fn window_closeout(graph: &TemporalGraph, group_id: &str) -> Result<usize> {
    // RFC3339 UTC strings sort lexicographically in chronological order, so the
    // `valid_to < now` string compare is a correct temporal compare (all temporal
    // columns are stored via `to_rfc3339()`, e.g. `graph/facts.rs`).
    let now = Utc::now().to_rfc3339();

    let guard = graph.begin_immediate_if_needed().await?;
    let result: Result<usize> = async {
        // Candidate SELECT — the double-handle guard (`expired_at IS NULL AND
        // invalid_at IS NULL`, P1.2) excludes any fact the ingest resolver already
        // superseded/invalidated; `is_dream_generated = 0` (F-4 anti-loop) excludes
        // dream output so a dream-retired fact is never re-retired.
        let mut rows = graph
            .conn
            .query(
                "SELECT id, valid_to FROM facts \
                 WHERE valid_to IS NOT NULL \
                   AND valid_to < ?1 \
                   AND expired_at IS NULL \
                   AND invalid_at IS NULL \
                   AND is_dream_generated = 0 \
                   AND group_id = ?2",
                libsql::params![now.clone(), group_id],
            )
            .await?;

        let mut candidates: Vec<WindowCloseoutCandidate> = Vec::new();
        while let Some(row) = rows.next().await? {
            let fact_id: i64 = row.get(0)?;
            // `valid_to` is NOT NULL by the SELECT predicate, so this is always Some.
            let valid_to: String = row.get(1)?;
            candidates.push(WindowCloseoutCandidate { fact_id, valid_to });
        }
        drop(rows);

        let mut retired = 0usize;
        for cand in &candidates {
            // Set `expired_at = valid_to` — the demonstrated window-close time, NOT
            // `now` (the fact expired when its world-time window ended, not when the
            // sweep noticed). `invalidate_fact` runs `UPDATE facts SET expired_at=?1
            // WHERE id=?2` (`graph/facts.rs:140-146`, verified).
            // Parse loudly: a malformed `valid_to` is a real data-corruption signal
            // (every temporal column is written via `to_rfc3339()`), so surface it as
            // `Error::Parse` — the dispatcher's warn-and-continue folds it into the
            // summary rather than silently skipping the fact (`llm-output-parse-loudly`).
            let valid_to_dt: DateTime<Utc> = DateTime::parse_from_rfc3339(&cand.valid_to)
                .map_err(|e| {
                    crate::core::error::Error::Parse(format!(
                        "window-closeout: fact {} has unparseable valid_to {:?}: {e}",
                        cand.fact_id, cand.valid_to
                    ))
                })?
                .with_timezone(&Utc);
            graph.invalidate_fact(cand.fact_id, valid_to_dt).await?;
            retired += 1;
        }
        Ok(retired)
    }
    .await;

    match result {
        Ok(retired) => {
            guard.commit().await?;
            // P1.4: source-attributed counter split by lane. This lane == the count
            // this op reports for the deterministic path; the o11y cross-check asserts
            // counter == OpReport.count.
            counter!(
                "kremory.dream.consolidation.supersessions_recorded_total",
                "lane" => "window_closeout",
            )
            .increment(retired as u64);
            tracing::info!(
                target: "kremory.dream.consolidation.supersession",
                group_id,
                retired,
                "window-closeout sweep complete"
            );
            Ok(retired)
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;
    use chrono::Duration;

    // ── Plant helpers (direct SQL — full control over temporal columns) ──────────
    //
    // `facts` carries the composite FK `(subject_id, subject_group_id) REFERENCES
    // entities(id, group_id)` (`graph/facts.rs:333-341`), and FK enforcement is ON,
    // so the subject entity row MUST exist first + the fact must stamp
    // `subject_group_id = group_id`. Value-object facts (`object_id = NULL`) do not
    // enforce the object FK.

    /// Ensure a minimal `entities` row exists for `(id, group_id)` (idempotent).
    /// Uses the real `insert_entity_with_group` API so the composite FK + the
    /// `entities_fts` shadow are satisfied correctly (`entities.label` was dropped
    /// in Migration 009 — hand-rolled INSERTs against the old shape fail).
    async fn ensure_entity(graph: &TemporalGraph, group_id: &str, id: &str) {
        use crate::core::graph::InsertEntityWithGroupParams;
        // INSERT OR IGNORE semantics: a repeat plant of the same (id, group_id) is a
        // benign duplicate — swallow it so multi-fact plants on one subject work.
        let _ = graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({}),
                group_id: Some(group_id),
            })
            .await;
    }

    /// Insert a `facts` row with explicit temporal columns. Returns its id.
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn plant_fact(
        graph: &TemporalGraph,
        group_id: &str,
        subject: &str,
        predicate: &str,
        object_value: &str,
        valid_from: DateTime<Utc>,
        valid_to: Option<DateTime<Utc>>,
        expired_at: Option<DateTime<Utc>>,
        invalid_at: Option<DateTime<Utc>>,
        is_dream_generated: i64,
    ) -> i64 {
        ensure_entity(graph, group_id, subject).await;
        let now = Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_value, valid_from, valid_to, recorded_at, \
                  expired_at, invalid_at, group_id, subject_group_id, confidence, is_dream_generated) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1.0, ?11)",
                libsql::params![
                    subject,
                    predicate,
                    object_value,
                    valid_from.to_rfc3339(),
                    valid_to.map(|v| v.to_rfc3339()),
                    now,
                    expired_at.map(|v| v.to_rfc3339()),
                    invalid_at.map(|v| v.to_rfc3339()),
                    group_id,
                    group_id,
                    is_dream_generated,
                ],
            )
            .await
            .expect("plant fact");
        let mut rows = graph
            .conn
            .query("SELECT last_insert_rowid()", ())
            .await
            .expect("rowid");
        rows.next()
            .await
            .expect("row")
            .expect("row present")
            .get::<i64>(0)
            .expect("id")
    }

    /// Read `(expired_at)` for a fact.
    async fn read_expired_at(graph: &TemporalGraph, fact_id: i64) -> Option<String> {
        let mut rows = graph
            .conn
            .query(
                "SELECT expired_at FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await
            .expect("query expired_at");
        let row = rows.next().await.expect("row").expect("present");
        row.get::<Option<String>>(0).expect("expired_at col")
    }

    fn budget() -> ConsolidationBudget {
        ConsolidationBudget::new(None)
    }

    // ── DoD-P1.3: emit-invariant unit test (pure function, no DB) ────────────────

    #[test]
    fn emit_invariant_rejects_time_inverted_nomination() {
        let older = Utc::now();
        let newer = older + Duration::hours(1);
        // Valid nomination: newer.valid_from strictly after older.valid_from.
        assert!(
            llm_nomination_is_time_ordered(older, newer),
            "newer-after-older is a valid nomination"
        );
        // Time-inverted: newer is actually BEFORE older → REJECT (an LLM cannot
        // invert the arrow of time).
        assert!(
            !llm_nomination_is_time_ordered(newer, older),
            "inverted (older supersedes newer) must be rejected"
        );
        // Equal `valid_from` → REJECT (`newer.valid_from <= older.valid_from`).
        assert!(
            !llm_nomination_is_time_ordered(older, older),
            "equal valid_from is not a strict supersession → reject"
        );
    }

    // ── DoD-P1.1: window-closeout retires a closed-window fact ───────────────────

    #[tokio::test]
    async fn window_closeout_retires_closed_window_fact() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let closed_valid_to = now - Duration::days(1); // window ended yesterday
        let id = plant_fact(
            &graph,
            "g1",
            "alice",
            "lives_in",
            "Boston",
            now - Duration::days(30),
            Some(closed_valid_to),
            None,
            None,
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(report.count, 1, "one closed-window fact retired");
        // expired_at == valid_to (the demonstrated close-out time, NOT now).
        let expired = read_expired_at(&graph, id).await.expect("expired set");
        let expired_dt = DateTime::parse_from_rfc3339(&expired)
            .expect("rfc3339")
            .with_timezone(&Utc);
        assert_eq!(
            expired_dt, closed_valid_to,
            "expired_at must equal valid_to (window-close time)"
        );
    }

    // ── DoD-P1.2: keep cases — still-valid + already-resolved (EXACT untouched) ──

    #[tokio::test]
    async fn window_closeout_keeps_still_valid_open_ended() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // valid_to IS NULL → open-ended, still valid (a multi-valued predicate's
        // live fact). MUST NOT be retired.
        let id = plant_fact(
            &graph,
            "g1",
            "alice",
            "likes",
            "coffee",
            now - Duration::days(30),
            None, // open-ended
            None,
            None,
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(report.count, 0, "open-ended fact must be KEPT (EXACT 0)");
        assert_eq!(
            read_expired_at(&graph, id).await,
            None,
            "expired_at must remain NULL — never touched"
        );
    }

    #[tokio::test]
    async fn window_closeout_keeps_already_resolved_at_ingest() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let preset_expired = now - Duration::days(5);
        // valid_to is closed BUT expired_at is already set (ingest resolver handled
        // it). The sweep MUST NOT touch it — expired_at must stay EXACTLY as preset.
        let id = plant_fact(
            &graph,
            "g1",
            "alice",
            "works_at",
            "AcmeOld",
            now - Duration::days(30),
            Some(now - Duration::days(2)),
            Some(preset_expired),
            None,
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(
            report.count, 0,
            "already-resolved fact must be KEPT (EXACT 0)"
        );
        let expired = read_expired_at(&graph, id).await.expect("still set");
        let expired_dt = DateTime::parse_from_rfc3339(&expired)
            .expect("rfc3339")
            .with_timezone(&Utc);
        assert_eq!(
            expired_dt, preset_expired,
            "preset expired_at must be UNCHANGED (never re-written)"
        );
    }

    #[tokio::test]
    async fn window_closeout_keeps_invalid_at_set_fact() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // valid_to closed but invalid_at set (contradiction resolver handled it) →
        // excluded by the `invalid_at IS NULL` guard.
        let id = plant_fact(
            &graph,
            "g1",
            "alice",
            "role",
            "Manager",
            now - Duration::days(30),
            Some(now - Duration::days(2)),
            None,
            Some(now - Duration::days(3)),
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(report.count, 0, "invalid_at-set fact must be KEPT");
        assert_eq!(read_expired_at(&graph, id).await, None);
    }

    // ── F-4 anti-loop: dream-generated facts are excluded ────────────────────────

    #[tokio::test]
    async fn window_closeout_excludes_dream_generated() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let id = plant_fact(
            &graph,
            "g1",
            "alice",
            "was",
            "intern",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            1, // is_dream_generated = 1
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(report.count, 0, "dream-generated fact excluded (anti-loop)");
        assert_eq!(read_expired_at(&graph, id).await, None);
    }

    // ── Group scoping: another namespace's closed fact is NOT touched ────────────

    #[tokio::test]
    async fn window_closeout_is_group_scoped() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let other_id = plant_fact(
            &graph,
            "g_other",
            "bob",
            "was",
            "student",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1", // sweep g1, not g_other
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(report.count, 0, "other group's fact not swept");
        assert_eq!(read_expired_at(&graph, other_id).await, None);
    }

    // ── DoD-P1.2 idempotency: second run retires 0 (already excluded) ────────────

    #[tokio::test]
    async fn window_closeout_is_idempotent_second_run_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        plant_fact(
            &graph,
            "g1",
            "alice",
            "lived_in",
            "Portland",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;

        let first = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("first");
        assert_eq!(first.count, 1, "first run retires the closed-window fact");

        let second = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .expect("second");
        assert_eq!(
            second.count, 0,
            "second run retires 0 — the fact now has expired_at set (excluded)"
        );
    }

    // ── LLM-nominate lane is stub-off: requesting it is a documented no-op ───────

    #[tokio::test]
    async fn llm_nominate_lane_is_stub_off_and_warns() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // Plant a closed-window fact so the deterministic lane still fires (proves
        // the LLM flag does not disable the deterministic lane).
        plant_fact(
            &graph,
            "g1",
            "alice",
            "was",
            "grad",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;

        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: true, // request the stub-off lane
            model_id: "gemma4:e4b",
        })
        .await
        .expect("supersession");

        assert_eq!(
            report.count, 1,
            "deterministic lane still retires the closed-window fact"
        );
        assert!(
            report.warnings.iter().any(|w| w.contains("stub-off")),
            "requesting the LLM lane emits a stub-off warning"
        );
    }
}

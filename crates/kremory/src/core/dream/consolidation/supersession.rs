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

use super::substrate::{
    emit_decision, ConsolidationBudget, ConsolidationOpKind, DecisionMode, DecisionRecord, OpReport,
};

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
        // Uniform decision record alongside the existing counter (ADR-070 Fork 2/3,
        // §3.3). No entity refs (the lane is a whole-op stub-off signal, not a
        // per-entity decision); supersession has no dry_run concept → `Applied`.
        emit_decision(DecisionRecord {
            op: ConsolidationOpKind::Supersession,
            mode: DecisionMode::Applied,
            outcome: "llm_nominate_stub_off",
            group_id,
            entity_refs: &[],
            debug_context: None,
        });
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
    let result: Result<Vec<i64>> = async {
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

        // Collect the retired fact ids; the per-fact decision records are emitted ONLY
        // after this txn durably commits (see the Ok arm) so a mid-sweep rollback never
        // leaves an un-revertable decision_total increment (Quinn Phase-B MED-1).
        let mut retired_ids: Vec<i64> = Vec::new();
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
            retired_ids.push(cand.fact_id);
        }
        Ok(retired_ids)
    }
    .await;

    match result {
        Ok(retired_ids) => {
            guard.commit().await?;
            let retired = retired_ids.len();
            // Post-commit emission of the per-fact decision records (ADR-070 Fork 2/3,
            // §3.3): fired ONLY after the txn durably commits so
            // decision_total{op=supersession,outcome=window_closeout} never overcounts
            // a mid-sweep rollback (Quinn Phase-B MED-1) — mirroring the post-commit
            // counter below, which is why the two stay equal. fact id is
            // HIGH-cardinality → trace-only via debug_context (Fork 3); no entity refs
            // are loaded on this fact-level lane; no dry_run → `Applied`.
            for fact_id in &retired_ids {
                emit_decision(DecisionRecord {
                    op: ConsolidationOpKind::Supersession,
                    mode: DecisionMode::Applied,
                    outcome: "window_closeout",
                    group_id,
                    entity_refs: &[],
                    debug_context: Some(format!("fact_id={fact_id}")),
                });
            }
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
        ConsolidationBudget::new(None, None)
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

    // ══════════════════════════════════════════════════════════════════════════
    // PROPERTY / INVARIANT TIER (INV1–INV8) — randomized-input safety proof.
    //
    // The op mutates USER facts, so its safety invariants must hold over RANDOMIZED
    // inputs, not just hand-picked corpus rows. Each of N iterations builds an
    // in-memory graph, plants K random facts across two namespaces, snapshots the
    // facts table BEFORE, runs `supersession` on "gA" only, snapshots AFTER, and
    // asserts eight invariants.
    //
    // PRNG choice: hand-rolled seeded SplitMix64 (NOT proptest's `proptest!` macro).
    // Rationale — the op is `async` (proptest's macro drives sync closures; wrapping
    // an async op per-case needs a runtime bridge that obscures the seed→case
    // mapping the task demands). SplitMix64 is a tiny, well-known, statistically
    // sound seedable generator. Seed = FIXED base ^ iteration index, so the whole
    // test is deterministic + reproducible in CI (no wall-clock / random-device
    // seeding). On any failure we print the exact `seed` + offending fact so the
    // case reproduces from that one line.
    // ══════════════════════════════════════════════════════════════════════════

    /// Deterministic SplitMix64 PRNG (Vigna, public-domain reference). Seedable,
    /// reproducible, no external dep. One `u64` of state; `next_u64` advances it.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `[0, n)` (n > 0). Modulo bias is negligible for the tiny
        /// ranges used here (pools of ≤4, counts ≤30).
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }

        /// Uniform in `[lo, hi]` inclusive.
        fn in_range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.below(hi - lo + 1)
        }

        /// `true` with probability `num/den`.
        fn chance(&mut self, num: u64, den: u64) -> bool {
            self.below(den) < num
        }
    }

    /// A full snapshot row of a planted fact — every column the invariants read.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FactRow {
        id: i64,
        group_id: String,
        valid_to: Option<String>,
        expired_at: Option<String>,
        invalid_at: Option<String>,
        is_dream_generated: i64,
    }

    /// Snapshot the ENTIRE facts table (all groups), ordered by id, into `FactRow`s.
    async fn snapshot_facts(graph: &TemporalGraph) -> Vec<FactRow> {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, group_id, valid_to, expired_at, invalid_at, is_dream_generated \
                 FROM facts ORDER BY id",
                (),
            )
            .await
            .expect("snapshot query");
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.expect("snapshot row") {
            out.push(FactRow {
                id: row.get(0).expect("id"),
                group_id: row.get(1).expect("group_id"),
                valid_to: row.get(2).expect("valid_to"),
                expired_at: row.get(3).expect("expired_at"),
                invalid_at: row.get(4).expect("invalid_at"),
                is_dream_generated: row.get(5).expect("is_dream_generated"),
            });
        }
        out
    }

    /// Would this row match the deterministic window-closeout predicate under `now`
    /// for the swept group `"gA"`? (INV6's mechanical predicate.)
    ///
    /// `valid_to NOT NULL AND valid_to < now AND expired_at NULL AND invalid_at NULL
    ///  AND is_dream_generated = 0 AND group_id = "gA"`.
    /// String compare on RFC3339 UTC == chronological compare (op relies on this).
    fn matches_closeout(row: &FactRow, now_rfc3339: &str) -> bool {
        row.group_id == "gA"
            && row.is_dream_generated == 0
            && row.expired_at.is_none()
            && row.invalid_at.is_none()
            && row.valid_to.as_deref().is_some_and(|vt| vt < now_rfc3339)
    }

    #[tokio::test]
    async fn property_supersession_invariants_over_random_inputs() {
        // FIXED base seed — deterministic + reproducible (no wall-clock seeding).
        const BASE_SEED: u64 = 0x5150_5450_4159_4100; // "PPTPAYA\0"-ish, arbitrary fixed.
        const ITERATIONS: u64 = 300; // ≥ 200 required.

        // Small pools so entities repeat and namespaces collide meaningfully.
        const SUBJECTS: [&str; 4] = ["alice", "bob", "carol", "dave"];
        const PREDICATES: [&str; 4] = ["lives_in", "works_at", "role", "likes"];
        const GROUPS: [&str; 2] = ["gA", "gB"];

        for iter in 0..ITERATIONS {
            // Seed = base ^ iteration index → each case is independently reproducible.
            let seed = BASE_SEED ^ iter;
            let mut rng = SplitMix64::new(seed);

            let graph = TemporalGraph::open_in_memory()
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: open graph: {e}"));

            // `now` for THIS case, captured once so BEFORE/AFTER classification and
            // the op's own `Utc::now()` agree to within test wall-clock (facts are
            // planted at fixed offsets from this anchor, far from the boundary).
            let now = Utc::now();

            // Plant K random facts (K ∈ 3..=30).
            let k = rng.in_range(3, 30);
            for _ in 0..k {
                let base_subject = SUBJECTS[rng.below(SUBJECTS.len() as u64) as usize];
                let predicate = PREDICATES[rng.below(PREDICATES.len() as u64) as usize];
                let group = GROUPS[rng.below(GROUPS.len() as u64) as usize];
                // Namespace the subject by group so the SAME name never crosses
                // namespaces. `insert_entity_with_group` guards against a
                // cross-namespace name collision (ADR-029b bypass surface #2) and
                // returns an error the `ensure_entity` helper swallows — which would
                // then leave the fact's composite FK `(subject, group)` unsatisfied.
                // Subjects still REPEAT within a group (the small pool + the group
                // prefix), so entity reuse is exercised; only cross-group aliasing is
                // avoided (not the property under test here).
                let subject = format!("{group}_{base_subject}");
                let subject = subject.as_str();

                // valid_from: random PAST (1..=365 days ago).
                let valid_from = now - Duration::days(rng.in_range(1, 365) as i64);

                // valid_to ∈ {NULL, random-past, random-future} — weighted so all
                // three occur (≈ 1/3 each). Past window is CLOSED; future is OPEN.
                let valid_to = match rng.below(3) {
                    0 => None,
                    1 => Some(now - Duration::hours(rng.in_range(1, 240) as i64)), // past
                    _ => Some(now + Duration::hours(rng.in_range(1, 240) as i64)), // future
                };

                // expired_at ∈ {NULL, random-past} — ~1/3 already handled.
                let expired_at = if rng.chance(1, 3) {
                    Some(now - Duration::hours(rng.in_range(1, 480) as i64))
                } else {
                    None
                };

                // invalid_at ∈ {NULL, random-past} — ~1/4 contradiction-handled.
                let invalid_at = if rng.chance(1, 4) {
                    Some(now - Duration::hours(rng.in_range(1, 480) as i64))
                } else {
                    None
                };

                // is_dream_generated ∈ {0, 1} — ~1/4 dream output (anti-loop input).
                let is_dream_generated = if rng.chance(1, 4) { 1 } else { 0 };

                let _id = plant_fact(
                    &graph,
                    group,
                    subject,
                    predicate,
                    "val",
                    valid_from,
                    valid_to,
                    expired_at,
                    invalid_at,
                    is_dream_generated,
                )
                .await;
            }

            // ── Snapshot BEFORE ──────────────────────────────────────────────────
            let before = snapshot_facts(&graph).await;

            // `now_rfc3339` used to classify BEFORE rows. Taken AFTER planting but
            // BEFORE the op runs; the op's own `Utc::now()` is a hair later, so any
            // row we classify as "past" (offset ≥ 1h from `now`) is unambiguously
            // past for the op too — no boundary flakiness.
            let now_rfc3339 = Utc::now().to_rfc3339();

            // ── Run the op on "gA" ONLY ──────────────────────────────────────────
            let report = supersession(SupersessionParams {
                graph: &graph,
                group_id: "gA",
                budget: &mut budget(),
                include_llm_nominate: false,
                model_id: "gemma4:e4b",
            })
            .await
            .unwrap_or_else(|e| panic!("seed={seed:#x}: supersession: {e}"));

            // ── Snapshot AFTER ───────────────────────────────────────────────────
            let after = snapshot_facts(&graph).await;

            // Index AFTER by id for O(1) lookup; op never inserts/deletes rows, so
            // BEFORE and AFTER have identical id sets.
            let after_by_id: std::collections::HashMap<i64, &FactRow> =
                after.iter().map(|r| (r.id, r)).collect();
            assert_eq!(
                before.len(),
                after.len(),
                "seed={seed:#x}: op must never insert or delete rows"
            );

            let mut expected_closeout_ids: Vec<i64> = Vec::new();

            for b in &before {
                let a = after_by_id
                    .get(&b.id)
                    .unwrap_or_else(|| panic!("seed={seed:#x}: fact {} vanished after op", b.id));

                let should_close = matches_closeout(b, &now_rfc3339);
                if should_close {
                    expected_closeout_ids.push(b.id);
                }

                // A row is "unchanged" iff every column the op could touch is equal.
                // The op only ever writes `expired_at`; assert the FULL row is equal
                // for KEEP cases (stronger — proves nothing else drifted either).
                let unchanged = **a == *b;

                // INV1 (safety): valid_to IS NULL → UNCHANGED.
                if b.valid_to.is_none() {
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV1 violated: open-ended fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }

                // INV2 (safety): valid_to >= now → UNCHANGED (future / equal window).
                if let Some(vt) = b.valid_to.as_deref() {
                    if vt >= now_rfc3339.as_str() {
                        assert!(
                            unchanged,
                            "seed={seed:#x} INV2 violated: still-valid (valid_to>=now) fact changed.\n  before={b:?}\n  after={a:?}"
                        );
                    }
                }

                // INV3 (no double-handle): already had expired_at OR invalid_at set
                // → UNCHANGED.
                if b.expired_at.is_some() || b.invalid_at.is_some() {
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV3 violated: already-resolved fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }

                // INV4 (namespace isolation): group "gB" → UNCHANGED.
                if b.group_id == "gB" {
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV4 violated: other-namespace (gB) fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }

                // INV5 (anti-loop): is_dream_generated = 1 → UNCHANGED.
                if b.is_dream_generated == 1 {
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV5 violated: dream-generated fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }

                // INV6 (correctness): a matching gA fact must have expired_at == its
                // valid_to AFTER, and NOTHING else changed.
                if should_close {
                    // expired_at now set to exactly valid_to.
                    assert_eq!(
                        a.expired_at, b.valid_to,
                        "seed={seed:#x} INV6 violated: retired fact's expired_at != its valid_to.\n  before={b:?}\n  after={a:?}"
                    );
                    // Nothing else changed: every OTHER column identical.
                    assert_eq!(a.group_id, b.group_id, "seed={seed:#x} INV6: group changed");
                    assert_eq!(
                        a.valid_to, b.valid_to,
                        "seed={seed:#x} INV6: valid_to changed"
                    );
                    assert_eq!(
                        a.invalid_at, b.invalid_at,
                        "seed={seed:#x} INV6: invalid_at changed"
                    );
                    assert_eq!(
                        a.is_dream_generated, b.is_dream_generated,
                        "seed={seed:#x} INV6: is_dream_generated changed"
                    );
                } else {
                    // The COMPLEMENT of INV6: any row that does NOT match the
                    // predicate must be fully unchanged (covers all KEEP paths at
                    // once — INV1–INV5 are the named sub-cases of this).
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV6-complement violated: non-matching fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }
            }

            // INV7 (count): report.count == number of rows matching the predicate.
            assert_eq!(
                report.count,
                expected_closeout_ids.len(),
                "seed={seed:#x} INV7 violated: report.count ({}) != matched-predicate count ({})",
                report.count,
                expected_closeout_ids.len()
            );

            // INV8 (idempotent): a SECOND run on "gA" retires 0 and changes nothing.
            let after_first = snapshot_facts(&graph).await;
            let second = supersession(SupersessionParams {
                graph: &graph,
                group_id: "gA",
                budget: &mut budget(),
                include_llm_nominate: false,
                model_id: "gemma4:e4b",
            })
            .await
            .unwrap_or_else(|e| panic!("seed={seed:#x}: second supersession: {e}"));
            let after_second = snapshot_facts(&graph).await;
            assert_eq!(
                second.count, 0,
                "seed={seed:#x} INV8 violated: second run retired {} (must be 0)",
                second.count
            );
            assert_eq!(
                after_first, after_second,
                "seed={seed:#x} INV8 violated: second run mutated the table"
            );
        }
    }

    // ── Crash / rollback-atomicity: a mid-sweep failure retires NOTHING ───────────
    //
    // The op's whole sweep runs inside ONE `BEGIN IMMEDIATE` (module doc lines
    // 137-138): the SELECT + the batch of `expired_at` writes are atomic, so a
    // mid-sweep failure must leave the table EXACTLY as it was (no partial
    // retirement). This proves RISK-006-class atomicity WITHOUT a production-op
    // seam — we drive the op's OWN documented error path (the `Error::Parse` raised
    // when a candidate's `valid_to` is unparseable, lines 184-191). One valid
    // closeout-eligible fact is written FIRST (its `expired_at` write lands inside
    // the txn), then a second candidate carries a malformed `valid_to` (planted via
    // a raw UPDATE that bypasses the `to_rfc3339` plant path). When the loop reaches
    // the malformed row it throws → the guard rolls back → NEITHER fact ends up with
    // `expired_at` set. Treat-cause note: this exercises a REAL failure path the op
    // already defines; no op contortion was needed to make it testable.
    #[tokio::test]
    async fn mid_sweep_failure_rolls_back_all_retirements() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();

        // Candidate A: a well-formed closed-window fact (lower id → processed first,
        // its retirement write lands inside the txn before the failure).
        let good = plant_fact(
            &graph,
            "g1",
            "alice",
            "lived_in",
            "Boston",
            now - Duration::days(30),
            Some(now - Duration::days(2)),
            None,
            None,
            0,
        )
        .await;

        // Candidate B: also closeout-eligible (valid closed-window plant so the
        // SELECT includes it), THEN corrupt its `valid_to` in-place via a raw UPDATE
        // to a string that is NOT RFC3339 — the op's parse (lines 184-191) will
        // raise `Error::Parse` when it reaches this row, aborting the sweep.
        //
        // The corrupt string MUST still satisfy the SELECT's `valid_to < now` string
        // compare (RFC3339 UTC strings sort lexicographically), or the row is
        // excluded before the parse is ever reached. A `2020-`-prefixed garbage
        // string sorts BEFORE `now` yet fails RFC3339 parse — the exact seam we want.
        let bad = plant_fact(
            &graph,
            "g1",
            "bob",
            "worked_at",
            "Acme",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;
        graph
            .conn
            .execute(
                "UPDATE facts SET valid_to = '2020-99-99garbage' WHERE id = ?1",
                libsql::params![bad],
            )
            .await
            .expect("corrupt valid_to");

        // Run the sweep — it MUST error out on the malformed candidate.
        let result = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await;
        assert!(
            result.is_err(),
            "sweep must fail on the unparseable valid_to (Error::Parse)"
        );

        // Atomicity: the WELL-FORMED fact's retirement was rolled back with the
        // failing one — its `expired_at` must still be NULL (no partial retirement).
        assert_eq!(
            read_expired_at(&graph, good).await,
            None,
            "rollback: the good fact's expired_at must be reverted to NULL"
        );
        // The malformed row is likewise untouched.
        assert_eq!(
            read_expired_at(&graph, bad).await,
            None,
            "rollback: the malformed fact was never retired"
        );
    }

    /// Quinn Phase-B MED-1 lock: the per-fact `decision_total{window_closeout}` records
    /// are emitted POST-COMMIT only, so a mid-sweep rollback must leave the metric at
    /// ZERO — never overcounting the rolled-back "good" candidate. Mirrors the
    /// atomicity test above but asserts the TELEMETRY side (the decision counter never
    /// diverges from what actually committed — the "counters that lie" failure mode).
    #[tokio::test]
    async fn window_closeout_decision_not_emitted_on_rollback() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // Good candidate (processed first, write lands inside the txn) + a candidate
        // whose corrupt `valid_to` throws Error::Parse mid-sweep → whole txn rolls back.
        let _good = plant_fact(
            &graph,
            "g1",
            "alice",
            "lived_in",
            "Boston",
            now - Duration::days(30),
            Some(now - Duration::days(2)),
            None,
            None,
            0,
        )
        .await;
        let bad = plant_fact(
            &graph,
            "g1",
            "bob",
            "worked_at",
            "Acme",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;
        graph
            .conn
            .execute(
                "UPDATE facts SET valid_to = '2020-99-99garbage' WHERE id = ?1",
                libsql::params![bad],
            )
            .await
            .expect("corrupt valid_to");

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let result = supersession(SupersessionParams {
            graph: &graph,
            group_id: "g1",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await;
        assert!(result.is_err(), "sweep must fail on the unparseable valid_to");

        let emitted = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .find_map(|(composite_key, _, _, value)| {
                let key = composite_key.key();
                if key.name() != "kremory.dream.consolidation.decision_total" {
                    return None;
                }
                let labels: std::collections::HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("op").copied() != Some("supersession")
                    || labels.get("outcome").copied() != Some("window_closeout")
                {
                    return None;
                }
                match value {
                    DebugValue::Counter(n) => Some(n),
                    _ => None,
                }
            })
            .unwrap_or(0);
        assert_eq!(
            emitted, 0,
            "window_closeout decision must NOT be emitted for a rolled-back sweep (post-commit only)"
        );
    }

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

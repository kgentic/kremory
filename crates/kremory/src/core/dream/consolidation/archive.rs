//! CONSOLIDATION op — fact archival (ADR-066 §2.4, spec P2).
//!
//! MOVE a long-expired (`expired_at < now - grace_days`), non-stranding fact from
//! live `facts` into the append-only `facts_archive` table (migration 019). For each
//! candidate, inside ONE `BEGIN IMMEDIATE`:
//!
//! 1. `INSERT INTO facts_archive (<explicit cols>) SELECT <matching cols>, now AS
//!    archived_at FROM facts WHERE id = ?` — a PROJECTED column list (NOT `SELECT *`)
//!    because `facts_archive` intentionally drops `embedding` + `access_count` and
//!    adds `archived_at` (spec P2.3, Quinn-P0 note).
//! 2. `DELETE FROM facts_fts WHERE fact_id = ?` — the inbound FTS shadow row
//!    (RISK-003; `facts_fts` is the SOLE inbound `facts.id` reference, analogous to
//!    the `entities_fts` delete on entity merge, `canonicalization.rs:582-593`).
//!    Omitting it orphans FTS hits.
//! 3. `DELETE FROM facts WHERE id = ?`.
//!
//! Zero-LLM. Deterministic. Gated by a ref-count "orphans nothing" guard (P2.2): a
//! fact whose archival would leave EITHER endpoint entity (`subject_id`, and
//! `object_id` when present) with ZERO live facts is KEPT (`sole_binding`, RISK-007).
//!
//! Idempotent (P2.4): a moved fact is gone from `facts`, so the candidate SELECT
//! self-excludes it on the next run (second run archives 0).
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P2.1–P2.4) + §6 (archive corpus) + ADR-066 §2.4.

use chrono::Utc;
use metrics::counter;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::OpReport;

/// A fact eligible for archival: id + both endpoints (for the ref-count guard).
#[derive(Debug, Clone)]
struct ArchiveCandidate {
    fact_id: i64,
    subject_id: String,
    /// `None` for literal-object facts (`object_id IS NULL`).
    object_id: Option<String>,
}

/// The 19 columns of `facts_archive` in order (migration 019, `defs_h.rs:560-569`).
/// `archived_at` is supplied by the INSERT…SELECT as `?1 AS archived_at`; the first
/// 18 are a direct projection of the matching `facts` columns. `facts_archive`
/// intentionally OMITS `facts.embedding` + `facts.access_count` (deprecated, always
/// 0) — hence a PROJECTED list, never `SELECT *` (Quinn-P0 note, P2.3).
const ARCHIVE_INSERT_SQL: &str = "INSERT INTO facts_archive \
     (id, subject_id, predicate, object_id, object_value, properties, \
      valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, \
      source_episode_id, memory_type, content_hash, subject_group_id, object_group_id, \
      is_dream_generated, archived_at) \
     SELECT id, subject_id, predicate, object_id, object_value, properties, \
      valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, \
      source_episode_id, memory_type, content_hash, subject_group_id, object_group_id, \
      is_dream_generated, ?1 AS archived_at \
     FROM facts WHERE id = ?2 AND group_id = ?3";

/// Run the fact-archival op over `group_id` (spec P2).
///
/// `grace_days` is the archival grace window (`DreamOpts.archive_grace_days`, P2.1):
/// only facts whose `expired_at` is older than `now - grace_days` are candidates.
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)` so
/// the deterministic corpus harness (`tests/consolidation_archive_test.rs`) can call
/// it under `feature = "test-utils"` (external test binaries cannot import
/// `pub(crate)` items — E0365). NOT part of the stable public API.
///
/// Args count = 3 (at the `too-many-arguments-threshold`, not over) — plain args,
/// no params struct needed.
#[doc(hidden)]
pub async fn archive(graph: &TemporalGraph, group_id: &str, grace_days: u32) -> Result<OpReport> {
    let mut report = OpReport::default();

    // Cutoff = now - grace_days, as an RFC3339 UTC string. RFC3339 UTC strings sort
    // lexicographically in chronological order (every temporal column is written via
    // `to_rfc3339()`), so `expired_at < cutoff` is a correct temporal compare — the
    // same string-compare soundness the P1 window-closeout lane relies on
    // (`supersession.rs:140-143`).
    let cutoff = archive_cutoff(Utc::now(), grace_days);

    // ── Candidate SELECT (P2.1) — deterministic, zero-LLM ────────────────────────
    // A candidate is expired (`expired_at IS NOT NULL`) and past the grace window
    // (`expired_at < cutoff`). We pull both endpoints for the downstream ref-count
    // guard (P2.2). Read the candidate set BEFORE opening the write transaction so a
    // long candidate list does not hold `BEGIN IMMEDIATE` open across reads.
    let mut rows = graph
        .conn
        .query(
            "SELECT id, subject_id, object_id FROM facts \
             WHERE expired_at IS NOT NULL \
               AND expired_at < ?1 \
               AND group_id = ?2",
            libsql::params![cutoff, group_id],
        )
        .await?;

    let mut candidates: Vec<ArchiveCandidate> = Vec::new();
    while let Some(row) = rows.next().await? {
        let fact_id: i64 = row.get(0)?;
        let subject_id: String = row.get(1)?;
        let object_id: Option<String> = row.get(2)?;
        candidates.push(ArchiveCandidate {
            fact_id,
            subject_id,
            object_id,
        });
    }
    drop(rows);

    if candidates.is_empty() {
        emit_archived_counter(0);
        return Ok(report);
    }

    // ── Move loop (P2.3) — each candidate moved inside its own atomic transaction ──
    let mut archived = 0usize;
    for cand in &candidates {
        // Ref-count guard (P2.2 / RISK-007): archiving this fact must not strand
        // either endpoint entity. Skip (KEEP) if it would leave subject — or object,
        // when present — with ZERO live facts remaining AFTER this fact is removed.
        if would_strand_endpoint(graph, group_id, cand).await? {
            continue;
        }

        let moved = move_fact(graph, cand.fact_id, group_id).await?;
        if moved {
            archived += 1;
        }
    }

    // P2.4 idempotency is structural: a moved fact is gone from `facts`, so the
    // candidate SELECT above excludes it on the next run (second run archives 0).
    emit_archived_counter(archived);
    tracing::info!(
        target: "kremory.dream.consolidation.archive",
        group_id,
        archived,
        candidates = candidates.len(),
        "fact-archival sweep complete"
    );

    report.count = archived;
    Ok(report)
}

/// Compute the archival cutoff instant as an RFC3339 UTC string: `now - grace_days`.
///
/// Pure function (unit-tested, L1). A fact with `expired_at < cutoff` is past the
/// grace window and eligible for archival; a fact at exactly the cutoff is NOT (the
/// SELECT uses strict `<`, so the boundary itself is retained — spec grace-boundary
/// test: `now - grace + 1s` KEEP, `now - grace - 1s` ARCHIVE).
fn archive_cutoff(now: chrono::DateTime<Utc>, grace_days: u32) -> String {
    (now - chrono::Duration::days(i64::from(grace_days))).to_rfc3339()
}

/// Ref-count guard (P2.2 / RISK-007): would archiving `cand` strand either endpoint?
///
/// An endpoint entity is STRANDED if removing this fact would leave it with ZERO live
/// facts. "Live" = `expired_at IS NULL AND invalid_at IS NULL`. Because `cand` is a
/// candidate it is itself already expired (`expired_at IS NOT NULL`), so it is NOT
/// live and cannot be its own endpoint's last LIVE fact — the guard therefore counts
/// live facts referencing the endpoint (as subject OR object) and skips archival only
/// when that count is ZERO. Pure structural SQL, zero-LLM.
///
/// Returns `true` when the fact must be KEPT (archiving it would strand an endpoint).
async fn would_strand_endpoint(
    graph: &TemporalGraph,
    group_id: &str,
    cand: &ArchiveCandidate,
) -> Result<bool> {
    if live_fact_count_for_entity(graph, group_id, &cand.subject_id).await? == 0 {
        return Ok(true);
    }
    if let Some(object_id) = &cand.object_id {
        if live_fact_count_for_entity(graph, group_id, object_id).await? == 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Count live facts (`expired_at IS NULL AND invalid_at IS NULL`) in `group_id` that
/// reference `entity_id` as EITHER `subject_id` or `object_id`. Used by the ref-count
/// guard (P2.2). A candidate fact is expired, so it never contributes to this count.
async fn live_fact_count_for_entity(
    graph: &TemporalGraph,
    group_id: &str,
    entity_id: &str,
) -> Result<i64> {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts \
             WHERE group_id = ?1 \
               AND (subject_id = ?2 OR object_id = ?2) \
               AND expired_at IS NULL \
               AND invalid_at IS NULL",
            libsql::params![group_id, entity_id],
        )
        .await?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| crate::core::error::Error::Parse("COUNT(*) returned no row".to_string()))?;
    Ok(row.get::<i64>(0)?)
}

/// Move ONE fact from `facts` → `facts_archive` inside a single `BEGIN IMMEDIATE`
/// (RISK-006 atomicity): INSERT the projected columns + `archived_at`, DELETE the
/// `facts_fts` shadow row (RISK-003), then DELETE the live row. All-or-nothing.
///
/// Returns `true` when a row was moved (`DELETE FROM facts` affected 1 row). A
/// candidate that vanished between the SELECT and this move (concurrent delete)
/// yields `false` — the INSERT…SELECT copied nothing and the DELETE affected nothing,
/// so `facts_archive` gains no orphan.
async fn move_fact(graph: &TemporalGraph, fact_id: i64, group_id: &str) -> Result<bool> {
    let now = Utc::now().to_rfc3339();
    let guard = graph.begin_immediate_if_needed().await?;
    let result: Result<bool> = async {
        // 1. INSERT projected columns + archived_at. Scoped by `id AND group_id`
        //    (Quinn-P2 Q2) so the INSERT and the DELETE (step 3) are group-symmetric:
        //    a fact_id whose group_id ≠ arg copies nothing here AND deletes nothing
        //    below → no orphan duplicate in facts_archive. If the row already vanished
        //    the INSERT…SELECT copies zero rows (no orphan created).
        graph
            .conn
            .execute(
                ARCHIVE_INSERT_SQL,
                libsql::params![now.clone(), fact_id, group_id],
            )
            .await?;
        // 2. DELETE the inbound FTS shadow row (RISK-003). `facts_fts` is the SOLE
        //    inbound `facts.id` reference — verified sole shadow to clean.
        graph
            .conn
            .execute(
                "DELETE FROM facts_fts WHERE fact_id = ?1",
                libsql::params![fact_id],
            )
            .await?;
        // 3. DELETE the live row (scoped by group_id defensively — the candidate was
        //    already group-filtered, but this makes the mutation self-documenting).
        let deleted = graph
            .conn
            .execute(
                "DELETE FROM facts WHERE id = ?1 AND group_id = ?2",
                libsql::params![fact_id, group_id],
            )
            .await?;
        Ok(deleted == 1)
    }
    .await;

    match result {
        Ok(moved) => {
            guard.commit().await?;
            Ok(moved)
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e)
        }
    }
}

/// Emit the source-attributed archival counter (P2.5). The o11y cross-check asserts
/// this equals `OpReport.count`.
fn emit_archived_counter(archived: usize) {
    counter!("kremory.dream.consolidation.facts_archived_total").increment(archived as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::InsertEntityWithGroupParams;
    use crate::core::schema::TemporalGraph;
    use chrono::{DateTime, Duration};

    // ── Plant helpers (direct SQL — full control over temporal columns) ──────────
    //
    // `facts` carries the composite FK `(subject_id) REFERENCES entities(id)`
    // (`schema.rs:505`); FK enforcement is ON, so the subject entity row MUST exist
    // first. Object-entity facts additionally reference `entities(id)` via
    // `object_id` (`schema.rs:506`), so an object entity is planted when present.

    async fn ensure_entity(graph: &TemporalGraph, group_id: &str, id: &str) {
        let _ = graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({}),
                group_id: Some(group_id),
            })
            .await;
    }

    /// Bundled fact-plant params (test helper — args-as-object to satisfy the
    /// threshold-3 lint WITHOUT a `#[allow]`).
    struct PlantFact<'a> {
        group_id: &'a str,
        subject: &'a str,
        predicate: &'a str,
        /// `Some` → relational fact (object_id set + object entity planted).
        object_id: Option<&'a str>,
        object_value: Option<&'a str>,
        valid_from: DateTime<Utc>,
        expired_at: Option<DateTime<Utc>>,
        invalid_at: Option<DateTime<Utc>>,
        /// When `true` also insert the `facts_fts` shadow row (mirrors real ingest,
        /// `facts.rs:370-377`) so FTS-shadow-deletion is observable.
        with_fts_shadow: bool,
    }

    /// Insert a `facts` row with explicit temporal columns; returns its id.
    async fn plant_fact(graph: &TemporalGraph, p: PlantFact<'_>) -> i64 {
        ensure_entity(graph, p.group_id, p.subject).await;
        if let Some(obj) = p.object_id {
            ensure_entity(graph, p.group_id, obj).await;
        }
        let now = Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_id, object_value, valid_from, recorded_at, \
                  expired_at, invalid_at, group_id, subject_group_id, object_group_id, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 1.0)",
                libsql::params![
                    p.subject,
                    p.predicate,
                    p.object_id,
                    p.object_value,
                    p.valid_from.to_rfc3339(),
                    now,
                    p.expired_at.map(|v| v.to_rfc3339()),
                    p.invalid_at.map(|v| v.to_rfc3339()),
                    p.group_id,
                    p.group_id,
                    p.object_id.map(|_| p.group_id),
                ],
            )
            .await
            .expect("plant fact");
        let mut rows = graph
            .conn
            .query("SELECT last_insert_rowid()", ())
            .await
            .expect("rowid");
        let fact_id: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("present")
            .get(0)
            .expect("id");
        drop(rows);

        if p.with_fts_shadow {
            graph
                .conn
                .execute(
                    "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
                    libsql::params![fact_id, p.predicate, p.object_value.unwrap_or("")],
                )
                .await
                .expect("plant fts shadow");
        }
        fact_id
    }

    async fn live_facts_count(graph: &TemporalGraph, group_id: &str) -> i64 {
        count(
            graph,
            "SELECT COUNT(*) FROM facts WHERE group_id = ?1",
            group_id,
        )
        .await
    }

    async fn archive_count(graph: &TemporalGraph, group_id: &str) -> i64 {
        count(
            graph,
            "SELECT COUNT(*) FROM facts_archive WHERE group_id = ?1",
            group_id,
        )
        .await
    }

    async fn fts_shadow_count_for(graph: &TemporalGraph, fact_id: i64) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM facts_fts WHERE fact_id = ?1",
                libsql::params![fact_id],
            )
            .await
            .expect("query facts_fts");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("count")
    }

    async fn count(graph: &TemporalGraph, sql: &str, group_id: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(sql, libsql::params![group_id])
            .await
            .expect("count query");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("count")
    }

    // ── L1 unit: cutoff computation (pure fn) ────────────────────────────────────

    #[test]
    fn archive_cutoff_is_now_minus_grace_days() {
        let now = DateTime::parse_from_rfc3339("2026-07-03T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let cutoff = archive_cutoff(now, 90);
        let expected = (now - Duration::days(90)).to_rfc3339();
        assert_eq!(cutoff, expected, "cutoff must be now - 90d");
        // Cutoff string sorts chronologically: a fact expired 91d ago (< cutoff) is
        // eligible; a fact expired 89d ago (> cutoff) is not.
        let expired_91d = (now - Duration::days(91)).to_rfc3339();
        let expired_89d = (now - Duration::days(89)).to_rfc3339();
        assert!(expired_91d < cutoff, "91d-expired is past the 90d grace");
        assert!(expired_89d > cutoff, "89d-expired is within the 90d grace");
    }

    #[test]
    fn archive_cutoff_zero_grace_is_now() {
        let now = Utc::now();
        let cutoff = archive_cutoff(now, 0);
        assert_eq!(cutoff, now.to_rfc3339(), "zero grace → cutoff == now");
    }

    // ── L2: long-expired-safe fact IS archived (happy path, DoD-P2.1/P2.3) ───────

    #[tokio::test]
    async fn archives_long_expired_safe_fact_and_removes_fts_shadow() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // A long-expired value-fact on `alice`. She has a SEPARATE live fact so
        // archiving this one does NOT strand her (ref-count guard passes).
        let archived_id = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "lived_in",
                object_id: None,
                object_value: Some("Boston"),
                valid_from: now - Duration::days(400),
                expired_at: Some(now - Duration::days(120)), // > 90d grace
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;
        // Live anchor so alice is not stranded.
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some("coffee"),
                valid_from: now - Duration::days(10),
                expired_at: None,
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;

        assert_eq!(fts_shadow_count_for(&graph, archived_id).await, 1);
        let facts_before = live_facts_count(&graph, "g1").await;

        let report = archive(&graph, "g1", 90).await.expect("archive");

        assert_eq!(report.count, 1, "one long-expired fact archived");
        // facts decreased by facts_archived.
        assert_eq!(
            live_facts_count(&graph, "g1").await,
            facts_before - 1,
            "live facts decreased by archived count"
        );
        // facts_archive increased by the same.
        assert_eq!(archive_count(&graph, "g1").await, 1, "one row in archive");
        // FTS shadow row removed for the archived id (RISK-003).
        assert_eq!(
            fts_shadow_count_for(&graph, archived_id).await,
            0,
            "facts_fts shadow row removed for the archived fact"
        );
        // The archived row's projected columns landed (spot-check subject + expired).
        let mut rows = graph
            .conn
            .query(
                "SELECT subject_id, object_value, expired_at, archived_at \
                 FROM facts_archive WHERE id = ?1",
                libsql::params![archived_id],
            )
            .await
            .expect("query archive row");
        let row = rows.next().await.expect("row").expect("present");
        let subject: String = row.get(0).expect("subject");
        let object_value: String = row.get(1).expect("object_value");
        let expired: Option<String> = row.get(2).expect("expired_at");
        let archived_at: String = row.get(3).expect("archived_at");
        assert_eq!(subject, "alice");
        assert_eq!(object_value, "Boston");
        assert!(expired.is_some(), "expired_at carried into archive");
        assert!(!archived_at.is_empty(), "archived_at stamped");
    }

    // ── L2: within-grace expired fact is KEPT (DoD-P2.1 boundary) ────────────────

    #[tokio::test]
    async fn keeps_within_grace_expired_fact() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let id = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "was",
                object_id: None,
                object_value: Some("intern"),
                valid_from: now - Duration::days(200),
                expired_at: Some(now - Duration::days(30)), // < 90d grace → KEEP
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;
        // live anchor so a strand-guard can never be the reason it's kept.
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some("tea"),
                valid_from: now - Duration::days(5),
                expired_at: None,
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(report.count, 0, "within-grace fact KEPT (EXACT 0)");
        assert_eq!(archive_count(&graph, "g1").await, 0, "nothing archived");
        assert_eq!(
            fts_shadow_count_for(&graph, id).await,
            1,
            "kept fact's shadow row untouched"
        );
    }

    // ── L2: unexpired fact is KEPT (expired_at NULL, DoD-P2.1) ───────────────────

    #[tokio::test]
    async fn keeps_unexpired_fact() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "lives_in",
                object_id: None,
                object_value: Some("Denver"),
                valid_from: now - Duration::days(400),
                expired_at: None, // never expired → not a candidate
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(report.count, 0, "unexpired fact KEPT (EXACT 0)");
        assert_eq!(archive_count(&graph, "g1").await, 0);
    }

    // ── L2: a WHOLLY-EXPIRED entity never drains (Quinn-P2 Q1) ────────────────────
    // An entity all of whose facts are expired-past-grace has ZERO live facts, so the
    // ref-count guard (DoD-P2.2: "would leave an endpoint with 0 live facts → keep")
    // KEEPS every one of them — a fully-quiet entity is never fully archived away, it
    // stays traceable in `facts`. Conservative + spec-conformant; asserted EXACT.
    #[tokio::test]
    async fn wholly_expired_entity_retains_all_its_facts() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // `quiet` has TWO long-expired facts and NO live fact.
        for pred in ["was", "did"] {
            plant_fact(
                &graph,
                PlantFact {
                    group_id: "g1",
                    subject: "quiet",
                    predicate: pred,
                    object_id: None,
                    object_value: Some("history"),
                    valid_from: now - Duration::days(500),
                    expired_at: Some(now - Duration::days(200)), // past the 90d grace
                    invalid_at: None,
                    with_fts_shadow: true,
                },
            )
            .await;
        }

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(
            report.count, 0,
            "wholly-expired entity: BOTH facts KEPT (archiving either strands `quiet` to 0 live) — EXACT 0"
        );
        assert_eq!(archive_count(&graph, "g1").await, 0, "nothing moved to archive");
    }

    // ── L2: sole-binding fact is KEPT (ref-count guard, RISK-007) ────────────────

    #[tokio::test]
    async fn keeps_sole_binding_fact_to_avoid_stranding_entity() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // `orphan` has EXACTLY ONE fact (long-expired). Archiving it would strand
        // `orphan` with zero live facts → the guard must KEEP it.
        let id = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "orphan",
                predicate: "was",
                object_id: None,
                object_value: Some("ghost"),
                valid_from: now - Duration::days(400),
                expired_at: Some(now - Duration::days(200)), // way past grace
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(
            report.count, 0,
            "sole-binding fact KEPT to avoid stranding (EXACT 0)"
        );
        assert_eq!(archive_count(&graph, "g1").await, 0, "nothing archived");
        assert_eq!(
            fts_shadow_count_for(&graph, id).await,
            1,
            "kept fact's shadow row untouched"
        );
    }

    // ── L2: relational sole-binding (object endpoint would strand) is KEPT ───────

    #[tokio::test]
    async fn keeps_fact_when_object_endpoint_would_strand() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // alice --employed_by--> acme (long-expired). alice has a live anchor, but
        // acme's ONLY reference is this expired fact → archiving would strand acme.
        let _employed_by = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "employed_by",
                object_id: Some("acme"),
                object_value: None,
                valid_from: now - Duration::days(400),
                expired_at: Some(now - Duration::days(150)),
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some("coffee"),
                valid_from: now - Duration::days(5),
                expired_at: None,
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(
            report.count, 0,
            "kept — object endpoint `acme` would be stranded (EXACT 0)"
        );
    }

    // ── L2: grace-window boundary (now-grace+1s KEEP, now-grace-1s ARCHIVE) ──────

    #[tokio::test]
    async fn grace_window_boundary_1s_either_side() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        // Live anchor for `alice` so neither boundary fact is kept for stranding.
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some("coffee"),
                valid_from: now - Duration::days(1),
                expired_at: None,
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;
        // Inside grace by 1s: expired at (now - 90d + 1s) → > cutoff → KEEP.
        let inside = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "a",
                object_id: None,
                object_value: Some("in"),
                valid_from: now - Duration::days(200),
                expired_at: Some(now - Duration::days(90) + Duration::seconds(1)),
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;
        // Past grace by 1s: expired at (now - 90d - 1s) → < cutoff → ARCHIVE.
        let outside = plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "b",
                object_id: None,
                object_value: Some("out"),
                valid_from: now - Duration::days(200),
                expired_at: Some(now - Duration::days(90) - Duration::seconds(1)),
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive");
        assert_eq!(report.count, 1, "only the past-grace fact archived");
        // The inside-grace fact stays live; the outside-grace fact is gone. Query by
        // id directly (the `count` helper binds group_id, not id).
        assert!(fact_exists(&graph, inside).await, "inside-grace fact KEPT");
        assert!(
            !fact_exists(&graph, outside).await,
            "past-grace fact ARCHIVED (gone from facts)"
        );
        assert!(
            archive_row_exists(&graph, outside).await,
            "past-grace fact landed in facts_archive"
        );
    }

    async fn fact_exists(graph: &TemporalGraph, id: i64) -> bool {
        row_exists(graph, "SELECT 1 FROM facts WHERE id = ?1", id).await
    }
    async fn archive_row_exists(graph: &TemporalGraph, id: i64) -> bool {
        row_exists(graph, "SELECT 1 FROM facts_archive WHERE id = ?1", id).await
    }
    async fn row_exists(graph: &TemporalGraph, sql: &str, id: i64) -> bool {
        let mut rows = graph
            .conn
            .query(sql, libsql::params![id])
            .await
            .expect("exists query");
        rows.next().await.expect("row").is_some()
    }

    // ── L2: idempotency (P2.4) — second run archives 0 ───────────────────────────

    #[tokio::test]
    async fn second_run_archives_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "lived_in",
                object_id: None,
                object_value: Some("Boston"),
                valid_from: now - Duration::days(400),
                expired_at: Some(now - Duration::days(120)),
                invalid_at: None,
                with_fts_shadow: true,
            },
        )
        .await;
        // live anchor.
        plant_fact(
            &graph,
            PlantFact {
                group_id: "g1",
                subject: "alice",
                predicate: "likes",
                object_id: None,
                object_value: Some("coffee"),
                valid_from: now - Duration::days(1),
                expired_at: None,
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;

        let first = archive(&graph, "g1", 90).await.expect("first");
        assert_eq!(first.count, 1, "first run archives the long-expired fact");
        let second = archive(&graph, "g1", 90).await.expect("second");
        assert_eq!(
            second.count, 0,
            "second run archives 0 — candidate already moved out of facts"
        );
        assert_eq!(archive_count(&graph, "g1").await, 1, "no double-archive");
    }

    // ── L2: group scoping — another namespace's expired fact is not touched ──────

    #[tokio::test]
    async fn archive_is_group_scoped() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let now = Utc::now();
        let other = plant_fact(
            &graph,
            PlantFact {
                group_id: "g_other",
                subject: "bob",
                predicate: "was",
                object_id: None,
                object_value: Some("student"),
                valid_from: now - Duration::days(400),
                expired_at: Some(now - Duration::days(200)),
                invalid_at: None,
                with_fts_shadow: false,
            },
        )
        .await;

        let report = archive(&graph, "g1", 90).await.expect("archive g1");
        assert_eq!(report.count, 0, "g1 sweep does not touch g_other");
        assert!(fact_exists(&graph, other).await, "g_other fact untouched");
    }

    // ── Empty group: archives 0 (no candidates) ──────────────────────────────────

    #[tokio::test]
    async fn empty_group_archives_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let report = archive(&graph, "g_empty", 90).await.expect("archive");
        assert_eq!(report.count, 0, "empty group → 0 archived");
        assert!(report.warnings.is_empty());
    }
}

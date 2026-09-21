use super::*;

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

/// Lock: the per-fact `decision_total{window_closeout}` records
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
    assert!(
        result.is_err(),
        "sweep must fail on the unparseable valid_to"
    );

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

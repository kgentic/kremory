use super::*;

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

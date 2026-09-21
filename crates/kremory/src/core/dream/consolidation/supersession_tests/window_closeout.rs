use super::*;

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

// ── TD-177: bulk-invalidation interlock ──────────────────────────────────────

/// The false-positive case FIRST (TD-177 explicit requirement): an ordinary
/// window close-out — a handful of closed-window facts in a namespace well
/// over the floor — must NOT trip the guard. 2 closed-window facts out of 20
/// live facts total (10%) is ordinary maintenance, not an incident.
#[tokio::test]
async fn window_closeout_ordinary_batch_under_threshold_is_not_blocked() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let now = Utc::now();

    // 18 padding facts, still open-ended (not closeout candidates).
    for i in 0..18 {
        plant_fact(
            &graph,
            "g1",
            &format!("entity{i}"),
            "status",
            "active",
            now - Duration::days(30),
            None,
            None,
            None,
            0,
        )
        .await;
    }
    // 2 closed-window candidates (10% of the 20 live facts).
    let closed1 = plant_fact(
        &graph,
        "g1",
        "bob",
        "lived_in",
        "Portland",
        now - Duration::days(30),
        Some(now - Duration::days(1)),
        None,
        None,
        0,
    )
    .await;
    let closed2 = plant_fact(
        &graph,
        "g1",
        "carol",
        "lived_in",
        "Seattle",
        now - Duration::days(30),
        Some(now - Duration::days(1)),
        None,
        None,
        0,
    )
    .await;

    let result = supersession(SupersessionParams {
        graph: &graph,
        group_id: "g1",
        budget: &mut budget(),
        include_llm_nominate: false,
        model_id: "gemma4:e4b",
    })
    .await
    .expect("supersession");

    assert_eq!(
        result.count, 2,
        "10% of the namespace is ordinary maintenance — must retire normally, not be blocked"
    );
    assert!(read_expired_at(&graph, closed1).await.is_some());
    assert!(read_expired_at(&graph, closed2).await.is_some());
}

/// The genuine-incident case: a single window-closeout sweep wanting to
/// retire well over 25% of a namespace's live facts in one pass is refused.
/// The candidate facts stay live (expired_at still NULL) rather than being
/// silently retired — the sweep completes (returns Ok(0)), it just applies
/// nothing this round.
#[tokio::test]
async fn window_closeout_blocks_a_bulk_wipe_and_leaves_facts_live() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let now = Utc::now();

    // 14 padding facts (still open-ended) + 6 closed-window candidates = 20
    // live facts total. 6/20 = 30%, above the 25% threshold.
    for i in 0..14 {
        plant_fact(
            &graph,
            "g1",
            &format!("entity{i}"),
            "status",
            "active",
            now - Duration::days(30),
            None,
            None,
            None,
            0,
        )
        .await;
    }
    let mut candidate_ids = Vec::new();
    for i in 0..6 {
        let id = plant_fact(
            &graph,
            "g1",
            &format!("closed_entity{i}"),
            "lived_in",
            "Nowhere",
            now - Duration::days(30),
            Some(now - Duration::days(1)),
            None,
            None,
            0,
        )
        .await;
        candidate_ids.push(id);
    }

    let result = supersession(SupersessionParams {
        graph: &graph,
        group_id: "g1",
        budget: &mut budget(),
        include_llm_nominate: false,
        model_id: "gemma4:e4b",
    })
    .await
    .expect("supersession must succeed even when the interlock blocks the batch");

    assert_eq!(
        result.count, 0,
        "TD-177: the interlock must block this batch (6 of 20 = 30%, above threshold)"
    );
    for id in candidate_ids {
        assert!(
            read_expired_at(&graph, id).await.is_none(),
            "blocked candidate {id} must remain live, not be silently retired"
        );
    }
}

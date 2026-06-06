#![allow(clippy::unwrap_used, clippy::expect_used)]
//! F2 SPIKE — see .ai-docs/adrs/rql/adr-029-namespace-policy/cycle-1-spike-f2-cross-namespace-dedup.md
//!
//! Originally: kremory's entity dedup was keyed on `normalize_name(label)` → used as
//! the `rql_entities.id` PRIMARY KEY (single TEXT PK, no namespace scope). Same name
//! in two namespaces would silently share the row (bypass surface #1).
//!
//! ADR-029b Decision 1 (migration 004): composite PK (id, group_id) — same name can
//! exist independently in each namespace.
//!
//! ADR-029b §3.2 (bypass surface #2): `insert_entity_with_group` now returns
//! `CrossNamespaceCollision` when the same `id` already exists under a DIFFERENT
//! `group_id` (to prevent accidental cross-namespace name reuse).
//!
//! These tests are regression anchors for both fixes (G_v015b_22).

use kremory::core::schema::TemporalGraph;

/// G_v015b_22 regression: same surface name into two namespaces.
///
/// Pre-029b: second insert silently reused / collided the row (bypass surface #1 + #2).
/// Post-029b: second insert returns `CrossNamespaceCollision` (bypass surface #2 closed).
/// ns_a still has its independent row; ns_b has no row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_same_name_two_namespaces() {
    let g = TemporalGraph::open_in_memory().await.expect("open in-mem");

    let entity_id = "acme corp"; // normalize_name("Acme Corp") -> "acme corp"
    let ns_a = "audit-ns-a";
    let ns_b = "tenant-b";

    // 1st write: ingest "Acme Corp" into namespace-A.
    let r1 = g
        .insert_entity_with_group(
            entity_id,
            0,
            serde_json::json!({"name": "Acme Corp", "context": "first mention in ns-a"}),
            Some(ns_a),
        )
        .await;
    println!("SPIKE F2 — INSERT 1 (ns-a) result: {:?}", r1);
    assert!(r1.is_ok(), "first insert should succeed");

    // 2nd write: same name, different namespace — must now return CrossNamespaceCollision.
    // ADR-029b §3.2 closes bypass surface #2: explicit error instead of silent merge.
    let r2 = g
        .insert_entity_with_group(
            entity_id,
            0,
            serde_json::json!({"name": "Acme Corp", "context": "second mention in ns-b"}),
            Some(ns_b),
        )
        .await;
    println!("SPIKE F2 — INSERT 2 (ns-b) result: {:?}", r2);
    assert!(
        r2.is_err(),
        "second insert into a different namespace must fail with CrossNamespaceCollision"
    );
    let err_msg = format!("{}", r2.unwrap_err());
    assert!(
        err_msg.contains("cross-namespace collision"),
        "expected CrossNamespaceCollision, got: {err_msg}"
    );

    // ns_a still has exactly 1 row; ns_b has 0 rows (insert was blocked).
    let ns_a_view = g.list_entities_in_group(ns_a).await.expect("list ns_a");
    let ns_b_view = g.list_entities_in_group(ns_b).await.expect("list ns_b");
    println!(
        "SPIKE F2 — ns_a view: {} entities ({:?})",
        ns_a_view.len(),
        ns_a_view
            .iter()
            .map(|e| (&e.id, &e.group_id))
            .collect::<Vec<_>>()
    );
    println!(
        "SPIKE F2 — ns_b view: {} entities ({:?})",
        ns_b_view.len(),
        ns_b_view
            .iter()
            .map(|e| (&e.id, &e.group_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(ns_a_view.len(), 1, "ns_a must retain its entity");
    assert_eq!(
        ns_b_view.len(),
        0,
        "ns_b must have no entity (insert was blocked)"
    );
}

/// Q4: does `reassign_entity_group_dangerous` succeed and overwrite the stored
/// group_id when `bypass_policy = true` (ADR-029b Decision 5)?
/// Composite-PK variant: requires `old_group_id` for the WHERE clause.
/// `bypass_policy = true` used here because this is a direct substrate test
/// (no production path — migration tooling only).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_reassign_entity_group_dangerous_overwrite() {
    let g = TemporalGraph::open_in_memory().await.expect("open");

    let entity_id = "acme corp";
    let ns_a = "audit-ns-a";
    let ns_b = "tenant-b";

    g.insert_entity_with_group(
        entity_id,
        0,
        serde_json::json!({"name": "Acme Corp"}),
        Some(ns_a),
    )
    .await
    .expect("seed insert into ns_a");

    // Probe: read back the initial group_id.
    let before = g
        .get_entity(entity_id)
        .await
        .expect("get")
        .expect("present");
    println!("SPIKE F2 Q4 — group_id BEFORE move: {:?}", before.group_id);

    // Move the entity to ns_b via bypass_policy=true (migration tooling path).
    // Production code uses bypass_policy=false and will get a policy check.
    let n = g
        .reassign_entity_group_dangerous(entity_id, ns_a, Some(ns_b), true)
        .await
        .expect("move group");
    println!(
        "SPIKE F2 Q4 — rows updated by reassign_entity_group_dangerous: {}",
        n
    );

    let after = g
        .get_entity(entity_id)
        .await
        .expect("get")
        .expect("present");
    println!("SPIKE F2 Q4 — group_id AFTER move: {:?}", after.group_id);
}

/// Q5 (control): second `insert_entity_with_group` for the same name in a DIFFERENT
/// namespace must now return `CrossNamespaceCollision` (ADR-029b §3.2 bypass surface #2
/// fix). The old behaviour was a swallowed UNIQUE constraint error; the new behaviour is
/// an explicit, loud error so callers know cross-namespace name collision was attempted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_second_insert_returns_cross_namespace_collision() {
    let g = TemporalGraph::open_in_memory().await.expect("open");

    let entity_id = "acme corp";

    g.insert_entity_with_group(
        entity_id,
        0,
        serde_json::json!({"name": "Acme Corp"}),
        Some("ns-a"),
    )
    .await
    .expect("first insert");

    let err = g
        .insert_entity_with_group(
            entity_id,
            0,
            serde_json::json!({"name": "Acme Corp"}),
            Some("ns-b"),
        )
        .await
        .expect_err("second insert into different namespace must fail (CrossNamespaceCollision)");

    println!("SPIKE F2 — second insert error: {}", err);
    let msg = format!("{err}");
    assert!(
        msg.contains("cross-namespace collision"),
        "expected CrossNamespaceCollision error, got: {msg}"
    );
}

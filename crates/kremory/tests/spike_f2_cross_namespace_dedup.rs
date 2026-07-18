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
//! ADR-029d (supersedes ADR-029b §3 real-entity fail-fast): entity identity is
//! per-namespace-open. `insert_entity_with_group` for the same `id` under a
//! DIFFERENT `group_id` now SUCCEEDS (two independent rows under the composite PK)
//! — the competitor-standard model that fixes multi-tenant/multi-conversation
//! ingest. Cross-namespace name reuse is a non-blocking observability signal
//! (`rql.entity.cross_namespace_collision_total` counter), not an error.
//!
//! These tests are regression anchors: composite-PK storage (029b Decision 1,
//! `G_v015b_7`: two rows) is now the end-to-end behaviour, not just storage-layer.

use kremory::core::graph::{InsertEntityWithGroupParams, ReassignEntityGroupDangerousParams};
use kremory::core::schema::TemporalGraph;

/// Regression: same surface name into two namespaces.
///
/// Pre-029b: second insert silently reused / collided the row (bypass surface #1).
/// 029b: second insert returned `CrossNamespaceCollision` (real-entity block).
/// ADR-029d (current): second insert SUCCEEDS — per-namespace-open; each namespace
/// keeps its own independent row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_same_name_two_namespaces() {
    let g = TemporalGraph::open_in_memory().await.expect("open in-mem");

    let entity_id = "acme corp"; // normalize_name("Acme Corp") -> "acme corp"
    let ns_a = "audit-ns-a";
    let ns_b = "tenant-b";

    // 1st write: ingest "Acme Corp" into namespace-A.
    let r1 = g
        .insert_entity_with_group(InsertEntityWithGroupParams { id: entity_id, entity_type_id: 0, properties: serde_json::json!({"name": "Acme Corp", "context": "first mention in ns-a"}), group_id: Some(ns_a) })
        .await;
    println!("SPIKE F2 — INSERT 1 (ns-a) result: {:?}", r1);
    assert!(r1.is_ok(), "first insert should succeed");

    // 2nd write: same name, different namespace — ADR-029d: SUCCEEDS (per-namespace-open).
    let r2 = g
        .insert_entity_with_group(InsertEntityWithGroupParams { id: entity_id, entity_type_id: 0, properties: serde_json::json!({"name": "Acme Corp", "context": "second mention in ns-b"}), group_id: Some(ns_b) })
        .await;
    println!("SPIKE F2 — INSERT 2 (ns-b) result: {:?}", r2);
    assert!(
        r2.is_ok(),
        "ADR-029d: second insert into a different namespace must SUCCEED (per-namespace-open)"
    );

    // ns_a and ns_b each retain their own independent row (composite PK, two rows).
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
        1,
        "ADR-029d: ns_b must have its own independent entity row (per-namespace-open)"
    );

    // ADR-029d P4: the two rows carry INDEPENDENT data (not a shared/collapsed row).
    // Each namespace's row keeps its own `properties.context`, proving they are two
    // distinct enriched rows keyed by (id, group_id), not one row seen twice.
    let ctx_a = ns_a_view[0].properties.get("context").and_then(|v| v.as_str());
    let ctx_b = ns_b_view[0].properties.get("context").and_then(|v| v.as_str());
    assert_eq!(ctx_a, Some("first mention in ns-a"), "ns_a keeps its own properties");
    assert_eq!(ctx_b, Some("second mention in ns-b"), "ns_b keeps its own properties");
    assert_eq!(ns_a_view[0].group_id.as_deref(), Some(ns_a));
    assert_eq!(ns_b_view[0].group_id.as_deref(), Some(ns_b));
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

    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: entity_id,
        entity_type_id: 0,
        properties: serde_json::json!({"name": "Acme Corp"}),
        group_id: Some(ns_a),
    })
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
        .reassign_entity_group_dangerous(ReassignEntityGroupDangerousParams {
            id: entity_id,
            old_group_id: ns_a,
            new_group_id: Some(ns_b),
            bypass_policy: true,
        })
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

/// Q5 (control): ADR-029d — second `insert_entity_with_group` for the same name in
/// a DIFFERENT namespace SUCCEEDS (per-namespace-open); both independent rows exist.
/// (Was `spike_f2_second_insert_returns_cross_namespace_collision` under 029b.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_second_insert_succeeds_per_namespace_open() {
    let g = TemporalGraph::open_in_memory().await.expect("open");

    let entity_id = "acme corp";

    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: entity_id,
        entity_type_id: 0,
        properties: serde_json::json!({"name": "Acme Corp"}),
        group_id: Some("ns-a"),
    })
    .await
    .expect("first insert");

    g.insert_entity_with_group(InsertEntityWithGroupParams {
        id: entity_id,
        entity_type_id: 0,
        properties: serde_json::json!({"name": "Acme Corp"}),
        group_id: Some("ns-b"),
    })
    .await
    .expect("ADR-029d: second insert into a different namespace must SUCCEED");

    // Both namespaces have their own independent row.
    let ns_a = g.list_entities_in_group("ns-a").await.expect("list ns-a");
    let ns_b = g.list_entities_in_group("ns-b").await.expect("list ns-b");
    assert_eq!(ns_a.len(), 1, "ns-a has its row");
    assert_eq!(ns_b.len(), 1, "ns-b has its own independent row (per-namespace-open)");
}

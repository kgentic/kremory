#![allow(clippy::unwrap_used, clippy::expect_used)]
//! F2 SPIKE — see .ai-docs/adrs/rql/adr-029-namespace-policy/cycle-1-spike-f2-cross-namespace-dedup.md
//!
//! Hypothesis (Vera review F2): kremory's entity dedup is keyed on
//! `normalize_name(label)` → used as the `rql_entities.id` PRIMARY KEY. The schema
//! has a single TEXT `group_id` column (no composite (id, group_id) PK). If a
//! consumer writes "Acme Corp" into namespace-A then "Acme Corp" into namespace-B,
//! does the row get shared (silently mutating namespace-A's data) or rejected?
//!
//! Observed via direct `TemporalGraph` access (Approach A) — bypasses the LLM
//! extraction + CascadeResolver tier so we isolate the storage decision.

use kremory::core::schema::TemporalGraph;

/// Q1+Q2+Q3+Q4: same surface name into two namespaces, observe row count + group_id.
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
            "Organisation",
            serde_json::json!({"name": "Acme Corp", "context": "first mention in ns-a"}),
            Some(ns_a),
        )
        .await;
    println!("SPIKE F2 — INSERT 1 (ns-a) result: {:?}", r1);
    assert!(r1.is_ok(), "first insert should succeed");

    // 2nd write: ingest "Acme Corp" again under namespace-B.
    let r2 = g
        .insert_entity_with_group(
            entity_id,
            "Organisation",
            serde_json::json!({"name": "Acme Corp", "context": "second mention in ns-b"}),
            Some(ns_b),
        )
        .await;
    println!("SPIKE F2 — INSERT 2 (ns-b) result: {:?}", r2);

    // Observation 1 (Q1): how many rows are there in rql_entities for this id?
    // Use the public list APIs — there is no direct conn access from outside the crate.
    let all_entities = g.list_entities().await.expect("list entities");
    let matching: Vec<_> = all_entities.iter().filter(|e| e.id == entity_id).collect();
    println!(
        "SPIKE F2 — Q1 row count for id={:?}: {} (total entities in DB: {})",
        entity_id,
        matching.len(),
        all_entities.len()
    );
    for e in &matching {
        println!(
            "SPIKE F2 — Q2/Q3 row dump: id={} label={} group_id={:?} props={}",
            e.id, e.label, e.group_id, e.properties
        );
    }

    // Observation 2: list per-namespace via the production list_entities_in_group
    // path to confirm what each namespace's "view" of the row looks like.
    let ns_a_view = g.list_entities_in_group(ns_a).await.expect("list ns_a");
    let ns_b_view = g.list_entities_in_group(ns_b).await.expect("list ns_b");
    println!(
        "SPIKE F2 — ns_a view: {} entities ({:?})",
        ns_a_view.len(),
        ns_a_view.iter().map(|e| (&e.id, &e.group_id)).collect::<Vec<_>>()
    );
    println!(
        "SPIKE F2 — ns_b view: {} entities ({:?})",
        ns_b_view.len(),
        ns_b_view.iter().map(|e| (&e.id, &e.group_id)).collect::<Vec<_>>()
    );
}

/// Q4: does `set_entity_group_only` succeed and overwrite the stored group_id
/// without any AppendOnly-style guard?
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_set_entity_group_only_overwrite() {
    let g = TemporalGraph::open_in_memory().await.expect("open");

    let entity_id = "acme corp";
    let ns_a = "audit-ns-a";
    let ns_b = "tenant-b";

    g.insert_entity_with_group(
        entity_id,
        "Organisation",
        serde_json::json!({"name": "Acme Corp"}),
        Some(ns_a),
    )
    .await
    .expect("seed insert into ns_a");

    // Probe: read back the initial group_id.
    let before = g.get_entity(entity_id).await.expect("get").expect("present");
    println!("SPIKE F2 Q4 — group_id BEFORE move: {:?}", before.group_id);

    // Move the entity to ns_b via the documented dual-store coupling API.
    let n = g
        .set_entity_group_only(entity_id, Some(ns_b))
        .await
        .expect("move group");
    println!("SPIKE F2 Q4 — rows updated by set_entity_group_only: {}", n);

    let after = g.get_entity(entity_id).await.expect("get").expect("present");
    println!("SPIKE F2 Q4 — group_id AFTER move: {:?}", after.group_id);
}

/// Q5 (control): write a real `INSERT OR IGNORE`-style stub via the ingest
/// pattern to confirm the comment in ingest.rs:300-303 ("INSERT OR IGNORE")
/// reflects observed behaviour. Spoiler from code: it's not actually OR IGNORE
/// — the caller swallows the Err in a `match`. This test exercises the raw
/// failure mode (UNIQUE constraint error) callers depend on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spike_f2_second_insert_returns_unique_violation() {
    let g = TemporalGraph::open_in_memory().await.expect("open");

    let entity_id = "acme corp";

    g.insert_entity_with_group(
        entity_id,
        "Organisation",
        serde_json::json!({"name": "Acme Corp"}),
        Some("ns-a"),
    )
    .await
    .expect("first insert");

    let err = g
        .insert_entity_with_group(
            entity_id,
            "Organisation",
            serde_json::json!({"name": "Acme Corp"}),
            Some("ns-b"),
        )
        .await
        .expect_err("second insert must fail (PK collision)");

    println!("SPIKE F2 — second insert error: {}", err);
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("unique") || msg.to_lowercase().contains("constraint"),
        "expected UNIQUE / constraint error, got: {msg}"
    );
}

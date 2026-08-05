#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Property-based tests for ADR-029b + ADR-029c invariants.
//!
//! Three property suites (Phase A of the 5-tier test pyramid):
//!
//! A1 — RRF composite-key dedup invariant
//!      `rrf_fuse_entities` and `rrf_fuse_facts` must never output more rows
//!      than the union cardinality of their two input lists (dedup by
//!      composite key). The union-cardinality bound is a pure function of
//!      the input keys — no SQL required.
//!
//! A2 — `ContradictionOverflow` sentinel exists and is well-typed
//!      The error variant `Error::ContradictionOverflow { count, ceiling }`
//!      must be constructable and its Display message must reference both
//!      fields. This guards ADR-029b Decision 3's safety ceiling contract:
//!      any production path that would emit this error is correctly typed.
//!
//! A3 — `NamespacePolicy` serde round-trip
//!      Arbitrary `NamespacePolicy` values must survive JSON round-trips
//!      without field loss. Covers `#[non_exhaustive]` + `#[serde(default)]`
//!      interactions introduced in v0.1.4 (ADR-029a).

use kremory::core::error::Error as CoreError;
use kremory::core::schema::{Entity, Fact};
use kremory::core::search::{rrf_fuse_entities_for_test, rrf_fuse_facts_for_test, SearchHit};
use kremory::{ImmutabilityLevel, NamespacePolicy};
use proptest::prelude::*;
use std::collections::HashSet;

use chrono::Utc;

// ── Helpers to construct minimal Entity / Fact stubs ─────────────────────────

fn make_entity(id: &str, group_id: Option<&str>) -> Entity {
    Entity {
        id: id.to_owned(),
        label: id.to_owned(),
        entity_type_id: 0,
        properties: serde_json::Value::Null,
        recorded_at: Utc::now(),
        updated_at: None,
        group_id: group_id.map(str::to_owned),
        access_count: 0,
    }
}

fn make_fact(id: i64, group_id: Option<&str>) -> Fact {
    Fact {
        id,
        subject_id: "s".to_owned(),
        predicate: "p".to_owned(),
        object_id: None,
        object_value: Some("o".to_owned()),
        properties: None,
        valid_from: Utc::now(),
        valid_to: None,
        recorded_at: Utc::now(),
        expired_at: None,
        invalid_at: None,
        group_id: group_id.map(str::to_owned),
        confidence: 1.0,
        source_episode_id: None,
        memory_type: None,
        content_hash: None,
        access_count: 0,
        subject_group_id: group_id.map(str::to_owned),
        object_group_id: None,
    }
}

fn hit_entity(id: &str, group_id: Option<&str>, score: f64) -> SearchHit<Entity> {
    SearchHit {
        item: make_entity(id, group_id),
        score,
    }
}

fn hit_fact(id: i64, group_id: Option<&str>, score: f64) -> SearchHit<Fact> {
    SearchHit {
        item: make_fact(id, group_id),
        score,
    }
}

// ── A1: RRF composite-key dedup invariant ────────────────────────────────────

// Entities: output count ≤ union of (id, group_id) keys from both inputs.
//
// The invariant: no output row appears more than once (dedup by composite key)
// AND output count is bounded by the union cardinality of all input keys.
proptest! {
    #[test]
    fn a1_rrf_entities_dedup_no_duplicates(
        // generate 0..8 (id_index, group_index, score) tuples for each input
        vec_hits in prop::collection::vec(
            (0usize..6, prop::option::of(0usize..4), 0.1f64..1.0f64),
            0..8
        ),
        fts_hits in prop::collection::vec(
            (0usize..6, prop::option::of(0usize..4), 0.1f64..1.0f64),
            0..8
        ),
    ) {
        let ns_labels = ["ns-a", "ns-b", "ns-c", "ns-d"];
        let entity_ids = ["e1", "e2", "e3", "e4", "e5", "e6"];

        let v: Vec<SearchHit<Entity>> = vec_hits.iter().map(|(ei, gi, sc)| {
            let gid = gi.map(|i| ns_labels[i]);
            hit_entity(entity_ids[*ei], gid, *sc)
        }).collect();

        let f: Vec<SearchHit<Entity>> = fts_hits.iter().map(|(ei, gi, sc)| {
            let gid = gi.map(|i| ns_labels[i]);
            hit_entity(entity_ids[*ei], gid, *sc)
        }).collect();

        // Compute expected union cardinality from input keys.
        let union_keys: HashSet<(String, Option<String>)> = v.iter().chain(f.iter())
            .map(|h| (h.item.id.clone(), h.item.group_id.clone()))
            .collect();

        let out = rrf_fuse_entities_for_test(v, f, 60.0);

        // 1. Output cardinality ≤ union of input keys.
        prop_assert!(out.len() <= union_keys.len(),
            "output {} > union key count {}", out.len(), union_keys.len());

        // 2. No duplicate composite keys in output.
        let out_keys: HashSet<(String, Option<String>)> = out.iter()
            .map(|h| (h.item.id.clone(), h.item.group_id.clone()))
            .collect();
        prop_assert_eq!(out.len(), out_keys.len(),
            "duplicate composite keys in rrf_fuse_entities output");

        // 3. All output keys are a subset of the union.
        for k in &out_keys {
            prop_assert!(union_keys.contains(k),
                "output key {:?} not in union of inputs", k);
        }

        // 4. Scores are positive.
        for h in &out {
            prop_assert!(h.score > 0.0, "rrf score must be positive, got {}", h.score);
        }
    }
}

// Facts: same composite-key dedup invariant using `(fact_id, group_id)`.
proptest! {
    #[test]
    fn a1_rrf_facts_dedup_no_duplicates(
        vec_hits in prop::collection::vec(
            (0i64..6, prop::option::of(0usize..4), 0.1f64..1.0f64),
            0..8
        ),
        fts_hits in prop::collection::vec(
            (0i64..6, prop::option::of(0usize..4), 0.1f64..1.0f64),
            0..8
        ),
    ) {
        let ns_labels = ["ns-a", "ns-b", "ns-c", "ns-d"];

        let v: Vec<SearchHit<Fact>> = vec_hits.iter().map(|(fi, gi, sc)| {
            let gid = gi.map(|i| ns_labels[i]);
            hit_fact(*fi, gid, *sc)
        }).collect();

        let f: Vec<SearchHit<Fact>> = fts_hits.iter().map(|(fi, gi, sc)| {
            let gid = gi.map(|i| ns_labels[i]);
            hit_fact(*fi, gid, *sc)
        }).collect();

        let union_keys: HashSet<(i64, Option<String>)> = v.iter().chain(f.iter())
            .map(|h| (h.item.id, h.item.group_id.clone()))
            .collect();

        let out = rrf_fuse_facts_for_test(v, f, 60.0);

        prop_assert!(out.len() <= union_keys.len(),
            "output {} > union key count {}", out.len(), union_keys.len());

        let out_keys: HashSet<(i64, Option<String>)> = out.iter()
            .map(|h| (h.item.id, h.item.group_id.clone()))
            .collect();
        prop_assert_eq!(out.len(), out_keys.len(),
            "duplicate composite keys in rrf_fuse_facts output");

        for k in &out_keys {
            prop_assert!(union_keys.contains(k),
                "output key {:?} not in union of inputs", k);
        }

        for h in &out {
            prop_assert!(h.score > 0.0, "rrf score must be positive, got {}", h.score);
        }
    }
}

/// Same-key cross-source fusion: entity present in both vector + FTS lists
/// gets a combined score higher than either individual contribution.
#[test]
fn a1_rrf_entities_same_key_score_combines() {
    // rank-0 in both lists → each contributes 1/(60+1) ≈ 0.01639
    let entity = hit_entity("alice", Some("ns-a"), 0.9);
    let v = vec![entity.clone()];
    let f = vec![entity];
    let out = rrf_fuse_entities_for_test(v.clone(), vec![], 60.0);
    let single_score = out[0].score;

    let out_combined = rrf_fuse_entities_for_test(v, f, 60.0);
    assert_eq!(out_combined.len(), 1, "should dedup to single entry");
    assert!(
        out_combined[0].score > single_score,
        "combined score {} must exceed single-list score {}",
        out_combined[0].score,
        single_score
    );
}

/// Different-namespace same-id entities stay separate in output.
#[test]
fn a1_rrf_entities_different_ns_same_id_kept_separate() {
    let alice_a = hit_entity("alice", Some("ns-a"), 0.9);
    let alice_b = hit_entity("alice", Some("ns-b"), 0.8);
    let out = rrf_fuse_entities_for_test(vec![alice_a], vec![alice_b], 60.0);
    assert_eq!(
        out.len(),
        2,
        "entities in different namespaces must NOT dedup"
    );
}

/// Legacy unkeyed entities (group_id = None) share the None-bucket correctly.
#[test]
fn a1_rrf_entities_none_group_id_deduplicates_within_bucket() {
    let legacy = hit_entity("alice", None, 0.9);
    let out = rrf_fuse_entities_for_test(vec![legacy.clone()], vec![legacy], 60.0);
    assert_eq!(out.len(), 1, "legacy None-group entities must dedup");
}

// ── A2: ContradictionOverflow error type contract ────────────────────────────

// The `ContradictionOverflow` error variant is constructable and its Display
// message contains both the count and ceiling values.
//
// ADR-029b Decision 3: `as_of_all` gates result set size behind a ceiling.
// This test asserts the error type contract so any future producer of this
// variant cannot change field names or the Display format without this test
// catching the regression.
proptest! {
    #[test]
    fn a2_contradiction_overflow_display_contains_count_and_ceiling(
        count in 1usize..10_000,
        ceiling in 0usize..9_999,
    ) {
        // Only coherent when count > ceiling.
        let count = count + ceiling; // ensure count > ceiling always
        let err = CoreError::ContradictionOverflow { count, ceiling };
        let msg = err.to_string();
        prop_assert!(msg.contains(&count.to_string()),
            "Display message must contain count {count}, got: {msg}");
        prop_assert!(msg.contains(&ceiling.to_string()),
            "Display message must contain ceiling {ceiling}, got: {msg}");
        // The error must be Send + Sync (required by anyhow + tokio).
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CoreError>();
    }
}

/// Spot-check: known values match expected format from error.rs.
#[test]
fn a2_contradiction_overflow_spot_check() {
    let err = CoreError::ContradictionOverflow {
        count: 1001,
        ceiling: 1000,
    };
    let msg = err.to_string();
    assert!(msg.contains("1001"), "count 1001 in: {msg}");
    assert!(msg.contains("1000"), "ceiling 1000 in: {msg}");
    assert!(
        msg.contains("as_of_all"),
        "message should mention as_of_all: {msg}"
    );
}

// ── A3: NamespacePolicy serde round-trip ─────────────────────────────────────

fn arb_immutability() -> impl Strategy<Value = ImmutabilityLevel> {
    prop_oneof![
        Just(ImmutabilityLevel::Mutable),
        Just(ImmutabilityLevel::AppendOnly),
    ]
}

fn arb_policy() -> impl Strategy<Value = NamespacePolicy> {
    arb_immutability().prop_flat_map(|immutability| {
        // AppendOnly requires forgettable=false and dream_eligible=false.
        // Mutable may have any combination.
        let (forgettable_range, dream_range) = match immutability {
            ImmutabilityLevel::AppendOnly => (Just(false).boxed(), Just(false).boxed()),
            _ => (any::<bool>().boxed(), any::<bool>().boxed()),
        };
        (forgettable_range, dream_range).prop_map(move |(forgettable, dream_eligible)| {
            NamespacePolicy::new()
                .with_immutability(immutability)
                .with_forgettable(forgettable)
                .with_dream_eligible(dream_eligible)
        })
    })
}

proptest! {
    #[test]
    fn a3_namespace_policy_serde_round_trip(policy in arb_policy()) {
        // Coherent policies must validate without error.
        policy.validate().expect("arb_policy must produce coherent policies");

        // JSON round-trip.
        let json = serde_json::to_string(&policy).expect("serialize NamespacePolicy");
        let back: NamespacePolicy = serde_json::from_str(&json).expect("deserialize NamespacePolicy");

        prop_assert_eq!(&back.immutability, &policy.immutability,
            "immutability survives round-trip");
        prop_assert_eq!(back.forgettable, policy.forgettable,
            "forgettable survives round-trip");
        prop_assert_eq!(back.dream_eligible, policy.dream_eligible,
            "dream_eligible survives round-trip");
    }

    #[test]
    fn a3_namespace_policy_json_missing_fields_deserialize_to_default(
        immutability in arb_immutability(),
    ) {
        // Backward-compat: JSON written before v0.1.4 has no `forgettable`/
        // `dream_eligible` fields.  `#[serde(default)]` must fill them in.
        let sparse = match immutability {
            ImmutabilityLevel::AppendOnly => r#"{"immutability":"append_only","forgettable":false,"dream_eligible":false}"#,
            _ => r#"{"immutability":"mutable"}"#,
        };
        let parsed: serde_json::Result<NamespacePolicy> = serde_json::from_str(sparse);
        // Must not error.
        prop_assert!(parsed.is_ok(),
            "sparse JSON parse failed for {:?}: {:?}", immutability, parsed.err());
        let p = parsed.expect("just checked Ok");
        // `mutable` sparse case: defaults kick in.
        if immutability == ImmutabilityLevel::Mutable {
            prop_assert_eq!(p.forgettable, true, "default forgettable=true");
            prop_assert_eq!(p.dream_eligible, true, "default dream_eligible=true");
        }
    }
}

/// APPEND_ONLY const survives round-trip and validate() accepts it.
#[test]
fn a3_append_only_const_round_trip() {
    let policy = NamespacePolicy::APPEND_ONLY;
    policy.validate().expect("APPEND_ONLY must be coherent");
    let json = serde_json::to_string(&policy).expect("serialize APPEND_ONLY");
    let back: NamespacePolicy = serde_json::from_str(&json).expect("deserialize APPEND_ONLY");
    assert_eq!(
        back, policy,
        "APPEND_ONLY const must survive JSON round-trip"
    );
}

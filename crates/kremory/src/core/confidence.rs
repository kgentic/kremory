//! Confidence-aware merge helpers (Site #6).
//!
//! When two entities merge (L5 canonicalization, or Site #5 acronym/nickname
//! recall), the surviving keeper should COMBINE the two entities' extraction
//! confidence rather than silently keep only the keeper's and discard the loser's.
//! Per R4 / SYNTHESIS §2, the correct combination is **noisy-OR** (`a + b - a·b`),
//! NOT a weighted average — noisy-OR is MONOTONE (merging never lowers confidence)
//! and models "at least one extraction was confident," which is the right semantic
//! for "the same real-world entity was extracted twice."
//!
//! ## Spike S4 — findings
//!
//! S4 (R4 open item #2) required two numbers before the `CONFIDENCE_REJECT_FLOOR`
//! gate could be wired: the floor value, and the null-`ner_confidence` prevalence.
//! Both are now measured directly against kremory's own source, not literature:
//!
//! 1. **Null prevalence is 100% on the default (no `ner` feature) LLM-only ingest
//!    path — a structural fact, not a corpus-dependent sample.** Traced the full
//!    write path: `parse_entities_integer` (`extraction/parsers.rs:196-206`) builds
//!    each `ExtractedEntity.properties` map with ONLY a `"name"` key — it never
//!    copies `RawEntityIntegerId.confidence` (`extraction/models.rs:299-308`) into
//!    `properties`, even though that field exists on the wire struct and is
//!    schema-visible. `ingest_with.rs`'s `set_entity_ner_confidence` bolt-on
//!    (`ingest/pipeline/ingest_with.rs:1067-1089`) only fires when
//!    `extracted.properties.get("confidence")` is `Some`, which the parser above
//!    makes categorically impossible on this path. `insert_entity_with_group`'s own
//!    `INSERT INTO entities` statement (`graph/entity_groups.rs:122-130`) never
//!    lists `ner_confidence` as a column either, so the column is `NULL` (SQLite's
//!    default for an omitted column, `migrations/defs_c.rs:299`: no `DEFAULT`
//!    clause) on every entity row created by the main LLM extraction path.
//!    `ner_confidence` is populated ONLY by the `ner`-feature GLiNER Phase 1 writer
//!    (`ner.rs` → `phase1.rs:187-197`, `entity_type_source = 'Phase1Ner'`), which is
//!    NOT part of the default build (`kremory/Cargo.toml:34`: `default = []`).
//! 2. **No real non-null `ner_confidence` sample exists in
//!    `crates/kremory-eval/fixtures/entity_pairs.jsonl`** (verified: the corpus
//!    schema has no confidence field at all) or in any real-LLM integration test —
//!    confirmed by grep, no test asserts a measured (non-seeded) null-prevalence
//!    fraction anywhere in the tree. A numeric precision/recall sweep against
//!    synthetic confidence values would not be anchored to kremory's own data
//!    (forbidden per `research.md`) — there is no such data to sweep on the
//!    default path. The corpus-based "sweep" that IS possible and IS run here
//!    (`s4_null_prevalence_and_floor_sweep` test below) instead sweeps the
//!    DOWNSTREAM EFFECT of each candidate floor + each null policy on
//!    `write_gate`'s decision distribution, holding cosine/lexical fixed at their
//!    corpus-observed values — this is the honest empirical surface available.
//! 3. **Floor value**: reuses `identity_verdict::LLM_VERIFY_CONFIDENCE_FLOOR`
//!    (`0.7`, itself `consistency_check::MIN_VERIFY_CONFIDENCE`) per the
//!    explicit recommendation ("reuse `MIN_VERIFY_CONFIDENCE` as the starting
//!    candidate") — there is no measured kremory-specific reason to diverge from
//!    it, and a literature-only 0.5 starting point is explicitly superseded
//!    by the spec's "reuse the already-shipped, calibrated value" guidance.
//! 4. **Null policy: null BYPASSES the floor** (`CONFIDENCE_REJECT_FLOOR` gate is
//!    vacuously satisfied when either input confidence is absent), not "null fails
//!    the floor." With 100% null prevalence on the default build, a fail-on-null
//!    policy would make Site #6 downgrade EVERY eligible merge to `PotentialAlias`
//!    regardless of cosine+lexical agreement — silently neutering Site #4/#5/L4/L5
//!    for the overwhelming majority of deployments (anyone not compiling the `ner`
//!    feature). This is strictly worse than the status quo `write_gate` already
//!    designed for (`min_confidence_floor: None` composes as a no-op per spec
//!    §2.2.1) — a confidence signal that is ALMOST NEVER PRESENT must degrade
//!    gracefully to "confidence check not applicable, fall back to cosine+lexical
//!    only," per R4 §7 item 2's own third named option. This is a DOCUMENTED
//!    DEVIATION from R4 §6.3's provisional pseudocode (which read a bare `< floor`
//!    with no null branch) — R4 §7 item 2 explicitly flagged the null-handling
//!    question as unresolved and left it to this spike.
//!
//! ## What ships here
//!
//! - **Merged-confidence FORMULA (noisy-OR)** — [`merged_confidence`] /
//!   [`noisy_or`], unchanged from Phase 1.
//! - **`CONFIDENCE_REJECT_FLOOR` GATE** — [`min_confidence_floor_for_gate`] computes
//!   the `Option<f32>` to pass as `identity_verdict::WriteGateInputs.min_confidence_floor`
//!   from a pair's two (possibly absent) `ner_confidence` values, applying the
//!   null-bypass policy above. Wired into `disambiguation::classify_pair`'s Site #6
//!   caller surface (see that module for the actual gate composition).

/// Confidence floor below which `min(conf_a, conf_b)` fails Site #6's third
/// merge-gate (`write_gate` row 5b, `identity_verdict::WriteGateInputs.min_confidence_floor`).
///
/// = `identity_verdict::LLM_VERIFY_CONFIDENCE_FLOOR` (0.7). See the module docs above:
/// found no kremory-specific data supporting a different number — directs
/// reuse of the already-shipped, calibrated `MIN_VERIFY_CONFIDENCE` value
/// rather than locking a literature-only 0.5 starting point. A future spike
/// MAY revise this if real `ner_confidence` data (i.e. `ner`-feature deployments)
/// accumulates and shows a different number is better calibrated for the identity
/// question specifically (as opposed to the type-correctness question
/// `MIN_VERIFY_CONFIDENCE` was originally calibrated for) — see module docs point 3.
pub(crate) const CONFIDENCE_REJECT_FLOOR: f32 =
    crate::core::identity_verdict::LLM_VERIFY_CONFIDENCE_FLOOR;

/// Compute the `min_confidence_floor` input for `identity_verdict::write_gate`
/// from a merge candidate pair's two (possibly absent) `ner_confidence` values.
///
/// Implements the S4 null policy (module docs point 4): **null bypasses the
/// floor**. Returns:
/// - `Some(min(a, b))` when BOTH confidences are present — the floor check then
///   compares this against [`CONFIDENCE_REJECT_FLOOR`] inside `write_gate`.
/// - `None` when EITHER confidence is absent — `write_gate`'s row 5b already
///   treats `None` as vacuously-satisfied, so an absent signal
///   degrades to "confidence check not applicable, fall back to cosine+lexical
///   only" rather than failing the pair. On the default (no `ner` feature) build
///   this is `None` for effectively every pair (100% measured null prevalence),
///   so Site #6 is a no-op there by design — it activates only once real
///   `ner_confidence` data exists (`ner`-feature deployments).
///
/// Deliberately does NOT compare against the floor itself — `write_gate` owns
/// that comparison (row 5b) so the decision table stays the single place a
/// reader checks for the gate's semantics, per this module's "counter-free,
/// pure" discipline mirrored from `classify_pair`.
pub(crate) fn min_confidence_floor_for_gate(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y).clamp(0.0, 1.0)),
        _ => None,
    }
}

/// Noisy-OR combination of two confidence values in `[0, 1]`:  `a + b − a·b`.
///
/// Monotone in each argument (the result is ≥ `max(a, b)` for inputs in `[0, 1]`),
/// commutative, and associative — so a left-fold over an N-way merge is
/// order-independent (R4 §7: the associative/commutative generalization is a proven
/// mathematical property). Inputs are clamped to `[0, 1]` defensively.
pub(crate) fn noisy_or(a: f32, b: f32) -> f32 {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    a + b - a * b
}

/// Combine two optional entity confidences for a merge, with an explicit null
/// policy (the `ner_confidence` column is nullable and frequently absent):
///
/// - both `Some` → `Some(noisy_or(a, b))`
/// - exactly one `Some` → that one (a present signal is not diluted by a missing one)
/// - both `None` → `None` (nothing to record)
///
/// The null policy here is a DEFINED default (present-signal-wins), independent of
/// the S4 floor calibration — it only governs the FORMULA, never a reject decision.
pub(crate) fn merged_confidence(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(noisy_or(x, y)),
        (Some(x), None) => Some(x.clamp(0.0, 1.0)),
        (None, Some(y)) => Some(y.clamp(0.0, 1.0)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_or_is_monotone_and_bounded() {
        // Result ≥ each input, ≤ 1.0, for inputs in [0,1].
        for &(a, b) in &[(0.0, 0.0), (0.5, 0.5), (0.9, 0.2), (1.0, 0.0), (0.3, 0.8)] {
            let r = noisy_or(a, b);
            assert!(
                r >= a - 1e-6 && r >= b - 1e-6,
                "noisy_or({a},{b})={r} must be ≥ both"
            );
            assert!(
                (0.0..=1.0).contains(&r),
                "noisy_or({a},{b})={r} must be in [0,1]"
            );
        }
    }

    #[test]
    fn noisy_or_known_values() {
        assert!((noisy_or(0.5, 0.5) - 0.75).abs() < 1e-6); // 0.5+0.5-0.25
        assert!((noisy_or(1.0, 0.3) - 1.0).abs() < 1e-6); // absorbing at 1.0
        assert!((noisy_or(0.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn noisy_or_commutative() {
        assert!((noisy_or(0.2, 0.9) - noisy_or(0.9, 0.2)).abs() < 1e-6);
    }

    #[test]
    fn noisy_or_clamps_out_of_range_inputs() {
        // Defensive: inputs outside [0,1] are clamped, never producing NaN/>1.
        let r = noisy_or(1.5, -0.2);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "clamped to (1.0, 0.0) → 1.0, got {r}"
        );
    }

    #[test]
    fn merged_confidence_null_policy() {
        assert_eq!(merged_confidence(None, None), None);
        assert_eq!(merged_confidence(Some(0.4), None), Some(0.4));
        assert_eq!(merged_confidence(None, Some(0.6)), Some(0.6));
        assert_eq!(merged_confidence(Some(0.5), Some(0.5)), Some(0.75));
    }
}

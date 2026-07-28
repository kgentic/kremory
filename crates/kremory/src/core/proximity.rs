//! Graph-proximity boost axis — ADR-062 (axis C), build-entry spec
//! `axis-c-read-time-relevance-spec-2026-07-01.md`, ADR-067 Phase 3.
//!
//! **Additive + bounded `[0, weight]`, matching [`crate::core::search::
//! graph_degree_bonus`] and [`crate::core::scoring::temporal::temporal_boost`]**
//! so a single `.min(1.0)` clamp covers all three axes at the insertion point
//! — no second normalization pass. ADR-067 **Amendment 1** (2026-07-20)
//! supersedes ADR-062's literal "post-RRF multiplicative boost" text: this
//! axis's own landing is the amendment's named migration trigger, and the
//! amendment already resolved that trigger to "stay additive" (score 132/135,
//! confidence HIGH) rather than migrate the whole chain to a normalized
//! multiplicative shape.
//!
//! Signal: the count of entities reachable from a seed within
//! `proximity_hop_bound` hops (`SearchConfig::proximity_hop_bound`, default
//! 2) — a WIDER, independently-bounded walk than
//! [`crate::core::search::graph_degree_bonus`]'s own 1-hop degree signal
//! (`expansion_hop_bound`, default 1). Distinct information: degree asks "how
//! many direct facts does this seed have"; proximity asks "is this seed near
//! a dense cluster it isn't directly part of" (ADR-062's own motivating
//! example: "a query 2 hops from a highly-connected entity gets no proximity
//! credit" under lexical-only ranking).
//!
//! **Read-side-pure (ADR-062 §7 RISK-003, spec NFR):** this module issues
//! **zero graph queries** — it is pure `usize -> f32` boost math over a count
//! the caller already fetched via a second, independently-bounded
//! `TemporalGraph::get_neighbours_at` call in
//! [`crate::core::context::Engine::contextualize`]'s per-seed loop (mirroring
//! how `graph_degree_bonus`/`temporal_boost` consume data the SAME loop's
//! primary `get_neighbours_at` call already fetched). The precise, mechanical
//! read-purity heuristic (ADR-062 §7, corrected in Cycle-1 review RISK-003):
//! grep this file for `execute(` (write) — a read-pure module must show zero
//! occurrences; a `use`-statement audit alone is insufficient because
//! `TemporalGraph` (the type the real graph-walk call lives on, in
//! `core/graph/queries.rs`) has its own inherent write methods
//! (`forget_entity`, `batch_forget`) reachable on the same `self`.
//!
//! **Fan-out cap (ADR-062 §8/ASMP-001, spike criterion 2a):** the caller
//! passes `max_visited: Some(SearchConfig::proximity_fan_out_cap)` to the
//! SAME `get_neighbours_at`, reusing ADR-067 Amendment 2's in-BFS visited cap
//! (originally built for TD-056's multi-hop expansion) rather than adding a
//! new capped traversal primitive — a high-degree hub seed's worst-case cost
//! is bounded by construction, not merely spike-measured.

/// Count value at which [`proximity_bonus`] saturates. Wider than
/// [`crate::core::search::GRAPH_DEGREE_SATURATION`] (10) because a
/// `proximity_hop_bound=2` walk naturally reaches more entities than a 1-hop
/// degree count for the same graph — using the same saturation point would
/// make the proximity axis saturate almost immediately, collapsing its
/// resolution.
pub(crate) const PROXIMITY_SATURATION: f32 = 20.0;

/// Additive graph-proximity bonus for a seed, in `[0, weight]`.
///
/// `neighbour_count` = entities reachable from the seed within
/// `proximity_hop_bound` hops (excluding the seed itself — see the caller in
/// `context.rs`, which computes this the same way `graph_degree_bonus`'s
/// caller computes 1-hop `degree`: `subgraph.entities.len().saturating_sub(1)`).
///
/// Boundary behaviour (unit-pinned below):
/// - `weight <= 0.0` → `0.0` (true no-op; callers should additionally skip
///   the graph query entirely at this weight — see `context.rs` — but this
///   function is safe to call unconditionally too).
/// - `neighbour_count == 0` → `0.0` (an isolated seed gets no proximity
///   credit).
/// - saturates at [`PROXIMITY_SATURATION`]: a hub seed's contribution is
///   capped at `weight`, never unbounded.
/// - monotonic: a more-connected seed never scores a smaller bonus than a
///   less-connected one, all else equal.
pub(crate) fn proximity_bonus(neighbour_count: usize, weight: f32) -> f32 {
    if weight <= 0.0 {
        return 0.0;
    }
    weight * (neighbour_count as f32 / PROXIMITY_SATURATION).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weight_is_no_op() {
        assert_eq!(proximity_bonus(5, 0.0), 0.0);
        // negative weight also clamps to no-op
        assert_eq!(proximity_bonus(5, -1.0), 0.0);
    }

    #[test]
    fn zero_neighbours_is_zero() {
        assert_eq!(proximity_bonus(0, 0.2), 0.0);
    }

    #[test]
    fn saturates_at_ceiling() {
        let saturated = proximity_bonus(PROXIMITY_SATURATION as usize, 0.2);
        let hub = proximity_bonus(100_000, 0.2);
        assert!(
            (saturated - 0.2).abs() < 1e-6,
            "must saturate to full weight at PROXIMITY_SATURATION, got {saturated}"
        );
        assert_eq!(
            saturated, hub,
            "a hub seed (100k neighbours) must not exceed the saturation ceiling's bonus"
        );
    }

    #[test]
    fn never_exceeds_weight() {
        const W: f32 = 0.2;
        for count in [0_usize, 1, 5, 10, 20, 50, 1_000] {
            let bonus = proximity_bonus(count, W);
            assert!(
                (0.0..=W + 1e-6).contains(&bonus),
                "count={count} produced bonus={bonus} outside [0, {W}]"
            );
        }
    }

    #[test]
    fn monotonic_below_saturation() {
        const W: f32 = 0.2;
        assert!(proximity_bonus(5, W) > proximity_bonus(1, W));
        assert!(proximity_bonus(15, W) > proximity_bonus(5, W));
    }

    /// ADR-062 §7 RISK-003 (corrected read-purity heuristic): this module
    /// must contain zero `execute(` call sites. Grepping its own source
    /// (rather than auditing `use` statements) is the precise, mechanical
    /// check the arch-design review settled on — `TemporalGraph` hosts write
    /// methods (`forget_entity`, `batch_forget`) on the SAME `self` as the
    /// read-only `get_neighbours_at` this axis's caller uses, so a
    /// `use`-statement check cannot distinguish read-pure from write-capable.
    #[test]
    fn proximity_module_is_read_pure_zero_execute_calls() {
        // The needle is assembled at runtime and comment lines are stripped,
        // because the naive `src.contains("...")` form of this guard was
        // SELF-TRIPPING: every occurrence of the literal in this file was the
        // guard's OWN text (the module doc, this test's doc, and the assertion
        // message), so it could never pass regardless of the implementation.
        // Verified 2026-07-27: the module is genuinely write-free — the guard's
        // MECHANISM was broken, not the code under test. Guard kept (ADR-062 §7
        // read-purity is load-bearing — a scoring axis that writes violates the
        // ratified design) and its mechanism fixed, per treat-cause-not-symptom.
        let needle = concat!("execu", "te(");
        let offending: Vec<(usize, &str)> = include_str!("proximity.rs")
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim_start().starts_with("//"))
            .filter(|(_, line)| line.contains(needle))
            .map(|(i, line)| (i + 1, line.trim()))
            .collect();
        assert!(
            offending.is_empty(),
            "core/proximity.rs must contain zero write call sites (RISK-003 read-purity, \
             ADR-062 §7) — this axis must never write. Offending: {offending:?}"
        );
    }
}

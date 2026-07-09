//! CONSOLIDATION op — cross-episode entity merge (ADR-066 §2.2, spec P3).
//!
//! Merge two entities in a `group_id` into one ONLY when ALL of:
//!
//! 1. **Same NORMALIZED label** (case-fold + whitespace-collapse, exact) OR
//!    **MinHash/Jaccard ≥ 0.9** on label token-shingles (fuzzy path). Deterministic,
//!    zero-embedding, zero-LLM.
//! 2. The pair is anchored across **≥ 2 DISTINCT episodes** (via
//!    `episodic_edges.episode_id`) — a same-episode-only pair is NOT cross-episode
//!    recurrence, it is one mention, and is left alone (spec §6
//!    `same_label_same_episode`).
//! 3. **STRUCTURAL CORROBORATION (the homonym guard, RISK-001 / DoD-P3.1b, MANDATORY
//!    on BOTH paths):** the two candidates SHARE STRUCTURE — either (i) a common third
//!    entity `c` such that both `a` and `b` reference `c` (as `subject_id` or
//!    `object_id`) in some fact, OR (ii) both assert an identical
//!    `(predicate, object_id/object_value)` fact. Same-label-ALONE with NO shared
//!    structure → probable homonym (two distinct real referents sharing a name, e.g.
//!    "John Smith" the lawyer vs the athlete) → **DO NOT MERGE**. The corroboration
//!    DECISION (ADR-067 F2) is a **rarity-weighted, fully-integer** score — a single
//!    RARE shared corroborator merges, but hub-shared corroborators (common third
//!    entities referenced by many others) are down-weighted and do NOT alone merge.
//!    Deterministic, zero-LLM, zero-float-at-the-boundary.
//!
//! 4. **CLIQUE-ONLY clustering (ADR-067 F1):** entities merge only as a maximal
//!    CLIQUE of the eligible-pair graph — every pair in a merged set must be
//!    DIRECTLY corroborated. A same-label triple `A~B`, `B~C`, `A≁C` (a "bridge
//!    homonym") is NOT a clique — connected-component transitive closure would
//!    wrongly fuse all three via the bridge `B`; clique-only merges only one of
//!    `{A,B}`/`{B,C}` (deterministic disjoint-cover) and defers the other.
//!
//! 5. **Provenance-anchored corroboration (ADR-067 §C0, the convergence fix):** a
//!    merge's endpoint remap (`apply_entity_merge` → `apply_merge_with_audit`)
//!    stamps every REWRITTEN fact `corroboration_inert = 1`. Corroboration reads
//!    (`neighbours_of` / `assertions_of`) filter `corroboration_inert = 0` — so an
//!    inherited (merge-created) edge can NEVER manufacture new corroboration
//!    eligibility on a later pass. This severs the cross-pass re-eligibility that
//!    clique-only ALONE does not fix (a deferred bridge partner would otherwise
//!    become eligible against the keeper on the NEXT pass, once the keeper inherits
//!    the loser's neighbours). The flag is monotone (merges only ever set it) → the
//!    corroboration-live edge set shrinks monotonically across passes → the op
//!    reaches a FIXPOINT in ≤ 1 merge-pass.
//!
//! When all hold: **keeper = the lowest entity id in the clique** (deterministic
//! tie-break); delegate the structural merge to the shared
//! [`crate::core::canonicalization::apply_entity_merge`] executor (spec DoD-P0.3) — NO
//! second merge code path (R-02).
//!
//! **Non-overlap with canonicalize (DoD-P3.3):** this op fires ONLY on exact/fuzzy
//! LABEL matches, NEVER on embedding cosine — a cosine-near-dup pair that is
//! lexically distinct is canonicalize's job, and cross_episode leaves it (spec §6
//! `cosine_near_dup_lexically_distinct`). No double-count.
//!
//! `cross_episode_merges` counts merge DECISIONS this pass, split
//! `{path=exact|fuzzy}` (DoD-P3.4). Under the ADR-070 shadow gate (`dry_run=true`)
//! the count is identical but no entity is fused — the `decision_total` `mode` label
//! (`shadow`/`applied`) distinguishes the two.
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P3.1–P3.4, incl. P3.1b) + §6 (cross_episode corpus) + ADR-066 §2.2 (REVISED),
//! superseded/extended by ADR-067 (clique F1 + rarity-weighted F2 + provenance C0):
//! `.ai-docs/adrs/adr-067-cross-episode-merge-safety-2026-07-03.md` +
//! `.ai-docs/specs/adr-067-cross-episode-merge-safety-impl-spec-2026-07-03.md`.

use std::collections::{BTreeMap, BTreeSet};

use metrics::counter;

use crate::core::canonicalization::apply_entity_merge;
use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::{
    emit_decision, ConsolidationOpKind, DecisionMode, DecisionRecord, OpReport,
};

/// Fuzzy-path admission threshold on the label token-shingle Jaccard (SYNTHESIS #9).
/// A pair below this on the fuzzy path is NOT admitted; the exact path (Jaccard == 1.0
/// on normalized labels) is the primary route.
const FUZZY_JACCARD_THRESHOLD: f64 = 0.9;

// ─── ADR-067 F1: clique-cover constants ──────────────────────────────────────────

/// Maximum size of a same-normalized-label connected component the op will attempt
/// clique enumeration over (impl-spec §C1 step 3 / §2). Above this, the component is
/// SKIPPED entirely (0 merges, a warning is emitted) rather than risk a pathological
/// enumeration — merge-safety over recall (R-03).
const MAX_LABEL_GROUP: usize = 32;

// ─── ADR-067 F2: rarity-weighted corroboration constants (V2 — fully integer) ────

/// Documentation-only, human-readable form of the corroboration threshold. The
/// DECISION uses [`SCALED_THRESHOLD`] (`u64`), never this `f64`. Provisional —
/// REQUIRES an ADR-063-style corpus spike before `include_cross_episode_merges`
/// may be enabled (R-01).
#[allow(dead_code)] // documentation-only constant; not read by the decision path
const CORROBORATION_THRESHOLD_NUM: f64 = 0.5;

/// Fixed common denominator for the scaled-integer weight LUT: `SCALE = 2^20`.
const SCALE: u64 = 1_048_576;

/// Hub cap: a corroborator with in-group degree/frequency STRICTLY GREATER than this
/// contributes integer weight `0` (a pure-integer branch, no LUT lookup). Provisional
/// — spike-gated (R-01).
const HUB_DEGREE_CAP: u32 = 8;

/// Pre-scaled, round-half-up weight lookup table: `WEIGHT_LUT_SCALED[f] =
/// floor((1/(1+log2(f))) × SCALE + 0.5)` for `f ∈ 1..=HUB_DEGREE_CAP` (index 0 unused).
/// Pinned exactly per impl-spec §3 — a `#[test]` re-derives this array from
/// `f64::log2` to guard against a transcription error (that test uses float; this
/// PRODUCTION array/decision does not).
const WEIGHT_LUT_SCALED: [u64; HUB_DEGREE_CAP as usize + 1] =
    [0, 1_048_576, 524_288, 405_645, 349_525, 315_653, 292_493, 275_408, 262_144];

/// Scaled decision threshold: `round(CORROBORATION_THRESHOLD_NUM × SCALE)`. The
/// decision is `Σ WEIGHT_LUT_SCALED[f] (as u64) >= SCALED_THRESHOLD` — pure integer,
/// no `f64` anywhere in the comparison (V2).
const SCALED_THRESHOLD: u64 = 524_288;

/// Which lexical route admitted a merge — labels the `{path}` counter (DoD-P3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergePath {
    /// Same normalized label (exact case-fold + whitespace-collapse match).
    Exact,
    /// Distinct labels but token-shingle Jaccard ≥ [`FUZZY_JACCARD_THRESHOLD`].
    Fuzzy,
}

impl MergePath {
    fn as_str(self) -> &'static str {
        match self {
            MergePath::Exact => "exact",
            MergePath::Fuzzy => "fuzzy",
        }
    }
}

/// An entity loaded for the cross-episode sweep: its id (name-slug) + normalized
/// label + the set of DISTINCT episodes it is anchored to.
///
/// **The label comparand is the entity id itself.** `entities.id` IS the entity
/// name-slug (canonicalization.rs:131 — "entity ids ARE normalized names") and the
/// legacy `entities.label` column was DROPPED by Migration 009 (`defs_b.rs:391`; the
/// runtime `COALESCE(et.name,'Entity')` "label" is the entity's TYPE name, not its
/// display name, so it is useless as an identity comparand). Two DISTINCT ids that
/// normalize identically (e.g. `"John Smith"` vs `"john  smith"`) are the exact-path
/// candidate pair; near-identical ids clear the fuzzy Jaccard path.
#[derive(Debug, Clone)]
struct EntitySlot {
    /// Entity id = name-slug (`entities.id`), also the `facts.subject_id`/`object_id`
    /// comparand and the keeper tie-break key.
    id: String,
    /// Case-folded, whitespace-collapsed entity id (the identity comparand).
    normalized: String,
    /// Distinct `episodic_edges.episode_id`s this entity appears in.
    episodes: BTreeSet<i64>,
}

// ─── L1 pure helpers (unit-tested) ───────────────────────────────────────────────

/// Normalize a label for exact-match comparison: lowercase (case-fold) + collapse all
/// internal whitespace runs to a single space + trim. Pure, deterministic.
///
/// `"  John   SMITH "` and `"john smith"` normalize identically → an exact-path match.
fn normalize_label(label: &str) -> String {
    label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Whitespace-token-shingle set for a normalized label. Tokens are the individual
/// whitespace-separated words of the normalized label — the shingle grain the fuzzy
/// Jaccard operates over. A single-word label yields a one-element set.
fn label_shingles(normalized: &str) -> BTreeSet<String> {
    normalized.split_whitespace().map(str::to_string).collect()
}

/// Jaccard similarity of two token-shingle sets: `|A ∩ B| / |A ∪ B|`. Returns 1.0 for
/// two empty sets (degenerate — never admitted as a fuzzy candidate because the exact
/// path already covers identical labels), 0.0 when one side is empty. Pure.
fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// The keeper of a candidate pair = the lexicographically-lowest id (deterministic
/// tie-break, DoD-P3.1b). Returns `(keeper, loser)`.
fn select_keeper<'a>(id_a: &'a str, id_b: &'a str) -> (&'a str, &'a str) {
    if id_a <= id_b {
        (id_a, id_b)
    } else {
        (id_b, id_a)
    }
}

// ─── Op entry-point ──────────────────────────────────────────────────────────────

/// Run the cross-episode merge op over `group_id` (spec P3).
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)` so the
/// deterministic corpus harness (`tests/consolidation_cross_episode_test.rs`) can call
/// it under `feature = "test-utils"` (external test binaries cannot import `pub(crate)`
/// items — E0365). NOT part of the stable public API.
///
/// `dry_run` (ADR-070 Stage 2 shadow gate): when `true`, every clique-cover + F2
/// corroboration decision is computed and emitted EXACTLY as it would live, but the
/// `apply_entity_merge` write is SKIPPED — no entity is fused. `report.count` is
/// incremented identically either way (it counts DECISIONS, not writes), so
/// `DreamSummary.cross_episode_merges` under shadow mode tells an operator "the op
/// WOULD have merged N pairs" (ADR-070 §2.3 DoD-2.3.1). The `mode` label on every
/// `emit_decision` (`shadow`/`applied`) is how a consumer distinguishes the two.
///
/// Args count = 3 — plain args, no params struct needed (TD-042 threshold 3).
#[doc(hidden)]
pub async fn cross_episode(
    graph: &TemporalGraph,
    group_id: &str,
    dry_run: bool,
) -> Result<OpReport> {
    let mut report = OpReport::default();

    // ── Load entity slots (id + normalized label + distinct episode set) ──────────
    let slots = load_entity_slots(graph, group_id).await?;
    if slots.len() < 2 {
        emit_counters(0, 0);
        return Ok(report);
    }

    // Decision-record mode for THIS op's decisions (ADR-070 Fork 1): in shadow mode
    // (`dry_run`) every decision is observed but no write commits, so ALL decisions
    // (merge + skips) carry `Shadow`; otherwise `Applied`. Filtering
    // decision_total{mode=shadow} gives an operator a shadow window's full decision
    // distribution — the Stage-2 observability use case.
    let mode = if dry_run {
        DecisionMode::Shadow
    } else {
        DecisionMode::Applied
    };

    // ── Phase 1: admit candidate pairs (exact + fuzzy), deterministic order ───────
    // Both paths computed over the SAME slot list; an exact pair is never re-admitted
    // by fuzzy (fuzzy only fires on DISTINCT normalized labels). Zero-embedding.
    let candidates = admit_candidates(&slots);

    // ── Phase 2: gate each pair against the STABLE before-state, collect the eligible
    // edges. Both gates are evaluated ONCE, on the pre-merge graph, so a merge applied
    // this run can never manufacture new eligibility mid-pass (the non-convergence
    // trap). Idempotency (P-INV5) then follows: after this run each identity cluster is
    // one entity, so the next run admits no eligible pair.
    let mut eligible: Vec<Candidate> = Vec::new();
    for cand in &candidates {
        // Episode-span gate (P3.2 / P-INV2): the pair must span ≥ 2 DISTINCT episodes.
        // A pair confined to ONE shared episode is a single mention, not recurrence.
        if !spans_distinct_episodes(&slots, &cand.keeper, &cand.loser) {
            continue;
        }
        // Structural-corroboration gate (P3.1b / RISK-001, MANDATORY): shared neighbour
        // OR identical (predicate, object) fact. No shared structure → homonym → DROP.
        match shares_structure(graph, group_id, (&cand.keeper, &cand.loser)).await? {
            StructureOutcome::Corroborated => {
                eligible.push(cand.clone());
            }
            rejected => {
                // Legacy skip counter — the SAME call, args, and fire condition as
                // before (once per rejected pair, for BOTH skip reasons); only its
                // enclosing `if !` became a `match` arm (bool→enum refactor, Risk #13).
                counter!(
                    "kremory.dream.consolidation.cross_episode_homonym_skip_total",
                    "path" => cand.path.as_str(),
                )
                .increment(1);
                // Uniform decision record carries the PRECISE reason so
                // decision_total{outcome} is a true partition — never double-counting
                // one rejected pair into two buckets (ADR-070 Fork 2/3, §3.3).
                let outcome = match rejected {
                    StructureOutcome::NoSharedStructure => "homonym_skip",
                    StructureOutcome::WeakCorroboration => "hub_skip",
                    StructureOutcome::Corroborated => {
                        unreachable!("Corroborated is handled in the arm above")
                    }
                };
                emit_decision(DecisionRecord {
                    op: ConsolidationOpKind::CrossEpisode,
                    mode,
                    // `outcome` is the computed `&'static str` from the match above —
                    // still type-enforced (a `String`/`format!` would not compile), but
                    // note it is field-init SHORTHAND, so the §3.2 `grep 'outcome:'`
                    // literal audit does NOT surface this site (the two skip literals
                    // live in the `let outcome` match arms above).
                    outcome,
                    group_id,
                    entity_refs: &[cand.keeper.as_str(), cand.loser.as_str()],
                    debug_context: None,
                });
                continue;
            }
        }
    }

    // ── Phase 3 (ADR-067 F1): CLIQUE-ONLY clustering, NOT connected-component
    // transitive closure. A same-label triple `A~B`, `B~C`, `A≁C` (a bridge homonym)
    // is NOT a clique — the A-C edge is missing — so connected-component union-find
    // would wrongly fuse all three via the bridge `B`. Clique-only requires every
    // pair in a merged set to be DIRECTLY corroborated.
    //
    // Build adjacency (BTreeMap<id, BTreeSet<id>>) from the eligible edges, partition
    // into connected components (cheap CC pre-partition — cliques never cross
    // components), enumerate maximal cliques per component via a deterministic
    // Bron–Kerbosch (lowest-id pivot), then take a deterministic GREEDY DISJOINT
    // clique cover: sort candidate cliques by (size desc, cumulative integer
    // corroboration-weight desc, min-member-id asc), greedily accept a clique iff ALL
    // members are unclaimed.
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut edge_path: BTreeMap<(String, String), MergePath> = BTreeMap::new();
    for e in &eligible {
        adjacency
            .entry(e.keeper.clone())
            .or_default()
            .insert(e.loser.clone());
        adjacency
            .entry(e.loser.clone())
            .or_default()
            .insert(e.keeper.clone());
        let key = edge_key(&e.keeper, &e.loser);
        edge_path.insert(key, e.path);
    }

    let mut all_cliques: Vec<BTreeSet<String>> = Vec::new();
    for component in connected_components(&adjacency) {
        if component.len() > MAX_LABEL_GROUP {
            report.warnings.push(format!(
                "cross_episode: label group of {} entities exceeds MAX_LABEL_GROUP={}; \
                 skipped for safety",
                component.len(),
                MAX_LABEL_GROUP
            ));
            counter!("kremory.dream.consolidation.cross_episode_group_skipped_total").increment(1);
            let group_refs: Vec<&str> = component.iter().map(|s| s.as_str()).collect();
            emit_decision(DecisionRecord {
                op: ConsolidationOpKind::CrossEpisode,
                mode,
                outcome: "group_size_skip",
                group_id,
                entity_refs: &group_refs,
                debug_context: None,
            });
            continue;
        }
        let component_adjacency: BTreeMap<String, BTreeSet<String>> = component
            .iter()
            .map(|id| {
                (
                    id.clone(),
                    adjacency.get(id).cloned().unwrap_or_default(),
                )
            })
            .collect();
        let cliques = bron_kerbosch_maximal_cliques(&component_adjacency);
        for c in cliques {
            if c.len() >= 2 {
                all_cliques.push(c);
            }
        }
    }

    // Deterministic greedy disjoint cover: sort by (size desc, weight desc, min-id asc).
    all_cliques.sort_by(|a, b| {
        let size_cmp = b.len().cmp(&a.len());
        if size_cmp != std::cmp::Ordering::Equal {
            return size_cmp;
        }
        let wa = clique_weight(a, &edge_path);
        let wb = clique_weight(b, &edge_path);
        let weight_cmp = wb.cmp(&wa);
        if weight_cmp != std::cmp::Ordering::Equal {
            return weight_cmp;
        }
        let min_a = a.iter().min().cloned().unwrap_or_default();
        let min_b = b.iter().min().cloned().unwrap_or_default();
        min_a.cmp(&min_b)
    });

    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut exact_merges = 0usize;
    let mut fuzzy_merges = 0usize;
    for clique in &all_cliques {
        if clique.iter().any(|m| claimed.contains(m)) {
            continue; // overlaps an already-accepted clique — deferred, not merged.
        }
        // Accept this clique: keeper = lowest member id (deterministic tie-break).
        let keeper = clique.iter().min().cloned().unwrap_or_default();
        // Path attribution: Exact if ANY intra-clique eligible edge was exact.
        let clique_path = clique_path(clique, &edge_path);

        // Merge every non-keeper member into the keeper, sorted for determinism.
        let mut losers: Vec<&String> = clique.iter().filter(|m| **m != keeper).collect();
        losers.sort();
        for loser in losers {
            // Shadow gate (ADR-070 §2.3): in `dry_run` the merge decision is still
            // counted (`report.count`) + emitted (`emit_decision`, mode=Shadow), but the
            // destructive `apply_entity_merge` fusion is SKIPPED — no entity is fused.
            if !dry_run {
                apply_entity_merge(graph, loser, &keeper).await?;
            }
            match clique_path {
                MergePath::Exact => exact_merges += 1,
                MergePath::Fuzzy => fuzzy_merges += 1,
            }
            emit_decision(DecisionRecord {
                op: ConsolidationOpKind::CrossEpisode,
                mode,
                outcome: "merged",
                group_id,
                entity_refs: &[keeper.as_str(), loser.as_str()],
                debug_context: None,
            });
            tracing::info!(
                target: "kremory.dream.consolidation.cross_episode",
                group_id,
                keeper = %keeper,
                loser = %loser,
                dry_run,
                "cross-episode merge decision" // covers both shadow + applied modes
            );
        }
        for m in clique {
            claimed.insert(m.clone());
        }
    }

    emit_counters(exact_merges, fuzzy_merges);
    tracing::info!(
        target: "kremory.dream.consolidation.cross_episode",
        group_id,
        exact_merges,
        fuzzy_merges,
        candidates = candidates.len(),
        eligible = eligible.len(),
        cliques = all_cliques.len(),
        "cross-episode sweep complete"
    );

    report.count = exact_merges + fuzzy_merges;
    Ok(report)
}

/// Deterministic edge key for `(a, b)` — sorted pair so `(keeper, loser)` and
/// `(loser, keeper)` map to the same key regardless of arg order.
fn edge_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Cumulative integer corroboration-weight of a clique's INTRA-CLIQUE eligible edges
/// (V3 disjoint-cover tiebreak): each edge that was gated eligible contributes
/// `SCALED_THRESHOLD` (the edge cleared the F2 gate; the exact per-edge score is not
/// retained past the gate, so the tiebreak uses a fixed per-edge weight — still
/// integer, still deterministic). Edges NOT present in `edge_path` (non-adjacent pair
/// within the clique — impossible for a true clique, but defensive) contribute 0.
fn clique_weight(clique: &BTreeSet<String>, edge_path: &BTreeMap<(String, String), MergePath>) -> u64 {
    let members: Vec<&String> = clique.iter().collect();
    let mut total: u64 = 0;
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            let key = edge_key(members[i], members[j]);
            if edge_path.contains_key(&key) {
                total += SCALED_THRESHOLD;
            }
        }
    }
    total
}

/// A clique's path attribution: `Exact` if ANY intra-clique eligible edge was exact,
/// else `Fuzzy` (mirrors the pre-ADR-067 `member_path` rule, now scoped per-clique).
fn clique_path(clique: &BTreeSet<String>, edge_path: &BTreeMap<(String, String), MergePath>) -> MergePath {
    let members: Vec<&String> = clique.iter().collect();
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            let key = edge_key(members[i], members[j]);
            if edge_path.get(&key).copied() == Some(MergePath::Exact) {
                return MergePath::Exact;
            }
        }
    }
    MergePath::Fuzzy
}

/// Partition `adjacency`'s node set into connected components (BTreeSet<String> per
/// component), via iterative BFS over a `BTreeMap` adjacency (deterministic — sorted
/// node iteration + sorted BFS frontier). Cliques never cross components, so this is
/// a cheap O(E) pre-partition before the bounded per-component Bron–Kerbosch.
fn connected_components(adjacency: &BTreeMap<String, BTreeSet<String>>) -> Vec<BTreeSet<String>> {
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut components: Vec<BTreeSet<String>> = Vec::new();
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        let mut component: BTreeSet<String> = BTreeSet::new();
        let mut frontier: Vec<String> = vec![start.clone()];
        while let Some(node) = frontier.pop() {
            if !component.insert(node.clone()) {
                continue;
            }
            visited.insert(node.clone());
            if let Some(neighbours) = adjacency.get(&node) {
                for n in neighbours {
                    if !component.contains(n) {
                        frontier.push(n.clone());
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

/// Deterministic Bron–Kerbosch with pivoting over a bounded (`≤ MAX_LABEL_GROUP`)
/// component adjacency (impl-spec §2). Pivot = lowest id among `P ∪ X` (a fixed rule,
/// never "max degree" which can tie nondeterministically). Returns every MAXIMAL
/// clique (as a `BTreeSet<String>`) found; the caller applies the deterministic
/// disjoint-cover sort/accept over this output, so BK's own emission order is
/// immaterial to the final result.
fn bron_kerbosch_maximal_cliques(
    adjacency: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<BTreeSet<String>> {
    let mut cliques: Vec<BTreeSet<String>> = Vec::new();
    let all_nodes: BTreeSet<String> = adjacency.keys().cloned().collect();
    bron_kerbosch_recurse(
        adjacency,
        BronKerboschState {
            r: BTreeSet::new(),
            p: all_nodes,
            x: BTreeSet::new(),
        },
        &mut cliques,
    );
    cliques
}

/// Bundled recursion state for [`bron_kerbosch_recurse`] — args-as-object (TD-042
/// threshold 3): `r` (current growing clique), `p` (candidates), `x` (excluded).
/// `adjacency` stays a lead positional param (shared/receiver-like dep, project
/// convention) and `out` stays a separate output-accumulator param.
struct BronKerboschState {
    r: BTreeSet<String>,
    p: BTreeSet<String>,
    x: BTreeSet<String>,
}

fn bron_kerbosch_recurse(
    adjacency: &BTreeMap<String, BTreeSet<String>>,
    state: BronKerboschState,
    out: &mut Vec<BTreeSet<String>>,
) {
    let BronKerboschState { r, mut p, mut x } = state;

    if p.is_empty() && x.is_empty() {
        if !r.is_empty() {
            out.push(r);
        }
        return;
    }

    // Deterministic pivot: lowest id among P ∪ X. `p ∪ x` is non-empty (checked by
    // the early-return above), so `.min()` over their chained iterators always
    // yields `Some` — but express that via the `if let` control-flow rather than
    // `.expect()` (no `.expect()`/`.unwrap()` in production src).
    let Some(pivot) = p.iter().chain(x.iter()).min().cloned() else {
        // Structurally unreachable (p/x non-empty here), but fail safely rather
        // than panic if that invariant is ever violated by a future edit.
        return;
    };
    let pivot_neighbours = adjacency.get(&pivot).cloned().unwrap_or_default();

    // Candidates to expand: P \ N(pivot), in sorted order (BTreeSet iteration).
    let candidates: Vec<String> = p
        .iter()
        .filter(|v| !pivot_neighbours.contains(*v))
        .cloned()
        .collect();

    for v in candidates {
        let v_neighbours = adjacency.get(&v).cloned().unwrap_or_default();
        let mut r_next = r.clone();
        r_next.insert(v.clone());
        let p_next: BTreeSet<String> = p.intersection(&v_neighbours).cloned().collect();
        let x_next: BTreeSet<String> = x.intersection(&v_neighbours).cloned().collect();
        bron_kerbosch_recurse(
            adjacency,
            BronKerboschState {
                r: r_next,
                p: p_next,
                x: x_next,
            },
            out,
        );
        p.remove(&v);
        x.insert(v);
    }
}

/// An admitted candidate pair (keeper + loser resolved) + which path admitted it.
#[derive(Debug, Clone)]
struct Candidate {
    keeper: String,
    loser: String,
    path: MergePath,
}

/// Admit exact + fuzzy candidate pairs over the slot list. Deterministic: slots are
/// pre-sorted by id, the pairwise scan is upper-triangular, and the result is stable.
/// Exact (normalized labels equal) is admitted first; the fuzzy loop then admits pairs
/// with DISTINCT normalized labels whose token-shingle Jaccard ≥ threshold, never
/// re-admitting an exact pair.
fn admit_candidates(slots: &[EntitySlot]) -> Vec<Candidate> {
    let shingle_cache: Vec<BTreeSet<String>> = slots
        .iter()
        .map(|s| label_shingles(&s.normalized))
        .collect();

    let mut out: Vec<Candidate> = Vec::new();
    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            let (keeper, loser) = select_keeper(&slots[i].id, &slots[j].id);
            let same_normalized = slots[i].normalized == slots[j].normalized;
            if same_normalized {
                out.push(Candidate {
                    keeper: keeper.to_string(),
                    loser: loser.to_string(),
                    path: MergePath::Exact,
                });
            } else {
                // Fuzzy path: DISTINCT normalized labels, Jaccard ≥ threshold.
                let sim = jaccard(&shingle_cache[i], &shingle_cache[j]);
                if sim >= FUZZY_JACCARD_THRESHOLD {
                    out.push(Candidate {
                        keeper: keeper.to_string(),
                        loser: loser.to_string(),
                        path: MergePath::Fuzzy,
                    });
                }
            }
        }
    }
    out
}

/// Do the two entities span ≥ 2 DISTINCT episodes between them (P3.2 / INV2)?
///
/// Union the two slots' distinct-episode sets; the pair qualifies as cross-episode
/// recurrence only when the union has ≥ 2 distinct `episode_id`s. A pair whose union
/// is a single episode (both mentioned once, in the same episode) is NOT cross-episode.
fn spans_distinct_episodes(slots: &[EntitySlot], id_a: &str, id_b: &str) -> bool {
    let mut union: BTreeSet<i64> = BTreeSet::new();
    for s in slots {
        if s.id == id_a || s.id == id_b {
            union.extend(s.episodes.iter().copied());
        }
    }
    union.len() >= 2
}

// ─── Structural-corroboration gate (P3.1b / RISK-001, ADR-067 F2 rarity-weighted) ─

/// The three-way result of the structural-corroboration gate ([`shares_structure`]).
/// Distinguishes the TWO rejection reasons so the caller can emit the precise
/// consolidation decision outcome (ADR-070 §3.3, Fork 2/3) — a pure homonym vs a
/// hub-weak pair — WITHOUT double-counting one rejected pair into two outcome
/// buckets. (The pre-ADR-070 code returned a bare `bool`, collapsing both reasons.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructureOutcome {
    /// Shared rare structure ≥ threshold → the pair is merge-eligible.
    Corroborated,
    /// No shared neighbours or assertions at all → pure homonym.
    NoSharedStructure,
    /// Shared structure exists but the rarity-weighted score is below threshold
    /// (dominated by hubs) → weak/hub corroboration.
    WeakCorroboration,
}

/// Do `a` and `b` share STRUCTURE ABOVE THE RARITY-WEIGHTED THRESHOLD (DoD-P3.1b,
/// MANDATORY homonym guard, ADR-067 F2)?
///
/// Every shared corroborator (a common third entity OR an identical
/// `(predicate, object)` assertion) contributes an INTEGER weight inversely
/// proportional to its in-group frequency — a RARE shared corroborator is strong
/// evidence; a HUB shared by many entities is weak (Adamic-Adar-shaped). The
/// decision is `Σ WEIGHT_LUT_SCALED[f] (as u64) >= SCALED_THRESHOLD` — fully
/// integer, no `f64` anywhere in the comparison (V2). `f64` appears ONLY in the
/// optional `KREMORY_DEBUG` log below, never in the gate.
///
/// `pair` = `(a, b)` bundled to keep the arg count at the TD-042 threshold-3.
async fn shares_structure(
    graph: &TemporalGraph,
    group_id: &str,
    pair: (&str, &str),
) -> Result<StructureOutcome> {
    let (a, b) = pair;
    let neighbours_a = neighbours_of(graph, group_id, a).await?;
    let neighbours_b = neighbours_of(graph, group_id, b).await?;

    // Shared third-entity corroborators, excluding the two candidates themselves (a
    // fact directly linking a↔b is NOT a shared third neighbour — it is a direct
    // edge, which does not corroborate a SHARED referent).
    let shared_neighbours: BTreeSet<String> = neighbours_a
        .intersection(&neighbours_b)
        .filter(|n| n.as_str() != a && n.as_str() != b)
        .cloned()
        .collect();

    // Shared identical (predicate, object) assertions. A RELATIONAL assertion's
    // object IS the neighbour that same fact contributes to `neighbours_of` — so a
    // relational key whose object already appears in `shared_neighbours` is the SAME
    // underlying corroborator counted twice (once as a shared third-entity, once as
    // an identical assertion), not two independent signals. Dedupe: skip a relational
    // assertion key when its object is already a counted shared neighbour. LITERAL
    // assertion keys (`ov:` prefix) can never collide with a neighbour id (neighbours
    // are always entity ids) so they always count independently.
    let assertions_a = assertions_of(graph, group_id, a).await?;
    let assertions_b = assertions_of(graph, group_id, b).await?;
    let shared_assertions: BTreeSet<String> = assertions_a
        .intersection(&assertions_b)
        .filter(|key| {
            let object_component = key.rsplit('\u{1F}').next().unwrap_or(key.as_str());
            object_component.starts_with("ov:") || !shared_neighbours.contains(object_component)
        })
        .cloned()
        .collect();

    if shared_neighbours.is_empty() && shared_assertions.is_empty() {
        return Ok(StructureOutcome::NoSharedStructure);
    }

    // Accumulate in sorted-id order (BTreeSet iteration = deterministic fixed
    // reduction sequence, though u64 addition is associative so order does not
    // affect the result). Each corroborator's weight = its IN-GROUP degree/frequency
    // looked up via WEIGHT_LUT_SCALED; `f > HUB_DEGREE_CAP` contributes integer 0.
    let mut acc: u64 = 0;
    for neighbour in &shared_neighbours {
        let deg = neighbour_degree(graph, group_id, neighbour).await?;
        acc += scaled_weight(deg);
    }
    for key in &shared_assertions {
        let freq = assertion_frequency(graph, group_id, key).await?;
        acc += scaled_weight(freq);
    }

    let outcome = acc >= SCALED_THRESHOLD;
    counter!(
        "kremory.dream.consolidation.cross_episode_corroboration_score_total",
        "outcome" => if outcome { "pass" } else { "fail" },
    )
    .increment(1);
    if std::env::var("KREMORY_DEBUG").is_ok() {
        // Float appears ONLY here (V2 §3 step 5) — human-readable, never the gate.
        tracing::debug!(
            target: "kremory.dream.consolidation.cross_episode",
            group_id,
            a,
            b,
            score = acc as f64 / SCALE as f64,
            scaled_acc = acc,
            scaled_threshold = SCALED_THRESHOLD,
            shared_neighbours = shared_neighbours.len(),
            shared_assertions = shared_assertions.len(),
            "cross_episode corroboration score"
        );
    }

    Ok(if outcome {
        StructureOutcome::Corroborated
    } else {
        StructureOutcome::WeakCorroboration
    })
}

/// Integer weight for a corroborator with in-group degree/frequency `f`: looks up
/// `WEIGHT_LUT_SCALED[f]` for `f ∈ 1..=HUB_DEGREE_CAP`, else (hub cap exceeded, or the
/// degenerate `f == 0`) contributes integer `0` — a pure-integer branch, no float.
fn scaled_weight(f: u32) -> u64 {
    if f == 0 || f > HUB_DEGREE_CAP {
        0
    } else {
        WEIGHT_LUT_SCALED[f as usize]
    }
}

/// In-group DEGREE of a shared neighbour `neighbour` within `group_id` (ADR-067 F2):
/// the count of DISTINCT entities that reference `neighbour` via a live,
/// corroboration-LIVE fact (`expired_at IS NULL AND invalid_at IS NULL AND
/// corroboration_inert = 0`) — same live-fact filter as [`neighbours_of`], INCLUDING
/// the C0 `corroboration_inert = 0` filter, so an inherited (merge-created) edge can
/// never inflate a neighbour's degree. Deterministic integer.
async fn neighbour_degree(graph: &TemporalGraph, group_id: &str, neighbour: &str) -> Result<u32> {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(DISTINCT ref) FROM ( \
                 SELECT subject_id AS ref FROM facts \
                 WHERE group_id = ?1 AND object_id = ?2 \
                   AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0 \
                 UNION \
                 SELECT object_id AS ref FROM facts \
                 WHERE group_id = ?1 AND subject_id = ?2 AND object_id IS NOT NULL \
                   AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0 \
             )",
            libsql::params![group_id, neighbour],
        )
        .await?;
    let count: i64 = rows
        .next()
        .await?
        .map(|row| row.get::<i64>(0))
        .transpose()?
        .unwrap_or(0);
    Ok(count.max(0) as u32)
}

/// In-group FREQUENCY of an identical `(predicate, object)` assertion key `key`
/// within `group_id` (ADR-067 F2): the count of DISTINCT `subject_id`s that assert
/// this exact key via a live, corroboration-LIVE fact. `key` is the same stable
/// string form produced by [`assertions_of`] (`"<predicate>\u{1F}<object>"`).
/// Deterministic integer.
async fn assertion_frequency(graph: &TemporalGraph, group_id: &str, key: &str) -> Result<u32> {
    let Some((predicate, object_key)) = key.split_once('\u{1F}') else {
        return Ok(0);
    };
    let mut rows = if let Some(object_value) = object_key.strip_prefix("ov:") {
        graph
            .conn
            .query(
                "SELECT COUNT(DISTINCT subject_id) FROM facts \
                 WHERE group_id = ?1 AND predicate = ?2 AND object_value = ?3 \
                   AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0",
                libsql::params![group_id, predicate, object_value],
            )
            .await?
    } else {
        graph
            .conn
            .query(
                "SELECT COUNT(DISTINCT subject_id) FROM facts \
                 WHERE group_id = ?1 AND predicate = ?2 AND object_id = ?3 \
                   AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0",
                libsql::params![group_id, predicate, object_key],
            )
            .await?
    };
    let count: i64 = rows
        .next()
        .await?
        .map(|row| row.get::<i64>(0))
        .transpose()?
        .unwrap_or(0);
    Ok(count.max(0) as u32)
}

/// The set of NEIGHBOUR entity ids of `entity` in `group_id`: every distinct
/// `subject_id`/`object_id` that co-occurs with `entity` in a fact (i.e. the OTHER
/// endpoint of any fact `entity` is an endpoint of). `entity` itself is not excluded
/// here (a self-fact would include it) — the intersection filter in
/// [`shares_structure`] drops `{a, b}` from the common set.
async fn neighbours_of(
    graph: &TemporalGraph,
    group_id: &str,
    entity: &str,
) -> Result<BTreeSet<String>> {
    // A fact touching `entity`: its OTHER endpoint (the neighbour) is `object_id` when
    // `entity` is the subject, or `subject_id` when `entity` is the object. Only
    // relational facts (`object_id IS NOT NULL`) contribute a graph neighbour.
    //
    // LIVE-FACTS ONLY (Quinn F3): a RETIRED (`expired_at`/`invalid_at` set) fact must
    // NOT corroborate an identity merge. On a bi-temporal substrate a superseded/
    // invalidated edge is no-longer-asserted structure; P1 supersession + P2 archive
    // run BEFORE P3 in the consolidation order (ADR-066 §2.6), so by the time P3 reads
    // corroboration the retired edges are already closed. Filtering them here is the
    // cause-fix — corroboration reflects the CURRENT graph, not its history.
    //
    // CORROBORATION-LIVE ONLY (ADR-067 §C0, the convergence fix): `corroboration_inert
    // = 0` on BOTH UNION arms — a merge-inherited fact endpoint must NOT corroborate a
    // NEW cross-episode eligibility (impl-spec §C0 DoD: "a neighbour inherited via the
    // loser's OBJECT-position fact is equally inert"). Without this, the keeper of a
    // deferred bridge would inherit the loser's neighbours and re-open eligibility
    // against the bridge partner on the NEXT pass.
    let mut rows = graph
        .conn
        .query(
            "SELECT object_id FROM facts \
             WHERE group_id = ?1 AND subject_id = ?2 AND object_id IS NOT NULL \
               AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0 \
             UNION \
             SELECT subject_id FROM facts \
             WHERE group_id = ?1 AND object_id = ?2 \
               AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0",
            libsql::params![group_id, entity],
        )
        .await?;
    let mut out: BTreeSet<String> = BTreeSet::new();
    while let Some(row) = rows.next().await? {
        let neighbour: Option<String> = row.get(0)?;
        if let Some(n) = neighbour {
            out.insert(n);
        }
    }
    Ok(out)
}

/// The set of `(predicate, object)` assertions made BY `entity` (as subject) in
/// `group_id`, each rendered as a stable string key `"<predicate>\u{1F}<object>"`
/// where object is `object_id` when relational or `ov:<object_value>` when literal.
/// Two entities asserting an identical key corroborate a shared referent (P3.1b ii).
async fn assertions_of(
    graph: &TemporalGraph,
    group_id: &str,
    entity: &str,
) -> Result<BTreeSet<String>> {
    // LIVE-FACTS ONLY (Quinn F3): mirror `neighbours_of` — a retired
    // (`expired_at`/`invalid_at` set) assertion is no-longer-asserted structure and
    // must NOT corroborate an identity merge. Governing spec: ADR-066 §1.2 (bi-temporal
    // columns) + §2.6 (P1/P2 close retired facts before P3 reads corroboration).
    //
    // CORROBORATION-LIVE ONLY (ADR-067 §C0): `corroboration_inert = 0` — mirrors
    // `neighbours_of`'s C0 filter (impl-spec §C0 step 3).
    let mut rows = graph
        .conn
        .query(
            "SELECT predicate, object_id, object_value FROM facts \
             WHERE group_id = ?1 AND subject_id = ?2 \
               AND expired_at IS NULL AND invalid_at IS NULL AND corroboration_inert = 0",
            libsql::params![group_id, entity],
        )
        .await?;
    let mut out: BTreeSet<String> = BTreeSet::new();
    while let Some(row) = rows.next().await? {
        let predicate: String = row.get(0)?;
        let object_id: Option<String> = row.get(1)?;
        let object_value: Option<String> = row.get(2)?;
        // Object rendering: relational (object_id) vs literal (ov:object_value). A fact
        // with neither a resolvable object_id NOR an object_value carries no corroborating
        // object → skip (an unanchored assertion is not evidence of a shared referent).
        let object_key = match (object_id, object_value) {
            (Some(oid), _) => oid,
            (None, Some(ov)) => format!("ov:{ov}"),
            (None, None) => continue,
        };
        out.insert(format!("{predicate}\u{1F}{object_key}"));
    }
    Ok(out)
}

// ─── Slot loading ────────────────────────────────────────────────────────────────

/// Load every entity in `group_id` with its normalized label + distinct-episode set.
/// Entities are returned SORTED by id (deterministic pairwise scan + keeper tie-break).
async fn load_entity_slots(graph: &TemporalGraph, group_id: &str) -> Result<Vec<EntitySlot>> {
    // Distinct episodes per entity via episodic_edges (namespace-scoped by
    // entity_group_id — entity ids are namespace-unique name-slugs). LEFT JOIN so an
    // entity with zero episode anchors still loads (it simply cannot span ≥2 episodes).
    // `e.id` IS the label comparand (the name-slug); `entities.label` was dropped by
    // Migration 009 (`defs_b.rs:391`). LEFT JOIN keeps entities with zero episode
    // anchors (they simply cannot span ≥2 episodes → never a candidate).
    let mut rows = graph
        .conn
        .query(
            "SELECT e.id, ee.episode_id \
             FROM entities e \
             LEFT JOIN episodic_edges ee \
               ON ee.entity_id = e.id \
              AND (ee.entity_group_id = e.group_id OR ee.entity_group_id IS NULL) \
             WHERE e.group_id = ?1 \
             ORDER BY e.id",
            libsql::params![group_id],
        )
        .await?;

    // Accumulate episodes per entity id (BTreeMap → deterministic id order).
    let mut acc: BTreeMap<String, (String, BTreeSet<i64>)> = BTreeMap::new();
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let episode_id: Option<i64> = row.get(1)?;
        let normalized = normalize_label(&id);
        let entry = acc
            .entry(id)
            .or_insert_with(|| (normalized, BTreeSet::new()));
        if let Some(ep) = episode_id {
            entry.1.insert(ep);
        }
    }

    Ok(acc
        .into_iter()
        .map(|(id, (normalized, episodes))| EntitySlot {
            id,
            normalized,
            episodes,
        })
        .collect())
}

/// Emit the source-attributed merge counter split `{path=exact|fuzzy}` (DoD-P3.4).
/// The o11y cross-check asserts the SUM equals `OpReport.count`.
fn emit_counters(exact_merges: usize, fuzzy_merges: usize) {
    counter!(
        "kremory.dream.consolidation.cross_episode_merges_total",
        "path" => "exact",
    )
    .increment(exact_merges as u64);
    counter!(
        "kremory.dream.consolidation.cross_episode_merges_total",
        "path" => "fuzzy",
    )
    .increment(fuzzy_merges as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{
        InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
    };
    use crate::core::schema::TemporalGraph;
    use chrono::Utc;

    // ── L1 unit: pure helpers (normalize / jaccard / keeper) ─────────────────────

    #[test]
    fn normalize_label_case_folds_and_collapses_whitespace() {
        assert_eq!(normalize_label("  John   SMITH "), "john smith");
        assert_eq!(normalize_label("john smith"), "john smith");
        assert_eq!(normalize_label("ACME\tCorp\n"), "acme corp");
        assert_eq!(normalize_label("Single"), "single");
        assert_eq!(normalize_label(""), "");
    }

    #[test]
    fn normalize_label_distinguishes_genuinely_different_labels() {
        assert_ne!(normalize_label("John Smith"), normalize_label("Jane Smith"));
        assert_ne!(normalize_label("Acme"), normalize_label("Acme Corp"));
    }

    #[test]
    fn jaccard_identical_shingles_is_one() {
        let a = label_shingles("john michael smith");
        let b = label_shingles("john michael smith");
        assert!((jaccard(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jaccard_high_overlap_crosses_threshold() {
        // "john michael smith" vs "john michael smyth-smith"? Use a clean ≥0.9 case:
        // 10-token labels sharing 9 → 9/11 ≈ 0.818 (below). Sharing 10 of 11 tokens:
        let a: BTreeSet<String> = (0..10).map(|n| format!("t{n}")).collect();
        let mut b = a.clone();
        b.remove("t0"); // b has 9 tokens, all in a; union = 10, inter = 9 → 0.9 exactly.
        assert!(
            (jaccard(&a, &b) - 0.9).abs() < 1e-12,
            "9 of 10 shared → Jaccard 0.9 (inclusive threshold)"
        );
        assert!(jaccard(&a, &b) >= FUZZY_JACCARD_THRESHOLD);
    }

    #[test]
    fn jaccard_low_overlap_below_threshold() {
        let a = label_shingles("alpha beta gamma");
        let b = label_shingles("alpha delta epsilon");
        // shared {alpha}=1, union {alpha,beta,gamma,delta,epsilon}=5 → 0.2.
        assert!((jaccard(&a, &b) - 0.2).abs() < 1e-12);
        assert!(jaccard(&a, &b) < FUZZY_JACCARD_THRESHOLD);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        let a = label_shingles("alpha");
        let b = label_shingles("omega");
        assert_eq!(jaccard(&a, &b), 0.0);
    }

    #[test]
    fn select_keeper_is_lowest_id_deterministic() {
        assert_eq!(select_keeper("ent-a", "ent-b"), ("ent-a", "ent-b"));
        assert_eq!(select_keeper("ent-b", "ent-a"), ("ent-a", "ent-b"));
        assert_eq!(select_keeper("zzz", "aaa"), ("aaa", "zzz"));
    }

    // ── DB plant helpers ─────────────────────────────────────────────────────────

    /// Insert an entity whose `id` IS the identity comparand (name-slug). The op
    /// normalizes the id (case-fold + whitespace-collapse), so two DISTINCT ids that
    /// normalize identically (e.g. `"John Smith"` / `"john  smith"`) are an exact-path
    /// candidate pair. `entities.label` is gone (Migration 009) — nothing to set.
    async fn insert_entity(graph: &TemporalGraph, gid: &str, id: &str) {
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({ "name": id }),
                group_id: Some(gid),
            })
            .await
            .expect("insert entity");
    }

    async fn new_episode(graph: &TemporalGraph) -> i64 {
        graph
            .insert_episode(InsertEpisodeParams {
                content: "episode content",
                timestamp: Utc::now(),
                source_type: Some("transcript"),
                metadata: None,
            })
            .await
            .expect("episode")
    }

    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn anchor(graph: &TemporalGraph, gid: &str, episode_id: i64, entity: &str) {
        graph
            .insert_episodic_edge(InsertEpisodicEdgeParams {
                episode_id,
                entity_id: entity,
                entity_group_id: Some(gid),
                role: "mention",
            })
            .await
            .expect("edge");
    }

    /// Plant a relational fact `subject --predicate--> object` in `gid`.
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn fact_rel(
        graph: &TemporalGraph,
        gid: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) {
        let now = Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
                  subject_group_id, object_group_id, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
                libsql::params![subject, predicate, object, now.clone(), now, gid, gid, gid],
            )
            .await
            .expect("plant relational fact");
    }

    /// Plant a literal fact `subject --predicate--> "value"` in `gid`.
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn fact_lit(
        graph: &TemporalGraph,
        gid: &str,
        subject: &str,
        predicate: &str,
        value: &str,
    ) {
        let now = Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_value, valid_from, recorded_at, group_id, \
                  subject_group_id, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1.0)",
                libsql::params![subject, predicate, value, now.clone(), now, gid, gid],
            )
            .await
            .expect("plant literal fact");
    }

    async fn entities_in_group(graph: &TemporalGraph, gid: &str) -> Vec<String> {
        graph
            .list_entities_in_group(gid)
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.id)
            .collect()
    }

    // NOTE ON TEST IDS: `entities.id` IS the identity comparand (name-slug; the
    // `label` column was dropped by Migration 009). A same-name pair therefore uses
    // two DISTINCT raw ids that NORMALIZE identically — e.g. `"John Smith"` (raw) and
    // `"john  smith"` (raw, double space) both normalize to `"john smith"`. They are
    // distinct `(id, group_id)` PK rows within one namespace, so both plant cleanly.

    // ── DoD-P3.1: exact-label + shared neighbour + ≥2 episodes → MERGE ───────────

    #[tokio::test]
    async fn exact_label_shared_neighbour_two_episodes_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // Two raw ids that normalize identically → exact-path candidate pair. Distinct
        // episodes + a shared neighbour `acme` → genuine cross-episode recurrence.
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "acme").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;

        // Shared neighbour: both work at acme.
        fact_rel(&graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "acme").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(report.count, 1, "exact-label corroborated pair merges once");

        let ids = entities_in_group(&graph, gid).await;
        // Keeper = lowest id: "John Smith" (uppercase 'J' 0x4A < lowercase 'j' 0x6A).
        assert!(ids.contains(&"John Smith".to_string()), "keeper survives");
        assert!(
            !ids.contains(&"john  smith".to_string()),
            "loser merged away"
        );
        assert!(ids.contains(&"acme".to_string()), "neighbour untouched");
    }

    // ── ADR-070 C1: shadow-mode (dry_run) gate ──────────────────────────────────

    /// Plant the canonical exact-label corroborated mergeable pair (mirrors
    /// `exact_label_shared_neighbour_two_episodes_merges`): 3 entities, two of which
    /// normalize identically + share neighbour `acme` across 2 episodes → 1 merge.
    async fn plant_mergeable_pair(graph: &TemporalGraph, gid: &str) {
        insert_entity(graph, gid, "John Smith").await;
        insert_entity(graph, gid, "john  smith").await;
        insert_entity(graph, gid, "acme").await;
        let e1 = new_episode(graph).await;
        let e2 = new_episode(graph).await;
        anchor(graph, gid, e1, "John Smith").await;
        anchor(graph, gid, e2, "john  smith").await;
        fact_rel(graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(graph, gid, "john  smith", "works_at", "acme").await;
    }

    /// Count `decision_total{op=cross_episode, mode, outcome=merged}` in a snapshot.
    fn merged_decision_count(
        snapshotter: &metrics_util::debugging::Snapshotter,
        mode: &str,
    ) -> u64 {
        use metrics_util::debugging::DebugValue;
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter_map(|(composite_key, _, _, value)| {
                let key = composite_key.key();
                if key.name() != "kremory.dream.consolidation.decision_total" {
                    return None;
                }
                let labels: std::collections::HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("op").copied() != Some("cross_episode")
                    || labels.get("mode").copied() != Some(mode)
                    || labels.get("outcome").copied() != Some("merged")
                {
                    return None;
                }
                match value {
                    DebugValue::Counter(n) => Some(n),
                    _ => None,
                }
            })
            .sum()
    }

    /// ADR-070 §2.3 DoD-2.3.1: the SAME mergeable fixture yields an IDENTICAL
    /// `report.count` under dry_run true/false (count reflects DECISIONS), but the
    /// entity row count is reduced by 1 under `false` (real fusion) and UNCHANGED
    /// under `true` (shadow skips the write). The `mode` label differs accordingly.
    #[tokio::test]
    async fn cross_episode_dry_run_skips_write_identical_count() {
        use metrics_util::debugging::DebuggingRecorder;

        // Applied (dry_run=false): the write commits → the loser is fused away.
        let applied = TemporalGraph::open_in_memory().await.expect("open");
        plant_mergeable_pair(&applied, "g1").await;
        let entities_before = entities_in_group(&applied, "g1").await.len();
        let rec_a = DebuggingRecorder::new();
        let snap_a = rec_a.snapshotter();
        let report_applied = {
            let _g = metrics::set_default_local_recorder(&rec_a);
            cross_episode(&applied, "g1", false).await.expect("applied")
        };
        let entities_after_applied = entities_in_group(&applied, "g1").await.len();

        // Shadow (dry_run=true): identical fixture on a fresh graph → same decision
        // count, but NO fusion.
        let shadow = TemporalGraph::open_in_memory().await.expect("open");
        plant_mergeable_pair(&shadow, "g1").await;
        let rec_s = DebuggingRecorder::new();
        let snap_s = rec_s.snapshotter();
        let report_shadow = {
            let _g = metrics::set_default_local_recorder(&rec_s);
            cross_episode(&shadow, "g1", true).await.expect("shadow")
        };
        let entities_after_shadow = entities_in_group(&shadow, "g1").await.len();

        // (a) report.count identical across modes (counts DECISIONS, not writes).
        assert_eq!(
            report_applied.count, report_shadow.count,
            "report.count is identical across dry_run true/false"
        );
        assert_eq!(report_applied.count, 1, "the corroborated pair is one decision");
        // (b) entity ROW count: applied fuses (−1); shadow leaves it untouched.
        assert_eq!(
            entities_after_applied,
            entities_before - 1,
            "applied (dry_run=false): the loser was fused away"
        );
        assert_eq!(
            entities_after_shadow, entities_before,
            "shadow (dry_run=true): no entity was fused — the write was skipped"
        );
        // (c) the `mode` label differs (shadow vs applied), with no cross-contamination.
        assert_eq!(
            merged_decision_count(&snap_a, "applied"),
            1,
            "applied run emits decision_total{{mode=applied,outcome=merged}}"
        );
        assert_eq!(
            merged_decision_count(&snap_s, "shadow"),
            1,
            "shadow run emits decision_total{{mode=shadow,outcome=merged}}"
        );
        assert_eq!(merged_decision_count(&snap_a, "shadow"), 0);
        assert_eq!(merged_decision_count(&snap_s, "applied"), 0);
    }

    /// ADR-070 §2.2 compound default: a first enablement of cross-episode merges lands
    /// in shadow mode until the operator explicitly opts into Stage 5 (apply).
    #[test]
    fn dream_opts_cross_episode_dry_run_defaults_true() {
        assert!(
            crate::memory::types::DreamOpts::default().cross_episode_dry_run,
            "DreamOpts::default().cross_episode_dry_run must be true (ADR-070 §2.2)"
        );
    }

    // ── DoD-P3.1b / RISK-001: same label, NO shared neighbour → DO NOT MERGE ─────
    // THE safety category: homonym (two distinct real referents sharing a name).

    #[tokio::test]
    async fn same_label_distinct_referent_no_corroboration_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "lawfirm").await;
        insert_entity(&graph, gid, "olympics").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;

        // DISJOINT structure: lawyer vs athlete — no shared neighbour, no identical
        // assertion. Same label alone must NOT merge.
        fact_rel(&graph, gid, "John Smith", "works_at", "lawfirm").await;
        fact_rel(&graph, gid, "john  smith", "competed_in", "olympics").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "homonym: same label + NO shared structure must NOT merge (EXACT 0)"
        );
        let ids = entities_in_group(&graph, gid).await;
        assert!(ids.contains(&"John Smith".to_string()), "both survive");
        assert!(ids.contains(&"john  smith".to_string()), "both survive");
    }

    // ── DoD-P3.1b (ii): identical (predicate, object) assertion corroborates ─────

    #[tokio::test]
    async fn same_label_identical_literal_assertion_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "Acme Corp").await;
        insert_entity(&graph, gid, "acme  corp").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "Acme Corp").await;
        anchor(&graph, gid, e2, "acme  corp").await;

        // Identical literal assertion (predicate + object_value) corroborates.
        fact_lit(&graph, gid, "Acme Corp", "headquartered_in", "Boston").await;
        fact_lit(&graph, gid, "acme  corp", "headquartered_in", "Boston").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 1,
            "identical literal assertion corroborates merge"
        );
    }

    // ── ADR-067 F2: rarity-weighted corroboration (integer-decided) ─────────────

    /// Plant `n` DISTINCT entities that all assert `subject --located_in--> object`
    /// (relational), raising `object`'s in-group degree to `n`. Used to construct a
    /// HUB shared neighbour whose degree exceeds `HUB_DEGREE_CAP`. Referencer ids are
    /// namespaced by `object` so planting hubs for TWO different objects in the same
    /// test (`two_hubs_shared_does_not_merge`) never collides on the `(id, group_id)`
    /// PK.
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn plant_hub_referencers(graph: &TemporalGraph, gid: &str, object: &str, n: u32) {
        for i in 0..n {
            let id = format!("hub_ref_{object}_{i}");
            insert_entity(graph, gid, &id).await;
            fact_rel(graph, gid, &id, "located_in", object).await;
        }
    }

    #[tokio::test]
    async fn single_hub_shared_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "Boston").await;
        // Boston's in-group degree > HUB_DEGREE_CAP (8): plant HUB_DEGREE_CAP+2 = 10
        // OTHER distinct referencers, THEN both candidates also reference Boston —
        // degree = 10 + 2 = 12 > 8.
        plant_hub_referencers(&graph, gid, "Boston", HUB_DEGREE_CAP + 2).await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        fact_rel(&graph, gid, "John Smith", "located_in", "Boston").await;
        fact_rel(&graph, gid, "john  smith", "located_in", "Boston").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "single HUB shared neighbour (deg > HUB_DEGREE_CAP) contributes integer \
             weight 0 → must NOT merge"
        );
    }

    #[tokio::test]
    async fn two_hubs_shared_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "Boston").await;
        insert_entity(&graph, gid, "MegaCorp").await;
        // BOTH corroborators are HUBS (deg > HUB_DEGREE_CAP): each contributes integer
        // weight 0, so the sum is 0 < SCALED_THRESHOLD — proves a flat "≥2 signals"
        // rule would wrongly merge here; integer rarity-weighting does not.
        plant_hub_referencers(&graph, gid, "Boston", HUB_DEGREE_CAP + 2).await;
        plant_hub_referencers(&graph, gid, "MegaCorp", HUB_DEGREE_CAP + 2).await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        fact_rel(&graph, gid, "John Smith", "located_in", "Boston").await;
        fact_rel(&graph, gid, "john  smith", "located_in", "Boston").await;
        fact_rel(&graph, gid, "John Smith", "member_of", "MegaCorp").await;
        fact_rel(&graph, gid, "john  smith", "member_of", "MegaCorp").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "TWO hub corroborators, BOTH deg > HUB_DEGREE_CAP → each weight 0 → \
             sum 0 < SCALED_THRESHOLD → must NOT merge"
        );
    }

    #[tokio::test]
    async fn single_rare_neighbour_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "niche_org").await;
        // "niche_org" is referenced ONLY by these two candidates → in-group degree 2 →
        // WEIGHT_LUT_SCALED[2] = 524_288 == SCALED_THRESHOLD → MERGE (the single-rare-
        // signal accept case).

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        fact_rel(&graph, gid, "John Smith", "works_at", "niche_org").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "niche_org").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 1,
            "one shared neighbour of degree 2 → scaled weight 524_288 >= threshold → MERGE"
        );
    }

    #[tokio::test]
    async fn accumulation_boundary_is_exact_integer() {
        // TWO deg-8 corroborators → WEIGHT_LUT_SCALED[8] + WEIGHT_LUT_SCALED[8] =
        // 262_144 + 262_144 = 524_288 == SCALED_THRESHOLD (exact integer `>=`) → MERGE.
        {
            let graph = TemporalGraph::open_in_memory().await.expect("open");
            let gid = "g1";
            insert_entity(&graph, gid, "John Smith").await;
            insert_entity(&graph, gid, "john  smith").await;
            insert_entity(&graph, gid, "org_a").await;
            insert_entity(&graph, gid, "org_b").await;
            // org_a, org_b each at degree EXACTLY HUB_DEGREE_CAP (8): plant
            // HUB_DEGREE_CAP-2 = 6 other referencers, then both candidates reference it
            // too → degree = 6 + 2 = 8.
            plant_hub_referencers(&graph, gid, "org_a", HUB_DEGREE_CAP - 2).await;
            plant_hub_referencers(&graph, gid, "org_b", HUB_DEGREE_CAP - 2).await;

            let e1 = new_episode(&graph).await;
            let e2 = new_episode(&graph).await;
            anchor(&graph, gid, e1, "John Smith").await;
            anchor(&graph, gid, e2, "john  smith").await;
            fact_rel(&graph, gid, "John Smith", "member_of", "org_a").await;
            fact_rel(&graph, gid, "john  smith", "member_of", "org_a").await;
            fact_rel(&graph, gid, "John Smith", "member_of", "org_b").await;
            fact_rel(&graph, gid, "john  smith", "member_of", "org_b").await;

            let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
            assert_eq!(
                report.count, 1,
                "TWO deg-8 corroborators sum to EXACTLY SCALED_THRESHOLD (integer `>=`) → MERGE"
            );
        }
        // ONE deg-8 corroborator → 262_144 < 524_288 → does NOT merge.
        {
            let graph = TemporalGraph::open_in_memory().await.expect("open");
            let gid = "g1";
            insert_entity(&graph, gid, "John Smith").await;
            insert_entity(&graph, gid, "john  smith").await;
            insert_entity(&graph, gid, "org_a").await;
            plant_hub_referencers(&graph, gid, "org_a", HUB_DEGREE_CAP - 2).await;

            let e1 = new_episode(&graph).await;
            let e2 = new_episode(&graph).await;
            anchor(&graph, gid, e1, "John Smith").await;
            anchor(&graph, gid, e2, "john  smith").await;
            fact_rel(&graph, gid, "John Smith", "member_of", "org_a").await;
            fact_rel(&graph, gid, "john  smith", "member_of", "org_a").await;

            let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
            assert_eq!(
                report.count, 0,
                "ONE deg-8 corroborator: 262_144 < SCALED_THRESHOLD 524_288 → must NOT merge"
            );
        }
    }

    /// `#[test]` re-derives `WEIGHT_LUT_SCALED` from `f64::log2` (round-half-up scaled
    /// to `SCALE`) and asserts BYTE-MATCH against the pinned production array. THIS
    /// test uses float; the PRODUCTION decision path never does (V2). Guards against
    /// a transcription error in the hand-pinned constant.
    #[test]
    fn weight_lut_scaled_matches_float_rederivation() {
        for f in 1..=HUB_DEGREE_CAP {
            let conceptual = 1.0 / (1.0 + (f as f64).log2());
            let rederived = (conceptual * SCALE as f64 + 0.5).floor() as u64;
            assert_eq!(
                WEIGHT_LUT_SCALED[f as usize], rederived,
                "WEIGHT_LUT_SCALED[{f}] pinned={} != float-rederived={rederived}",
                WEIGHT_LUT_SCALED[f as usize]
            );
        }
        // Index 0 is unused (f starts at 1) — pinned as 0.
        assert_eq!(WEIGHT_LUT_SCALED[0], 0, "index 0 is unused, pinned as 0");
        // SCALED_THRESHOLD == round(CORROBORATION_THRESHOLD_NUM * SCALE).
        let rederived_threshold = (CORROBORATION_THRESHOLD_NUM * SCALE as f64 + 0.5).floor() as u64;
        assert_eq!(
            SCALED_THRESHOLD, rederived_threshold,
            "SCALED_THRESHOLD must equal round(CORROBORATION_THRESHOLD_NUM * SCALE)"
        );
    }

    // ── DoD-P3.2: fuzzy Jaccard ≥0.9 + corroboration → MERGE ─────────────────────

    #[tokio::test]
    async fn fuzzy_jaccard_high_corroborated_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // id_a = 10 tokens; id_b = 9 of those tokens (subset) → union 10,
        // intersection 9 → Jaccard 0.9 (== inclusive fuzzy threshold). The ids ARE the
        // labels, so token-shingle Jaccard is computed over the id words directly.
        let id_a = "a b c d e f g h i j";
        let id_b = "a b c d e f g h i";
        insert_entity(&graph, gid, id_a).await;
        insert_entity(&graph, gid, id_b).await;
        insert_entity(&graph, gid, "org").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, id_a).await;
        anchor(&graph, gid, e2, id_b).await;
        fact_rel(&graph, gid, id_a, "member_of", "org").await;
        fact_rel(&graph, gid, id_b, "member_of", "org").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(report.count, 1, "fuzzy ≥0.9 corroborated pair merges");
    }

    // ── DoD-P3.1b / RISK-001: fuzzy high Jaccard but NO corroboration → NO merge ─

    #[tokio::test]
    async fn fuzzy_jaccard_high_no_corroboration_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        let id_a = "a b c d e f g h i j";
        let id_b = "a b c d e f g h i";
        insert_entity(&graph, gid, id_a).await;
        insert_entity(&graph, gid, id_b).await;
        insert_entity(&graph, gid, "lawfirm").await;
        insert_entity(&graph, gid, "olympics").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, id_a).await;
        anchor(&graph, gid, e2, id_b).await;
        // Disjoint structure — high label similarity but no shared referent.
        fact_rel(&graph, gid, id_a, "works_at", "lawfirm").await;
        fact_rel(&graph, gid, id_b, "competed_in", "olympics").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "fuzzy label match with NO shared structure must NOT merge (EXACT 0)"
        );
    }

    // ── DoD-P3.3: exact label but only ONE shared episode → NOT cross-episode ────

    #[tokio::test]
    async fn same_label_same_episode_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "acme").await;

        // BOTH anchored to the SAME single episode → not cross-episode recurrence.
        let e1 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e1, "john  smith").await;
        // Even WITH corroboration, the single-episode gate blocks the merge.
        fact_rel(&graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "acme").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "same-episode-only pair is not cross-episode recurrence (EXACT 0)"
        );
    }

    // ── DoD-P3.3: cosine-near-dup but lexically distinct → canonicalize's job ────

    #[tokio::test]
    async fn cosine_near_dup_lexically_distinct_is_left_alone() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // Genuinely different labels (low Jaccard) that an embedding cosine MIGHT
        // consider similar — cross_episode fires on LABEL ONLY, so it leaves them.
        insert_entity(&graph, gid, "big blue").await;
        insert_entity(&graph, gid, "ibm").await;
        insert_entity(&graph, gid, "acme").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "big blue").await;
        anchor(&graph, gid, e2, "ibm").await;
        // Even a shared neighbour must not cause a merge — the labels are lexically
        // distinct, so no candidate pair is admitted at all.
        fact_rel(&graph, gid, "big blue", "makes", "acme").await;
        fact_rel(&graph, gid, "ibm", "makes", "acme").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "lexically-distinct pair is canonicalize's cosine job, not cross_episode"
        );
    }

    // ── DoD-P3.1b / transitivity: 3-entity cluster, all edges corroborated → fuse ─
    // Deterministic 3-entity fixture (Quinn F4). The corpus harness Row is 2-entity
    // only, so the transitive union-find path (spec impl-spec §3 P3.1b + ADR-066 §2.2:
    // "A~C + B~C collapses {A,B,C} to one keeper") has no corpus coverage — pinned here.

    #[tokio::test]
    async fn transitive_all_corroborated_fuses_triple_to_root() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // Three raw ids that all normalize to "john smith" → three exact-path candidate
        // pairs (A~B, A~C, B~C). Distinct episodes each. ADR-067 F2 (rarity-weighted):
        // a corroborator shared by ALL THREE has in-group degree 3 →
        // `WEIGHT_LUT_SCALED[3] = 405_645 < SCALED_THRESHOLD (524_288)` — below
        // threshold ALONE. So each pair gets its OWN additional distinct rare
        // corroborator (degree exactly 2 — only that pair references it) so EVERY
        // pairwise edge independently clears F2, while "acme" (shared by all three,
        // the genuine transitivity signal) is ALSO present on every pair. This keeps
        // the test's intent — verify clique-transitivity fuses a genuine 3-clique —
        // without weakening any assertion (Rule 8: the corroboration signal must be
        // strengthened to legitimately clear the ratified F2 arithmetic, not the
        // assertion loosened).
        insert_entity(&graph, gid, "John Smith").await; // 'J' 0x4A → lowest id (root)
        insert_entity(&graph, gid, "john  smith").await; // double space variant
        insert_entity(&graph, gid, "john   smith").await; // triple space variant
        insert_entity(&graph, gid, "acme").await;
        insert_entity(&graph, gid, "pair_ab_rare").await;
        insert_entity(&graph, gid, "pair_ac_rare").await;
        insert_entity(&graph, gid, "pair_bc_rare").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        let e3 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        anchor(&graph, gid, e3, "john   smith").await;
        // All three share neighbour "acme" (the transitivity signal, degree 3 alone).
        fact_rel(&graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john   smith", "works_at", "acme").await;
        // Per-pair distinct rare corroborator (degree exactly 2 each) so every
        // pairwise edge clears SCALED_THRESHOLD independently of the group-wide hub.
        fact_rel(&graph, gid, "John Smith", "knows", "pair_ab_rare").await;
        fact_rel(&graph, gid, "john  smith", "knows", "pair_ab_rare").await;
        fact_rel(&graph, gid, "John Smith", "knows", "pair_ac_rare").await;
        fact_rel(&graph, gid, "john   smith", "knows", "pair_ac_rare").await;
        fact_rel(&graph, gid, "john  smith", "knows", "pair_bc_rare").await;
        fact_rel(&graph, gid, "john   smith", "knows", "pair_bc_rare").await;

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 2,
            "transitive: 3 same-label corroborated entities fuse to ONE root (2 losers)"
        );
        let ids = entities_in_group(&graph, gid).await;
        // Keeper = lowest raw id. "John Smith" ('J'=0x4A) < "john  smith"/"john   smith"
        // ('j'=0x6A) → "John Smith" is the surviving root.
        assert!(ids.contains(&"John Smith".to_string()), "root survives");
        assert!(!ids.contains(&"john  smith".to_string()), "loser 1 fused");
        assert!(!ids.contains(&"john   smith".to_string()), "loser 2 fused");
        assert!(ids.contains(&"acme".to_string()), "neighbour untouched");
    }

    // ── ADR-067 F1 fix (V5) — bridge-homonym does NOT fuse via bridge ────────────
    // Cause-fix, not test-loosen: the old test PINNED a bug (connected-component
    // transitive closure fusing A,C via bridge B). Resolving it is the ADR-066
    // P3-ENABLEMENT blocker per Rule 8 — the wrong-behaviour assertion is corrected,
    // not relaxed. Renamed per impl-spec §C1 DoD.

    #[tokio::test]
    async fn bridge_homonym_triple_does_not_fuse_via_bridge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // Three raw ids normalizing to "john smith". Structure: A~B share neighbour "x",
        // B~C share neighbour "y", but A and C have DISJOINT neighbours (A:{x}, C:{y},
        // B:{x,y}). So A~C is NOT directly corroborated — only B bridges them. Neither
        // "x" nor "y" is a hub here (in-group degree 2 each — the exact single-rare-
        // neighbour accept case, ADR-067 F2), so both A~B and B~C DO clear the
        // corroboration gate; the safety property under test is CLIQUE-ONLY clustering
        // (F1), not F2.
        insert_entity(&graph, gid, "John Smith").await; // A — lowest id (keeper)
        insert_entity(&graph, gid, "john  smith").await; // B — the bridge
        insert_entity(&graph, gid, "john   smith").await; // C
        insert_entity(&graph, gid, "x").await;
        insert_entity(&graph, gid, "y").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        let e3 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        anchor(&graph, gid, e3, "john   smith").await;
        // A~B corroborated via "x"; B~C corroborated via "y"; A~C DISJOINT.
        fact_rel(&graph, gid, "John Smith", "knows", "x").await; // A → x
        fact_rel(&graph, gid, "john  smith", "knows", "x").await; // B → x  (A~B share x)
        fact_rel(&graph, gid, "john  smith", "knows", "y").await; // B → y
        fact_rel(&graph, gid, "john   smith", "knows", "y").await; // C → y  (B~C share y)

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        // Clique-only + disjoint-cover: eligible cliques are {A,B} and {B,C} (both
        // sharing B). Sort by (size desc [tie: both 2], weight desc [tie: both one
        // SCALED_THRESHOLD edge], min-id asc): "John Smith" (0x4A...) < "john  smith"
        // (0x6A...) → {A,B} sorts first, is accepted (keeper=A), B is claimed. {B,C}
        // overlaps (B claimed) → DEFERRED, not merged. EXACTLY 1 merge (A absorbs B);
        // C survives distinct from A. The bridge NEVER fuses A and C.
        assert_eq!(
            report.count, 1,
            "clique-only: exactly ONE of {{A,B}}/{{B,C}} merges; the bridge never fuses A,C"
        );
        let ids = entities_in_group(&graph, gid).await;
        // EXACT survivor set (V5 — not "one of B/C survives"): A and C BOTH survive,
        // B (the bridge) vanished.
        let survivors: BTreeSet<String> = ids.into_iter().collect();
        let expected: BTreeSet<String> = ["John Smith", "john   smith", "x", "y"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(
            survivors, expected,
            "EXACT survivor set: \"John Smith\" (A, keeper) AND \"john   smith\" (C) both \
             survive; \"john  smith\" (B, the bridge) vanished"
        );
    }

    /// The V1 run-to-fixpoint proof (ADR-067 §C0 / §6): repeatedly run `cross_episode`
    /// over the bridge plant until it merges 0. WITHOUT the C0 `corroboration_inert`
    /// stamp, pass 2 would inherit B's `knows→y` fact onto A (the remap makes the
    /// keeper A own neighbour `y`), making `A~C` newly eligible (both A and C now
    /// reference `y`) → C merges into A on pass 2, and A,C — distinct referents —
    /// co-merge. This test is the concrete regression guard for that convergence bug.
    #[tokio::test]
    async fn bridge_homonym_reaches_fixpoint_without_co_merging_a_c() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await; // A
        insert_entity(&graph, gid, "john  smith").await; // B — the bridge
        insert_entity(&graph, gid, "john   smith").await; // C
        insert_entity(&graph, gid, "x").await;
        insert_entity(&graph, gid, "y").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        let e3 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        anchor(&graph, gid, e3, "john   smith").await;
        fact_rel(&graph, gid, "John Smith", "knows", "x").await;
        fact_rel(&graph, gid, "john  smith", "knows", "x").await;
        fact_rel(&graph, gid, "john  smith", "knows", "y").await;
        fact_rel(&graph, gid, "john   smith", "knows", "y").await;

        // Run REPEATEDLY until the op merges 0 (the fixpoint), bounded to guard
        // against a runaway loop if the convergence proof is ever violated.
        let mut pass_counts: Vec<usize> = Vec::new();
        for pass in 0..5 {
            let report = cross_episode(&graph, gid, false)
                .await
                .unwrap_or_else(|e| panic!("pass {pass}: cross_episode: {e}"));
            pass_counts.push(report.count);
            if report.count == 0 {
                break;
            }
        }

        assert_eq!(
            pass_counts.first().copied(),
            Some(1),
            "pass 1 merges exactly 1 (A absorbs B): {pass_counts:?}"
        );
        assert_eq!(
            pass_counts.get(1).copied(),
            Some(0),
            "FIXPOINT REACHED IN ≤1 MERGE-PASS: pass 2 must merge 0. If this is 1, the \
             C0 corroboration_inert stamp is missing/broken and A inherited B's neighbour \
             `y`, re-opening eligibility against C (the V1 convergence bug): {pass_counts:?}"
        );
        assert_eq!(
            pass_counts.len(),
            2,
            "fixpoint reached in exactly 2 passes (1 merge-pass + 1 zero-confirmation): \
             {pass_counts:?}"
        );

        // A and C are NEVER co-merged: both survive, as DISTINCT entities.
        let ids = entities_in_group(&graph, gid).await;
        let survivors: BTreeSet<String> = ids.into_iter().collect();
        assert!(
            survivors.contains("John Smith"),
            "A must survive at the fixpoint"
        );
        assert!(
            survivors.contains("john   smith"),
            "C must survive at the fixpoint — A,C must NEVER co-merge across any number \
             of repeated invocations"
        );
        assert!(
            !survivors.contains("john  smith"),
            "B (the bridge) must have been merged away in pass 1"
        );
    }

    // ── Idempotency: second run merges 0 ─────────────────────────────────────────

    #[tokio::test]
    async fn second_run_merges_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        insert_entity(&graph, gid, "John Smith").await;
        insert_entity(&graph, gid, "john  smith").await;
        insert_entity(&graph, gid, "acme").await;
        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        fact_rel(&graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "acme").await;

        let first = cross_episode(&graph, gid, false).await.expect("first");
        assert_eq!(first.count, 1, "first run merges the recurring pair");
        let second = cross_episode(&graph, gid, false).await.expect("second");
        assert_eq!(second.count, 0, "second run merges 0 (loser gone)");
    }

    // ── Namespace isolation: never merge across group_id ─────────────────────────

    #[tokio::test]
    async fn cross_episode_is_namespace_isolated() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Same label + structure but in DIFFERENT namespaces → never merge across.
        // Each namespace holds only ONE john-smith slug, so no in-namespace pair exists.
        // Neighbour names differ per namespace — `insert_entity_with_group` refuses
        // the SAME name-slug across namespaces (ADR-029b cross-namespace collision
        // guard), which is orthogonal to the property under test here.
        insert_entity(&graph, "gA", "John Smith").await;
        insert_entity(&graph, "gB", "john  smith").await;
        insert_entity(&graph, "gA", "acme a").await;
        insert_entity(&graph, "gB", "acme b").await;
        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, "gA", e1, "John Smith").await;
        anchor(&graph, "gB", e2, "john  smith").await;
        fact_rel(&graph, "gA", "John Smith", "works_at", "acme a").await;
        fact_rel(&graph, "gB", "john  smith", "works_at", "acme b").await;

        let report = cross_episode(&graph, "gA", false).await.expect("cross_episode gA");
        assert_eq!(
            report.count, 0,
            "gA sweep sees only one john-smith → no pair"
        );
        // Both survive their own namespaces.
        assert!(entities_in_group(&graph, "gA")
            .await
            .contains(&"John Smith".to_string()));
        assert!(entities_in_group(&graph, "gB")
            .await
            .contains(&"john  smith".to_string()));
    }

    // ── Empty / single-entity group: 0 merges ────────────────────────────────────

    #[tokio::test]
    async fn empty_group_merges_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let report = cross_episode(&graph, "g_empty", false)
            .await
            .expect("cross_episode");
        assert_eq!(report.count, 0);
        assert!(report.warnings.is_empty());
    }

    // ── ADR-067 F1: label group > MAX_LABEL_GROUP is SKIPPED (safety over recall) ──

    #[tokio::test]
    async fn oversized_label_group_skipped() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // MAX_LABEL_GROUP + 1 = 33 same-normalized-label entities, CHAINED into one
        // connected component: entity[i] and entity[i+1] share a DISTINCT rare
        // neighbour n_i (in-group degree exactly 2, well under HUB_DEGREE_CAP, so each
        // adjacent pair individually clears the F2 gate) — a path graph, not a full
        // clique, but connectivity alone is what the MAX_LABEL_GROUP check gates on
        // (impl-spec §C1 step 3 fires on connected-COMPONENT size, before clique
        // enumeration). Component size 33 > MAX_LABEL_GROUP=32 → the WHOLE component
        // is skipped: 0 merges + a warning.
        let n = MAX_LABEL_GROUP + 1;
        let mut ids: Vec<String> = Vec::new();
        for i in 0..n {
            // Trailing-space variants all normalize to "john smith".
            let id = format!("John Smith{}", " ".repeat(i));
            insert_entity(&graph, gid, &id).await;
            ids.push(id);
        }

        // Chain: entity[i] --knows--> n_i <--knows-- entity[i+1] (n_i degree = 2).
        for i in 0..(n - 1) {
            let neighbour = format!("chain_nb_{i}");
            insert_entity(&graph, gid, &neighbour).await;
            let ep_a = new_episode(&graph).await;
            let ep_b = new_episode(&graph).await;
            anchor(&graph, gid, ep_a, &ids[i]).await;
            anchor(&graph, gid, ep_b, &ids[i + 1]).await;
            fact_rel(&graph, gid, &ids[i], "knows", &neighbour).await;
            fact_rel(&graph, gid, &ids[i + 1], "knows", &neighbour).await;
        }

        let report = cross_episode(&graph, gid, false).await.expect("cross_episode");
        assert_eq!(
            report.count, 0,
            "a same-label component of {n} entities (> MAX_LABEL_GROUP={MAX_LABEL_GROUP}) \
             must be SKIPPED entirely, not merged"
        );
        assert_eq!(
            report.warnings.len(),
            1,
            "exactly one skip-warning must be emitted for the oversized group"
        );
        assert!(
            report.warnings[0].contains("exceeds MAX_LABEL_GROUP"),
            "warning must name the MAX_LABEL_GROUP cap: {:?}",
            report.warnings[0]
        );
        // ALL entities survive — precision 100% via skip, recall 0% for this group.
        let survivors = entities_in_group(&graph, gid).await;
        for id in &ids {
            assert!(
                survivors.contains(id),
                "entity {id:?} must survive the skipped oversized group"
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PROPERTY / INVARIANT TIER (P-INV1..P-INV6) — randomized-input safety proof.
    //
    // cross_episode DESTRUCTIVELY merges USER entities, so its safety invariants must
    // hold over RANDOMIZED graphs, not just hand-picked cases. Each of N iterations
    // builds an in-memory graph across TWO namespaces ("gA" swept, "gB" untouched),
    // plants random entities (some pairs normalize identically, some don't), random
    // episode anchors, and random facts (shared-neighbour vs disjoint), snapshots
    // BEFORE, runs `cross_episode` on "gA", snapshots AFTER, and asserts:
    //
    //   P-INV1 (homonym safety, THE critical one): never merge two entities with NO
    //          shared neighbour AND no identical (predicate,object) assertion.
    //   P-INV2 (episode-span): never merge two entities confined to ONE single episode.
    //   P-INV3 (namespace isolation): never merge across group_id (gB untouched).
    //   P-INV4 (delegate-correctness): after a merge, the loser is GONE and its facts /
    //          episodic edges are remapped to the keeper (no dangling loser refs).
    //   P-INV5 (idempotent): a second run on the same graph merges 0.
    //   P-INV6 (count): report.count == number of entities that disappeared.
    //
    // PRNG: hand-rolled seeded SplitMix64 (mirrors `supersession.rs`) — the op is
    // async, seed = fixed base ^ iteration index (deterministic + reproducible), and
    // the exact `seed` is printed on any failure so the case reproduces from one line.
    // ══════════════════════════════════════════════════════════════════════════

    /// Deterministic SplitMix64 PRNG (Vigna, public-domain reference). Mirrors the
    /// `supersession.rs` property harness — seedable, reproducible, no external dep.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
        fn in_range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.below(hi - lo + 1)
        }
        fn chance(&mut self, num: u64, den: u64) -> bool {
            self.below(den) < num
        }
    }

    /// A planted entity in the random graph: its raw id, namespace, and the base name
    /// it was derived from (its normalized identity within the namespace).
    #[derive(Debug, Clone)]
    struct PlantedEntity {
        raw_id: String,
        group: String,
        normalized: String,
        episodes: BTreeSet<i64>,
        /// The distinct set of neighbour ids this entity references via facts.
        neighbours: BTreeSet<String>,
        /// The `(predicate, object)` assertion keys this entity makes.
        assertions: BTreeSet<String>,
    }

    /// Snapshot of an `entities` row for AFTER-state assertions.
    async fn snapshot_entity_ids(graph: &TemporalGraph, group: &str) -> BTreeSet<String> {
        let mut rows = graph
            .conn
            .query(
                "SELECT id FROM entities WHERE group_id = ?1 ORDER BY id",
                libsql::params![group],
            )
            .await
            .expect("snapshot entities");
        let mut out = BTreeSet::new();
        while let Some(row) = rows.next().await.expect("row") {
            out.insert(row.get::<String>(0).expect("id"));
        }
        out
    }

    /// Count facts still referencing `entity` as subject OR object in `group`.
    async fn fact_refs_to(graph: &TemporalGraph, group: &str, entity: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM facts \
                 WHERE group_id = ?1 AND (subject_id = ?2 OR object_id = ?2)",
                libsql::params![group, entity],
            )
            .await
            .expect("fact refs");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("count")
    }

    /// Count episodic edges still referencing `entity` in `group`.
    async fn edge_refs_to(graph: &TemporalGraph, group: &str, entity: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM episodic_edges \
                 WHERE entity_id = ?1 AND (entity_group_id = ?2 OR entity_group_id IS NULL)",
                libsql::params![entity, group],
            )
            .await
            .expect("edge refs");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("count")
    }

    /// In-group degree of a shared neighbour `n` among `population` (ADR-067 F2), computed
    /// DIRECTLY from plant records — the count of DISTINCT planted entities in
    /// `population` whose `neighbours` set contains `n`. This mirrors the op's
    /// `neighbour_degree` but reasons on the IMMUTABLE plant data, so it NEVER counts a
    /// corroborator that would have been inherited via a prior merge (the oracle's model
    /// of C0's `corroboration_inert = 0` filter — impl-spec §4 step 1).
    fn oracle_neighbour_degree(population: &[&PlantedEntity], n: &str) -> u32 {
        population
            .iter()
            .filter(|p| p.neighbours.contains(n))
            .count() as u32
    }

    /// In-group frequency of an identical assertion key `k` among `population`, computed
    /// directly from plant records (mirrors [`oracle_neighbour_degree`] for assertions).
    fn oracle_assertion_frequency(population: &[&PlantedEntity], k: &str) -> u32 {
        population
            .iter()
            .filter(|p| p.assertions.contains(k))
            .count() as u32
    }

    /// Independent rarity-weighted corroboration decision (ADR-067 F2), recomputed from
    /// plant records over DIRECTLY-asserted structure only — the oracle's model of C0.
    /// Uses the SAME `WEIGHT_LUT_SCALED`/`SCALED_THRESHOLD` integer decision as the op
    /// (impl-spec §3), so op and oracle agree bit-for-bit on eligibility.
    fn oracle_shares_structure(
        population: &[&PlantedEntity],
        a: &PlantedEntity,
        b: &PlantedEntity,
    ) -> bool {
        let shared_neighbours: BTreeSet<&String> = a
            .neighbours
            .intersection(&b.neighbours)
            .filter(|n| n.as_str() != a.raw_id && n.as_str() != b.raw_id)
            .collect();
        // Same dedup as the op's `shares_structure` (impl-spec §C2 step 4 + the
        // corroborator-identity fix): a relational assertion key's object IS the
        // shared neighbour that same fact contributes — skip it here to avoid
        // double-counting one underlying fact as two corroborators. Literal
        // assertion keys (`ov:` prefix) never collide with an entity id.
        let shared_assertions: BTreeSet<&String> = a
            .assertions
            .intersection(&b.assertions)
            .filter(|key| {
                let object_component = key.rsplit('\u{1F}').next().unwrap_or(key.as_str());
                object_component.starts_with("ov:") || !shared_neighbours.contains(&object_component.to_string())
            })
            .collect();
        if shared_neighbours.is_empty() && shared_assertions.is_empty() {
            return false;
        }
        let mut acc: u64 = 0;
        for n in shared_neighbours {
            acc += scaled_weight(oracle_neighbour_degree(population, n));
        }
        for k in shared_assertions {
            acc += scaled_weight(oracle_assertion_frequency(population, k));
        }
        acc >= SCALED_THRESHOLD
    }

    /// Is `(a, b)` oracle-eligible: label-match (exact-normalized OR fuzzy Jaccard ≥
    /// threshold) AND spans ≥2 distinct episodes AND clears the rarity-weighted
    /// corroboration decision (over DIRECTLY-asserted structure in `population`)?
    fn is_oracle_eligible(population: &[&PlantedEntity], a: &PlantedEntity, b: &PlantedEntity) -> bool {
        let label_match = a.normalized == b.normalized || {
            let sa = label_shingles(&a.normalized);
            let sb = label_shingles(&b.normalized);
            a.normalized != b.normalized && jaccard(&sa, &sb) >= FUZZY_JACCARD_THRESHOLD
        };
        if !label_match {
            return false;
        }
        let mut eps = a.episodes.clone();
        eps.extend(b.episodes.iter().copied());
        if eps.len() < 2 {
            return false;
        }
        oracle_shares_structure(population, a, b)
    }

    /// **Independent clique cover (ADR-067 §4, C3)** — genuinely independent of the
    /// op's Bron–Kerbosch: brute-force enumeration over subsets (bounded population,
    /// `2^N` scan keeping maximal fully-connected subsets), NOT the op's BK algorithm.
    /// The op and oracle share ONLY the deterministic disjoint-cover sort/accept RULE
    /// (a spec, not an algorithm) — mirrors the op's `(size desc, weight desc,
    /// min-id asc)` sort exactly (impl-spec §4 step 2).
    ///
    /// Returns the FIXPOINT vanished set: for each accepted (disjoint) clique of size
    /// ≥ 2, every non-keeper member — because the oracle reasons on directly-asserted
    /// structure only, this cover already IS the fixpoint (no inheritance can add an
    /// edge a second oracle pass would find).
    fn oracle_fixpoint_vanished(population: &[&PlantedEntity]) -> BTreeSet<String> {
        let n = population.len();
        let ids: Vec<&String> = population.iter().map(|p| &p.raw_id).collect();

        // Brute-force adjacency via oracle eligibility (independent of the op's BK).
        let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if is_oracle_eligible(population, population[i], population[j]) {
                    adjacency
                        .entry(ids[i].clone())
                        .or_default()
                        .insert(ids[j].clone());
                    adjacency
                        .entry(ids[j].clone())
                        .or_default()
                        .insert(ids[i].clone());
                }
            }
        }

        // Brute-force MAXIMAL clique enumeration over subsets (2^N scan). Population in
        // this property harness is small (≤8 entities), well within brute-force bounds
        // (impl-spec §4 step 2: "brute force over subsets for small N (N ≤ ~12)").
        let all_ids: Vec<String> = adjacency.keys().cloned().collect();
        let m = all_ids.len();
        let mut cliques: Vec<BTreeSet<String>> = Vec::new();
        if m > 0 && m <= 20 {
            for mask in 1u32..(1u32 << m) {
                let subset: Vec<&String> = (0..m)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| &all_ids[i])
                    .collect();
                if subset.len() < 2 {
                    continue;
                }
                // Is `subset` fully connected (a clique)?
                let is_clique = subset.iter().enumerate().all(|(i, a)| {
                    subset
                        .iter()
                        .skip(i + 1)
                        .all(|b| adjacency.get(*a).is_some_and(|adj| adj.contains(*b)))
                });
                if !is_clique {
                    continue;
                }
                // Is `subset` MAXIMAL (no node outside it is adjacent to ALL members)?
                let subset_set: BTreeSet<String> = subset.iter().map(|s| s.to_string()).collect();
                let is_maximal = !all_ids.iter().any(|candidate| {
                    !subset_set.contains(candidate)
                        && subset_set.iter().all(|m| {
                            adjacency
                                .get(candidate)
                                .is_some_and(|adj| adj.contains(m))
                        })
                });
                if is_maximal {
                    cliques.push(subset_set);
                }
            }
        }

        // SAME disjoint-cover sort rule as the op: (size desc, weight desc, min-id asc).
        cliques.sort_by(|a, b| {
            let size_cmp = b.len().cmp(&a.len());
            if size_cmp != std::cmp::Ordering::Equal {
                return size_cmp;
            }
            let weight = |c: &BTreeSet<String>| -> u64 {
                let members: Vec<&String> = c.iter().collect();
                let mut total = 0u64;
                for i in 0..members.len() {
                    for j in (i + 1)..members.len() {
                        if adjacency
                            .get(members[i])
                            .is_some_and(|adj| adj.contains(members[j]))
                        {
                            total += SCALED_THRESHOLD;
                        }
                    }
                }
                total
            };
            let weight_cmp = weight(b).cmp(&weight(a));
            if weight_cmp != std::cmp::Ordering::Equal {
                return weight_cmp;
            }
            let min_a = a.iter().min().cloned().unwrap_or_default();
            let min_b = b.iter().min().cloned().unwrap_or_default();
            min_a.cmp(&min_b)
        });

        let mut claimed: BTreeSet<String> = BTreeSet::new();
        let mut vanished: BTreeSet<String> = BTreeSet::new();
        for clique in &cliques {
            if clique.iter().any(|m| claimed.contains(m)) {
                continue;
            }
            let keeper = clique.iter().min().cloned().unwrap_or_default();
            for member in clique {
                if member != &keeper {
                    vanished.insert(member.clone());
                }
            }
            for m in clique {
                claimed.insert(m.clone());
            }
        }
        vanished
    }

    #[tokio::test]
    async fn property_cross_episode_invariants_over_random_inputs() {
        const BASE_SEED: u64 = 0x584D_5247_5F58_4550; // arbitrary fixed base.
        const ITERATIONS: u64 = 300; // ≥ 200 required.

        // Base names → normalized identity. Variant generators derive DISTINCT raw ids
        // that normalize to the SAME base (case + whitespace) so same-name pairs arise.
        const BASE_NAMES: [&str; 3] = ["john smith", "acme corp", "mary jones"];
        const PREDICATES: [&str; 3] = ["works_at", "member_of", "located_in"];
        const NEIGHBOURS: [&str; 3] = ["orgx", "orgy", "orgz"];
        const GROUPS: [&str; 2] = ["gA", "gB"];

        for iter in 0..ITERATIONS {
            let seed = BASE_SEED ^ iter;
            let mut rng = SplitMix64::new(seed);

            let graph = TemporalGraph::open_in_memory()
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: open graph: {e}"));

            // Track planted entities per (group, raw_id) so the oracle can recompute
            // eligibility independently of the op.
            let mut planted: Vec<PlantedEntity> = Vec::new();
            // Distinct-variant counter per (group, base) so raw ids never collide as a
            // PK within a namespace (two variants of the same base must differ).
            let mut variant_seq: BTreeMap<(String, String), u64> = BTreeMap::new();

            // Pre-create a shared episode pool (ids reused across entities so that
            // same-episode and distinct-episode collisions both occur).
            let mut episodes: Vec<i64> = Vec::new();
            for _ in 0..3 {
                episodes.push(new_episode(&graph).await);
            }

            // ── NEW (V4/C3): plant a deterministic BRIDGE TRIPLE seed every iteration
            // (impl-spec §C3 DoD / §4 step 4) — A~B, B~C, A≁C, both edges via a RARE
            // (deg-2) neighbour so they clear F2. If the op ever regresses to CC
            // bridging OR to inheritance-driven pass-2 co-merge, the fixpoint
            // assertions below FAIL. This seed is included ALONGSIDE the random plant
            // in the SAME "gA" namespace every iteration.
            // All three normalize to the SAME base (`normalize_label` only lower-cases
            // + collapses whitespace — it does NOT ignore letters), so they differ ONLY
            // by case/whitespace, never by letter content, to stay same-label-eligible.
            let bridge_a = format!("bseed{iter}_A");
            let bridge_b = format!("bseed{iter}_a ");
            let bridge_base = format!("bseed{iter}_a"); // == normalize_label(bridge_a/_b/_c).
            let bridge_c = format!("bseed{iter}_A  ");
            let bx = format!("bx{iter}");
            let by = format!("by{iter}");
            for (id, nb) in [(&bridge_a, &bx), (&bridge_b, &bx)] {
                insert_entity(&graph, "gA", id).await;
                let _ = graph
                    .insert_entity_with_group(InsertEntityWithGroupParams {
                        id: nb,
                        entity_type_id: 0,
                        properties: serde_json::json!({}),
                        group_id: Some("gA"),
                    })
                    .await;
            }
            insert_entity(&graph, "gA", &bridge_c).await;
            let _ = graph
                .insert_entity_with_group(InsertEntityWithGroupParams {
                    id: &by,
                    entity_type_id: 0,
                    properties: serde_json::json!({}),
                    group_id: Some("gA"),
                })
                .await;
            let bridge_e1 = new_episode(&graph).await;
            let bridge_e2 = new_episode(&graph).await;
            let bridge_e3 = new_episode(&graph).await;
            anchor(&graph, "gA", bridge_e1, &bridge_a).await;
            anchor(&graph, "gA", bridge_e2, &bridge_b).await;
            anchor(&graph, "gA", bridge_e3, &bridge_c).await;
            fact_rel(&graph, "gA", &bridge_a, "knows", &bx).await;
            fact_rel(&graph, "gA", &bridge_b, "knows", &bx).await;
            fact_rel(&graph, "gA", &bridge_b, "knows", &by).await;
            fact_rel(&graph, "gA", &bridge_c, "knows", &by).await;
            let bridge_population: Vec<PlantedEntity> = [
                (&bridge_a, BTreeSet::from([bridge_e1])),
                (&bridge_b, BTreeSet::from([bridge_e2])),
                (&bridge_c, BTreeSet::from([bridge_e3])),
            ]
            .into_iter()
            .map(|(raw, eps)| {
                let neighbours: BTreeSet<String> = if raw == &bridge_a {
                    BTreeSet::from([bx.clone()])
                } else if raw == &bridge_b {
                    BTreeSet::from([bx.clone(), by.clone()])
                } else {
                    BTreeSet::from([by.clone()])
                };
                PlantedEntity {
                    raw_id: raw.clone(),
                    group: "gA".to_string(),
                    normalized: bridge_base.clone(),
                    episodes: eps,
                    neighbours,
                    assertions: BTreeSet::new(),
                }
            })
            .collect();
            planted.extend(bridge_population);

            let n_entities = rng.in_range(2, 8);
            for _ in 0..n_entities {
                let group = GROUPS[rng.below(GROUPS.len() as u64) as usize].to_string();
                let base = BASE_NAMES[rng.below(BASE_NAMES.len() as u64) as usize].to_string();

                // Derive a DISTINCT raw id that normalizes to `base`. Append a growing
                // run of trailing spaces (normalized away) + random case on the first
                // char, keyed by a per-(group,base) sequence so raw ids never repeat.
                let seq = variant_seq
                    .entry((group.clone(), base.clone()))
                    .or_insert(0);
                let this_seq = *seq;
                *seq += 1;
                // Trailing spaces collapse away under normalize; leading case flip too.
                let mut raw = base.clone();
                for _ in 0..this_seq {
                    raw.push(' ');
                }
                if rng.chance(1, 2) {
                    // Uppercase the first char (case-folds back to `base`).
                    let mut chars: Vec<char> = raw.chars().collect();
                    if let Some(c) = chars.first_mut() {
                        *c = c.to_ascii_uppercase();
                    }
                    raw = chars.into_iter().collect();
                }
                debug_assert_eq!(
                    normalize_label(&raw),
                    base,
                    "variant must normalize to base"
                );

                // Insert entity (skip on the cross-namespace collision guard — a base
                // may already exist in the OTHER namespace; that's fine, we just skip).
                let inserted = graph
                    .insert_entity_with_group(InsertEntityWithGroupParams {
                        id: &raw,
                        entity_type_id: 0,
                        properties: serde_json::json!({ "name": raw }),
                        group_id: Some(&group),
                    })
                    .await
                    .is_ok();
                if !inserted {
                    continue;
                }

                // Anchor to a random subset of episodes (1..=3 distinct).
                let n_eps = rng.in_range(1, episodes.len() as u64);
                let mut ep_set: BTreeSet<i64> = BTreeSet::new();
                for _ in 0..n_eps {
                    let ep = episodes[rng.below(episodes.len() as u64) as usize];
                    ep_set.insert(ep);
                    anchor(&graph, &group, ep, &raw).await;
                }

                // Facts: with ~1/2 chance give a relational fact to a random neighbour
                // (creates shared-neighbour structure), and with ~1/2 a literal fact
                // (creates identical-assertion structure). The neighbour entity is
                // planted on demand.
                let mut neighbours: BTreeSet<String> = BTreeSet::new();
                let mut assertions: BTreeSet<String> = BTreeSet::new();
                if rng.chance(1, 2) {
                    let nb = NEIGHBOURS[rng.below(NEIGHBOURS.len() as u64) as usize];
                    let nb_id = format!("{group}_{nb}"); // namespace-unique neighbour.
                    let _ = graph
                        .insert_entity_with_group(InsertEntityWithGroupParams {
                            id: &nb_id,
                            entity_type_id: 0,
                            properties: serde_json::json!({}),
                            group_id: Some(&group),
                        })
                        .await;
                    let pred = PREDICATES[rng.below(PREDICATES.len() as u64) as usize];
                    fact_rel(&graph, &group, &raw, pred, &nb_id).await;
                    neighbours.insert(nb_id.clone());
                    assertions.insert(format!("{pred}\u{1F}{nb_id}"));
                }
                if rng.chance(1, 2) {
                    let pred = PREDICATES[rng.below(PREDICATES.len() as u64) as usize];
                    let value = format!("v{}", rng.below(3));
                    fact_lit(&graph, &group, &raw, pred, &value).await;
                    assertions.insert(format!("{pred}\u{1F}ov:{value}"));
                }

                planted.push(PlantedEntity {
                    raw_id: raw,
                    group,
                    normalized: base,
                    episodes: ep_set,
                    neighbours,
                    assertions,
                });
            }

            // ── Snapshot BEFORE ──────────────────────────────────────────────────
            let before_ids_ga = snapshot_entity_ids(&graph, "gA").await;
            let before_ids_gb = snapshot_entity_ids(&graph, "gB").await;

            // ── Run the op REPEATEDLY to a FIXPOINT (V4 — upgraded from a single
            // second-run check): repeat until it merges 0, recording each pass's
            // count. The op's CUMULATIVE vanished set must equal the oracle's
            // FIXPOINT vanished set, AND the fixpoint must be reached in ≤1
            // merge-pass (pass 1 merges; pass 2 is the 0-merge confirmation). ────
            let mut pass_counts: Vec<usize> = Vec::new();
            const MAX_PASSES: usize = 6; // bounded guard against a runaway loop.
            for pass in 0..MAX_PASSES {
                let report = cross_episode(&graph, "gA", false)
                    .await
                    .unwrap_or_else(|e| panic!("seed={seed:#x} pass={pass}: cross_episode: {e}"));
                pass_counts.push(report.count);
                if report.count == 0 {
                    break;
                }
            }

            // ── Snapshot AFTER (at the fixpoint) ─────────────────────────────────
            let after_ids_ga = snapshot_entity_ids(&graph, "gA").await;
            let after_ids_gb = snapshot_entity_ids(&graph, "gB").await;

            // P-INV3 (namespace isolation): gB entity set UNCHANGED.
            assert_eq!(
                before_ids_gb, after_ids_gb,
                "seed={seed:#x} P-INV3 violated: gB (unswept namespace) entities changed"
            );

            // Compute the set of gA entities that DISAPPEARED (were merged away),
            // CUMULATIVE across all passes to the fixpoint.
            let vanished: BTreeSet<String> =
                before_ids_ga.difference(&after_ids_ga).cloned().collect();

            // P-INV6 (count): cumulative merge count == number of gA entities vanished.
            let cumulative_count: usize = pass_counts.iter().sum();
            assert_eq!(
                cumulative_count,
                vanished.len(),
                "seed={seed:#x} P-INV6 violated: cumulative report.count ({cumulative_count}) \
                 != vanished entities ({}); pass_counts={pass_counts:?}",
                vanished.len()
            );

            // P-INV5 (fixpoint bound, V4 upgrade): reached in ≤1 merge-pass — i.e. at
            // most the FIRST pass merges anything; every subsequent pass merges 0.
            assert!(
                pass_counts.len() <= 1 || pass_counts[1..].iter().all(|&c| c == 0),
                "seed={seed:#x} P-INV5 violated: fixpoint NOT reached in ≤1 merge-pass \
                 (pass 2+ merged something): pass_counts={pass_counts:?}"
            );

            // ── Independent oracle (C3): brute-force clique cover (NOT the op's BK),
            // computing the FIXPOINT vanished set from DIRECTLY-asserted structure
            // only (never counting an inherited/merge-created edge — the oracle's
            // model of C0). ──────────────────────────────────────────────────────
            let ga_planted: Vec<&PlantedEntity> =
                planted.iter().filter(|p| p.group == "gA").collect();
            let oracle_vanished = oracle_fixpoint_vanished(&ga_planted);

            // P-INV1 (homonym safety, THE critical one — now FIXPOINT-strength): the
            // op's cumulative vanished set must equal the oracle's independent
            // fixpoint vanished set EXACTLY. A false-merge (op vanishes something the
            // oracle's directly-asserted clique cover does not) OR an inheritance
            // co-merge (op vanishes MORE across passes than the oracle's fixpoint)
            // both fail here.
            assert_eq!(
                vanished, oracle_vanished,
                "seed={seed:#x} P-INV1 violated: op's cumulative vanished set != oracle's \
                 independent fixpoint vanished set (op={vanished:?} oracle={oracle_vanished:?}); \
                 pass_counts={pass_counts:?}"
            );

            for loser_id in &vanished {
                // P-INV4 (delegate-correctness): loser GONE + no dangling refs.
                assert!(
                    !after_ids_ga.contains(loser_id),
                    "seed={seed:#x} P-INV4: loser {loser_id} must be deleted"
                );
                assert_eq!(
                    fact_refs_to(&graph, "gA", loser_id).await,
                    0,
                    "seed={seed:#x} P-INV4: no fact may still reference loser {loser_id}"
                );
                assert_eq!(
                    edge_refs_to(&graph, "gA", loser_id).await,
                    0,
                    "seed={seed:#x} P-INV4: no episodic edge may still reference loser {loser_id}"
                );
            }

            // Bridge-seed specific assertion (impl-spec §C3 DoD): the planted bridge's
            // A and C must NEVER co-merge — both survive at the fixpoint, distinct.
            let (bridge_keeper, bridge_other) = if bridge_a <= bridge_c {
                (&bridge_a, &bridge_c)
            } else {
                (&bridge_c, &bridge_a)
            };
            assert!(
                after_ids_ga.contains(bridge_keeper) && after_ids_ga.contains(bridge_other),
                "seed={seed:#x} bridge-seed violated: A ({bridge_a:?}) and C ({bridge_c:?}) \
                 must NEVER co-merge; after_ids_ga={after_ids_ga:?}"
            );

            // P-INV5 (idempotent): one MORE run past the fixpoint still merges 0.
            let confirm = cross_episode(&graph, "gA", false)
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: confirm cross_episode: {e}"));
            let after_confirm_ga = snapshot_entity_ids(&graph, "gA").await;
            assert_eq!(
                confirm.count, 0,
                "seed={seed:#x} P-INV5 violated: post-fixpoint run merged {} (must be 0)",
                confirm.count
            );
            assert_eq!(
                after_ids_ga, after_confirm_ga,
                "seed={seed:#x} P-INV5 violated: post-fixpoint run mutated the gA entity set"
            );
        }
    }
}

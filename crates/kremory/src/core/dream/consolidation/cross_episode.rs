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
//!    "John Smith" the lawyer vs the athlete) → **DO NOT MERGE**. Deterministic,
//!    zero-LLM.
//!
//! When all three hold: **keeper = the lowest entity id** (deterministic tie-break);
//! delegate the structural merge to the shared
//! [`crate::core::canonicalization::apply_entity_merge`] executor (spec DoD-P0.3) — NO
//! second merge code path (R-02).
//!
//! **Non-overlap with canonicalize (DoD-P3.3):** this op fires ONLY on exact/fuzzy
//! LABEL matches, NEVER on embedding cosine — a cosine-near-dup pair that is
//! lexically distinct is canonicalize's job, and cross_episode leaves it (spec §6
//! `cosine_near_dup_lexically_distinct`). No double-count.
//!
//! `cross_episode_merges` counts merges THIS pass applied, split
//! `{path=exact|fuzzy}` (DoD-P3.4).
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P3.1–P3.4, incl. P3.1b) + §6 (cross_episode corpus) + ADR-066 §2.2 (REVISED).

use std::collections::{BTreeMap, BTreeSet};

use metrics::counter;

use crate::core::canonicalization::apply_entity_merge;
use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::OpReport;

/// Fuzzy-path admission threshold on the label token-shingle Jaccard (SYNTHESIS #9).
/// A pair below this on the fuzzy path is NOT admitted; the exact path (Jaccard == 1.0
/// on normalized labels) is the primary route.
const FUZZY_JACCARD_THRESHOLD: f64 = 0.9;

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
/// Args count = 2 — plain args, no params struct needed (TD-042 threshold 3).
#[doc(hidden)]
pub async fn cross_episode(graph: &TemporalGraph, group_id: &str) -> Result<OpReport> {
    let mut report = OpReport::default();

    // ── Load entity slots (id + normalized label + distinct episode set) ──────────
    let slots = load_entity_slots(graph, group_id).await?;
    if slots.len() < 2 {
        emit_counters(0, 0);
        return Ok(report);
    }

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
        if !shares_structure(graph, group_id, (&cand.keeper, &cand.loser)).await? {
            counter!(
                "kremory.dream.consolidation.cross_episode_homonym_skip_total",
                "path" => cand.path.as_str(),
            )
            .increment(1);
            continue;
        }
        eligible.push(cand.clone());
    }

    // ── Phase 3: union-find cluster the eligible edges, then merge each cluster into
    // its root (lowest id, deterministic — DoD-P3.1b). Transitive: A~C + B~C collapses
    // {A,B,C} to one keeper even if A~B was not DIRECTLY corroborated (identity is
    // transitive). Mirrors canonicalize's transitive-chain resolution
    // (`canonicalization.rs:205-247`) but with the corroboration gate upstream.
    let mut uf = UnionFind::new();
    // Attribute each member's merge to a path: exact wins over fuzzy when a member has
    // any exact-eligible edge (an exact recurrence is the stronger signal).
    let mut member_path: BTreeMap<String, MergePath> = BTreeMap::new();
    for e in &eligible {
        uf.union(&e.keeper, &e.loser);
        for id in [&e.keeper, &e.loser] {
            let entry = member_path.entry(id.clone()).or_insert(e.path);
            if e.path == MergePath::Exact {
                *entry = MergePath::Exact;
            }
        }
    }

    let mut exact_merges = 0usize;
    let mut fuzzy_merges = 0usize;
    // For each entity that is NOT its cluster root, merge it into the root. Iterate in
    // sorted id order for deterministic application (BTreeMap over slot ids).
    let sorted_ids: Vec<String> = slots.iter().map(|s| s.id.clone()).collect();
    for id in &sorted_ids {
        let root = uf.find(id);
        if &root == id {
            continue; // this id IS the cluster root (or a singleton) — nothing to merge.
        }
        // `id` is a non-root member → merge it (loser) into `root` (keeper).
        apply_entity_merge(graph, id, &root).await?;
        match member_path.get(id).copied().unwrap_or(MergePath::Fuzzy) {
            MergePath::Exact => exact_merges += 1,
            MergePath::Fuzzy => fuzzy_merges += 1,
        }
        tracing::info!(
            target: "kremory.dream.consolidation.cross_episode",
            group_id,
            keeper = %root,
            loser = %id,
            "cross-episode merge applied"
        );
    }

    emit_counters(exact_merges, fuzzy_merges);
    tracing::info!(
        target: "kremory.dream.consolidation.cross_episode",
        group_id,
        exact_merges,
        fuzzy_merges,
        candidates = candidates.len(),
        eligible = eligible.len(),
        "cross-episode sweep complete"
    );

    report.count = exact_merges + fuzzy_merges;
    Ok(report)
}

/// Minimal string-keyed union-find with path-compression + lowest-id root election.
/// The ROOT of a set is always its lexicographically-smallest member (the keeper
/// tie-break, DoD-P3.1b), so `find` returns a deterministic keeper for any cluster.
struct UnionFind {
    parent: BTreeMap<String, String>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: BTreeMap::new(),
        }
    }

    /// Find the representative (lowest-id root) of `x`, inserting `x` as its own root
    /// if unseen. Iterative path-walk (no recursion) with a bounded loop.
    fn find(&mut self, x: &str) -> String {
        // Ensure `x` is present.
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }
        let mut cur = x.to_string();
        // Walk to the root (parent == self). Bounded by the set size.
        let bound = self.parent.len() + 1;
        for _ in 0..bound {
            let p = self
                .parent
                .get(&cur)
                .cloned()
                .unwrap_or_else(|| cur.clone());
            if p == cur {
                break;
            }
            cur = p;
        }
        // Path-compress `x` directly onto the root.
        self.parent.insert(x.to_string(), cur.clone());
        cur
    }

    /// Union the sets containing `a` and `b`; the merged root is the LOWER of the two
    /// roots (lowest-id keeper election).
    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (root, child) = if ra <= rb { (ra, rb) } else { (rb, ra) };
        self.parent.insert(child, root);
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

// ─── Structural-corroboration gate (P3.1b / RISK-001) ────────────────────────────

/// Do `a` and `b` share STRUCTURE (DoD-P3.1b, MANDATORY homonym guard)?
///
/// Returns `true` when EITHER:
///   (i) a common third entity `c` is referenced (as `subject_id` OR `object_id`) by
///       a fact touching `a` AND by a fact touching `b` (`c ∉ {a, b}`), OR
///   (ii) `a` and `b` each assert an IDENTICAL `(predicate, object_id/object_value)`
///        fact.
///
/// No shared structure → probable homonym → the caller drops the pair. Pure structural
/// SQL over `facts`, zero-LLM.
///
/// `pair` = `(a, b)` bundled to keep the arg count at the TD-042 threshold-3.
async fn shares_structure(
    graph: &TemporalGraph,
    group_id: &str,
    pair: (&str, &str),
) -> Result<bool> {
    let (a, b) = pair;
    let neighbours_a = neighbours_of(graph, group_id, a).await?;
    let neighbours_b = neighbours_of(graph, group_id, b).await?;

    // (i) common third entity: intersection of the two neighbour sets, excluding the
    // two candidates themselves (a fact directly linking a↔b is NOT a shared third
    // neighbour — it is a direct edge, which does not corroborate a SHARED referent).
    if neighbours_a
        .intersection(&neighbours_b)
        .any(|n| n != a && n != b)
    {
        return Ok(true);
    }

    // (ii) identical (predicate, object) assertion: both assert the same edge/literal.
    let assertions_a = assertions_of(graph, group_id, a).await?;
    let assertions_b = assertions_of(graph, group_id, b).await?;
    if assertions_a.intersection(&assertions_b).next().is_some() {
        return Ok(true);
    }

    Ok(false)
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
    let mut rows = graph
        .conn
        .query(
            "SELECT object_id FROM facts \
             WHERE group_id = ?1 AND subject_id = ?2 AND object_id IS NOT NULL \
               AND expired_at IS NULL AND invalid_at IS NULL \
             UNION \
             SELECT subject_id FROM facts \
             WHERE group_id = ?1 AND object_id = ?2 \
               AND expired_at IS NULL AND invalid_at IS NULL",
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
    let mut rows = graph
        .conn
        .query(
            "SELECT predicate, object_id, object_value FROM facts \
             WHERE group_id = ?1 AND subject_id = ?2 \
               AND expired_at IS NULL AND invalid_at IS NULL",
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
        assert_eq!(
            report.count, 1,
            "identical literal assertion corroborates merge"
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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
        // pairs (A~B, A~C, B~C). Distinct episodes each. ALL three share the common
        // neighbour "acme" → EVERY pair is directly corroborated (intended transitivity).
        insert_entity(&graph, gid, "John Smith").await; // 'J' 0x4A → lowest id (root)
        insert_entity(&graph, gid, "john  smith").await; // double space variant
        insert_entity(&graph, gid, "john   smith").await; // triple space variant
        insert_entity(&graph, gid, "acme").await;

        let e1 = new_episode(&graph).await;
        let e2 = new_episode(&graph).await;
        let e3 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "John Smith").await;
        anchor(&graph, gid, e2, "john  smith").await;
        anchor(&graph, gid, e3, "john   smith").await;
        // All three share neighbour "acme" → A~B, A~C, B~C ALL directly corroborated.
        fact_rel(&graph, gid, "John Smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john  smith", "works_at", "acme").await;
        fact_rel(&graph, gid, "john   smith", "works_at", "acme").await;

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
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

    // ── KNOWN over-merge surface (Quinn F1) — bridge-homonym, pinned + tracked ────

    #[tokio::test]
    async fn bridge_homonym_triple_fuses_via_bridge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g1";
        // Three raw ids normalizing to "john smith". Structure: A~B share neighbour "x",
        // B~C share neighbour "y", but A and C have DISJOINT neighbours (A:{x}, C:{y},
        // B:{x,y}). So A~C is NOT directly corroborated — only B bridges them.
        insert_entity(&graph, gid, "John Smith").await; // A — lowest id (root)
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

        let report = cross_episode(&graph, gid).await.expect("cross_episode");
        // CURRENT behavior fuses A,C via bridge B — a KNOWN over-merge surface (Quinn
        // F1); pinned here so it's visible + regression-tracked; resolving it (clique vs
        // connected-component) is a P3-ENABLEMENT blocker, see ADR-066.
        assert_eq!(
            report.count, 2,
            "CURRENT (connected-component): bridge B fuses A,C into ONE cluster (2 losers) \
             — KNOWN over-merge, not asserted safe"
        );
        let ids = entities_in_group(&graph, gid).await;
        assert!(ids.contains(&"John Smith".to_string()), "root survives");
        assert!(!ids.contains(&"john  smith".to_string()), "bridge B fused");
        assert!(
            !ids.contains(&"john   smith".to_string()),
            "C fused via bridge B (the over-merge)"
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

        let first = cross_episode(&graph, gid).await.expect("first");
        assert_eq!(first.count, 1, "first run merges the recurring pair");
        let second = cross_episode(&graph, gid).await.expect("second");
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

        let report = cross_episode(&graph, "gA").await.expect("cross_episode gA");
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
        let report = cross_episode(&graph, "g_empty")
            .await
            .expect("cross_episode");
        assert_eq!(report.count, 0);
        assert!(report.warnings.is_empty());
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

    /// Do two planted entities share structure (mirrors the op's
    /// [`shares_structure`] predicate independently, for oracle comparison)?
    fn planted_share_structure(a: &PlantedEntity, b: &PlantedEntity) -> bool {
        // (i) common third neighbour (excluding a, b themselves).
        let common: BTreeSet<&String> = a.neighbours.intersection(&b.neighbours).collect();
        if common.iter().any(|n| **n != a.raw_id && **n != b.raw_id) {
            return true;
        }
        // (ii) identical assertion.
        a.assertions.intersection(&b.assertions).next().is_some()
    }

    /// **Oracle limitation (Quinn F4b, honest test-doc):** the P-INV1 transitive
    /// invariant validates that the op merges exactly the CONNECTED COMPONENTS of the
    /// eligible-edge graph — the oracle is itself a union-find over `is_oracle_eligible`
    /// edges (a CONNECTED-COMPONENT oracle). It therefore CANNOT independently catch a
    /// **bridge-homonym over-merge**: a same-label triple `A~B`, `B~C`, `A≁C` where only
    /// `B` bridges two disjoint referents. The oracle bridges `A`,`C` via `B` IDENTICALLY
    /// to the op, so both agree and the invariant passes — it cannot distinguish
    /// "correct transitivity" from "over-merge through a bridge". That specific behaviour
    /// is pinned instead by the deterministic `bridge_homonym_triple_fuses_via_bridge`
    /// fixture (Quinn F4), NOT by this oracle. Resolving bridge-homonymy (clique vs
    /// connected-component clustering + an oracle independent of the op's own clustering)
    /// is a P3-ENABLEMENT blocker — see ADR-066 §Consequences "P3 enablement blockers".
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

            // ── Run the op on "gA" only ──────────────────────────────────────────
            let report = cross_episode(&graph, "gA")
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: cross_episode: {e}"));

            // ── Snapshot AFTER ───────────────────────────────────────────────────
            let after_ids_ga = snapshot_entity_ids(&graph, "gA").await;
            let after_ids_gb = snapshot_entity_ids(&graph, "gB").await;

            // P-INV3 (namespace isolation): gB entity set UNCHANGED.
            assert_eq!(
                before_ids_gb, after_ids_gb,
                "seed={seed:#x} P-INV3 violated: gB (unswept namespace) entities changed"
            );

            // Compute the set of gA entities that DISAPPEARED (were merged away).
            let vanished: Vec<String> = before_ids_ga.difference(&after_ids_ga).cloned().collect();

            // P-INV6 (count): report.count == number of gA entities that vanished.
            assert_eq!(
                report.count,
                vanished.len(),
                "seed={seed:#x} P-INV6 violated: report.count ({}) != vanished entities ({})",
                report.count,
                vanished.len()
            );

            // Build the ORACLE eligible-edge graph over gA planted entities: a pair is
            // eligible iff it is (same-normalized OR fuzzy Jaccard ≥ threshold) AND spans
            // ≥2 distinct episodes AND shares structure — the EXACT op gate, recomputed
            // independently from the plant records. The op merges the connected
            // components of this graph (union-find, transitive identity), so the oracle
            // must validate against COMPONENTS, not just direct pairs.
            let ga_planted: Vec<&PlantedEntity> =
                planted.iter().filter(|p| p.group == "gA").collect();

            let is_oracle_eligible = |a: &PlantedEntity, b: &PlantedEntity| -> bool {
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
                planted_share_structure(a, b)
            };

            // Oracle union-find over eligible edges (component membership = "may merge").
            let mut oracle = UnionFind::new();
            for p in &ga_planted {
                let _ = oracle.find(&p.raw_id);
            }
            for i in 0..ga_planted.len() {
                for j in (i + 1)..ga_planted.len() {
                    if is_oracle_eligible(ga_planted[i], ga_planted[j]) {
                        oracle.union(&ga_planted[i].raw_id, &ga_planted[j].raw_id);
                    }
                }
            }

            for loser_id in &vanished {
                let loser = ga_planted
                    .iter()
                    .find(|p| &p.raw_id == loser_id)
                    .unwrap_or_else(|| {
                        panic!("seed={seed:#x}: vanished id {loser_id} not planted")
                    });
                let loser_root = oracle.find(loser_id);

                // P-INV1 (homonym safety, THE critical one): a vanished entity MUST sit
                // in an ORACLE component of size > 1 — i.e. it is transitively connected
                // to ≥1 other entity by corroborated + episode-spanning edges. An entity
                // merged despite being a SINGLETON in the oracle graph is a label-alone
                // (homonym) false-merge.
                let component_members: Vec<&&PlantedEntity> = ga_planted
                    .iter()
                    .filter(|p| oracle.find(&p.raw_id) == loser_root)
                    .collect();
                assert!(
                    component_members.len() > 1,
                    "seed={seed:#x} P-INV1 violated: {loser_id} was merged but is a SINGLETON \
                     in the corroborated-eligible graph (label-alone / homonym false-merge)\n  \
                     loser={loser:?}"
                );

                // The surviving keeper is the LOWEST id in the component — assert it is
                // a same-cluster survivor (never a cross-cluster merge).
                let keeper_id: &String = component_members
                    .iter()
                    .map(|p| &p.raw_id)
                    .min()
                    .expect("non-empty component");
                assert!(
                    after_ids_ga.contains(keeper_id),
                    "seed={seed:#x} P-INV1: the component's lowest-id keeper {keeper_id} must survive"
                );

                // P-INV2 (episode-span) is a precondition of every oracle edge, so a
                // size>1 component already guarantees the loser participated in a
                // ≥2-episode edge. Assert directly on the loser's own linking edge too:
                // it must share a ≥2-episode eligible edge with SOME component member.
                let has_span_edge = component_members
                    .iter()
                    .any(|other| other.raw_id != loser.raw_id && is_oracle_eligible(loser, other));
                assert!(
                    has_span_edge,
                    "seed={seed:#x} P-INV2 violated: {loser_id} has no ≥2-episode corroborated \
                     edge to any component member"
                );

                // P-INV4 (delegate-correctness): loser GONE + no dangling refs.
                assert!(
                    !after_ids_ga.contains(&loser.raw_id),
                    "seed={seed:#x} P-INV4: loser {loser_id} must be deleted"
                );
                assert_eq!(
                    fact_refs_to(&graph, "gA", &loser.raw_id).await,
                    0,
                    "seed={seed:#x} P-INV4: no fact may still reference loser {loser_id}"
                );
                assert_eq!(
                    edge_refs_to(&graph, "gA", &loser.raw_id).await,
                    0,
                    "seed={seed:#x} P-INV4: no episodic edge may still reference loser {loser_id}"
                );
            }

            // P-INV5 (idempotent): a SECOND run on gA merges 0 and changes nothing.
            let second = cross_episode(&graph, "gA")
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: second cross_episode: {e}"));
            let after_second_ga = snapshot_entity_ids(&graph, "gA").await;
            assert_eq!(
                second.count, 0,
                "seed={seed:#x} P-INV5 violated: second run merged {} (must be 0)",
                second.count
            );
            assert_eq!(
                after_ids_ga, after_second_ga,
                "seed={seed:#x} P-INV5 violated: second run mutated the gA entity set"
            );
        }
    }
}

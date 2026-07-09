//! CONSOLIDATION op — community detection (ADR-066 §2.1, spec P4).
//!
//! Build a `petgraph::UnGraph` of a namespace's entity CO-OCCURRENCE (two entities
//! that share ≥1 episode), run the DETERMINISTIC synchronous weighted label
//! propagation proven by the compile-spike (`spike/community_detect.rs`), and persist
//! the partition into `entity_communities` + `community_summaries` (migration 019).
//! `communities_updated` counts the communities whose sorted-member SHA-256
//! (`community_member_hash`, substrate F-2) CHANGED vs the prior persisted state.
//! Zero-LLM, zero-embedding — a pure function of the co-occurrence topology.
//!
//! ## Graph model (DoD-P4.1)
//!
//! - **Nodes** = NON-catch-all entities (`entity_type_id != 0`) in `group_id`. The
//!   `id=0 "Entity"` catch-all (`defs_b.rs:215`) is a generic placeholder, not a real
//!   referent, so it never participates in community structure (C-INV5).
//! - **Edges** = a pair of entities that both anchor to the SAME `episode_id` via
//!   `episodic_edges`. **Edge weight = the number of DISTINCT episodes the pair
//!   co-occurs in** (a pair that recurs across many episodes binds harder). Built by a
//!   flat SQL fetch of `(episode_id, entity_id)` anchors + an in-Rust pairing pass —
//!   ZERO recursive-CTE (ADR §A5, spike C4).
//!
//! ## top_labels_json (DoD-P4.2, ADR §A6 — NO LLM summary)
//!
//! The community summary carries a DETERMINISTIC aggregate ONLY: the N most-frequent
//! entity TYPE labels among the community's members (`COALESCE(entity_types.name,
//! 'Entity')`, the runtime type-name — `entities.label` was dropped by Migration 009).
//! Ties broken by label string (smallest first) so the JSON is a pure function of
//! membership. NO LLM report is generated (R-16 scope guard).
//!
//! ## Determinism (DoD-P4.4, load-bearing)
//!
//! The whole op is deterministic: the SQL fetch is `ORDER BY`-stable, node insertion
//! into the `UnGraph` follows sorted entity-id order (so `NodeIndex` ↔ entity-id is a
//! fixed bijection per input), and the label-propagation itself is the compile-spiked
//! algorithm (sorted node iteration, `BTreeMap` tallies, smallest-label tie-break).
//! Running twice on the same graph yields the IDENTICAL partition.
//!
//! ## HAIRBALL CAVEAT (DoD-P4.6 / RISK-004 — documented known limitation, NOT a bug)
//!
//! Synchronous label propagation DEGENERATES on near-complete co-occurrence graphs.
//! When one big episode makes every entity co-occur with every other (a "hairball"),
//! every node sees the same neighbour-label distribution and the algorithm collapses
//! the whole graph into ONE community. **This is the HONEST answer** — a near-complete
//! graph genuinely has no sub-structure to surface ("everyone appeared together, so
//! there is one community"), NOT garbage. The modularity go/no-go spike
//! (`.ai-docs/research/p4-community-modularity-spike-2026-07-03.md`) confirmed:
//! STRUCTURED graphs yield meaningful communities (Q≈0.65/0.71), the hairball honestly
//! collapses to `count == 1`. Higher-modularity / incremental clustering (Leiden) that
//! would tease sub-structure out of a hairball is a deferred follow-up — see the ADR-066
//! "P4 hairball follow-up TD" (§A3/§A4). This op ships default-off (opt-in via
//! `DreamOpts.include_community_detection`).
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P4.1–P4.6) + §6 (communities corpus) + ADR-066 §2.1.

use std::collections::{BTreeMap, BTreeSet};

use chrono::Utc;
use metrics::counter;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::{
    community_member_hash, emit_decision, ConsolidationOpKind, DecisionMode, DecisionRecord,
    OpReport,
};

/// How many top type-labels to record per community in `top_labels_json` (DoD-P4.2).
/// Deterministic aggregate: the N most-frequent labels, ties broken smallest-string.
const TOP_LABELS_N: usize = 3;

/// Iteration cap for label propagation — matches the ratified spike (`community_detect.rs`).
/// The algorithm halts on fixpoint well before this on realistic graphs; the cap only
/// bounds a pathological non-converging input (deterministically).
const LABEL_PROP_MAX_ITERS: usize = 50;

/// Run community detection over `group_id` (spec P4).
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)` so the
/// deterministic fixture harness (`tests/consolidation_communities_test.rs`) can call it
/// under `feature = "test-utils"` (external test binaries cannot import `pub(crate)`
/// items — E0365). NOT part of the stable public API.
///
/// Args count = 2 — plain args, no params struct needed (TD-042 threshold 3).
#[doc(hidden)]
pub async fn communities(graph: &TemporalGraph, group_id: &str) -> Result<OpReport> {
    let mut report = OpReport::default();

    // ── 1. Build the in-memory co-occurrence graph (DoD-P4.1) ────────────────────
    let built = build_cooccurrence_graph(graph, group_id).await?;
    if built.node_id.is_empty() {
        // No non-catch-all entities → nothing to cluster. A namespace whose entities
        // are all catch-all (or empty) has no community structure — 0 updated.
        emit_updated_counter(0);
        return Ok(report);
    }

    // ── 2. Deterministic label propagation (DoD-P4.2 — the C1 spike algorithm) ────
    let labels = label_propagation(&built.graph, LABEL_PROP_MAX_ITERS);

    // Group node indices by community label. `BTreeMap` keys iterate ascending, so the
    // community-id renumbering below is a deterministic function of the partition.
    let mut by_label: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&node_idx, &label) in &labels {
        by_label.entry(label).or_default().push(node_idx);
    }

    // Renumber communities to dense 0..K ids in ascending-label order (stable, dense,
    // human-friendly persisted ids independent of the raw label seeds).
    let now = Utc::now().to_rfc3339();
    let mut community_id: i64 = 0;
    let mut updated = 0usize;

    // Load the PRIOR persisted member_hash per community_id so we can count only the
    // communities whose membership CHANGED (DoD-P4.3 / P4.5 idempotency). Keyed by the
    // member_hash VALUE (not the id) — community ids are renumbered each run, so hash
    // is the stable identity of a membership SET.
    let prior_hashes = load_prior_member_hashes(graph, group_id).await?;

    // Wipe the prior persisted partition for this namespace, then re-persist the fresh
    // one. Full-recompute (SYNTHESIS #15) — the whole namespace partition is rewritten;
    // `communities_updated` reflects how many of the NEW communities are new/changed.
    clear_persisted_partition(graph, group_id).await?;

    for node_indices in by_label.values() {
        // Member entity ids for this community, sorted (member_hash is order-independent
        // but sorting also gives a deterministic persisted row order + top-labels tie-break).
        let mut member_ids: Vec<&str> = node_indices
            .iter()
            .map(|&ni| built.node_id[ni].as_str())
            .collect();
        member_ids.sort_unstable();

        let member_hash = community_member_hash(&member_ids);
        // A community whose exact membership SET existed last run (same hash) is
        // unchanged; a new/changed membership set counts toward `communities_updated`.
        // Both arms emit a uniform decision record (ADR-070 Fork 2/3, §3.3): the
        // `unchanged_passthrough` arm is a GENUINELY-NEW observability point — the
        // idempotency no-op path previously emitted no signal at all. Communities has
        // no dry_run concept → always `Applied`.
        if !prior_hashes.contains(&member_hash) {
            updated += 1;
            emit_decision(DecisionRecord {
                op: ConsolidationOpKind::Communities,
                mode: DecisionMode::Applied,
                outcome: "updated",
                group_id,
                entity_refs: &member_ids,
                debug_context: None,
            });
        } else {
            emit_decision(DecisionRecord {
                op: ConsolidationOpKind::Communities,
                mode: DecisionMode::Applied,
                outcome: "unchanged_passthrough",
                group_id,
                entity_refs: &member_ids,
                debug_context: None,
            });
        }

        let top_labels = top_type_labels(&member_ids, &built.type_label);
        let top_labels_json = serde_json::to_string(&top_labels).unwrap_or_else(|_| "[]".into());

        persist_community(
            graph,
            PersistCommunityParams {
                group_id,
                community_id,
                member_ids: &member_ids,
                member_count: member_ids.len(),
                top_labels_json: &top_labels_json,
                member_hash: &member_hash,
                now: &now,
            },
        )
        .await?;

        community_id += 1;
    }

    emit_updated_counter(updated);
    tracing::info!(
        target: "kremory.dream.consolidation.communities",
        group_id,
        nodes = built.node_id.len(),
        communities = by_label.len(),
        updated,
        "community-detection sweep complete"
    );

    report.count = updated;
    Ok(report)
}

// ─── Graph construction (DoD-P4.1) ───────────────────────────────────────────────

/// The built co-occurrence graph + the parallel arrays mapping `NodeIndex` → entity id
/// and entity id → its type-name label.
struct BuiltGraph {
    graph: UnGraph<u32, f32>,
    /// `node_id[node_index]` = the entity id (name-slug) of that node. Node insertion
    /// order is sorted entity-id order, so `NodeIndex(i)` ↔ `node_id[i]` is a fixed
    /// bijection per input (determinism).
    node_id: Vec<String>,
    /// `type_label[entity_id]` = the entity's runtime TYPE-name label
    /// (`COALESCE(entity_types.name, 'Entity')`) — the top_labels aggregate grain.
    type_label: BTreeMap<String, String>,
}

/// Build the entity co-occurrence `UnGraph` for `group_id` (DoD-P4.1).
///
/// Nodes = NON-catch-all entities (`entity_type_id != 0`, C-INV5). An edge joins two
/// entities that both anchor to the SAME `episode_id`; the edge WEIGHT is the number of
/// DISTINCT episodes the pair co-occurs in. Pure SQL fetch + in-Rust pairing, no LLM,
/// no recursive CTE (ADR §A5).
async fn build_cooccurrence_graph(graph: &TemporalGraph, group_id: &str) -> Result<BuiltGraph> {
    // ── 1a. The node set: non-catch-all entities + their type-name label ─────────
    // `entities.entity_type_id != 0` excludes the catch-all (defs_b.rs:215). The
    // type-name label is `COALESCE(et.name, 'Entity')` joined on entity_type_id
    // (mirrors `list_entities_in_group`, entities.rs:380). ORDER BY id → deterministic.
    let mut rows = graph
        .conn
        .query(
            "SELECT e.id, COALESCE(et.name, 'Entity') AS type_label \
             FROM entities e \
             LEFT JOIN entity_types et \
               ON et.group_id = e.group_id AND et.id = e.entity_type_id \
             WHERE e.group_id = ?1 AND e.entity_type_id != 0 \
             ORDER BY e.id",
            libsql::params![group_id],
        )
        .await?;

    let mut node_id: Vec<String> = Vec::new();
    let mut node_index_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut type_label: BTreeMap<String, String> = BTreeMap::new();
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let label: String = row.get(1)?;
        node_index_of.insert(id.clone(), node_id.len());
        type_label.insert(id.clone(), label);
        node_id.push(id);
    }
    drop(rows);

    let mut ug: UnGraph<u32, f32> = UnGraph::new_undirected();
    let node_handles: Vec<NodeIndex> = (0..node_id.len())
        .map(|i| ug.add_node(u32::try_from(i).unwrap_or(u32::MAX)))
        .collect();

    if node_id.len() < 2 {
        // 0 or 1 node → no pair can co-occur; return the (edge-less) graph as built.
        return Ok(BuiltGraph {
            graph: ug,
            node_id,
            type_label,
        });
    }

    // ── 1b. Episode → member-set map, over NON-catch-all entities only ───────────
    // Fetch `(episode_id, entity_id)` anchors for entities in the node set. Namespace-
    // scoped by `entity_group_id` (entity ids are namespace-unique name-slugs; the
    // `IS NULL` arm tolerates pre-migration-A edges). ORDER BY for a stable pairing pass.
    let mut anchor_rows = graph
        .conn
        .query(
            "SELECT ee.episode_id, ee.entity_id \
             FROM episodic_edges ee \
             JOIN entities e \
               ON e.id = ee.entity_id \
              AND (ee.entity_group_id = e.group_id OR ee.entity_group_id IS NULL) \
             WHERE e.group_id = ?1 AND e.entity_type_id != 0 \
             ORDER BY ee.episode_id, ee.entity_id",
            libsql::params![group_id],
        )
        .await?;

    // Per episode, the DISTINCT set of member entity-ids (only those in the node set).
    let mut episode_members: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    while let Some(row) = anchor_rows.next().await? {
        let episode_id: i64 = row.get(0)?;
        let entity_id: String = row.get(1)?;
        // Only entities in the node set contribute (the JOIN already enforces
        // non-catch-all, but guard against an anchor to a since-removed entity).
        if node_index_of.contains_key(&entity_id) {
            episode_members
                .entry(episode_id)
                .or_default()
                .insert(entity_id);
        }
    }
    drop(anchor_rows);

    // ── 1c. Pair co-occurrences → weighted edges (weight = distinct-episode count) ─
    // For each episode, every unordered pair of its (distinct) members co-occurs once.
    // Accumulate per-pair episode counts in a deterministic `BTreeMap` keyed by the
    // SORTED (lo, hi) id pair, then emit one weighted edge per pair.
    let mut pair_weight: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    for members in episode_members.values() {
        let ids: Vec<&String> = members.iter().collect(); // BTreeSet → sorted, distinct.
        for a in 0..ids.len() {
            for b in (a + 1)..ids.len() {
                let ia = node_index_of[ids[a]];
                let ib = node_index_of[ids[b]];
                let key = if ia <= ib { (ia, ib) } else { (ib, ia) };
                *pair_weight.entry(key).or_insert(0) += 1;
            }
        }
    }

    for (&(ia, ib), &weight) in &pair_weight {
        ug.add_edge(node_handles[ia], node_handles[ib], weight as f32);
    }

    Ok(BuiltGraph {
        graph: ug,
        node_id,
        type_label,
    })
}

// ─── Deterministic label propagation (DoD-P4.2 — VERBATIM from the C1 spike) ──────

/// Deterministic synchronous WEIGHTED label propagation — the exact algorithm ratified
/// by the compile-spike (`spike/community_detect.rs` / `spike/community_quality.rs`).
///
/// Each node adopts the neighbour label with the greatest summed EDGE WEIGHT; ties are
/// broken by the SMALLEST label id. Nodes are processed in sorted `NodeIndex` order and
/// tallies are kept in a `BTreeMap` (ascending key order), so the whole function is a
/// pure, reproducible function of the graph topology (C-INV1 determinism). Halts on a
/// fixpoint (no label changed in a pass) or at `max_iters`.
///
/// Weighted == unweighted when all edge weights are 1.0 (the spike's determinism proof
/// holds either way); the weighted variant lets a many-episode co-occurrence pull harder.
fn label_propagation(g: &UnGraph<u32, f32>, max_iters: usize) -> BTreeMap<usize, usize> {
    // Seed: each node is its own community (node index as the initial label).
    let mut labels: BTreeMap<usize, usize> =
        g.node_indices().map(|n| (n.index(), n.index())).collect();
    for _ in 0..max_iters {
        let mut changed = false;
        // Deterministic: iterate node indices in ascending order.
        let ordered: Vec<NodeIndex> = {
            let mut v: Vec<NodeIndex> = g.node_indices().collect();
            v.sort_by_key(|n| n.index());
            v
        };
        for n in ordered {
            // Tally neighbour labels weighted by edge weight (BTreeMap → det. order).
            let mut tally: BTreeMap<usize, f64> = BTreeMap::new();
            for e in g.edges(n) {
                let nb = if e.source() == n {
                    e.target()
                } else {
                    e.source()
                };
                *tally.entry(labels[&nb.index()]).or_insert(0.0) += f64::from(*e.weight());
            }
            if tally.is_empty() {
                continue; // isolated node keeps its own singleton label.
            }
            // Most-weighted label; tie → smallest label id (BTreeMap iterates ascending).
            let best = tally
                .iter()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.0.cmp(a.0))
                })
                .map(|(l, _)| *l)
                .unwrap_or(n.index());
            if labels[&n.index()] != best {
                labels.insert(n.index(), best);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    labels
}

// ─── top_labels aggregate (DoD-P4.2, ADR §A6 — deterministic, NO LLM) ─────────────

/// The N most-frequent TYPE labels among a community's members, most-frequent first,
/// ties broken by label string (ascending). Pure deterministic aggregate — the ADR §A6
/// alternative to an LLM summary (R-16 scope guard).
fn top_type_labels(member_ids: &[&str], type_label: &BTreeMap<String, String>) -> Vec<String> {
    // Count label frequency in a BTreeMap (ascending-label key order → deterministic
    // tie-break for equal counts).
    let mut freq: BTreeMap<String, usize> = BTreeMap::new();
    for id in member_ids {
        if let Some(label) = type_label.get(*id) {
            *freq.entry(label.clone()).or_insert(0) += 1;
        }
    }
    // Sort by (descending count, ascending label). The BTreeMap iteration is already
    // ascending-label; a stable sort by descending count preserves that tie-break.
    let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(TOP_LABELS_N)
        .map(|(label, _)| label)
        .collect()
}

// ─── Persistence (DoD-P4.2/P4.3) ──────────────────────────────────────────────────

/// Load the set of PRIOR persisted `member_hash` values for `group_id`. A fresh
/// community whose sorted-member hash is already in this set is UNCHANGED and does not
/// count toward `communities_updated` (DoD-P4.3 / P4.5 idempotency).
async fn load_prior_member_hashes(
    graph: &TemporalGraph,
    group_id: &str,
) -> Result<BTreeSet<String>> {
    let mut rows = graph
        .conn
        .query(
            "SELECT member_hash FROM community_summaries WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await?;
    let mut out: BTreeSet<String> = BTreeSet::new();
    while let Some(row) = rows.next().await? {
        out.insert(row.get::<String>(0)?);
    }
    Ok(out)
}

/// Wipe the prior persisted partition for `group_id` (full-recompute, SYNTHESIS #15).
/// Both `entity_communities` and `community_summaries` rows for the namespace are
/// cleared before the fresh partition is written, so stale membership never lingers.
async fn clear_persisted_partition(graph: &TemporalGraph, group_id: &str) -> Result<()> {
    graph
        .conn
        .execute(
            "DELETE FROM entity_communities WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await?;
    graph
        .conn
        .execute(
            "DELETE FROM community_summaries WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await?;
    Ok(())
}

/// Bundled params for [`persist_community`] — args-as-object per TD-042
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
struct PersistCommunityParams<'a> {
    group_id: &'a str,
    community_id: i64,
    member_ids: &'a [&'a str],
    member_count: usize,
    top_labels_json: &'a str,
    member_hash: &'a str,
    now: &'a str,
}

/// Persist ONE community: an `entity_communities` row per member + a single
/// `community_summaries` row (member_count, top_labels_json, member_hash). Deterministic.
async fn persist_community(graph: &TemporalGraph, p: PersistCommunityParams<'_>) -> Result<()> {
    for member in p.member_ids {
        graph
            .conn
            .execute(
                "INSERT OR REPLACE INTO entity_communities \
                 (group_id, entity_id, community_id, updated_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![p.group_id, *member, p.community_id, p.now],
            )
            .await?;
    }
    graph
        .conn
        .execute(
            "INSERT OR REPLACE INTO community_summaries \
             (group_id, community_id, member_count, top_labels_json, member_hash, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![
                p.group_id,
                p.community_id,
                p.member_count as i64,
                p.top_labels_json,
                p.member_hash,
                p.now,
            ],
        )
        .await?;
    Ok(())
}

/// Emit the source-attributed `communities_updated` counter (Rule 19). The o11y
/// cross-check asserts this equals `OpReport.count`.
fn emit_updated_counter(updated: usize) {
    counter!("kremory.dream.consolidation.communities_updated_total").increment(updated as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{
        InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
    };
    use crate::core::schema::TemporalGraph;
    use chrono::Utc;

    // ── DB plant helpers ─────────────────────────────────────────────────────────

    /// Insert a NON-catch-all entity (`entity_type_id = 1` unless overridden) so it
    /// participates in community structure. `entity_type_id = 0` is the catch-all and
    /// is excluded from the node set (C-INV5).
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn insert_typed_entity(graph: &TemporalGraph, gid: &str, id: &str, type_id: u32) {
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: type_id,
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

    /// The persisted `(entity_id -> community_id)` map for `group_id` (from
    /// `entity_communities`), for membership assertions.
    async fn persisted_membership(graph: &TemporalGraph, gid: &str) -> BTreeMap<String, i64> {
        let mut rows = graph
            .conn
            .query(
                "SELECT entity_id, community_id FROM entity_communities \
                 WHERE group_id = ?1 ORDER BY entity_id",
                libsql::params![gid],
            )
            .await
            .expect("membership");
        let mut out = BTreeMap::new();
        while let Some(row) = rows.next().await.expect("row") {
            out.insert(
                row.get::<String>(0).expect("entity"),
                row.get::<i64>(1).expect("community"),
            );
        }
        out
    }

    /// Distinct community count from `community_summaries` for `group_id`.
    async fn persisted_community_count(graph: &TemporalGraph, gid: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM community_summaries WHERE group_id = ?1",
                libsql::params![gid],
            )
            .await
            .expect("count");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("count")
    }

    // ── L1 unit: label_propagation reuses the C1 spike (two-triangle bridge) ──────

    #[test]
    fn label_propagation_two_triangle_bridge_is_deterministic() {
        // Mirror the spike topology: two triangles {a,b,c} dense, bridged to {d}.
        let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(a, c, 1.0);
        g.add_edge(c, d, 1.0);
        let r1 = label_propagation(&g, LABEL_PROP_MAX_ITERS);
        let r2 = label_propagation(&g, LABEL_PROP_MAX_ITERS);
        assert_eq!(
            r1, r2,
            "label propagation must be deterministic across runs"
        );
        // Every node assigned exactly once.
        assert_eq!(r1.len(), 4, "all four nodes labelled");
    }

    #[test]
    fn label_propagation_isolated_nodes_are_own_singletons() {
        // Two disconnected components → at least two distinct communities.
        let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        g.add_edge(a, b, 1.0); // component 1
        g.add_edge(c, d, 1.0); // component 2
        let labels = label_propagation(&g, LABEL_PROP_MAX_ITERS);
        let comms: BTreeSet<usize> = labels.values().copied().collect();
        assert!(
            comms.len() >= 2,
            "two disconnected edges → ≥2 communities, got {}",
            comms.len()
        );
    }

    // ── L1 unit: top_type_labels is a deterministic frequency aggregate ──────────

    #[test]
    fn top_type_labels_most_frequent_first_ties_smallest_string() {
        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        labels.insert("e1".into(), "Person".into());
        labels.insert("e2".into(), "Person".into());
        labels.insert("e3".into(), "Company".into());
        labels.insert("e4".into(), "City".into());
        let members = ["e1", "e2", "e3", "e4"];
        let top = top_type_labels(&members, &labels);
        // Person=2 (most frequent) first; City & Company tie at 1 → smallest string first.
        assert_eq!(top, vec!["Person", "City", "Company"]);
    }

    // ── L2 fixture: two-triangle-bridge → 2 communities (spec §6) ────────────────

    #[tokio::test]
    async fn two_triangle_bridge_yields_two_communities() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_bridge";
        // Two dense triangles sharing episodes, linked by ONE weak cross-episode edge.
        // Triangle 1: a,b,c co-occur in e1 (+ pairwise reinforcement in e2/e3).
        // Triangle 2: d,e,f co-occur in e4 (+ e5/e6). Bridge: c,d co-occur in ONE episode.
        for id in ["a", "b", "c", "d", "e", "f"] {
            insert_typed_entity(&graph, gid, id, 1).await;
        }
        // Triangle 1 — a,b,c share three episodes (heavy intra-cluster weight).
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["a", "b", "c"] {
                anchor(&graph, gid, ep, id).await;
            }
        }
        // Triangle 2 — d,e,f share three episodes.
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["d", "e", "f"] {
                anchor(&graph, gid, ep, id).await;
            }
        }
        // Bridge — c,d co-occur in exactly ONE episode (weight 1).
        let bridge = new_episode(&graph).await;
        anchor(&graph, gid, bridge, "c").await;
        anchor(&graph, gid, bridge, "d").await;

        let report = communities(&graph, gid).await.expect("communities");
        assert_eq!(
            persisted_community_count(&graph, gid).await,
            2,
            "two dense triangles weakly bridged → 2 communities"
        );
        // First run: both communities are new → updated == community count.
        assert_eq!(
            report.count, 2,
            "first run: both communities counted as updated"
        );

        // Membership: {a,b,c} share one community, {d,e,f} another; the two differ.
        let m = persisted_membership(&graph, gid).await;
        assert_eq!(m["a"], m["b"], "a,b same community");
        assert_eq!(m["b"], m["c"], "b,c same community");
        assert_eq!(m["d"], m["e"], "d,e same community");
        assert_eq!(m["e"], m["f"], "e,f same community");
        assert_ne!(m["a"], m["d"], "the two triangles are DISTINCT communities");
    }

    // ── L2 fixture: HAIRBALL → 1 HONEST community (DoD-P4.6, documented collapse) ─
    // The dense_episode_hairball quality case (RISK-004): one big episode makes every
    // entity co-occur with every other → near-complete graph → synchronous label
    // propagation HONESTLY collapses to ONE community. Assert count == 1 (NOT >1) — the
    // documented known limitation, not garbage. Must NOT crash.

    #[tokio::test]
    async fn dense_episode_hairball_collapses_to_one_honest_community() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_hairball";
        // 8 entities ALL appearing in ONE big episode → near-complete co-occurrence.
        let ids = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"];
        for id in ids {
            insert_typed_entity(&graph, gid, id, 1).await;
        }
        let big = new_episode(&graph).await;
        for id in ids {
            anchor(&graph, gid, big, id).await;
        }

        // Must NOT crash + HONESTLY reports 1 community (documented hairball collapse).
        let report = communities(&graph, gid)
            .await
            .expect("communities (no crash)");
        assert_eq!(
            persisted_community_count(&graph, gid).await,
            1,
            "hairball HONESTLY collapses to ONE community (documented caveat, not a bug)"
        );
        assert_eq!(report.count, 1, "one community, first run → updated == 1");
        // All 8 entities land in the SAME single community.
        let m = persisted_membership(&graph, gid).await;
        let comms: BTreeSet<i64> = m.values().copied().collect();
        assert_eq!(comms.len(), 1, "all members in the single honest community");
        assert_eq!(m.len(), 8, "every non-catch-all entity assigned");
    }

    // ── L2 fixture: symmetric-tie → deterministic smallest-label tie-break ───────

    #[tokio::test]
    async fn symmetric_tie_resolves_deterministically() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_tie";
        // A symmetric "path" a-b-c where b is pulled equally by a and c. The
        // smallest-label tie-break must resolve it to a fixed, reproducible partition.
        for id in ["a", "b", "c"] {
            insert_typed_entity(&graph, gid, id, 1).await;
        }
        let e1 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "a").await;
        anchor(&graph, gid, e1, "b").await;
        let e2 = new_episode(&graph).await;
        anchor(&graph, gid, e2, "b").await;
        anchor(&graph, gid, e2, "c").await;

        // Run twice — the persisted membership must be IDENTICAL (deterministic).
        communities(&graph, gid).await.expect("run 1");
        let m1 = persisted_membership(&graph, gid).await;
        communities(&graph, gid).await.expect("run 2");
        let m2 = persisted_membership(&graph, gid).await;
        assert_eq!(
            m1, m2,
            "symmetric-tie partition is reproducible (exact membership)"
        );
        // Every node assigned exactly one community.
        assert_eq!(
            m1.len(),
            3,
            "all three nodes assigned exactly one community"
        );
    }

    // ── L2 fixture: unchanged-rerun → communities_updated == 0 (DoD-P4.5) ────────

    #[tokio::test]
    async fn unchanged_rerun_reports_zero_updated() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_idem";
        for id in ["a", "b", "c", "d", "e", "f"] {
            insert_typed_entity(&graph, gid, id, 1).await;
        }
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["a", "b", "c"] {
                anchor(&graph, gid, ep, id).await;
            }
        }
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["d", "e", "f"] {
                anchor(&graph, gid, ep, id).await;
            }
        }

        let first = communities(&graph, gid).await.expect("first");
        assert!(first.count >= 1, "first run counts new communities");
        // Second run on the UNCHANGED graph → every member_hash matches → 0 updated.
        let second = communities(&graph, gid).await.expect("second");
        assert_eq!(
            second.count, 0,
            "unchanged graph rerun → communities_updated == 0 (idempotent, DoD-P4.5)"
        );
    }

    /// ADR-070 §4 matrix: the idempotency-hash-unchanged path (previously SILENT) now
    /// emits `decision_total{op=communities,outcome=unchanged_passthrough}` — the
    /// "first-class observability closes a real gap" half of Fork 2/3.
    #[tokio::test]
    async fn communities_unchanged_passthrough_emits_new_decision_signal() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_idem_decision";
        for id in ["a", "b", "c", "d", "e", "f"] {
            insert_typed_entity(&graph, gid, id, 1).await;
        }
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["a", "b", "c"] {
                anchor(&graph, gid, ep, id).await;
            }
        }
        for _ in 0..3 {
            let ep = new_episode(&graph).await;
            for id in ["d", "e", "f"] {
                anchor(&graph, gid, ep, id).await;
            }
        }

        // First run persists the partition (communities are NEW → `updated`).
        let _first = communities(&graph, gid).await.expect("first");

        // Capture the SECOND run: every member_hash matches → unchanged_passthrough.
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let second = communities(&graph, gid).await.expect("second");
        assert_eq!(second.count, 0, "unchanged rerun → 0 updated");

        let passthrough = snapshotter.snapshot().into_vec().into_iter().find_map(
            |(composite_key, _, _, value)| {
                let key = composite_key.key();
                if key.name() != "kremory.dream.consolidation.decision_total" {
                    return None;
                }
                let labels: std::collections::HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("op").copied() != Some("communities")
                    || labels.get("outcome").copied() != Some("unchanged_passthrough")
                {
                    return None;
                }
                match value {
                    DebugValue::Counter(n) => Some(n),
                    _ => None,
                }
            },
        );
        assert!(
            matches!(passthrough, Some(n) if n >= 1),
            "unchanged_passthrough decision must fire on the idempotent rerun (was silent)"
        );
    }

    // ── L2 fixture: catch-all entities (type 0) excluded (C-INV5) ────────────────

    #[tokio::test]
    async fn catch_all_entities_are_excluded_from_communities() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_catchall";
        // Two real (typed) entities + one catch-all (type 0). The catch-all must NOT
        // appear in any community even though it co-occurs with the real ones.
        insert_typed_entity(&graph, gid, "real_a", 1).await;
        insert_typed_entity(&graph, gid, "real_b", 1).await;
        insert_typed_entity(&graph, gid, "junk", 0).await; // catch-all — excluded.
        let e1 = new_episode(&graph).await;
        for id in ["real_a", "real_b", "junk"] {
            anchor(&graph, gid, e1, id).await;
        }
        let e2 = new_episode(&graph).await;
        for id in ["real_a", "real_b", "junk"] {
            anchor(&graph, gid, e2, id).await;
        }

        communities(&graph, gid).await.expect("communities");
        let m = persisted_membership(&graph, gid).await;
        assert!(m.contains_key("real_a"), "typed entity assigned");
        assert!(m.contains_key("real_b"), "typed entity assigned");
        assert!(
            !m.contains_key("junk"),
            "catch-all entity (type 0) MUST be excluded from communities (C-INV5)"
        );
    }

    // ── Empty / single-entity namespace: 0 communities, no crash ─────────────────

    #[tokio::test]
    async fn empty_namespace_reports_zero() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let report = communities(&graph, "g_empty").await.expect("communities");
        assert_eq!(report.count, 0, "empty namespace → 0 updated");
        assert_eq!(persisted_community_count(&graph, "g_empty").await, 0);
        assert!(report.warnings.is_empty());
    }

    #[tokio::test]
    async fn single_entity_namespace_forms_one_singleton() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_single";
        insert_typed_entity(&graph, gid, "lonely", 1).await;
        let e1 = new_episode(&graph).await;
        anchor(&graph, gid, e1, "lonely").await;

        let report = communities(&graph, gid).await.expect("communities");
        // One node, no edges → one singleton community.
        assert_eq!(persisted_community_count(&graph, gid).await, 1);
        assert_eq!(report.count, 1, "the lone entity forms its own community");
        let m = persisted_membership(&graph, gid).await;
        assert_eq!(m.len(), 1);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PROPERTY / INVARIANT TIER (C-INV1..C-INV5) — randomized-input safety proof.
    //
    // community detection PERSISTS a partition read by downstream consumers, so its
    // invariants must hold over RANDOMIZED entity/episode graphs, not just hand-picked
    // cases. Each of N iterations builds a graph across TWO namespaces ("gA" clustered,
    // "gB" untouched), plants random typed + catch-all entities and random episode
    // anchors, runs `communities` on "gA", and asserts:
    //
    //   C-INV1 (determinism): running label-prop twice on the SAME built graph yields
    //          the IDENTICAL partition.
    //   C-INV2 (total assignment): every non-catch-all gA entity is in exactly ONE
    //          persisted community.
    //   C-INV3 (namespace isolation): no gB entity ever appears in a gA community, and
    //          gB's persisted partition is untouched by the gA sweep.
    //   C-INV4 (idempotent): a second run on the unchanged graph → communities_updated==0
    //          and every persisted member_hash is unchanged.
    //   C-INV5 (catch-all exclusion): a type-0 entity is NEVER in any community.
    //
    // PRNG: hand-rolled seeded SplitMix64 (mirrors `cross_episode.rs`) — the op is async,
    // seed = fixed base ^ iteration index (deterministic + reproducible), and the exact
    // `seed` is printed on any failure so the case reproduces from one line.
    // ══════════════════════════════════════════════════════════════════════════

    /// Deterministic SplitMix64 PRNG (Vigna, public-domain reference). Mirrors the
    /// `cross_episode.rs` property harness — seedable, reproducible, no external dep.
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

    /// The set of persisted `member_hash` values for `group_id`.
    async fn persisted_member_hashes(graph: &TemporalGraph, gid: &str) -> BTreeSet<String> {
        let mut rows = graph
            .conn
            .query(
                "SELECT member_hash FROM community_summaries WHERE group_id = ?1",
                libsql::params![gid],
            )
            .await
            .expect("hashes");
        let mut out = BTreeSet::new();
        while let Some(row) = rows.next().await.expect("row") {
            out.insert(row.get::<String>(0).expect("hash"));
        }
        out
    }

    #[tokio::test]
    async fn property_communities_invariants_over_random_inputs() {
        const BASE_SEED: u64 = 0x434F_4D4D_5F50_3400; // "COMM_P4\0"-ish, arbitrary fixed.
        const ITERATIONS: u64 = 300; // ≥ 200 required.

        const GROUPS: [&str; 2] = ["gA", "gB"];

        for iter in 0..ITERATIONS {
            let seed = BASE_SEED ^ iter;
            let mut rng = SplitMix64::new(seed);

            let graph = TemporalGraph::open_in_memory()
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: open graph: {e}"));

            // Shared episode pool (ids reused across entities → same-episode and
            // distinct-episode co-occurrences both arise).
            let n_episodes = rng.in_range(1, 4);
            let mut episodes: Vec<i64> = Vec::new();
            for _ in 0..n_episodes {
                episodes.push(new_episode(&graph).await);
            }

            // Track, per namespace: the set of NON-catch-all entity ids planted, and the
            // set of catch-all ids planted (for C-INV5).
            let mut typed_ids: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
            let mut catchall_ids: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();

            let n_entities = rng.in_range(2, 10);
            for e in 0..n_entities {
                let group = GROUPS[rng.below(GROUPS.len() as u64) as usize];
                // ~1/4 of entities are catch-all (type 0); the rest are typed (1..=3).
                let is_catchall = rng.chance(1, 4);
                let type_id = if is_catchall {
                    0
                } else {
                    u32::try_from(rng.in_range(1, 3)).unwrap_or(1)
                };
                // Namespace-unique id — the cross-namespace collision guard refuses a
                // repeated name-slug, so key the id on the namespace + a sequence.
                let id = format!("{group}_ent{e}");
                let inserted = graph
                    .insert_entity_with_group(InsertEntityWithGroupParams {
                        id: &id,
                        entity_type_id: type_id,
                        properties: serde_json::json!({ "name": id }),
                        group_id: Some(group),
                    })
                    .await
                    .is_ok();
                if !inserted {
                    continue;
                }
                if is_catchall {
                    catchall_ids.entry(group).or_default().insert(id.clone());
                } else {
                    typed_ids.entry(group).or_default().insert(id.clone());
                }

                // Anchor to a random subset of episodes (1..=all).
                let n_eps = rng.in_range(1, episodes.len() as u64);
                for _ in 0..n_eps {
                    let ep = episodes[rng.below(episodes.len() as u64) as usize];
                    anchor(&graph, group, ep, &id).await;
                }
            }

            // ── C-INV1 (determinism): build the gA graph, run label-prop twice ───
            let built = build_cooccurrence_graph(&graph, "gA")
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: build gA graph: {e}"));
            let p1 = label_propagation(&built.graph, LABEL_PROP_MAX_ITERS);
            let p2 = label_propagation(&built.graph, LABEL_PROP_MAX_ITERS);
            assert_eq!(
                p1, p2,
                "seed={seed:#x} C-INV1 violated: label propagation is non-deterministic"
            );

            // ── Run the op on gA (persists the partition) ─────────────────────────
            let first = communities(&graph, "gA")
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: communities gA: {e}"));

            let membership_ga = persisted_membership(&graph, "gA").await;
            let empty = BTreeSet::new();
            let typed_ga = typed_ids.get("gA").unwrap_or(&empty);
            let catchall_ga = catchall_ids.get("gA").unwrap_or(&empty);

            // ── C-INV2 (total assignment): every non-catch-all gA entity in exactly
            //    ONE community. `entity_communities` PK is (group_id, entity_id), so a
            //    present key IS "exactly one" — assert every typed id is present. ──────
            for id in typed_ga {
                assert!(
                    membership_ga.contains_key(id),
                    "seed={seed:#x} C-INV2 violated: typed gA entity {id} not assigned to a community"
                );
            }
            // No EXTRA member rows beyond the planted typed set (guards against a stray
            // catch-all or gB leak sneaking in).
            assert_eq!(
                membership_ga.len(),
                typed_ga.len(),
                "seed={seed:#x} C-INV2: gA community membership must be EXACTLY the typed entity set"
            );

            // ── C-INV5 (catch-all exclusion): no type-0 gA entity in any community ─
            for id in catchall_ga {
                assert!(
                    !membership_ga.contains_key(id),
                    "seed={seed:#x} C-INV5 violated: catch-all gA entity {id} appears in a community"
                );
            }

            // ── C-INV3 (namespace isolation): no gB entity in a gA community, and gB's
            //    own partition is untouched (the gA sweep never wrote gB rows). ────────
            let gb_all: BTreeSet<String> = typed_ids
                .get("gB")
                .unwrap_or(&empty)
                .union(catchall_ids.get("gB").unwrap_or(&empty))
                .cloned()
                .collect();
            for id in &gb_all {
                assert!(
                    !membership_ga.contains_key(id),
                    "seed={seed:#x} C-INV3 violated: gB entity {id} appears in a gA community"
                );
            }
            assert_eq!(
                persisted_community_count(&graph, "gB").await,
                0,
                "seed={seed:#x} C-INV3: gA sweep must not persist any gB community"
            );

            // ── C-INV4 (idempotent): second run → 0 updated, hashes unchanged ─────
            let hashes_before = persisted_member_hashes(&graph, "gA").await;
            let second = communities(&graph, "gA")
                .await
                .unwrap_or_else(|e| panic!("seed={seed:#x}: communities gA rerun: {e}"));
            assert_eq!(
                second.count, 0,
                "seed={seed:#x} C-INV4 violated: unchanged rerun reported {} updated (expected 0)",
                second.count
            );
            let hashes_after = persisted_member_hashes(&graph, "gA").await;
            assert_eq!(
                hashes_before, hashes_after,
                "seed={seed:#x} C-INV4 violated: member_hashes changed on an unchanged rerun"
            );

            // First-run sanity: `communities_updated` never exceeds the community count
            // (each community counts at most once).
            let comm_count = persisted_community_count(&graph, "gA").await as usize;
            assert!(
                first.count <= comm_count,
                "seed={seed:#x}: first-run updated ({}) exceeds community count ({comm_count})",
                first.count
            );
        }
    }
}

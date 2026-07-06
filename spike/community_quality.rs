// THROWAWAY SPIKE (ADR-066 §2.6 / DoD-P4.6 / RISK-004) — MODULARITY GO/NO-GO.
//
// The compile-spike `community_detect.rs` already PROVED the label-propagation
// algorithm compiles + is deterministic on petgraph 0.7. The OPEN question this
// spike answers empirically: does deterministic synchronous label propagation
// produce USEFUL communities on realistic kremory entity co-occurrence graphs,
// or collapse everything into ONE giant "hairball" community (which would make
// `communities_updated` a meaningless counter)?
//
// We compute, per topology:
//   1. Newman-Girvan modularity Q of the resulting partition (weighted).
//        Q ~ 0   -> no better than random / hairball.
//        Q > 0.3 -> meaningful community structure (standard threshold).
//   2. Community count + size distribution.
//
// Topologies (constructed as (nodes, weighted edges), mirroring the ADR §2.1
// co-occurrence model: nodes = non-catch-all entities in a namespace; an edge
// exists between two entities that share >=1 episode, weight = shared-episode
// count):
//   A  — clear planted community structure (3 dense cliques weakly bridged).
//   B  — dense/hairball (one big episode → near-complete co-occurrence). THE risk.
//   C  — sparse chain.
//   D  — realistic mixed: mirrors kremory's actual `whole_project_e2e` ingest —
//        several distinct-domain episodes, each a small entity clique, slug
//        overlap near-zero between domains (so the co-occurrence graph is a set
//        of disjoint per-episode cliques + a couple of recurring bridge entities).
//
// Determinism note: the algorithm is the EXACT one from community_detect.rs
// (BTreeMap-ordered, sorted node iteration, smallest-label tie-break). We run
// each topology TWICE and assert the partition is identical.
//
// Compile:
//   rustc --edition 2021 -L target/debug/deps \
//     --extern petgraph=$(ls -t target/debug/deps/libpetgraph-*.rlib | head -1) \
//     spike/community_quality.rs -o /tmp/community_quality && /tmp/community_quality

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, BTreeSet};

/// Deterministic synchronous label propagation — VERBATIM from
/// `spike/community_detect.rs` (the ratified algorithm). Weighted variant:
/// neighbour votes are summed by EDGE WEIGHT, not by count, so a heavy edge
/// pulls harder. (Unweighted == weighted when all weights are 1.0.)
fn label_propagation(g: &UnGraph<u32, f32>, max_iters: usize) -> BTreeMap<usize, usize> {
    let mut labels: BTreeMap<usize, usize> =
        g.node_indices().map(|n| (n.index(), n.index())).collect();
    for _ in 0..max_iters {
        let mut changed = false;
        let ordered: Vec<NodeIndex> = {
            let mut v: Vec<NodeIndex> = g.node_indices().collect();
            v.sort_by_key(|n| n.index());
            v
        };
        for n in ordered {
            // Tally neighbour labels weighted by edge weight (BTreeMap => det. order).
            let mut tally: BTreeMap<usize, f64> = BTreeMap::new();
            for e in g.edges(n) {
                let nb = if e.source() == n { e.target() } else { e.source() };
                *tally.entry(labels[&nb.index()]).or_insert(0.0) += *e.weight() as f64;
            }
            if tally.is_empty() {
                continue;
            }
            // Most-weighted label; tie -> smallest label id.
            let best = tally
                .iter()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.0.cmp(a.0))
                })
                .map(|(l, _)| *l)
                .unwrap();
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

/// Newman-Girvan weighted modularity of a partition.
/// Q = (1/2m) * Σ_ij [ A_ij - k_i k_j / 2m ] δ(c_i, c_j)
/// where A_ij is edge weight, k_i is weighted degree, m is total edge weight,
/// δ = 1 iff i and j in the same community.
fn modularity(g: &UnGraph<u32, f32>, labels: &BTreeMap<usize, usize>) -> f64 {
    // total edge weight m (sum of weights; each undirected edge counted once)
    let mut m = 0.0f64;
    for e in g.edge_references() {
        m += *e.weight() as f64;
    }
    if m == 0.0 {
        return 0.0;
    }
    // weighted degree per node
    let mut deg: BTreeMap<usize, f64> = g.node_indices().map(|n| (n.index(), 0.0)).collect();
    for e in g.edge_references() {
        let (s, t) = (e.source().index(), e.target().index());
        let w = *e.weight() as f64;
        *deg.get_mut(&s).unwrap() += w;
        *deg.get_mut(&t).unwrap() += w;
    }
    let two_m = 2.0 * m;
    // group nodes by community
    let mut comms: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&node, &c) in labels {
        comms.entry(c).or_default().push(node);
    }
    // within-community edge weight (l_c) and community weighted-degree sum (d_c)
    let mut q = 0.0f64;
    for (_c, members) in &comms {
        let mset: BTreeSet<usize> = members.iter().copied().collect();
        let mut l_c = 0.0f64; // sum of weights of edges INSIDE this community
        for e in g.edge_references() {
            let (s, t) = (e.source().index(), e.target().index());
            if mset.contains(&s) && mset.contains(&t) {
                l_c += *e.weight() as f64;
            }
        }
        let d_c: f64 = members.iter().map(|n| deg[n]).sum();
        // contribution: l_c/m - (d_c/2m)^2
        q += l_c / m - (d_c / two_m) * (d_c / two_m);
    }
    q
}

struct Topo {
    name: &'static str,
    desc: &'static str,
    g: UnGraph<u32, f32>,
}

/// A — clear planted structure: 3 dense 4-cliques, each bridged to the next by
/// a single weight-1 edge. Ground truth = 3 communities.
fn topo_a() -> Topo {
    let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
    let n: Vec<NodeIndex> = (0..12u32).map(|i| g.add_node(i)).collect();
    // three 4-cliques: {0..3}, {4..7}, {8..11}, intra weight 5 (co-occur a lot)
    for base in [0usize, 4, 8] {
        for i in base..base + 4 {
            for j in (i + 1)..base + 4 {
                g.add_edge(n[i], n[j], 5.0);
            }
        }
    }
    // weak bridges (single shared episode)
    g.add_edge(n[3], n[4], 1.0);
    g.add_edge(n[7], n[8], 1.0);
    Topo { name: "A", desc: "3 dense 4-cliques weakly bridged (planted GT=3)", g }
}

/// B — dense/hairball: 12 entities ALL appearing in one big episode →
/// near-complete co-occurrence graph, every pair weight 1. THE risk case.
fn topo_b() -> Topo {
    let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
    let n: Vec<NodeIndex> = (0..12u32).map(|i| g.add_node(i)).collect();
    for i in 0..12usize {
        for j in (i + 1)..12usize {
            g.add_edge(n[i], n[j], 1.0);
        }
    }
    Topo { name: "B", desc: "12-entity near-complete clique (one giant episode) — HAIRBALL", g }
}

/// C — sparse chain: 12 entities in a line, weight 1.
fn topo_c() -> Topo {
    let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
    let n: Vec<NodeIndex> = (0..12u32).map(|i| g.add_node(i)).collect();
    for i in 0..11usize {
        g.add_edge(n[i], n[i + 1], 1.0);
    }
    Topo { name: "C", desc: "12-entity sparse chain", g }
}

/// D — realistic mixed, mirroring kremory `whole_project_e2e` ingest:
/// 4 distinct-domain episodes, each producing a small entity clique with almost
/// no slug overlap between domains, plus TWO recurring bridge entities that
/// appear in 2 episodes each (the realistic minority cross-episode recurrence).
///
/// Domains (from the actual EPISODES fixture):
///   company : Acme Corporation, Jane Smith, Ohio, industrial robots        (4)
///   river   : Amazon River, Brazil, Peru, Atlantic Ocean                   (4)
///   scientist: Marie Curie, Warsaw, Physics, Chemistry, radioactivity      (5)
///   recipe  : banana bread, flour, baking soda, bananas                    (4)
/// Bridges (plausible recurrences): "Brazil" also in scientist-ish note;
///   "Physics" also in a company R&D note. Modeled as 2 bridge edges.
fn topo_d() -> Topo {
    let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
    // node ids: company 0..3, river 4..7, scientist 8..12, recipe 13..16
    let n: Vec<NodeIndex> = (0..17u32).map(|i| g.add_node(i)).collect();
    let clique = |g: &mut UnGraph<u32, f32>, ids: &[usize], w: f32| {
        for a in 0..ids.len() {
            for b in (a + 1)..ids.len() {
                g.add_edge(n[ids[a]], n[ids[b]], w);
            }
        }
    };
    clique(&mut g, &[0, 1, 2, 3], 3.0); // company
    clique(&mut g, &[4, 5, 6, 7], 3.0); // river
    clique(&mut g, &[8, 9, 10, 11, 12], 3.0); // scientist
    clique(&mut g, &[13, 14, 15, 16], 3.0); // recipe
    // realistic minority cross-episode bridges (weight 1 = single shared episode)
    g.add_edge(n[5], n[9], 1.0); // Brazil (river) ~ recurs near scientist domain
    g.add_edge(n[2], n[10], 1.0); // Ohio/Physics faint bridge
    Topo {
        name: "D",
        desc: "realistic mixed: 4 distinct-domain episode-cliques + 2 minority bridges (mirrors whole_project_e2e)",
        g,
    }
}

fn size_dist(labels: &BTreeMap<usize, usize>) -> (usize, Vec<usize>) {
    let mut comms: BTreeMap<usize, usize> = BTreeMap::new();
    for &c in labels.values() {
        *comms.entry(c).or_insert(0) += 1;
    }
    let mut sizes: Vec<usize> = comms.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a)); // descending
    (comms.len(), sizes)
}

fn report(t: &Topo) {
    let r1 = label_propagation(&t.g, 50);
    let r2 = label_propagation(&t.g, 50);
    let deterministic = r1 == r2;
    let q = modularity(&t.g, &r1);
    let (count, sizes) = size_dist(&r1);
    let n = t.g.node_count();
    let biggest = sizes.first().copied().unwrap_or(0);
    let collapsed = count == 1; // single community over >1 node = hairball collapse
    let dominant_frac = biggest as f64 / n as f64;
    println!(
        "TOPO {} — {}\n  nodes={} edges={} | Q={:.4} | communities={} sizes={:?} | biggest={}/{} ({:.0}%) | collapsed_to_1={} | deterministic={}",
        t.name, t.desc, n, t.g.edge_count(), q, count, sizes, biggest, n,
        dominant_frac * 100.0, collapsed, deterministic
    );
    // machine-readable line for the findings doc
    println!(
        "  RESULT_LINE topo={} Q={:.4} community_count={} collapsed_hairball={} deterministic={}",
        t.name, q, count, collapsed, deterministic
    );
}

fn main() {
    println!("=== ADR-066 P4 MODULARITY GO/NO-GO SPIKE (weighted label propagation) ===\n");
    let topos = [topo_a(), topo_b(), topo_c(), topo_d()];
    for t in &topos {
        report(t);
        println!();
    }

    // Also re-run B UNWEIGHTED vs WEIGHTED is identical here (all weights 1), so
    // additionally test B with a slight weight gradient to see if any weight
    // signal at all rescues the hairball: give a 4-node sub-core weight 3 edges.
    println!("=== B-variant: hairball WITH an embedded weighted sub-core (does weight rescue?) ===");
    let mut gb: UnGraph<u32, f32> = UnGraph::new_undirected();
    let nb: Vec<NodeIndex> = (0..12u32).map(|i| gb.add_node(i)).collect();
    for i in 0..12usize {
        for j in (i + 1)..12usize {
            gb.add_edge(nb[i], nb[j], 1.0);
        }
    }
    // embed a heavier 4-clique {0,1,2,3} at weight 3 (they co-occur in 3 episodes)
    for i in 0..4usize {
        for j in (i + 1)..4usize {
            // find & bump existing edge by adding a parallel heavier one is messy;
            // simplest: add extra weight edges (UnGraph allows parallel edges,
            // and the weighted tally sums them → effective weight 1+3=4).
            gb.add_edge(nb[i], nb[j], 3.0);
        }
    }
    let bt = Topo { name: "B2", desc: "hairball + embedded weight-3 sub-core", g: gb };
    report(&bt);
}

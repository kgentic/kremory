// Compile-spike (ADR-066): does petgraph 0.7 support the graph primitives needed
// for deterministic in-Rust community detection (label propagation) on MSRV 1.86?
// We do NOT rely on an external Leiden crate — label propagation is hand-rolled
// over petgraph::UnGraph so the algorithm is fully deterministic (sorted node
// iteration + sorted neighbour tie-break) for VCR reproducibility.
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::BTreeMap;

/// Deterministic synchronous label propagation. Each node adopts the most common
/// label among neighbours; ties broken by smallest label id. Nodes processed in
/// sorted index order. Halts on fixpoint or max_iters. Returns node_index -> community_id.
fn label_propagation(g: &UnGraph<u32, f32>, max_iters: usize) -> BTreeMap<usize, usize> {
    // Seed: each node its own community (index as label).
    let mut labels: BTreeMap<usize, usize> = g.node_indices().map(|n| (n.index(), n.index())).collect();
    for _ in 0..max_iters {
        let mut changed = false;
        // Deterministic: iterate node indices in ascending order.
        let ordered: Vec<NodeIndex> = {
            let mut v: Vec<NodeIndex> = g.node_indices().collect();
            v.sort_by_key(|n| n.index());
            v
        };
        for n in ordered {
            // Tally neighbour labels (BTreeMap => deterministic key order).
            let mut tally: BTreeMap<usize, usize> = BTreeMap::new();
            for e in g.edges(n) {
                let nb = if e.source() == n { e.target() } else { e.source() };
                *tally.entry(labels[&nb.index()]).or_insert(0) += 1;
            }
            if tally.is_empty() { continue; }
            // Most common; tie -> smallest label (BTreeMap iterates ascending).
            let best = tally.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|(l, _)| *l).unwrap();
            if labels[&n.index()] != best {
                labels.insert(n.index(), best);
                changed = true;
            }
        }
        if !changed { break; }
    }
    labels
}

fn main() {
    let mut g: UnGraph<u32, f32> = UnGraph::new_undirected();
    let a = g.add_node(0);
    let b = g.add_node(1);
    let c = g.add_node(2);
    let d = g.add_node(3);
    // Two triangles bridged: {a,b,c} dense, {d} loose.
    g.add_edge(a, b, 1.0);
    g.add_edge(b, c, 1.0);
    g.add_edge(a, c, 1.0);
    g.add_edge(c, d, 1.0);
    let r1 = label_propagation(&g, 20);
    let r2 = label_propagation(&g, 20);
    assert_eq!(r1, r2, "label propagation must be deterministic across runs");
    // count distinct communities
    let comms: std::collections::BTreeSet<usize> = r1.values().copied().collect();
    println!("SPIKE PASS: communities={} deterministic={}", comms.len(), r1 == r2);
}

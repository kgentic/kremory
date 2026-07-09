//! ADR-070 C1.5 compile-spike (RISK-002 / Risk #17): does the FULL clique-loop
//! borrow interaction compile when a threaded `Option<&Arc<dyn EnrichmentEventSink>>`
//! sink fires `on_merge_proposed` co-located with the post-await `emit_decision`
//! entity_refs build?
//!
//! Reproduces the exact shape of cross_episode.rs's clique loop:
//!   - `keeper` is an OWNED String (from `.cloned()`), `loser` is `&String` (into the
//!     immutably-borrowed `all_cliques`)
//!   - both are borrowed across `apply_entity_merge(graph, loser, &keeper).await`
//!   - THEN a slice `&[keeper.as_str(), loser.as_str()]` is built for emit_decision
//!   - AND `sink` (Option<&Arc<dyn Trait>>) fires `on_merge_proposed(MergeProposed{..})`
//!     with borrows of group_id/loser/keeper, co-located
//!
//! PASS (compiles) → C2 as written (thread the sink into cross_episode's params).
//! FAIL → orchestrator-fires fallback (Risk #17 contingency).
//!
//! Run: `rustc --edition 2021 spike/adr070_c15_sink_borrow.rs -o /tmp/c15 && echo PASS`

use std::sync::Arc;

// Mirrors kremory's `MergeProposed<'a>` args-as-object event (borrowed fields).
#[derive(Debug, Clone, Copy)]
struct MergeProposed<'a> {
    group_id: &'a str,
    loser: &'a str,
    keeper: &'a str,
    dry_run: bool,
}

// Mirrors the EnrichmentEventSink method under test (default no-op, dyn-safe).
trait EnrichmentEventSink {
    fn on_merge_proposed(&self, _event: MergeProposed<'_>) {}
}

// Mirrors emit_decision's DecisionRecord entity_refs shape.
struct DecisionRecord<'a> {
    entity_refs: &'a [&'a str],
}
fn emit_decision(_r: DecisionRecord<'_>) {}

// Mirrors apply_entity_merge: async, borrows graph + two &str, returns Result.
async fn apply_entity_merge(_graph: &Graph, _loser: &str, _keeper: &str) -> Result<(), ()> {
    Ok(())
}

struct Graph;
struct TestSink;
impl EnrichmentEventSink for TestSink {}

/// The clique-loop shape with dry_run gate + emit_decision + co-located sink fire.
async fn cross_episode(
    graph: &Graph,
    group_id: &str,
    dry_run: bool,
    sink: Option<&Arc<dyn EnrichmentEventSink>>,
) -> Result<usize, ()> {
    // Stand-in for `all_cliques: Vec<BTreeSet<String>>` — immutably borrowed by the loop.
    let all_cliques: Vec<Vec<String>> = vec![vec!["a".into(), "b".into(), "c".into()]];
    let mut count = 0usize;

    for clique in &all_cliques {
        // keeper: OWNED String (mirrors `.iter().min().cloned()`).
        let keeper = clique.iter().min().cloned().unwrap_or_default();
        // losers: &String into the immutably-borrowed clique.
        let losers: Vec<&String> = clique.iter().filter(|m| **m != keeper).collect();
        for loser in losers {
            if !dry_run {
                apply_entity_merge(graph, loser, &keeper).await?;
            }
            count += 1;
            // Post-await entity_refs slice build (the RISK-002 concern).
            emit_decision(DecisionRecord {
                entity_refs: &[keeper.as_str(), loser.as_str()],
            });
            // Co-located sink fire with borrowed-struct arg.
            if let Some(s) = sink {
                s.on_merge_proposed(MergeProposed {
                    group_id,
                    loser,
                    keeper: &keeper,
                    dry_run,
                });
            }
        }
    }
    Ok(count)
}

fn main() {
    // Prove it composes with a real Arc<dyn Trait> at a call site.
    let sink: Arc<dyn EnrichmentEventSink> = Arc::new(TestSink);
    let fut = cross_episode(&Graph, "g1", true, Some(&sink));
    // Poll once via a trivial executor to satisfy the async fn (no tokio dep in spike).
    let _ = fut;
    println!("compile-spike constructed cross_episode future OK");
}

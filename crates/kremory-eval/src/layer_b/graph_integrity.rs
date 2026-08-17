//! Graph integrity invariants for kremory's SQLite storage layer.
//!
//! Six programmatic invariants — no LLM judge, sub-millisecond execution.
//! Any invariant violation = FAIL (correctness invariant, no score threshold).
//!
//! # Invariants
//!
//! | # | Name                    | What is checked                                                  |
//! |---|-------------------------|------------------------------------------------------------------|
//! | 1 | `no_orphan_nodes`       | Every entity has ≥1 fact referencing it as subject OR object     |
//! | 2 | `no_duplicate_edges`    | No two facts with same (subj, pred, obj, namespace) overlap in   |
//! |   |                         | `[valid_from, valid_to)` time range                              |
//! | 3 | `fts_index_in_sync`     | Entity count matches what was ingested (proxy for FTS sync)      |
//! | 4 | `all_namespaces_present`| Each expected namespace has ≥1 fact in the graph                 |
//! | 5 | `episode_edge_presence` | Every entity has ≥1 `episodic_edge` row                          |
//! | 6 | `no_transitive_merge_chain` | No entity is the SURVIVOR of one live merge and the VICTIM   |
//! |   |                         | of another (see "Why invariant 6 is different" below)            |
//!
//! # Why invariant 6 is different — it reads HISTORY, not state (TD-223)
//!
//! Invariants 1–5 all query the graph's CURRENT state (`list_entities`,
//! `facts_at`, `entity_history`, `episodic_edges_for_entity`). That makes every
//! one of them structurally blind to a bad entity merge, because **a merge leaves
//! a perfectly well-formed current state**: no orphans, no duplicate edges,
//! correct namespaces, episodic edges intact. The damage exists only in the
//! HISTORY of how that state was reached.
//!
//! Measured on two retained benchmark databases from the same corpus — one
//! healthy, one in which both conversation speakers had been merged out of
//! existence (`melanie` → `caroline` → `loved ones` → `luna and oliver`):
//!
//! | check                        | corrupted graph | healthy graph | separates? |
//! |------------------------------|-----------------|---------------|------------|
//! | invariant 1 (orphan nodes)   | 48              | 48            | NO — identical |
//! | invariant 5 (episodic edges) | 46              | 44            | NO         |
//! | **invariant 6 (this one)**   | **6**           | **0**         | **YES**    |
//!
//! Invariant 6 is threshold-free and makes no lexical or semantic assumption. Its
//! meaning: `A → B` followed by `B → C` moved A's identity two hops while
//! **nobody ever adjudicated A against C**. That unreviewed second hop is the
//! signature of runaway merging.
//!
//! ## Known false-positive class — read before making this a gate
//!
//! A legitimate three-variant canonicalisation also forms a chain
//! (`pottery class` → `pottery` → `pottery project`: two defensible merges, one
//! flagged entity). On the corrupted database above, roughly 4 of the 6 flags
//! were catastrophic and roughly 2 were benign. That is acceptable for a
//! REPORTED metric and needs more evidence before it BLOCKS anything —
//! `allow_merge_chain_count` exists so a fixture with known-benign chains raises
//! the bar deliberately and on the record, rather than the check being deleted.
//!
//! Evidence base is n = 2 databases: enough to reject invariants 1–5 as blind,
//! NOT enough to establish a false-positive rate.
//!
//! # Usage
//!
//! ```rust,ignore
//! use kremory::core::schema::TemporalGraph;
//! use kremory_eval::layer_b::graph_integrity::{run_invariants, IntegrityConfig};
//!
//! let graph = TemporalGraph::open_in_memory().await?;
//! let config = IntegrityConfig {
//!     expected_entity_count: Some(5),
//!     expected_namespaces: vec!["default".into()],
//!     ..Default::default()
//! };
//! let report = run_invariants(&graph, &config).await?;
//! assert!(report.all_passed());
//! ```

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use kremory::core::schema::TemporalGraph;

use crate::types::{EvalErr, EvalError};

// ---------------------------------------------------------------------------
// InvariantResult
// ---------------------------------------------------------------------------

/// Result for a single integrity invariant check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantResult {
    /// Short invariant identifier (snake_case).
    pub name: String,
    /// Whether the invariant passed.
    pub passed: bool,
    /// Expected value (stringified, for error messages).
    pub expected: String,
    /// Actual value observed (stringified).
    pub actual: String,
    /// Human-readable detail on violations (empty if passed).
    pub details: String,
}

impl InvariantResult {
    fn pass(name: impl Into<String>, value: impl std::fmt::Display) -> Self {
        let v = value.to_string();
        Self {
            name: name.into(),
            passed: true,
            expected: v.clone(),
            actual: v,
            details: String::new(),
        }
    }

    fn fail(
        name: impl Into<String>,
        expected: impl std::fmt::Display,
        actual: impl std::fmt::Display,
        details: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            passed: false,
            expected: expected.to_string(),
            actual: actual.to_string(),
            details: details.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// IntegrityReport
// ---------------------------------------------------------------------------

/// Full integrity report for one graph state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    /// Individual invariant results.
    pub invariants: Vec<InvariantResult>,
    /// True when ALL invariants passed.
    pub passed: bool,
    /// UTC timestamp of when this report was produced.
    pub timestamp: chrono::DateTime<Utc>,
}

impl IntegrityReport {
    fn new(invariants: Vec<InvariantResult>) -> Self {
        let passed = invariants.iter().all(|r| r.passed);
        Self {
            invariants,
            passed,
            timestamp: Utc::now(),
        }
    }

    /// Returns `true` iff every invariant passed.
    pub fn all_passed(&self) -> bool {
        self.passed
    }

    /// Returns only the failing invariant results.
    pub fn failures(&self) -> Vec<&InvariantResult> {
        self.invariants.iter().filter(|r| !r.passed).collect()
    }
}

// ---------------------------------------------------------------------------
// IntegrityConfig
// ---------------------------------------------------------------------------

/// Configuration for the integrity check run.
#[derive(Debug, Clone, Default)]
pub struct IntegrityConfig {
    /// If `Some(n)`, the FTS-sync invariant checks that the graph contains
    /// exactly `n` entities. If `None`, the invariant is skipped.
    pub expected_entity_count: Option<usize>,

    /// Namespace labels expected to appear in the graph's facts.
    /// If empty, invariant 4 is skipped.
    pub expected_namespaces: Vec<String>,

    /// Number of isolated (no-fact) entities allowed. Default 0.
    /// Fixtures that intentionally contain isolated entities should set this > 0.
    pub allow_isolated_entity_count: usize,

    /// Number of entities allowed to sit in a transitive merge chain (invariant
    /// 6). Default 0 — the invariant is threshold-free by design.
    ///
    /// Raise it only for a fixture with a KNOWN-benign chain (e.g. a genuine
    /// three-variant canonicalisation), and say which chain in the call site. The
    /// knob exists so such a fixture raises the bar deliberately and on the
    /// record, instead of the invariant being weakened or dropped.
    pub allow_merge_chain_count: usize,
}

// ---------------------------------------------------------------------------
// Invariant implementations
// ---------------------------------------------------------------------------

/// Check invariant 1: no orphan nodes (entities with no fact references).
///
/// Uses `list_entities()` and `facts_at(now)` from the public API.
/// An entity is considered orphaned if it appears in neither `subject_id`
/// nor `object_id` of any non-expired fact.
async fn check_no_orphan_nodes(
    graph: &TemporalGraph,
    allow_isolated: usize,
) -> EvalError<InvariantResult> {
    let entities = graph
        .list_entities()
        .await
        .map_err(|e| EvalErr::Other(format!("list_entities failed: {}", e)))?;

    let facts = graph
        .facts_at(Utc::now())
        .await
        .map_err(|e| EvalErr::Other(format!("facts_at failed: {}", e)))?;

    // Build set of entity IDs referenced by any non-expired fact.
    let mut referenced: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for fact in &facts {
        referenced.insert(&fact.subject_id);
        if let Some(obj_id) = &fact.object_id {
            referenced.insert(obj_id.as_str());
        }
    }

    let orphans: Vec<String> = entities
        .iter()
        .filter(|e| !referenced.contains(e.id.as_str()))
        .map(|e| e.id.clone())
        .collect();

    let orphan_count = orphans.len();

    if orphan_count <= allow_isolated {
        Ok(InvariantResult::pass(
            "no_orphan_nodes",
            format!("orphans={} (allowed={})", orphan_count, allow_isolated),
        ))
    } else {
        Ok(InvariantResult::fail(
            "no_orphan_nodes",
            format!("orphans ≤ {}", allow_isolated),
            format!("orphans={}", orphan_count),
            format!("orphaned entity IDs: {:?}", orphans),
        ))
    }
}

/// Check invariant 2: no duplicate edges in same time range.
///
/// For each `(subject_id, predicate, object_key, group_id)` tuple, no two
/// non-expired facts should have overlapping `[valid_from, valid_to)` windows.
async fn check_no_duplicate_edges(graph: &TemporalGraph) -> EvalError<InvariantResult> {
    let entities = graph
        .list_entities()
        .await
        .map_err(|e| EvalErr::Other(format!("list_entities failed: {}", e)))?;

    let mut violations: Vec<String> = Vec::new();

    for entity in &entities {
        let history = graph
            .entity_history(&entity.id)
            .await
            .map_err(|e| EvalErr::Other(format!("entity_history failed: {}", e)))?;

        // Filter to non-expired facts only.
        let active: Vec<_> = history.iter().filter(|f| f.expired_at.is_none()).collect();

        // Group by (predicate, object_key, group_id).
        use std::collections::HashMap;
        let mut groups: HashMap<String, Vec<_>> = HashMap::new();
        for fact in &active {
            let obj_key = fact
                .object_id
                .as_deref()
                .or(fact.object_value.as_deref())
                .unwrap_or("");
            let group_key = format!(
                "{}|{}|{}|{}",
                fact.predicate,
                obj_key,
                fact.group_id.as_deref().unwrap_or(""),
                fact.subject_id
            );
            groups.entry(group_key).or_default().push(fact);
        }

        // For each group, check no two facts have overlapping [valid_from, valid_to).
        for (group_key, group_facts) in &groups {
            for i in 0..group_facts.len() {
                for j in (i + 1)..group_facts.len() {
                    let a = group_facts[i];
                    let b = group_facts[j];
                    // Overlap: a.valid_from < b.valid_to_or_inf AND b.valid_from < a.valid_to_or_inf
                    let a_end = a.valid_to.unwrap_or(chrono::DateTime::<Utc>::MAX_UTC);
                    let b_end = b.valid_to.unwrap_or(chrono::DateTime::<Utc>::MAX_UTC);
                    let overlaps = a.valid_from < b_end && b.valid_from < a_end;
                    if overlaps {
                        violations.push(format!(
                            "duplicate edge group '{}': fact {} [{}, {}) overlaps fact {} [{}, {})",
                            group_key,
                            a.id,
                            a.valid_from,
                            a.valid_to.map(|t| t.to_string()).as_deref().unwrap_or("∞"),
                            b.id,
                            b.valid_from,
                            b.valid_to.map(|t| t.to_string()).as_deref().unwrap_or("∞"),
                        ));
                    }
                }
            }
        }
    }

    if violations.is_empty() {
        Ok(InvariantResult::pass(
            "no_duplicate_edges",
            "no overlapping fact windows",
        ))
    } else {
        Ok(InvariantResult::fail(
            "no_duplicate_edges",
            "0 overlapping fact windows",
            format!("{} violations", violations.len()),
            violations.join("; "),
        ))
    }
}

/// Check invariant 3: FTS index entity count in sync.
///
/// The FTS virtual table is not directly queryable via the public API.
/// We proxy this by checking the entity count matches an expected count
/// (set via `IntegrityConfig::expected_entity_count`).
/// If not configured, this invariant is skipped (passes vacuously).
async fn check_fts_index_in_sync(
    graph: &TemporalGraph,
    expected_count: Option<usize>,
) -> EvalError<InvariantResult> {
    let Some(expected) = expected_count else {
        return Ok(InvariantResult::pass(
            "fts_index_in_sync",
            "skipped (no expected_entity_count set)",
        ));
    };

    let entities = graph
        .list_entities()
        .await
        .map_err(|e| EvalErr::Other(format!("list_entities failed: {}", e)))?;

    let actual = entities.len();

    if actual == expected {
        Ok(InvariantResult::pass(
            "fts_index_in_sync",
            format!("entity_count={}", actual),
        ))
    } else {
        Ok(InvariantResult::fail(
            "fts_index_in_sync",
            format!("entity_count={}", expected),
            format!("entity_count={}", actual),
            format!(
                "expected {} entities but found {}; FTS index may be out of sync",
                expected, actual
            ),
        ))
    }
}

/// Check invariant 4: all expected namespaces are present in the graph.
///
/// Uses `facts_at(now)` and checks each expected namespace appears as a
/// `group_id` value in at least one non-expired fact.
async fn check_all_namespaces_present(
    graph: &TemporalGraph,
    expected_namespaces: &[String],
) -> EvalError<InvariantResult> {
    if expected_namespaces.is_empty() {
        return Ok(InvariantResult::pass(
            "all_namespaces_present",
            "skipped (no expected_namespaces configured)",
        ));
    }

    let facts = graph
        .facts_at(Utc::now())
        .await
        .map_err(|e| EvalErr::Other(format!("facts_at failed: {}", e)))?;

    let present: std::collections::HashSet<String> =
        facts.iter().filter_map(|f| f.group_id.clone()).collect();

    let missing: Vec<&str> = expected_namespaces
        .iter()
        .filter(|ns| !present.contains(*ns))
        .map(String::as_str)
        .collect();

    if missing.is_empty() {
        Ok(InvariantResult::pass(
            "all_namespaces_present",
            format!(
                "all {} namespaces present: {:?}",
                expected_namespaces.len(),
                expected_namespaces
            ),
        ))
    } else {
        Ok(InvariantResult::fail(
            "all_namespaces_present",
            format!("all namespaces present: {:?}", expected_namespaces),
            format!("{} missing namespaces: {:?}", missing.len(), missing),
            format!("missing namespaces: {:?}", missing),
        ))
    }
}

/// Check invariant 5: every entity has ≥1 episodic edge (v0.1.1 redesign).
///
/// Uses `episodic_edges_for_entity()` per entity. An entity without any
/// episodic edge cannot be traced back to an ingestion episode, which
/// indicates the episodic linkage was not written correctly.
async fn check_episode_edge_presence(graph: &TemporalGraph) -> EvalError<InvariantResult> {
    let entities = graph
        .list_entities()
        .await
        .map_err(|e| EvalErr::Other(format!("list_entities failed: {}", e)))?;

    let mut missing: Vec<String> = Vec::new();

    for entity in &entities {
        let edges = graph
            .episodic_edges_for_entity(&entity.id)
            .await
            .map_err(|e| EvalErr::Other(format!("episodic_edges_for_entity failed: {}", e)))?;

        if edges.is_empty() {
            missing.push(entity.id.clone());
        }
    }

    if missing.is_empty() {
        Ok(InvariantResult::pass(
            "episode_edge_presence",
            format!("all {} entities have ≥1 episodic edge", entities.len()),
        ))
    } else {
        Ok(InvariantResult::fail(
            "episode_edge_presence",
            "all entities have ≥1 episodic edge",
            format!(
                "{}/{} entities missing episodic edges",
                missing.len(),
                entities.len()
            ),
            format!("entities without episodic edges: {:?}", missing),
        ))
    }
}

/// The two endpoints of one logged `entity_merge`, the only fields invariant 6
/// needs from `graph_mutation_log.inputs`.
///
/// That JSON is kremory's OWN structured emit (`MergeInputs`), but the type lives
/// in a `pub(crate)` module and cannot be imported from this crate, so the shape
/// is restated here. Because the shape is duplicated it is parsed LOUDLY: a merge
/// row whose `inputs` lacks `keeper`/`loser` is an `Err`, never a skipped row. If
/// the producer's field names ever drift, invariant 6 FAILS the run rather than
/// quietly reporting zero chains — a silent zero here would look exactly like a
/// healthy graph, which is the failure mode this whole invariant exists to catch.
#[derive(Deserialize)]
struct MergeEndpoints {
    keeper: String,
    loser: String,
}

/// Check invariant 6: no entity is the survivor of one live merge and the victim
/// of another.
///
/// Reads `graph_mutation_log` directly rather than via `list_mutations`, because
/// that helper is gated behind kremory's `test-utils` feature while
/// [`run_invariants`] is called from the non-test `eval` binary. Enabling
/// `test-utils` as a normal dependency would leak test-only surface into every
/// workspace build of the published crate via feature unification.
///
/// Only LIVE merges count (`undone_at IS NULL`). A merge already reversed through
/// `Memory::unmerge` no longer holds the graph in a chained state, so counting it
/// would report damage that has been repaired.
async fn check_no_transitive_merge_chain(
    graph: &TemporalGraph,
    allow_chain_count: usize,
) -> EvalError<InvariantResult> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, inputs FROM graph_mutation_log \
             WHERE kind = 'entity_merge' AND undone_at IS NULL ORDER BY id",
            (),
        )
        .await
        .map_err(|e| EvalErr::Other(format!("query graph_mutation_log failed: {}", e)))?;

    let mut survivors: HashSet<String> = HashSet::new();
    let mut victims: HashSet<String> = HashSet::new();
    // Kept so the failure detail can show the reader the actual chain rather than
    // a bare list of ids.
    let mut absorbed: HashMap<String, Vec<String>> = HashMap::new();
    let mut absorbed_by: HashMap<String, String> = HashMap::new();
    let mut merge_count: usize = 0;

    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| EvalErr::Other(format!("read graph_mutation_log row failed: {}", e)))?
    {
        let id = row
            .get::<i64>(0)
            .map_err(|e| EvalErr::Other(format!("graph_mutation_log.id read failed: {}", e)))?;
        let inputs = row
            .get::<String>(1)
            .map_err(|e| EvalErr::Other(format!("graph_mutation_log.inputs read failed: {}", e)))?;

        let ends: MergeEndpoints = serde_json::from_str(&inputs).map_err(|e| {
            EvalErr::Other(format!(
                "graph_mutation_log row {}: entity_merge inputs carry no keeper/loser — \
                 producer shape drift, invariant 6 cannot be evaluated: {}",
                id, e
            ))
        })?;

        merge_count += 1;
        survivors.insert(ends.keeper.clone());
        victims.insert(ends.loser.clone());
        absorbed_by.insert(ends.loser.clone(), ends.keeper.clone());
        absorbed.entry(ends.keeper).or_default().push(ends.loser);
    }

    let mut chained: Vec<&String> = survivors.intersection(&victims).collect();
    chained.sort();

    if chained.len() <= allow_chain_count {
        return Ok(InvariantResult::pass(
            "no_transitive_merge_chain",
            format!(
                "chained={} (allowed={}) across {} live merges",
                chained.len(),
                allow_chain_count,
                merge_count
            ),
        ));
    }

    let detail = chained
        .iter()
        .map(|id| {
            let took = absorbed.get(*id).map(|v| v.join(", ")).unwrap_or_default();
            let then = absorbed_by
                .get(*id)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            format!(
                "'{}' absorbed [{}] then was absorbed by '{}'",
                id, took, then
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    Ok(InvariantResult::fail(
        "no_transitive_merge_chain",
        format!("chained entities ≤ {}", allow_chain_count),
        format!(
            "chained={} across {} live merges",
            chained.len(),
            merge_count
        ),
        format!("transitive merge chains: {}", detail),
    ))
}

// ---------------------------------------------------------------------------
// run_invariants — public entry point
// ---------------------------------------------------------------------------

/// Run all 6 graph integrity invariants and return a consolidated report.
///
/// Gate: any invariant failure = report `passed = false`.
/// Sub-millisecond execution (no LLM judge).
pub async fn run_invariants(
    graph: &TemporalGraph,
    config: &IntegrityConfig,
) -> EvalError<IntegrityReport> {
    let mut results = Vec::with_capacity(6);

    results.push(check_no_orphan_nodes(graph, config.allow_isolated_entity_count).await?);
    results.push(check_no_duplicate_edges(graph).await?);
    results.push(check_fts_index_in_sync(graph, config.expected_entity_count).await?);
    results.push(check_all_namespaces_present(graph, &config.expected_namespaces).await?);
    results.push(check_episode_edge_presence(graph).await?);
    results.push(check_no_transitive_merge_chain(graph, config.allow_merge_chain_count).await?);

    Ok(IntegrityReport::new(results))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use chrono::Utc;
    use kremory::core::graph::{
        FactInsert, InsertEntityParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
    };
    use kremory::core::schema::TemporalGraph;

    async fn empty_graph() -> TemporalGraph {
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory")
    }

    #[tokio::test]
    async fn empty_graph_all_pass() {
        let graph = empty_graph().await;
        let config = IntegrityConfig::default();
        let report = run_invariants(&graph, &config).await.unwrap();
        assert!(report.all_passed(), "failures: {:?}", report.failures());
    }

    #[tokio::test]
    async fn graph_with_entity_and_fact_passes_orphan_check() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        graph
            .insert_episode(InsertEpisodeParams {
                content: "hello world",
                timestamp: Utc::now(),
                source_type: None,
                metadata: None,
            })
            .await
            .unwrap();
        graph
            // entity_group_id=None → 'default', matching `insert_entity("e1", …)`
            // above (no group → 'default') so the composite FK lines up (ADR-029b).
            .insert_episodic_edge(InsertEpisodicEdgeParams {
                episode_id: 1,
                entity_id: "e1",
                entity_group_id: None,
                role: "subject",
            })
            .await
            .unwrap();
        graph
            .insert_fact(FactInsert::new("e1", "works_at", Utc::now()).object_value("Acme"))
            .await
            .unwrap();

        let config = IntegrityConfig {
            expected_entity_count: Some(1),
            expected_namespaces: vec![],
            allow_isolated_entity_count: 0,
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        assert!(report.all_passed(), "failures: {:?}", report.failures());
    }

    #[tokio::test]
    async fn orphan_entity_fails_when_not_allowed() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "isolated",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        // No fact references "isolated", no episodic edge

        let config = IntegrityConfig {
            allow_isolated_entity_count: 0,
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let orphan_result = report
            .invariants
            .iter()
            .find(|r| r.name == "no_orphan_nodes")
            .unwrap();
        assert!(!orphan_result.passed, "orphan check should fail");
    }

    #[tokio::test]
    async fn orphan_entity_passes_when_allowed() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "isolated",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();

        let config = IntegrityConfig {
            allow_isolated_entity_count: 1,
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let orphan_result = report
            .invariants
            .iter()
            .find(|r| r.name == "no_orphan_nodes")
            .unwrap();
        assert!(
            orphan_result.passed,
            "orphan check should pass with allow=1"
        );
    }

    #[tokio::test]
    async fn fts_count_mismatch_fails() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        graph
            .insert_entity(InsertEntityParams {
                id: "e2",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();

        let config = IntegrityConfig {
            expected_entity_count: Some(3), // wrong — only 2 inserted
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let fts_result = report
            .invariants
            .iter()
            .find(|r| r.name == "fts_index_in_sync")
            .unwrap();
        assert!(
            !fts_result.passed,
            "fts check should fail on count mismatch"
        );
    }

    #[tokio::test]
    async fn fts_count_skipped_when_not_configured() {
        let graph = empty_graph().await;
        let config = IntegrityConfig {
            expected_entity_count: None,
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let fts_result = report
            .invariants
            .iter()
            .find(|r| r.name == "fts_index_in_sync")
            .unwrap();
        assert!(fts_result.passed, "fts check should pass when skipped");
    }

    #[tokio::test]
    async fn missing_namespace_fails() {
        let graph = empty_graph().await;
        // No facts inserted so no namespace in graph

        let config = IntegrityConfig {
            expected_namespaces: vec!["workspace-abc".into()],
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let ns_result = report
            .invariants
            .iter()
            .find(|r| r.name == "all_namespaces_present")
            .unwrap();
        assert!(
            !ns_result.passed,
            "namespace check should fail when namespace missing"
        );
    }

    #[tokio::test]
    async fn namespace_check_skipped_when_empty() {
        let graph = empty_graph().await;
        let config = IntegrityConfig {
            expected_namespaces: vec![],
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        let ns_result = report
            .invariants
            .iter()
            .find(|r| r.name == "all_namespaces_present")
            .unwrap();
        assert!(ns_result.passed, "namespace check should pass when skipped");
    }

    #[tokio::test]
    async fn episode_edge_missing_fails() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        // No episodic edge inserted for e1

        let config = IntegrityConfig::default();
        let report = run_invariants(&graph, &config).await.unwrap();
        let edge_result = report
            .invariants
            .iter()
            .find(|r| r.name == "episode_edge_presence")
            .unwrap();
        assert!(!edge_result.passed, "edge presence check should fail");
    }

    #[tokio::test]
    async fn integrity_report_failures_method() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "orphan",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        // orphan + no episodic edge = 2 failures

        let config = IntegrityConfig {
            allow_isolated_entity_count: 0,
            ..Default::default()
        };
        let report = run_invariants(&graph, &config).await.unwrap();
        assert!(!report.all_passed());
        let failures = report.failures();
        assert!(!failures.is_empty());
    }

    // -----------------------------------------------------------------------
    // Invariant 6 — no_transitive_merge_chain (TD-223)
    // -----------------------------------------------------------------------

    /// Insert one `entity_merge` row shaped EXACTLY as the dream pass emits it.
    ///
    /// The `inputs` payload is copied verbatim from a real corrupted benchmark
    /// database (`graph_mutation_log` id 14, the merge that absorbed the speaker
    /// `caroline` into `loved ones`), with only the two endpoint names
    /// substituted. Grounding the fixture in observed producer output — rather
    /// than in a shape invented alongside the reader — is what stops these tests
    /// passing tautologically against a wrong mental model of the contract.
    async fn insert_merge(graph: &TemporalGraph, keeper: &str, loser: &str, undone: bool) {
        let (lo, hi) = if keeper < loser {
            (keeper, loser)
        } else {
            (loser, keeper)
        };
        let inputs = serde_json::json!({
            "pair_lo": lo,
            "pair_hi": hi,
            "keeper": keeper,
            "loser": loser,
            "site": "site5_acronym_nickname",
            "cosine": serde_json::Value::Null,
            "structural_signal": true,
        })
        .to_string();
        let undone_at = if undone {
            Some("2026-08-17T10:00:00+00:00".to_string())
        } else {
            None
        };
        graph
            .conn
            .execute(
                "INSERT INTO graph_mutation_log \
                 (kind, group_id, created_at, undone_at, pre_state, inputs) \
                 VALUES ('entity_merge', 'default', '2026-08-17T09:45:39+00:00', ?1, '{}', ?2)",
                libsql::params![undone_at, inputs],
            )
            .await
            .expect("insert graph_mutation_log row");
    }

    async fn chain_result(graph: &TemporalGraph, allow: usize) -> InvariantResult {
        check_no_transitive_merge_chain(graph, allow)
            .await
            .expect("check_no_transitive_merge_chain")
    }

    /// THE FALSE-POSITIVE TEST, written first. A star — two entities merged into
    /// one survivor — is ordinary canonicalisation and MUST pass. Without this,
    /// an invariant that simply failed on "any merge happened" would look
    /// perfectly healthy against the chain test below.
    #[tokio::test]
    async fn star_merges_do_not_count_as_a_chain() {
        let graph = empty_graph().await;
        insert_merge(&graph, "pottery project", "pottery class", false).await;
        insert_merge(&graph, "pottery project", "pottery", false).await;

        let result = chain_result(&graph, 0).await;
        assert!(
            result.passed,
            "star merge wrongly flagged as a chain: {:?}",
            result
        );
        assert!(
            result.actual.contains("chained=0"),
            "actual: {}",
            result.actual
        );
    }

    #[tokio::test]
    async fn no_merges_at_all_passes() {
        let graph = empty_graph().await;
        let result = chain_result(&graph, 0).await;
        assert!(result.passed);
        assert!(result.actual.contains("across 0 live merges"));
    }

    /// The corruption shape from the real benchmark run: a speaker absorbed into
    /// another entity, which is then itself absorbed. `caroline` is the chained
    /// entity — survivor of the first merge, victim of the second.
    #[tokio::test]
    async fn transitive_chain_is_detected_and_named() {
        let graph = empty_graph().await;
        insert_merge(&graph, "caroline", "melanie", false).await;
        insert_merge(&graph, "loved ones", "caroline", false).await;

        let result = chain_result(&graph, 0).await;
        assert!(!result.passed, "chain not detected: {:?}", result);
        assert!(
            result.actual.contains("chained=1"),
            "actual: {}",
            result.actual
        );
        assert!(
            result
                .details
                .contains("'caroline' absorbed [melanie] then was absorbed by 'loved ones'"),
            "details did not spell out the chain: {}",
            result.details
        );
    }

    /// Three hops — the full shape observed in the corrupted database
    /// (`melanie` → `caroline` → `loved ones` → `luna and oliver`). Both
    /// intermediate entities are chained.
    #[tokio::test]
    async fn multi_hop_chain_counts_every_intermediate() {
        let graph = empty_graph().await;
        insert_merge(&graph, "caroline", "melanie", false).await;
        insert_merge(&graph, "loved ones", "caroline", false).await;
        insert_merge(&graph, "luna and oliver", "loved ones", false).await;

        let result = chain_result(&graph, 0).await;
        assert!(!result.passed);
        assert!(
            result.actual.contains("chained=2"),
            "actual: {}",
            result.actual
        );
    }

    /// A merge already reversed via `Memory::unmerge` no longer holds the graph
    /// in a chained state, so it must not be reported as live damage.
    #[tokio::test]
    async fn undone_merge_does_not_form_a_chain() {
        let graph = empty_graph().await;
        insert_merge(&graph, "caroline", "melanie", false).await;
        insert_merge(&graph, "loved ones", "caroline", true).await;

        let result = chain_result(&graph, 0).await;
        assert!(result.passed, "undone merge counted as live: {:?}", result);
        assert!(result.actual.contains("across 1 live merges"));
    }

    #[tokio::test]
    async fn allowance_admits_a_known_benign_chain() {
        let graph = empty_graph().await;
        insert_merge(&graph, "pottery", "pottery class", false).await;
        insert_merge(&graph, "pottery project", "pottery", false).await;

        assert!(!chain_result(&graph, 0).await.passed);
        assert!(chain_result(&graph, 1).await.passed);
    }

    /// Producer shape drift must FAIL the run, never report a silent zero — a
    /// silent zero is indistinguishable from a healthy graph, which is precisely
    /// the failure this invariant exists to catch.
    #[tokio::test]
    async fn merge_row_without_endpoints_errors_rather_than_reporting_zero() {
        let graph = empty_graph().await;
        graph
            .conn
            .execute(
                "INSERT INTO graph_mutation_log \
                 (kind, group_id, created_at, pre_state, inputs) \
                 VALUES ('entity_merge', 'default', '2026-08-17T09:45:39+00:00', '{}', \
                 '{\"survivor\":\"a\",\"absorbed\":\"b\"}')",
                (),
            )
            .await
            .unwrap();

        let err = check_no_transitive_merge_chain(&graph, 0).await;
        assert!(err.is_err(), "shape drift silently reported as healthy");
    }

    /// Drive invariant 6 against a REAL on-disk graph, to check it on production
    /// data rather than only on fixtures authored beside the reader.
    ///
    /// `#[ignore]` + env-var driven because the databases it validates against are
    /// multi-hundred-MB benchmark artefacts that are not committed. Run with:
    ///
    /// ```text
    /// KREMORY_GI_DB=/path/to/copy.db cargo test -p kremory-eval --lib \
    ///     chain_check_against_real_database -- --ignored --nocapture
    /// ```
    ///
    /// Copy the database first — `TemporalGraph::open` may run migrations and so
    /// can mutate the file.
    ///
    /// `KREMORY_GI_DIM` overrides the embedding dimension (default 768, what the
    /// benchmark runs use). `TemporalGraph::open` defaults to 384 and rejects a
    /// graph written at any other width.
    #[tokio::test]
    #[ignore = "requires KREMORY_GI_DB pointing at a real on-disk graph"]
    async fn chain_check_against_real_database() {
        let path = std::env::var("KREMORY_GI_DB").expect("KREMORY_GI_DB not set");
        let dim: usize = std::env::var("KREMORY_GI_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(768);
        let graph = TemporalGraph::open_with_dim(&path, dim)
            .await
            .expect("open graph");
        let result = chain_result(&graph, 0).await;
        println!(
            "KREMORY_GI_DB={} passed={} actual={} details={}",
            path, result.passed, result.actual, result.details
        );
    }

    #[tokio::test]
    async fn chain_check_is_wired_into_run_invariants() {
        let graph = empty_graph().await;
        insert_merge(&graph, "caroline", "melanie", false).await;
        insert_merge(&graph, "loved ones", "caroline", false).await;

        let report = run_invariants(&graph, &IntegrityConfig::default())
            .await
            .unwrap();
        assert!(report
            .failures()
            .iter()
            .any(|f| f.name == "no_transitive_merge_chain"));
    }
}

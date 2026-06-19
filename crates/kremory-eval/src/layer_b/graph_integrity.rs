//! Graph integrity invariants for kremory's SQLite storage layer.
//!
//! Five programmatic invariants — no LLM judge, sub-millisecond execution.
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
//!     allow_isolated_entity_count: 0,
//! };
//! let report = run_invariants(&graph, &config).await?;
//! assert!(report.all_passed());
//! ```

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

// ---------------------------------------------------------------------------
// run_invariants — public entry point
// ---------------------------------------------------------------------------

/// Run all 5 graph integrity invariants and return a consolidated report.
///
/// Gate: any invariant failure = report `passed = false`.
/// Sub-millisecond execution (no LLM judge).
pub async fn run_invariants(
    graph: &TemporalGraph,
    config: &IntegrityConfig,
) -> EvalError<IntegrityReport> {
    let mut results = Vec::with_capacity(5);

    results.push(check_no_orphan_nodes(graph, config.allow_isolated_entity_count).await?);
    results.push(check_no_duplicate_edges(graph).await?);
    results.push(check_fts_index_in_sync(graph, config.expected_entity_count).await?);
    results.push(check_all_namespaces_present(graph, &config.expected_namespaces).await?);
    results.push(check_episode_edge_presence(graph).await?);

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
}

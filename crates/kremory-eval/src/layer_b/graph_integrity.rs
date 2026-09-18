//! Graph integrity invariants for kremory's SQLite storage layer.
//!
//! Seven programmatic invariants — no LLM judge, sub-millisecond execution.
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
//! |   |                         | of another (see "Why invariant 6 is different" below). Also      |
//! |   |                         | REPORTS fan-ins (TD-256) — N entities absorbed into one keeper   |
//! |   |                         | that is never itself absorbed. A fan-in forms no chain, so this  |
//! |   |                         | invariant was blind to it; on the full corpus that is 4 chains   |
//! |   |                         | vs 20 fan-ins. Reported in the value string, NEVER failed.       |
//! | 7 | `world_time_grounding`  | `valid_from` shows real episode-anchored date resolution, not    |
//! |   |                         | ingest wall-clock (see below — breaks convention)                |
//!
//! # Why invariant 6 is different — it reads HISTORY, not state
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
//! # Why invariant 7 deliberately breaks `empty_graph_all_pass`
//!
//! `world_time_grounding`'s G-POP guard fails any run with fewer than 200
//! episode-linked facts — INCLUDING an empty graph. Every other invariant in
//! this file passes vacuously on an empty graph (nothing to violate). This one
//! doesn't, on purpose: a graph too small/empty to say anything is exactly the
//! shape that makes a metric report a perfect score while measuring nothing.
//! `empty_graph_all_pass` (the test) now expects exactly one failure,
//! `world_time_grounding`, and asserts the other six still pass — don't "fix"
//! that test by loosening G-POP.
//!
//! # Why `world_time_grounding` measures `valid_from`, not `valid_to`
//!
//! Commits `4e4eaa3e` and `f6df1d25` thread the episode's own declared date
//! into extraction and persist the resolved date into `valid_from` — when a
//! fact STARTS being true in world time. They do not touch `valid_to` — when
//! a fact STOPS being true in world time — which remains NULL on every fact
//! measured so far. A metric that reasoned about `valid_to` would report a
//! perfect score on every database forever, by construction — the same
//! vacuity pattern this file's guards exist to prevent. Do not extend this
//! invariant to score supersession/archive in world time until `valid_to`
//! extraction exists — there is nothing to measure yet.
//!
//! # The inverted-vacuity trap — same-day agreement is BANNED from the pass condition
//!
//! Measured directly, same-day `date(valid_from) == date(episode
//! timestamp)` agreement scores **HIGHER on the BROKEN corpus (100%, all
//! 5,106 facts in `.context/full-corpus.db`) than on the FIXED one (95.2%,
//! 359/377 facts in `.context/td186a-variance/dream-on-adj.db`)** — because
//! the pre-fix bug stamps every fact's `valid_from` from ingest wall-clock,
//! which trivially always equals the episode's own (also wall-clock, pre-fix)
//! timestamp. If same-day agreement ever appears in this invariant's pass
//! condition, a regression back to the pre-fix bug would make the score go
//! UP, not down. The four guards below (G-POP, G-DAYS, G-BACKREF, G-FWD) were
//! chosen specifically because none of them can be satisfied by the
//! same-day-collapse shape.
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

/// Distinct losers absorbed by one keeper before it counts as a fan-in (TD-256).
///
/// 2 is the lowest value that can detect anything, and is also the value at which
/// a legitimate two-variant canonicalisation is flagged — which is exactly why
/// fan-ins are reported rather than failed. Raising this to 3 drops the signal on
/// `.context/full-corpus.db` from 20 findings to 6 and hides every 2-way date
/// collapse. Must stay in lockstep with `FANIN_MIN_LOSERS` in
/// `bench/locomo/harness.py`; the shared fixture test below is what holds them
/// together.
const FANIN_MIN_LOSERS: usize = 2;

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

    // FAN-IN (TD-256) — mirrors `check_graph_integrity` in `bench/locomo/harness.py`.
    // The chain signal above needs one entity to be BOTH survivor and victim. N
    // entities absorbed into ONE keeper that is never itself absorbed forms no
    // chain, so the intersection is empty and the graph reports clean. That is the
    // shape that dominates real data: on `.context/full-corpus.db` (113 live
    // merges) there are 4 chained entities and 20 fan-ins, 19 of them collapsing
    // DISTINCT CALENDAR DATES ('3 july 2023' absorbed 9 other July dates).
    //
    // REPORTED, NEVER FAILED. At the only threshold that detects anything (2
    // distinct losers) a legitimate two-variant canonicalisation is
    // indistinguishable from damage without reading the names — which is what
    // `star_merge_is_not_a_chain` pins. The pass/fail decision below stays on
    // chains alone; the fan-in numbers ride along in the value string so a reader
    // of a GREEN report still sees them.
    //
    // Losers are DE-DUPLICATED: the log genuinely records the same loser twice
    // (two keepers on the full corpus do this), and counting rows rather than
    // distinct entities would promote a one-loser merge into a fan-in.
    let mut fanin_entities = 0usize;
    let mut worst_fanin = 0usize;
    for losers in absorbed.values() {
        let distinct: HashSet<&String> = losers.iter().collect();
        if distinct.len() >= FANIN_MIN_LOSERS {
            fanin_entities += 1;
            worst_fanin = worst_fanin.max(distinct.len());
        }
    }
    let fanin_summary = format!("fanins={} (worst={})", fanin_entities, worst_fanin);

    if chained.len() <= allow_chain_count {
        return Ok(InvariantResult::pass(
            "no_transitive_merge_chain",
            format!(
                "chained={} (allowed={}) across {} live merges, {}",
                chained.len(),
                allow_chain_count,
                merge_count,
                fanin_summary
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
            "chained={} across {} live merges, {}",
            chained.len(),
            merge_count,
            fanin_summary
        ),
        format!("transitive merge chains: {}", detail),
    ))
}

/// Check invariant 7: `valid_from` shows real episode-anchored world-time
/// resolution, not ingest wall-clock.
///
/// Joins every live-or-superseded fact to its source episode
/// (`f.source_episode_id = e.id`) and compares `date(f.valid_from)` against
/// `date(e.timestamp)` — the episode's OWN declared date, not
/// `e.recorded_at` (ingest wall-clock in every database measured, pre- and
/// post-fix — do not trust `recorded_at` as a world-time anchor).
/// `predicate = 'potential_alias'` facts are excluded — a dream-generated
/// system assertion, correctly ingest-dated, not a grounding claim.
///
/// Deliberately queries the RAW `facts` table via `graph.conn` rather than
/// `facts_at(now)` (used by invariants 1–5): this invariant is about whether
/// the EXTRACTOR grounded `valid_from` correctly at insertion time, which
/// holds or doesn't for every fact ever produced — including ones since
/// superseded. Restricting to only-currently-live facts via `facts_at(now)`
/// would arbitrarily undercount and could mask a grounding failure specific
/// to the superseded population.
///
/// Four non-vacuity guards, all of which must be satisfied. See the module
/// doc "The inverted-vacuity trap" section for why same-day agreement can
/// never appear here:
///
/// - **G-POP**: `n_linked >= 200` — an empty/tiny graph must NOT pass.
///   Deliberately breaks `empty_graph_all_pass` (see module doc).
/// - **G-DAYS**: `distinct_days >= 10` — measured: broken corpus 1, fixed
///   corpus 31.
/// - **G-BACKREF** (load-bearing): `back >= 8` AND `back_pct >= 2.0` — a
///   pipeline with no date anchor CANNOT date a fact before its own episode.
///   Measured: broken corpus 0, fixed corpus 18 (4.8%).
/// - **G-FWD**: `fwd_pct <= 3.0` — hallucinated-future ceiling. Measured:
///   both corpora 0.0%, reported as "untested by this corpus", not
///   "confirmed working".
async fn check_world_time_grounding(graph: &TemporalGraph) -> EvalError<InvariantResult> {
    const NAME: &str = "world_time_grounding";
    const MIN_LINKED: i64 = 200;
    const MIN_DISTINCT_DAYS: i64 = 10;
    const MIN_BACK: i64 = 8;
    const MIN_BACK_PCT: f64 = 2.0;
    const MAX_FWD_PCT: f64 = 3.0;

    let mut rows = graph
        .conn
        .query(
            "SELECT
                COUNT(1) AS n_linked,
                COUNT(DISTINCT date(f.valid_from)) AS distinct_days,
                SUM(CASE WHEN date(f.valid_from) < date(e.timestamp) THEN 1 ELSE 0 END) AS back,
                SUM(CASE WHEN date(f.valid_from) = date(e.timestamp) THEN 1 ELSE 0 END) AS same_day,
                SUM(CASE WHEN date(f.valid_from) > date(e.timestamp) THEN 1 ELSE 0 END) AS fwd
             FROM facts f
             JOIN episodes e ON f.source_episode_id = e.id
             WHERE f.predicate != 'potential_alias'",
            (),
        )
        .await
        .map_err(|e| EvalErr::Other(format!("world_time_grounding query failed: {}", e)))?;

    let row = rows
        .next()
        .await
        .map_err(|e| EvalErr::Other(format!("world_time_grounding row read failed: {}", e)))?
        .ok_or_else(|| {
            EvalErr::Other("world_time_grounding query returned no row (should be impossible — it's an unconditional aggregate)".into())
        })?;

    let n_linked = row
        .get::<i64>(0)
        .map_err(|e| EvalErr::Other(format!("world_time_grounding n_linked read failed: {}", e)))?;
    let distinct_days = row.get::<i64>(1).map_err(|e| {
        EvalErr::Other(format!("world_time_grounding distinct_days read failed: {}", e))
    })?;
    let back = row
        .get::<Option<i64>>(2)
        .map_err(|e| EvalErr::Other(format!("world_time_grounding back read failed: {}", e)))?
        .unwrap_or(0);
    let same_day = row
        .get::<Option<i64>>(3)
        .map_err(|e| EvalErr::Other(format!("world_time_grounding same_day read failed: {}", e)))?
        .unwrap_or(0);
    let fwd = row
        .get::<Option<i64>>(4)
        .map_err(|e| EvalErr::Other(format!("world_time_grounding fwd read failed: {}", e)))?
        .unwrap_or(0);

    let back_pct = if n_linked > 0 {
        100.0 * back as f64 / n_linked as f64
    } else {
        0.0
    };
    let fwd_pct = if n_linked > 0 {
        100.0 * fwd as f64 / n_linked as f64
    } else {
        0.0
    };

    let mut violations: Vec<String> = Vec::new();
    if n_linked < MIN_LINKED {
        violations.push(format!(
            "G-POP: n_linked={} < {} (population too small/absent to say anything — TD-224 defence)",
            n_linked, MIN_LINKED
        ));
    }
    if distinct_days < MIN_DISTINCT_DAYS {
        violations.push(format!(
            "G-DAYS: distinct_days={} < {} (broken corpus measured 1; fixed corpus measured 31)",
            distinct_days, MIN_DISTINCT_DAYS
        ));
    }
    if back < MIN_BACK || back_pct < MIN_BACK_PCT {
        violations.push(format!(
            "G-BACKREF: back={} ({:.2}%) — needs back>={} AND back_pct>={:.1}% (a pipeline with no date anchor cannot date a fact before its own episode)",
            back, back_pct, MIN_BACK, MIN_BACK_PCT
        ));
    }
    if fwd_pct > MAX_FWD_PCT {
        violations.push(format!(
            "G-FWD: fwd_pct={:.2}% > {:.1}% (hallucinated-future ceiling)",
            fwd_pct, MAX_FWD_PCT
        ));
    }

    let actual = format!(
        "n_linked={} distinct_days={} back={} ({:.2}%) same_day={} ({:.2}%) fwd={} ({:.2}%)",
        n_linked,
        distinct_days,
        back,
        back_pct,
        same_day,
        if n_linked > 0 {
            100.0 * same_day as f64 / n_linked as f64
        } else {
            0.0
        },
        fwd,
        fwd_pct,
    );

    if violations.is_empty() {
        Ok(InvariantResult::pass(NAME, actual))
    } else {
        Ok(InvariantResult::fail(
            NAME,
            format!(
                "n_linked>={} AND distinct_days>={} AND (back>={} AND back_pct>={:.1}%) AND fwd_pct<={:.1}%",
                MIN_LINKED, MIN_DISTINCT_DAYS, MIN_BACK, MIN_BACK_PCT, MAX_FWD_PCT
            ),
            actual,
            violations.join("; "),
        ))
    }
}

// ---------------------------------------------------------------------------
// run_invariants — public entry point
// ---------------------------------------------------------------------------

/// Run all 7 graph integrity invariants and return a consolidated report.
///
/// Gate: any invariant failure = report `passed = false`.
/// Sub-millisecond execution (no LLM judge).
pub async fn run_invariants(
    graph: &TemporalGraph,
    config: &IntegrityConfig,
) -> EvalError<IntegrityReport> {
    let mut results = Vec::with_capacity(7);

    results.push(check_no_orphan_nodes(graph, config.allow_isolated_entity_count).await?);
    results.push(check_no_duplicate_edges(graph).await?);
    results.push(check_fts_index_in_sync(graph, config.expected_entity_count).await?);
    results.push(check_all_namespaces_present(graph, &config.expected_namespaces).await?);
    results.push(check_episode_edge_presence(graph).await?);
    results.push(check_no_transitive_merge_chain(graph, config.allow_merge_chain_count).await?);
    results.push(check_world_time_grounding(graph).await?);

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

    /// NOT `all_passed()` any more — invariant 7 (`world_time_grounding`)
    /// deliberately fails on an empty graph (G-POP; see the module doc "Why
    /// invariant 7 deliberately breaks `empty_graph_all_pass`").
    /// This test now proves the OTHER SIX still pass vacuously on nothing,
    /// while the new one correctly refuses to.
    #[tokio::test]
    async fn empty_graph_all_pass_except_world_time_grounding() {
        let graph = empty_graph().await;
        let config = IntegrityConfig::default();
        let report = run_invariants(&graph, &config).await.unwrap();
        assert!(
            !report.all_passed(),
            "world_time_grounding should fail on an empty graph, but everything passed"
        );
        let failures = report.failures();
        assert_eq!(
            failures.len(),
            1,
            "expected exactly one failure (world_time_grounding), got: {:?}",
            failures
        );
        assert_eq!(failures[0].name, "world_time_grounding");
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
            // above (no group → 'default') so the composite FK lines up.
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
        // world_time_grounding fails here too (1 fact, nowhere near G-POP's
        // 200 floor) — this test is specifically about the orphan check, so
        // assert THAT one directly rather than `all_passed()`.
        let orphan_result = report
            .invariants
            .iter()
            .find(|r| r.name == "no_orphan_nodes")
            .unwrap();
        assert!(
            orphan_result.passed,
            "orphan check should pass: {:?}",
            orphan_result
        );
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
    // Invariant 6 — no_transitive_merge_chain
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

    /// TD-256, THE BLIND SPOT. Three dates absorbed into one keeper that is never
    /// itself absorbed: `survivors ∩ victims` is EMPTY, so the chain signal reads
    /// clean. The invariant still PASSES — fan-ins never fail a run — but the
    /// count must reach the report, because a reader of a green report is exactly
    /// who needs to see it.
    #[tokio::test]
    async fn fanin_is_reported_but_does_not_fail_the_invariant() {
        let graph = empty_graph().await;
        insert_merge(&graph, "3 july 2023", "5 july 2023", false).await;
        insert_merge(&graph, "3 july 2023", "6 july 2023", false).await;
        insert_merge(&graph, "3 july 2023", "20 july 2023", false).await;

        let result = chain_result(&graph, 0).await;
        assert!(result.passed, "fan-ins must never fail a run: {:?}", result);
        assert!(
            result.actual.contains("chained=0"),
            "no chain exists — that is the whole point: {}",
            result.actual
        );
        assert!(
            result.actual.contains("fanins=1 (worst=3)"),
            "fan-in must reach a GREEN report: {}",
            result.actual
        );
    }

    /// The log really does record the same loser twice (two keepers do this on
    /// `.context/full-corpus.db`). Counting rows rather than DISTINCT entities
    /// would promote a one-loser merge into a fan-in.
    #[tokio::test]
    async fn duplicate_loser_rows_do_not_inflate_a_fanin() {
        let graph = empty_graph().await;
        insert_merge(&graph, "1 february 2023", "4 february 2023", false).await;
        insert_merge(&graph, "1 february 2023", "4 february 2023", false).await;

        let result = chain_result(&graph, 0).await;
        assert!(
            result.actual.contains("fanins=0"),
            "one distinct loser is not a fan-in: {}",
            result.actual
        );
    }

    /// TD-256. The two implementations agree ON ONE FIXTURE, checked mechanically.
    ///
    /// This function and `test_shared_fixture_matches_rust_implementation` in
    /// `bench/locomo/test_locomo_scorer.py` read the SAME file and assert the SAME
    /// `expected` block. Before it existed, the only thing keeping the mirrors
    /// aligned was a doc-comment asking a future editor to remember — and this
    /// repo has already lost four weeks to exactly that (TD-173, the REST fusion
    /// copy that never received the library's cap fix).
    ///
    /// `include_str!` is deliberate: if the fixture moves, this fails to COMPILE
    /// rather than silently testing nothing.
    #[tokio::test]
    async fn shared_fixture_matches_python_implementation() {
        const FIXTURE: &str =
            include_str!("../../../../bench/locomo/fixtures/graph_integrity_shared.json");
        let fixture: serde_json::Value =
            serde_json::from_str(FIXTURE).expect("shared fixture is valid JSON");

        let graph = empty_graph().await;
        for pair in fixture["merges"].as_array().expect("merges is an array") {
            let keeper = pair[0].as_str().expect("keeper is a string");
            let loser = pair[1].as_str().expect("loser is a string");
            insert_merge(&graph, keeper, loser, false).await;
        }

        let expected = &fixture["expected"];
        let result = chain_result(&graph, 0).await;

        // Numbers come FROM THE FILE, never hardcoded here — otherwise this test
        // would keep passing after the fixture changed underneath it.
        for (key, fragment) in [
            ("live_merges", format!("across {} live merges", expected["live_merges"])),
            ("chained_entities", format!("chained={}", expected["chained_entities"])),
            (
                "fanin_entities",
                format!(
                    "fanins={} (worst={})",
                    expected["fanin_entities"], expected["worst_fanin"]
                ),
            ),
        ] {
            assert!(
                result.actual.contains(&fragment),
                "{key}: expected {fragment:?} in {:?}",
                result.actual
            );
        }

        assert_eq!(
            result.passed,
            expected["clean"].as_bool().expect("clean is a bool"),
            "pass/fail must track the fixture's `clean`: {:?}",
            result
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

    // -----------------------------------------------------------------------
    // Invariant 7 — world_time_grounding
    // -----------------------------------------------------------------------

    /// G-POP directly: an empty graph has 0 episode-linked facts, nowhere
    /// near the 200 floor. An empty/tiny graph must NOT pass a metric that's
    /// supposed to say something meaningful.
    #[tokio::test]
    async fn world_time_check_fails_on_empty_graph() {
        let graph = empty_graph().await;
        let result = check_world_time_grounding(&graph).await.unwrap();
        assert!(!result.passed, "empty graph should fail G-POP: {:?}", result);
        assert!(
            result.details.contains("G-POP"),
            "failure should name G-POP: {:?}",
            result
        );
    }

    /// Plants a one-timestamp graph — every fact dated the same single day as
    /// its episode, same shape as the pre-fix bug (ingest wall-clock stamping
    /// both `valid_from` and the episode's own `timestamp` identically). Small
    /// population too, so this trips both G-POP and G-DAYS. The point isn't
    /// which guard fires — it's proving `run_invariants` actually surfaces
    /// `world_time_grounding` in `failures()`, catching "exists but never
    /// pushed into the results vec".
    #[tokio::test]
    async fn world_time_check_is_wired_into_run_invariants() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        let anchor = Utc::now();
        let episode_id = graph
            .insert_episode(InsertEpisodeParams {
                content: "single conversation",
                timestamp: anchor,
                source_type: None,
                metadata: None,
            })
            .await
            .unwrap();
        for i in 0..5 {
            let obj = format!("thing-{}", i);
            graph
                .insert_fact(
                    FactInsert::new("e1", "mentions", anchor)
                        .object_value(&obj)
                        .source_episode_id(episode_id),
                )
                .await
                .unwrap();
        }

        let report = run_invariants(&graph, &IntegrityConfig::default())
            .await
            .unwrap();
        assert!(
            report
                .failures()
                .iter()
                .any(|f| f.name == "world_time_grounding"),
            "world_time_grounding not surfaced in failures(): {:?}",
            report.failures()
        );
    }

    /// Back-dated facts across many distinct days, well over every threshold.
    /// One fixed-timestamp episode ("2024-06-15") anchors 25 distinct days of
    /// facts (10 per day): day offset 0 is same-day (10 facts), offsets 1–24
    /// are all BACK-references (240 facts, 96%) — exactly the shape only
    /// possible when the extractor resolved relative dates against the
    /// episode anchor rather than stamping ingest wall-clock. Catches an
    /// invariant that fails on everything (a check that's accidentally
    /// inverted, or whose SQL never matches a real row).
    #[tokio::test]
    async fn world_time_check_passes_on_grounded_fixture() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        let anchor: chrono::DateTime<Utc> = "2024-06-15T12:00:00Z".parse().unwrap();
        let episode_id = graph
            .insert_episode(InsertEpisodeParams {
                content: "a long conversation covering many past events",
                timestamp: anchor,
                source_type: None,
                metadata: None,
            })
            .await
            .unwrap();

        for day_offset in 0..25i64 {
            let valid_from = anchor - chrono::Duration::days(day_offset);
            for i in 0..10 {
                let obj = format!("event-{}-{}", day_offset, i);
                graph
                    .insert_fact(
                        FactInsert::new("e1", "recalls", valid_from)
                            .object_value(&obj)
                            .source_episode_id(episode_id),
                    )
                    .await
                    .unwrap();
            }
        }

        let result = check_world_time_grounding(&graph).await.unwrap();
        assert!(result.passed, "grounded fixture should pass: {:?}", result);
        assert!(
            result.actual.contains("n_linked=250"),
            "expected 250 linked facts: {:?}",
            result
        );
    }

    /// G-FWD directly, isolated from the other three guards. Per S-9 ("a
    /// non-vacuity guard cannot be made to fail on demand — it is
    /// decorative"): every OTHER test in this module that exercises a
    /// failure (`world_time_check_fails_on_empty_graph`, the real-database
    /// broken-corpus control) trips G-POP and/or G-DAYS/G-BACKREF — none of
    /// them, nor either real corpus measured this session, ever produced a
    /// `fwd_pct` above 0.0%. Without this test, G-FWD's `> 3.0` branch has
    /// literally never been exercised RED, which is exactly what S-9 forbids
    /// trusting.
    ///
    /// Fixture: population 300 across 30 distinct days — 200 back-dated
    /// (offsets 1–20, satisfies G-BACKREF comfortably), 10 same-day, and 90
    /// FORWARD-dated (offsets −1..−9, 30% of the population — comfortably
    /// over the 3.0% ceiling). G-POP (300≥200) and G-DAYS (30≥10) also pass,
    /// so G-FWD is the ONLY guard expected to fire — isolating it, not just
    /// proving "some guard can fail".
    #[tokio::test]
    async fn world_time_check_fails_when_forward_dates_exceed_ceiling() {
        let graph = empty_graph().await;
        graph
            .insert_entity(InsertEntityParams {
                id: "e1",
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
        let anchor: chrono::DateTime<Utc> = "2024-06-15T12:00:00Z".parse().unwrap();
        let episode_id = graph
            .insert_episode(InsertEpisodeParams {
                content: "a conversation with an implausible amount of future-planning",
                timestamp: anchor,
                source_type: None,
                metadata: None,
            })
            .await
            .unwrap();

        // Same-day: 10 facts at offset 0.
        for i in 0..10 {
            let obj = format!("same-{}", i);
            graph
                .insert_fact(
                    FactInsert::new("e1", "recalls", anchor)
                        .object_value(&obj)
                        .source_episode_id(episode_id),
                )
                .await
                .unwrap();
        }
        // Back-dated: offsets 1..=20, 10 facts/day = 200 facts.
        for day_offset in 1..=20i64 {
            let valid_from = anchor - chrono::Duration::days(day_offset);
            for i in 0..10 {
                let obj = format!("back-{}-{}", day_offset, i);
                graph
                    .insert_fact(
                        FactInsert::new("e1", "recalls", valid_from)
                            .object_value(&obj)
                            .source_episode_id(episode_id),
                    )
                    .await
                    .unwrap();
            }
        }
        // Forward-dated: offsets 1..=9 INTO THE FUTURE, 10 facts/day = 90 facts.
        for day_offset in 1..=9i64 {
            let valid_from = anchor + chrono::Duration::days(day_offset);
            for i in 0..10 {
                let obj = format!("fwd-{}-{}", day_offset, i);
                graph
                    .insert_fact(
                        FactInsert::new("e1", "plans", valid_from)
                            .object_value(&obj)
                            .source_episode_id(episode_id),
                    )
                    .await
                    .unwrap();
            }
        }

        let result = check_world_time_grounding(&graph).await.unwrap();
        assert!(
            !result.passed,
            "fixture with 30% forward-dated facts should fail G-FWD: {:?}",
            result
        );
        assert!(
            result.details.contains("G-FWD"),
            "failure should name G-FWD specifically: {:?}",
            result
        );
        assert!(
            !result.details.contains("G-POP")
                && !result.details.contains("G-DAYS")
                && !result.details.contains("G-BACKREF"),
            "G-FWD should be the ONLY guard failing on this fixture (n_linked=300, \
             distinct_days=30, back=200 all comfortably clear their thresholds) — \
             a co-failure here would mean the fixture doesn't actually isolate \
             G-FWD: {:?}",
            result
        );
    }

    /// Drive invariant 7 against a REAL on-disk graph, on the same two
    /// artefacts used to validate `world_time_grounding` above — this is the
    /// G-CONTROL guard: proof the instrument can actually detect both the
    /// broken and fixed state, not just pass by construction.
    ///
    /// Unlike `chain_check_against_real_database` (print-only), this test
    /// ASSERTS the verdict, so a `cargo test` run's own exit code is the
    /// observable signal:
    ///
    /// ```text
    /// cp .context/full-corpus.db /tmp/ctl-broken.db
    /// cp .context/td186a-variance/dream-on-adj.db /tmp/ctl-ok.db
    /// KREMORY_GI_DIM=768 KREMORY_GI_DB=/tmp/ctl-broken.db cargo test -p kremory-eval --lib \
    ///     world_time_check_against_real_database -- --ignored   # MUST FAIL (red)
    /// KREMORY_GI_DIM=768 KREMORY_GI_DB=/tmp/ctl-ok.db     cargo test -p kremory-eval --lib \
    ///     world_time_check_against_real_database -- --ignored   # MUST PASS (green)
    /// ```
    ///
    /// Copy the database first — `TemporalGraph::open` may run migrations and
    /// so can mutate the file.
    #[tokio::test]
    #[ignore = "requires KREMORY_GI_DB pointing at a real on-disk graph"]
    async fn world_time_check_against_real_database() {
        let path = std::env::var("KREMORY_GI_DB").expect("KREMORY_GI_DB not set");
        let dim: usize = std::env::var("KREMORY_GI_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(768);
        let graph = TemporalGraph::open_with_dim(&path, dim)
            .await
            .expect("open graph");
        let result = check_world_time_grounding(&graph).await.expect("check_world_time_grounding");
        println!(
            "KREMORY_GI_DB={} passed={} actual={} details={}",
            path, result.passed, result.actual, result.details
        );
        assert!(
            result.passed,
            "world_time_grounding verdict against KREMORY_GI_DB={}: expected={} actual={} details={}",
            path, result.expected, result.actual, result.details
        );
    }
}

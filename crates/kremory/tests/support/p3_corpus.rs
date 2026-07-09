//! ADR-071 Item 1 — P3 `cross_episode` corpus-calibration gate.
//!
//! **Zero-LLM, code-generated, planted-by-design corpus** (impl-spec
//! `.ai-docs/specs/adr-071-dream-phase-hardening-impl-spec-2026-07-06.md` §Item 1 /
//! §1b). Ground truth = the planted GENERATIVE RULE, not a rater's judgment — no
//! cross-model-family labeling step applies (per the spec's citation of
//! `feedback_fair_adversarial_corpus_methodology`: "planted labels are a stronger
//! guarantee than any rater").
//!
//! This is a DIFFERENT, larger, planted-by-design corpus from the existing
//! 14-row `consolidation_cross_episode_adversarial.jsonl` hand-planted unit fixture
//! — do NOT conflate or overwrite that file, which stays as-is for its own
//! unit-test purpose (`consolidation_cross_episode_test.rs`).
//!
//! Each [`CaseSpec`] describes one planted pair; [`plant_case`] builds it into a
//! FRESH isolated `TemporalGraph::open_in_memory()` (mirrors
//! `consolidation_cross_episode_test.rs`'s per-row-fresh-graph convention — cheap,
//! ~18ms/graph empirically, and eliminates any cross-case contamination risk
//! entirely rather than relying on `group_id` scoping alone).
//!
//! **Corroboration mechanics planted here (mirrors `cross_episode.rs`'s
//! `shares_structure`/`neighbour_degree` read paths exactly, verified this
//! session):**
//! - A "shared neighbour" `N` of `(entity_a, entity_b)` = a third entity BOTH `a`
//!   and `b` assert a relational fact to (`subject=a|b, predicate=X,
//!   object_id=N`). `neighbours_of(a) ∩ neighbours_of(b)` then contains `N`.
//! - `N`'s in-group DEGREE = the count of DISTINCT entities in the SAME
//!   `group_id` that reference `N` via a live fact (any direction). The minimum
//!   possible degree for a genuinely-shared `N` is 2 (only `a` and `b` reference
//!   it) — which is `WEIGHT_LUT_SCALED[2] == SCALED_THRESHOLD` EXACTLY (a
//!   boundary-exact case, not a margin case; see [`Category::Boundary`] below).
//! - `MERGE`-category cases therefore plant TWO independent degree-2 shared
//!   neighbours (not one) so `Σ scaled_weight = 2 × WEIGHT_LUT_SCALED[2] =
//!   1_048_576`, clearing `SCALED_THRESHOLD = 524_288` with a full 2× margin —
//!   never sitting exactly on the boundary (impl-spec Risk #2).
//! - `HubShared`/`TwoHubShared` cases add 8 FILLER entities that also reference
//!   the shared neighbour(s), pushing degree to 10 (> `HUB_DEGREE_CAP = 8`, a
//!   comfortable margin over the cap) so `scaled_weight` returns integer `0` for
//!   that neighbour (verified `cross_episode.rs:802-808`: `f > HUB_DEGREE_CAP` →
//!   `0`).

#![allow(dead_code)]
// not every symbol is used by every consuming test binary
// Test-support file (CLAUDE.md rule 5 exempts test files from strict-typing +
// args-as-object lints) — mirrors `consolidation_cross_episode_test.rs`'s own
// `#![allow(clippy::too_many_arguments)]` for its structurally-identical
// plant-helper signatures (explicit temporal/structural columns as positional
// args reads more clearly here than a bundled params struct would).
#![allow(clippy::too_many_arguments, clippy::unwrap_used, clippy::expect_used)]

use chrono::Utc;

use kremory::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
};
use kremory::core::schema::TemporalGraph;

// ─── Planted category + ground truth ─────────────────────────────────────────

/// The six planted structural categories (impl-spec §1b, ADR-063 spec's category
/// table). `RareShared` and `GenuineDup` share an IDENTICAL generative rule (two
/// independent degree-2 shared neighbours, margin-clearing) — they are reported as
/// separate rows in the confusion table for auditability, per the spec's own
/// "SHOULD_MERGE (RareShared/GenuineDup)" naming, but are mechanically the same
/// planted structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    RareShared,
    GenuineDup,
    HubShared,
    TwoHubShared,
    ZeroShared,
    Boundary,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::RareShared => "RareShared",
            Category::GenuineDup => "GenuineDup",
            Category::HubShared => "HubShared",
            Category::TwoHubShared => "TwoHubShared",
            Category::ZeroShared => "ZeroShared",
            Category::Boundary => "Boundary",
        }
    }
}

/// Ground-truth verdict, from the planted generative rule (never a rater).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Merge,
    NotMerge,
}

/// Declarative spec for one planted case (data only — [`plant_case`] realises it).
#[derive(Debug, Clone)]
pub struct CaseSpec {
    pub id: String,
    pub category: Category,
    pub expected: Verdict,
    /// `true` ⟺ this case counts in the PASS/CONCERNS/FAIL precision-gate
    /// denominator (RareShared/GenuineDup/HubShared/TwoHubShared). `false` ⟺
    /// sanity-only, reported but never gated (ZeroShared/Boundary — impl-spec
    /// §1a corpus-sizing table).
    pub gated: bool,
}

/// A planted case realised into a graph, with the RAW structural facts recorded
/// so the mandatory sanity test (Risk #11) can independently verify the graph
/// actually has the intended shape BEFORE the gate trusts any op output.
pub struct PlantedCase {
    pub spec: CaseSpec,
    pub entity_a: String,
    pub entity_b: String,
    /// `(neighbour_id, intended_degree)` for every shared neighbour planted for
    /// this case. Empty for [`Category::ZeroShared`] (no shared neighbour at
    /// all, by design).
    pub shared_neighbours: Vec<(String, u32)>,
}

// ─── Corpus generation (declarative — no I/O) ─────────────────────────────────

/// Generate the full corpus spec list. Sizes exceed the impl-spec §1a minimums
/// with margin:
///
/// | category                    | count | minimum | gated |
/// |------------------------------|-------|---------|-------|
/// | RareShared                   |    65 |         | yes   |
/// | GenuineDup                   |    65 |  ≥120*  | yes   |
/// | HubShared                    |    32 |    ≥30  | yes   |
/// | TwoHubShared                 |    32 |    ≥30  | yes   |
/// | ZeroShared (sanity-only)     |    12 |    ≥10  | no    |
/// | Boundary (sanity-only)       |    12 |    ≥10  | no    |
///
/// `*` RareShared + GenuineDup combined = 130 ≥ 120.
pub fn generate_corpus() -> Vec<CaseSpec> {
    let mut specs = Vec::new();
    push_n(&mut specs, Category::RareShared, Verdict::Merge, true, 65);
    push_n(&mut specs, Category::GenuineDup, Verdict::Merge, true, 65);
    push_n(&mut specs, Category::HubShared, Verdict::NotMerge, true, 32);
    push_n(
        &mut specs,
        Category::TwoHubShared,
        Verdict::NotMerge,
        true,
        32,
    );
    push_n(
        &mut specs,
        Category::ZeroShared,
        Verdict::NotMerge,
        false,
        12,
    );
    push_n(&mut specs, Category::Boundary, Verdict::Merge, false, 12);
    specs
}

fn push_n(specs: &mut Vec<CaseSpec>, category: Category, expected: Verdict, gated: bool, n: usize) {
    for i in 0..n {
        specs.push(CaseSpec {
            id: format!("p3-{}-{i:04}", category.as_str().to_lowercase()),
            category,
            expected,
            gated,
        });
    }
}

// ─── Realisation (I/O — plants into a fresh isolated graph) ──────────────────

/// Two DISTINCT raw entity ids that normalize identically under
/// `cross_episode::normalize_label` (lowercase + whitespace-collapse): differing
/// only in case, no whitespace at all (so whitespace-collapse is a no-op and
/// case-fold alone makes them equal). Distinct raw strings satisfy the
/// `entities.id TEXT PRIMARY KEY` uniqueness constraint; identical normalized
/// form admits them on the EXACT merge path.
fn label_pair(case_id: &str) -> (String, String) {
    let base = format!("P3{}", case_id.replace('-', "X"));
    (base.clone(), base.to_lowercase())
}

async fn insert_entity(graph: &TemporalGraph, gid: &str, id: &str) {
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id }),
            group_id: Some(gid),
        })
        .await
        .unwrap_or_else(|e| panic!("insert entity {id}: {e}"));
}

async fn new_episode(graph: &TemporalGraph) -> i64 {
    graph
        .insert_episode(InsertEpisodeParams {
            content: "p3-corpus episode",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: None,
        })
        .await
        .expect("insert_episode")
}

async fn anchor(graph: &TemporalGraph, gid: &str, episode_id: i64, entity: &str) {
    graph
        .insert_episodic_edge(InsertEpisodicEdgeParams {
            episode_id,
            entity_id: entity,
            entity_group_id: Some(gid),
            role: "mention",
        })
        .await
        .unwrap_or_else(|e| panic!("insert_episodic_edge for {entity}: {e}"));
}

/// Plant one relational fact `subject --predicate--> object` (raw SQL, mirrors
/// `consolidation_cross_episode_test.rs::fact_rel` exactly — this op reads
/// `facts` directly, no higher-level `FactInsert` builder needed for a pure
/// relational corroborator). `corroboration_inert` defaults to `0` (live) per
/// `migrate_020` (`facts.corroboration_inert INTEGER NOT NULL DEFAULT 0`) — not
/// set explicitly here, same as the existing harness.
async fn fact_rel(graph: &TemporalGraph, gid: &str, subject: &str, predicate: &str, object: &str) {
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
        .unwrap_or_else(|e| panic!("plant relational fact {subject}->{object}: {e}"));
}

/// Realise one [`CaseSpec`] into a FRESH isolated `TemporalGraph`. Returns the
/// graph, its `group_id`, and the [`PlantedCase`] record (raw structural facts
/// for the Risk #11 sanity check).
pub async fn plant_case(spec: CaseSpec) -> (TemporalGraph, String, PlantedCase) {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let gid = format!("p3gate-{}", spec.id);

    let (entity_a, entity_b) = label_pair(&spec.id);
    insert_entity(&graph, &gid, &entity_a).await;
    insert_entity(&graph, &gid, &entity_b).await;

    // Episode-span gate (P3.2): entity_a -> ep1, entity_b -> ep2, ALWAYS ≥2
    // distinct episodes, for EVERY category. This isolates the structural-
    // corroboration signal as the ONLY thing that varies across categories —
    // the episode-span gate itself is never the differentiator in this corpus.
    let ep1 = new_episode(&graph).await;
    let ep2 = new_episode(&graph).await;
    anchor(&graph, &gid, ep1, &entity_a).await;
    anchor(&graph, &gid, ep2, &entity_b).await;

    let mut shared_neighbours: Vec<(String, u32)> = Vec::new();

    match spec.category {
        Category::RareShared | Category::GenuineDup => {
            // TWO independent shared neighbours, each degree=2 (only a,b
            // reference). Sum clears SCALED_THRESHOLD with a full 2x margin —
            // never boundary-exact (impl-spec Risk #2).
            for idx in 0..2 {
                let n = format!("{}-n{idx}", spec.id);
                insert_entity(&graph, &gid, &n).await;
                fact_rel(&graph, &gid, &entity_a, "corroborates", &n).await;
                fact_rel(&graph, &gid, &entity_b, "corroborates", &n).await;
                shared_neighbours.push((n, 2));
            }
        }
        Category::HubShared => {
            let n = format!("{}-hub", spec.id);
            insert_entity(&graph, &gid, &n).await;
            fact_rel(&graph, &gid, &entity_a, "corroborates", &n).await;
            fact_rel(&graph, &gid, &entity_b, "corroborates", &n).await;
            // 8 filler entities push degree to 10 (>8 cap, comfortable margin).
            for f in 0..8 {
                let filler = format!("{}-filler{f}", spec.id);
                insert_entity(&graph, &gid, &filler).await;
                fact_rel(&graph, &gid, &filler, "corroborates", &n).await;
            }
            shared_neighbours.push((n, 10));
        }
        Category::TwoHubShared => {
            let n1 = format!("{}-hub1", spec.id);
            let n2 = format!("{}-hub2", spec.id);
            insert_entity(&graph, &gid, &n1).await;
            insert_entity(&graph, &gid, &n2).await;
            fact_rel(&graph, &gid, &entity_a, "corroborates", &n1).await;
            fact_rel(&graph, &gid, &entity_b, "corroborates", &n1).await;
            fact_rel(&graph, &gid, &entity_a, "corroborates", &n2).await;
            fact_rel(&graph, &gid, &entity_b, "corroborates", &n2).await;
            // Shared filler pool referencing BOTH hubs — pushes both degrees to
            // 10 with half the entity count of two independent filler pools.
            for f in 0..8 {
                let filler = format!("{}-filler{f}", spec.id);
                insert_entity(&graph, &gid, &filler).await;
                fact_rel(&graph, &gid, &filler, "corroborates", &n1).await;
                fact_rel(&graph, &gid, &filler, "corroborates", &n2).await;
            }
            shared_neighbours.push((n1, 10));
            shared_neighbours.push((n2, 10));
        }
        Category::ZeroShared => {
            // Distinct, unrelated third entities — NO shared structure at all
            // (NoSharedStructure path, not the weak-corroboration path).
            let na = format!("{}-na", spec.id);
            let nb = format!("{}-nb", spec.id);
            insert_entity(&graph, &gid, &na).await;
            insert_entity(&graph, &gid, &nb).await;
            fact_rel(&graph, &gid, &entity_a, "corroborates", &na).await;
            fact_rel(&graph, &gid, &entity_b, "corroborates", &nb).await;
            // no shared_neighbours pushed — intentionally empty.
        }
        Category::Boundary => {
            // ONE shared neighbour, degree EXACTLY 2 — the boundary-exact case
            // (weight == SCALED_THRESHOLD, not merely >=). Deliberately
            // unmargined: this is the "calibration crux" the spec asks to
            // REPORT, not gate (current constants merge this).
            let n = format!("{}-boundary", spec.id);
            insert_entity(&graph, &gid, &n).await;
            fact_rel(&graph, &gid, &entity_a, "corroborates", &n).await;
            fact_rel(&graph, &gid, &entity_b, "corroborates", &n).await;
            shared_neighbours.push((n, 2));
        }
    }

    let planted = PlantedCase {
        spec,
        entity_a,
        entity_b,
        shared_neighbours,
    };
    (graph, gid, planted)
}

// ─── Independent raw-structure query (Risk #11 — NEVER calls into the SUT) ───

/// In-group DEGREE of `entity` within `group_id`: the count of DISTINCT entities
/// that reference `entity` via a live, corroboration-live fact (any direction).
///
/// **Deliberately reimplemented, not imported** — this MUST be an INDEPENDENT
/// check of the raw planted graph, never a call into `cross_episode.rs`'s own
/// (private) `neighbour_degree`. Calling the SUT's own function to verify the
/// SUT's own input would be circular and could not catch a corpus-generator bug
/// that happens to agree with a matching SUT bug. The query mirrors
/// `cross_episode.rs:816-839`'s SQL exactly (verified this session) BY DESIGN —
/// same bi-temporal + corroboration-live filter — but is a fresh independent
/// implementation for this purpose.
pub async fn raw_degree(graph: &TemporalGraph, group_id: &str, entity: &str) -> u32 {
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
            libsql::params![group_id, entity],
        )
        .await
        .expect("raw_degree query");
    let count: i64 = rows
        .next()
        .await
        .expect("raw_degree row")
        .map(|r| r.get::<i64>(0))
        .transpose()
        .expect("raw_degree col")
        .unwrap_or(0);
    count.max(0) as u32
}

/// The RAW set of neighbour entity ids shared between `a` and `b` within
/// `group_id` — an independent reimplementation of `cross_episode.rs`'s
/// `neighbours_of` intersection (same rationale as [`raw_degree`]: never call
/// into the SUT to verify the SUT's own input).
pub async fn raw_shared_neighbours(
    graph: &TemporalGraph,
    group_id: &str,
    a: &str,
    b: &str,
) -> std::collections::BTreeSet<String> {
    async fn neighbours_of(
        graph: &TemporalGraph,
        group_id: &str,
        entity: &str,
    ) -> std::collections::BTreeSet<String> {
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
            .await
            .expect("neighbours_of query");
        let mut out = std::collections::BTreeSet::new();
        while let Some(row) = rows.next().await.expect("neighbours_of row") {
            let n: Option<String> = row.get(0).expect("neighbours_of col");
            if let Some(n) = n {
                out.insert(n);
            }
        }
        out
    }

    let na = neighbours_of(graph, group_id, a).await;
    let nb = neighbours_of(graph, group_id, b).await;
    na.intersection(&nb)
        .filter(|n| n.as_str() != a && n.as_str() != b)
        .cloned()
        .collect()
}

//! L5 vector canonicalization (dream-phase post-pass).
//!
//! Periodic batch task that runs pairwise cosine similarity across all entity
//! embeddings within a `group_id`.  Pairs above [`L5_CANONICALIZATION_THRESHOLD`]
//! are merged: the entity with the longer `description` field (LightRAG heuristic)
//! is kept; the other is remapped and deleted.
//!
//! ## Design notes
//!
//! - Pairwise complexity is O(N²) in Rust but the inner similarity computation
//!   is done via libsql's `vector_distance_cos`; only pairs with embeddings are
//!   considered (NULL embeddings are excluded from the listing).
//! - Each merge is wrapped in a `BEGIN IMMEDIATE` transaction so partial failures
//!   cannot leave the graph in an inconsistent state.
//! - A `loser → keeper` map with chain resolution handles transitivity correctly
//!   when merging chains of similar entities.
//! - Threshold comparison is **strict** (`>`): a score of exactly
//!   `L5_CANONICALIZATION_THRESHOLD` does NOT trigger a merge.
//!
//! ## Caller
//!
//! Invoke from the dream-phase pipeline after L4 disambiguation has run.
//! The function is intentionally exposed as a stand-alone method on
//! [`crate::core::schema::TemporalGraph`] (via the `canonicalize_surface_forms`
//! free-function) so it can also be unit-tested without a full dream-phase harness.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use metrics::counter;
use tracing;

use crate::core::dream::provenance::{
    EntityMergePreState, EpisodicEdgeCols, EpisodicEdgeSnapshot, FactEndpoint, KeeperPre,
    LoserEntityRow, MergeInputs, MergeSite, RepointedFact,
};
use crate::core::error::{Error, Result};
use crate::core::provider::DynEmbeddingProvider;
use crate::core::schema::TemporalGraph;

// ─── Threshold constant ───────────────────────────────────────────────────────

/// Cosine similarity **above which** two entity embeddings are considered surface
/// variants of the same real-world entity and are merged.
///
/// Comparison is **strict** (`>`): a score equal to the threshold does NOT merge.
///
/// Source: LightRAG post-extraction canonicalization design (empirically derived).
pub const L5_CANONICALIZATION_THRESHOLD: f32 = 0.8;

// ─── Report struct ────────────────────────────────────────────────────────────

/// Summary returned by [`canonicalize_surface_forms`].
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalizationReport {
    /// The `group_id` that was canonicalized.
    pub group_id: String,
    /// Number of entity pairs examined (upper triangle of the similarity matrix).
    pub pairs_examined: usize,
    /// Number of merge operations applied (loser → keeper remappings).
    pub merges_applied: usize,
    /// The threshold value used for this run.
    pub threshold_used: f32,
}

// ─── Internal type ────────────────────────────────────────────────────────────

/// Minimal entity representation loaded by [`canonicalize_surface_forms`].
/// Only the fields needed for the similarity check + keeper selection.
struct EntitySlot {
    id: String,
    description_len: usize,
}

// ─── Public entry-point ───────────────────────────────────────────────────────

/// Run L5 surface-form canonicalization over all entities in `group_id`.
///
/// # Algorithm
///
/// 1. Fetch all entity rows with non-NULL embeddings from `group_id`.
/// 2. For every pair (i, j) where i < j, compute cosine similarity via
///    libsql's `vector_distance_cos`.
/// 3. Pairs with `similarity > threshold` are scheduled for merge:
///    - **Keeper** = the entity with the longer `description` in `properties`.
///    - **Loser**  = the other entity.
/// 4. Union-find collapses transitive chains so a loser that was already merged
///    into a keeper is not processed twice.
/// 5. Each merge is committed inside a single `BEGIN IMMEDIATE` transaction:
///    - Remap `facts.subject_id` / `facts.object_id` from loser → keeper.
///    - Remap `episodic_edges.entity_id` from loser → keeper.
///    - Accumulate loser's `access_count` into keeper.
///    - Delete loser from `entities_fts`, `entities`.
///
/// # No-op conditions
///
/// Returns a zero-merge report when:
/// - `group_id` has fewer than 2 entities with embeddings.
/// - No pair scores above `threshold`.
///
/// # Idempotency
///
/// After a successful run, no pair in `group_id` will have similarity above
/// `threshold` (assuming embeddings are stable), so a second call returns
/// `merges_applied = 0`.
pub async fn canonicalize_surface_forms(
    graph: &TemporalGraph,
    group_id: &str,
    threshold: f32,
) -> Result<CanonicalizationReport> {
    canonicalize_surface_forms_with_embedder(
        graph,
        CanonicalizeSurfaceFormsParams {
            group_id,
            threshold,
            embedder: None,
        },
    )
    .await
}

/// Bundled parameters for [`canonicalize_surface_forms_with_embedder`] —
/// args-as-object per TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
/// `graph` stays a lead positional param (receiver-like dep, project
/// convention — mirrors [`ApplyMergeWithAuditParams`]).
pub struct CanonicalizeSurfaceFormsParams<'a> {
    pub group_id: &'a str,
    pub threshold: f32,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// every merge this pass applies recomputes + persists the keeper's name
    /// embedding post-commit (see [`EntityMergeParams::embedder`]). `None`
    /// preserves [`canonicalize_surface_forms`]'s pre-TD-112 behavior (stored
    /// embedding left stale after a merge).
    pub embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// [`canonicalize_surface_forms`] extended with an embedder handle (TD-112,
/// `.ai-docs/tech-debt/tech-debt-register.md:2547`) so L5's own pairwise-cosine
/// merges re-embed the keeper instead of leaving its stored embedding stale.
/// Additive — the bare 3-arg [`canonicalize_surface_forms`] is this entry
/// point's `embedder: None` degradation (same "extended with an optional
/// param" shape as [`apply_merge_with_audit`]); all pre-existing call sites
/// and tests keep working unchanged.
pub async fn canonicalize_surface_forms_with_embedder(
    graph: &TemporalGraph,
    params: CanonicalizeSurfaceFormsParams<'_>,
) -> Result<CanonicalizationReport> {
    let CanonicalizeSurfaceFormsParams {
        group_id,
        threshold,
        embedder,
    } = params;
    // ── Step 1: load entity slots with embeddings ─────────────────────────────
    let slots = load_entity_slots(graph, group_id).await?;

    if slots.len() < 2 {
        tracing::debug!(
            target: "kremory.l5",
            group_id,
            entity_count = slots.len(),
            "kremory.l5.skip_too_few_entities"
        );
        return Ok(CanonicalizationReport {
            group_id: group_id.to_string(),
            pairs_examined: 0,
            merges_applied: 0,
            threshold_used: threshold,
        });
    }

    // ── Step 2: pairwise cosine similarity via SQL ────────────────────────────
    let raw_pairs = find_merge_pairs(graph, &slots, threshold).await?;
    let pairs_examined = (slots.len() * (slots.len().saturating_sub(1))) / 2;

    // ADR-057: the L5 batch merge is the MORE dangerous destructive-merge site —
    // its 0.80 threshold sits below the ~0.90 anisotropic-cosine floor of bare-name
    // embeddings, so a weak consumer embedder would merge nearly every entity in a
    // group (`tests/spike_td080_embedder_cosine.rs`). Apply the SAME deterministic
    // name-compatibility gate as L4: a cosine-high pair whose names are lexically
    // incompatible is anisotropy, not identity — drop it. Entity ids ARE normalized
    // names, so they are the correct lexical comparand.
    let pairs_to_merge: Vec<(String, String)> = raw_pairs
        .into_iter()
        .filter(|(loser_id, keeper_id)| {
            let compatible =
                crate::core::disambiguation::names_lexically_compatible(loser_id, keeper_id);
            if !compatible {
                counter!("kremory.l5.merge_blocked_lexical_total").increment(1);
                tracing::info!(
                    target: "kremory.l5",
                    loser_id = %loser_id,
                    keeper_id = %keeper_id,
                    "kremory.l5.merge_blocked_lexical"
                );
            }
            compatible
        })
        .collect();

    if pairs_to_merge.is_empty() {
        tracing::debug!(
            target: "kremory.l5",
            group_id,
            pairs_examined,
            "kremory.l5.no_merges_needed"
        );
        return Ok(CanonicalizationReport {
            group_id: group_id.to_string(),
            pairs_examined,
            merges_applied: 0,
            threshold_used: threshold,
        });
    }

    // ── Step 3: Resolve unique (loser → keeper) pairs ────────────────────────
    //
    // The `pairs_to_merge` upper-triangle list may contain the same loser in
    // multiple pairs (e.g., for a triplet A↔B, A↔C, B↔C all above threshold).
    // Strategy:
    //   a) Build a `loser → best_keeper` map: for each loser keep the keeper with
    //      the longest description (preserves LightRAG heuristic transitively).
    //   b) Iterate unique losers, resolving the best_keeper transitively (if the
    //      chosen keeper is itself a loser in the map, follow the chain until we
    //      reach a non-loser root).
    //   c) Skip if resolved keeper == loser (degenerate self-merge).

    // Build description_len lookup for fast access.
    let desc_len_map: std::collections::HashMap<String, usize> = slots
        .iter()
        .map(|s| (s.id.clone(), s.description_len))
        .collect();

    // loser_id → keeper_id (the keeper with the longest description seen so far
    // for this loser across all pairs it appears in).
    let mut loser_to_keeper: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for (loser_id, keeper_id) in &pairs_to_merge {
        loser_to_keeper
            .entry(loser_id.clone())
            .and_modify(|existing_keeper| {
                // Keep the longer-description keeper.
                let new_len = desc_len_map.get(keeper_id).copied().unwrap_or(0);
                let existing_len = desc_len_map
                    .get(existing_keeper.as_str())
                    .copied()
                    .unwrap_or(0);
                if new_len > existing_len {
                    *existing_keeper = keeper_id.clone();
                }
            })
            .or_insert_with(|| keeper_id.clone());
    }

    // Resolve transitive chains: if keeper_X is itself a loser, follow the map.
    // Bounded iteration (max chain length = number of unique losers).
    fn resolve_keeper(
        _loser: &str,
        keeper: &str,
        map: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut current = keeper.to_string();
        // Safety bound: stop after map.len() hops to prevent infinite loops.
        for _ in 0..=map.len() {
            if let Some(next) = map.get(&current) {
                if next == &current {
                    break;
                }
                current = next.clone();
            } else {
                break;
            }
        }
        current
    }

    // Anti-re-merge nogood (Site #2 of THREE, spec §6.2 V3 fix): load the
    // group's split-pair bans ONCE before the merge loop; a pair `unmerge`
    // recorded must NOT be re-merged here on the next `dream()`.
    let nogoods =
        crate::core::dream::provenance::reversal::load_merge_nogoods(graph, group_id).await?;

    let mut merges_applied = 0usize;

    for (loser_id, raw_keeper_id) in &loser_to_keeper {
        let effective_keeper = resolve_keeper(loser_id, raw_keeper_id, &loser_to_keeper);

        // Skip self-merges (shouldn't happen, but guard defensively).
        if loser_id == &effective_keeper {
            continue;
        }

        // Nogood guard (spec §6.2, Site #2): drop a split pair before the merge.
        if nogoods.contains(&crate::core::dream::provenance::reversal::sorted_pair(
            loser_id,
            &effective_keeper,
        )) {
            counter!(
                "kremory.graph.merge_nogood_skip_total",
                "site" => "canonicalize",
            )
            .increment(1);
            continue;
        }

        tracing::info!(
            target: "kremory.l5",
            group_id,
            loser_id = %loser_id,
            keeper_id = %effective_keeper,
            "kremory.l5.merge"
        );

        apply_merge(
            graph,
            ApplyMergeParams {
                loser_id,
                keeper_id: &effective_keeper,
                embedder,
            },
        )
        .await?;
        merges_applied += 1;
    }

    counter!("kremory.l5.merges_applied_total").increment(merges_applied as u64);
    counter!("kremory.l5.canonicalization_runs_total").increment(1);

    tracing::info!(
        target: "kremory.l5",
        group_id,
        pairs_examined,
        merges_applied,
        threshold,
        "kremory.l5.done"
    );

    Ok(CanonicalizationReport {
        group_id: group_id.to_string(),
        pairs_examined,
        merges_applied,
        threshold_used: threshold,
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Fetch all entity IDs and their description lengths from `group_id`.
/// Only entities with a non-NULL embedding are returned (no embedding = can't
/// compute cosine similarity).
async fn load_entity_slots(graph: &TemporalGraph, group_id: &str) -> Result<Vec<EntitySlot>> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, properties \
             FROM entities \
             WHERE group_id = ?1 AND embedding IS NOT NULL",
            libsql::params![group_id],
        )
        .await?;

    let mut slots = Vec::new();
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let props_text: Option<String> = row.get(1)?;
        let description_len = props_text
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| v.get("description").and_then(|d| d.as_str()).map(str::len))
            .unwrap_or(0);
        slots.push(EntitySlot {
            id,
            description_len,
        });
    }
    Ok(slots)
}

/// For every (i, j) pair with i < j, compute cosine similarity via SQL.
/// Returns `(loser_id, keeper_id)` pairs where similarity > threshold.
///
/// Keeper selection: longer description wins (LightRAG heuristic).
async fn find_merge_pairs(
    graph: &TemporalGraph,
    slots: &[EntitySlot],
    threshold: f32,
) -> Result<Vec<(String, String)>> {
    let mut merge_pairs: Vec<(String, String)> = Vec::new();

    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            let id_a = &slots[i].id;
            let id_b = &slots[j].id;

            // Use SQL vector_distance_cos: distance = 1 - similarity (cosine distance).
            // similarity = 1.0 - distance
            // NULL is returned when either vector has zero magnitude — skip those.
            let mut rows = graph
                .conn
                .query(
                    "SELECT vector_distance_cos(a.embedding, b.embedding) \
                     FROM entities a, entities b \
                     WHERE a.id = ?1 AND b.id = ?2",
                    libsql::params![id_a.clone(), id_b.clone()],
                )
                .await?;

            let Some(row) = rows.next().await? else {
                continue;
            };

            let distance_opt: Option<f64> = row.get(0)?;
            let Some(distance) = distance_opt else {
                // Zero-magnitude vector(s) — skip.
                continue;
            };

            let similarity = (1.0_f32 - distance as f32).clamp(0.0, 1.0);

            if similarity > threshold {
                // Keeper = longer description (LightRAG heuristic).
                let (keeper_idx, loser_idx) =
                    if slots[i].description_len >= slots[j].description_len {
                        (i, j)
                    } else {
                        (j, i)
                    };
                merge_pairs.push((slots[loser_idx].id.clone(), slots[keeper_idx].id.clone()));
            }
        }
    }

    Ok(merge_pairs)
}

/// Apply a single merge: remap all edges from `loser_id` → `keeper_id`,
/// accumulate `access_count`, then delete the loser row.
///
/// Wrapped in a single `BEGIN IMMEDIATE` transaction — partial failure rolls back.
///
/// Thin wrapper over [`apply_merge_with_audit`] with `audit = None` — L5's own
/// pairwise cosine merges are not LLM-adjudicated, so no `identity_verdict_audit`
/// row is written for them (ADR-063 spec §5.2: "audit only LLM-touched decisions").
async fn apply_merge(graph: &TemporalGraph, params: ApplyMergeParams<'_>) -> Result<()> {
    let ApplyMergeParams {
        loser_id,
        keeper_id,
        embedder,
    } = params;
    apply_entity_merge(
        graph,
        EntityMergeParams {
            loser_id,
            keeper_id,
            site: MergeSite::Canonicalize,
            // L5 surface-form merges are pairwise-cosine + lexical-variant gated,
            // NOT structural-corroboration driven (spec §2.3, Quinn L3).
            structural_signal: false,
            embedder,
        },
    )
    .await
}

/// Bundled parameters for [`apply_merge`] — args-as-object per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`). `graph` stays a lead
/// positional param (receiver-like dep, project convention).
struct ApplyMergeParams<'a> {
    loser_id: &'a str,
    keeper_id: &'a str,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// the keeper's name embedding is recomputed + persisted post-commit so it
    /// reflects the merged identity instead of going stale. `None` preserves
    /// pre-TD-112 behavior (no re-embed).
    embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// The shared structural entity-merge executor (ADR-066 spec DoD-P0.3).
///
/// Remaps all edges `loser_id → keeper_id` (`facts.subject_id`/`object_id`,
/// `episodic_edges.entity_id`), accumulates the loser's **`Entity.access_count`**
/// into the keeper (DENT-001 — the LIVE search-tracking field `schema.rs:106`, NOT
/// the deprecated `facts.access_count`; retained verbatim, test T9 asserts it),
/// combines `ner_confidence` via noisy-OR, then deletes the loser from
/// `entities_fts` + `entities`. All inside one `BEGIN IMMEDIATE`.
///
/// Extracted so BOTH `canonicalize_surface_forms` (L5 reconciliation) AND the
/// dream CONSOLIDATION `cross_episode_merges` op (ADR-066 §2.2 / spec P3) call ONE
/// merge code path — no second, drift-prone executor (R-02). This is the
/// `audit = None` (no `identity_verdict_audit` row) surface — the pairwise-cosine /
/// structural-corroboration merges these two callers perform are not LLM-adjudicated.
pub(crate) async fn apply_entity_merge(
    graph: &TemporalGraph,
    params: EntityMergeParams<'_>,
) -> Result<()> {
    let EntityMergeParams {
        loser_id,
        keeper_id,
        site,
        structural_signal,
        embedder,
    } = params;
    apply_merge_with_audit(
        graph,
        ApplyMergeWithAuditParams {
            loser_id,
            keeper_id,
            audit: None,
            site,
            structural_signal,
            embedder,
        },
    )
    .await
}

/// Bundled parameters for [`apply_entity_merge`] — args-as-object per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`; `#[allow]` banned in src).
/// Threads the merge `site` (reversible-graph-mutations spec §2.3) through the
/// shared structural-merge entry without exceeding the arg-count bar.
pub(crate) struct EntityMergeParams<'a> {
    pub(crate) loser_id: &'a str,
    pub(crate) keeper_id: &'a str,
    pub(crate) site: MergeSite,
    /// Whether a deterministic STRUCTURAL corroboration signal drove this merge
    /// (reversible-graph-mutations spec §2.3 — recorded into
    /// `graph_mutation_log.inputs.structural_signal`, the SEE-surface honesty
    /// the inspect view reports, Quinn L3). `true` for cross-episode structural
    /// merges (the corroboration gate); `false` for L5's pairwise-cosine surface
    /// merges (no structural signal). Threaded from the call-site rather than
    /// derived from `audit` (which is `None` on both non-LLM sites).
    pub(crate) structural_signal: bool,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// the keeper's name embedding is recomputed from its canonical id and
    /// persisted post-commit, so the surviving entity's stored embedding
    /// reflects its post-merge identity instead of going stale. `None`
    /// preserves pre-TD-112 behavior (embedding left untouched) — the shape
    /// callers without embedder access (e.g. Site #5's embedder-independent
    /// design) use today.
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// One `identity_verdict_audit` row to write INSIDE the same `BEGIN IMMEDIATE`
/// transaction as a destructive merge (ADR-063 spec §5.1 RISK-003 — "the audit
/// row and the destructive remap are the same logical event; a rollback undoes
/// the audit row along with the remap, which is correct"). Shared by Site #5
/// (`core::dream::acronym_nickname_recall`) and any future LLM-adjudicated
/// caller of [`apply_merge_with_audit`].
pub(crate) struct IdentityVerdictAuditRow<'a> {
    /// `'site5_acronym_nickname'` | `'site3_type_registry'` (spec §5.1).
    pub(crate) site: &'a str,
    pub(crate) group_id: &'a str,
    pub(crate) candidate_a: &'a str,
    pub(crate) candidate_b: &'a str,
    /// `None` for Site #5 (no meaningful cosine signal, spec §3.3/§5.1).
    pub(crate) cosine: Option<f32>,
    /// The write_gate's `DeterministicSignal` input value for this pair.
    pub(crate) structural_signal: bool,
    /// The LLM verdict that authorized this merge. `Merge` via `write_gate`
    /// row 5 always carries a verdict for Site #5 (row 1 never fires there —
    /// `cosine` is always `0.0`, spec §3.3).
    pub(crate) verdict: Option<&'a crate::core::identity_verdict::IdentityVerdictItem>,
    /// `'merge'` — this row always documents a `Merge` decision (the
    /// `PotentialAlias`/`Reject` arms never call `apply_merge_with_audit`).
    pub(crate) decision: &'a str,
    pub(crate) run_id: &'a str,
}

/// Bundled parameters for [`apply_merge_with_audit`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments, threshold 3). `graph` stays
/// a lead positional param (receiver-like dep, project convention).
pub(crate) struct ApplyMergeWithAuditParams<'a> {
    pub(crate) loser_id: &'a str,
    pub(crate) keeper_id: &'a str,
    pub(crate) audit: Option<IdentityVerdictAuditRow<'a>>,
    /// Which merge-producing site fired this merge — recorded into the
    /// `graph_mutation_log` row's `inputs.site` (reversible-graph-mutations
    /// spec §2.3), the provenance the nogood + undo consume in later sub-phases.
    pub(crate) site: MergeSite,
    /// Whether a deterministic STRUCTURAL corroboration signal drove this merge
    /// — recorded into `inputs.structural_signal` (spec §2.3, Quinn L3). `true`
    /// for the cross-episode structural gate and Site #5's structural pre-filter;
    /// `false` for L5 pairwise cosine. Threaded from the call-site (not derived
    /// from `audit`, which is `None` on the non-LLM sites).
    pub(crate) structural_signal: bool,
    /// TD-112: when `Some`, the keeper's name embedding is recomputed +
    /// persisted (best-effort, post-commit) so it reflects the merged
    /// identity. See [`EntityMergeParams::embedder`] doc for the full rationale.
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// [`apply_merge`] extended with an OPTIONAL `identity_verdict_audit` INSERT
/// (ADR-063 spec §3.3/§5.1) inside the SAME `BEGIN IMMEDIATE` transaction as
/// the destructive remap. `audit = None` preserves `apply_merge`'s exact
/// prior behavior (existing L5 callers); `audit = Some(..)` is the new Site #5
/// path. This is the "extended `apply_merge` with an optional audit param"
/// approach named in the implementation brief — additive, no change to the
/// existing `apply_merge` call site's signature or semantics.
pub(crate) async fn apply_merge_with_audit(
    graph: &TemporalGraph,
    params: ApplyMergeWithAuditParams<'_>,
) -> Result<()> {
    let ApplyMergeWithAuditParams {
        loser_id,
        keeper_id,
        audit,
        site,
        structural_signal,
        embedder,
    } = params;
    let guard = graph.begin_immediate_if_needed().await?;
    // Whether THIS guard opened the txn — decides post-commit o11y placement
    // (spec §8.1: the snapshot counter fires only after the durable commit).
    let committed_here = guard.opened();

    // Collect loser's access_count + ner_confidence before deletion. Site #6
    // (ADR-063): the loser's confidence is combined into the keeper via noisy-OR,
    // not discarded (SYNTHESIS §2 — merging the same entity must never lower it).
    let mut rows = graph
        .conn
        .query(
            "SELECT access_count, ner_confidence FROM entities WHERE id = ?1",
            libsql::params![loser_id],
        )
        .await;

    let (loser_access_count, loser_ner_confidence): (i64, Option<f32>) = match rows {
        Ok(ref mut r) => match r.next().await {
            Ok(Some(row)) => (
                row.get::<i64>(0).unwrap_or(0),
                row.get::<Option<f64>>(1).ok().flatten().map(|v| v as f32),
            ),
            _ => (0, None),
        },
        Err(_) => (0, None),
    };

    // ─── Reversible-graph-mutations snapshot (sub-phase 1b, spec §4) ─────────
    // Capture the COMPLETE pre-state and INSERT the `graph_mutation_log` row
    // BEFORE the destructive block below, on `graph.conn` — the SAME connection
    // the destructive statements use, inside the `guard` transaction opened
    // above. The snapshot INSERT therefore commits atomically with the merge
    // (or rolls back with it via the r1..r7 chain / the early-rollback here), so
    // provenance can never diverge from what the merge actually destroyed
    // (spec §1 / §8.1). Capture-BEFORE-destroy is load-bearing: the loser row,
    // its facts' prior `corroboration_inert`, the keeper's pre-overwrite values,
    // and the episodic-edge collision flags only exist until the UPDATEs below
    // run (spec §2.3 / §2.4 DERIVATION deps).
    let snapshot_group_id = match snapshot_merge_pre_state(
        graph,
        MergeSnapshotParams {
            loser_id,
            keeper_id,
            site,
            audit: audit.as_ref(),
            structural_signal,
        },
    )
    .await
    {
        Ok(group_id) => group_id,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };

    // Remap facts.subject_id — ADR-067 §C0 (V1): also stamp `corroboration_inert = 1`
    // on every rewritten row. The endpoint remap makes the keeper INHERIT the
    // loser's fact-neighbours; without this stamp `cross_episode`'s corroboration
    // reads (`neighbours_of`/`assertions_of`, which filter `corroboration_inert = 0`)
    // would treat the inherited structure as directly-asserted on the NEXT pass,
    // re-opening a deferred bridge partner's eligibility (the V1 convergence bug).
    // The flag is monotone (set here, never cleared) — see impl-spec §6 fixpoint proof.
    let r1 = graph
        .conn
        .execute(
            "UPDATE facts SET subject_id = ?1, corroboration_inert = 1 WHERE subject_id = ?2",
            libsql::params![keeper_id, loser_id],
        )
        .await;

    // Remap facts.object_id — same C0 stamp, object-position analogue. Both UNION
    // arms of `neighbours_of` (and `assertions_of`) filter `corroboration_inert = 0`,
    // so a neighbour reached via the loser's OBJECT-position fact must be equally
    // inerted (impl-spec §C0 DoD: "BOTH arms of neighbours_of's UNION").
    let r2 = if r1.is_ok() {
        graph
            .conn
            .execute(
                "UPDATE facts SET object_id = ?1, corroboration_inert = 1 WHERE object_id = ?2",
                libsql::params![keeper_id, loser_id],
            )
            .await
    } else {
        r1
    };

    // Remap episodic_edges.entity_id onto the keeper.
    // Migration 017: episodic_edges carries UNIQUE(episode_id, entity_id,
    // entity_group_id). When keeper and loser both link the same episode the
    // remap collides; `UPDATE OR IGNORE` skips those rows (the keeper already
    // owns that presence edge — the loser's is the same presence fact about the
    // now-merged entity), then `DELETE` removes the now-orphaned loser rows so no
    // dangling entity_id survives the merge.
    let r3 = if r2.is_ok() {
        match graph
            .conn
            .execute(
                "UPDATE OR IGNORE episodic_edges SET entity_id = ?1 WHERE entity_id = ?2",
                libsql::params![keeper_id, loser_id],
            )
            .await
        {
            Ok(_) => {
                graph
                    .conn
                    .execute(
                        "DELETE FROM episodic_edges WHERE entity_id = ?1",
                        libsql::params![loser_id],
                    )
                    .await
            }
            Err(e) => Err(e),
        }
    } else {
        r2
    };

    // Accumulate access_count into keeper
    let r4 = if r3.is_ok() && loser_access_count > 0 {
        graph
            .conn
            .execute(
                "UPDATE entities SET access_count = access_count + ?1 WHERE id = ?2",
                libsql::params![loser_access_count, keeper_id],
            )
            .await
    } else {
        r3
    };

    // Site #6 (ADR-063 §"six sites" #6 / SYNTHESIS §2): combine the loser's
    // ner_confidence into the keeper via noisy-OR (a + b − a·b), null-safe — merging
    // the same real-world entity must never LOWER its confidence. This is the
    // deterministic merged-confidence FORMULA half; the reject-FLOOR gate is deferred
    // (S4-blocked — the floor value + null-prevalence are both unmeasured, R4 open
    // item; building it now would hardcode a guessed floor).
    let r4b = if r4.is_ok() {
        let keeper_ner_confidence: Option<f32> = match graph
            .conn
            .query(
                "SELECT ner_confidence FROM entities WHERE id = ?1",
                libsql::params![keeper_id],
            )
            .await
        {
            Ok(mut kr) => match kr.next().await {
                Ok(Some(row)) => row.get::<Option<f64>>(0).ok().flatten().map(|v| v as f32),
                _ => None,
            },
            Err(_) => None,
        };
        match crate::core::confidence::merged_confidence(
            keeper_ner_confidence,
            loser_ner_confidence,
        ) {
            Some(merged) => {
                graph
                    .conn
                    .execute(
                        "UPDATE entities SET ner_confidence = ?1 WHERE id = ?2",
                        libsql::params![f64::from(merged), keeper_id],
                    )
                    .await
            }
            None => Ok(0), // both null — nothing to combine
        }
    } else {
        r4
    };

    // Delete loser FTS entry
    let r5 = if r4b.is_ok() {
        graph
            .conn
            .execute(
                "DELETE FROM entities_fts WHERE entity_id = ?1",
                libsql::params![loser_id],
            )
            .await
    } else {
        r4b
    };

    // Delete loser entity row
    let r6 = if r5.is_ok() {
        graph
            .conn
            .execute(
                "DELETE FROM entities WHERE id = ?1",
                libsql::params![loser_id],
            )
            .await
    } else {
        r5
    };

    // Optional identity_verdict_audit INSERT — spec §5.1 RISK-003: runs INSIDE
    // this same BEGIN IMMEDIATE transaction, not as a separate best-effort
    // write. `r7` folds into the same all-must-succeed chain as r1-r6 so a
    // failed audit insert rolls back the merge too (same commit/rollback
    // boundary as the remap itself).
    let r7: std::result::Result<u64, libsql::Error> = if r6.is_ok() {
        if let Some(a) = audit.as_ref() {
            graph
                .conn
                .execute(
                    "INSERT INTO identity_verdict_audit \
                     (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
                      llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    libsql::params![
                        a.site,
                        a.group_id,
                        a.candidate_a,
                        a.candidate_b,
                        a.cosine.map(f64::from),
                        a.structural_signal,
                        a.verdict.map(|v| v.is_same_entity),
                        a.verdict.map(|v| f64::from(v.confidence)),
                        a.verdict.map(|v| v.reasoning.clone()),
                        a.decision,
                        a.run_id,
                    ],
                )
                .await
        } else {
            Ok(0)
        }
    } else {
        r6.map(|_| 0)
    };

    match r7 {
        Ok(_) => {
            guard.commit().await?;
            // Spec §8.1: the `graph_mutation_log` row is a durable in-txn write
            // (correct on rollback); its SUMMARY counter must fire only AFTER the
            // real commit, never mid-txn. When `committed_here` is false the
            // durable commit belongs to an outer txn (no current caller nests —
            // all three merge sites open their own txn) so the counter would move
            // to that outer committer; today it is always `true`.
            if committed_here {
                // Spec §8.2 canonical mutation-log counter (renamed from the
                // sub-phase-1b `kremory.dream.provenance.snapshot_total`). Label set
                // is `{kind, source}` ONLY — `group_id` is UNBOUNDED cardinality and
                // MUST NOT be a metric label (ADR-071 Item 5 discipline; spec §8.2
                // reconciled). It is carried as a `tracing::` event FIELD instead, so
                // per-namespace attribution stays observable without exploding the
                // metric's label dimension.
                counter!(
                    "kremory.graph.mutation_logged_total",
                    "kind" => "entity_merge",
                    "source" => "entity_merge_executor",
                )
                .increment(1);
                tracing::debug!(
                    target: "kremory.graph.provenance",
                    kind = "entity_merge",
                    group_id = %snapshot_group_id,
                    source = "entity_merge_executor",
                    "kremory.graph.mutation_logged: reversible-mutation provenance row committed"
                );
                // TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): recompute
                // + persist the keeper's name embedding AFTER the durable commit —
                // the merge itself already succeeded, so a re-embed failure here is
                // best-effort (never rolls back a durable merge) and must not hold
                // the write-lock open across an external embedder call. Without this,
                // the keeper's stored embedding stays at its PRE-merge value forever
                // (register-verified: no production dream/reconciliation site
                // re-persisted a recomputed embedding — `discover_types.rs:735` only
                // recomputes for a similarity comparison, never persists).
                if let Some(emb) = embedder {
                    // Attribute the re-embed outcome per merge site (Quinn L3) so
                    // once multiple sites thread the embedder, their re-embed
                    // success/failure rates stay distinguishable. `site` is a
                    // BOUNDED enum tag (3 variants) — safe as a metric label.
                    let site_label = site.as_str();
                    match emb.embed_dyn(keeper_id).await {
                        Ok(fresh_embedding) => {
                            match graph
                                .update_entity_embedding(keeper_id, &fresh_embedding)
                                .await
                            {
                                Ok(()) => {
                                    counter!(
                                        "kremory.graph.merge_reembed_total",
                                        "outcome" => "success",
                                        "site" => site_label,
                                    )
                                    .increment(1);
                                }
                                Err(e) => {
                                    counter!(
                                        "kremory.graph.merge_reembed_total",
                                        "outcome" => "persist_failed",
                                        "site" => site_label,
                                    )
                                    .increment(1);
                                    tracing::warn!(
                                        target: "kremory.graph.merge_reembed",
                                        keeper_id = %keeper_id,
                                        site = %site_label,
                                        error = %e,
                                        "kremory.graph.merge_reembed_persist_failed: keeper \
                                         embedding left at its stale pre-merge value"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            counter!(
                                "kremory.graph.merge_reembed_total",
                                "outcome" => "embed_failed",
                                "site" => site_label,
                            )
                            .increment(1);
                            tracing::warn!(
                                target: "kremory.graph.merge_reembed",
                                keeper_id = %keeper_id,
                                site = %site_label,
                                error = %e,
                                "kremory.graph.merge_reembed_embed_failed: keeper embedding \
                                 left at its stale pre-merge value"
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e.into())
        }
    }
}

/// Capture the COMPLETE `entity_merge` pre-state (spec §2.3) and INSERT the
/// `graph_mutation_log` row, on `graph.conn` — the SAME connection/txn as the
/// destructive merge statements (the caller holds an open `BeginGuard`), so the
/// snapshot commits atomically with the merge (spec §4 / §8.1). MUST be called
/// BEFORE the destructive block: every captured value (loser row, per-fact
/// prior `corroboration_inert`, keeper's pre-overwrite `access_count` /
/// `ner_confidence`, per-edge collision flag) only exists until the merge's
/// UPDATEs run (spec §2.4 DERIVATION deps).
///
/// A missing loser or keeper entity is a hard `Error` (parse-loudly, spec §2.1):
/// a merge whose endpoints we cannot snapshot could not be reversed, so we fail
/// loudly rather than persist an un-undoable log row.
/// Bundled parameters for [`snapshot_merge_pre_state`] — args-as-object per
/// TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
struct MergeSnapshotParams<'a> {
    loser_id: &'a str,
    keeper_id: &'a str,
    site: MergeSite,
    audit: Option<&'a IdentityVerdictAuditRow<'a>>,
    /// The authoritative structural-corroboration bool for `inputs.structural_signal`
    /// (spec §2.3, Quinn L3) — threaded from the call-site, NOT derived from
    /// `audit` (which is `None` on the non-LLM structural sites).
    structural_signal: bool,
}

async fn snapshot_merge_pre_state(
    graph: &TemporalGraph,
    params: MergeSnapshotParams<'_>,
) -> Result<String> {
    let MergeSnapshotParams {
        loser_id,
        keeper_id,
        site,
        audit,
        structural_signal,
    } = params;
    // (1) loser entity row — all 11 live columns (spec §2.3(1); no `label`,
    //     dropped Mig 009). The merge's loser DELETE is `WHERE id = loser` (no
    //     group filter), so the snapshot SELECT matches that predicate exactly.
    let loser_entity_row = {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, group_id, properties, embedding, recorded_at, updated_at, \
                        access_count, entity_type_id, entity_type_source, \
                        entity_type_assigned_at, ner_confidence \
                 FROM entities WHERE id = ?1",
                libsql::params![loser_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "reversible-mutations snapshot: loser entity `{loser_id}` not found — \
                 cannot capture a reversible pre-state for this merge"
            )));
        };
        // `embedding` is an F32_BLOB; capture the raw bytes faithfully and
        // base64-encode for JSON transport (spec §2.3(1)). A missed byte here is
        // silent embedding loss on undo — the exact failure this sub-phase guards.
        let embedding: Option<Vec<u8>> = row.get::<Option<Vec<u8>>>(3)?;
        LoserEntityRow {
            id: row.get::<String>(0)?,
            group_id: row.get::<String>(1)?,
            properties: row.get::<Option<String>>(2)?,
            embedding_b64: embedding.as_deref().map(|b| BASE64_STANDARD.encode(b)),
            recorded_at: row.get::<String>(4)?,
            updated_at: row.get::<Option<String>>(5)?,
            access_count: row.get::<i64>(6)?,
            entity_type_id: row.get::<i64>(7)?,
            entity_type_source: row.get::<Option<String>>(8)?,
            entity_type_assigned_at: row.get::<Option<String>>(9)?,
            ner_confidence: row.get::<Option<f64>>(10)?,
        }
    };
    let group_id = loser_entity_row.group_id.clone();

    // (2) keeper's PRE-merge access_count + ner_confidence (spec §2.3(2)) —
    //     captured BEFORE the merge's accumulate + noisy-OR overwrite (both
    //     non-invertible), so undo restores these exact values, not a subtraction.
    let keeper_pre = {
        let mut rows = graph
            .conn
            .query(
                "SELECT access_count, ner_confidence FROM entities WHERE id = ?1",
                libsql::params![keeper_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "reversible-mutations snapshot: keeper entity `{keeper_id}` not found — \
                 cannot capture a reversible pre-state for this merge"
            )));
        };
        KeeperPre {
            id: keeper_id.to_string(),
            access_count: row.get::<i64>(0)?,
            ner_confidence: row.get::<Option<f64>>(1)?,
        }
    };

    // (3) facts re-pointed loser→keeper (spec §2.3(3)). The merge stamps each
    //     `corroboration_inert = 1`; capture the PRIOR flag per (fact_id,
    //     endpoint) so undo restores it — a fact already inert from an EARLIER
    //     merge must NOT be cleared (the monotone-undo trap, §12 CH-3). The two
    //     SELECT predicates match the merge's endpoint UPDATEs exactly (subject
    //     then object), no group filter. A self-referential fact (loser on BOTH
    //     endpoints) is captured twice, once per endpoint — correct, undo reverts
    //     both.
    let mut repointed_facts: Vec<RepointedFact> = Vec::new();
    {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, corroboration_inert FROM facts WHERE subject_id = ?1",
                libsql::params![loser_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            repointed_facts.push(RepointedFact {
                fact_id: row.get::<i64>(0)?,
                endpoint: FactEndpoint::Subject,
                prior_corroboration_inert: row.get::<i64>(1)?,
            });
        }
    }
    {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, corroboration_inert FROM facts WHERE object_id = ?1",
                libsql::params![loser_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            repointed_facts.push(RepointedFact {
                fact_id: row.get::<i64>(0)?,
                endpoint: FactEndpoint::Object,
                prior_corroboration_inert: row.get::<i64>(1)?,
            });
        }
    }

    // (4) loser episodic_edges remapped/orphan-deleted by the merge (spec
    //     §2.3(4)). Capture every loser edge FIRST (drain the cursor), THEN
    //     compute its `collided` flag: does the keeper ALREADY own an edge for
    //     the same (episode_id, entity_group_id)? Under
    //     UNIQUE(episode_id, entity_id, entity_group_id) (Mig 017) the merge's
    //     `UPDATE OR IGNORE` drops a colliding loser edge (undo re-INSERTs it
    //     under the loser) and re-points a non-colliding one (undo re-points it
    //     back) — §4.2 step 4. Cursors never overlap: the loser edges are drained
    //     into a Vec before the per-edge collision sub-queries run.
    let loser_edges: Vec<(i64, String, String, String, String)> = {
        let mut rows = graph
            .conn
            .query(
                "SELECT episode_id, entity_group_id, entity_id, role, recorded_at \
                 FROM episodic_edges WHERE entity_id = ?1",
                libsql::params![loser_id],
            )
            .await?;
        let mut v = Vec::new();
        while let Some(row) = rows.next().await? {
            v.push((
                row.get::<i64>(0)?,
                row.get::<String>(1)?,
                row.get::<String>(2)?,
                row.get::<String>(3)?,
                row.get::<String>(4)?,
            ));
        }
        v
    };
    let mut episodic_edges: Vec<EpisodicEdgeSnapshot> = Vec::with_capacity(loser_edges.len());
    for (episode_id, entity_group_id, entity_id, role, recorded_at) in loser_edges {
        let collided = {
            let mut rows = graph
                .conn
                .query(
                    "SELECT 1 FROM episodic_edges \
                     WHERE entity_id = ?1 AND episode_id = ?2 AND entity_group_id = ?3 \
                     LIMIT 1",
                    libsql::params![keeper_id, episode_id, entity_group_id.clone()],
                )
                .await?;
            rows.next().await?.is_some()
        };
        episodic_edges.push(EpisodicEdgeSnapshot {
            episode_id,
            entity_group_id,
            collided,
            cols: EpisodicEdgeCols {
                entity_id,
                role,
                recorded_at,
            },
        });
    }

    let pre_state = EntityMergePreState {
        loser_entity_row,
        keeper_pre,
        repointed_facts,
        episodic_edges,
    };

    // `inputs`: the SORTED unordered pair is the nogood key (spec §6.2 / §2.3) —
    // a keeper/loser role-flip between passes cannot evade the anti-re-merge ban.
    // `cosine` comes from the `audit` row when present (Site #5 passes it; the
    // structural/cosine sites pass `audit = None`). `structural_signal` is the
    // AUTHORITATIVE bool threaded from the call-site (Quinn L3) — it is accurate
    // even on the `audit = None` cross-episode path, so the inspect SEE-surface
    // reports the merge's real provenance.
    let (pair_lo, pair_hi) = if loser_id <= keeper_id {
        (loser_id.to_string(), keeper_id.to_string())
    } else {
        (keeper_id.to_string(), loser_id.to_string())
    };
    let inputs = MergeInputs {
        pair_lo,
        pair_hi,
        keeper: keeper_id.to_string(),
        loser: loser_id.to_string(),
        site,
        cosine: audit.and_then(|a| a.cosine).map(f64::from),
        structural_signal,
    };

    // Our OWN structured emit (never LLM-authored, spec §2.1) — serialization is
    // deterministic; a failure is a hard error, never a silent skip.
    let pre_state_json = serde_json::to_string(&pre_state)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize merge pre_state: {e}")))?;
    let inputs_json = serde_json::to_string(&inputs)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize merge inputs: {e}")))?;

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.graph.provenance",
            kind = "entity_merge",
            group_id = %group_id,
            pre_state = %pre_state_json,
            inputs = %inputs_json,
            "reversible-mutations merge snapshot captured (pre-destroy)"
        );
    }

    // INSERT on `graph.conn` = the SAME txn as the destructive block (spec §2.1
    // / §8.1). `undone_at` (NULL = live) + `enabled_at_time` (DEFAULT 1) take
    // their DDL defaults; `kind` is the fixed `entity_merge` string.
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                "entity_merge",
                group_id.clone(),
                now,
                pre_state_json,
                inputs_json
            ],
        )
        .await?;

    // Return the namespace this merge scoped, for the post-commit
    // `mutation_logged_total{group_id}` label (spec §8.2).
    Ok(group_id)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::InsertEntityWithGroupParams;
    use crate::core::schema::TemporalGraph;

    // ── Helper ────────────────────────────────────────────────────────────────

    /// Insert an entity with a given embedding and description into `group_id`.
    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` seeders.
    #[allow(clippy::too_many_arguments)]
    async fn insert_entity_with_embedding(
        graph: &TemporalGraph,
        id: &str,
        group_id: &str,
        description: &str,
        embedding: &[f32],
    ) {
        let props = serde_json::json!({ "name": id, "description": description });
        // entity_type_id = 0 is the "Entity" catch-all sentinel.
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0u32,
                properties: props,
                group_id: Some(group_id),
            })
            .await
            .expect("insert entity");
        graph
            .set_entity_embedding(id, embedding)
            .await
            .expect("set embedding");
    }

    /// Build a unit-normalised f32 embedding of given dimension where all components
    /// are equal. Useful as "same direction" vectors for high cosine similarity.
    fn unit_vec(dim: usize) -> Vec<f32> {
        let v = 1.0_f32 / (dim as f32).sqrt();
        vec![v; dim]
    }

    /// Build a unit-normalised vector orthogonal to `unit_vec` in 384 dims.
    /// Sets component 0 to 1 and all others to 0.
    fn orthogonal_vec() -> Vec<f32> {
        let mut v = vec![0.0_f32; 384];
        v[0] = 1.0;
        v
    }

    /// TD-112 test helper: cosine distance between an entity's PERSISTED
    /// embedding and an ad-hoc probe vector, via the same `vector_distance_cos`
    /// SQL function `find_merge_pairs` uses in production. Reading back through
    /// SQL (rather than raw bytes) sidesteps endianness/precision concerns —
    /// only the semantic (cosine) equality matters for this assertion.
    /// Returns `None` if the entity has no embedding (zero-magnitude vector).
    async fn entity_embedding_distance(
        graph: &TemporalGraph,
        id: &str,
        probe: &[f32],
    ) -> Option<f32> {
        let vec_str = format!(
            "vector32('[{}]')",
            probe
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut rows = graph
            .conn
            .query(
                &format!(
                    "SELECT vector_distance_cos(embedding, {vec_str}) FROM entities WHERE id = ?1"
                ),
                libsql::params![id],
            )
            .await
            .expect("query entity embedding distance");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("entity must exist for embedding-distance probe");
        row.get::<Option<f64>>(0)
            .expect("distance column")
            .map(|d| d as f32)
    }

    // ── T1: Empty group ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn t1_empty_group_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let report = canonicalize_surface_forms(&graph, "g_empty", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 0);
        assert_eq!(report.group_id, "g_empty");
        assert_eq!(report.threshold_used, L5_CANONICALIZATION_THRESHOLD);
    }

    /// Migration 017 interaction: when keeper and loser BOTH have a presence edge
    /// to the same episode, remapping loser→keeper collides on
    /// UNIQUE(episode_id, entity_id, entity_group_id). `apply_merge`'s
    /// `UPDATE OR IGNORE` + cleanup `DELETE` must leave exactly one surviving edge
    /// on the keeper and zero dangling loser edges — no FK violation, no error.
    #[tokio::test]
    async fn merge_collapses_duplicate_episodic_edges() {
        use crate::core::graph::{InsertEpisodeParams, InsertEpisodicEdgeParams};
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "keeper", "g1", "keeper", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "loser", "g1", "loser", &unit_vec(384)).await;
        let ep = graph
            .insert_episode(InsertEpisodeParams {
                content: "Keeper and loser appear together.",
                timestamp: chrono::Utc::now(),
                source_type: Some("transcript"),
                metadata: None,
            })
            .await
            .expect("episode");
        // Both entities link the SAME episode → post-merge they would collide.
        for ent in ["keeper", "loser"] {
            graph
                .insert_episodic_edge(InsertEpisodicEdgeParams {
                    episode_id: ep,
                    entity_id: ent,
                    entity_group_id: Some("g1"),
                    role: "mention",
                })
                .await
                .expect("edge");
        }

        apply_merge(
            &graph,
            ApplyMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                embedder: None,
            },
        )
        .await
        .expect("merge must not error on episodic-edge collision");

        let keeper_edges = graph.episodic_edges_for_entity("keeper").await.unwrap();
        assert_eq!(
            keeper_edges.len(),
            1,
            "keeper keeps exactly one presence edge after merge; got {keeper_edges:?}"
        );
        let loser_edges = graph.episodic_edges_for_entity("loser").await.unwrap();
        assert!(
            loser_edges.is_empty(),
            "no dangling loser edges after merge; got {loser_edges:?}"
        );
    }

    // ── T2: Single entity ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn t2_single_entity_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "e1", "g_single", "A person.", &unit_vec(384)).await;
        let report = canonicalize_surface_forms(&graph, "g_single", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 0);
    }

    // ── T3: No similar pairs (orthogonal embeddings) ──────────────────────────

    #[tokio::test]
    async fn t3_orthogonal_embeddings_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Five entities with mutually orthogonal embeddings.
        for i in 0..5usize {
            let mut emb = vec![0.0_f32; 384];
            emb[i] = 1.0; // each entity sits on a different axis
            insert_entity_with_embedding(
                &graph,
                &format!("orth_{i}"),
                "g_orth",
                &format!("Entity {i}"),
                &emb,
            )
            .await;
        }
        let report = canonicalize_surface_forms(&graph, "g_orth", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 10); // C(5,2) = 10
    }

    // ── T3b: noisy-OR ner_confidence combination on merge (ADR-063 Site #6) ────

    #[tokio::test]
    async fn merge_combines_ner_confidence_via_noisy_or() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Same surface-variant pair as T4 (merges); keeper = longer description.
        insert_entity_with_embedding(
            &graph,
            "alice johnson",
            "g_conf",
            "A detailed description of Alice Johnson, software engineer.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(&graph, "alice j", "g_conf", "Alice.", &unit_vec(384)).await;

        // Seed ner_confidence: keeper 0.6, loser 0.8 → noisy_or(0.6,0.8) = 0.92.
        graph
            .conn
            .execute(
                "UPDATE entities SET ner_confidence = 0.6 WHERE id = 'alice johnson'",
                (),
            )
            .await
            .expect("set keeper conf");
        graph
            .conn
            .execute(
                "UPDATE entities SET ner_confidence = 0.8 WHERE id = 'alice j'",
                (),
            )
            .await
            .expect("set loser conf");

        let report = canonicalize_surface_forms(&graph, "g_conf", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 1, "the surface-variant pair merges");

        // Keeper ("alice johnson", longer desc) now carries noisy_or(0.6, 0.8) = 0.92
        // — the loser's confidence was COMBINED, not discarded (Site #6 formula half).
        let mut rows = graph
            .conn
            .query(
                "SELECT ner_confidence FROM entities WHERE id = 'alice johnson'",
                (),
            )
            .await
            .expect("query keeper conf");
        let conf: f64 = rows
            .next()
            .await
            .expect("row")
            .expect("keeper survives")
            .get::<Option<f64>>(0)
            .expect("conf col")
            .expect("keeper ner_confidence is set after merge");
        assert!(
            (conf - 0.92).abs() < 1e-4,
            "keeper ner_confidence must be noisy_or(0.6, 0.8) = 0.92, got {conf}"
        );
    }

    // ── T4: One pair above threshold → 1 merge, keeper = longer description ───

    #[tokio::test]
    async fn t4_one_pair_above_threshold_keeper_is_longer_description() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // Two very similar embeddings: both unit vectors → cosine ≈ 1.0 (above 0.8).
        // ADR-057: entity ids are SURFACE VARIANTS of the same name (Jaccard ≥ 0.5),
        // so the lexical merge gate permits the merge (this is what L5 is FOR — merging
        // variants of one entity, never unrelated names). "alice j" → "alice johnson".
        // e_short has the shorter description → it should become the loser.
        insert_entity_with_embedding(
            &graph,
            "alice johnson",
            "g_pair",
            "A detailed description of Alice Johnson, software engineer at Acme Corp.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(&graph, "alice j", "g_pair", "Alice.", &unit_vec(384)).await;

        let report = canonicalize_surface_forms(&graph, "g_pair", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");

        assert_eq!(report.merges_applied, 1, "expected exactly 1 merge");
        assert_eq!(report.pairs_examined, 1);

        // Verify: "alice johnson" (longer desc) survives, "alice j" is gone.
        let entities = graph.list_entities_in_group("g_pair").await.expect("list");
        let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"alice johnson"), "keeper must survive");
        assert!(!ids.contains(&"alice j"), "loser must be deleted");
    }

    // ── T5: Multi-pair chain (A↔B, B↔C, A↔C) → all coalesce to 1 keeper ─────

    #[tokio::test]
    async fn t5_triplet_all_similar_coalesces_to_one_keeper() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // All three entities have the same unit vector → pairwise cosine = 1.0 (>0.8).
        // ADR-057: surface variants of one name (shared {alice, johnson} significant
        // tokens → Jaccard ≥ 0.5 pairwise) so the lexical gate permits all merges.
        // Longest description → "alice johnson engineer" is the keeper.
        insert_entity_with_embedding(&graph, "alice johnson", "g_tri", "Alice.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "alice johnson m",
            "g_tri",
            "Alice Johnson.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(
            &graph,
            "alice johnson engineer",
            "g_tri",
            "Alice Johnson, software engineer.",
            &unit_vec(384),
        )
        .await;

        let report = canonicalize_surface_forms(&graph, "g_tri", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");

        // 2 merges: a_short → keeper, b_mid → keeper
        assert_eq!(report.merges_applied, 2, "2 losers should be merged");
        assert_eq!(report.pairs_examined, 3); // C(3,2) = 3

        // Only 1 entity survives.
        let entities = graph.list_entities_in_group("g_tri").await.expect("list");
        assert_eq!(entities.len(), 1, "only 1 keeper must remain");
    }

    // ── T6: Idempotency — second run applies 0 merges ─────────────────────────

    #[tokio::test]
    async fn t6_idempotent_second_run_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "sam carter", "g_idem", "Short.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "sam carter phd",
            "g_idem",
            "Longer description here.",
            &unit_vec(384),
        )
        .await;

        let r1 = canonicalize_surface_forms(&graph, "g_idem", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("first run");
        assert_eq!(r1.merges_applied, 1, "first run must merge 1 pair");

        let r2 = canonicalize_surface_forms(&graph, "g_idem", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("second run");
        assert_eq!(r2.merges_applied, 0, "second run must be a no-op");
    }

    // ── T7: Threshold boundary — strict `>` (0.8 does NOT merge, >0.8 does) ───

    #[tokio::test]
    async fn t7_threshold_boundary_strict_greater_than() {
        // Use a threshold of 0.99 and unit vectors (similarity ≈ 1.0) → should merge.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "omega corp", "g_bound", "Alpha.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "omega corp ltd",
            "g_bound",
            "Alpha extended.",
            &unit_vec(384),
        )
        .await;

        // Threshold 0.99 — unit vectors have cosine = 1.0 > 0.99 → merge.
        let r = canonicalize_surface_forms(&graph, "g_bound", 0.99)
            .await
            .expect("canonicalize");
        assert_eq!(
            r.merges_applied, 1,
            "cosine=1.0 > threshold=0.99 must merge"
        );
    }

    #[tokio::test]
    async fn t7b_orthogonal_below_threshold_no_merge() {
        // Orthogonal vectors → cosine = 0.0; way below any sane threshold.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "o1", "g_bound2", "Alpha.", &orthogonal_vec()).await;
        let mut perp = vec![0.0_f32; 384];
        perp[1] = 1.0; // orthogonal to o1
        insert_entity_with_embedding(&graph, "o2", "g_bound2", "Beta.", &perp).await;

        let r = canonicalize_surface_forms(&graph, "g_bound2", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(r.merges_applied, 0, "orthogonal vectors must not merge");
    }

    // ── T8: Cross-group isolation ─────────────────────────────────────────────

    #[tokio::test]
    async fn t8_cross_group_isolation() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // group_a: two surface variants (Jaccard 2/3 ≥ 0.5) → merge within a (ADR-057).
        insert_entity_with_embedding(&graph, "nova labs", "group_a", "Short.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "nova labs inc",
            "group_a",
            "Longer description.",
            &unit_vec(384),
        )
        .await;

        // group_b: one entity only (no pair → no merge).
        insert_entity_with_embedding(
            &graph,
            "gb1",
            "group_b",
            "Unrelated entity.",
            &unit_vec(384),
        )
        .await;

        // Canonicalize group_a only.
        let ra = canonicalize_surface_forms(&graph, "group_a", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("group_a");
        assert_eq!(ra.merges_applied, 1, "group_a: 1 merge");

        // group_b must be untouched.
        let rb = canonicalize_surface_forms(&graph, "group_b", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("group_b");
        assert_eq!(rb.merges_applied, 0, "group_b: no merges");

        let gb_entities = graph.list_entities_in_group("group_b").await.expect("list");
        assert_eq!(gb_entities.len(), 1, "gb1 must still exist");
    }

    // ── T9: access_count is accumulated on keeper ─────────────────────────────

    #[tokio::test]
    async fn t9_access_count_accumulated_on_keeper() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "zeta group", "g_ac", "Short.", &unit_vec(384)).await;
        insert_entity_with_embedding(
            &graph,
            "zeta group holdings",
            "g_ac",
            "Longer desc.",
            &unit_vec(384),
        )
        .await;

        // Artificially bump access_count on loser via SQL.
        graph
            .conn
            .execute(
                "UPDATE entities SET access_count = 7 WHERE id = 'zeta group'",
                (),
            )
            .await
            .expect("bump loser access_count");

        let r = canonicalize_surface_forms(&graph, "g_ac", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(r.merges_applied, 1);

        // Keeper should have accumulated loser's access_count (0 + 7 = 7).
        let entities = graph.list_entities_in_group("g_ac").await.expect("list");
        let keeper = entities
            .iter()
            .find(|e| e.id == "zeta group holdings")
            .expect("keeper");
        assert_eq!(
            keeper.access_count, 7,
            "keeper must accumulate loser access_count"
        );
    }

    // ── ADR-067 §C0: apply_entity_merge stamps corroboration_inert on remapped facts ──

    /// Insert a bare entity (no embedding needed — these tests exercise the merge
    /// executor directly via `apply_entity_merge`, not the cosine-gated
    /// `canonicalize_surface_forms` path).
    async fn insert_bare_entity(graph: &TemporalGraph, id: &str, group_id: &str) {
        graph
            .insert_entity_with_group(crate::core::graph::InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({ "name": id }),
                group_id: Some(group_id),
            })
            .await
            .expect("insert bare entity");
    }

    /// Plant a relational fact `subject --predicate--> object` directly via SQL
    /// (mirrors `cross_episode.rs`'s test `fact_rel` helper).
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn plant_fact_rel(
        graph: &TemporalGraph,
        gid: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
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

    async fn corroboration_inert_of(graph: &TemporalGraph, fact_id: i64) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT corroboration_inert FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await
            .expect("query corroboration_inert");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("corroboration_inert col")
    }

    async fn fact_id_of(graph: &TemporalGraph, subject: &str, predicate: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT id FROM facts WHERE subject_id = ?1 AND predicate = ?2",
                libsql::params![subject, predicate],
            )
            .await
            .expect("query fact id");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("id col")
    }

    #[tokio::test]
    async fn apply_entity_merge_stamps_corroboration_inert_on_subject_endpoint() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_c0_subj";
        insert_bare_entity(&graph, "keeper", gid).await;
        insert_bare_entity(&graph, "loser", gid).await;
        insert_bare_entity(&graph, "untouched_subject", gid).await;
        insert_bare_entity(&graph, "neighbour", gid).await;

        // loser's fact (loser is SUBJECT) — endpoint gets remapped to keeper.
        plant_fact_rel(&graph, gid, "loser", "knows", "neighbour").await;
        // An UNRELATED fact (neither endpoint touches the merge) must stay live (=0).
        plant_fact_rel(&graph, gid, "untouched_subject", "knows", "neighbour").await;

        let remapped_id = fact_id_of(&graph, "loser", "knows").await;
        let untouched_id = fact_id_of(&graph, "untouched_subject", "knows").await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        // The remapped fact's endpoint is now `keeper` (subject_id rewritten) AND it
        // must be stamped corroboration_inert = 1.
        assert_eq!(
            corroboration_inert_of(&graph, remapped_id).await,
            1,
            "fact whose subject_id endpoint was rewritten by the merge must be \
             corroboration_inert = 1"
        );
        // The untouched fact (no endpoint rewritten) must remain corroboration_inert = 0.
        assert_eq!(
            corroboration_inert_of(&graph, untouched_id).await,
            0,
            "fact NOT touched by the merge remap must remain corroboration_inert = 0"
        );
    }

    #[tokio::test]
    async fn apply_entity_merge_stamps_corroboration_inert_on_object_endpoint() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_c0_obj";
        insert_bare_entity(&graph, "keeper", gid).await;
        insert_bare_entity(&graph, "loser", gid).await;
        insert_bare_entity(&graph, "asserter", gid).await;

        // `asserter --knows--> loser`: loser is the OBJECT here. Verifies the C0 stamp
        // fires equally on the object_id UPDATE branch (impl-spec §C0 DoD: "a neighbour
        // inherited via the loser's OBJECT-position fact is equally inert").
        plant_fact_rel(&graph, gid, "asserter", "knows", "loser").await;
        let remapped_id = fact_id_of(&graph, "asserter", "knows").await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        assert_eq!(
            corroboration_inert_of(&graph, remapped_id).await,
            1,
            "fact whose object_id endpoint was rewritten by the merge must be \
             corroboration_inert = 1"
        );
        // object_id itself must have been remapped to keeper.
        let mut rows = graph
            .conn
            .query(
                "SELECT object_id FROM facts WHERE id = ?1",
                libsql::params![remapped_id],
            )
            .await
            .expect("query object_id");
        let object_id: Option<String> = rows
            .next()
            .await
            .expect("row")
            .expect("present")
            .get(0)
            .expect("object_id col");
        assert_eq!(
            object_id.as_deref(),
            Some("keeper"),
            "object_id must be remapped"
        );
    }

    // ── TD-112: keeper re-embed on merge ──────────────────────────────────────

    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): a merge that
    /// carries an embedder must recompute + persist the KEEPER's embedding from
    /// its canonical id, replacing whatever stale value it held before the
    /// merge. Before the fix, `apply_merge_with_audit` never touched
    /// `entities.embedding` — the keeper's stored vector stayed at its
    /// pre-merge value forever.
    #[tokio::test]
    async fn apply_entity_merge_reembeds_keeper_when_embedder_present() {
        use crate::core::provider::{DeterministicEmbeddingProvider, EmbeddingProvider};

        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_td112_reembed";
        // Keeper starts with a STALE embedding unrelated to its post-merge
        // identity — simulates an entity embedded before this merge ever ran.
        insert_entity_with_embedding(&graph, "keeper", gid, "keeper", &unit_vec(384)).await;
        insert_bare_entity(&graph, "loser", gid).await;

        let embedder = DeterministicEmbeddingProvider::new(384);
        let fresh_keeper_embedding = embedder.embed("keeper").await.expect("embed keeper id");

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: Some(&embedder),
            },
        )
        .await
        .expect("merge");

        // The keeper's stored embedding must now match a fresh embed of its own
        // id — NOT the stale pre-merge value.
        let distance_to_fresh =
            entity_embedding_distance(&graph, "keeper", &fresh_keeper_embedding).await;
        assert!(
            distance_to_fresh.map(|d| d < 1e-4).unwrap_or(false),
            "keeper's stored embedding must equal a fresh embed of its id post-merge \
             (TD-112); distance was {distance_to_fresh:?}"
        );

        // Regression guard: it must have actually CHANGED from the stale value
        // (rules out a no-op that happens to also satisfy the first assertion).
        let distance_to_stale = entity_embedding_distance(&graph, "keeper", &unit_vec(384)).await;
        assert!(
            distance_to_stale.map(|d| d > 1e-4).unwrap_or(false),
            "keeper's stored embedding must have CHANGED from its stale pre-merge \
             value (TD-112 regression guard); distance was {distance_to_stale:?}"
        );
    }

    /// TD-112 counterpart: `embedder: None` (the pre-fix / degraded-mode shape)
    /// must leave the keeper's embedding untouched — the fix is additive, never
    /// a mandatory re-embed.
    #[tokio::test]
    async fn apply_entity_merge_leaves_keeper_embedding_untouched_when_no_embedder() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_td112_no_embedder";
        insert_entity_with_embedding(&graph, "keeper", gid, "keeper", &unit_vec(384)).await;
        insert_bare_entity(&graph, "loser", gid).await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        let distance_to_stale = entity_embedding_distance(&graph, "keeper", &unit_vec(384)).await;
        assert!(
            distance_to_stale.map(|d| d < 1e-4).unwrap_or(false),
            "with embedder: None, keeper's embedding must be UNCHANGED from its \
             pre-merge value; distance was {distance_to_stale:?}"
        );
    }
}

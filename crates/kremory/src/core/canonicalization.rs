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

use metrics::counter;
use tracing;

use crate::core::error::Result;
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
    let pairs_to_merge = find_merge_pairs(graph, &slots, threshold).await?;
    let pairs_examined = (slots.len() * (slots.len().saturating_sub(1))) / 2;

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
                let existing_len = desc_len_map.get(existing_keeper.as_str()).copied().unwrap_or(0);
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

    let mut merges_applied = 0usize;

    for (loser_id, raw_keeper_id) in &loser_to_keeper {
        let effective_keeper = resolve_keeper(loser_id, raw_keeper_id, &loser_to_keeper);

        // Skip self-merges (shouldn't happen, but guard defensively).
        if loser_id == &effective_keeper {
            continue;
        }

        tracing::info!(
            target: "kremory.l5",
            group_id,
            loser_id = %loser_id,
            keeper_id = %effective_keeper,
            "kremory.l5.merge"
        );

        apply_merge(graph, loser_id, &effective_keeper).await?;
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
        slots.push(EntitySlot { id, description_len });
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
                let (keeper_idx, loser_idx) = if slots[i].description_len >= slots[j].description_len {
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
async fn apply_merge(graph: &TemporalGraph, loser_id: &str, keeper_id: &str) -> Result<()> {
    let guard = graph.begin_immediate_if_needed().await?;

    // Collect loser's access_count before deletion.
    let mut rows = graph
        .conn
        .query(
            "SELECT access_count FROM entities WHERE id = ?1",
            libsql::params![loser_id],
        )
        .await;

    let loser_access_count: i64 = match rows {
        Ok(ref mut r) => match r.next().await {
            Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
            _ => 0,
        },
        Err(_) => 0,
    };

    // Remap facts.subject_id
    let r1 = graph
        .conn
        .execute(
            "UPDATE facts SET subject_id = ?1 WHERE subject_id = ?2",
            libsql::params![keeper_id, loser_id],
        )
        .await;

    // Remap facts.object_id
    let r2 = if r1.is_ok() {
        graph
            .conn
            .execute(
                "UPDATE facts SET object_id = ?1 WHERE object_id = ?2",
                libsql::params![keeper_id, loser_id],
            )
            .await
    } else {
        r1
    };

    // Remap episodic_edges.entity_id
    let r3 = if r2.is_ok() {
        graph
            .conn
            .execute(
                "UPDATE episodic_edges SET entity_id = ?1 WHERE entity_id = ?2",
                libsql::params![keeper_id, loser_id],
            )
            .await
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

    // Delete loser FTS entry
    let r5 = if r4.is_ok() {
        graph
            .conn
            .execute(
                "DELETE FROM entities_fts WHERE entity_id = ?1",
                libsql::params![loser_id],
            )
            .await
    } else {
        r4
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

    match r6 {
        Ok(_) => {
            guard.commit().await?;
            Ok(())
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e.into())
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::schema::TemporalGraph;

    // ── Helper ────────────────────────────────────────────────────────────────

    /// Insert an entity with a given embedding and description into `group_id`.
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
            .insert_entity_with_group(id, 0u32, props, Some(group_id))
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

    // ── T4: One pair above threshold → 1 merge, keeper = longer description ───

    #[tokio::test]
    async fn t4_one_pair_above_threshold_keeper_is_longer_description() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // Two very similar embeddings: both unit vectors → cosine ≈ 1.0 (above 0.8).
        // e_short has the shorter description → it should become the loser.
        insert_entity_with_embedding(
            &graph,
            "e_long",
            "g_pair",
            "A detailed description of Alice Johnson, software engineer at Acme Corp.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(
            &graph,
            "e_short",
            "g_pair",
            "Alice.",
            &unit_vec(384),
        )
        .await;

        let report = canonicalize_surface_forms(&graph, "g_pair", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");

        assert_eq!(report.merges_applied, 1, "expected exactly 1 merge");
        assert_eq!(report.pairs_examined, 1);

        // Verify: e_long still exists, e_short is gone.
        let entities = graph.list_entities_in_group("g_pair").await.expect("list");
        let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e_long"), "keeper e_long must survive");
        assert!(!ids.contains(&"e_short"), "loser e_short must be deleted");
    }

    // ── T5: Multi-pair chain (A↔B, B↔C, A↔C) → all coalesce to 1 keeper ─────

    #[tokio::test]
    async fn t5_triplet_all_similar_coalesces_to_one_keeper() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // All three entities have the same unit vector → pairwise cosine = 1.0 (>0.8).
        // Longest description → "c_long" is the keeper.
        insert_entity_with_embedding(&graph, "a_short", "g_tri", "Alice.", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "b_mid", "g_tri", "Alice Johnson.", &unit_vec(384)).await;
        insert_entity_with_embedding(
            &graph,
            "c_long",
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
        insert_entity_with_embedding(&graph, "x1", "g_idem", "Short.", &unit_vec(384)).await;
        insert_entity_with_embedding(
            &graph,
            "x2",
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
        insert_entity_with_embedding(&graph, "b1", "g_bound", "Alpha.", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "b2", "g_bound", "Alpha extended.", &unit_vec(384)).await;

        // Threshold 0.99 — unit vectors have cosine = 1.0 > 0.99 → merge.
        let r = canonicalize_surface_forms(&graph, "g_bound", 0.99)
            .await
            .expect("canonicalize");
        assert_eq!(r.merges_applied, 1, "cosine=1.0 > threshold=0.99 must merge");
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

        // group_a: two very similar entities (should merge within a).
        insert_entity_with_embedding(&graph, "ga1", "group_a", "Short.", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "ga2", "group_a", "Longer description.", &unit_vec(384)).await;

        // group_b: one entity only (no pair → no merge).
        insert_entity_with_embedding(&graph, "gb1", "group_b", "Unrelated entity.", &unit_vec(384)).await;

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

        insert_entity_with_embedding(&graph, "ac_loser", "g_ac", "Short.", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "ac_keeper", "g_ac", "Longer desc.", &unit_vec(384)).await;

        // Artificially bump access_count on loser via SQL.
        graph
            .conn
            .execute(
                "UPDATE entities SET access_count = 7 WHERE id = 'ac_loser'",
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
        let keeper = entities.iter().find(|e| e.id == "ac_keeper").expect("keeper");
        assert_eq!(keeper.access_count, 7, "keeper must accumulate loser access_count");
    }
}

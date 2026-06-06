//! L7 dream-phase entity reclassification.
//!
//! Batch pass that runs during the dream phase for each `group_id`.  Identifies
//! entities that were committed with the catch-all `entity_type_id=0` ("Entity")
//! and attempts to assign a concrete type by re-asking the LLM with accumulated
//! episode context as evidence.
//!
//! ## Algorithm
//!
//! For each entity in the group where `entity_type_id == 0`:
//! 1. Fetch related episodes in the same `group_id`.
//! 2. If fewer than [`L7_RECLASSIFY_MIN_EPISODES`] episodes exist → skip.
//! 3. Render a reclassification prompt (entity name + episode window + registry table).
//! 4. Call the LLM via [`StructuredCallBuilder`] with [`SCHEMA_RECLASSIFY`].
//! 5. Validate the returned `entity_type_id` via [`EntityTypeRegistry::validate_or_fallback`].
//! 6. If validated id > 0 → update the entity row via `graph.update_entity_type_id`.
//!
//! ## Metrics
//!
//! Per-entity outcomes (`rql.dream.l7_reclassify`) — `outcome` + `reason` pair:
//!
//! - `{outcome=updated, reason=ok}` — entity successfully reclassified.
//! - `{outcome=invalid, reason=registry_rejected}` — LLM returned unregistered id.
//! - `{outcome=skipped, reason=already_typed}` — no-op, entity had concrete type.
//! - `{outcome=skipped, reason=insufficient_episodes}` — L7 throttled below threshold.
//!
//! Per-pass roll-up (emitted by [`run_dream_phase_passes`]):
//!
//! - `rql.dream.l7_pass_started_total` — counter, one increment per pass invocation.
//! - `rql.dream.l7_eligible_entities_per_pass` — histogram, count of `entity_type_id=0`
//!   candidates observed before the reclassify loop fires. Denominator for the
//!   per-entity outcome counters.
//!
//! Cardinality: bounded — 4 outcome×reason combinations, no group_id labels.
//!
//! ## Dream-phase wiring
//!
//! Called from [`run_dream_phase_passes`] after alias resolution and before
//! L5 canonicalization.  All three passes operate on the same `group_id` scope.

use metrics::{counter, histogram};
use tracing;

use crate::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use crate::core::disambiguation::resolve_pending_aliases;
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::error::Result;
use crate::core::extraction::prompts::render_reclassify_prompt;
use crate::core::extraction::schemas::SCHEMA_RECLASSIFY;
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::provider::{chat_msg_user, ChatProvider};
use crate::core::schema::{Entity, Episode, TemporalGraph};

// ─── Threshold constants ──────────────────────────────────────────────────────

/// Minimum number of episodes in the `group_id` required before the L7 pass
/// will attempt to reclassify an entity.
///
/// Below this threshold there is insufficient context for the LLM to distinguish
/// between registered types reliably.  The entity remains at `entity_type_id=0`
/// until more episodes accumulate.
pub const L7_RECLASSIFY_MIN_EPISODES: usize = 3;

// ─── Reclassification wrapper ─────────────────────────────────────────────────

/// Parsed response from the reclassification LLM call.
///
/// Mirrors [`ReclassifyWrapper`] in `extraction/schemas.rs` — duplicated here
/// so the reclassification module does not depend on `pub(crate)` schema internals.
#[derive(Debug, serde::Deserialize, Default)]
struct ReclassifyResponse {
    entity_type_id: u32,
}

// ─── Core reclassification function ──────────────────────────────────────────

/// Attempt to reclassify a single entity from `entity_type_id=0` to a concrete
/// registered type using accumulated episode context.
///
/// ## Returns
///
/// - `Ok(Some(new_id))` — entity was reclassified; DB row updated.
/// - `Ok(None)` — entity already typed, too few episodes, or LLM returned id=0.
/// - `Err(_)` — DB or LLM error.
pub async fn reclassify_entity_type_in_dream_phase<L: ChatProvider>(
    graph: &TemporalGraph,
    llm: &L,
    entity: &Entity,
    related_episodes: &[Episode],
    registry: &EntityTypeRegistry,
) -> Result<Option<u32>> {
    // Guard 1: already typed — nothing to do. Distinct from insufficient_episodes:
    // this is a no-op (entity had concrete type before L7 ever needed to run),
    // not active throttling of L7. Strategy reasoning requires the distinction.
    if entity.entity_type_id != 0 {
        counter!(
            "rql.dream.l7_reclassify",
            "outcome" => "skipped",
            "reason" => "already_typed",
        )
        .increment(1);
        return Ok(None);
    }

    // Guard 2: insufficient context. L7 actively chose not to run this entity yet.
    if related_episodes.len() < L7_RECLASSIFY_MIN_EPISODES {
        counter!(
            "rql.dream.l7_reclassify",
            "outcome" => "skipped",
            "reason" => "insufficient_episodes",
        )
        .increment(1);
        return Ok(None);
    }

    // Build prompt.
    let prompt = render_reclassify_prompt(entity, related_episodes, registry);
    let msgs = vec![chat_msg_user(&prompt)];

    // Call LLM with integer-id schema enforcement.
    let value = StructuredCallBuilder::new(llm, &SCHEMA_RECLASSIFY, "Reclassify")
        .messages(msgs)
        .call()
        .await?;

    // Deserialise response.
    let response: ReclassifyResponse = serde_json::from_value(value).unwrap_or_default();
    let candidate_id = response.entity_type_id;

    // Validate via registry (maps out-of-range / unknown ids → 0).
    let validated = registry.validate_or_fallback(candidate_id);
    if validated == 0 {
        counter!(
            "rql.dream.l7_reclassify",
            "outcome" => "invalid",
            "reason" => "registry_rejected",
        )
        .increment(1);
        tracing::debug!(
            target: "kremory.l7",
            entity_id = %entity.id,
            candidate_id,
            "rql.dream.l7_reclassify.invalid"
        );
        return Ok(None);
    }

    // Update entity row.
    graph.update_entity_type_id(&entity.id, validated).await?;
    counter!(
        "rql.dream.l7_reclassify",
        "outcome" => "updated",
        "reason" => "ok",
    )
    .increment(1);
    tracing::info!(
        target: "kremory.l7",
        entity_id = %entity.id,
        old_type_id = entity.entity_type_id,
        new_type_id = validated,
        "rql.dream.l7_reclassify.updated"
    );

    Ok(Some(validated))
}

// ─── Dream-phase report ───────────────────────────────────────────────────────

/// Aggregated outcome of a single `run_dream_phase_passes` invocation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DreamPhaseReport {
    /// `group_id` that was processed.
    pub group_id: String,
    /// Number of `potential_alias` facts that were resolved (merged or revoked).
    pub aliases_resolved: usize,
    /// Number of entities reclassified from `entity_type_id=0` to a concrete type.
    pub entities_reclassified: usize,
    /// Number of entity merges performed by L5 canonicalization.
    pub canonicalization_merges: usize,
}

// ─── Dream-phase orchestrator ─────────────────────────────────────────────────

/// Run the three L7 dream-phase passes for a single `group_id`.
///
/// Pass order:
/// 1. [`resolve_pending_aliases`] — merge or revoke `potential_alias` facts.
/// 2. [`reclassify_entity_type_in_dream_phase`] — promote `entity_type_id=0` entities.
/// 3. [`canonicalize_surface_forms`] — merge near-duplicate entities by embedding sim.
///
/// ## Isolation
///
/// This free function does not interact with the memory-layer `GraphHandle` trait
/// or the `graph_run_consolidation` path (which is F-01 LOCKED at `NotImplemented`).
/// It operates directly on a `TemporalGraph` reference, keeping it testable without
/// the full async `Engine` infrastructure.
pub async fn run_dream_phase_passes<L: ChatProvider>(
    graph: &TemporalGraph,
    group_id: &str,
    llm: &L,
    registry: &EntityTypeRegistry,
) -> Result<DreamPhaseReport> {
    // Pass 1: resolve pending aliases.
    let aliases_resolved = resolve_pending_aliases(graph, group_id).await?;

    // Pass 2: reclassify untyped entities.
    let episodes = graph.get_episodes_in_group(group_id).await?;
    let entities = graph.list_entities_in_group(group_id).await?;

    // Eligibility snapshot (TD-019 Gap 3 Tier C): denominator for per-entity outcomes.
    // group_id intentionally NOT included as a label — unbounded cardinality.
    let eligible_count = entities.iter().filter(|e| e.entity_type_id == 0).count();
    counter!("rql.dream.l7_pass_started_total").increment(1);
    histogram!("rql.dream.l7_eligible_entities_per_pass").record(eligible_count as f64);

    let mut entities_reclassified = 0usize;
    for entity in &entities {
        if entity.entity_type_id == 0 {
            let outcome = reclassify_entity_type_in_dream_phase(
                graph,
                llm,
                entity,
                &episodes,
                registry,
            )
            .await?;
            if outcome.is_some() {
                entities_reclassified += 1;
            }
        }
    }

    // Pass 3: L5 canonicalization.
    let canon_report =
        canonicalize_surface_forms(graph, group_id, L5_CANONICALIZATION_THRESHOLD).await?;

    Ok(DreamPhaseReport {
        group_id: group_id.to_owned(),
        aliases_resolved,
        entities_reclassified,
        canonicalization_merges: canon_report.merges_applied,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::core::entity_types::{EntityTypeRegistry, EntityTypeSpec};
    use crate::core::provider::MockChatProvider;
    use crate::core::schema::{Entity, Episode, TemporalGraph};
    use chrono::Utc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_registry() -> EntityTypeRegistry {
        EntityTypeRegistry::from_specs(vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "Catch-all.".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A named human individual.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "A company, institution, or formal group.".to_string(),
            },
        ])
    }

    fn make_entity(entity_type_id: u32) -> Entity {
        Entity {
            id: "test-entity".to_string(),
            label: if entity_type_id == 0 {
                "Entity".to_string()
            } else {
                "Person".to_string()
            },
            entity_type_id,
            properties: serde_json::json!({"name": "Alice", "description": "A software engineer."}),
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: Some("test-group".to_string()),
            access_count: 0,
        }
    }

    fn make_episodes(n: usize) -> Vec<Episode> {
        (0..n)
            .map(|i| Episode {
                id: i as i64,
                content: format!("Alice attended meeting {}.", i),
                timestamp: Utc::now(),
                source_type: None,
                metadata: None,
                group_id: Some("test-group".to_string()),
                saga_id: None,
                sequence_number: None,
                content_hash: None,
                recorded_at: None,
            })
            .collect()
    }

    // ── T1: entity already typed → returns Ok(None) ───────────────────────────

    #[tokio::test]
    async fn t1_already_typed_returns_none() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let llm = MockChatProvider::null();
        let entity = make_entity(1); // already Person
        let episodes = make_episodes(5);
        let registry = make_registry();

        let result =
            reclassify_entity_type_in_dream_phase(&graph, &llm, &entity, &episodes, &registry)
                .await
                .expect("reclassify");
        assert!(
            result.is_none(),
            "entity already typed (id=1) must return None"
        );
    }

    // ── T2: fewer than 3 episodes → returns Ok(None) ─────────────────────────

    #[tokio::test]
    async fn t2_insufficient_episodes_returns_none() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let llm = MockChatProvider::null();
        let entity = make_entity(0); // untyped
        let episodes = make_episodes(2); // below minimum
        let registry = make_registry();

        let result =
            reclassify_entity_type_in_dream_phase(&graph, &llm, &entity, &episodes, &registry)
                .await
                .expect("reclassify");
        assert!(
            result.is_none(),
            "fewer than L7_RECLASSIFY_MIN_EPISODES episodes must return None"
        );
    }

    // ── T3: LLM returns valid id → entity updated; returns Some(id) ──────────

    #[tokio::test]
    async fn t3_valid_id_updates_entity_and_returns_some() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Seed entity row with entity_type_id=0.
        graph
            .insert_entity_with_group(
                "test-entity",
                0,
                serde_json::json!({"name": "Alice", "description": "A software engineer."}),
                Some("test-group"),
            )
            .await
            .expect("insert entity");

        // LLM returns {"entity_type_id": 1} → Person.
        let llm = MockChatProvider::with_response("", r#"{"entity_type_id": 1}"#);
        let entity = make_entity(0);
        let episodes = make_episodes(3); // meets minimum
        let registry = make_registry();

        let result =
            reclassify_entity_type_in_dream_phase(&graph, &llm, &entity, &episodes, &registry)
                .await
                .expect("reclassify");

        assert_eq!(
            result,
            Some(1),
            "valid LLM id=1 must return Some(1)"
        );

        // Verify DB was updated.
        let updated = graph
            .get_entity("test-entity")
            .await
            .expect("get entity")
            .expect("entity must exist");
        assert_eq!(
            updated.entity_type_id, 1,
            "entity_type_id in DB must be updated to 1"
        );
    }

    // ── T4: LLM returns out-of-range id → registry fallback to 0 → None ──────

    #[tokio::test]
    async fn t4_out_of_range_id_returns_none_and_no_db_update() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        graph
            .insert_entity_with_group(
                "test-entity",
                0,
                serde_json::json!({"name": "Alice"}),
                Some("test-group"),
            )
            .await
            .expect("insert entity");

        // LLM returns out-of-range id (99 — not registered).
        let llm = MockChatProvider::with_response("", r#"{"entity_type_id": 99}"#);
        let entity = make_entity(0);
        let episodes = make_episodes(5);
        let registry = make_registry(); // max id = 2

        let result =
            reclassify_entity_type_in_dream_phase(&graph, &llm, &entity, &episodes, &registry)
                .await
                .expect("reclassify");

        assert!(
            result.is_none(),
            "out-of-range id must fall back to 0 and return None"
        );

        // Verify DB was NOT updated.
        let unchanged = graph
            .get_entity("test-entity")
            .await
            .expect("get entity")
            .expect("entity must exist");
        assert_eq!(
            unchanged.entity_type_id, 0,
            "entity_type_id must remain 0 when LLM returns invalid id"
        );
    }

    // ── T5: DreamPhaseReport default is all-zero ───────────────────────────────

    #[test]
    fn t5_dream_phase_report_default() {
        let report = DreamPhaseReport::default();
        assert_eq!(report.aliases_resolved, 0);
        assert_eq!(report.entities_reclassified, 0);
        assert_eq!(report.canonicalization_merges, 0);
    }
}

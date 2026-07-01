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
//! Cardinality: bounded — 4 outcome×reason combinations, no group_id labels.
//!
//! ## Dream-phase wiring
//!
//! [`reclassify_entity_type_in_dream_phase`] is the single-entity reclassify
//! primitive. The live dream pass chain is orchestrated by the facade
//! (`mem.dream()` → `facade::dream::execute_blocking`), which runs the canonical
//! §D3 five-pass ordering directly. The former free-function orchestrator
//! `run_dream_phase_passes` (and its `l7_pass_*` roll-up counters, which were
//! only ever emitted from that never-wired orchestrator) was removed in the
//! TD-094 Phase-6 cleanup (2026-07-01) — the facade path is the sole caller.

use metrics::counter;
use tracing;

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

/// Bundled non-generic parameters for `reclassify_entity_type_in_dream_phase`,
/// args-as-object per TD-042 (rust-conventions §too_many_arguments). The generic
/// `llm: &L` stays a lead positional argument.
pub struct ReclassifyEntityTypeParams<'a> {
    /// The temporal graph to update on a successful reclassification.
    pub graph: &'a TemporalGraph,
    /// The entity to reclassify (must currently be `entity_type_id=0`).
    pub entity: &'a Entity,
    /// Episodes providing context for the reclassification LLM call.
    pub related_episodes: &'a [Episode],
    /// Registry used to validate the LLM-emitted candidate type id.
    pub registry: &'a EntityTypeRegistry,
}

/// Attempt to reclassify a single entity from `entity_type_id=0` to a concrete
/// registered type using accumulated episode context.
///
/// ## Returns
///
/// - `Ok(Some(new_id))` — entity was reclassified; DB row updated.
/// - `Ok(None)` — entity already typed, too few episodes, or LLM returned id=0.
/// - `Err(_)` — DB or LLM error.
pub async fn reclassify_entity_type_in_dream_phase<L: ChatProvider>(
    llm: &L,
    params: ReclassifyEntityTypeParams<'_>,
) -> Result<Option<u32>> {
    let ReclassifyEntityTypeParams {
        graph,
        entity,
        related_episodes,
        registry,
    } = params;
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::entity_types::{EntityTypeRegistry, EntityTypeSpec};
    use crate::core::graph::InsertEntityWithGroupParams;
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
                source_id: None,
                source_uri: None,
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

        let result = reclassify_entity_type_in_dream_phase(
            &llm,
            ReclassifyEntityTypeParams {
                graph: &graph,
                entity: &entity,
                related_episodes: &episodes,
                registry: &registry,
            },
        )
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

        let result = reclassify_entity_type_in_dream_phase(
            &llm,
            ReclassifyEntityTypeParams {
                graph: &graph,
                entity: &entity,
                related_episodes: &episodes,
                registry: &registry,
            },
        )
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
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id: "test-entity",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Alice", "description": "A software engineer."}),
                group_id: Some("test-group"),
            })
            .await
            .expect("insert entity");

        // LLM returns {"entity_type_id": 1} → Person.
        let llm = MockChatProvider::with_response("", r#"{"entity_type_id": 1}"#);
        let entity = make_entity(0);
        let episodes = make_episodes(3); // meets minimum
        let registry = make_registry();

        let result = reclassify_entity_type_in_dream_phase(
            &llm,
            ReclassifyEntityTypeParams {
                graph: &graph,
                entity: &entity,
                related_episodes: &episodes,
                registry: &registry,
            },
        )
        .await
        .expect("reclassify");

        assert_eq!(result, Some(1), "valid LLM id=1 must return Some(1)");

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
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id: "test-entity",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Alice"}),
                group_id: Some("test-group"),
            })
            .await
            .expect("insert entity");

        // LLM returns out-of-range id (99 — not registered).
        let llm = MockChatProvider::with_response("", r#"{"entity_type_id": 99}"#);
        let entity = make_entity(0);
        let episodes = make_episodes(5);
        let registry = make_registry(); // max id = 2

        let result = reclassify_entity_type_in_dream_phase(
            &llm,
            ReclassifyEntityTypeParams {
                graph: &graph,
                entity: &entity,
                related_episodes: &episodes,
                registry: &registry,
            },
        )
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
}

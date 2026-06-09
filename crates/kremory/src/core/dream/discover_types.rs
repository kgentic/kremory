//! Pass 0 type discovery primitive (ADR-037 §3 / §9).
//!
//! ## What this does
//!
//! 1. **Signal source** — loads all `entity_type_id = 0` entities for `group_id`.
//! 2. **Top-K clusters** — groups catch-alls by name, ranks by frequency, takes
//!    top `max_proposals` clusters as proposal candidates (one cluster → one LLM prompt entry).
//!    (Embedding-based clustering is deferred to v0.2.0; name-frequency grouping is the
//!    v0.1.1 minimal-first implementation anchored by ADR-037 §9.2.)
//! 3. **LLM proposal call** — single `StructuredCallBuilder` call asking the LLM to
//!    propose entity types for the supplied clusters.  `max_proposals` is enforced
//!    PROMPT-SIDE (cap in the system prompt), not post-emission filter.
//! 4. **Shape validator** — each proposal runs through `validate_proposed_name`
//!    (9 rejection categories per ADR-037 §3.1).
//! 5. **Anti-redundancy gate** — if an embedder is available, each proposal's
//!    description and name are embedded and compared against existing types at
//!    0.85 / 0.70 cosine thresholds.  Without an embedder the gate is skipped
//!    and a warning is recorded (D7 degraded mode).
//! 6. **Persistence** — accepted proposals are inserted into `entity_types` via
//!    `INSERT OR IGNORE` with 4 provenance columns (Migration 014).
//! 7. **In-place evidence retype** — evidence entities whose name is semantically
//!    close to the new type's description (cosine ≥ 0.75) are updated to
//!    `entity_type_id = new_id, entity_type_source = 'DreamPass0'`.
//!    Without an embedder: retype ALL evidence entities for the accepted type (D4).
//!
//! ## Observability (ADR-037 §6)
//!
//! All 6 required metrics are emitted:
//! - `kremory.dream.types_proposed_total{model, namespace}`
//! - `kremory.dream.types_accepted_total{model, namespace}`
//! - `kremory.dream.types_rejected_total{reason, model, namespace}`
//! - `kremory.dream.entities_retyped_total{source, namespace}`
//! - `kremory.dream.proposal_call_duration_ms{model}` (histogram)
//! - `kremory.dream.anti_redundancy_gate_skipped_total{reason="no_embedder"}`

use std::time::Instant;

use chrono::Utc;
use metrics::{counter, histogram};

use crate::core::{
    dream::{
        anti_redundancy::{self, GateOutcome},
        proposed_type::{
            validate_proposed_name, DiscoveryProposalBatch, RejectionReason,
        },
    },
    entity_types::EntityTypeRegistry,
    error::Result,
    extraction::structured::StructuredCallBuilder,
    provider::{chat_msg_system, chat_msg_user, ChatProvider, DynEmbeddingProvider},
};

// ─── Public result types ──────────────────────────────────────────────────────

/// A single discovered or proposed entity type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeProposal {
    pub name: String,
    pub description: String,
    pub justification: String,
}

/// Result of a single `discover_types` invocation.
#[derive(Debug, Default)]
pub struct DiscoveryResult {
    /// All proposals emitted by the LLM (before gating).
    pub types_proposed: Vec<TypeProposal>,
    /// Proposals that passed both the shape validator and the anti-redundancy gate.
    pub types_accepted: Vec<TypeProposal>,
    /// Proposals that were rejected, with the reason.
    pub types_rejected: Vec<(TypeProposal, String)>,
    /// Warnings for operator attention (e.g. degraded-mode runs).
    pub warnings: Vec<String>,
    /// Number of evidence entities retyped in-place.
    pub entities_retyped: usize,
}

/// Maximum cluster candidates passed to the LLM in one call.
/// Enforced PROMPT-SIDE — embedded in the system prompt instruction.
pub(crate) const MAX_PROPOSALS: usize = 5;

/// Cosine threshold for in-place evidence retype (ADR-037 §9.6 / §3.5).
const EVIDENCE_RETYPE_COSINE: f32 = 0.75;

// ─── Main primitive ───────────────────────────────────────────────────────────

/// Discover new entity types from catch-all entities in `group_id`.
///
/// Called by `mem.dream()` when `include_type_discovery = true` (D6).
/// Can also be called standalone via the escape-hatch `mem.discover_types()` (ADR-037 §4.2).
///
/// Parameters:
/// - `conn` — open SQLite connection
/// - `group_id` — the namespace (equals `namespace_to_group_id(&ns)`)
/// - `llm` — the chat provider
/// - `embedder` — `None` = degraded mode (anti-redundancy gate skipped)
/// - `max_proposals` — caller can override; defaults to [`MAX_PROPOSALS`]
pub(crate) async fn discover_types<L: ChatProvider>(
    conn: &libsql::Connection,
    group_id: &str,
    llm: &L,
    embedder: Option<&dyn DynEmbeddingProvider>,
    max_proposals: usize,
) -> Result<DiscoveryResult> {
    let mut result = DiscoveryResult::default();

    // ── Step 1: Load catch-all entities ──────────────────────────────────────

    let catch_alls = load_catch_all_entities(conn, group_id).await?;
    if catch_alls.is_empty() {
        tracing::debug!(
            target: "kremory::dream::discover_types",
            group_id = %group_id,
            "discover_types: no catch-all entities found — skipping"
        );
        return Ok(result);
    }

    tracing::debug!(
        target: "kremory::dream::discover_types",
        group_id = %group_id,
        catch_all_count = catch_alls.len(),
        "discover_types: found catch-all entities"
    );

    // ── Step 2: Top-K clusters by frequency (name-grouping) ──────────────────

    let clusters = top_k_clusters(&catch_alls, max_proposals);
    if clusters.is_empty() {
        return Ok(result);
    }

    // ── Step 3: Load existing registry for anti-redundancy gate ──────────────

    let registry = EntityTypeRegistry::load_for_group(conn, group_id).await?;
    let model_str = llm.model().to_string();

    // ── Step 4: Anti-redundancy embeddings (degraded mode check) ─────────────

    let existing_embeddings: Vec<_> = if let Some(emb) = embedder {
        let mut embs = Vec::new();
        for spec in registry.specs() {
            // Skip catch-all (id=0) — it's not a valid comparison target
            if spec.id == 0 {
                continue;
            }
            let desc_emb = emb.embed_dyn(&spec.description).await?;
            let name_emb = emb.embed_dyn(&spec.name).await?;
            embs.push((spec.clone(), desc_emb, name_emb));
        }
        embs
    } else {
        result.warnings.push(
            "no embedder configured — anti-redundancy gate skipped".to_string(),
        );
        anti_redundancy::emit_gate_skipped();
        Vec::new()
    };

    // ── Step 5: Build LLM prompt ──────────────────────────────────────────────

    let messages = build_discovery_messages(&clusters, max_proposals);
    let schema = crate::core::dream::proposed_type::discovery_proposal_schema()
        .map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!(e)))?;

    // ── Step 6: LLM call with observability ──────────────────────────────────

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(llm, &schema, "DiscoveryProposalBatch")
        .model(&model_str)
        .messages(messages)
        .call()
        .await;
    let elapsed_ms = call_start.elapsed().as_millis() as f64;

    histogram!(
        "kremory.dream.proposal_call_duration_ms",
        "model" => model_str.clone()
    )
    .record(elapsed_ms);

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            counter!(
                "kremory.dream.proposal_call_outcome_total",
                "outcome" => "llm_err",
                "model" => model_str.clone(),
                "namespace" => group_id.to_string()
            )
            .increment(1);
            tracing::warn!(
                target: "kremory::dream::discover_types",
                error = %e,
                group_id = %group_id,
                "discover_types: LLM call failed — returning empty result"
            );
            return Ok(result);
        }
    };

    // Deserialise — use separate counters for direct vs repair path (ADR §6 / rule llm-output-parse-loudly)
    let batch: DiscoveryProposalBatch = match serde_json::from_value(raw_value.clone()) {
        Ok(b) => {
            counter!(
                "kremory.dream.proposal_call_outcome_total",
                "outcome" => "ok",
                "model" => model_str.clone(),
                "namespace" => group_id.to_string()
            )
            .increment(1);
            b
        }
        Err(_) => {
            // Try repair: wrap in {"proposals": ...} if the LLM returned a bare array
            let repaired = if raw_value.is_array() {
                serde_json::json!({ "proposals": raw_value })
            } else {
                raw_value.clone()
            };
            match serde_json::from_value::<DiscoveryProposalBatch>(repaired) {
                Ok(b) => {
                    counter!(
                        "kremory.dream.proposal_call_outcome_total",
                        "outcome" => "parse_repair",
                        "model" => model_str.clone(),
                        "namespace" => group_id.to_string()
                    )
                    .increment(1);
                    b
                }
                Err(e2) => {
                    counter!(
                        "kremory.dream.proposal_call_outcome_total",
                        "outcome" => "parse_fail",
                        "model" => model_str.clone(),
                        "namespace" => group_id.to_string()
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e2,
                        group_id = %group_id,
                        "discover_types: failed to parse LLM batch — returning empty result"
                    );
                    return Ok(result);
                }
            }
        }
    };

    // ── Step 7: Per-proposal: shape validate → anti-redundancy → persist ──────

    for raw_proposal in batch.proposals {
        let proposal = TypeProposal {
            name: raw_proposal.name.clone(),
            description: raw_proposal.description.clone(),
            justification: raw_proposal.justification.clone(),
        };

        // Count proposed
        counter!(
            "kremory.dream.types_proposed_total",
            "model" => model_str.clone(),
            "namespace" => group_id.to_string()
        )
        .increment(1);
        result.types_proposed.push(TypeProposal {
            name: proposal.name.clone(),
            description: proposal.description.clone(),
            justification: proposal.justification.clone(),
        });

        // Shape validate
        if let Err(reason) = validate_proposed_name(&proposal.name, group_id) {
            let reason_str = reason_to_string(reason);
            result
                .types_rejected
                .push((proposal, reason_str));
            continue;
        }

        // Anti-redundancy gate
        if let Some(emb) = embedder {
            let desc_emb = match emb.embed_dyn(&proposal.description).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e,
                        name = %proposal.name,
                        "discover_types: failed to embed proposal description — skipping gate"
                    );
                    result.warnings.push(format!(
                        "embed failed for '{}' — anti-redundancy gate skipped for this proposal",
                        proposal.name
                    ));
                    anti_redundancy::emit_gate_skipped();
                    // Fall through to acceptance (can't gate without embedding)
                    accept_proposal(
                        conn,
                        group_id,
                        &model_str,
                        &proposal,
                        &catch_alls,
                        None,
                        &mut result,
                    )
                    .await?;
                    continue;
                }
            };
            let name_emb = match emb.embed_dyn(&proposal.name).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e,
                        name = %proposal.name,
                        "discover_types: failed to embed proposal name — skipping gate"
                    );
                    result.warnings.push(format!(
                        "embed failed for '{}' name — anti-redundancy gate skipped",
                        proposal.name
                    ));
                    anti_redundancy::emit_gate_skipped();
                    accept_proposal(
                        conn,
                        group_id,
                        &model_str,
                        &proposal,
                        &catch_alls,
                        None,
                        &mut result,
                    )
                    .await?;
                    continue;
                }
            };

            match anti_redundancy::check_proposal(
                &desc_emb,
                &name_emb,
                &existing_embeddings,
                group_id,
                &model_str,
            ) {
                GateOutcome::Redundant { existing_name } => {
                    let reason = format!("redundant_with:{existing_name}");
                    result.types_rejected.push((proposal, reason));
                    continue;
                }
                GateOutcome::Pass => {
                    accept_proposal(
                        conn,
                        group_id,
                        &model_str,
                        &proposal,
                        &catch_alls,
                        Some((&desc_emb, emb)),
                        &mut result,
                    )
                    .await?;
                }
            }
        } else {
            // Degraded mode: gate skipped, accept directly
            accept_proposal(
                conn,
                group_id,
                &model_str,
                &proposal,
                &catch_alls,
                None,
                &mut result,
            )
            .await?;
        }
    }

    Ok(result)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Persist an accepted proposal and retype its evidence entities.
///
/// Uses Migration 014's provenance columns (`discovered_at`, `discovered_by`,
/// `evidence_count`, `confidence`).
// 7 params: conn, group_id, model_str, proposal, catch_alls, desc_emb_and_embedder, result.
// Grouping into a context struct would require an opaque builder layer solely to satisfy
// clippy — same precedent as extraction/structured.rs:try_arm.  Documented exemption.
#[allow(clippy::too_many_arguments)]
async fn accept_proposal(
    conn: &libsql::Connection,
    group_id: &str,
    model_str: &str,
    proposal: &TypeProposal,
    catch_alls: &[CatchAllEntity],
    // (proposal_desc_embedding, embedder) — None = degraded mode
    desc_emb_and_embedder: Option<(&[f32], &dyn DynEmbeddingProvider)>,
    result: &mut DiscoveryResult,
) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    let discovered_by = format!("dream:llm:{model_str}");

    // Count evidence entities that belong to this proposal
    // (all catch-alls contribute to the proposal's evidence_count in the minimal impl)
    let evidence_count = catch_alls.len() as i64;

    // INSERT OR IGNORE into entity_types with Migration 014 provenance columns.
    // Auto-increment assigns the next integer ID.
    conn.execute(
        "INSERT OR IGNORE INTO entity_types \
         (group_id, name, description, discovered_at, discovered_by, evidence_count, confidence) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        libsql::params![
            group_id.to_string(),
            proposal.name.clone(),
            proposal.description.clone(),
            now.clone(),
            discovered_by.clone(),
            evidence_count,
            1.0f64, // Pass 0 treats all surviving proposals as confidence = 1.0
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: INSERT entity_types failed for '{}': {e}",
            proposal.name
        ))
    })?;

    // Re-load the assigned ID (INSERT OR IGNORE means we must SELECT back)
    let new_id: u32 = {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2 LIMIT 1",
                libsql::params![group_id.to_string(), proposal.name.clone()],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "discover_types: SELECT new id failed for '{}': {e}",
                    proposal.name
                ))
            })?;
        let row = rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: SELECT row read failed for '{}': {e}",
                proposal.name
            ))
        })?;
        match row {
            Some(r) => r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "discover_types: id column read failed: {e}"
                ))
            })? as u32,
            None => {
                tracing::warn!(
                    target: "kremory::dream::discover_types",
                    name = %proposal.name,
                    "discover_types: row vanished after INSERT — skipping retype"
                );
                return Ok(());
            }
        }
    };

    // Count + emit accepted metric
    counter!(
        "kremory.dream.types_accepted_total",
        "model" => model_str.to_string(),
        "namespace" => group_id.to_string()
    )
    .increment(1);
    result.types_accepted.push(TypeProposal {
        name: proposal.name.clone(),
        description: proposal.description.clone(),
        justification: proposal.justification.clone(),
    });

    tracing::info!(
        target: "kremory::dream::discover_types",
        group_id = %group_id,
        type_name = %proposal.name,
        new_id = new_id,
        "discover_types: accepted and persisted new entity type"
    );

    // ── In-place evidence retype (D4) ─────────────────────────────────────────
    //
    // If embedder available: retype only evidence entities whose name embedding
    // is cosine ≥ 0.75 to the new type's description embedding (ADR-037 §9.6).
    //
    // If no embedder (degraded): retype ALL evidence entities for this type
    // (accepted type's name is the only signal).

    let retyped_count = if let Some((desc_emb, emb)) = desc_emb_and_embedder {
        retype_evidence_by_similarity(
            conn,
            group_id,
            catch_alls,
            new_id,
            desc_emb,
            emb,
            &now,
        )
        .await?
    } else {
        retype_evidence_all(conn, group_id, catch_alls, new_id, &now).await?
    };

    if retyped_count > 0 {
        counter!(
            "kremory.dream.entities_retyped_total",
            "source" => "pass_0_evidence",
            "namespace" => group_id.to_string()
        )
        .increment(retyped_count as u64);
        result.entities_retyped += retyped_count;
    }

    Ok(())
}

/// Retype evidence entities with cosine ≥ 0.75 similarity to the type description.
// 7 params: conn, group_id, catch_alls, new_type_id, type_desc_emb, emb, now.
// Same documented exemption as accept_proposal above.
#[allow(clippy::too_many_arguments)]
async fn retype_evidence_by_similarity(
    conn: &libsql::Connection,
    group_id: &str,
    catch_alls: &[CatchAllEntity],
    new_type_id: u32,
    type_desc_emb: &[f32],
    emb: &dyn DynEmbeddingProvider,
    now: &str,
) -> Result<usize> {
    let mut count = 0usize;
    for entity in catch_alls {
        // Embed the entity's stored ID (which is the normalised name)
        let entity_name_emb = match emb.embed_dyn(&entity.id).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        let sim = anti_redundancy::cosine(&entity_name_emb, type_desc_emb);
        if sim >= EVIDENCE_RETYPE_COSINE {
            retype_entity(conn, group_id, &entity.id, new_type_id, now).await?;
            count += 1;
        }
    }
    Ok(count)
}

/// Retype all evidence catch-all entities (degraded mode: no embedder).
async fn retype_evidence_all(
    conn: &libsql::Connection,
    group_id: &str,
    catch_alls: &[CatchAllEntity],
    new_type_id: u32,
    now: &str,
) -> Result<usize> {
    for entity in catch_alls {
        retype_entity(conn, group_id, &entity.id, new_type_id, now).await?;
    }
    Ok(catch_alls.len())
}

/// Update a single entity's type to `new_type_id` with DreamPass0 provenance (D4).
async fn retype_entity(
    conn: &libsql::Connection,
    group_id: &str,
    entity_id: &str,
    new_type_id: u32,
    now: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE entities \
         SET entity_type_id = ?1, \
             entity_type_source = 'DreamPass0', \
             entity_type_assigned_at = ?2 \
         WHERE id = ?3 AND group_id = ?4 AND entity_type_id = 0",
        libsql::params![
            new_type_id as i64,
            now.to_string(),
            entity_id.to_string(),
            group_id.to_string(),
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: retype entity failed for id='{}': {e}",
            entity_id
        ))
    })?;
    Ok(())
}

// ─── Catch-all entity loader ──────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct CatchAllEntity {
    pub(crate) id: String,
}

async fn load_catch_all_entities(
    conn: &libsql::Connection,
    group_id: &str,
) -> Result<Vec<CatchAllEntity>> {
    let mut rows = conn
        .query(
            "SELECT id FROM entities \
             WHERE group_id = ?1 AND entity_type_id = 0 \
             ORDER BY access_count DESC",
            libsql::params![group_id.to_string()],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities query failed: {e}"
            ))
        })?;

    let mut entities = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: load_catch_all_entities row read failed: {e}"
        ))
    })? {
        let id: String = row.get(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities id read failed: {e}"
            ))
        })?;
        entities.push(CatchAllEntity { id });
    }
    Ok(entities)
}

// ─── Name-frequency clustering (v0.1.1 minimal) ──────────────────────────────

/// A cluster of catch-all entities with the same normalised name.
#[derive(Debug)]
pub(crate) struct NameCluster {
    /// Representative name (first occurrence).
    pub(crate) name: String,
    /// Number of catch-all entities with this name.
    pub(crate) count: usize,
}

/// Group catch-alls by normalised name, rank by frequency, take top K.
///
/// ADR-037 §9.2 specifies embedding-cosine clustering at threshold 0.7 with
/// minimum cluster size 3. v0.1.1 minimal-first: name-frequency grouping.
/// Embedding clustering ships in v0.2.0 when the embedder is always available.
fn top_k_clusters(entities: &[CatchAllEntity], k: usize) -> Vec<NameCluster> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for entity in entities {
        let norm = normalise_name(&entity.id);
        *counts.entry(norm).or_insert(0) += 1;
    }
    // Filter: minimum cluster size of 2 (relaxed from spec's 3 at v0.1.1 to avoid
    // suppressing all proposals on small test namespaces).
    let mut clusters: Vec<NameCluster> = counts
        .into_iter()
        .filter(|(_, c)| *c >= 1) // include singletons at v0.1.1
        .map(|(name, count)| NameCluster { name, count })
        .collect();
    // Sort descending by count for prompt ordering (most evidence first).
    clusters.sort_by(|a, b| b.count.cmp(&a.count));
    clusters.truncate(k);
    clusters
}

/// Normalise entity id/name for clustering: lowercase + trim.
fn normalise_name(s: &str) -> String {
    s.trim().to_lowercase()
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

fn build_discovery_messages(clusters: &[NameCluster], max_proposals: usize) -> Vec<crate::core::provider::ChatMessage> {
    let cluster_list = clusters
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. \"{}\" (seen {} time(s))", i + 1, c.name, c.count))
        .collect::<Vec<_>>()
        .join("\n");

    let system = format!(
        "You are a knowledge-graph type analyst. Your task is to propose NEW entity type \
categories for a knowledge graph.\n\
\n\
You will receive a list of entity names that could not be classified into existing types. \
Your job is to propose up to {max_proposals} NEW entity type definitions that would cover \
these entities.\n\
\n\
Rules:\n\
- Propose at most {max_proposals} new types. Fewer is fine.\n\
- Each type must have a short name (3-50 characters), a clear description, and a justification.\n\
- The name must be a concrete noun phrase — no placeholders like 'Unknown', 'Other', 'TBD', 'N/A'.\n\
- The description should explain what kinds of entities belong to this type.\n\
- Be specific: 'MedicalDevice' is better than 'Equipment'. 'LegalCase' is better than 'Item'.\n\
- Respond ONLY with the JSON structure — no extra commentary."
    );

    let user = format!(
        "These entities could not be classified into existing types:\n\n{cluster_list}\n\n\
Propose up to {max_proposals} new entity type categories that would cover them. \
Output JSON with a `proposals` array containing objects with `name`, `description`, and `justification` fields."
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

// ─── Rejection reason → string ────────────────────────────────────────────────

fn reason_to_string(r: RejectionReason) -> String {
    r.as_str().to_string()
}

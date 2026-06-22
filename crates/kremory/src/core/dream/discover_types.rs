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
        proposed_type::{validate_proposed_name, DiscoveryProposalBatch, RejectionReason},
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
/// Bundled non-generic parameters for [`discover_types`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). The generic `llm: &L` stays a
/// lead positional param (brief rule 4); the remaining args bundle here.
///
/// Fields:
/// - `conn` — open SQLite connection
/// - `group_id` — the namespace (equals `namespace_to_group_id(&ns)`)
/// - `embedder` — `None` = degraded mode (anti-redundancy gate skipped)
/// - `max_proposals` — caller can override; defaults to [`MAX_PROPOSALS`]
pub(crate) struct DiscoverTypesParams<'a> {
    pub(crate) conn: &'a libsql::Connection,
    pub(crate) group_id: &'a str,
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
    pub(crate) max_proposals: usize,
}

/// Discover new entity types from catch-all entities in `group_id`.
///
/// `llm` — the chat provider (generic lead positional param).
pub(crate) async fn discover_types<L: ChatProvider>(
    llm: &L,
    params: DiscoverTypesParams<'_>,
) -> Result<DiscoveryResult> {
    let DiscoverTypesParams {
        conn,
        group_id,
        embedder,
        max_proposals,
    } = params;
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
        result
            .warnings
            .push("no embedder configured — anti-redundancy gate skipped".to_string());
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
            result.types_rejected.push((proposal, reason_str));
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
                    accept_proposal(AcceptProposalParams {
                        conn,
                        group_id,
                        model_str: &model_str,
                        proposal: &proposal,
                        catch_alls: &catch_alls,
                        desc_emb_and_embedder: None,
                        result: &mut result,
                    })
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
                    accept_proposal(AcceptProposalParams {
                        conn,
                        group_id,
                        model_str: &model_str,
                        proposal: &proposal,
                        catch_alls: &catch_alls,
                        desc_emb_and_embedder: None,
                        result: &mut result,
                    })
                    .await?;
                    continue;
                }
            };

            match anti_redundancy::check_proposal(anti_redundancy::CheckProposalParams {
                proposal_desc_emb: &desc_emb,
                proposal_name_emb: &name_emb,
                existing_type_embeddings: &existing_embeddings,
                namespace: group_id,
                model: &model_str,
            }) {
                GateOutcome::Redundant { existing_name } => {
                    let reason = format!("redundant_with:{existing_name}");
                    result.types_rejected.push((proposal, reason));
                    continue;
                }
                GateOutcome::Pass => {
                    accept_proposal(AcceptProposalParams {
                        conn,
                        group_id,
                        model_str: &model_str,
                        proposal: &proposal,
                        catch_alls: &catch_alls,
                        desc_emb_and_embedder: Some((&desc_emb, emb)),
                        result: &mut result,
                    })
                    .await?;
                }
            }
        } else {
            // Degraded mode: gate skipped, accept directly
            accept_proposal(AcceptProposalParams {
                conn,
                group_id,
                model_str: &model_str,
                proposal: &proposal,
                catch_alls: &catch_alls,
                desc_emb_and_embedder: None,
                result: &mut result,
            })
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
///
/// Bundled parameters for [`accept_proposal`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Same precedent as
/// `extraction/structured.rs:TryArmParams`. A plain field-literal struct (all
/// fields required) — no builder layer needed.
struct AcceptProposalParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    model_str: &'a str,
    proposal: &'a TypeProposal,
    catch_alls: &'a [CatchAllEntity],
    /// (proposal_desc_embedding, embedder) — `None` = degraded mode.
    desc_emb_and_embedder: Option<(&'a [f32], &'a dyn DynEmbeddingProvider)>,
    result: &'a mut DiscoveryResult,
}

async fn accept_proposal(params: AcceptProposalParams<'_>) -> Result<()> {
    let AcceptProposalParams {
        conn,
        group_id,
        model_str,
        proposal,
        catch_alls,
        desc_emb_and_embedder,
        result,
    } = params;
    let now = Utc::now().to_rfc3339();
    let discovered_by = format!("dream:llm:{model_str}");

    // Count evidence entities that belong to this proposal
    // (all catch-alls contribute to the proposal's evidence_count in the minimal impl)
    let evidence_count = catch_alls.len() as i64;

    // INSERT OR IGNORE into entity_types with Migration 014 provenance columns.
    //
    // `id` MUST be allocated explicitly: `entity_types` has a COMPOSITE primary
    // key `(group_id, id)` (migrations/defs_b.rs:207), so `id` is NOT a rowid
    // alias and does NOT auto-assign. Omitting it inserts NULL → `NOT NULL
    // constraint failed: entity_types.id`. We mirror `label_to_id_or_register`
    // (entity_types.rs:460/491): allocate `COALESCE(MAX(id),0)+1` per group_id
    // as a subquery inside the INSERT (atomic at statement level). New custom
    // types land above any seeded range; id=0 catch-all is never touched. On a
    // name clash, UNIQUE(group_id, name) triggers OR IGNORE and the SELECT-back
    // below returns the existing id. (Fixes TD-051.)
    conn.execute(
        "INSERT OR IGNORE INTO entity_types \
         (group_id, id, name, description, discovered_at, discovered_by, evidence_count, confidence) \
         VALUES (?1, (SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1), \
                 ?2, ?3, ?4, ?5, ?6, ?7)",
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
        retype_evidence_by_similarity(RetypeBySimilarityParams {
            conn,
            group_id,
            catch_alls,
            new_type_id: new_id,
            type_desc_emb: desc_emb,
            emb,
            now: &now,
        })
        .await?
    } else {
        retype_evidence_all(RetypeAllParams {
            conn,
            group_id,
            catch_alls,
            new_type_id: new_id,
            now: &now,
        })
        .await?
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

/// Bundled parameters for [`retype_evidence_by_similarity`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). Same precedent as
/// `AcceptProposalParams` above.
struct RetypeBySimilarityParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    catch_alls: &'a [CatchAllEntity],
    new_type_id: u32,
    type_desc_emb: &'a [f32],
    emb: &'a dyn DynEmbeddingProvider,
    now: &'a str,
}

/// Retype evidence entities with cosine ≥ 0.75 similarity to the type description.
async fn retype_evidence_by_similarity(params: RetypeBySimilarityParams<'_>) -> Result<usize> {
    let RetypeBySimilarityParams {
        conn,
        group_id,
        catch_alls,
        new_type_id,
        type_desc_emb,
        emb,
        now,
    } = params;
    let mut count = 0usize;
    for entity in catch_alls {
        // Embed the entity's stored ID (which is the normalised name)
        let entity_name_emb = match emb.embed_dyn(&entity.id).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        let sim = anti_redundancy::cosine(&entity_name_emb, type_desc_emb);
        if sim >= EVIDENCE_RETYPE_COSINE {
            retype_entity(RetypeEntityParams {
                conn,
                group_id,
                entity_id: &entity.id,
                new_type_id,
                now,
            })
            .await?;
            count += 1;
        }
    }
    Ok(count)
}

/// Bundled parameters for [`retype_evidence_all`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct RetypeAllParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    catch_alls: &'a [CatchAllEntity],
    new_type_id: u32,
    now: &'a str,
}

/// Retype all evidence catch-all entities (degraded mode: no embedder).
async fn retype_evidence_all(params: RetypeAllParams<'_>) -> Result<usize> {
    let RetypeAllParams {
        conn,
        group_id,
        catch_alls,
        new_type_id,
        now,
    } = params;
    for entity in catch_alls {
        retype_entity(RetypeEntityParams {
            conn,
            group_id,
            entity_id: &entity.id,
            new_type_id,
            now,
        })
        .await?;
    }
    Ok(catch_alls.len())
}

/// Bundled parameters for [`retype_entity`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct RetypeEntityParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    entity_id: &'a str,
    new_type_id: u32,
    now: &'a str,
}

/// Update a single entity's type to `new_type_id` with DreamPass0 provenance (D4).
async fn retype_entity(params: RetypeEntityParams<'_>) -> Result<()> {
    let RetypeEntityParams {
        conn,
        group_id,
        entity_id,
        new_type_id,
        now,
    } = params;
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

fn build_discovery_messages(
    clusters: &[NameCluster],
    max_proposals: usize,
) -> Vec<crate::core::provider::ChatMessage> {
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

// ─── TD-051 regression tests ──────────────────────────────────────────────────
//
// `accept_proposal` is the ADR-037 Pass-0 persistence step. It was buried below
// the LLM call + clustering + anti-redundancy gate, so the only test exercising
// it was the `#[ignore]`d real-LLM smoke (`tests/phase_d_pass_0.rs`) — which hid
// TD-051 (INSERT omitted `id` on a composite-PK table → runtime NOT NULL crash).
// These tests drive the REAL `accept_proposal` deterministically (no LLM, no
// embedder; empty `catch_alls` makes retype a no-op) so the persistence/id
// allocation is asserted in the default `cargo test` gate.
#[cfg(test)]
mod td051_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::schema::TemporalGraph;

    async fn accepted_type_id(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
                libsql::params![group_id, name],
            )
            .await
            .expect("select discovered type");
        rows.next()
            .await
            .expect("row iter")
            .expect("discovered-type row must exist — INSERT must not have crashed")
            .get::<i64>(0)
            .expect("id column")
    }

    /// First discovered type allocates `id = 11` — above the seeded range (0..=10).
    /// Before the TD-051 fix this panicked with `NOT NULL constraint failed`.
    #[tokio::test]
    async fn accept_proposal_allocates_id_above_seed_range() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g1")
            .await
            .expect("seed defaults 0..=10");

        let proposal = TypeProposal {
            name: "Vehicle".to_string(),
            description: "A car, truck, or other conveyance.".to_string(),
            justification: "Several catch-all entities were vehicles.".to_string(),
        };
        let mut result = DiscoveryResult::default();

        accept_proposal(AcceptProposalParams {
            conn: &conn,
            group_id: "g1",
            model_str: "test-model",
            proposal: &proposal,
            catch_alls: &[],
            desc_emb_and_embedder: None,
            result: &mut result,
        })
        .await
        .expect("accept_proposal must persist the discovered type (TD-051)");

        assert_eq!(
            accepted_type_id(&conn, "g1", "Vehicle").await,
            11,
            "first discovered type allocates id=11 (above seeded 0..=10)"
        );
        assert_eq!(result.types_accepted.len(), 1, "one type accepted");
        assert_eq!(result.types_accepted[0].name, "Vehicle");
    }

    /// Successive discoveries climb `MAX(id)+1` (11, 12) without colliding with
    /// the seeded range or the id=0 catch-all.
    #[tokio::test]
    async fn accept_proposal_increments_id_across_discoveries() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g1")
            .await
            .expect("seed");

        for (name, expected_id) in [("Vehicle", 11i64), ("Statute", 12i64)] {
            let proposal = TypeProposal {
                name: name.to_string(),
                description: format!("description for {name}"),
                justification: "j".to_string(),
            };
            let mut result = DiscoveryResult::default();
            accept_proposal(AcceptProposalParams {
                conn: &conn,
                group_id: "g1",
                model_str: "m",
                proposal: &proposal,
                catch_alls: &[],
                desc_emb_and_embedder: None,
                result: &mut result,
            })
            .await
            .expect("accept_proposal persists");
            assert_eq!(
                accepted_type_id(&conn, "g1", name).await,
                expected_id,
                "{name} must allocate id={expected_id}"
            );
        }

        // id=0 catch-all is untouched by discovery.
        assert_eq!(accepted_type_id(&conn, "g1", "Entity").await, 0);
    }
}

// ─── TD-050 deterministic full-workflow test ──────────────────────────────────
//
// The TD-051 tests above drive `accept_proposal` (the persistence STEP) in
// isolation. The only test exercising the FULL chain (load catch-alls → cluster
// → LLM proposal → shape-validate → accept → retype evidence) was the `#[ignore]`d
// real-LLM smoke (`tests/phase_d_pass_0.rs`), which is stochastic + shape-only
// ("types_discovered is stochastic — type-shape check only"). So the SEMANTIC
// correctness of discovery — does it grow the table by the proposed type AND
// retype the catch-all evidence with `entity_type_source='DreamPass0'`? — was
// unasserted in the default `cargo test` gate (the TD-050 gap).
//
// This test closes it deterministically: a scripted `ChatProvider` returns a
// KNOWN proposal batch and `embedder = None` takes the degraded-mode path
// (anti-redundancy gate skipped → no embedding-similarity nondeterminism), so
// the OUTCOME is fully determined and asserted. No live LLM; runs in CI.
#[cfg(test)]
mod td050_full_workflow_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::provider::{
        ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;

    /// Scripted `ChatProvider` that returns one fixed discovery-proposal batch,
    /// ignoring the prompt entirely (mirrors `tests/helpers/scripted_llm.rs`).
    #[derive(Debug)]
    struct ScriptedProposalProvider {
        json: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedProposalProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: self.json.clone(),
            }))
        }

        fn model(&self) -> &str {
            "scripted-test"
        }
    }

    /// Full workflow: discover_types proposes a KNOWN type, grows `entity_types`
    /// above the seeded range, and retypes ALL catch-all evidence in-place with
    /// `entity_type_source='DreamPass0'`. Asserts the OUTCOME, not the shape.
    #[tokio::test]
    async fn discover_types_grows_table_and_retypes_evidence_deterministically() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "legal")
            .await
            .expect("seed defaults 0..=10");

        // Three out-of-vocab entities parked as catch-all (entity_type_id = 0) —
        // the residue Pass-0 discovery operates on. Minimal-column INSERT per the
        // precedent at schema.rs (id, entity_type_id, recorded_at, group_id).
        let now = Utc::now().to_rfc3339();
        for id in ["vanguard therapeutics", "acme capital", "nexus ventures"] {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![id.to_string(), now.clone(), "legal".to_string()],
            )
            .await
            .expect("insert catch-all entity");
        }

        let llm = ScriptedProposalProvider {
            json: r#"{"proposals":[{"name":"Company","description":"A business organisation, firm, or investment fund.","justification":"Vanguard Therapeutics, Acme Capital and Nexus Ventures are all companies."}]}"#
                .to_string(),
        };

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "legal",
                embedder: None,
                max_proposals: 3,
            },
        )
        .await
        .expect("discover_types must succeed");

        // ── Outcome of the discovery result (not shape) ───────────────────────
        assert_eq!(result.types_proposed.len(), 1, "exactly one type proposed");
        assert_eq!(result.types_accepted.len(), 1, "exactly one type accepted");
        assert_eq!(
            result.types_accepted[0].name, "Company",
            "the accepted type is the one the scripted LLM proposed"
        );
        assert!(
            result.types_rejected.is_empty(),
            "'Company' is a valid name — must not be rejected, got {:?}",
            result.types_rejected
        );
        assert_eq!(
            result.entities_retyped, 3,
            "all 3 catch-all entities retyped in degraded mode"
        );

        // ── entity_types table grew by the KNOWN type, id above seeded range ──
        let new_id: i64 = {
            let mut rows = conn
                .query(
                    "SELECT id FROM entity_types WHERE group_id = 'legal' AND name = 'Company'",
                    (),
                )
                .await
                .expect("query discovered type");
            rows.next()
                .await
                .expect("row iter")
                .expect("'Company' row must exist — discovery must have persisted it")
                .get::<i64>(0)
                .expect("id column")
        };
        assert!(
            new_id > 10,
            "discovered type id {new_id} must be above the seeded 0..=10 range"
        );

        // ── every catch-all entity retyped to the new id with DreamPass0 ──────
        let mut rows = conn
            .query(
                "SELECT entity_type_id, entity_type_source FROM entities \
                 WHERE group_id = 'legal'",
                (),
            )
            .await
            .expect("query retyped entities");
        let mut checked = 0usize;
        while let Some(r) = rows.next().await.expect("row iter") {
            let tid: i64 = r.get(0).expect("entity_type_id");
            let src: String = r.get(1).expect("entity_type_source");
            assert_eq!(
                tid, new_id,
                "every catch-all entity must be retyped to the discovered type id"
            );
            assert_eq!(
                src, "DreamPass0",
                "retype provenance must be 'DreamPass0' (ADR-037 D4)"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 3,
            "all 3 entities present and retyped — none left at id=0"
        );
    }
}

// ─── TD-050 real-LLM discovery-OUTCOME test ───────────────────────────────────
//
// Closes the gaps the existing `#[ignore]`d smoke (`tests/phase_d_pass_0.rs`)
// leaves open: that smoke (a) can pass VACUOUSLY (if stochastic extraction
// produced zero catch-alls, discovery early-returns and every assertion still
// passes), and (b) never asserts that evidence was RETYPED. This test drives a
// REAL model through the discovery path with a DETERMINISTIC trigger:
//
// - The catch-all bucket is seeded directly (5 drugs) — discovery cannot run
//   vacuously; the trigger is asserted to exist before the call.
// - Drugs are a type genuinely ABSENT from the 11 defaults, so a competent
//   model's proposal is NOT (correctly) rejected by anti-redundancy.
// - `embedder = None` skips the gate and takes `retype_evidence_all`, making the
//   RETYPE COUNT deterministic. The only stochastic element is "did the real
//   model propose >=1 valid type for an unambiguous drug cluster" — which is
//   exactly the discovery-quality signal this test exists to surface (and is
//   reliable for gemma4-e2b on a clear cluster).
//
// `#[ignore]` + `feature = "llm-integration"`: needs live Ollama. Run with:
//   OLLAMA_CHAT_MODEL=gemma4:e4b cargo test -p kremory \
//     --features llm-integration --lib discover_types_real_llm -- --ignored --nocapture
//
// MODEL TIER (load-bearing finding, 2026-06-22): defaults to `gemma4:e4b` — the
// DEFERRED-phase QUALITY model (90%, Phase 2 per tests/llm_integration.rs:1-25),
// NOT the interactive `gemma4-e2b`. Discovery is a background/quality task. The
// smoke run that built this test showed `gemma4-e2b` proposing a placeholder
// name `"..."` → rejected (`ellipsis_placeholder`) → ZERO types discovered,
// while `gemma4:e4b` proposes "Over-the-Counter Pain Reliever" → accepted → all
// evidence retyped. A consumer wiring only the fast interactive model for dreams
// gets silent zero-discovery. See [[project_dream_discovery_needs_deferred_quality_model]].
//
// Governed by ADR-037 (Pass-0 discovery, §9.6 D4 provenance). Complements the
// deterministic `td050_full_workflow_tests` (scripted proposal) by proving the
// REAL model end of the chain.
#[cfg(all(test, feature = "llm-integration"))]
mod td050_real_llm_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::schema::TemporalGraph;
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use std::sync::Arc;

    async fn count_catch_alls(conn: &libsql::Connection, group_id: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE group_id = ?1 AND entity_type_id = 0",
                libsql::params![group_id],
            )
            .await
            .expect("count catch-alls");
        rows.next()
            .await
            .expect("row")
            .expect("count row")
            .get::<i64>(0)
            .expect("count col")
    }

    async fn type_id_by_name(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
                libsql::params![group_id, name],
            )
            .await
            .expect("select type id");
        rows.next()
            .await
            .expect("row")
            .expect("discovered-type row must exist")
            .get::<i64>(0)
            .expect("id col")
    }

    #[tokio::test]
    #[ignore]
    async fn discover_types_real_llm_proposes_accepts_and_retypes() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "med")
            .await
            .expect("seed defaults 0..=10");

        // Deterministic catch-all trigger: 5 drugs (a type ABSENT from the 11
        // defaults). entity_type_id = 0 = catch-all.
        let now = Utc::now().to_rfc3339();
        let drugs = [
            "aspirin",
            "ibuprofen",
            "paracetamol",
            "metformin",
            "atorvastatin",
        ];
        for d in drugs {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![d.to_string(), now.clone(), "med".to_string()],
            )
            .await
            .expect("insert catch-all entity");
        }

        // Non-vacuous guarantee: the discovery trigger MUST exist.
        assert_eq!(
            count_catch_alls(&conn, "med").await,
            5,
            "5 catch-all entities must exist before discovery — guards against a vacuous pass"
        );

        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        // Deferred-phase QUALITY model (Phase 2, 90% per tests/llm_integration.rs:1-25).
        // gemma4-e2b (interactive) is too weak for discovery — see module doc.
        let chat_model =
            std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());
        let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
            .base_url(&base_url)
            .model(&chat_model)
            .timeout_seconds(120)
            .keep_alive("1h")
            .build()
            .expect("Ollama LLM builder must succeed");

        // embedder = None → anti-redundancy gate skipped + retype_evidence_all
        // (deterministic retype count). Discovery itself is fully real.
        let result = discover_types(
            &*llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "med",
                embedder: None,
                max_proposals: 3,
            },
        )
        .await
        .expect("discover_types must not error with a live model");

        // Surface what the real model actually discovered (operator observability).
        eprintln!(
            "[td050-real-llm] model={chat_model} proposed={:?} accepted={:?} rejected={:?} retyped={}",
            result
                .types_proposed
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            result
                .types_accepted
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            result
                .types_rejected
                .iter()
                .map(|(t, r)| format!("{}:{r}", t.name))
                .collect::<Vec<_>>(),
            result.entities_retyped,
        );

        // Gap: the real model actually produced a usable proposal (not vacuous,
        // not scripted). Reliable for an unambiguous drug cluster; if this flaps,
        // that IS the discovery-quality signal this test surfaces.
        assert!(
            !result.types_proposed.is_empty(),
            "real model must propose >=1 type for an unambiguous Drug cluster"
        );
        assert!(
            !result.types_accepted.is_empty(),
            "the proposal must survive the shape validator and be accepted; rejected={:?}",
            result.types_rejected
        );

        // Gap: evidence retyped (deterministic in degraded mode → all 5).
        assert_eq!(
            result.entities_retyped, 5,
            "degraded-mode accept retypes ALL catch-all evidence"
        );

        // Consistency: every accepted type persisted above the seeded range.
        for t in &result.types_accepted {
            let id = type_id_by_name(&conn, "med", &t.name).await;
            assert!(
                id > 10,
                "discovered type '{}' must allocate id>10 (above seeded 0..=10), got {id}",
                t.name
            );
        }

        // Retype provenance: every drug entity now non-catch-all with DreamPass0.
        let mut rows = conn
            .query(
                "SELECT entity_type_id, entity_type_source FROM entities WHERE group_id = 'med'",
                (),
            )
            .await
            .expect("query retyped entities");
        let mut n = 0usize;
        while let Some(r) = rows.next().await.expect("row") {
            let tid: i64 = r.get(0).expect("entity_type_id");
            let src: String = r.get(1).expect("entity_type_source");
            assert!(
                tid > 10,
                "every drug entity must be retyped above the seeded range, got {tid}"
            );
            assert_eq!(src, "DreamPass0", "retype provenance must be 'DreamPass0'");
            n += 1;
        }
        assert_eq!(n, 5, "all 5 drug entities retyped — none left at id=0");
    }
}

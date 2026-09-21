use chrono::Utc;
use metrics::counter;

use crate::core::{
    dream::anti_redundancy,
    error::Result,
    provider::DynEmbeddingProvider,
};

use super::cluster::CatchAllEntity;
use super::types::{DiscoveryResult, TypeProposal};

/// Cosine threshold for in-place evidence retype.
const EVIDENCE_RETYPE_COSINE: f32 = 0.75;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Persist an accepted proposal and retype its evidence entities.
///
/// Uses Migration 014's provenance columns (`discovered_at`, `discovered_by`,
/// `evidence_count`, `confidence`).
///
/// Bundled parameters for [`accept_proposal`] — args-as-object
/// (rust-conventions §too_many_arguments). Same precedent as
/// `extraction/structured.rs:TryArmParams`. A plain field-literal struct (all
/// fields required) — no builder layer needed.
pub(super) struct AcceptProposalParams<'a> {
    pub(super) conn: &'a libsql::Connection,
    pub(super) group_id: &'a str,
    pub(super) model_str: &'a str,
    pub(super) proposal: &'a TypeProposal,
    pub(super) catch_alls: &'a [CatchAllEntity],
    /// (proposal_desc_embedding, embedder) — `None` = degraded mode.
    pub(super) desc_emb_and_embedder: Option<(&'a [f32], &'a dyn DynEmbeddingProvider)>,
    /// `DreamOpts::include_evidence_retype_by_similarity` (default
    /// `false`). Gates ONLY the `desc_emb_and_embedder = Some(..)` cosine-only
    /// retype path (see module docs); irrelevant when `desc_emb_and_embedder`
    /// is `None` (degraded mode always uses `retype_evidence_all`, unaffected).
    pub(super) evidence_retype_by_similarity: bool,
    pub(super) result: &'a mut DiscoveryResult,
}

pub(super) async fn accept_proposal(params: AcceptProposalParams<'_>) -> Result<()> {
    let AcceptProposalParams {
        conn,
        group_id,
        model_str,
        proposal,
        catch_alls,
        desc_emb_and_embedder,
        evidence_retype_by_similarity,
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
    // below returns the existing id.
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
    // If embedder available AND `evidence_retype_by_similarity` opted in: retype
    // only evidence entities whose name embedding is cosine ≥ 0.75 to the new
    // type's description embedding. This guard (default OFF): this
    // bare-name-vs-type-description comparison is an unspiked degeneracy risk
    // (module docs) — when the flag is off, evidence entities are left as
    // catch-all here; Pass 2 `reclassify` (LLM + confidence-gated) picks them up
    // safely in the same `mem.dream()` call.
    //
    // If no embedder (degraded): retype ALL evidence entities for this type
    // (accepted type's name is the only signal) — unaffected by the flag.

    let retyped_count = if let Some((desc_emb, emb)) = desc_emb_and_embedder {
        if evidence_retype_by_similarity {
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
            counter!(
                "kremory.dream.evidence_retype_by_similarity_skipped_total",
                "namespace" => group_id.to_string()
            )
            .increment(1);
            tracing::debug!(
                target: "kremory::dream::discover_types",
                group_id = %group_id,
                type_name = %proposal.name,
                new_id = new_id,
                "discover_types: skipping cosine-only evidence retype \
                 (DreamOpts::include_evidence_retype_by_similarity is false); \
                 evidence entities remain catch-all for Pass 2 reclassify"
            );
            0
        }
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

/// Bundled parameters for [`retype_evidence_by_similarity`] — args-as-object
/// (rust-conventions §too_many_arguments). Same precedent as
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
///
/// **TD-123 verdict (2026-09-15): verdict-cleared for precision, NOT guarded.**
/// This compares an entity NAME embedding to a type DESCRIPTION embedding —
/// structurally different from the four `names_lexically_compatible`-guarded
/// sites (name-vs-name), so that helper does not apply here; a token-Jaccard
/// check between a bare name and a prose description would reject nearly
/// every legitimate pair by construction, not just the bad ones.
///
/// Empirically spiked against the real `nomic-embed-text` embedder (the
/// shipped default) with 10 adversarial name/description pairs drawn from
/// `DEFAULT_ENTITY_TYPES`' own real descriptions (e.g. `Boston` vs `Person`,
/// `Alice Chen` vs `Organisation`): worst-case cosine was **0.4922**, a 0.26
/// margin below this threshold — no evidence of the anisotropic
/// same-embedder-different-comparison-shape degeneracy that made bare
/// name-vs-name cosine unusable (TD-097/098, `cos(Ria,Morocco)=1.0000`). So
/// the false-positive (precision) risk TD-123 was auditing for is not
/// observed here.
///
/// **Separate, NOT-fixed finding surfaced by the same spike, filed as
/// TD-256**: none of 7 LEGITIMATE pairs cleared 0.75 either (best:
/// `Acme Corp` vs `Organisation` = 0.6604) — the threshold may be
/// miscalibrated for this comparison shape under this embedder, silently
/// dropping evidence retypes rather than wrongly accepting them. A recall
/// problem, not the precision problem this function's guard-audit covers;
/// out of scope for TD-123, tracked separately.
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

/// Bundled parameters for [`retype_evidence_all`] — args-as-object
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

/// Bundled parameters for [`retype_entity`] — args-as-object
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


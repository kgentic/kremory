use chrono::Utc;

use crate::core::provider::{ChatProvider, EmbeddingProvider};

use crate::core::graph::{EpisodeInsert, InsertEpisodicEdgeParams};
use crate::core::ingest::{Engine, SourceParams};

use super::types::{EntityCandidate, IngestPhase1Result, ResolvedDecision, UpsertedEntities};

// EntityExtractor trait must be in scope for the `gliner.extract(..)` call in the
// `ner`-gated branch below; unused (and thus removed) on the default feature set.
#[cfg(feature = "ner")]
use crate::core::intelligence::EntityExtractor;

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Phase 1 ingest: store episode + run NER, return candidates.
    ///
    /// Writes exactly ONE row to `episodes` and ZERO rows to `entities`. Entity
    /// writes are deferred to [`write_verified_entities`] after the verify stage
    /// has resolved the NER candidates.
    ///
    /// Per C6 spec §5.5 — the hot path (`ingest_phase1_ner`) must be callable
    /// without an LLM (NER is GLiNER or empty-candidates; no LLM call fires here).
    /// Using `MockChatProvider::null()` with this method is safe and correct.
    ///
    /// # Returns
    /// `IngestPhase1Result { episode_id, candidates }` where `episode_id > 0` and
    /// `candidates` is a (possibly empty) Vec of NER span extractions.
    /// Phase 1 ingest entry point — see full doc on the function above.
    // Phase A: pub so integration tests (external crates) can call it.
    // Phase B: verify_stage.rs calls this from within the crate (pub(crate) would suffice
    // then, but pub is required now for the integration test boundary).
    pub async fn ingest_phase1_ner(
        &self,
        text: &str,
        source_params: SourceParams,
    ) -> crate::core::error::Result<IngestPhase1Result> {
        let ref_time = Utc::now();

        // 1. Insert episode row — same path as ingest_with Step 1.
        let mut episode = EpisodeInsert::new(text, ref_time).source_type("ingest_phase1_ner");
        if let Some(source_id) = source_params.source_id.as_deref() {
            episode = episode.source_id(source_id);
        }
        if let Some(source_uri) = source_params.source_uri.as_deref() {
            episode = episode.source_uri(source_uri);
        }
        if let Some(recorded_at) = source_params.recorded_at {
            episode = episode.recorded_at(recorded_at);
        }
        let episode_id = self.graph.insert_episode_with_group(episode, None).await?;

        // 2. NER only — no LLM, no entity writes.
        //    When the `ner` feature is active, run GLiNER to obtain span candidates.
        //    Without the feature, return empty candidates (caller proceeds to verify
        //    with nothing to verify, which is correct — no entities, no verify needed).
        #[cfg(feature = "ner")]
        let candidates = {
            // MNT-001: use the process-wide singleton from ner::ner_singleton() so
            // both ingest_phase1_ner and ingest_with share a single loaded model
            // (~650 MB INT8). Previously each path had its own OnceLock, risking
            // double model load (~1.3 GB) when both paths were used in the same process.
            let gliner = crate::core::ner::ner_singleton()?;

            // Load registry for entity_type_id resolution.
            let effective_gid = "default";
            crate::core::entity_types::ensure_default_types_seeded(&self.graph.conn, effective_gid)
                .await?;
            let registry = crate::core::entity_types::EntityTypeRegistry::load_for_group(
                &self.graph.conn,
                effective_gid,
            )
            .await?;

            let allowed: Vec<String> = registry.specs().iter().map(|s| s.name.clone()).collect();
            let ctx = crate::core::intelligence::ExtractionContext {
                allowed_entity_types: &allowed,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &[],
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: crate::core::config::ContentType::Text,
                registry_specs: registry.specs(),
                existing_graph_entities: &[],
                arm_budget_ms: self.config.extraction_arm_budget_ms,
            };
            let extraction = gliner.extract(text, &ctx).await?;
            extraction
                .entities
                .into_iter()
                .map(|e| {
                    // Resolve entity_type_id from label; fall back to 0 (catch-all).
                    let type_id = registry
                        .name_to_id(&e.label)
                        .map(|id| id as i64)
                        .unwrap_or(0i64);
                    let confidence = e
                        .properties
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .map(|c| c as f32)
                        .unwrap_or(0.0_f32);
                    EntityCandidate {
                        name: e.name,
                        entity_type_id_raw: type_id,
                        ner_confidence: confidence,
                        span: (0, 0), // GLiNER span resolution not needed for Phase A
                    }
                })
                .collect::<Vec<_>>()
        };

        #[cfg(not(feature = "ner"))]
        let candidates: Vec<EntityCandidate> = Vec::new();

        Ok(IngestPhase1Result {
            episode_id,
            candidates,
        })
    }

    /// Write entity rows for the given NER candidates based on verify decisions.
    ///
    /// Per C6 spec §5.5 decision semantics:
    /// - `Confirm { candidate_idx }` → write with `candidate.entity_type_id_raw`
    /// - `Correct { candidate_idx, new_type_id }` → write with `new_type_id`
    /// - `Demote { candidate_idx }` → write with `entity_type_id = 0` (catch-all)
    ///
    /// Entity ids are derived via `normalize_name(candidate.name)` matching the
    /// existing pipeline convention. Duplicate entity ids are silently skipped
    /// (INSERT OR IGNORE semantics — entity already exists from a prior ingest).
    ///
    /// # Parameters
    /// - `episode_id` — FK linking entities to their source episode (from Phase 1)
    /// - `candidates` — NER candidates produced by `ingest_phase1_ner`
    /// - `decisions` — verify decisions, one per relevant candidate
    // Phase A: pub so integration tests (external crates) can call it.
    pub async fn write_verified_entities(
        &self,
        episode_id: i64,
        candidates: &[EntityCandidate],
        decisions: &[ResolvedDecision],
    ) -> crate::core::error::Result<UpsertedEntities> {
        use crate::core::resolver::normalize_name;

        let now = Utc::now().to_rfc3339();
        let mut result = UpsertedEntities::default();

        for decision in decisions {
            let (candidate_idx, entity_type_id) = match decision {
                ResolvedDecision::Confirm { candidate_idx } => {
                    let candidate = &candidates[*candidate_idx];
                    (*candidate_idx, candidate.entity_type_id_raw)
                }
                ResolvedDecision::Correct {
                    candidate_idx,
                    new_type_id,
                } => (*candidate_idx, *new_type_id),
                ResolvedDecision::Demote { candidate_idx } => (*candidate_idx, 0i64),
            };

            let candidate = &candidates[candidate_idx];
            let entity_id = normalize_name(&candidate.name);
            let props = serde_json::json!({ "name": candidate.name });
            let props_str = serde_json::to_string(&props)
                .map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!("{e}")))?;

            // INSERT OR IGNORE — if entity already exists from a prior ingest,
            // skip silently. Per ingest_with pattern: first-mention-wins.
            self.graph
                .conn
                .execute(
                    "INSERT OR IGNORE INTO entities \
                     (id, entity_type_id, properties, recorded_at, group_id, \
                      entity_type_source, entity_type_assigned_at, ner_confidence) \
                     VALUES (?1, ?2, ?3, ?4, 'default', 'Phase1Ner', ?4, ?5)",
                    libsql::params![
                        entity_id.clone(),
                        entity_type_id,
                        props_str.clone(),
                        now.clone(),
                        candidate.ner_confidence as f64
                    ],
                )
                .await
                .map_err(|e| {
                    crate::core::error::Error::Other(anyhow::anyhow!(
                        "write_verified_entities INSERT entity '{}': {e}",
                        entity_id
                    ))
                })?;

            // FTS insert (entities_fts row required per graph.rs invariant).
            // INSERT OR IGNORE — mirrors the FTS insert in insert_entity.
            self.graph
                .conn
                .execute(
                    "INSERT OR IGNORE INTO entities_fts(entity_id, label, properties) \
                     VALUES (?1, '', ?2)",
                    libsql::params![entity_id.clone(), props_str],
                )
                .await
                .map_err(|e| {
                    crate::core::error::Error::Other(anyhow::anyhow!(
                        "write_verified_entities FTS INSERT entity '{}': {e}",
                        entity_id
                    ))
                })?;

            // Episodic edge: link entity to its source episode.
            // `write_verified_entities` persists entities under `group_id =
            // 'default'` (the INSERT above), so the edge MUST reference the same
            // namespace for the Migration 006 composite FK to resolve.
            // `None` ⇒ `'default'`.
            self.graph
                .insert_episodic_edge(InsertEpisodicEdgeParams {
                    episode_id,
                    entity_id: &entity_id,
                    entity_group_id: None,
                    role: "mention",
                })
                .await
                .ok();

            metrics::counter!(
                "rql.ingest.entity_persisted_total",
                "source" => "phase1_verified",
            )
            .increment(1);

            result.entity_ids.push(entity_id);
        }

        Ok(result)
    }
}

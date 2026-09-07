// ─── Foundational NER-candidate types ────────────────────────────────────────

/// A NER candidate returned by Phase 1 (`ingest_phase1_ner`).
///
/// Represents a single entity span extracted by NER (GLiNER when the `ner`
/// feature is active, or empty Vec without it). Passed to Stage 2 verify and
/// then to `write_verified_entities` after decisions are resolved.
///
/// Callers MUST NOT write entity rows until after
/// `write_verified_entities` is called with the resolved decisions.
#[derive(Debug, Clone)]
pub struct EntityCandidate {
    /// Entity surface form as extracted by NER.
    pub name: String,
    /// Raw entity type id assigned by the NER stage. For GLiNER, this is the
    /// integer id from the entity type registry. `0` = catch-all ("Entity").
    pub entity_type_id_raw: i64,
    /// NER span confidence score. Populated by GLiNER; 0.0 on non-NER paths.
    pub ner_confidence: f32,
    /// Byte-offset span `(start, end)` of the entity mention in the source text.
    pub span: (usize, usize),
}

/// Result of `ingest_phase1_ner`: episode stored + NER candidates surfaced.
///
/// Contains exactly one episode row (already written to the DB) and zero
/// entity rows — entity writes are deferred to `write_verified_entities`.
///
/// `episode_id` is the FK used by `write_verified_entities`
/// to link written entities back to their source episode.
#[derive(Debug)]
pub struct IngestPhase1Result {
    /// Id of the episode row written by `ingest_phase1_ner`.
    pub episode_id: i64,
    /// NER candidates found in the episode text. May be empty when the `ner`
    /// feature is not active or when no candidates are found.
    pub candidates: Vec<EntityCandidate>,
}

/// A decision about a single `EntityCandidate` from the verify stage.
///
/// Used by `write_verified_entities` to determine which entity_type_id to
/// write for each candidate:
/// - `Confirm` — write with `candidate.entity_type_id_raw` (NER was correct)
/// - `Correct` — write with `new_type_id` (verify corrected the NER type)
/// - `Demote` — write with `entity_type_id = 0` (catch-all; NER was wrong,
///   no better type known)
#[derive(Debug, Clone)]
pub enum ResolvedDecision {
    /// NER type was correct — persist `candidate.entity_type_id_raw` verbatim.
    Confirm {
        /// Index into the candidates slice passed to `write_verified_entities`.
        candidate_idx: usize,
    },
    /// NER type was wrong — persist `new_type_id` instead.
    Correct {
        /// Index into the candidates slice.
        candidate_idx: usize,
        /// Verified correct entity type id from the registry.
        new_type_id: i64,
    },
    /// NER type was wrong and no better type is known — demote to catch-all (id=0).
    Demote {
        /// Index into the candidates slice.
        candidate_idx: usize,
    },
}

/// Entities written by `write_verified_entities`.
#[derive(Debug, Default)]
pub struct UpsertedEntities {
    /// Normalized entity ids (= `normalize_name(candidate.name)`) that were
    /// successfully written to the `entities` table.
    pub entity_ids: Vec<String>,
}

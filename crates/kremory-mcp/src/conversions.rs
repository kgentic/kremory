//! Bidirectional conversions between MCP wire types (`params.rs`) and
//! `kremory` facade types.
//!
//! ## Error surface
//!
//! Conversions from wire → facade can fail on:
//! - Empty `namespace`
//! - Malformed RFC 3339 timestamps (`published_at`, `structured_facts[].valid_at`
//!   / `invalid_at`, `as_of`)
//!
//! These map to [`ConversionError`], which `handlers.rs` maps to
//! `crate::handlers::ToolError::InvalidParams` — `lib.rs` then converts that
//! to MCP `ErrorData::invalid_params` (and the REST bin maps it to HTTP 422).
//! `source_kind` / recall `format` / `template` are typed wire enums (see
//! `params.rs`) — an unknown string there is rejected by `Parameters<T>`'s
//! own JSON-schema deserialization before it ever reaches this module, so no
//! `ConversionError` variant is needed for them.
//!
//! ## facade → wire
//!
//! Always infallible — facade enum variants map 1:1 to wire strings/enums;
//! `DateTime<Utc>` always renders to RFC 3339.

use chrono::{DateTime, Utc};
use kremory::{
    DeleteEntityOutcome, DeleteFactOutcome, EditEntityOutcome, MutationKind, MutationRecord,
    Namespace, RecallTemplate, RetrievedContext, RetrievedFact, SourceKind, SourceRef,
    StructuredFact, UndoOutcome, UnmergeOutcome,
};
use thiserror::Error;

use crate::params::{
    ConsolidationOpsRanWire, DeleteEntityOutcomeWire, DeleteFactOutcomeWire, DreamOutput,
    DreamParams, EditEntityOutcomeWire, ListMutationsParams, MutationRecordWire, RecallFormat,
    RecallParams, RecallTemplateWire, RememberOutput, RememberParams, RetrievedContextWire,
    RetrievedFactWire, SourceKindWire, SourceRefWire, StructuredFactWire, TypeProposalWire,
    UndoOutcomeWire, UndoParams, UnmergeOutcomeWire,
};

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("namespace must not be empty")]
    EmptyNamespace,
    #[error("malformed RFC 3339 timestamp in {field}: {value:?} ({source})")]
    Timestamp {
        field: &'static str,
        value: String,
        #[source]
        source: chrono::ParseError,
    },
    #[error(
        "unknown mutation kind {raw:?} — expected one of: entity_merge, fact_supersede, \
         fact_archive, entity_edit, entity_delete, fact_delete, community_assign, canonical_form"
    )]
    UnknownMutationKind { raw: String },
}

// ─── primitive conversions ─────────────────────────────────────────────

pub(crate) fn source_kind_wire_to_facade(k: SourceKindWire) -> SourceKind {
    match k {
        SourceKindWire::Document => SourceKind::Document,
        SourceKindWire::Chat => SourceKind::Chat,
        // `RememberRequest::from_note` maps to `SourceKind::Document` at the
        // facade too — "note" is a caller-facing synonym, not a distinct
        // substrate kind.
        SourceKindWire::Note => SourceKind::Document,
    }
}

fn source_kind_facade_to_wire(k: SourceKind) -> String {
    match k {
        SourceKind::Meeting => "meeting".to_string(),
        SourceKind::Document => "document".to_string(),
        SourceKind::Chat => "chat".to_string(),
        SourceKind::Episode => "episode".to_string(),
        // `SourceKind` is `#[non_exhaustive]` in kremory so it can add
        // variants without a SemVer break. This arm is forward-compat, not
        // an expected runtime path — surface it loudly rather than silently
        // mislabeling a future variant.
        other => {
            tracing::warn!(
                ?other,
                "unknown SourceKind variant (kremory added a new variant?) — \
                 wire output defaulting to debug format"
            );
            format!("{other:?}").to_lowercase()
        }
    }
}

/// Parse the `kremory_list_mutations` wire `kind` string to `kremory::MutationKind`
/// (parse-loudly, [[llm-output-parse-loudly]] extended to caller-supplied wire
/// strings — an unrecognised kind is a hard `invalid_params`, never silently
/// dropped or defaulted to "any kind").
pub(crate) fn mutation_kind_wire_to_facade(raw: &str) -> Result<MutationKind, ConversionError> {
    match raw {
        "entity_merge" => Ok(MutationKind::EntityMerge),
        "fact_supersede" => Ok(MutationKind::FactSupersede),
        "fact_archive" => Ok(MutationKind::FactArchive),
        "entity_edit" => Ok(MutationKind::EntityEdit),
        "entity_delete" => Ok(MutationKind::EntityDelete),
        "fact_delete" => Ok(MutationKind::FactDelete),
        "community_assign" => Ok(MutationKind::CommunityAssign),
        "canonical_form" => Ok(MutationKind::CanonicalForm),
        other => Err(ConversionError::UnknownMutationKind {
            raw: other.to_string(),
        }),
    }
}

/// `kremory::MutationKind` (`#[non_exhaustive]`) → the wire snake_case tag.
/// Mirrors `source_kind_facade_to_wire`'s forward-compat posture: known
/// variants map to their exact `graph_mutation_log.kind` tag; an unrecognised
/// future variant (kremory added a 9th kind) falls back to the debug format
/// rather than panicking on the mandatory `#[non_exhaustive]` wildcard arm.
fn mutation_kind_facade_to_wire(kind: MutationKind) -> String {
    match kind {
        MutationKind::EntityMerge => "entity_merge".to_string(),
        MutationKind::FactSupersede => "fact_supersede".to_string(),
        MutationKind::FactArchive => "fact_archive".to_string(),
        MutationKind::EntityEdit => "entity_edit".to_string(),
        MutationKind::EntityDelete => "entity_delete".to_string(),
        MutationKind::FactDelete => "fact_delete".to_string(),
        MutationKind::CommunityAssign => "community_assign".to_string(),
        MutationKind::CanonicalForm => "canonical_form".to_string(),
        other => {
            tracing::warn!(
                ?other,
                "unknown MutationKind variant (kremory added a new variant?) — \
                 wire output defaulting to debug format"
            );
            format!("{other:?}").to_lowercase()
        }
    }
}

pub(crate) fn recall_template_wire_to_facade(t: RecallTemplateWire) -> RecallTemplate {
    match t {
        RecallTemplateWire::Entities => RecallTemplate::Entities,
        RecallTemplateWire::EdgeSummary => RecallTemplate::EdgeSummary,
        RecallTemplateWire::TemporalFacts => RecallTemplate::TemporalFacts,
    }
}

pub(crate) fn parse_iso8601(
    field: &'static str,
    raw: &str,
) -> Result<DateTime<Utc>, ConversionError> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ConversionError::Timestamp {
            field,
            value: raw.to_string(),
            source: e,
        })
}

pub(crate) fn build_namespace(
    namespace: &str,
    thread: Option<&str>,
) -> Result<Namespace, ConversionError> {
    if namespace.is_empty() {
        return Err(ConversionError::EmptyNamespace);
    }
    Ok(match thread {
        Some(t) if !t.is_empty() => Namespace::new(namespace).with_thread(t),
        _ => Namespace::new(namespace),
    })
}

// ─── kremory_remember ───────────────────────────────────────────────────

/// Resolved (validated + facade-typed) form of [`RememberParams`]. Pure —
/// `lib.rs` applies this to the live `Memory::remember(...)` builder chain.
#[derive(Debug)]
pub(crate) struct ResolvedRemember {
    pub namespace: Namespace,
    pub content: String,
    pub source_kind: Option<SourceKind>,
    pub source_id: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub facts: Vec<StructuredFact>,
    pub skip_extraction: bool,
}

impl RememberParams {
    pub(crate) fn resolve(self) -> Result<ResolvedRemember, ConversionError> {
        let namespace = build_namespace(&self.namespace, self.thread.as_deref())?;
        let published_at = self
            .published_at
            .as_deref()
            .map(|raw| parse_iso8601("published_at", raw))
            .transpose()?;
        let mut facts = Vec::with_capacity(self.structured_facts.len());
        for f in self.structured_facts {
            facts.push(f.try_into_facade()?);
        }
        Ok(ResolvedRemember {
            namespace,
            content: self.content,
            source_kind: self.source_kind.map(source_kind_wire_to_facade),
            source_id: self.source_id,
            published_at,
            facts,
            skip_extraction: self.skip_extraction,
        })
    }
}

impl StructuredFactWire {
    fn try_into_facade(self) -> Result<StructuredFact, ConversionError> {
        let valid_from = self
            .valid_at
            .as_deref()
            .map(|raw| parse_iso8601("structured_facts[].valid_at", raw))
            .transpose()?;
        let valid_to = self
            .invalid_at
            .as_deref()
            .map(|raw| parse_iso8601("structured_facts[].invalid_at", raw))
            .transpose()?;
        Ok(StructuredFact {
            subject: self.subject,
            predicate: self.predicate,
            object: self.object,
            valid_from,
            valid_to,
            memory_type: None,
        })
    }
}

impl From<kremory::EpisodeCommit> for RememberOutput {
    fn from(c: kremory::EpisodeCommit) -> Self {
        Self {
            run_id: c.run_id.map(|u| u.to_string()),
            episode_entity_id: c.episode_entity_id,
            committed_at: c.committed_at.to_rfc3339(),
            stub_entities_inserted: c.stub_entities_inserted,
        }
    }
}

// ─── kremory_recall ─────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ResolvedRecall {
    pub namespace: Namespace,
    pub query: String,
    pub k: Option<usize>,
    pub as_of: Option<DateTime<Utc>>,
    pub format: RecallFormat,
    pub template: RecallTemplate,
    /// Mirrors `RecallParams::rerank_k`.
    pub rerank_k: Option<usize>,
}

impl RecallParams {
    pub(crate) fn resolve(self) -> Result<ResolvedRecall, ConversionError> {
        let namespace = build_namespace(&self.namespace, self.thread.as_deref())?;
        let as_of = self
            .as_of
            .as_deref()
            .map(|raw| parse_iso8601("as_of", raw))
            .transpose()?;
        Ok(ResolvedRecall {
            namespace,
            query: self.query,
            k: self.k,
            as_of,
            format: self.format,
            template: recall_template_wire_to_facade(self.template),
            rerank_k: self.rerank_k,
        })
    }
}

impl From<RetrievedContext> for RetrievedContextWire {
    fn from(r: RetrievedContext) -> Self {
        Self {
            entity_id: r.entity_id,
            entity_name: r.entity_name,
            summary: r.summary,
            score: r.score,
            incomplete: r.incomplete,
            entity_type_id: r.entity_type_id,
            entity_type_name: r.entity_type_name,
            namespace: r.namespace.as_ref().map(|ns| ns.namespace.clone()),
            source_refs: r.source_refs.into_iter().map(SourceRefWire::from).collect(),
            facts: r.facts.into_iter().map(RetrievedFactWire::from).collect(),
        }
    }
}

impl From<RetrievedFact> for RetrievedFactWire {
    fn from(f: RetrievedFact) -> Self {
        Self {
            fact_id: f.fact_id,
            fact: f.fact,
            subject: f.subject,
            predicate: f.predicate,
            object: f.object,
            object_is_entity: f.object_is_entity,
            valid_at: f.valid_at.to_rfc3339(),
            invalid_at: f.invalid_at.map(|d| d.to_rfc3339()),
            recorded_at: f.recorded_at.to_rfc3339(),
            expired_at: f.expired_at.map(|d| d.to_rfc3339()),
            confidence: f.confidence,
            source_episode_ids: f.source_episode_ids,
            score: f.score,
        }
    }
}

impl From<SourceRef> for SourceRefWire {
    fn from(s: SourceRef) -> Self {
        Self {
            kind: source_kind_facade_to_wire(s.kind),
            id: s.id,
            occurred_at: s.occurred_at.to_rfc3339(),
            published_at: s.published_at.map(|t| t.to_rfc3339()),
        }
    }
}

// ─── kremory::memory::Renderable{Context,Fact,SourceRef} ─────────
//
// Wire-side halves of the ONE prompt-block renderer implementation in
// `kremory::memory` (`render_entities` / `render_edge_summary` /
// `render_temporal_facts`). Previously `kremory-mcp/src/bin/kremory-http.rs`
// hand-mirrored those three functions field-for-field against this crate's
// own `RetrievedContextWire` DTO — the exact two-implementations-must-agree
// shape that caused a silent `.max()` fusion regression to reappear here for
// rendering. These impls let `kremory-http.rs` call the real
// `kremory::memory::render_*` functions directly instead.

impl kremory::memory::RenderableFact for RetrievedFactWire {
    fn fact_text(&self) -> &str {
        &self.fact
    }
    fn valid_at_rfc3339(&self) -> String {
        // Already RFC-3339 on the wire type (stamped at the `From<RetrievedFact>`
        // boundary above) — no re-format needed, just clone the owned String out.
        self.valid_at.clone()
    }
    fn invalid_at_rfc3339(&self) -> Option<String> {
        self.invalid_at.clone()
    }
}

impl kremory::memory::RenderableSourceRef for SourceRefWire {
    fn kind_label(&self) -> &str {
        // Wire `kind` is already the human-readable label string (mapped by
        // `source_kind_facade_to_wire` above), not an enum — no lookup needed.
        &self.kind
    }
    fn ref_id(&self) -> &str {
        &self.id
    }
    fn occurred_at_rfc3339(&self) -> String {
        self.occurred_at.clone()
    }
}

impl kremory::memory::RenderableContext for RetrievedContextWire {
    type Fact = RetrievedFactWire;
    type SourceRef = SourceRefWire;

    fn entity_name(&self) -> &str {
        &self.entity_name
    }
    fn summary(&self) -> &str {
        &self.summary
    }
    fn namespace_group_id(&self) -> Option<&str> {
        // `/search` takes exactly one `namespace` per request (`RecallParams::
        // namespace: String`), so every item in one response necessarily
        // carries the SAME value here (or `None` for a synthesized
        // content-passage item — see `content_passage_into_context_wire` in
        // `kremory-http.rs`). The core renderers' multi-namespace `[ns:...]`
        // prefix therefore can never fire for this type: not because this
        // accessor hardcodes it away, but because the underlying data can't
        // vary within one response. Returning the real field (rather than a
        // hardcoded `None`) keeps this impl honest about what it's reading.
        self.namespace.as_deref()
    }
    fn facts(&self) -> &[RetrievedFactWire] {
        &self.facts
    }
    fn source_refs(&self) -> &[SourceRefWire] {
        &self.source_refs
    }
}

// ─── kremory_dream ──────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ResolvedDream {
    pub namespace: Namespace,
    pub batch_id: Option<String>,
}

impl DreamParams {
    pub(crate) fn resolve(self) -> Result<ResolvedDream, ConversionError> {
        let namespace = build_namespace(&self.namespace, self.thread.as_deref())?;
        Ok(ResolvedDream {
            namespace,
            batch_id: self.batch_id,
        })
    }
}

impl From<kremory::DreamSummary> for DreamOutput {
    fn from(d: kremory::DreamSummary) -> Self {
        Self {
            communities_updated: d.communities_updated,
            cross_episode_would_merge: d.cross_episode_would_merge,
            cross_episode_merged: d.cross_episode_merged,
            supersessions_recorded: d.supersessions_recorded,
            facts_archived: d.facts_archived,
            entities_reclassified: d.entities_reclassified,
            aliases_resolved: d.aliases_resolved,
            canonicalization_merges: d.canonicalization_merges,
            acronym_nickname_merges: d.acronym_nickname_merges,
            type_registry_merges: d.type_registry_merges,
            consistency_check_corrected: d.consistency_check_corrected,
            types_discovered: d
                .types_discovered
                .into_iter()
                .map(|t| TypeProposalWire {
                    name: t.name,
                    description: t.description,
                    justification: t.justification,
                })
                .collect(),
            consolidation_ops_ran: ConsolidationOpsRanWire {
                community: d.consolidation_ops_ran.community,
                cross_episode: d.consolidation_ops_ran.cross_episode,
                archival: d.consolidation_ops_ran.archival,
                supersession_sweep: d.consolidation_ops_ran.supersession_sweep,
            },
            duration_ms: d.duration_ms,
            budget_exhausted: d.budget_exhausted,
            warnings: d.warnings,
        }
    }
}

// ─── kremory_list_mutations ─────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ResolvedListMutations {
    pub namespace: Namespace,
    pub entity_id: Option<String>,
    pub kind: Option<MutationKind>,
    pub since: Option<DateTime<Utc>>,
    pub include_undone: bool,
}

impl ListMutationsParams {
    pub(crate) fn resolve(self) -> Result<ResolvedListMutations, ConversionError> {
        let namespace = build_namespace(&self.namespace, self.thread.as_deref())?;
        let kind = self
            .kind
            .as_deref()
            .map(mutation_kind_wire_to_facade)
            .transpose()?;
        let since = self
            .since
            .as_deref()
            .map(|raw| parse_iso8601("since", raw))
            .transpose()?;
        Ok(ResolvedListMutations {
            namespace,
            entity_id: self.entity_id,
            kind,
            since,
            include_undone: self.include_undone.unwrap_or(false),
        })
    }
}

impl From<MutationRecord> for MutationRecordWire {
    fn from(r: MutationRecord) -> Self {
        Self {
            mutation_id: r.mutation_id,
            kind: mutation_kind_facade_to_wire(r.kind),
            created_at: r.created_at,
            undone: r.undone,
            group_id: r.group_id,
            affected_entities: r.affected_entities,
            summary: r.summary,
        }
    }
}

// ─── kremory_undo ───────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ResolvedUndo {
    pub namespace: Namespace,
    pub mutation_id: i64,
}

impl UndoParams {
    pub(crate) fn resolve(self) -> Result<ResolvedUndo, ConversionError> {
        let namespace = build_namespace(&self.namespace, self.thread.as_deref())?;
        Ok(ResolvedUndo {
            namespace,
            mutation_id: self.mutation_id,
        })
    }
}

impl From<UnmergeOutcome> for UnmergeOutcomeWire {
    fn from(o: UnmergeOutcome) -> Self {
        Self {
            restored_entity: o.restored_entity,
            keeper: o.keeper,
            facts_repointed: o.facts_repointed,
            edges_restored: o.edges_restored,
            entities_reopened: o.entities_reopened,
            nogood_recorded: o.nogood_recorded,
            already_undone: o.already_undone,
        }
    }
}

impl From<EditEntityOutcome> for EditEntityOutcomeWire {
    fn from(o: EditEntityOutcome) -> Self {
        Self {
            entity_id: o.entity_id,
            rekeyed: o.rekeyed,
            retyped: o.retyped,
            facts_repointed: o.facts_repointed,
            archived_repointed: o.archived_repointed,
            edges_repointed: o.edges_repointed,
            communities_repointed: o.communities_repointed,
            entities_reopened: o.entities_reopened,
            mutation_id: o.mutation_id,
            already_undone: o.already_undone,
        }
    }
}

impl From<DeleteEntityOutcome> for DeleteEntityOutcomeWire {
    fn from(o: DeleteEntityOutcome) -> Self {
        Self {
            entity_id: o.entity_id,
            facts_retracted: o.facts_retracted,
            edges_removed: o.edges_removed,
            communities_removed: o.communities_removed,
            neighbors_retracted: o.neighbors_retracted,
            entities_reopened: o.entities_reopened,
            mutation_id: o.mutation_id,
            already_undone: o.already_undone,
        }
    }
}

impl From<DeleteFactOutcome> for DeleteFactOutcomeWire {
    fn from(o: DeleteFactOutcome) -> Self {
        Self {
            fact_id: o.fact_id,
            fact_restored: o.fact_restored,
            neighbors_retracted: o.neighbors_retracted,
            entities_reopened: o.entities_reopened,
            mutation_id: o.mutation_id,
            already_undone: o.already_undone,
        }
    }
}

/// `kremory::UndoOutcome` (`#[non_exhaustive]`) → the internally-tagged wire
/// enum. Fallible rather than lossy: a future 5th log-dispatchable
/// `MutationKind` would arrive as a variant this crate doesn't yet mirror,
/// and per [[llm-output-parse-loudly]] / [[treat-cause-not-symptom]] that must
/// surface as a loud error (a version-skew bug for the operator to fix), never
/// as a silently-dropped/defaulted wire payload.
impl TryFrom<UndoOutcome> for UndoOutcomeWire {
    type Error = String;

    fn try_from(outcome: UndoOutcome) -> Result<Self, Self::Error> {
        match outcome {
            UndoOutcome::Unmerge(o) => Ok(Self::Unmerge(o.into())),
            UndoOutcome::EditEntity(o) => Ok(Self::EditEntity(o.into())),
            UndoOutcome::DeleteEntity(o) => Ok(Self::DeleteEntity(o.into())),
            UndoOutcome::DeleteFact(o) => Ok(Self::DeleteFact(o.into())),
            UndoOutcome::RestoreArchived(o) => Ok(Self::RestoreArchived(
                crate::params::RestoreArchivedOutcomeWire {
                    restored_fact_id: o.restored_fact_id,
                    already_live: o.already_live,
                },
            )),
            other => Err(format!(
                "kremory_undo: UndoOutcome carries a variant kremory-mcp's UndoOutcomeWire \
                 does not yet mirror ({other:?}) — kremory added a new log-dispatchable \
                 MutationKind that this crate has not caught up with yet"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SourceKindWire;

    /// Pin the JSON an agent actually receives when it undoes a dream's fact
    /// archival.
    ///
    /// ⚠️ The CONVERSION itself cannot be unit-tested from this crate:
    /// `kremory::core::dream::provenance` is a private module and
    /// `RestoreArchivedOutcome` is `#[non_exhaustive]`, so the input value can be
    /// destructured here (which is why the `TryFrom` arm compiles) but never
    /// constructed. What is testable — and what the agent depends on — is that the
    /// variant exists and serializes under the same internally-tagged shape as its
    /// four siblings.
    ///
    /// The gap this covers was not a missing feature but a WRONG ANSWER: without
    /// the arm, `UndoOutcomeWire::try_from` fell through to its catch-all `Err`,
    /// and that conversion runs AFTER `do_undo` has already `.execute()`d. A
    /// committed, successful restore was reported as `internal_error`, so the
    /// correct agent response was to retry something that had already happened.
    #[test]
    fn the_restore_archived_wire_variant_serializes_like_its_siblings() {
        let wire = UndoOutcomeWire::RestoreArchived(crate::params::RestoreArchivedOutcomeWire {
            restored_fact_id: 42,
            already_live: false,
        });

        let json = serde_json::to_value(&wire).expect("wire type must serialize");

        assert_eq!(
            json.get("restored_fact_id").and_then(serde_json::Value::as_i64),
            Some(42),
            "the agent needs the fact id it just restored: {json}"
        );
        assert_eq!(
            json.get("already_live").and_then(serde_json::Value::as_bool),
            Some(false),
            "an idempotent no-op must stay distinguishable from a real restore: {json}"
        );
    }

    #[test]
    fn source_kind_wire_maps_note_and_document_to_facade_document() {
        assert_eq!(
            source_kind_wire_to_facade(SourceKindWire::Document),
            SourceKind::Document
        );
        assert_eq!(
            source_kind_wire_to_facade(SourceKindWire::Note),
            SourceKind::Document
        );
        assert_eq!(
            source_kind_wire_to_facade(SourceKindWire::Chat),
            SourceKind::Chat
        );
    }

    #[test]
    fn parse_iso8601_accepts_rfc3339() {
        let t = parse_iso8601("published_at", "2026-07-14T10:30:00Z").unwrap();
        assert_eq!(t.to_rfc3339().as_str()[..10], *"2026-07-14");
    }

    #[test]
    fn parse_iso8601_rejects_malformed() {
        let err = parse_iso8601("published_at", "not a date").unwrap_err();
        match err {
            ConversionError::Timestamp { field, value, .. } => {
                assert_eq!(field, "published_at");
                assert_eq!(value, "not a date");
            }
            other => panic!("expected Timestamp error, got: {other:?}"),
        }
    }

    #[test]
    fn build_namespace_rejects_empty() {
        let err = build_namespace("", None).unwrap_err();
        assert!(matches!(err, ConversionError::EmptyNamespace));
    }

    #[test]
    fn build_namespace_with_thread() {
        let ns = build_namespace("ws-1", Some("thread-a")).unwrap();
        assert_eq!(ns.namespace, "ws-1");
        assert_eq!(ns.thread.as_deref(), Some("thread-a"));
    }

    #[test]
    fn build_namespace_without_thread() {
        let ns = build_namespace("ws-1", None).unwrap();
        assert!(ns.thread.is_none());
    }

    #[test]
    fn remember_params_resolve_happy_path() {
        let p = RememberParams {
            namespace: "ws-1".into(),
            thread: None,
            content: "hello".into(),
            source_kind: Some(SourceKindWire::Note),
            source_id: Some("doc-1".into()),
            published_at: Some("2026-07-14T00:00:00Z".into()),
            structured_facts: vec![StructuredFactWire {
                subject: "alice".into(),
                predicate: "leads".into(),
                object: "design".into(),
                valid_at: None,
                invalid_at: None,
            }],
            skip_extraction: true,
        };
        let resolved = p.resolve().unwrap();
        assert_eq!(resolved.namespace.namespace, "ws-1");
        assert_eq!(resolved.source_kind, Some(SourceKind::Document));
        assert_eq!(resolved.source_id.as_deref(), Some("doc-1"));
        assert!(resolved.published_at.is_some());
        assert_eq!(resolved.facts.len(), 1);
        assert!(resolved.skip_extraction);
    }

    #[test]
    fn remember_params_resolve_rejects_bad_timestamp_in_structured_fact() {
        let p = RememberParams {
            namespace: "ws-1".into(),
            thread: None,
            content: "hello".into(),
            source_kind: None,
            source_id: None,
            published_at: None,
            structured_facts: vec![StructuredFactWire {
                subject: "s".into(),
                predicate: "p".into(),
                object: "o".into(),
                valid_at: Some("not a date".into()),
                invalid_at: None,
            }],
            skip_extraction: false,
        };
        let err = p.resolve().unwrap_err();
        match err {
            ConversionError::Timestamp { field, .. } => {
                assert_eq!(field, "structured_facts[].valid_at")
            }
            other => panic!("expected Timestamp error, got: {other:?}"),
        }
    }

    #[test]
    fn recall_params_resolve_defaults() {
        let p = RecallParams {
            namespace: "ws-1".into(),
            thread: Some("t-1".into()),
            query: "go-live".into(),
            k: Some(5),
            as_of: None,
            format: RecallFormat::Structured,
            template: RecallTemplateWire::Entities,
            rerank_k: None,
        };
        let resolved = p.resolve().unwrap();
        assert_eq!(resolved.namespace.thread.as_deref(), Some("t-1"));
        assert_eq!(resolved.query, "go-live");
        assert_eq!(resolved.k, Some(5));
        assert_eq!(resolved.format, RecallFormat::Structured);
        assert_eq!(resolved.template, RecallTemplate::Entities);
    }

    #[test]
    fn dream_params_resolve() {
        let p = DreamParams {
            namespace: "ws-1".into(),
            thread: None,
            batch_id: Some("batch-1".into()),
        };
        let resolved = p.resolve().unwrap();
        assert_eq!(resolved.namespace.namespace, "ws-1");
        assert_eq!(resolved.batch_id.as_deref(), Some("batch-1"));
    }

    #[test]
    fn retrieved_context_wire_maps_fields() {
        use chrono::TimeZone;
        let ctx = RetrievedContext::new(kremory::RetrievedContextNewParams {
            entity_id: "ent-1".into(),
            entity_name: "Alice".into(),
            summary: "leads design".into(),
            score: 0.9,
            source_refs: vec![SourceRef {
                kind: SourceKind::Document,
                id: "doc-1".into(),
                occurred_at: Utc.with_ymd_and_hms(2026, 7, 14, 0, 0, 0).unwrap(),
                published_at: None,
            }],
        });
        let wire: RetrievedContextWire = ctx.into();
        assert_eq!(wire.entity_id, "ent-1");
        assert_eq!(wire.entity_name, "Alice");
        assert_eq!(wire.source_refs.len(), 1);
        assert_eq!(wire.source_refs[0].kind, "document");
    }

    /// G2 — the `DreamSummary.types_discovered: Vec<TypeProposal>` detail must
    /// survive the projection to the MCP `DreamOutput` wire (name + description +
    /// justification), NOT be flattened to a count. Guards against the
    /// parity-drop class that shipped the recall→facts bug. A NON-EMPTY input is
    /// load-bearing: the handler_roundtrip mock LLM discovers zero types, so only
    /// a unit test with real proposals proves the detail is carried.
    #[test]
    fn dream_output_carries_type_proposal_detail() {
        let summary = kremory::DreamSummary {
            communities_updated: 0,
            cross_episode_would_merge: 0,
            cross_episode_merged: 0,
            supersessions_recorded: 0,
            facts_archived: 0,
            consolidation_ops_ran: kremory::ConsolidationOpsRan::default(),
            duration_ms: 0,
            types_discovered: vec![kremory::TypeProposal {
                name: "Firm".into(),
                description: "A commercial organization".into(),
                justification: "Recurring 'Firm' entities lacked a type".into(),
            }],
            entities_reclassified: 0,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            warnings: Vec::new(),
            budget_exhausted: false,
        };
        let wire: DreamOutput = summary.into();
        assert_eq!(wire.types_discovered.len(), 1);
        assert_eq!(wire.types_discovered[0].name, "Firm");
        assert_eq!(
            wire.types_discovered[0].description,
            "A commercial organization"
        );
        assert_eq!(
            wire.types_discovered[0].justification,
            "Recurring 'Firm' entities lacked a type"
        );
    }

    // ─── kremory_list_mutations / kremory_undo primitive conversions ─────
    //
    // `MutationRecord` / `UnmergeOutcome` / `EditEntityOutcome` /
    // `DeleteEntityOutcome` / `DeleteFactOutcome` / `UndoOutcome` are all
    // `#[non_exhaustive]` structs (or enums wrapping `#[non_exhaustive]`
    // structs) — kremory-mcp is an external crate to kremory, so none of
    // these can be struct-literal-constructed here (the exact restriction
    // `#[non_exhaustive]` enforces). `MutationKind`'s eight variants are all
    // UNIT variants, which non_exhaustive does NOT block constructing from
    // outside the crate, so the kind mapping is unit-testable directly; the
    // per-field wire mapping for `MutationRecordWire` / `UndoOutcomeWire` is
    // instead proven through a real facade round-trip in
    // `tests/handler_roundtrip.rs` (`list_mutations_and_undo_roundtrip_entity_edit`).

    #[test]
    fn mutation_kind_wire_to_facade_round_trips_all_known_tags() {
        let cases = [
            ("entity_merge", MutationKind::EntityMerge),
            ("fact_supersede", MutationKind::FactSupersede),
            ("fact_archive", MutationKind::FactArchive),
            ("entity_edit", MutationKind::EntityEdit),
            ("entity_delete", MutationKind::EntityDelete),
            ("fact_delete", MutationKind::FactDelete),
            ("community_assign", MutationKind::CommunityAssign),
            ("canonical_form", MutationKind::CanonicalForm),
        ];
        for (tag, expected) in cases {
            assert_eq!(
                mutation_kind_wire_to_facade(tag).unwrap(),
                expected,
                "tag {tag:?} must parse to {expected:?}"
            );
            assert_eq!(
                mutation_kind_facade_to_wire(expected),
                tag,
                "{expected:?} must render back to tag {tag:?}"
            );
        }
    }

    #[test]
    fn mutation_kind_wire_to_facade_rejects_unknown_tag() {
        let err = mutation_kind_wire_to_facade("not_a_real_kind").unwrap_err();
        match err {
            ConversionError::UnknownMutationKind { raw } => {
                assert_eq!(raw, "not_a_real_kind");
            }
            other => panic!("expected UnknownMutationKind error, got: {other:?}"),
        }
    }

    #[test]
    fn list_mutations_params_resolve_routes_entity_id_and_parses_kind() {
        use crate::params::ListMutationsParams;

        let p = ListMutationsParams {
            namespace: "ws-1".into(),
            thread: None,
            entity_id: Some("alice".into()),
            kind: Some("entity_edit".into()),
            since: Some("2026-07-14T00:00:00Z".into()),
            include_undone: Some(true),
        };
        let resolved = p.resolve().unwrap();
        assert_eq!(resolved.namespace.namespace, "ws-1");
        assert_eq!(resolved.entity_id.as_deref(), Some("alice"));
        assert_eq!(resolved.kind, Some(MutationKind::EntityEdit));
        assert!(resolved.since.is_some());
        assert!(resolved.include_undone);
    }

    #[test]
    fn list_mutations_params_resolve_defaults_include_undone_false() {
        use crate::params::ListMutationsParams;

        let p = ListMutationsParams {
            namespace: "ws-1".into(),
            thread: None,
            entity_id: None,
            kind: None,
            since: None,
            include_undone: None,
        };
        let resolved = p.resolve().unwrap();
        assert!(resolved.entity_id.is_none());
        assert!(resolved.kind.is_none());
        assert!(!resolved.include_undone);
    }

    #[test]
    fn list_mutations_params_resolve_rejects_unknown_kind() {
        use crate::params::ListMutationsParams;

        let p = ListMutationsParams {
            namespace: "ws-1".into(),
            thread: None,
            entity_id: None,
            kind: Some("bogus".into()),
            since: None,
            include_undone: None,
        };
        let err = p.resolve().unwrap_err();
        assert!(matches!(err, ConversionError::UnknownMutationKind { .. }));
    }

    #[test]
    fn list_mutations_params_resolve_rejects_empty_namespace() {
        use crate::params::ListMutationsParams;

        let p = ListMutationsParams {
            namespace: "".into(),
            thread: None,
            entity_id: None,
            kind: None,
            since: None,
            include_undone: None,
        };
        let err = p.resolve().unwrap_err();
        assert!(matches!(err, ConversionError::EmptyNamespace));
    }

    #[test]
    fn undo_params_resolve_happy_path() {
        use crate::params::UndoParams;

        let p = UndoParams {
            namespace: "ws-1".into(),
            thread: Some("t-1".into()),
            mutation_id: 42,
        };
        let resolved = p.resolve().unwrap();
        assert_eq!(resolved.namespace.namespace, "ws-1");
        assert_eq!(resolved.namespace.thread.as_deref(), Some("t-1"));
        assert_eq!(resolved.mutation_id, 42);
    }

    #[test]
    fn undo_params_resolve_rejects_empty_namespace() {
        use crate::params::UndoParams;

        let p = UndoParams {
            namespace: "".into(),
            thread: None,
            mutation_id: 1,
        };
        let err = p.resolve().unwrap_err();
        assert!(matches!(err, ConversionError::EmptyNamespace));
    }
}

//! Bidirectional conversions between MCP wire types (this crate) and
//! kremory core types (`kremory::memory::*`).
//!
//! Per D.4 scaffold rationale (lib.rs header): the MCP-side types carry
//! `JsonSchema` derives required by rmcp's macros; kremory-side types do
//! not (keeps kremory free of schemars). This module is the conversion
//! boundary.
//!
//! ## Error surface
//!
//! Conversions from MCP → rqlm can fail on:
//! - Unknown enum string values (`source_ref_kind` not in
//!   {meeting,document,chat}; `template` not in
//!   {entities,edge_summary,temporal_facts}; `source_kind` filter same)
//! - Malformed ISO-8601 timestamps (`source_ref_occurred_at`,
//!   `valid_at`/`invalid_at`, `as_of`)
//! - Empty `workspace_id`
//!
//! These map to `ConversionError`. Handler bodies in `lib.rs` translate
//! these into MCP `ErrorData::invalid_params` so the JSON-RPC client
//! receives a precise diagnostic.
//!
//! ## rqlm → MCP
//!
//! Always infallible — rqlm's enum variants map 1:1 to lower-case
//! strings; `DateTime<Utc>` always renders to RFC 3339.

use chrono::{DateTime, Utc};
use kremory::memory::{
    ContextTemplate, DreamPhaseResult, IngestResult, RetrievedContext, SearchOpts, SourceKind,
    SourceRef, StructuredFact, WorkspaceScope,
};
use thiserror::Error;

use crate::{
    ContextBlockParameters, IngestEpisodeOutput, IngestEpisodeParameters, RetrievedContextOutput,
    RunDreamPhaseOutput, RunDreamPhaseParameters, SearchParameters, SearchResultsOutput,
    SourceRefOutput, StructuredFactInput,
};

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("workspace_id must not be empty")]
    EmptyWorkspaceId,
    #[error("unknown source_ref_kind: {0:?} (expected meeting | document | chat)")]
    UnknownSourceKind(String),
    #[error("unknown template: {0:?} (expected entities | edge_summary | temporal_facts)")]
    UnknownTemplate(String),
    #[error("malformed ISO-8601 timestamp in {field}: {value:?} ({source})")]
    Timestamp {
        field: &'static str,
        value: String,
        #[source]
        source: chrono::ParseError,
    },
}

// ─── primitive conversions ─────────────────────────────────────────────

pub(crate) fn parse_source_kind(s: &str) -> Result<SourceKind, ConversionError> {
    match s {
        "meeting" => Ok(SourceKind::Meeting),
        "document" => Ok(SourceKind::Document),
        "chat" => Ok(SourceKind::Chat),
        other => Err(ConversionError::UnknownSourceKind(other.to_string())),
    }
}

pub(crate) fn source_kind_to_wire(k: SourceKind) -> &'static str {
    match k {
        SourceKind::Meeting => "meeting",
        SourceKind::Document => "document",
        SourceKind::Chat => "chat",
    }
}

pub(crate) fn parse_template(s: &str) -> Result<ContextTemplate, ConversionError> {
    match s {
        "entities" => Ok(ContextTemplate::Entities),
        "edge_summary" => Ok(ContextTemplate::EdgeSummary),
        "temporal_facts" => Ok(ContextTemplate::TemporalFacts),
        other => Err(ConversionError::UnknownTemplate(other.to_string())),
    }
}

pub(crate) fn parse_timestamp(
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

// ─── compound conversions ──────────────────────────────────────────────

pub(crate) fn build_scope(
    workspace_id: &str,
    thread_id: Option<&str>,
) -> Result<WorkspaceScope, ConversionError> {
    if workspace_id.is_empty() {
        return Err(ConversionError::EmptyWorkspaceId);
    }
    Ok(match thread_id {
        Some(t) => WorkspaceScope::with_thread(workspace_id, t),
        None => WorkspaceScope::new(workspace_id),
    })
}

impl IngestEpisodeParameters {
    pub fn into_rqlm(
        self,
    ) -> Result<(WorkspaceScope, SourceRef, Vec<StructuredFact>, String), ConversionError> {
        let scope = build_scope(&self.workspace_id, self.thread_id.as_deref())?;
        let kind = parse_source_kind(&self.source_ref_kind)?;
        let occurred_at = parse_timestamp("source_ref_occurred_at", &self.source_ref_occurred_at)?;
        let source_ref = SourceRef {
            kind,
            id: self.source_ref_id,
            occurred_at,
        };
        let mut facts = Vec::with_capacity(self.structured_facts.len());
        for sf in self.structured_facts {
            facts.push(sf.try_into_rqlm()?);
        }
        Ok((scope, source_ref, facts, self.content))
    }
}

impl StructuredFactInput {
    fn try_into_rqlm(self) -> Result<StructuredFact, ConversionError> {
        let valid_at = self
            .valid_at
            .as_deref()
            .map(|raw| parse_timestamp("valid_at", raw))
            .transpose()?;
        let invalid_at = self
            .invalid_at
            .as_deref()
            .map(|raw| parse_timestamp("invalid_at", raw))
            .transpose()?;
        Ok(StructuredFact {
            subject: self.subject,
            predicate: self.predicate,
            object: self.object,
            valid_at,
            invalid_at,
        })
    }
}

impl RunDreamPhaseParameters {
    pub fn into_rqlm(self) -> Result<WorkspaceScope, ConversionError> {
        build_scope(&self.workspace_id, self.thread_id.as_deref())
    }
}

impl SearchParameters {
    pub fn into_rqlm(self) -> Result<(WorkspaceScope, String, SearchOpts), ConversionError> {
        let scope = build_scope(&self.workspace_id, self.thread_id.as_deref())?;
        let as_of = self
            .as_of
            .as_deref()
            .map(|raw| parse_timestamp("as_of", raw))
            .transpose()?;
        let source_kind = self
            .source_kind
            .as_deref()
            .map(parse_source_kind)
            .transpose()?;
        let opts = SearchOpts {
            limit: self.limit,
            as_of,
            source_kind,
        };
        Ok((scope, self.query, opts))
    }
}

impl ContextBlockParameters {
    pub fn into_rqlm(self) -> Result<(Vec<RetrievedContext>, ContextTemplate), ConversionError> {
        let template = parse_template(&self.template)?;
        let results = self
            .results
            .into_iter()
            .map(RetrievedContext::from)
            .collect();
        Ok((results, template))
    }
}

// ─── rqlm → MCP (infallible) ───────────────────────────────────────────

impl From<IngestResult> for IngestEpisodeOutput {
    fn from(r: IngestResult) -> Self {
        Self {
            entities_added: r.entities_added,
            edges_added: r.edges_added,
            facts_invalidated: r.facts_invalidated,
            duration_ms: r.duration_ms,
        }
    }
}

impl From<DreamPhaseResult> for RunDreamPhaseOutput {
    fn from(r: DreamPhaseResult) -> Self {
        Self {
            communities_recomputed: r.communities_recomputed,
            cross_meeting_merges: r.cross_meeting_merges,
            supersessions_recorded: r.supersessions_recorded,
            facts_archived: r.facts_archived,
            duration_ms: r.duration_ms,
        }
    }
}

impl From<Vec<RetrievedContext>> for SearchResultsOutput {
    fn from(results: Vec<RetrievedContext>) -> Self {
        Self {
            results: results.into_iter().map(RetrievedContextOutput::from).collect(),
        }
    }
}

impl From<RetrievedContext> for RetrievedContextOutput {
    fn from(c: RetrievedContext) -> Self {
        Self {
            entity_id: c.entity_id,
            entity_name: c.entity_name,
            summary: c.summary,
            score: c.score,
            source_refs: c
                .source_refs
                .into_iter()
                .map(SourceRefOutput::from)
                .collect(),
        }
    }
}

impl From<SourceRef> for SourceRefOutput {
    fn from(r: SourceRef) -> Self {
        Self {
            kind: source_kind_to_wire(r.kind).to_string(),
            id: r.id,
            occurred_at: r.occurred_at.to_rfc3339(),
        }
    }
}

// MCP wire SourceRefOutput → rqlm SourceRef. Round-trip support so a
// caller-supplied `ContextBlockParameters.results` (which may have been
// fetched via a prior `rqlm_search` call and round-tripped through the
// MCP boundary) re-enters rqlm with the correct enum/timestamp types.
//
// Unlike the other MCP-side wire types, the `source_refs` inside a
// `ContextBlockParameters.results[i]` get re-converted to `SourceRef`
// here (infallible-via-expect rejected — bad input from a caller MUST
// not panic). Returns a partial `SourceRef` with `chat` fallback +
// Utc::now() for malformed inputs and logs a warning; alternative would
// be threading TryFrom through context_block's signature, but that
// changes rqlm's pure-fn shape just for an MCP boundary case.
impl From<SourceRefOutput> for SourceRef {
    fn from(o: SourceRefOutput) -> Self {
        let kind = parse_source_kind(&o.kind).unwrap_or_else(|_| {
            tracing::warn!(
                wire_kind = %o.kind,
                "MCP SourceRefOutput → SourceRef: unknown kind, defaulting to Chat"
            );
            SourceKind::Chat
        });
        let occurred_at = parse_timestamp("source_refs[].occurred_at", &o.occurred_at)
            .unwrap_or_else(|_| {
                tracing::warn!(
                    wire_ts = %o.occurred_at,
                    "MCP SourceRefOutput → SourceRef: malformed timestamp, defaulting to now"
                );
                Utc::now()
            });
        Self {
            kind,
            id: o.id,
            occurred_at,
        }
    }
}

impl From<RetrievedContextOutput> for RetrievedContext {
    fn from(c: RetrievedContextOutput) -> Self {
        Self {
            entity_id: c.entity_id,
            entity_name: c.entity_name,
            summary: c.summary,
            score: c.score,
            source_refs: c.source_refs.into_iter().map(SourceRef::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    // ─── primitive conversions ──────────────────────────────────────

    #[test]
    fn source_kind_parses_lowercase_strings() {
        assert_eq!(parse_source_kind("meeting").unwrap(), SourceKind::Meeting);
        assert_eq!(parse_source_kind("document").unwrap(), SourceKind::Document);
        assert_eq!(parse_source_kind("chat").unwrap(), SourceKind::Chat);
    }

    #[test]
    fn source_kind_rejects_unknown_value() {
        let err = parse_source_kind("Meeting").unwrap_err();
        assert!(matches!(err, ConversionError::UnknownSourceKind(_)));
        let msg = format!("{err}");
        assert!(msg.contains("Meeting"), "error must carry offending value: {msg}");
    }

    #[test]
    fn source_kind_to_wire_round_trips() {
        for kind in [SourceKind::Meeting, SourceKind::Document, SourceKind::Chat] {
            let wire = source_kind_to_wire(kind);
            let back = parse_source_kind(wire).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn template_parses_snake_case_strings() {
        assert_eq!(parse_template("entities").unwrap(), ContextTemplate::Entities);
        assert_eq!(
            parse_template("edge_summary").unwrap(),
            ContextTemplate::EdgeSummary
        );
        assert_eq!(
            parse_template("temporal_facts").unwrap(),
            ContextTemplate::TemporalFacts
        );
    }

    #[test]
    fn template_rejects_unknown_value() {
        let err = parse_template("EdgeSummary").unwrap_err();
        assert!(matches!(err, ConversionError::UnknownTemplate(_)));
    }

    #[test]
    fn parse_timestamp_accepts_rfc3339() {
        use chrono::Datelike;
        use chrono::Timelike;
        let t = parse_timestamp("source_ref_occurred_at", "2026-05-19T10:30:00Z").unwrap();
        assert_eq!(t.year(), 2026);
        assert_eq!(t.month(), 5);
        assert_eq!(t.day(), 19);
        assert_eq!(t.hour(), 10);
        assert_eq!(t.minute(), 30);
        // Round-trip via .to_rfc3339() — what wire-output conversion uses.
        let round = t.to_rfc3339();
        assert!(round.starts_with("2026-05-19T10:30:00"));
    }

    #[test]
    fn parse_timestamp_rejects_malformed() {
        let err = parse_timestamp("source_ref_occurred_at", "yesterday at noon").unwrap_err();
        match err {
            ConversionError::Timestamp { field, value, .. } => {
                assert_eq!(field, "source_ref_occurred_at");
                assert_eq!(value, "yesterday at noon");
            }
            other => panic!("expected Timestamp error, got: {other:?}"),
        }
    }

    // ─── compound conversions ───────────────────────────────────────

    #[test]
    fn build_scope_with_thread() {
        let s = build_scope("ws-1", Some("thread-a")).unwrap();
        assert_eq!(s.workspace_id, "ws-1");
        assert_eq!(s.thread_id.as_deref(), Some("thread-a"));
    }

    #[test]
    fn build_scope_without_thread() {
        let s = build_scope("ws-1", None).unwrap();
        assert!(s.thread_id.is_none());
    }

    #[test]
    fn build_scope_rejects_empty_workspace_id() {
        let err = build_scope("", None).unwrap_err();
        assert!(matches!(err, ConversionError::EmptyWorkspaceId));
    }

    #[test]
    fn ingest_parameters_into_rqlm_happy_path() {
        let p = IngestEpisodeParameters {
            workspace_id: "ws-1".into(),
            thread_id: None,
            content: "transcript chunk".into(),
            source_ref_kind: "meeting".into(),
            source_ref_id: "mtg-1".into(),
            source_ref_occurred_at: "2026-05-19T10:00:00Z".into(),
            structured_facts: vec![StructuredFactInput {
                subject: "alice".into(),
                predicate: "leads".into(),
                object: "design-team".into(),
                valid_at: Some("2026-05-01T00:00:00Z".into()),
                invalid_at: None,
            }],
        };
        let (scope, source_ref, facts, content) = p.into_rqlm().unwrap();
        assert_eq!(scope.workspace_id, "ws-1");
        assert_eq!(source_ref.kind, SourceKind::Meeting);
        assert_eq!(source_ref.id, "mtg-1");
        assert_eq!(content, "transcript chunk");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "alice");
        assert!(facts[0].valid_at.is_some());
        assert!(facts[0].invalid_at.is_none());
    }

    #[test]
    fn ingest_parameters_into_rqlm_rejects_bad_kind() {
        let p = IngestEpisodeParameters {
            workspace_id: "ws-1".into(),
            thread_id: None,
            content: "x".into(),
            source_ref_kind: "podcast".into(),
            source_ref_id: "id".into(),
            source_ref_occurred_at: "2026-05-19T10:00:00Z".into(),
            structured_facts: vec![],
        };
        let err = p.into_rqlm().unwrap_err();
        assert!(matches!(err, ConversionError::UnknownSourceKind(_)));
    }

    #[test]
    fn ingest_parameters_into_rqlm_rejects_bad_timestamp_in_structured_fact() {
        let p = IngestEpisodeParameters {
            workspace_id: "ws-1".into(),
            thread_id: None,
            content: "x".into(),
            source_ref_kind: "meeting".into(),
            source_ref_id: "id".into(),
            source_ref_occurred_at: "2026-05-19T10:00:00Z".into(),
            structured_facts: vec![StructuredFactInput {
                subject: "s".into(),
                predicate: "p".into(),
                object: "o".into(),
                valid_at: Some("not a date".into()),
                invalid_at: None,
            }],
        };
        let err = p.into_rqlm().unwrap_err();
        match err {
            ConversionError::Timestamp { field, .. } => assert_eq!(field, "valid_at"),
            other => panic!("expected Timestamp error, got: {other:?}"),
        }
    }

    #[test]
    fn search_parameters_into_rqlm_passes_through_filters() {
        let p = SearchParameters {
            workspace_id: "ws-1".into(),
            thread_id: Some("t-1".into()),
            query: "go-live decisions".into(),
            limit: Some(5),
            as_of: Some("2026-05-01T00:00:00Z".into()),
            source_kind: Some("document".into()),
        };
        let (scope, q, opts) = p.into_rqlm().unwrap();
        assert_eq!(scope.thread_id.as_deref(), Some("t-1"));
        assert_eq!(q, "go-live decisions");
        assert_eq!(opts.limit, Some(5));
        assert!(opts.as_of.is_some());
        assert_eq!(opts.source_kind, Some(SourceKind::Document));
    }

    #[test]
    fn context_block_parameters_into_rqlm() {
        let p = ContextBlockParameters {
            results: vec![RetrievedContextOutput {
                entity_id: "ent-1".into(),
                entity_name: "Test".into(),
                summary: "Body".into(),
                score: 0.9,
                source_refs: vec![SourceRefOutput {
                    kind: "meeting".into(),
                    id: "m-1".into(),
                    occurred_at: "2026-05-19T10:00:00Z".into(),
                }],
            }],
            template: "entities".into(),
        };
        let (results, template) = p.into_rqlm().unwrap();
        assert_eq!(template, ContextTemplate::Entities);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_refs[0].kind, SourceKind::Meeting);
    }

    #[test]
    fn context_block_parameters_rejects_unknown_template() {
        let p = ContextBlockParameters {
            results: vec![],
            template: "bogus".into(),
        };
        assert!(matches!(
            p.into_rqlm().unwrap_err(),
            ConversionError::UnknownTemplate(_)
        ));
    }

    // ─── rqlm → MCP round trips ─────────────────────────────────────

    #[test]
    fn ingest_result_to_wire_preserves_counts() {
        let r = IngestResult {
            entities_added: 3,
            edges_added: 7,
            facts_invalidated: 1,
            duration_ms: 42,
        };
        let wire: IngestEpisodeOutput = r.into();
        assert_eq!(wire.entities_added, 3);
        assert_eq!(wire.edges_added, 7);
        assert_eq!(wire.facts_invalidated, 1);
        assert_eq!(wire.duration_ms, 42);
    }

    #[test]
    fn retrieved_context_round_trips_through_wire() {
        let original = RetrievedContext {
            entity_id: "ent-1".into(),
            entity_name: "Roadmap Decision".into(),
            summary: "Q3 priorities locked".into(),
            score: 0.92,
            source_refs: vec![SourceRef {
                kind: SourceKind::Meeting,
                id: "mtg-1".into(),
                occurred_at: Utc.with_ymd_and_hms(2026, 5, 19, 10, 0, 0).unwrap(),
            }],
        };
        let wire: RetrievedContextOutput = original.clone().into();
        assert_eq!(wire.source_refs[0].kind, "meeting");
        assert_eq!(wire.source_refs[0].id, "mtg-1");
        let back: RetrievedContext = wire.into();
        assert_eq!(back.entity_id, original.entity_id);
        assert_eq!(back.source_refs.len(), 1);
        assert_eq!(back.source_refs[0].kind, original.source_refs[0].kind);
        assert_eq!(back.source_refs[0].id, original.source_refs[0].id);
        assert_eq!(
            back.source_refs[0].occurred_at,
            original.source_refs[0].occurred_at
        );
    }

    #[test]
    fn source_ref_output_unknown_kind_defaults_to_chat_with_warn() {
        let bad = SourceRefOutput {
            kind: "podcast".into(),
            id: "id".into(),
            occurred_at: "2026-05-19T10:00:00Z".into(),
        };
        // No panic; logs warn + defaults. Verifies the fallback contract
        // (matches the trace comment in the From impl).
        let r: SourceRef = bad.into();
        assert_eq!(r.kind, SourceKind::Chat);
    }
}

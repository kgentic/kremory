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
//! These map to [`ConversionError`], which `lib.rs` maps to
//! [`crate::ToolError::InvalidParams`] → MCP `ErrorData::invalid_params`.
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
    Namespace, RecallTemplate, RetrievedContext, RetrievedFact, SourceKind, SourceRef,
    StructuredFact,
};
use thiserror::Error;

use crate::params::{
    ConsolidationOpsRanWire, DreamOutput, DreamParams, RecallFormat, RecallParams,
    RecallTemplateWire, RememberOutput, RememberParams, RetrievedContextWire, RetrievedFactWire,
    SourceKindWire, SourceRefWire, StructuredFactWire,
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
            types_discovered_count: d.types_discovered.len(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SourceKindWire;

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
}

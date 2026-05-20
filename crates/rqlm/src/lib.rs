//! rqlm — Zep-equivalent orchestration layer over rqlc (Graphiti-equivalent).
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface" — 4 public async functions
//! plus the scoping types in `types`. Internal orchestration modules added
//! in D.2 stay `pub(crate)`; external consumers use the 4 entry points only.
//!
//! ## Greenfield contract
//!
//! D.1b ships scaffolding only. The 4 public functions return
//! `RqlmError::Unimplemented(...)` until D.2's TDD lane lands the real
//! implementations. This keeps the public surface visible to downstream
//! consumers (the-host-application, rqlm-mcp) while the implementation is built.
//!
//! ## BYOM contract
//!
//! `ChatProvider` is the canonical LLM abstraction across all rql layers
//! (per `.claude/rules/rql-layer-ownership.md § Cross-Cutting LLM
//! Abstraction`). rqlm re-exports the trait so consumers can implement it
//! against any backend (OpenAI, Anthropic, Bedrock, vLLM, local GGUF, …).
//!
//! ## Layer ownership
//!
//! Per the canonical Zep / Graphiti split (verified 2026-05-13 via context7;
//! recorded in `project_zep_graphiti_split_canonical.md`):
//!
//! - rqlc owns per-episode work: `add_episode` cycle = LLM entity / edge
//!   extraction + dedup + fact invalidation + temporal validity inference
//!   + community detection primitive + hybrid retrieval primitives.
//! - rqlm owns cross-episode wrappers: multi-tenant scoping + packaged
//!   batch consolidation recipe (`run_dream_phase`) + opinionated retrieval
//!   defaults over rqlc's hybrid search + context-block templates.

pub mod graph;
pub mod types;

pub use graph::GraphHandle;
pub use types::{
    ContextTemplate, DreamPhaseResult, IngestResult, RetrievedContext, Result, RqlmError,
    SearchOpts, SourceKind, SourceRef, StructuredFact, WorkspaceScope,
};

// Re-export the canonical LLM abstraction trait so SDK consumers depend on
// rqlm only and still get the BYOM contract surface. Per ADR-Phase-D.0 §
// "rqlm public API surface".
pub use autoagents_llm::chat::ChatProvider;

use std::sync::Arc;

/// Ingest one episode into the graph. Thin orchestration wrapper over
/// [`GraphHandle::graph_ingest_episode`].
///
/// The `_valid_at` parameter from D.1b's exploratory signature was
/// dropped in D.2b-impl — episode time is carried via
/// `source_ref.occurred_at` per Graphiti's canonical add_episode shape.
/// The duplicate had no consumer (rqlm-mcp's conversion layer never
/// surfaced it) and the trait signature is the single source of truth.
pub async fn ingest_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    scope: WorkspaceScope,
) -> Result<IngestResult> {
    graph
        .graph_ingest_episode(&scope, &source_ref, content, &structured_facts, provider)
        .await
}

/// Run the packaged batch consolidation recipe over the scoped graph.
/// Thin orchestration wrapper over [`GraphHandle::graph_run_consolidation`].
///
/// Consumer-triggered (the host application fires on meeting-end; aidocs fires on
/// doc-batch flush). NOT a daemon.
pub async fn run_dream_phase(
    graph: &dyn GraphHandle,
    scope: WorkspaceScope,
    provider: Arc<dyn ChatProvider>,
) -> Result<DreamPhaseResult> {
    graph.graph_run_consolidation(&scope, provider).await
}

/// Query the graph with rqlm's opinionated retrieval defaults. Thin
/// orchestration wrapper over [`GraphHandle::graph_search`].
pub async fn search(
    graph: &dyn GraphHandle,
    query: &str,
    scope: WorkspaceScope,
    opts: SearchOpts,
) -> Result<Vec<RetrievedContext>> {
    graph.graph_search(&scope, query, &opts).await
}

/// Render `results` into the final string handed to the LLM, per the
/// requested template strategy. Closest to Zep's `%{user_summary}` /
/// `%{edges}` / `%{entities}` template placeholders.
///
/// Implemented for all 3 [`ContextTemplate`] variants:
/// - `Entities` — one block per entity: name, summary, source pointers
/// - `EdgeSummary` — one line per entity-source-edge for compact context
/// - `TemporalFacts` — flattens source_refs with `valid_at` annotations
///
/// Empty `results` always renders an empty string. Order is preserved from
/// the input — callers are expected to pass results already sorted by score.
///
/// This is the only rqlm public fn that has no rqlc primitive dependency,
/// so D.2a (this commit) ships it ahead of the rest. The other three fns
/// (`ingest_episode`, `run_dream_phase`, `search`) wrap rqlc primitives
/// and ship in D.2b once the `RqlGraph<L, Emb>` API shape (generic vs
/// trait-object vs concrete wrapper) is locked.
pub fn context_block(results: &[RetrievedContext], template: ContextTemplate) -> String {
    if results.is_empty() {
        return String::new();
    }
    match template {
        ContextTemplate::Entities => render_entities(results),
        ContextTemplate::EdgeSummary => render_edge_summary(results),
        ContextTemplate::TemporalFacts => render_temporal_facts(results),
    }
}

fn render_entities(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        out.push_str("## ");
        out.push_str(&r.entity_name);
        out.push('\n');
        out.push_str(&r.summary);
        if !r.source_refs.is_empty() {
            out.push_str("\n\nSources: ");
            for (j, sr) in r.source_refs.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                out.push_str(source_kind_label(sr.kind));
                out.push(':');
                out.push_str(&sr.id);
            }
        }
    }
    out
}

fn render_edge_summary(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for r in results {
        for sr in &r.source_refs {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("- ");
            out.push_str(&r.entity_name);
            out.push_str(" <- ");
            out.push_str(source_kind_label(sr.kind));
            out.push(':');
            out.push_str(&sr.id);
        }
    }
    out
}

fn render_temporal_facts(results: &[RetrievedContext]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        for sr in &r.source_refs {
            out.push_str(&r.entity_name);
            out.push_str(" (valid_at=");
            out.push_str(&sr.occurred_at.to_rfc3339());
            out.push_str(") — ");
            out.push_str(&r.summary);
            out.push('\n');
        }
    }
    out.trim_end_matches('\n').to_string()
}

fn source_kind_label(k: SourceKind) -> &'static str {
    match k {
        SourceKind::Meeting => "meeting",
        SourceKind::Document => "document",
        SourceKind::Chat => "chat",
    }
}

/// Re-export of the schema module from rqlc — rqlm consumers should not
/// need to depend on rqlc directly for the common types they round-trip.
pub use rql_core::schema as core_schema;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_scope_constructors_round_trip() {
        let s = WorkspaceScope::new("ws-1");
        assert_eq!(s.workspace_id, "ws-1");
        assert!(s.thread_id.is_none());

        let s = WorkspaceScope::with_thread("ws-2", "thread-a");
        assert_eq!(s.workspace_id, "ws-2");
        assert_eq!(s.thread_id.as_deref(), Some("thread-a"));
    }

    #[test]
    fn source_kind_serde_round_trips_snake_case() {
        let v = serde_json::to_string(&SourceKind::Meeting).expect("serialize");
        assert_eq!(v, "\"meeting\"");

        let back: SourceKind = serde_json::from_str("\"document\"").expect("deserialize");
        assert_eq!(back, SourceKind::Document);
    }

    #[test]
    fn context_template_serde_round_trips_snake_case() {
        let v = serde_json::to_string(&ContextTemplate::EdgeSummary).expect("serialize");
        assert_eq!(v, "\"edge_summary\"");
    }

    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Stub graph handle for D.2b delegation tests. Records the params
    /// each method was called with so we can assert the rqlm wrappers
    /// pass them through correctly.
    #[derive(Default)]
    struct StubGraphHandle {
        last_ingest_scope: Mutex<Option<WorkspaceScope>>,
        last_ingest_content: Mutex<Option<String>>,
        last_ingest_source_id: Mutex<Option<String>>,
        last_ingest_facts_count: Mutex<Option<usize>>,
        last_search_scope: Mutex<Option<WorkspaceScope>>,
        last_search_query: Mutex<Option<String>>,
        last_search_limit: Mutex<Option<usize>>,
        last_consolidation_scope: Mutex<Option<WorkspaceScope>>,
    }

    #[async_trait]
    impl GraphHandle for StubGraphHandle {
        async fn graph_ingest_episode(
            &self,
            scope: &WorkspaceScope,
            source_ref: &SourceRef,
            content: &str,
            structured_facts: &[StructuredFact],
            _provider: Arc<dyn ChatProvider>,
        ) -> Result<IngestResult> {
            *self.last_ingest_scope.lock().unwrap() = Some(scope.clone());
            *self.last_ingest_content.lock().unwrap() = Some(content.to_string());
            *self.last_ingest_source_id.lock().unwrap() = Some(source_ref.id.clone());
            *self.last_ingest_facts_count.lock().unwrap() = Some(structured_facts.len());
            Ok(IngestResult {
                entities_added: 2,
                edges_added: 3,
                facts_invalidated: 0,
                duration_ms: 42,
            })
        }

        async fn graph_search(
            &self,
            scope: &WorkspaceScope,
            query: &str,
            opts: &SearchOpts,
        ) -> Result<Vec<RetrievedContext>> {
            *self.last_search_scope.lock().unwrap() = Some(scope.clone());
            *self.last_search_query.lock().unwrap() = Some(query.to_string());
            *self.last_search_limit.lock().unwrap() = opts.limit;
            Ok(vec![RetrievedContext {
                entity_id: "ent-stub".into(),
                entity_name: "Stub Entity".into(),
                summary: "from StubGraphHandle".into(),
                score: 0.5,
                source_refs: vec![],
            }])
        }

        async fn graph_run_consolidation(
            &self,
            scope: &WorkspaceScope,
            _provider: Arc<dyn ChatProvider>,
        ) -> Result<DreamPhaseResult> {
            *self.last_consolidation_scope.lock().unwrap() = Some(scope.clone());
            Ok(DreamPhaseResult {
                communities_recomputed: 1,
                cross_meeting_merges: 0,
                supersessions_recorded: 0,
                facts_archived: 0,
                duration_ms: 10,
            })
        }
    }

    /// rqlc's `MockChatProvider::null()` already implements the
    /// canonical `autoagents_llm::chat::ChatProvider` trait — it's gated
    /// behind rqlc's `llm` feature which rqlm enables in Cargo.toml.
    /// The stub graph never invokes the provider so the canned-response
    /// behaviour is irrelevant; we just need an `Arc<dyn ChatProvider>`
    /// for delegation tests.
    fn null_provider() -> Arc<dyn ChatProvider> {
        Arc::new(rql_core::provider::MockChatProvider::null())
    }

    #[tokio::test]
    async fn ingest_episode_delegates_to_graph_handle() {
        use chrono::Utc;
        let graph = StubGraphHandle::default();
        let scope = WorkspaceScope::with_thread("ws-1", "thread-a");
        let source_ref = SourceRef {
            kind: SourceKind::Meeting,
            id: "mtg-42".into(),
            occurred_at: Utc::now(),
        };
        let facts = vec![StructuredFact {
            subject: "alice".into(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_at: None,
            invalid_at: None,
        }];

        let result = ingest_episode(
            &graph,
            "transcript content",
            source_ref,
            facts,
            null_provider(),
            scope.clone(),
        )
        .await
        .expect("ingest_episode should succeed via stub");

        assert_eq!(result.entities_added, 2);
        assert_eq!(result.edges_added, 3);
        assert_eq!(graph.last_ingest_scope.lock().unwrap().as_ref(), Some(&scope));
        assert_eq!(
            graph.last_ingest_content.lock().unwrap().as_deref(),
            Some("transcript content")
        );
        assert_eq!(
            graph.last_ingest_source_id.lock().unwrap().as_deref(),
            Some("mtg-42")
        );
        assert_eq!(*graph.last_ingest_facts_count.lock().unwrap(), Some(1));
    }

    #[tokio::test]
    async fn search_delegates_with_opts() {
        let graph = StubGraphHandle::default();
        let scope = WorkspaceScope::new("ws-2");
        let opts = SearchOpts {
            limit: Some(25),
            as_of: None,
            source_kind: Some(SourceKind::Document),
        };

        let hits = search(&graph, "go-live", scope.clone(), opts)
            .await
            .expect("search delegates cleanly");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity_id, "ent-stub");
        assert_eq!(graph.last_search_scope.lock().unwrap().as_ref(), Some(&scope));
        assert_eq!(
            graph.last_search_query.lock().unwrap().as_deref(),
            Some("go-live")
        );
        assert_eq!(*graph.last_search_limit.lock().unwrap(), Some(25));
    }

    #[tokio::test]
    async fn run_dream_phase_delegates_to_consolidation() {
        let graph = StubGraphHandle::default();
        let scope = WorkspaceScope::new("ws-3");

        let result = run_dream_phase(&graph, scope.clone(), null_provider())
            .await
            .expect("run_dream_phase delegates cleanly");

        assert_eq!(result.communities_recomputed, 1);
        assert_eq!(
            graph.last_consolidation_scope.lock().unwrap().as_ref(),
            Some(&scope)
        );
    }
}

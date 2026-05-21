//! Handler-level round-trip integration tests for kremory-mcp.
//!
//! These exercise each MCP tool handler in-process: construct a server
//! (bound or unbound), call the handler directly with `Parameters(...)`,
//! and assert the returned `CallToolResult` carries the expected
//! structured payload (or that an error is returned with the right shape
//! for invalid input + unbound state).
//!
//! Subprocess JSON-RPC plumbing is exercised by `cargo build -p kremory-mcp`
//! producing a runnable binary + by rmcp's own integration suite. These
//! tests focus on the kremory-mcp-specific surface: conversions + delegation
//! + error mapping.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::ErrorCode;
use kremory::core::provider::MockChatProvider;
use kremory::memory::{
    ChatProvider, DreamPhaseResult, GraphHandle, IngestResult, RetrievedContext, Result as RqlmResult,
    SearchOpts, SourceKind, SourceRef, StructuredFact, WorkspaceScope,
};
use kremory_mcp::{
    ContextBlockParameters, IngestEpisodeParameters, RetrievedContextOutput, KremoryMcpServer,
    RunDreamPhaseParameters, SearchParameters, SourceRefOutput, StructuredFactInput,
};

/// Stub graph handle that records the params each method received +
/// returns deterministic canned results. Mirrors kremory's internal stub
/// but in the kremory-mcp test surface for explicit ownership.
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
    ) -> RqlmResult<IngestResult> {
        *self.last_ingest_scope.lock().unwrap() = Some(scope.clone());
        *self.last_ingest_content.lock().unwrap() = Some(content.to_string());
        *self.last_ingest_source_id.lock().unwrap() = Some(source_ref.id.clone());
        *self.last_ingest_facts_count.lock().unwrap() = Some(structured_facts.len());
        Ok(IngestResult {
            entities_added: 5,
            edges_added: 8,
            facts_invalidated: 1,
            duration_ms: 123,
        })
    }

    async fn graph_search(
        &self,
        scope: &WorkspaceScope,
        query: &str,
        opts: &SearchOpts,
    ) -> RqlmResult<Vec<RetrievedContext>> {
        *self.last_search_scope.lock().unwrap() = Some(scope.clone());
        *self.last_search_query.lock().unwrap() = Some(query.to_string());
        *self.last_search_limit.lock().unwrap() = opts.limit;
        Ok(vec![RetrievedContext {
            entity_id: "ent-roadmap".into(),
            entity_name: "Q3 Roadmap".into(),
            summary: "Locked priorities".into(),
            score: 0.87,
            source_refs: vec![SourceRef {
                kind: SourceKind::Meeting,
                id: "mtg-7".into(),
                occurred_at: Utc::now(),
            }],
        }])
    }

    async fn graph_run_consolidation(
        &self,
        scope: &WorkspaceScope,
        _provider: Arc<dyn ChatProvider>,
    ) -> RqlmResult<DreamPhaseResult> {
        *self.last_consolidation_scope.lock().unwrap() = Some(scope.clone());
        Ok(DreamPhaseResult {
            communities_recomputed: 2,
            cross_meeting_merges: 1,
            supersessions_recorded: 1,
            facts_archived: 3,
            duration_ms: 50,
        })
    }
}

fn bound_server() -> (KremoryMcpServer, Arc<StubGraphHandle>) {
    let graph = Arc::new(StubGraphHandle::default());
    let provider: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let server = KremoryMcpServer::new(graph.clone() as Arc<dyn GraphHandle>, provider);
    (server, graph)
}

fn sample_ingest_params() -> IngestEpisodeParameters {
    IngestEpisodeParameters {
        workspace_id: "ws-1".into(),
        thread_id: Some("thread-a".into()),
        content: "transcript chunk".into(),
        source_ref_kind: "meeting".into(),
        source_ref_id: "mtg-42".into(),
        source_ref_occurred_at: "2026-05-19T10:00:00Z".into(),
        structured_facts: vec![StructuredFactInput {
            subject: "alice".into(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_at: None,
            invalid_at: None,
        }],
    }
}

// ─── bound-server happy paths ───────────────────────────────────────────

#[tokio::test]
async fn ingest_episode_bound_delegates_and_returns_structured_output() {
    let (server, graph) = bound_server();
    let result = server
        .kremory_ingest_episode(Parameters(sample_ingest_params()))
        .await
        .expect("ingest_episode bound call");

    let structured = result.structured_content.expect("structured payload present");
    assert_eq!(result.is_error, Some(false));
    assert_eq!(structured["entities_added"], 5);
    assert_eq!(structured["edges_added"], 8);
    assert_eq!(structured["facts_invalidated"], 1);
    assert_eq!(structured["duration_ms"], 123);

    // Verify kremory forwarded the params to the graph handle.
    let scope = graph.last_ingest_scope.lock().unwrap().clone().unwrap();
    assert_eq!(scope.workspace_id, "ws-1");
    assert_eq!(scope.thread_id.as_deref(), Some("thread-a"));
    assert_eq!(
        graph.last_ingest_content.lock().unwrap().as_deref(),
        Some("transcript chunk")
    );
    assert_eq!(
        graph.last_ingest_source_id.lock().unwrap().as_deref(),
        Some("mtg-42")
    );
    assert_eq!(*graph.last_ingest_facts_count.lock().unwrap(), Some(1));
}

#[tokio::test]
async fn search_bound_delegates_and_returns_results_payload() {
    let (server, graph) = bound_server();
    let params = SearchParameters {
        workspace_id: "ws-9".into(),
        thread_id: None,
        query: "roadmap decisions".into(),
        limit: Some(5),
        as_of: None,
        source_kind: None,
    };
    let result = server
        .kremory_search(Parameters(params))
        .await
        .expect("search bound call");

    let structured = result.structured_content.expect("structured payload");
    let results = structured["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["entity_id"], "ent-roadmap");
    assert_eq!(results[0]["entity_name"], "Q3 Roadmap");

    assert_eq!(
        graph.last_search_query.lock().unwrap().as_deref(),
        Some("roadmap decisions")
    );
    assert_eq!(*graph.last_search_limit.lock().unwrap(), Some(5));
}

#[tokio::test]
async fn run_dream_phase_bound_delegates_and_returns_counts() {
    let (server, graph) = bound_server();
    let params = RunDreamPhaseParameters {
        workspace_id: "ws-1".into(),
        thread_id: None,
    };
    let result = server
        .kremory_run_dream_phase(Parameters(params))
        .await
        .expect("dream phase bound call");

    let structured = result.structured_content.expect("structured payload");
    assert_eq!(structured["communities_recomputed"], 2);
    assert_eq!(structured["cross_meeting_merges"], 1);
    assert_eq!(structured["supersessions_recorded"], 1);
    assert_eq!(structured["facts_archived"], 3);

    let scope = graph.last_consolidation_scope.lock().unwrap().clone().unwrap();
    assert_eq!(scope.workspace_id, "ws-1");
}

#[tokio::test]
async fn context_block_renders_entities_template_without_graph() {
    // context_block is pure — works on the unbound server too.
    let server = KremoryMcpServer::unbound();
    let params = ContextBlockParameters {
        results: vec![RetrievedContextOutput {
            entity_id: "ent-1".into(),
            entity_name: "Roadmap".into(),
            summary: "Q3 priorities locked".into(),
            score: 0.9,
            source_refs: vec![SourceRefOutput {
                kind: "meeting".into(),
                id: "mtg-1".into(),
                occurred_at: "2026-05-19T10:00:00Z".into(),
            }],
        }],
        template: "entities".into(),
    };
    let result = server
        .kremory_context_block(Parameters(params))
        .await
        .expect("context_block unbound call");

    let structured = result.structured_content.expect("structured payload");
    let rendered = structured["rendered"].as_str().expect("rendered str");
    assert!(
        rendered.contains("## Roadmap"),
        "entities template should render entity header: {rendered}"
    );
    assert!(
        rendered.contains("meeting:mtg-1"),
        "entities template should include source ref: {rendered}"
    );
}

// ─── unbound-server error paths ──────────────────────────────────────────

#[tokio::test]
async fn ingest_episode_unbound_returns_graph_not_bound_error() {
    let server = KremoryMcpServer::unbound();
    let err = server
        .kremory_ingest_episode(Parameters(sample_ingest_params()))
        .await
        .expect_err("unbound server must error on ingest");
    assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
    assert!(
        err.message.contains("kremory_ingest_episode"),
        "error must carry tool name: {}",
        err.message
    );
    assert!(
        err.message.contains("KremoryMcpServer::new"),
        "error must point at bound constructor: {}",
        err.message
    );
}

#[tokio::test]
async fn search_unbound_returns_graph_not_bound_error() {
    let server = KremoryMcpServer::unbound();
    let params = SearchParameters {
        workspace_id: "ws-1".into(),
        thread_id: None,
        query: "x".into(),
        limit: None,
        as_of: None,
        source_kind: None,
    };
    let err = server
        .kremory_search(Parameters(params))
        .await
        .expect_err("unbound server must error on search");
    assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
    assert!(err.message.contains("kremory_search"));
}

#[tokio::test]
async fn run_dream_phase_unbound_returns_graph_not_bound_error() {
    let server = KremoryMcpServer::unbound();
    let params = RunDreamPhaseParameters {
        workspace_id: "ws-1".into(),
        thread_id: None,
    };
    let err = server
        .kremory_run_dream_phase(Parameters(params))
        .await
        .expect_err("unbound server must error on dream phase");
    assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
    assert!(err.message.contains("kremory_run_dream_phase"));
}

// ─── conversion-error → invalid_params ──────────────────────────────────

#[tokio::test]
async fn ingest_episode_rejects_bad_source_kind_with_invalid_params() {
    let (server, _) = bound_server();
    let mut params = sample_ingest_params();
    params.source_ref_kind = "podcast".into();
    let err = server
        .kremory_ingest_episode(Parameters(params))
        .await
        .expect_err("bad source_ref_kind must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("podcast"),
        "invalid_params must echo the offending value: {}",
        err.message
    );
}

#[tokio::test]
async fn ingest_episode_rejects_malformed_timestamp_with_invalid_params() {
    let (server, _) = bound_server();
    let mut params = sample_ingest_params();
    params.source_ref_occurred_at = "yesterday at noon".into();
    let err = server
        .kremory_ingest_episode(Parameters(params))
        .await
        .expect_err("malformed timestamp must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("source_ref_occurred_at"),
        "invalid_params must name the offending field: {}",
        err.message
    );
}

#[tokio::test]
async fn context_block_rejects_unknown_template_with_invalid_params() {
    let server = KremoryMcpServer::unbound();
    let params = ContextBlockParameters {
        results: vec![],
        template: "EdgeSummary".into(), // wrong casing — must be snake_case
    };
    let err = server
        .kremory_context_block(Parameters(params))
        .await
        .expect_err("unknown template must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("EdgeSummary"),
        "error must echo the offending template: {}",
        err.message
    );
}

#[tokio::test]
async fn search_rejects_empty_workspace_id_with_invalid_params() {
    let (server, _) = bound_server();
    let params = SearchParameters {
        workspace_id: "".into(),
        thread_id: None,
        query: "x".into(),
        limit: None,
        as_of: None,
        source_kind: None,
    };
    let err = server
        .kremory_search(Parameters(params))
        .await
        .expect_err("empty workspace_id must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(err.message.to_lowercase().contains("workspace"));
}

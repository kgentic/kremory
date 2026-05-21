//! `StubGraphHandle` — a do-nothing `GraphHandle` implementation for test infra.
//!
//! ## Purpose
//!
//! Downstream consumers (the host application integration tests, aidocs SDK test suites,
//! kremory integration tests) need a concrete `&dyn GraphHandle` without the
//! overhead of spinning up a real libSQL database. `StubGraphHandle` provides
//! that — every method panics with `unimplemented!()` so consumers discover
//! at test time which methods their code path actually calls, and supply a
//! more specific stub for those.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use kremory::memory::{GraphHandle, StubGraphHandle};
//!
//! let handle: Arc<dyn GraphHandle> = Arc::new(StubGraphHandle);
//! ```
//!
//! ## Shape stability
//!
//! `StubGraphHandle` is a shape-stability sentinel: it must implement every
//! method in the `GraphHandle` trait. If a new required method is added to the
//! trait, this struct's compile failure surfaces the gap immediately — the same
//! compiler-enforced invariant as the contract-pin tests.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::error::IngestStatus;

use super::{
    events::EnrichmentEventSink,
    graph::GraphHandle,
    types::{
        BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamOpts, DreamPhaseResult,
        DreamStatus, EpisodeCommit, RetrievedContext, SearchOpts, SourceRef, StructuredFact,
        SubmitOpts, WorkspaceScope,
    },
    ChatProvider, Result,
};

/// Do-nothing `GraphHandle` implementation for test infrastructure.
///
/// Every method panics with `unimplemented!()`. Use this as a stand-in when
/// constructing a `&dyn GraphHandle` or `Arc<dyn GraphHandle>` in tests that
/// don't exercise the graph storage layer directly.
///
/// For tests that DO exercise specific methods, implement a targeted stub by
/// constructing a local type and implementing only the needed methods —
/// `StubGraphHandle` can serve as the fallback for the remaining ones by
/// delegation or by using it as the base in your own impl.
pub struct StubGraphHandle;

#[async_trait]
impl GraphHandle for StubGraphHandle {
    async fn graph_ingest_episode(
        &self,
        _scope: &WorkspaceScope,
        _source_ref: &SourceRef,
        _content: &str,
        _structured_facts: &[StructuredFact],
        _provider: Arc<dyn ChatProvider>,
        _batch_id: Option<String>,
        _opts: SubmitOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<EpisodeCommit> {
        unimplemented!("StubGraphHandle::graph_ingest_episode — provide a concrete stub");
    }

    async fn graph_ingest_status(&self, _run_id: Uuid) -> Result<IngestStatus> {
        unimplemented!("StubGraphHandle::graph_ingest_status — provide a concrete stub");
    }

    async fn graph_cancel(&self, _run_id: Uuid) -> Result<CancelOutcome> {
        unimplemented!("StubGraphHandle::graph_cancel — provide a concrete stub");
    }

    async fn graph_submit_dream(
        &self,
        _scope: &WorkspaceScope,
        _provider: Arc<dyn ChatProvider>,
        _batch_id: Option<String>,
        _opts: DreamOpts,
        _sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<DreamHandle> {
        unimplemented!("StubGraphHandle::graph_submit_dream — provide a concrete stub");
    }

    async fn graph_dream_status(&self, _run_id: Uuid) -> Result<DreamStatus> {
        unimplemented!("StubGraphHandle::graph_dream_status — provide a concrete stub");
    }

    async fn graph_batch_status(&self, _batch_id: &str) -> Result<BatchStatus> {
        unimplemented!("StubGraphHandle::graph_batch_status — provide a concrete stub");
    }

    async fn graph_last_consolidated_at(
        &self,
        _scope: &WorkspaceScope,
    ) -> Result<Option<DateTime<Utc>>> {
        unimplemented!("StubGraphHandle::graph_last_consolidated_at — provide a concrete stub");
    }

    async fn graph_episodes_since_last_dream(&self, _scope: &WorkspaceScope) -> Result<usize> {
        unimplemented!(
            "StubGraphHandle::graph_episodes_since_last_dream — provide a concrete stub"
        );
    }

    async fn graph_is_consolidating(&self, _scope: &WorkspaceScope) -> Result<bool> {
        unimplemented!("StubGraphHandle::graph_is_consolidating — provide a concrete stub");
    }

    async fn graph_search(
        &self,
        _scope: &WorkspaceScope,
        _query: &str,
        _opts: &SearchOpts,
    ) -> Result<Vec<RetrievedContext>> {
        unimplemented!("StubGraphHandle::graph_search — provide a concrete stub");
    }

    async fn graph_run_consolidation(
        &self,
        _scope: &WorkspaceScope,
        _provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult> {
        unimplemented!("StubGraphHandle::graph_run_consolidation — provide a concrete stub");
    }
}

/// `CancelOutcome` for stubs that need a do-nothing cancel response.
///
/// Convenience constructor returning a non-rolled-back, empty cancel outcome.
/// Use when the cancel method must return a value but the test doesn't care
/// about the specifics.
impl CancelOutcome {
    /// Returns a stub `CancelOutcome` with `CancelledPhase::Enrichment`,
    /// `rolled_back = false`, `partial = []`.
    pub fn stub() -> Self {
        Self {
            cancelled_phase: CancelledPhase::Enrichment,
            rolled_back: false,
            partial: vec![],
        }
    }
}

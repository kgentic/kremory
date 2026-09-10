//! kremory — bi-temporal knowledge graph engine + agent memory orchestration.
//!
//! # Modules
//!
//! - [`facade`] — fluent `Memory` facade: the recommended public API for most consumers.
//! - [`core`] — graph primitives: episode ingest, extraction, contradiction handling,
//!   hybrid retrieval, bi-temporal queries, community detection.
//! - [`memory`] — orchestration layer: multi-tenant scoping, batch consolidation
//!   recipe (`run_dream_phase`), opinionated retrieval defaults, context-block
//!   templates.
//!
//! # BYOM contract
//!
//! `ChatProvider` is the canonical LLM abstraction. Re-exported from
//! [`memory`] so SDK consumers depend on `kremory` only.
//!
//! # Quick start
//!
//! Mirrors the README Quickstart — kept compile-verified so it can't silently drift.
//!
//! ```rust,no_run
//! use kremory::{Memory, Namespace};
//! # async fn ex() -> kremory::memory::Result<()> {
//! // Env-detected provider (OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY):
//! let mem = Memory::auto("./agent.db").await?;
//! let ns = Namespace::new("user-jim");
//! // A namespace is required — pass per-call (or set a Memory::open builder default).
//! mem.remember("User prefers concise replies")
//!     .in_namespace(ns.clone())
//!     .await?;
//! let context: String = mem
//!     .recall("what does the user prefer?")
//!     .in_namespace(ns)
//!     .await?;
//! # let _ = context;
//! # Ok(())
//! # }
//! ```

// Per CLAUDE.md testing-policy.md: implementation code is strictly typed (no
// unwrap/expect on Result), test code is exempt. This applies that policy
// structurally at the crate root so inline `#[cfg(test)] mod tests` blocks
// don't each need their own `#![allow(...)]` header.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod core;
pub mod facade;
pub mod memory;

// Convenience re-exports from core
pub use core::error::{Error as CoreError, Result as CoreResult};
// Custom entity-type seed registry (custom-entity-type-registry spec §5.2).
pub use core::entity_types::{
    EntityTypeSpec, NamespaceRegistrationError, NamespaceSeed, SeedOutcome,
};
pub use core::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestGuard, IngestSendError, IngestorConfig,
};
// Process-global graph singleton — spec §2 step 3 (consumer-facing, intentional).
pub use core::engine::{engine, engine_init};

// ── Facade re-exports (Tier 1 / Tier 2 public surface — Story A.8) ───────────
pub use facade::{
    ConsolidationOpsRan, DreamFireAndForget, DreamRequest, DreamSummary, EpisodeEntryBuilder,
    ForgetOutcome, ForgetRequest, Memory, MemoryBuilder, NoEmbedder, NoLlm, RecallRawRequest,
    RecallRequest,
    RecallTemplate, RememberBatchBuilder, RememberRequest, SupersedeOutcome, SupersedeRequest,
    WithEmbedder, WithLlm, WithLlmTrackedParams,
};
// Reversible-graph-mutations (ADR-073 Tier-1) consumer surface — the honest
// reversal outcome types + the inspect view. Re-exported at the crate root so
// the napi binding (kremory-napi) can name them 1:1, matching the crate-root
// convention already used for `SupersedeOutcome` / `DreamSummary`.
pub use facade::{
    DeleteEntityOutcome, DeleteFactOutcome, EditEntityOutcome, MutationFilter, MutationKind,
    MutationRecord, RestoreArchivedOutcome, UndoOutcome, UndoRequest, UnmergeOutcome,
    UnsupersedeOutcome,
};
// Dream scheduler + pass API (Phase C, v0.1.1)
pub use core::ingest::DreamPassOpts;
pub use memory::scheduler::{DreamSchedule, DreamSchedulerHandle};
// Dream Pass 0 type discovery (ADR-037 §3, v0.1.1)
pub use core::dream::TypeProposal;

// Convenience re-exports from memory
pub use memory::init_telemetry;
pub use memory::ChatProvider;
pub use memory::{
    ContextTemplate, DreamPhaseResult, GraphAssertEntityTypeParams, GraphHandle, GraphSearchParams,
    IngestResult, MemoryError, MemoryType, Namespace, Result as MemoryResult, RetrievedContext,
    RetrievedContextNewParams, RetrievedFact, RetrievedFactNewParams, SearchOpts, SourceKind,
    SourceRef, StructuredFact, TelemetryConfig, TelemetryHandle, TelemetryInitError,
};
// ADR-029a (v0.1.4): namespace policy primitives.
pub use memory::types::{ImmutabilityLevel, InvalidPolicyError, NamespacePolicy};
// Handle / lifecycle types (facade + substrate consumers)
pub use memory::types::{
    AwaitOpts, BatchStatus, CancelOutcome, CancelledPhase, CrossEpisodeMode, DreamHandle,
    DreamMode, DreamOpts, DreamStatus, EpisodeCommit, SubmitOpts,
};
pub use memory::IngestStatus;
// Event sinks + event types (Tier 2 consumers)
pub use core::sink::{ContradictionDetected, IngestEventSink, IngestionError, OnEdgeAddedParams};
pub use memory::events::BatchPhase2Complete;
pub use memory::events::EnrichmentEventSink;
// BYOM provider traits
pub use core::provider::{ArcEmbedder, DynEmbeddingProvider, EmbeddingProvider};
// The third BYOM trait. `MemoryBuilder` REQUIRES an extractor to reach a
// buildable state without an LLM (`.with_extractor(..)` is the only path to
// `WithLlm` for an offline consumer), so a trait a caller MUST implement
// belongs beside the other two rather than behind `core::intelligence::`.
// TD-245: writing the first offline example is what surfaced it — the example
// had to reach through a module path no doc mentions.
pub use core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};

// Convenience re-exports from core::config (ADR D15)
pub use core::config::Config as CoreConfig;

// Embedding observability wrapper (ADR D10)
pub use core::embedding::TokenTrackingEmbedder;

// Caller-facing pre-chunking helper (TD-232 / TD-234) — see the module doc
// comment on `core::chunking` for why this exists and what it deliberately
// does NOT do (kremory never calls it automatically).
pub use core::chunking::split_for_embedding;

/// Observability primitives (v0.1.2): token tracking wrappers, provider rates, init.
///
/// Use these types to wire custom ChatProvider / EmbeddingProvider instances
/// with full metric emission (tokens, cost, duration). The Tier 1 shortcuts
/// (`with_ollama`, `with_openai`, `with_anthropic`) auto-wrap via these types.
pub mod observability {
    pub use crate::core::chat_tracking::{llm_error_type, TokenTrackingChatProvider};
    pub use crate::core::embedding::TokenTrackingEmbedder;
    pub use crate::core::rates::{
        CostUsdParams, ProviderRateEntry, ProviderRates, RatesError, PROVIDER_RATES,
    };
    pub use crate::memory::{init_telemetry, TelemetryConfig, TelemetryHandle, TelemetryInitError};
}

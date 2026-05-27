//! kremory — bi-temporal knowledge graph engine + agent memory orchestration.
//!
//! # Modules
//!
//! - [`facade`] — fluent `Memory` facade: the recommended public API for most consumers.
//! - [`core`] — graph primitives: episode ingest, extraction, contradiction handling,
//!   hybrid retrieval, bi-temporal queries, community detection. (was `rql-core`)
//! - [`memory`] — orchestration layer: multi-tenant scoping, batch consolidation
//!   recipe (`run_dream_phase`), opinionated retrieval defaults, context-block
//!   templates. (was `rql-memory`)
//!
//! # BYOM contract
//!
//! `ChatProvider` is the canonical LLM abstraction. Re-exported from
//! [`memory`] so SDK consumers depend on `kremory` only.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use kremory::{Memory, Namespace};
//! # async fn ex() -> kremory::memory::Result<()> {
//! // Env-detected provider (OLLAMA_HOST or OPENAI_API_KEY):
//! let mem = Memory::auto("./agent.db")
//!     .await?;
//! mem.remember("User prefers concise replies").await?;
//! let context = mem.recall("what does user prefer?").await?;
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod facade;
pub mod memory;

// Convenience re-exports from core
pub use core::error::{Error as CoreError, Result as CoreResult};
pub use core::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestGuard, IngestSendError, IngestorConfig,
};
// Process-global graph singleton — spec §2 step 3 (consumer-facing, intentional).
pub use core::engine::{engine, engine_init};

// ── Facade re-exports (Tier 1 / Tier 2 public surface — Story A.8) ───────────
pub use facade::{
    DreamFireAndForget, DreamRequest, DreamSummary, EpisodeEntryBuilder, ForgetRequest, Memory,
    MemoryBuilder, NoEmb, NoLlm, RecallRawRequest, RecallRequest, RecallTemplate,
    RememberBatchBuilder, RememberRequest, WithEmb, WithLlm,
};

// Convenience re-exports from memory
pub use memory::init_telemetry;
pub use memory::ChatProvider;
pub use memory::{
    ContextTemplate, DreamPhaseResult, GraphHandle, IngestResult, MemoryError, MemoryType,
    Namespace, Result as MemoryResult, RetrievedContext, SearchOpts, SourceKind, SourceRef,
    StructuredFact, TelemetryConfig, TelemetryHandle, TelemetryInitError,
};
// Handle / lifecycle types (facade + substrate consumers)
pub use memory::types::{
    AwaitOpts, BatchStatus, CancelOutcome, CancelledPhase, DreamHandle, DreamMode, DreamOpts,
    DreamStatus, EpisodeCommit, SubmitOpts,
};
pub use memory::IngestStatus;
// Event sinks + event types (Tier 2 consumers)
pub use core::sink::{ContradictionDetected, IngestEventSink, IngestionError};
pub use memory::events::BatchPhase2Complete;
pub use memory::events::EnrichmentEventSink;
// BYOM provider traits
pub use core::provider::{ArcEmbedder, DynEmbeddingProvider, EmbeddingProvider};

// Convenience re-exports from core::config (ADR D15)
pub use core::config::Config as CoreConfig;

// Embedding observability wrapper (ADR D10)
pub use core::embedding::TokenTrackingEmbedder;

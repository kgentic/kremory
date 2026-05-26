//! kremory — bi-temporal knowledge graph engine + agent memory orchestration.
//!
//! # Modules
//!
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

pub mod core;
pub mod memory;

// Convenience re-exports from core
pub use core::error::{Error as CoreError, Result as CoreResult};
pub use core::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestGuard, IngestSendError, IngestorConfig,
};

// Convenience re-exports from memory
pub use memory::init_telemetry;
pub use memory::ChatProvider;
pub use memory::{
    ContextTemplate, DreamPhaseResult, GraphHandle, IngestResult, MemoryError, MemoryType,
    Result as MemoryResult, RetrievedContext, SearchOpts, SourceKind, SourceRef, StructuredFact,
    TelemetryConfig, TelemetryHandle, WorkspaceScope,
};

// Convenience re-exports from core::config (ADR D15)
pub use core::config::Config as CoreConfig;

// Embedding observability wrapper (ADR D10)
pub use core::embedding::TokenTrackingEmbedder;

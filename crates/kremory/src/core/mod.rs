//! kremory::core — bi-temporal knowledge graph primitives.
//!
//! This module contains the per-episode graph primitives: entity/edge extraction,
//! dedup, fact invalidation, contradiction handling, hybrid retrieval, bi-temporal
//! queries, and community detection.
//!
//! Corresponds to the former `rql-core` crate (crates/rqlc/).
//!
//! # BYOM invariant
//!
//! `core` consumes the `ChatProvider` trait from `autoagents-llm` ONLY — never
//! the concrete `autoagents-llamacpp` impl. Consumers wire their own backend.
//!
//! # Lock ordering
//!
//! All locks in this module must be acquired in the following fixed order to
//! prevent deadlocks. Never acquire a lock with a higher index while holding
//! one with a lower index.
//!
//! | Index | Lock | Location | Guards |
//! |-------|------|----------|--------|
//! | 1 | `write_lock: tokio::sync::Mutex<()>` | `TemporalGraph` (ADR-022) | Serialises concurrent write transactions (BEGIN IMMEDIATE). Acquired first — before any sub-lock — on every write path. |
//! | 2 | `session: std::sync::Mutex<ort::Session>` | `OrtEmbeddingProvider`, `NerModel` | Guards the ORT inference session. Short critical section; never held across await points. |
//! | 3 | `entries: std::sync::Mutex<HashMap>` | `SpeculativeCache` | Guards speculative-cache entries. Never held while acquiring index-2 locks. |
//! | 4 | `error_rx: std::sync::Mutex<Receiver<IngestError>>` | `BackgroundIngestor` | Guards the ingest-error channel receiver. Never held while acquiring index-1, -2, or -3 locks. |
//!
//! ## Rules
//!
//! 1. The ADR-022 `write_lock` (index 1) is **always acquired first** on any
//!    write path through `TemporalGraph`. It is held for the duration of the
//!    SQLite write transaction and released only after `COMMIT` or `ROLLBACK`.
//! 2. ORT session locks (index 2) are held only during model inference, never
//!    across an `.await`. Do not call any async function while holding one.
//! 3. `SpeculativeCache` (index 3) and the error-channel receiver (index 4)
//!    are structurally independent of each other and of ORT sessions. Acquire
//!    at most one of these at a time on any single code path.
//! 4. When adding new `Mutex` / `RwLock` fields to any type in this module,
//!    assign an index higher than any lock it may be nested inside, and update
//!    this table before merging.

pub mod arena;
pub mod background;
pub mod config;
pub mod context;
pub mod contradiction;
pub mod embedding;
pub mod error;
pub mod extraction;
pub mod extraction_window;
pub mod graph;
pub mod grounding;
pub mod hybrid_extractor;
pub mod ingest;
pub mod intelligence;
pub mod migrations;
#[cfg(feature = "ner")]
pub mod ner;
pub mod provider;
pub mod resolver;
pub mod schema;
pub mod search;
pub mod sink;
pub mod speculative_cache;
pub mod text_utils;

pub use background::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestGuard, IngestSendError, IngestorConfig,
};
pub use error::{ContradictionResolution, IngestStatus, IngestionErrorKind, Result, RqlError};
pub use sink::{
    ContradictionDetected, EntityId, EntityOrEdgeRef, Fact as SinkFact, IngestEventSink,
    IngestionError as SinkIngestionError,
};

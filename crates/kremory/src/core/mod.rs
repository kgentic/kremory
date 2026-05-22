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

pub mod arena;
pub mod background;
pub mod extraction_window;
pub mod config;
pub mod context;
pub mod contradiction;
pub mod embedding;
pub mod error;
pub mod extraction;
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

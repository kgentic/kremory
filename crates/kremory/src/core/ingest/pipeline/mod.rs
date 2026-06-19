//! Ingest pipeline — entity extraction, resolution, contradiction, persistence.
//!
//! Split from a single 2141-LoC `pipeline.rs` (TD-045 Wave 2) into per-method
//! `impl Engine` continuation files. Originally split from `ingest.rs` (TD-001 E0-C).

mod deferred;
mod ingest_with;
mod phase1;
mod types;

pub use deferred::IngestDeferredParams;
pub use ingest_with::IngestWithParams;
pub use phase1::WriteVerifiedEntitiesParams;
pub use types::{EntityCandidate, IngestPhase1Result, ResolvedDecision, UpsertedEntities};

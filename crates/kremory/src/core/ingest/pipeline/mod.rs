//! Ingest pipeline — entity extraction, resolution, contradiction, persistence.
//!
//! Split from a single 2141-LoC `pipeline.rs` into per-method `impl Engine`
//! continuation files. Originally split from `ingest.rs`.

mod deferred;
mod deferred_emissions;
mod entity_rules;
mod entity_type_registry;
mod entity_upsert;
mod fact_rules;
mod forward_refs;
mod ingest_with;
mod phase1;
mod pre_pinned;
mod types;

pub use deferred::IngestDeferredParams;
pub use ingest_with::IngestWithParams;
pub use phase1::WriteVerifiedEntitiesParams;
pub use types::{EntityCandidate, IngestPhase1Result, ResolvedDecision, UpsertedEntities};

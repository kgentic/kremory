//! `kremory-napi` — Node.js binding for the kremory `Memory` facade.
//!
//! Exposes `kremory::Memory` to TypeScript/JavaScript via napi-rs derive macros.
//! Single binding, no PyO3, no wasm-bindgen.
//!
//! # Binding surface
//!
//! - `JsMemory` wraps `kremory::Memory`. Async methods delegate to a tokio
//!   multi-thread runtime via napi-rs `async` feature.
//! - Plain data structs (`JsOpenOptions`, `JsRecallOptions`, `JsRememberOptions`,
//!   `JsStructuredFact`, `JsRetrievedContext`, `JsIngestResult`) are `#[napi(object)]` — napi-rs
//!   emits TS `interface` declarations for each.
//!
//! # Error mapping
//!
//! All `kremory::MemoryError` values are converted to `napi::Error::from_reason`
//! so they surface as JS `Error` rejections with a descriptive message.

#![deny(clippy::all)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod bridge;
mod convert;
mod memory;

pub use convert::{
    JsBatchOptions, JsBatchStatus, JsCancelOutcome, JsConsolidationOpsRan, JsDeleteEntityOutcome,
    JsDeleteFactOutcome, JsDreamOpts, JsDreamPassOpts, JsDreamStatusResult, JsDreamSummary,
    JsEditEntityOptions, JsEditEntityOutcome, JsEpisode, JsForgetOutcome, JsIngestResult,
    JsIngestStatusResult, JsMetadataFilter, JsMutationFilter, JsMutationRecord, JsOpenOptions, JsRecallOptions,
    JsRememberOptions, JsRestoreArchivedOutcome, JsRetrievedContext, JsStructuredFact,
    JsSupersedeOutcome, JsTypeProposal, JsUndoOutcome, JsUnmergeOutcome, JsUnsupersedeOutcome,
};

pub use memory::JsMemory;

//! Type-safe conversions between kremory Rust types and napi-rs JS objects.
//!
//! All `#[napi(object)]` structs here generate TypeScript `interface` declarations
//! in `index.d.ts` via the napi-rs derive macro pipeline.
//!
//! Split by domain (TD-243 file-size ratchet): `ingest` (open/remember/episode/batch/status
//! types), `recall` (retrieval/context/source types), `dream` (dream options/summary/status
//! types), `mutations` (supersede/undo/forget/edit/delete/mutation-record types). Re-exported
//! flat here so every `convert::X` path used elsewhere in this crate keeps resolving unchanged.

mod dream;
mod ingest;
mod mutations;
mod recall;

pub use dream::*;
pub use ingest::*;
pub use mutations::*;
pub use recall::*;

#[cfg(test)]
mod tests;

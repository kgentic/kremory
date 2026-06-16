//! Shared test-harness support modules (v0.2.4).
//!
//! Included by integration test binaries via `mod support;`. Each integration
//! test binary compiles its own copy of these modules, so a binary that uses
//! only part of the surface would otherwise trip the crate's strict
//! `unused`/dead-code lints — hence the module-level `allow` below. This mirrors
//! the existing `tests/helpers/` convention.
#![allow(dead_code)]

pub mod test_log;

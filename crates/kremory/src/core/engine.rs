//! Process-global `TemporalGraph` singleton. Story #5.
//!
//! `init_engine` initialises the singleton once; subsequent calls return
//! `Err(Error::EngineAlreadyInitialised)` to surface double-init bugs at
//! startup rather than silently succeeding with a stale handle. Call
//! `engine()` anywhere after init to obtain the `Arc<TemporalGraph>`.
//!
//! # Why OnceLock
//!
//! One process = one SQLite file = one connection. `OnceLock` gives a
//! zero-cost read path (no lock on hot reads once set) while making the
//! write path safe — `std::sync::OnceLock::set` is an atomic CAS. The
//! alternative (`lazy_static` or `once_cell::Lazy`) would initialise
//! eagerly or hide the `Result` path.
//!
//! # BYOM invariant
//!
//! This module is pure graph infrastructure — no LLM calls, no autoagents
//! concrete types. Embedding providers are wired by the caller before
//! handing the `Arc<TemporalGraph>` to upper layers.

use std::sync::{Arc, OnceLock};

use crate::core::error::{Error, Result};
use crate::core::schema::TemporalGraph;

static ENGINE: OnceLock<Arc<TemporalGraph>> = OnceLock::new();

/// Initialise the process-global `TemporalGraph` singleton.
///
/// Opens the graph at `path` and stores it in `ENGINE`. Safe to call
/// from any async context — the underlying `TemporalGraph::open` is `async`
/// and this function does not block the thread.
///
/// # Errors
///
/// - Returns `Err(Error::EngineAlreadyInitialised)` if `init_engine` was
///   already called successfully in this process.
/// - Propagates any `TemporalGraph::open` error (I/O, migration failure).
pub async fn init_engine(path: &str) -> Result<Arc<TemporalGraph>> {
    let graph = Arc::new(TemporalGraph::open(path).await?);
    ENGINE
        .set(graph.clone())
        .map_err(|_| Error::EngineAlreadyInitialised)?;
    Ok(graph)
}

/// Return a clone of the process-global `Arc<TemporalGraph>`.
///
/// # Errors
///
/// Returns `Err(Error::EngineNotInitialised)` when called before
/// `init_engine` has successfully returned.
pub fn engine() -> Result<Arc<TemporalGraph>> {
    ENGINE.get().cloned().ok_or(Error::EngineNotInitialised)
}

#[cfg(test)]
mod tests {
    // OnceLock is process-global and cannot be reset between tests.
    // Engine tests therefore run in isolated integration-test binaries,
    // not in the same process as unit tests that may race on init.
    // This module holds compile-time contract checks only.

    use super::*;

    #[test]
    fn engine_returns_not_initialised_before_init() {
        // In a fresh test binary ENGINE is always uninitialised.
        // If another test in this binary already initialised it, skip.
        if ENGINE.get().is_none() {
            let result = engine();
            assert!(
                matches!(result, Err(Error::EngineNotInitialised)),
                "expected EngineNotInitialised"
            );
        }
    }
}

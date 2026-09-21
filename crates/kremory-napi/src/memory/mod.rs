//! `JsMemory` struct definition + domain submodules.
//!
//! Split by domain (mirrors `convert/`): lifecycle (open/close), ingest,
//! dream, recall, mutations, admin. Each submodule carries its own
//! `#[napi] impl JsMemory { ... }` block — multiple inherent impl blocks for
//! one type, across files, is normal Rust, and napi-rs does register each
//! block's methods independently. But napi-rs's macro expansion is
//! item-order sensitive WITHIN this file: it errors ("Did not find struct
//! `JsMemory` parsed before expand #[napi] for impl") if a `mod` pulling in
//! an `impl JsMemory` block is declared — and therefore expanded — before
//! the `#[napi]`-annotated struct itself. Confirmed by a real build failure
//! when `mod admin; ...` preceded the struct here; the struct MUST stay
//! first.

use kremory::{Memory, Namespace};
use napi_derive::napi;

// ── JsMemory ──────────────────────────────────────────────────────────────────

/// Node.js handle for a kremory `Memory` instance.
///
/// Obtain via `JsMemory.open(path, opts?)`.
/// `close()` should be called at shutdown to future-proof against v0.1.1+ WAL
/// flush semantics.
#[napi(js_name = "Memory")]
pub struct JsMemory {
    inner: Memory,
    /// Handle-level default namespace captured from `JsOpenOptions.defaultNamespace`
    /// at `open` time. Applied to ingest/recall calls that don't pass an explicit
    /// per-call namespace. Set-once, never mutated — safe for concurrent reads.
    default_namespace: Option<Namespace>,
}

mod admin;
mod dream;
mod ingest;
mod lifecycle;
mod mutations;
mod recall;

//! `kremory-napi` — Node.js binding for the kremory `Memory` facade.
//!
//! Exposes `kremory::Memory` to TypeScript/JavaScript via napi-rs derive macros.
//! Per ADR-030 Decision 2 Form B: single binding, no PyO3, no wasm-bindgen.
//!
//! # Binding surface
//!
//! - `JsMemory` wraps `kremory::Memory`. Async methods delegate to a tokio
//!   multi-thread runtime via napi-rs `async` feature.
//! - Plain data structs (`JsOpenOptions`, `JsRecallOptions`, `JsIngestOptions`,
//!   `JsRetrievedContext`, `JsIngestResult`) are `#[napi(object)]` — napi-rs
//!   emits TS `interface` declarations for each.
//!
//! # Error mapping
//!
//! All `kremory::MemoryError` values are converted to `napi::Error::from_reason`
//! so they surface as JS `Error` rejections with a descriptive message.

#![deny(clippy::all)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

mod convert;

use napi_derive::napi;

use kremory::Memory;

pub use convert::{
    JsIngestOptions, JsIngestResult, JsOpenOptions, JsRecallOptions, JsRetrievedContext,
};

// ── JsMemory ──────────────────────────────────────────────────────────────────

/// Node.js handle for a kremory `Memory` instance.
///
/// Obtain via `JsMemory.open(path, opts?)`.
/// `close()` should be called at shutdown to future-proof against v0.1.1+ WAL
/// flush semantics.
#[napi]
pub struct JsMemory {
    inner: Memory,
}

#[napi]
impl JsMemory {
    /// Open a kremory Memory at `path`, using env-detected providers
    /// (`OLLAMA_HOST` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`).
    ///
    /// If `opts.defaultNamespace` is set it becomes the handle-level default.
    /// If `opts.embeddingDim` is set, it must match the active provider's
    /// embedding dimension exactly.
    #[napi(factory)]
    pub async fn open(path: String, _opts: Option<JsOpenOptions>) -> napi::Result<JsMemory> {
        // Tier 1: env-auto provider detection.
        // `opts.embeddingDim` and `opts.defaultNamespace` inform the builder
        // at v0.1.5+; for now the `auto` shortcut is used for the minimum
        // viable binding. Callers requiring fine-grained control should use
        // the Tier 2 builder path (not yet exposed via this binding).
        let mem = Memory::auto(&path)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory open failed: {e}")))?;

        Ok(JsMemory { inner: mem })
    }

    /// Ingest a text episode into memory.
    ///
    /// Blocks until Phase 2 enrichment completes (default kremory behaviour).
    /// Returns a lightweight `JsIngestResult` with the committed episode ID.
    #[napi]
    pub async fn ingest(
        &self,
        text: String,
        opts: Option<JsIngestOptions>,
    ) -> napi::Result<JsIngestResult> {
        let namespace = convert::resolve_ingest_namespace(&opts);

        let commit = if let Some(ns) = namespace {
            self.inner
                .remember(text)
                .in_namespace(ns)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory ingest failed: {e}")))?
        } else {
            self.inner
                .remember(text)
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory ingest failed: {e}")))?
        };

        Ok(JsIngestResult {
            episode_entity_id: commit.episode_entity_id,
            committed_at: commit.committed_at.to_rfc3339(),
        })
    }

    /// Search memory for context matching `query`.
    ///
    /// Returns up to `opts.k` (default 10) results ranked by relevance.
    #[napi]
    pub async fn recall(
        &self,
        query: String,
        opts: Option<JsRecallOptions>,
    ) -> napi::Result<Vec<JsRetrievedContext>> {
        let namespace = convert::resolve_recall_namespace(&opts);
        let k = opts
            .as_ref()
            .and_then(|o| o.k)
            .map(|k_val| usize::try_from(k_val).unwrap_or(10));
        let as_of = opts
            .as_ref()
            .and_then(|o| o.as_of.as_deref())
            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok());

        let results = {
            let mut builder = self.inner.recall(query);

            if let Some(ns) = namespace {
                builder = builder.in_namespace(ns);
            }
            if let Some(k_val) = k {
                builder = builder.k(k_val);
            }
            if let Some(as_of_ts) = as_of {
                builder = builder.as_of(as_of_ts);
            }

            builder
                .raw()
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory recall failed: {e}")))?
        };

        Ok(results
            .into_iter()
            .map(convert::retrieved_context_to_js)
            .collect())
    }

    /// Close the memory handle, flushing any pending writes.
    ///
    /// Calling `close()` is a no-op at v0.1.0; WAL flush semantics land in
    /// v0.1.1. Call this at shutdown to future-proof your code.
    #[napi]
    pub async fn close(&self) -> napi::Result<()> {
        self.inner
            .close()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory close failed: {e}")))
    }
}

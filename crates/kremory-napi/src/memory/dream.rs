//! Dream/consolidation: `dream`, sync single-pass dreaming, ghost-episode
//! listing, entity-type pinning (protects against dream reclassification),
//! and the await/cancel surface for in-flight dream handles.

use napi_derive::napi;
use super::JsMemory;
use crate::convert;
use crate::convert::*;
use kremory::Namespace;

#[napi]
impl JsMemory {
    /// Trigger the dream-phase batch consolidation.
    ///
    /// Blocks until the dream completes. Returns a `JsDreamSummary` with per-phase
    /// accounting (incl. the `crossEpisodeWouldMerge`/`crossEpisodeMerged` split
    /// and the `consolidationOpsRan` ran-signal). Wraps `Memory::dream()`,
    /// threading the full `DreamOptions` consolidation-control surface — every
    /// consolidation knob (community / archival / supersession / cross-episode mode /
    /// budgets / grace / warn-floor) is now reachable from JS. An unknown
    /// `opts.crossEpisodeMode` rejects the Promise (loud parse).
    #[napi]
    pub async fn dream(&self, opts: Option<JsDreamOpts>) -> napi::Result<JsDreamSummary> {
        // Resolve namespace BEFORE opts is moved into the conversion (namespace is
        // NOT a DreamOpts field — it routes via `.in_namespace(ns)`).
        let ns = opts
            .as_ref()
            .and_then(|o| o.namespace.as_deref())
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let rust_opts = convert::js_dream_opts_to_rust(opts)?;

        let mut req = self.inner.dream().with_opts(rust_opts);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }

        let summary = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory dream failed: {e}")))?;

        Ok(convert::dream_summary_to_js(summary))
    }

    /// Run a single dream pass synchronously.
    ///
    /// Lower-level than `dream()` — calls the engine's `run_dream_pass_sync`
    /// directly with explicit pass options. Concurrent calls serialize via an
    /// internal `Mutex` on `Engine`. Wraps `Memory::run_dream_pass_sync`.
    #[napi]
    pub async fn run_dream_pass_sync(
        &self,
        opts: Option<JsDreamPassOpts>,
    ) -> napi::Result<JsDreamSummary> {
        let rust_opts = convert::js_dream_pass_opts_to_rust(opts);
        let summary = self
            .inner
            .run_dream_pass_sync(rust_opts)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory run_dream_pass_sync failed: {e}"))
            })?;
        Ok(convert::dream_summary_to_js(summary))
    }

    /// Return episode IDs where Phase 1 ingest succeeded but Phase 2 produced
    /// no facts (ghost episodes).
    ///
    /// `group_id` restricts the query to one namespace. Omit/`null` to return
    /// ghost episodes across all namespaces. Wraps `Memory::ghost_episodes`.
    #[napi]
    pub async fn ghost_episodes(&self, group_id: Option<String>) -> napi::Result<Vec<i64>> {
        self.inner
            .ghost_episodes(group_id.as_deref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory ghost_episodes failed: {e}")))
    }

    /// Pin an entity as `ConsumerPinned`, protecting it from dream
    /// reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
    /// Wraps `Memory::assert_entity_type`.
    #[napi]
    pub async fn assert_entity_type(
        &self,
        entity_id: String,
        entity_type_id: u32,
        group_id: Option<String>,
    ) -> napi::Result<()> {
        self.inner
            .assert_entity_type(kremory::GraphAssertEntityTypeParams {
                entity_id: &entity_id,
                entity_type_id,
                group_id: group_id.as_deref(),
            })
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory assert_entity_type failed: {e}"))
            })
    }

    /// Block until the dream-phase run identified by `handleId` reaches a
    /// terminal status.
    ///
    /// Wraps `Memory::await_dream`. `timeoutMs` is mandatory.
    /// Returns `DreamStatusResult` with `status` one of:
    /// `"pending"` | `"processing"` | `"complete"` | `"failed"`.
    ///
    /// # Currently unreachable
    ///
    /// No public JS (or Rust facade) call currently PRODUCES a `handleId` —
    /// `Memory.dream()` always blocks inline and returns a `DreamSummary`
    /// directly (see `parity-skip.toml`'s `DreamRequest::fire_and_forget`
    /// entry: "Async fire-and-forget dream; `Memory.dream` always blocks
    /// inline"). Calling this method with any UUID today will time out or
    /// error — there is no way to obtain a live `handleId` first. A
    /// fire-and-forget dream entry point that returns a real handle is a
    /// separate, larger change (mirroring the entry point, not just this
    /// wrapper); tracked, not implemented here.
    #[napi]
    pub async fn await_dream(
        &self,
        handle_id: String,
        timeout_ms: i64,
    ) -> napi::Result<JsDreamStatusResult> {
        let run_id = uuid::Uuid::parse_str(&handle_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let handle = kremory::DreamHandle {
            run_id,
            namespace: kremory::Namespace::new(""),
            submitted_at: chrono::Utc::now(),
            batch_id: None,
        };

        let timeout = std::time::Duration::from_millis(timeout_ms as u64);

        let status = self
            .inner
            .await_dream(&handle, timeout)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory awaitDream failed: {e}")))?;

        Ok(convert::dream_status_to_js(status))
    }

    /// Cancel a dream-phase run by its handle UUID.
    ///
    /// Wraps `Memory::cancel_dream`. `handleId` is the RFC-4122 UUID string of
    /// the dream run. Returns `CancelOutcome`.
    ///
    /// # Currently unreachable
    ///
    /// Same gap as `awaitDream` above: no public call produces a `handleId`
    /// to cancel, because `Memory.dream()` always blocks inline. See that
    /// method's doc comment for the full explanation.
    #[napi]
    pub async fn cancel_dream(&self, handle_id: String) -> napi::Result<JsCancelOutcome> {
        let run_id = uuid::Uuid::parse_str(&handle_id)
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;

        let handle = kremory::DreamHandle {
            run_id,
            namespace: kremory::Namespace::new(""),
            submitted_at: chrono::Utc::now(),
            batch_id: None,
        };

        let outcome =
            self.inner.cancel_dream(&handle).await.map_err(|e| {
                napi::Error::from_reason(format!("kremory cancelDream failed: {e}"))
            })?;

        Ok(convert::cancel_outcome_to_js(outcome))
    }

}

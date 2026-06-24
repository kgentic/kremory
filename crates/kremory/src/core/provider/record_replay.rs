// ---------------------------------------------------------------------------
// RecordReplayChatProvider — trait-seam VCR decorator (v0.2.4 Component 1).
//
// A record/replay decorator over `Arc<dyn ChatProvider>` for deterministic,
// no-Ollama CI runs of the golden-path smoke test (Tier 2 of the 3-tier test
// pyramid). See `.ai-docs/specs/v0-2-4-test-infra-o11y-harness-arch-spec-2026-06-15.md`.
//
// Three modes:
//   Record      — delegate to the wrapped real provider, capture
//                 (fingerprint, response_text) into the shared cassette buffer,
//                 flush to disk via an explicit `.flush()` (NOT Drop — the
//                 decorator Arc is cloned into the Phase-2 worker thread, so
//                 Drop ordering is non-deterministic).
//   Replay      — inner is None; match the request fingerprint against a loaded
//                 cassette and return the recorded response via a per-fingerprint
//                 cursor. On MISS: LOUD error naming the unmatched fingerprint
//                 (NO silent default, NO fallthrough-to-live — Decision D3).
//   Passthrough — delegate, no capture.
//
// Thread-safety (NEW-201): cassette state (replay cursor + record buffer) is
// held behind a SINGLE `Arc<Mutex<CassetteState>>`, NOT RefCell/Cell, because
// `ChatProvider: Send + Sync` and the decorator Arc is shared across the
// background worker thread.
//
// model resolution (Option-1, 2026-06-23 — supersedes the former ASMP-002
// `ChatProvider::model()` override, removed when published autoagents-llm 0.3.7
// dropped the trait method): the model is a caller-supplied field exposed via the
// inherent `model_id()`. In Record/Passthrough it is the value passed to
// `record()`/`passthrough()`; in Replay it is the cassette header's stored `model`.
// The header is a plain (non-Mutex) field so `model_id()` can return `&str` directly.
//
// Fingerprint (§4.2): Sha256 of (model || messages-json || schema-json ||
// tools-count-marker), stable across runs/platforms.
//
// Gated: test-infra only — never part of the production public API.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use autoagents_llm::chat::{ChatMessage, ChatProvider, StructuredOutputFormat};
use autoagents_llm::error::LLMError;

use super::MockChatResponse;

/// One recorded chat response, keyed under a fingerprint in call order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CassetteEntry {
    /// 0-based position of this response among repeated identical-fingerprint
    /// calls within one journey. Replayed in ascending `call_index` order.
    pub call_index: usize,
    /// The raw response text the provider returned (the LLM's structured-output
    /// JSON, or whatever `ChatResponse::text()` yielded).
    pub response_text: String,
}

/// On-disk cassette file format (§4.1). The top-level `model` header is
/// LOAD-BEARING (DENT-001; Option-1 2026-06-23): Replay-mode `model_id()` returns
/// it so extraction builders route to the correct capability arm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cassette {
    /// Format version (currently 1).
    pub version: u32,
    /// REQUIRED, load-bearing model header — returned by `model()` in Replay
    /// mode so `capability_of(model)` selects `FormatSchema` not `PromptOnly`.
    pub model: String,
    /// Review/staleness-audit metadata only — NOT part of matching.
    pub recorded_at: String,
    /// fingerprint-hex → ordered list of recorded responses.
    pub entries: std::collections::BTreeMap<String, Vec<CassetteEntry>>,
}

/// Shared, mutable cassette state behind a single `Mutex` (NEW-201).
///
/// In Record mode `entries` is the capture buffer (appended to from any thread,
/// including the Phase-2 worker thread). In Replay mode `cursors` tracks how many
/// responses have already been popped per fingerprint, and `entries` is the
/// loaded cassette's response lists.
#[derive(Debug, Default)]
struct CassetteState {
    /// fingerprint-hex → ordered list of recorded/loaded responses.
    entries: std::collections::BTreeMap<String, Vec<CassetteEntry>>,
    /// fingerprint-hex → next index to return (Replay cursor).
    cursors: HashMap<String, usize>,
}

/// VCR mode for [`RecordReplayChatProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcrMode {
    /// Delegate to the inner provider and capture responses.
    Record,
    /// Match against the loaded cassette; loud error on miss.
    Replay,
    /// Delegate to the inner provider, capture nothing.
    Passthrough,
}

/// Trait-seam record/replay decorator over an `Arc<dyn ChatProvider>`.
///
/// See the module-level comment above for the full design. Construct via
/// [`RecordReplayChatProvider::record`], [`RecordReplayChatProvider::replay`],
/// or [`RecordReplayChatProvider::passthrough`].
pub struct RecordReplayChatProvider {
    mode: VcrMode,
    /// `Some` for Record/Passthrough (the real provider); `None` for Replay.
    inner: Option<Arc<dyn ChatProvider>>,
    /// Shared cassette state (capture buffer + replay cursor) — see NEW-201.
    state: Arc<Mutex<CassetteState>>,
    /// Stored model header (Option-1 2026-06-23): plain field so `model_id()`
    /// returns `&str` directly without holding the `Mutex`. In Record/Passthrough
    /// this is the caller-supplied model passed to `record()`/`passthrough()`; in
    /// Replay it is the cassette header `model`.
    model: String,
    /// Cassette file path (written by `flush()` in Record mode; loaded from in
    /// Replay mode). `None` in Passthrough mode.
    cassette_path: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for RecordReplayChatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordReplayChatProvider")
            .field("mode", &self.mode)
            .field("has_inner", &self.inner.is_some())
            .field("model", &self.model)
            .field("cassette_path", &self.cassette_path)
            .finish()
    }
}

/// Bundled parameters for [`RecordReplayChatProvider::fingerprint`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
struct FingerprintParams<'a> {
    /// Chat messages contributing to the fingerprint.
    messages: &'a [ChatMessage],
    /// Optional tool definitions (only the count is fingerprinted).
    tools: Option<&'a [autoagents_llm::chat::Tool]>,
    /// Optional structured-output schema.
    json_schema: Option<&'a StructuredOutputFormat>,
}

impl RecordReplayChatProvider {
    /// Record mode: wrap a real provider and capture responses to `cassette_path`
    /// on [`Self::flush`].
    ///
    /// Option-1 (2026-06-23): `model` is supplied by the caller (it can no longer
    /// be read off `inner.model()` — that trait method does not exist on published
    /// `autoagents-llm` 0.3.7). The string is load-bearing: it feeds the request
    /// fingerprint + cassette header, so record/replay must use the same value.
    pub fn record(
        inner: Arc<dyn ChatProvider>,
        cassette_path: impl Into<std::path::PathBuf>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            mode: VcrMode::Record,
            inner: Some(inner),
            state: Arc::new(Mutex::new(CassetteState::default())),
            model: model.into(),
            cassette_path: Some(cassette_path.into()),
        }
    }

    /// Replay mode: load the cassette at `cassette_path`. No inner provider.
    /// The decorator's `model()` returns the cassette header's `model` string.
    ///
    /// # Errors
    /// Returns [`LLMError::ProviderError`] if the cassette file cannot be read
    /// or fails to parse (loud, per [[llm-output-parse-loudly]]).
    pub fn replay(
        cassette_path: impl Into<std::path::PathBuf>,
    ) -> std::result::Result<Self, LLMError> {
        let path = cassette_path.into();
        let bytes = std::fs::read(&path).map_err(|e| {
            LLMError::ProviderError(format!(
                "RecordReplayChatProvider: failed to read cassette {}: {e}",
                path.display()
            ))
        })?;
        let cassette: Cassette = serde_json::from_slice(&bytes).map_err(|e| {
            LLMError::ProviderError(format!(
                "RecordReplayChatProvider: failed to parse cassette {}: {e}",
                path.display()
            ))
        })?;
        let model = cassette.model.clone();
        let state = CassetteState {
            entries: cassette.entries,
            cursors: std::collections::HashMap::new(),
        };
        Ok(Self {
            mode: VcrMode::Replay,
            inner: None,
            state: Arc::new(Mutex::new(state)),
            model,
            cassette_path: Some(path),
        })
    }

    /// Passthrough mode: delegate to a real provider, capture nothing.
    ///
    /// Option-1 (2026-06-23): `model` is caller-supplied (no longer read off
    /// `inner.model()` — absent on published 0.3.7).
    pub fn passthrough(inner: Arc<dyn ChatProvider>, model: impl Into<String>) -> Self {
        Self {
            mode: VcrMode::Passthrough,
            inner: Some(inner),
            state: Arc::new(Mutex::new(CassetteState::default())),
            model: model.into(),
            cassette_path: None,
        }
    }

    /// kremory-inherent model accessor (Option-1, 2026-06-23). Returns the model
    /// string captured at construction (cassette header in Replay; caller-supplied
    /// in Record/Passthrough). Replaces the former `ChatProvider::model()` override.
    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// Deterministic fingerprint for a chat request (§4.2).
    ///
    /// `Sha256(model || 0x1F || json(messages) || 0x1F || json(schema) ||
    /// 0x1F || "tools:N")`, hex-encoded. Stable across Rust releases and
    /// platforms (unlike `DefaultHasher`).
    fn fingerprint(&self, params: FingerprintParams<'_>) -> std::result::Result<String, LLMError> {
        let FingerprintParams {
            messages,
            tools,
            json_schema,
        } = params;
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.model.as_bytes());
        hasher.update([0x1F]);
        let messages_json = serde_json::to_vec(messages).map_err(|e| {
            LLMError::ProviderError(format!(
                "RecordReplayChatProvider: failed to serialize messages for fingerprint: {e}"
            ))
        })?;
        hasher.update(&messages_json);
        hasher.update([0x1F]);
        // Fingerprint determinism depends on `serde_json::Value::Object` being
        // BTreeMap-backed (key-sorted) so that `schema.schema` serializes in a
        // stable byte order across runs. This holds because this crate uses
        // serde_json WITHOUT the `preserve_order` feature. If a future transitive
        // dep enables `preserve_order` via feature unification, `Value::Object`
        // flips to an insertion-order IndexMap and committed cassette fingerprints
        // silently break (same logical schema → different bytes → cassette MISS).
        let schema_json = match json_schema {
            Some(s) => serde_json::to_vec(s).map_err(|e| {
                LLMError::ProviderError(format!(
                    "RecordReplayChatProvider: failed to serialize schema for fingerprint: {e}"
                ))
            })?,
            None => b"null".to_vec(),
        };
        hasher.update(&schema_json);
        hasher.update([0x1F]);
        let tools_marker = format!("tools:{}", tools.map_or(0, <[_]>::len));
        hasher.update(tools_marker.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for b in digest {
            use std::fmt::Write as _;
            // Writing to a String is infallible; the result is discarded
            // deliberately. `?` would require a fallible error type we don't have.
            let _ = write!(hex, "{b:02x}");
        }
        Ok(hex)
    }

    /// Flush the recorded cassette to disk (NEW-202).
    ///
    /// MUST be called explicitly in the Record path AFTER `wait_for_processing`
    /// returns (so Phase-2 worker-thread writes have landed in the shared buffer)
    /// and BEFORE any cassette-entry-count assertion. Drop-based flush is NOT
    /// sufficient because the worker thread holds a clone of the decorator `Arc`.
    ///
    /// No-op in Replay/Passthrough modes (returns `Ok(())`).
    ///
    /// # Errors
    /// Returns [`LLMError::ProviderError`] if the cassette directory cannot be
    /// created or the file cannot be written/serialized.
    pub fn flush(&self) -> std::result::Result<(), LLMError> {
        if self.mode != VcrMode::Record {
            return Ok(());
        }
        let path = self.cassette_path.as_ref().ok_or_else(|| {
            LLMError::ProviderError(
                "RecordReplayChatProvider::flush: Record mode without a cassette path".to_string(),
            )
        })?;
        let entries = {
            let guard = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.entries.clone()
        };
        let cassette = Cassette {
            version: 1,
            model: self.model.clone(),
            recorded_at: "1970-01-01T00:00:00Z".to_string(),
            entries,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LLMError::ProviderError(format!(
                    "RecordReplayChatProvider::flush: failed to create {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let json = serde_json::to_vec_pretty(&cassette).map_err(|e| {
            LLMError::ProviderError(format!(
                "RecordReplayChatProvider::flush: failed to serialize cassette: {e}"
            ))
        })?;
        std::fs::write(path, json).map_err(|e| {
            LLMError::ProviderError(format!(
                "RecordReplayChatProvider::flush: failed to write {}: {e}",
                path.display()
            ))
        })?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ChatProvider for RecordReplayChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[autoagents_llm::chat::Tool]>,
        json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<
        Box<dyn autoagents_llm::chat::ChatResponse>,
        autoagents_llm::error::LLMError,
    > {
        match self.mode {
            VcrMode::Passthrough => {
                let inner = self.inner.as_ref().ok_or_else(|| {
                    LLMError::ProviderError(
                        "RecordReplayChatProvider: Passthrough mode without an inner provider"
                            .to_string(),
                    )
                })?;
                inner.chat_with_tools(messages, tools, json_schema).await
            }
            VcrMode::Record => {
                let inner = self.inner.as_ref().ok_or_else(|| {
                    LLMError::ProviderError(
                        "RecordReplayChatProvider: Record mode without an inner provider"
                            .to_string(),
                    )
                })?;
                let fp = self.fingerprint(FingerprintParams {
                    messages,
                    tools,
                    json_schema: json_schema.as_ref(),
                })?;
                let response = inner.chat_with_tools(messages, tools, json_schema).await?;
                let text = response.text().unwrap_or_default();
                {
                    let mut guard = self
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let list = guard.entries.entry(fp).or_default();
                    let call_index = list.len();
                    list.push(CassetteEntry {
                        call_index,
                        response_text: text,
                    });
                }
                Ok(response)
            }
            VcrMode::Replay => {
                let fp = self.fingerprint(FingerprintParams {
                    messages,
                    tools,
                    json_schema: json_schema.as_ref(),
                })?;
                let mut guard = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let idx = guard.cursors.get(&fp).copied().unwrap_or(0);
                let recorded = guard
                    .entries
                    .get(&fp)
                    .and_then(|list| list.get(idx))
                    .map(|e| e.response_text.clone());
                match recorded {
                    Some(text) => {
                        guard.cursors.insert(fp, idx + 1);
                        Ok(Box::new(MockChatResponse { text }))
                    }
                    None => {
                        // Observability counter (debug builds only): surface the
                        // cumulative miss count so a CI failure shows more than
                        // the first error.
                        #[cfg(debug_assertions)]
                        {
                            let cassette = self
                                .cassette_path
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_default();
                            metrics::counter!(
                                "kremory.test.cassette_miss_total",
                                "cassette" => cassette
                            )
                            .increment(1);
                        }
                        let path = self
                            .cassette_path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "<none>".to_string());
                        Err(LLMError::ProviderError(format!(
                            "RecordReplayChatProvider cassette MISS: no recorded response for \
                             fingerprint={fp} (model={model}, call_index={idx}, messages_len={n}). \
                             Re-record with KREMORY_VCR=record against live Ollama. Cassette: {path}",
                            model = self.model,
                            n = messages.len(),
                        )))
                    }
                }
            }
        }
    }

    // Option-1 (2026-06-23): the `ChatProvider::model()` override is removed (the
    // trait method does not exist on published 0.3.7). The model is now a
    // caller-supplied field exposed via the inherent `model_id()`.
}

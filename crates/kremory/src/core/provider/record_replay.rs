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

/// Matches a FULL RFC3339 timestamp — date **and** time, optional fractional
/// seconds, `Z` or `±HH:MM` offset. Deliberately NOT a bare `YYYY-MM-DD`.
///
/// The narrowness is the safety property. `chrono`'s `to_rfc3339()` — the only
/// thing that renders wall-clock into a prompt here — always emits the full
/// form, while episode PROSE routinely contains bare dates ("we met on
/// 2023-05-07"). Matching bare dates would rewrite corpus content and
/// re-fingerprint cassettes that work today.
/// `Option` rather than an `unwrap`/`expect`: `#[allow(clippy::…)]` is banned in
/// `src/` (rust-conventions), and a fail-safe degrade is better here anyway — a
/// non-compiling pattern disables canonicalisation (fingerprints stay MORE
/// specific) instead of panicking inside a hash. `canonicalization_pattern_compiles`
/// below pins that it does in fact compile, so the `None` arm is unreachable in
/// practice rather than silently load-bearing.
static RFC3339_IN_PROMPT: std::sync::LazyLock<Option<regex::Regex>> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
            .ok()
    });

/// Replace absolute timestamps with ORDER-PRESERVING rank tokens for hashing.
///
/// # Why this exists
///
/// The fingerprint hashes the whole request. The contradiction-detection prompt
/// interpolates `fact.valid_from.to_rfc3339()` at microsecond precision
/// (`core/contradiction.rs:137-171`), and for LLM-extracted facts `valid_from`
/// is the episode's `ref_time` — wall-clock at ingest, **by deliberate design**
/// (TD-187 decision, "Consequences accepted": `as_of()` is document-granular for
/// extracted facts). So a freshly-ingested fixture yields a different request on
/// every run, the SHA256 never repeats, and re-recording writes a key that can
/// never be hit again. Proven live 2026-08-13.
///
/// Wiring `published_at` into the extraction path to make this deterministic was
/// considered and **REJECTED**: `reference_time: None` is a load-bearing guard
/// that keeps 303 committed cassettes byte-identical (TD-187 §F1). Fixing five
/// cassettes by invalidating three hundred is not a fix.
///
/// # Why RANK rather than erasure
///
/// Erasing timestamps would make two requests that differ ONLY in *which fact is
/// earlier* collide — and that is precisely the distinction the prompt asks the
/// model to make ("Use the 'valid from' timestamps to judge which is later").
/// A cassette that silently replayed the wrong verdict there would be a guard
/// masking a real bug. Ranking keeps the ordering inside the hash: absolute
/// values drop out, relative order does not.
///
/// Timestamps that fail to parse are left verbatim — failing toward MORE
/// specificity, never less.
///
/// # Prior art
///
/// Standard VCR practice: `vcr` (Ruby), `betamax`, and `nock` all ship request
/// matchers that ignore volatile fields. `regex` is already a dependency
/// (`Cargo.toml:216`, prompt-injection sanitiser), so this adds no supply-chain
/// surface.
fn canonicalize_timestamps_for_fingerprint(content: &str) -> String {
    let Some(re) = RFC3339_IN_PROMPT.as_ref() else {
        return content.to_string();
    };
    let found: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
    if found.is_empty() {
        return content.to_string();
    }

    // Rank by INSTANT, not by string: `2023-01-01T00:00:00Z` and
    // `2023-01-01T01:00:00+01:00` are the same moment and must share a rank,
    // or an offset change alone would move the fingerprint.
    let mut instants: Vec<chrono::DateTime<chrono::Utc>> = found
        .iter()
        .filter_map(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .collect();
    instants.sort_unstable();
    instants.dedup();

    re.replace_all(content, |caps: &regex::Captures<'_>| {
        let raw = &caps[0];
        match chrono::DateTime::parse_from_rfc3339(raw) {
            Ok(dt) => {
                let utc = dt.with_timezone(&chrono::Utc);
                match instants.binary_search(&utc) {
                    Ok(rank) => format!("<T{rank}>"),
                    // Unreachable: `utc` was collected into `instants`
                    // above. Degrade to the literal rather than panic.
                    Err(_) => raw.to_string(),
                }
            }
            Err(_) => raw.to_string(),
        }
    })
    .into_owned()
}

/// Max characters of request text quoted in a replay-MISS error.
///
/// Large enough to show a drifting timestamp or a reordered candidate list —
/// the two things that actually move a fingerprint — and small enough that a
/// failing test's output stays readable.
const MISS_REQUEST_QUOTE_CHARS: usize = 1500;

/// Render the request messages for a replay-MISS error, truncated on a char
/// boundary.
///
/// Spec `v0-2-4-test-infra-o11y-harness-arch-spec-2026-06-15.md` §4.3 requires
/// the miss error to name "the (truncated) request so the failure is debuggable
/// without re-running with eprintln". The code sample beneath that sentence
/// omitted it and the implementation copied the sample; this closes the gap.
///
/// The cut is taken via `char_indices` because byte-slicing a `String` at a
/// fixed offset panics mid-codepoint, and the amount omitted is reported so a
/// silent truncation cannot be mistaken for a short request.
fn render_request_for_miss(messages: &[ChatMessage]) -> String {
    let rendered = messages
        .iter()
        .enumerate()
        .map(|(i, m)| format!("  [{i}] {:?}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let total = rendered.chars().count();
    if total <= MISS_REQUEST_QUOTE_CHARS {
        return rendered;
    }
    let cut = rendered
        .char_indices()
        .nth(MISS_REQUEST_QUOTE_CHARS)
        .map_or(rendered.len(), |(byte_idx, _)| byte_idx);
    format!(
        "{}\n  … [truncated {omitted} of {total} chars]",
        &rendered[..cut],
        omitted = total - MISS_REQUEST_QUOTE_CHARS,
    )
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
        // Absolute wall-clock timestamps are replaced by ORDER-PRESERVING rank
        // tokens before hashing (see `canonicalize_timestamps_for_fingerprint`).
        // Without this, any prompt carrying `to_rfc3339()` output — notably
        // contradiction detection, `core/contradiction.rs:137-171` — produces a
        // fresh fingerprint on every run and can never replay. The message sent
        // to the PROVIDER is untouched; only the hash input is canonicalised.
        let canonical: Vec<ChatMessage> = messages
            .iter()
            .map(|m| ChatMessage {
                content: canonicalize_timestamps_for_fingerprint(&m.content),
                ..m.clone()
            })
            .collect();
        let messages_json = serde_json::to_vec(&canonical).map_err(|e| {
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
                        // Spec §4.3 requires the REQUEST here, not just its
                        // length: the fingerprint hashes the full request, and
                        // the cassette stores only {call_index, response_text},
                        // so without this the input that drifted is recoverable
                        // from neither side and re-recording just writes a new
                        // hash over the same blind spot.
                        Err(LLMError::ProviderError(format!(
                            "RecordReplayChatProvider cassette MISS: no recorded response for \
                             fingerprint={fp} (model={model}, call_index={idx}, messages_len={n}). \
                             Re-record with KREMORY_VCR=record against live Ollama. Cassette: {path}\n\
                             live request was:\n{request}",
                            model = self.model,
                            n = messages.len(),
                            request = render_request_for_miss(messages),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::provider::chat_msg_user;

    /// A replay MISS must quote the REQUEST, not merely report its length.
    ///
    /// This module shipped with ZERO tests, and the spec's own P3 checklist
    /// (`v0-2-4-test-infra-o11y-harness-arch-spec-2026-06-15.md` §9) called for
    /// a miss test that was never written — which is why the following survived
    /// for months:
    ///
    /// §4.3 of that spec says, in prose, that the error names "the (truncated)
    /// request so the failure is debuggable without re-running with eprintln".
    /// The code sample printed DIRECTLY BENEATH that sentence formats only
    /// `messages_len={n}` — a length. The implementation copied the sample, not
    /// the sentence.
    ///
    /// The consequence is not cosmetic. The cassette is keyed by a SHA256 over
    /// the full request, and stores only `{call_index, response_text}` — never
    /// the request. So on a miss, neither the live request NOR the recorded one
    /// was recoverable, and "the fingerprint changed" was undiagnosable by
    /// construction. Re-recording cannot help: it writes a new hash and discards
    /// the input again. Five llm-integration tests sat unexplained on exactly
    /// this.
    #[tokio::test]
    async fn cassette_miss_error_quotes_the_request_not_just_its_length() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("empty-cassette.json");
        let cassette = Cassette {
            version: 1,
            model: "test-model".to_string(),
            recorded_at: "1970-01-01T00:00:00Z".to_string(),
            entries: std::collections::BTreeMap::new(),
        };
        std::fs::write(&path, serde_json::to_vec(&cassette).expect("serialize"))
            .expect("write cassette");

        let provider = RecordReplayChatProvider::replay(&path).expect("replay mode");

        // A distinctive marker plus a rendered timestamp — the shape that
        // actually drifts in the contradiction-detection prompt, where
        // `fact.valid_from.to_rfc3339()` is interpolated into the hashed bytes.
        const SENTINEL: &str = "PINEAPPLE-ON-PIZZA-SENTINEL";
        let request = format!("[1] alice -> works_at -> acme (valid from: 2023-01-01) {SENTINEL}");

        let err = provider
            .chat_with_tools(&[chat_msg_user(request)], None, None)
            .await
            .expect_err("an empty cassette must MISS");

        let msg = err.to_string();
        assert!(
            msg.contains(SENTINEL),
            "the miss error must quote the request so drift is visible without \
             re-running with eprintln (spec §4.3). Got: {msg}"
        );
    }

    /// The `None` arm of `RFC3339_IN_PROMPT` must be unreachable in practice.
    ///
    /// It exists so a bad pattern degrades instead of panicking inside a hash
    /// (`#[allow(clippy::…)]` is banned in `src/`). But a fail-safe that is
    /// silently ALWAYS taken would disable canonicalisation everywhere and read
    /// as working. This pins that the pattern compiles.
    #[test]
    fn canonicalization_pattern_compiles() {
        assert!(
            RFC3339_IN_PROMPT.is_some(),
            "the RFC3339 pattern must compile, else canonicalisation is silently off"
        );
    }

    /// Absolute timestamps must drop out of the fingerprint.
    ///
    /// This is the defect proven live on 2026-08-13: the same logical request,
    /// re-ingested, carries a different wall-clock and therefore a different
    /// SHA256, so the cassette key can never repeat.
    #[test]
    fn same_order_different_absolute_times_canonicalize_identically() {
        let monday = "  [1] a → p → b (valid from: 2026-08-13T15:11:21.276196+00:00)\n  \
                      [2] a → p → c (valid from: 2026-08-13T16:00:00.000000+00:00)";
        let tuesday = "  [1] a → p → b (valid from: 2019-01-02T03:04:05.000001+00:00)\n  \
                       [2] a → p → c (valid from: 2020-06-07T08:09:10.999999+00:00)";

        assert_eq!(
            canonicalize_timestamps_for_fingerprint(monday),
            canonicalize_timestamps_for_fingerprint(tuesday),
            "absolute wall-clock must not reach the hash when the ORDER is the same"
        );
    }

    /// THE ANTI-MASKING TEST — the load-bearing half.
    ///
    /// Erasing timestamps outright would make these two collide, and they must
    /// not: the contradiction prompt asks the model to "use the 'valid from'
    /// timestamps to judge which is later", so a cassette that replayed one
    /// verdict for both orderings would silently answer the wrong question.
    /// That would be a guard masking a real bug — the thing
    /// [[audit-what-guards-mask-before-deleting]] exists to prevent.
    #[test]
    fn swapping_which_fact_is_earlier_still_changes_the_canonical_form() {
        let b_first = "[1] a → p → b (valid from: 2023-01-01T00:00:00+00:00)\n\
                       [2] a → p → c (valid from: 2024-01-01T00:00:00+00:00)";
        let c_first = "[1] a → p → b (valid from: 2024-01-01T00:00:00+00:00)\n\
                       [2] a → p → c (valid from: 2023-01-01T00:00:00+00:00)";

        assert_ne!(
            canonicalize_timestamps_for_fingerprint(b_first),
            canonicalize_timestamps_for_fingerprint(c_first),
            "relative ORDER must survive canonicalisation, or the cassette could \
             replay the wrong supersession verdict"
        );
    }

    /// The same INSTANT written in two offsets must share a rank.
    ///
    /// Otherwise a purely cosmetic timezone change would move the fingerprint,
    /// reintroducing the bug in a narrower form.
    #[test]
    fn equal_instants_in_different_offsets_share_a_rank() {
        let utc = "at 2023-01-01T00:00:00+00:00 and 2023-06-01T00:00:00+00:00";
        let offset = "at 2023-01-01T01:00:00+01:00 and 2023-06-01T00:00:00+00:00";
        assert_eq!(
            canonicalize_timestamps_for_fingerprint(utc),
            canonicalize_timestamps_for_fingerprint(offset),
        );
    }

    /// BARE dates must be left ALONE — the blast-radius guard.
    ///
    /// Episode prose routinely contains `2023-05-07`. Rewriting those would
    /// change the hashed content of corpora that replay correctly today and
    /// re-fingerprint the 303 committed cassettes — turning a five-cassette
    /// problem into a three-hundred-cassette one, which is exactly why TD-187
    /// §F1 gated its own prompt change.
    #[test]
    fn bare_dates_in_prose_are_untouched() {
        let prose = "We met on 2023-05-07 and again on 1999-12-31.";
        assert_eq!(
            canonicalize_timestamps_for_fingerprint(prose),
            prose,
            "bare YYYY-MM-DD is corpus content, not a rendered wall-clock"
        );
    }

    /// A request with no timestamps at all must hash exactly as before.
    ///
    /// This is what keeps every existing extraction cassette valid: TD-187 gates
    /// its date block on `reference_time.is_some()`, which the facade never
    /// sets, so those prompts carry no RFC3339 and take the early return.
    #[test]
    fn timestamp_free_content_is_returned_verbatim() {
        let plain = "Extract entities from: Alice works at Acme Corp.";
        assert_eq!(canonicalize_timestamps_for_fingerprint(plain), plain);
    }

    /// Non-vacuity guard for the test above: prove the assertion can FAIL.
    ///
    /// A miss error that quoted EVERY request unconditionally would satisfy the
    /// test above while telling you nothing. This pins the other direction — a
    /// request that does NOT contain the sentinel must not report it — so the
    /// first test is measuring the request's content rather than a constant.
    #[tokio::test]
    async fn cassette_miss_error_quotes_this_request_not_a_canned_string() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("empty-cassette.json");
        let cassette = Cassette {
            version: 1,
            model: "test-model".to_string(),
            recorded_at: "1970-01-01T00:00:00Z".to_string(),
            entries: std::collections::BTreeMap::new(),
        };
        std::fs::write(&path, serde_json::to_vec(&cassette).expect("serialize"))
            .expect("write cassette");

        let provider = RecordReplayChatProvider::replay(&path).expect("replay mode");

        let err = provider
            .chat_with_tools(
                &[chat_msg_user("an entirely different request")],
                None,
                None,
            )
            .await
            .expect_err("an empty cassette must MISS");

        let msg = err.to_string();
        assert!(
            !msg.contains("PINEAPPLE-ON-PIZZA-SENTINEL"),
            "the error must quote THIS request, not a canned string; got: {msg}"
        );
        assert!(
            msg.contains("an entirely different request"),
            "the error must quote THIS request; got: {msg}"
        );
    }
}

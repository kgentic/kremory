//! Ingest helpers — context snippet extraction + SimpleGraph type alias.
//!
//! Split from `ingest.rs`.

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

#[cfg(any(test, feature = "test-utils"))]
use std::sync::Arc;

#[cfg(any(test, feature = "test-utils"))]
use crate::core::provider::{MockChatProvider, NullEmbeddingProvider};
#[cfg(any(test, feature = "test-utils"))]
use crate::core::schema::TemporalGraph;

#[cfg(any(test, feature = "test-utils"))]
use super::{Engine, EngineNewParams};

#[cfg(any(test, feature = "test-utils"))]
use crate::core::config::PipelineConfig;
#[cfg(any(test, feature = "test-utils"))]
use crate::core::error::Result;

// ─── Context snippet helper ───────────────────────────────────────────────────

/// Extract a verbatim context snippet for an entity's first mention.
///
/// Searches for `name` (case-insensitive) in `text` and returns a ±`half_window`
/// character window around the first match. If the name is not found in `text`
/// (e.g. LLM hallucinated the entity), returns the first `half_window * 2` chars
/// of `text` as a fallback so the property is never empty.
pub(super) fn extract_context_snippet(text: &str, name: &str, half_window: usize) -> String {
    /// Walk a byte offset forward to the nearest valid UTF-8 char boundary (ceiling).
    fn ceil_char_boundary(s: &str, byte_pos: usize) -> usize {
        let mut pos = byte_pos.min(s.len());
        while pos < s.len() && !s.is_char_boundary(pos) {
            pos += 1;
        }
        pos
    }
    /// Walk a byte offset backward to the nearest valid UTF-8 char boundary (floor).
    fn floor_char_boundary(s: &str, byte_pos: usize) -> usize {
        let mut pos = byte_pos.min(s.len());
        while pos > 0 && !s.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    let lower_text = text.to_lowercase();
    let lower_name = name.to_lowercase();
    if let Some(pos) = lower_text.find(lower_name.as_str()) {
        let raw_start = pos.saturating_sub(half_window);
        let raw_end = (pos + name.len() + half_window).min(text.len());
        let start = floor_char_boundary(text, raw_start);
        let end = ceil_char_boundary(text, raw_end);
        text[start..end].to_owned()
    } else {
        // Name not found in text — use the opening portion as a fallback.
        let raw_end = half_window.saturating_mul(2).min(text.len());
        let end = ceil_char_boundary(text, raw_end);
        text[..end].to_owned()
    }
}

// ─── SimpleGraph ──────────────────────────────────────────────────────────────

/// Convenience type alias for tests and simple usage (no LLM/embedding).
///
/// Gated: test-infra only — not part of the production public API.
#[cfg(any(test, feature = "test-utils"))]
pub type SimpleGraph = Engine<MockChatProvider, NullEmbeddingProvider>;

#[cfg(any(test, feature = "test-utils"))]
impl SimpleGraph {
    /// Open an in-memory graph with null providers and default config.
    pub async fn open_in_memory_simple() -> Result<Self> {
        Self::open_in_memory_with_search_config(|_| {}).await
    }

    /// Like [`Self::open_in_memory_simple`] but lets a test tweak the
    /// [`crate::core::config::SearchConfig`] before construction — e.g. set a
    /// non-default `floor_threshold` / `temporal_weight` to exercise the
    /// recall-v2 scoring axes (which are behaviourally-neutral no-ops at their
    /// defaults). Reuses the exact same null-provider wiring.
    pub async fn open_in_memory_with_search_config(
        tweak: impl FnOnce(&mut crate::core::config::SearchConfig),
    ) -> Result<Self> {
        let graph = Arc::new(TemporalGraph::open_in_memory().await?);
        let mut config = PipelineConfig::builder().build()?;
        tweak(&mut config.search);
        Ok(Self::new(EngineNewParams {
            graph,
            llm: Arc::new(MockChatProvider::null()),
            embedder: Arc::new(NullEmbeddingProvider {
                dim: config.embedding_dim.0,
            }),
            config,
            // Null provider → no model.
            model: None,
        }))
    }
}

//! Extraction-prompt-window splitter (formerly `chunker`).
//!
//! **This is a kind-2 chunker only** — see `.ai-docs/research/proxy-pointer-rag-integration.md` §0
//! and ADR-Phase-D.0:88 for the kind-1 vs kind-2 distinction.
//!
//! Slices an episode's text into LLM-extraction-prompt-sized windows so the extractor LLM
//! can read oversized episode bodies within its context budget. The slices are **throwaway**:
//! never stored, never embedded, never retrieved. They exist only as local-loop variables
//! consumed by [`crate::core::ingest::Engine::ingest_with`] and dropped at function return.
//!
//! Kremory does NOT do kind-1 retrieval chunking (slicing for vector-index rows). Episodes
//! are stored whole and embeddings are per-entity-name + per-fact, never per-window-slice.
//! Caller-side retrieval chunking is App-layer concern per `feedback_rql_does_not_chunk_caller_owns`.
//!
//! The early-return at `min_words` (default 100) makes this splitter a no-op for normally-sized
//! episodes — it only fires when the input exceeds the configured extraction-prompt budget.

use crate::core::config::{ContentType, ExtractionWindowConfig};

pub(crate) struct ExtractionWindowSplitter {
    config: ExtractionWindowConfig,
}

impl ExtractionWindowSplitter {
    pub(crate) fn new(config: ExtractionWindowConfig) -> Self {
        Self { config }
    }

    /// Split text into extraction-prompt-sized windows based on content type and density.
    pub fn split(&self, text: &str, content_type: &ContentType) -> Vec<String> {
        let chunks = match content_type {
            ContentType::Text | ContentType::Document => self.split_by_density(text),
            ContentType::Message => self.split_by_lines(text),
            ContentType::Json => vec![text.to_string()],
        };
        self.apply_overlap(chunks)
    }

    fn apply_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if self.config.overlap_words == 0 || chunks.len() <= 1 {
            return chunks;
        }
        let mut result = Vec::with_capacity(chunks.len());
        result.push(chunks[0].clone());
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i - 1].split_whitespace().collect();
            let overlap_start = prev_words.len().saturating_sub(self.config.overlap_words);
            let overlap: String = prev_words[overlap_start..].join(" ");
            if overlap.is_empty() {
                result.push(chunks[i].clone());
            } else {
                result.push(format!("{overlap} {}", chunks[i]));
            }
        }
        result
    }

    fn split_by_density(&self, text: &str) -> Vec<String> {
        let token_count = text.split_whitespace().count();
        if token_count < self.config.min_words {
            return vec![text.to_string()];
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let window_size = (self.config.max_words / 2).max(1);
        let mut split_points: Vec<usize> = Vec::new();

        // Scan text with a sliding window, looking for high-density regions.
        let mut i = 0;
        while i + window_size < words.len() {
            let window = &words[i..i + window_size];
            let capitalized_count = window
                .iter()
                .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                .count();
            let density = capitalized_count as f64 / window_size as f64;

            if density > self.config.density_threshold {
                // Find a split point near the end of this window in the original text.
                let approx_char_pos = words[..i + window_size]
                    .iter()
                    .map(|w| w.len() + 1)
                    .sum::<usize>();
                let split_pos = find_sentence_boundary(text, approx_char_pos);
                if split_pos > 0 && split_pos < text.len() {
                    split_points.push(split_pos);
                }
                // Advance past this window to avoid overlapping splits.
                i += window_size;
            } else {
                i += (window_size / 4).max(1);
            }
        }

        // Build chunks from density-driven split points (if any).
        let density_chunks: Vec<String> = if split_points.is_empty() {
            vec![text.to_string()]
        } else {
            // Deduplicate and sort split points.
            split_points.sort_unstable();
            split_points.dedup();

            let mut chunks: Vec<String> = Vec::new();
            let mut prev = 0;
            for &pos in &split_points {
                let chunk = text[prev..pos].trim().to_string();
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }
                prev = pos;
            }
            let remainder = text[prev..].trim().to_string();
            if !remainder.is_empty() {
                chunks.push(remainder);
            }
            chunks
        };

        // Always enforce max_words per chunk by word-boundary splitting.
        let mut final_chunks: Vec<String> = Vec::new();
        for chunk in density_chunks {
            final_chunks.extend(enforce_max_words(&chunk, self.config.max_words));
        }

        if final_chunks.is_empty() {
            vec![text.to_string()]
        } else {
            final_chunks
        }
    }

    fn split_by_lines(&self, text: &str) -> Vec<String> {
        // Split on double-newlines (paragraph boundaries).
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if paragraphs.is_empty() {
            return vec![text.to_string()];
        }

        // Merge consecutive short paragraphs until reaching max_words.
        let mut merged: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut current_tokens = 0;

        for para in paragraphs {
            let para_tokens = para.split_whitespace().count();
            if current_tokens + para_tokens > self.config.max_words && !current.is_empty() {
                merged.push(current.trim().to_string());
                current = para.to_string();
                current_tokens = para_tokens;
            } else {
                if !current.is_empty() {
                    current.push('\n');
                    current.push('\n');
                }
                current.push_str(para);
                current_tokens += para_tokens;
            }
        }
        if !current.trim().is_empty() {
            merged.push(current.trim().to_string());
        }

        // Enforce max_words per merged chunk by word-boundary splitting.
        let mut chunks: Vec<String> = Vec::new();
        for chunk in merged {
            chunks.extend(enforce_max_words(&chunk, self.config.max_words));
        }

        if chunks.is_empty() {
            vec![text.to_string()]
        } else {
            chunks
        }
    }
}

// UTF-8 boundary snapping lives in `core::text_utils` — ONE implementation
// for the whole crate. Four private copies of this idea is exactly how the
// 2026-07-28 crash happened: `core/ingest/helpers.rs` already had a correct
// floor/ceil pair and this module never inherited it.
use crate::core::text_utils::floor_char_boundary;

/// Find the nearest sentence boundary (period, exclamation, question mark) at or before
/// `approx_pos`, searching up to 200 bytes backwards.
///
/// Both ends of the search window are snapped to character boundaries first —
/// see `floor_char_boundary` for why that is mandatory rather than defensive.
fn find_sentence_boundary(text: &str, approx_pos: usize) -> usize {
    let pos = floor_char_boundary(text, approx_pos);
    let search_start = floor_char_boundary(text, pos.saturating_sub(200));
    let slice = &text[search_start..pos];

    // Walk backwards looking for sentence-ending punctuation.
    let boundary_rel = slice
        .char_indices()
        .rev()
        .find(|(_, c)| *c == '.' || *c == '!' || *c == '?')
        .map(|(i, c)| i + c.len_utf8());

    match boundary_rel {
        Some(rel) => search_start + rel,
        None => pos,
    }
}

/// Split a single chunk at word boundaries so no piece exceeds max_words words.
fn enforce_max_words(text: &str, max_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + max_words).min(words.len());
        chunks.push(words[i..end].join(" "));
        i = end;
    }
    chunks
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::ExtractionWindowConfig;

    fn config_with(
        min_words: usize,
        max_words: usize,
        density_threshold: f64,
    ) -> ExtractionWindowConfig {
        ExtractionWindowConfig {
            min_words,
            max_words,
            density_threshold,
            overlap_words: 0,
        }
    }

    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` builders.
    #[allow(clippy::too_many_arguments)]
    fn config_with_overlap(
        min: usize,
        max: usize,
        density: f64,
        overlap: usize,
    ) -> ExtractionWindowConfig {
        ExtractionWindowConfig {
            min_words: min,
            max_words: max,
            density_threshold: density,
            overlap_words: overlap,
        }
    }

    #[test]
    fn test_short_text_single_chunk() {
        let cfg = config_with(500, 1000, 0.15);
        let splitter = ExtractionWindowSplitter::new(cfg);
        let text = "Alice works at Acme Corp.";
        let chunks = splitter.split(text, &ContentType::Text);
        assert_eq!(
            chunks.len(),
            1,
            "short text under min_tokens should produce one chunk"
        );
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_long_text_splits() {
        // max_tokens=50 so 200-token text must be split.
        let cfg = config_with(10, 50, 0.10);
        let splitter = ExtractionWindowSplitter::new(cfg);
        // Build text of ~200 tokens, dense with capitalized words.
        let sentence =
            "Alice Bob Carol David Emma Frank George Hannah Ivan Julia worked at Acme Corp. ";
        let text = sentence.repeat(10);
        let chunks = splitter.split(&text, &ContentType::Text);
        assert!(
            chunks.len() > 1,
            "long high-density text should split into multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_json_passthrough() {
        let cfg = config_with(500, 1000, 0.15);
        let splitter = ExtractionWindowSplitter::new(cfg);
        let json = r#"{"name": "Alice", "label": "Person"}"#;
        let chunks = splitter.split(json, &ContentType::Json);
        assert_eq!(chunks.len(), 1, "JSON should always return a single chunk");
        assert_eq!(chunks[0], json);
    }

    #[test]
    fn test_message_splits_on_paragraphs() {
        // max_tokens=6 so each paragraph (8+ tokens) cannot merge with the other.
        let cfg = config_with(1, 6, 0.15);
        let splitter = ExtractionWindowSplitter::new(cfg);
        let para1 = "Alice works at Acme Corp last year.";
        let para2 = "Bob manages the engineering team there.";
        let text = format!("{}\n\n{}", para1, para2);
        let chunks = splitter.split(&text, &ContentType::Message);
        assert!(
            chunks.len() >= 2,
            "paragraphs exceeding max_tokens should not merge; got {} chunks",
            chunks.len()
        );
        // Verify content is preserved.
        let all_text = chunks.join(" ");
        assert!(all_text.contains("Alice"), "Alice must appear in output");
        assert!(all_text.contains("Bob"), "Bob must appear in output");
    }

    #[test]
    fn test_message_merges_short_paragraphs() {
        let cfg = config_with(1, 200, 0.15);
        let splitter = ExtractionWindowSplitter::new(cfg);
        // Two very short paragraphs that together fit within max_tokens.
        let text = "Hello world.\n\nGoodbye world.";
        let chunks = splitter.split(text, &ContentType::Message);
        assert_eq!(
            chunks.len(),
            1,
            "short paragraphs within max_tokens should merge"
        );
    }

    #[test]
    fn test_chunks_respect_max_tokens() {
        let max = 20;
        let cfg = config_with(5, max, 0.05);
        let splitter = ExtractionWindowSplitter::new(cfg);
        // Generate text well above min and max tokens (no capitals → density won't trigger,
        // but enforce_max_tokens must still split).
        let words: Vec<String> = (0..300).map(|i| format!("word{}", i)).collect();
        let text = words.join(" ");
        let chunks = splitter.split(&text, &ContentType::Text);
        for (i, chunk) in chunks.iter().enumerate() {
            let token_count = chunk.split_whitespace().count();
            assert!(
                token_count <= max,
                "chunk {} has {} tokens which exceeds max_tokens={}",
                i,
                token_count,
                max
            );
        }
    }

    #[test]
    fn test_message_respects_max_tokens_per_chunk() {
        let max = 10;
        let cfg = config_with(1, max, 0.15);
        let splitter = ExtractionWindowSplitter::new(cfg);
        // Build a single paragraph that individually exceeds max_tokens.
        let big_para: String = (0..30).map(|i| format!("word{} ", i)).collect();
        let text = big_para.trim().to_string();
        let chunks = splitter.split(&text, &ContentType::Message);
        for (i, chunk) in chunks.iter().enumerate() {
            let count = chunk.split_whitespace().count();
            assert!(
                count <= max,
                "message chunk {} has {} tokens exceeding max={}",
                i,
                count,
                max
            );
        }
    }

    #[test]
    fn test_overlap_between_chunks() {
        // max_tokens=100, overlap=50, min_tokens=1 so a 250-word text splits.
        let cfg = config_with_overlap(1, 100, 0.05, 50);
        let splitter = ExtractionWindowSplitter::new(cfg);
        // Generate 250 unique lowercase words so density never triggers a split
        // before enforce_max_tokens does.
        let words: Vec<String> = (0..250).map(|i| format!("word{:03}", i)).collect();
        let text = words.join(" ");
        let chunks = splitter.split(&text, &ContentType::Text);
        assert!(
            chunks.len() >= 2,
            "250-word text with max_tokens=100 should produce at least 2 chunks, got {}",
            chunks.len()
        );
        // The last 50 words of chunk[0] should appear at the start of chunk[1].
        let chunk0_words: Vec<&str> = chunks[0].split_whitespace().collect();
        let overlap_start = chunk0_words.len().saturating_sub(50);
        let expected_prefix: String = chunk0_words[overlap_start..].join(" ");
        assert!(
            chunks[1].starts_with(&expected_prefix),
            "chunk[1] should start with the last 50 words of chunk[0]"
        );
    }

    #[test]
    fn test_zero_overlap_is_noop() {
        let cfg_no_overlap = config_with(1, 100, 0.05);
        let cfg_with_overlap_val = config_with_overlap(1, 100, 0.05, 0);
        let splitter_a = ExtractionWindowSplitter::new(cfg_no_overlap);
        let splitter_b = ExtractionWindowSplitter::new(cfg_with_overlap_val);
        let words: Vec<String> = (0..250).map(|i| format!("word{:03}", i)).collect();
        let text = words.join(" ");
        let chunks_a = splitter_a.split(&text, &ContentType::Text);
        let chunks_b = splitter_b.split(&text, &ContentType::Text);
        assert_eq!(
            chunks_a, chunks_b,
            "overlap_tokens=0 should produce identical output to a config with no overlap"
        );
    }

    #[test]
    fn test_overlap_single_chunk_noop() {
        // Short text that fits in one chunk — overlap must not be applied.
        let cfg = config_with_overlap(500, 1000, 0.15, 50);
        let splitter = ExtractionWindowSplitter::new(cfg);
        let text = "Alice works at Acme Corp.";
        let chunks = splitter.split(text, &ContentType::Text);
        assert_eq!(chunks.len(), 1, "single-chunk text should stay one chunk");
        assert_eq!(chunks[0], text, "single chunk content must be unchanged");
    }
}

#[cfg(test)]
mod utf8_boundary_tests {
    use super::*;
    use crate::core::config::ExtractionWindowConfig;

    /// Reproduces the panic that took kremory-http down mid-run against
    /// LongMemEval on 2026-07-28:
    ///   `byte index 6809 is not a char boundary; it is inside 'è'`
    ///
    /// The split offsets are reconstructed by BYTE arithmetic (a sum of
    /// `str::len()` plus one assumed separator byte per word, then a flat
    /// `- 200` for the backward search window), so on multi-byte text either
    /// end of `&text[search_start..pos]` can land mid-character and panic.
    /// LoCoMo never hit it because that corpus is effectively ASCII.
    ///
    /// Built to trip the density heuristic (many Capitalised words) with
    /// accented characters dense enough that a byte-derived offset is
    /// overwhelmingly unlikely to land on a boundary by luck.
    #[test]
    fn chunking_does_not_panic_on_multibyte_text() {
        let sentence = "Català València Perpinyà Andorra Eivissa Menorca Mallorca \
                        Lleida Girona Tarragona Sabadell Terrassa Badalona Mataró. ";
        let text = sentence.repeat(80);
        assert!(
            text.len() > 6809,
            "fixture must exceed the observed panic offset"
        );
        assert!(
            !text.is_char_boundary(text.len() / 2 + 1) || text.chars().any(|c| c.len_utf8() > 1),
            "fixture must actually contain multi-byte characters"
        );

        for (min, max, density) in [(10, 60, 0.3), (5, 40, 0.2), (20, 120, 0.5)] {
            let splitter = ExtractionWindowSplitter::new(ExtractionWindowConfig {
                min_words: min,
                max_words: max,
                density_threshold: density,
                overlap_words: 0,
            });
            // Pre-fix this panics rather than returning.
            let chunks: Vec<String> = splitter.split(&text, &ContentType::Text);
            assert!(!chunks.is_empty());
            // No chunk may be empty or have sliced a character in half — a
            // successful round-trip through String proves every boundary held.
            for c in &chunks {
                assert!(!c.is_empty());
                assert!(c.is_char_boundary(0));
            }
        }
    }

    /// `floor_char_boundary` must never return an index that would panic, and
    /// must never round UP (callers rely on `split_pos < text.len()`).
    #[test]
    fn floor_char_boundary_is_safe_and_monotone() {
        let text = "aè b€ c𝄞 d";
        for i in 0..=text.len() + 5 {
            let f = floor_char_boundary(text, i);
            assert!(
                text.is_char_boundary(f),
                "returned non-boundary {f} for {i}"
            );
            assert!(f <= i.min(text.len()), "rounded UP: {f} > {i}");
            let _ = &text[..f]; // would panic if f were not a boundary
        }
    }

    /// ASCII behaviour must be byte-identical to before the fix — on ASCII,
    /// every byte index is already a character boundary, so the snap is a no-op.
    #[test]
    fn floor_char_boundary_is_identity_on_ascii() {
        let text = "the quick brown fox. jumps over the lazy dog.";
        for i in 0..=text.len() {
            assert_eq!(floor_char_boundary(text, i), i);
        }
    }
}

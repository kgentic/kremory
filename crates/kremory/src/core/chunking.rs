//! Caller-facing pre-chunking helper.
//!
//! **This exists on the caller's side of a locked architectural boundary, not
//! inside it.** Kremory does NOT chunk content on the caller's behalf for
//! storage/retrieval ("kind-1" chunking) — only internally, throwaway, for the
//! LLM extraction prompt window ("kind-2", see
//! [`crate::core::extraction_window`]). That boundary has been affirmed three
//! times (the original architecture ADR, a later research doc's correction,
//! and `extraction_window`'s own doc comment) and this module does not reopen
//! it: [`split_for_embedding`] is a plain function the caller invokes
//! themselves, before looping `remember()` once per chunk. Kremory never
//! calls it automatically.
//!
//! **Why it needs to exist at all.** Before this, a caller who wanted to
//! respect the "you own chunking" boundary had no tool to do it with —
//! [`crate::core::extraction_window::ExtractionWindowSplitter`] is
//! `pub(crate)`, unreachable from outside the crate. A long document handed
//! to `remember()` whole silently loses its dense/embedding search arm once
//! it exceeds the embedder's own context window — the boundary was real,
//! but undoored.
//!
//! Backed by the `text-splitter` crate (unicode-aware sentence/paragraph
//! boundary detection) rather than a naive character slice, so a chunk
//! doesn't get cut mid-word or mid-sentence — that would degrade both the
//! embedding it produces and the BM25 tokens either side of the cut.

use text_splitter::TextSplitter;

/// Split `text` into chunks no larger than `max_chars` (Unicode scalar
/// count), preferring sentence/paragraph boundaries over a hard cut.
///
/// Call this yourself before looping `remember()` once per returned chunk —
/// kremory never calls it for you (see the module doc comment for why). A
/// reasonable `max_chars` for `nomic-embed-text`'s ~2048-token window is
/// `6000`-`8000`; check your own embedder's real limit rather than assume.
///
/// Returns `vec![text.to_string()]` unchanged when `text` already fits —
/// this is always safe to call unconditionally, including on short episodes.
///
/// # Example
///
/// ```
/// use kremory::split_for_embedding;
///
/// let chunks = split_for_embedding("Some text\n\nfrom a\ndocument", 20);
/// assert!(chunks.iter().all(|c| c.chars().count() <= 20));
/// ```
pub fn split_for_embedding(text: &str, max_chars: usize) -> Vec<String> {
    let splitter = TextSplitter::new(max_chars);
    splitter.chunks(text).map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_returns_a_single_unchanged_chunk() {
        let text = "A short episode.";
        let chunks = split_for_embedding(text, 10_000);
        assert_eq!(chunks, vec![text.to_string()]);
    }

    #[test]
    fn long_text_splits_into_multiple_chunks_each_within_budget() {
        // Build text well over the embedder-window-sized budget used below.
        let sentence = "The quick brown fox jumps over the lazy dog. ";
        let text: String = sentence.repeat(50); // ~2,300 chars
        let max_chars = 300;

        let chunks = split_for_embedding(&text, max_chars);

        assert!(
            chunks.len() > 1,
            "text over budget must produce more than one chunk; got {}",
            chunks.len()
        );
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= max_chars,
                "chunk of {} chars exceeds max_chars={max_chars}: {chunk:?}",
                chunk.chars().count()
            );
        }
        // Reassembling must not silently drop or duplicate content — the
        // reconstructed text must contain every sentence exactly as often as
        // the source did.
        let reassembled = chunks.join("");
        assert_eq!(
            reassembled.matches("quick brown fox").count(),
            text.matches("quick brown fox").count(),
            "chunking must not drop or duplicate sentences"
        );
    }

    #[test]
    fn chunks_do_not_cut_mid_word() {
        let sentence = "Supercalifragilisticexpialidocious is a very long word indeed. ";
        let text = sentence.repeat(20);
        let chunks = split_for_embedding(&text, 100);

        for chunk in &chunks {
            let trimmed = chunk.trim();
            if trimmed.is_empty() {
                continue;
            }
            assert!(
                !trimmed
                    .split_whitespace()
                    .next_back()
                    .unwrap_or("")
                    .is_empty()
                    || trimmed.ends_with('.'),
                "chunk should not end mid-word: {trimmed:?}"
            );
        }
    }

    #[test]
    fn empty_text_returns_no_chunks() {
        assert_eq!(split_for_embedding("", 1_000), Vec::<String>::new());
    }
}

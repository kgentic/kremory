//! Nomic-embed-text asymmetric task-prefix helpers (TD-143,
//! `.ai-docs/tech-debt/tech-debt-register.md` §TD-143).
//!
//! `nomic-embed-text` (the default kremory embedder, `facade/providers.rs`)
//! is an ASYMMETRIC embedding model: it requires a task prefix prepended to
//! the raw text before embedding — `search_document: ` for text that gets
//! STORED and later searched against, `search_query: ` for text that is a
//! QUERY compared (via vector search) against that stored corpus. Embedding
//! both sides with no prefix (kremory's pre-TD-143 behaviour) collapses the
//! query↔document asymmetry the model was trained on.
//!
//! Gated behind [`crate::core::config::SearchConfig::embed_task_prefix_enabled`]
//! (default `false` — every call below is then a no-op passthrough,
//! byte-identical to pre-TD-143 behaviour). This module intentionally stays
//! OUTSIDE the `EmbeddingProvider` / `DynEmbeddingProvider` trait contract —
//! per TD-143's DoD, the prefix is nomic-specific, so it is applied by CALL
//! SITES via these two helpers rather than baked into the (provider-agnostic,
//! BYOM) embedder trait. A consumer using a different, non-nomic embedder
//! never flips the knob and is completely unaffected.
//!
//! # Correctness — prefix pairing must be consistent within one vector index
//!
//! A query embedded WITH the prefix must only ever be compared against
//! documents embedded WITH the prefix — mixing prefixed and unprefixed
//! vectors in the SAME `entities` / `episodes` / `facts` vector index is a
//! silent correctness bug (the vectors then live in different task spaces,
//! so cosine similarity between them is meaningless, not merely degraded).
//! Every call site that embeds text destined for, or compared against, a
//! persisted `entities.embedding` / `episodes.embedding` / `facts.embedding`
//! column MUST route through [`document_embed_text`] (write side) or
//! [`query_embed_text`] (read side) — see the TD-143 tech-debt register
//! entry for the full call-site inventory this module's introduction
//! required (13 call sites across ingest, recall, and entity resolution).
//!
//! Flipping this knob on an EXISTING corpus makes every previously-stored
//! embedding stale (a document-prefixed write and an unprefixed write occupy
//! different task spaces) — re-embed a FRESH corpus copy first (episodes via
//! `Memory::reembed_all_episode_embeddings`, free + local against an Ollama
//! embedder — NOT `backfill_episode_embeddings`, whose `WHERE embedding IS
//! NULL` paging only fills a gap and can never overwrite an already-embedded
//! row; entities/facts via a full re-ingest) before measuring.

use std::borrow::Cow;

/// Prefix `text` with nomic's `search_document: ` task prefix when `enabled`.
///
/// No-op (borrows `text` unchanged) when `enabled` is `false` — byte-identical
/// to pre-TD-143 behaviour. Apply at every call site that embeds text
/// destined to be STORED into a persisted `entities` / `episodes` / `facts`
/// embedding column (ingest-time writes, the backfill subcommand, and the
/// dream-phase merge re-embed).
pub(crate) fn document_embed_text(text: &str, enabled: bool) -> Cow<'_, str> {
    if enabled {
        Cow::Owned(format!("search_document: {text}"))
    } else {
        Cow::Borrowed(text)
    }
}

/// Prefix `text` with nomic's `search_query: ` task prefix when `enabled`.
///
/// No-op (borrows `text` unchanged) when `enabled` is `false` — byte-identical
/// to pre-TD-143 behaviour. Apply at every call site that embeds text to be
/// COMPARED, via vector search, against a persisted, document-prefixed
/// `entities` / `episodes` / `facts` embedding column — this includes the
/// recall query path AND entity-resolution probes (L4 disambiguation,
/// ADR-075 candidate blocking) that search the same `entities` index.
pub(crate) fn query_embed_text(text: &str, enabled: bool) -> Cow<'_, str> {
    if enabled {
        Cow::Owned(format!("search_query: {text}"))
    } else {
        Cow::Borrowed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default-off byte-identical passthrough (TD-143 DoD item 1) ──────────

    #[test]
    fn document_embed_text_disabled_is_passthrough() {
        assert_eq!(document_embed_text("hello world", false), "hello world");
    }

    #[test]
    fn query_embed_text_disabled_is_passthrough() {
        assert_eq!(query_embed_text("hello world", false), "hello world");
    }

    // ── Knob-on applies the correct prefix on each side ──────────────────────

    #[test]
    fn document_embed_text_enabled_applies_search_document_prefix() {
        assert_eq!(
            document_embed_text("Boston", true),
            "search_document: Boston"
        );
    }

    #[test]
    fn query_embed_text_enabled_applies_search_query_prefix() {
        assert_eq!(query_embed_text("Boston", true), "search_query: Boston");
    }

    // ── Asymmetry: query and document sides must get DIFFERENT prefixes ─────

    #[test]
    fn document_and_query_prefixes_are_asymmetric_for_the_same_input() {
        let doc = document_embed_text("same text", true);
        let query = query_embed_text("same text", true);
        assert_ne!(
            doc, query,
            "query and document embeds of identical raw text must diverge once \
             prefixed — a shared prefix would defeat the asymmetric task space \
             nomic-embed-text requires"
        );
        assert!(doc.starts_with("search_document: "));
        assert!(query.starts_with("search_query: "));
    }

    #[test]
    fn empty_text_is_still_prefixed_when_enabled() {
        assert_eq!(document_embed_text("", true), "search_document: ");
        assert_eq!(query_embed_text("", true), "search_query: ");
    }
}

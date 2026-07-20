//! Query-intent classification for recall scoring (TD-066 #77, recall-v2 spec
//! `Decision 1`). A zero-LLM regex/keyword heuristic that labels a recall query
//! as [`Intent::Factual`], [`Intent::Relational`], or [`Intent::Broad`] so the
//! recall scoring pipeline can pick per-intent default weights for its post-RRF
//! boost axes (graph-degree / temporal / truth-boost / axis-C proximity).
//!
//! # HALF-FEATURE (2026-07-20) — classified + observed, not yet scored
//! [`classify_intent`] IS wired into `core::context::Engine::contextualize`
//! (runs before FTS/vector search per spec `Decision 1`), but PHASE 1 only
//! CLASSIFIES + EMITS OBSERVABILITY (`kremory.recall.intent_total` counter +
//! trace) — it does **not yet alter scoring**. TD-066 phase 2 consumes
//! [`Intent`] to select per-axis boost-weight defaults. The unit tests below
//! pin the classification behaviour so phase-2 wiring inherits a known-correct
//! signal rather than a silently-drifting one (per
//! `feedback_half_implemented_optimizations_ship_dormant`). Emitting intent now
//! also feeds the pattern-tuning the spec (`Decision 5`) defers to build-time
//! eval feedback. Reversibility (spec `Decision 1`): delete the call site and
//! every consumer falls back to [`Intent::Broad`] — no schema/data coupling.
//!
//! # Patterns are v1 defaults, not structural
//! Per `per-cell-policy-is-product-ux-space` + spec `Decision 5`: the EXACT
//! keyword patterns are product-UX-tunable v1 defaults — only the MECHANISM
//! (the [`Intent`] enum + downstream weight-override lookup) is structural. A
//! misclassification only shifts default weights within recall; it never gates
//! correctness (all axes still run for every intent).

/// The recall query's intent, used downstream to select per-axis boost-weight
/// defaults in the recall scoring pipeline.
///
/// [`Intent::Broad`] is the safe default: an unrecognised or ambiguous query
/// classifies as `Broad`, which runs every scoring axis with neutral weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Intent {
    /// Single-fact lookup — "what is X", "who is X", "when did X". Downstream
    /// this favours precise, high-confidence direct hits.
    Factual,
    /// Relation-seeking — "how does X relate to Y", "connection between X and
    /// Y". Downstream this favours graph-proximity / multi-hop signal.
    Relational,
    /// Standing / open recall — "what do we know about X", "tell me about X",
    /// and everything not matched above. Neutral defaults.
    Broad,
}

impl Intent {
    /// Stable lowercase label for metrics / trace dimensions.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Intent::Factual => "factual",
            Intent::Relational => "relational",
            Intent::Broad => "broad",
        }
    }
}

/// Classify a recall `query` into an [`Intent`] via zero-LLM keyword heuristics.
///
/// Precedence (first match wins): **Relational** markers → **Broad** standing-
/// recall phrases → **Factual** interrogatives → default **Broad**.
///
/// - Relational is checked first because its markers ("connection between") are
///   the most specific and can co-occur with a "what is" lead.
/// - The Broad-phrase check precedes Factual on purpose so "what do we know
///   about X" is `Broad`, not captured by the "what" interrogative.
pub(crate) fn classify_intent(query: &str) -> Intent {
    let q = query.to_lowercase();

    // Relation-seeking markers — most specific, checked first.
    const RELATIONAL: &[&str] = &[
        "relationship between",
        "connection between",
        "related to",
        "relate to",
        "how does it relate",
        "how do they relate",
        "connected to",
        "link between",
        "links between",
        "association between",
        "how are they connected",
    ];
    if RELATIONAL.iter().any(|m| q.contains(m)) {
        return Intent::Relational;
    }

    // Standing / open-recall phrases — checked BEFORE Factual so a broad
    // "what do we know about X" is not captured by the "what" interrogative.
    const BROAD: &[&str] = &[
        "what do we know",
        "what do you know",
        "tell me about",
        "tell me everything",
        "everything about",
        "everything we know",
        "give me an overview",
        "overview of",
        "summarise",
        "summarize",
        "summary of",
    ];
    if BROAD.iter().any(|m| q.contains(m)) {
        return Intent::Broad;
    }

    // Single-fact interrogatives.
    const FACTUAL: &[&str] = &[
        "what is",
        "what are",
        "what was",
        "what were",
        "what's",
        "who is",
        "who are",
        "who was",
        "who were",
        "who's",
        "when is",
        "when was",
        "when did",
        "where is",
        "where was",
        "where did",
        "which ",
        "how many",
        "how much",
        "how old",
    ];
    if FACTUAL.iter().any(|m| q.contains(m)) {
        return Intent::Factual;
    }

    Intent::Broad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factual_interrogatives_classify_factual() {
        for q in [
            "What is Caroline's job?",
            "who is Melanie?",
            "When did the flood happen?",
            "How many kids does she have?",
            "where was the pottery class",
            "What's her favourite book?",
        ] {
            assert_eq!(classify_intent(q), Intent::Factual, "query: {q:?}");
        }
    }

    #[test]
    fn relation_seeking_classifies_relational() {
        for q in [
            "How does Caroline relate to Melanie?",
            "the relationship between Jon and Gina",
            "what is the connection between the flood and the move?",
            "who is she connected to",
        ] {
            assert_eq!(classify_intent(q), Intent::Relational, "query: {q:?}");
        }
    }

    #[test]
    fn standing_recall_and_unmatched_classify_broad() {
        for q in [
            "What do we know about Caroline?",
            "tell me about the road trip",
            "everything about Melanie's art",
            "Caroline pottery sweden", // no interrogative -> default Broad
            "",
        ] {
            assert_eq!(classify_intent(q), Intent::Broad, "query: {q:?}");
        }
    }

    #[test]
    fn relational_precedence_beats_factual_lead() {
        // "what is the connection between ..." leads with the Factual "what is"
        // but the Relational marker must win (checked first).
        assert_eq!(
            classify_intent("What is the connection between X and Y?"),
            Intent::Relational,
        );
    }

    #[test]
    fn broad_phrase_precedence_beats_factual_what() {
        // "what do we know" leads with "what" but must classify Broad, not
        // Factual — the Broad-phrase check precedes the Factual interrogatives.
        assert_eq!(classify_intent("what do we know about X"), Intent::Broad);
    }
}

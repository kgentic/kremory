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

    /// MEASUREMENT (2026-07-27, `.ai-docs/research/intent-classifier-routing-
    /// viability-2026-07-27.md`): does `classify_intent`'s Factual/Relational/
    /// Broad taxonomy correspond to LoCoMo's category taxonomy well enough to
    /// gate a retrieval arm on it? Reads the real dataset, calls the REAL
    /// `classify_intent` (not a reimplementation), cross-tabs category ×
    /// intent, and prints the 15 multi-hop questions the classifier does NOT
    /// label `Relational` (the failure shape a paraphrase-fragile keyword
    /// matcher produces). `#[ignore]`d so the default suite stays green — run
    /// explicitly:
    /// `cargo test -p kremory --lib core::intent::tests::locomo_intent_category_crosstab -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement-only: prints cross-tab, not a pass/fail gate"]
    fn locomo_intent_category_crosstab() {
        use std::collections::BTreeMap;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../bench/locomo/data/locomo10.json"
        );
        let raw = std::fs::read_to_string(path).expect("read locomo10.json");
        let data: serde_json::Value = serde_json::from_str(&raw).expect("parse locomo10.json");
        let convs = data.as_array().expect("top-level array of conversations");

        // Category ID -> name mapping, copied verbatim from
        // `bench/locomo/harness.py::CATEGORY_NAMES` (the harness is the
        // authority for this mapping; ids are NOT guessed here).
        const CATEGORY_NAMES: &[(i64, &str)] = &[
            (1, "single-hop"),
            (2, "temporal"),
            (3, "multi-hop"),
            (4, "open-domain"),
            (5, "adversarial"),
        ];
        let cat_name = |raw_cat: i64| -> &'static str {
            CATEGORY_NAMES
                .iter()
                .find(|(id, _)| *id == raw_cat)
                .map(|(_, n)| *n)
                .unwrap_or("unknown")
        };

        let mut crosstab: BTreeMap<&'static str, BTreeMap<&'static str, u32>> = BTreeMap::new();
        let mut total = 0u32;
        let mut multi_hop_not_relational: Vec<(String, &'static str)> = Vec::new();

        for conv in convs {
            let qa_list = conv
                .get("qa")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            for qa in qa_list {
                let question = qa.get("question").and_then(|v| v.as_str()).unwrap_or("");
                let raw_cat = qa.get("category").and_then(|v| v.as_i64()).unwrap_or(0);
                let cname = cat_name(raw_cat);
                let intent = classify_intent(question);

                *crosstab
                    .entry(cname)
                    .or_default()
                    .entry(intent.as_str())
                    .or_insert(0) += 1;
                total += 1;

                if cname == "multi-hop" && intent != Intent::Relational {
                    multi_hop_not_relational.push((question.to_string(), intent.as_str()));
                }
            }
        }

        println!("=== LoCoMo category x Intent cross-tab (n={total}) ===");
        for (cat, inner) in &crosstab {
            let row_total: u32 = inner.values().sum();
            print!("{cat:12} n={row_total:4}  ");
            for intent_name in ["factual", "relational", "broad"] {
                let c = inner.get(intent_name).copied().unwrap_or(0);
                let pct = 100.0 * f64::from(c) / f64::from(row_total);
                print!("{intent_name}={c:4} ({pct:5.1}%)  ");
            }
            println!();
        }

        println!(
            "\n=== sample multi-hop questions NOT classified Relational ({} of {}) ===",
            multi_hop_not_relational.len(),
            crosstab
                .get("multi-hop")
                .map(|m| m.values().sum::<u32>())
                .unwrap_or(0)
        );
        for (q, got) in multi_hop_not_relational.iter().take(15) {
            println!("  [{got:10}] {q}");
        }

        assert_eq!(total, 1986, "expected 1986 LoCoMo QA pairs (all 10 convs)");
    }
}

//! Which names in the extracted facts have no entity behind them (TD-045 split out
//! of `ingest_with.rs`).
//!
//! A model routinely writes a fact about something it did not list as an entity —
//! `("Ines", "works_at", "Lysfjord")` where only Ines was extracted. Lysfjord is a
//! FORWARD REFERENCE: the fact needs a row to point at, or its `subject_id` /
//! `object_id` dangles.
//!
//! The decision of WHICH names need a stub is pure set arithmetic, and it used to
//! be interleaved with the INSERT that acts on it — so the rule could only be
//! checked by running an ingest against a database and inspecting what came out.
//! It is separated here so it can be tested by calling it.
//!
//! Ordering is first-seen, never a set iteration order: the stubs are inserted in
//! this order and the counters are emitted per insert, so a `HashSet`'s arbitrary
//! order would make the resulting metrics and logs non-reproducible run to run.

use std::collections::HashSet;

use crate::core::intelligence::{ExtractedEntity, ExtractedFact};
use crate::core::resolver::normalize_name;

/// Normalized names referenced by `facts` that `entities` does not account for,
/// first-seen order, deduplicated.
///
/// Only the SUBJECT is always a name; the object counts only when the fact says it
/// is an entity reference (`is_entity_ref`) — otherwise it is a literal value like
/// "42" or "blue", which must never become an entity.
pub(super) fn forward_reference_names(
    entities: &[ExtractedEntity],
    facts: &[ExtractedFact],
) -> Vec<String> {
    let extracted: HashSet<String> = entities
        .iter()
        .map(|e| normalize_name(&e.name))
        .collect();

    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for fact in facts {
        let mut candidates = vec![normalize_name(&fact.subject)];
        if fact.is_entity_ref {
            candidates.push(normalize_name(&fact.object));
        }
        for name in candidates {
            if extracted.contains(&name) {
                // The entity loop owns it — it will be upserted properly there.
                continue;
            }
            if seen.insert(name.clone()) {
                out.push(name);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(name: &str) -> ExtractedEntity {
        ExtractedEntity {
            label: "Entity".to_string(),
            name: name.to_string(),
            properties: serde_json::Value::Null,
        }
    }

    fn fact(subject: &str, object: &str, is_entity_ref: bool) -> ExtractedFact {
        ExtractedFact {
            subject: subject.to_string(),
            predicate: "rel".to_string(),
            object: object.to_string(),
            is_entity_ref,
            confidence: 1.0,
            valid_at: None,
        }
    }

    #[test]
    fn a_name_the_extractor_listed_is_not_a_forward_reference() {
        let got = forward_reference_names(&[entity("Ines")], &[fact("Ines", "Lysfjord", false)]);
        assert_eq!(got, Vec::<String>::new());
    }

    #[test]
    fn an_unlisted_subject_is_a_forward_reference() {
        let got = forward_reference_names(&[], &[fact("Ines", "auditor", false)]);
        assert_eq!(got, vec![normalize_name("Ines")]);
    }

    #[test]
    fn a_literal_object_never_becomes_an_entity() {
        // is_entity_ref = false → "42" is a value, not a thing.
        let got = forward_reference_names(&[entity("Ines")], &[fact("Ines", "42", false)]);
        assert!(
            got.is_empty(),
            "a literal object must not be stubbed as an entity; got {got:?}"
        );
    }

    #[test]
    fn an_entity_ref_object_is_a_forward_reference() {
        let got = forward_reference_names(&[entity("Ines")], &[fact("Ines", "Lysfjord", true)]);
        assert_eq!(got, vec![normalize_name("Lysfjord")]);
    }

    #[test]
    fn the_same_name_is_returned_once_however_often_it_appears() {
        let got = forward_reference_names(
            &[],
            &[
                fact("Lysfjord", "x", false),
                fact("Lysfjord", "y", false),
                fact("Bergen", "Lysfjord", true),
            ],
        );
        assert_eq!(got, vec![normalize_name("Lysfjord"), normalize_name("Bergen")]);
    }

    #[test]
    fn order_is_first_seen_because_the_inserts_and_their_metrics_follow_it() {
        let got = forward_reference_names(
            &[],
            &[fact("Zulu", "x", false), fact("Alpha", "y", false)],
        );
        assert_eq!(
            got,
            vec![normalize_name("Zulu"), normalize_name("Alpha")],
            "first-seen, NOT sorted — set order would make the emitted counters \
             non-reproducible between runs"
        );
    }

    #[test]
    fn names_are_matched_after_normalization_not_verbatim() {
        // The extractor said "Ines"; the fact says "  ines  ". Same thing.
        let got = forward_reference_names(&[entity("Ines")], &[fact("  ines  ", "x", false)]);
        assert!(
            got.is_empty(),
            "normalization must be applied to BOTH sides; got {got:?}"
        );
    }
}

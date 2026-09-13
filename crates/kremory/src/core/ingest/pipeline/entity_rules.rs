//! The decisions the entity side of ingest makes, as functions (TD-045 split out
//! of `ingest_with.rs`).
//!
//! Sibling to `fact_rules.rs`, and here for the same measured reason: of the
//! modules previously lifted out of this pipeline, only the ones that extracted a
//! PURE RULE carry tests. A unit that still takes `&self` needs a live database
//! and a language model to exercise, so moving bulk into a method shortens the
//! file and adds no coverage.
//!
//! Neither rule below ever needed the database. Each was an inline expression
//! wrapped around an `await`, and the `await` is the only thing that made it
//! unreachable from a test:
//!
//! * [`top_n_display_names`] decides which already-known entities the extraction
//!   prompt shows the model, and **under which spelling**. Show the normalized id
//!   instead of the stored display name and the model answers in kind, inventing
//!   a variant ("alice_johnson" beside "Alice Johnson") — the precise duplication
//!   the injection exists to prevent.
//! * [`duplicate_extracted_names`] reports which names one extraction batch
//!   repeated. It is deliberately a REPORT rather than an error; the reasoning is
//!   at the call site and predates this split.
//!
//! Nothing here touches `self`, the graph or the network. The callers keep the I/O.

use std::collections::HashSet;

use crate::core::intelligence::ExtractedEntity;
use crate::core::resolver::normalize_name;
use crate::core::schema::Entity;

/// Pick the top-`limit` entities to inject into the extraction prompt, as
/// `(display_name, label)` pairs.
///
/// Ranked by `access_count` descending — most-used first — because the prompt has
/// room for a sample, not the namespace, and the entities a caller touches most
/// are the ones an extractor is most likely to mention again.
///
/// **The display name is the point.** `Entity::id` is normalized (lowercased,
/// underscored); `properties["name"]` holds the original casing. The model is
/// being shown these names so it will REUSE them verbatim, so the stored
/// display form wins wherever it exists, and the id is only a fallback for rows
/// that never carried one.
///
/// The sort is stable, so entities tied on `access_count` keep the order the
/// caller's query returned them in.
pub(super) fn top_n_display_names(
    mut entities: Vec<Entity>,
    limit: usize,
) -> Vec<(String, String)> {
    entities.sort_by(|a, b| b.access_count.cmp(&a.access_count));
    entities.truncate(limit);
    entities
        .into_iter()
        .map(|e| {
            let display_name = e
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned())
                .unwrap_or(e.id);
            (display_name, e.label)
        })
        .collect()
}

/// Report the normalized names a single extraction batch emitted more than once.
///
/// Returns one entry per REPEAT, not per distinct offender: a name appearing
/// three times yields two entries. That is what makes the caller's `dup_count`
/// a count of surplus rows rather than a count of colliding names, and the two
/// differ exactly when a noisy extractor loops.
///
/// Names are returned in their NORMALIZED form, because that is the form the
/// dedup downstream actually collides on — reporting the raw spellings would
/// name a different set than the one being removed.
pub(super) fn duplicate_extracted_names(entities: &[ExtractedEntity]) -> Vec<String> {
    let mut seen_this_call: HashSet<String> = HashSet::new();
    let mut dup_names: Vec<String> = Vec::new();
    for extracted in entities {
        let id = normalize_name(&extracted.name);
        if !seen_this_call.insert(id.clone()) {
            dup_names.push(id);
        }
    }
    dup_names
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    /// Minimal `Entity` — only `id`, `properties` and `access_count` carry
    /// meaning for these rules. `label` is fixed rather than passed: the one test
    /// that varies it uses struct-update syntax, which keeps this helper inside
    /// the 3-argument cap without an `#[allow]`.
    fn entity(id: &str, access_count: i64, name_prop: Option<&str>) -> Entity {
        Entity {
            id: id.to_owned(),
            label: "Person".to_owned(),
            entity_type_id: 0,
            properties: match name_prop {
                Some(n) => json!({ "name": n }),
                None => json!({}),
            },
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: None,
            access_count,
        }
    }

    fn extracted(name: &str) -> ExtractedEntity {
        ExtractedEntity {
            label: "Person".to_owned(),
            name: name.to_owned(),
            properties: json!({}),
        }
    }

    // ── top_n_display_names ─────────────────────────────────────────────────

    #[test]
    fn most_used_entities_come_first() {
        let got = top_n_display_names(
            vec![
                entity("alice", 1, None),
                entity("bob", 9, None),
                entity("carol", 5, None),
            ],
            10,
        );

        let order: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            order,
            vec!["bob", "carol", "alice"],
            "prompt injection is a sample, so it must be ranked by use"
        );
    }

    #[test]
    fn the_limit_truncates_after_ranking_not_before() {
        let got = top_n_display_names(
            vec![entity("seldom", 1, None), entity("often", 99, None)],
            1,
        );

        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].0, "often",
            "truncating before the sort would keep whichever row the query happened to return first"
        );
    }

    #[test]
    fn the_stored_display_name_beats_the_normalized_id() {
        let got = top_n_display_names(vec![entity("alice_johnson", 1, Some("Alice Johnson"))], 10);

        assert_eq!(
            got[0].0, "Alice Johnson",
            "the model is shown these names so it reuses them verbatim; showing the id teaches it the wrong spelling"
        );
    }

    #[test]
    fn a_missing_name_property_falls_back_to_the_id() {
        let got = top_n_display_names(vec![entity("alice_johnson", 1, None)], 10);

        assert_eq!(got[0].0, "alice_johnson");
    }

    #[test]
    fn a_non_string_name_property_falls_back_to_the_id() {
        let mut e = entity("alice_johnson", 1, None);
        e.properties = json!({ "name": 42 });

        let got = top_n_display_names(vec![e], 10);

        assert_eq!(
            got[0].0, "alice_johnson",
            "a numeric name is not a display name; it must not be stringified into the prompt"
        );
    }

    #[test]
    fn the_label_travels_with_the_name() {
        let acme = Entity {
            label: "Organization".to_owned(),
            ..entity("acme", 1, Some("ACME Corp"))
        };

        let got = top_n_display_names(vec![acme], 10);

        assert_eq!(got[0], ("ACME Corp".to_owned(), "Organization".to_owned()));
    }

    #[test]
    fn an_empty_namespace_injects_nothing() {
        assert!(top_n_display_names(vec![], 50).is_empty());
    }

    #[test]
    fn a_limit_above_the_population_returns_everything() {
        let got = top_n_display_names(vec![entity("alice", 1, None), entity("bob", 2, None)], 50);

        assert_eq!(got.len(), 2);
    }

    // ── duplicate_extracted_names ───────────────────────────────────────────

    #[test]
    fn a_clean_batch_reports_nothing() {
        let got = duplicate_extracted_names(&[extracted("Alice"), extracted("Bob")]);

        assert!(got.is_empty());
    }

    #[test]
    fn names_colliding_only_after_normalization_are_still_duplicates() {
        let got =
            duplicate_extracted_names(&[extracted("Alice Johnson"), extracted("alice johnson")]);

        assert_eq!(
            got.len(),
            1,
            "the dedup downstream collides on the normalized form, so the report must too"
        );
    }

    #[test]
    fn the_report_names_the_normalized_form_not_the_raw_spelling() {
        let got =
            duplicate_extracted_names(&[extracted("Alice Johnson"), extracted("Alice Johnson")]);

        assert_eq!(got, vec![normalize_name("Alice Johnson")]);
    }

    #[test]
    fn three_occurrences_report_two_repeats() {
        let got =
            duplicate_extracted_names(&[extracted("car"), extracted("car"), extracted("car")]);

        assert_eq!(
            got.len(),
            2,
            "the count is of surplus rows, not of colliding names — a looping extractor is the case this was softened for"
        );
    }

    #[test]
    fn the_first_occurrence_is_never_reported() {
        let got =
            duplicate_extracted_names(&[extracted("car"), extracted("bike"), extracted("car")]);

        assert_eq!(got, vec![normalize_name("car")]);
    }

    #[test]
    fn an_empty_batch_reports_nothing() {
        assert!(duplicate_extracted_names(&[]).is_empty());
    }
}

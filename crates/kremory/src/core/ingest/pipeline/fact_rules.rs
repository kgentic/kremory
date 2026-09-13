//! The decisions the Phase-2 fact loop makes, as functions (TD-045 split out of
//! `ingest_with.rs`).
//!
//! Same motivation as `forward_refs.rs`: each of these rules was an inline
//! expression inside a 510-line loop that also opens transactions, calls an LLM
//! contradiction detector and writes rows — so the only way to ask "does the
//! pool_b filter keep a cross-referencing fact?" was to run a full ingest against
//! a database with a live model and infer the answer from what landed. Four of
//! the six encode a measured, load-bearing decision (the pool_b cost filter, the
//! full-triple duplicate check, the outcome/source labelling that keeps the cost
//! filter falsifiable, and the refusal to emit a contradiction payload whose prior
//! fact cannot be evidenced), and none of them could be checked directly.
//!
//! Nothing here touches `self`, the graph or the network. The loop keeps the I/O.

use std::collections::HashSet;

use chrono::{DateTime, Utc};

use crate::core::error::ContradictionResolution;
use crate::core::intelligence::ExtractedFact;
use crate::core::resolver::normalize_name;
use crate::core::schema::Fact;
use crate::core::sink::{ContradictionDetected, EntityId, SinkFact};

/// Where a fact's subject and object point once the merge map has had its say.
///
/// `object_id` and `object_value` are mutually exclusive by construction: a fact
/// object is EITHER an entity reference or a literal ("42", "blue"), never both.
pub(super) struct FactEndpoints {
    /// Resolved subject entity id.
    pub(super) subject_id: String,
    /// Resolved object entity id — `Some` only when the fact is an entity ref.
    pub(super) object_id: Option<String>,
    /// Literal object text — `Some` only when the fact is NOT an entity ref.
    pub(super) object_value: Option<String>,
}

/// Resolve a fact's endpoints through the entity loop's merge map.
///
/// A name absent from `name_to_id` falls back to its own normalized form rather
/// than being dropped: the forward-reference pre-scan has already inserted a stub
/// under exactly that id, so the fallback lands on a real row.
pub(super) fn resolve_fact_endpoints(
    fact: &ExtractedFact,
    name_to_id: &std::collections::HashMap<String, String>,
) -> FactEndpoints {
    let resolve = |name: &str| {
        let norm = normalize_name(name);
        name_to_id.get(&norm).cloned().unwrap_or(norm)
    };

    FactEndpoints {
        subject_id: resolve(&fact.subject),
        object_id: fact.is_entity_ref.then(|| resolve(&fact.object)),
        object_value: (!fact.is_entity_ref).then(|| fact.object.clone()),
    }
}

/// Keep only the FTS hits that share an entity with the new fact.
///
/// pool_b is an FTS search on the PREDICATE ALONE, so unfiltered it returns other
/// subjects' facts entirely — "Alice likes tea" pulls in "Bob likes coffee" and
/// buys a full LLM round-trip to ask whether they contradict. They cannot.
///
/// Measured over 8 Groq sessions before this filter existed: pool_a contributed to
/// 85 LLM-reachable checks -> 33 contradictions + 4 duplicates; pool_b contributed
/// to 63 -> 0 and 0. That is a 95% CI upper bound of 4.8% on pool_b's hit rate
/// against ~43% for pool_a — pure cost.
///
/// This is a COST fix, not a capability removal. Same-subject contradiction is
/// pool_a's job and is untouched. What it drops is the CROSS-ENTITY case, which
/// was never designed: genuine cross-entity contradiction ("X is CEO of Acme" vs
/// "Y is CEO of Acme") needs predicate CARDINALITY, which kremory does not model.
/// Build that deliberately if wanted; do not leave it as an accident of a text
/// search.
///
/// A candidate whose OBJECT is our SUBJECT (or vice versa) is still about the same
/// entity, so it survives — see the `cross_ref` arm.
pub(super) fn pool_b_sharing_an_entity(
    hits: Vec<Fact>,
    subject_id: &str,
    object_id: Option<&str>,
) -> Vec<Fact> {
    hits.into_iter()
        .filter(|f| {
            let shares_subject = f.subject_id == subject_id;
            let shares_object = match (f.object_id.as_deref(), object_id) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            let cross_ref = f.object_id.as_deref() == Some(subject_id)
                || object_id == Some(f.subject_id.as_str());
            shares_subject || shares_object || cross_ref
        })
        .collect()
}

/// The `outcome` and `source` labels for `kremory.ingest.contradiction_outcome_total`.
///
/// Contradiction detection is 38.4% of all ingest LLM calls (361/941 measured over
/// 20 real sessions) and had no success metric of any kind, so the single largest
/// consumer of the ingest budget could not be shown to find anything.
///
/// `source` is the load-bearing label and is what keeps [`pool_b_sharing_an_entity`]
/// falsifiable: pool_a is scoped to (subject, predicate) while pool_b is a
/// predicate-only FTS hit, so splitting outcomes by which pool supplied the
/// candidates answers with data whether the predicate-only arm earns its cost. If
/// pool_b ever starts earning its keep, `source="pool_b"` will show it.
///
/// Cardinality is bounded: 3 outcomes x 4 sources.
pub(super) fn contradiction_labels(
    pool_a: &[Fact],
    contradictions: &[i64],
    duplicates: &[i64],
) -> (&'static str, &'static str) {
    let a_ids: HashSet<i64> = pool_a.iter().map(|f| f.id).collect();
    let (mut from_a, mut from_b) = (false, false);
    for id in contradictions.iter().chain(duplicates.iter()) {
        if a_ids.contains(id) {
            from_a = true;
        } else {
            from_b = true;
        }
    }
    let source = match (from_a, from_b) {
        (true, true) => "mixed",
        (true, false) => "pool_a",
        (false, true) => "pool_b",
        (false, false) => "none",
    };
    let outcome = if !contradictions.is_empty() {
        "contradiction"
    } else if !duplicates.is_empty() {
        "duplicate"
    } else {
        "no_conflict"
    };
    (outcome, source)
}

/// The episode-scoped identity of the fact about to be written — args-as-object
/// to keep [`is_within_episode_duplicate`] under clippy's too_many_arguments
/// threshold (`clippy.toml` sets it to 3).
///
/// `object_id` and `object_value` come from [`FactEndpoints`] and are mutually
/// exclusive; carrying both is what makes the check a FULL-triple comparison
/// rather than the subject+predicate one that dropped set-valued facts.
pub(super) struct EpisodeTriple<'a> {
    /// The episode currently being ingested.
    pub(super) episode_id: i64,
    /// Resolved object entity id, when the fact object is an entity reference.
    pub(super) object_id: Option<&'a str>,
    /// Literal object text, when the fact object is not an entity reference.
    pub(super) object_value: Option<&'a str>,
}

/// Has THIS episode already asserted this EXACT triple?
///
/// The check must be on the FULL triple. It once compared subject+predicate only,
/// and `pool_a` comes from `get_facts_by_subject_predicate` — i.e. it is
/// object-agnostic — so one episode asserting "Alice speaks English", "Alice speaks
/// French", "Alice speaks Spanish" stored ONLY THE FIRST. Facts 2 and 3 matched an
/// existing same-episode row on the pair and were dropped, with no count surfaced
/// to the caller.
///
/// Within a single episode there is no temporal ordering that could make one
/// assertion supersede another — they are co-asserted, so differing objects are
/// multiple values, not a contradiction. Cross-episode supersession remains the
/// separate, temporally-ordered mechanism and is unaffected.
pub(super) fn is_within_episode_duplicate(pool_a: &[Fact], candidate: EpisodeTriple<'_>) -> bool {
    let EpisodeTriple {
        episode_id,
        object_id,
        object_value,
    } = candidate;
    pool_a.iter().any(|f| {
        f.source_episode_id == Some(episode_id)
            && f.object_id.as_deref() == object_id
            && f.object_value.as_deref() == object_value
    })
}

/// Is this episode asserting a SECOND value for the same subject+predicate?
///
/// Detection kept as a counter, not a policy. A set-valued predicate is
/// indistinguishable from a contradiction without predicate-cardinality knowledge
/// the engine does not have, so every value is stored and the phenomenon is
/// measured instead — data loss is irreversible, detection is additive. Design
/// policy against real counts, not assumptions.
///
/// Callers must check [`is_within_episode_duplicate`] FIRST and `continue` on it,
/// or an exact repeat counts as a multivalue.
pub(super) fn is_within_episode_multivalue(pool_a: &[Fact], episode_id: i64) -> bool {
    pool_a
        .iter()
        .any(|f| f.source_episode_id == Some(episode_id))
}

/// Everything [`contradiction_event`] needs to describe a supersession.
///
/// An args-as-object because the rule reads seven things and `clippy.toml` caps
/// arguments at 3 — the same reason [`EpisodeTriple`] exists.
pub(super) struct ContradictionEventInput<'a> {
    /// Correctly-scoped candidate pool (same subject).
    pub(super) pool_a: &'a [Fact],
    /// Predicate-matched pool, already narrowed by [`pool_b_sharing_an_entity`].
    pub(super) pool_b: &'a [Fact],
    /// The prior fact being superseded, by id.
    pub(super) prior_fact_id: i64,
    /// The newly extracted triple that supersedes it.
    pub(super) new_fact: &'a ExtractedFact,
    /// Resolved subject entity id, shared by both facts.
    pub(super) subject_id: &'a str,
    /// Episode reference time — the new fact's world clock.
    pub(super) ref_time: DateTime<Utc>,
    /// Wall clock for the event itself. Passed in rather than read inside so the
    /// rule is deterministic and therefore testable.
    pub(super) detected_at: DateTime<Utc>,
}

/// Build the `ContradictionDetected` payload for a superseded fact, or `None`
/// when the prior fact is not in either pool.
///
/// **`None` means skip the emission, never fabricate one.** The payload's
/// `prior_fact` has to be a faithful snapshot of a row that actually existed; if
/// the id cannot be found in the pools that were just searched, there is nothing
/// truthful to report and a default-filled payload would be a lie in an audit
/// trail. This is the parse-loudly rule applied on the way out.
///
/// Resolution is always `Superseded` — `ingest_with` has no Retained or Merged
/// branch on this path.
///
/// An EMPTY object is legitimate and is not the same failure. A fact may be a
/// unary predicate carrying neither `object_id` nor `object_value`, and `""` is
/// the faithful representation of that prior fact, not a silently defaulted
/// required field.
pub(super) fn contradiction_event(
    input: ContradictionEventInput<'_>,
) -> Option<Box<ContradictionDetected>> {
    let ContradictionEventInput {
        pool_a,
        pool_b,
        prior_fact_id,
        new_fact,
        subject_id,
        ref_time,
        detected_at,
    } = input;

    pool_a
        .iter()
        .chain(pool_b.iter())
        .find(|f| f.id == prior_fact_id)
        .map(|prior| {
            let prior_object = prior
                .object_id
                .clone()
                .or_else(|| prior.object_value.clone())
                .unwrap_or_default();
            Box::new(ContradictionDetected {
                entity_id: EntityId(subject_id.to_owned()),
                prior_fact: SinkFact {
                    subject: prior.subject_id.clone(),
                    predicate: prior.predicate.clone(),
                    object: prior_object,
                    valid_at: Some(prior.valid_from),
                },
                new_fact: SinkFact {
                    subject: new_fact.subject.clone(),
                    predicate: new_fact.predicate.clone(),
                    object: new_fact.object.clone(),
                    valid_at: Some(ref_time),
                },
                resolution: ContradictionResolution::Superseded,
                detected_at,
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    /// Minimal `Fact` — only the fields these rules read carry meaning. `id` is
    /// fixed at 1 because only [`contradiction_labels`] distinguishes facts by id;
    /// those tests use [`fact_with_id`].
    fn fact(subject: &str, object_id: Option<&str>, episode: Option<i64>) -> Fact {
        Fact {
            id: 1,
            subject_id: subject.to_owned(),
            predicate: "likes".to_owned(),
            object_id: object_id.map(str::to_owned),
            object_value: None,
            properties: None,
            valid_from: Utc::now(),
            valid_to: None,
            recorded_at: Utc::now(),
            expired_at: None,
            invalid_at: None,
            group_id: None,
            confidence: 1.0,
            source_episode_id: episode,
            memory_type: None,
            content_hash: None,
            access_count: 0,
            subject_group_id: None,
            object_group_id: None,
        }
    }

    /// A pool_a member with a specific id, for the source-attribution tests.
    fn fact_with_id(id: i64, subject: &str) -> Fact {
        Fact {
            id,
            ..fact(subject, None, None)
        }
    }

    fn literal_fact(subject: &str, value: &str, episode: Option<i64>) -> Fact {
        Fact {
            object_value: Some(value.to_owned()),
            ..fact(subject, None, episode)
        }
    }

    fn extracted(subject: &str, object: &str, is_entity_ref: bool) -> ExtractedFact {
        ExtractedFact {
            subject: subject.to_owned(),
            predicate: "likes".to_owned(),
            object: object.to_owned(),
            is_entity_ref,
            confidence: 1.0,
            valid_at: None,
        }
    }

    // ── resolve_fact_endpoints ──────────────────────────────────────────────

    #[test]
    fn endpoints_route_both_names_through_the_merge_map() {
        let mut map = HashMap::new();
        map.insert("alice".to_owned(), "alice_johnson".to_owned());
        map.insert("bob".to_owned(), "robert_smith".to_owned());

        let e = resolve_fact_endpoints(&extracted("Alice", "Bob", true), &map);

        assert_eq!(e.subject_id, "alice_johnson");
        assert_eq!(e.object_id.as_deref(), Some("robert_smith"));
        assert!(e.object_value.is_none());
    }

    #[test]
    fn endpoints_fall_back_to_the_normalized_name_when_unmapped() {
        // The forward-reference pre-scan inserted a stub under exactly this id,
        // so the fallback lands on a real row rather than dangling.
        let e = resolve_fact_endpoints(&extracted("Lysfjord", "Ines", true), &HashMap::new());

        assert_eq!(e.subject_id, normalize_name("Lysfjord"));
        assert_eq!(e.object_id, Some(normalize_name("Ines")));
    }

    #[test]
    fn a_literal_object_never_becomes_an_entity_id() {
        let e = resolve_fact_endpoints(&extracted("Alice", "42", false), &HashMap::new());

        assert!(e.object_id.is_none());
        assert_eq!(e.object_value.as_deref(), Some("42"));
    }

    #[test]
    fn a_literal_object_keeps_its_original_casing_and_spacing() {
        // object_value is stored verbatim — normalizing it would corrupt values.
        let e = resolve_fact_endpoints(&extracted("Alice", "Dark Roast", false), &HashMap::new());

        assert_eq!(e.object_value.as_deref(), Some("Dark Roast"));
    }

    // ── pool_b_sharing_an_entity ────────────────────────────────────────────

    #[test]
    fn pool_b_drops_an_unrelated_subject() {
        // The regression this filter exists for: "Bob likes coffee" reached the
        // LLM as a contradiction candidate for "Alice likes tea".
        let kept = pool_b_sharing_an_entity(vec![fact("bob", None, None)], "alice", None);

        assert!(kept.is_empty());
    }

    #[test]
    fn pool_b_keeps_a_shared_subject() {
        let kept = pool_b_sharing_an_entity(vec![fact("alice", None, None)], "alice", None);

        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn pool_b_keeps_a_shared_object() {
        let kept =
            pool_b_sharing_an_entity(vec![fact("bob", Some("acme"), None)], "alice", Some("acme"));

        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn pool_b_keeps_a_candidate_whose_object_is_our_subject() {
        // cross_ref arm, direction 1: (bob, likes, alice) vs a new fact about alice.
        let kept = pool_b_sharing_an_entity(vec![fact("bob", Some("alice"), None)], "alice", None);

        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn pool_b_keeps_a_candidate_whose_subject_is_our_object() {
        // cross_ref arm, direction 2 — the one an `a == b`-only filter drops.
        let kept = pool_b_sharing_an_entity(vec![fact("acme", None, None)], "alice", Some("acme"));

        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn two_literal_object_facts_do_not_share_an_object() {
        // `(None, None)` must NOT count as a match, or every literal-object fact
        // in the FTS window survives the filter and the cost fix is undone.
        let kept = pool_b_sharing_an_entity(vec![literal_fact("bob", "tea", None)], "alice", None);

        assert!(kept.is_empty());
    }

    // ── contradiction_labels ────────────────────────────────────────────────

    #[test]
    fn labels_attribute_a_hit_to_the_pool_that_supplied_it() {
        let pool_a = vec![fact_with_id(7, "alice")];

        assert_eq!(
            contradiction_labels(&pool_a, &[7], &[]),
            ("contradiction", "pool_a")
        );
        assert_eq!(
            contradiction_labels(&pool_a, &[99], &[]),
            ("contradiction", "pool_b")
        );
        assert_eq!(
            contradiction_labels(&pool_a, &[7, 99], &[]),
            ("contradiction", "mixed")
        );
    }

    #[test]
    fn labels_report_no_conflict_and_no_source_when_nothing_fired() {
        assert_eq!(contradiction_labels(&[], &[], &[]), ("no_conflict", "none"));
    }

    #[test]
    fn a_duplicate_still_attributes_its_source_pool() {
        // Duplicates count toward `source` as well as contradictions — both are
        // evidence the pool earned its LLM call.
        let pool_a = vec![fact_with_id(7, "alice")];

        assert_eq!(
            contradiction_labels(&pool_a, &[], &[7]),
            ("duplicate", "pool_a")
        );
        assert_eq!(
            contradiction_labels(&pool_a, &[], &[99]),
            ("duplicate", "pool_b")
        );
    }

    #[test]
    fn contradiction_outranks_duplicate_in_the_outcome_label() {
        let pool_a = vec![fact_with_id(7, "alice")];

        assert_eq!(contradiction_labels(&pool_a, &[7], &[8]).0, "contradiction");
    }

    // ── is_within_episode_duplicate ─────────────────────────────────────────

    #[test]
    fn set_valued_predicates_survive_the_duplicate_check() {
        // The exact regression: "Alice speaks English/French/Spanish" in ONE
        // episode kept only the first, because the check ignored the object.
        let pool_a = vec![fact("alice", Some("english"), Some(42))];

        assert!(!is_within_episode_duplicate(
            &pool_a,
            EpisodeTriple {
                episode_id: 42,
                object_id: Some("french"),
                object_value: None
            }
        ));
    }

    #[test]
    fn an_exact_repeated_triple_is_a_duplicate() {
        let pool_a = vec![fact("alice", Some("english"), Some(42))];

        assert!(is_within_episode_duplicate(
            &pool_a,
            EpisodeTriple {
                episode_id: 42,
                object_id: Some("english"),
                object_value: None
            }
        ));
    }

    #[test]
    fn the_same_triple_from_another_episode_is_not_a_duplicate() {
        // Cross-episode supersession is the separate temporally-ordered
        // mechanism; this check must not pre-empt it.
        let pool_a = vec![fact("alice", Some("english"), Some(41))];

        assert!(!is_within_episode_duplicate(
            &pool_a,
            EpisodeTriple {
                episode_id: 42,
                object_id: Some("english"),
                object_value: None
            }
        ));
    }

    #[test]
    fn literal_objects_compare_on_value_not_id() {
        let pool_a = vec![literal_fact("alice", "tea", Some(42))];

        assert!(is_within_episode_duplicate(
            &pool_a,
            EpisodeTriple {
                episode_id: 42,
                object_id: None,
                object_value: Some("tea")
            }
        ));
        assert!(!is_within_episode_duplicate(
            &pool_a,
            EpisodeTriple {
                episode_id: 42,
                object_id: None,
                object_value: Some("coffee")
            }
        ));
    }

    // ── is_within_episode_multivalue ────────────────────────────────────────

    #[test]
    fn a_second_value_in_the_same_episode_is_multivalue() {
        let pool_a = vec![fact("alice", Some("english"), Some(42))];

        assert!(is_within_episode_multivalue(&pool_a, 42));
    }

    #[test]
    fn a_prior_episodes_value_is_not_multivalue() {
        let pool_a = vec![fact("alice", Some("english"), Some(41))];

        assert!(!is_within_episode_multivalue(&pool_a, 42));
    }

    #[test]
    fn an_empty_pool_is_neither_duplicate_nor_multivalue() {
        assert!(!is_within_episode_duplicate(
            &[],
            EpisodeTriple {
                episode_id: 42,
                object_id: Some("english"),
                object_value: None
            }
        ));
        assert!(!is_within_episode_multivalue(&[], 42));
    }

    // ── contradiction_event ─────────────────────────────────────────────────

    /// A fixed instant, so payload timestamps are assertable rather than merely
    /// present. `Utc::now()` inside a helper would make the clock untestable —
    /// which is exactly why `contradiction_event` takes `detected_at` instead of
    /// reading the clock itself.
    fn ref_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).expect("fixed timestamp is valid")
    }

    /// When the contradiction was NOTICED — deliberately a DIFFERENT instant from
    /// [`ref_time`].
    ///
    /// ⚠️ These two were the same value here, and that alone made a whole class of
    /// bug invisible: a rule that assigned `detected_at` where `ref_time` belonged
    /// would still satisfy an assertion named "valid_at is the episode ref_time",
    /// because the fixture made the two indistinguishable. A fixture that collapses
    /// two semantically distinct fields cannot tell them apart, and neither can any
    /// assertion built on it.
    fn detected_at() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_009_999, 0).expect("fixed timestamp is valid")
    }

    /// The fixed inputs every payload test shares; each test varies one. Pools
    /// travel as a pair to stay inside the 3-argument cap without an `#[allow]`.
    fn event_input<'a>(
        pools: (&'a [Fact], &'a [Fact]),
        prior_fact_id: i64,
        new_fact: &'a ExtractedFact,
    ) -> ContradictionEventInput<'a> {
        ContradictionEventInput {
            pool_a: pools.0,
            pool_b: pools.1,
            prior_fact_id,
            new_fact,
            subject_id: "alice",
            ref_time: ref_time(),
            detected_at: detected_at(),
        }
    }

    #[test]
    fn a_prior_fact_in_neither_pool_emits_nothing() {
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[], &[]), 999, &new_fact));

        assert!(
            got.is_none(),
            "a payload whose prior_fact cannot be evidenced is a lie in an audit trail; skip beats fabricate"
        );
    }

    #[test]
    fn a_prior_fact_found_in_pool_a_is_snapshotted() {
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert_eq!(got.prior_fact.subject, "alice");
        assert_eq!(got.prior_fact.predicate, "likes");
        assert_eq!(got.new_fact.object, "Berlin");
    }

    #[test]
    fn a_prior_fact_found_only_in_pool_b_is_still_snapshotted() {
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[], &[prior]), 7, &new_fact));

        assert!(
            got.is_some(),
            "pool_b is searched too — restricting the lookup to pool_a would silently drop predicate-matched supersessions"
        );
    }

    #[test]
    fn an_objectless_prior_fact_reports_an_empty_object_rather_than_skipping() {
        // Neither object_id nor object_value — a unary predicate.
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("an objectless prior fact is legitimate, not a missing field");

        assert_eq!(
            got.prior_fact.object, "",
            "\"\" is the faithful representation of an objectless triple, not a silent default"
        );
    }

    #[test]
    fn a_literal_object_is_reported_when_there_is_no_object_id() {
        let prior = Fact {
            id: 7,
            ..literal_fact("alice", "blue", None)
        };
        let new_fact = extracted("Alice", "red", false);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert_eq!(got.prior_fact.object, "blue");
    }

    #[test]
    fn an_entity_object_outranks_a_literal_on_the_same_prior_fact() {
        let prior = Fact {
            id: 7,
            object_id: Some("berlin".to_owned()),
            object_value: Some("Berlin, DE".to_owned()),
            ..fact("alice", None, None)
        };
        let new_fact = extracted("Alice", "Munich", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert_eq!(
            got.prior_fact.object, "berlin",
            "the entity reference is the resolved identity; the literal is only a fallback"
        );
    }

    #[test]
    fn the_resolution_on_this_path_is_always_superseded() {
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert!(matches!(
            got.resolution,
            ContradictionResolution::Superseded
        ));
    }

    #[test]
    fn the_new_facts_valid_at_is_the_episode_ref_time() {
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert_eq!(
            got.new_fact.valid_at,
            Some(ref_time()),
            "the new fact's world clock is when it became TRUE, not when the contradiction was noticed"
        );
        assert_ne!(
            got.new_fact.valid_at,
            Some(detected_at()),
            "if these two clocks were equal the assertion above would prove nothing"
        );
    }

    #[test]
    fn the_events_detected_at_is_the_wall_clock_not_the_episode_ref_time() {
        let prior = fact_with_id(7, "alice");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[prior], &[]), 7, &new_fact))
            .expect("prior fact is in pool_a");

        assert_eq!(got.detected_at, detected_at());
        assert_ne!(
            got.detected_at, ref_time(),
            "the two clocks are separate arguments precisely so they can differ"
        );
    }

    #[test]
    fn the_first_matching_id_wins_when_a_fact_sits_in_both_pools() {
        let in_a = fact_with_id(7, "alice_from_pool_a");
        let in_b = fact_with_id(7, "alice_from_pool_b");
        let new_fact = extracted("Alice", "Berlin", true);

        let got = contradiction_event(event_input((&[in_a], &[in_b]), 7, &new_fact))
            .expect("prior fact is in both pools");

        assert_eq!(
            got.prior_fact.subject, "alice_from_pool_a",
            "pool_a is the correctly-scoped arm and is chained first, so it wins the tie"
        );
    }
}

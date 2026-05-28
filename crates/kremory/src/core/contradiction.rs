use chrono::{DateTime, Utc};
use std::sync::Arc;

use crate::core::error::Result;
use crate::core::intelligence::ExtractedFact;
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::core::schema::Fact;

// ---------------------------------------------------------------------------
// Temporal Overlap
// ---------------------------------------------------------------------------

/// Check if two temporal intervals overlap.
/// Intervals are half-open: [start, end) where None end = +infinity.
///
/// Two intervals [a_start, a_end) and [b_start, b_end) overlap when:
///   a_start < b_end AND b_start < a_end
/// Where None end means +infinity (always satisfies the < comparison).
pub(crate) fn temporal_overlap(
    existing_start: &DateTime<Utc>,
    existing_end: Option<&DateTime<Utc>>,
    new_start: &DateTime<Utc>,
    new_end: Option<&DateTime<Utc>>,
) -> bool {
    // existing starts before new ends (or new has no end — +infinity)
    let a_before_b_end = new_end.is_none_or(|be| existing_start < be);
    // new starts before existing ends (or existing has no end — +infinity)
    let b_before_a_end = existing_end.is_none_or(|ae| new_start < ae);

    a_before_b_end && b_before_a_end
}

// ---------------------------------------------------------------------------
// ContradictionOutcome
// ---------------------------------------------------------------------------

/// Outcome of resolving a contradiction under a known namespace policy.
///
/// Returned by `resolve_contradiction` (ADR-029b Decision 2) to distinguish
/// Mutable-path mutation from AppendOnly-path insertion. Callers use this to
/// audit which resolution strategy was applied.
///
/// The `resolve_contradiction` function that returns this type is a 029b
/// deliverable wired into the contradiction resolver in a follow-up step.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(dead_code)]
pub(crate) enum ContradictionOutcome {
    /// Mutable-ns path: `prior_fact_id`'s `valid_to` was updated to
    /// `new_fact.valid_from`; the new fact was inserted.
    Superseded { prior_fact_id: i64 },
    /// AppendOnly-ns path: a superseding fact was appended; the prior fact
    /// (`prior_fact_id`) was NOT mutated. Overlap is resolved by the
    /// `recorded_at DESC LIMIT 1` tie-breaker (Decision 3).
    AppendedSuperseder { prior_fact_id: i64 },
}

// ---------------------------------------------------------------------------
// ContradictionResult
// ---------------------------------------------------------------------------

/// Result of contradiction detection for a single fact.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ContradictionResult {
    /// Facts identified as contradictions (should be invalidated).
    pub(crate) contradictions: Vec<i64>,
    /// Facts identified as duplicates (same content, should be merged).
    pub(crate) duplicates: Vec<i64>,
    /// Whether the new fact is consistent with all existing facts.
    pub(crate) is_consistent: bool,
}

impl ContradictionResult {
    pub(crate) fn no_conflicts() -> Self {
        Self {
            contradictions: vec![],
            duplicates: vec![],
            is_consistent: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Dual-List Prompt Builder
// ---------------------------------------------------------------------------

/// Build a dual-list indexed prompt for contradiction classification.
/// Pool A (same-endpoint facts) and Pool B (semantically similar facts)
/// are numbered with continuous indexing across both lists.
///
/// Returns (prompt_string, index_to_fact_id_map).
pub(crate) fn build_dual_list_prompt(
    new_fact: &ExtractedFact,
    pool_a: &[Fact],
    pool_b: &[Fact],
) -> (String, Vec<i64>) {
    let mut prompt = format!(
        "A new fact has been extracted:\n  Subject: {}\n  Predicate: {}\n  Object: {}\n\n",
        new_fact.subject, new_fact.predicate, new_fact.object
    );

    let mut index_map: Vec<i64> = Vec::new();
    let mut idx = 1usize;

    // Pool A: same-endpoint facts
    prompt.push_str("List 1 — Facts with same subject and predicate:\n");
    if pool_a.is_empty() {
        prompt.push_str("  (none)\n");
    }
    for fact in pool_a {
        let obj = fact
            .object_value
            .as_deref()
            .or(fact.object_id.as_deref())
            .unwrap_or("(unknown)");
        prompt.push_str(&format!(
            "  [{}] {} → {} → {} (valid from: {})\n",
            idx,
            fact.subject_id,
            fact.predicate,
            obj,
            fact.valid_from.to_rfc3339()
        ));
        index_map.push(fact.id);
        idx += 1;
    }

    // Pool B: semantically similar facts (skip any already in Pool A)
    prompt.push_str("\nList 2 — Semantically similar facts:\n");
    let pool_b_deduped: Vec<&Fact> = pool_b
        .iter()
        .filter(|f| !index_map.contains(&f.id))
        .collect();
    if pool_b_deduped.is_empty() {
        prompt.push_str("  (none)\n");
    }
    for fact in &pool_b_deduped {
        let obj = fact
            .object_value
            .as_deref()
            .or(fact.object_id.as_deref())
            .unwrap_or("(unknown)");
        prompt.push_str(&format!(
            "  [{}] {} → {} → {} (valid from: {})\n",
            idx,
            fact.subject_id,
            fact.predicate,
            obj,
            fact.valid_from.to_rfc3339()
        ));
        index_map.push(fact.id);
        idx += 1;
    }

    prompt.push_str("\nFor each existing fact that the new fact CONTRADICTS (makes false or outdated), return its index number.\nIf the new fact is an UPDATE (same relationship but newer value), return that index too.\nIf no contradictions, return an empty array [].\n\nOutput a JSON array of index numbers.");

    (prompt, index_map)
}

// ---------------------------------------------------------------------------
// Index Parser
// ---------------------------------------------------------------------------

/// Parse LLM response containing a JSON array of index numbers.
/// Handles: "[1, 3]", "[]", "[1]", and malformed responses (returns empty).
pub(crate) fn parse_index_list(json: &str) -> Vec<usize> {
    serde_json::from_str::<Vec<usize>>(json.trim()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// GBNF Grammar for Index Lists
// ---------------------------------------------------------------------------

/// GBNF grammar for a JSON array of integers.
pub const GBNF_INDEX_LIST: &str = r#"root ::= "[" ws "]" | "[" ws number ("," ws number)* ws "]"
number ::= [1-9] [0-9]*
ws ::= [ \t\n]*"#;

// ---------------------------------------------------------------------------
// TwoPoolDetector
// ---------------------------------------------------------------------------

/// Contradiction detector using two candidate pools and temporal overlap filtering.
pub(crate) struct TwoPoolDetector<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> TwoPoolDetector<L> {
    pub(crate) fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }

    /// Detect contradictions for a new fact against existing facts.
    ///
    /// pool_a: facts with same subject + predicate (already fetched by caller)
    /// pool_b: semantically similar facts (already fetched by caller)
    /// reference_time: when the new fact becomes valid
    pub async fn detect(
        &self,
        new_fact: &ExtractedFact,
        pool_a: &[Fact],
        pool_b: &[Fact],
        reference_time: &DateTime<Utc>,
    ) -> Result<ContradictionResult> {
        // Filter to temporally overlapping facts only
        let overlapping_a: Vec<&Fact> = pool_a
            .iter()
            .filter(|f| temporal_overlap(&f.valid_from, f.valid_to.as_ref(), reference_time, None))
            .collect();

        let overlapping_b: Vec<&Fact> = pool_b
            .iter()
            .filter(|f| temporal_overlap(&f.valid_from, f.valid_to.as_ref(), reference_time, None))
            // dedup: exclude anything already in pool_a
            .filter(|f| !pool_a.iter().any(|a| a.id == f.id))
            .collect();

        // Check for exact duplicates (same endpoints + same predicate + same object value)
        let mut duplicates: Vec<i64> = Vec::new();
        for fact in &overlapping_a {
            let obj = fact
                .object_value
                .as_deref()
                .or(fact.object_id.as_deref())
                .unwrap_or("");
            if obj == new_fact.object {
                duplicates.push(fact.id);
            }
        }

        // If all overlapping Pool-A facts are duplicates and Pool B is empty, skip LLM
        if overlapping_a.len() == duplicates.len() && overlapping_b.is_empty() {
            return Ok(ContradictionResult {
                contradictions: vec![],
                duplicates,
                is_consistent: true,
            });
        }

        // If nothing overlaps temporally at all, trivially consistent
        if overlapping_a.is_empty() && overlapping_b.is_empty() {
            return Ok(ContradictionResult::no_conflicts());
        }

        // Build dual-list prompt with overlapping facts only
        let overlapping_a_owned: Vec<Fact> = overlapping_a.iter().map(|f| (*f).clone()).collect();
        let overlapping_b_owned: Vec<Fact> = overlapping_b.iter().map(|f| (*f).clone()).collect();
        let (prompt, index_map) =
            build_dual_list_prompt(new_fact, &overlapping_a_owned, &overlapping_b_owned);

        // LLM classification
        let contradiction_msgs = vec![
            chat_msg_system("You are a fact consistency checker. Identify which existing facts are contradicted by a new fact."),
            chat_msg_user(prompt),
        ];
        let response = self
            .llm
            .chat_with_tools(&contradiction_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let response_text = response.text().unwrap_or_default();

        let indices = parse_index_list(&response_text);
        let contradictions: Vec<i64> = indices
            .iter()
            .filter_map(|&i| {
                if i >= 1 && i <= index_map.len() {
                    Some(index_map[i - 1]) // 1-indexed to 0-indexed
                } else {
                    None
                }
            })
            .collect();

        Ok(ContradictionResult {
            is_consistent: contradictions.is_empty() && duplicates.is_empty(),
            contradictions,
            duplicates,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::provider::MockChatProvider;
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn dt(offset_days: i64) -> DateTime<Utc> {
        Utc::now() + Duration::days(offset_days)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_fact(
        id: i64,
        subject_id: &str,
        predicate: &str,
        object_id: Option<&str>,
        object_value: Option<&str>,
        valid_from: DateTime<Utc>,
        valid_to: Option<DateTime<Utc>>,
    ) -> Fact {
        let now = Utc::now();
        Fact {
            id,
            subject_id: subject_id.to_string(),
            predicate: predicate.to_string(),
            object_id: object_id.map(str::to_string),
            object_value: object_value.map(str::to_string),
            properties: None,
            valid_from,
            valid_to,
            recorded_at: now,
            expired_at: None,
            invalid_at: None,
            group_id: None,
            confidence: 1.0,
            source_episode_id: None,
            memory_type: None,
            content_hash: None,
            access_count: 0,
            subject_group_id: None,
            object_group_id: None,
        }
    }

    fn make_extracted(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
        ExtractedFact {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            is_entity_ref: true,
            confidence: 1.0,
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    // ── Temporal Overlap ─────────────────────────────────────────────────────

    #[test]
    fn test_overlap_both_open_ended() {
        // [t0, ∞) and [t1, ∞) — both infinite, must overlap
        let t0 = dt(-10);
        let t1 = dt(-5);
        assert!(temporal_overlap(&t0, None, &t1, None));
    }

    #[test]
    fn test_overlap_non_overlapping() {
        // [t0, t1) and [t2, t3) where t1 < t2 — gap between intervals
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(!temporal_overlap(&t0, Some(&t1), &t2, Some(&t3)));
    }

    #[test]
    fn test_overlap_adjacent() {
        // [t0, t1) and [t1, t2) — share a boundary point, half-open so they don't overlap
        let t0 = dt(-10);
        let t1 = dt(-5);
        let t2 = dt(0);
        assert!(!temporal_overlap(&t0, Some(&t1), &t1, Some(&t2)));
    }

    #[test]
    fn test_overlap_contained() {
        // [t0, t3) contains [t1, t2) — overlap
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(temporal_overlap(&t0, Some(&t3), &t1, Some(&t2)));
    }

    #[test]
    fn test_overlap_partial() {
        // [t0, t2) and [t1, t3) where t0 < t1 < t2 < t3 — partial overlap
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(temporal_overlap(&t0, Some(&t2), &t1, Some(&t3)));
    }

    // ── Dual-List Prompt ──────────────────────────────────────────────────────

    #[test]
    fn test_dual_list_continuous_indexing() {
        let new_fact = make_extracted("alice", "works_at", "newco");
        let now = Utc::now();
        let fact_a = make_fact(1, "alice", "works_at", Some("acme"), None, now, None);
        let fact_b = make_fact(2, "alice", "employed_by", Some("acme"), None, now, None);

        let (prompt, index_map) = build_dual_list_prompt(&new_fact, &[fact_a], &[fact_b]);

        // Indices in the prompt should be [1] then [2]
        assert!(prompt.contains("[1]"), "Pool A should start at index 1");
        assert!(prompt.contains("[2]"), "Pool B should start at index 2");
        assert_eq!(index_map.len(), 2);
        assert_eq!(index_map[0], 1);
        assert_eq!(index_map[1], 2);
    }

    #[test]
    fn test_dual_list_empty_pools() {
        let new_fact = make_extracted("alice", "works_at", "newco");
        let (prompt, index_map) = build_dual_list_prompt(&new_fact, &[], &[]);

        assert!(prompt.contains("(none)"), "empty pools should show (none)");
        // (none) appears twice — once per list
        assert_eq!(
            prompt.matches("(none)").count(),
            2,
            "both empty lists should display (none)"
        );
        assert!(index_map.is_empty());
    }

    #[test]
    fn test_dual_list_dedup_across_pools() {
        // Fact with id=1 appears in both Pool A and Pool B — should only be listed once
        let new_fact = make_extracted("alice", "works_at", "newco");
        let now = Utc::now();
        let shared_fact = make_fact(1, "alice", "works_at", Some("acme"), None, now, None);
        let unique_b = make_fact(2, "alice", "employed_by", Some("acme"), None, now, None);

        let pool_a = vec![shared_fact.clone()];
        let pool_b = vec![shared_fact, unique_b];

        let (prompt, index_map) = build_dual_list_prompt(&new_fact, &pool_a, &pool_b);

        // index_map should only have 2 entries: fact 1 (from A) and fact 2 (unique B)
        assert_eq!(
            index_map.len(),
            2,
            "dedup should prevent double-counting fact 1"
        );
        assert_eq!(index_map[0], 1);
        assert_eq!(index_map[1], 2);
        assert!(prompt.contains("[1]"));
        assert!(prompt.contains("[2]"));
        // Should not have index [3]
        assert!(!prompt.contains("[3]"));
    }

    // ── Index Parser ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_index_list_valid() {
        assert_eq!(parse_index_list("[1, 3]"), vec![1usize, 3]);
    }

    #[test]
    fn test_parse_index_list_empty() {
        assert_eq!(parse_index_list("[]"), Vec::<usize>::new());
    }

    #[test]
    fn test_parse_index_list_malformed() {
        assert_eq!(parse_index_list("garbage"), Vec::<usize>::new());
    }

    // ── TwoPoolDetector ───────────────────────────────────────────────────────

    #[test]
    fn test_no_overlap_consistent() {
        // Pool A fact ended long before the reference time — no temporal overlap
        let t_old_start = dt(-30);
        let t_old_end = dt(-20);
        let reference = dt(-10);

        let fact = make_fact(
            1,
            "alice",
            "works_at",
            Some("acme"),
            None,
            t_old_start,
            Some(t_old_end),
        );
        let new_fact = make_extracted("alice", "works_at", "newco");

        let client = Arc::new(MockChatProvider::new(HashMap::new()));
        let detector = TwoPoolDetector::new(client);

        let result = block_on(detector.detect(&new_fact, &[fact], &[], &reference)).unwrap();
        assert!(result.is_consistent);
        assert!(result.contradictions.is_empty());
        assert!(result.duplicates.is_empty());
    }

    #[test]
    fn test_duplicate_detected() {
        // Pool A fact has the same object as the new fact — should be a duplicate
        let t_start = dt(-10);
        let reference = dt(0);

        let fact = make_fact(42, "alice", "works_at", Some("acme"), None, t_start, None);
        // new_fact object matches fact.object_id
        let new_fact = make_extracted("alice", "works_at", "acme");

        let client = Arc::new(MockChatProvider::new(HashMap::new()));
        let detector = TwoPoolDetector::new(client);

        let result = block_on(detector.detect(&new_fact, &[fact], &[], &reference)).unwrap();
        assert!(result.contradictions.is_empty());
        assert_eq!(result.duplicates, vec![42i64]);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_contradiction_via_mock_llm() {
        // Pool A has one fact that overlaps temporally and has a different object.
        // MockChatProvider returns "[1]" for any prompt containing "works_at".
        let t_start = dt(-10);
        let reference = dt(0);

        let fact = make_fact(99, "alice", "works_at", Some("acme"), None, t_start, None);
        let new_fact = make_extracted("alice", "works_at", "newco");

        let mut responses = HashMap::new();
        responses.insert("works_at".to_string(), "[1]".to_string());
        let client = Arc::new(MockChatProvider::new(responses));
        let detector = TwoPoolDetector::new(client);

        let result = block_on(detector.detect(&new_fact, &[fact], &[], &reference)).unwrap();

        // Index 1 maps to fact id 99
        assert_eq!(result.contradictions, vec![99i64]);
        assert!(result.duplicates.is_empty());
        assert!(!result.is_consistent);
    }
}

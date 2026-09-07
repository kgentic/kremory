use chrono::{DateTime, Utc};
use metrics::counter;
use std::sync::Arc;

use crate::core::error::Result;
use crate::core::extraction::schemas::{ContradictionVerdictWrapper, SCHEMA_CONTRADICTION_VERDICT};
use crate::core::extraction::structured;
use crate::core::intelligence::ExtractedFact;
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::core::schema::Fact;

// ---------------------------------------------------------------------------
// Temporal Overlap
// ---------------------------------------------------------------------------

/// Bundled parameters for [`temporal_overlap`] — args-as-object to keep the
/// function under clippy's `too_many_arguments` threshold.
///
/// Both intervals are half-open: `[start, end)` where a `None` end = +infinity.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TemporalOverlapParams<'a> {
    /// Start of the existing interval.
    pub existing_start: &'a DateTime<Utc>,
    /// End of the existing interval (`None` = +infinity).
    pub existing_end: Option<&'a DateTime<Utc>>,
    /// Start of the new interval.
    pub new_start: &'a DateTime<Utc>,
    /// End of the new interval (`None` = +infinity).
    pub new_end: Option<&'a DateTime<Utc>>,
}

/// Check if two temporal intervals overlap.
/// Intervals are half-open: [start, end) where None end = +infinity.
///
/// Two intervals [a_start, a_end) and [b_start, b_end) overlap when:
///   a_start < b_end AND b_start < a_end
/// Where None end means +infinity (always satisfies the < comparison).
pub(crate) fn temporal_overlap(params: TemporalOverlapParams<'_>) -> bool {
    let TemporalOverlapParams {
        existing_start,
        existing_end,
        new_start,
        new_end,
    } = params;
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
/// Returned by `resolve_contradiction` to distinguish
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

    // The previous instruction was:
    //
    //   "For each existing fact that the new fact CONTRADICTS ... return its index.
    //    If the new fact is an UPDATE (same relationship but newer value), return
    //    that index too."
    //
    // That second sentence describes EVERY item in a list, so it instructed the
    // model to destroy set-valued facts: a festival's 2nd..6th performer each
    // superseded the previous. Measured on the production model
    // (openai/gpt-oss-120b, temp 0): **7 of 8** genuinely set-valued
    // pairs were superseded. Note the direction — the WEAKER local model scored
    // 5/8. Capability made it WORSE, because the defect was in the INSTRUCTION,
    // not the judgement: a better model follows a wrong instruction more
    // faithfully. Model upgrades would have degraded this further.
    //
    // Replaced with a COEXISTENCE question, which measured **0 of 8** destroyed
    // on the same model — no cardinality model needed, declared or derived
    // (deriving it is impossible here: 67% of predicates appear exactly once).
    //
    // The coexistence question alone then MISSED 4 of 6 genuine updates
    // (`works_at`, `current_job_title`, ...) — because the model is arguably
    // right that both COULD be true (a person can hold two jobs). "Can both be
    // true?" is world knowledge; "did they mean ALSO or INSTEAD?" is TEMPORAL.
    // So the second half below leans on the `valid from:` timestamps already
    // rendered above. Same two-signal shape as Graphiti (LLM verdict + temporal
    // order, edge_operations.py:538-573) but with the RIGHT question in the LLM
    // slot — Graphiti asks "does this contradict?", which is what produces its
    // own version of this bug.
    //
    // Worked examples of BOTH outcomes are deliberate. Every competitor audited
    // (Graphiti, CORE, mem0's legacy path) shows the model only a
    // supersession example, which is exactly what biases it toward destruction.
    prompt.push_str(
        "\nDecide, for each existing fact above, whether the NEW fact REPLACES it.\n\
         \n\
         Step 1 — Can both facts be true AT THE SAME TIME?\n\
         Many relationships naturally hold SEVERAL values at once: a festival has many\n\
         performers, a recipe has many ingredients, a system processes many inputs, a\n\
         place is found in many regions. If the new fact is simply ANOTHER value of\n\
         such a relationship, it is an ADDITION — do NOT return that index.\n\
         \n\
         Step 2 — Only if both CANNOT hold at once, is this a replacement?\n\
         Some relationships hold one value at a time: a current employer, a current\n\
         job title, a scheduled time, a status, a place of residence. When the new\n\
         fact states a later value for such a relationship, it REPLACES the earlier\n\
         one — return that index. Use the 'valid from' timestamps to judge which is\n\
         later: a replacement supersedes an EARLIER fact, never a later one.\n\
         \n\
         When genuinely unsure, prefer ADDITION (an empty list). Keeping a stale fact\n\
         is recoverable; deleting a correct one is not.\n\
         \n\
         Examples:\n\
           existing: festival -> has_performer -> billie eilish\n\
           new:      festival -> has_performer -> the 1975\n\
           => {\"indices\": [], \"reason\": \"a festival has many performers; this is an additional value\"}\n\
         \n\
           existing: alice -> works_at -> acme corp   (valid from: 2023-01-01)\n\
           new:      alice -> works_at -> globex      (valid from: 2024-06-01)\n\
           => {\"indices\": [1], \"reason\": \"one current employer at a time; the later fact replaces the earlier\"}\n\
         \n\
         Output JSON in this exact format: {\"indices\": [n, n, ...], \"reason\": \"<brief justification>\"} \
         where each n is the 1-based index of a fact the new fact REPLACES. \
         If nothing is replaced: {\"indices\": [], \"reason\": \"<why both can coexist>\"}.",
    );

    (prompt, index_map)
}

// ---------------------------------------------------------------------------
// Index Parser
// ---------------------------------------------------------------------------

/// Parse LLM response containing a JSON index list.
///
/// Requires the FULL wrapped shape `{"indices": [1, 3], "reason": "..."}` —
/// deserialised directly as [`ContradictionVerdictWrapper`], whose `reason`
/// field carries no `#[serde(default)]`:
/// a verdict missing its audit-trail justification is a PARSE FAILURE, not a
/// degraded-but-usable result, so the fallback ladder can retry rather than
/// silently accepting an incomplete response. This deliberately does NOT
/// route through the shared shape-tolerant `parse_items` helper used by the
/// other extraction parsers in this crate — that helper only inspects the
/// `indices` key and has no concept of `reason` at all, so "shape-tolerant"
/// there would silently mean "reason-optional" here, which is exactly the
/// regression this function must not reintroduce. A bare `[1, 3]` array has
/// no `reason` field by construction and is rejected on the same footing as
/// a wrapped object that omits `reason`.
///
/// Any parse failure (missing `reason`, bare array, wrong key, malformed
/// JSON) is LOUD, never silent: a bounded-cardinality
/// `reason` label distinguishes the failure shape, plus the same
/// `rql.extraction.silent_drop_suspected` signal every other extraction
/// parser emits on a suspected drop. The caller still receives an empty list
/// — treated as "no contradictions detected" — so a malformed verdict never
/// panics the pipeline; the metric/log surface is what makes the drop
/// observable instead of invisible.
///
/// The `u32` values found are converted to `usize` for use as 1-based indices
/// into the caller's `index_map` slice.
pub(crate) fn parse_index_list(json: &str) -> Vec<usize> {
    let trimmed = json.trim();

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.extraction.parsers",
            parser = "contradiction_indices",
            len = trimmed.len(),
            raw_input = %trimmed,
            "parse_index_list raw input"
        );
    }

    match serde_json::from_str::<ContradictionVerdictWrapper>(trimmed) {
        Ok(wrapper) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            wrapper.indices.into_iter().map(|i| i as usize).collect()
        }
        Err(_) => {
            let reason = classify_index_list_parse_failure(trimmed);
            counter!("rql.extraction.json_parse_fail", "reason" => reason).increment(1);
            counter!("rql.extraction.silent_drop_suspected", "parser" => "contradiction_indices")
                .increment(1);
            tracing::warn!(
                parser = "contradiction_indices",
                reason,
                "kremory.extraction.json_parse_fail"
            );
            Vec::new()
        }
    }
}

/// Classify why [`parse_index_list`] failed to deserialise `s` into the full
/// [`ContradictionVerdictWrapper`], for a bounded-cardinality metric label
/// (never raw text in a label).
fn classify_index_list_parse_failure(s: &str) -> &'static str {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(serde_json::Value::Array(_)) => "bare_array",
        Ok(serde_json::Value::Object(map)) => {
            if !map.contains_key("indices") {
                "missing_indices"
            } else if !matches!(map.get("reason"), Some(serde_json::Value::String(_))) {
                "missing_reason"
            } else {
                "malformed_indices"
            }
        }
        _ => "malformed",
    }
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

/// Bundled parameters for [`TwoPoolDetector::detect`] — args-as-object to
/// keep the function under clippy's `too_many_arguments` threshold.
pub struct DetectParams<'a> {
    /// The newly extracted fact to check.
    pub new_fact: &'a ExtractedFact,
    /// Facts with the same subject + predicate (already fetched by caller).
    pub pool_a: &'a [Fact],
    /// Semantically similar facts (already fetched by caller).
    pub pool_b: &'a [Fact],
    /// When the new fact becomes valid.
    pub reference_time: &'a DateTime<Utc>,
}

/// Contradiction detector using two candidate pools and temporal overlap filtering.
pub(crate) struct TwoPoolDetector<L: ChatProvider> {
    llm: Arc<L>,
    /// Consumer-supplied model identifier. Set by the
    /// Engine via [`with_model`](Self::with_model) at construction — NOT read off
    /// `llm.model()`. Drives capability detection + metric labels for the
    /// contradiction-verdict call. `None`/empty → `PromptOnly`.
    model: Option<String>,
}

impl<L: ChatProvider> TwoPoolDetector<L> {
    pub(crate) fn new(llm: Arc<L>) -> Self {
        Self { llm, model: None }
    }

    /// Set the consumer-supplied model identifier. Chainable; the
    /// Engine calls this with `self.model.clone()` at construction.
    pub(crate) fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    /// Detect contradictions for a new fact against existing facts.
    ///
    /// pool_a: facts with same subject + predicate (already fetched by caller)
    /// pool_b: semantically similar facts (already fetched by caller)
    /// reference_time: when the new fact becomes valid
    pub async fn detect(&self, params: DetectParams<'_>) -> Result<ContradictionResult> {
        let DetectParams {
            new_fact,
            pool_a,
            pool_b,
            reference_time,
        } = params;
        // Filter to temporally overlapping facts only
        let overlapping_a: Vec<&Fact> = pool_a
            .iter()
            .filter(|f| {
                temporal_overlap(TemporalOverlapParams {
                    existing_start: &f.valid_from,
                    existing_end: f.valid_to.as_ref(),
                    new_start: reference_time,
                    new_end: None,
                })
            })
            .collect();

        let overlapping_b: Vec<&Fact> = pool_b
            .iter()
            .filter(|f| {
                temporal_overlap(TemporalOverlapParams {
                    existing_start: &f.valid_from,
                    existing_end: f.valid_to.as_ref(),
                    new_start: reference_time,
                    new_end: None,
                })
            })
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
        let contradiction_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &SCHEMA_CONTRADICTION_VERDICT,
            "ContradictionVerdict",
        )
        .messages(contradiction_msgs)
        .model(self.model.as_deref().unwrap_or(""))
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let response_text = serde_json::to_string(&contradiction_value).unwrap_or_default();

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

    /// Find a `{"indices": [<non-empty>], "reason": ...}` example in the prompt.
    ///
    /// Returns the example's text so the caller can assert on it. Deliberately
    /// matches the SHAPE (populated index list) rather than specific digits —
    /// see the caller for why. Char-boundary-safe: only ever slices at byte
    /// offsets returned by `find`, which are always boundaries.
    fn regex_lite_find_populated_indices_example(prompt: &str) -> Option<&str> {
        let mut from = 0usize;
        while let Some(rel) = prompt[from..].find(r#"{"indices": ["#) {
            let start = from + rel;
            let after_open = start + r#"{"indices": ["#.len();
            let close = after_open + prompt[after_open..].find(']')?;
            // Populated = at least one digit between the brackets.
            if prompt[after_open..close]
                .chars()
                .any(|c| c.is_ascii_digit())
            {
                let end = close + prompt[close..].find('}').map_or(1, |i| i + 1);
                return Some(&prompt[start..end.min(prompt.len())]);
            }
            from = after_open;
        }
        None
    }

    fn make_extracted(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
        ExtractedFact {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            is_entity_ref: true,
            confidence: 1.0,
            valid_at: None,
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
        assert!(temporal_overlap(TemporalOverlapParams {
            existing_start: &t0,
            existing_end: None,
            new_start: &t1,
            new_end: None,
        }));
    }

    #[test]
    fn test_overlap_non_overlapping() {
        // [t0, t1) and [t2, t3) where t1 < t2 — gap between intervals
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(!temporal_overlap(TemporalOverlapParams {
            existing_start: &t0,
            existing_end: Some(&t1),
            new_start: &t2,
            new_end: Some(&t3),
        }));
    }

    #[test]
    fn test_overlap_adjacent() {
        // [t0, t1) and [t1, t2) — share a boundary point, half-open so they don't overlap
        let t0 = dt(-10);
        let t1 = dt(-5);
        let t2 = dt(0);
        assert!(!temporal_overlap(TemporalOverlapParams {
            existing_start: &t0,
            existing_end: Some(&t1),
            new_start: &t1,
            new_end: Some(&t2),
        }));
    }

    #[test]
    fn test_overlap_contained() {
        // [t0, t3) contains [t1, t2) — overlap
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(temporal_overlap(TemporalOverlapParams {
            existing_start: &t0,
            existing_end: Some(&t3),
            new_start: &t1,
            new_end: Some(&t2),
        }));
    }

    #[test]
    fn test_overlap_partial() {
        // [t0, t2) and [t1, t3) where t0 < t1 < t2 < t3 — partial overlap
        let t0 = dt(-20);
        let t1 = dt(-10);
        let t2 = dt(-5);
        let t3 = dt(0);
        assert!(temporal_overlap(TemporalOverlapParams {
            existing_start: &t0,
            existing_end: Some(&t2),
            new_start: &t1,
            new_end: Some(&t3),
        }));
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

    #[test]
    fn test_dual_list_prompt_requires_reason_in_output_instruction() {
        // Producer/consumer contract regression guard: `parse_index_list`
        // deserialises into `ContradictionVerdictWrapper` whose `reason`
        // field carries NO `#[serde(default)]` — a verdict omitting `reason`
        // is a parse FAILURE, not a degraded-but-accepted result. If the prompt
        // ever again asks only for `{"indices": [...]}` without `reason`,
        // a compliant model will omit the field and the parser will reject
        // ~every verdict — exactly the drop measured on the live conv0 run
        // (243/344 contradiction verdicts silently failing to parse). This
        // test pins the prompt's output-format instruction to always name
        // `reason` as part of the required output shape, so the prompt can
        // never again drift out of sync with the parser's mandatory field.
        let new_fact = make_extracted("alice", "works_at", "newco");
        let now = Utc::now();
        let fact_a = make_fact(1, "alice", "works_at", Some("acme"), None, now, None);

        let (prompt, _index_map) = build_dual_list_prompt(&new_fact, &[fact_a], &[]);

        assert!(
            prompt.contains("\"reason\""),
            "output-format instruction must require a `reason` field \
             (matching ContradictionVerdictWrapper's mandatory `reason`) — \
             prompt was: {prompt}"
        );
        // Both the populated example and the empty-indices example must
        // also demonstrate `reason` — a model that only sees `reason` in
        // one example may omit it in the other case.
        // Asserts the INVARIANT ("a populated-indices example carries reason"),
        // not the literal digits the prompt happens to use — a prior prompt
        // rewording changed its example to `[1]`, a single replacement, which
        // is what that example now demonstrates. Pinning the exact digits made
        // this test fail on a legitimate rewording while protecting nothing
        // extra: the property that matters is that BOTH the populated and the
        // empty case show `reason`, which is checked here at full strength.
        let populated_example = regex_lite_find_populated_indices_example(&prompt);
        assert!(
            populated_example.is_some_and(|e| e.contains("\"reason\"")),
            "populated-indices example must include reason — prompt was: {prompt}"
        );
        assert!(
            prompt.contains(r#"{"indices": [], "reason":"#),
            "empty-indices example must include reason — prompt was: {prompt}"
        );
    }

    // ── Index Parser ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_index_list_wrapped_valid() {
        // New contract: LLM must return wrapped form {"indices": [1, 3], "reason": "..."}.
        // `reason` is REQUIRED (no serde-default); missing reason = parse
        // error = empty fallback.
        assert_eq!(
            parse_index_list(r#"{"indices": [1, 3], "reason": "overlap on object"}"#),
            vec![1usize, 3]
        );
    }

    #[test]
    fn test_parse_index_list_wrapped_empty() {
        // Genuinely empty indices, `reason` present — a valid verdict of "no
        // contradictions", not a rejected parse.
        assert_eq!(
            parse_index_list(r#"{"indices": [], "reason": "no overlap"}"#),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_parse_index_list_bare_array_rejected_missing_reason() {
        // Parse-layer hardening regression fix: a bare `[1, 3]` array has NO
        // `reason` field by construction, so it can never satisfy the
        // `ContradictionVerdictWrapper` contract (`reason` is required, no
        // `#[serde(default)]`). `parse_index_list` MUST reject it — same
        // footing as any other reason-less verdict — rather than silently
        // accepting it via a shape-tolerant fallback (the original regression
        // routed through the generic `parse_items` helper, which only checks
        // `indices` and ignores `reason` entirely).
        assert_eq!(parse_index_list("[1, 3]"), Vec::<usize>::new());
    }

    #[test]
    fn test_parse_index_list_wrapped_missing_reason_rejected() {
        // A wrapped object with `indices` but no `reason` must be rejected —
        // missing `reason` is a parse failure, not a silently-accepted
        // degraded result.
        assert_eq!(
            parse_index_list(r#"{"indices": [1, 3]}"#),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_parse_index_list_malformed() {
        assert_eq!(parse_index_list("garbage"), Vec::<usize>::new());
    }

    #[test]
    fn parse_index_list_rejection_emits_loud_observability() {
        // This fix must not silently drop a reason-less verdict. Verify
        // BOTH the missing-reason and bare-array rejection paths increment
        // the loud-failure counters
        // (`json_parse_fail` + `silent_drop_suspected`), using a local
        // (non-global) `DebuggingRecorder` — same pattern as
        // `tests/b1_observability.rs`.
        use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

        for input in [r#"{"indices": [1, 3]}"#, "[1, 3]"] {
            let recorder = DebuggingRecorder::new();
            let snapshotter: Snapshotter = recorder.snapshotter();

            let result = metrics::with_local_recorder(&recorder, || parse_index_list(input));
            assert!(result.is_empty(), "rejected verdict must yield empty list");

            let names: Vec<String> = snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .map(|(key, _, _, _)| key.key().name().to_string())
                .collect();

            assert!(
                names.iter().any(|n| n == "rql.extraction.json_parse_fail"),
                "input {input:?} must increment json_parse_fail — got {names:?}"
            );
            assert!(
                names
                    .iter()
                    .any(|n| n == "rql.extraction.silent_drop_suspected"),
                "input {input:?} must increment silent_drop_suspected — got {names:?}"
            );
        }
    }

    // ── L1: Adversarial parse_index_list — "valid but useless" inputs ─────────
    //
    // These tests mirror the L1 philosophy for parse_json_lenient: the parser
    // must handle adversarial LLM outputs gracefully.  parse_index_list is used
    // in the contradiction-detection path; a bad parse here silently drops
    // contradiction signals.  The adversarial cases below pin the safe-fallback
    // contract (return empty on anything unexpected).

    #[test]
    fn parse_index_list_wrapped_null_indices_returns_empty() {
        // LLM returns {"indices": null} — null is not an array, must return empty.
        let result = parse_index_list(r#"{"indices": null}"#);
        assert!(
            result.is_empty(),
            "null indices must produce empty list — got {result:?}"
        );
    }

    #[test]
    fn parse_index_list_empty_json_object_returns_empty() {
        // LLM returns {} with no indices field.
        let result = parse_index_list("{}");
        assert!(
            result.is_empty(),
            "missing indices field must produce empty list — got {result:?}"
        );
    }

    #[test]
    fn parse_index_list_value_zero_in_array_is_preserved() {
        // Index value 0 is not a valid 1-based contradiction index, but the parser
        // must NOT discard it — the caller ignores out-of-range indices when it
        // walks index_map, so the parser's job is faithful extraction only.
        // {"indices": [0], "reason": "..."} → vec![0]
        // T1.7 (sprint plan v0-2-0-phase-b-prep): reason is REQUIRED per llm-output-parse-loudly.
        let result = parse_index_list(r#"{"indices": [0], "reason": "edge case"}"#);
        assert_eq!(
            result,
            vec![0usize],
            "zero index value must be preserved as-is"
        );
    }

    #[test]
    fn parse_index_list_empty_array_returns_empty() {
        // {"indices": [], "reason": "..."} is a valid wrapped form (reason
        // present) but contains no indices. Parser must return empty vec —
        // not panic, not error.
        let result = parse_index_list(r#"{"indices": [], "reason": "nothing to flag"}"#);
        assert!(
            result.is_empty(),
            "empty indices array must return empty vec — got {result:?}"
        );
    }

    #[test]
    fn parse_index_list_large_index_is_preserved() {
        // A very large index (9999) is out of range for any real index_map,
        // but the parser must not truncate or reject it — the caller handles bounds.
        // T1.7 (sprint plan v0-2-0-phase-b-prep): reason is REQUIRED per llm-output-parse-loudly.
        let result = parse_index_list(r#"{"indices": [1, 9999], "reason": "bounds preservation"}"#);
        assert_eq!(result, vec![1usize, 9999]);
    }

    #[test]
    fn parse_index_list_nested_object_returns_empty() {
        // "valid but useless" — structurally valid JSON but wrong schema.
        // parse_index_list must return empty (safe fallback), not panic.
        let result = parse_index_list(r#"{"result":{"contradictions":[1,2]}}"#);
        assert!(
            result.is_empty(),
            "wrong schema must produce empty list — got {result:?}"
        );
    }

    #[test]
    fn parse_index_list_prose_wrapped_in_json_returns_empty() {
        // LLM wraps a text answer instead of returning indices.
        let result = parse_index_list(r#"{"indices": "no contradictions found"}"#);
        assert!(
            result.is_empty(),
            "string value for indices must produce empty list — got {result:?}"
        );
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

        let result = block_on(detector.detect(DetectParams {
            new_fact: &new_fact,
            pool_a: &[fact],
            pool_b: &[],
            reference_time: &reference,
        }))
        .unwrap();
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

        let result = block_on(detector.detect(DetectParams {
            new_fact: &new_fact,
            pool_a: &[fact],
            pool_b: &[],
            reference_time: &reference,
        }))
        .unwrap();
        assert!(result.contradictions.is_empty());
        assert_eq!(result.duplicates, vec![42i64]);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_contradiction_via_mock_llm() {
        // Pool A has one fact that overlaps temporally and has a different object.
        // MockChatProvider returns the wrapped form {"indices":[1],"reason":"..."}
        // for prompts containing "works_at" — `reason` is required; a
        // bare-array form "[1]" is REJECTED (no `reason` field possible) — see
        // `test_parse_index_list_bare_array_rejected_missing_reason`.
        let t_start = dt(-10);
        let reference = dt(0);

        let fact = make_fact(99, "alice", "works_at", Some("acme"), None, t_start, None);
        let new_fact = make_extracted("alice", "works_at", "newco");

        // T1.7 (sprint plan v0-2-0-phase-b-prep): reason is REQUIRED on ContradictionVerdictWrapper
        // per `llm-output-parse-loudly`; mock must emit it to exercise the live deserializer path.
        let mut responses = HashMap::new();
        responses.insert(
            "works_at".to_string(),
            r#"{"indices":[1], "reason":"alice cannot work at two orgs simultaneously"}"#
                .to_string(),
        );
        let client = Arc::new(MockChatProvider::new(responses));
        let detector = TwoPoolDetector::new(client);

        let result = block_on(detector.detect(DetectParams {
            new_fact: &new_fact,
            pool_a: &[fact],
            pool_b: &[],
            reference_time: &reference,
        }))
        .unwrap();

        // Index 1 maps to fact id 99
        assert_eq!(result.contradictions, vec![99i64]);
        assert!(result.duplicates.is_empty());
        assert!(!result.is_consistent);
    }
}

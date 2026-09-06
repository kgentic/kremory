//! ADR-076 (TD-127) Pass 2: batched entity resolution.
//!
//! Collapses the O(ambiguous × candidates) pairwise `ResolutionVerdict` LLM
//! fan-out (`resolver.rs` Tier 3, called once per `(extracted, candidate)`
//! pair) into one structured-output call per window over the AMBIGUOUS
//! remainder — entities that survived ADR-075 candidate blocking but were
//! NOT resolved by the cheap deterministic tiers (`CascadeResolver::
//! resolve_deterministic`, Pass 1).
//!
//! Adapted from Graphiti's `nodes()` prompt (`graphiti_core/prompts/
//! dedupe_nodes.py`) — NOT copied verbatim: no episode-context injection
//! (ADR-076 RISK-005 — candidates carry name + type only, parity with the
//! pairwise `resolve()` Tier-3 prompt), and the map-back guards below are
//! kremory-specific hardening the Graphiti reference does not need (its
//! `NodeResolutions` response is trusted as-is; ours is not — see
//! `load-bearing-invariants-at-emit-not-prompt`).
//!
//! Every map-back guard rejects to conservative-NEW, never to a merge — a
//! malformed/adversarial response can only ever UNDER-merge (which `dream()`
//! fixes), never over-merge. See ADR-076 §Decision Pass 2 for the guard
//! order (reject-before-accept) and rationale for each.

use std::collections::{HashMap, HashSet};

use metrics::{counter, histogram};

use crate::core::extraction::schemas::{BatchedNodeResolutions, SCHEMA_BATCHED_RESOLUTION};
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::intelligence::ExtractedEntity;
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::core::resolver::{entity_name, normalize_name};
use crate::core::schema::Entity;

/// One entity from the ambiguous worklist paired with its global index (the
/// position in the ingest's full `all_entities` list — NOT the window-local
/// `id` presented to the LLM) and its own ADR-075 candidate block.
///
/// `resolve_batched`'s caller (`ingest_with.rs` Pass 1) builds this list;
/// `resolve_batched` windows it and assigns window-local `id`s 0..K-1 for
/// the prompt. The own-block `Vec<&Entity>` is retained per-entity (not
/// flattened into the shared pool) so the own-block map-back guard
/// (ADR-076 RISK-002) can check candidate reachability per pairwise
/// semantics even though the prompt presents one flat shared pool.
pub(crate) type AmbiguousEntity<'a> = (usize, &'a ExtractedEntity, Vec<&'a Entity>);

/// ADR-076 Pass 2 entry point: resolve the ambiguous remainder in windows of
/// at most `max_window` entities, one batched structured-output call per
/// window. Returns `global_index -> existing_entity_id` for every entity
/// that received a CONFIDENT merge; an absent entry means NEW (either the
/// model said `-1`, or the response failed a map-back guard, or the call
/// itself failed — all degrade to conservative-NEW per RISK-001).
///
/// `llm` + `model` are threaded separately from a `CascadeResolver` (rather
/// than taking `&CascadeResolver<L>`) because Pass 2 makes zero use of the
/// deterministic tiers — those already ran in Pass 1 (the caller's
/// `resolve_deterministic` loop) before an entity is considered ambiguous
/// enough to reach here.
///
/// Args bundled per rust-conventions §too_many_arguments (clippy.toml
/// threshold 3) — `#[allow]` is banned in `src/`.
pub(crate) struct ResolveBatchedParams<'a, L: ChatProvider + ?Sized> {
    pub(crate) llm: &'a L,
    pub(crate) model: Option<&'a str>,
    pub(crate) ambiguous: &'a [AmbiguousEntity<'a>],
    pub(crate) max_window: usize,
}

pub(crate) async fn resolve_batched<L: ChatProvider + ?Sized>(
    params: ResolveBatchedParams<'_, L>,
) -> HashMap<usize, String> {
    let ResolveBatchedParams {
        llm,
        model,
        ambiguous,
        max_window,
    } = params;

    let mut resolved: HashMap<usize, String> = HashMap::new();
    if ambiguous.is_empty() {
        return resolved;
    }
    let max_window = max_window.max(1);

    for window in ambiguous.chunks(max_window) {
        let window_result = resolve_window(llm, model, window).await;
        resolved.extend(window_result);
    }

    resolved
}

/// Resolve a single window: build the shared candidate pool, prompt the LLM
/// once, and map the response back via [`map_back`]. Any call/parse failure
/// degrades the WHOLE window to conservative-NEW (empty map) rather than
/// propagating an error — see ADR-076 RISK-001.
async fn resolve_window<L: ChatProvider + ?Sized>(
    llm: &L,
    model: Option<&str>,
    window: &[AmbiguousEntity<'_>],
) -> HashMap<usize, String> {
    let window_len = window.len();
    histogram!("kremory.resolution.window_size").record(window_len as f64);

    // Shared candidate pool: stable-sorted union (by existing-entity id) of
    // every entity's own ADR-075 block. Determinism here is load-bearing
    // (VCR key + A/B reproducibility, ADR-076 ASMP-002) — sorting by id
    // fixes the candidate_id assignment regardless of block-discovery order.
    let mut pool: Vec<&Entity> = Vec::new();
    let mut seen_pool: HashSet<&str> = HashSet::new();
    for (_, _, candidates) in window {
        for &c in candidates {
            if seen_pool.insert(c.id.as_str()) {
                pool.push(c);
            }
        }
    }
    pool.sort_by(|a, b| a.id.cmp(&b.id));
    histogram!("kremory.resolution.pool_size").record(pool.len() as f64);

    let prompt = build_prompt(window, &pool);
    let messages = vec![
        chat_msg_system(
            "You are an entity deduplication assistant. NEVER fabricate entity names \
             or mark distinct entities as duplicates.",
        ),
        chat_msg_user(prompt.as_str()),
    ];

    let call_result =
        StructuredCallBuilder::new(llm, &SCHEMA_BATCHED_RESOLUTION, "BatchedResolution")
            .messages(messages)
            .model(model.unwrap_or(""))
            .call()
            .await;

    counter!("kremory.resolution.batched_calls_total").increment(1);

    let value = match call_result {
        Ok(v) => v,
        Err(_) => {
            // RISK-001: ladder-exhausted / transport error — NEVER propagate.
            // The whole window degrades to conservative-NEW.
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "parse_failure"
            )
            .increment(window_len as u64);
            return HashMap::new();
        }
    };

    let resolutions: BatchedNodeResolutions = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(_) => {
            // A malformed INNER row (missing required field) fails the whole
            // Vec's deserialisation per `llm-output-parse-loudly` — treat the
            // same as a call failure: window-wide conservative-NEW.
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "parse_failure"
            )
            .increment(window_len as u64);
            return HashMap::new();
        }
    };

    map_back(&resolutions, window, &pool)
}

/// Build the batched-resolution prompt: the window's ambiguous entities
/// (window-local `id` 0..K-1) + the shared candidate pool (`candidate_id`
/// 0..M-1). Name + type ONLY — no context snippet (ADR-076 RISK-005). The
/// model is instructed to echo each entity's `name` verbatim (the
/// reject-only misindex checksum consumed by [`map_back`]).
fn build_prompt(window: &[AmbiguousEntity<'_>], pool: &[&Entity]) -> String {
    let k = window.len();

    // Real JSON serialization (not Debug-formatting) so entity names with
    // quotes/unicode/control characters round-trip safely into the prompt.
    let entities_json: Vec<serde_json::Value> = window
        .iter()
        .enumerate()
        .map(|(local_id, (_, entity, _))| {
            serde_json::json!({"id": local_id, "name": entity.name, "type": entity.label})
        })
        .collect();
    let pool_json: Vec<serde_json::Value> = pool
        .iter()
        .enumerate()
        .map(|(cid, cand)| {
            serde_json::json!({
                "candidate_id": cid,
                "name": entity_name(cand),
                "type": cand.label,
            })
        })
        .collect();
    let entities_str = serde_json::to_string(&entities_json).unwrap_or_default();
    let pool_str = serde_json::to_string(&pool_json).unwrap_or_default();

    format!(
        r#"<ENTITIES>
{entities_str}
</ENTITIES>

<EXISTING ENTITIES>
{pool_str}
</EXISTING ENTITIES>

Each of the above ENTITIES was extracted from the same source text. For each
entity, determine if it is a duplicate of any EXISTING ENTITY. Entities
should only be considered duplicates if they refer to the *same real-world
object or concept*.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains {k} entities with IDs 0 through {last}.
Your response MUST include EXACTLY {k} resolutions with IDs 0 through {last}. Do not skip or add IDs.

For every entity, provide:
- `id`: integer id from ENTITIES
- `name`: copy the entity's `name` EXACTLY as given — do not rephrase or correct it
- `duplicate_candidate_id`: the `candidate_id` of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate

Output ONLY the JSON object matching the schema — no reasoning, no explanation, no markdown fences, no prose before or after. Start your response with `{{` and end it with `}}`.

<EXAMPLE>
ENTITY: {{"id": 0, "name": "NYC", "type": "Location"}}
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "New York City", "type": "Location"}}, {{"candidate_id": 1, "name": "New York Knicks", "type": "Organization"}}]
Result: {{"id": 0, "name": "NYC", "duplicate_candidate_id": 0}} (same location, abbreviated name)

ENTITY: {{"id": 1, "name": "Java", "type": "Technology"}}
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Java", "type": "Location"}}]
Result: {{"id": 1, "name": "Java", "duplicate_candidate_id": -1}} (same name but distinct real-world things — an island, not the programming language)
</EXAMPLE>
"#,
        last = k.saturating_sub(1),
    )
}

/// Map a batched LLM response back to `global_index -> existing_entity_id`,
/// applying every guard in ADR-076 §Decision Pass 2's order (reject BEFORE
/// accept). Pure + sync so it is directly unit-testable with canned input —
/// see the `#[cfg(test)]` module below.
///
/// `worklist` is the WINDOW (not the full ambiguous list) — `worklist[i].0`
/// is the global index the caller keys its result map on; `i` itself is the
/// window-local id the LLM was shown (0..K-1).
fn map_back(
    resolutions: &BatchedNodeResolutions,
    worklist: &[AmbiguousEntity<'_>],
    pool: &[&Entity],
) -> HashMap<usize, String> {
    let k = worklist.len();
    let m = pool.len();
    let mut resolved: HashMap<usize, String> = HashMap::new();
    let mut seen_ids: HashSet<u32> = HashSet::new();
    let mut covered: HashSet<u32> = HashSet::new();

    for row in &resolutions.entity_resolutions {
        // (1) Duplicate-id guard (RISK-003) — first row for a given `id`
        // wins, regardless of whether it (or the later dupe) is in range.
        if !seen_ids.insert(row.id) {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "duplicate_row"
            )
            .increment(1);
            continue;
        }

        // (2) Range guard on `id`.
        if row.id as usize >= k {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "out_of_range_id"
            )
            .increment(1);
            continue;
        }
        covered.insert(row.id);
        let (global_idx, entity, own_block) = &worklist[row.id as usize];

        // (3) Explicit "no duplicate" — a genuine NEW decision, not a guard
        // rejection; no counter (this is the model doing its job).
        if row.duplicate_candidate_id == -1 {
            continue;
        }

        // (4) Range guard on `duplicate_candidate_id` (anything outside
        // {-1} ∪ [0,M)). Short-circuits before the `as usize` cast so a
        // negative-but-not-`-1` value never wraps into a false in-range hit.
        if row.duplicate_candidate_id < -1 || row.duplicate_candidate_id as usize >= m {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "out_of_range_cid"
            )
            .increment(1);
            continue;
        }

        // (5) Name-echo checksum (RISK-004) — reject-only misindex signal.
        if normalize_name(&row.name) != normalize_name(&entity.name) {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "name_mismatch"
            )
            .increment(1);
            continue;
        }

        // (6) Own-block guard (RISK-002) — the primary over-merge defence.
        // An entity may only resolve to a candidate that was in ITS OWN
        // ADR-075 block, never one that entered the shared pool only via a
        // sibling's ANN search.
        let cand = pool[row.duplicate_candidate_id as usize];
        if !own_block.iter().any(|c| c.id == cand.id) {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "cross_assignment"
            )
            .increment(1);
            continue;
        }

        // (7) Accept — all guards passed.
        resolved.insert(*global_idx, cand.id.clone());
    }

    // (8) Missing row — any ambiguous entity the response never mentioned at
    // all degrades to conservative-NEW (parity with pairwise's "uncertain ->
    // Different").
    for local_id in 0..k as u32 {
        if !covered.contains(&local_id) {
            counter!(
                "kremory.resolution.conservative_new_total",
                "reason" => "missing_row"
            )
            .increment(1);
        }
    }

    resolved
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extraction::schemas::BatchedNodeResolution;
    use chrono::Utc;

    fn make_entity(id: &str, label: &str, name: &str) -> Entity {
        Entity {
            id: id.to_string(),
            label: label.to_string(),
            entity_type_id: 0,
            properties: serde_json::json!({"name": name}),
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: None,
            access_count: 0,
        }
    }

    fn make_extracted(label: &str, name: &str) -> ExtractedEntity {
        ExtractedEntity {
            label: label.to_string(),
            name: name.to_string(),
            properties: serde_json::json!({"name": name}),
        }
    }

    fn resolutions(rows: Vec<(u32, &str, i32)>) -> BatchedNodeResolutions {
        BatchedNodeResolutions {
            entity_resolutions: rows
                .into_iter()
                .map(|(id, name, cid)| BatchedNodeResolution {
                    id,
                    name: name.to_string(),
                    duplicate_candidate_id: cid,
                })
                .collect(),
        }
    }

    // (a) out-of-range `id` is skipped — no entry, no panic.
    #[test]
    fn map_back_out_of_range_id_skipped() {
        let extracted = make_extracted("Person", "Alice");
        let existing = make_entity("alice-1", "Person", "Alice Johnson");
        let worklist: Vec<AmbiguousEntity> = vec![(0, &extracted, vec![&existing])];
        let pool: Vec<&Entity> = vec![&existing];

        // id=5 is out of range for a 1-entity window (valid range 0..1).
        let resp = resolutions(vec![(5, "Alice", 0)]);
        let out = map_back(&resp, &worklist, &pool);

        assert!(out.is_empty(), "out-of-range id must not produce a merge");
    }

    // (b) out-of-range `duplicate_candidate_id` -> NEW (no merge recorded).
    #[test]
    fn map_back_out_of_range_cid_is_new() {
        let extracted = make_extracted("Person", "Alice");
        let existing = make_entity("alice-1", "Person", "Alice Johnson");
        let worklist: Vec<AmbiguousEntity> = vec![(0, &extracted, vec![&existing])];
        let pool: Vec<&Entity> = vec![&existing];

        // pool has 1 candidate (valid cid range 0..1); cid=7 is out of range.
        let resp = resolutions(vec![(0, "Alice", 7)]);
        let out = map_back(&resp, &worklist, &pool);

        assert!(
            out.is_empty(),
            "out-of-range candidate_id must resolve to NEW"
        );
    }

    // (c) duplicate `id` rows -> first row wins.
    #[test]
    fn map_back_duplicate_id_first_row_wins() {
        let extracted = make_extracted("Person", "Alice");
        let e0 = make_entity("alice-1", "Person", "Alice Johnson");
        let e1 = make_entity("alice-2", "Person", "Alice Smith");
        let worklist: Vec<AmbiguousEntity> = vec![(0, &extracted, vec![&e0, &e1])];
        let mut pool: Vec<&Entity> = vec![&e0, &e1];
        pool.sort_by(|a, b| a.id.cmp(&b.id)); // deterministic pool order

        // Two rows for id=0: first picks whichever candidate_id maps to
        // pool[0], second (duplicate) must be ignored even though it names
        // a different, otherwise-valid candidate.
        let cid_first = pool.iter().position(|e| e.id == "alice-1").unwrap() as i32;
        let cid_second = pool.iter().position(|e| e.id == "alice-2").unwrap() as i32;
        let resp = resolutions(vec![(0, "Alice", cid_first), (0, "Alice", cid_second)]);
        let out = map_back(&resp, &worklist, &pool);

        assert_eq!(out.get(&0), Some(&"alice-1".to_string()));
    }

    // (d) name-echo mismatch -> NEW.
    #[test]
    fn map_back_name_mismatch_is_new() {
        let extracted = make_extracted("Person", "Alice");
        let existing = make_entity("alice-1", "Person", "Alice Johnson");
        let worklist: Vec<AmbiguousEntity> = vec![(0, &extracted, vec![&existing])];
        let pool: Vec<&Entity> = vec![&existing];

        // Model echoed the wrong name for id=0 — misindex signal.
        let resp = resolutions(vec![(0, "Bob", 0)]);
        let out = map_back(&resp, &worklist, &pool);

        assert!(out.is_empty(), "name-echo mismatch must resolve to NEW");
    }

    // (e) cross-assignment: candidate belongs to a SIBLING's block, not this
    // entity's own block -> NEW.
    #[test]
    fn map_back_cross_assignment_is_new() {
        let extracted_a = make_extracted("Person", "Alice");
        let extracted_b = make_extracted("Location", "Java");
        // Alice's own block is [alice-1]; java-island only entered the shared
        // pool via Java's own block, never Alice's.
        let alice_existing = make_entity("alice-1", "Person", "Alice Johnson");
        let java_existing = make_entity("java-island", "Location", "Java");

        let worklist: Vec<AmbiguousEntity> = vec![
            (0, &extracted_a, vec![&alice_existing]),
            (1, &extracted_b, vec![&java_existing]),
        ];
        let mut pool: Vec<&Entity> = vec![&alice_existing, &java_existing];
        pool.sort_by(|a, b| a.id.cmp(&b.id));

        let java_cid = pool.iter().position(|e| e.id == "java-island").unwrap() as i32;
        // Model (incorrectly) points Alice (id=0) at java-island's candidate_id.
        let resp = resolutions(vec![(0, "Alice", java_cid)]);
        let out = map_back(&resp, &worklist, &pool);

        assert!(
            out.is_empty(),
            "cross-block assignment must be rejected to NEW, never accepted"
        );
    }

    // (f) missing row -> NEW (no entry for that entity).
    #[test]
    fn map_back_missing_row_is_new() {
        let extracted_a = make_extracted("Person", "Alice");
        let extracted_b = make_extracted("Person", "Bob");
        let existing = make_entity("alice-1", "Person", "Alice Johnson");
        let worklist: Vec<AmbiguousEntity> = vec![
            (0, &extracted_a, vec![&existing]),
            (1, &extracted_b, vec![]),
        ];
        let pool: Vec<&Entity> = vec![&existing];

        // Response only covers id=0; id=1 (Bob) has no row at all.
        let resp = resolutions(vec![(0, "Alice", 0)]);
        let out = map_back(&resp, &worklist, &pool);

        assert_eq!(out.get(&0), Some(&"alice-1".to_string()));
        assert_eq!(out.get(&1), None);
    }

    // (g) a fully-valid row produces the correct merge.
    #[test]
    fn map_back_valid_row_merges() {
        let extracted = make_extracted("Person", "Alice");
        let existing = make_entity("alice-1", "Person", "Alice Johnson");
        let worklist: Vec<AmbiguousEntity> = vec![(0, &extracted, vec![&existing])];
        let pool: Vec<&Entity> = vec![&existing];

        let resp = resolutions(vec![(0, "Alice", 0)]);
        let out = map_back(&resp, &worklist, &pool);

        assert_eq!(out.len(), 1);
        assert_eq!(out.get(&0), Some(&"alice-1".to_string()));
    }

    // Determinism: identical inputs must yield identical candidate_id
    // assignment / output across repeated invocations (ADR-076 ASMP-002).
    #[test]
    fn map_back_deterministic_across_runs() {
        let extracted_a = make_extracted("Person", "Alice");
        let extracted_b = make_extracted("Location", "Java");
        let e0 = make_entity("zeta", "Person", "Alice Johnson");
        let e1 = make_entity("alpha", "Location", "Javanese Island");

        let worklist: Vec<AmbiguousEntity> =
            vec![(0, &extracted_a, vec![&e0]), (1, &extracted_b, vec![&e1])];
        let mut pool: Vec<&Entity> = vec![&e0, &e1];
        pool.sort_by(|a, b| a.id.cmp(&b.id));
        // Sorted by id: "alpha" (e1) before "zeta" (e0).
        assert_eq!(pool[0].id, "alpha");
        assert_eq!(pool[1].id, "zeta");

        let zeta_cid = pool.iter().position(|e| e.id == "zeta").unwrap() as i32;
        let resp = resolutions(vec![(0, "Alice", zeta_cid)]);

        let out1 = map_back(&resp, &worklist, &pool);
        let out2 = map_back(&resp, &worklist, &pool);
        assert_eq!(out1, out2);
        assert_eq!(out1.get(&0), Some(&"zeta".to_string()));
    }

    // Windowing: >max_window ambiguous entities split into ceil(K/max) calls.
    // (Structural check on chunking only — no LLM in this test.)
    #[test]
    fn windowing_splits_by_max_window() {
        let e = make_entity("e1", "Person", "Someone");
        let extracted: Vec<ExtractedEntity> = (0..5)
            .map(|i| make_extracted("Person", &format!("Person{i}")))
            .collect();
        let ambiguous: Vec<AmbiguousEntity> = extracted
            .iter()
            .enumerate()
            .map(|(i, ex)| (i, ex, vec![&e]))
            .collect();

        let chunks: Vec<&[AmbiguousEntity]> = ambiguous.chunks(2).collect();
        assert_eq!(chunks.len(), 3, "5 entities / max_window=2 -> 3 windows");
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 1);
    }
}

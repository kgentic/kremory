//! JSON-to-domain parsers for extraction output.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).
//!
//! # Shape-tolerant parsing (2026-07-20 parse-layer hardening)
//!
//! Every parser in this module that deserialises LLM-emitted item lists routes
//! through [`parse_items`] — a single shared helper that accepts BOTH the
//! wrapped-object shape strict schemas advertise (`{"items": [...]}`) AND the
//! bare-array shape lenient providers still emit (`[...]`). See
//! `.ai-docs/specs/extraction-parse-layer-shape-robustness-observability-2026-07-20.md`
//! for the full design + Vera Cycle-1 review that hardened it further
//! (SCOPE-003 wrapper discipline, ASMP-001 ordering invariant, RISK-001
//! partial-drop visibility, DENT-001 metric-family reuse).

use metrics::counter;
use tracing;

use super::json_repair::repair_to_array;
use super::models::{RawEntityIntegerId, RawEntitySimple, RawFact};
use crate::core::intelligence::{ExtractedEntity, ExtractedFact};

// ─── Shared shape-tolerant parse helper (C1) ─────────────────────────────────

/// Which shape matched during [`parse_items`] parsing.
///
/// Exposed so callers that need per-path success tracking (Vera DENT-001 —
/// post-repair success is suspicious and must be tracked separately from a
/// clean wrapped/bare-array parse, per `observability-first-class` cardinal
/// failure mode #9) can label their own metrics accordingly. Callers that
/// don't need the breakdown may ignore the returned value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParsePath {
    /// The wrapped-object shape (`{wrapper_key: [...]}`) matched on the
    /// original, unrepaired string.
    Wrapped,
    /// The bare-array shape (`[...]`) matched on the original, unrepaired
    /// string.
    BareArray,
    /// Either shape matched only after [`repair_to_array`] ran — a suspicious
    /// success (Rule 20) regardless of which of the two shapes it was.
    PostRepair,
    /// Neither shape matched at all (`deserialize_ok == false`).
    None,
}

/// Shape-tolerant parse of an LLM-emitted item list into `Vec<T>`.
///
/// Accepts three input shapes, tried in this exact order:
///   1. **Wrapped object**: `{wrapper_key: [...]}` — extracted via
///      [`serde_json::Value::get`] on the parsed `Value`, NEVER via a
///      `#[serde(default)]` wrapper struct. A default-wrapper struct cannot
///      distinguish a wrong/missing top-level key from a genuine empty array
///      (Vera SCOPE-003) — `Value::get` can, because it only matches on the
///      exact key.
///   2. **Bare array**: `[...]` — lenient providers (e.g. Ollama) still emit
///      this shape even when the schema advertises a wrapper.
///   3. **Post-repair retry of (1) and (2)** on the output of
///      [`repair_to_array`] (markdown fences, single-object, trailing commas).
///
/// # Ordering invariant (Vera ASMP-001)
/// The wrapper-key extraction (step 1) MUST run on the **original** string
/// before any repair pass. `repair_to_array` wraps a bare `{...}` object in an
/// array (`{...}` → `[{...}]`) — if repair ran first, a genuine
/// `{"items": []}` would become `[{"items": []}]`, and the wrapper-key lookup
/// would then run against an *array* (which has no string keys), silently
/// losing the "this was actually a valid empty wrapper" signal. Trying the
/// original string first means a genuine empty wrapper is recognised
/// immediately, with zero repair passes.
///
/// # Return value
/// `(items, raw_count, deserialize_ok, path)`:
/// - `raw_count` — length of the array actually found by whichever shape
///   succeeded. A genuine empty payload (`[]` or `{wrapper_key: []}`) yields
///   `raw_count == 0` — this is what lets callers distinguish "nothing to
///   extract" from "something was dropped" (`raw_count > 0`, `emitted == 0`).
/// - `deserialize_ok` — `true` iff ANY of the three attempts structurally
///   parsed a JSON array (even an empty one). `false` means nothing parsed at
///   all (wrong top-level key with no matching array shape, malformed JSON,
///   etc.) — this is the signal that distinguishes total parse failure from a
///   genuine empty result, which `raw_count` alone cannot do (both are `0`).
/// - `path` — [`ParsePath`] naming WHICH of the three shapes matched, so
///   callers that track a `path` metric label (e.g. `parse_entities_integer`)
///   can report the real shape instead of a hardcoded constant.
pub(crate) fn parse_items<T: serde::de::DeserializeOwned>(
    json: &str,
    wrapper_key: &str,
) -> (Vec<T>, usize, bool, ParsePath) {
    let trimmed = json.trim();

    // Attempt on the ORIGINAL string first — ASMP-001 ordering invariant.
    if let Some((items, count, path)) = try_parse_shape::<T>(trimmed, wrapper_key) {
        return (items, count, true, path);
    }

    // Fall back to repair, then retry both shapes on the repaired string.
    // Whichever shape matches here is reported as `PostRepair` — the
    // suspicious-success signal is "did repair have to run", not which of the
    // two shapes it landed on afterwards.
    let repaired = repair_to_array(trimmed);
    if let Some((items, count, _shape)) = try_parse_shape::<T>(&repaired, wrapper_key) {
        return (items, count, true, ParsePath::PostRepair);
    }

    (Vec::new(), 0, false, ParsePath::None)
}

/// Try the wrapped-object shape then the bare-array shape against `s`, in
/// that order. Returns `None` if neither shape structurally parses — the
/// caller (`parse_items`) decides what to try next.
fn try_parse_shape<T: serde::de::DeserializeOwned>(
    s: &str,
    wrapper_key: &str,
) -> Option<(Vec<T>, usize, ParsePath)> {
    // Wrapped: {wrapper_key: [...]}. `Value::get` — never a
    // `#[serde(default)]` wrapper struct (Vera SCOPE-003): a default wrapper
    // can't distinguish a wrong/missing key from a genuine empty array.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(s) {
        if let Some(inner) = value.get(wrapper_key) {
            if let Ok(items) = serde_json::from_value::<Vec<T>>(inner.clone()) {
                let count = items.len();
                return Some((items, count, ParsePath::Wrapped));
            }
        }
    }

    // Bare array: [...].
    if let Ok(items) = serde_json::from_str::<Vec<T>>(s) {
        let count = items.len();
        return Some((items, count, ParsePath::BareArray));
    }

    None
}

// ─── Always-on yield observability (C2 / RISK-001) ───────────────────────────

/// Emit the always-on raw-vs-emitted yield metrics for a single parser call.
///
/// Extends the existing `rql.extraction.*` metric family (Vera DENT-001 — no
/// parallel namespace). `parser` MUST be a bounded label drawn from the small
/// fixed set of parser names in this crate — never raw text (Rule 19 /
/// cardinality safety).
///
/// - `rql.extraction.parse_raw_items{parser}` — items found before filtering.
/// - `rql.extraction.parse_emitted{parser}` — usable items after filtering.
/// - **Tripwire** (`raw_count > 0 && emitted == 0`): unconditional WARN +
///   `rql.extraction.silent_drop_suspected{parser}` counter — the parser
///   received items but produced none, which is the exact silent-drop shape
///   that caused the empty benchmark fact graph this spec fixes. A genuinely
///   empty extraction has `raw_count == 0`, so this never false-fires on a
///   fact-less episode.
/// - **Partial-drop** (Vera RISK-001, `emitted > 0 && emitted < raw_count`):
///   an INFO log — this repo has no CI/dashboards, so a lone counter would
///   never be seen; the log line is the operator-visible signal.
pub(crate) fn emit_parse_yield_metrics(parser: &'static str, raw_count: usize, emitted: usize) {
    counter!("rql.extraction.parse_raw_items", "parser" => parser).increment(raw_count as u64);
    counter!("rql.extraction.parse_emitted", "parser" => parser).increment(emitted as u64);

    if raw_count > 0 && emitted == 0 {
        tracing::warn!(
            parser,
            raw_count,
            "extraction parser received items but emitted none — possible schema/parser shape drift"
        );
        counter!("rql.extraction.silent_drop_suspected", "parser" => parser).increment(1);
    } else if emitted > 0 && emitted < raw_count {
        tracing::info!(
            parser,
            raw_count,
            emitted,
            "extraction parser dropped some items"
        );
    }
}

// ─── Relation name parser ─────────────────────────────────────────────────────

pub(crate) fn parse_relation_names(json: &str) -> anyhow::Result<Vec<String>> {
    let trimmed = json.trim();

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.extraction.parsers",
            len = trimmed.len(),
            raw_input = %trimmed,
            "parse_relation_names raw input"
        );
    }

    // T = serde_json::Value (not String): the schema (`RelTypeListWrapper`)
    // wraps in `{"items": [...]}`, but some providers emit non-string
    // elements (numbers, nested values) alongside valid names. Deserialising
    // to `Value` first — rather than `String` directly — lets `raw_count`
    // reflect the true array length while the filter below drops non-string
    // elements individually, preserving the pre-existing lenient behaviour
    // (a single bad element no longer fails the whole batch).
    let (raw, raw_count, deserialize_ok, _path): (Vec<serde_json::Value>, usize, bool, ParsePath) =
        parse_items(trimmed, "items");

    if deserialize_ok {
        counter!("rql.extraction.json_parse_ok", "parser" => "relation_names").increment(1);
    } else {
        counter!("rql.extraction.json_parse_fail", "parser" => "relation_names").increment(1);
        tracing::warn!(
            parser = "relation_names",
            "kremory.extraction.json_parse_fail"
        );
    }

    let mut dropped_non_string = 0u64;
    let names: Vec<String> = raw
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(s),
            other => {
                let as_str = other.as_str().map(|s| s.to_string());
                if as_str.is_none() {
                    dropped_non_string += 1;
                }
                as_str
            }
        })
        .collect();

    if dropped_non_string > 0 {
        counter!("rql.extraction.item_dropped", "parser" => "relation_names", "reason" => "non_string")
            .increment(dropped_non_string);
    }
    emit_parse_yield_metrics("relation_names", raw_count, names.len());

    Ok(names)
}

// ─── Entity parsers ───────────────────────────────────────────────────────────

/// Legacy string-label entity parser (pre-TD-013 L1 path).
///
/// Production extractors now use `parse_entities_integer`.  This function is
/// retained as a test fixture for the string-label parse path (used by
/// `test_parse_entities_*` tests).  Rustc dead-code analysis doesn't count
/// `#[cfg(test)]` callers from the production-code vantage point.
#[allow(dead_code)]
pub(crate) fn parse_entities(json: &str) -> anyhow::Result<Vec<ExtractedEntity>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    let raw: Vec<RawEntitySimple> = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok", "parser" => "entities_legacy").increment(1);
            v
        }
        Err(_) => {
            let repaired = repair_to_array(trimmed);
            match serde_json::from_str(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok", "parser" => "entities_legacy")
                        .increment(1);
                    v
                }
                Err(e) => {
                    counter!("rql.extraction.json_parse_fail", "parser" => "entities_legacy")
                        .increment(1);
                    tracing::warn!(error = %e, parser = "entities", "kremory.extraction.json_parse_fail");
                    return Ok(vec![]);
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter(|e| !e.name.is_empty() && !e.label.is_empty())
        .map(|e| {
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            ExtractedEntity {
                name: e.name,
                label: e.label,
                properties: serde_json::Value::Object(props),
            }
        })
        .collect())
}

/// Parse the integer-ID entity JSON emitted by the L1 extraction path (TD-013).
///
/// Accepts two input shapes:
/// 1. Wrapped: `{"entities": [{"name": "Alice", "entity_type_id": 1}, ...]}` — primary.
/// 2. Bare array: `[{"name": "Alice", "entity_type_id": 1}, ...]` — repair fallback.
/// 3. Single object: `{"name": "Alice", "entity_type_id": 1}` — wrapped via repair_to_array.
///
/// For each `RawEntityIntegerId`:
/// - Skips entries with empty `name`.
/// - Validates `entity_type_id` via `EntityTypeRegistry::validate_or_fallback`
///   (out-of-range or unknown ids → id=0 catch-all "Entity").
/// - Resolves the integer id to a label string via `EntityTypeRegistry::id_to_name`.
/// - Builds `ExtractedEntity { name, label, properties }` — downstream contract preserved.
///
/// Routes through the shared [`parse_items`] helper (Vera SCOPE-003): the
/// previous `EntityListIntegerWrapper` + `#[serde(default)]` deserialize was a
/// live wrong-key hazard — `{"result": [...]}` would silently deserialize to
/// an empty wrapper, indistinguishable from a genuine `{"entities": []}`.
pub(crate) fn parse_entities_integer(
    json: &str,
    registry: &crate::core::entity_types::EntityTypeRegistry,
) -> anyhow::Result<Vec<ExtractedEntity>> {
    let trimmed = json.trim();

    // KREMORY_DEBUG=1: emit raw stage1 LLM output for diagnosis (Rule 19).
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.extraction.parsers",
            len = trimmed.len(),
            raw_input = %trimmed,
            "parse_entities_integer raw input"
        );
    }

    let (raw, raw_count, deserialize_ok, parse_path): (Vec<RawEntityIntegerId>, usize, bool, ParsePath) =
        parse_items(trimmed, "entities");

    if deserialize_ok {
        // Path label restored (Vera DENT-001): a post-repair success is
        // suspicious and must stay distinguishable from a clean
        // wrapped/bare_array parse (Rule 20 — post-repair success is
        // suspicious, track separately). `parse_items` now reports which of
        // the three shapes actually matched via `ParsePath`.
        let path_label = match parse_path {
            ParsePath::Wrapped => "wrapped",
            ParsePath::BareArray => "bare_array",
            ParsePath::PostRepair => "post_repair",
            // Unreachable when `deserialize_ok` is true — `parse_items` only
            // returns `ParsePath::None` alongside `deserialize_ok == false`.
            ParsePath::None => "wrapped",
        };
        counter!("rql.extraction.json_parse_ok", "parser" => "entities_integer", "path" => path_label)
            .increment(1);
    } else {
        counter!("rql.extraction.json_parse_fail", "parser" => "entities_integer").increment(1);
        tracing::warn!(
            parser = "entities_integer",
            "kremory.extraction.json_parse_fail"
        );
    }

    let entities: Vec<ExtractedEntity> = raw
        .into_iter()
        .filter_map(|e| {
            if e.name.is_empty() {
                counter!("rql.extraction.entity_rejected", "reason" => "empty_name").increment(1);
                counter!("rql.extraction.item_dropped", "parser" => "entities_integer", "reason" => "empty_name")
                    .increment(1);
                return None;
            }
            // Shape-validate the name. Repair paths can splice JSON fragments
            // into the name field; reject any name containing JSON syntax
            // chars so garbage entities never reach persistence.
            if name_looks_like_json_fragment(&e.name) {
                counter!("rql.extraction.entity_rejected", "reason" => "name_json_fragment")
                    .increment(1);
                counter!("rql.extraction.item_dropped", "parser" => "entities_integer", "reason" => "name_json_fragment")
                    .increment(1);
                tracing::warn!(
                    raw_name = %e.name,
                    "kremory.extraction.entity_rejected.name_json_fragment"
                );
                return None;
            }
            let validated_id = registry.validate_or_fallback(e.entity_type_id);
            let label = registry.id_to_name(validated_id).to_string();
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            Some(ExtractedEntity {
                name: e.name,
                label,
                properties: serde_json::Value::Object(props),
            })
        })
        .collect();

    emit_parse_yield_metrics("entities_integer", raw_count, entities.len());

    Ok(entities)
}

/// Returns true if `name` contains characters that suggest it is a spliced JSON
/// fragment rather than a real entity name. Repair paths (`repair_to_array` +
/// `llm_json::repair_json`) can produce parseable output where a truncated
/// object's tail bleeds into the next entity's `name`. A real entity name will
/// never legitimately contain unescaped JSON structural characters.
pub(crate) fn name_looks_like_json_fragment(name: &str) -> bool {
    name.contains('"')
        || name.contains('\\')
        || name.contains('{')
        || name.contains('}')
        || name.contains("entity_type_id")
}

// ─── Fact parser ──────────────────────────────────────────────────────────────

pub(crate) fn parse_facts(json: &str) -> anyhow::Result<Vec<ExtractedFact>> {
    Ok(parse_facts_with_raw_count(json)?.0)
}

/// Same as [`parse_facts`] but also returns the raw item count found before
/// filtering, so stage-level callers (e.g. `default_extractor.rs`) can emit
/// the raw-vs-persisted gap metric (`rql.extraction.raw_facts_from_llm`)
/// without re-parsing the payload.
pub(crate) fn parse_facts_with_raw_count(
    json: &str,
) -> anyhow::Result<(Vec<ExtractedFact>, usize)> {
    let trimmed = json.trim();
    // KREMORY_DEBUG=1: emit raw stage3 triplet output to stderr for diagnosis (Rule 19).
    // Closes the observability gap identified in TD-080: an empty fact list could be a
    // genuine `[]` emission OR a malformed payload silently swallowed below — this dump
    // distinguishes them. Mirrors the stage1 dump in `parse_entities_integer`.
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.extraction.parsers",
            len = trimmed.len(),
            raw_input = %trimmed,
            "parse_facts raw input"
        );
    }

    // Root-cause fix (2026-07-20, corrects ADR-077): the strict schema
    // `TripletListWrapper` makes providers emit `{"items":[...]}`, but this parser
    // previously only accepted a bare `[...]`, so every wrapped emission from a
    // strict provider (e.g. Groq gpt-oss) silently dropped to 0 facts — the cause
    // of the empty benchmark fact graph. Now routed through the shared
    // `parse_items` helper (generalizes the fix; accepts BOTH shapes).
    let (raw, raw_count, deserialize_ok, _path): (Vec<RawFact>, usize, bool, ParsePath) =
        parse_items(trimmed, "items");

    if deserialize_ok {
        counter!("rql.extraction.json_parse_ok", "parser" => "facts").increment(1);
    } else {
        counter!("rql.extraction.json_parse_fail", "parser" => "facts").increment(1);
        tracing::warn!(parser = "facts", "kremory.extraction.json_parse_fail");
    }

    let mut dropped_empty_field = 0u64;
    let facts: Vec<ExtractedFact> = raw
        .into_iter()
        .filter(|f| {
            let keep = !f.subject.is_empty() && !f.predicate.is_empty() && !f.object.is_empty();
            if !keep {
                dropped_empty_field += 1;
            }
            keep
        })
        .map(|f| ExtractedFact {
            subject: f.subject,
            predicate: f.predicate,
            object: f.object,
            is_entity_ref: f.is_entity_ref,
            confidence: f.confidence,
        })
        .collect();

    if dropped_empty_field > 0 {
        counter!("rql.extraction.item_dropped", "parser" => "facts", "reason" => "empty_field")
            .increment(dropped_empty_field);
    }
    emit_parse_yield_metrics("facts", raw_count, facts.len());

    Ok((facts, raw_count))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_items: wrapped-shape happy path ─────────────────────────────

    #[test]
    fn parse_items_wrapped_n_items_parses_all() {
        let (items, raw_count, ok, path) = parse_items::<u32>(r#"{"items":[1,2,3]}"#, "items");
        assert_eq!(items, vec![1, 2, 3]);
        assert_eq!(raw_count, 3);
        assert!(ok);
        assert_eq!(path, ParsePath::Wrapped);
    }

    #[test]
    fn parse_items_bare_array_n_items_parses_all() {
        let (items, raw_count, ok, path) = parse_items::<u32>("[1,2,3]", "items");
        assert_eq!(items, vec![1, 2, 3]);
        assert_eq!(raw_count, 3);
        assert!(ok);
        assert_eq!(path, ParsePath::BareArray);
    }

    #[test]
    fn parse_items_post_repair_shape_reports_post_repair_path() {
        // A trailing comma before `]` is invalid JSON — `serde_json` rejects
        // it on the first (unrepaired) attempt, so this only parses after
        // `repair_to_array` strips the trailing comma. The eventual shape
        // that matches is a bare array, but the OUTER path label must be
        // `PostRepair` (Vera DENT-001: post-repair success is suspicious and
        // must stay distinguishable from a clean first-attempt parse).
        let (items, raw_count, ok, path) = parse_items::<u32>("[1,2,3,]", "items");
        assert_eq!(items, vec![1, 2, 3]);
        assert_eq!(raw_count, 3);
        assert!(ok);
        assert_eq!(
            path,
            ParsePath::PostRepair,
            "trailing-comma input must only parse after repair, not on the first attempt"
        );
    }

    // ── ASMP-001: ordering invariant ───────────────────────────────────────

    #[test]
    fn parse_items_wrapped_empty_yields_zero_raw_count_no_repair_needed() {
        // A genuine `{"items": []}` must be recognised as a structurally-valid
        // empty result WITHOUT falling through to repair_to_array. If the
        // wrapper-key extraction ran AFTER repair (the ASMP-001 bug),
        // repair_to_array would wrap the whole object in an array
        // (`[{"items":[]}]`), the wrapper-key lookup would then run against
        // an array (no string keys), and the parse would fail entirely
        // (deserialize_ok=false) rather than recognising the genuine empty
        // wrapper (deserialize_ok=true). Asserting BOTH raw_count==0 AND
        // ok==true is what actually pins the ordering — raw_count alone is
        // 0 either way (total failure also reports raw_count=0).
        let (items, raw_count, ok, path) = parse_items::<u32>(r#"{"items":[]}"#, "items");
        assert!(items.is_empty());
        assert_eq!(raw_count, 0);
        assert!(
            ok,
            "a genuine empty wrapper must structurally parse (ok=true), not fail"
        );
        assert_eq!(path, ParsePath::Wrapped);
    }

    #[test]
    fn parse_items_bare_empty_array_yields_zero_raw_count() {
        let (items, raw_count, ok, path) = parse_items::<u32>("[]", "items");
        assert!(items.is_empty());
        assert_eq!(raw_count, 0);
        assert!(ok);
        assert_eq!(path, ParsePath::BareArray);
    }

    // ── SCOPE-003: wrong-key hazard ─────────────────────────────────────────

    #[test]
    fn parse_items_wrong_top_level_key_does_not_silently_succeed_as_empty() {
        // `{"result": [...]}` when the caller asked for wrapper_key="items"
        // must NOT be indistinguishable from a genuine `{"items": []}`. Both
        // report an empty `items` vec, but `deserialize_ok` differs: a
        // genuine empty wrapper is `ok=true`; a wrong/missing key with no
        // fallback shape available is `ok=false`. That boolean is the
        // "distinguishing signal" a `#[serde(default)]` wrapper struct cannot
        // provide (it would silently report ok looking identical to genuine
        // empty).
        let (items, raw_count, ok, path) = parse_items::<u32>(r#"{"result":[1,2,3]}"#, "items");
        assert!(items.is_empty());
        assert_eq!(raw_count, 0);
        assert!(
            !ok,
            "wrong top-level key must be distinguishable from genuine empty (ok=false)"
        );
        assert_eq!(path, ParsePath::None);
    }

    #[test]
    fn parse_items_malformed_json_reports_not_ok() {
        let (items, raw_count, ok, path) = parse_items::<u32>("not json at all", "items");
        assert!(items.is_empty());
        assert_eq!(raw_count, 0);
        assert!(!ok);
        assert_eq!(path, ParsePath::None);
    }

    // ── parse_facts: wrapped + bare + empty ─────────────────────────────────

    #[test]
    fn parse_facts_wrapped_n_facts_parses_all() {
        let json = r#"{"items":[
            {"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9},
            {"subject":"Bob","predicate":"manages","object":"Team","is_entity_ref":false,"confidence":0.8}
        ]}"#;
        let facts = parse_facts(json).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject, "Alice");
        assert_eq!(facts[1].subject, "Bob");
    }

    #[test]
    fn parse_facts_wrapped_empty_yields_zero_facts_no_tripwire_state() {
        let (facts, raw_count) = parse_facts_with_raw_count(r#"{"items":[]}"#).unwrap();
        assert!(facts.is_empty());
        assert_eq!(
            raw_count, 0,
            "genuine empty wrapper must report raw_count=0"
        );
    }

    #[test]
    fn parse_facts_with_raw_count_reports_raw_before_filter() {
        // One fact has an empty object field and gets filtered — raw_count
        // must reflect the pre-filter total (2), not the post-filter total (1).
        let json = r#"{"items":[
            {"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9},
            {"subject":"Bob","predicate":"manages","object":"","is_entity_ref":false,"confidence":0.8}
        ]}"#;
        let (facts, raw_count) = parse_facts_with_raw_count(json).unwrap();
        assert_eq!(facts.len(), 1, "the empty-object fact must be filtered out");
        assert_eq!(raw_count, 2, "raw_count must count both facts pre-filter");
    }

    // ── TD-133 PREVENTION: parse_facts CONTRACT regression gate ─────────────
    //
    // These pin the exact wrapper-drift bug that caused the 262/520 Groq parse
    // loss (SYSTEM-PRIMER §5) so it can never silently regress locally, plus
    // the `llm-output-parse-loudly` contract: a triplet MISSING a required
    // content field must FAIL parse loudly (deserialize_ok=false → the fallback
    // ladder retries), NOT silently emit an empty-field fact that gets dropped.
    // Uses real Groq-observed shapes (strict-provider wrapped `{"items":[...]}`
    // and lenient bare `[...]`).

    #[test]
    fn parse_facts_wrapped_groq_shape_yields_one_fact() {
        // (a) The strict-provider wrapped shape `{"items":[...]}` — the exact
        // shape Groq gpt-oss emits and that the original TD-133 bug dropped to 0
        // facts. Must parse to exactly one fact.
        let json = r#"{"items":[{"subject":"a","predicate":"p","object":"b","is_entity_ref":false,"confidence":1.0}]}"#;

        // parse_items-level: assert the underlying deserialize_ok=true contract.
        let (raw, raw_count, ok, path) = parse_items::<RawFact>(json, "items");
        assert_eq!(raw.len(), 1);
        assert_eq!(raw_count, 1);
        assert!(ok, "well-formed wrapped triplet must report deserialize_ok=true");
        assert_eq!(path, ParsePath::Wrapped);

        // public parse_facts surface: exactly one fact, fields preserved.
        let (facts, raw_count2) = parse_facts_with_raw_count(json).unwrap();
        assert_eq!(facts.len(), 1, "wrapped Groq shape must parse to exactly 1 fact");
        assert_eq!(raw_count2, 1);
        assert_eq!(facts[0].subject, "a");
        assert_eq!(facts[0].predicate, "p");
        assert_eq!(facts[0].object, "b");
    }

    #[test]
    fn parse_facts_bare_array_groq_shape_yields_one_fact() {
        // (b) The lenient bare-array shape `[...]` — still emitted by some
        // providers even when the schema advertises the wrapper.
        let json = r#"[{"subject":"a","predicate":"p","object":"b","is_entity_ref":false,"confidence":1.0}]"#;
        let (facts, raw_count) = parse_facts_with_raw_count(json).unwrap();
        assert_eq!(facts.len(), 1, "bare-array Groq shape must parse to exactly 1 fact");
        assert_eq!(raw_count, 1);
        assert_eq!(facts[0].subject, "a");
    }

    #[test]
    fn parse_facts_missing_predicate_fails_loud_not_silent_empty_field() {
        // (c) A triplet MISSING the required `predicate` field must drive the
        // whole payload to a LOUD parse failure so the fallback ladder retries —
        // NOT silently deserialise with predicate="" and get dropped by the
        // empty-field filter. Before removing #[serde(default)] from RawFact,
        // this payload deserialised (predicate="") and was silently filtered;
        // now it must fail parse (llm-output-parse-loudly).
        let json = r#"{"items":[{"subject":"a","object":"b","is_entity_ref":false,"confidence":1.0}]}"#;

        // parse_items-level: the whole array fails to deserialize → loud signal.
        let (raw, raw_count, ok, path) = parse_items::<RawFact>(json, "items");
        assert!(raw.is_empty(), "a triplet missing `predicate` must not yield a RawFact");
        assert_eq!(raw_count, 0);
        assert!(
            !ok,
            "missing required content field must be a loud parse failure (deserialize_ok=false), \
             not a silent empty-field default"
        );
        assert_eq!(path, ParsePath::None);

        // public parse_facts surface: 0 facts emitted (routes to json_parse_fail).
        let (facts, raw_count2) = parse_facts_with_raw_count(json).unwrap();
        assert!(
            facts.is_empty(),
            "public parse_facts must emit 0 facts for a required-field-missing payload"
        );
        assert_eq!(raw_count2, 0);
    }

    #[test]
    fn parse_facts_missing_predicate_increments_json_parse_fail_not_ok() {
        // Companion o11y assertion for (c): the loud failure must be visible on
        // the `rql.extraction.json_parse_fail{parser=facts}` counter, not masked
        // as a success. This is the metric that would have surfaced the 262
        // silent drops as an actual failure signal.
        let json = r#"{"items":[{"subject":"a","object":"b","is_entity_ref":false,"confidence":1.0}]}"#;
        let fails = captured_counter_label_values(
            "rql.extraction.json_parse_fail",
            "parser",
            || {
                let (facts, _raw) = parse_facts_with_raw_count(json).unwrap();
                assert!(facts.is_empty());
            },
        );
        assert_eq!(
            fails,
            vec!["facts".to_string()],
            "missing-required-field payload must increment json_parse_fail{{parser=facts}}, \
             got {fails:?}"
        );
    }

    // ── parse_relation_names: wrapped + bare + empty (new coverage — this
    // parser previously accepted bare-array only) ──────────────────────────

    #[test]
    fn parse_relation_names_wrapped_n_names_parses_all() {
        let names = parse_relation_names(r#"{"items":["works_at","manages"]}"#).unwrap();
        assert_eq!(names, vec!["works_at".to_string(), "manages".to_string()]);
    }

    #[test]
    fn parse_relation_names_bare_array_still_works() {
        let names = parse_relation_names(r#"["works_at","manages"]"#).unwrap();
        assert_eq!(names, vec!["works_at".to_string(), "manages".to_string()]);
    }

    #[test]
    fn parse_relation_names_wrapped_empty_yields_zero_names() {
        let names = parse_relation_names(r#"{"items":[]}"#).unwrap();
        assert!(names.is_empty());
    }

    // ── parse_entities_integer: wrong-key hazard closed ─────────────────────

    #[test]
    fn parse_entities_integer_wrong_key_does_not_silently_succeed() {
        let registry = crate::core::entity_types::EntityTypeRegistry::from_specs(vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
        ]);
        // Wrong top-level key ("result" instead of "entities") — must not be
        // silently treated the same as a genuine `{"entities": []}`. Both
        // yield an empty Vec<ExtractedEntity> at this call's public surface
        // (anyhow::Result<Vec<ExtractedEntity>>), so the distinguishing
        // signal lives in the `rql.extraction.json_parse_fail` counter this
        // call increments (verified via the underlying `parse_items` unit
        // tests above, which assert `deserialize_ok` directly).
        let entities = parse_entities_integer(
            r#"{"result":[{"name":"Alice","entity_type_id":0}]}"#,
            &registry,
        )
        .unwrap();
        assert!(
            entities.is_empty(),
            "wrong-key payload must not resolve to entities"
        );
    }

    // ── parse_entities_integer: `path` label restored (Vera DENT-001) ───────
    //
    // These pin the actual shape reported on `rql.extraction.json_parse_ok`'s
    // `path` label — Rule 20: a post-repair success is suspicious and must
    // stay distinguishable from a clean wrapped/bare_array parse. Uses
    // `metrics_util::debugging::DebuggingRecorder` (library-safe local
    // recorder, no global state) — same pattern as `tests/b1_observability.rs`.

    fn registry_with_person() -> crate::core::entity_types::EntityTypeRegistry {
        crate::core::entity_types::EntityTypeRegistry::from_specs(vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
        ])
    }

    /// Snapshot every `rql.extraction.json_parse_ok{path=...}` label value
    /// recorded while `f` runs, under a local (non-global) recorder.
    fn captured_json_parse_ok_paths(f: impl FnOnce()) -> Vec<String> {
        use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

        let recorder = DebuggingRecorder::new();
        let snapshotter: Snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, f);

        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter(|(key, _, _, _)| key.key().name() == "rql.extraction.json_parse_ok")
            .filter_map(|(key, _, _, _)| {
                key.key()
                    .labels()
                    .find(|l| l.key() == "path")
                    .map(|l| l.value().to_string())
            })
            .collect()
    }

    #[test]
    fn parse_entities_integer_wrapped_input_reports_wrapped_path() {
        let registry = registry_with_person();
        let paths = captured_json_parse_ok_paths(|| {
            let entities = parse_entities_integer(
                r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#,
                &registry,
            )
            .unwrap();
            assert_eq!(entities.len(), 1);
        });
        assert_eq!(
            paths,
            vec!["wrapped".to_string()],
            "a clean wrapped-object input must report path=wrapped, got {paths:?}"
        );
    }

    #[test]
    fn parse_entities_integer_bare_array_input_reports_bare_array_path() {
        let registry = registry_with_person();
        let paths = captured_json_parse_ok_paths(|| {
            let entities =
                parse_entities_integer(r#"[{"name":"Alice","entity_type_id":1}]"#, &registry)
                    .unwrap();
            assert_eq!(entities.len(), 1);
        });
        assert_eq!(
            paths,
            vec!["bare_array".to_string()],
            "a clean bare-array input must report path=bare_array, got {paths:?}"
        );
    }

    // ── B1: json_parse_fail / json_parse_ok carry a `parser` label ──────────
    //
    // Prior to this fix, `rql.extraction.json_parse_fail` / `_ok` were emitted
    // BARE (no `parser` label) at the relation_names/entities_integer/facts
    // sites, collapsing all three sources into one un-attributable aggregate —
    // making it impossible to tell which parser was failing (B1 debugging
    // blocker). These pin the `parser` label now present on both counters for
    // each of the three call sites, using the same local-recorder pattern as
    // `captured_json_parse_ok_paths` above (module-private parsers can't be
    // exercised from the external `tests/b1_observability.rs` integration
    // crate — `pub(crate)` visibility — so unit tests here are the correct,
    // compiling location).

    /// Snapshot every recorded value of `label_key` on counter `metric_name`
    /// while `f` runs, under a local (non-global) recorder. Generalizes
    /// `captured_json_parse_ok_paths` for the parser-attribution tests below.
    fn captured_counter_label_values(
        metric_name: &'static str,
        label_key: &'static str,
        f: impl FnOnce(),
    ) -> Vec<String> {
        use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

        let recorder = DebuggingRecorder::new();
        let snapshotter: Snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, f);

        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter(|(key, _, _, _)| key.key().name() == metric_name)
            .filter_map(|(key, _, _, _)| {
                key.key()
                    .labels()
                    .find(|l| l.key() == label_key)
                    .map(|l| l.value().to_string())
            })
            .collect()
    }

    #[test]
    fn parse_facts_malformed_input_reports_json_parse_fail_with_parser_facts_label() {
        let parsers = captured_counter_label_values(
            "rql.extraction.json_parse_fail",
            "parser",
            || {
                let (facts, raw_count) =
                    parse_facts_with_raw_count("not json at all").unwrap();
                assert!(facts.is_empty());
                assert_eq!(raw_count, 0);
            },
        );
        assert_eq!(
            parsers,
            vec!["facts".to_string()],
            "json_parse_fail must carry parser=facts for the fact parser, got {parsers:?}"
        );
    }

    #[test]
    fn parse_relation_names_malformed_input_reports_json_parse_fail_with_parser_relation_names_label()
     {
        let parsers = captured_counter_label_values(
            "rql.extraction.json_parse_fail",
            "parser",
            || {
                let names = parse_relation_names("not json at all").unwrap();
                assert!(names.is_empty());
            },
        );
        assert_eq!(
            parsers,
            vec!["relation_names".to_string()],
            "json_parse_fail must carry parser=relation_names for the relation-names \
             parser, got {parsers:?}"
        );
    }

    #[test]
    fn parse_entities_integer_malformed_input_reports_json_parse_fail_with_parser_label() {
        let registry = registry_with_person();
        let parsers = captured_counter_label_values(
            "rql.extraction.json_parse_fail",
            "parser",
            || {
                let entities =
                    parse_entities_integer("not json at all", &registry).unwrap();
                assert!(entities.is_empty());
            },
        );
        assert_eq!(
            parsers,
            vec!["entities_integer".to_string()],
            "json_parse_fail must carry parser=entities_integer for the integer-id \
             entity parser, got {parsers:?}"
        );
    }

    #[test]
    fn parse_facts_wrapped_input_reports_json_parse_ok_with_parser_facts_label() {
        let parsers = captured_counter_label_values(
            "rql.extraction.json_parse_ok",
            "parser",
            || {
                let facts = parse_facts(r#"{"items":[]}"#).unwrap();
                assert!(facts.is_empty());
            },
        );
        assert_eq!(
            parsers,
            vec!["facts".to_string()],
            "json_parse_ok must carry parser=facts for the fact parser, got {parsers:?}"
        );
    }

    #[test]
    fn parse_relation_names_wrapped_input_reports_json_parse_ok_with_parser_relation_names_label()
     {
        let parsers = captured_counter_label_values(
            "rql.extraction.json_parse_ok",
            "parser",
            || {
                let names = parse_relation_names(r#"{"items":[]}"#).unwrap();
                assert!(names.is_empty());
            },
        );
        assert_eq!(
            parsers,
            vec!["relation_names".to_string()],
            "json_parse_ok must carry parser=relation_names for the relation-names \
             parser, got {parsers:?}"
        );
    }

    #[test]
    fn parse_entities_integer_post_repair_input_reports_post_repair_path() {
        let registry = registry_with_person();
        // A single bare object (no "entities" wrapper key, not itself an
        // array) only parses after `repair_to_array` wraps it in `[...]`.
        let paths = captured_json_parse_ok_paths(|| {
            let entities =
                parse_entities_integer(r#"{"name":"Alice","entity_type_id":1}"#, &registry)
                    .unwrap();
            assert_eq!(entities.len(), 1);
        });
        assert_eq!(
            paths,
            vec!["post_repair".to_string()],
            "a shape that only parses after repair must report path=post_repair \
             (Rule 20: post-repair success is suspicious), got {paths:?}"
        );
    }
}

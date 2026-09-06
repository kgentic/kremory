//! JSON repair and NuExtract-specific parsing utilities.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

use metrics::counter;
use tracing;

use super::models::LlmExtractionOutput;
use crate::core::intelligence::{ExtractedEntity, ExtractedFact, ExtractionContext};

// ─── JSON object extraction ──────────────────────────────────────────────────

/// Extract the first balanced JSON object `{...}` from `s`.
///
/// Some LLMs (e.g. llama3.2) emit valid JSON followed by prose ("Note: ..."),
/// or emit multiple JSON objects separated by whitespace. This function returns
/// a slice covering only the first complete `{...}` block. If no balanced
/// object is found, returns the original input unchanged.
pub(super) fn extract_first_json_object(s: &str) -> &str {
    let bytes = s.as_bytes();
    let len = bytes.len();

    // Find the first `{`
    let start = match bytes.iter().position(|&b| b == b'{') {
        Some(pos) => pos,
        None => return s,
    };

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut i = start;

    while i < len {
        match bytes[i] {
            b'"' if !in_string => {
                in_string = true;
                i += 1;
            }
            b'"' if in_string => {
                in_string = false;
                i += 1;
            }
            b'\\' if in_string => {
                // skip escape sequence
                i += 2;
            }
            b'{' if !in_string => {
                depth += 1;
                i += 1;
            }
            b'}' if !in_string => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return &s[start..i];
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // No balanced object found — return original.
    s
}

// ─── Unclosed-string fixer ───────────────────────────────────────────────────

/// Pre-process NuExtract JSON to fix unclosed string values before `}`.
///
/// Some LLMs (e.g. `gemma4-e2b`) omit the closing `"` on string values when
/// followed immediately by `}`, producing output like:
///   `{"name": "Bob", "label": "Person}`
/// instead of the correct:
///   `{"name": "Bob", "label": "Person"}`
///
/// This pattern breaks `serde_json` parsing before `jsonrepair::repair_json` can
/// help, because the `}` gets absorbed into the unclosed string, mangling the
/// rest of the document structure.
///
/// Strategy: scan for `"}` where the `"` was intended as opening-quote-already-in-
/// progress. More concretely, find any byte sequence that matches `"<word>}` where
/// `<word>` contains no `"`, `\`, `{`, `}`, or newline — and insert the missing `"`.
pub(crate) fn fix_unclosed_string_before_brace(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = Vec::with_capacity(len + 8);
    let mut i = 0;

    while i < len {
        // Look for the pattern: `"` followed by 1+ safe chars followed by `}`
        // where no closing `"` precedes the `}`.
        if bytes[i] == b'"' {
            // Scan forward to find the end of this potential string.
            let start = i; // points at opening `"`
            i += 1;
            let mut found_close = false;
            while i < len {
                match bytes[i] {
                    b'"' => {
                        // Properly closed string — copy verbatim up to and including `"`.
                        found_close = true;
                        i += 1;
                        break;
                    }
                    b'\\' => {
                        // Escape sequence — skip both chars.
                        i += 2;
                    }
                    b'}' if !found_close => {
                        // `}` inside an unclosed string — insert missing `"` before `}`.
                        out.extend_from_slice(&bytes[start..i]);
                        out.push(b'"'); // close the string
                        out.push(b'}'); // then the brace
                        i += 1;
                        found_close = true; // consumed this token
                        break;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            if !found_close && i >= len {
                // Ran off the end without closing — just copy remainder as-is.
                out.extend_from_slice(&bytes[start..i]);
            } else if found_close {
                // Copy up to current position if we broke on a real close quote.
                // (already pushed in the `}` branch above; for the `"` branch copy.)
                // For the `"` branch we need to push the range start..i.
                // Check: did we push via the `}` branch (already done)?
                // We can tell because `bytes[i-1]` would be `}` (pushed above).
                if i > 0 && bytes[i - 1] != b'}' {
                    out.extend_from_slice(&bytes[start..i]);
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(out).unwrap_or_else(|_| s.to_owned())
}

// ─── NuExtract response parser ───────────────────────────────────────────────

pub(crate) fn parse_nuextract_response(
    json: &str,
    ctx: &ExtractionContext<'_>,
) -> anyhow::Result<(Vec<ExtractedEntity>, Vec<ExtractedFact>)> {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        counter!("rql.extraction.json_parse_fail").increment(1);
        tracing::warn!(
            parser = "nuextract",
            "kremory.extraction.json_parse_fail empty response"
        );
        return Ok((vec![], vec![]));
    }

    // Extract the first balanced `{...}` block. Some LLMs (e.g. llama3.2) emit
    // valid JSON followed by prose ("Note: ...") or multiple JSON objects.
    // Feeding multiple objects or trailing prose to serde/repair_json produces
    // "invalid type: sequence, expected a string" errors.
    let first_obj = extract_first_json_object(trimmed);

    // If no JSON object found at all (pure prose response), return empty gracefully.
    if !first_obj.trim_start().starts_with('{') {
        counter!("rql.extraction.json_parse_fail").increment(1);
        tracing::warn!(
            parser = "nuextract",
            "kremory.extraction.json_parse_fail no JSON object in response"
        );
        return Ok((vec![], vec![]));
    }

    // Pre-process: some LLMs omit closing `"` before `}` in string values.
    // Fix this before attempting serde / llm_json repair.
    let preprocessed = fix_unclosed_string_before_brace(first_obj);
    let trimmed = preprocessed.as_ref();

    // Try direct parse, then llm_json repair if malformed.
    let output: LlmExtractionOutput = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            v
        }
        Err(_) => {
            // repair_to_array wraps in array — but NuExtract returns an object.
            // Try object-level repair first, then array-unwrap fallback.
            let repaired = jsonrepair::repair_json(trimmed, &jsonrepair::Options::default())
                .unwrap_or_else(|_| trimmed.to_owned());
            // KREMORY_DEBUG: dump raw before/after the repair so the operator can see what
            // malformed JSON the model emitted and what repair produced (R1.3 json_repair
            // supplemental). The unconditional warn on the failure paths below is the
            // always-on operational signal; this gated debug carries the raw payload.
            if std::env::var("KREMORY_DEBUG").is_ok() {
                tracing::debug!(
                    target: "kremory.extraction.json_repair",
                    raw_before = %trimmed,
                    repaired_after = %repaired,
                    "nuextract JSON repair invoked"
                );
            }
            match serde_json::from_str::<LlmExtractionOutput>(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok").increment(1);
                    v
                }
                Err(_) => {
                    // Last resort: repair_to_array may have wrapped the object.
                    let array_repaired = repair_to_array(trimmed);
                    match serde_json::from_str::<serde_json::Value>(&array_repaired) {
                        Ok(v) => {
                            // Unwrap single-element array back to object.
                            let obj = if let Some(arr) = v.as_array() {
                                if arr.len() == 1 {
                                    &arr[0]
                                } else {
                                    &v
                                }
                            } else {
                                &v
                            };
                            match serde_json::from_value::<LlmExtractionOutput>(obj.clone()) {
                                Ok(v) => {
                                    counter!("rql.extraction.json_parse_ok").increment(1);
                                    v
                                }
                                Err(e) => {
                                    counter!("rql.extraction.json_parse_fail").increment(1);
                                    tracing::warn!(error = %e, parser = "nuextract", "kremory.extraction.json_parse_fail after repair");
                                    return Ok((vec![], vec![]));
                                }
                            }
                        }
                        Err(e) => {
                            counter!("rql.extraction.json_parse_fail").increment(1);
                            tracing::warn!(error = %e, parser = "nuextract", "kremory.extraction.json_parse_fail array repair");
                            return Ok((vec![], vec![]));
                        }
                    }
                }
            }
        }
    };

    // Convert serde structs → domain types, filtering empty names.
    // Store entity name in `properties["name"]` so the FTS index (which indexes
    // the `properties` column) can match queries against the entity's original
    // case name (e.g. "Alice"). Without this, FTS MATCH "Alice" returns nothing
    // because entity_id is UNINDEXED and label is the type ("Person"), not the name.
    let mut entities: Vec<ExtractedEntity> = output
        .entities
        .into_iter()
        .filter(|e| !e.name.is_empty())
        .map(|e| {
            let label = if e.label.is_empty() {
                "Entity".to_string()
            } else {
                e.label
            };
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            ExtractedEntity {
                name: e.name,
                label,
                properties: serde_json::Value::Object(props),
            }
        })
        .collect();

    // Apply exclusion filter.
    if !ctx.excluded_entity_types.is_empty() {
        entities.retain(|e| !ctx.excluded_entity_types.contains(&e.label));
    }

    // Build entity name set for is_entity_ref resolution.
    let entity_names: std::collections::HashSet<String> =
        entities.iter().map(|e| e.name.to_lowercase()).collect();

    // Convert relationships, filtering empty required fields.
    let facts: Vec<ExtractedFact> = output
        .relationships
        .into_iter()
        .filter(|r| !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty())
        .map(|r| {
            let is_entity_ref = entity_names.contains(&r.object.to_lowercase());
            ExtractedFact {
                subject: r.subject,
                predicate: r.predicate,
                object: r.object,
                is_entity_ref,
                confidence: r.confidence,
                // TD-187 round 2: the repair path reconstructs from
                // `RawRelationship`, which has no date field — see the same note
                // in `programmatic.rs`. `None` ⇒ episode `ref_time` fallback.
                valid_at: None,
            }
        })
        .collect();

    Ok((entities, facts))
}

// ─── JSON Repair ─────────────────────────────────────────────────────────────

/// Repair messy LLM JSON into a parseable array.
/// Handles: markdown fences, missing array brackets, trailing commas,
/// single quotes, Python booleans (True/False/None).
/// Strip a markdown code fence (```` ```json … ``` ````) from an LLM response.
///
/// Returns `raw` trimmed and unchanged when there is no opening fence, so it is
/// safe to call unconditionally.
///
/// # Why this is its own function (TD-192)
///
/// The fence-strip used to live INSIDE [`repair_to_array`], welded to that
/// function's other job — wrapping a bare `{...}` in an array. That coupling
/// meant `structured.rs::parse_response_to_value` could not reuse it: it must
/// preserve object-vs-array shape exactly (the contradiction verdict is a
/// wrapper OBJECT, and turning it into an array silently loses `reason`). So
/// `parse_response_to_value` had no fence handling at all, and a fenced payload
/// fell through to first-balanced-OBJECT extraction — which discarded every
/// element of a fenced ARRAY after the first. See that call site for the full
/// failure.
///
/// Hand-rolled rather than taking a dependency (Rule 33): the grammar is a
/// formally-specified, ~10-line, edge-case-free prefix/suffix strip, and the
/// crate's existing JSON-repair dependency (`llm_json`) demonstrably does NOT
/// handle fences — it was already in the fallback chain at the call site and
/// failed on exactly this input.
pub(crate) fn strip_code_fences(raw: &str) -> &str {
    let s = raw.trim();
    let Some(after_ticks) = s.strip_prefix("```") else {
        return s;
    };
    // Drop the optional language tag, which runs to the first newline. A fence
    // opener with no newline at all is not a fenced block — leave it alone.
    let Some(nl) = after_ticks.find('\n') else {
        return s;
    };
    let body = &after_ticks[nl + 1..];
    // A missing CLOSING fence is common in truncated responses; keep the body.
    body.trim_end().strip_suffix("```").unwrap_or(body).trim()
}

pub(crate) fn repair_to_array(raw: &str) -> String {
    // Behaviour-preserving: this is the extracted fence-strip that used to be
    // inlined here (TD-192). Single source of truth, shared with
    // `structured.rs::parse_response_to_value`.
    let s = strip_code_fences(raw);
    if s.is_empty() || s == "[]" {
        return "[]".to_string();
    }

    // Wrap in array brackets if the output starts with `{` (objects without wrapper).
    let mut wrapped = if s.starts_with('{') {
        format!("[{s}]")
    } else {
        s.to_string()
    };

    // Strip trailing commas before `]` — llm_json mishandles these.
    while wrapped.contains(",]") {
        wrapped = wrapped.replace(",]", "]");
    }
    while wrapped.contains(", ]") {
        wrapped = wrapped.replace(", ]", "]");
    }

    // Use llm_json for fine-grained repairs (quotes, booleans, commas).
    let repaired =
        jsonrepair::repair_json(&wrapped, &jsonrepair::Options::default()).unwrap_or(wrapped);

    // llm_json sometimes reduces arrays to a single object — re-wrap if needed.
    let repaired = repaired.trim();
    if repaired.starts_with('{') {
        format!("[{repaired}]")
    } else {
        repaired.to_string()
    }
}

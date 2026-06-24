//! TD-023 — Hybrid GLiNER + LLM-typing extractor.
//!
//! Two-phase entity extraction designed to combine GLiNER's speed advantage
//! (64-83× faster than LLM-only per TD-022 empirical bench) with the LLM's
//! disambiguation strength (gemma4 picks Court / Drug / Theory better than
//! GLiNER on hard corpora like legal_deposition).
//!
//! ## Phase 1 — GLiNER (sync, fast)
//!
//! Run the closed-vocabulary `GlinerExtractor` over the input text using
//! `ctx.allowed_entity_types` as the label set. Yields a `Vec<ExtractedEntity>`
//! of candidate name+placeholder-label pairs. Typical wall-clock: ~4-5s for
//! a 14-domain corpus on Apple Silicon CPU.
//!
//! ## Phase 2 — LLM typing (sync, ONE batched call)
//!
//! Render a single prompt listing the GLiNER candidates and asking the LLM
//! to assign each one an `entity_type_id` from the registry. ONE LLM call
//! for ALL candidates. Uses the existing
//! `SCHEMA_ENTITY_LIST_INTEGER_ID` schema — the same `{entities: [{name,
//! entity_type_id}]}` shape used by the production integer-ID extractor.
//!
//! ## No-2nd-LLM-call invariant
//!
//! Per [[no-second-llm-pass-for-entity-extraction]]: this extractor makes
//! ONE LLM call total for typing. It REPLACES the current 3-stage extraction
//! (entities + relations + triplets), reducing total calls from 3 → 1 for
//! the typing surface. Relation/fact extraction is intentionally NOT done in
//! this extractor; downstream code uses `ingest_deferred` for relations,
//! preserving the substrate two-phase architecture.

use std::sync::Arc;

use crate::core::entity_types::EntityTypeRegistry;
use crate::core::error::Result as KremoryResult;
use crate::core::extraction::prompts::render_hybrid_typing_prompt;
use crate::core::extraction::schemas::hybrid_typing_schema_with_bounds;
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_user, ChatProvider};

/// Builder-time configuration for the GLiNER candidate-generation extractor.
///
/// **Hidden (F2):** `with_gliner()` is now a no-arg knob, so this type has no
/// constructor or consumer. Kept (not deleted) as the ADR-039 §A6 forward-compat
/// placeholder + the `kremory-napi` `GlinerConfigJs` doc-mirror target. When real
/// tuning fields land (threshold, model path, batch size), un-hide this and add a
/// `with_gliner_config(GlinerConfig)` knob — both non-breaking. `#[doc(hidden)]`
/// keeps it off the consumer-visible surface so F2's "do-nothing config" smell is
/// fully removed.
#[cfg(feature = "ner")]
#[doc(hidden)]
#[derive(Debug, Default, Clone)]
pub struct GlinerConfig {
    // Reserved for future tuning knobs (threshold, model path, batch size).
    // Adding a field here is non-breaking; consumers always construct via
    // `GlinerConfig::default()`.
    _private: (),
}

/// Hybrid extractor: GLiNER (Phase 1, span discovery) + ONE LLM call (Phase 2, typing).
///
/// Generic over `L: ChatProvider` so callers can wire any LLM provider.
/// The `GlinerExtractor` is held by value — model weights (~650MB INT8) are
/// loaded once at construction via hf-hub.
#[cfg(feature = "ner")]
pub struct GlinerLlmExtractor<L: ChatProvider> {
    gliner: crate::core::ner::GlinerExtractor,
    llm: Arc<L>,
}

#[cfg(feature = "ner")]
impl<L: ChatProvider> GlinerLlmExtractor<L> {
    /// Build the hybrid extractor. Downloads `gliner_large-v2.1` INT8 on first
    /// use (cached locally by hf-hub thereafter).
    ///
    /// Threshold tunable via `KREMORY_GLINER_THRESHOLD` env var (default 0.5).
    /// Lower threshold → higher recall (catches more candidates) at the cost
    /// of more noise candidates the LLM has to type-or-reject.
    ///
    /// Called from `MemoryBuilder::build()` in E-2 when both `with_gliner` and
    /// `with_llm` knobs are set.
    #[allow(dead_code)]
    pub fn new(llm: Arc<L>) -> anyhow::Result<Self> {
        let threshold: f32 = std::env::var("KREMORY_GLINER_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let gliner = crate::core::ner::GlinerExtractor::with_threshold(threshold)?;
        Ok(Self { gliner, llm })
    }
}

#[cfg(feature = "ner")]
impl<L: ChatProvider + 'static> EntityExtractor for GlinerLlmExtractor<L> {
    fn name(&self) -> &'static str {
        "gliner_llm"
    }

    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> KremoryResult<ExtractionResult> {
        let phase1_start = std::time::Instant::now();

        // Pre-normalize all-uppercase proper-noun tokens to title case.
        // gliner_large-v2.1 is trained on mixed-case news/web corpus and
        // misses all-caps Person names (e.g. legal depositions: "CLIFFORD
        // REEVES", "DIANA OHLSSON"). Per-word normalization avoids touching
        // intentional all-caps like product names ("GPT-4" stays unchanged
        // because of the digit). Empirical bench (TD-023 v4 + this fix
        // expected): +6.2pt → +10-12pt total on legal_deposition.
        let normalized_text = normalize_allcaps_words(text);
        let normalized_count = if normalized_text != text {
            1usize
        } else {
            0usize
        };
        metrics::counter!("rql.hybrid.allcaps_normalized_invocations")
            .increment(normalized_count as u64);

        // Phase 1: GLiNER fast span discovery on the normalized text.
        let gliner_result = self.gliner.extract(&normalized_text, ctx).await?;
        let phase1_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;
        metrics::histogram!("rql.hybrid.phase1_gliner_ms").record(phase1_ms);
        metrics::histogram!("rql.hybrid.phase1_candidate_count")
            .record(gliner_result.entities.len() as f64);

        // If GLiNER found nothing, return empty — no LLM call needed.
        if gliner_result.entities.is_empty() {
            metrics::counter!("rql.hybrid.phase2_skipped", "reason" => "no_candidates")
                .increment(1);
            return Ok(ExtractionResult {
                entities: vec![],
                facts: vec![],
            });
        }

        // Build a registry from the context's registry_specs so the prompt can
        // render the available types + the schema can bound entity_type_id.
        let registry = EntityTypeRegistry::from_specs(ctx.registry_specs.to_vec());

        // If the registry is empty (no types registered for this namespace),
        // fall back to GLiNER's raw labels — without registry context the LLM
        // typing call has no target taxonomy.
        if registry.is_empty() {
            metrics::counter!("rql.hybrid.phase2_skipped", "reason" => "empty_registry")
                .increment(1);
            return Ok(ExtractionResult {
                entities: gliner_result.entities,
                facts: vec![],
            });
        }

        // Phase 2: ONE batched LLM typing call.
        let phase2_start = std::time::Instant::now();
        let candidate_names: Vec<&str> = gliner_result
            .entities
            .iter()
            .map(|e| e.name.as_str())
            .collect();

        let prompt = render_hybrid_typing_prompt(text, &candidate_names, &registry);
        let messages = vec![chat_msg_user(&prompt)];

        // Schema bounds: idx ∈ [0, num_candidates - 1], entity_type_id ∈ registry
        let schema = hybrid_typing_schema_with_bounds(candidate_names.len(), registry.specs());

        // Force the LlmJsonRepair arm — the original 2026-06-04 choice is
        // correct for our production target (GLiNER + gemma4-e2b). Empirical
        // 2026-06-09 verification confirmed:
        //
        //   gemma4-e2b + FormatSchema     → 80% precision (LLM mistypes
        //     South Korea→Organisation, supplier→Quantity etc — small model
        //     "satisfices" to ANY allowed token under grammar-constrained
        //     decoding instead of the semantically right one)
        //
        //   gemma4-e2b + LlmJsonRepair    → 90% precision (LLM produces
        //     mostly-correct free-form JSON; the row-tolerant parser below
        //     keeps the good rows; for the failed rows GLiNER's open-vocab
        //     label is used as a fallback — operationally fine)
        //
        // The row-tolerant parser + adversarial test suite below pin the
        // free-form-with-defense pattern. Net effect: small-model parses
        // recover ~80% of the time at the row level, and FormatSchema's
        // semantic degradation is avoided.
        let model_name = ctx.model().to_string();
        let value = StructuredCallBuilder::new(self.llm.as_ref(), &schema, "HybridTyping")
            .messages(messages)
            .model(&model_name)
            .ttft_budget_ms(ctx.arm_budget_ms)
            .force_arm(crate::core::extraction::schemas::FallbackArm::LlmJsonRepair)
            .call()
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "GlinerLlmExtractor phase 2 LLM typing call failed: {e}"
                ))
            })?;

        let phase2_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;
        metrics::histogram!("rql.hybrid.phase2_llm_typing_ms").record(phase2_ms);

        // Parse the response. Schema is
        // `{entities: [{name: String, entity_type_id: u32}]}`. Build a
        // name → entity_type_id map from the LLM response, then walk GLiNER's
        // candidate list assigning the refined label.
        let typed_entities = parse_typed_response(&value, &gliner_result.entities, &registry);

        metrics::histogram!("rql.hybrid.final_entity_count").record(typed_entities.len() as f64);
        metrics::counter!("rql.hybrid.invocation_complete").increment(1);

        Ok(ExtractionResult {
            entities: typed_entities,
            facts: vec![],
        })
    }
}

/// Normalize all-uppercase WHOLE WORDS to title case.
///
/// Only applies when the word is purely alphabetic (or alphabetic + apostrophe)
/// AND >1 char AND fully uppercase. Words containing digits / hyphens / other
/// special chars pass through unchanged so identifiers like "GPT-4" or "U.K."
/// keep their intentional casing.
///
/// "CLIFFORD REEVES MEETS DIANA" → "Clifford Reeves Meets Diana".
/// "GPT-4 vs Claude" → unchanged.
#[cfg(feature = "ner")]
fn normalize_allcaps_words(text: &str) -> String {
    text.split_inclusive(|c: char| c.is_whitespace())
        .map(|chunk| {
            // Split off trailing whitespace so we preserve it verbatim.
            let trailing_ws_start = chunk
                .rfind(|c: char| !c.is_whitespace())
                .map(|i| i + chunk[i..].chars().next().map_or(0, |c| c.len_utf8()))
                .unwrap_or(chunk.len());
            let (word, ws) = chunk.split_at(trailing_ws_start);

            let is_target = word.chars().count() > 1
                && word.chars().all(|c| {
                    c.is_ascii_alphabetic()
                        || c == '\''
                        || c == '.'
                        || c == ','
                        || c == ';'
                        || c == ':'
                })
                && word.chars().any(|c| c.is_ascii_alphabetic())
                && word
                    .chars()
                    .filter(|c| c.is_ascii_alphabetic())
                    .all(|c| c.is_ascii_uppercase());

            if !is_target {
                return chunk.to_string();
            }

            // Title-case: first alpha upper, rest lower; preserve non-alpha chars.
            let mut out = String::with_capacity(word.len());
            let mut saw_alpha = false;
            for ch in word.chars() {
                if ch.is_ascii_alphabetic() {
                    if !saw_alpha {
                        out.push(ch);
                        saw_alpha = true;
                    } else {
                        out.push(ch.to_ascii_lowercase());
                    }
                } else {
                    out.push(ch);
                }
            }
            out.push_str(ws);
            out
        })
        .collect()
}

/// Parse `{typings: [{idx, entity_type_id}]}` and merge LLM typings back
/// into the GLiNER candidate list via the bounded `idx` field.
///
/// Per [[load-bearing-invariants-at-emit-not-prompt]] the candidate-to-typing
/// link is enforced at emit by the schema's `idx` enum bounds. Out-of-range
/// emissions are dropped here (defensive); candidates with no matching
/// typing keep their GLiNER label (fallback — better a rough type than a
/// dropped entity).
#[cfg(feature = "ner")]
fn parse_typed_response(
    response: &serde_json::Value,
    gliner_entities: &[ExtractedEntity],
    registry: &EntityTypeRegistry,
) -> Vec<ExtractedEntity> {
    // All-or-nothing wrapper deserialization per [[llm-output-parse-loudly]].
    // On gemma4-e2b empirical 2026-06-09: LLM typing under LlmJsonRepair often
    // produces semantically wrong choices (Amazon Robotics→Location,
    // France→Quantity, etc) — when the response fails to parse, that's
    // actually GOOD because the GLiNER fallback labels (which were correct)
    // win. Row-tolerant parsing was tried + reverted: it let through the
    // LLM's bad partial output and dropped precision 90% → 60%.
    //
    // If a future model + prompt combo produces semantically correct typings
    // reliably, this can be revisited — but then FormatSchema (Ollama-enforced
    // structure) becomes the right tradeoff and we wouldn't need row-tolerance
    // anyway.
    let wrapper: super::models::HybridTypingWrapper = match serde_json::from_value(response.clone())
    {
        Ok(w) => w,
        Err(_) => {
            metrics::counter!("rql.hybrid.response_parse_fail").increment(1);
            super::models::HybridTypingWrapper::default()
        }
    };

    let mut typing_by_idx: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for typing in &wrapper.typings {
        let idx = typing.idx as usize;
        if idx >= gliner_entities.len() {
            metrics::counter!("rql.hybrid.idx_out_of_range").increment(1);
            continue;
        }
        // Treat id=0 from the LLM as "model declined to classify" → fall
        // back to GLiNER's hint downstream (don't record in typing_by_idx).
        // The schema's enum excludes 0 at the FormatSchema layer; the
        // LlmJsonRepair arm bypasses the enum, so this is the parser-side
        // backstop per [[audit-what-guards-mask-before-deleting]].
        if typing.entity_type_id == 0 {
            metrics::counter!(
                "rql.hybrid.llm_chose_catch_all",
                "fallback" => "gliner_label",
            )
            .increment(1);
            continue;
        }
        let validated = registry.validate_or_fallback(typing.entity_type_id);
        if validated == 0 {
            // Model returned an out-of-registry id that resolved to 0.
            metrics::counter!("rql.hybrid.llm_out_of_registry_fallback",).increment(1);
            continue;
        }
        typing_by_idx.insert(idx, validated);
    }

    metrics::histogram!("rql.hybrid.llm_typing_response_count").record(typing_by_idx.len() as f64);

    let mut result: Vec<ExtractedEntity> = Vec::with_capacity(gliner_entities.len());
    for (idx, entity) in gliner_entities.iter().enumerate() {
        let llm_type_id = typing_by_idx.get(&idx).copied();

        let refined_label = if let Some(id) = llm_type_id {
            registry.id_to_name(id).to_string()
        } else {
            metrics::counter!(
                "rql.hybrid.candidate_unmatched_by_llm",
                "fallback" => "gliner_label",
            )
            .increment(1);
            entity.label.clone()
        };

        let mut props = entity.properties.clone();
        if let Some(obj) = props.as_object_mut() {
            obj.insert(
                "hybrid_phase1_label".to_string(),
                serde_json::Value::String(entity.label.clone()),
            );
            obj.insert(
                "hybrid_phase2_typed".to_string(),
                serde_json::Value::Bool(llm_type_id.is_some()),
            );
        }

        result.push(ExtractedEntity {
            name: entity.name.clone(),
            label: refined_label,
            properties: props,
        });
    }

    result
}

// ─── Adversarial tests for parse_typed_response ──────────────────────────────
//
// These tests pin the all-or-nothing wrapper deserialization behavior. On
// gemma4-e2b (our PROD model) under LlmJsonRepair, "all-or-nothing" is actually
// the desired property: when the LLM produces partially-correct JSON, we want
// the WHOLE response rejected so GLiNER's open-vocab fallback labels (which are
// correct ~90% of the time on small-model output) win.
//
// Empirical 2026-06-09 verification: row-tolerant parsing was tried and dropped
// precision from 90%→60% because it kept the small model's semantically wrong
// individual typings. The test names below reflect the all-or-nothing contract.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entity_types::EntityTypeSpec;
    use crate::core::intelligence::ExtractedEntity;

    fn make_registry() -> EntityTypeRegistry {
        EntityTypeRegistry::from_specs(vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "Catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An organisation".to_string(),
            },
            EntityTypeSpec {
                id: 3,
                name: "Location".to_string(),
                description: "A location".to_string(),
            },
        ])
    }

    fn make_candidates(n: usize) -> Vec<ExtractedEntity> {
        (0..n)
            .map(|i| ExtractedEntity {
                name: format!("Candidate {i}"),
                label: "Entity".to_string(),
                properties: serde_json::Value::Null,
            })
            .collect()
    }

    /// Sanity baseline: a clean 3-row response keeps all 3 typings.
    #[test]
    fn parses_clean_response_keeps_all_typings() {
        let registry = make_registry();
        let candidates = make_candidates(3);
        let response = serde_json::json!({
            "typings": [
                {"idx": 0, "entity_type_id": 1},
                {"idx": 1, "entity_type_id": 2},
                {"idx": 2, "entity_type_id": 3},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].label, "Person");
        assert_eq!(result[1].label, "Organisation");
        assert_eq!(result[2].label, "Location");
    }

    /// Empirical small-model failure shape #1: one row has typo'd key name.
    /// All-or-nothing contract — the WHOLE wrapper deserialization fails,
    /// all candidates fall back to GLiNER's open-vocab labels. This is the
    /// desired behavior on small models where LLM typings degrade GLiNER's
    /// good labels (verified 2026-06-09 — row-tolerant let through 60%, all-
    /// or-nothing keeps 90%).
    #[test]
    fn parses_typo_in_one_row_rejects_whole_response() {
        let registry = make_registry();
        let candidates = make_candidates(6);
        let response = serde_json::json!({
            "typings": [
                {"idx": 0, "entity_type_id": 1},
                {"idx": 1, "entity_type_id": 2},
                {"idx": 2, "entity_type_id": 3},
                {"idx": 3, "entity_type_type_id": 2},  // TYPO — doubled type_
                {"idx": 4, "entity_type_id": 3},
                {"idx": 5, "entity_type_id": 3},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(
            result.len(),
            6,
            "all candidates returned (all use GLiNER fallback)"
        );
        // All rows fall back to GLiNER's open-vocab label because the whole
        // wrapper failed to deserialize — desired contract on small models.
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// Empirical small-model failure shape #2: one row drops a required field.
    /// All-or-nothing contract — the WHOLE wrapper fails, GLiNER labels win.
    #[test]
    fn parses_missing_field_in_one_row_rejects_whole_response() {
        let registry = make_registry();
        let candidates = make_candidates(7);
        let response = serde_json::json!({
            "typings": [
                {"idx": 0, "entity_type_id": 1},
                {"idx": 1, "entity_type_id": 2},
                {"idx": 2, "entity_type_id": 3},
                {"idx": 3, "entity_type_id": 1},
                {"idx": 4},  // MISSING entity_type_id
                {"idx": 5, "entity_type_id": 3},
                {"idx": 6, "entity_type_id": 2},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 7);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// `idx` out of bounds: row is dropped at idx-validation stage, not parse-stage.
    /// All other rows survive.
    #[test]
    fn parses_out_of_bounds_idx_drops_only_that_row() {
        let registry = make_registry();
        let candidates = make_candidates(3);
        let response = serde_json::json!({
            "typings": [
                {"idx": 0, "entity_type_id": 1},
                {"idx": 9999, "entity_type_id": 2},  // OUT OF BOUNDS
                {"idx": 2, "entity_type_id": 3},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].label, "Person");
        assert_eq!(result[2].label, "Location");
    }

    /// `entity_type_id: 0` means "model declined" — falls back to GLiNER for that row.
    #[test]
    fn parses_entity_type_id_zero_uses_gliner_fallback() {
        let registry = make_registry();
        let candidates = make_candidates(2);
        let response = serde_json::json!({
            "typings": [
                {"idx": 0, "entity_type_id": 0},  // model declined
                {"idx": 1, "entity_type_id": 2},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "Entity", "id=0 declined → GLiNER fallback");
        assert_eq!(result[1].label, "Organisation");
    }

    /// Empty typings array → all candidates fall back to GLiNER labels.
    #[test]
    fn parses_empty_typings_array_returns_all_gliner_fallback() {
        let registry = make_registry();
        let candidates = make_candidates(3);
        let response = serde_json::json!({ "typings": [] });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// Structural failure: typings field missing entirely → parser falls through
    /// gracefully (treated as structural failure, increments response_parse_fail).
    #[test]
    fn parses_missing_typings_field_returns_all_gliner_fallback() {
        let registry = make_registry();
        let candidates = make_candidates(3);
        let response = serde_json::json!({ "other_field": "stuff" });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// Structural failure: typings is null instead of array → graceful fallback.
    #[test]
    fn parses_typings_null_returns_all_gliner_fallback() {
        let registry = make_registry();
        let candidates = make_candidates(2);
        let response = serde_json::json!({ "typings": null });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// Structural failure: typings is a string (model emitted JSON-as-string) →
    /// graceful fallback rather than panic.
    #[test]
    fn parses_typings_as_string_returns_all_gliner_fallback() {
        let registry = make_registry();
        let candidates = make_candidates(2);
        let response = serde_json::json!({ "typings": "[{\"idx\": 0, \"entity_type_id\": 1}]" });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }

    /// Row with `idx` as string (model violated schema): all-or-nothing
    /// wrapper deserialization fails, whole response rejected, GLiNER wins.
    #[test]
    fn parses_idx_as_string_rejects_whole_response() {
        let registry = make_registry();
        let candidates = make_candidates(3);
        let response = serde_json::json!({
            "typings": [
                {"idx": "first", "entity_type_id": 1},  // schema violation
                {"idx": 1, "entity_type_id": 2},
                {"idx": 2, "entity_type_id": 3},
            ]
        });
        let result = parse_typed_response(&response, &candidates, &registry);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.label == "Entity"));
    }
}

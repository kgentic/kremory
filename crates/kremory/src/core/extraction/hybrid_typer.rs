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

        // Pass the model name so the builder's capability routing + metric
        // labels work. Force the LlmJsonRepair arm: empirical data
        // (2026-06-04) shows gemma4:e4b under constrained decoding
        // (FormatSchema) misclassifies proper nouns as Quantity for legal
        // text. Free-form JSON + llm_json repair gives noticeably better
        // semantic choices (93.8% vs 87.5% on legal_deposition). Accept
        // slightly higher arm-fail rate for better semantic accuracy.
        let model_name = self.llm.model().to_string();
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
    // Deserialise via the typed wrapper. Per [[llm-output-parse-loudly]] the
    // strict no-`#[serde(default)]` fields fail loudly on malformed input so
    // the fallback ladder can retry. If the response is too garbled to parse,
    // we fall through with an empty typing map and every candidate keeps its
    // GLiNER label (recorded via the candidate_unmatched_by_llm counter).
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

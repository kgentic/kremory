//! NER-based entity extraction using GLiNER models via direct ort inference.
//!
//! Replaces LLM-based extraction for entity span detection (50-500x faster).
//! LLM is still used for relationship extraction, dedup, and contradiction.
//!
//! This module implements GLiNER inference directly against `ort` rc.12 without
//! depending on the `gline-rs` crate (which pins an incompatible ort version).
//!
//! Input format derived from gline-rs source (fbilhaut/gline-rs):
//!   Prompt: `<<ENT>> type1 <<ENT>> type2 <<SEP>> word1 word2 ... wordN`
//!   where each word is independently sub-word tokenized.
//!
//! Model: `onnx-community/gliner_large-v2.1` (span mode)
//!
//! Inputs:
//!   input_ids:      [batch, seq_len]           int64
//!   attention_mask: [batch, seq_len]           int64
//!   words_mask:     [batch, seq_len]           int64
//!   text_lengths:   [batch, 1]                 int64
//!   span_idx:       [batch, num_spans, 2]      int64  — (start_word, end_word) pairs
//!   span_mask:      [batch, num_spans]         bool   — true for valid (non-padding) spans
//!
//! Output tensor `logits` shape: `[batch, num_spans, num_classes]`
//!   Each entry is a raw score (pre-sigmoid) for that span being that entity class.

#[cfg(feature = "ner")]
mod inner {
    use crate::core::intelligence::{
        EntityExtractor, ExtractedEntity, ExtractionContext, ExtractionResult,
    };
    use anyhow::Context as _;
    use ndarray::{Array2, Array3};
    use std::sync::Mutex;

    /// (span_idx [batch, num_spans, 2], span_mask [batch, num_spans], spans per text)
    type SpanTensors = (Array3<i64>, Array2<bool>, Vec<Vec<(usize, usize)>>);

    /// (input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask, spans per text)
    type GlinerInputs = (
        Array2<i64>,
        Array2<i64>,
        Array2<i64>,
        Array2<i64>,
        Array3<i64>,
        Array2<bool>,
        Vec<Vec<(usize, usize)>>,
    );

    // Special token IDs in GLiNER's vocabulary (DeBERTa-based tokenizer).
    // These are confirmed by gline-rs test assertions in encoded.rs:
    //   ids1 = [1, 128002, <entity_tokens>, 128003, <text_tokens>, 2, 0...]
    const TOKEN_BOS: i64 = 1;
    const TOKEN_EOS: i64 = 2;
    const TOKEN_ENT: i64 = 128002; // <<ENT>> marker before each entity type
    const TOKEN_SEP: i64 = 128003; // <<SEP>> marker separating entities from text

    // ONNX tensor names
    const INPUT_IDS: &str = "input_ids";
    const ATTENTION_MASK: &str = "attention_mask";
    const WORDS_MASK: &str = "words_mask";
    const TEXT_LENGTHS: &str = "text_lengths";
    const SPAN_IDX: &str = "span_idx";
    const SPAN_MASK: &str = "span_mask";
    const LOGITS: &str = "logits";

    // Maximum entity width in words (inclusive).
    // Matches gline-rs default Parameters::max_width = 12.
    const MAX_SPAN_WIDTH: usize = 12;

    // Default confidence threshold for accepting entity spans.
    // Matches gline-rs default Parameters (threshold = 0.5).
    const DEFAULT_THRESHOLD: f32 = 0.5;

    // No default entity types — the consumer must provide domain-specific types
    // via `PipelineConfig::allowed_entity_types`. GLiNER is a closed-vocabulary
    // model and cannot run open-ended; embedding domain assumptions here would
    // make the crate consumer-specific.

    // ---------------------------------------------------------------------------
    // Tokenizer helpers
    // ---------------------------------------------------------------------------

    /// Encode a single word into sub-word token IDs using the HuggingFace tokenizer,
    /// without adding BOS/EOS — matches how gline-rs calls its tokenizer word-by-word.
    fn encode_word(tokenizer: &tokenizers::Tokenizer, word: &str) -> anyhow::Result<Vec<i64>> {
        let enc = tokenizer
            .encode(word, false)
            .map_err(|e| anyhow::anyhow!("tokenise word '{}': {}", word, e))?;
        Ok(enc.get_ids().iter().map(|&id| id as i64).collect())
    }

    // ---------------------------------------------------------------------------
    // Input preparation
    // ---------------------------------------------------------------------------

    /// A single tokenized prompt ready to be packed into tensors.
    struct EncodedPrompt {
        /// Flat sequence of sub-word token IDs including BOS/EOS and padding zeros.
        input_ids: Vec<i64>,
        /// Attention mask (1 for real tokens, 0 for padding).
        attention_mask: Vec<i64>,
        /// Word-level mask: 0 for entity-prefix tokens, 1..=N for text word tokens.
        words_mask: Vec<i64>,
        /// Number of words in the text part (not counting entity prefix).
        text_length: i64,
    }

    /// Build encoded prompts for a single text against a list of entity types.
    ///
    /// Prompt token sequence (word-level before sub-word tokenization):
    /// `<<ENT>> type1 <<ENT>> type2 ... <<ENT>> typeK <<SEP>> word1 word2 ... wordN`
    ///
    /// This mirrors gline-rs `PromptInput` + `EncodedInput` exactly.
    fn encode_prompt(
        tokenizer: &tokenizers::Tokenizer,
        text: &str,
        entity_types: &[&str],
        max_seq_len: usize,
    ) -> anyhow::Result<EncodedPrompt> {
        // Split text into words (whitespace split — same as gline-rs RegexSplitter default).
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let text_len = text_words.len();

        // Build the entity prefix sub-word token sequence:
        // [<<ENT>>, type1_tok0, type1_tok1, ..., <<ENT>>, type2_tok0, ..., <<SEP>>]
        let mut entity_prefix_ids: Vec<i64> = Vec::new();
        for et in entity_types {
            entity_prefix_ids.push(TOKEN_ENT);
            entity_prefix_ids.extend(encode_word(tokenizer, et)?);
        }
        entity_prefix_ids.push(TOKEN_SEP);
        let entity_prefix_len = entity_prefix_ids.len();

        // Build the text sub-word token sequence and track word boundaries.
        // For each word we record how many sub-word tokens it produces so we can
        // set the words_mask (only the first sub-token of each word gets the word ID).
        let mut text_token_ids: Vec<i64> = Vec::new();
        // (word_index_1based, first_subtoken_position_in_text_tokens)
        let mut word_starts: Vec<(i64, usize)> = Vec::new();
        for (word_idx, word) in text_words.iter().enumerate() {
            let subwords = encode_word(tokenizer, word)?;
            word_starts.push(((word_idx + 1) as i64, text_token_ids.len()));
            text_token_ids.extend(subwords);
        }

        // Total token count = BOS + entity_prefix + text_tokens + EOS
        let total = 1 + entity_prefix_len + text_token_ids.len() + 1;

        // Allocate padded vectors
        let mut input_ids = vec![0i64; max_seq_len];
        let mut attention_mask = vec![0i64; max_seq_len];
        let mut words_mask = vec![0i64; max_seq_len];

        let mut pos: usize = 0;

        // BOS
        input_ids[pos] = TOKEN_BOS;
        attention_mask[pos] = 1;
        pos += 1;

        // Entity prefix tokens (words_mask stays 0 — these are not text words)
        for &id in &entity_prefix_ids {
            if pos >= max_seq_len {
                break;
            }
            input_ids[pos] = id;
            attention_mask[pos] = 1;
            pos += 1;
        }

        // The offset in the flat token array where text tokens begin.
        let text_offset = pos;

        // Text tokens
        // We need to place words_mask entries: for the first sub-token of each word,
        // words_mask[pos] = word_index (1-based). Other positions remain 0.
        let mut word_start_iter = word_starts.iter().peekable();
        for (text_tok_idx, &id) in text_token_ids.iter().enumerate() {
            if pos >= max_seq_len {
                break;
            }
            input_ids[pos] = id;
            attention_mask[pos] = 1;

            // Check if this sub-token position is the first token of a word.
            if let Some(&&(word_id, start_subtoken)) = word_start_iter.peek() {
                if text_tok_idx == start_subtoken {
                    words_mask[pos] = word_id;
                    word_start_iter.next();
                }
            }

            pos += 1;
        }

        // EOS
        if pos < max_seq_len {
            input_ids[pos] = TOKEN_EOS;
            attention_mask[pos] = 1;
            // words_mask stays 0 for EOS
            pos += 1;
        }

        // Sanity: pos should equal min(total, max_seq_len)
        let _ = pos;
        let _ = total;
        let _ = text_offset;

        Ok(EncodedPrompt {
            input_ids,
            attention_mask,
            words_mask,
            text_length: text_len as i64,
        })
    }

    /// Generate all (start_word, end_word) span pairs for a single text of `num_words` words.
    ///
    /// For each start word index, we generate ends from start up to
    /// min(start + max_width, num_words) — exclusive on the right because word indices are
    /// 0-based and inclusive ends are stored.  This mirrors gline-rs `EncodedInput` span
    /// generation exactly.
    ///
    /// Returns a Vec of (start, end) pairs where both indices are 0-based and inclusive.
    pub(crate) fn generate_spans(num_words: usize, max_width: usize) -> Vec<(usize, usize)> {
        // Fixed grid: exactly num_words × max_width entries.
        // The model reshapes span_idx to [batch, num_words, max_width, ...],
        // so we MUST produce exactly num_words * max_width spans in row-major order.
        let mut spans = Vec::with_capacity(num_words * max_width);
        for start in 0..num_words {
            for offset in 0..max_width {
                let end = start + offset;
                if end < num_words {
                    spans.push((start, end));
                } else {
                    spans.push((0, 0)); // padding — masked out by span_mask
                }
            }
        }
        spans
    }

    /// Build the span_idx and span_mask tensors for a batch.
    ///
    /// Produces a fixed grid of `num_words × MAX_SPAN_WIDTH` per text.
    /// The model internally reshapes to `[batch, num_words, max_width, hidden]`.
    /// Invalid spans (end >= num_words) have span_mask=false.
    ///
    /// Returns `(span_idx [batch, num_spans, 2] i64, span_mask [batch, num_spans] bool)`
    /// and the list of per-text span lists (needed by the decoder).
    fn build_span_tensors(word_counts: &[usize], max_width: usize) -> anyhow::Result<SpanTensors> {
        let batch = word_counts.len();

        let all_spans: Vec<Vec<(usize, usize)>> = word_counts
            .iter()
            .map(|&n| generate_spans(n, max_width))
            .collect();

        // Fixed grid: every text has exactly num_words * max_width spans.
        // For batching, pad to the max across the batch.
        let max_num_spans = all_spans.iter().map(|s| s.len()).max().unwrap_or(1).max(1);

        let mut span_idx_data: Vec<i64> = vec![0i64; batch * max_num_spans * 2];
        let mut span_mask_data: Vec<bool> = vec![false; batch * max_num_spans];

        for (b, (spans, &num_words)) in all_spans.iter().zip(word_counts.iter()).enumerate() {
            for (s, &(start, end)) in spans.iter().enumerate() {
                let base = b * max_num_spans * 2 + s * 2;
                span_idx_data[base] = start as i64;
                span_idx_data[base + 1] = end as i64;
                // Valid if end < num_words (padding entries have end >= num_words mapped to (0,0))
                // We check the original offset: position s within the grid,
                // offset = s % max_width, start_word = s / max_width
                let offset = s % max_width;
                let start_word = s / max_width;
                span_mask_data[b * max_num_spans + s] =
                    start_word < num_words && (start_word + offset) < num_words;
            }
        }

        let span_idx_arr = Array3::from_shape_vec((batch, max_num_spans, 2), span_idx_data)
            .context("failed to build span_idx array")?;
        let span_mask_arr = Array2::from_shape_vec((batch, max_num_spans), span_mask_data)
            .context("failed to build span_mask array")?;

        Ok((span_idx_arr, span_mask_arr, all_spans))
    }

    /// Build the six 2-D/3-D input tensors for a batch of texts.
    ///
    /// Returns `(input_ids, attention_mask, words_mask, text_lengths,
    ///           span_idx, span_mask, spans_per_text)`
    /// where `spans_per_text[i]` holds the (start, end) word pairs for text i.
    fn build_tensors(
        tokenizer: &tokenizers::Tokenizer,
        texts: &[&str],
        entity_types: &[&str],
    ) -> anyhow::Result<GlinerInputs> {
        // We use 512 as a hard cap (standard for DeBERTa-based models).
        const MAX_SEQ_LEN: usize = 512;

        let batch = texts.len();
        let mut word_counts: Vec<usize> = Vec::with_capacity(batch);
        let mut encoded: Vec<EncodedPrompt> = Vec::with_capacity(batch);

        for text in texts {
            let ep = encode_prompt(tokenizer, text, entity_types, MAX_SEQ_LEN)?;
            word_counts.push(ep.text_length as usize);
            encoded.push(ep);
        }

        // All prompts are already padded to MAX_SEQ_LEN.
        let mut input_ids_rows: Vec<i64> = Vec::with_capacity(batch * MAX_SEQ_LEN);
        let mut attention_mask_rows: Vec<i64> = Vec::with_capacity(batch * MAX_SEQ_LEN);
        let mut words_mask_rows: Vec<i64> = Vec::with_capacity(batch * MAX_SEQ_LEN);
        let mut text_lengths_rows: Vec<i64> = Vec::with_capacity(batch);

        for ep in &encoded {
            input_ids_rows.extend_from_slice(&ep.input_ids);
            attention_mask_rows.extend_from_slice(&ep.attention_mask);
            words_mask_rows.extend_from_slice(&ep.words_mask);
            text_lengths_rows.push(ep.text_length);
        }

        let input_ids_arr = Array2::from_shape_vec((batch, MAX_SEQ_LEN), input_ids_rows)
            .context("failed to build input_ids array")?;
        let attention_mask_arr = Array2::from_shape_vec((batch, MAX_SEQ_LEN), attention_mask_rows)
            .context("failed to build attention_mask array")?;
        let words_mask_arr = Array2::from_shape_vec((batch, MAX_SEQ_LEN), words_mask_rows)
            .context("failed to build words_mask array")?;
        let text_lengths_arr = Array2::from_shape_vec((batch, 1), text_lengths_rows)
            .context("failed to build text_lengths array")?;

        let (span_idx_arr, span_mask_arr, spans_per_text) =
            build_span_tensors(&word_counts, MAX_SPAN_WIDTH)
                .context("failed to build span tensors")?;

        Ok((
            input_ids_arr,
            attention_mask_arr,
            words_mask_arr,
            text_lengths_arr,
            span_idx_arr,
            span_mask_arr,
            spans_per_text,
        ))
    }

    // ---------------------------------------------------------------------------
    // Sigmoid
    // ---------------------------------------------------------------------------

    #[inline]
    pub(crate) fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // ---------------------------------------------------------------------------
    // Span decoding
    // ---------------------------------------------------------------------------

    /// A decoded entity span.
    #[derive(Debug)]
    pub(crate) struct DecodedSpan {
        pub(crate) word_start: usize,
        pub(crate) word_end: usize,
        pub(crate) class_idx: usize,
        pub(crate) score: f32,
    }

    /// Decode the span-mode `logits` tensor into entity spans.
    ///
    /// `logits` shape: `[batch, num_spans, num_classes]`
    ///   Each entry is a raw logit (pre-sigmoid) for that span being that entity class.
    ///
    /// `spans` is the list of (start_word, end_word) pairs that correspond to the
    /// span dimension of the logits tensor — produced by `generate_spans`.
    ///
    /// For each span, we take the class with the highest sigmoid score above `threshold`.
    /// Greedy non-overlapping selection is then applied (highest score wins).
    pub(crate) fn decode_logits(
        logits: &ndarray::ArrayViewD<f32>,
        batch_idx: usize,
        spans: &[(usize, usize)],
        num_classes: usize,
        threshold: f32,
    ) -> Vec<DecodedSpan> {
        let shape = logits.shape();
        // GLiNER span-mode output: [batch, num_words, max_width, num_classes]
        // The span grid is row-major: span index = word * max_width + offset
        let (num_words_dim, max_width_dim, num_classes_dim) = match shape.len() {
            4 => (shape[1], shape[2], shape[3]),
            3 => (shape[1], 1, shape[2]), // fallback for test tensors
            _ => return vec![],
        };
        let effective_classes = num_classes.min(num_classes_dim);

        let mut candidates: Vec<DecodedSpan> = Vec::new();

        for (span_pos, &(word_start, word_end)) in spans.iter().enumerate() {
            // Map span grid position to (word, offset) indices for 4D logits
            let word_idx = span_pos / max_width_dim;
            let offset_idx = span_pos % max_width_dim;
            if word_idx >= num_words_dim || offset_idx >= max_width_dim {
                continue;
            }
            // Skip padding spans (both mapped to (0,0) with end >= num_words)
            if word_start == 0 && word_end == 0 && span_pos > 0 {
                // Could be padding — check if this is a genuine (0,0) span
                // Only the very first span (0,0) at position 0 is genuine
                let expected_start = word_idx;
                let expected_end = word_idx + offset_idx;
                if expected_end != word_end || expected_start != word_start {
                    continue; // padding entry
                }
            }

            // Find the best class for this span
            let mut best_score = threshold;
            let mut best_class = usize::MAX;
            for class in 0..effective_classes {
                let score = if shape.len() == 4 {
                    sigmoid(logits[[batch_idx, word_idx, offset_idx, class]])
                } else {
                    sigmoid(logits[[batch_idx, span_pos, class]])
                };
                if score > best_score {
                    best_score = score;
                    best_class = class;
                }
            }
            if best_class != usize::MAX {
                candidates.push(DecodedSpan {
                    word_start,
                    word_end,
                    class_idx: best_class,
                    score: best_score,
                });
            }
        }

        // Greedy non-overlapping selection: sort by score descending, then remove overlaps.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected: Vec<DecodedSpan> = Vec::new();
        'outer: for span in candidates {
            for sel in &selected {
                // Overlapping if ranges intersect
                if span.word_start <= sel.word_end && span.word_end >= sel.word_start {
                    continue 'outer;
                }
            }
            selected.push(span);
        }

        selected
    }

    /// Reconstruct the entity text from the original text words.
    pub(crate) fn reconstruct_text(words: &[&str], word_start: usize, word_end: usize) -> String {
        let raw = words[word_start..=word_end.min(words.len().saturating_sub(1))].join(" ");
        // Strip trailing punctuation that gets attached to the last word
        raw.trim_end_matches(['.', ',', ';', ':']).to_string()
    }

    // ---------------------------------------------------------------------------
    // GlinerExtractor
    // ---------------------------------------------------------------------------

    /// GLiNER-based zero-shot NER extractor using direct ort inference.
    ///
    /// Entity types are specified at inference time — no retraining needed.
    /// Downloads `onnx-community/gliner_large-v2.1` from HuggingFace Hub on
    /// first use (cached locally by hf-hub).
    ///
    /// Uses the INT8 quantised model (`onnx/model_int8.onnx`, ~653 MB) for
    /// faster CPU inference.
    pub struct GlinerExtractor {
        session: Mutex<ort::session::Session>,
        tokenizer: tokenizers::Tokenizer,
        threshold: f32,
    }

    impl GlinerExtractor {
        /// Download the GLiNER model from HuggingFace Hub and build the ONNX session.
        ///
        /// Uses `onnx-community/gliner_large-v2.1` with the INT8 quantised model.
        /// The model and tokenizer are cached by hf-hub after the first download.
        pub fn new() -> anyhow::Result<Self> {
            Self::with_threshold(DEFAULT_THRESHOLD)
        }

        /// Create a GlinerExtractor with a custom confidence threshold.
        pub fn with_threshold(threshold: f32) -> anyhow::Result<Self> {
            let api = hf_hub::api::sync::Api::new().context("failed to init hf-hub API")?;

            let repo = api.model("onnx-community/gliner_large-v2.1".to_string());

            let model_path = repo.get("onnx/model_int8.onnx").context(
                "failed to download onnx/model_int8.onnx from onnx-community/gliner_large-v2.1",
            )?;

            let tokenizer_path = repo.get("tokenizer.json").context(
                "failed to download tokenizer.json from onnx-community/gliner_large-v2.1",
            )?;

            let session = ort::session::Session::builder()
                .context("failed to create ORT session builder")?
                .commit_from_file(&model_path)
                .context("failed to load GLiNER ONNX model")?;

            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                threshold,
            })
        }

        /// Synchronous inference path: build tensors, run ONNX, decode spans.
        fn extract_sync(
            &self,
            text: &str,
            entity_types: &[&str],
        ) -> anyhow::Result<Vec<ExtractedEntity>> {
            let texts = &[text];

            let (
                input_ids_arr,
                attention_mask_arr,
                words_mask_arr,
                text_lengths_arr,
                span_idx_arr,
                span_mask_arr,
                spans_per_text,
            ) = build_tensors(&self.tokenizer, texts, entity_types)
                .context("failed to build input tensors")?;

            let input_ids_ref = ort::value::TensorRef::from_array_view(input_ids_arr.view())
                .context("failed to create input_ids tensor")?;
            let attention_mask_ref =
                ort::value::TensorRef::from_array_view(attention_mask_arr.view())
                    .context("failed to create attention_mask tensor")?;
            let words_mask_ref = ort::value::TensorRef::from_array_view(words_mask_arr.view())
                .context("failed to create words_mask tensor")?;
            let text_lengths_ref = ort::value::TensorRef::from_array_view(text_lengths_arr.view())
                .context("failed to create text_lengths tensor")?;
            let span_idx_ref = ort::value::TensorRef::from_array_view(span_idx_arr.view())
                .context("failed to create span_idx tensor")?;
            let span_mask_ref = ort::value::TensorRef::from_array_view(span_mask_arr.view())
                .context("failed to create span_mask tensor")?;

            let decoded_spans = {
                let mut session = self
                    .session
                    .lock()
                    .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;

                let outputs = session
                    .run(ort::inputs![
                        INPUT_IDS      => input_ids_ref,
                        ATTENTION_MASK => attention_mask_ref,
                        WORDS_MASK     => words_mask_ref,
                        TEXT_LENGTHS   => text_lengths_ref,
                        SPAN_IDX       => span_idx_ref,
                        SPAN_MASK      => span_mask_ref
                    ])
                    .context("GLiNER ONNX inference failed")?;

                let logits: ndarray::ArrayViewD<f32> = outputs[LOGITS]
                    .try_extract_array()
                    .context("failed to extract logits tensor")?;

                // Diagnostic: print the actual output shape to stderr to aid debugging.
                // This verifies our span-mode assumption against the real model output.
                let shape_str = logits
                    .shape()
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!("[ner] GLiNER logits shape: [{}]", shape_str);

                let num_classes = entity_types.len();
                let batch_spans = &spans_per_text[0];

                decode_logits(&logits, 0, batch_spans, num_classes, self.threshold)
            };

            // Reconstruct entity text from original words.
            let text_words: Vec<&str> = text.split_whitespace().collect();

            let entities: Vec<ExtractedEntity> = decoded_spans
                .into_iter()
                .map(|span| {
                    let name = reconstruct_text(&text_words, span.word_start, span.word_end);
                    let label = entity_types
                        .get(span.class_idx)
                        .copied()
                        .unwrap_or("unknown")
                        .to_string();
                    ExtractedEntity {
                        name,
                        label,
                        properties: serde_json::json!({ "confidence": span.score }),
                    }
                })
                .collect();

            Ok(entities)
        }
    }

    impl EntityExtractor for GlinerExtractor {
        fn name(&self) -> &'static str {
            "gliner"
        }

        async fn extract<'a>(
            &'a self,
            text: &'a str,
            ctx: &'a ExtractionContext<'a>,
        ) -> crate::core::error::Result<ExtractionResult> {
            let start = std::time::Instant::now();

            // Consumer must provide entity types — GLiNER is closed-vocabulary.
            if ctx.allowed_entity_types.is_empty() {
                return Err(crate::core::error::Error::Config(
                    "GlinerExtractor requires allowed_entity_types — \
                     the model cannot run open-ended. Set PipelineConfig::allowed_entity_types \
                     with domain-specific entity labels."
                        .to_string(),
                ));
            }
            let effective_types: Vec<&str> = ctx
                .allowed_entity_types
                .iter()
                .map(String::as_str)
                .collect();

            // Remove excluded labels.
            let entity_types: Vec<&str> = effective_types
                .into_iter()
                .filter(|t| {
                    !ctx.excluded_entity_types
                        .iter()
                        .any(|excluded| excluded.as_str() == *t)
                })
                .collect();

            if entity_types.is_empty() {
                return Ok(ExtractionResult {
                    entities: vec![],
                    facts: vec![],
                });
            }

            let entities = self.extract_sync(text, &entity_types)?;

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let entity_count = entities.len();
            metrics::histogram!("rql.ner.extraction_ms").record(elapsed_ms);
            metrics::histogram!("rql.ner.entity_count").record(entity_count as f64);
            tracing::info!(elapsed_ms, entity_count, "kremory.ner.extraction");

            Ok(ExtractionResult {
                entities,
                facts: vec![],
            })
        }
    }
}

#[cfg(feature = "ner")]
pub use inner::GlinerExtractor;

#[cfg(all(test, feature = "ner"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::inner::{decode_logits, generate_spans, reconstruct_text, sigmoid};
    use super::GlinerExtractor;
    use crate::core::intelligence::{EntityExtractor, ExtractionContext};

    // -----------------------------------------------------------------------
    // sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_boundaries() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    // -----------------------------------------------------------------------
    // reconstruct_text
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_text_single_word() {
        let words = vec!["Alice"];
        assert_eq!(reconstruct_text(&words, 0, 0), "Alice");
    }

    #[test]
    fn test_reconstruct_text_multi_word() {
        let words = vec!["Alice", "works", "at", "Acme", "Corp"];
        assert_eq!(reconstruct_text(&words, 3, 4), "Acme Corp");
    }

    #[test]
    fn test_reconstruct_text_clamps_end() {
        let words = vec!["hello", "world"];
        assert_eq!(reconstruct_text(&words, 0, 99), "hello world");
    }

    // -----------------------------------------------------------------------
    // generate_spans
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_spans_single_word() {
        // 1 word, max_width=12 → fixed grid of 1×12 = 12 entries
        // Only (0,0) is valid; rest are (0,0) padding
        let spans = generate_spans(1, 12);
        assert_eq!(spans.len(), 12); // fixed grid: num_words × max_width
        assert_eq!(spans[0], (0, 0)); // valid: word 0, offset 0
                                      // Remaining 11 entries are padding (0,0) with end >= num_words
    }

    #[test]
    fn test_generate_spans_two_words() {
        // 2 words, max_width=3 → grid of 2×3 = 6 entries
        // word 0: (0,0)✓ (0,1)✓ (0,2=padding)
        // word 1: (1,1)✓ (1,2=padding) (1,3=padding)
        let spans = generate_spans(2, 3);
        assert_eq!(spans.len(), 6);
        assert_eq!(spans[0], (0, 0)); // word 0, offset 0
        assert_eq!(spans[1], (0, 1)); // word 0, offset 1
        assert_eq!(spans[2], (0, 0)); // word 0, offset 2 → padding (end=2 >= 2)
        assert_eq!(spans[3], (1, 1)); // word 1, offset 0
        assert_eq!(spans[4], (0, 0)); // word 1, offset 1 → padding (end=2 >= 2)
        assert_eq!(spans[5], (0, 0)); // word 1, offset 2 → padding (end=3 >= 2)
    }

    #[test]
    fn test_generate_spans_respects_max_width() {
        // 4 words, max_width=2 → grid of 4×2 = 8 entries
        // word 0: (0,0)✓ (0,1)✓
        // word 1: (1,1)✓ (1,2)✓
        // word 2: (2,2)✓ (2,3)✓
        // word 3: (3,3)✓ (3,4=padding)
        let spans = generate_spans(4, 2);
        assert_eq!(spans.len(), 8);
        assert_eq!(spans[0], (0, 0));
        assert_eq!(spans[1], (0, 1));
        assert_eq!(spans[2], (1, 1));
        assert_eq!(spans[3], (1, 2));
        assert_eq!(spans[4], (2, 2));
        assert_eq!(spans[5], (2, 3));
        assert_eq!(spans[6], (3, 3));
        assert_eq!(spans[7], (0, 0)); // padding
    }

    #[test]
    fn test_generate_spans_empty_text() {
        let spans = generate_spans(0, 12);
        assert!(spans.is_empty());
    }

    // -----------------------------------------------------------------------
    // decode_logits — span mode [batch, num_spans, num_classes]
    // -----------------------------------------------------------------------

    /// Helper: build a [1, num_spans, num_classes] ArrayD filled with `fill`.
    fn make_logits(num_spans: usize, num_classes: usize, fill: f32) -> ndarray::ArrayD<f32> {
        ndarray::ArrayD::from_shape_vec(
            vec![1, num_spans, num_classes],
            vec![fill; num_spans * num_classes],
        )
        .unwrap()
    }

    #[test]
    fn test_decode_logits_empty_below_threshold() {
        // All logits are -5.0 → sigmoid ≈ 0.007, well below threshold 0.5
        let spans = vec![(0usize, 0usize), (0, 1), (1, 1)];
        let arr = make_logits(3, 2, -5.0);
        let result = decode_logits(&arr.view(), 0, &spans, 2, 0.5);
        assert!(
            result.is_empty(),
            "all logits below threshold should yield no spans"
        );
    }

    #[test]
    fn test_decode_logits_single_word_entity() {
        // Span (0,0) with class 0 has a high score; all others are -5.
        // Spans: (0,0)=span0, (0,1)=span1, (1,1)=span2  →  3 spans, 1 class
        let spans_list = vec![(0usize, 0usize), (0, 1), (1, 1)];
        let num_spans = 3;
        let num_classes = 1;
        let mut data = vec![-5.0_f32; num_spans * num_classes];
        // span 0, class 0 → high score
        data[0] = 5.0;

        let arr = ndarray::ArrayD::from_shape_vec(vec![1, num_spans, num_classes], data).unwrap();

        let result = decode_logits(&arr.view(), 0, &spans_list, num_classes, 0.5);
        assert_eq!(result.len(), 1, "should find exactly one entity span");
        assert_eq!(result[0].word_start, 0);
        assert_eq!(result[0].word_end, 0);
        assert_eq!(result[0].class_idx, 0);
        assert!(result[0].score > 0.9);
    }

    #[test]
    fn test_decode_logits_multi_word_entity() {
        // Span (1,2) should be decoded as a 2-word entity.
        // Spans: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)  for 3 words, max_width=12
        let spans_list = generate_spans(3, 12);
        let num_spans = spans_list.len(); // 6
        let num_classes = 1;
        let mut data = vec![-5.0_f32; num_spans * num_classes];

        // Find the index of span (1,2) and give it a high score.
        let target_span_idx = spans_list
            .iter()
            .position(|&s| s == (1, 2))
            .expect("span (1,2) must be present");
        data[target_span_idx * num_classes] = 5.0;

        let arr = ndarray::ArrayD::from_shape_vec(vec![1, num_spans, num_classes], data).unwrap();

        let result = decode_logits(&arr.view(), 0, &spans_list, num_classes, 0.5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].word_start, 1);
        assert_eq!(result[0].word_end, 2);
        assert_eq!(result[0].class_idx, 0);
    }

    #[test]
    fn test_decode_logits_greedy_non_overlapping() {
        // Span (0,0) with class 0 has score ~0.99 (logit 5.0).
        // Span (0,1) with class 1 has score ~0.73 (logit 1.0) — overlaps with (0,0).
        // Span (2,2) with class 0 has score ~0.99 — non-overlapping, should also be selected.
        //
        // Expected result: (0,0)+class0 and (2,2)+class0 selected; (0,1)+class1 suppressed.
        let spans_list = generate_spans(3, 12);
        // spans: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        let num_spans = spans_list.len();
        let num_classes = 2;
        let mut data = vec![-5.0_f32; num_spans * num_classes];

        let idx_0_0 = spans_list.iter().position(|&s| s == (0, 0)).unwrap();
        let idx_0_1 = spans_list.iter().position(|&s| s == (0, 1)).unwrap();
        let idx_2_2 = spans_list.iter().position(|&s| s == (2, 2)).unwrap();

        // (0,0) class 0 → score ~0.99
        data[idx_0_0 * num_classes] = 5.0;
        // (0,1) class 1 → score ~0.73 (overlaps with (0,0))
        data[idx_0_1 * num_classes + 1] = 1.0;
        // (2,2) class 0 → score ~0.99 (non-overlapping)
        data[idx_2_2 * num_classes] = 5.0;

        let arr = ndarray::ArrayD::from_shape_vec(vec![1, num_spans, num_classes], data).unwrap();
        let result = decode_logits(&arr.view(), 0, &spans_list, num_classes, 0.5);

        assert_eq!(
            result.len(),
            2,
            "two non-overlapping spans should be selected"
        );
        // The greedy sort puts the highest scores first.
        let has_0_0 = result
            .iter()
            .any(|s| s.word_start == 0 && s.word_end == 0 && s.class_idx == 0);
        let has_2_2 = result
            .iter()
            .any(|s| s.word_start == 2 && s.word_end == 2 && s.class_idx == 0);
        let has_0_1 = result.iter().any(|s| s.word_start == 0 && s.word_end == 1);
        assert!(has_0_0, "span (0,0) class 0 should be selected");
        assert!(has_2_2, "span (2,2) class 0 should be selected");
        assert!(!has_0_1, "overlapping span (0,1) should be suppressed");
    }

    #[test]
    fn test_decode_logits_best_class_per_span() {
        // Span (0,0) has class 1 scoring higher than class 0.
        // Decoder must pick class 1 (best class per span), not class 0.
        let spans_list = vec![(0usize, 0usize)];
        let num_classes = 2;
        let mut data = vec![-5.0_f32; num_classes];
        data[0] = 1.0; // class 0: sigmoid ~0.73
        data[1] = 5.0; // class 1: sigmoid ~0.99

        let arr = ndarray::ArrayD::from_shape_vec(vec![1, 1, num_classes], data).unwrap();
        let result = decode_logits(&arr.view(), 0, &spans_list, num_classes, 0.5);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].class_idx, 1, "highest-scoring class should win");
        assert!(result[0].score > 0.9);
    }

    // -----------------------------------------------------------------------
    // empty allowed_entity_types must error
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_gliner_errors_without_entity_types() {
        let extractor = GlinerExtractor::new().expect("model load failed");
        let ctx = ExtractionContext::default(); // allowed_entity_types is empty
        let result = extractor.extract("Alice joined Acme Corp", &ctx).await;
        assert!(result.is_err(), "should fail when no entity types provided");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("allowed_entity_types"),
            "error should mention allowed_entity_types, got: {err}",
        );
    }
}

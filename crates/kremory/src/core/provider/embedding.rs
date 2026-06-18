// ---------------------------------------------------------------------------
// Concrete embedding provider implementations.
//
// Trait declarations (`EmbeddingProvider`, `DynEmbeddingProvider`, blanket impl)
// live in the parent module (`super` / `core::provider`).  This file contains
// the concrete types that implement those traits:
//
//   - `ArcEmbedder`                    — dyn-dispatch wrapper (production)
//   - `NullEmbeddingProvider`          — zero-vector stub
//   - `DeterministicEmbeddingProvider` — FNV-1a hashed vectors (production)
//   - `MockEmbeddingProvider`          — FNV-1a hashed vectors (test-utils gated)
//   - `OnnxEmbeddingProvider`          — ONNX all-MiniLM-L6-v2 (feature-gated)
//
// Re-exported from `super` — callers see the unchanged path
// `crate::core::provider::NullEmbeddingProvider` etc.
// ---------------------------------------------------------------------------

use std::future::Future;
use std::sync::Arc;

use crate::core::error::Result;

use super::{DynEmbeddingProvider, EmbeddingProvider};

// ---------------------------------------------------------------------------
// ArcEmbedder
// ---------------------------------------------------------------------------

/// Wrapper that implements `EmbeddingProvider` by delegating to
/// `Arc<dyn DynEmbeddingProvider>`. Used by the facade to turn a
/// `Arc<dyn DynEmbeddingProvider>` back into something generic code can use.
pub struct ArcEmbedder(pub Arc<dyn DynEmbeddingProvider>);

impl EmbeddingProvider for ArcEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        self.0.embed_dyn(text)
    }
    fn last_usage_tokens(&self) -> Option<u64> {
        self.0.last_usage_tokens_dyn()
    }
}

// ---------------------------------------------------------------------------
// NullEmbeddingProvider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NullEmbeddingProvider {
    pub dim: usize,
}

impl EmbeddingProvider for NullEmbeddingProvider {
    fn embed<'a>(&'a self, _text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move { Ok(vec![0.0_f32; dim]) }
    }
}

// ---------------------------------------------------------------------------
// DeterministicEmbeddingProvider — production Anthropic fallback
//
// Uses inline FNV-1a (matching MockEmbeddingProvider pattern; zero new dep).
// NOT gated — available in production builds. Named `Deterministic` (not Mock)
// per F-04 resolution: `MockEmbeddingProvider` remains test-utils gated.
// ---------------------------------------------------------------------------

/// Production-safe deterministic embedding provider.
///
/// Uses FNV-1a hashing to produce a fixed-dimension float vector from any string.
/// Embeddings are deterministic (same input → same output) but NOT semantic
/// (similar inputs produce unrelated vectors). Suitable only for structural recall
/// (exact-match entity lookup) where no embedding model API is available.
///
/// Used by `Memory::with_anthropic` — Anthropic has no embedding API.
///
/// Default `dim` = 384 — matches `NullEmbeddingProvider` and `OnnxEmbeddingProvider`
/// output dimension to preserve vector-column compatibility.
///
/// # Example
///
/// ```rust
/// use kremory::core::provider::DeterministicEmbeddingProvider;
/// let provider = DeterministicEmbeddingProvider::new(384);
/// ```
#[derive(Debug, Clone)]
pub struct DeterministicEmbeddingProvider {
    pub dim: usize,
}

impl DeterministicEmbeddingProvider {
    /// Create a new provider with the given output dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn hash_text(text: &str) -> u64 {
        // FNV-1a 64-bit inline (matches MockEmbeddingProvider pattern; zero new dep)
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        let base_hash = Self::hash_text(text);

        async move {
            // Per-dimension hash: XOR base_hash with a dimension-specific seed
            // BEFORE any multiply, so each element starts from a completely
            // different state. Adding `i` to a large hash product (10^19+) loses
            // precision in f32 conversion — all elements collapse to nearly the
            // same value. XOR-first avoids that collapse.
            //
            // LCG multiplier (Knuth) chosen to spread dimension index across all
            // bits without depending on FNV magnitude.
            const LCG_MUL: u64 = 6364136223846793005;
            let mut vec = Vec::with_capacity(dim);
            for i in 0..dim {
                // Step 1: mix dimension index into a seed that differs by bits,
                //         not magnitude.
                let dim_seed = (i as u64)
                    .wrapping_mul(LCG_MUL)
                    .wrapping_add(1442695040888963407);
                // Step 2: XOR with text hash so same dimension → different text →
                //         different value.
                let h = base_hash ^ dim_seed;
                // Step 3: one more FNV-like avalanche to spread the bits.
                let h = h
                    .wrapping_mul(1099511628211_u64)
                    .wrapping_add(dim_seed.wrapping_mul(2654435761));
                // Map u64 to [-1.0, 1.0]
                let val = (h as f32 / u64::MAX as f32) * 2.0 - 1.0;
                vec.push(val);
            }
            Ok(vec)
        }
    }
}

// ---------------------------------------------------------------------------
// MockEmbeddingProvider
//
// Gated: test-infra only — not part of the production public API.
// ---------------------------------------------------------------------------

/// Returns a deterministic embedding by using a simple FNV-1a hash over the
/// input bytes to seed each dimension.  Same text always → same vector; different
/// texts produce different vectors with overwhelming probability.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    pub dim: usize,
}

#[cfg(any(test, feature = "test-utils"))]
impl MockEmbeddingProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn hash_text(text: &str) -> u64 {
        // FNV-1a 64-bit
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed<'a>(&'a self, text: &'a str) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        let base_hash = Self::hash_text(text);

        async move {
            // Same per-dimension hash as DeterministicEmbeddingProvider — XOR-first
            // avoids the f32 precision-collapse that occurs when adding small `i` to a
            // large product (~10^19), which made all elements nearly identical.
            const LCG_MUL: u64 = 6364136223846793005;
            let mut vec = Vec::with_capacity(dim);
            for i in 0..dim {
                let dim_seed = (i as u64)
                    .wrapping_mul(LCG_MUL)
                    .wrapping_add(1442695040888963407);
                let h = base_hash ^ dim_seed;
                let h = h
                    .wrapping_mul(1099511628211_u64)
                    .wrapping_add(dim_seed.wrapping_mul(2654435761));
                let val = (h as f32 / u64::MAX as f32) * 2.0 - 1.0;
                vec.push(val);
            }
            Ok(vec)
        }
    }
}

// ---------------------------------------------------------------------------
// OnnxEmbeddingProvider
// ---------------------------------------------------------------------------

/// Real embedding provider backed by all-MiniLM-L6-v2 (ONNX).
/// Produces 384-dimensional L2-normalised sentence embeddings.
/// Requires the `embeddings` feature flag.
#[cfg(feature = "embeddings")]
pub struct OnnxEmbeddingProvider {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "embeddings")]
impl OnnxEmbeddingProvider {
    /// Download the ONNX model + tokenizer from HuggingFace Hub (cached after
    /// first call) and build the inference session.
    pub fn new() -> Result<Self> {
        use anyhow::Context as _;

        let api = hf_hub::api::sync::Api::new().context("failed to init hf-hub API")?;

        let onnx_repo = api.model("optimum/all-MiniLM-L6-v2".to_string());
        let model_path = onnx_repo
            .get("model.onnx")
            .context("failed to download model.onnx from optimum/all-MiniLM-L6-v2")?;

        let tokenizer_repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
        let tokenizer_path = tokenizer_repo
            .get("tokenizer.json")
            .context("failed to download tokenizer.json")?;

        let session = ort::session::Session::builder()
            .context("failed to create ORT session builder")?
            .commit_from_file(&model_path)
            .context("failed to load ONNX model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
        })
    }

    /// Synchronous embed — tokenize, run ONNX inference, mean pool, L2 normalise.
    fn embed_sync(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use anyhow::Context as _;
        use ndarray::Array2;

        const MAX_SEQ_LEN: usize = 128;

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenisation failed: {e}"))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len().min(MAX_SEQ_LEN);

        let input_ids_data: Vec<i64> = ids[..seq_len].iter().map(|&x| x as i64).collect();
        let attention_mask_data: Vec<i64> = attention_mask[..seq_len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let token_type_ids_data: Vec<i64> = type_ids[..seq_len].iter().map(|&x| x as i64).collect();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids_data)
            .context("failed to build input_ids array")?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask_data)
            .context("failed to build attention_mask array")?;
        let token_type_ids_arr = Array2::from_shape_vec((1, seq_len), token_type_ids_data)
            .context("failed to build token_type_ids array")?;

        let input_ids_ref = ort::value::TensorRef::from_array_view(input_ids_arr.view())
            .context("failed to create input_ids tensor")?;
        let attention_mask_ref = ort::value::TensorRef::from_array_view(attention_mask_arr.view())
            .context("failed to create attention_mask tensor")?;
        let token_type_ids_ref = ort::value::TensorRef::from_array_view(token_type_ids_arr.view())
            .context("failed to create token_type_ids tensor")?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids"      => input_ids_ref,
                "attention_mask" => attention_mask_ref,
                "token_type_ids" => token_type_ids_ref
            ])
            .context("ONNX inference failed")?;

        let hidden: ndarray::ArrayViewD<f32> = outputs["last_hidden_state"]
            .try_extract_array()
            .context("failed to extract last_hidden_state")?;

        let shape = hidden.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 3,
            "expected 3-D hidden state, got {:?}",
            shape
        );
        let (_batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);

        // Mean pooling (attention-mask-weighted)
        let mut pooled = vec![0.0_f32; hidden_size];
        let mut mask_sum = 0.0_f32;
        for t in 0..seq {
            let m = attention_mask_arr[[0, t]] as f32;
            mask_sum += m;
            for d in 0..hidden_size {
                pooled[d] += hidden[[0, t, d]] * m;
            }
        }
        if mask_sum > 0.0 {
            for v in &mut pooled {
                *v /= mask_sum;
            }
        }

        // L2 normalise
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingProvider for OnnxEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_sync(text)
            .map_err(crate::core::error::Error::from)
    }
}

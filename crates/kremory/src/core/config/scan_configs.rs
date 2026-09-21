/// The content type of a document being ingested into the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    /// Plain unstructured text (e.g. transcripts, notes).
    Text,
    /// A discrete conversational message (e.g. chat turn, email).
    Message,
    /// Structured JSON payload; entity extraction is schema-aware.
    Json,
    /// A standalone document (e.g. markdown file, report, wiki page).
    /// Used by `ingest_document()` — stored as a searchable entity with
    /// full-text embedding in addition to extracted sub-entities.
    Document,
}

/// LLM-extraction-prompt-window parameters.
///
/// **This is kind-2 chunking only** — slices an oversized episode body into prompt-sized
/// windows so the extractor LLM can read it within its context budget. Slices are throwaway
/// and never enter storage/embedding. See `core/extraction_window.rs` module docstring
/// for the kind-1 vs kind-2 distinction.
#[derive(Debug, Clone)]
pub struct ExtractionWindowConfig {
    /// Default: 100 words. Shorter text rarely benefits from splitting; below
    /// this the overhead of extra chunks exceeds the gain.  Graphiti ratio:
    /// min/max ≈ 33%.
    pub min_words: usize,

    /// Default: 0.15. Empirically, chunks with >15% of tokens being entity
    /// spans lose inter-entity context when kept whole; splitting at this
    /// threshold keeps entity co-occurrence coherent.
    pub density_threshold: f64,

    /// Default: 300 words (~1500 chars, ~400 BPE tokens).  Sized so 3 chunks
    /// fit in the default 4096-token context with room for system prompt,
    /// query, and generation.  Optimised for latency on the real-time meeting
    /// assistant path.  Must stay aligned with `max_chunk_chars` in
    /// the host application's pipeline config (1500 chars).
    ///
    /// Increase it via `PipelineConfig::builder().max_words(n)` (verified against
    /// the setter at `config.rs:767`; siblings: `min_words`, `overlap_words`,
    /// `density_threshold`).
    /// This previously said to set `LLM_CONTEXT_SIZE` +
    /// `CHUNK_MAX_WORDS` env vars. **Neither has any effect on the pipeline** —
    /// `PipelineConfig` builds this struct via `Default` (see its `Default` impl
    /// below), never via [`from_env`](Self::from_env), and `LLM_CONTEXT_SIZE`
    /// appears in no source file at all. The builder setters are the supported path.
    pub max_words: usize,

    /// Number of words from the end of `chunk[i]` to prepend to `chunk[i+1]`.
    /// Default: 50 words.  Graphiti uses 200/3000 (6.7%); ours is 50/300
    /// (16.7%) — slightly higher overlap compensates for smaller chunks.
    pub overlap_words: usize,
}

impl ExtractionWindowConfig {
    /// Build from environment variables, falling back to sensible defaults.
    ///
    /// ⚠️ **The kremory pipeline does NOT call this.**
    /// `PipelineConfig` constructs its `extraction_window` via `Default`
    /// (`config.rs:708`), so **setting the env vars below changes nothing** unless a
    /// consumer calls `from_env()` themselves and passes the result in. This method
    /// has zero callers in the workspace. The supported way to tune chunking is the
    /// `PipelineConfig` builder (`max_words` / `min_words` / `overlap_words` /
    /// `density_threshold`).
    ///
    /// Kept rather than deleted because it is `pub` and works correctly *if called* —
    /// but it is documented here as opt-in, not as ambient configuration, because the
    /// previous wording sent users to set env vars that silently did nothing.
    ///
    /// | Env var | Default | Rationale |
    /// |---------|---------|-----------|
    /// | `CHUNK_MAX_WORDS` (or legacy `CHUNK_MAX_TOKENS`) | 300 | ~1500 chars, fits 3 chunks in 4096-ctx prompt |
    /// | `CHUNK_MIN_WORDS` (or legacy `CHUNK_MIN_TOKENS`) | 100 | Don't chunk short text (Graphiti min/max ≈ 33%) |
    /// | `CHUNK_OVERLAP_WORDS` (or legacy `CHUNK_OVERLAP_TOKENS`) | 50 | Context continuity between chunks |
    /// | `CHUNK_DENSITY_THRESHOLD` | 0.15 | Entity-dense regions trigger splitting |
    pub fn from_env() -> Self {
        Self {
            max_words: env_usize_or("CHUNK_MAX_WORDS", "CHUNK_MAX_TOKENS", 300),
            min_words: env_usize_or("CHUNK_MIN_WORDS", "CHUNK_MIN_TOKENS", 100),
            overlap_words: env_usize_or("CHUNK_OVERLAP_WORDS", "CHUNK_OVERLAP_TOKENS", 50),
            density_threshold: env_f64("CHUNK_DENSITY_THRESHOLD", 0.15),
        }
    }
}

impl Default for ExtractionWindowConfig {
    fn default() -> Self {
        Self {
            min_words: 100,
            density_threshold: 0.15,
            max_words: 300,
            overlap_words: 50,
        }
    }
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read `new_var`, falling back to the legacy `old_var` name, then `default`.
///
/// Field-rename compatibility shim (word-based `ExtractionWindowConfig` fields
/// were renamed from `*_tokens` to `*_words` — see rename PR): existing
/// deployments setting `CHUNK_MAX_TOKENS` etc. keep working unchanged.
fn env_usize_or(new_var: &str, old_var: &str, default: usize) -> usize {
    std::env::var(new_var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| env_usize(old_var, default))
}

fn env_f64(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// MinHash / LSH parameters used for near-duplicate entity detection.
#[derive(Debug, Clone)]
pub struct MinHashConfig {
    /// Default: 32. Rationale: 32 permutations give ~3% Jaccard estimation
    /// error at manageable memory cost (~256 bytes per sketch).
    pub num_permutations: usize,

    /// Default: 3. Rationale: character 3-grams balance sensitivity to small
    /// edits (typos, abbreviations) against noise from very short substrings.
    pub shingle_size: usize,

    /// Default: 4. Rationale: with 32 permutations and bands of 4, we get
    /// 8 bands, yielding a good probability curve around the 0.9 threshold
    /// (P(candidate) ≈ 0.99 at threshold, ~0.01 false-positive rate at 0.5).
    pub band_size: usize,

    /// Default: 0.9. Rationale: entity surface forms that share ≥90% of their
    /// 3-gram shingles are treated as the same entity; below 0.9 too many
    /// distinct entities collapse.
    pub jaccard_threshold: f64,
}

impl Default for MinHashConfig {
    fn default() -> Self {
        Self {
            num_permutations: 32,
            shingle_size: 3,
            band_size: 4,
            jaccard_threshold: 0.9,
        }
    }
}

/// Entropy-based pre-filter that gates whether a token is fed into MinHash.
/// Low-entropy strings (e.g. "Inc.", "Ltd.") are common suffixes that would
/// inflate false-positive collision rates if hashed directly.
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Default: 6. Rationale: entity names shorter than 6 characters are almost
    /// always abbreviations or stop-words; hashing them adds noise without value.
    pub min_name_length: usize,

    /// Default: 2. Rationale: a single-token string is almost never a meaningful
    /// multi-word entity; requiring at least 2 whitespace-delimited tokens
    /// removes most numeric codes and single-letter abbreviations.
    pub min_token_count: usize,

    /// Default: 1.5. Rationale: Shannon entropy of 1.5 bits corresponds roughly
    /// to strings that repeat fewer than 3 distinct characters — effectively
    /// keyboard-mash or padded identifiers that carry no semantic content.
    pub entropy_threshold: f64,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        }
    }
}

/// What [`Engine::ingest_with`](crate::core::ingest::pipeline::ingest_with)'s
/// secret scan (TD-061) does with a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecretScanMode {
    /// Leave the episode text untouched; record the hit (structured log +
    /// counter, see `core::secret_scan`) so it is never silently dropped.
    /// **Default** — matches the register's "flag+log" resolution: redaction
    /// mutates stored content, which is a bigger behavioural change than most
    /// consumers opt into implicitly, so the safer default surfaces the leak
    /// without altering what was ingested.
    #[default]
    FlagOnly,
    /// Replace every detected secret span in the episode text with a
    /// non-reversible marker (`[REDACTED_SECRET]`) BEFORE the episode is
    /// inserted, extracted, or embedded — the redacted text is what gets
    /// persisted. A hit is still logged + counted (same as `FlagOnly`); this
    /// mode only changes what ends up in storage, not whether the hit is
    /// observable.
    Redact,
}

/// Ingest-boundary secret/API-key/token detection (TD-061).
///
/// Gates [`core::secret_scan::scan_ingest_text`](crate::core::secret_scan::scan_ingest_text),
/// called from `ingest_with` on the raw episode text BEFORE it is inserted,
/// extracted, or embedded (see that module's doc comment for the crates.io
/// evaluation behind the underlying scanner).
#[derive(Debug, Clone)]
pub struct SecretScanConfig {
    /// Default: `true`. `false` skips the scan entirely (byte-identical to
    /// pre-TD-061 behaviour) — the off switch is required for the same reason
    /// every other ingest-gating knob on this struct has one: a lever with no
    /// control arm cannot be measured, and a consumer whose corpus is
    /// synthetic test fixtures full of fake-but-shaped secrets (JWTs, PEM
    /// blocks in fixture data) needs a way to turn the noise off.
    pub enabled: bool,
    /// Default: [`SecretScanMode::FlagOnly`]. See that enum's variants for
    /// the flag-vs-redact tradeoff.
    pub mode: SecretScanMode,
}

impl Default for SecretScanConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: SecretScanMode::FlagOnly,
        }
    }
}

use std::time::Duration;

use crate::core::error::{Error, Result};

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
/// and never enter storage/embedding. See `core/extraction_window.rs` module docstring and
/// ADR-Phase-D.0:88 for the kind-1 vs kind-2 distinction.
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
    /// `density_threshold`). CFG-2
    /// (V1-CANONICAL §6c): this previously said to set `LLM_CONTEXT_SIZE` +
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
    /// ⚠️ **The kremory pipeline does NOT call this.** CFG-2 (V1-CANONICAL §6c):
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

/// Hybrid search parameters controlling how BM25 and vector scores are combined.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Default: 0.5. Rationale: equal weighting between BM25 and vector search
    /// is a safe baseline; BM25 handles keyword precision while vector search
    /// handles semantic similarity. Must sum to 1.0 with `vector_weight`.
    pub bm25_weight: f64,

    /// Default: 0.5. Rationale: see `bm25_weight`. Must sum to 1.0 with
    /// `bm25_weight`.
    pub vector_weight: f64,

    /// Default: 60. Rationale: the standard RRF constant from Cormack et al.
    /// (2009); k=60 was shown to be near-optimal across a wide range of
    /// retrieval tasks.
    pub rrf_k: usize,

    /// Default: 10. Rationale: top-10 is the conventional precision@k cut-off
    /// for downstream synthesis; returning more increases context window cost
    /// without commensurate quality gain.
    pub top_k: usize,

    // ── recall-v2 post-RRF scoring axes (recall-v2-architecture-2026-07-03) ──
    //
    // All fields below feed the staged post-RRF boost pipeline in
    // `core::context::Engine::contextualize`. Every default preserves TODAY's
    // behaviour: only `graph_degree_weight` is a live tested axis (its default
    // is its already-shipped constant, NOT 0.0); all NEW axes default to their
    // no-op value so recall output is byte-identical to pre-change at defaults.
    /// Weight of the additive graph-degree bonus (recall-v2 Phase 2a; formerly
    /// the `search::GRAPH_DEGREE_WEIGHT` const). **Default 0.05, NOT 0.0** —
    /// graph-degree is the one already-live boost axis (TD-066 Change 2), so
    /// `0.0` would silently disable a shipped, tested feature. "Today's
    /// behaviour" for this axis *is* the 0.05 bonus. Bounded contribution:
    /// `weight * min(degree/saturation, 1)`.
    pub graph_degree_weight: f32,

    /// Weight of the additive temporal-recency boost (recall-v2 Phase 2b).
    /// **Default 0.0 = off** — new axis, behaviourally neutral until an
    /// eval-calibrated value is set. Additive + bounded `[0, weight]`, matching
    /// graph-degree so a single `.min(1.0)` clamp covers both (no second
    /// normalization pass).
    pub temporal_weight: f32,

    /// Decay rate `lambda` for the temporal boost's `exp(-lambda * age_days)`
    /// (recall-v2 Phase 2b). Inert while `temporal_weight == 0.0`. Default 0.01
    /// ≈ a ~69-day half-life; a placeholder pending Phase-7 eval calibration.
    pub temporal_decay_lambda: f64,

    /// Minimum post-boost score a seed must reach to survive to output
    /// (recall-v2 Phase 5, Decision 6). **Default 0.0 = no-op** (nothing is
    /// floored out). Applied AFTER all boosts, BEFORE the access-count
    /// increment, so dropped entities are never counted as recalled.
    pub floor_threshold: f32,

    /// Max BFS hop depth for the 1-hop expansion (recall-v2 Phase 4, TD-056).
    /// **Default 1 = today's behaviour.** Widening this without an in-BFS
    /// visited cap (`expansion_fan_out_cap`) reintroduces hub-explosion (spec
    /// R1), so the two ship together.
    pub expansion_hop_bound: u32,

    /// Cap on entities visited during a seed's BFS expansion (recall-v2 Phase 4,
    /// TD-056). **Default 8 = the former `context::MAX_NEIGHBOURS_PER_SEED`
    /// const.** Threaded into the BFS itself (not just the caller-side trim) so
    /// it bounds hop≥2 traversal, not only the final result set.
    pub expansion_fan_out_cap: usize,

    /// Score-decay factor applied to a 1-hop neighbour relative to the seed
    /// that surfaced it (recall-v2 Phase 4; formerly the
    /// `context::NEIGHBOUR_SCORE_DECAY` const). **Default 0.5** — mid-range of
    /// the literature's `[0.3, 0.7]` weighted-expansion band (HippoRAG PPR
    /// damping is also 0.5).
    pub neighbour_score_decay: f32,

    /// Weight of the additive graph-**proximity** boost — ADR-062 (axis C),
    /// build-entry spec `axis-c-read-time-relevance-spec-2026-07-01.md`,
    /// ADR-067 Phase 3. **Default 0.0 = off** — new axis, behaviourally
    /// neutral until an eval-calibrated value is set (same shape as
    /// `temporal_weight`).
    ///
    /// ADR-067 **Amendment 1** (2026-07-20) supersedes ADR-062's literal
    /// "post-RRF multiplicative boost" text: this axis ships **additive +
    /// bounded `[0, weight]`**, composed into the SAME single `.min(1.0)`
    /// clamp as `graph_degree_weight`/`temporal_weight` — the amendment's own
    /// migration trigger ("axis-C proximity lands and would coexist with the
    /// additive axes") is this axis landing, and the amendment already
    /// decided the outcome: stay additive, migrate all axes to a normalized
    /// multiplicative chain later ONLY if a future eval shows additive
    /// underperforms. No dead multiplicative plumbing ships.
    ///
    /// Distinct signal from `graph_degree_weight`: degree reads the seed's
    /// OWN 1-hop neighbour count (`expansion_hop_bound`, default 1);
    /// proximity reads a WIDER, independently-bounded walk
    /// (`proximity_hop_bound`, default 2) — "is this seed near a dense
    /// cluster it isn't directly part of", not "how many direct facts does
    /// it have". `<= 0.0` skips the second graph query entirely (no latency
    /// cost when off, not just a no-op score).
    pub proximity_weight: f32,

    /// Bounded-hop distance for the graph-proximity walk (ADR-062 §5 Q2/Q3).
    /// **Default 2** — the spec's own starting recommendation: `hop_bound=1`
    /// collapses to a near-duplicate of `graph_degree_weight`'s own 1-hop
    /// signal; `hop_bound=3+` widens cost without a calibrated benefit yet.
    /// Inert while `proximity_weight <= 0.0` (the second graph query is
    /// skipped entirely, not just discounted).
    pub proximity_hop_bound: u32,

    /// In-BFS visited-entity cap for the proximity walk (ADR-062 §8/ASMP-001,
    /// spike criterion 2a). **Default 8** — matches `expansion_fan_out_cap`'s
    /// default. `get_neighbours_at`'s `max_visited` (ADR-067 Amendment 2)
    /// early-exits the BFS once the cap is hit, bounding a high-degree hub
    /// seed's worst-case cost at `proximity_hop_bound >= 2` — the same
    /// mechanism TD-056's multi-hop expansion cap already uses, reused here
    /// rather than adding a new capped traversal primitive.
    pub proximity_fan_out_cap: usize,

    /// Per-stream weight applied to the ADR-072 `content_search` BM25 stream's
    /// RRF contribution in `search::rrf_fuse_with_content` (TD-066 Increment
    /// 2, `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md`
    /// §3 Increment 2). **Default 1.0 = today's equal-weight RRF fusion**
    /// (Increment 1's behaviour, byte-identical) — this is a config-only
    /// calibration knob, not a new algorithm. `content`-alone beats
    /// equal-weight `hybrid` on 3 of 4 LoCoMo categories (spec §1.1), so a
    /// value `> 1.0` favours the content stream over the entity-graph stream
    /// in the fused ranking; `<= 0.0` degrades content's contribution to
    /// (effectively) zero, without removing the stream's dedup/id-space
    /// participation. The calibrated value is a Phase-7-style bench-sweep
    /// output, not a value guessed at design time — ship the knob, measure
    /// the value.
    pub content_stream_weight: f32,

    /// TD-136 (dense episode retrieval): when `true`, recall runs a dense
    /// (embedding) vector arm over `episodes.embedding` and RRF-fuses it with
    /// the ADR-072 BM25 `content_search` episode arm BEFORE the fused content
    /// stream enters `search::rrf_fuse_with_content`. Also gates ingest-time
    /// episode embedding (`core/ingest/pipeline`). **Default `false` = today's
    /// BM25-only behaviour, byte-identical** — the whole dense arm (read +
    /// write) is inert until enabled, so `content-search` builds keep the exact
    /// pre-TD-136 recall + ingest surface at defaults. Wired from the
    /// `KREMORY_EPISODE_DENSE` boot override
    /// (`facade::providers::search_env_overrides`), mirroring
    /// `content_stream_weight`'s `KREMORY_CONTENT_WEIGHT` A/B knob, so the
    /// dense-vs-BM25 comparison costs a server restart, not a rebuild.
    pub episode_dense_enabled: bool,

    /// TD-139 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-139 DoD item
    /// 2): when `true`, recall runs a THIRD dense (embedding) arm —
    /// `TemporalGraph::vector_search_facts` over `facts.embedding` — and
    /// RRF-fuses it into the entity+content stream `fuse_content_stream`
    /// already produces. Mirrors `episode_dense_enabled`'s rollout shape
    /// exactly (gate → degrade-on-failure → metrics → env → builder), but is
    /// its OWN knob, not a sub-case of the episode arm: facts and episodes
    /// are fused via different mechanisms
    /// (`core::search::rrf_fuse_with_facts`, not `rrf_fuse_content_streams`
    /// — see that fn's doc comment for why a fact cannot be represented as a
    /// `ContentPassage`). **Default `false` = today's behaviour, byte-
    /// identical** — facts remain reachable ONLY via 1-hop expansion from an
    /// already-matched entity (TD-139's "why it matters" section) until this
    /// is enabled. Ingest-time fact embedding (`ingest_with.rs:~1774`,
    /// `deferred.rs:~411`) is UNCHANGED by this knob — that write path
    /// already runs unconditionally (TD-139 discovery: "we already do all of
    /// that except the recall wiring"). Wired from the `KREMORY_FACT_DENSE`
    /// boot override (`facade::providers::search_env_overrides`), mirroring
    /// `episode_dense_enabled`'s `KREMORY_EPISODE_DENSE`, so the fact-dense
    /// A/B costs a server restart, not a rebuild.
    ///
    /// ⚠️ TD-137 risk (fact/predicate quality): noisy predicates
    /// (`greeting`, `session_timestamp`) embed as junk and may surface as
    /// dense-arm noise, potentially making the fact stream's already-net-
    /// negative shape WORSE — precisely why this ships default-OFF behind a
    /// measured gate, never defaulted on without an A/B.
    pub fact_dense_enabled: bool,

    /// TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143): when
    /// `true`, every embed call on the recall/ingest search surface is
    /// prefixed with nomic-embed-text's REQUIRED asymmetric task prefix —
    /// `search_document: ` for text that gets STORED into
    /// `entities`/`episodes`/`facts.embedding` (ingest-time writes, plus the
    /// backfill subcommand), `search_query: ` for text that is a QUERY
    /// compared (via vector search) against that stored corpus (recall +
    /// entity-resolution probes). See `core::embed_prefix` for the two
    /// helpers every call site routes through and the full call-site
    /// inventory. **Default `false` = today's bare-text embedding,
    /// byte-identical** — nomic-embed-text is an ASYMMETRIC model; without
    /// the prefix, queries and passages collapse into the same task space,
    /// which TD-143 identifies as the prime suspect for the measured
    /// vocabulary/abstraction breadth gap (`.ai-docs/research/
    /// first-stage-retrieval-landscape-2026-07-27.md`).
    ///
    /// ⚠️ Nomic-specific, NOT a generic embedder feature: a consumer using a
    /// different (non-nomic) BYOM embedder must never flip this on — the
    /// prefix is meaningless (or actively harmful) for a symmetric or
    /// differently-prefixed model. This is why the prefix is applied at CALL
    /// SITES via `core::embed_prefix`, not baked into the `EmbeddingProvider`
    /// / `DynEmbeddingProvider` trait contract, which stays provider-agnostic.
    ///
    /// ⚠️ CORRECTNESS — flipping this on an EXISTING corpus makes every
    /// already-stored embedding stale (a document-prefixed write and an
    /// unprefixed write occupy DIFFERENT task spaces — cosine similarity
    /// between them is meaningless, not just degraded). Never flip this knob
    /// against a live/measured database. The safe sequence: (1) flip on a
    /// FRESH corpus copy, (2) re-embed it in full — episodes via
    /// `Memory::reembed_all_episode_embeddings` (free + local against an
    /// Ollama embedder; NOT `backfill_episode_embeddings`, whose `WHERE
    /// embedding IS NULL` paging only fills a gap and can never overwrite a
    /// row that already has a vector — see that method's doc), entities/facts
    /// via a full re-ingest — (3) THEN measure.
    /// Wired from the `KREMORY_EMBED_TASK_PREFIX` boot override
    /// (`facade::providers::search_env_overrides`), mirroring
    /// `episode_dense_enabled`'s `KREMORY_EPISODE_DENSE`.
    pub embed_task_prefix_enabled: bool,

    /// Reranker latency lever 1 (`.ai-docs/tech-debt/` cross-encoder latency
    /// spike, 2026-07-28): caps the SUMMARY portion of each rerank candidate's
    /// text (`entity_name + " " + summary`, `facade::recall::apply_rerank_with`)
    /// at this many `char`s before it is handed to the cross-encoder.
    /// `entity_name` is never truncated. **Default `0` = unlimited (today's
    /// behaviour, byte-identical)** — a content-fused candidate's `summary` can
    /// carry the FULL episode body, and BGE-reranker-base's own tokenizer caps
    /// at 512 tokens, so anything beyond that is silently truncated by the
    /// tokenizer today anyway; this knob truncates on a cheap `char` boundary
    /// BEFORE tokenization, which is the measured majority of reranker latency
    /// (attention cost grows ~quadratically with sequence length). Truncation
    /// is on a Unicode scalar-value boundary (`str::chars`), never a byte
    /// index, so it cannot panic or produce invalid UTF-8 on multi-byte input.
    /// Wired from the `KREMORY_RERANK_CANDIDATE_MAX_CHARS` boot override
    /// (`facade::providers::search_env_overrides`), mirroring `rrf_k`'s
    /// `KREMORY_RRF_K`, so a truncation-length sweep costs a restart, not a
    /// rebuild.
    pub rerank_candidate_max_chars: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 0.5,
            vector_weight: 0.5,
            rrf_k: 60,
            top_k: 10,
            // recall-v2 axes — every default preserves today's behaviour
            // (0.05 graph-degree live; all new axes at their no-op value).
            graph_degree_weight: 0.05,
            temporal_weight: 0.0,
            temporal_decay_lambda: 0.01,
            floor_threshold: 0.0,
            expansion_hop_bound: 1,
            expansion_fan_out_cap: 8,
            neighbour_score_decay: 0.5,
            // ADR-062 / ADR-067 Phase 3 — proximity OFF by default: the
            // second bounded-hop graph query never fires until
            // KREMORY_PROXIMITY_WEIGHT / with_proximity_weight flips it on.
            proximity_weight: 0.0,
            proximity_hop_bound: 2,
            proximity_fan_out_cap: 8,
            // TD-066 Increment 2 — 1.0 = neutral/no-op, today's equal-weight
            // RRF fusion (Increment 1's behaviour, byte-identical).
            content_stream_weight: 1.0,
            // TD-136 dense episode arm. **DEFAULT-ON since 2026-07-28 (ADR-078).**
            //
            // It shipped OFF and stayed OFF while every published benchmark set
            // `KREMORY_EPISODE_DENSE=1` — so the measured numbers described a
            // configuration no consumer received. Measured worth: **+5.1
            // recall@10 at full corpus.** It adds NO dependency: an embedder is
            // already a hard requirement of `Memory` (entities and facts are
            // embedded regardless), so this is more calls of something the
            // consumer already supplies, not a new one. Costs: one embed per
            // episode at ingest, one per query at recall (~3.8ms vector search,
            // measured).
            //
            // Turn it off with `.with_episode_dense_enabled(false)` or
            // `KREMORY_EPISODE_DENSE=0`. Existing corpora ingested before this
            // flip have no episode vectors until `backfill-episode-embeddings`
            // runs; the arm degrades to BM25-only until then rather than failing.
            episode_dense_enabled: true,
            // TD-139 DoD item 2 — dense fact arm OFF by default: facts stay
            // reachable only via 1-hop entity expansion until KREMORY_FACT_DENSE
            // flips it on.
            fact_dense_enabled: false,
            // TD-143 — nomic task-prefix OFF by default: every embed call sends
            // bare text, byte-identical to pre-TD-143, until KREMORY_EMBED_TASK_PREFIX
            // flips it on (and the corpus has been re-embedded to match).
            embed_task_prefix_enabled: false,
            // Reranker latency lever 1 — 0 = unlimited (today's behaviour,
            // byte-identical) until KREMORY_RERANK_CANDIDATE_MAX_CHARS /
            // with_rerank_candidate_max_chars sets a positive cap.
            rerank_candidate_max_chars: 0,
        }
    }
}

/// Explicit, per-knob overrides for [`SearchConfig`] set programmatically via
/// [`MemoryBuilder`](crate::facade::MemoryBuilder) (TD-141:
/// `.ai-docs/tech-debt/tech-debt-register.md` §TD-141). Each field is `None`
/// unless the consumer called the matching `MemoryBuilder::with_*` setter —
/// `None` means "not set programmatically", NOT "use this field's shipped
/// default".
///
/// # Why a sparse overlay, not `Option<SearchConfig>`
///
/// TD-141's design decision (b) sketches threading `search: Option<SearchConfig>`
/// through the `open_graph*` params. This type deviates from that literal shape:
/// TD-141 decision (a) requires PER-FIELD precedence (explicit programmatic
/// config > env override > default — see [`apply`](Self::apply)), and decision
/// (d) requires per-knob setters, not one `with_search_config(SearchConfig)`.
/// A consumer who calls only `.with_content_stream_weight(v)` must NOT
/// silently clobber a `KREMORY_RRF_K` / `KREMORY_EPISODE_DENSE` env override
/// for the other two fields. A monolithic `SearchConfig` cannot express
/// "unset" per field once materialized (every field always holds a concrete
/// value), so wrapping it in `Option` would force an all-or-nothing choice —
/// either every explicit config always wins outright (clobbering env knobs
/// the consumer never touched) or the whole thing is applied before env
/// (defeating the seam's purpose). The sparse `Option<T>`-per-field overlay
/// is the shape that actually satisfies (a) and (d) together.
#[derive(Debug, Clone, Default)]
pub(crate) struct SearchConfigOverrides {
    /// Explicit override for [`SearchConfig::content_stream_weight`].
    pub content_stream_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::rrf_k`].
    pub rrf_k: Option<usize>,
    /// Explicit override for [`SearchConfig::episode_dense_enabled`].
    pub episode_dense_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::fact_dense_enabled`] (TD-139).
    pub fact_dense_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::embed_task_prefix_enabled`] (TD-143).
    pub embed_task_prefix_enabled: Option<bool>,
    /// Explicit override for [`SearchConfig::proximity_weight`] (ADR-062 /
    /// ADR-067 Phase 3). `proximity_hop_bound` / `proximity_fan_out_cap` are
    /// deliberately NOT exposed here — config-default-only, mirroring
    /// `expansion_hop_bound`/`expansion_fan_out_cap`'s own precedent (only
    /// the weight is the A/B lever that needs a restart-not-rebuild seam).
    pub proximity_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::temporal_weight`] (TD-157).
    ///
    /// Added 2026-07-28. This axis has had working compute behind it all along
    /// (`core/context.rs`, `core/scoring/temporal.rs`) and was **unreachable**:
    /// unlike its sibling `proximity_weight` it had no builder method, no env
    /// override and no config-file path, so its `0.0` default could not be
    /// changed without editing this crate's source. A computed axis multiplied
    /// by an unreachable zero is dead weight, not a default.
    pub temporal_weight: Option<f32>,
    /// Explicit override for [`SearchConfig::rerank_candidate_max_chars`]
    /// (reranker latency lever 1).
    pub rerank_candidate_max_chars: Option<usize>,
}

impl SearchConfigOverrides {
    /// Apply only the `Some` fields onto `builder`, each WINNING over
    /// whatever env override (`facade::providers::search_env_overrides`) was
    /// already applied earlier in the same chain — TD-141 precedence:
    /// explicit programmatic config > env override > default. Fields left
    /// `None` here pass `builder` through unchanged, so an env override for a
    /// knob the consumer never touched programmatically still applies. An
    /// all-`None` (default-constructed) `SearchConfigOverrides` is a strict
    /// no-op, so unset-by-default callers get byte-identical behaviour to
    /// pre-TD-141 (env-only / default).
    pub(crate) fn apply(&self, mut builder: PipelineConfigBuilder) -> PipelineConfigBuilder {
        if let Some(v) = self.content_stream_weight {
            builder = builder.content_stream_weight(v);
        }
        if let Some(v) = self.rrf_k {
            builder = builder.rrf_k(v);
        }
        if let Some(v) = self.episode_dense_enabled {
            builder = builder.episode_dense_enabled(v);
        }
        if let Some(v) = self.fact_dense_enabled {
            builder = builder.fact_dense_enabled(v);
        }
        if let Some(v) = self.embed_task_prefix_enabled {
            builder = builder.embed_task_prefix_enabled(v);
        }
        if let Some(v) = self.proximity_weight {
            builder = builder.proximity_weight(v);
        }
        if let Some(v) = self.temporal_weight {
            builder = builder.temporal_weight(v);
        }
        if let Some(v) = self.rerank_candidate_max_chars {
            builder = builder.rerank_candidate_max_chars(v);
        }
        builder
    }
}

/// Newtype wrapper for the embedding dimension to make API signatures
/// self-documenting and prevent accidental dimension mismatches.
///
/// Default: 384. Rationale: matches the all-MiniLM-L6-v2 output dimension,
/// which is the default bundled embedding model. Changing this requires
/// a full index rebuild.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingDim(pub usize);

impl Default for EmbeddingDim {
    fn default() -> Self {
        Self(384)
    }
}

/// Entity-resolution call-shape strategy (ADR-076 / TD-127).
///
/// Selects how the ambiguous remainder of ingest-time entity resolution (the
/// entities that survive ADR-075 candidate blocking but are NOT resolved by
/// the cheap deterministic tiers — exact-normalize + MinHash) reaches the LLM:
///
/// - `Batched` (default): one structured-output call per window resolves ALL
///   ambiguous entities against a shared candidate pool at once, collapsing
///   the O(ambiguous × candidates) pairwise fan-out to O(windows). See
///   `resolver_batched.rs`.
/// - `Pairwise`: the pre-ADR-076 behaviour — one LLM `ResolutionVerdict` call
///   per (entity, candidate) pair via `CascadeResolver::resolve`. Retained for
///   A/B comparison and as an instant rollback (config flip, no code revert).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionStrategy {
    /// Pre-ADR-076 pairwise `resolve()` fan-out.
    Pairwise,
    /// ADR-076 batched structured-output resolution (default).
    #[default]
    Batched,
}

/// Top-level pipeline configuration aggregating all sub-configs.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Dimensionality of embedding vectors produced by the model.
    pub embedding_dim: EmbeddingDim,
    /// LLM-extraction-prompt-window splitting parameters (kind-2 chunking).
    pub extraction_window: ExtractionWindowConfig,
    /// MinHash LSH parameters for near-duplicate detection.
    pub minhash: MinHashConfig,
    /// Entropy pre-filter parameters.
    pub entropy: EntropyConfig,
    /// Hybrid search fusion parameters.
    pub search: SearchConfig,
    /// Entity types the pipeline will extract and index (empty = all types).
    pub allowed_entity_types: Vec<String>,
    /// Relation / edge types the pipeline will resolve (empty = all types).
    pub allowed_edge_types: Vec<String>,
    /// Entity types that are explicitly excluded even if matched by extraction.
    pub excluded_entity_types: Vec<String>,
    /// Default: 300s. Rationale: 5-minute cache TTL balances freshness against
    /// the cost of re-embedding; longer than a typical meeting turn cadence.
    pub cache_ttl: Duration,
    /// Default: 1000. Rationale: caps resident memory for the speculative cache;
    /// at ~1.5 KB per entry this is ~1.5 MB, acceptable on constrained hardware.
    pub cache_max_entries: usize,
    /// Per-arm wall-clock cap for structured-output extraction (ms).
    ///
    /// Default: 30_000 (30s — production fail-fast). The HTTP layer bounds
    /// individual requests (~10s via LLMBuilder); this caps an arm even if
    /// HTTP succeeds-then-hangs.
    ///
    /// Override for slow local LLMs (qwen2.5:14b ~80-130s per call on Apple
    /// M4 Max 36GB with 32k context) by setting 180_000-300_000 via
    /// `.extraction_arm_budget_ms(value)` on the builder. The benchmark
    /// suite sets this to 300_000 to accommodate qwen2.5:14b warm-up latency.
    pub extraction_arm_budget_ms: u64,
    /// TD-NEW-A (Lane C): how many per-chunk extraction LLM calls to run
    /// concurrently within a single `ingest_with` call (`futures::buffered`,
    /// order-preserving for determinism). `1` = the pre-change sequential
    /// behaviour. Extraction is read-only (no persisted-graph dependency
    /// between chunks — `known_entities` becomes window-local when >1, the
    /// staleness the dream L5 backstop absorbs), so overlapping the calls
    /// cuts ingest wall-time once resolution is no longer the bottleneck
    /// (ADR-076). Default: 5 (conservative vs provider concurrent-rate limits;
    /// raise via `KREMORY_EXTRACTION_CONCURRENCY` on a higher-limit provider).
    pub extraction_concurrency: usize,
    /// ADR-075 (TD-124): entity-resolution candidate-blocking width. When a
    /// group has MORE than this many existing entities, ingest resolution
    /// compares each newly-extracted entity only against a bounded candidate
    /// set — exact normalized-name matches UNION the embedding-ANN top-`k`
    /// nearest existing entities — instead of every existing entity. This
    /// collapses the LLM `ResolutionVerdict` fan-out from O(new × existing) to
    /// O(k). Groups with ≤ this many entities keep the exhaustive (pre-ADR-075)
    /// comparison, so the change is a no-op on small graphs. Dream-phase L5
    /// canonicalization is the completeness backstop for any match blocking
    /// misses. Default: 10.
    pub resolution_block_k: usize,
    /// ADR-075 P1 (TD-124): auto-different cosine floor for the embedding-ANN
    /// resolution arm. When a blocked ANN candidate's cosine similarity to the
    /// newly-extracted entity is **below** this floor, it is dropped from the
    /// candidate set (treated as `Different`) WITHOUT an LLM `ResolutionVerdict`
    /// call — the model would almost always say "different" for an embedding-far
    /// pair anyway, so the call is pure cost. This can only ever REDUCE merges
    /// (never create a false one), and dream-phase L5 canonicalization is the
    /// completeness backstop. Exact normalized-name matches are unaffected (they
    /// bypass the floor). Range `[0.0, 1.0]`. Default: `0.0` — OFF, i.e. pure
    /// P0 behaviour (every blocked candidate still reaches `resolve()`); raise
    /// (e.g. `0.5`) to trade a little recall for far fewer resolution LLM calls.
    pub resolution_min_cosine: f32,
    /// ADR-076 (TD-127): entity-resolution call-shape. `Batched` (default)
    /// collapses the ambiguous-remainder pairwise LLM fan-out into one
    /// structured call per window; `Pairwise` retains the pre-ADR-076
    /// per-(entity, candidate) `ResolutionVerdict` call for A/B + rollback.
    pub resolution_strategy: ResolutionStrategy,
    /// ADR-076 (TD-127): maximum number of ambiguous entities packed into a
    /// single batched-resolution window. Mirrors the `resolution_block_k`
    /// pattern — a safe-default overflow-cap knob, not a token-budget
    /// estimator (YAGNI per ADR-076 §Decision). Default: 32 — conservative
    /// on any 4k-or-larger-context model (32 short name+type lines, plus
    /// pooled candidates and system prompt, totals roughly 2-3k tokens).
    /// Raise on a large-context model to shrink call count further.
    pub resolution_batch_max_entities: usize,
    /// TD-167: run LLM contradiction detection during ingest. **Default: `false`.**
    ///
    /// Defaulted OFF because the mechanism currently DESTROYS SET-VALUED FACTS.
    /// It treats every predicate as functional (one value per subject), so
    /// ingesting a list supersedes all but the last member. Measured on 8
    /// LongMemEval sessions, 2026-07-29: 81 of 1,021 facts invalidated, of
    /// which ≥31% are provably multi-valued rather than contradictory —
    /// `has_performer: billie eilish / tove lo / lana del rey` all superseded
    /// by `the 1975`; `contain: rolled oats` superseded by `seeds`.
    ///
    /// The prompt specifies the behaviour (`core/contradiction.rs:176` — "if
    /// the new fact is an UPDATE (same relationship but newer value), return
    /// that index too"), and no predicate-cardinality model exists anywhere in
    /// the crate, so this is a design gap rather than a tuning problem.
    ///
    /// Turning it OFF is SAFE and REVERSIBLE:
    ///   * supersession is a SOFT delete (`invalid_at` is set; the row stays),
    ///     so no data written under the old default was destroyed;
    ///   * `core::dream::provenance::reversal::unsupersede` (ADR-073) already
    ///     exists to reverse individual false positives;
    ///   * genuine temporal supersession ("works_at Acme" → "works_at Globex")
    ///     is the capability being deferred — real and wanted, but currently
    ///     net-negative. Opt back in with `KREMORY_CONTRADICTION_DETECTION=1`
    ///     once TD-167 lands derived predicate cardinality.
    pub contradiction_detection_enabled: bool,
}

impl PipelineConfig {
    /// Returns a new [`PipelineConfigBuilder`] populated with all defaults.
    pub fn builder() -> PipelineConfigBuilder {
        PipelineConfigBuilder {
            inner: PipelineConfig {
                embedding_dim: EmbeddingDim::default(),
                extraction_window: ExtractionWindowConfig::default(),
                minhash: MinHashConfig::default(),
                entropy: EntropyConfig::default(),
                search: SearchConfig::default(),
                allowed_entity_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                excluded_entity_types: Vec::new(),
                cache_ttl: Duration::from_secs(300),
                cache_max_entries: 1000,
                extraction_arm_budget_ms: 30_000,
                extraction_concurrency: 5,
                resolution_block_k: 10,
                resolution_min_cosine: 0.0,
                resolution_strategy: ResolutionStrategy::default(),
                resolution_batch_max_entities: 32,
                // TD-167 / ADR-079 rev.2: ON. Was default-OFF for ~4h on
                // 2026-07-29 while the destruction bug was open; the prompt fix
                // (coexistence + temporal ordering) measured 0/8 set-valued
                // destroyed and a corpus run confirmed it (contradiction rate
                // 38.8% -> 11.3%, no multi-valued predicate destroyed). Scope
                // moved MVP -> v1, and LongMemEval's knowledge-update category
                // (78 of 500 questions) TESTS this mechanism — shipping the
                // benchmark with it disabled would publish a number with the
                // relevant feature switched off.
                contradiction_detection_enabled: true,
            },
        }
    }
}

/// Fluent builder for [`PipelineConfig`].
pub struct PipelineConfigBuilder {
    inner: PipelineConfig,
}

impl PipelineConfigBuilder {
    // ── EmbeddingDim ─────────────────────────────────────────────────────────

    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.inner.embedding_dim = EmbeddingDim(dim);
        self
    }

    // ── ExtractionWindowConfig ───────────────────────────────────────────────

    pub fn min_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.min_words = v;
        self
    }

    pub fn density_threshold(mut self, v: f64) -> Self {
        self.inner.extraction_window.density_threshold = v;
        self
    }

    pub fn max_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.max_words = v;
        self
    }

    pub fn overlap_words(mut self, v: usize) -> Self {
        self.inner.extraction_window.overlap_words = v;
        self
    }

    // ── MinHashConfig ────────────────────────────────────────────────────────

    pub fn num_permutations(mut self, v: usize) -> Self {
        self.inner.minhash.num_permutations = v;
        self
    }

    pub fn shingle_size(mut self, v: usize) -> Self {
        self.inner.minhash.shingle_size = v;
        self
    }

    pub fn band_size(mut self, v: usize) -> Self {
        self.inner.minhash.band_size = v;
        self
    }

    pub fn jaccard_threshold(mut self, v: f64) -> Self {
        self.inner.minhash.jaccard_threshold = v;
        self
    }

    // ── EntropyConfig ────────────────────────────────────────────────────────

    pub fn min_name_length(mut self, v: usize) -> Self {
        self.inner.entropy.min_name_length = v;
        self
    }

    pub fn min_token_count(mut self, v: usize) -> Self {
        self.inner.entropy.min_token_count = v;
        self
    }

    pub fn entropy_threshold(mut self, v: f64) -> Self {
        self.inner.entropy.entropy_threshold = v;
        self
    }

    // ── SearchConfig ─────────────────────────────────────────────────────────

    pub fn bm25_weight(mut self, v: f64) -> Self {
        self.inner.search.bm25_weight = v;
        self
    }

    pub fn vector_weight(mut self, v: f64) -> Self {
        self.inner.search.vector_weight = v;
        self
    }

    pub fn rrf_k(mut self, v: usize) -> Self {
        self.inner.search.rrf_k = v;
        self
    }

    pub fn top_k(mut self, v: usize) -> Self {
        self.inner.search.top_k = v;
        self
    }

    /// TD-066 Increment 2 / S0-infra sweep knob: per-stream weight applied to
    /// the ADR-072 `content_search` BM25 stream's RRF contribution. Default
    /// `1.0` (equal-weight fusion, byte-identical). Wired from the
    /// `KREMORY_CONTENT_WEIGHT` env override at server boot
    /// (`facade::providers::search_env_overrides`) so weight sweeps cost a
    /// restart, not a rebuild.
    pub fn content_stream_weight(mut self, v: f32) -> Self {
        self.inner.search.content_stream_weight = v;
        self
    }

    /// TD-136 dense-episode A/B knob: enable the dense (embedding) episode
    /// retrieval arm + ingest-time episode embedding. Default `false`
    /// (BM25-only, byte-identical). Wired from the `KREMORY_EPISODE_DENSE` env
    /// override at server boot (`facade::providers::search_env_overrides`) so
    /// the dense-vs-BM25 comparison costs a restart, not a rebuild.
    pub fn episode_dense_enabled(mut self, v: bool) -> Self {
        self.inner.search.episode_dense_enabled = v;
        self
    }

    /// TD-139 DoD item 2 dense-fact A/B knob: enable the dense (embedding)
    /// fact retrieval arm (`TemporalGraph::vector_search_facts`, RRF-fused
    /// via `core::search::rrf_fuse_with_facts`). Default `false` (facts
    /// reachable only via 1-hop entity expansion, byte-identical). Wired
    /// from the `KREMORY_FACT_DENSE` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the fact-dense A/B
    /// costs a restart, not a rebuild.
    pub fn fact_dense_enabled(mut self, v: bool) -> Self {
        self.inner.search.fact_dense_enabled = v;
        self
    }

    /// TD-143 nomic task-prefix A/B knob: enable `search_document:` /
    /// `search_query:` prefixing on every embed call site (see
    /// `core::embed_prefix`). Default `false` (bare text, byte-identical).
    /// Wired from the `KREMORY_EMBED_TASK_PREFIX` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the prefix A/B costs a
    /// restart, not a rebuild. ⚠️ Flipping this on an existing corpus requires
    /// a re-embed — see [`SearchConfig::embed_task_prefix_enabled`].
    pub fn embed_task_prefix_enabled(mut self, v: bool) -> Self {
        self.inner.search.embed_task_prefix_enabled = v;
        self
    }

    /// ADR-062 / ADR-067 Phase 3 axis-C A/B knob: weight of the additive
    /// graph-proximity boost. Default `0.0` (off, byte-identical — the second
    /// bounded-hop query never fires). Wired from the
    /// `KREMORY_PROXIMITY_WEIGHT` env override at server boot
    /// (`facade::providers::search_env_overrides`) so the proximity A/B costs
    /// a restart, not a rebuild.
    pub fn proximity_weight(mut self, v: f32) -> Self {
        self.inner.search.proximity_weight = v;
        self
    }

    /// ADR-067 temporal-recency axis weight (`SearchConfig::temporal_weight`),
    /// default `0.0` = axis off (byte-identical to pre-TD-157 behaviour).
    ///
    /// TD-157 (2026-07-28): this axis shipped with working compute
    /// (`core/context.rs`, `core/scoring/temporal.rs`) and **no way to enable
    /// it** — no builder method, no env override, and `SearchConfig` derives no
    /// `Deserialize`, so there was no config-file path either. Its sibling
    /// `proximity_weight` has all three. The value was therefore pinned at
    /// `0.0` for every consumer, and the axis was computed and then multiplied
    /// by an unreachable zero. That is dead weight, not a default: a knob
    /// nobody can turn is indistinguishable from an unimplemented feature, and
    /// the axis has consequently NEVER been measured. This adds the missing
    /// seam so it can be A/B'd; the default is unchanged.
    pub fn temporal_weight(mut self, v: f32) -> Self {
        self.inner.search.temporal_weight = v;
        self
    }

    /// ADR-062 axis-C hop bound — the parameter that controls the proximity
    /// signal's SELECTIVITY, and therefore the one that actually needed to be
    /// sweepable. Added 2026-07-27 after the first axis-C A/B measured flat at
    /// three weights: the per-recall trace showed `seed_count=50
    /// boosted_count=48` at the default `hop_bound = 2`, i.e. the walk reaches
    /// neighbours for ~96% of seeds, making the boost a near-uniform additive
    /// offset that cannot discriminate at ANY weight. Wired from
    /// `KREMORY_PROXIMITY_HOP_BOUND` (`facade::providers::search_env_overrides`)
    /// so the selectivity sweep costs a restart, not a rebuild — the TD-141
    /// lesson (a knob you cannot set is a knob you cannot evaluate) applied to
    /// axis-C's own tuning surface.
    pub fn proximity_hop_bound(mut self, v: u32) -> Self {
        self.inner.search.proximity_hop_bound = v;
        self
    }

    /// Reranker latency lever 1: cap the SUMMARY portion of each rerank
    /// candidate at `v` chars. Default `0` (unlimited, byte-identical).
    /// Wired from the `KREMORY_RERANK_CANDIDATE_MAX_CHARS` env override at
    /// server boot (`facade::providers::search_env_overrides`) so a
    /// truncation-length sweep costs a restart, not a rebuild.
    pub fn rerank_candidate_max_chars(mut self, v: usize) -> Self {
        self.inner.search.rerank_candidate_max_chars = v;
        self
    }

    // ── Ontology ─────────────────────────────────────────────────────────────

    pub fn allowed_entity_types(mut self, v: Vec<String>) -> Self {
        self.inner.allowed_entity_types = v;
        self
    }

    pub fn allowed_edge_types(mut self, v: Vec<String>) -> Self {
        self.inner.allowed_edge_types = v;
        self
    }

    pub fn excluded_entity_types(mut self, v: Vec<String>) -> Self {
        self.inner.excluded_entity_types = v;
        self
    }

    // ── Cache ────────────────────────────────────────────────────────────────

    pub fn cache_ttl(mut self, v: Duration) -> Self {
        self.inner.cache_ttl = v;
        self
    }

    pub fn cache_max_entries(mut self, v: usize) -> Self {
        self.inner.cache_max_entries = v;
        self
    }

    // ── Extraction budget ────────────────────────────────────────────────────

    /// Set the per-arm wall-clock budget for structured-output extraction (ms).
    ///
    /// Default: 30_000 (30s — production fail-fast).
    /// Override for slow local LLMs: e.g. `300_000` for qwen2.5:14b on Apple
    /// silicon (~80-130s per call at 32k context).
    pub fn extraction_arm_budget_ms(mut self, ms: u64) -> Self {
        self.inner.extraction_arm_budget_ms = ms;
        self
    }

    /// TD-NEW-A Lane C: concurrent per-chunk extraction calls per ingest
    /// (`futures::buffered`, order-preserving). `1` = sequential. Default 5.
    pub fn extraction_concurrency(mut self, n: usize) -> Self {
        self.inner.extraction_concurrency = n;
        self
    }

    /// TD-167: enable LLM contradiction detection during ingest. Default OFF.
    ///
    /// OFF because it currently treats every predicate as functional and so
    /// SUPERSEDES SET-VALUED FACTS — a festival's second performer supersedes
    /// the first. Enable only if you have verified your predicates are
    /// single-valued, or once TD-167 lands derived cardinality.
    pub fn contradiction_detection_enabled(mut self, on: bool) -> Self {
        self.inner.contradiction_detection_enabled = on;
        self
    }

    // ── Resolution candidate blocking (ADR-075 / TD-124) ──────────────────────

    /// Set the entity-resolution candidate-blocking width `k`. Groups with more
    /// than `k` existing entities compare each new entity against only the
    /// exact-name matches + embedding-ANN top-`k`; groups with ≤`k` keep the
    /// exhaustive comparison. Default: 10. Set to `usize::MAX` to disable
    /// blocking entirely (restore pre-ADR-075 behaviour).
    pub fn resolution_block_k(mut self, k: usize) -> Self {
        self.inner.resolution_block_k = k;
        self
    }

    /// Set the auto-different cosine floor for the embedding-ANN resolution arm
    /// (ADR-075 P1). Blocked ANN candidates below this cosine similarity are
    /// dropped without an LLM verdict. `0.0` (default) = off / pure P0. Clamped
    /// to `[0.0, 1.0]`.
    pub fn resolution_min_cosine(mut self, floor: f32) -> Self {
        self.inner.resolution_min_cosine = floor.clamp(0.0, 1.0);
        self
    }

    // ── Batched resolution (ADR-076 / TD-127) ──────────────────────────────────

    /// Select the entity-resolution call-shape strategy. `Batched` (default)
    /// collapses the ambiguous-remainder LLM fan-out into one structured call
    /// per window; `Pairwise` restores the pre-ADR-076 per-pair behaviour.
    pub fn resolution_strategy(mut self, v: ResolutionStrategy) -> Self {
        self.inner.resolution_strategy = v;
        self
    }

    /// Set the maximum number of ambiguous entities packed into a single
    /// batched-resolution window. Default: 32. Raise on a large-context model
    /// to shrink call count further; see `resolution_batch_max_entities` docs.
    pub fn resolution_batch_max_entities(mut self, v: usize) -> Self {
        self.inner.resolution_batch_max_entities = v;
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Validates the configuration and returns a [`PipelineConfig`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following invariants are violated:
    /// - `jaccard_threshold` must be in `(0.0, 1.0]`
    /// - `bm25_weight + vector_weight` must equal `1.0` (within 1e-9)
    /// - `embedding_dim` must be greater than 0
    /// - `min_words` must be greater than 0
    /// - `max_words` must be >= `min_words`
    /// - `density_threshold` must be in `(0.0, 1.0]`
    pub fn build(self) -> Result<PipelineConfig> {
        let c = &self.inner;

        let jt = c.minhash.jaccard_threshold;
        if jt <= 0.0 || jt > 1.0 {
            return Err(Error::Config(format!(
                "jaccard_threshold must be in (0.0, 1.0], got {}",
                jt
            )));
        }

        if c.embedding_dim.0 == 0 {
            return Err(Error::EmbeddingDimZero);
        }

        if c.extraction_window.min_words == 0 {
            return Err(Error::Config("min_words must be greater than 0".into()));
        }

        if c.extraction_window.max_words < c.extraction_window.min_words {
            return Err(Error::TokenWindowInvalid {
                min: c.extraction_window.min_words,
                got: c.extraction_window.max_words,
            });
        }

        let dt = c.extraction_window.density_threshold;
        if dt <= 0.0 || dt > 1.0 {
            return Err(Error::Config(format!(
                "density_threshold must be in (0.0, 1.0], got {}",
                dt
            )));
        }

        Ok(self.inner)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_builds() {
        let result = PipelineConfig::builder().build();
        assert!(result.is_ok(), "default config should build without error");
    }

    /// recall-improvement-e2e-spec-2026-07-22 §S0-infra (R4c): the S0-infra
    /// sweep knobs' defaults are unchanged — no-env recall output is
    /// byte-identical (DoD #1). `rrf_k=60` (Cormack et al.),
    /// `content_stream_weight=1.0` (equal-weight fusion). Guards against a
    /// default drift silently changing the baseline sweep point.
    #[test]
    fn search_sweep_knob_defaults_unchanged() {
        let search = SearchConfig::default();
        assert_eq!(search.rrf_k, 60, "default RRF k must stay 60");
        assert_eq!(
            search.content_stream_weight, 1.0,
            "default content_stream_weight must stay 1.0 (equal-weight fusion)"
        );
        // The builder path (used by open_graph before env overrides) must agree.
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.rrf_k, 60);
        assert_eq!(built.content_stream_weight, 1.0);
        // The new builder setter threads the value.
        let tuned = PipelineConfig::builder()
            .content_stream_weight(2.5)
            .rrf_k(1)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.content_stream_weight, 2.5);
        assert_eq!(tuned.rrf_k, 1);
    }

    /// Reranker latency lever 1 — `0` (unlimited) is the default, byte-
    /// identical to pre-lever behaviour, and the builder setter reaches the
    /// live `SearchConfig`.
    #[test]
    fn rerank_candidate_max_chars_defaults_to_zero_unlimited() {
        assert_eq!(
            SearchConfig::default().rerank_candidate_max_chars,
            0,
            "default must stay 0 (unlimited, byte-identical pre-lever)"
        );
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.rerank_candidate_max_chars, 0);
        let tuned = PipelineConfig::builder()
            .rerank_candidate_max_chars(512)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.rerank_candidate_max_chars, 512);
    }

    /// ADR-062 / ADR-067 Phase 3 — proximity ships OFF by default (byte-
    /// identical): `proximity_weight <= 0.0` skips the second graph query
    /// entirely (see `core::context::Engine::contextualize`).
    #[test]
    fn proximity_weight_defaults_to_zero_off() {
        assert_eq!(
            SearchConfig::default().proximity_weight,
            0.0,
            "default proximity_weight must stay 0.0 (axis OFF, byte-identical pre-ADR-062)"
        );
        assert_eq!(
            SearchConfig::default().proximity_hop_bound,
            2,
            "default proximity_hop_bound must stay the spec's starting value (2)"
        );
        assert_eq!(
            SearchConfig::default().proximity_fan_out_cap,
            8,
            "default proximity_fan_out_cap must match expansion_fan_out_cap's default (8)"
        );
        // The builder path (used by open_graph before env overrides) must agree.
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.proximity_weight, 0.0);
        // The new builder setter threads the value.
        let tuned = PipelineConfig::builder()
            .proximity_weight(0.2)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.proximity_weight, 0.2);
    }

    #[test]
    fn test_invalid_jaccard_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(2.0).build();
        assert!(result.is_err(), "jaccard_threshold > 1.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_jaccard_zero_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(0.0).build();
        assert!(result.is_err(), "jaccard_threshold = 0.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_weights_accept_any_positive_values() {
        let result = PipelineConfig::builder()
            .bm25_weight(0.7)
            .vector_weight(0.3)
            .build();
        assert!(
            result.is_ok(),
            "RRF weights are independent multipliers; sum-to-1.0 no longer required"
        );

        let result2 = PipelineConfig::builder()
            .bm25_weight(2.0)
            .vector_weight(1.0)
            .build();
        assert!(result2.is_ok(), "any positive values accepted");
    }

    #[test]
    fn test_invalid_embedding_dim_rejected() {
        let result = PipelineConfig::builder().embedding_dim(0).build();
        assert!(result.is_err(), "embedding_dim = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("embedding_dim"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_min_tokens_rejected() {
        let result = PipelineConfig::builder().min_words(0).build();
        assert!(result.is_err(), "min_words = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("min_words"), "error should mention field name");
    }

    #[test]
    fn test_max_less_than_min_rejected() {
        let result = PipelineConfig::builder()
            .min_words(800)
            .max_words(400)
            .build();
        assert!(result.is_err(), "max_words < min_words must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("max_words") || msg.contains("min_words"),
            "error should mention word-window fields"
        );
    }

    #[test]
    fn test_custom_config_builds() {
        let cfg = PipelineConfig::builder()
            .embedding_dim(768)
            .min_words(200)
            .max_words(600)
            .density_threshold(0.2)
            .num_permutations(64)
            .shingle_size(4)
            .band_size(8)
            .jaccard_threshold(0.85)
            .min_name_length(4)
            .min_token_count(1)
            .entropy_threshold(1.0)
            .bm25_weight(0.7)
            .vector_weight(0.3)
            .rrf_k(30)
            .top_k(5)
            .cache_ttl(Duration::from_secs(60))
            .cache_max_entries(500)
            .build()
            .expect("custom config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(768));
        assert_eq!(cfg.extraction_window.min_words, 200);
        assert_eq!(cfg.extraction_window.max_words, 600);
        assert!((cfg.extraction_window.density_threshold - 0.2).abs() < 1e-12);
        assert_eq!(cfg.minhash.num_permutations, 64);
        assert_eq!(cfg.minhash.shingle_size, 4);
        assert_eq!(cfg.minhash.band_size, 8);
        assert!((cfg.minhash.jaccard_threshold - 0.85).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 4);
        assert_eq!(cfg.entropy.min_token_count, 1);
        assert!((cfg.entropy.entropy_threshold - 1.0).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.7).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.3).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 30);
        assert_eq!(cfg.search.top_k, 5);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(60));
        assert_eq!(cfg.cache_max_entries, 500);
    }

    #[test]
    fn test_default_values_correct() {
        let cfg = PipelineConfig::builder()
            .build()
            .expect("default config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(384));
        assert_eq!(cfg.extraction_window.min_words, 100);
        assert!((cfg.extraction_window.density_threshold - 0.15).abs() < 1e-12);
        assert_eq!(cfg.extraction_window.max_words, 300);
        assert_eq!(cfg.minhash.num_permutations, 32);
        assert_eq!(cfg.minhash.shingle_size, 3);
        assert_eq!(cfg.minhash.band_size, 4);
        assert!((cfg.minhash.jaccard_threshold - 0.9).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 6);
        assert_eq!(cfg.entropy.min_token_count, 2);
        assert!((cfg.entropy.entropy_threshold - 1.5).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.5).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.5).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 60);
        assert_eq!(cfg.search.top_k, 10);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(300));
        assert_eq!(cfg.cache_max_entries, 1000);
    }

    #[test]
    fn test_ontology_config() {
        let allowed = vec!["Person".to_string(), "Organization".to_string()];
        let excluded = vec!["StopWord".to_string()];
        let edges = vec!["WORKS_AT".to_string()];

        let cfg = PipelineConfig::builder()
            .allowed_entity_types(allowed.clone())
            .excluded_entity_types(excluded.clone())
            .allowed_edge_types(edges.clone())
            .build()
            .expect("ontology config should build");

        assert_eq!(cfg.allowed_entity_types, allowed);
        assert_eq!(cfg.excluded_entity_types, excluded);
        assert_eq!(cfg.allowed_edge_types, edges);

        // Verify that allowed and excluded are independent — a type can
        // appear in excluded without being in allowed.
        assert!(!cfg.allowed_entity_types.contains(&"StopWord".to_string()));
        assert!(cfg.excluded_entity_types.contains(&"StopWord".to_string()));
    }

    #[test]
    fn test_extraction_arm_budget_ms_default() {
        let cfg = PipelineConfig::builder()
            .build()
            .expect("default config should build");
        assert_eq!(
            cfg.extraction_arm_budget_ms, 30_000,
            "default extraction_arm_budget_ms must be 30_000 (production fail-fast)"
        );
    }

    #[test]
    fn test_extraction_arm_budget_ms_custom() {
        let cfg = PipelineConfig::builder()
            .extraction_arm_budget_ms(180_000)
            .build()
            .expect("custom extraction_arm_budget_ms should build");
        assert_eq!(
            cfg.extraction_arm_budget_ms, 180_000,
            "extraction_arm_budget_ms builder method must propagate the custom value"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Config — core-layer telemetry prefix config (ADR D15)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core-layer telemetry configuration.
///
/// Controls the `metrics_prefix` and `span_prefix` namespace so callers can
/// co-deploy multiple kremory instances without metric label collision (ADR D15).
///
/// Default prefixes match the canonical names in `monitoring/kremory-memory-slos.toml`.
/// Override only when running multiple kremory deployments in the same Prometheus
/// namespace (e.g. staging vs prod scraping into one cluster).
///
/// # Cardinality note (ADR D7)
///
/// Prefixes are `Option<String>` set once at startup — not per-request strings.
/// The prefix is prepended to the base metric name at registration time, not at
/// emit time, so there is no per-call allocation overhead.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Optional prefix prepended to all `metrics::counter!/histogram!/gauge!` names.
    ///
    /// Example: `Some("kremory_prod".to_string())` → `kremory_prod_core_tokens_total`.
    /// `None` (default) uses the canonical `kremory_core_*` namespace.
    pub metrics_prefix: Option<String>,

    /// Optional prefix prepended to all `tracing::info!/warn!/error!` span names.
    ///
    /// Example: `Some("prod".to_string())` → `prod.kremory.embed completed`.
    /// `None` (default) uses the canonical `kremory.*` span namespace.
    pub span_prefix: Option<String>,
}

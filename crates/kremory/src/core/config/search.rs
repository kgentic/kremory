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

    /// Default: 1, flipped from 60 (Cormack et al.'s 2009 general-purpose
    /// constant) after a measured LoCoMo A/B: `k=1` beat `k=60`
    /// on every category and every metric (recall@10 +3.5, nDCG@10 +1.8, MRR
    /// +1.6, hit-rate +3.2). Smaller `k` sharpens RRF's
    /// `1/(k+rank+1)` weighting, so a confident top candidate from either the
    /// entity-graph or content stream dominates the fused ranking more; `k=60`
    /// spreads influence toward parity across ranks. kremory's two streams
    /// evidently produce strong top candidates often enough that sharpening
    /// helps. Single-corpus, single-run result — tunable via `with_rrf_k` /
    /// `KREMORY_RRF_K` if a consumer's own corpus disagrees.
    pub rrf_k: usize,

    /// Default: 10. Rationale: top-10 is the conventional precision@k cut-off
    /// for downstream synthesis; returning more increases context window cost
    /// without commensurate quality gain.
    pub top_k: usize,

    // ── Post-RRF scoring axes ────────────────────────────────────────────────
    //
    // All fields below feed the staged post-RRF boost pipeline in
    // `core::context::Engine::contextualize`. Every default preserves TODAY's
    // behaviour: only `graph_degree_weight` is a live tested axis (its default
    // is its already-shipped constant, NOT 0.0); all NEW axes default to their
    // no-op value so recall output is byte-identical to pre-change at defaults.
    /// Weight of the additive graph-degree bonus (formerly
    /// the `search::GRAPH_DEGREE_WEIGHT` const). **Default 0.05, NOT 0.0** —
    /// graph-degree is the one already-live boost axis, so
    /// `0.0` would silently disable a shipped, tested feature. "Today's
    /// behaviour" for this axis *is* the 0.05 bonus. Bounded contribution:
    /// `weight * min(degree/saturation, 1)`.
    pub graph_degree_weight: f32,

    /// Weight of the additive temporal-recency boost.
    /// **Default 0.0 = off** — new axis, behaviourally neutral until an
    /// eval-calibrated value is set. Additive + bounded `[0, weight]`, matching
    /// graph-degree so a single `.min(1.0)` clamp covers both (no second
    /// normalization pass).
    pub temporal_weight: f32,

    /// Decay rate `lambda` for the temporal boost's `exp(-lambda * age_days)`.
    /// Inert while `temporal_weight == 0.0`. Default 0.01
    /// ≈ a ~69-day half-life; a placeholder pending eval calibration.
    pub temporal_decay_lambda: f64,

    /// Minimum post-boost score a seed must reach to survive to output.
    /// **Default 0.0 = no-op** (nothing is
    /// floored out). Applied AFTER all boosts, BEFORE the access-count
    /// increment, so dropped entities are never counted as recalled.
    pub floor_threshold: f32,

    /// Max BFS hop depth for the 1-hop expansion.
    /// **Default 1 = today's behaviour.** Widening this without an in-BFS
    /// visited cap (`expansion_fan_out_cap`) reintroduces hub-explosion, so
    /// the two ship together.
    pub expansion_hop_bound: u32,

    /// Cap on entities visited during a seed's BFS expansion.
    /// **Default 8 = the former `context::MAX_NEIGHBOURS_PER_SEED`
    /// const.** Threaded into the BFS itself (not just the caller-side trim) so
    /// it bounds hop≥2 traversal, not only the final result set.
    pub expansion_fan_out_cap: usize,

    /// Score-decay factor applied to a 1-hop neighbour relative to the seed
    /// that surfaced it (formerly the
    /// `context::NEIGHBOUR_SCORE_DECAY` const). **Default 0.5** — mid-range of
    /// the literature's `[0.3, 0.7]` weighted-expansion band (HippoRAG PPR
    /// damping is also 0.5).
    pub neighbour_score_decay: f32,

    /// Weight of the additive graph-**proximity** boost (axis C).
    /// **Default 0.0 = off** — new axis, behaviourally
    /// neutral until an eval-calibrated value is set (same shape as
    /// `temporal_weight`).
    ///
    /// This axis ships **additive +
    /// bounded `[0, weight]`**, composed into the SAME single `.min(1.0)`
    /// clamp as `graph_degree_weight`/`temporal_weight` — stay additive,
    /// migrate all axes to a normalized
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

    /// Bounded-hop distance for the graph-proximity walk.
    /// **Default 2** — the starting recommendation: `hop_bound=1`
    /// collapses to a near-duplicate of `graph_degree_weight`'s own 1-hop
    /// signal; `hop_bound=3+` widens cost without a calibrated benefit yet.
    /// Inert while `proximity_weight <= 0.0` (the second graph query is
    /// skipped entirely, not just discounted).
    pub proximity_hop_bound: u32,

    /// In-BFS visited-entity cap for the proximity walk. **Default 8** —
    /// matches `expansion_fan_out_cap`'s
    /// default. `get_neighbours_at`'s `max_visited`
    /// early-exits the BFS once the cap is hit, bounding a high-degree hub
    /// seed's worst-case cost at `proximity_hop_bound >= 2` — the same
    /// mechanism the multi-hop expansion cap already uses, reused here
    /// rather than adding a new capped traversal primitive.
    pub proximity_fan_out_cap: usize,

    /// Per-stream weight applied to the `content_search` BM25 stream's
    /// RRF contribution in `search::rrf_fuse_with_content`.
    /// **Default 1.0 = today's equal-weight RRF fusion**
    /// (the unweighted fusion's behaviour, byte-identical) — this is a
    /// config-only calibration knob, not a new algorithm. `content`-alone
    /// beats equal-weight `hybrid` on 3 of 4 LoCoMo categories, so a
    /// value `> 1.0` favours the content stream over the entity-graph stream
    /// in the fused ranking; `<= 0.0` degrades content's contribution to
    /// (effectively) zero, without removing the stream's dedup/id-space
    /// participation. The calibrated value is a bench-sweep
    /// output, not a value guessed at design time — ship the knob, measure
    /// the value.
    pub content_stream_weight: f32,

    /// The dense episode retrieval knob: when `true`, recall runs a dense
    /// (embedding) vector arm over `episodes.embedding` and RRF-fuses it with
    /// the BM25 `content_search` episode arm BEFORE the fused content
    /// stream enters `search::rrf_fuse_with_content`. Also gates ingest-time
    /// episode embedding (`core/ingest/pipeline`). **Default `false` = today's
    /// BM25-only behaviour, byte-identical** — the whole dense arm (read +
    /// write) is inert until enabled, so `content-search` builds keep the exact
    /// pre-dense-arm recall + ingest surface at defaults. Wired from the
    /// `KREMORY_EPISODE_DENSE` boot override
    /// (`facade::providers::search_env_overrides`), mirroring
    /// `content_stream_weight`'s `KREMORY_CONTENT_WEIGHT` A/B knob, so the
    /// dense-vs-BM25 comparison costs a server restart, not a rebuild.
    pub episode_dense_enabled: bool,

    /// The dense fact retrieval knob: when `true`, recall runs a THIRD dense
    /// (embedding) arm —
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
    /// already-matched entity until this
    /// is enabled. Ingest-time fact embedding (`ingest_with.rs:~1774`,
    /// `deferred.rs:~411`) is UNCHANGED by this knob — that write path
    /// already runs unconditionally ("we already do all of
    /// that except the recall wiring"). Wired from the `KREMORY_FACT_DENSE`
    /// boot override (`facade::providers::search_env_overrides`), mirroring
    /// `episode_dense_enabled`'s `KREMORY_EPISODE_DENSE`, so the fact-dense
    /// A/B costs a server restart, not a rebuild.
    ///
    /// ⚠️ Risk (fact/predicate quality): noisy predicates
    /// (`greeting`, `session_timestamp`) embed as junk and may surface as
    /// dense-arm noise, potentially making the fact stream's already-net-
    /// negative shape WORSE — precisely why this ships default-OFF behind a
    /// measured gate, never defaulted on without an A/B.
    pub fact_dense_enabled: bool,

    /// The nomic task-prefix knob: when
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
    /// which is the prime suspect for the measured
    /// vocabulary/abstraction breadth gap.
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

    /// Reranker latency lever 1 (cross-encoder latency
    /// spike): caps the SUMMARY portion of each rerank candidate's
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
            rrf_k: 1,
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
            // Proximity OFF by default: the
            // second bounded-hop graph query never fires until
            // KREMORY_PROXIMITY_WEIGHT / with_proximity_weight flips it on.
            proximity_weight: 0.0,
            proximity_hop_bound: 2,
            proximity_fan_out_cap: 8,
            // 1.0 = neutral/no-op, today's equal-weight
            // RRF fusion (the unweighted fusion's behaviour, byte-identical).
            content_stream_weight: 1.0,
            // The dense episode arm. **DEFAULT-ON.**
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
            // Dense fact arm OFF by default: facts stay
            // reachable only via 1-hop entity expansion until KREMORY_FACT_DENSE
            // flips it on.
            fact_dense_enabled: false,
            // Nomic task-prefix OFF by default: every embed call sends
            // bare text, byte-identical to before this knob shipped, until
            // KREMORY_EMBED_TASK_PREFIX flips it on (and the corpus has been
            // re-embedded to match).
            embed_task_prefix_enabled: false,
            // Reranker latency lever 1 — 0 = unlimited (today's behaviour,
            // byte-identical) until KREMORY_RERANK_CANDIDATE_MAX_CHARS /
            // with_rerank_candidate_max_chars sets a positive cap.
            rerank_candidate_max_chars: 0,
        }
    }
}

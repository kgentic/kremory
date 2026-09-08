# Feature flags


kremory's `default` feature set is **`["content-search"]`** (ADR-078, 2026-07-28 — it was previously
empty). Opt into the surfaces below as needed:

| Feature | Default | Enables |
|---|---|---|
| *(default)* | — | Full `Memory` facade, bi-temporal graph, hybrid recall, dream phase, reversibility. BYOM LLM + embedder always available. |
| `content-search` | **ON** | Three things, not one: (a) the BM25/FTS5 **content arm that `recall()` RRF-fuses in automatically**, (b) the dense episode arm, and (c) Migrations 022 + 026 (`episodes_fts`, `episodes.embedding`). It also enables the explicit `mem.recall(q).content()` terminal + `ContentPassage` type ([Recall](./recall.md), ADR-072) — but that terminal is the *smallest* part of it. **Turning this off costs −32.2pt ex-adversarial substring recall** (measured, ADR-078); it is a default, not an extra. Zero additional dependencies. |
| `ner` | off | GLiNER hybrid extractor (`.with_gliner()` + `.with_llm(...)`); auto-downloads the ONNX GLiNER model on first use (`ort` / `ndarray` / `tokenizers` / `hf-hub`). |
| `embeddings` | off | Local ONNX embedding-provider support (`ort` / `ndarray` / `tokenizers` / `hf-hub`). BYOM embedders work without it. |
| `otel` | off | OTLP export — `tracing-subscriber` + `tracing-opentelemetry` + OTLP exporter; enables `init_telemetry(...)` (see [observability.md](observability.md)). |
| `trace` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "trace")]` sites exist in the crate (`Cargo.toml:59`). Was intended for hot-path span emission (ADR D3). |
| `unstable-graph` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "unstable-graph")]` sites exist (`Cargo.toml:80-87`). The planned v0.1.6 `Memory::get_related` traversal (G9) **never landed**; this row previously described it as if it had. |
| `unstable-tags` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "unstable-tags")]` sites exist (`Cargo.toml:88-93`). The planned `episode_tags` junction + `with_tags` / `filter_tag_any` / `filter_tag_all` (G3.b) **never landed**; this row previously described it as if it had. |
| `test-utils` / `llm-smoke` / `llm-integration` | off | Test-harness gating only — not part of the stable consumer surface. |

```toml
# Example: content recall + OTLP export.
# `content-search` is listed explicitly for clarity, but it is ON by default
# since ADR-078 (2026-07-28) — you only need to name it if you have set
# `default-features = false`.
kremory = { version = "0.8", features = ["content-search", "otel"] }
```

### Advanced tuning knobs (`MemoryBuilder`)

Beyond the setters already shown elsewhere in this reference, `MemoryBuilder` has a cluster of
recall-scoring and extraction-tuning knobs — each an explicit override that wins over both its
compiled-in default AND any matching `KREMORY_*` env var, for that one `Memory` instance. Verify
the live, in-effect values via `Memory::search_config()` / `Memory::contradiction_detection_enabled()`
(see [Advanced — substrate composition](./advanced.md), Introspection):

| Setter | Tunes |
|---|---|
| `with_content_stream_weight(f32)` | RRF weight of the BM25 `content-search` stream (default `1.0`) |
| `with_graph_degree_weight(f32)` | Weight of the additive graph-degree bonus (default `0.05`, live) |
| `with_proximity_weight(f32)` | Weight of the additive graph-proximity boost (default `0.0`, off) |
| `with_temporal_weight(f32)` | Weight of the additive temporal-recency boost (default `0.0`, off) |
| `with_rrf_k(usize)` | RRF fusion constant `k` (default `1`) |
| `with_episode_dense_enabled(bool)` | Dense (embedding) episode retrieval arm (default `false`) |
| `with_fact_dense_enabled(bool)` | Dense (embedding) fact retrieval arm (default `false`) |
| `with_embed_task_prefix_enabled(bool)` | nomic `search_document:`/`search_query:` task-prefixing (default `false`; nomic-specific) |
| `with_rerank_candidate_max_chars(usize)` | Truncates each rerank candidate's summary text before the cross-encoder (default `0` = unlimited) |
| `with_contradiction_detection_enabled(bool)` | Ingest-time contradiction detection — gates a DESTRUCTIVE supersession path (default `true`) |
| `extraction_arm_budget_ms(u64)` | Per-arm wall-clock cap for structured-output extraction (default `30_000`; raise for slow local LLMs) |
| `prior_turn_replay_depth(usize)` | Preceding episodes replayed into extraction for reference resolution (see [Recall](./recall.md), "Prior-turn replay"; default `10`) |
| `allowed_entity_types(Vec<String>)` | Entity type names the extractor may emit (required for non-empty `ner`-feature output) |
| `episode_content_warn_threshold(Option<usize>)` | Soft warn threshold (chars) for oversize episode content (default `Some(10_000)`) |
| `with_extractor(Arc<Ext>)` | Plug in a custom entity extractor (BYOE); mutually exclusive with `.with_gliner()` |
| `with_await_extraction(bool)` / `with_await_extraction_timeout(Duration)` | Block `remember()` on Phase 2 LLM extraction instead of the default fire-and-forget-with-poll behaviour |

**Why some of these are plain `bool` and one (`.cross_episode(CrossEpisodeMode)`, see [Dream](./dream.md)) is a
tri-state enum instead:** the bools above each toggle ONE independent, orthogonal on/off
decision — enabling `with_fact_dense_enabled(true)` doesn't change what any other setter means.
`CrossEpisodeMode` exists because cross-episode merging is controlled by **two COUPLED raw
bools** (`include_cross_episode_merges` + `cross_episode_dry_run`) where certain combinations are
ambiguous or contradictory to read at a call site (what does `dry_run: true` mean when
`include: false`?) — the tri-state enum (`Off` / `Shadow` / `Apply`) makes every valid
combination a single, self-explaining value and the invalid one unrepresentable. That is the
dividing line: a lone, independent knob stays a `bool`; two knobs whose meanings interact become
one enum.

```rust
let mem = Memory::open("./agent.db")
    .with_llm(llm)
    .with_embedder(emb)
    .with_rrf_k(60)                 // override the default k=1 for this instance
    .with_temporal_weight(0.1)      // turn on the temporal-recency axis
    .extraction_arm_budget_ms(180_000)  // slow local model — raise the per-arm timeout
    .await?;
```

---

# RQL Spike Runbook

## Build

```bash
# Library + unit tests (no external deps)
cargo build

# With LLM inference (llama-cpp-2, Metal GPU)
cargo build --features llm

# With real embeddings (ONNX Runtime, all-MiniLM-L6-v2)
cargo build --features embeddings

# Both
cargo build --features llm,embeddings
```

## Test

### Integration tests (no models required, always run)

```bash
# Full suite — 1782 tests (measured 2026-08-06, `content-search,test-utils`).
# Prefer nextest: it is process-per-test and ~16x faster than `cargo test` here (TD-109).
cargo nextest run --workspace --features content-search,test-utils

# Doctests are a SEPARATE tier — nextest does not run them.
cargo test --doc -p kremory --features content-search,test-utils

# Retrieval benchmark with detailed output
cargo test --test it retrieval_benchmark:: -- --nocapture
```

### E2E tests (require local model files, opt-in)

```bash
# Real embedding retrieval (downloads model on first run)
cargo test --features embeddings --test it retrieval_semantic:: -- --ignored --nocapture
```

> ⚠️ **WRONG-DOC CORRECTION, 2026-08-06.** This section previously listed four further
> GGUF-model commands (`e2e_local_llm::test_e2e_ingest`, `e2e_local_llm::test_e2e_nuextract`,
> `model_comparison::`, `multi_domain_extraction::`). **None of them could run.** All four
> target modules carry `#![cfg(any())]` — `any()` with no arguments is always false, so the
> whole file compiles to nothing and the named tests do not exist. They have been parked
> since 2026-05-18 (the D.1a BYOM strict gate removed `autoagents-llamacpp` from the crate,
> so the tests' concrete `LlamaCppProvider` dependency vanished). The commands were carried
> forward untouched through the 2026-08-06 `tests/it/` consolidation — the paths were even
> mechanically rewritten to the new `--test it <module>::` form, which made them look
> *more* current while remaining unrunnable.
>
> Thirteen such permanently-dead modules exist (8,223 LoC). Their disposition is tracked
> under **TD-109** ("delete the 13 permanently-dead `#![cfg(any())]` files"). Do not restore
> a command here without first confirming its module is not `cfg(any())`-gated.

> `RQL_MODEL_PATH`, `RQL_NUEXTRACT_MODEL_PATH` and `RQL_BENCHMARK_MODELS` (documented under
> Environment variables below) are read only by those dead modules. They are inert today.

## Metrics

### How metrics work

The crate uses the `metrics` facade (v0.24). Library code emits via `histogram!()` and `counter!()` macros. Tests install a `DebuggingRecorder` to capture metrics, then export via `MetricsExporter` to timestamped JSON files in `logs/`.

52 metrics across 5 layers: DB (22), search (12), extraction (6), LLM (6), ingest (6).

### Metrics report script

```bash
# Show latest run summary (grouped by layer, with p50/p99)
python3 scripts/metrics_report.py --latest

# Compare two runs (flags regressions, exit code 1 if >20% slower — CI-safe)
python3 scripts/metrics_report.py --compare logs/RUN_A.json logs/RUN_B.json

# Show trend across all runs for a label (sparklines + bar charts)
python3 scripts/metrics_report.py --history retrieval-benchmark

# Fact-specific benchmark history
python3 scripts/metrics_report.py --history fact-retrieval-benchmark
```

The script is stdlib-only Python 3 — no pip install needed. ANSI color output (green = improvement, red = regression >20%, yellow = warning >10%).

### Metrics JSON format

Exported to `logs/{unix_timestamp}-{label}-metrics.json`:

```json
{
  "timestamp_unix": 1774823093,
  "timestamp_iso": "2026-03-29T22:24:53Z",
  "label": "retrieval-benchmark",
  "histograms": {
    "rql.db.insert_entity_ms": {
      "count": 208,
      "sum": 11.6,
      "min": 0.044,
      "max": 0.309,
      "mean": 0.056,
      "values": [0.044, 0.048, ...]
    }
  },
  "counters": {},
  "gauges": {}
}
```

### Key performance baseline (208 entities, 194 facts, in-memory)

| Operation | Mean | Notes |
|-----------|------|-------|
| `insert_entity` | 56μs | INSERT + FTS5 tokenization |
| `insert_fact` | 40μs | Slightly faster (less indexing) |
| `set_embedding` | 472μs | DiskANN index update dominates |
| `get_entity` | 17μs | Primary key lookup |
| `fts_search` | 127μs | BM25 full-text |
| `vector_search` | 615μs | DiskANN cosine similarity |
| `hybrid_search` | 796μs | FTS + vector via RRF fusion |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RQL_MODEL_PATH` | _(unset)_ | **INERT** — read only by `cfg(any())`-dead modules. See the Test section warning. |
| `RQL_NUEXTRACT_MODEL_PATH` | _(unset)_ | **INERT** — same. |
| `RQL_BENCHMARK_MODELS` | _(unset)_ | **INERT** — same. |

## Cargo Features

| Feature | Enables | Required for |
|---|---|---|
| `llm` | llama-cpp-2 (Metal) | LLM extraction, E2E tests |
| `embeddings` | ort, tokenizers, hf-hub | Real ONNX embedding provider |

## Models

Models live in the project-root `models/` directory:

| File | Size | Use |
|---|---|---|
| `Phi-4-mini-instruct.Q4_K_M.gguf` | 3.8B | Primary extraction model |
| `NuExtract-2.0-4B-Q4_K_M.gguf` | 4B | Template-based extraction |
| `qwen2.5-3b-instruct-q4_k_m.gguf` | 3B | Alternative extractor |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.5B | Lightweight alternative |

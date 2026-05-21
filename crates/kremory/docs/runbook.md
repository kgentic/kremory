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
# Full suite — 153 lib tests + 7 retrieval benchmark + spike tests
cargo test

# Retrieval benchmark with detailed output
cargo test --test retrieval_benchmark -- --nocapture
```

### E2E tests (require local model files, opt-in)

```bash
# E2E ingest with LLM extraction
RQL_MODEL_PATH=../models/Phi-4-mini-instruct.Q4_K_M.gguf \
  cargo test --features llm --test e2e_local_llm test_e2e_ingest -- --ignored --nocapture

# NuExtract template extraction benchmark
RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
  cargo test --features llm --test e2e_local_llm test_e2e_nuextract -- --ignored --nocapture

# Multi-model comparison (all available models)
RQL_BENCHMARK_MODELS=1 \
RQL_MODEL_PATH=../models/Phi-4-mini-instruct.Q4_K_M.gguf \
RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
  cargo test --features llm --test model_comparison -- --ignored --nocapture

# Multi-domain extraction (14 fixtures, ground truth validation)
RQL_MODEL_PATH=../models/Phi-4-mini-instruct.Q4_K_M.gguf \
  cargo test --features llm --test multi_domain_extraction -- --ignored --nocapture

# Real embedding retrieval (downloads model on first run)
cargo test --features embeddings --test retrieval_semantic -- --ignored --nocapture
```

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
| `RQL_MODEL_PATH` | _(unset)_ | Path to GGUF model for E2E LLM tests |
| `RQL_NUEXTRACT_MODEL_PATH` | _(unset)_ | Path to NuExtract GGUF model |
| `RQL_BENCHMARK_MODELS` | _(unset)_ | Set to `1` to opt in to multi-model comparison |

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

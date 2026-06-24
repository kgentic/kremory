# Local-model benchmark

Reproducible benchmark for choosing kremory's Ollama models. Measures, per model,
the full extraction-classification surface — **true precision, recall, F1, per-label
confusion, over-extraction** — plus **per-call latency vs the inline 30s budget**,
total ingest wall-clock, model size, and thinking-capability. All from one inline
ingest run through the existing `label_precision_benchmark` harness (no extra LLM cost).

## Prerequisites
- [Ollama](https://ollama.com) running (default `http://localhost:11434`)
- `ollama pull nomic-embed-text` (embeddings)
- Rust toolchain (the harness compiles under `--features llm-integration`)
- Models are **pulled on demand**; tags that 404 are recorded, never recommended.

## Run

```sh
# one model
scripts/model-benchmark/bench-model.sh qwen2.5:7b            # default domain: mock_interview
scripts/model-benchmark/bench-model.sh gemma4:e4b legal_deposition

# disable reasoning on a thinking model (sends Ollama think:false)
KREMORY_BENCH_THINK=false scripts/model-benchmark/bench-model.sh gemma4:e4b

# full sweep (non-thinking once; thinking native + think:false), then collate
scripts/model-benchmark/run-batch.sh
python3 scripts/model-benchmark/collate.py
```

## Output
Default output dir: `.ai-docs/research/local-model-benchmark-2026-06-24/` (override with `BENCH_OUT`).
- `results.tsv` — one row per model run
- `reports/<model>.<domain>[.nothink].json` — full per-model metrics incl. confusion + extracted set
- `logs/<model>.<domain>[.nothink].log` — full harness output (latency histograms, diagnostics)
- `results.md` — collated, F1-sorted table + the two recommended picks (`collate.py`)

## How it maps to kremory's two model slots
- **interactive** (`Memory::with_ollama` default) serves the inline 30s-budget path → pick the
  highest-F1 **non-thinking** model whose slowest single call < 30s, smallest reasonable size.
- **deferred/dream quality** runs in the background (no tight budget) → pick the highest-F1
  **thinking** model (or a thinking model with `think:false` if that wins).

Latency numbers are **Apple Silicon M4 Max only**; precision/recall/F1 are hardware-independent
(same gguf weights → same quality anywhere). Cross-hardware latency is parked.

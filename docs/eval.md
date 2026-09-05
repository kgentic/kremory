# kremory quality evaluation harness

`kremory-eval` is an internal crate (`publish = false`) that runs two layers
of quality evaluation against kremory: published-comparable benchmarks (Layer
A) and diagnostic metrics (Layer B). It is not part of the public API.

This document covers how to run the harness, the environment variables it
honours, and where each scorer's design is decided.

> **Positioning at v0.1.x** — the Layer B regression-prevention numbers are
> shipped per release. The Layer A LongMemEval Oracle headline baseline is
> intentionally **deferred until kremory's public API stabilises** (post-v1.0.0).
> A published number against a moving API surface is misleading to adopters,
> and peer numbers (Mem0, Zep, Emergence) are all cloud-stack runs that a
> local-14B kremory baseline would not be directly comparable to. Smoke runs
> against local fixtures are the keeper signal for now.

> For the dataset and fixture inventory see
> [`docs/eval-fixtures.md`](./eval-fixtures.md).

---

## Quick start

### CI-safe smoke (no model required)

Layer B with the `MockJudge::always_correct()` judge — runs in seconds, no
network, no GPU:

```bash
cargo run -p kremory-eval --release --bin eval -- layer-b
cargo run -p kremory-eval --release --bin eval -- layer-b --determinism
```

Layer A LongMemEval against local synthetic fixtures (mock judge, no HF
download):

```bash
cargo run -p kremory-eval --release --bin eval -- layer-a longmemeval --sample 5 --judge mock
```

### Live runs (real models)

Requires:

- A running Ollama instance with `qwen2.5:14b` (chat) + `nomic-embed-text`
  (embedder) pulled
- A local Gemma 4 E2B GGUF for the judge (see [judge model](#judge-model))

Layer A LongMemEval with live Gemma judge against local smoke fixtures:

```bash
KREMORY_EVAL_LIVE_LLM=1 \
  cargo run -p kremory-eval --release --bin eval -- \
  layer-a longmemeval --sample 3 --smoke --judge gemma
```

Full LongMemEval Oracle baseline against HuggingFace dataset:

```bash
KREMORY_EVAL_LIVE_LLM=1 \
  cargo run -p kremory-eval --release --bin eval -- \
  layer-a longmemeval --judge gemma
```

Layer B with live Gemma judge:

```bash
KREMORY_EVAL_LIVE_LLM=1 \
  cargo run -p kremory-eval --release --bin eval -- layer-b
```

Reports are written to `crates/kremory-eval/output/<label>-<timestamp>.json`
(gitignored).

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `KREMORY_EVAL_LIVE_LLM` | unset | Set to `1` to enable live LLM paths (Gemma judge, Ollama providers). When unset, eval falls back to `MockJudge` and Layer A smoke fixtures. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL used by the kremory `Memory` handle in Layer A live runs. |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:14b` | Chat model used by `Memory` for entity extraction in Layer A live runs. Conventional name shared with litellm / ollama-haystack / langchain-ollama. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model used by `Memory` for recall in Layer A live runs. Must be a 768-dim model when paired with the default `embedding_dim(768)` on the builder. |
| `KREMORY_EVAL_JUDGE_MODEL_PATH` | `~/.cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf` | Path to the GGUF file used by `GemmaJudge`. |

### Why are eval-time model defaults different from `Memory::with_ollama`?

`Memory::with_ollama` is the kremory Tier-1.5 shortcut for application code
and hardcodes `llama3.2`. The 3.2B llama produces duplicate `VerbatimString`
entity names on LongMemEval transcripts, which trips kremory's intra-batch
dedup invariant. The eval harness avoids this by constructing the `Memory`
handle via the Tier-2 builder `Memory::open(path).with_llm(...).with_embedder(...)`
with explicit Ollama providers — the BYOM contract kremory advertises.

Production application code is free to:

- Keep using `Memory::with_ollama` (llama3.2) for low-latency local recall
  where extraction quality is not the bottleneck.
- Switch to the Tier-2 builder when extraction quality matters, mirroring the
  pattern the eval harness uses.

---

## Layer A — LongMemEval

### What it scores

Five named question types plus an abstention split, derived from the
`xiaowu0162/longmemeval-cleaned` dataset on HuggingFace. The scorer is a
Rust port of upstream `evaluate_qa.py` at SHA
`9e0b455f4ef0e2ab8f2e582289761153549043fc`. Prompt templates
(`PROMPT_STANDARD`, `PROMPT_TEMPORAL`, `PROMPT_KNOWLEDGE_UPDATE`,
`PROMPT_PREFERENCE`, `PROMPT_ABSTENTION`) are copied verbatim from
upstream — see [`crates/kremory-eval/baselines/v0.1.4-scorer-decision.md`](../crates/kremory-eval/baselines/v0.1.4-scorer-decision.md).

### Output JSON

Per-sample rows include `question_id`, `question_type`, `ability_category`,
`score` (0.0 / 1.0), and the raw judge response. Aggregates include
`overall_accuracy`, `per_type_accuracy` (6 categories), and
`abstention_accuracy`.

### Judge

`GemmaJudge` runs a local Gemma 4 E2B-IT Q4_K_M GGUF via
`autoagents-llamacpp`. The judge is asked for a JSON object with
`is_correct`, `is_partial`, `reasoning`. The scorer uses the structured
`is_correct` bool directly — not a substring match on free-form reasoning
text — because some small models (Gemma 4 E2B observed 2026-05-28) produce
positive reasoning without an explicit "yes" prefix.

For unit tests, `MockJudge` returns a fixed verdict so the scorer logic can
be exercised without a model on disk.

### CLI flags

| Flag | Purpose |
|---|---|
| `--sample N` | Limit to first N samples (useful for ballpark runs before committing to a full ~500-question pass). |
| `--judge mock\|gemma` | Pick the judge. Default is `mock`. `gemma` requires `KREMORY_EVAL_LIVE_LLM=1`. |
| `--smoke` | Use local synthetic 6-fixture set instead of HuggingFace download. Compatible with both judges. |

---

## Layer B — diagnostic metrics

### Metrics

| Metric | Source | Notes |
|---|---|---|
| Entity P/R/F1 | `layer_b::entity_extraction` | Per-fixture and aggregate. |
| RAGAS faithfulness, answer_relevancy, context_precision, context_recall, context_entities_recall, hallucination | `layer_b::ragas` | 20 fixtures. Judge-driven (MockJudge by default, GemmaJudge with `KREMORY_EVAL_LIVE_LLM=1`). |
| Graph integrity invariants | `layer_b::graph_integrity` | Structural checks (FTS counts, namespace presence, orphan entities). |
| Determinism (optional) | `--determinism` flag | Runs Layer B three times and asserts pairwise variance ≤ 0.02. |

### Output JSON

Aggregates per metric. Determinism JSON section appears only when
`--determinism` is passed.

---

## Judge model

The judge defaults to a Gemma 4 E2B-IT Q4_K_M GGUF at
`~/.cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf`. Override with
`KREMORY_EVAL_JUDGE_MODEL_PATH`. The model file is ~3 GB.

The default seed is `42`, `temperature=0`, `max_tokens=1024`. These match
the calibration runs in `crates/kremory-eval/baselines/`.

---

## BYOM invariant

`kremory` itself depends ONLY on the `autoagents-llm` trait crate. The
`autoagents-llamacpp` concrete provider lives in `kremory-eval`'s `Cargo.toml`
and is never pulled into the `kremory` dependency tree. Verify via:

```bash
cargo tree -p kremory --edges normal | grep autoagents-llamacpp
# (must print nothing)
```

---

## Decisions and source-of-truth

- Scorer fork decision (Option A — Rust port): `crates/kremory-eval/baselines/v0.1.4-scorer-decision.md`
- Fixture inventory: `docs/eval-fixtures.md`

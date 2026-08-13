# LoCoMo Benchmark for Codemem

> Adapted from [cogniplex/codemem](https://github.com/cogniplex/codemem) (Apache-2.0) —
> `harness.py`'s `CodememClient` now drives kremory over `kremory-http` instead
> of codemem's own API server. Run instructions below updated accordingly;
> the results table is codemem's own historical numbers, kept for reference.

Evaluates codemem's long-term conversational memory against the [LoCoMo benchmark](https://snap-research.github.io/locomo/) (ACL 2024).

## Results

Full dataset: 10 conversations, 1,986 questions.

| System | Accuracy | Recall Limit | Evidence Oracle | Embedding Fallback |
|--------|----------|-------------|----------------|-------------------|
| **Codemem** (OpenAI embed) | **91.64%** | **10** | No | No |
| **Codemem** (local BERT) | **89.58%** | **10** | No | No |
| Published SOTA | 90.53% | 50-100 | Yes | Yes |
| CORE | 88.24% | -- | -- | -- |

### Embedding model comparison

| Mode | OpenAI text-embedding-3-small | Local BERT (bge-base-en-v1.5) | Delta |
|------|------|------|-------|
| codemem | 91.64% | 89.58% | +2.06% |
| codemem-graph | 91.49% | 91.49% | 0% |

Graph expansion closes the gap entirely for BERT (89.58% → 91.49%) but doesn't help OpenAI (91.64% → 91.49%). Better embeddings already retrieve what graph expansion would find, making the two modes converge.

### Breakdown by category (OpenAI codemem, 1,986 questions)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Adversarial | 446 | 446 | 100.0% |
| Open-domain | 833 | 841 | 99.0% |
| Temporal | 270 | 321 | 84.1% |
| Single-hop | 215 | 282 | 76.2% |
| Multi-hop | 56 | 96 | 58.3% |
| **Overall** | **1820** | **1986** | **91.64%** |

### Why this matters

Most published LoCoMo benchmarks inflate their scores through techniques that bypass actual retrieval quality:

- **Evidence oracle**: Using ground-truth evidence IDs from the dataset to directly fetch the memories containing the answer, rather than relying on the retrieval system to find them.
- **High recall limits**: Retrieving 50-100 memories per question out of ~100-400 total. At that ratio, you're returning most of the conversation.
- **Embedding similarity fallback**: When word overlap fails, falling back to cosine similarity, effectively adding a second retrieval pass.
- **Low thresholds**: Word overlap thresholds as low as 0.30, which produces false positives.

Codemem's benchmark takes a stricter approach:

- **No evidence oracle** -- retrieval must find the right memories on its own
- **Recall limit of 10** -- forces the system to retrieve precisely, not exhaustively
- **No embedding fallback** -- word overlap and substring matching only
- **Standard thresholds** -- 0.50 for single-hop, 0.35 for multi-hop
- **Chunked ingestion** -- 4-turn chunks (~100 memories per conversation) vs per-turn storage (~400+)

Despite stricter conditions, codemem scores higher. The graph-vector hybrid scoring (vector similarity + BM25 + graph centrality + temporal signals) retrieves the right information with 5-10x fewer results.

## Setup

```bash
cd bench/locomo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

```bash
mkdir -p data
wget -O data/locomo10.json https://huggingface.co/datasets/snap-research/locomo/resolve/main/locomo10.json
```

## Run

```bash
# Start kremory-http server (from the pattaya workspace root)
# --features prometheus,content-search is REQUIRED, not optional. Cargo.toml:49 calls
# this "the HTTP/bench build"; this README omitted it until 2026-08-06 and that omission
# cost a wrong conclusion (see below).
KREMORY_MCP_DB_PATH=./bench.db cargo run -p kremory-mcp --bin kremory-http \
  --features prometheus,content-search

# Full benchmark (10 conversations, ~1,986 questions)
python3 harness.py

# Quick test (single conversation)
python3 harness.py --conversations 0

# Different modes
python3 harness.py --mode baseline         # Full context upper bound
python3 harness.py --mode codemem          # Hybrid recall
python3 harness.py --mode codemem-graph    # Hybrid + graph expansion

# Custom recall limit
python3 harness.py --recall-limit 20
```

### ⚠️ Why the feature flags are not optional (added 2026-08-06)

Built WITHOUT `prometheus`, the server has no `GET /metrics`, so:

- `harness.py` prints `[warn] /metrics absent — server built without --features prometheus`
- the `o11y` block in **every** result JSON is silently `{}`
- **every kremory counter is invisible** — counters are metrics, not tracing, so they
  never reach the server log either

That last point cost a wrong conclusion on 2026-08-06. A verified-working ingest fix
appeared not to have run, because `grep stub_embedded_total server.log` came back
empty — and an empty grep reads as "the code did not execute". The counter was fine;
the exporter simply was not built in. The database (`SELECT COUNT(*) ... WHERE
embedding IS NULL` → 0) proved the fix had worked all along.

**The general trap**: with `prometheus` off, an absent counter is indistinguishable
from an absent code path. Build with the feature, or verify against the DB — never
treat a silent grep as evidence.

Without `content-search`, BM25 + dense content recall are compiled out entirely
(ADR-078). Benchmarking that build measures a product no consumer receives — this is
exactly how 0.5.0 shipped to crates.io with search off and ~32 points unmeasured.

## 🛡️ Run-integrity guards (added 2026-08-13)

Five guards make a *bad paid run* structurally hard rather than merely unlikely. Full
rationale + the residual risks nothing removes:
`.ai-docs/plans/paid-bench-run-integrity-build-2026-08-13.md`.

**Before believing any recall number — this is a gate, not a nicety:**

```bash
python3 evidence_eval.py <run.json> --self-test    # exits NON-ZERO if the metric is order-blind
```

It shuffles each retrieved list and re-scores; a rank-aware metric MUST move. Until
2026-08-13 it printed `FAIL … order-BLIND` and **exited 0**, so wiring it into a script
gave a false green. It is now safe to use in a `&&` chain.

**Smoke before you batch** (smoke-one-before-batch — the guard the plan prescribed and
nothing implemented):

```bash
python3 qa_eval.py answer-gen results.json -o answers.jsonl --sample-n 10
python3 qa_eval.py answer-judge answers.jsonl -o verdicts.jsonl
python3 qa_eval.py answer-tally results.json --verdicts verdicts.jsonl --allow-sampled
```

The sample marker travels gen → judge → tally. **The tally fails closed on it**: without
`--allow-sampled` it refuses, and with it the output is branded `NOT QUOTABLE` and the
summary JSON carries `locomo_qa_gen_accuracy_SAMPLE`, not the real metric name. Same
`--sample-n`/`--sample-seed` picks the same questions, so a re-run replays from cache for $0.

**Pin the sweep point** when grading two arms in one session — crossing their result files
is silent, plausible, and otherwise undetectable:

```bash
python3 qa_eval.py answer-tally results.json --verdicts v.jsonl --expect rrf_k=60 --expect content_stream_weight=1.0
```

**Never run a paid ingest by hand.** Use the script, which asserts five preconditions and
aborts *before* spending — including that the active model is priced in
`provider-rates.toml`, because an unpriced model emits **no cost metric at all** and would
leave the spend guard reading `$0.00` forever:

```bash
KREMORY_BENCH_COST_CEILING_USD=25.00 ./run_paid_ingest.sh <label>
```

It arms `scripts/bench-spend-guard.sh`, which polls the server's own
`kremory_core_cost_usd_total` and kills the run at the ceiling, and writes
`<db>.provenance.json` recording which binary and model built the corpus — because the DB
carries no such column and **its mtime is not its ingest time** (a read is a write,
RECALL-LEDGER §4.20).

Every guard has a RED-proof: `test_spend_guard.sh`, `test_paid_ingest_preconditions.sh`,
and the `test_*.py` files. Run them with `python3 -m pytest` plus
`bash test_spend_guard.sh && bash test_paid_ingest_preconditions.sh`.

## Question Categories

| Category | Count | Description |
|----------|-------|-------------|
| Single-hop | 282 | Simple fact retrieval from one memory |
| Temporal | 321 | Time-based queries requiring date/sequence awareness |
| Multi-hop | 96 | Connecting information across multiple memories |
| Open-domain | 841 | General knowledge questions grounded in conversation |
| Adversarial | 446 | Questions about things never discussed (should abstain) |

## Evaluation

Retrieval accuracy: for each question, the harness checks whether the gold answer text is findable in the recalled memories using:

1. **Exact substring match** (case-insensitive)
2. **Word overlap** with basic stemming (threshold: 0.50 single-hop, 0.35 multi-hop)
3. **Fuzzy date matching** for temporal questions
4. **Abstention detection** for adversarial questions (low overlap = correct)

No LLM judge -- scoring is deterministic and reproducible.

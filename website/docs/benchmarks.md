# LoCoMo benchmark — full methodology and results

[LoCoMo](https://github.com/mem0ai/memory-benchmarks) is the standard long-conversation memory
benchmark: a dataset of long, multi-session conversations with 1,540 real questions attached,
each requiring recall of something from earlier in the conversation to answer correctly. A score
is produced in three steps: kremory's `recall()` retrieves candidate memories (free, deterministic),
an "answerer" LLM generates an answer using only those memories (paid), and a "judge" LLM grades
the generated answer against the known-correct gold answer (paid).

All numbers below are measured on the **full 1,540-question corpus**, not a sample, and are
build-verified against kremory's shipped default feature set (`content-search`) — not an
enhanced or unreleased configuration.

## Results

| configuration | score | answerer / judge | memories shown (k) |
|---|---|---|---|
| Mem0 Platform (hosted, paid competitor) | 92.5% | `gpt-5` / `gpt-5` | 200 |
| **kremory — Mem0's exact protocol reproduced** | **92.1%** (1418/1540) | `gpt-5` / `gpt-5` | 200 |
| kremory — cheaper model, matched context size | 92.1% (1419/1540) | `gpt-4o-mini` / `gpt-4o-mini` | 200 |
| kremory — earlier measurement | 91.7% (1412/1540) | `gpt-4o-mini` / `gpt-4o-mini` | 50 |

kremory comes within 0.4 points of a funded, hosted competitor, reproduced under that
competitor's own exact evaluation method (same model, same context size).

**The result that matters most isn't either number alone — it's that they agree.** Matching
context size alone (cheap model) scored 92.1%; matching context size *and* model *and* prompts
(their own methodology) also scored 92.1%. Swapping in a materially more expensive AI model
changed almost nothing once both systems were given the same amount of context to work with.
That agreement is why the result is trustworthy rather than a favorable measurement picked after
the fact.

## The honest caveat — this is not the out-of-the-box default

`.recall()` defaults to 10 memories per query if you don't specify otherwise. The numbers above
used `.recall(query).k(200)` — explicitly requesting a much larger context window, matching what
the comparison required. Out-of-the-box accuracy (no configuration) is closer to **89%**
(measured at `k=20`, the closest setting actually benchmarked to the true default) — still
strong, but not the headline number without the explicit `.k()` call:

```rust
let memories = mem.recall("What did we discuss about pricing?").k(200).await?;
```

## The one real known weak spot: multi-hop questions

Questions requiring the model to connect two separate facts score noticeably lower than every
other category:

| category | accuracy |
|---|---|
| open-domain | 95.1% |
| single-hop | 91.8% |
| temporal | 89.7% |
| **multi-hop** | **74.0%** |

This was investigated directly rather than left unexplained. The finding, in short: **it is not
a retrieval problem.** Checking the actual evidence cited as "correct answer" for every
multi-hop failure showed the right information was retrieved in 96% of cases — the model simply
didn't connect it correctly. The remaining failures split roughly into genuinely hard inference
(the answer requires outside knowledge the conversation never states), legitimately ambiguous
open-ended questions with more than one reasonable answer, and a small residual rate (0.65%
corpus-wide) of the model returning an empty response under tight reasoning constraints. Two
fix hypotheses were tested (giving the model more reasoning budget; strengthening an
anti-hedging instruction) and both were found to be weak, inconclusive levers rather than real
fixes — reported honestly rather than shipped as a claimed improvement that doesn't hold up.

## Reproducing this

The benchmark harness lives in `bench/locomo/` (`harness.py` for retrieval capture, `qa_eval.py`
for the paid answer-generation and judging steps). Retrieval capture is free and deterministic;
the answer-generation and judging steps call a real LLM API and cost real money proportional to
corpus size and model choice.

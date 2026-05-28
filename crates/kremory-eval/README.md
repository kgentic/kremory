# kremory-eval

Internal quality evaluation harness for the kremory crate. **Not published to crates.io** (`publish = false`).

Implements two evaluation layers:
- **Layer A** — published-comparable benchmarks (LongMemEval, DMR, LoCoMo) with single-number accuracy + per-category breakdown.
- **Layer B** — diagnostic metrics (entity P/R/F1 against 14-domain fixtures + RAGAS Faithfulness/Relevancy/Precision/Recall/Hallucination + graph integrity invariants + custom G-Eval).

Strategy: `.ai-docs/planning/quality-eval-strategy-2026-05-28.md`
Plan: `.ai-docs/plans/rql/plan-v014-quality-eval-phase-1-2026-05-28.md`

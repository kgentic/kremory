# Eval fixture inventory

Every fixture the `kremory-eval` harness depends on, where it lives, and how
it was sourced. For how to run the harness, see [`docs/eval.md`](./eval.md).

---

## Layer A — LongMemEval

### Dataset (live runs)

- **Source**: `xiaowu0162/longmemeval-cleaned` on HuggingFace Hub
- **Variant**: `oracle` (the only one the harness wires up at v0.1.4)
- **Cached path**: `~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/<snapshot>/longmemeval_oracle.json`
- **Sample count**: ~500 questions, six `question_type` values plus
  `_abs`-suffixed abstention variants
- **License**: per upstream (MIT at time of writing)

### Smoke fixtures (local, no network)

`crates/kremory-eval/src/bin/eval.rs::make_smoke_fixtures` defines six
in-memory `LongMemEvalRecord`s covering each question type plus one
abstention case. They are deliberately trivial — they validate the harness
wiring, not the score quality. Used by:

- `--judge mock` (default path when not live)
- `--judge gemma --smoke` (live judge, no HF download)

---

## Layer B — diagnostic metrics

### Entity extraction fixtures

`crates/kremory-eval/fixtures/*.txt` (14 files) — short to medium-length
domain transcripts spanning news, podcast, legal, board meeting, medical,
support, sales, standup, product review, academic, slack, mock interview,
long report, and a short canary snippet.

Ground truth lives in `crates/kremory-eval/fixtures/ground_truth.json`
keyed by fixture filename. P/R/F1 is computed per fixture and aggregated.

### RAGAS fixtures

`crates/kremory-eval/fixtures/ragas/fixture-001.json` … `fixture-020.json`
(20 fixtures). Each fixture carries:

- `question`, `expected_answer`, `expected_contexts`, `expected_entities`
- Used to score faithfulness, answer_relevancy, context_precision,
  context_recall, context_entities_recall, hallucination

The runner builds a synthetic `RagasOutput` by echoing each fixture's
expected fields, so MockJudge runs deterministically. Live runs swap in
`GemmaJudge`.

### Graph integrity invariants

No on-disk fixtures — the test opens `TemporalGraph::open_in_memory()` and
runs structural invariants (`run_invariants`) with `IntegrityConfig`.
Used as a non-quantitative GATE: any invariant failure aborts the Layer B
run.

### Calibration-spike QA pairs

`crates/kremory-eval/fixtures/calibration-spike/qa-pairs.json` — used by
the `calibration_spike` bin to verify GemmaJudge calibration against a
small hand-labelled set. Not part of the regular eval pipeline.

### English dictionary (entity extraction support)

`crates/kremory-eval/fixtures/dictionaries/en_US.aff` and
`en_US.dic` — Hunspell dictionary used by the kremory entity extractor's
OOV detection. Required only by the kremory crate's own tests; included
here for proximity to the diagnostic fixtures that exercise the same code
path.

---

## Baselines

`crates/kremory-eval/baselines/` holds the locked baselines that downstream
runs compare against:

- `v0.1.4-judge-calibration.json` — Gemma judge calibration snapshot
- `v0.1.4-diagnostic-entity-extraction.json` — Layer B entity P/R/F1
  reference
- `v0.1.4-ci-ram-canary.json` — RAM canary baseline (used by the deferred
  `eval-canary.yml` workflow)
- `v0.1.4-scorer-decision.md` — Option A locking note + O4/O6 reconciliation

---

## Updating fixtures

- New entity-extraction fixture: drop a `.txt` file into `fixtures/`, add
  the matching key to `ground_truth.json`, re-run the entity baseline to
  refresh `v0.1.4-diagnostic-entity-extraction.json`.
- New RAGAS fixture: add `fixture-NNN.json` matching the existing schema.
  Note that adding a fixture changes the aggregate — document the shift in
  the next release's CHANGELOG.
- LongMemEval dataset is upstream-managed; do not check it into the repo.

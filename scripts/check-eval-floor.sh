#!/usr/bin/env bash
# check-eval-floor.sh — the eval regression guard TD-242 exists to provide.
#
# NAMING: this gates the eval harness's per-question PASS RATE (the `correct`
# field the substring scorer writes), NOT retrieval recall@k. The repo's own
# note is explicit that the eval is QA-gen, not a retrieval proxy, so calling
# this a "recall floor" would conflate two different numbers. `eval` is the
# repo's established word for this harness (`kremory-eval`, docs/eval.md).
#
# kremory has eval benchmarks and, until this script, NO committed floor that
# anything could fail on. ADR-078's content-search default flip moved the number
# by 32.2pt and not one of ~1,800 deterministic tests noticed, because no test
# can see retrieval quality. TD-099 (CSR traversal) and TD-100 (`graph_edges`
# view) both change how the graph arm is read. This is their safety net.
#
# ── PRIOR ART, AND ITS FAILURE ───────────────────────────────────────────────
# Modelled on GraphQLite's `scripts/check-tck-baseline.sh` (teardown 2026-09-08).
# Their mechanism is right and their floor is broken: it sits 442 scenarios BELOW
# their own advertised number, so their gate cannot fire — a green control
# enforcing nothing. Hence: an unpopulated baseline FAILS LOUDLY; a large
# improvement NAGS to re-baseline in-tool; `--update` says commit it deliberately.
#
# ── IT READS THE HARNESS'S OWN AGGREGATE, NOT THE RAW JSONL ──────────────────
# The first cut recomputed the pass rate from `run-*.jsonl`. That was WRONG and
# would have committed a floor nobody quotes: every record carries `correct`
# INCLUDING the 446 adversarial questions the harness deliberately excludes, so
# recomputing scored 1986 and gave 69.13% where the harness headline is
# 1373/1540 = 89.2%. A rival denominator, silently. See TD-217 — every benchmark
# % has a denominator; state it or it is not comparable. So this reads
# `category_stats` / `scored_questions` from the harness JSON: derive from the
# producer, never re-derive alongside it.
#
# It also pins PROVENANCE (git sha, rrf_k, weights, features, scorer,
# server_mode). A floor compared across different retrieval configs is
# meaningless, so config drift is reported loudly, not folded into the delta.
#
# ── ONE JOB ──────────────────────────────────────────────────────────────────
# A RATCHET, not a target. Fails only when the pass rate drops BELOW the floor,
# overall or in any single category. A guard that fires on ordinary work gets
# disabled, and a disabled guard is how you end up with no guard at all.
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#   bash scripts/check-eval-floor.sh results/floor/<run>.json
#   bash scripts/check-eval-floor.sh --update results/floor/<run>.json
#
# Produce a run with (server on :3179, corpus already ingested — NO paid ingest):
#   bench/locomo/.venv/bin/python bench/locomo/harness.py --skip-ingest \
#     --mode codemem --server-mode hybrid --recall-limit 20 --scorer substring \
#     --base-url http://localhost:3179 --output results/floor/<run>.json
#
# No CI exists in this project. Run via scripts/check-all.sh --results <run.json>.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
BASELINE="scripts/eval-floor-baseline.json"
NAG_THRESHOLD=2.0

UPDATE=0
if [[ "${1:-}" == "--update" ]]; then UPDATE=1; shift; fi
RUN="${1:-}"
if [[ -z "$RUN" ]]; then
  echo "FAIL: no harness results JSON given." >&2
  echo "  usage: bash $0 [--update] results/floor/<run>.json" >&2
  exit 1
fi
[[ -f "$RUN" ]] || { echo "FAIL: results file not found: $RUN" >&2; exit 1; }

KREMORY_BASELINE="$BASELINE" KREMORY_RUN="$RUN" KREMORY_UPDATE="$UPDATE" \
KREMORY_NAG="$NAG_THRESHOLD" python3 scripts/lib/eval_floor.py

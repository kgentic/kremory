#!/usr/bin/env bash
# check-eval-floor.sh — the eval regression guard TD-242 exists to provide.
#
# NAMING: this gates the eval harness's per-question PASS RATE (the `correct`
# field a QA judge writes), NOT retrieval recall@k. The repo's own note is
# explicit that the eval is QA-gen, not a retrieval proxy, so calling this a
# "recall floor" would conflate two different numbers. `eval` is the repo's
# established word for this harness (`kremory-eval`, docs/eval.md), so it is
# borrowed rather than a third vocabulary being minted.
#
# kremory has recall benchmarks and, until this script, NO committed floor that
# anything could fail on. ADR-078 demonstrated the cost: a default-feature flip
# moved recall by 32.2pt and not one of ~1,800 tests noticed, because no test
# can see recall quality. TD-099 (CSR traversal) and TD-100 (`graph_edges` view)
# both change how the graph arm is read. This is their safety net, so it lands
# first.
#
# ── PRIOR ART, AND ITS FAILURE ───────────────────────────────────────────────
# Modelled on GraphQLite's `scripts/check-tck-baseline.sh` (teardown
# 2026-09-08). Their mechanism is right and their floor is broken: it sits 442
# scenarios BELOW their own advertised number, so their gate cannot fire — a
# green control enforcing nothing. That is the failure this script is shaped to
# avoid, which is why:
#
#   - an unpopulated or missing baseline FAILS LOUDLY; it never passes by default
#   - a large improvement over the floor NAGS you to re-baseline, in-tool,
#     rather than trusting a DoD line nobody reads
#   - `--update` prints that the re-baseline must be committed deliberately
#
# ── ONE JOB ──────────────────────────────────────────────────────────────────
# Compare a results file against the committed floor. It is a RATCHET, not a
# target: it fails only when accuracy drops BELOW the floor, overall or in any
# single category. A guard that fires on ordinary work gets disabled, and a
# disabled guard is how you end up with no guard at all.
#
#   overall >= floor AND every category >= its floor  -> PASS
#   overall <  floor                                  -> FAIL (named, with delta)
#   any category < its floor                          -> FAIL (named, with delta)
#   baseline missing / unpopulated                    -> FAIL (with instructions)
#   overall >= floor + NAG_THRESHOLD                  -> PASS + re-baseline nag
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#   bash scripts/check-eval-floor.sh results/run-<stamp>.jsonl
#   bash scripts/check-eval-floor.sh --update results/run-<stamp>.jsonl
#
# No CI exists in this project (org Actions billing suspended). Run it manually
# before a push, like `check-dual-emit.sh` and `check-file-size-ratchet.sh`.
#
# ⛔ Producing a results file requires a benchmark run, which is a standing HITL
#    gate in CLAUDE.md (the ingest path uses the PAID extraction model). This
#    script never runs a benchmark; it only reads a results file you produced.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
BASELINE="scripts/eval-floor-baseline.json"
NAG_THRESHOLD=2.0   # percentage points above floor before it asks to re-baseline

UPDATE=0
if [[ "${1:-}" == "--update" ]]; then UPDATE=1; shift; fi

RESULTS="${1:-}"
if [[ -z "$RESULTS" ]]; then
  echo "FAIL: no results file given." >&2
  echo "  usage: bash $0 [--update] results/run-<stamp>.jsonl" >&2
  exit 1
fi
if [[ ! -f "$RESULTS" ]]; then
  echo "FAIL: results file not found: $RESULTS" >&2
  exit 1
fi

export KREMORY_BASELINE="$BASELINE" KREMORY_RESULTS="$RESULTS" \
       KREMORY_UPDATE="$UPDATE" KREMORY_NAG="$NAG_THRESHOLD"

python3 - <<'PY'
import json, os, sys, datetime

baseline_path = os.environ["KREMORY_BASELINE"]
results_path  = os.environ["KREMORY_RESULTS"]
update        = os.environ["KREMORY_UPDATE"] == "1"
nag           = float(os.environ["KREMORY_NAG"])

# ── score the results file ───────────────────────────────────────────────────
rows = []
with open(results_path) as fh:
    for n, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"FAIL: {results_path}:{n} is not valid JSON ({e})", file=sys.stderr)
            sys.exit(1)

if not rows:
    print(f"FAIL: {results_path} contained zero records. Refusing to score an empty run —"
          f" an empty file must never read as a pass.", file=sys.stderr)
    sys.exit(1)

# `correct` absent is NOT the same as `correct: false`. Absence means the
# harness did not report, and must never fall through to a passing branch.
missing = [r.get("question_id", "?") for r in rows if "correct" not in r]
if missing:
    print(f"FAIL: {len(missing)} record(s) have no `correct` field "
          f"(first: {missing[0]}). Cannot score an unreported run.", file=sys.stderr)
    sys.exit(1)

def pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0

overall = pct(sum(1 for r in rows if r["correct"]), len(rows))
by_cat, cats = {}, {}
for r in rows:
    c = r.get("category", "uncategorised")
    cats.setdefault(c, []).append(bool(r["correct"]))
for c, v in cats.items():
    by_cat[c] = pct(sum(v), len(v))

# ── --update: write the floor ────────────────────────────────────────────────
if update:
    payload = {
        "_comment": "Recall floor for scripts/check-eval-floor.sh (TD-242). "
                    "A RATCHET, not a target. Re-baseline deliberately after a "
                    "measured improvement and COMMIT IT — an unrefreshed floor "
                    "is a green gate enforcing nothing (see GraphQLite, teardown "
                    "2026-09-08).",
        "populated": True,
        "generated_from": results_path,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question_count": len(rows),
        "overall_correct_pct": overall,
        "by_category_correct_pct": dict(sorted(by_cat.items())),
    }
    with open(baseline_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"re-baselined from {results_path}: overall {overall}% over {len(rows)} questions")
    for c, v in sorted(by_cat.items()):
        print(f"    {c:<14} {v}%")
    print("COMMIT THIS DELIBERATELY — a silent re-baseline turns the ratchet into decoration.")
    sys.exit(0)

# ── read + validate the floor ────────────────────────────────────────────────
try:
    with open(baseline_path) as fh:
        base = json.load(fh)
except FileNotFoundError:
    print(f"FAIL: {baseline_path} missing. Create it with:\n"
          f"  bash scripts/check-eval-floor.sh --update <results.jsonl>", file=sys.stderr)
    sys.exit(1)

if not base.get("populated", False):
    print(f"FAIL: {baseline_path} is UNPOPULATED — no floor has been set.\n"
          f"  This is deliberate: the mechanism ships without a floor because\n"
          f"  generating one requires a benchmark run, which is a standing HITL\n"
          f"  gate (paid extraction model). Populate it with:\n"
          f"      bash scripts/check-eval-floor.sh --update <results.jsonl>\n"
          f"  It fails rather than passes so an unset floor can never be mistaken\n"
          f"  for a green gate.", file=sys.stderr)
    sys.exit(1)

floor_overall = base["overall_correct_pct"]
floor_cats    = base.get("by_category_correct_pct", {})

print(f"floor  : {floor_overall}% overall, from {base.get('generated_from','?')} "
      f"({base.get('generated_at','?')}, {base.get('question_count','?')} questions)")
print(f"current: {overall}% overall, from {results_path} ({len(rows)} questions)")

failures = []
if overall < floor_overall:
    failures.append(f"OVERALL {overall}% < floor {floor_overall}% "
                    f"({round(overall - floor_overall, 2)}pt)")
for c, fv in sorted(floor_cats.items()):
    if c not in by_cat:
        failures.append(f"category '{c}' present in floor but ABSENT from results "
                        f"— a vanished category must not read as a pass")
        continue
    if by_cat[c] < fv:
        failures.append(f"category '{c}' {by_cat[c]}% < floor {fv}% "
                        f"({round(by_cat[c] - fv, 2)}pt)")

for c in sorted(set(by_cat) - set(floor_cats)):
    print(f"  note: category '{c}' is new since the floor was set ({by_cat[c]}%) — "
          f"not gated until you re-baseline")

if failures:
    print("\nFAIL — recall regressed below the committed floor:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    print("\nFix the regression. Do NOT re-baseline to make this pass — that is how a\n"
          "gate becomes decoration.", file=sys.stderr)
    sys.exit(1)

print("\nPASS — no category regressed below the floor.")
if overall >= floor_overall + nag:
    print(f"\n⚠️  RE-BASELINE DUE: overall is {round(overall - floor_overall, 2)}pt above the "
          f"floor (>{nag}pt).\n"
          f"    An unrefreshed floor stops protecting anything — GraphQLite's sits 442\n"
          f"    scenarios below its own advertised number and therefore cannot fire.\n"
          f"    Refresh it, on the record:\n"
          f"        bash scripts/check-eval-floor.sh --update {results_path}")
sys.exit(0)
PY

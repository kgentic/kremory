#!/usr/bin/env python3
"""Eval floor ratchet — see scripts/check-eval-floor.sh for the why.

Reads the LoCoMo harness's own aggregate (category_stats / scored_questions)
rather than recomputing from the raw jsonl, because the jsonl carries `correct`
for the 446 adversarial questions the harness deliberately excludes. Scoring all
1986 gives 69.13%; the harness headline is 1373/1540 = 89.2%. Deriving from the
producer avoids inventing a rival denominator (TD-217).
"""
import json
import os
import sys
import datetime

BPATH = os.environ["KREMORY_BASELINE"]
RPATH = os.environ["KREMORY_RUN"]
UPDATE = os.environ["KREMORY_UPDATE"] == "1"
NAG = float(os.environ["KREMORY_NAG"])

# Config keys that make two runs comparable. A change in any of them means the
# floor and the run measured different systems.
PROV_KEYS = ("rrf_k", "content_stream_weight", "graph_degree_weight",
             "temporal_weight", "rerank_enabled", "features")


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def pct(n, d):
    return round(100.0 * n / d, 2)


try:
    run = json.load(open(RPATH))
except Exception as e:  # noqa: BLE001 - any parse failure is fatal and must say so
    die(f"{RPATH} is not readable JSON ({e})")

# A missing field must never fall through to a passing branch.
for k in ("category_stats", "scored_questions", "headline", "provenance", "scorer"):
    if k not in run:
        die(f"{RPATH} has no '{k}'. This does not look like a harness run JSON "
            f"(bench/locomo/harness.py --output).")

cs = run["category_stats"]
if not cs:
    die(f"{RPATH} has an EMPTY category_stats. Refusing to score an empty run — "
        f"emptiness must never read as a pass.")

# ── GRAPH INTEGRITY MUST HAVE RUN (TD-256 / P1.2) ────────────────────────────
# `results/floor/eval-floor-20260908.json` — the run the live floor was built
# from — records:
#     graph_integrity: {"status":"skipped",
#                       "reason":"no --graph-db and no KREMORY_MCP_DB_PATH"}
# The tool existed, was wired in, and was not used; the floor was then committed
# as though the graph had been verified. RECALL-LEDGER §1ter is the reason this
# matters: a graph in which BOTH SPEAKERS of the conversation had been merged out
# of existence scored 98.0%, IDENTICALLY to the healthy one, and the rank-aware
# metric slightly FAVOURED the destroyed graph. No scorer at any tier can see it.
#
# So a skipped integrity check is not a missing nicety — it is the only signal
# that would distinguish "89.2% on a healthy graph" from "89.2% on rubble".
# It FAILS rather than warns, for the same reason the unpopulated-baseline branch
# above fails: an absent check that reports success is the defect being guarded.
gi = run.get("graph_integrity")
if gi is None:
    die(f"{RPATH} has no 'graph_integrity' block at all.\n"
        f"  Produced by a harness older than TD-223. Re-run the harness so the\n"
        f"  graph can be verified — a score on an unverified graph is not evidence\n"
        f"  the memory survived consolidation (RECALL-LEDGER 1ter).")

gi_status = gi.get("status")
if gi_status != "checked":
    die(f"graph integrity was NOT VERIFIED for this run (status={gi_status!r}).\n"
        f"  reason: {gi.get('reason', '<none given>')}\n"
        f"\n"
        f"  FIX: re-run the harness with the graph database, e.g.\n"
        f"      bench/locomo/.venv/bin/python bench/locomo/harness.py --skip-ingest \\\n"
        f"        --graph-db $PWD/.context/full-corpus.db \\\n"
        f"        --mode codemem --server-mode hybrid --recall-limit 20 \\\n"
        f"        --scorer substring --output {RPATH}\n"
        f"  (or export KREMORY_MCP_DB_PATH before the run)\n"
        f"\n"
        f"  WHY THIS BLOCKS: a corrupted graph scores the SAME as a healthy one.\n"
        f"  On 2026-08-17 a run scored 149/152 = 98.0% on a graph where both\n"
        f"  speakers of the dialogue had been merged away. A skipped check is\n"
        f"  indistinguishable from a passed one, so it must never be either.")

# Reported, never gated (plan 2026-09-18, R5). Fan-ins in particular include
# legitimate canonicalisation at this threshold — a human reads the names.
print(f"graph  : {gi['chained_entities']} chained, {gi['fanin_entities']} fan-ins "
      f"(worst={gi['worst_fanin']}) across {gi['live_merges']} live merges")
if gi["chained_entities"] or gi["fanin_entities"]:
    print("         ⚠️  the graph this score was measured on shows merge damage; "
          "the score CANNOT see it (RECALL-LEDGER 1ter)")

# ── WHICH SURFACE WAS MEASURED (P1.3) ────────────────────────────────────────
# `mode=hybrid` runs a SECOND independent BM25 pass and rrf_merges it on top of
# the library's already-fused list, at the REST layer, in code no library
# consumer executes (kremory-http.rs). Measured 82.1 recall@10 vs 77.2 for the
# library path. A floor built on hybrid does not gate what `cargo add kremory`
# ships, and the number must say so wherever it is quoted.
# `provenance.binary` is a DICT (exe / crate_version / features / mtime), not a
# string — a first cut here formatted the whole blob into the surface line. Take
# the path and fall back loudly rather than rendering a dict into a report.
binary_prov = run["provenance"].get("binary")
if isinstance(binary_prov, dict):
    binary = binary_prov.get("exe", "<no exe recorded>")
    # Repo-relative: an absolute path pins one machine's checkout into a
    # committed floor file and stops reading as a comparison across runs.
    binary = binary.split("/pattaya/", 1)[-1] if "/pattaya/" in binary else binary
else:
    binary = binary_prov or "<unrecorded>"
server_mode = run.get("server_mode", "<unrecorded>")
surface = f"{binary} (server_mode={server_mode})"
print(f"surface: {surface}")
if server_mode != "recall":
    print(f"         ⚠️  NOT the shipped library path. server_mode={server_mode!r} "
          f"adds a REST-layer BM25 arm + a second fusion that no library consumer\n"
          f"         runs. This floor gates the SERVER, not `cargo add kremory`.\n"
          f"         A like-for-like mode=recall comparison is an OPEN QUESTION — "
          f"it needs a bench run and is deliberately not run here.")

corr = sum(v["correct"] for v in cs.values())
tot = sum(v["total"] for v in cs.values())
if tot == 0:
    die(f"{RPATH} scored zero questions.")
if tot != run["scored_questions"]:
    die(f"category_stats totals {tot} but scored_questions says "
        f"{run['scored_questions']} — the denominator disagrees with itself.")

overall = pct(corr, tot)
by_cat = {k: pct(v["correct"], v["total"]) for k, v in cs.items()}
prov = {k: run["provenance"].get(k) for k in PROV_KEYS}
prov["git_sha"] = run["provenance"].get("git_sha")
prov["scorer"] = run.get("scorer")
prov["server_mode"] = run.get("server_mode")

if UPDATE:
    payload = {
        "_comment": "Eval floor for scripts/check-eval-floor.sh (TD-242). A RATCHET, "
                    "not a target. Re-baseline deliberately after a measured "
                    "improvement and COMMIT IT — an unrefreshed floor is a green gate "
                    "enforcing nothing (see GraphQLite, teardown 2026-09-08).",
        "populated": True,
        "generated_from": RPATH,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": run["headline"],
        # P1.3 — the floor must carry WHICH SURFACE produced it. `mode=hybrid` on
        # kremory-http is not the library path a `cargo add kremory` consumer gets
        # (+4.9 recall@10 from a REST-layer second fusion). Recorded here so the
        # qualifier travels with the committed number, the same way TD-217 built
        # the denominator into `headline`.
        "measured_surface": surface,
        "graph_integrity": {
            "status": gi_status,
            "chained_entities": gi["chained_entities"],
            "fanin_entities": gi["fanin_entities"],
            "worst_fanin": gi["worst_fanin"],
            "live_merges": gi["live_merges"],
        },
        "scored_questions": tot,
        "unscored_questions": run.get("unscored_questions"),
        "overall_correct_pct": overall,
        "by_category_correct_pct": dict(sorted(by_cat.items())),
        "provenance": prov,
    }
    with open(BPATH, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"re-baselined from {RPATH}\n  {run['headline']}")
    for c, v in sorted(by_cat.items()):
        print(f"    {c:<14} {v}%")
    print(f"  provenance: rrf_k={prov['rrf_k']} features={prov['features']} "
          f"scorer={prov['scorer']} server_mode={prov['server_mode']}")
    print("COMMIT THIS DELIBERATELY — a silent re-baseline turns the ratchet into "
          "decoration.")
    sys.exit(0)

try:
    base = json.load(open(BPATH))
except FileNotFoundError:
    die(f"{BPATH} missing. Create it with:\n"
        f"  bash scripts/check-eval-floor.sh --update <run.json>")

if not base.get("populated", False):
    die(f"{BPATH} is UNPOPULATED — no floor has been set.\n"
        f"  Populate with: bash scripts/check-eval-floor.sh --update <run.json>\n"
        f"  It FAILS rather than passes so an unset floor can never be mistaken\n"
        f"  for a green gate.")

floor_overall = base["overall_correct_pct"]
floor_cats = base.get("by_category_correct_pct", {})
print(f"floor  : {floor_overall}% over {base.get('scored_questions','?')} scored "
      f"({base.get('generated_at','?')})")
print(f"current: {overall}% over {tot} scored — {run['headline']}")

# Config drift is reported, never silently absorbed into the delta.
bprov = base.get("provenance", {})
drift = [f"{k}: floor={bprov.get(k)!r} current={prov[k]!r}"
         for k in PROV_KEYS if bprov.get(k) != prov[k]]
if drift:
    print("\n⚠️  CONFIG DRIFT vs the floor — this comparison is NOT like-for-like:")
    for d in drift:
        print(f"    - {d}")
    print("    Re-baseline deliberately if the new config is the intended default.")

fails = []
if overall < floor_overall:
    fails.append(f"OVERALL {overall}% < floor {floor_overall}% "
                 f"({round(overall - floor_overall, 2)}pt)")
for c, fv in sorted(floor_cats.items()):
    if c not in by_cat:
        fails.append(f"category '{c}' in floor but ABSENT from run — a vanished "
                     f"category must not read as a pass")
    elif by_cat[c] < fv:
        fails.append(f"category '{c}' {by_cat[c]}% < floor {fv}% "
                     f"({round(by_cat[c] - fv, 2)}pt)")
for c in sorted(set(by_cat) - set(floor_cats)):
    print(f"  note: category '{c}' is new since the floor ({by_cat[c]}%) — not gated "
          f"until you re-baseline")

if fails:
    print("\nFAIL — eval regressed below the committed floor:", file=sys.stderr)
    for f in fails:
        print(f"  - {f}", file=sys.stderr)
    print("\nFix the regression. Do NOT re-baseline to make this pass — that is how a\n"
          "gate becomes decoration.", file=sys.stderr)
    sys.exit(1)

print("\nPASS — no category regressed below the floor.")
if overall >= floor_overall + NAG:
    print(f"\n⚠️  RE-BASELINE DUE: overall is {round(overall - floor_overall, 2)}pt "
          f"above the floor (>{NAG}pt).\n"
          f"    An unrefreshed floor stops protecting anything.\n"
          f"    Refresh it, on the record:\n"
          f"        bash scripts/check-eval-floor.sh --update {RPATH}")
sys.exit(0)

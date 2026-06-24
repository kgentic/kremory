#!/usr/bin/env python3
"""Collate the model-benchmark TSV + per-model JSON reports into results.md.

Sorts by F1, splits non-thinking (interactive-default candidates) vs thinking
(deferred-quality candidates), applies the recommendation rule, and appends a
per-model confusion breakdown. Reproducible: reads BENCH_OUT (default research dir).

Usage: python3 scripts/model-benchmark/collate.py [out_dir]
"""
import csv, json, os, subprocess, sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(["git", "-C", str(DIR), "rev-parse", "--show-toplevel"]).decode().strip())
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get(
    "BENCH_OUT", REPO / ".ai-docs/research/local-model-benchmark-2026-06-24"))
TSV, MD = OUT / "results.tsv", OUT / "results.md"


def num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


rows = []
with open(TSV) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        r["_f1"] = num(r.get("f1"))
        r["_prec"] = num(r.get("precision"))
        rows.append(r)

rows.sort(key=lambda r: (r["_f1"] is None, -(r["_f1"] or 0)))
nonthink = [r for r in rows if r["thinking_cap"] == "NO" and r["think_off"] != "yes"]
think = [r for r in rows if r["thinking_cap"] == "YES" or r["think_off"] == "yes"]

COLS = ["model", "think_off", "size", "recall", "precision", "f1", "over_extr",
        "facts", "min_rel", "stage_ms_max", "fits_30s", "wallclock_s", "pullable"]


def table(rs):
    h = "| " + " | ".join(COLS) + " |\n|" + "|".join(["---"] * len(COLS)) + "|\n"
    return h + "".join("| " + " | ".join(str(r.get(c, "?")) for c in COLS) + " |\n" for r in rs)


# Recommendation rule
fit = [r for r in nonthink if r["fits_30s"] == "YES" and r["_f1"] is not None and r["pullable"] == "YES"]
interactive = max(fit, key=lambda r: (r["_f1"], -num(r["size"].rstrip("GBMB") or 0) if r["size"][:-2].replace(".", "").isdigit() else 0), default=None)
thinkable = [r for r in think if r["_f1"] is not None and r["pullable"] == "YES"]
deferred = max(thinkable, key=lambda r: r["_f1"], default=None)

lines = [
    "# Local-model benchmark — results",
    "",
    f"Hardware: Apple Silicon M4 Max, 36GB · latency is M4-Max-only (cross-hardware parked).",
    f"Fixture: mock_interview (10 ground-truth entities). `precision`=true precision (correct/extracted), `recall`=found/expected.",
    "",
    "## Recommendation",
    f"- **interactive `with_ollama` default** → `{interactive['model'] if interactive else 'NONE FIT'}`"
    + (f" (F1 {interactive['f1']}%, precision {interactive['precision']}%, {interactive['size']}, slowest call {interactive['stage_ms_max']}ms < 30s)" if interactive else ""),
    f"- **deferred/dream quality** → `{deferred['model'] if deferred else 'NONE'}`"
    + (f" (F1 {deferred['f1']}%, think_off={deferred['think_off']})" if deferred else ""),
    "",
    "## Non-thinking (interactive candidates), sorted by F1",
    table(nonthink),
    "## Thinking (deferred candidates; native + think:false), sorted by F1",
    table(think),
    "## Per-model confusion (from JSON reports)",
]

for r in rows:
    rp = r.get("report", "")
    if not rp or not Path(rp).exists():
        continue
    d = json.loads(Path(rp).read_text())
    conf = d.get("confusion", [])
    bad = [c for c in conf if c["outcome"] != "correct"]
    tag = f"{r['model']} (think_off={r['think_off']})"
    lines.append(f"\n**{tag}** — recall {d.get('recall',0)*100:.0f}% precision {d.get('precision',0)*100:.0f}% · misses/wrong: "
                 + (", ".join(f"{c['name']}[{c['expected']}]→{c['outcome']}" for c in bad) if bad else "none (all correct)"))

MD.write_text("\n".join(lines) + "\n")
print(f"wrote {MD}")
if interactive:
    print(f"INTERACTIVE → {interactive['model']} (F1 {interactive['f1']}%)")
if deferred:
    print(f"DEFERRED → {deferred['model']} (F1 {deferred['f1']}%, think_off={deferred['think_off']})")

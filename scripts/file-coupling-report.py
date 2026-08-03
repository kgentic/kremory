#!/usr/bin/env python3
"""file-coupling-report.py — per-file coupling + size census for split planning.

Joins three independent sources so a split plan is ranked by evidence, not by line count:

  1. LOC            — `wc -l` over tracked .rs files
  2. COUPLING       — graphify's AST graph (graphify-out/graph.json): nodes per file,
                      fan-in / fan-out, distinct external files coupled, community spread
  3. DEFECT DENSITY — mentions of each file in the tech-debt register

Why coupling and not LOC: a 3,000-line file that nothing else references is a bad file;
a 3,000-line file that 40 other files reach into is an architectural problem. Only the
second one is urgent, and LOC cannot tell them apart.

Why community spread: graphify runs community detection over the symbol graph. When one
file's symbols land in many distinct communities, those communities are candidate module
boundaries — i.e. the split seams are already computed.

Usage:  scripts/file-coupling-report.py [--graph graphify-out/graph.json] [--min-loc 800]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REGISTER = Path(".ai-docs/tech-debt/tech-debt-register.md")


def loc_census(min_loc: int) -> dict[str, int]:
    """LOC per tracked .rs file, excluding build artefacts."""
    out = subprocess.run(
        ["find", "crates", "-name", "*.rs", "-not", "-path", "*/target/*"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    census = {}
    for f in out:
        try:
            n = sum(1 for _ in open(f, encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if n >= min_loc:
            census[f] = n
    return census


def graph_coupling(graph_path: Path) -> dict[str, dict]:
    """Per-file coupling metrics derived from the graphify AST graph."""
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))

    node_file: dict[str, str] = {}
    node_comm: dict[str, int] = {}
    for n in nodes:
        nid = n.get("id")
        src = n.get("source_file") or ""
        if nid and src:
            node_file[nid] = src
            if n.get("community") is not None:
                node_comm[nid] = n["community"]

    stats: dict[str, dict] = defaultdict(
        lambda: {"nodes": 0, "internal": 0, "fan_in": 0, "fan_out": 0,
                 "peers_in": set(), "peers_out": set(), "communities": set()}
    )
    for nid, src in node_file.items():
        stats[src]["nodes"] += 1
        if nid in node_comm:
            stats[src]["communities"].add(node_comm[nid])

    for e in links:
        s, t = e.get("source"), e.get("target")
        fs, ft = node_file.get(s), node_file.get(t)
        if not fs or not ft:
            continue
        if fs == ft:
            stats[fs]["internal"] += 1
        else:
            stats[fs]["fan_out"] += 1
            stats[fs]["peers_out"].add(ft)
            stats[ft]["fan_in"] += 1
            stats[ft]["peers_in"].add(fs)
    return stats


def register_mentions() -> dict[str, list[str]]:
    """Map file basename/path -> TD ids that mention it in the register."""
    if not REGISTER.exists():
        return {}
    text = REGISTER.read_text(encoding="utf-8", errors="replace")
    # Split into per-TD blocks so a mention is attributed to the right id.
    blocks = re.split(r"^###\s+(TD-\d+)", text, flags=re.MULTILINE)
    mentions: dict[str, set[str]] = defaultdict(set)
    for i in range(1, len(blocks) - 1, 2):
        td, body = blocks[i], blocks[i + 1]
        for m in re.finditer(r"[\w/]+\.rs", body):
            mentions[m.group(0)].add(td)
    return {k: sorted(v) for k, v in mentions.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="graphify-out/graph.json")
    ap.add_argument("--min-loc", type=int, default=800)
    args = ap.parse_args()

    gp = Path(args.graph)
    if not gp.exists():
        print(f"error: {gp} not found — run `graphify extract . --code-only` first")
        return 1

    loc = loc_census(args.min_loc)
    coup = graph_coupling(gp)
    mentions = register_mentions()

    rows = []
    for path, n_loc in loc.items():
        c = coup.get(path, {})
        peers_in = len(c.get("peers_in", ()) or ())
        peers_out = len(c.get("peers_out", ()) or ())
        comms = len(c.get("communities", ()) or ())
        tds = sorted({td for key, v in mentions.items()
                      if path.endswith(key) or Path(path).name == key for td in v})
        # Rank: coupling breadth x size, with defect density as a multiplier.
        score = (peers_in + peers_out) * (n_loc / 1000) * (1 + len(tds) * 0.5)
        rows.append({
            "file": path, "loc": n_loc, "nodes": c.get("nodes", 0),
            "fan_in": c.get("fan_in", 0), "fan_out": c.get("fan_out", 0),
            "peers_in": peers_in, "peers_out": peers_out,
            "communities": comms, "tds": tds, "score": round(score, 1),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    print(f"{'score':>7} {'LOC':>5} {'nodes':>5} {'in':>4} {'out':>4} {'comms':>5}  file / TDs")
    print("-" * 100)
    for r in rows[:30]:
        print(f"{r['score']:>7} {r['loc']:>5} {r['nodes']:>5} {r['peers_in']:>4} "
              f"{r['peers_out']:>4} {r['communities']:>5}  {r['file']}")
        if r["tds"]:
            print(f"{'':>34}  └─ {', '.join(r['tds'])}")

    Path("graphify-out/file-coupling.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nwrote graphify-out/file-coupling.json ({len(rows)} files >= {args.min_loc} LOC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

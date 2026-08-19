#!/usr/bin/env python3
"""Apply the PRE-REGISTERED dream keep-or-cut rule to two completed arms.

Rule (frozen 2026-08-19, before any measurement):
`.ai-docs/decisions/dream-keep-or-cut-prereg-2026-08-19.md`

This script does NOT decide anything new. It reads the two arms, runs the
validity gate, computes the difference, and prints the outcome the rule already
committed to. Its whole value is that it cannot be argued with afterwards.

Usage:
    dream_ab_decide.py --on-db  <path> --off-db  <path> \
                       --on-res <path> --off-res <path> \
                       [--on-verdicts <path> --off-verdicts <path>]

Without verdict files it reports the SUBSTRING scorer only and refuses to issue
a verdict, because substring is explicitly NOT decisive under the rule.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from math import comb
from pathlib import Path

BAR_PP = 5.0  # pre-registered


def merge_rows(db: Path) -> int:
    """Live entity-merge rows. NOT COUNT(*) on a vector-indexed table — but
    graph_mutation_log is a plain table, so COUNT(*) is safe here. Counting a
    named column anyway, because the blanket rule exists precisely so nobody has
    to make this judgement per-table (SYSTEM-PRIMER gotcha #1)."""
    if not db.exists():
        return -1
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT kind FROM graph_mutation_log "
            "WHERE kind = 'entity_merge' AND undone_at IS NULL"
        )
        return len(cur.fetchall())
    except sqlite3.Error as e:
        print(f"  ! could not read {db.name}: {e}", file=sys.stderr)
        return -1
    finally:
        con.close()


def per_question(results: Path) -> dict[str, bool]:
    """question-id -> correct?, from a harness results.json (substring scorer)."""
    # Field is `is_correct` — VERIFIED against a real results.json, not guessed.
    # The first draft looked for `correct`/`answer_in_memories`, neither of which
    # exists; it would have returned {} and reported "NO SHARED QUESTIONS", which
    # reads like a finding rather than a broken parser.
    #
    # 47 of the 199 questions are adversarial and deliberately UNSCORED for
    # correctness (`is_correct` absent/None). They are excluded here rather than
    # counted as wrong — scoring them would put the same constant in both arms and
    # dilute the very difference this comparison is measuring.
    d = json.loads(results.read_text())
    rows = d.get("results") or []
    out = {}
    for r in rows:
        qid = r.get("question_id")
        if qid is None:
            continue
        val = r.get("is_correct")
        if val is None:
            continue
        out[str(qid)] = bool(val)
    return out


def verdict_map(path: Path) -> dict[str, bool]:
    """question-id -> CORRECT?, from a qa_eval answer-judge verdicts.jsonl."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = str(r.get("question_id") or r.get("qid") or r.get("question"))
        v = r.get("verdict") or r.get("judgement") or r.get("label")
        if v is None:
            continue
        out[qid] = str(v).strip().upper().startswith("CORRECT")
    return out


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(on: dict[str, bool], off: dict[str, bool], label: str):
    shared = sorted(set(on) & set(off))
    if not shared:
        print(f"  {label}: NO SHARED QUESTIONS — cannot compare")
        return None
    on_n = sum(on[q] for q in shared)
    off_n = sum(off[q] for q in shared)
    on_pct = 100.0 * on_n / len(shared)
    off_pct = 100.0 * off_n / len(shared)
    b = sum(1 for q in shared if on[q] and not off[q])   # dream-on only
    c = sum(1 for q in shared if off[q] and not on[q])   # dream-off only
    delta = on_pct - off_pct
    print(f"  {label}: n={len(shared)}  on={on_pct:.1f}% ({on_n})  "
          f"off={off_pct:.1f}% ({off_n})  delta={delta:+.1f}pp")
    print(f"    discordant: on-only={b}  off-only={c}  "
          f"McNemar exact p={mcnemar_exact_p(b, c):.4f}")
    if b + c == 0:
        print("    ⚠ ZERO discordant pairs — the lever changed no answer at all.")
    return delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-db", type=Path, required=True)
    ap.add_argument("--off-db", type=Path, required=True)
    ap.add_argument("--on-res", type=Path, required=True)
    ap.add_argument("--off-res", type=Path, required=True)
    ap.add_argument("--on-verdicts", type=Path)
    ap.add_argument("--off-verdicts", type=Path)
    a = ap.parse_args()

    print("=== VALIDITY GATE (checked before any score is read) ===")
    on_merges, off_merges = merge_rows(a.on_db), merge_rows(a.off_db)
    print(f"  dream-on  live entity_merge rows: {on_merges}  (rule: must be >= 1)")
    print(f"  dream-off live entity_merge rows: {off_merges} (rule: must be == 0)")
    gate_ok = on_merges >= 1 and off_merges == 0
    if not gate_ok:
        print("\n  GATE FAILED — the toggle did not change the graph, so the")
        print("  comparison is VOID and the scores below mean nothing.")

    print("\n=== SCORES ===")
    sub = compare(per_question(a.on_res), per_question(a.off_res),
                  "substring (NOT decisive)")

    qa = None
    if a.on_verdicts and a.off_verdicts:
        qa = compare(verdict_map(a.on_verdicts), verdict_map(a.off_verdicts),
                     "qa-gen   (PRIMARY)")
    else:
        print("  qa-gen   (PRIMARY): NOT RUN — no verdict files supplied")

    print("\n=== PRE-REGISTERED OUTCOME ===")
    if not gate_ok:
        print("  CUT — validity gate failed; unanswerable in this session.")
        return 0
    if qa is None:
        print("  NO VERDICT — the primary metric was not measured.")
        print("  Substring alone is explicitly non-decisive under the rule.")
        return 2
    if qa >= BAR_PP:
        print(f"  KEEP (provisional) — qa-gen delta {qa:+.1f}pp >= +{BAR_PP}pp bar.")
        print("  State dream's wall-clock cost alongside this number.")
    else:
        print(f"  CUT — qa-gen delta {qa:+.1f}pp < +{BAR_PP}pp bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

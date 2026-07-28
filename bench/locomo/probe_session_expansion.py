#!/usr/bin/env python3
"""W3.1 — size CORE's `replaceWithCompacts` (session expansion) against our own misses.

THE MECHANISM BEING SIZED
-------------------------
CORE's read path, on a hit for any turn of a session, swaps in that session's
compacted narrative — i.e. a hit ANYWHERE in a session makes the WHOLE session
reachable, index-free and at zero retrieval cost.

THE QUESTION THIS ANSWERS
-------------------------
For the evidence turns we MISS, does a sibling turn from the SAME session already
appear in the top-k? If yes, session expansion would have surfaced the answer for
free. If no, the mechanism has nothing to expand from and the direction is dead.

WHY THE FIRST ATTEMPT WAS VOID
------------------------------
It grouped by kremory's own columns: `source_id` is unique per episode and
`saga_id` is all NULL, so kremory has no session grouping to test. The dataset
does — `dia_id` is literally `D<session>:<turn>` — so grouping is done on the
DATASET's sessions here, which is what CORE's mechanism keys on anyway.

Reuses `evidence_eval`'s turn-matching verbatim so the "was this turn retrieved"
decision is identical to the metric everything else in this campaign is scored on.

USAGE
  python3 probe_session_expansion.py results/<run>.json [--k 10]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evidence_eval import evidence_ids, load_turns, norm, turn_in


def session_of(dia_id: str) -> str:
    """`D12:3` -> `D12`. The dataset's own session grouping, which is exactly the
    unit CORE's compaction keys on."""
    return str(dia_id).split(":", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).parent / "data" / "locomo10.json")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    turns_by_sample = load_turns(args.dataset)
    rows = json.loads(args.run.read_text())["results"]

    # Per-question outcomes.
    n_scorable = 0
    n_total_miss = 0            # hit-rate 0: no evidence turn anywhere in top-k
    n_total_miss_rescuable = 0  # ... but a retrieved item sits in a gold session
    n_partial = 0               # some but not all evidence retrieved
    n_partial_rescuable = 0
    missed_turns = 0
    missed_turns_in_touched_session = 0
    rescue_examples: list[tuple[str, str]] = []

    for row in rows:
        if row.get("category") == "adversarial":
            continue
        turns = turns_by_sample.get(row.get("sample_id"), {})
        ev_ids = [e for e in evidence_ids(row) if e in turns]
        if not ev_ids:
            continue
        n_scorable += 1

        mems = [norm(m) for m in (row.get("recalled_memories") or [])][: args.k]

        # Which of the CONVERSATION's turns does the retrieved set carry? That set
        # (not just the gold ones) is what determines which sessions were touched.
        touched_sessions: set[str] = set()
        for did, text in turns.items():
            if any(turn_in(text, m) for m in mems):
                touched_sessions.add(session_of(did))

        found = {e for e in ev_ids if any(turn_in(turns[e], m) for m in mems)}
        missing = [e for e in ev_ids if e not in found]

        missed_turns += len(missing)
        missed_turns_in_touched_session += sum(
            1 for e in missing if session_of(e) in touched_sessions
        )

        if not found:
            n_total_miss += 1
            if any(session_of(e) in touched_sessions for e in ev_ids):
                n_total_miss_rescuable += 1
                if len(rescue_examples) < 5:
                    rescue_examples.append((row.get("question", "")[:90], ",".join(ev_ids)))
        elif missing:
            n_partial += 1
            if any(session_of(e) in touched_sessions for e in missing):
                n_partial_rescuable += 1

    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:5.1f}%" if b else "   n/a"

    print(f"=== session-expansion ceiling — {args.run.name} (k={args.k}, n={n_scorable}) ===")
    print()
    print("  THE KILL CRITERION is the first row: complete misses that a session")
    print("  hit would have rescued. <5% => the mechanism has nothing to expand from.")
    print()
    print(f"  complete misses (no evidence in top-k)      {n_total_miss:5d}")
    print(f"    ... rescuable by session expansion        {n_total_miss_rescuable:5d}"
          f"   {pct(n_total_miss_rescuable, n_total_miss)} of misses"
          f"   {pct(n_total_miss_rescuable, n_scorable)} of corpus")
    print()
    print(f"  partial misses (some evidence retrieved)    {n_partial:5d}")
    print(f"    ... completable by session expansion      {n_partial_rescuable:5d}"
          f"   {pct(n_partial_rescuable, n_partial)} of partials")
    print()
    print(f"  missed evidence TURNS                       {missed_turns:5d}")
    print(f"    ... in a session already touched          {missed_turns_in_touched_session:5d}"
          f"   {pct(missed_turns_in_touched_session, missed_turns)}")
    if rescue_examples:
        print()
        print("  sample rescuable complete-misses:")
        for q, ev in rescue_examples:
            print(f"    [{ev}] {q}")


if __name__ == "__main__":
    main()

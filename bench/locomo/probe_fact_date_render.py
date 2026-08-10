#!/usr/bin/env python3
"""Kill-gate probe: would rendering `RetrievedFact.valid_at` into the wire string
make the temporal category measurable on `scorer=substring`?

WHY THIS EXISTS
---------------
`crates/kremory-mcp/src/bin/kremory-http.rs:458` renders a recall result as

    "{entity_name}: {summary}" + ". " + ".".join(f.fact for f in facts)

`RetrievedFactWire.valid_at` (`crates/kremory-mcp/src/params.rs:97`, a
NON-optional `String`) is therefore dropped at the wire boundary. RECALL-LEDGER
§4.17-RESULT measured "0 of 199 questions contain any ISO date" and concluded the
temporal category has a hard floor of 33/37 — correct observation, but the cause
is this dropped field, not an unwinnable question class.

Before building the one-line render fix, this probe answers the two
PRE-REGISTERED conditions (CLAUDE.md Rule 43):

  PR-1  >=2 of the 4 gold-"2022" questions must have a 2022-dated fact anchored
        to an entity in their top-k. Below 2 => the recall justification is
        withdrawn.
  PR-2  FALSE-POSITIVE CONTROL. How many ALREADY-answered questions would gain a
        bare "2022" token from date text alone? A high count means any measured
        gain is an artefact of the same class as the WITHDRAWN 91.6% row
        (RECALL-LEDGER §"THE HEADLINE"), where `"" in anything` scored 446 free
        points.

RESULT (2026-08-10, `.context/td187-round2-bench/`): PR-1 passes 2/4; **PR-2
fires at 195/199**. See RECALL-LEDGER §4.19.

Usage:
    python3 probe_fact_date_render.py <run.json> <run.db>
Cost: FREE. Pure offline SQL + JSON. No LLM call, no server.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict


def entity_anchor_dates(db: str) -> dict[str, set[str]]:
    """entity id -> distinct `YYYY-MM-DD` of every fact it anchors.

    Both sides: an entity anchors a fact as its SUBJECT or as its OBJECT, and
    `flatten_result_content` joins whatever `RetrievedContextWire.facts` carries
    for that entity — which is the entity's connected facts, either direction.
    """
    con = sqlite3.connect(db)
    dates: dict[str, set[str]] = defaultdict(set)
    for eid, vf in con.execute("SELECT subject_id, valid_from FROM facts"):
        dates[eid].add(vf[:10])
    for eid, vf in con.execute(
        "SELECT object_id, valid_from FROM facts WHERE object_id IS NOT NULL"
    ):
        dates[eid].add(vf[:10])
    con.close()
    return dates


def result_entity(memory: str) -> str:
    """Recover the entity id from a flattened wire string.

    `flatten_result_content` emits `"{entity_name}: {summary}. {fact}. ..."`, so
    the text before the FIRST colon is the entity. Lowercased because
    `entities.id` is a slug.
    """
    return memory.split(":", 1)[0].strip().lower()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    run_path, db_path = sys.argv[1], sys.argv[2]

    rows = json.load(open(run_path))["results"]
    dates = entity_anchor_dates(db_path)

    # The token every gold answer in the target set is, verbatim.
    TARGET = "2022"
    anchors = {e for e, ds in dates.items() if any(d.startswith(TARGET) for d in ds)}

    print(f"entities anchoring a {TARGET} fact: {sorted(anchors)}")

    gold_target = [r for r in rows if str(r.get("expected_answer", "")).strip() == TARGET]

    # ---- PR-1 -----------------------------------------------------------
    pr1 = 0
    print(f"\nPR-1  gold=='{TARGET}' questions ({len(gold_target)}):")
    for r in gold_target:
        present = {result_entity(m) for m in r.get("recalled_memories") or []} & anchors
        if present:
            pr1 += 1
        print(
            f"  {r['question_id']:8} correct={r['is_correct']!s:5} "
            f"anchor_in_topk={sorted(present) or 'NO'}"
        )
    print(f"  => PR-1: {pr1}/{len(gold_target)} "
          f"({'PASS' if pr1 >= 2 else 'FAIL — withdraw the recall justification'})")

    # ---- PR-2 -----------------------------------------------------------
    contaminated = [
        r for r in rows
        if {result_entity(m) for m in r.get("recalled_memories") or []} & anchors
    ]
    n, total = len(contaminated), len(rows)
    already_ok = sum(1 for r in contaminated if r["is_correct"])
    print(f"\nPR-2  questions whose top-k contains a {TARGET}-anchoring entity: "
          f"{n}/{total} ({100 * n / total:.1f}%)")
    print(f"      of those, ALREADY correct: {already_ok}")
    print(f"      => every one of those {n} would newly contain the bare token "
          f"'{TARGET}' after the render fix, regardless of whether the date "
          f"answers the question.")
    verdict = "FIRED — substring gain is uninterpretable" if n > total * 0.5 else "clear"
    print(f"  => PR-2: {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Kill-gate probe: do `format=structured` and `format=text` retrieve the SAME items?

WHY THIS EXISTS
---------------
The proposed benchmark fix (RECALL-LEDGER §4.19(e)) is to have the harness ask for
`format=text&template=temporal_facts` — the rendering a real MCP agent gets — instead
of the default `format=structured`. That is only a clean A/B if the two formats differ
in RENDERING and not in RETRIEVAL.

Reading the source says they may NOT be equivalent at the HTTP layer (§4.19(f)):

  * `format=text` returns at `kremory-http.rs:499`, BEFORE the `mode` match at `:520`.
  * `RecallParams` — the type `do_recall` consumes — has NO `mode` field
    (`params.rs:193-205`), so `mode` is pure HTTP dispatch.
  * `format=structured&mode=hybrid` runs `hybrid_mode_results` (`:695`), which RRF-merges
    an ALREADY-content-fused recall arm against a SECOND raw-BM25 arm.

So `structured&hybrid` plausibly takes TWO doses of the content stream and `text&hybrid`
takes one.

THE DECISIVE TEST NEEDS NO ID-INFERENCE
---------------------------------------
`format=text` returns one opaque string, so its item ids cannot be read directly. But the
claim "`format=text` ignores `mode`" makes a crisp, checkable prediction:

    P1  text&mode=hybrid  ==  text&mode=recall     (BYTE-IDENTICAL)
    P2  structured&mode=hybrid  !=  structured&mode=recall

If both hold, `text&hybrid` corresponds to the RECALL-mode retrieval, not hybrid's — and
switching the harness to `format=text` silently changes retrieval as well as rendering.

PRE-REGISTERED (written before running):
    The rendering fix is a CLEAN A/B only if P1 is FALSE or P2 is FALSE.
    If P1 is TRUE **and** P2 is TRUE, the "rendering-only" framing is REFUTED and the
    retrieval-parity defect must be fixed BEFORE the harness is switched.

Cost: one local server boot. ZERO LLM calls. £0. No ingest, no paid path.

Usage:
    KREMORY_MCP_DB_PATH=<db> PORT=3181 target/release/kremory-http &
    python3 probe_render_retrieval_parity.py <base-url> <namespace>
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

K = 50

# Mixed categories. The first four are the gold-"2022" temporal questions from
# RECALL-LEDGER §4.19(c) — the set the rendering change is supposed to reach.
QUESTIONS = [
    "When did Melanie paint a sunrise?",
    'When did Melanie read the book "nothing is impossible"?',
    "When did Caroline and Melanie go to a pride fesetival together?",
    "When did Melanie's friend adopt a child?",
    "What is Caroline's relationship status?",
    "Where did Caroline move from 4 years ago?",
]


def fetch(base: str, ns: str, q: str, mode: str, fmt: str | None) -> dict:
    params = {"q": q, "namespace": ns, "k": K, "mode": mode}
    if fmt:
        params["format"] = fmt
        if fmt == "text":
            params["template"] = "temporal_facts"
    url = f"{base}/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return {"ok": True, "body": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "body": e.read().decode()[:200]}


def ids_of(body: dict) -> list[str]:
    return [r["id"] for r in body.get("results", [])]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base, ns = sys.argv[1], sys.argv[2]

    p1_identical = p1_total = 0
    p2_differ = p2_total = 0
    rows = []

    for q in QUESTIONS:
        sh = fetch(base, ns, q, "hybrid", "structured")
        sr = fetch(base, ns, q, "recall", "structured")
        th = fetch(base, ns, q, "hybrid", "text")
        tr = fetch(base, ns, q, "recall", "text")

        if not all(x["ok"] for x in (sh, sr, th, tr)):
            bad = [n for n, x in zip("sh sr th tr".split(), (sh, sr, th, tr)) if not x["ok"]]
            rows.append(f"  ! {q[:45]:47} REQUEST FAILED: {bad} "
                        f"{[x for x in (sh,sr,th,tr) if not x['ok']][0]}")
            continue

        # P1 — does `format=text` ignore `mode`?
        same_text = th["body"] == tr["body"]
        p1_total += 1
        p1_identical += int(same_text)

        # P2 — does `format=structured` respect `mode`?
        ids_h, ids_r = ids_of(sh["body"]), ids_of(sr["body"])
        diff_struct = ids_h != ids_r
        p2_total += 1
        p2_differ += int(diff_struct)

        overlap = len(set(ids_h) & set(ids_r))
        rows.append(
            f"  {q[:45]:47} | text h==r: {str(same_text):5} | "
            f"struct n_hybrid={len(ids_h):3} n_recall={len(ids_r):3} "
            f"overlap={overlap:3} differ={diff_struct}"
        )

    print(f"namespace={ns}  k={K}\n")
    print("\n".join(rows))

    print(f"\nP1  text&hybrid == text&recall (mode IGNORED for text): "
          f"{p1_identical}/{p1_total}")
    print(f"P2  structured&hybrid != structured&recall (mode RESPECTED): "
          f"{p2_differ}/{p2_total}")

    refuted = p1_total and p2_total and p1_identical == p1_total and p2_differ == p2_total
    print("\n" + ("=" * 72))
    if refuted:
        print("VERDICT: 'rendering-only' framing REFUTED.")
        print("  `format=text` ignores `mode`, while `format=structured` honours it, so")
        print("  switching the harness to text ALSO changes retrieval. Fix the parity")
        print("  defect BEFORE switching the harness (RECALL-LEDGER §4.19(f)).")
    else:
        print("VERDICT: not refuted on this sample — the two formats did NOT show the")
        print("  predicted asymmetry. Re-read the traces in §4.19(f) before proceeding.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

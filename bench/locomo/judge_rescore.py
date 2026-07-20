#!/usr/bin/env python3
"""Workstream A — offline cross-family LLM-judge rescoring for the LoCoMo harness.

The harness scores recall with a STRICT substring / word-overlap matcher
(`check_answer_in_memories`). It scores 0 for semantically-correct answers that
recall DID surface but phrased differently (gold "2022" vs memory "last year";
gold "Single" vs memory "not seeing anyone"). Two consequences:
  1. Not comparable to competitors — mem0 / Zep / LoCoMo-SOTA report using an
     LLM-as-judge; comparing our strict number to their judged number
     UNDERCOUNTS us.
  2. Can't tell a real recall gap from a scoring artifact — so we can't
     prioritize the recall levers (C/D/B/E) against MEASURED gaps.

This script adds a cross-family LLM-judge pass DECOUPLED from ingest/recall.
The harness now persists `recalled_memories` (the actual texts) per question,
so any results JSON is re-scorable offline with NO re-ingest and NO re-recall.

Subcommands
-----------
  prepare   results.json[...]  -> judge-batch JSONL of ANSWERABLE questions
            {key, question_id, category, question, gold, memories}. Adversarial
            questions are EXCLUDED (they keep the harness's abstention logic —
            there is nothing for a judge to support). Entries already present
            in --cache are skipped (idempotent; no re-spend on re-runs).

  tally     results.json[...] --verdicts v.jsonl  -> per-category and overall
            SUBSTRING vs JUDGE correctness + the delta. The delta IS the
            measurement-artifact size; it is surfaced, not hidden (A5).

  prompt    print the canonical strict judge prompt (single source of truth for
            the orchestrator's sub-agent dispatch — a load-bearing invariant).

The judge itself
----------------
The judge is run by the ORCHESTRATOR (Claude sub-agents — option J1: $0,
cross-family vs the gpt-oss extraction model, no key/HITL, per
orchestrator-as-llm-substitute + research.md independence). This script calls
NO LLM: it prepares the batch and tallies verdicts, so the judge model is
swappable (J1 Claude sub-agents now; J2 Gemini 2.5 Flash later for a cached
reproducible artifact — NOT an OpenAI-lineage model, which is weakly
independent from gpt-oss).

Verdict record shape (one JSON object per line in the --verdicts file):
  {"key": "<qid>::<sha16>", "question_id": "...",
   "supported": true|false, "reason": "..."}

Cache/verdict key = f"{question_id}::{sha256(gold + US + '\\n'.join(memories))[:16]}"
(US = unit separator). A verdict is reused only while BOTH the gold answer and
the exact recalled memories are unchanged; change recall and the question is
re-judged automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Categories whose "correct" answer is to ABSTAIN — recall SHOULD find nothing
# relevant. There is no gold fact to support, so the judge does not score them;
# tally reuses the harness's substring abstention verdict for these.
ABSTENTION_CATEGORIES = {"adversarial"}

_US = "\x1f"  # unit separator between gold and memories in the hash preimage

# The canonical strict, anti-inflation judge prompt (spec A2 + research.md
# feedback_independent_verification_catches_self_scoring_inflation). This is the
# EXACT instruction the orchestrator must give each judge sub-agent so verdicts
# are reproducible and not silently inflated. Keep it here as the single source
# of truth — do not paraphrase it per dispatch.
JUDGE_PROMPT = """\
You are a STRICT grader for a memory-recall benchmark. For each item you are \
given a QUESTION, the GOLD answer, and the list of MEMORIES that a retrieval \
system returned for that question. Decide whether the memories actually support \
the gold answer.

Answer supported=true ONLY IF the recalled memories actually contain, state, or \
DIRECTLY ENTAIL the gold answer. In particular:
  - A paraphrase or synonym of the gold answer counts as SUPPORTED.
  - A relative date/time that resolves unambiguously to the gold's absolute \
date counts as SUPPORTED (e.g. gold "2022" and a memory dated within 2022, or \
"last year" when the conversation year makes it 2022).
  - A memory that is merely on-topic but does NOT answer the question counts as \
NOT supported.
  - If the gold answer is a LIST, it is supported only if the memories cover the \
substantive items (a clear majority), not just one.

Do NOT use any outside knowledge — judge ONLY from the provided memories. When \
in doubt, answer false. Return supported=false rather than guessing.

Return a JSON array, one object per item, in the SAME ORDER as the input, each: \
{"question_id": "<id>", "supported": true|false, "reason": "<one short clause>"}.
Output ONLY the JSON array, nothing else.\
"""


def cache_key(sample_id: str, question_id: str, gold: str,
              memories: list[str]) -> str:
    """Stable key over (sample_id, question_id, gold, exact recalled memories).

    sample_id (the conversation, e.g. "conv-26") is REQUIRED: question_id is
    only LOCAL to a conversation ("q_0", "q_1", ...) and all conversations
    reuse the same q_N namespace, so a key WITHOUT sample_id collides across
    conversations and silently collapses 497 questions to 199.
    """
    preimage = (str(gold) + _US + "\n".join(memories)).encode("utf-8")
    return f"{sample_id}::{question_id}::{hashlib.sha256(preimage).hexdigest()[:16]}"


def load_results(paths: list[Path]) -> list[dict]:
    """Flatten `results` arrays across one or more harness output JSONs.

    Deduplicates by question_id so the same conversation scored twice (e.g. a
    recall pass then a hybrid pass) does not double-count — the LAST occurrence
    wins (later files / later passes override earlier ones).
    """
    by_key: dict[tuple[str, str], dict] = {}
    for p in paths:
        doc = json.loads(p.read_text())
        for r in doc.get("results", []):
            # (sample_id, question_id) — question_id alone is conv-local and
            # collides across conversations (see cache_key docstring).
            by_key[(r.get("sample_id", ""), r["question_id"])] = r
    return list(by_key.values())


def load_verdicts(path: Path | None) -> dict[str, dict]:
    """Load verdict JSONL keyed by `key`. Missing file -> empty (no verdicts)."""
    verdicts: dict[str, dict] = {}
    if path is None or not path.exists():
        return verdicts
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        verdicts[v["key"]] = v
    return verdicts


def cmd_prepare(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    cached = load_verdicts(args.cache)

    batch: list[dict] = []
    n_abstain = n_cached = n_no_gold = 0
    for r in results:
        category = r.get("category", "")
        if category in ABSTENTION_CATEGORIES:
            n_abstain += 1
            continue
        gold = r.get("expected_answer", "")
        memories = r.get("recalled_memories")
        if memories is None:
            # results JSON predates the harness `recalled_memories` capture —
            # cannot be judged offline. Surface loudly rather than silently drop.
            n_no_gold += 1
            continue
        sample_id = r.get("sample_id", "")
        key = cache_key(sample_id, r["question_id"], gold, memories)
        if key in cached:
            n_cached += 1
            continue
        batch.append(
            {
                "key": key,
                "sample_id": sample_id,
                "question_id": r["question_id"],
                "category": category,
                "question": r.get("question", ""),
                "gold": gold,
                "memories": memories,
            }
        )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in batch:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"prepare: {len(batch)} question(s) to judge -> {out}\n"
        f"  skipped: {n_cached} already-cached, {n_abstain} abstention "
        f"(not judged), {n_no_gold} missing recalled_memories "
        f"(re-run harness with the recalled_memories capture)",
        file=sys.stderr,
    )
    if n_no_gold:
        print(
            f"[WARN] {n_no_gold} result(s) lack recalled_memories and were "
            f"EXCLUDED from judging — the run predates the capture. Re-run "
            f"`harness.py --skip-ingest` to regenerate with texts.",
            file=sys.stderr,
        )
    return 0


def _fmt_row(label: str, sc: int, jc: int, tot: int) -> str:
    sp = sc / tot * 100 if tot else 0.0
    jp = jc / tot * 100 if tot else 0.0
    return (
        f"{label:<14} {sc:>4}/{tot:<4} {sp:>6.1f}%   "
        f"{jc:>4}/{tot:<4} {jp:>6.1f}%   {jp - sp:>+6.1f}"
    )


def cmd_tally(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    verdicts = load_verdicts(args.verdicts)

    # per-category counters: [substring_correct, judge_correct, total]
    cats: dict[str, list[int]] = {}
    unjudged: list[str] = []
    for r in results:
        category = r.get("category", "")
        row = cats.setdefault(category, [0, 0, 0])
        row[2] += 1
        sub = bool(r.get("is_correct", False))
        if sub:
            row[0] += 1

        if category in ABSTENTION_CATEGORIES:
            # not judged — reuse the harness's abstention verdict verbatim
            if sub:
                row[1] += 1
            continue

        key = cache_key(r.get("sample_id", ""), r["question_id"],
                        r.get("expected_answer", ""),
                        r.get("recalled_memories") or [])
        v = verdicts.get(key)
        if v is None:
            unjudged.append(f"{r.get('sample_id','')}/{r['question_id']}")
            # unjudged counts as NOT judge-correct (fail-loud: a missing verdict
            # must not silently inflate — it depresses the judge number until
            # judged, and is reported below).
            continue
        if v.get("supported"):
            row[1] += 1

    # ---- report -----------------------------------------------------------
    print("=" * 72)
    print("LoCoMo recall — SUBSTRING vs LLM-JUDGE (delta = scoring-artifact size)")
    print("=" * 72)
    print(f"{'category':<14} {'substring':>13}   {'llm-judge':>13}   {'Δpts':>6}")
    print("-" * 72)
    tot_s = tot_j = tot_n = 0
    for cat in sorted(cats):
        sc, jc, tot = cats[cat]
        tot_s += sc
        tot_j += jc
        tot_n += tot
        print(_fmt_row(cat, sc, jc, tot))
    print("-" * 72)
    print(_fmt_row("OVERALL", tot_s, tot_j, tot_n))
    print("=" * 72)

    if unjudged:
        print(
            f"\n[WARN] {len(unjudged)} answerable question(s) have NO verdict "
            f"and are counted as judge-incorrect. Run `prepare` -> judge -> add "
            f"to --verdicts, then re-tally. First few: {unjudged[:5]}",
            file=sys.stderr,
        )
    return 0


def cmd_prompt(_args: argparse.Namespace) -> int:
    print(JUDGE_PROMPT)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="emit a judge-batch JSONL from results")
    pp.add_argument("results", type=Path, nargs="+",
                    help="one or more harness results JSON files")
    pp.add_argument("-o", "--output", type=Path, required=True,
                    help="judge-batch JSONL to write")
    pp.add_argument("--cache", type=Path, default=None,
                    help="existing verdicts JSONL to skip already-judged items")
    pp.set_defaults(func=cmd_prepare)

    pt = sub.add_parser("tally", help="print substring-vs-judge scores")
    pt.add_argument("results", type=Path, nargs="+",
                    help="one or more harness results JSON files")
    pt.add_argument("--verdicts", type=Path, required=True,
                    help="verdicts JSONL produced by the judge")
    pt.set_defaults(func=cmd_tally)

    pr = sub.add_parser("prompt", help="print the canonical strict judge prompt")
    pr.set_defaults(func=cmd_prompt)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

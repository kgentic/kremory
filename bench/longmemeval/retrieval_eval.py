#!/usr/bin/env python3
"""LongMemEval RETRIEVAL evaluation — rank-aware, $0, no API key.

WHY THIS EXISTS (and why it runs BEFORE anything paid)
------------------------------------------------------
LongMemEval's headline metric is generative QA: an LLM answers from retrieved
context and a judge scores it. Both calls are paid. But paying to measure an
answerer sitting on top of retrieval you have never verified is backwards — if
recall never surfaces the evidence session, the answerer cannot succeed and the
spend measures nothing.

LongMemEval ships the qrel needed to check that for free: `answer_session_ids`
names the session(s) containing the evidence. `ingest_sessions` writes
`[Session {sid}]` as the first line of every stored memory, so the retrieved
text joins back to a session id by construction. That makes retrieval quality
measurable with zero LLM calls.

METRIC CLASS (CLAUDE.md Rule 36 — match the metric to the mechanism)
--------------------------------------------------------------------
Reported metrics are RANK-AWARE: recall@k, MRR, nDCG@k, and mean gold rank.
This is deliberate and is the lesson from the LoCoMo campaign: that harness
gated every retrieval lever on a substring scorer which is invariant under
permutation (an empirical probe found 0 of 1986 verdicts changed by shuffling
the result list), so every reordering lever — reranking, fusion weights,
score calibration — would have A/B'd at exactly 0.000 no matter how well it
worked. Set metrics (recall@k) can only see membership changes; MRR/nDCG can
see ordering. Both are reported so a lever of either class is measurable.

Abstention questions (`_abs`) are EXCLUDED from the scored denominator: they
have no gold session by construction, and scoring them here would repeat the
category-5 mistake of crediting an absence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from kremory_client import CodememClient, KremoryStalled  # noqa: E402

import harness  # noqa: E402  (ingest_sessions, parse_haystack_date)

SESSION_RE = re.compile(r"\[Session\s+([^\]]+)\]")


def gold_rank(retrieved: list[str], gold_ids: set[str]) -> int | None:
    """1-based rank of the first retrieved memory from a gold session."""
    for i, mem in enumerate(retrieved, 1):
        m = SESSION_RE.search(str(mem))
        if m and m.group(1).strip() in gold_ids:
            return i
    return None


def ndcg_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    """Graded-free nDCG@k with binary relevance over session ids.

    Ideal DCG assumes every gold session could occupy the top slots, so a
    question with 2 gold sessions is not penalised for only being able to place
    2 relevant items.
    """
    dcg = 0.0
    seen: set[str] = set()
    for i, mem in enumerate(retrieved[:k], 1):
        m = SESSION_RE.search(str(mem))
        sid = m.group(1).strip() if m else None
        if sid in gold_ids and sid not in seen:
            seen.add(sid)
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).parent / "data" / "longmemeval_s.json")
    ap.add_argument("--base-url", default="http://localhost:3179")
    ap.add_argument("--max-questions", type=int, default=5)
    ap.add_argument("--k", type=int, default=10, help="recall depth requested from the server")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse a corpus already ingested by a prior run")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "results" / "retrieval-eval.json")
    ap.add_argument("--smallest-first", action="store_true",
                    help="order questions by corpus size so a short run covers more of them")
    args = ap.parse_args()

    data = json.loads(args.dataset.read_text())
    # Abstention questions carry no gold session — excluded, never scored.
    scorable = [x for x in data if x.get("answer_session_ids")
                and not x["question_id"].endswith("_abs")]
    n_abs = len(data) - len(scorable)
    if args.smallest_first:
        scorable.sort(key=lambda x: sum(
            len(t.get("content", "") or "") for s in x["haystack_sessions"] for t in s))
    batch = scorable[: args.max_questions] if args.max_questions else scorable

    client = CodememClient(args.base_url)
    if not client.health():
        print(f"[FAIL] kremory-http not reachable at {args.base_url}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.output.with_suffix(".jsonl")
    jf = open(jsonl, "a")

    rows: list[dict] = []
    t_start = time.time()
    for n, item in enumerate(batch, 1):
        qid = item["question_id"]
        ns = f"lme-ret-{qid}"
        gold = {s.strip() for s in item["answer_session_ids"]}

        t0 = time.time()
        stored = 0
        if not args.skip_ingest:
            client.delete_namespace(ns)
            try:
                stored = harness.ingest_sessions(client, ns, item)
            except KremoryStalled as e:
                print(f"\n[FAIL-LOUD] ingest stalled on {qid}: {e}", file=sys.stderr)
                jf.close()
                return 5
            if stored == 0:
                print(f"  [warn] {qid}: stored 0 sessions", file=sys.stderr)
        ingest_s = time.time() - t0

        mems = client.recall(item["question"], ns, args.k)
        texts = [m.get("content", "") if isinstance(m, dict) else str(m) for m in mems]
        rank = gold_rank(texts, gold)

        # kremory returns a MIX of `kind=episode` and `kind=entity`. Only
        # episodes carry the `[Session sid]` marker, and `source_episode_id`
        # comes back NULL on every row (the ADR-078 Phase A provenance field is
        # present in the wire shape but unpopulated on this path), so an entity
        # hit cannot be attributed to a session at all.
        #
        # They are still counted in the RANK, deliberately: an entity occupying
        # a top-k slot is a real cost to the answerer, which receives that slot
        # instead of an evidence session. Tracking the share separately makes
        # the trade visible rather than folding it into a single number.
        kinds: dict[str, int] = {}
        for r_ in (client.last_results or []):
            kinds[r_.get("kind") or "unknown"] = kinds.get(r_.get("kind") or "unknown", 0) + 1

        row = {
            "question_id": qid,
            "question_type": item.get("question_type", ""),
            "n_sessions": len(item.get("haystack_sessions", [])),
            "stored": stored,
            "retrieved": len(texts),
            "gold_sessions": sorted(gold),
            "gold_rank": rank,
            "hit_at_1": rank == 1,
            "hit_at_5": rank is not None and rank <= 5,
            "hit_at_10": rank is not None and rank <= 10,
            "rr": (1.0 / rank) if rank else 0.0,
            "ndcg_at_10": ndcg_at_k(texts, gold, 10),
            "result_kinds": kinds,
            "ingest_s": round(ingest_s, 1),
        }
        rows.append(row)
        jf.write(json.dumps(row) + "\n")
        jf.flush()
        print(f"[{n}/{len(batch)}] {qid:<12} {row['question_type']:<26} "
              f"stored={stored:<3} rank={rank}  ndcg={row['ndcg_at_10']:.3f}  "
              f"({ingest_s:.0f}s)", flush=True)

    jf.close()
    if not rows:
        print("[FAIL] no questions scored", file=sys.stderr)
        return 1

    # FAIL-LOUD: retrieval returning NOTHING is unmeasurable, not zero.
    # Without this, `--skip-ingest` against a namespace that was never
    # populated prints a pristine 0.000 across every metric and reads exactly
    # like a measured result. That is the same absence-as-measurement defect
    # that produced the LoCoMo category-5 false 100%, inverted — and it fired
    # on this script's very first smoke run, which is why the guard exists.
    n_empty = sum(1 for r in rows if r["retrieved"] == 0)
    if n_empty == len(rows):
        print(f"\n[FAIL-LOUD] every one of {len(rows)} questions retrieved ZERO memories.\n"
              f"  This is a retrieval/ingest failure, not a score of 0.000.\n"
              f"  Most likely: --skip-ingest against a corpus that was never ingested\n"
              f"  (a plain run deletes each namespace unless --keep-corpus was set).\n"
              f"  Refusing to emit a summary.", file=sys.stderr)
        return 4
    if n_empty:
        print(f"\n[warn] {n_empty}/{len(rows)} questions retrieved zero memories — "
              f"those are retrieval failures, not misses.", file=sys.stderr)

    n = len(rows)
    # An ingest that stored nothing makes retrieval unmeasurable, not zero —
    # count it separately rather than letting it depress the mean silently.
    broken = [r for r in rows if not args.skip_ingest and r["stored"] == 0]
    summary = {
        "metric": "longmemeval-retrieval",
        "scorer": "session-qrel (answer_session_ids), rank-aware — NO LLM, $0",
        "k": args.k,
        "n_scored": n,
        "n_abstention_excluded": n_abs,
        "n_ingest_empty": len(broken),
        "recall_at_1": round(sum(r["hit_at_1"] for r in rows) / n, 4),
        "recall_at_5": round(sum(r["hit_at_5"] for r in rows) / n, 4),
        "recall_at_10": round(sum(r["hit_at_10"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        "ndcg_at_10": round(sum(r["ndcg_at_10"] for r in rows) / n, 4),
        "mean_gold_rank_when_found": round(
            sum(r["gold_rank"] for r in rows if r["gold_rank"]) /
            max(1, sum(1 for r in rows if r["gold_rank"])), 2),
        "mean_ingest_s": round(sum(r["ingest_s"] for r in rows) / n, 1),
        "wall_clock_s": round(time.time() - t_start, 1),
        "result_kind_share": {},
        "per_type": {},
        "rows": rows,
    }
    _tot_kinds: dict[str, int] = {}
    for r in rows:
        for kk, vv in (r.get("result_kinds") or {}).items():
            _tot_kinds[kk] = _tot_kinds.get(kk, 0) + vv
    _kind_total = sum(_tot_kinds.values()) or 1
    summary["result_kind_share"] = {
        kk: round(vv / _kind_total, 4) for kk, vv in sorted(_tot_kinds.items())
    }
    for r in rows:
        t = r["question_type"]
        b = summary["per_type"].setdefault(t, {"n": 0, "hit_at_10": 0, "ndcg": 0.0})
        b["n"] += 1
        b["hit_at_10"] += int(r["hit_at_10"])
        b["ndcg"] += r["ndcg_at_10"]
    for t, b in summary["per_type"].items():
        b["recall_at_10"] = round(b.pop("hit_at_10") / b["n"], 4)
        b["ndcg_at_10"] = round(b.pop("ndcg") / b["n"], 4)

    args.output.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*62}\nLongMemEval RETRIEVAL (no LLM, $0)\n{'='*62}")
    print(f"  scored              {n}  ({n_abs} abstention questions excluded)")
    if broken:
        print(f"  ⚠ ingest stored 0  {len(broken)} — retrieval unmeasurable for these")
    print(f"  recall@1 / @5 / @10 {summary['recall_at_1']:.3f} / "
          f"{summary['recall_at_5']:.3f} / {summary['recall_at_10']:.3f}")
    print(f"  MRR                 {summary['mrr']:.3f}")
    print(f"  nDCG@10             {summary['ndcg_at_10']:.3f}")
    print(f"  mean gold rank      {summary['mean_gold_rank_when_found']}")
    print(f"  mean ingest         {summary['mean_ingest_s']}s/question")
    print(f"  top-k composition   {summary['result_kind_share']}")
    print(f"    (only kind=episode can match a session qrel; entity slots are")
    print(f"     counted in the rank because they cost the answerer a slot)")
    print(f"\n  {'type':<28} {'n':>4} {'recall@10':>10} {'nDCG@10':>9}")
    for t in sorted(summary["per_type"]):
        b = summary["per_type"][t]
        print(f"  {t:<28} {b['n']:>4} {b['recall_at_10']:>10.3f} {b['ndcg_at_10']:>9.3f}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

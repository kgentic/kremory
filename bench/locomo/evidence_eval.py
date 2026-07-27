#!/usr/bin/env python3
"""Rank-aware retrieval evaluation against LoCoMo's OWN ground-truth evidence.

WHY THIS EXISTS
---------------
The default harness scorer (`check_answer_in_memories`) asks "does the gold ANSWER
STRING appear anywhere in the top-k retrieved text". That is a SET function: it is
provably invariant under reordering (measured 2026-07-27 — shuffling the retrieved
list for all 1986 questions flipped 0 verdicts). It therefore cannot measure any
lever whose output is a permutation — cross-encoder reranker, fusion weights that
only change intra-k order, RRF tie-breaks. See CLAUDE.md Rule 36 /
`~/.claude/rules/verify-metric-sensitivity-before-gating-decisions.md`.

It is also blind in a second way: ~72% of remaining failures are questions whose
gold answer is DERIVED (inference, multi-episode aggregation, date arithmetic) and
never appears verbatim, e.g. "What would Caroline's political leaning likely be?"
-> "Liberal", evidence=['D12:1']. Answer-string matching cannot tell whether
retrieval succeeded on those; evidence matching can.

LoCoMo ships per-turn ids (`dia_id`) and per-question `evidence` lists. That is a
qrels set. This module scores retrieval the way the IR literature requires for
ranking work — rank-aware metrics against labelled relevance (BEIR standardises
nDCG@10; Nogueira et al. arXiv:1910.14424 on first-stage recall bounding rerankers).

TD-139 MEASUREMENT PREREQUISITE (`.ai-docs/tech-debt/tech-debt-register.md`
"TD-139", the "MEASUREMENT PREREQUISITE" block)
-------------------------------------------------------------------------------
A retrieved FACT string (e.g. "Caroline attended LGBTQ_support_group") does not
contain LoCoMo turn text verbatim, so scoring it the same way as an entity/
episode item would read it as irrelevant even when perfectly on-point — the same
class of blindness TD-134 hit. `facts.source_episode_id` (`core/schema.rs:129`)
lets a fact be resolved to the EPISODE it was asserted from; this module then
scores the fact against THAT episode's evidence turns instead of the fact's own
(short, non-verbatim) text.

A row MAY carry an OPTIONAL, index-aligned `recalled_memory_provenance` list
(same length/order as `recalled_memories`) of `{"kind": "entity"|"episode"|
"fact", "source_episode_id": int|None}` dicts — the REST wire shape a future
harness capture would persist (`SearchResultWire.kind` /
`.source_episode_id`, `crates/kremory-mcp/src/bin/kremory-http.rs`). No live
arm emits `kind == "fact"` yet (TD-139 DoD item 2, deliberately unwired), so
no existing run JSON carries this field — `score_row` falls back to its
pre-TD-139 behaviour (score the item's own text) whenever provenance is
absent, kind isn't `"fact"`, or `--validate-db` wasn't supplied. This keeps
every already-recorded run's numbers byte-identical; `--self-test` exercises
the resolution path with synthetic fixtures instead (no DB required).

METRICS (all @k, over the retrieved list as ordered by the server)
  recall@k   fraction of a question's evidence turns present in the top-k
  nDCG@k     graded gain = number of distinct evidence turns a retrieved item
             carries (episodes are turn-chunks, so one item can cover several),
             discounted by log2(rank+1), normalised by the ideal ordering
  MRR        1 / rank of the first item carrying any evidence turn
  first_rank mean rank of the first evidence-bearing item (retrieval "depth to hit")

Adversarial questions are excluded: they are abstention checks, not retrieval tasks.

USAGE
  python3 bench/locomo/evidence_eval.py <run.json> [run2.json ...] \
      [--dataset bench/locomo/data/locomo10.json] [--k 10] [--validate-db PATH]
      [--self-test]

`--validate-db` checks what fraction of evidence turns are locatable in the ingested
corpus at all — the instrument's own validation step. If that coverage is low, every
number below is bounded by an ingestion gap, not a retrieval one, and must not be
read as a retrieval result. It is ALSO (TD-139) the source of per-episode content
used to resolve a `kind == "fact"` retrieved item to its source episode's evidence
turns — omitting it makes fact resolution a no-op (falls back to pre-TD-139 scoring),
it does NOT error.

`--self-test` shuffles each retrieved list and re-scores. A rank-aware metric MUST
move. If it does not, this instrument is as blind as the one it replaces. It also
runs a synthetic (no-DB) TD-139 fact-resolution check: a fact resolved to a source
episode that DOES contain an evidence turn must be credited; one resolved to an
episode that does NOT must not be.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# Turn texts shorter than this are matched ONLY in "Speaker: text" form — a bare
# "Yes." or "Haha" would otherwise match almost any episode and fabricate hits.
MIN_BARE_MATCH_LEN = 25


def norm(s: str) -> str:
    """Lowercase + collapse whitespace. Episodes join turns with newlines and the
    dataset stores them unwrapped, so whitespace is the only real divergence."""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def load_turns(dataset_path: Path) -> dict[str, dict[str, str]]:
    """sample_id -> {dia_id: matchable text}. Prefers "Speaker: text" (how episodes
    render a turn); keeps the bare text as a fallback for long turns."""
    data = json.loads(Path(dataset_path).read_text())
    out: dict[str, dict[str, str]] = {}
    for conv in data:
        turns: dict[str, str] = {}
        c = conv["conversation"]
        for key, val in c.items():
            if not (key.startswith("session_") and isinstance(val, list)):
                continue
            for t in val:
                did, speaker, text = t.get("dia_id"), t.get("speaker", ""), t.get("text", "")
                if did:
                    turns[did] = norm(f"{speaker}: {text}")
        out[conv["sample_id"]] = turns
    return out


def bare(text: str) -> str:
    """The turn text with the 'speaker: ' prefix stripped."""
    return text.split(": ", 1)[1] if ": " in text else text


def turn_in(turn: str, hay: str) -> bool:
    if turn in hay:
        return True
    b = bare(turn)
    return len(b) >= MIN_BARE_MATCH_LEN and b in hay


def evidence_ids(row: dict) -> list[str]:
    raw = row.get("evidence_ids")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return [str(x) for x in (raw or [])]


def resolve_haystack(mem: str, provenance: dict | None,
                      episode_content: dict[int, str] | None) -> str:
    """TD-139: the text a single retrieved item is scored against.

    A `kind == "fact"` item (the REST wire's `SearchResultWire.kind` /
    `.source_episode_id`, `crates/kremory-mcp/src/bin/kremory-http.rs`)
    resolves to its SOURCE EPISODE's content via `source_episode_id`
    (`facts.source_episode_id`, `core/schema.rs:129`) — a fact's own short
    triple string ("Caroline attended LGBTQ_support_group") does not contain
    LoCoMo turn text verbatim, so scoring it directly reads a perfectly
    on-point fact as irrelevant.

    Every other case falls back to the item's OWN retrieved text, UNCHANGED
    — this is the pre-TD-139 behaviour and is what makes every existing
    (provenance-free) run file score byte-identically: `episode_content` is
    `None` (no `--validate-db`), `provenance` is `None`/absent (no
    `recalled_memory_provenance` field), `provenance["kind"] != "fact"`
    (entity/episode items), or `source_episode_id` is `None`.

    A `source_episode_id` absent from `episode_content` (deleted row / DB
    mismatch) resolves to `""` — `turn_in("", t)` never matches, so the item
    is correctly NOT credited rather than silently falling back to its own
    (non-verbatim) text, which would re-introduce the exact blindness this
    resolution exists to fix.
    """
    if episode_content is None or not provenance:
        return mem
    if provenance.get("kind") != "fact":
        return mem
    source_episode_id = provenance.get("source_episode_id")
    if source_episode_id is None:
        return mem
    return episode_content.get(int(source_episode_id), "")


def score_row(row: dict, turns: dict[str, str], k: int,
              episode_content: dict[int, str] | None = None) -> dict | None:
    """Per-question rank-aware scores, or None if the row carries no usable qrels.

    `episode_content` (TD-139, optional): `{episode_id: normalised content}`,
    used ONLY to resolve `kind == "fact"` retrieved items — see
    [`resolve_haystack`]. `None` (the default) reproduces pre-TD-139 scoring
    exactly, regardless of what `row` carries.
    """
    ev = [turns[e] for e in evidence_ids(row) if e in turns]
    if not ev:
        return None
    mems = [norm(m) for m in (row.get("recalled_memories") or [])][:k]
    if not mems:
        return {"recall": 0.0, "ndcg": 0.0, "rr": 0.0, "first_rank": None, "n_ev": len(ev)}

    # TD-139: OPTIONAL index-aligned provenance, same length/order as
    # `recalled_memories` — see the module docstring. Absent for every
    # existing run file, so `haystacks == mems` and scoring is unchanged.
    provenance_list = (row.get("recalled_memory_provenance") or [])[:k]
    haystacks = [
        resolve_haystack(
            m,
            provenance_list[i] if i < len(provenance_list) else None,
            episode_content,
        )
        for i, m in enumerate(mems)
    ]

    # gain[i] = how many DISTINCT evidence turns item i carries (episodes are
    # turn-chunks, so one retrieved item can legitimately cover several).
    gains = [sum(1 for t in ev if turn_in(t, h)) for h in haystacks]
    found = {t for t in ev if any(turn_in(t, h) for h in haystacks)}

    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g)
    # Ideal: all evidence turns packed into the earliest positions, respecting that
    # a single item can carry several (use the observed gain multiset, best-first).
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal) if g)

    first = next((i + 1 for i, g in enumerate(gains) if g), None)
    return {
        "recall": len(found) / len(ev),
        "ndcg": (dcg / idcg) if idcg else 0.0,
        "rr": (1.0 / first) if first else 0.0,
        "first_rank": first,
        "n_ev": len(ev),
    }


def evaluate(rows: list[dict], turns_by_sample: dict[str, dict[str, str]], k: int,
             shuffle: bool = False, seed: int = 1234,
             episode_content: dict[int, str] | None = None) -> tuple[dict, dict]:
    rng = random.Random(seed)
    overall: dict[str, list] = defaultdict(list)
    per_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cat = row.get("category", "?")
        if cat == "adversarial":
            continue
        turns = turns_by_sample.get(row.get("sample_id"), {})
        if shuffle:
            row = dict(row)
            mems = list(row.get("recalled_memories") or [])
            # Fisher-Yates on an index permutation, not `mems` directly, so a
            # TD-139 `recalled_memory_provenance` list (if present) can be
            # reindexed by the SAME permutation and stay index-aligned with
            # the shuffled `recalled_memories` — shuffling one without the
            # other would silently misattribute a fact's source episode to
            # whatever item lands at its old index. `rng.shuffle` draws are a
            # pure function of list LENGTH, not contents, so this produces
            # the identical permutation (and RNG stream) `rng.shuffle(mems)`
            # did before — byte-identical self-test numbers on every
            # existing (provenance-free) run.
            order = list(range(len(mems)))
            rng.shuffle(order)
            row["recalled_memories"] = [mems[i] for i in order]
            prov = row.get("recalled_memory_provenance")
            if prov is not None:
                row["recalled_memory_provenance"] = [
                    prov[i] if i < len(prov) else None for i in order
                ]
        s = score_row(row, turns, k, episode_content=episode_content)
        if s is None:
            continue
        for m in ("recall", "ndcg", "rr"):
            overall[m].append(s[m])
            per_cat[cat][m].append(s[m])
        if s["first_rank"]:
            overall["first_rank"].append(s["first_rank"])
            per_cat[cat]["first_rank"].append(s["first_rank"])
        overall["hit"].append(1.0 if s["first_rank"] else 0.0)
        per_cat[cat]["hit"].append(1.0 if s["first_rank"] else 0.0)
    return overall, per_cat


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def report(label: str, overall: dict, per_cat: dict, k: int) -> None:
    n = len(overall["recall"])
    print(f"\n=== {label}  (k={k}, n={n} scorable ex-adversarial) ===")
    print(f"  recall@{k}   {mean(overall['recall']) * 100:.1f}%   "
          f"(fraction of evidence turns retrieved)")
    print(f"  nDCG@{k}     {mean(overall['ndcg']) * 100:.1f}%   "
          f"(rank-aware — THIS is what a reordering lever moves)")
    print(f"  MRR         {mean(overall['rr']) * 100:.1f}%")
    print(f"  hit-rate    {mean(overall['hit']) * 100:.1f}%   "
          f"(>=1 evidence turn anywhere in top-{k})")
    print(f"  mean first-hit rank {mean(overall['first_rank']):.2f} "
          f"(over the {len(overall['first_rank'])} questions with a hit)")
    print(f"  {'category':<13} {'recall':>8} {'nDCG':>8} {'MRR':>8} {'hit':>8}")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        print(f"  {cat:<13} {mean(c['recall']) * 100:>7.1f}% {mean(c['ndcg']) * 100:>7.1f}% "
              f"{mean(c['rr']) * 100:>7.1f}% {mean(c['hit']) * 100:>7.1f}%")


def validate_corpus(db: str, turns_by_sample: dict[str, dict[str, str]],
                    rows: list[dict]) -> None:
    """Instrument validation: are the evidence turns even PRESENT in what we ingested?

    NB project a NON-indexed column — a PK-only/COUNT projection on this
    vector-indexed table returns 0 rows under foreign sqlite3 (libsql vector-index trap).
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    corpus: dict[str, str] = defaultdict(str)
    for group_id, content in con.execute("SELECT group_id, content FROM episodes"):
        corpus[group_id or ""] += "\n" + norm(content)
    con.close()

    wanted: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("category") == "adversarial":
            continue
        sid = row.get("sample_id")
        for e in evidence_ids(row):
            t = turns_by_sample.get(sid, {}).get(e)
            if t:
                wanted[sid].add(t)

    tot = found = 0
    worst: list[tuple[str, float]] = []
    for sid, ts in wanted.items():
        hay = corpus.get(f"locomo-bench-{sid}", "")
        f = sum(1 for t in ts if turn_in(t, hay))
        tot += len(ts)
        found += f
        worst.append((sid, f / len(ts) if ts else 1.0))
    print("=== INSTRUMENT VALIDATION — evidence-turn coverage in the ingested corpus ===")
    print(f"  {found}/{tot} distinct evidence turns locatable ({found / max(tot,1) * 100:.1f}%)")
    for sid, frac in sorted(worst, key=lambda x: x[1])[:5]:
        print(f"    worst: {sid} {frac * 100:.1f}%")
    if found / max(tot, 1) < 0.9:
        print("  WARNING: coverage <90% — retrieval numbers below are bounded by an "
              "INGESTION gap, not a retrieval one. Do not read them as retrieval results.")


def load_episode_content(db: str) -> dict[int, str]:
    """TD-139: `{episode id: normalised content}` for [`resolve_haystack`]'s
    fact -> source-episode resolution (`facts.source_episode_id`,
    `core/schema.rs:129`).

    Same libsql vector-index COUNT(*) trap as `validate_corpus` above — `id`/
    `content` are non-indexed columns on `episodes`, so a plain projection
    (not a COUNT) is safe under foreign sqlite3.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {int(eid): norm(content) for eid, content in con.execute("SELECT id, content FROM episodes")}
    con.close()
    return out


def self_test_fact_resolution() -> bool:
    """TD-139: a `kind == "fact"` retrieved item must be scored against its
    SOURCE EPISODE's evidence turns, not its own (short, non-verbatim) triple
    text. Synthetic — no DB needed, so this runs unconditionally under
    `--self-test` regardless of whether `--validate-db` was supplied.

    Three assertions:
      1. BASELINE — with no `episode_content` (the pre-TD-139 / no
         `--validate-db` case), the fact's own snippet does NOT contain the
         evidence turn verbatim: a MISS. Proves the fixture is honest (the
         fact string really doesn't already match) before claiming the fix
         turns it into a hit.
      2. POSITIVE — resolved against its ACTUAL source episode (which does
         contain the evidence turn): must be credited (recall == 1.0).
      3. NEGATIVE — resolved against a DIFFERENT episode that does NOT
         contain the evidence turn: must NOT be credited (recall == 0.0) —
         resolution must not fabricate hits for facts anchored elsewhere.
    """
    turns = {"D1:1": norm("Caroline: I attend a LGBTQ_support_group every week")}
    episode_content = {
        101: norm("Caroline: I attend a LGBTQ_support_group every week. It helps a lot."),
        102: norm("Melanie: I like hiking on weekends."),
    }
    base_row = {
        "sample_id": "s1",
        "category": "single-hop",
        "evidence_ids": ["D1:1"],
        "recalled_memories": ["Caroline attended LGBTQ_support_group"],
    }
    hit_row = {**base_row, "recalled_memory_provenance": [
        {"kind": "fact", "source_episode_id": 101}]}
    miss_row = {**base_row, "recalled_memory_provenance": [
        {"kind": "fact", "source_episode_id": 102}]}

    baseline = score_row(hit_row, turns, k=10, episode_content=None)
    hit = score_row(hit_row, turns, k=10, episode_content=episode_content)
    miss = score_row(miss_row, turns, k=10, episode_content=episode_content)

    ok = True
    if baseline is None or baseline["recall"] != 0.0:
        print(f"  [self-test:fact] FAIL — fixture's fact snippet unexpectedly matched "
              f"its own text with no resolution applied (baseline must be a MISS): {baseline}")
        ok = False
    if hit is None or hit["recall"] != 1.0:
        print(f"  [self-test:fact] FAIL — a fact resolved to its ACTUAL source episode "
              f"must be credited (recall==1.0): {hit}")
        ok = False
    if miss is None or miss["recall"] != 0.0:
        print(f"  [self-test:fact] FAIL — a fact resolved to an episode with NO evidence "
              f"turns must NOT be credited (recall==0.0): {miss}")
        ok = False
    print("  [self-test:fact] " + (
        "PASS — kind==\"fact\" items resolve to their source episode's evidence turns "
        "(a non-verbatim-but-on-point fact is credited; a wrongly-anchored one is not)."
        if ok else "see FAIL(s) above — TD-139 fact resolution is broken."))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data" / "locomo10.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--validate-db", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="shuffle retrieved lists and re-score; a rank-aware metric MUST move")
    args = ap.parse_args()

    turns = load_turns(args.dataset)

    if args.self_test:
        self_test_fact_resolution()

    # TD-139: `--validate-db` doubles as the source of per-episode content
    # for fact -> source-episode resolution (see `resolve_haystack`). `None`
    # when omitted, which makes resolution a no-op — every run scores
    # exactly as it did pre-TD-139.
    episode_content = load_episode_content(args.validate_db) if args.validate_db else None

    for i, run in enumerate(args.runs):
        rows = json.loads(run.read_text())["results"]
        if args.validate_db and i == 0:
            validate_corpus(args.validate_db, turns, rows)
        overall, per_cat = evaluate(rows, turns, args.k, episode_content=episode_content)
        report(run.name, overall, per_cat, args.k)

        if args.self_test:
            sh_overall, _ = evaluate(rows, turns, args.k, shuffle=True,
                                      episode_content=episode_content)
            d_ndcg = (mean(sh_overall["ndcg"]) - mean(overall["ndcg"])) * 100
            d_recall = (mean(sh_overall["recall"]) - mean(overall["recall"])) * 100
            print(f"  [self-test] shuffled: nDCG {mean(sh_overall['ndcg']) * 100:.1f}% "
                  f"({d_ndcg:+.1f}), recall {mean(sh_overall['recall']) * 100:.1f}% ({d_recall:+.1f})")
            print("  [self-test] " + (
                "PASS — nDCG responds to reordering (recall correctly does not)."
                if abs(d_ndcg) > 0.05 else
                "FAIL — nDCG did not move under shuffle; this instrument is order-BLIND."))


if __name__ == "__main__":
    main()

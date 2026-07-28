#!/usr/bin/env python3
"""Pre-build ceiling probe for doc2query write-time enrichment (Candidate 1).

Spec: `.ai-docs/specs/breadth-gap-structural-architecture-2026-07-28.md` §Candidate 1
"What would disconfirm it (and the cheap pre-build probes)".

THE QUESTION: kremory's largest recall gap is BREADTH — evidence that never
enters the 50-candidate pool — and 77% of those misses are vocabulary/abstraction
mismatch (question asks "relationship status", stored text says "single parent").
doc2query proposes generating pseudo-questions at write time so the abstract
question matches an abstract index entry.

Before building it: would the generated questions ACTUALLY have matched the real
questions we miss? If the LLM's phrasing diverges from LoCoMo's abstraction
level, the whole approach is dead and no build is warranted.

Two targets, same methodology (per the spec's two probes):
  EPISODE target — generate questions from the gold TURN text.
  FACT target    — generate questions from the linked fact TRIPLE
                   (resolved via facts.source_episode_id, wired in TD-139).

Baseline for comparison: the real question's overlap with the RAW gold turn,
which is what we index today. The probe is only encouraging if generated
questions score materially ABOVE that baseline — otherwise enrichment adds
nothing the current index doesn't already have.

Free/near-free: ~50 Groq calls (<$0.02) + local Ollama embeddings ($0).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import statistics
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_ee():
    spec = importlib.util.spec_from_file_location("ee", HERE / "evidence_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STOP = set(
    "a an the is are was were be been being do does did what when where who whom which how why "
    "to of in on at for with and or from that this it his her their they he she i you we my your "
    "would will can could should have has had not no yes about as by if then than so".split()
)


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def overlap(question: str, candidate: str) -> float:
    """Fraction of the QUESTION's content words present in the candidate text."""
    q = content_words(question)
    if not q:
        return 0.0
    return len(q & content_words(candidate)) / len(q)


def groq_chat(prompt: str, model: str, key: str, base: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def ollama_embed(text: str) -> list[float] | None:
    body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("embedding")
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def main() -> int:
    ee = load_ee()
    turns_by_sample = ee.load_turns(HERE / "data" / "locomo10.json")

    results_path = HERE / "results" / "conv0-dense-rerank50.json"
    rows = json.load(open(results_path))["results"]

    # ── Collect the misses: evidence turns absent from the 50-pool ──────────
    misses = []
    for r in rows:
        if r.get("category") == "adversarial":
            continue
        turns = turns_by_sample.get(r.get("sample_id"), {})
        ev = [(e, turns[e]) for e in ee.evidence_ids(r) if e in turns]
        if not ev:
            continue
        mems = [ee.norm(m) for m in (r.get("recalled_memories") or [])][:50]
        for _eid, txt in ev:
            if not any(ee.turn_in(txt, m) for m in mems):
                misses.append({"question": r["question"], "gold_turn": txt})

    print(f"misses to probe: {len(misses)}")
    if not misses:
        print("no misses found — nothing to probe")
        return 1

    # ── Resolve each gold turn to its episode, then to its linked facts ─────
    con = sqlite3.connect(f"file:{REPO / 'bench-dense.db'}?mode=ro", uri=True)
    episodes = [(r[0], ee.norm(r[1] or "")) for r in con.execute("SELECT id, content FROM episodes")]
    facts_by_ep: dict[int, list[str]] = {}
    for eid, s, p, o, ov in con.execute(
        "SELECT source_episode_id, subject_id, predicate, object_id, object_value FROM facts "
        "WHERE source_episode_id IS NOT NULL"
    ):
        obj = o or ov or ""
        facts_by_ep.setdefault(eid, []).append(f"{s} {p} {obj}".strip())

    for m in misses:
        gold_eps = [eid for eid, content in episodes if ee.turn_in(m["gold_turn"], content)]
        m["facts"] = [f for eid in gold_eps for f in facts_by_ep.get(eid, [])][:6]

    with_facts = sum(1 for m in misses if m["facts"])
    print(f"misses whose gold episode has >=1 linked fact: {with_facts}/{len(misses)}")

    key = os.environ.get("KREMORY_MCP_CHAT_API_KEY")
    base = os.environ.get("KREMORY_MCP_CHAT_BASE_URL")
    model = os.environ.get("PROBE_MODEL", "openai/gpt-oss-120b")
    if not key or not base:
        print("MISSING KREMORY_MCP_CHAT_API_KEY / _BASE_URL — cannot generate", file=sys.stderr)
        return 2

    ep_scores, fact_scores, base_scores = [], [], []
    ep_cos, fact_cos, base_cos = [], [], []
    samples = []

    for i, m in enumerate(misses):
        q_real = m["question"]
        base_ov = overlap(q_real, m["gold_turn"])
        base_scores.append(base_ov)

        ep_prompt = (
            "Below is a line from a personal conversation. Write 3 short questions that this "
            "line would answer, as a person might later ask them. Output only the questions, "
            "one per line, no numbering.\n\nLine: " + m["gold_turn"]
        )
        try:
            ep_qs = [l.strip(" -*") for l in groq_chat(ep_prompt, model, key, base).splitlines() if l.strip()][:3]
        except Exception as e:
            print(f"  [{i}] episode-gen failed: {e}", file=sys.stderr)
            ep_qs = []
        ep_best = max((overlap(q_real, q) for q in ep_qs), default=0.0)
        ep_scores.append(ep_best)

        fact_best = 0.0
        fact_qs = []
        if m["facts"]:
            fact_prompt = (
                "Below are structured facts extracted from a conversation. Write 3 short "
                "questions these facts would answer, as a person might later ask them. Output "
                "only the questions, one per line, no numbering.\n\nFacts:\n"
                + "\n".join(m["facts"])
            )
            try:
                fact_qs = [l.strip(" -*") for l in groq_chat(fact_prompt, model, key, base).splitlines() if l.strip()][:3]
            except Exception as e:
                print(f"  [{i}] fact-gen failed: {e}", file=sys.stderr)
            fact_best = max((overlap(q_real, q) for q in fact_qs), default=0.0)
        fact_scores.append(fact_best)

        # Semantic check with the SAME embedder production uses.
        qv = ollama_embed(q_real)
        if qv:
            gv = ollama_embed(m["gold_turn"])
            base_cos.append(cosine(qv, gv) if gv else 0.0)
            if ep_qs:
                vs = [ollama_embed(q) for q in ep_qs]
                ep_cos.append(max((cosine(qv, v) for v in vs if v), default=0.0))
            if fact_qs:
                vs = [ollama_embed(q) for q in fact_qs]
                fact_cos.append(max((cosine(qv, v) for v in vs if v), default=0.0))

        if len(samples) < 4:
            samples.append((q_real, m["gold_turn"][:90], ep_qs[:2], fact_qs[:2]))
        print(f"  [{i+1}/{len(misses)}] base={base_ov:.2f} ep={ep_best:.2f} fact={fact_best:.2f}")

    def stat(xs):
        return f"mean {100*statistics.mean(xs):5.1f}%  median {100*statistics.median(xs):5.1f}%" if xs else "n/a"

    print("\n" + "=" * 78)
    print("DOC2QUERY CEILING PROBE — would generated questions have matched the real ones?")
    print("=" * 78)
    print(f"n misses = {len(misses)}   (fact-linked: {with_facts})")
    print("\nLEXICAL overlap with the REAL missed question (higher = better match):")
    print(f"  BASELINE  raw gold turn (what we index today) : {stat(base_scores)}")
    print(f"  EPISODE   generated questions from the turn   : {stat(ep_scores)}")
    print(f"  FACT      generated questions from the facts  : {stat(fact_scores)}")
    print("\nSEMANTIC cosine vs the real question (nomic-embed-text, production embedder):")
    print(f"  BASELINE  raw gold turn                       : {stat(base_cos)}")
    print(f"  EPISODE   generated questions                  : {stat(ep_cos)}")
    print(f"  FACT      generated questions                  : {stat(fact_cos)}")
    print("\nVERDICT GUIDE: enrichment is only worth building if EPISODE or FACT scores")
    print("materially EXCEED the BASELINE — otherwise the generated text adds nothing")
    print("the current index does not already contain.")
    print("\n--- samples ---")
    for q, turn, eq, fq in samples:
        print(f"\nREAL Q : {q}")
        print(f"GOLD   : {turn}")
        print(f"GEN-EP : {eq}")
        print(f"GEN-FA : {fq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

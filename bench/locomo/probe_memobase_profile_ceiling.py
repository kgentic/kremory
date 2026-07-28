#!/usr/bin/env python3
"""Ceiling probe for a memobase-style ALWAYS-INJECTED profile (RECALL-LEDGER §5.6).

WHY THIS ONE, AND WHY NOW
-------------------------
§4bis established that **99.1% of gold evidence already sits inside a depth-200
pool** — the remaining gap is ORDERING, not findability. That demotes every
write-side idea whose value proposition is "now the abstract question matches
something" (doc2query, EnrichIndex, session compaction): they make content more
findable, and findability is not the constraint.

memobase is the exception, and that is exactly why it is the strongest remaining
candidate: **it is not a retrieval lever at all.** The profile has no embedding
column, is never ranked, never fused, never competes for a top-k slot — it is
injected into context up to a token budget on every call. It therefore sidesteps
the one bottleneck we have repeatedly failed to beat with better candidates
(bigger cross-encoder −0.3 nDCG, deeper pool −0.3 nDCG, only 43% of the oracle
reordering gain captured). It also costs ZERO read-time retrieval, which fits the
finding that the cross-encoder is 92% of a lookup.

Its schema is `topic × sub_topic` and the extraction prompt tells the model to
**infer what is implied** — `demographics → marital_status`, `psychological →
personality` are literally our own worked failure examples from the abstraction-
mismatch failure class.

WHAT THIS MEASURES (and what it does NOT)
-----------------------------------------
A CEILING, not an end-to-end score: for the questions our BEST retrieval config
currently gets wrong, would the answer have been PRESENT in an always-injected
profile? If the profile does not contain the answers, no amount of injection
machinery can help and the direction dies for ~$0.05. If it does, the direction
is licensed and earns a real design cycle (`/ship-decision`, per the ledger).

It scores with the harness's OWN `check_answer_in_memories`, so a "rescued"
verdict here means exactly what "correct" means everywhere else in this campaign
— no new, friendlier scorer is introduced to flatter the candidate.

It also reports the profile's TOKEN COST, because "always injected" means the
consumer pays it on every single call: a profile that rescues 30% of misses but
costs 4k tokens per request is a different product decision from one that costs 400.

USAGE
  python3 probe_memobase_profile_ceiling.py [--run results/w1-jina1-k50.json]
                                            [--conversations 0]
  (needs KREMORY_MCP_CHAT_API_KEY / KREMORY_MCP_CHAT_BASE_URL from .env)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# memobase's shape: a FIXED schema, not free-form summarisation. The fixed schema
# is the mechanism — it forces the model to fill slots it would not volunteer,
# which is what turns an implied fact into a retrievable one.
PROFILE_SCHEMA = {
    "basic_info": ["name", "age", "gender", "location", "occupation", "education"],
    "demographics": ["marital_status", "family", "children", "living_situation"],
    "interest": ["hobbies", "sports", "music", "food", "travel", "media"],
    "psychological": ["personality", "values", "emotional_patterns", "goals"],
    "life_event": ["milestones", "losses", "achievements", "health"],
    "work": ["role", "employer", "projects", "career_goals"],
    "relationship": ["friends", "partners", "community", "pets"],
}

PROMPT = """You are building a long-term user profile from a conversation, in the
style of a structured memory system.

Fill the schema below for EACH speaker. Two rules that matter more than coverage:

1. INFER WHAT IS IMPLIED. If someone mentions raising a child alone, record
   marital_status. If they describe how they react under stress, record
   personality. Do not restrict yourself to facts stated verbatim — the whole
   point of this profile is to hold what the raw text only implies.
2. Every value must be traceable to something in the conversation. Do not invent.
   Omit a slot rather than guess it.

Schema (topic -> sub_topics):
{schema}

Return ONLY a JSON object of the form:
  {{"<speaker>": {{"<topic>": {{"<sub_topic>": "<value>", ...}}, ...}}, ...}}

Conversation:
{conversation}
"""


def groq_chat(prompt: str, model: str, key: str, base: str, max_tokens: int = 4000) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Groq's edge returns 403 for the default `Python-urllib/3.x` agent on
            # bodies of this size — the identical request succeeds under curl.
            # Diagnosed 2026-07-28 by isolating auth (both keys returned 200 via
            # curl) before touching the payload. `probe_doc2query_ceiling.py` has
            # the same urllib shape and has never been run, so it will hit this too.
            "User-Agent": "kremory-bench-probe/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        # gpt-oss emits a separate `reasoning` channel; an empty `content` means
        # the budget was spent there. Fail LOUDLY rather than returning "" and
        # letting the caller record a 0% rescue rate that is really a truncation.
        raise RuntimeError(
            f"empty `content` (finish likely truncated; reasoning head: "
            f"{(msg.get('reasoning') or '')[:200]!r}) — raise --max-tokens"
        )
    return content


def flatten_profile(profile: dict) -> list[str]:
    """One 'memory' string per filled slot — the shape the scorer already expects,
    so the profile is scored exactly as a retrieved memory would be."""
    out = []
    for speaker, topics in (profile or {}).items():
        if not isinstance(topics, dict):
            continue
        for topic, subs in topics.items():
            if not isinstance(subs, dict):
                out.append(f"{speaker} {topic}: {subs}")
                continue
            for sub, val in subs.items():
                out.append(f"{speaker} {topic} {sub}: {val}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "results" / "w1-jina1-k50.json")
    ap.add_argument("--dataset", type=Path, default=HERE / "data" / "locomo10.json")
    ap.add_argument("--conversations", type=int, nargs="*", default=[0])
    ap.add_argument("--out", type=Path, default=HERE / "results" / "probe-memobase.json")
    # The run file was scored over its FULL retrieved list (50). A consumer sees
    # k. Re-deciding "is this a miss" at the real k is the fair comparison —
    # an always-injected profile competes against what the consumer actually
    # gets, not against a 50-deep pool they never see.
    ap.add_argument("--k", type=int, default=None,
                    help="re-decide correctness over the top-k retrieved memories only")
    ap.add_argument("--reuse-profile", action="store_true",
                    help="load the profile from --out instead of paying for a new LLM call")
    args = ap.parse_args()

    harness = load_mod("harness", HERE / "harness.py")

    key = os.environ.get("KREMORY_MCP_CHAT_API_KEY")
    base = os.environ.get("KREMORY_MCP_CHAT_BASE_URL")
    model = os.environ.get("PROBE_MODEL", "openai/gpt-oss-120b")
    if not (key and base):
        print("FATAL: KREMORY_MCP_CHAT_API_KEY / _BASE_URL not set", file=sys.stderr)
        return 2

    data = json.loads(args.dataset.read_text())
    rows = json.loads(args.run.read_text())["results"]

    schema_txt = "\n".join(f"  {t}: {', '.join(s)}" for t, s in PROFILE_SCHEMA.items())
    report = []

    for conv_idx in args.conversations:
        conv = data[conv_idx]
        sample_id = conv["sample_id"]
        c = conv["conversation"]

        lines = []
        for key_name in sorted(
            (k for k in c if k.startswith("session_") and isinstance(c[k], list)),
            key=lambda k: int(k.split("_")[1]),
        ):
            for t in c[key_name]:
                lines.append(f"{t.get('speaker','')}: {t.get('text','')}")
        conversation = "\n".join(lines)

        print(f"[{sample_id}] conversation: {len(lines)} turns, ~{len(conversation)//4} tokens")
        if args.reuse_profile and args.out.exists():
            cached = {e["sample_id"]: e["profile"] for e in json.loads(args.out.read_text())}
            if sample_id in cached:
                raw = json.dumps(cached[sample_id])
                print("  (reusing cached profile — no LLM call)")
            else:
                raw = groq_chat(PROMPT.format(schema=schema_txt, conversation=conversation),
                                model, key, base)
        else:
            raw = groq_chat(
                PROMPT.format(schema=schema_txt, conversation=conversation),
                model, key, base,
            )
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.split("```")[1]
            txt = txt[4:] if txt.startswith("json") else txt
        try:
            profile = json.loads(txt)
        except json.JSONDecodeError as e:
            print(f"  profile did not parse: {e}\n  raw head: {txt[:300]}")
            continue

        slots = flatten_profile(profile)
        profile_text = "\n".join(slots)
        approx_tokens = len(profile_text) // 4
        print(f"  profile: {len(slots)} filled slots, ~{approx_tokens} tokens "
              f"(this is the ALWAYS-INJECTED per-call cost)")

        conv_rows = [r for r in rows
                     if r.get("sample_id") == sample_id and r.get("category") != "adversarial"]

        def is_hit(r: dict) -> bool:
            """Correctness AT THE DEPTH THE CONSUMER SEES. The run file's own
            `is_correct` was decided over its full retrieved list; re-deciding at
            --k is what makes the comparison fair, since the profile competes
            against the top-k a caller actually receives."""
            if args.k is None:
                return str(r.get("is_correct")) == "True"
            mems = (r.get("recalled_memories") or [])[: args.k]
            return harness.check_answer_in_memories(
                r["expected_answer"], mems, r.get("category", "")
            )[0]

        misses = [r for r in conv_rows if not is_hit(r)]
        depth = f"top-{args.k}" if args.k else "full retrieved list"
        print(f"  scorable: {len(conv_rows)}   current misses at {depth}: {len(misses)}")

        rescued, rescued_examples = 0, []
        for r in misses:
            ok, _conf, _why = harness.check_answer_in_memories(
                r["expected_answer"], slots, r.get("category", "")
            )
            if ok:
                rescued += 1
                if len(rescued_examples) < 8:
                    rescued_examples.append((r["category"], r["question"][:70],
                                             str(r["expected_answer"])[:40]))

        # Control: how many ALREADY-CORRECT questions does the profile alone also
        # answer? A profile that "answers" nearly everything is matching loosely,
        # not recalling — this is the probe's own false-positive check.
        already = [r for r in conv_rows if is_hit(r)]
        control = sum(
            1 for r in already
            if harness.check_answer_in_memories(r["expected_answer"], slots, r.get("category", ""))[0]
        )

        pct = 100.0 * rescued / len(misses) if misses else 0.0
        print(f"  ── RESCUED {rescued}/{len(misses)} misses ({pct:.1f}%)"
              f"   = +{100.0 * rescued / len(conv_rows):.1f}pt on this conversation")
        print(f"  ── control: profile alone also covers {control}/{len(already)} "
              f"already-correct ({100.0 * control / len(already) if already else 0:.1f}%)")
        for cat, q, a in rescued_examples:
            print(f"       [{cat}] {q}  ->  {a}")

        report.append({
            "sample_id": sample_id, "slots": len(slots), "approx_tokens": approx_tokens,
            "scorable": len(conv_rows), "misses": len(misses), "rescued": rescued,
            "control_covered": control, "control_total": len(already),
            "profile": profile,
        })

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten -> {args.out}")
    print("\nKILL CRITERION: rescue rate <10% of misses => always-injection cannot")
    print("reach our failure class; direction dies here for the price of one LLM call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
SPIKE — can the LLM decide entity identity from CONTEXT, given a WIDER candidate set?

Throwaway. Not wired to anything. Delete when the question is answered.

## The question this exists to answer

The shipped L5 adjudicator (2fd9a48e) shows the model only two names + two 400-char
descriptions, and only ever shows it pairs that already share >=50% of their words.
So (a) it is not "reading context" in any real sense, and (b) `IBM`/`International
Business Machines` and `Bob`/`Robert` are filtered out before it sees anything.

Removing those two cages is what the project owner asked for. The measured argument
AGAINST is TD-222: Site #5 already does "wide candidates + LLM decides", and it
merged unrelated co-occurring entities on one verdict, answering true to 19 of 19
pairs. The argument FOR is that Site #5's failure may have been weak candidates and
thin context rather than the model's judgement — today's shipped run rejected 6 of 7
when given descriptions.

This spike separates those. Same pairs, two context levels:

    THIN — name + description            (what ships today)
    RICH — name + description + the entity's facts + episode text it appears in

## SCOPE — read this before believing any number here

This probes the MODEL + PROMPT. It does NOT exercise the Rust pipeline, the
write_gate, or the merge executor. A pass here means "the model can make this call",
NOT "our code does". The Rust change would still need its own verification.

## Truth set

Source: `dream-off.db` — the arm where NOTHING was merged, so every entity survives
with its original name and real context.

NEGATIVES (correct answer: NOT the same entity)
    Pairs where one name's token set is a strict SUBSET of the other's — i.e. "X" vs
    "X + extra word". That is the hypernym signature that cost 9.9 nDCG
    (`pottery class`/`pottery`). Every candidate is PRINTED for inspection, because
    the author labelling its own test set is exactly the tautology this project has
    been bitten by; some superset pairs are genuinely ambiguous and are EXCLUDED by
    name rather than silently labelled.

POSITIVES (correct answer: SAME entity)
    Synthetic, and same-by-construction: take a REAL entity and corrupt its NAME
    (misspelling, or abbreviate to an initial), while giving it that real entity's
    OWN context. conv0 contains no natural `alice j`/`alice johnson` pair, which is
    precisely the gap that makes this necessary — and it is the case a wider
    candidate set is supposed to unlock.

A model that scores well on negatives but badly on positives is a rejector, not a
discriminator, and is useless for widening the candidate set. Both directions matter.
"""

import json
import re
import sqlite3
import sys
import urllib.request

DB = ".context/td186a-variance/dream-off.db"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

# ── The truth set is HAND-CURATED, and that is deliberate ────────────────────
#
# The first version of this spike generated negatives mechanically: "one name's
# tokens are a strict subset of the other's". It produced 123 pairs and the labels
# were WRONG on several — `pride parade` vs `lgbtq pride parade` and `lgbtq center`
# vs `lgbtq youth center` are plausibly the SAME entity, but the rule labelled them
# "not same". A model answering those correctly would have scored as wrong, and the
# conclusion would have been "the model is bad" when the LABELS were bad.
#
# That is the author-labels-its-own-test-set tautology. A heuristic does not escape
# it — it just hides it. So the labels are explicit, and both lists are PRINTED at
# run time so they can be challenged rather than trusted.

# Correct answer: NOT the same entity. Each is a category/topic vs a specific thing
# belonging to it, or two distinct instances (different sessions, different dates).
CURATED_NEGATIVES = [
    ("pottery", "pottery class"),
    ("pottery", "pottery project"),
    ("painting", "horse painting"),
    ("painting", "melanies painting"),
    ("lgbtq", "lgbtq conference"),
    ("lgbtq", "lgbtq art show"),
    ("lgbtq", "lgbtq activist group"),
    ("community", "lgbtq community"),
    ("community", "trans community"),
    ("family", "family support"),
    ("family", "family camping trip"),
    ("nature", "nature and family time"),
    ("adoption", "adoption agencies"),
    ("adoption", "adoption advice"),
    ("meeting", "council meeting"),
    ("journey", "transgender journey"),
    ("summer", "summer traditions"),
    ("show", "talent show"),
    ("lake", "lake sunrise"),
    ("session 1", "session 11"),
    ("8 may 2023", "25 may 2023"),
    ("3 july 2023", "12 july 2023"),
]

# Correct answer: SAME entity — NATURAL positives found in the corpus itself
# (singular/plural of one referent), not synthesised.
CURATED_POSITIVES_NATURAL = [
    ("adoption agency", "adoption agencies"),
]

# Genuinely arguable — EXCLUDED from scoring entirely, by name, so the exclusion is
# visible. Including them would score the model against a coin-flip.
AMBIGUOUS_EXCLUDE = {
    ("pride parade", "lgbtq pride parade"),      # plausibly the same event
    ("lgbtq center", "lgbtq youth center"),      # plausibly the same place
    ("camping trip", "melanies camping trip"),   # plausibly the same trip
    ("kids", "those kids"),                      # determiner only
    ("family", "family love"),
    ("family", "having a family"),
    ("a photo", "photo of"),                     # junk entities, meaningless either way
}


def norm_tokens(name):
    return {t for t in re.split(r"\s+", name.lower().strip()) if len(t) >= 2}


def load_entities(conn):
    out = {}
    for eid, props in conn.execute(
        "SELECT id, properties FROM entities WHERE recorded_at IS NOT NULL"
    ):
        desc = ""
        if props:
            try:
                desc = (json.loads(props) or {}).get("description") or ""
            except json.JSONDecodeError:
                desc = ""
        out[eid] = desc
    return out


def entity_facts(conn, eid, limit=6):
    rows = conn.execute(
        "SELECT predicate, COALESCE(object_id, object_value) FROM facts "
        "WHERE subject_id = ? AND expired_at IS NULL LIMIT ?",
        (eid, limit),
    ).fetchall()
    rows += conn.execute(
        "SELECT predicate, subject_id FROM facts "
        "WHERE object_id = ? AND expired_at IS NULL LIMIT ?",
        (eid, limit),
    ).fetchall()
    return [f"{p} {o}" for p, o in rows if o]


def entity_episodes(conn, eid, limit=2, window=300):
    rows = conn.execute(
        "SELECT e.content FROM episodes e JOIN episodic_edges ee ON ee.episode_id = e.id "
        "WHERE ee.entity_id = ? LIMIT ?",
        (eid, limit),
    ).fetchall()
    out = []
    for (content,) in rows:
        if not content:
            continue
        # Centre the snippet on the first mention so the window is informative
        # rather than always the episode header.
        idx = content.lower().find(eid.split()[0].lower())
        start = max(0, idx - window // 3) if idx >= 0 else 0
        out.append(content[start : start + window].replace("\n", " "))
    return out


def block(conn, display_name, real_id, entities, rich):
    """One side of a pair. `display_name` may be a corrupted form of `real_id`;
    context is always drawn from `real_id`, which is what makes a synthetic
    positive same-entity BY CONSTRUCTION."""
    desc = entities.get(real_id, "")
    s = f'name="{display_name}" description="{desc[:400]}"'
    if rich:
        facts = entity_facts(conn, real_id)
        eps = entity_episodes(conn, real_id)
        if facts:
            s += "\n      facts: " + "; ".join(facts)
        if eps:
            s += "\n      mentioned in: " + " ||| ".join(eps)
    return s


SYSTEM = (
    "You are a knowledge-graph entity-resolution analyst. For each pair below, decide "
    "whether the two entries denote the SAME real-world entity.\n\n"
    "Two names can share most of their words and still be DIFFERENT entities, because "
    "one names a category, topic or activity that the other belongs to: 'pottery class' "
    "is not 'pottery'; 'chess club' is not 'chess'.\n\n"
    "Conversely two names can share NO words and still be the SAME entity: an acronym, "
    "a nickname, an initial or a misspelling of the same referent. Use the supplied "
    "context to decide which situation you are in.\n\n"
    "A wrong merge destroys data irreversibly; a missed merge is retried later. If "
    "genuinely unsure, answer false with low confidence.\n\n"
    "Respond ONLY with JSON: {\"verdicts\":[{\"pair_id\":0,\"is_same_entity\":true,"
    "\"confidence\":0.9,\"reasoning\":\"...\"}]}"
)


def ask(pairs, conn, entities, rich):
    lines = []
    for i, p in enumerate(pairs):
        a = block(conn, p["a_name"], p["a_real"], entities, rich)
        b = block(conn, p["b_name"], p["b_real"], entities, rich)
        lines.append(f"Pair {i}:\n  A: {a}\n  B: {b}")
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join(lines)},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [parse-fail] raw: {raw[:300]}")
        return {}
    return {v["pair_id"]: v for v in parsed.get("verdicts", []) if "pair_id" in v}


def main():
    conn = sqlite3.connect(DB)
    entities = load_entities(conn)
    ids = list(entities)
    print(f"[load] {len(ids)} entities from {DB}\n")

    # ── NEGATIVES: hand-curated, both entities must really exist ────────────
    negatives, skipped = [], []
    for a, b in CURATED_NEGATIVES:
        if (a, b) in AMBIGUOUS_EXCLUDE or (b, a) in AMBIGUOUS_EXCLUDE:
            continue
        if a in entities and b in entities:
            negatives.append({"a_name": a, "a_real": a, "b_name": b, "b_real": b})
        else:
            skipped.append((a, b, a in entities, b in entities))
    if skipped:
        print("[skip] curated negatives whose entities are absent from this graph:")
        for a, b, ha, hb in skipped:
            print(f"    {a!r}({'y' if ha else 'N'})  vs  {b!r}({'y' if hb else 'N'})")
        print()

    # ── POSITIVES: same entity by construction, name corrupted ──────────────
    def misspell(n):
        # drop one interior vowel from the longest token
        toks = n.split()
        i = max(range(len(toks)), key=lambda k: len(toks[k]))
        t = toks[i]
        for j in range(1, len(t) - 1):
            if t[j] in "aeiou":
                toks[i] = t[:j] + t[j + 1 :]
                break
        return " ".join(toks)

    def abbreviate(n):
        toks = n.split()
        return f"{toks[0]} {toks[1][0]}" if len(toks) >= 2 else None

    positives = []
    # Natural positives from the corpus first — no synthesis, no author judgement
    # beyond "a singular and its plural denote one referent".
    for a, b in CURATED_POSITIVES_NATURAL:
        if a in entities and b in entities:
            positives.append({"a_name": a, "a_real": a, "b_name": b, "b_real": a})

    # Synthetic, same-by-construction: the NAME is corrupted, the CONTEXT stays the
    # real entity's. This is the `alice j`/`alice johnson` case conv0 does not
    # contain, and it is exactly what a wider candidate set is meant to unlock —
    # note token-Jaccard for an abbreviation is BELOW the shipped 0.5 gate, so the
    # production pipeline would never show these to the model at all.
    for e in ids:
        if len(e.split()) >= 2 and len(e) > 8 and len(positives) < 11:
            ab = abbreviate(e)
            if ab and ab != e:
                positives.append({"a_name": e, "a_real": e, "b_name": ab, "b_real": e})
    for e in ids:
        if len(e) > 6 and " " in e and len(positives) < 22:
            ms = misspell(e)
            if ms != e:
                positives.append({"a_name": e, "a_real": e, "b_name": ms, "b_real": e})

    print(f"=== NEGATIVES ({len(negatives)}) — correct answer: NOT same. Inspect these: ===")
    for p in negatives:
        print(f"    {p['a_name']!r}  vs  {p['b_name']!r}")
    print(f"\n=== POSITIVES ({len(positives)}) — correct answer: SAME (name corrupted, context is the real entity's) ===")
    for p in positives[:8]:
        print(f"    {p['a_name']!r}  vs  {p['b_name']!r}")
    print(f"    ... and {max(0, len(positives) - 8)} more\n")

    if not negatives or not positives:
        print("FATAL: one side of the truth set is empty — the probe cannot discriminate.")
        sys.exit(1)

    for rich in (False, True):
        arm = "RICH (name+desc+facts+episodes)" if rich else "THIN (name+desc only)"
        print(f"\n{'='*70}\n  ARM: {arm}\n{'='*70}")
        results = {}
        for label, pairs, want_same in (("NEG", negatives, False), ("POS", positives, True)):
            correct = wrong = missing = 0
            for i in range(0, len(pairs), 8):  # chunk, mirroring production
                chunk = pairs[i : i + 8]
                verdicts = ask(chunk, conn, entities, rich)
                for j, p in enumerate(chunk):
                    v = verdicts.get(j)
                    if v is None:
                        missing += 1
                        continue
                    got = bool(v.get("is_same_entity"))
                    ok = got == want_same
                    correct += ok
                    wrong += not ok
                    if not ok:
                        print(
                            f"    [{label} WRONG] {p['a_name']!r} vs {p['b_name']!r} "
                            f"-> same={got} conf={v.get('confidence')} :: {str(v.get('reasoning'))[:110]}"
                        )
            total = correct + wrong + missing
            rate = 100.0 * correct / total if total else 0.0
            results[label] = rate
            print(f"  {label}: {correct}/{total} correct ({rate:.1f}%), {missing} no-verdict")
        print(f"  --> reject-rate on hypernyms {results['NEG']:.1f}%  |  accept-rate on same-entity {results['POS']:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LoCoMo QA-generation prompts — the answerer + answer-judge stage.

This is the missing stage that makes kremory's LoCoMo number *comparable* to
mem0 / Zep / etc. Those systems report a QA-answer-generation score:
  retrieve top-k  ->  an ANSWERER LLM generates an answer from the memories
                  ->  a JUDGE LLM labels the generated answer CORRECT/WRONG vs gold.
kremory's existing harness scores a retrieval PROXY (is the gold text present in
the recalled memories). Same retrieval, different scorer -> non-comparable number.

The ANSWER_GENERATION_PROMPT and the JUDGE prompt below are ported VERBATIM from
mem0's open-source benchmark harness so the judged number is a like-for-like
comparison. Source (saved for provenance at .context/mem0-ref/mem0-locomo-prompts.py):
  https://github.com/mem0ai/memory-benchmarks
  benchmarks/locomo/prompts.py @ edcd6f1d4240 (2026-04-09)

NUANCE (plan assumption A2): this is mem0's CURRENT (top-200 "2026 suite") prompt
— its own headline numbers with THIS prompt run ~92% at top-200. We feed only
k=10 memories into it (peer-comparable s=10 budget), so the honest expectation is
~55-70%, bounded above by recall@10 (~64% on conv0). The judge is deliberately
LENIENT by design (partial credit, paraphrase-ok, +-14d dates, same-referent);
that is why the driver validates it against a hand-labeled control set before the
aggregate is trusted (guards the "Benchmark Theatre" trap, plan A5).

ADAPTATION: mem0's `get_answer_generation_prompt` expects memory DICTS
({memory, created_at}) it sorts chronologically. kremory's recalled memories are
plain STRINGS (session-tagged dialogue excerpts with inline dates). The prompt
*strings* are ported verbatim (identical reasoning/judge logic = comparability);
only the memory-formatting helper differs to fit string memories.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Category mapping (mem0 uses int 1-5; kremory's harness uses these strings).
# Only the open-domain preprocessing keys on category (gold before ';').
# adversarial (5) is excluded from scoring and never reaches this stage.
# ---------------------------------------------------------------------------
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
_NAME_TO_ID = {v: k for k, v in CATEGORY_NAMES.items()}


# ===========================================================================
# ANSWER GENERATION PROMPT — ported verbatim from mem0 (edcd6f1d4240)
# ===========================================================================
ANSWER_GENERATION_PROMPT = """You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
"""


# ===========================================================================
# JUDGE — ported verbatim from mem0 (no-evidence variant; we do NOT use the
# evidence oracle, per the plan's honesty rules). JUDGE_SYSTEM_PROMPT +
# the resolved no-evidence template (evidence_section / evidence_rule /
# evidence_wrong_clause all empty).
# ===========================================================================
JUDGE_SYSTEM_PROMPT = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


# ===========================================================================
# kremory adapters — string memories + string categories
# ===========================================================================
def preprocess_answer(category: str, answer: str) -> str:
    """Gold-answer preprocessing (mem0 parity).

    open-domain (mem0 cat 3): use only the part before the first ';'. The gold
    for open-domain questions is `<canonical answer>; <supporting explanation>`
    — the judge compares against the canonical part only.
    """
    answer = str(answer)  # gold is occasionally an int (e.g. a bare year 2022)
    if category == "open-domain" and ";" in answer:
        return answer.split(";")[0].strip()
    return answer


_INLINE_DATE = re.compile(
    r"\[[^\]]*?on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\]"
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        1,
    )
}


def _inline_date_key(mem: str) -> tuple[int, int, int] | None:
    """Parse the `[... on 9 June, 2023]` stamp kremory renders inline.

    Returns `None` when there is no stamp — entity profiles and extracted facts
    carry none (measured: 16% of retrieved items on conv0).
    """
    g = _INLINE_DATE.search(str(mem))
    if not g:
        return None
    month = _MONTHS.get(g.group(2).lower())
    if month is None:
        return None
    return (int(g.group(3)), month, int(g.group(1)))


def format_memories(memories: list[str], *, chronological: bool = False) -> str:
    """Format kremory's plain-string recalled memories into the {memories} slot.

    ## `chronological` — restores mem0's ordering (opt-in)

    mem0's `get_answer_generation_prompt` takes dict memories `{memory,
    created_at}` and **sorts them chronologically** before the answerer sees
    them. Our adaptation fed them in RETRIEVAL-RANK order instead, justified on
    the grounds that kremory's excerpts "already carry inline session dates" so
    the answerer could order them itself.

    **That justification is empirically false**, and our own paid smoke
    (2026-08-11, 10 questions, $0.03) caught it: asked *"When did Caroline meet
    up with her friends, family and mentors?"*, kremory retrieved the correct
    evidence at **rank 1** (`[Session 3] [7:55 pm on 9 June, 2023]`) — and the
    answerer replied "5-11 July 2023", having read the *third* memory
    (`12 July, 2023`). Correct retrieval, wrong answer, purely from presentation
    order.

    Enabling this moves us TOWARD the reference implementation, so it increases
    comparability rather than gaming it — the distinction that matters, since the
    prompts themselves are ported verbatim precisely to keep the numbers
    comparable.

    **Opt-in, default OFF, deliberately.** Flipping the default would silently
    re-base every historical qa-gen number on this page. Measure the delta, then
    decide to re-base explicitly.

    Undated items (entity profiles, extracted facts — no inline stamp) keep their
    relevance order and lead, as orientation; dated dialogue excerpts follow in
    time order. That matches both mem0's chronological principle for dialogue and
    the `_SECTIONS` philosophy below of putting distilled output first.
    """
    if not memories:
        return "(No relevant memories found)"

    ordered = memories
    if chronological:
        undated = [m for m in memories if _inline_date_key(m) is None]
        dated = [m for m in memories if _inline_date_key(m) is not None]
        # `sorted` is stable, so equal-dated items keep relevance order.
        dated.sort(key=lambda m: _inline_date_key(m))  # type: ignore[arg-type,return-value]
        ordered = undated + dated

    lead = (
        "The following memories were retrieved from past conversations "
        "(each is a dialogue excerpt with inline dates). Read every one:"
    )
    if chronological:
        lead = (
            "The following memories were retrieved from past conversations. "
            "The dated dialogue excerpts are in CHRONOLOGICAL ORDER (earliest "
            "first); any undated profiles or facts appear first. Read every one:"
        )
    parts = [lead, ""]
    for i, mem in enumerate(ordered, 1):
        parts.append(f"[Memory {i}]\n{mem}")
    return "\n\n".join(parts)


# Wire `kind` values (`SearchResultKindWire`, kremory-http.rs) -> the section a
# retrieved item belongs in. Order is deliberate: verbatim conversation LAST,
# because it is the longest and the answerer is instructed to read to the end;
# the distilled graph output goes first, as orientation.
_SECTIONS = [
    ("Fact", "STRUCTURED FACTS extracted from these conversations",
     "Each line is a fact the memory system derived. Use them to orient, but "
     "prefer the conversation excerpts below where they disagree."),
    ("Entity", "ENTITY PROFILES",
     "Short profiles of the people, places and things mentioned."),
    ("Episode", "CONVERSATION EXCERPTS (verbatim, with inline dates)",
     "The source dialogue. These carry the dates and the exact wording."),
]


def format_memories_structured(
    memories: list[str],
    provenance: list[dict] | None,
) -> str:
    """ADR-078 Phase A - group retrieved items by WHAT THEY ARE, rather than
    flattening entity summaries, facts and verbatim turns into one anonymous list.

    Why this exists: measured on conv0, adding the graph to retrieval moves
    single-hop +15.6pt (entity lookup - the graph doing its job) but temporal
    -8.1pt and multi-hop -7.7pt. Of the questions it flips right->wrong, **8 of 8
    had the gold evidence present in context anyway** - so the loss is not
    eviction, it is DILUTION: the model gets 20 undifferentiated items and must
    work out for itself which are source text and which are derived summaries.

    kremory already ships a structured renderer for exactly this
    (`RecallTemplate::{Entities, EdgeSummary, TemporalFacts}` / `as_prompt_text`)
    and no benchmark has ever used it. This is the cheap harness-side equivalent,
    so the FORMAT can be A/B'd before anything is built.

    Falls back to the flat format when provenance is absent or short, so any run
    file without `recalled_memory_provenance` scores exactly as before.
    """
    if not memories:
        return "(No relevant memories found)"
    if not provenance or len(provenance) < len(memories):
        return format_memories(memories)

    buckets: dict[str, list[str]] = {}
    for mem, prov in zip(memories, provenance):
        kind = (prov or {}).get("kind") or "Episode"
        buckets.setdefault(str(kind), []).append(mem)

    known = {k for k, _, _ in _SECTIONS}
    # Anything unrecognised joins the verbatim section rather than vanishing - a
    # silently-dropped item would be a measurement bug, not a formatting choice.
    for k in list(buckets):
        if k not in known:
            buckets.setdefault("Episode", []).extend(buckets.pop(k))

    out: list[str] = [
        "The following was retrieved from past conversations, grouped by what "
        "each item IS. Read every section to the end:",
    ]
    for kind, title, blurb in _SECTIONS:
        items = buckets.get(kind) or []
        if not items:
            continue
        out.append(f"### {title}\n({blurb})")
        for i, mem in enumerate(items, 1):
            out.append(f"[{kind} {i}]\n{mem}")
    return "\n\n".join(out)


def format_text_block(block: str | None) -> str:
    """Pass through kremory's OWN server-rendered prompt-ready block
    (`GET /search?format=text&template=temporal_facts` — RECALL-LEDGER §4.19
    / TD-155) VERBATIM into the {memories} slot, instead of reconstructing a
    rendering harness-side from `recalled_memories` strings. This is what an
    MCP tool consumer actually receives — including `valid_at`, which
    `format_memories()`/`format_memories_structured()` never see because
    `flatten_result_content` (the REST `structured` renderer) drops it.

    `None` or empty means the harness didn't capture this question's second
    request (not run with `--capture-text-block`, `--server-mode content`,
    or the request failed) — falls back to the same "no context" sentinel
    `format_memories()` uses, so the answerer prompt is well-formed either way.
    """
    if not block:
        return "(No relevant memories found)"
    return block


def build_answer_prompt(
    question: str,
    memories: list[str],
    reference_date: str = "2023",
    provenance: list[dict] | None = None,
    structured: bool = False,
    text_block: str | None = None,
    chronological: bool = False,
) -> str:
    """Fill the ported ANSWER_GENERATION_PROMPT for kremory string memories.

    `structured=True` groups memories by wire `kind` (see
    [`format_memories_structured`]); the default is byte-identical to the flat
    format every published number was measured on.

    `text_block`, when not `None`, TAKES PRECEDENCE over both `structured`
    and the flat format: it is kremory's own server-rendered prompt-ready
    block (see [`format_text_block`]), passed through rather than
    reconstructed here. Mutually exclusive with `structured` at the caller
    (`qa_eval.py`) — this function does not itself enforce that, it simply
    prefers `text_block` when given.
    """
    if text_block is not None:
        rendered = format_text_block(text_block)
    else:
        rendered = (
            format_memories_structured(memories, provenance)
            if structured
            else format_memories(memories, chronological=chronological)
        )
    return ANSWER_GENERATION_PROMPT.format(
        memories=rendered,
        question=question,
        reference_date=reference_date,
    )


def build_judge_prompt(question: str, gold: str, response: str, category: str) -> str:
    """Fill the ported JUDGE_PROMPT (gold preprocessed for open-domain)."""
    return JUDGE_PROMPT.format(
        question=question,
        answer=preprocess_answer(category, gold),
        response=response,
    )

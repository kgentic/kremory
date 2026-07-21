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


def format_memories(memories: list[str]) -> str:
    """Format kremory's plain-string recalled memories into the {memories} slot.

    mem0 shows dated dict-memories chronologically; kremory memories are dialogue
    excerpts that already carry inline session dates, so we present them as an
    explicitly-numbered, position-labelled block. The delimiter is immaterial to
    the answerer's reasoning instructions (which reference "every memory below");
    numbering matches the longmemeval answerer's proven multi-line format.
    """
    if not memories:
        return "(No relevant memories found)"
    parts = [
        "The following memories were retrieved from past conversations "
        "(each is a dialogue excerpt with inline dates). Read every one:",
        "",
    ]
    for i, mem in enumerate(memories, 1):
        parts.append(f"[Memory {i}]\n{mem}")
    return "\n\n".join(parts)


def build_answer_prompt(
    question: str,
    memories: list[str],
    reference_date: str = "2023",
) -> str:
    """Fill the ported ANSWER_GENERATION_PROMPT for kremory string memories."""
    return ANSWER_GENERATION_PROMPT.format(
        memories=format_memories(memories),
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

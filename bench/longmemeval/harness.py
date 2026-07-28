#!/usr/bin/env python3
# NOTICE: Adapted from cogniplex/codemem (Apache-2.0) for kremory.
# Original: https://github.com/cogniplex/codemem/tree/main/bench/longmemeval
# Change: CodememClient retargeted at kremory-http's bare (no /api prefix)
# REST surface — POST /memories, GET /search, DELETE /namespaces/{ns},
# POST /consolidation/{cycle}?namespace= (namespace REQUIRED — kremory's
# dream() is always namespace-scoped, unlike codemem's global consolidation).
# graph_neighbors/get_memory are stubbed (no graph-traversal REST tool yet).
"""LongMemEval benchmark harness for codemem.

LongMemEval (ICLR 2025) evaluates long-term conversational memory across 6 question
types testing 5 core abilities:
- Single-session user: Recall user-stated info
- Single-session assistant: Recall assistant-stated info
- Single-session preference: Extract implicit preferences
- Multi-session: Synthesize across sessions
- Knowledge-update: Handle info that changed over time
- Temporal-reasoning: Time-based reasoning

Dataset: 500 questions, each with ~40 haystack sessions (~115K tokens).
Each question is evaluated independently: ingest → recall → generate → score → cleanup.

Requires OPENAI_API_KEY for answer generation.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm import tqdm

CODEMEM_BASE = "http://localhost:3179"  # kremory-http: bare routes, NO /api prefix
DEFAULT_DATASET = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"

# Answer-failure circuit breaker (mirrors bench/locomo/harness.py CIRCUIT_*).
# A paid run must not grind through 500 questions turning an outage into a
# score. Checked once, after the first CIRCUIT_MIN questions.
CIRCUIT_MIN = 20
CIRCUIT_FAILURE_RATE = 0.30


class AnswerGenerationFailed(RuntimeError):
    """The answerer never produced an answer (LLM error, timeout, bad key).

    This MUST NOT be collapsed into a string. The previous code returned the
    literal "I don't know." here, which `ABSTENTION_PHRASES` matches and
    `quick_score` scores CORRECT for any `_abs` question — so an OpenAI outage
    rendered as a perfect abstention score. An absence of measurement was read
    as a measurement. Same defect class as the LoCoMo category-5 false 100%
    (fixed 2026-07-28); this is its sixth instance in the benchmark suite.
    """


# Returned (not raised) when retrieval legitimately found nothing. Deliberately
# contains NO substring from ABSTENTION_PHRASES, so it cannot be mistaken for a
# demonstrated abstention. Mirrors the LoCoMo W0.2 rule: empty recall is a
# RETRIEVAL failure, not a demonstrated abstention, and is scored False so it
# surfaces in the number instead of hiding behind it.
NO_RECALL_MARKER = "<<retrieval returned zero memories>>"


@dataclass
class Config:
    base_url: str = CODEMEM_BASE
    dataset_path: Path = DEFAULT_DATASET
    recall_limit: int = 10
    graph_depth: int = 2
    mode: str = "codemem"  # baseline | codemem | codemem-graph
    llm_model: str = "gpt-4o"
    eval_model: str = "gpt-4o"
    use_llm_eval: bool = False
    max_questions: int = 0  # 0 = all
    skip_ingest: bool = False
    keep_corpus: bool = False  # survive the run so --skip-ingest can re-score
    output: Path | None = None


# ---------------------------------------------------------------------------
# kremory HTTP client — SHARED, not forked
# ---------------------------------------------------------------------------
#
# This file used to carry its own adapted copy of CodememClient. That fork
# never inherited the hardenings the LoCoMo side learned the hard way, most
# importantly `store_timeout = 90.0` and a try/except around POST /memories:
# kremory's ingest fans out to ~20 sequential LLM calls per store, so httpx's
# 30s default times out MID-STORE and the fork raised an uncaught
# httpx.ReadTimeout that killed the run. Reproduced 2026-07-28 on the first
# real call ever made through this harness. De-forked rather than patched a
# fourth time — see bench/common/kremory_client.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from kremory_client import CodememClient, KremoryStalled  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "samples", "questions"):
            if key in data:
                return data[key]
        return [data]
    return data


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def parse_haystack_date(date_str: str) -> str | None:
    """LongMemEval `haystack_dates` -> RFC3339, for kremory's `published_at`.

    Corpus format is `2023/05/25 (Thu) 20:21` — the weekday is parenthesised
    and must be stripped before parsing. Returns None (rather than a guessed
    timestamp) if the shape is unrecognised, so an unparseable date degrades to
    "no world-time" instead of silently inventing one.
    """
    if not date_str:
        return None
    cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", str(date_str)).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    print(f"  [warn] unparseable haystack date {date_str!r}", file=sys.stderr)
    return None


def ingest_sessions(
    client: CodememClient,
    namespace: str,
    item: dict,
) -> int:
    """Ingest haystack sessions for one question into codemem.

    Stores one memory per session (all turns concatenated).
    """
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    session_dates = item.get("haystack_dates", [])
    stored = 0

    for i, (turns, sid, date_str) in enumerate(
        zip(sessions, session_ids, session_dates, strict=True)
    ):
        if not turns:
            continue

        lines = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")

        session_content = f"[Session {sid}] [{date_str}]\n" + "\n".join(lines)

        tags = [
            f"question:{item['question_id']}",
            f"session:{sid}",
        ]

        mid = client.store_memory(
            content=session_content,
            namespace=namespace,
            memory_type="Context",
            importance=0.5,
            tags=tags,
            # WORLD-time for this session. Previously the date reached kremory
            # ONLY as text inside `session_content`, so the temporal axis was
            # never exercised by any benchmark — the server has always accepted
            # `published_at` and no harness had ever sent it. `haystack_dates`
            # is 1:1 with sessions in 500/500 corpus records (verified), so
            # this mapping is total.
            published_at=parse_haystack_date(date_str),
        )
        if mid:
            stored += 1

    return stored


# ---------------------------------------------------------------------------
# Recall strategies
# ---------------------------------------------------------------------------

def recall_codemem(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
) -> list[str]:
    results = client.recall(question, namespace, limit=limit)
    return [r.get("content", "") for r in results if r.get("content")]


def recall_codemem_graph(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
    graph_depth: int = 2,
) -> list[str]:
    results = client.recall(question, namespace, limit=limit)

    contents = []
    seen_ids = set()
    for r in results:
        content = r.get("content", "")
        node_id = r.get("id", r.get("node_id", ""))
        if content:
            contents.append(content)
        if node_id:
            seen_ids.add(node_id)

    seed_ids = [
        r.get("id", r.get("node_id", ""))
        for r in results[:10]
        if r.get("id") or r.get("node_id")
    ]
    for seed_id in seed_ids:
        if not seed_id:
            continue
        neighbors = client.graph_neighbors(seed_id, depth=graph_depth)
        for node in neighbors:
            memory_id = node.get("memory_id", node.get("id", ""))
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            if node.get("kind") == "Memory":
                mem = client.get_memory(memory_id)
                if mem and mem.get("content"):
                    contents.append(mem["content"])

    return contents[:limit * 2]


def recall_baseline(item: dict) -> list[str]:
    """Baseline: return all session text."""
    contents = []
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    session_dates = item.get("haystack_dates", [])

    for turns, sid, date_str in zip(sessions, session_ids, session_dates, strict=True):
        if not turns:
            continue
        lines = [f"[Session {sid}] [{date_str}]"]
        for turn in turns:
            role = turn.get("role", "unknown")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.get('content', '')}")
        contents.append("\n".join(lines))
    return contents


# ---------------------------------------------------------------------------
# Answer generation (LLM required)
# ---------------------------------------------------------------------------

def generate_answer(
    openai_client: OpenAI,
    question: str,
    memories: list[str],
    question_date: str,
    model: str = "gpt-4o",
) -> str:
    """Generate an answer using LLM with recalled memories as context.

    Raises `AnswerGenerationFailed` if the LLM call fails. It does NOT return a
    plausible-looking string on failure — see that exception's docstring.
    """
    if not memories:
        return NO_RECALL_MARKER

    context_parts = []
    for i, mem in enumerate(memories, 1):
        context_parts.append(f"[Memory {i}]\n{mem}")
    context = "\n\n".join(context_parts)

    # Truncate context to ~100K chars to stay within token limits
    if len(context) > 100_000:
        context = context[:100_000] + "\n\n[...truncated...]"

    prompt = f"""You are answering a question based on recalled conversation memories.

First, extract relevant information from each memory excerpt.
Then, reason about the answer based on the extracted information.
Finally, provide a concise answer.

If the information is not available in the provided memories, respond with "I don't know."

Memories:
{context}

Question (asked on {question_date}): {question}

Step 1 - Extract relevant information:
Step 2 - Reasoning:
Step 3 - Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        full_answer = response.choices[0].message.content.strip()

        # Extract Step 3 answer
        step3_match = re.search(
            r"Step\s*3(?:\s*[\.\-:]|\s)*\s*(?:Answer|Final Answer)?\s*[:\-]?\s*",
            full_answer,
            flags=re.IGNORECASE,
        )
        if step3_match:
            trailing = full_answer[step3_match.end():]
            next_step = re.search(r"\n\s*Step\s*\d+\s*[\.\-:]", trailing, flags=re.IGNORECASE)
            answer_part = trailing[:next_step.start()] if next_step else trailing
            answer_part = re.sub(
                r"^\s*(?:Answer|Final Answer)\s*[:\-]?\s*", "", answer_part, flags=re.IGNORECASE
            ).strip()
            if answer_part:
                return answer_part

        return full_answer

    except Exception as e:
        # Do NOT return "I don't know." here. That string is in
        # ABSTENTION_PHRASES, so every failed call on an `_abs` question was
        # scored CORRECT — an outage read as a perfect abstention score.
        # Fail loudly and let the caller record the question as UNSCORED.
        raise AnswerGenerationFailed(f"answer generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

ABSTENTION_PHRASES = [
    "i don't know", "i do not know", "cannot determine",
    "not enough information", "no information", "not mentioned",
    "unable to determine", "no relevant", "cannot be determined",
    "isn't mentioned", "not available", "don't have enough",
    "do not have enough", "no memory", "no record",
]


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an|is|was|were|are|am)\b", " ", s)
    for word, digit in NUMBER_WORDS.items():
        s = re.sub(rf"\b{word}\b", digit, s)
    return re.sub(r"\s+", " ", s).strip()


def f1_token_overlap(hypothesis: str, reference: str) -> float:
    h_tokens = normalize_text(hypothesis).split()
    r_tokens = normalize_text(reference).split()
    if not h_tokens or not r_tokens:
        return 0.0
    common = sum((Counter(h_tokens) & Counter(r_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(h_tokens)
    recall = common / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def list_item_overlap_score(hypothesis: str, reference: str) -> float | None:
    """Partial-credit score for LongMemEval references that are themselves a
    LIST of items (e.g. "Paris, London, Berlin") — the fraction of gold
    items individually found in the hypothesis.

    Bug B (list-answer partial credit, W0.2): f1_token_overlap() treats the
    reference as one bag of tokens, so a multi-word item that's entirely
    missing loses proportionally more weight than a missing single-word
    item. This gives each ITEM equal weight instead.

    Returns None when `reference` doesn't look like a list (fewer than 2
    comma-separated items) — callers fall back to existing single-answer
    scoring unchanged. Item match is substring-on-normalized-text OR
    per-item F1 >= 0.5 (mirrors f1_token_overlap's own threshold, applied
    per-item instead of over the whole joined string).
    """
    items = [i.strip() for i in reference.split(",") if i.strip()]
    if len(items) < 2:
        return None
    h_norm = normalize_text(hypothesis)
    matched = sum(
        1
        for item in items
        if normalize_text(item) in h_norm or f1_token_overlap(hypothesis, item) >= 0.5
    )
    return matched / len(items)


# Phrases must be normalized with the SAME function as the hypothesis, or the
# comparison is asymmetric and silently partial. `normalize_text` maps
# `[^\w\s]` to a space, so "i don't know" -> "i don t know": comparing the RAW
# phrase against the NORMALIZED hypothesis meant the three apostrophe-bearing
# entries — including "i don't know", the single most common way a model
# abstains — could NEVER match, and abstention was scored as missed.
#
# This bug was also MASKING the one directly above it: `generate_answer` used
# to return the literal "I don't know." on any LLM error, which would have been
# credited as a correct abstention had this comparison worked. Normalizing the
# phrase list WITHOUT also fixing that handler would have armed it. Both are
# fixed together, deliberately.
ABSTENTION_PHRASES_NORMALIZED = [normalize_text(p) for p in ABSTENTION_PHRASES]


def is_abstention_question(question_id: str) -> bool:
    return question_id.endswith("_abs")


def quick_score(hypothesis: str, reference: str, question_id: str) -> dict:
    """Local scoring: exact match, substring, F1 token overlap."""
    # An EMPTY hypothesis is a data-loss bug, never a scoreable answer: it is a
    # substring of everything, so it would score a false "substring match"
    # below AND a false abstention above. Refuse it rather than default.
    if not str(hypothesis).strip():
        raise ValueError(
            f"empty hypothesis for question {question_id!r} — refusing to score. "
            "The answerer produced nothing; record the question as UNSCORED."
        )

    if is_abstention_question(question_id):
        # Zero recall is a RETRIEVAL failure, not a demonstrated abstention:
        # the system never had content to reason over and correctly reject.
        # Crediting it would let a stalled retrieval path score a perfect
        # abstention number (LoCoMo W0.2, Bug A — same rule, same reason).
        if hypothesis == NO_RECALL_MARKER:
            return {
                "is_correct": False,
                "f1": 0.0,
                "explanation": "abstention credit withheld: recall returned zero "
                               "memories (retrieval failure, not a demonstrated abstention)",
            }
        h_lower = normalize_text(hypothesis)
        abstained = any(phrase in h_lower for phrase in ABSTENTION_PHRASES_NORMALIZED)
        return {"is_correct": abstained, "f1": 0.0, "explanation": f"abstention={'correct' if abstained else 'missed'}"}

    h_norm = normalize_text(hypothesis)
    r_norm = normalize_text(reference)

    # Exact match
    if h_norm == r_norm:
        return {"is_correct": True, "f1": 1.0, "explanation": "exact match"}

    # Substring match
    if r_norm in h_norm or h_norm in r_norm:
        return {"is_correct": True, "f1": 1.0, "explanation": "substring match"}

    # F1 token overlap
    f1 = f1_token_overlap(hypothesis, reference)

    # List-answer partial credit (Bug B, W0.2). Only ever RAISES f1 (never
    # lowers it), so a genuinely single-item reference (list_item_overlap_score
    # returns None, <2 comma items) is scored exactly as before this fix.
    list_score = list_item_overlap_score(hypothesis, reference)
    explanation = f"f1={f1:.3f}"
    if list_score is not None and list_score > f1:
        f1 = list_score
        explanation = f"list-item overlap={f1:.3f} (fraction of gold items matched)"

    is_correct = f1 >= 0.5
    return {"is_correct": is_correct, "f1": round(f1, 4), "explanation": explanation}


def llm_evaluate(
    openai_client: OpenAI,
    question: str,
    hypothesis: str,
    reference: str,
    question_id: str,
    model: str = "gpt-4o",
    is_preference: bool = False,
) -> dict:
    """GPT-4o binary judge matching LongMemEval paper methodology."""
    if is_abstention_question(question_id):
        h_lower = normalize_text(hypothesis)
        abstained = any(phrase in h_lower for phrase in ABSTENTION_PHRASES_NORMALIZED)
        return {"is_correct": abstained, "confidence": 0.9 if abstained else 0.1, "explanation": "abstention check"}

    if is_preference:
        prompt = f"""You are evaluating whether an AI assistant's answer aligns with a user's known preferences.

Question: {question}

User preference rubric: {reference}

Assistant's answer: {hypothesis}

Judge whether the assistant's answer is consistent with the user's known preferences described in the rubric. The answer is correct if it reflects the user's preferences (e.g., mentioning the right tools, topics, or approaches). It does NOT need to match the rubric text — it should demonstrate awareness of the user's preferences.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""
    else:
        prompt = f"""You are evaluating whether an AI assistant's answer to a question is correct.

Question: {question}

Reference answer: {reference}

Assistant's answer: {hypothesis}

Judge whether the assistant's answer is correct. The answer doesn't need to be word-for-word identical, but it should convey the same key information as the reference answer. Minor variations in phrasing, additional context, or slightly different formatting are acceptable as long as the core answer is correct.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences
        fence = re.match(r"^\s*```[a-zA-Z]*\s*\n(?P<body>.*)\n\s*```\s*$", content, re.S)
        if fence:
            content = fence.group("body").strip()
        result = json.loads(content)
        return {
            "is_correct": result.get("correct", False),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        # Do NOT silently fall back to quick_score and present the result as a
        # judge verdict. On an `_abs` question the hypothesis reaching this
        # point is often itself a failure artifact, so the fallback laundered
        # TWO failures into one "correct". The fallback verdict is still
        # computed (it is better than nothing) but is now explicitly LABELLED
        # as degraded, so a run can be audited for how many of its verdicts
        # were never actually judged.
        qs = quick_score(hypothesis, reference, question_id)
        return {
            "is_correct": qs["is_correct"],
            "confidence": 0.5,
            "explanation": f"LLM eval failed ({e}), quick score fallback",
            "judge_degraded": True,
        }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

QUESTION_TYPE_NAMES = {
    "single-session-user": "Single-Session (User)",
    "single-session-assistant": "Single-Session (Asst)",
    "single-session-preference": "Single-Session (Pref)",
    "multi-session": "Multi-Session",
    "knowledge-update": "Knowledge Update",
    "temporal-reasoning": "Temporal Reasoning",
}


def run_benchmark(config: Config) -> dict:
    client = CodememClient(config.base_url)
    openai_client = OpenAI()

    # Health check
    if config.mode != "baseline" and not client.health():
        print("ERROR: kremory-http server not reachable at", config.base_url, file=sys.stderr)
        print(
            "Start it with: KREMORY_MCP_DB_PATH=./bench.db cargo run -p kremory-mcp --bin kremory-http",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load dataset
    if not config.dataset_path.exists():
        print(f"ERROR: Dataset not found at {config.dataset_path}", file=sys.stderr)
        print("Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(config.dataset_path)
    print(f"Loaded {len(dataset)} questions")

    if config.max_questions > 0:
        dataset = dataset[:config.max_questions]
        print(f"Limited to {len(dataset)} questions")

    all_results = []
    type_stats: dict[str, dict] = {}
    # Questions retrieved but NOT scored (the answerer failed) — kept OUT of
    # the accuracy denominator and reported separately, so an outage shows up
    # as missing measurement rather than as a score.
    unscored_stats: dict[str, int] = {}
    answer_failures = 0
    q_done = 0

    # W0.3: crash-safe per-question write. The 500-question LongMemEval run
    # is paid (OpenAI generation + judge calls per question) — previously
    # this harness only wrote the full batch via json.dump() at the very
    # end (see below), so a mid-run crash lost ALL progress + spend. Mirror
    # LoCoMo's incremental jsonl write (bench/locomo/harness.py) — one
    # flushed line per question, written as each result is produced.
    import datetime
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    jsonl_path = (config.output.parent if config.output else Path("results")) / f"run-{run_ts}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_f = open(jsonl_path, "a")
    print(f"  [o11y] per-question records -> {jsonl_path}", file=sys.stderr)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question_id = item["question_id"]
        question = item["question"]
        reference = item.get("answer", "")
        question_type = item.get("question_type", "unknown")
        question_date = item.get("question_date", "")
        namespace = f"longmemeval-{question_id}"

        # 1. Ingest
        if config.mode != "baseline" and not config.skip_ingest:
            client.delete_namespace(namespace)
            time.sleep(0.3)
            try:
                stored = ingest_sessions(client, namespace, item)
            except KremoryStalled as e:
                # Fail LOUD and stop. A stalled ingest means every subsequent
                # question would score against an empty namespace — i.e. the
                # run would complete and report a number that measures nothing.
                # This question is a ~40-session ingest against a server doing
                # ~20 sequential LLM calls per store, so a stall is a real
                # operational risk, not a theoretical one.
                jsonl_f.flush()
                print(f"\n[FAIL-LOUD] kremory ingest stalled on {question_id}: {e}\n"
                      f"  Aborting rather than scoring against an empty namespace.\n"
                      f"  Records so far: {jsonl_path}", file=sys.stderr)
                sys.exit(5)
            # Build graph edges between related sessions
            if config.mode == "codemem-graph":
                client.consolidate("creative", namespace)
            time.sleep(0.5)
        else:
            stored = 0

        # An ingest that stored NOTHING is a retrieval-failure run, not a
        # zero-score run. Surface it immediately instead of letting every
        # question score False against an empty namespace.
        if config.mode != "baseline" and not config.skip_ingest and stored == 0:
            print(f"  [warn] {question_id}: ingest stored 0 of "
                  f"{len(item.get('haystack_sessions', []))} sessions", file=sys.stderr)

        # 2. Recall — use higher limits for multi-session and temporal questions
        recall_limit = config.recall_limit
        if question_type in ("multi-session", "knowledge-update", "temporal-reasoning"):
            recall_limit = max(recall_limit, 15)

        if config.mode == "baseline":
            memories = recall_baseline(item)
        elif config.mode == "codemem-graph":
            memories = recall_codemem_graph(
                client, question, namespace, recall_limit, config.graph_depth,
            )
        else:
            memories = recall_codemem(client, question, namespace, recall_limit)

        # 3. Generate answer. A failure here is UNSCORED, never a verdict —
        # the answerer producing nothing is an absence of measurement.
        try:
            hypothesis = generate_answer(
                openai_client, question, memories, question_date, config.llm_model,
            )
            answer_error = None
        except AnswerGenerationFailed as e:
            hypothesis, answer_error = None, str(e)
            answer_failures += 1
            print(f"  [UNSCORED] {question_id}: {e}", file=sys.stderr)

        # 4. Score — preference questions always use LLM eval since their
        # references are qualitative rubrics, not factual answers.
        if answer_error is not None:
            score_result = {
                "is_correct": None,
                "confidence": None,
                "explanation": f"not scored: {answer_error}",
            }
        else:
            use_llm = config.use_llm_eval or question_type == "single-session-preference"
            if use_llm:
                score_result = llm_evaluate(
                    openai_client, question, hypothesis, reference, question_id, config.eval_model,
                    is_preference=(question_type == "single-session-preference"),
                )
            else:
                score_result = quick_score(hypothesis, reference, question_id)

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "reference": reference,
            "hypothesis": hypothesis,
            "is_correct": score_result["is_correct"],
            "confidence": score_result.get("confidence", score_result.get("f1", 0)),
            "explanation": score_result["explanation"],
            "judge_degraded": score_result.get("judge_degraded", False),
            "answer_error": answer_error,
            "memories_recalled": len(memories),
            "memories_stored": stored,
            "mode": config.mode,
        }
        all_results.append(result)

        # --- o11y: per-question structured record, written incrementally
        # (crash-safe) — W0.3. Same record shape as the final results.json
        # entry, so a crash mid-run still leaves every question scored so
        # far inspectable/resumable without re-spending on OpenAI calls.
        jsonl_f.write(json.dumps(result) + "\n")
        jsonl_f.flush()

        # Track per-type. An UNSCORED question stays out of the denominator —
        # the failure this guards against is precisely a category that inflates
        # the total while measuring nothing.
        if score_result["is_correct"] is None:
            unscored_stats[question_type] = unscored_stats.get(question_type, 0) + 1
        else:
            if question_type not in type_stats:
                type_stats[question_type] = {"correct": 0, "total": 0}
            type_stats[question_type]["total"] += 1
            if score_result["is_correct"]:
                type_stats[question_type]["correct"] += 1

        # --- fail-loud: don't turn an outage into a score (mirrors LoCoMo) ---
        q_done += 1
        if q_done == CIRCUIT_MIN:
            rate = answer_failures / q_done
            if rate > CIRCUIT_FAILURE_RATE:
                jsonl_f.flush()
                print(f"\n[FAIL-LOUD] SYSTEMIC ANSWERER FAILURE after {q_done} questions: "
                      f"{answer_failures}/{q_done} answers failed ({rate:.0%}). "
                      f"Aborting before spending the rest of the run. "
                      f"Records: {jsonl_path}", file=sys.stderr)
                sys.exit(2)

        # 5. Cleanup (per-question, like LongMemEval expects).
        #
        # `--keep-corpus` suppresses this so the ingested haystack SURVIVES the
        # run and a later `--skip-ingest` pass can re-score against it.
        # Without it, `--skip-ingest` is unusable by construction: a normal run
        # deletes every namespace it created, so there is never a corpus for it
        # to point at, and each re-score pays a full 25,112-session re-ingest
        # (~61M tokens through the local extractor). That is the expensive
        # resource in this benchmark — far more than the OpenAI spend.
        # Mandated by W0.4 ("ingest-once/--skip-ingest across bench/*/harness.py").
        if config.mode != "baseline" and not config.skip_ingest and not config.keep_corpus:
            client.delete_namespace(namespace)

    jsonl_f.close()

    # Summary
    total_correct = sum(v["correct"] for v in type_stats.values())
    total_questions = sum(v["total"] for v in type_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"LongMemEval Results — Mode: {config.mode}")
    print(f"{'='*60}")
    print(f"\n{'Type':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        name = QUESTION_TYPE_NAMES.get(qt, qt)
        print(f"{name:<30} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")
    for qt in sorted(unscored_stats.keys()):
        name = QUESTION_TYPE_NAMES.get(qt, qt)
        print(f"{name:<30} {'—':>8} {unscored_stats[qt]:>8} {'UNSCORED':>10}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL':<30} {total_correct:>8} {total_questions:>8} {overall:>9.1f}%")

    n_degraded = sum(1 for r in all_results if r.get("judge_degraded"))
    if answer_failures or n_degraded:
        print(f"\n⚠ INTEGRITY")
        if answer_failures:
            print(f"  {answer_failures} question(s) UNSCORED — the answerer failed and was")
            print(f"    excluded from the denominator above rather than credited. Before")
            print(f"    2026-07-28 these returned the literal \"I don't know.\", which")
            print(f"    ABSTENTION_PHRASES matches, so every failure scored CORRECT on an")
            print(f"    `_abs` question — an outage rendered as a perfect abstention score.")
        if n_degraded:
            print(f"  {n_degraded} verdict(s) DEGRADED — the judge call failed and the local")
            print(f"    quick_score stood in. These are NOT judge verdicts; treat the")
            print(f"    headline as provisional until they are re-judged.")

    print(f"\n--- Landscape ---")
    print(f"  Oracle gpt-4o:     82.4%")
    print(f"  Zep:               71.2%")
    print(f"  Naive RAG:         52.0%")
    print(f"  Best Guess:        18.8%")

    output = {
        "mode": config.mode,
        "metric": "longmemeval",
        "llm_model": config.llm_model,
        "recall_limit": config.recall_limit,
        "total_questions": len(all_results),
        "type_stats": type_stats,
        "unscored_stats": unscored_stats,
        "answer_failures": answer_failures,
        "judge_degraded_count": sum(1 for r in all_results if r.get("judge_degraded")),
        "scored_denominator": total_questions,
        "overall_accuracy": round(overall, 2),
        "results": all_results,
    }

    # Save
    out_path = config.output
    if not out_path:
        out_path = Path(__file__).parent / "results" / f"{config.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


# ---------------------------------------------------------------------------
# Rescore existing results
# ---------------------------------------------------------------------------

def rescore_results(results_path: Path, args):
    """Re-score an existing results file with different scoring settings.

    Useful for re-evaluating with LLM judge or preference-aware scoring
    without re-running the full ingest/recall/generation pipeline.
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    type_stats: dict[str, dict] = {}
    new_results = []

    for r in tqdm(results, desc="Re-scoring"):
        question_type = r["question_type"]
        hypothesis = r["hypothesis"]
        reference = r["reference"]
        question_id = r["question_id"]
        question = r["question"]

        use_llm = args.llm_eval or question_type == "single-session-preference"
        if use_llm:
            score_result = llm_evaluate(
                openai_client, question, hypothesis, reference, question_id,
                args.eval_model,
                is_preference=(question_type == "single-session-preference"),
            )
        else:
            score_result = quick_score(hypothesis, reference, question_id)

        r["is_correct"] = score_result["is_correct"]
        r["confidence"] = score_result.get("confidence", score_result.get("f1", 0))
        r["explanation"] = score_result["explanation"]
        new_results.append(r)

        if question_type not in type_stats:
            type_stats[question_type] = {"correct": 0, "total": 0}
        type_stats[question_type]["total"] += 1
        if score_result["is_correct"]:
            type_stats[question_type]["correct"] += 1

    total_correct = sum(v["correct"] for v in type_stats.values())
    total_questions = sum(v["total"] for v in type_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"Re-scored: {results_path.name}")
    print(f"{'='*60}")
    print(f"\n{'Type':<35s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10}")

    for qt in sorted(type_stats.keys()):
        name = QUESTION_TYPE_NAMES.get(qt, qt)
        s = type_stats[qt]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"{name:<35s} {s['correct']:>8d} {s['total']:>8d} {acc:>9.1f}%")

    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL':<35s} {total_correct:>8d} {total_questions:>8d} {overall:>9.1f}%")

    # Save rescored results
    data["results"] = new_results
    data["type_stats"] = type_stats
    data["overall_accuracy"] = overall
    out_path = args.output or results_path.with_suffix(".rescored.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nRe-scored results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # W0.3: unbuffered stdout so long runs stream progress live instead of
    # buffering until process exit/flush. In-script so it's robust
    # regardless of invocation (no reliance on `python3 -u` / PYTHONUNBUFFERED=1).
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="LongMemEval benchmark harness for codemem")
    parser.add_argument("--mode", default="codemem",
                        choices=["baseline", "codemem", "codemem-graph"],
                        help="Recall strategy to benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--base-url", default=CODEMEM_BASE,
                        help="Codemem API base URL")
    parser.add_argument("--recall-limit", type=int, default=10,
                        help="Number of memories to recall per question")
    parser.add_argument("--graph-depth", type=int, default=2,
                        help="Graph expansion depth for codemem-graph mode")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="LLM model for answer generation")
    parser.add_argument("--eval-model", default="gpt-4o",
                        help="LLM model for evaluation judge")
    parser.add_argument("--llm-eval", action="store_true",
                        help="Use GPT-4o judge instead of quick_score")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion, reuse memories from a prior --keep-corpus run")
    parser.add_argument("--keep-corpus", action="store_true",
                        help="Do NOT delete each namespace after scoring, so the ingested "
                             "haystack survives and --skip-ingest can re-score against it. "
                             "Use this on the FIRST (expensive) ingest pass.")
    parser.add_argument("--output", type=Path,
                        help="Output file path for results JSON")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSON",
                        help="Re-score existing results file (skip ingest/recall/generation)")
    args = parser.parse_args()

    if args.rescore:
        rescore_results(args.rescore, args)
    else:
        config = Config(
            base_url=args.base_url,
            dataset_path=args.dataset,
            recall_limit=args.recall_limit,
            graph_depth=args.graph_depth,
            mode=args.mode,
            llm_model=args.llm_model,
            eval_model=args.eval_model,
            use_llm_eval=args.llm_eval,
            max_questions=args.max_questions,
            skip_ingest=args.skip_ingest,
            keep_corpus=args.keep_corpus,
            output=args.output,
        )
        run_benchmark(config)


if __name__ == "__main__":
    main()

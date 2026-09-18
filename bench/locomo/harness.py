#!/usr/bin/env python3
# NOTICE: Adapted from cogniplex/codemem (Apache-2.0) for kremory.
# Original: https://github.com/cogniplex/codemem/tree/main/bench/locomo
# Change: CodememClient retargeted at kremory-http's bare (no /api prefix)
# REST surface — POST /memories, GET /search, DELETE /namespaces/{ns},
# POST /consolidation/{cycle}?namespace= (namespace REQUIRED — kremory's
# dream() is always namespace-scoped, unlike codemem's global consolidation).
# start_session/end_session/get_namespaces are no-ops/stubs (kremory has no
# session or namespace-listing REST tools); graph_neighbors/get_memory are
# stubbed (no graph-traversal REST tool yet).
"""LoCoMo benchmark harness for codemem.

Ingests LoCoMo conversations into codemem, runs recall for each question,
and collects answers for scoring.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

# Single source of truth for which categories no presence-style scorer may
# score. Defined in judge_rescore.py and already imported by qa_eval.py; the
# harness imports the SAME constant rather than declaring a second one, so the
# substring scorer and the LLM-judge scorer cannot drift onto different
# denominators (they previously did: n=1986 here vs n=1540 there).
from judge_rescore import ABSTENTION_CATEGORIES

# recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1]: provenance stamp helper.
# Sibling module so both the harness (build) and the tally/gate (assert) share it.
try:
    from provenance import build_provenance
except ImportError:  # invoked as a package / from another CWD
    from bench.locomo.provenance import build_provenance

CODEMEM_BASE = "http://localhost:3179"  # kremory-http: bare routes, NO /api prefix
DEFAULT_DATASET = Path(__file__).parent / "data" / "locomo10.json"
NAMESPACE_PREFIX = "locomo-bench"

# fail-fast-and-loud, but SLOW != STUCK. `INGEST_BUDGET_S` is now a GENEROUS
# TOTAL per-conversation BACKSTOP (default 2h), not a tight budget — a genuine
# hang is caught per-chunk (see `stall_s` in `ingest_conversation` + the per-POST
# httpx timeout), so this only stops a conversation that is PROGRESSING but too
# slow to justify including in a multi-conversation run. A slow-but-progressing
# conversation (kremory's remember() fans out to ~20 sequential LLM calls per
# chunk — minutes per conversation) now RUNS TO COMPLETION instead of aborting.
# Override via KREMORY_INGEST_BUDGET_S; tighten the hang guard via KREMORY_INGEST_STALL_S.
import os as _os

# INGEST_BUDGET_S and KremoryStalled are imported from the shared client below,
# NOT redefined here. Defining a local `class KremoryStalled` while the shared
# client raises its own would produce two DISTINCT classes, so this module's
# `except KremoryStalled` would silently stop catching a stalled ingest — the
# fail-fast path would go dead while still looking present in the source.


@dataclass
class Config:
    base_url: str = CODEMEM_BASE
    dataset_path: Path = DEFAULT_DATASET
    recall_limit: int = 10
    recall_limit_temporal: int = 10
    recall_limit_multihop: int = 10
    graph_depth: int = 2
    mode: str = "codemem"  # baseline | rag | codemem | codemem-graph
    conversations: list[int] = field(default_factory=list)
    skip_ingest: bool = False
    output: Path | None = None
    # DREAM KEEP-OR-CUT (2026-08-19). When True, `client.consolidate(...)` is NOT
    # called after ingest, so the dream phase never runs. This is the ONLY
    # difference between the two arms of the pre-registered A/B
    # (`.ai-docs/decisions/dream-keep-or-cut-prereg-2026-08-19.md`).
    #
    # Deliberately a HARNESS flag, not a new library knob: the switch already
    # existed as "do not call the endpoint", so per contract-first there is
    # nothing to add to kremory's public surface. It also means the measured
    # dream-on arm is byte-for-byte the shipped behaviour, not a special build.
    no_dream: bool = False
    # TD-223/TD-224. Path to the SQLite graph this run wrote, so the report can
    # state whether consolidation merged entities out of existence. Defaults to
    # KREMORY_MCP_DB_PATH (what the server was started with). When unknown the
    # check reports "skipped" — never a pass.
    graph_db: str | None = None
    # kremory-http's GET /search accepts an optional ?mode=recall|content|hybrid
    # (R-lane, commit d6ccd56) selecting which retrieval path the server uses.
    # DISTINCT from `mode` above (baseline/rag/codemem/codemem-graph — the
    # harness's OWN recall-strategy selector). Default "recall" preserves the
    # pre-existing wire shape byte-for-byte (see CodememClient.recall()).
    server_mode: str = "recall"
    # Workstream A: scorer selector. "substring" is the strict word-overlap
    # matcher (check_answer_in_memories) — default, kept for reproducibility.
    # "llm-judge" is an OFFLINE cross-family LLM-as-judge pass (judge_rescore.py)
    # that credits semantic matches the substring matcher misses; the harness
    # still runs substring inline and always persists `recalled_memories`, so
    # both scorers report over the same run. Recorded in output metadata only.
    scorer: str = "substring"
    # OPT-IN, default OFF (RECALL-LEDGER §4.19 / TD-155 / TD-196). When True,
    # the harness issues a SECOND `GET /search` per question — `format=text&
    # template=temporal_facts`, kremory's own prompt-ready rendering, WITH
    # dates — alongside the existing `recall()` call, which is completely
    # unchanged. Persisted as `recalled_text_block`; `recalled_memories` /
    # `recalled_memory_provenance` / the inline substring score are byte-
    # identical whether this is on or off. Default OFF because the second
    # call is NOT retrieval-equivalent to the primary one under
    # `--server-mode hybrid` (TD-196) — enabling it there measures two
    # different retrievals, not one retrieval rendered two ways.
    capture_text_block: bool = False


# ---------------------------------------------------------------------------
# kremory HTTP client — SHARED, not inline
# ---------------------------------------------------------------------------
#
# CodememClient moved to bench/common/kremory_client.py on 2026-07-28, VERBATIM.
# This file was the ONLY copy that had ever run against kremory, and
# bench/longmemeval carried a divergent fork of it missing `store_timeout`, the
# POST /memories try/except, `total_http_errors` and `scrape_metrics` — a fork
# that could not have completed a single run. Sharing one implementation is
# what stops that recurring; leaving a second copy here would have re-created
# the very drift the extraction exists to remove.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from kremory_client import (  # noqa: E402
    INGEST_BUDGET_S,
    CodememClient,
    KremoryStalled,
)

# ---------------------------------------------------------------------------
# Answer checking (AutoMem-style: check if gold answer is in recalled memories)
# ---------------------------------------------------------------------------

import re
import string


def normalize_text(s: str) -> str:
    """Normalize text for matching: lowercase, remove punctuation, basic stemming."""
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Basic stemming
    for suffix in ["ing", "tion", "ment", "ness", "ed", "ly", "er", "est", "ies"]:
        if len(s) > len(suffix) + 3:
            s = re.sub(rf"\b(\w+){suffix}\b", r"\1", s)
    return " ".join(s.split())


def word_overlap_score(expected: str, text: str) -> float:
    """Compute word overlap between expected answer and text."""
    expected_words = set(normalize_text(expected).split())
    text_words = set(normalize_text(text).split())
    if not expected_words:
        return 0.0
    overlap = expected_words & text_words
    return len(overlap) / len(expected_words)


def list_item_overlap_score(expected: str, text: str) -> float | None:
    """Partial-credit score for gold answers that are themselves a LIST of
    items (e.g. "pottery, camping, painting, swimming") — the fraction of
    gold items individually found in the combined recalled text.

    Bug B (list-answer partial credit, W0.2): word_overlap_score() weighs
    each gold WORD equally regardless of which item it belongs to, so a
    missing multi-word item (e.g. "classic children's books") can tank the
    score disproportionately while a missing single-word item barely moves
    it. This gives each ITEM equal weight instead, which is the metric the
    W0.2 fix asks for.

    Returns None when `expected` doesn't look like a list (fewer than 2
    comma-separated items) — callers fall back to existing single-answer
    scoring unchanged. Item match is substring OR >=0.5 word-overlap on
    that single item (mirrors the existing word_overlap_score semantics,
    just applied per-item instead of over the whole joined string).
    """
    items = [i.strip() for i in expected.split(",") if i.strip()]
    if len(items) < 2:
        return None
    text_lower = text.lower()
    matched = sum(
        1
        for item in items
        if item.lower() in text_lower or word_overlap_score(item, text) >= 0.5
    )
    return matched / len(items)


def check_answer_in_memories(
    expected_answer: str,
    memories: list[str],
    category: str,
) -> tuple[bool, float, str]:
    """Check if the expected answer can be found in recalled memories.

    Returns (is_correct, confidence, explanation).
    This matches AutoMem's evaluation approach.
    """
    # An abstention category is NOT SCOREABLE by a retrieval-presence metric,
    # and this refusal is structural rather than a caller convention (an
    # `if` at one of N call sites is a check you can forget at the N+1th).
    #
    # Every LoCoMo category-5 question is a SPEAKER-ATTRIBUTION FALSE PREMISE:
    # "What did Caroline realize after HER charity race?" — the race was
    # MELANIE's. All 446 carry real `evidence`, and `adversarial_answer` is a
    # true fact re-attributed to the wrong person. So a good retriever SHOULD
    # surface that turn; detecting the false premise is an ANSWERER judgement
    # made over retrieved text, not a property of retrieval.
    #
    # A presence metric is therefore BLIND to abstention, and every number it
    # can emit here is an artefact of which branch fires, not a measurement:
    # steps 1-2 below return True on ANY match (so finding the distractor
    # "passes"), while the old step-3 block returned False in the 0.2-0.5 band
    # and True below 0.2 — a non-monotonic step function where high overlap and
    # low overlap both score correct and only the middle fails. On the
    # 2026-07-28 out-of-the-box run those three readings gave 100%, 98.4% and
    # 0.9% over the SAME retrievals. `qa_eval.py` already excludes this
    # category via `ABSTENTION_CATEGORIES`; the substring scorer now agrees, so
    # both scorers share the n=1540 denominator and are directly comparable.
    # Recall still runs and `recalled_memories` is still persisted, so a real
    # abstention judge can score these offline without re-running retrieval.
    if category in ABSTENTION_CATEGORIES:
        raise ValueError(
            f"category {category!r} is an abstention category and cannot be scored "
            "by this retrieval-presence scorer — abstention is an answerer property. "
            "Exclude it from the scored denominator (see ABSTENTION_CATEGORIES)."
        )

    expected = str(expected_answer)

    # An EMPTY gold is a data-loss bug in the caller, never a scoreable input:
    # `"" in anything` is True, so step 1 below would short-circuit EVERY such
    # question to "correct, exact substring match" and the category branches
    # further down would never run. This is exactly the defect that forced
    # AutoMem to retract their published LoCoMo figure (see the note by the
    # baselines block below) — and this harness shipped it too, scoring 446/446
    # adversarial because category-5 gold lives under `adversarial_answer`,
    # not `answer`. Fail LOUDLY rather than default: a missing gold must abort
    # the run, not silently inflate it.
    if not expected.strip():
        raise ValueError(
            f"empty gold answer for a {category!r} question — refusing to score. "
            "The caller lost the gold field (LoCoMo category-5 gold is under "
            "`adversarial_answer`, not `answer`); fix extraction, do not default."
        )

    combined_text = "\n".join(memories)

    # 1. Exact substring match (case-insensitive)
    if expected.lower() in combined_text.lower():
        return True, 1.0, "exact substring match"

    # 2. Word overlap across all memories
    best_score = 0.0
    best_memory_idx = -1
    for i, mem in enumerate(memories):
        score = word_overlap_score(expected, mem)
        if score > best_score:
            best_score = score
            best_memory_idx = i

    # For multi-hop, also check combined text with lower threshold
    combined_score = word_overlap_score(expected, combined_text)

    if category == "multi-hop":
        threshold = 0.35
        score = max(best_score, combined_score)
    else:
        threshold = 0.50
        score = best_score

    # 2b. List-answer partial credit (Bug B, W0.2). Gated to categories where
    # a comma really does mean "multiple distinct items": single-hop,
    # multi-hop, open-domain. Excluded on purpose:
    #   - "adversarial": expects near-zero overlap by design (list credit
    #     here would defeat the abstention check below).
    #   - "temporal": commas there are date-formatting artifacts
    #     ("19 January, 2023" is ONE date, not a 2-item list) — already
    #     handled by the dedicated fuzzy-date block further down.
    # Only ever RAISES the score (never lowers it), so a genuinely
    # single-item answer (list_item_overlap_score returns None) is scored
    # exactly as before this fix.
    used_list_credit = False
    if category not in ("adversarial", "temporal"):
        list_score = list_item_overlap_score(expected, combined_text)
        if list_score is not None and list_score > score:
            score = list_score
            used_list_credit = True

    if score >= threshold:
        if used_list_credit:
            return True, score, f"list-item overlap {score:.2f} (fraction of gold items matched)"
        return True, score, f"word overlap {score:.2f} in memory {best_memory_idx}"

    # 3. (was: adversarial abstention credit — DELETED 2026-07-28.)
    #
    # This block tried to infer abstention from low overlap, but was
    # unreachable above the 0.5 threshold, so the composite behaviour was
    # non-monotonic and unusable (see the refusal at the top of this
    # function). Patching its thresholds would have produced a fourth
    # arbitrary number rather than a measurement, so the category is now
    # refused outright instead. Its one sound instinct — never credit an
    # abstention that a ZERO-memory retrieval failure produced — is preserved
    # by the `recall_empty` circuit-breaker, which aborts a run whose empty
    # rate crosses CIRCUIT_EMPTY_RATE.

    # 4. Fuzzy date matching for temporal questions
    if category == "temporal":
        try:
            from dateutil import parser as date_parser
            expected_dates = []
            for word in expected.split():
                try:
                    expected_dates.append(date_parser.parse(word, fuzzy=True))
                except (ValueError, OverflowError):
                    pass
            if not expected_dates:
                try:
                    expected_dates.append(date_parser.parse(expected, fuzzy=True))
                except (ValueError, OverflowError):
                    pass

            if expected_dates:
                for mem in memories:
                    for ed in expected_dates:
                        if ed.strftime("%d") in mem and (
                            ed.strftime("%B") in mem or ed.strftime("%b") in mem
                        ):
                            return True, 0.95, "fuzzy date match"
        except ImportError:
            pass

    return False, score, f"best word overlap {score:.2f} below threshold {threshold}"


def build_headline(scored_correct: int, scored_total: int,
                   unscored_stats: dict[str, int]) -> str:
    """The single quotable line for a run — the percentage AND its denominator.

    TD-217(b). A run's headline is the thing that gets pasted into a doc, a
    commit message, or a comparison table, and by the time it lands there the
    `unscored_stats` JSON key is nowhere near it. `98.0%` next to a competitor's
    number computed over ALL questions is a category error that nobody can see.

    So the qualifier is built into the string rather than printed beside it:
    a caller cannot quote the number without quoting what it excludes.
    Pure + module-level so it is directly testable (see test_locomo_scorer.py).
    """
    pct = scored_correct / scored_total * 100 if scored_total else 0.0
    base = f"{scored_correct}/{scored_total} = {pct:.1f}%"
    if not unscored_stats:
        return base
    n_unscored = sum(unscored_stats.values())
    parts = ", ".join(f"{n} {cat}" for cat, n in sorted(unscored_stats.items()))
    return (f"{base} ({parts} NOT SCORED for correctness; "
            f"{scored_total + n_unscored} questions asked — their RETRIEVAL is "
            f"scored by evidence_eval.py)")


# ---------------------------------------------------------------------------
# Graph integrity (TD-223 / TD-224 / TD-256)
# ---------------------------------------------------------------------------

#: Distinct losers absorbed by one keeper before it counts as a fan-in.
#:
#: 2 is deliberate and is the lowest value that can detect anything. It is also
#: the value at which a legitimate two-variant canonicalisation is flagged — so
#: fan-ins are REPORTED, never used to fail a run. Raising this to 3 would drop
#: the corpus signal from 20 findings to 6 and hide every 2-way date collapse.
FANIN_MIN_LOSERS = 2


def check_graph_integrity(db_path: str | None) -> dict:
    """Did this run's consolidation merge entities out of existence?

    TD-223. A run once scored 149/152 = 98.0% on a graph in which BOTH speakers
    of the dialogue had been merged away (``melanie`` -> ``caroline`` ->
    ``loved ones`` -> ``luna and oliver``), identically to the graph where all
    three survived — and the rank-aware metric slightly FAVOURED the destroyed
    one. No scorer at any tier could see it; it was found by reading
    ``graph_mutation_log`` by hand.

    The FIRST signal is a TRANSITIVE MERGE CHAIN: an entity that is the SURVIVOR of
    one live merge and the VICTIM of another. ``A -> B`` then ``B -> C`` moved A's
    identity two hops while **nobody ever adjudicated A against C**.

    The SECOND signal is a FAN-IN, added 2026-09-18 (TD-256), and on real data it
    is the dominant one. ``B -> A``, ``C -> A``, ``D -> A`` forms no chain — A was
    never absorbed — so the chain test above calls it clean. Measured on
    ``.context/full-corpus.db``: **4 chained entities, 20 fan-ins**, 19 of which
    collapse distinct calendar dates ('3 july 2023' swallowed 9 other July dates).
    A gate that saw only chains was blind to 19 of the 23 findings on the very
    database the release floor was computed from.

    Fan-ins are REPORTED, never used to fail a run: at the only threshold that can
    detect anything (2 losers) an ordinary two-variant canonicalisation is
    indistinguishable from damage without reading the entity names. Judging that
    is a human's job; surfacing it is this function's.

    Measured on the two retained databases (one healthy, one destroyed): 6 chained
    entities vs 0. This mirrors invariant 6 in
    ``crates/kremory-eval/src/layer_b/graph_integrity.rs``, which is the reference
    implementation; both read the same contract, that an ``entity_merge`` row's
    ``inputs`` carries ``keeper`` and ``loser``.

    Known false-positive class: a legitimate three-variant canonicalisation forms
    a chain too (``pottery class`` -> ``pottery`` -> ``pottery project``). On the
    destroyed database 5 of 6 flags were real damage and 1 was benign. Reported,
    never used to fail the run.

    Returns ``status="skipped"`` when no database is known — NEVER a pass. An
    absent check that reports success is the defect this exists to prevent.
    """
    if not db_path:
        return {
            "status": "skipped",
            "reason": "no --graph-db and no KREMORY_MCP_DB_PATH; "
                      "cannot verify the graph survived consolidation",
        }
    if not Path(db_path).exists():
        return {"status": "skipped", "reason": f"graph db not found: {db_path}"}

    import sqlite3
    # Read-only URI: the server may hold this file open, and a benchmark report
    # must never mutate the artefact it is describing.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, inputs FROM graph_mutation_log "
            "WHERE kind = 'entity_merge' AND undone_at IS NULL ORDER BY id"
        ).fetchall()
    except sqlite3.Error as e:
        return {"status": "error", "reason": f"reading graph_mutation_log: {e}"}
    finally:
        conn.close()

    survivors: set[str] = set()
    victims: set[str] = set()
    absorbed: dict[str, list[str]] = {}
    absorbed_by: dict[str, str] = {}

    for row_id, inputs in rows:
        try:
            d = json.loads(inputs)
            keeper, loser = d["keeper"], d["loser"]
        except (ValueError, KeyError, TypeError) as e:
            # PARSE LOUDLY. A silently skipped row makes a corrupted graph look
            # clean, which is exactly the failure mode this check exists for.
            return {
                "status": "error",
                "reason": f"graph_mutation_log row {row_id}: entity_merge inputs "
                          f"carry no keeper/loser — producer shape drift, "
                          f"integrity cannot be evaluated ({e})",
            }
        survivors.add(keeper)
        victims.add(loser)
        absorbed.setdefault(keeper, []).append(loser)
        absorbed_by[loser] = keeper

    chained = sorted(survivors & victims)

    # FAN-IN (TD-256). The chain signal above fires only when one entity is BOTH a
    # survivor and a victim. The dominant real damage shape makes no chain at all:
    # N entities absorbed into ONE keeper that is never itself absorbed, so
    # `survivors & victims` is empty and the graph reports clean.
    #
    # Measured on `.context/full-corpus.db` (2026-09-18, 113 live merges):
    # 4 chained entities — and 20 fan-ins, 19 of which collapse DISTINCT CALENDAR
    # DATES into one another ('3 july 2023' absorbed 9 other July dates). The
    # signal is 95% precise there: exactly one of the 20 ('minnesota wolves') is
    # not a date collapse.
    #
    # Threshold is 2 distinct losers, and losers are DE-DUPLICATED first: the log
    # records the same loser twice for two keepers ('1 february 2023',
    # '3 august 2023'), so counting rows rather than distinct entities would
    # inflate a 1-loser merge into a fan-in.
    fanins = {
        keeper: sorted(set(losers))
        for keeper, losers in absorbed.items()
        if len(set(losers)) >= FANIN_MIN_LOSERS
    }
    # Worst first — a 9-way collapse and a 2-way variant merge are not the same
    # finding, and a flat count hides which one you are looking at.
    worst = sorted(fanins.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    return {
        "status": "checked",
        "graph_db": db_path,
        "live_merges": len(rows),
        "chained_entities": len(chained),
        # NB: `clean` means "no transitive merge chain" — NOT "the graph is
        # healthy". It deliberately does NOT account for fan-ins: a two-variant
        # canonicalisation ('pottery class' + 'pottery' -> 'pottery project') is
        # ordinary housekeeping and is guarded as a false-positive by
        # test_star_merges_are_not_a_chain. Fan-ins are REPORTED for a human to
        # judge, never used to fail a run (plan 2026-09-18, risk R5).
        "clean": not chained,
        "chains": [
            f"'{e}' absorbed [{', '.join(absorbed.get(e, []))}] "
            f"then was absorbed by '{absorbed_by.get(e, '<unknown>')}'"
            for e in chained
        ],
        "fanin_entities": len(fanins),
        "worst_fanin": len(worst[0][1]) if worst else 0,
        "fanins": [
            f"'{keeper}' absorbed {len(losers)}: [{', '.join(losers)}]"
            for keeper, losers in worst
        ],
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some versions wrap in a top-level key
        for key in ("data", "samples", "conversations"):
            if key in data:
                return data[key]
        return [data]
    return data


def parse_session_datetime(raw: str) -> str | None:
    """Parse a LoCoMo session header date into RFC3339, or None if unparseable.

    LoCoMo stores it as e.g. `"1:56 pm on 8 May, 2023"`. `dateutil` handles every
    part of that EXCEPT the literal `" on "` separator, which it reads as a token
    and rejects — so strip it first.

    WHY THIS EXISTS (TD-187): this value is the episode's world-time ANCHOR, and
    until now the harness embedded it in the content STRING only and never sent it
    as `published_at`. That mattered because kremory's extractor is never told when
    an episode happened, so a turn saying "I went yesterday" was stored as an
    undated assertion — the date was simply not in the graph to retrieve. Four of
    the six questions that fail in ALL six conv0 runs are "when did X happen"
    (q_1, q_7, q_26, q_50).

    Returning None on failure is deliberate and load-bearing: kremory renders the
    anchor into the extraction prompt ONLY when the caller declared one, because a
    wall-clock fallback would change the VCR fingerprint on every run. A None here
    must therefore stay None all the way down — never substitute `now()`.

    The returned string MUST carry a UTC offset. kremory parses `published_at`
    with `DateTime::parse_from_rfc3339` (`crates/kremory-mcp/src/conversions.rs:150`),
    which REJECTS a naive timestamp — the accepted shape is pinned by the test at
    `conversions.rs:551` (`"2026-07-14T10:30:00Z"`). An earlier draft of this
    function returned `.isoformat()` on a naive datetime; every store would have
    been rejected. Caught by smoke-testing the parser against the real corpus
    before running the bench, not by review.

    ASSUMPTION (stated, not hidden): LoCoMo session headers carry no timezone, so
    they are interpreted as UTC. The corpus gives us nothing better, and internal
    consistency is what matters for resolving "yesterday" against the anchor.
    """
    if not raw or not raw.strip():
        return None
    try:
        from datetime import timezone

        from dateutil import parser as date_parser
        dt = date_parser.parse(raw.replace(" on ", " "))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (ValueError, OverflowError, TypeError):
        return None


def extract_sessions(conversation: dict) -> list[dict]:
    """Extract ordered sessions from a LoCoMo conversation.

    Returns list of {session_num, datetime, turns: [{speaker, text, dia_id}]}.
    """
    sessions = []
    conv_data = conversation.get("conversation", conversation)
    session_num = 1

    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conv_data:
            break
        turns = []
        for turn in conv_data[key]:
            text = turn.get("text", "")
            speaker = turn.get("speaker", "unknown")
            dia_id = turn.get("dia_id", "")
            if turn.get("blip_caption"):
                text += f" [Image: {turn['blip_caption']}]"
            turns.append({"speaker": speaker, "text": text, "dia_id": dia_id})
        sessions.append({
            "session_num": session_num,
            "datetime": conv_data.get(date_key, ""),
            "turns": turns,
        })
        session_num += 1

    return sessions


# LoCoMo category IDs → names
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def extract_questions(conversation: dict) -> list[dict]:
    """Extract QA pairs from a LoCoMo conversation.

    Returns list of {question_id, question, answer, category, evidence_ids}.

    `answer` is the string this run SCORES AGAINST, and its meaning is
    category-dependent:

      * categories 1-4 — the gold answer, from `answer`. Present on all 1540.
      * category 5 (adversarial) — the DISTRACTOR, from `adversarial_answer`:
        the plausible-but-wrong answer a naive system gives by retrieving the
        topically-adjacent turn (which does exist — all 446 carry `evidence`).
        It is carried here for offline abstention judging only:
        `check_answer_in_memories` REFUSES this category outright, because a
        retrieval-presence metric cannot see abstention.
        444 of 446 category-5 records have NO `answer` key at all; the 2 that
        carry both make the roles explicit (`answer: "No"` /
        `adversarial_answer: "Yes"` on a false-premise yes/no question).

    Reading `answer` unconditionally is what produced a false 446/446
    adversarial score: the `""` default fed an empty gold into a substring
    check that matches everything. Every question in locomo10 has a gold under
    exactly one of the two fields, so an empty result here is a bug — raise.
    """
    qa_list = conversation.get("qa", [])
    questions = []
    for i, qa in enumerate(qa_list):
        raw_cat = qa.get("category", 0)
        category = CATEGORY_NAMES.get(raw_cat, f"cat-{raw_cat}")
        gold = qa.get("adversarial_answer") if category == "adversarial" else qa.get("answer")
        if gold is None or not str(gold).strip():
            raise ValueError(
                f"question {i} (category {raw_cat}) has no usable gold: "
                f"answer={qa.get('answer')!r} "
                f"adversarial_answer={qa.get('adversarial_answer')!r}"
            )
        questions.append({
            "question_id": f"q_{i}",
            "question": qa.get("question", ""),
            "answer": gold,
            "category": category,
            "evidence": qa.get("evidence", []),
        })
    return questions


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_conversation(
    client: CodememClient,
    namespace: str,
    sessions: list[dict],
    sample_id: str,
    no_dream: bool = False,
) -> int:
    """Store conversation into codemem at turn-group granularity.

    Stores groups of ~4 consecutive turns per memory (not whole sessions).
    This gives retrieval enough context per chunk while keeping embeddings
    focused enough to match specific questions.
    """
    count = 0
    session_id = client.start_session(namespace)
    turns_per_chunk = 4
    ingest_start = time.monotonic()
    # fail-fast-and-loud, but distinguish STUCK from merely SLOW. The old
    # aggregate per-conversation budget killed a conversation that was
    # PROGRESSING (107 chunks ingested) just for being slow — wrong: slow-but-
    # progressing is not a hang. Guard on PER-CHUNK stall instead (resets each
    # chunk), with a generous TOTAL backstop so a pathologically-slow-but-
    # progressing conversation still bounds. (A genuinely hung chunk is also
    # caught upstream by the per-POST httpx timeout — this is belt-and-suspenders
    # + the backstop.)
    stall_s = float(_os.environ.get("KREMORY_INGEST_STALL_S", "180"))

    for sess in sessions:
        turns = sess["turns"]
        # TD-187: the session header carries the episode's world-time anchor. Send
        # it as `published_at` so kremory can ground relative time expressions
        # ("yesterday") at extraction. None when unparseable — and it must STAY
        # None rather than falling back to now(), see parse_session_datetime().
        sess_published_at = parse_session_datetime(sess.get("datetime", ""))
        # Split session into chunks of turns
        for chunk_start in range(0, len(turns), turns_per_chunk):
            chunk_turns = turns[chunk_start:chunk_start + turns_per_chunk]
            if not chunk_turns:
                continue

            lines = []
            for turn in chunk_turns:
                lines.append(f"{turn['speaker']}: {turn['text']}")

            chunk_content = (
                f"[Session {sess['session_num']}] [{sess['datetime']}]\n"
                + "\n".join(lines)
            )

            tags = [
                f"sample:{sample_id}",
                f"session:{sess['session_num']}",
            ]
            speakers = {t["speaker"] for t in chunk_turns}
            for sp in speakers:
                tags.append(f"speaker:{sp}")

            chunk_t0 = time.monotonic()
            # ADR-080: thread every chunk of this conversation under ONE
            # source id so the server can replay preceding turns into the
            # extraction prompt. Scoped to the CONVERSATION (not the session)
            # because LoCoMo questions span sessions, matching mem0's
            # session-scope shape.
            #
            # ⚠️ This CHANGES the ingest protocol relative to the published
            # 92.08% figure, which was measured with no source id at all (every
            # chunk got a server-side uuid). Any figure produced with this on
            # must say so. Set KREMORY_BENCH_NO_THREAD=1 to reproduce the old
            # protocol exactly.
            thread_id = None if _os.environ.get("KREMORY_BENCH_NO_THREAD") else f"locomo/{sample_id}"
            mid = client.store_memory(
                content=chunk_content,
                namespace=namespace,
                memory_type="Context",
                importance=0.5,
                tags=tags,
                published_at=sess_published_at,
                source_id=thread_id,
            )
            chunk_elapsed = time.monotonic() - chunk_t0
            if mid:
                count += 1

            # STALL guard (per-chunk, resets each chunk): a SINGLE chunk that
            # blows past the stall window is a genuine hang, not just slow — abort
            # loud. Slow-but-progressing ingest sails through (each chunk is well
            # under `stall_s`), which is the whole point.
            if chunk_elapsed > stall_s:
                raise KremoryStalled(
                    f"ingest of {sample_id} STALLED — one chunk took "
                    f"{chunk_elapsed:.0f}s > {stall_s:.0f}s stall window (chunk {count}) — "
                    f"a genuine hang, not merely slow"
                )
            # Generous TOTAL backstop (default 2h via KREMORY_INGEST_BUDGET_S):
            # even a progressing-but-pathologically-slow conversation must bound so
            # it can't eat an entire multi-conversation run (TD-128 intent).
            elapsed = time.monotonic() - ingest_start
            if elapsed > INGEST_BUDGET_S:
                raise KremoryStalled(
                    f"ingest of {sample_id} exceeded {INGEST_BUDGET_S:.0f}s TOTAL "
                    f"backstop after {count} chunks ({elapsed:.0f}s) — progressing but "
                    f"too slow to include; raise KREMORY_INGEST_BUDGET_S or fix throughput (TD-127)"
                )

    if session_id:
        client.end_session(session_id, summary=f"Ingested {count} turn chunks for {sample_id}")

    # Run creative consolidation to build SHARES_THEME edges between memories.
    #
    # DREAM KEEP-OR-CUT (2026-08-19): this single call IS the dream phase from the
    # harness's point of view. `--no-dream` skips it. Logged either way — a silent
    # skip would make the two arms indistinguishable in the run log, which is
    # precisely the confound the pre-registration's validity gate exists to catch.
    if no_dream:
        print(f"[{sample_id}] --no-dream: SKIPPING consolidation (dream-off arm)", flush=True)
    else:
        print(f"[{sample_id}] running consolidation (dream-on arm)", flush=True)
        client.consolidate("creative", namespace)

    return count


# ---------------------------------------------------------------------------
# Recall strategies
# ---------------------------------------------------------------------------

def get_recall_limit(category: str, config: Config) -> int:
    if category == "temporal":
        return config.recall_limit_temporal
    if category == "multi-hop":
        return config.recall_limit_multihop
    return config.recall_limit


def recall_codemem(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
) -> list[str]:
    """Standard codemem hybrid recall (vector + BM25 + graph scoring)."""
    results = client.recall(question, namespace, limit=limit)
    return [r.get("content", "") for r in results if r.get("content")]


def recall_codemem_graph(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
    graph_depth: int = 2,
) -> list[str]:
    """Codemem recall with explicit graph expansion (like AutoMem's bridge discovery)."""
    results = client.recall(question, namespace, limit=limit)

    # Collect initial content
    contents = []
    seen_ids = set()
    for r in results:
        content = r.get("content", "")
        node_id = r.get("id", r.get("node_id", ""))
        if content:
            contents.append(content)
        if node_id:
            seen_ids.add(node_id)

    # Graph expand: follow edges from top results to find bridge memories
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
            # Use memory_id to fetch full content (label is truncated to 80 chars)
            memory_id = node.get("memory_id", node.get("id", ""))
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            if node.get("kind") == "Memory":
                mem = client.get_memory(memory_id)
                if mem and mem.get("content"):
                    contents.append(mem["content"])

    return contents[:limit * 2]  # cap total context


def recall_baseline(sessions: list[dict]) -> list[str]:
    """Baseline: return all session text (simulates long-context LLM)."""
    contents = []
    for sess in sessions:
        parts = [f"[Session {sess['session_num']}] [{sess['datetime']}]"]
        for turn in sess["turns"]:
            parts.append(f"{turn['speaker']}: {turn['text']}")
        contents.append("\n".join(parts))
    return contents


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(config: Config) -> dict:
    client = CodememClient(config.base_url, server_mode=config.server_mode)

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
        print("Download it first — see README.md", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(config.dataset_path)
    print(f"Loaded {len(dataset)} conversations")

    # Filter conversations if specified
    if config.conversations:
        dataset = [dataset[i] for i in config.conversations if i < len(dataset)]
        print(f"Filtered to {len(dataset)} conversations: {config.conversations}")

    all_results = []
    category_stats: dict[str, dict] = {}
    # Categories retrieved but deliberately NOT scored (see ABSTENTION_CATEGORIES).
    unscored_stats: dict[str, int] = {}

    # --- Observability-first + fail-fast-and-loud instrumentation ---
    # A multi-hour benchmark must be diagnosable (WHY is a score low: retrieval
    # broken vs model-too-weak vs scoring?) and must never silently grind on
    # garbage or hang. Per-question records are written incrementally (crash-safe
    # + resumable-inspectable); a circuit-breaker stops loud on systemic failure;
    # a liveness probe aborts loud if the server/Ollama dies.
    import datetime
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    jsonl_path = (config.output.parent if config.output else Path("results")) / f"run-{run_ts}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_f = open(jsonl_path, "a")
    print(f"  [o11y] per-question records -> {jsonl_path}", file=sys.stderr)
    q_done = 0
    recall_empty_count = 0
    CIRCUIT_MIN = 20            # evaluate the breaker after this many questions
    CIRCUIT_EMPTY_RATE = 0.80  # >80% empty recalls = systemic ingest/recall failure
    CIRCUIT_HTTP_RATE = 0.20   # >20% HTTP errors = systemic
    LIVENESS_EVERY = 25

    # TD-128 (B6): per-conversation ingest ISOLATION. A budget-abort on ONE
    # conversation must not halt a multi-conversation run — record it and move
    # on, so the run degrades to a partial matrix (only fully-ingested
    # conversations scored) instead of a dead run. Fail-loud is preserved: the
    # aborted set is reported loudly at the end, and a run where EVERY attempted
    # conversation aborts is still a hard `sys.exit(4)` (systemic ingest failure).
    # This folds the shell-level per-conv isolation of `.context/robust-3conv.sh`
    # into the harness so a single `--conversations 0 1 2` invocation is robust.
    ingest_aborted: list[dict] = []
    ingest_attempted = 0

    for conv_idx, conversation in enumerate(dataset):
        sample_id = conversation.get("sample_id", f"conv_{conv_idx}")
        namespace = f"{NAMESPACE_PREFIX}-{sample_id}"
        sessions = extract_sessions(conversation)
        questions = extract_questions(conversation)

        print(f"\n--- Conversation {conv_idx}: {sample_id} ---")
        print(f"  {len(sessions)} sessions, {len(questions)} questions")

        # Ingest
        if config.mode != "baseline" and not config.skip_ingest:
            # Clean previous run
            client.delete_namespace(namespace)
            time.sleep(0.5)

            ingest_attempted += 1
            # Streamed, not summarised at the end: a multi-hour paid run that
            # dies at conversation 6 of 10 must leave a readable trail of what
            # already succeeded, and an end-of-run summary is written exactly
            # never in that case. The per-QUESTION stream below already had
            # this; the ingest half only ever emitted a record when it FAILED,
            # so a partial run's completed conversations were invisible.
            ingest_started_at = time.time()
            jsonl_f.write(json.dumps({
                "event": "ingest_start", "conv_idx": conv_idx,
                "sample_id": sample_id, "namespace": namespace,
                "sessions": len(sessions), "questions": len(questions),
            }) + "\n")
            jsonl_f.flush()
            try:
                mem_count = ingest_conversation(
                    client, namespace, sessions, sample_id, no_dream=config.no_dream
                )
            except KremoryStalled as e:
                # TD-128 (B6): ISOLATE — record + skip this conversation's
                # questions (its namespace is only partially ingested; scoring it
                # would pollute the matrix), then continue to the next. Still
                # LOUD. The all-aborted case is caught after the loop (exit 4).
                print(
                    f"\n{'='*64}\n"
                    f"FAIL-FAST ABORT (ingestion) — conversation ISOLATED, run continues: {e}\n"
                    f"  conversation={conv_idx} sample={sample_id}  "
                    f"http_errors={client.total_http_errors}\n"
                    f"  → kremory ingest stalled/too-slow for THIS conversation — "
                    f"excluding it from the matrix, moving to the next. Fix ingest "
                    f"throughput or raise KREMORY_INGEST_BUDGET_S to include it.\n{'='*64}",
                    file=sys.stderr,
                )
                ingest_aborted.append({
                    "conv_idx": conv_idx, "sample_id": sample_id,
                    "reason": str(e), "http_errors": client.total_http_errors,
                })
                jsonl_f.write(json.dumps({
                    "event": "ingest_aborted", "conv_idx": conv_idx,
                    "sample_id": sample_id, "namespace": namespace,
                    "reason": str(e), "http_errors": client.total_http_errors,
                }) + "\n")
                jsonl_f.flush()
                continue
            ingest_elapsed_s = round(time.time() - ingest_started_at, 1)
            print(f"  Stored {mem_count} memories in {ingest_elapsed_s}s")
            jsonl_f.write(json.dumps({
                "event": "ingest_done", "conv_idx": conv_idx,
                "sample_id": sample_id, "namespace": namespace,
                "memories": mem_count, "elapsed_s": ingest_elapsed_s,
                "http_errors": client.total_http_errors,
            }) + "\n")
            jsonl_f.flush()
            # Brief pause for enrichment
            time.sleep(1.0)
        elif config.skip_ingest:
            print("  Skipping ingestion (--skip-ingest)")

        # Evaluate each question
        for qa in tqdm(questions, desc=f"  Evaluating", leave=False):
            category = qa["category"]
            limit = get_recall_limit(category, config)

            # Recall based on mode (TD-132: time it — client-side wall-clock is
            # the per-question recall latency / TTFB for these non-streaming JSON
            # responses).
            recall_t0 = time.monotonic()
            if config.mode == "baseline":
                memories = recall_baseline(sessions)
            elif config.mode == "rag":
                # Use only vector search by setting a very high k
                # (codemem's /search endpoint does hybrid by default,
                #  but with low k the vector component dominates)
                memories = recall_codemem(client, qa["question"], namespace, limit)
            elif config.mode == "codemem-graph":
                memories = recall_codemem_graph(
                    client, qa["question"], namespace, limit, config.graph_depth
                )
            else:  # codemem (default)
                memories = recall_codemem(client, qa["question"], namespace, limit)
            recall_latency_ms = round((time.monotonic() - recall_t0) * 1000.0, 1)

            # OPT-IN, default OFF (Config.capture_text_block — RECALL-LEDGER
            # §4.19 / TD-155 / TD-196). SECOND request, in ADDITION to the
            # primary recall above — `memories` / `recall_latency_ms` /
            # `is_correct` below are computed from the UNCHANGED primary
            # call only. This is best-effort instrumentation: a failure here
            # (see `recall_text_block`'s own try/except) degrades to `None`
            # and never touches `client.total_http_errors` or aborts the run.
            recalled_text_block = None
            if config.capture_text_block and config.mode != "baseline":
                recalled_text_block = client.recall_text_block(
                    qa["question"], namespace, limit,
                )

            # Check if gold answer is in recalled memories (AutoMem-style).
            # Abstention categories are retrieved + persisted but NOT scored —
            # a presence metric cannot see abstention (see the refusal in
            # check_answer_in_memories). `is_correct` is None, not False, so
            # "unscored" is distinguishable downstream from "scored wrong".
            if category in ABSTENTION_CATEGORIES:
                is_correct, confidence, explanation = (
                    None, None,
                    "not scored: abstention is an answerer property, invisible to a "
                    "retrieval-presence scorer",
                )
            else:
                is_correct, confidence, explanation = check_answer_in_memories(
                    qa["answer"], memories, category,
                )

            result = {
                "sample_id": sample_id,
                "question_id": qa["question_id"],
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "category": category,
                "evidence_ids": qa["evidence"],
                "memories_recalled": len(memories),
                # Workstream A (LLM-judge scorer): persist the actual recalled
                # memory TEXTS per question so the run JSON is re-scorable
                # offline by a cross-family LLM judge (judge_rescore.py) without
                # re-running recall. The strict substring scorer above only
                # consumes len(memories); the judge needs the strings.
                "recalled_memories": memories,
                # TD-139 / ADR-078 Phase A: index-aligned with `recalled_memories`
                # (only for the codemem paths, which flatten `client.last_results`
                # one-for-one; empty for `baseline`/`graph` which build their own
                # lists). `evidence_eval.py` already reads this field, and
                # `qa_eval.py --structured` uses it to group the answerer's
                # context into labelled blocks instead of one anonymous list.
                "recalled_memory_provenance": (
                    [
                        # TD-216: `id` was DROPPED here, and it is the ONLY link
                        # from an episode-kind result back to a gold evidence turn
                        # — `source_episode_id` is populated ONLY for `kind=fact`
                        # by contract (for an episode item the `id` IS the episode
                        # id; see `SearchResultWire::source_episode_id`). Without
                        # it, 0 of 9950 provenance entries in the 2026-08-11 full
                        # run carried ANY usable linkage, so no rank-aware metric
                        # (nDCG/MRR/mean-gold-rank) was computable by anyone, from
                        # any run — which is why every retrieval lever has been
                        # judged on a substring scorer proven blind to ordering
                        # (0 of 1986 verdicts changed by shuffling the top-10).
                        {
                            "kind": r.get("kind"),
                            "id": r.get("id"),
                            "source_episode_id": r.get("source_episode_id"),
                        }
                        for r in client.last_results
                        if r.get("content")
                    ]
                    if config.mode in ("codemem", "rag")
                    else []
                ),
                # RECALL-LEDGER §4.19 / TD-155 / Candidate A (dual capture):
                # kremory's own `format=text&template=temporal_facts`
                # rendering for THIS question — `None` when
                # `--capture-text-block` was not passed, `--server-mode
                # content` (the template has no meaning there), or the
                # second request failed. NEVER derived from `memories`
                # above; a second, independent server round-trip.
                # `qa_eval.py --text-block` consumes this verbatim in place
                # of `format_memories()`/`format_memories_structured()`.
                "recalled_text_block": recalled_text_block,
                "is_correct": is_correct,
                "confidence": None if confidence is None else round(confidence, 4),
                "explanation": explanation,
                "mode": config.mode,
                "recall_latency_ms": recall_latency_ms,
            }
            all_results.append(result)

            # --- o11y: per-question structured record, written incrementally (crash-safe) ---
            recall_empty = (len(memories) == 0)
            if recall_empty:
                recall_empty_count += 1
            q_done += 1
            jsonl_f.write(json.dumps({
                "question_id": qa["question_id"], "category": category,
                "namespace": namespace, "recall_returned": len(memories),
                "recall_empty": recall_empty, "http_errors": client.total_http_errors,
                "correct": is_correct, "recall_latency_ms": recall_latency_ms,
                # RECALL-LEDGER §4.19: observability-only, NOT part of the
                # circuit breaker (see comment on total_text_block_errors in
                # kremory_client.py's __init__).
                "text_block_errors": client.total_text_block_errors,
            }) + "\n")
            jsonl_f.flush()

            # --- fail-loud: systemic-failure circuit-breaker (don't grind 1,986 Qs into zeros) ---
            if q_done == CIRCUIT_MIN:
                empty_rate = recall_empty_count / q_done
                http_rate = client.total_http_errors / q_done
                if empty_rate > CIRCUIT_EMPTY_RATE or http_rate > CIRCUIT_HTTP_RATE:
                    jsonl_f.flush()
                    print(f"\n[FAIL-LOUD] SYSTEMIC FAILURE after {q_done} questions: "
                          f"{recall_empty_count}/{q_done} recalls empty ({empty_rate:.0%}), "
                          f"{client.total_http_errors} HTTP errors ({http_rate:.0%}). "
                          f"Ingest/recall likely broken — aborting before wasting the full run. "
                          f"Records: {jsonl_path}", file=sys.stderr)
                    sys.exit(2)

            # --- fail-loud: liveness probe (abort if server/Ollama died, don't hang) ---
            if q_done % LIVENESS_EVERY == 0 and config.mode != "baseline" and not client.health():
                print(f"\n[FAIL-LOUD] kremory-http /health failed at question {q_done} — "
                      f"server/Ollama down. Aborting. Records: {jsonl_path}", file=sys.stderr)
                sys.exit(3)

            # Track per-category. Unscored categories are counted in a SEPARATE
            # bucket so they stay visible in the summary without silently
            # entering the accuracy denominator — the failure mode being fixed
            # here is precisely a category that inflated the total while
            # measuring nothing.
            if category in ABSTENTION_CATEGORIES:
                unscored_stats[category] = unscored_stats.get(category, 0) + 1
            else:
                if category not in category_stats:
                    category_stats[category] = {"correct": 0, "total": 0}
                category_stats[category]["total"] += 1
                if is_correct:
                    category_stats[category]["correct"] += 1

    jsonl_f.close()

    # TD-128 (B6): report ingest isolation outcome LOUDLY, and enforce fail-loud
    # for the systemic case — if every conversation we attempted to ingest
    # aborted, that is an ingest-broken run, not a partial matrix: exit 4.
    if ingest_aborted:
        print(f"\n{'='*64}\n[ISOLATION] {len(ingest_aborted)}/{ingest_attempted} "
              f"conversation(s) aborted ingest and were EXCLUDED from the matrix:",
              file=sys.stderr)
        for a in ingest_aborted:
            print(f"    - conv {a['conv_idx']} ({a['sample_id']}): {a['reason']}",
                  file=sys.stderr)
        print(f"{'='*64}", file=sys.stderr)
        if ingest_attempted > 0 and len(ingest_aborted) == ingest_attempted:
            print(f"[FAIL-LOUD] ALL {ingest_attempted} attempted conversations "
                  f"aborted ingest — systemic ingest failure, not a partial run. "
                  f"Exiting 4.", file=sys.stderr)
            sys.exit(4)

    # Summary with scores
    total_correct = sum(v["correct"] for v in category_stats.values())
    total_questions = sum(v["total"] for v in category_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"Mode: {config.mode}")
    print(f"{'='*60}")
    print(f"\n{'Category':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"{cat:<25} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")
    for cat in sorted(unscored_stats.keys()):
        print(f"{cat:<25} {'—':>8} {unscored_stats[cat]:>8} {'NOT SCORED':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL (scored only)':<25} {total_correct:>8} {total_questions:>8} "
          f"{overall:>9.1f}%")
    # TD-217(b): the ONE line anyone copies out of a run must carry its own
    # denominator caveat. A bare "98.0%" is one line of JSON away from looking
    # like it covered all 199 questions; built here so the qualifier travels
    # with the number instead of relying on a reader scrolling to the prose.
    headline = build_headline(total_correct, total_questions, unscored_stats)
    print(f"\nHEADLINE: {headline}")

    # TD-223/TD-224 — printed right beside the headline on purpose. The score and
    # the graph's survival are independent claims; quoting the first without the
    # second is what let a 98.0% be reported on a graph with both speakers gone.
    graph_integrity = check_graph_integrity(
        config.graph_db or _os.environ.get("KREMORY_MCP_DB_PATH")
    )
    if graph_integrity["status"] == "checked":
        n_chain = graph_integrity["chained_entities"]
        n_fan = graph_integrity["fanin_entities"]
        if not n_chain and not n_fan:
            print(f"GRAPH INTEGRITY: clean — 0 chained entities, 0 fan-ins across "
                  f"{graph_integrity['live_merges']} live merges")
        else:
            # Both counts print unconditionally once EITHER is non-zero. A run with
            # 0 chains and 20 fan-ins previously printed the word "clean".
            print(f"GRAPH INTEGRITY: ⚠️  {n_chain} chained entities, {n_fan} fan-ins "
                  f"across {graph_integrity['live_merges']} live merges — this score "
                  f"was measured on a graph that may have lost referents:")
            for chain in graph_integrity["chains"]:
                print(f"  [chain]  {chain}")
            # Fan-ins are advisory: at this threshold an ordinary two-variant
            # canonicalisation looks identical to damage. Read the names.
            for fan in graph_integrity["fanins"]:
                print(f"  [fan-in] {fan}")
    else:
        # LOUD. A skipped check is not a passed check — the 2026-09-08 floor run
        # recorded status="skipped" and was read as a clean bill of health.
        print(f"GRAPH INTEGRITY: ⚠️  {graph_integrity['status'].upper()} — NOT "
              f"VERIFIED — {graph_integrity['reason']}")
    if unscored_stats:
        n_unscored = sum(unscored_stats.values())
        print(f"\n  {n_unscored} question(s) EXCLUDED from the denominator above "
              f"({', '.join(sorted(unscored_stats))}) — "
              f"{total_questions}/{total_questions + n_unscored} questions scored.")
        print(f"  These are NOT unmeasured (TD-217, corrected 2026-08-14). Every")
        print(f"  category-5 question carries REAL evidence turns — they are")
        print(f"  speaker-attribution false premises, not questions about things")
        print(f"  never discussed — so their RETRIEVAL is scored by")
        print(f"  `evidence_eval.py` (recall@k / nDCG@k against the gold turn ids)")
        print(f"  and reported as its own category row there. What this scorer")
        print(f"  cannot see is the ANSWERER's refusal of the false premise, which")
        print(f"  is not a retrieval property. `recalled_memories` is persisted, so")
        print(f"  an abstention judge can score that offline. Denominator matches")
        print(f"  `qa_eval.py`, which already excluded the same set.")
    # Baselines — protocol-matched ONLY. Canonical SoT:
    #   .ai-docs/specs/locomo-benchmark-protocol-2026-07-27.md
    #   .ai-docs/research/locomo-competitor-baselines-protocol-audit-2026-07-27.md
    #
    # The score printed ABOVE is the SUBSTRING retrieval-proxy scorer (no answer
    # generation), so only retrieval-proxy systems may sit beside it. kremory's
    # externally-quotable headline is the QA-GEN number from `qa_eval.py`
    # (answerer + judge), NOT this one — they differed by ~17pt on the same run
    # (94.8% substring vs 77.5% qa-gen), so conflating them is a real hazard.
    #
    # REMOVED 2026-07-27 — both prior lines were uncited AND wrong:
    #   "AutoMem: 90.53%" — RETRACTED at source ("That number was wrong",
    #       https://automem.ai/blog/benchmarking-honesty; re-fetched + confirmed).
    #       Cause included a category-5 bug scoring answers against EMPTY STRINGS.
    #       THIS HARNESS SHIPPED THE IDENTICAL BUG until 2026-07-28: category-5
    #       gold lives under `adversarial_answer`, `extract_questions` read
    #       `answer` with a `""` default, and `"" in anything` is True — so all
    #       446 scored a false "exact substring match". Recording the other
    #       project's defect in a comment did not prevent our own; only the
    #       structural refusal now at the top of check_answer_in_memories does.
    #   "CORE: 88.24%"    — vendor marketing figure; CORE's own reproducible repo
    #       reports 85% on a different 1,247-question subset, and CORE is
    #       generate-then-judge, so it never belonged beside a substring score.
    print(f"\n--- Baselines (retrieval-proxy protocol only — same family as this scorer) ---")
    print(f"  AutoMem:  84.74%  (1683/1986, all 5 categories)")
    print(f"            hybrid: cats 1-4 via check_answer_in_memories (use_llm_extraction=False),")
    print(f"            judge only for cat-5.  https://automem.ai/benchmarks/")
    print(f"            ⚠ DIFFERENT DENOMINATOR: theirs is n=1986 including a")
    print(f"              JUDGED cat-5; the score above is n=1540 with cat-5")
    print(f"              excluded. Not directly comparable without rescoring one.")
    print(f"\n  NB generate-then-judge systems (CORE 85% @1247q, Mem0 92.5%) are NOT")
    print(f"     comparable to the number above — compare those against `qa_eval.py")
    print(f"     answer-tally`, minding the k / judge deltas (we run k=10 + gpt-4o")
    print(f"     judge; CORE runs k=20, Mem0 judges with gpt-4o-mini).")

    # TD-132 o11y: final /metrics scrape (for a single fresh-DB conversation the
    # cumulative counters ARE this run's totals) + client-side recall-latency
    # aggregate. Degrades to {} if the server lacks the `prometheus` feature.
    o11y_metrics = client.scrape_metrics()
    recall_latencies = sorted(
        r["recall_latency_ms"] for r in all_results if r.get("recall_latency_ms") is not None
    )

    def _pct(xs, p):
        if not xs:
            return None
        i = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return xs[i]

    o11y = {
        "metrics_endpoint_present": bool(o11y_metrics),
        "tokens_total": o11y_metrics.get("kremory_core_tokens_total"),
        "cost_usd_total": o11y_metrics.get("kremory_core_cost_usd_total"),
        "recall_latency_ms_p50": _pct(recall_latencies, 50),
        "recall_latency_ms_p95": _pct(recall_latencies, 95),
        "recall_latency_ms_mean": (
            round(sum(recall_latencies) / len(recall_latencies), 1) if recall_latencies else None
        ),
        "raw_metric_totals": o11y_metrics,
    }
    print(f"\n--- o11y (TD-132) ---")
    print(f"  tokens_total:      {o11y['tokens_total']}")
    print(f"  cost_usd_total:    {o11y['cost_usd_total']}")
    print(
        f"  recall latency ms: p50={o11y['recall_latency_ms_p50']} "
        f"p95={o11y['recall_latency_ms_p95']} mean={o11y['recall_latency_ms_mean']}"
    )
    if not o11y["metrics_endpoint_present"]:
        print("  [warn] /metrics absent — server built without --features prometheus", file=sys.stderr)

    output = {
        "mode": config.mode,
        "scorer": config.scorer,
        "server_mode": config.server_mode,
        # RECALL-LEDGER §4.19 / TD-155: whether the SECOND, opt-in
        # `recalled_text_block` capture ran for this file. `False` on every
        # pre-2026-08-10 run (field absent entirely) and on any run that
        # didn't pass --capture-text-block.
        "capture_text_block": config.capture_text_block,
        # recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1] + TD-135: stamp the
        # exact sweep point (git SHA + the server's ACTIVE scoring config +
        # build-feature flags, read from its own GET /health — the single source of
        # truth, NOT the harness env, which can silently diverge). answer-tally/gate
        # can then refuse to score a stale/mismatched recall file (assert_provenance
        # in bench/locomo/provenance.py).
        "provenance": build_provenance(server=config.base_url),
        "total_questions": len(all_results),
        "o11y": o11y,
        "category_stats": category_stats,
        "unscored_stats": unscored_stats,
        # TD-217(b): the denominator, stated. `total_questions` counts every
        # question ASKED; `category_stats` sums only the ones SCORED, and the
        # two differ by `unscored_stats`. A consumer computing an accuracy from
        # this file previously had to notice that on its own.
        "scored_questions": sum(v["total"] for v in category_stats.values()),
        "unscored_questions": sum(unscored_stats.values()),
        "headline": headline,
        # TD-223/TD-224: a recall score says nothing about whether the graph it
        # was measured on survived consolidation. Those are independent claims,
        # and until now only the first was ever measured.
        "graph_integrity": graph_integrity,
        "results": all_results,
        # TD-128 (B6): conversations excluded from this matrix due to ingest abort.
        "ingest_aborted": ingest_aborted,
        "ingest_attempted": ingest_attempted,
    }

    # Save results
    if config.output:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {config.output}")
    else:
        default_output = Path(__file__).parent / "results" / f"{config.mode}.json"
        default_output.parent.mkdir(parents=True, exist_ok=True)
        with open(default_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {default_output}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # W0.3: unbuffered stdout so long runs stream progress (tqdm bars, per-
    # conversation status prints) live instead of buffering until process
    # exit/flush. In-script so it's robust regardless of invocation (no
    # reliance on callers remembering `python3 -u` or PYTHONUNBUFFERED=1).
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="LoCoMo benchmark harness for codemem")
    parser.add_argument("--mode", default="codemem",
                        choices=["baseline", "rag", "codemem", "codemem-graph"],
                        help="Recall strategy to benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to locomo10.json")
    parser.add_argument("--base-url", default=CODEMEM_BASE,
                        help="Codemem API base URL")
    parser.add_argument("--recall-limit", type=int, default=50,
                        help="Recall depth applied to ALL categories unless a "
                             "per-category override below is given. (Previously "
                             "temporal/multi-hop were silently pinned at 10 "
                             "regardless of this flag — a depth bug.)")
    parser.add_argument("--recall-limit-temporal", type=int, default=None,
                        help="Override recall depth for temporal questions "
                             "(default: --recall-limit)")
    parser.add_argument("--recall-limit-multihop", type=int, default=None,
                        help="Override recall depth for multi-hop questions "
                             "(default: --recall-limit)")
    parser.add_argument("--server-mode", default="recall",
                        choices=["recall", "content", "hybrid"],
                        help="kremory-http GET /search server-side retrieval mode "
                             "(distinct from --mode, the harness's own recall strategy)")
    parser.add_argument("--graph-depth", type=int, default=2,
                        help="Graph expansion depth for codemem-graph mode")
    parser.add_argument("--conversations", type=int, nargs="*", default=[],
                        help="Specific conversation indices to evaluate (default: all)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion, reuse existing memories")
    parser.add_argument("--output", type=Path,
                        help="Output file path for results JSON")
    parser.add_argument("--scorer", default="substring",
                        choices=["substring", "llm-judge"],
                        help="Scoring path. 'substring' = strict word-overlap "
                             "matcher (default, reproducible). 'llm-judge' = "
                             "the run captures recalled_memories for an OFFLINE "
                             "cross-family LLM judge (see judge_rescore.py); "
                             "substring still runs inline so both are reported.")
    parser.add_argument("--graph-db", default=None,
                        help="TD-223/TD-224. Path to the SQLite graph this run "
                             "writes, so the report can state whether "
                             "consolidation merged entities out of existence. "
                             "Defaults to $KREMORY_MCP_DB_PATH. When unknown the "
                             "check reports 'skipped' — never a pass.")
    parser.add_argument("--no-dream", action="store_true",
                        help="Skip the post-ingest consolidation call, so the dream "
                             "phase never runs. The dream-off arm of the pre-registered "
                             "keep-or-cut A/B (.ai-docs/decisions/"
                             "dream-keep-or-cut-prereg-2026-08-19.md).")
    parser.add_argument("--capture-text-block", action="store_true",
                        help="OPT-IN, default OFF (RECALL-LEDGER §4.19 / "
                             "TD-155). Also issue a SECOND GET /search per "
                             "question with format=text&template=temporal_facts "
                             "— kremory's own prompt-ready, DATED rendering, "
                             "the one MCP tool consumers have always received "
                             "and this harness has never captured. Persisted "
                             "as `recalled_text_block`, alongside the existing "
                             "`recalled_memories` (byte-identical, unaffected). "
                             "NOT retrieval-equivalent to the primary call "
                             "under --server-mode hybrid (TD-196) — the two "
                             "renderings fuse the BM25 content stream a "
                             "different number of times at the HTTP layer. "
                             "Retrieval-equivalent under --server-mode recall.")
    args = parser.parse_args()

    config = Config(
        base_url=args.base_url,
        dataset_path=args.dataset,
        recall_limit=args.recall_limit,
        # Depth-bug fix: --recall-limit now applies to temporal/multi-hop too
        # (they were hardcoded to 10, silently capping every deep run). A
        # per-category flag still overrides when explicitly given.
        recall_limit_temporal=(
            args.recall_limit_temporal
            if args.recall_limit_temporal is not None
            else args.recall_limit
        ),
        recall_limit_multihop=(
            args.recall_limit_multihop
            if args.recall_limit_multihop is not None
            else args.recall_limit
        ),
        graph_depth=args.graph_depth,
        mode=args.mode,
        conversations=args.conversations or [],
        skip_ingest=args.skip_ingest,
        output=args.output,
        server_mode=args.server_mode,
        scorer=args.scorer,
        graph_db=args.graph_db,
        no_dream=args.no_dream,
        capture_text_block=args.capture_text_block,
    )
    run_benchmark(config)


if __name__ == "__main__":
    main()

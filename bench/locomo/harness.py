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
INGEST_BUDGET_S = float(_os.environ.get("KREMORY_INGEST_BUDGET_S", "7200"))


class KremoryStalled(RuntimeError):
    """Raised when kremory ingest stalls (store timeout) or blows the wall-clock
    budget. Bubbles to run_benchmark's fail-fast handler — never swallowed."""


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


# ---------------------------------------------------------------------------
# Codemem API client
# ---------------------------------------------------------------------------

class CodememClient:
    def __init__(self, base_url: str, timeout: float = 30.0, server_mode: str = "recall"):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)
        # kremory-http GET /search server-side retrieval mode (recall/content/
        # hybrid) — see Config.server_mode docstring above for why this is a
        # separate axis from --mode.
        self.server_mode = server_mode
        # Raw wire rows from the most recent `recall()` — see the comment there.
        self.last_results: list[dict] = []
        # kremory's POST /memories runs a SYNCHRONOUS 3-stage LLM extraction
        # pipeline (entities -> relations -> triplets, then per-entity
        # ResolutionVerdict dedup calls) on every store — this is not a
        # cheap embed-only write. Measured against gemma4:e4b (the fast
        # default model): kremory.ingest.completed total_ms observed at
        # 15.7s / 25.1s / 21.6s for successful stores in a smoke run, and
        # one store still had ~6 pending ResolutionVerdict calls after 30s
        # of stage time — tripping this client's default 30.0s timeout
        # with an httpx.ReadTimeout mid-store. 120s gives headroom for a
        # slow store plus one ladder-arm fallback retry.
        self.store_timeout = 90.0
        # kremory's POST /consolidation/{cycle} runs the full dream()
        # reconciliation (discover/aliases/reclassify/consistency_check/
        # canonicalize passes) over every episode in the namespace — scales
        # with namespace size, not per-call content size. Generous ceiling
        # so a 19-session conversation's worth of memories doesn't trip the
        # same class of timeout.
        # Raised from 300s (conv0's dream exceeded it → ReadTimeout aborted
        # consolidation → multi-hop under-served). Env-overridable.
        self.consolidate_timeout = float(_os.environ.get("KREMORY_CONSOLIDATE_TIMEOUT_S", "1800"))
        # Per-source HTTP-error counter (observability-first-class: an
        # aggregate accuracy score alone can't tell you WHY it's low —
        # retrieval-broken vs model-too-weak vs scoring-bug. This is the
        # cheap half of that signal; recall_empty tracked alongside it in
        # run_benchmark() is the other half).
        self.total_http_errors = 0

    def health(self) -> bool:
        try:
            r = self.http.get("/health")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def store_memory(
        self,
        content: str,
        namespace: str,
        memory_type: str = "Context",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str | None:
        # kremory's POST /memories only accepts {content, namespace,
        # published_at?} — memory_type/importance/tags are codemem-only
        # fields kremory ignores; kept in the signature so callers below
        # don't need to change, just not sent over the wire.
        try:
            r = self.http.post(
                "/memories",
                json={
                    "content": content,
                    "namespace": namespace,
                },
                timeout=self.store_timeout,
            )
        except httpx.TimeoutException as e:
            # A store TIMEOUT means kremory's ingest stalled (per-remember() fans
            # out to ~20 sequential LLM calls). Per fail-fast-and-loud: do NOT
            # grind or swallow — abort the whole run immediately with a clear
            # diagnostic. Bubbles to run_benchmark's KremoryStalled handler.
            raise KremoryStalled(
                f"POST /memories timed out after {self.store_timeout:.0f}s "
                f"(namespace={namespace}, content_len={len(content)}) — "
                f"kremory ingest stalled"
            ) from e
        except httpx.HTTPError as e:
            raise KremoryStalled(
                f"POST /memories transport error ({type(e).__name__}: {e}) "
                f"(namespace={namespace})"
            ) from e
        if r.status_code == 201:
            return r.json().get("id")
        self.total_http_errors += 1
        print(f"  [warn] store failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
        return None

    def recall(self, query: str, namespace: str, limit: int = 15) -> list[dict]:
        r = self.http.get(
            "/search",
            params={
                "q": query,
                "namespace": namespace,
                "k": limit,
                # ALWAYS send `mode` explicitly (fixed 2026-07-28, ADR-078).
                # This previously omitted the param whenever `server_mode ==
                # "recall"`, on the belief that the server's default was also
                # `recall`. It is NOT: `SearchMode`'s `#[default]` is `Hybrid`
                # (`kremory-http.rs`), deliberately, because the entity-graph
                # `recall` surface judges 40.2% vs hybrid's 71.4%. So every run
                # this harness has ever labelled `server_mode: recall` was in
                # fact measuring HYBRID — the label lied, and `--server-mode
                # recall` was unreachable over the wire. Sending it explicitly
                # makes the provenance stamp true and the flag actually work.
                "mode": self.server_mode,
            },
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            # ADR-078 Phase A: stash the RAW wire rows so the caller can persist
            # per-item provenance (`kind`, `source_episode_id` — TD-139's
            # `SearchResultWire`) alongside the flattened content strings. The
            # flattening below is lossy: it turns entity summaries, facts and
            # verbatim episode turns into one anonymous list, which is exactly
            # the format the answerer has always been fed. Keeping the kinds
            # lets us test whether LABELLED context beats the flat blob.
            self.last_results = results
            return results
        self.last_results = []
        self.total_http_errors += 1
        print(f"  [warn] recall failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
        return []

    def scrape_metrics(self) -> dict[str, float]:
        """Scrape GET /metrics (Prometheus text) and sum each metric across its
        label sets → {metric_name: total}. TD-132: surfaces the server-side
        `kremory_core_*` counters (tokens, cost, request latency, dream duration)
        the kremory-http `prometheus` feature exposes. Returns {} if the endpoint
        is absent (server built without the feature) — bench degrades gracefully.
        """
        try:
            r = self.http.get("/metrics", timeout=10.0)
        except Exception:
            return {}
        if r.status_code != 200:
            return {}
        totals: dict[str, float] = {}
        for line in r.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Prometheus text: `name{labels} value`  OR  `name value`.
            try:
                left, val = line.rsplit(" ", 1)
                value = float(val)
            except ValueError:
                continue
            name = left.split("{", 1)[0]
            totals[name] = totals.get(name, 0.0) + value
        return totals

    def graph_neighbors(self, node_id: str, depth: int = 2) -> list[dict]:
        # kremory-http has no graph-traversal REST tool yet — stub so
        # --mode codemem-graph degrades to plain recall instead of 404ing.
        return []

    def get_memory(self, memory_id: str) -> dict | None:
        # No GET /memories/{id} route on kremory-http yet — stub.
        return None

    def consolidate(self, cycle: str, namespace: str) -> bool:
        # kremory's POST /consolidation/{cycle} REQUIRES ?namespace= — dream()
        # is always namespace-scoped (unlike codemem's global consolidation);
        # omitting it is a loud 422, not a silent no-op.
        try:
            r = self.http.post(
                f"/consolidation/{cycle}",
                params={"namespace": namespace},
                timeout=self.consolidate_timeout,
            )
        except httpx.HTTPError as e:
            # Consolidation is best-effort enrichment (SHARES_THEME edges) — a
            # timeout must NOT nuke the run, but MUST be loud (not a raw crash).
            self.total_http_errors += 1
            print(
                f"  [warn] consolidation '{cycle}' failed after "
                f"{self.consolidate_timeout:.0f}s: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return False
        return r.status_code == 200

    def start_session(self, namespace: str) -> str | None:
        # kremory-http has no session concept/route — no-op (matches the
        # already-no-op contract: return None, ingest_conversation() treats a
        # falsy session_id as "skip end_session").
        return None

    def end_session(self, session_id: str, summary: str = "") -> bool:
        # No-op — see start_session.
        return True

    def delete_namespace(self, namespace: str) -> bool:
        r = self.http.delete(f"/namespaces/{namespace}")
        return r.status_code in (200, 404)

    def get_namespaces(self) -> list[dict]:
        # kremory-http has no namespace-listing route yet — stub.
        return []


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
            mid = client.store_memory(
                content=chunk_content,
                namespace=namespace,
                memory_type="Context",
                importance=0.5,
                tags=tags,
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

    # Run creative consolidation to build SHARES_THEME edges between memories
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
            try:
                mem_count = ingest_conversation(client, namespace, sessions, sample_id)
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
            print(f"  Stored {mem_count} memories")
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
                        {"kind": r.get("kind"), "source_episode_id": r.get("source_episode_id")}
                        for r in client.last_results
                        if r.get("content")
                    ]
                    if config.mode in ("codemem", "rag")
                    else []
                ),
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
    print(f"{'OVERALL':<25} {total_correct:>8} {total_questions:>8} {overall:>9.1f}%")
    if unscored_stats:
        n_unscored = sum(unscored_stats.values())
        print(f"\n  {n_unscored} question(s) EXCLUDED from the denominator above "
              f"({', '.join(sorted(unscored_stats))}).")
        print(f"  Abstention is an answerer property and is invisible to this")
        print(f"  retrieval-presence scorer, so no number is emitted for it in")
        print(f"  either direction. `recalled_memories` is still persisted, so an")
        print(f"  abstention judge can score these offline. Denominator now")
        print(f"  matches `qa_eval.py`, which already excluded the same set.")
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
    )
    run_benchmark(config)


if __name__ == "__main__":
    main()

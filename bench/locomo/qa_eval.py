#!/usr/bin/env python3
"""LoCoMo QA-generation eval — the offline answerer + answer-judge stage.

Turns kremory's retrieval-PROXY LoCoMo number into a field-COMPARABLE
QA-answer-generation number (the metric mem0 / Zep / etc. actually report).

Pipeline (all offline; reads memories already captured by the harness — no
re-ingest, no re-recall):
  answer-gen   <batch.jsonl | results.json> -o answers.jsonl
      For each question: take its recalled memories (sliced to k), call the
      ANSWERER LLM (gpt-4o-mini) with mem0's answer-generation prompt -> a
      generated answer string.
  answer-judge answers.jsonl -o verdicts.jsonl
      For each generated answer: call the JUDGE LLM (gpt-4o) with mem0's judge
      prompt over {question, gold, generated_answer} -> CORRECT/WRONG.
  answer-tally <batch.jsonl | results.json> --verdicts verdicts.jsonl
      QA-gen accuracy per category (+ overall), + honesty metadata.

Design mirrors judge_rescore.py (decoupled prepare/tally, stable cache_key,
idempotent resume) and reuses .context/judge_run.py's urllib call + retry +
ThreadPoolExecutor pattern. Transport is pure-stdlib urllib against
api.openai.com (OpenAI-compatible /chat/completions) — no SDK/venv needed; the
plan explicitly sanctions "point base_url at api.openai.com".

Verbatim mem0 prompts live in prompts_qa.py (provenance in its docstring).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# same-dir imports (script dir is on sys.path[0] when run directly)
import prompts_qa
from judge_rescore import ABSTENTION_CATEGORIES, cache_key, load_results
from provenance import (ProvenanceMismatch, ShippedDefaultsMismatch,
                        assert_provenance, assert_shipped_defaults)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Published per-1M-token rates (USD), for a rough cost estimate only. Verify
# against the live pricing page before quoting — these drift.
_RATES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    # verified live 2026-09-04 (OpenAI's own pricing page drifts — re-check before quoting)
    "gpt-5": (0.625, 5.00),
}

# ---------------------------------------------------------------------------
# Content-addressed response cache (VCR) — replay identical calls for $0.
# Key = sha256(model | system | user | json_mode). A cache HIT costs nothing
# and is deterministic, so re-tallies, format iterations, and re-runs over the
# SAME (model, prompts) never re-spend. NOTE: because a hit is an exact replay,
# caching is INCOMPATIBLE with measuring run-to-run LLM variance (Phase 3 N=5) —
# for a genuine independent sample, vary an input (fresh DB/recall) or --no-cache.
# ---------------------------------------------------------------------------
_CACHE_DIR: Path | None = None
_CACHE_LOCK = __import__("threading").Lock()
_CACHE_STATS = {"hit": 0, "miss": 0}


def _cache_path(model: str, system: str, user: str, json_mode: bool) -> Path | None:
    if _CACHE_DIR is None:
        return None
    h = hashlib.sha256(
        f"{model}\x1f{system}\x1f{user}\x1f{json_mode}".encode()
    ).hexdigest()
    return _CACHE_DIR / f"{h}.json"


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # fall back to a .env in cwd or any parent (repo root)
    here = Path.cwd()
    for d in [here, *here.parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(r"\s*OPENAI_API_KEY\s*=\s*(.+)", line)
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    print("no OPENAI_API_KEY in env or .env", file=sys.stderr)
    raise SystemExit(1)


def _chat(key: str, model: str, system: str, user: str, *,
          json_mode: bool = False, max_tokens: int = 1024,
          max_retry: int = 5) -> tuple[str, int, int]:
    """One chat.completions call via urllib. Returns (content, prompt_tok, completion_tok).

    Raises after `max_retry` exhausted so the caller can fail-loud (a missing
    answer/verdict must not be silently swallowed).
    """
    # cache HIT — deterministic replay for $0 (tokens returned as 0 so cost
    # accounting reflects FRESH spend only).
    cpath = _cache_path(model, system, user, json_mode)
    if cpath is not None and cpath.exists():
        try:
            c = json.loads(cpath.read_text())
            with _CACHE_LOCK:
                _CACHE_STATS["hit"] += 1
            return (c["content"], 0, 0)
        except Exception:  # noqa: BLE001 - corrupt cache entry -> refetch
            pass

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {"model": model, "messages": messages}
    # The gpt-5 family rejects both `max_tokens` (renamed `max_completion_tokens`)
    # and a non-default `temperature`. Added 2026-07-28 so we can run kremory
    # under Mem0's EXACT published protocol — their 92.5% uses gpt-5 as both
    # answerer and judge (verified from `mem0ai/memory-benchmarks` result
    # metadata, TD-161). Without this the whole batch 400s and reports zero
    # answers, which is what happened on the first attempt.
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        payload["max_completion_tokens"] = max_tokens
        if model.startswith("gpt-5"):
            # Reasoning-token exhaustion: gpt-5 bills hidden reasoning against the
            # SAME max_completion_tokens budget as the visible answer, and on a
            # non-trivial fraction of questions it can spend the whole budget
            # reasoning and return a valid response with EMPTY content — not an
            # error, just nothing to parse. Measured 2026-09-04 on a 15-question
            # smoke (a 15-question diagnostic, since discarded/re-run): 9/15 (60%) empty
            # answers, 2/15 unparsed judge verdicts (both raw-content == "" when
            # traced through the response cache). `reasoning_effort=minimal` is
            # OpenAI-documented as supported specifically for gpt-5/-mini/-nano
            # on Chat Completions to skip extensive reasoning for a task this
            # simple (a short factual answer / a CORRECT-WRONG label) — verified
            # against developers.openai.com/api/docs/guides/reasoning, not
            # assumed. See RECALL-LEDGER.md and locomo-benchmark-protocol
            # §4bis for the full writeup.
            payload["reasoning_effort"] = "minimal"
    else:
        payload["temperature"] = 0
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(
                OPENAI_URL, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json",
                         "User-Agent": "curl/8.4.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            content = d["choices"][0]["message"]["content"]
            usage = d.get("usage", {})
            ptok = usage.get("prompt_tokens", 0)
            ctok = usage.get("completion_tokens", 0)
            if cpath is not None:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = cpath.with_suffix(".tmp")
                tmp.write_text(json.dumps(
                    {"content": content, "prompt_tokens": ptok,
                     "completion_tokens": ctok, "model": model}))
                tmp.replace(cpath)  # atomic — no partial cache file on crash
                with _CACHE_LOCK:
                    _CACHE_STATS["miss"] += 1
            return (content, ptok, ctok)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            # 429 / 5xx are retryable; 4xx (except 429) is fatal
            if e.code not in (429, 500, 502, 503, 504):
                break
            time.sleep(min(2 ** attempt, 20))
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))
        except Exception as e:  # noqa: BLE001 - surface + backoff
            last = f"{type(e).__name__}: {e}"
            time.sleep(1 + attempt)
    raise RuntimeError(f"chat failed after {max_retry} tries: {last}")


_WARNED_UNPRICED: set[str] = set()


def _cost(model: str, ptok: int, ctok: int) -> float:
    """USD estimate, or 0.0 with a LOUD one-time warning for an unpriced model.

    Returning a silent 0.0 for an unknown model is the same defect class this
    harness hit twice on 2026-07-28 — a meter reporting a number it never
    measured. The gpt-5 run for the Mem0-protocol comparison printed
    `est $0.000` on ~4.8M tokens for exactly this reason. The value stays 0.0
    (there is no honest number to invent) but it can no longer pass as measured.
    """
    r = _RATES.get(model)
    if not r:
        if model not in _WARNED_UNPRICED:
            _WARNED_UNPRICED.add(model)
            print(
                f"[cost] WARNING: no rate for {model!r} — reported spend is NOT "
                f"measured and will read $0.000. Token counts below are real; the "
                f"dollar figure is absent, not zero. Add a rate to _RATES.",
                file=sys.stderr, flush=True,
            )
        return 0.0
    return ptok / 1e6 * r[0] + ctok / 1e6 * r[1]


# ---------------------------------------------------------------------------
# input loading — accept a prepared batch JSONL OR a raw harness results JSON
# ---------------------------------------------------------------------------
def gate_shipped_defaults(path: Path, *, allow_unverified: bool) -> None:
    """ROADMAP W0.2 — refuse to score a run that was not measured on the SHIPPED
    DEFAULT build.

    qa-gen is the only externally-quotable protocol (W0.3), so this is the gate
    that matters. ADR-078 is the failure it exists to prevent: a +30.2pt measured
    win sat above `content-search = []` for months, every published benchmark ran
    with the feature ON, and no consumer received it.

    `--allow-unverified-build` exists deliberately. A hard, unoverridable refusal
    on a diagnostic sweep would get the gate switched off, and a control people
    route around protects nothing — but the override prints a banner rather than
    passing quietly, so a non-default run can never be MISTAKEN for a quotable one.
    """
    if path.suffix != ".json":
        # A prepared .jsonl batch carries no stamp — it was derived from a results
        # JSON upstream. Say so; do not pretend the build was verified.
        print(
            "[W0.2] input is a prepared batch (.jsonl) — it carries no provenance "
            "stamp, so the BUILD IS UNVERIFIED here. Gate the upstream results "
            ".json instead if this number is going to be quoted.",
            file=sys.stderr,
        )
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[W0.2] could not read {path} for the build gate: {e}", file=sys.stderr)
        return
    stamp = data.get("provenance")
    if stamp is None:
        msg = (
            f"[W0.2] {path} carries NO provenance stamp — the build that produced "
            f"it is unknown, so the number is not attributable to any kremory "
            f"configuration."
        )
        if not allow_unverified:
            raise SystemExit(msg + "  Re-run the harness, or pass --allow-unverified-build.")
        print(msg + "  (--allow-unverified-build: continuing)", file=sys.stderr)
        return
    try:
        assert_shipped_defaults(stamp)
    except ShippedDefaultsMismatch as e:
        if not allow_unverified:
            raise SystemExit(
                f"{e}\n\nThis number is NOT quotable. Re-run on the default build, "
                f"or pass --allow-unverified-build to score it as a diagnostic."
            ) from e
        print(
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            " NON-DEFAULT BUILD — DIAGNOSTIC ONLY, NOT A QUOTABLE NUMBER\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"{e}\n",
            file=sys.stderr,
        )
        return
    print("[W0.2] build verified: run matches the shipped default feature set.",
          file=sys.stderr)


def parse_expect(pairs: list[str] | None) -> dict[str, object]:
    """`["rrf_k=60", "content_stream_weight=1.5"]` -> typed dict.

    Mirrors `provenance.py`'s own `_main` typing: int, then float, then str.
    """
    out: dict[str, object] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"[W0.3] bad --expect {pair!r}, want key=value")
        k, v = pair.split("=", 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def gate_recall_provenance(path: Path, expect: dict[str, object]) -> None:
    """W0.3 — refuse to score a recall file whose provenance stamp is not the
    SWEEP POINT the caller intended.

    `provenance.assert_provenance` has existed since the recall-improvement
    e2e spec (§S0-infra G1) and, until 2026-08-13, **had zero callers outside
    its own test and `__main__`.** It was written to stop exactly the
    13.9%-class stale-file defect and had never once run on a path that spends
    money.

    Deliberately INERT without `--expect`, per `over-blocking-is-a-security-
    failure`: an always-on sweep-point assertion would reject every legitimate
    historical results file (they were measured at other sweep points, which is
    the point of a sweep). Stamp PRESENCE and the shipped-default build check
    are already enforced unconditionally by `gate_shipped_defaults` above — this
    adds the *which configuration* half, and only when the caller states one.

    It earns its keep the moment two arms are graded in one session (e.g.
    library vs HTTP-shim fusion): crossing their result files is a silent,
    plausible, and completely undetectable error without it.
    """
    if not expect:
        return
    if path.suffix == ".jsonl":
        print("[W0.3] input is a prepared batch (.jsonl) and carries no "
              "provenance stamp — --expect cannot be checked here. Gate the "
              "upstream results .json instead.", file=sys.stderr)
        return
    stamp = assert_provenance(path, expect)   # raises ProvenanceMismatch
    print(f"[W0.3] sweep point verified: "
          f"{ {k: stamp.get(k) for k in expect} }", file=sys.stderr)


def load_batch(path: Path, *, allow_unverified: bool = False,
               expect: dict[str, object] | None = None) -> list[dict]:
    """Return answerable {key, sample_id, question_id, category, question, gold,
    memories} records. .jsonl = already-prepared batch (judge_rescore prepare
    shape); .json = raw harness results -> filter answerable + build keys."""
    gate_shipped_defaults(path, allow_unverified=allow_unverified)
    gate_recall_provenance(path, expect or {})
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # harness results JSON -> mirror judge_rescore.cmd_prepare filtering
    batch, n_abstain, n_no_mem = [], 0, 0
    for r in load_results([path]):
        category = r.get("category", "")
        if category in ABSTENTION_CATEGORIES:
            n_abstain += 1
            continue
        memories = r.get("recalled_memories")
        if memories is None:
            n_no_mem += 1
            continue
        gold = r.get("expected_answer", "")
        sample_id = r.get("sample_id", "")
        batch.append({
            "key": cache_key(sample_id, r["question_id"], gold, memories),
            "sample_id": sample_id, "question_id": r["question_id"],
            "category": category, "question": r.get("question", ""),
            "gold": gold, "memories": memories,
            # ADR-078 Phase A: index-aligned wire `kind` / `source_episode_id`
            # (TD-139). Absent on every pre-2026-07-28 run file, in which case
            # `--structured` degrades to the flat format rather than erroring.
            "provenance": r.get("recalled_memory_provenance") or [],
            # RECALL-LEDGER §4.19 / TD-155: kremory's own server-rendered
            # `format=text&template=temporal_facts` block for this question.
            # `None` on every pre-2026-08-10 run file and on any run not
            # started with `--capture-text-block` — `--text-block` degrades
            # to an ABORT (not a silent flat-format fallback; see
            # cmd_answer_gen) rather than scoring a run that never captured it.
            "text_block": r.get("recalled_text_block"),
        })
    if n_abstain or n_no_mem:
        print(f"[load] {len(batch)} answerable ({n_abstain} adversarial skipped, "
              f"{n_no_mem} missing recalled_memories)", file=sys.stderr)
    return batch


# ---------------------------------------------------------------------------
# G1 — deterministic sampling, and a sampled number that CANNOT be quoted.
#
# The readiness plan prescribes "smoke ~10 questions ($0.03), inspect, THEN the
# full run" (smoke-one-before-batch). Until 2026-08-13 there was NO FLAG that
# did it: the prescribed guard was an intention, not a command, and the only way
# to obey it was to hand-slice a batch file.
#
# Two halves, and the second is the load-bearing one:
#   1. `--sample-n` makes the smoke a command.
#   2. The sample marker travels answer-gen -> judge -> tally, and the tally
#      REFUSES to print a headline from sampled verdicts without `--allow-sampled`,
#      then brands the output NOT QUOTABLE. A 10-question number that reads like
#      a 1531-question number is worse than no smoke at all.
# ---------------------------------------------------------------------------

SAMPLE_KEY = "_sample"


def apply_sample(batch: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministically take `n` rows and stamp each with a sample marker.

    Determinism matters because the smoke must be REPEATABLE: re-running the
    same (n, seed) has to hit the same questions, or the cache never replays and
    a "free" re-run silently costs money again. Rows are sorted by `key` first
    so the selection does not depend on the input file's row order.

    `n` None or >= len(batch) returns the batch untouched and UNMARKED — a
    "sample" that is the whole population is not a sample, and marking it would
    brand a full run NOT QUOTABLE.
    """
    total = len(batch)
    if n is None or n >= total:
        if n is not None and n >= total:
            print(f"[sample] --sample-n {n} >= {total} available rows — running "
                  f"the FULL set, not marking it as sampled.", file=sys.stderr)
        return batch
    if n <= 0:
        raise SystemExit(f"[sample] --sample-n must be > 0, got {n}")
    ordered = sorted(batch, key=lambda b: b["key"])
    picked = random.Random(seed).sample(ordered, n)
    marker = {"n": n, "seed": seed, "of_total": total}
    for b in picked:
        b[SAMPLE_KEY] = marker
    print(f"[sample] SMOKE RUN — {n} of {total} questions (seed={seed}). "
          f"Any number from this run is a DIAGNOSTIC, not a result.",
          file=sys.stderr)
    return picked


def sample_marker_of(rows: list[dict]) -> dict | None:
    """The sample marker carried by `rows`, or None. Used by the tally to detect
    sampling from the VERDICTS side — the tally's own batch comes from the full
    results JSON and therefore never carries one."""
    for r in rows:
        m = r.get(SAMPLE_KEY)
        if m:
            return m
    return None


def _load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()}


# ---------------------------------------------------------------------------
# answer-gen
# ---------------------------------------------------------------------------
def _extract_answer(text: str) -> str:
    """The answerer ends with 'ANSWER:' — take the final segment (mem0 parity)."""
    if "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[-1].strip()
    return text.strip()


def cmd_answer_gen(a: argparse.Namespace) -> int:
    key = _load_api_key()
    batch = load_batch(a.input,
                       allow_unverified=getattr(a, 'allow_unverified_build', False),
                       expect=parse_expect(getattr(a, 'expect', None)))
    # Sample BEFORE the context-format flags suffix `key` below, so the same
    # (n, seed) selects the same QUESTIONS whichever format is under test —
    # otherwise a --structured smoke and a flat smoke would compare different
    # questions and the A/B would measure the sample, not the format.
    batch = apply_sample(batch, getattr(a, "sample_n", None),
                         getattr(a, "sample_seed", 1234))
    structured = getattr(a, "structured", False)
    text_block_flag = getattr(a, "text_block", False)
    if structured and text_block_flag:
        print("[answer-gen] ABORT: --structured and --text-block are mutually "
              "exclusive context sources (grouped-by-kind reconstruction vs "
              "kremory's own server-rendered block) — pick one.",
              file=sys.stderr)
        return 2
    if structured:
        # The cache key must include the CONTEXT FORMAT, or a structured run
        # silently replays the flat run's cached answers and the A/B reads null.
        # That is the same class of defect as a probe reporting a number it did
        # not measure — make it impossible rather than remembering not to.
        n_prov = sum(1 for b in batch if b.get("provenance"))
        for b in batch:
            b["key"] = f"{b['key']}|structured"
        print(f"[answer-gen] STRUCTURED context: {n_prov}/{len(batch)} rows carry "
              f"provenance", flush=True)
        if n_prov == 0:
            print("[answer-gen] ABORT: --structured requested but NO row carries "
                  "`recalled_memory_provenance` — every prompt would silently fall "
                  "back to the flat format and the A/B would measure nothing. "
                  "Re-run the harness to capture provenance first.", file=sys.stderr)
            return 2
    if text_block_flag:
        # Same discipline as --structured immediately above (RECALL-LEDGER
        # §4.19 / TD-155): the cache key MUST carry the context format, or a
        # --text-block run silently replays --structured's (or the flat
        # run's) cached dateless answers under a new label — the exact
        # failure this suffix scheme exists to make impossible.
        n_tb = sum(1 for b in batch if b.get("text_block"))
        for b in batch:
            b["key"] = f"{b['key']}|textblock"
        print(f"[answer-gen] TEXT-BLOCK context: {n_tb}/{len(batch)} rows carry "
              f"recalled_text_block", flush=True)
        if n_tb == 0:
            print("[answer-gen] ABORT: --text-block requested but NO row carries "
                  "`recalled_text_block` — every prompt would silently fall back "
                  "to the flat format and the A/B would measure nothing. "
                  "Re-run the harness with --capture-text-block first.",
                  file=sys.stderr)
            return 2
    chronological = getattr(a, "chrono", False)
    if chronological:
        # SAME cache-key discipline as --structured / --text-block above. The
        # ordering of memories IS the context format here: without the suffix a
        # --chrono run replays the rank-ordered run's cached answers and the A/B
        # reads a false null — the exact "measured nothing, recorded a result"
        # failure the sibling suffixes exist to prevent.
        n_dated = sum(
            1 for b in batch
            if any(prompts_qa._inline_date_key(m) is not None
                   for m in (b.get("memories") or []))
        )
        for b in batch:
            b["key"] = f"{b['key']}|chrono"
        print(f"[answer-gen] CHRONOLOGICAL context: {n_dated}/{len(batch)} rows "
              f"carry at least one inline-dated memory", flush=True)
        if n_dated == 0:
            print("[answer-gen] ABORT: --chrono requested but NO row carries an "
                  "inline-dated memory — every prompt would be byte-identical to "
                  "the rank-ordered format and the A/B would measure nothing.",
                  file=sys.stderr)
            return 2
    done = _load_done_keys(a.output) if a.resume else set()
    todo = [b for b in batch if b["key"] not in done]
    print(f"[answer-gen] {len(batch)} questions, {len(todo)} to do "
          f"({len(done)} cached), k={a.k}, model={a.model}, "
          f"concurrency={a.concurrency}", flush=True)
    if not todo:
        return 0

    def work(b: dict) -> dict:
        mems = (b.get("memories") or [])[: a.k]
        prompt = prompts_qa.build_answer_prompt(
            b["question"], mems, reference_date=a.reference_date,
            provenance=(b.get("provenance") or [])[: a.k],
            structured=structured,
            # Deliberately NOT sliced to `a.k` — this is ONE pre-rendered
            # server string (F5: `format=text` returns `{block: string}`,
            # not per-item rows), rendered against the harness's own
            # per-category recall limit at capture time, not this offline
            # replay's `--k`. `None` when the flag is off, which routes
            # `build_answer_prompt` back to `structured`/flat unchanged.
            text_block=(b.get("text_block") if text_block_flag else None),
            chronological=chronological)
        content, ptok, ctok = _chat(key, a.model, "", prompt, max_tokens=a.max_tokens)
        return {"key": b["key"], "sample_id": b.get("sample_id", ""),
                "question_id": b["question_id"], "category": b.get("category", ""),
                "question": b["question"], "gold": b.get("gold", ""),
                "generated_answer": _extract_answer(content),
                "raw": content if a.keep_raw else None,
                "k": a.k, "answerer_model": a.model,
                # G1: the sample marker must SURVIVE into the answers file, or
                # the judge and the tally have no way to know this run was a
                # smoke — and an unmarked 10-question number is indistinguishable
                # from a full-corpus one.
                **({SAMPLE_KEY: b[SAMPLE_KEY]} if b.get(SAMPLE_KEY) else {}),
                "_ptok": ptok, "_ctok": ctok}

    a.output.parent.mkdir(parents=True, exist_ok=True)
    t0, done_n, fails = time.time(), 0, 0
    pt = ct = 0
    mode = "a" if (a.resume and a.output.exists()) else "w"
    with open(a.output, mode) as f, ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(work, b): b for b in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"[answer-gen] FAIL {futs[fut]['key']}: {e}", file=sys.stderr)
                continue
            pt += rec.pop("_ptok"); ct += rec.pop("_ctok")
            f.write(json.dumps({k: v for k, v in rec.items() if v is not None},
                               ensure_ascii=False) + "\n")
            f.flush()
            done_n += 1
            if done_n % 20 == 0 or done_n == len(todo):
                print(f"[answer-gen] {done_n}/{len(todo)}  {time.time()-t0:.0f}s  "
                      f"~${_cost(a.model, pt, ct):.3f}", flush=True)
    print(f"[answer-gen] wrote {done_n} answers to {a.output} "
          f"({fails} failed) in {time.time()-t0:.0f}s  "
          f"tokens in/out={pt}/{ct}  est ${_cost(a.model, pt, ct):.3f}  "
          f"cache hit/miss={_CACHE_STATS['hit']}/{_CACHE_STATS['miss']}", flush=True)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# answer-judge
# ---------------------------------------------------------------------------
def _parse_label(content: str) -> tuple[bool | None, str]:
    """Return (correct, reason). correct=None when the label can't be parsed."""
    label, reason = None, ""
    try:
        obj = json.loads(content)
        reason = str(obj.get("reasoning", ""))[:300]
        raw = str(obj.get("label", "")).strip().upper()
        if "CORRECT" in raw and "WRONG" not in raw:
            label = True
        elif "WRONG" in raw:
            label = False
    except Exception:  # noqa: BLE001 - fall back to regex on the text
        up = content.upper()
        if "WRONG" in up and "CORRECT" not in up:
            label = False
        elif "CORRECT" in up:
            label = True
        reason = content[:300]
    return label, reason


def cmd_answer_judge(a: argparse.Namespace) -> int:
    key = _load_api_key()
    answers = [json.loads(l) for l in a.input.read_text().splitlines() if l.strip()]
    done = _load_done_keys(a.output) if a.resume else set()
    todo = [r for r in answers if r["key"] not in done]
    print(f"[answer-judge] {len(answers)} answers, {len(todo)} to do "
          f"({len(done)} cached), model={a.model}, concurrency={a.concurrency}",
          flush=True)
    if not todo:
        return 0

    def work(r: dict) -> dict:
        prompt = prompts_qa.build_judge_prompt(
            r["question"], r.get("gold", ""), r.get("generated_answer", ""),
            r.get("category", ""))
        # 300 is enough headroom for gpt-4o-*'s short {"label", "reasoning"} JSON.
        # gpt-5 gets more even with reasoning_effort=minimal (set in _chat) as a
        # defense-in-depth margin against the same exhaustion bug — see _chat's
        # comment on the 2026-09-04 smoke finding.
        judge_max_tokens = 600 if a.model.startswith("gpt-5") else 300
        content, ptok, ctok = _chat(key, a.model, prompts_qa.JUDGE_SYSTEM_PROMPT,
                                    prompt, json_mode=True, max_tokens=judge_max_tokens)
        correct, reason = _parse_label(content)
        return {"key": r["key"], "question_id": r["question_id"],
                "category": r.get("category", ""),
                "correct": correct, "reason": reason,
                "judge_model": a.model,
                # G1: propagate, do not re-sample. The judge deliberately has NO
                # `--sample-n`: its input is already whatever answer-gen produced,
                # so a second sampling stage could only compound into a subset
                # nobody chose. To smoke the JUDGE alone, slice its input file
                # (`head -n 10 answers.jsonl`) — that is honest and visible.
                **({SAMPLE_KEY: r[SAMPLE_KEY]} if r.get(SAMPLE_KEY) else {}),
                "_ptok": ptok, "_ctok": ctok}

    a.output.parent.mkdir(parents=True, exist_ok=True)
    t0, done_n, unparsed, fails = time.time(), 0, 0, 0
    pt = ct = 0
    mode = "a" if (a.resume and a.output.exists()) else "w"
    with open(a.output, mode) as f, ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"[answer-judge] FAIL {futs[fut]['key']}: {e}", file=sys.stderr)
                continue
            pt += rec.pop("_ptok"); ct += rec.pop("_ctok")
            if rec["correct"] is None:
                unparsed += 1
                print(f"[answer-judge] UNPARSED label {rec['key']}", file=sys.stderr)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done_n += 1
            if done_n % 20 == 0 or done_n == len(todo):
                print(f"[answer-judge] {done_n}/{len(todo)}  {time.time()-t0:.0f}s  "
                      f"~${_cost(a.model, pt, ct):.3f}", flush=True)
    print(f"[answer-judge] wrote {done_n} verdicts to {a.output} "
          f"({unparsed} unparsed, {fails} failed) in {time.time()-t0:.0f}s  "
          f"tokens in/out={pt}/{ct}  est ${_cost(a.model, pt, ct):.3f}  "
          f"cache hit/miss={_CACHE_STATS['hit']}/{_CACHE_STATS['miss']}", flush=True)
    # ⚠️ This was `return 0` unconditionally, and call failures were not even
    # counted — so a judge pass in which EVERY call 401'd/429'd/timed out wrote
    # zero verdicts and reported SUCCESS. Found 2026-08-13 while proving the
    # smoke flow end to end.
    #
    # It is RECALL-LEDGER §8 method rule 13, which this repo earned the hard way:
    # "A probe that can print a number when it measured NOTHING is a defect, not
    # an instrument ... a total outage must never be reportable as a null
    # result." The doc2query probe reported a clean 0.0% after every LLM call
    # 403'd. Same shape here.
    #
    # `cmd_answer_gen` already got this right (`return 1 if fails else 0`); the
    # judge was the asymmetric half. Downstream, `answer-tally`'s >50%-unjudged
    # abort would eventually catch the empty case — but "a later stage happens to
    # notice" is not the same as this stage telling the truth about its own run,
    # and an operator watching a `&&` chain sees only the exit code.
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# answer-tally
# ---------------------------------------------------------------------------
def cmd_answer_tally(a: argparse.Namespace) -> int:
    batch = load_batch(a.input,
                       allow_unverified=getattr(a, 'allow_unverified_build', False),
                       expect=parse_expect(getattr(a, 'expect', None)))
    verdicts: dict[str, dict] = {}
    for l in a.verdicts.read_text().splitlines():
        if l.strip():
            v = json.loads(l)
            verdicts[v["key"]] = v
    # `answer-gen --structured` suffixes its cache keys so the two context
    # formats cannot share cached answers. The tally MUST apply the same suffix
    # or every row is "unjudged" and the report prints 0.0% — which is what
    # happened on this flag's first use (2026-07-28). Detected rather than
    # declared: if none of the plain keys match but the suffixed ones do, use
    # those, so the caller cannot get it wrong by forgetting a flag.
    if verdicts and not any(b["key"] in verdicts for b in batch):
        if any(f"{b['key']}|structured" in verdicts for b in batch):
            for b in batch:
                b["key"] = f"{b['key']}|structured"
            print("[answer-tally] matched STRUCTURED verdict keys", file=sys.stderr)
        elif any(f"{b['key']}|textblock" in verdicts for b in batch):
            for b in batch:
                b["key"] = f"{b['key']}|textblock"
            print("[answer-tally] matched TEXT-BLOCK verdict keys", file=sys.stderr)

    # ---- G1: sampled runs are DIAGNOSTICS, never headlines ------------------
    # The tally's `batch` comes from the FULL results JSON, so it never carries a
    # sample marker; the verdicts do (answer-gen -> judge propagation). Detect
    # from that side, restrict the batch to the sampled keys, and fail CLOSED.
    #
    # Without the restriction the existing >50%-unjudged abort at the bottom of
    # this function fires on every smoke (10 verdicts vs 1531 rows) — correct,
    # but it means a smoke has no reporting step at all, which is why the
    # prescribed "inspect, THEN the full run" step had nowhere to land.
    sample = sample_marker_of(list(verdicts.values()))
    if sample:
        if not getattr(a, "allow_sampled", False):
            print(
                f"\nABORT: these verdicts come from a SAMPLED run "
                f"({sample['n']} of {sample['of_total']} questions, "
                f"seed={sample['seed']}).\n"
                f"       A sampled accuracy is a DIAGNOSTIC and must never be "
                f"quoted as a corpus number.\n"
                f"       Pass --allow-sampled to print it, branded NOT QUOTABLE.",
                file=sys.stderr)
            return 2
        judged_keys = set(verdicts)
        batch = [b for b in batch if b["key"] in judged_keys]
        if not batch:
            print("\nABORT: sampled verdicts matched NO row in the batch — the "
                  "verdicts belong to a different run.", file=sys.stderr)
            return 2

    cats: dict[str, list[int]] = {}     # category -> [correct, total]
    unjudged, unparsed = [], 0
    for b in batch:
        cat = b.get("category", "")
        row = cats.setdefault(cat, [0, 0])
        row[1] += 1
        v = verdicts.get(b["key"])
        if v is None:
            unjudged.append(f"{b.get('sample_id','')}/{b['question_id']}")
            continue
        if v.get("correct") is None:
            unparsed += 1
            continue  # unparsed = not-correct (fail-loud, depresses the number)
        if v.get("correct"):
            row[0] += 1

    # Instrument validation BEFORE any number is printed. A tally where most
    # rows found no verdict has measured nothing, and MUST NOT render a score —
    # an unmatched join and a genuinely-wrong system look identical on the
    # output (0.0% across every category). Same defect class as a probe
    # reporting 0.0% after its LLM calls all failed; both were real, same day.
    if batch and len(unjudged) > len(batch) // 2:
        print(f"\nABORT: {len(unjudged)}/{len(batch)} rows have NO matching verdict "
              f"— this tally measured NOTHING and will not print a score.\n"
              f"       The verdicts file almost certainly belongs to a different run, "
              f"or was generated with a different --structured setting.\n"
              f"       first unmatched: {unjudged[:3]}", file=sys.stderr)
        return 2

    print("=" * 66)
    if sample:
        print("⚠️  SAMPLED RUN — NOT QUOTABLE ⚠️")
        print(f"    {sample['n']} of {sample['of_total']} questions "
              f"(seed={sample['seed']}). This is a smoke DIAGNOSTIC.")
        print("=" * 66)
    print(f"LoCoMo QA-GEN accuracy [{'SAMPLE' if sample else 'HEADLINE'}] "
          f"(answerer={a.answerer_label}, "
          f"judge={a.judge_label}, k={a.k_label})")
    print("NB: the harness inline SUBSTRING scorer is a separate, much stricter "
          "floor\n    (~2x under-credits, esp. open-domain) — do NOT confuse it "
          "with this number.")
    print("=" * 66)
    print(f"{'category':<14} {'correct':>8} {'total':>6} {'acc':>8}")
    print("-" * 66)
    per_cat = {}
    tc = tt = 0
    for cat in sorted(cats):
        c, t = cats[cat]
        tc += c; tt += t
        acc = c / t * 100 if t else 0.0
        per_cat[cat] = {"correct": c, "total": t, "accuracy_pct": round(acc, 2)}
        print(f"{cat:<14} {c:>8} {t:>6} {acc:>7.1f}%")
    print("-" * 66)
    overall = tc / tt * 100 if tt else 0.0
    print(f"{'OVERALL':<14} {tc:>8} {tt:>6} {overall:>7.1f}%")
    print("=" * 66)
    if unjudged:
        print(f"\n[WARN] {len(unjudged)} question(s) have NO verdict (counted "
              f"incorrect). First few: {unjudged[:5]}", file=sys.stderr)
    if unparsed:
        print(f"[WARN] {unparsed} verdict(s) had an unparseable label "
              f"(counted incorrect).", file=sys.stderr)

    # ---- first-class o11y: emit a machine-readable summary JSON so the JUDGED
    # number is a durable metric artifact, not a printed table (per the "the
    # judged number must emit metrics/o11y" gap). ----
    summary = {
        # G1: a SAMPLED run gets a DIFFERENT metric name, so a downstream
        # consumer cannot mistake a smoke for the corpus number by reading the
        # JSON alone. Same discipline as branding the printed table.
        "metric": ("locomo_qa_gen_accuracy_SAMPLE" if sample
                   else "locomo_qa_gen_accuracy"),
        "quotable": not sample,
        **({"sample": sample} if sample else {}),
        "scorer": "qa-gen (answerer+judge)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"answerer": a.answerer_label, "judge": a.judge_label,
                   "k": a.k_label, "input": str(a.input),
                   "verdicts": str(a.verdicts)},
        "overall": {"correct": tc, "total": tt,
                    "accuracy_pct": round(overall, 2)},
        "per_category": per_cat,
        "integrity": {"unjudged": len(unjudged), "unparsed_labels": unparsed,
                      "note": "unjudged + unparsed are counted incorrect (fail-loud)"},
    }
    out = a.summary or a.verdicts.with_name(a.verdicts.stem + "-summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"[o11y] summary metric written -> {out}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("answer-gen", help="generate answers from recalled memories")
    g.add_argument("input", type=Path, help="batch .jsonl or harness results .json")
    g.add_argument("-o", "--output", type=Path, required=True)
    g.add_argument("--model", default="gpt-4o-mini")
    g.add_argument("--k", type=int, default=10, help="top-k memories to feed")
    g.add_argument("--reference-date", default="2023")
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--keep-raw", action="store_true", help="store full CoT text")
    g.add_argument("--allow-unverified-build", action="store_true",
                   help="score a run measured on a NON-default build as a "
                        "diagnostic. Prints a banner; the number is not quotable "
                        "(ROADMAP W0.2).")
    g.add_argument("--structured", action="store_true",
                   help="group the answerer's context by wire `kind` (facts / "
                        "entities / conversation excerpts) instead of one flat "
                        "list; requires `recalled_memory_provenance` on the run")
    g.add_argument("--chrono", action="store_true",
                   help="order dated memories CHRONOLOGICALLY in the answerer "
                        "context, as mem0's reference harness does (it sorts its "
                        "dict-memories by created_at; our adaptation fed them in "
                        "retrieval-rank order instead). Default OFF so historical "
                        "qa-gen numbers are not silently re-based.")
    g.add_argument("--text-block", action="store_true",
                   help="use kremory's own server-rendered prompt-ready block "
                        "(`recalled_text_block` — format=text&template="
                        "temporal_facts, RECALL-LEDGER §4.19) as the ENTIRE "
                        "answerer context, verbatim, instead of reconstructing "
                        "one harness-side; requires the harness to have been "
                        "run with --capture-text-block. Mutually exclusive "
                        "with --structured.")
    g.add_argument("--expect", action="append", metavar="KEY=VALUE",
                   help="W0.3: assert the recall file's provenance stamp matches "
                        "this sweep point (repeatable, e.g. --expect rrf_k=60). "
                        "Refuses to score on mismatch.")
    g.add_argument("--sample-n", type=int, default=None,
                   help="SMOKE: answer only N questions, chosen deterministically. "
                        "Marks the output so the tally refuses to headline it.")
    g.add_argument("--sample-seed", type=int, default=1234,
                   help="seed for --sample-n; same (n, seed) picks the same "
                        "questions, so a re-run replays from cache for $0")
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--cache-dir", default="results/qa/.cache",
                   help="content-addressed response cache (VCR); replays identical calls for $0")
    g.add_argument("--no-cache", dest="cache", action="store_false")
    g.set_defaults(resume=True, cache=True, func=cmd_answer_gen)

    j = sub.add_parser("answer-judge", help="judge generated answers vs gold")
    j.add_argument("input", type=Path, help="answers.jsonl from answer-gen")
    j.add_argument("-o", "--output", type=Path, required=True)
    j.add_argument("--model", default="gpt-4o")
    j.add_argument("--concurrency", type=int, default=8)
    j.add_argument("--no-resume", dest="resume", action="store_false")
    j.add_argument("--cache-dir", default="results/qa/.cache",
                   help="content-addressed response cache (VCR); replays identical calls for $0")
    j.add_argument("--no-cache", dest="cache", action="store_false")
    j.set_defaults(resume=True, cache=True, func=cmd_answer_judge)

    t = sub.add_parser("answer-tally", help="QA-gen accuracy per category")
    t.add_argument("input", type=Path, help="batch .jsonl or harness results .json")
    t.add_argument("--allow-unverified-build", action="store_true",
                   help="tally a run whose build could not be verified (no stamp) or "
                        "does not match the declared defaults. Prints a banner; the "
                        "number is not quotable (ROADMAP W0.2).")
    t.add_argument("--verdicts", type=Path, required=True)
    t.add_argument("--expect", action="append", metavar="KEY=VALUE",
                   help="W0.3: assert the recall file's provenance stamp matches "
                        "this sweep point (repeatable). Refuses to tally on "
                        "mismatch — the guard against crossing two arms' files.")
    t.add_argument("--allow-sampled", action="store_true",
                   help="permit tallying verdicts from a --sample-n smoke run. "
                        "The output is branded NOT QUOTABLE and the summary JSON "
                        "carries a different metric name.")
    t.add_argument("--answerer-label", default="gpt-4o-mini")
    t.add_argument("--judge-label", default="gpt-4o")
    t.add_argument("--k-label", default="10")
    t.add_argument("--summary", type=Path, default=None,
                   help="machine-readable metric JSON (default: <verdicts>-summary.json)")
    t.set_defaults(func=cmd_answer_tally)

    args = p.parse_args()
    global _CACHE_DIR
    if getattr(args, "cache", False) and getattr(args, "cache_dir", None):
        _CACHE_DIR = Path(args.cache_dir)
    try:
        return args.func(args)
    except ProvenanceMismatch as e:
        # W0.3: a refusal, not a crash. A traceback reads as "the tool broke"
        # and invites a retry with the gate disabled; a clean non-zero exit with
        # the mismatch spelled out reads as "you are about to score the wrong
        # file", which is what actually happened.
        print(f"\n[W0.3] REFUSING TO SCORE — provenance mismatch:\n{e}",
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

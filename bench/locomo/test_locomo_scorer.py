"""Regression guards for the LoCoMo substring scorer.

These exist because of a defect class this repo has now hit FIVE times: an
ABSENCE (a missing field, a skipped file, an unmeasurable category) read as a
MEASUREMENT. The specific instance guarded here scored 446/446 adversarial
questions "correct" for months — category-5 gold lives under
`adversarial_answer`, `extract_questions` read `answer` with a `""` default,
and `"" in anything` is `True`.

Two of the four tests drive the REAL corpus rather than a hand-built fixture,
deliberately: a fixture encodes the author's mental model of the data, which is
the same model that produced the bug. Only the real file can disagree with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

# Both bench harnesses are named `harness`, so a plain `import harness` binds to
# whichever directory happens to be first on sys.path — the other suite then
# silently tests the WRONG module. Load this one by explicit path under a unique
# name so the suites can run together.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))
_spec = _ilu.spec_from_file_location(
    "_locomo_harness", _Path(__file__).resolve().parent / "harness.py"
)
harness = _ilu.module_from_spec(_spec)
_sys.modules["_locomo_harness"] = harness
_spec.loader.exec_module(harness)

DATA = Path(__file__).parent / "data" / "locomo10.json"


def _corpus() -> list[dict]:
    if not DATA.exists():
        pytest.skip(f"corpus not present: {DATA}")
    return json.loads(DATA.read_text())


def test_empty_gold_raises_instead_of_matching_everything() -> None:
    """`"" in anything` is True — an empty gold must never reach the scorer.

    This is the exact shape of the original bug: absence of a gold silently
    became a perfect score.
    """
    with pytest.raises(ValueError, match="empty gold"):
        harness.check_answer_in_memories("", ["any retrieved text at all"], "single-hop")
    with pytest.raises(ValueError, match="empty gold"):
        harness.check_answer_in_memories("   ", [], "temporal")


def test_abstention_category_is_refused_not_scored() -> None:
    """Abstention is an answerer property; a presence metric cannot see it.

    Refusal is structural (in the function) rather than a caller convention,
    so a future call site cannot reintroduce the number by forgetting a guard.
    """
    for category in sorted(harness.ABSTENTION_CATEGORIES):
        with pytest.raises(ValueError, match="abstention category"):
            harness.check_answer_in_memories("self-care is important", ["x"], category)


def test_every_corpus_question_yields_a_non_empty_gold() -> None:
    """Real corpus, not a fixture. 446 of 1986 questions carry their gold under
    a DIFFERENT key, which is what the original default silently swallowed."""
    seen = 0
    for sample in _corpus():
        for q in harness.extract_questions(sample):
            seen += 1
            assert str(q["answer"]).strip(), (
                f"empty gold survived extraction for {q['category']} "
                f"question {q['question_id']}"
            )
    assert seen == 1986, f"expected the full LoCoMo corpus, got {seen} questions"


def test_adversarial_gold_comes_from_the_adversarial_answer_field() -> None:
    """Real corpus. Asserts the distractor is carried through AND that scoring
    it is refused — carrying it is for offline abstention judging only."""
    advs = [
        q
        for sample in _corpus()
        for q in harness.extract_questions(sample)
        if q["category"] == "adversarial"
    ]
    assert len(advs) == 446, f"expected 446 adversarial questions, got {len(advs)}"

    raw = {
        (s.get("sample_id", ""), f"q_{i}"): q
        for s in _corpus()
        for i, q in enumerate(s.get("qa", []))
    }
    # Spot-check the join on the canonical example: the charity race was
    # MELANIE's, so "What did Caroline realize after her charity race?" is a
    # speaker-attribution false premise and "self-care is important" is the
    # distractor, not the gold.
    assert any(
        q["answer"] == "self-care is important" for q in advs
    ), "known distractor missing — extraction is not reading `adversarial_answer`"

    for q in advs:
        assert str(q["answer"]).strip()
        with pytest.raises(ValueError):
            harness.check_answer_in_memories(q["answer"], ["some text"], q["category"])


def test_shared_client_identity_is_not_shadowed() -> None:
    """Both harnesses must use the SAME classes from bench/common.

    Guards a bug introduced and caught during the 2026-07-28 de-fork: the
    harness kept a local `class KremoryStalled` while the shared client raised
    its own, producing two DISTINCT classes. `except KremoryStalled` then
    silently stops catching a stalled ingest — the fail-fast path goes dead
    while still looking present in the source. Identity, not just name.
    """
    import kremory_client

    assert harness.CodememClient is kremory_client.CodememClient
    assert harness.KremoryStalled is kremory_client.KremoryStalled
    assert harness.INGEST_BUDGET_S == kremory_client.INGEST_BUDGET_S


# ── TD-187: temporal grounding — the session anchor must reach kremory ────────


def test_parse_session_datetime_emits_utc_offset() -> None:
    """The anchor MUST carry a UTC offset or kremory rejects every store.

    kremory parses `published_at` with `DateTime::parse_from_rfc3339`
    (`crates/kremory-mcp/src/conversions.rs:150`), which REJECTS a naive
    timestamp; the accepted shape is pinned by `conversions.rs:551`. A first
    draft of `parse_session_datetime` returned `.isoformat()` on a naive
    datetime — every store would have 4xx'd and the whole bench run would have
    been lost. This test is the tripwire for that regression.
    """
    import re

    got = harness.parse_session_datetime("1:56 pm on 8 May, 2023")
    assert got is not None
    assert re.search(r"([+-]\d{2}:\d{2}|Z)$", got), f"no UTC offset: {got}"
    assert got.startswith("2023-05-08T13:56:00")


def test_parse_session_datetime_returns_none_not_now_on_failure() -> None:
    """Unparseable MUST stay None — never a wall-clock substitute.

    Load-bearing, not cosmetic: kremory renders the anchor into the extraction
    prompt ONLY when the caller declared one, because the VCR fingerprint
    (`core/provider/record_replay.rs:225-263`) hashes the message list. A
    wall-clock fallback here would change the prompt on every run and make all
    303 chat cassettes permanently un-replayable.
    """
    assert harness.parse_session_datetime("") is None
    assert harness.parse_session_datetime("   ") is None
    assert harness.parse_session_datetime("not a date at all") is None


def test_parse_session_datetime_covers_the_whole_real_corpus() -> None:
    """Every session header in the shipped corpus must parse — 272 of 272.

    Drives the REAL corpus, not a hand-made fixture: a hand-shaped string would
    only validate the author's mental model of the format
    (`smoke-one-before-batch-llm-validation`, subagent-fixture clause).
    """
    import json
    from pathlib import Path

    corpus = Path(__file__).parent / "data" / "locomo10.json"
    if not corpus.exists():
        pytest.skip("locomo10.json not present")

    data = json.loads(corpus.read_text())
    convs = data if isinstance(data, list) else [data]
    total = 0
    parsed = 0
    for conv in convs:
        for sess in harness.extract_sessions(conv):
            total += 1
            if harness.parse_session_datetime(sess["datetime"]):
                parsed += 1

    assert total > 0, "corpus yielded no sessions — extract_sessions broke"
    assert parsed == total, f"only {parsed} of {total} session headers parsed"

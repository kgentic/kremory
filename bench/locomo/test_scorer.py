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

import harness

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

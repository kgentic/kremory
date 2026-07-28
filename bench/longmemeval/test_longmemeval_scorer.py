"""Regression guards for the LongMemEval answerer + scorer.

Guards the SIXTH instance of this repo's recurring defect class: an ABSENCE
read as a MEASUREMENT.

The instance: `generate_answer` returned the literal "I don't know." on any LLM
exception. "i don't know" is the first entry in ABSTENTION_PHRASES, and
`quick_score` scores an `_abs` question correct iff the answer contains an
abstention phrase. So every OpenAI outage, rate-limit or bad key scored CORRECT
on all 30 abstention questions — a failure rendered as a perfect score, on a
PAID run.
"""
from __future__ import annotations

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
    "_lme_harness", _Path(__file__).resolve().parent / "harness.py"
)
harness = _ilu.module_from_spec(_spec)
_sys.modules["_lme_harness"] = harness
_spec.loader.exec_module(harness)


def test_answer_failure_raises_rather_than_faking_an_abstention() -> None:
    """The old code returned "I don't know." here. That string is scored as a
    correct abstention, so the failure was invisible."""

    class BoomClient:
        class chat:  # noqa: N801 - mimics the OpenAI client shape
            class completions:
                @staticmethod
                def create(**_kw):
                    raise RuntimeError("simulated OpenAI outage")

    with pytest.raises(harness.AnswerGenerationFailed):
        harness.generate_answer(
            BoomClient(), "what did I say?", ["[Memory 1] some text"], "2023/06/01", "gpt-4o"
        )


def test_the_old_failure_string_scores_as_an_abstention_now() -> None:
    """Pins WHY the answerer must not fabricate this string.

    Historical note, because it is easy to get backwards: before 2026-07-28
    this did NOT score correct — `normalize_text` turns "I don't know." into
    "i don t know", while ABSTENTION_PHRASES were compared UN-normalized, so
    the apostrophe entries never matched. That second bug MASKED the first.
    Now that the comparison is symmetric, the string is a valid abstention —
    which is exactly why `generate_answer` must raise instead of returning it.
    """
    assert harness.quick_score("I don't know.", "gold", "q_abs")["is_correct"] is True


def test_abstention_phrases_are_stable_under_normalization() -> None:
    """Structural guard on the asymmetry itself.

    Both sides of the membership test must be normalized with the SAME
    function. Comparing a raw phrase list against a normalized hypothesis
    silently disabled 3 of 14 phrases — including the most common one.
    """
    for raw, norm in zip(harness.ABSTENTION_PHRASES, harness.ABSTENTION_PHRASES_NORMALIZED):
        assert norm == harness.normalize_text(raw)
        assert harness.normalize_text(norm) == norm, (
            f"{raw!r} -> {norm!r} is not normalization-stable; the membership "
            "test would still be asymmetric"
        )
    # And the end-to-end property that actually matters.
    for raw in harness.ABSTENTION_PHRASES:
        assert harness.quick_score(f"Sorry, {raw}.", "gold", "q_abs")["is_correct"] is True, (
            f"abstention phrase {raw!r} does not match its own normalized form"
        )


def test_zero_recall_is_not_credited_as_an_abstention() -> None:
    """Empty retrieval is a RETRIEVAL failure, not a demonstrated abstention:
    the system never had content to reason over and correctly reject."""
    verdict = harness.quick_score(harness.NO_RECALL_MARKER, "gold", "q_abs")
    assert verdict["is_correct"] is False
    assert "retrieval failure" in verdict["explanation"]


def test_no_recall_marker_contains_no_abstention_phrase() -> None:
    """Structural: the marker must not be mistakable for an abstention.

    Guards against someone later rewording it to something like
    "no information found", which ABSTENTION_PHRASES would match.
    """
    lowered = harness.NO_RECALL_MARKER.lower()
    for phrase in harness.ABSTENTION_PHRASES:
        assert phrase not in lowered, f"marker contains abstention phrase {phrase!r}"


def test_empty_hypothesis_is_refused_not_scored() -> None:
    """`"" in anything` is True — an empty hypothesis would score a false
    substring match AND a false abstention. Same shape as the LoCoMo
    category-5 bug fixed the same day."""
    with pytest.raises(ValueError, match="empty hypothesis"):
        harness.quick_score("", "gold", "q_normal")
    with pytest.raises(ValueError, match="empty hypothesis"):
        harness.quick_score("   ", "gold", "q_abs")


def test_a_real_abstention_still_scores_correct() -> None:
    """The fix must not break the behaviour it protects."""
    assert harness.quick_score(
        "I don't have enough information to answer that.", "gold", "q_abs"
    )["is_correct"] is True
    assert harness.quick_score("The answer is Paris.", "gold", "q_abs")["is_correct"] is False


def test_normal_scoring_is_unchanged() -> None:
    assert harness.quick_score("Paris", "Paris", "q1")["is_correct"] is True
    assert harness.quick_score("the answer is Paris", "Paris", "q1")["is_correct"] is True
    assert harness.quick_score("Berlin", "Paris", "q1")["is_correct"] is False

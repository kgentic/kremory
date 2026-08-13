"""G0 — `evidence_eval.py --self-test` must EXIT NON-ZERO when a self-test FAILS.

Regression guard for a defect found 2026-08-13: `main()` was annotated `-> None`,
the module contained no `sys.exit` and no non-zero return, and the caller at the
fact-resolution self-test discarded its boolean. So `--self-test` printed

    FAIL — nDCG did not move under shuffle; this instrument is order-BLIND.

and exited **0**.

Why that is a gate and not a nicety: the standing project rule is "ALWAYS run
`--self-test` before believing any recall number" (SYSTEM-PRIMER, RECALL-LEDGER
§8 rule 1). With a 0 exit on failure, that rule was enforceable only by a human
reading the terminal — any script wiring it in as a precondition got a false
green. See `a-record-of-work-is-not-the-work` Face 2: a script reporting success
having detected failure.

These tests drive `main()` itself, not the self-test helpers, because THE WIRING
was the bug — both helpers already computed the right answer and were simply not
connected to an exit code.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import evidence_eval  # noqa: E402


# --------------------------------------------------------------------------
# Minimal fixtures. Deliberately tiny — this file tests the exit-code contract,
# not scoring behaviour (that is `test_locomo_scorer.py`'s job).
# --------------------------------------------------------------------------

_SAMPLE = "conv-test"
_EVIDENCE_TURN = "Alice: I adopted a greyhound called Pip in March"


def _write_dataset(tmp_path: Path) -> Path:
    """A one-conversation LoCoMo-shaped dataset with two turns."""
    data = [{
        "sample_id": _SAMPLE,
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Alice",
                 "text": "I adopted a greyhound called Pip in March"},
                {"dia_id": "D1:2", "speaker": "Bob",
                 "text": "Unrelated filler about the weather"},
            ],
        },
    }]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data))
    return p


def _write_run(tmp_path: Path, *, gold_rank: int) -> Path:
    """One scorable question whose evidence turn sits at `gold_rank` (1-based)
    in a 6-item pool, so nDCG has room to move under a shuffle."""
    pool = [f"filler memory number {i}" for i in range(6)]
    pool[gold_rank - 1] = _EVIDENCE_TURN
    run = {"results": [{
        "sample_id": _SAMPLE,
        "category": "single_hop",
        # ⚠️ The key is `evidence_ids` (see `evidence_ids()`), NOT `evidence`.
        # The first cut of this fixture used `evidence`, so `score_row` returned
        # None for every row, NOTHING was scored, and every metric read 0.0 —
        # which made the "healthy" test assert PASS over an EMPTY result set
        # while also (correctly, but uselessly) tripping the order-blind check.
        # A fixture that scores nothing cannot prove anything; hence the
        # non-vacuity assertion in `_assert_scored`.
        "evidence_ids": ["D1:1"],
        "recalled_memories": pool,
    }]}
    p = tmp_path / "run.json"
    p.write_text(json.dumps(run))
    return p


def _assert_scored(out: str) -> None:
    """NON-VACUITY: the run must have scored at least one question.

    `report()` prints `(k=10, n=N scorable ex-adversarial)`. If N is 0 every
    metric is 0.0, nDCG cannot move under shuffle, and the tests below would be
    measuring an empty set rather than the behaviour they claim to test.
    """
    m = re.search(r"n=(\d+) scorable", out)
    assert m, f"could not find the scorable-count line in output:\n{out}"
    assert int(m.group(1)) > 0, (
        "VACUOUS FIXTURE — 0 questions scored, so this test proves nothing. "
        f"Check the row keys against `score_row`/`evidence_ids`.\n{out}")


def _run_main(monkeypatch, run: Path, dataset: Path, *, self_test: bool = True) -> int:
    argv = ["evidence_eval.py", str(run), "--dataset", str(dataset)]
    if self_test:
        argv.append("--self-test")
    monkeypatch.setattr(sys, "argv", argv)
    return evidence_eval.main()


# --------------------------------------------------------------------------
# Non-vacuity: the healthy path must exit 0. Without this, an implementation
# that returned 1 unconditionally would satisfy every failure test below.
# --------------------------------------------------------------------------

def test_healthy_self_test_exits_zero(tmp_path, monkeypatch, capsys):
    rc = _run_main(monkeypatch, _write_run(tmp_path, gold_rank=1),
                   _write_dataset(tmp_path))
    out = capsys.readouterr().out
    _assert_scored(out)
    assert "[self-test] PASS" in out, out
    assert rc == 0


def test_without_self_test_flag_exits_zero(tmp_path, monkeypatch):
    """Scoring a run without `--self-test` is unaffected by this change."""
    rc = _run_main(monkeypatch, _write_run(tmp_path, gold_rank=1),
                   _write_dataset(tmp_path), self_test=False)
    assert rc == 0


# --------------------------------------------------------------------------
# The two failure paths. Each asserts BOTH that the failure is printed AND that
# it reaches the exit code — printing it was never the broken half.
# --------------------------------------------------------------------------

def test_order_blind_metric_exits_nonzero(tmp_path, monkeypatch, capsys):
    """Force the scorer order-BLIND. This is the exact condition the self-test
    exists to detect, and the one it used to report as a success."""
    real_score_row = evidence_eval.score_row

    def order_blind(row, turns, k, episode_content=None):
        # Score the pool as an unordered SET: rank information is destroyed, so
        # nDCG cannot move under shuffle.
        s = real_score_row(row, turns, k, episode_content=episode_content)
        if s is not None:
            s["ndcg"] = s["recall"]
            s["rr"] = 0.0
            s["first_rank"] = None
        return s

    monkeypatch.setattr(evidence_eval, "score_row", order_blind)
    rc = _run_main(monkeypatch, _write_run(tmp_path, gold_rank=1),
                   _write_dataset(tmp_path))
    captured = capsys.readouterr()
    # Non-vacuity matters MOST here: an empty result set is also order-blind, so
    # without this the test would pass for the wrong reason.
    _assert_scored(captured.out)
    assert "order-BLIND" in captured.out, captured.out
    assert rc == 1, "an order-blind instrument MUST fail the gate, not pass it"


def test_failing_fact_resolution_exits_nonzero(tmp_path, monkeypatch, capsys):
    """The fact-resolution self-test's boolean was discarded at its call site;
    assert it now reaches the exit code."""
    monkeypatch.setattr(evidence_eval, "self_test_fact_resolution", lambda: False)
    rc = _run_main(monkeypatch, _write_run(tmp_path, gold_rank=1),
                   _write_dataset(tmp_path))
    err = capsys.readouterr().err
    assert "OVERALL: FAIL" in err, err
    assert rc == 1


def test_main_is_annotated_to_return_an_exit_code():
    """`main() -> None` is how this defect shipped. A future refactor that drops
    the return value disarms every caller that gates on it, silently."""
    import inspect
    # `from __future__ import annotations` stringizes annotations, so this is
    # the literal "int", not the type object. Comparing against `int` here
    # fails even on correct code — checked, not assumed.
    ann = inspect.signature(evidence_eval.main).return_annotation
    assert ann in (int, "int"), f"main() must return an exit code, got {ann!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

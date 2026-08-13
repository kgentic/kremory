"""G1 — `--sample-n` makes the prescribed smoke a COMMAND, and a sampled number
can never be printed as a headline.

Background. `.ai-docs/plans/paid-bench-readiness-target-2026-08-11.md` prescribes
"run `qa_eval.py` on ~10 questions ($0.03), inspect, THEN the full run"
(smoke-one-before-batch). Until 2026-08-13 **no flag did that** — the guard was
an intention, and obeying it meant hand-slicing a batch file.

Two halves are tested here, and the second is the load-bearing one:
  1. sampling is DETERMINISTIC (same seed -> same questions, so a re-run replays
     from the response cache for $0 instead of re-spending);
  2. the marker survives answer-gen -> judge -> tally, and the tally FAILS
     CLOSED on it. A 10-question accuracy that renders identically to a
     1531-question accuracy is worse than having no smoke at all.

No network. `_chat` is never called: these drive `apply_sample` and
`cmd_answer_tally` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import qa_eval  # noqa: E402


def _batch(n: int) -> list[dict]:
    return [{"key": f"k{i:03d}", "question_id": i, "sample_id": "conv-test",
             "category": "single_hop", "question": f"q{i}", "gold": f"g{i}",
             "memories": ["m"]} for i in range(n)]


# --------------------------------------------------------------------------
# apply_sample
# --------------------------------------------------------------------------

def test_sample_is_deterministic_for_a_given_seed():
    a = qa_eval.apply_sample(_batch(50), 10, 7)
    b = qa_eval.apply_sample(_batch(50), 10, 7)
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert len(a) == 10


def test_sample_is_independent_of_input_row_order():
    """Rows are sorted by key before selection, so a differently-ordered results
    file must still yield the same sample — otherwise a re-run against a
    re-serialised file silently re-spends."""
    forward = qa_eval.apply_sample(_batch(50), 10, 7)
    reversed_in = qa_eval.apply_sample(list(reversed(_batch(50))), 10, 7)
    assert sorted(r["key"] for r in forward) == sorted(r["key"] for r in reversed_in)


def test_different_seeds_select_different_questions():
    """NON-VACUITY for the determinism tests above: if sampling ignored the seed
    entirely they would still pass."""
    a = {r["key"] for r in qa_eval.apply_sample(_batch(50), 10, 1)}
    b = {r["key"] for r in qa_eval.apply_sample(_batch(50), 10, 2)}
    assert a != b


def test_sample_marks_every_selected_row():
    rows = qa_eval.apply_sample(_batch(50), 10, 7)
    assert all(r[qa_eval.SAMPLE_KEY] == {"n": 10, "seed": 7, "of_total": 50}
               for r in rows)


def test_sample_n_none_returns_batch_unmarked():
    rows = qa_eval.apply_sample(_batch(5), None, 7)
    assert len(rows) == 5
    assert not any(qa_eval.SAMPLE_KEY in r for r in rows)


def test_sample_n_at_or_above_population_is_not_a_sample():
    """A "sample" of the whole population is the whole population. Marking it
    would brand a legitimate full run NOT QUOTABLE."""
    rows = qa_eval.apply_sample(_batch(5), 5, 7)
    assert len(rows) == 5
    assert not any(qa_eval.SAMPLE_KEY in r for r in rows), \
        "a full-population --sample-n must NOT mark the run as sampled"


def test_sample_n_zero_is_rejected():
    with pytest.raises(SystemExit):
        qa_eval.apply_sample(_batch(5), 0, 7)


# --------------------------------------------------------------------------
# tally gate
# --------------------------------------------------------------------------

def _write_tally_inputs(tmp_path: Path, *, n_batch: int, n_judged: int,
                        sampled: bool) -> tuple[Path, Path]:
    """A results .json the tally can load, plus a verdicts .jsonl covering the
    first `n_judged` of it."""
    results = {"results": [
        {"sample_id": "conv-test", "question_id": i, "category": "single_hop",
         "question": f"q{i}", "expected_answer": f"g{i}",
         "recalled_memories": ["m"]}
        for i in range(n_batch)
    ]}
    rj = tmp_path / "results.json"
    rj.write_text(json.dumps(results))

    batch = qa_eval.load_batch(rj, allow_unverified=True)
    marker = {"n": n_judged, "seed": 7, "of_total": n_batch}
    lines = []
    for b in batch[:n_judged]:
        v = {"key": b["key"], "question_id": b["question_id"],
             "category": b["category"], "correct": True, "reason": "ok",
             "judge_model": "test"}
        if sampled:
            v[qa_eval.SAMPLE_KEY] = marker
        lines.append(json.dumps(v))
    vj = tmp_path / "verdicts.jsonl"
    vj.write_text("\n".join(lines))
    return rj, vj


def _tally_args(rj: Path, vj: Path, tmp_path: Path, **kw) -> argparse.Namespace:
    return argparse.Namespace(
        input=rj, verdicts=vj, answerer_label="a", judge_label="j", k_label="10",
        summary=tmp_path / "summary.json", allow_unverified_build=True,
        allow_sampled=kw.get("allow_sampled", False))


def test_tally_refuses_sampled_verdicts_by_default(tmp_path, capsys):
    rj, vj = _write_tally_inputs(tmp_path, n_batch=50, n_judged=10, sampled=True)
    rc = qa_eval.cmd_answer_tally(_tally_args(rj, vj, tmp_path))
    err = capsys.readouterr().err
    assert rc == 2
    assert "SAMPLED" in err and "10 of 50" in err, err


def test_tally_prints_sampled_run_branded_not_quotable(tmp_path, capsys):
    rj, vj = _write_tally_inputs(tmp_path, n_batch=50, n_judged=10, sampled=True)
    rc = qa_eval.cmd_answer_tally(
        _tally_args(rj, vj, tmp_path, allow_sampled=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT QUOTABLE" in out, out
    assert "[SAMPLE]" in out and "[HEADLINE]" not in out, out
    # The batch must be restricted to judged rows, or the accuracy is diluted by
    # 40 unjudged questions counted incorrect.
    assert "OVERALL" in out
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["metric"] == "locomo_qa_gen_accuracy_SAMPLE"
    assert summary["quotable"] is False
    assert summary["sample"]["n"] == 10
    assert summary["overall"]["total"] == 10, \
        "a sampled tally must score the SAMPLE, not the full batch"


def test_unsampled_tally_is_unchanged_and_quotable(tmp_path, capsys):
    """NON-VACUITY: an implementation that refused everything, or branded every
    run NOT QUOTABLE, would pass every test above and fail this one."""
    rj, vj = _write_tally_inputs(tmp_path, n_batch=10, n_judged=10, sampled=False)
    rc = qa_eval.cmd_answer_tally(_tally_args(rj, vj, tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "[HEADLINE]" in out, out
    assert "NOT QUOTABLE" not in out
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["metric"] == "locomo_qa_gen_accuracy"
    assert summary["quotable"] is True
    assert "sample" not in summary


def test_sampled_verdicts_matching_no_batch_row_abort(tmp_path, capsys):
    rj, vj = _write_tally_inputs(tmp_path, n_batch=50, n_judged=10, sampled=True)
    rows = [json.loads(l) for l in vj.read_text().splitlines()]
    for r in rows:
        r["key"] = "does-not-exist-" + r["key"]
    vj.write_text("\n".join(json.dumps(r) for r in rows))
    rc = qa_eval.cmd_answer_tally(
        _tally_args(rj, vj, tmp_path, allow_sampled=True))
    assert rc == 2
    assert "different run" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

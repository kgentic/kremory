"""Phase 2 — the SMOKE FLOW driven end to end through the real qa_eval CLI.

Everything else in this suite tests a function. This drives
`answer-gen -> answer-judge -> answer-tally` through `qa_eval.main()` exactly as
an operator would, and proves the sample marker survives all three stages. That
matters because the guard this proves is a PROPAGATION property, and the unit
tests each verify one end of it — the failure mode is the join, which is
precisely what `assert_provenance` demonstrated by being written, tested, and
wired to nothing for months.

COST: zero, three ways, belt and braces.
  1. The response cache is pre-seeded, so every `_chat` is a replay.
  2. `OPENAI_API_KEY` is set to an obviously-invalid value. Should a cache miss
     occur, the call 401s — `_chat` treats 4xx (bar 429) as fatal and raises
     rather than retrying, so a miss FAILS LOUDLY and CANNOT be billed.
  3. No network is required for the happy path at all.

Point 2 is the load-bearing one: it makes "this test accidentally spent money"
structurally impossible rather than something to be careful about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import prompts_qa  # noqa: E402
import qa_eval  # noqa: E402

INVALID_KEY = "sk-invalid-this-test-must-never-spend"
ANSWERER = "gpt-4o-mini"
JUDGE = "gpt-4o"
N_TOTAL, N_SAMPLE = 40, 6


def _results_json(tmp_path: Path) -> Path:
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"results": [
        {"sample_id": "conv-test", "question_id": i, "category": "single_hop",
         "question": f"question number {i}?", "expected_answer": f"answer {i}",
         "recalled_memories": [f"memory {i}a", f"memory {i}b"]}
        for i in range(N_TOTAL)
    ]}))
    return p


def _seed_cache(cache_dir: Path, batch: list[dict], k: int) -> None:
    """Write the exact cache entries `_chat` will look up, for both stages.

    Uses qa_eval's own `_cache_path` and prompts_qa's own builders rather than
    re-deriving either, so a drift in the real prompt-building code cannot leave
    this test silently passing against prompts nobody uses.
    """
    qa_eval._CACHE_DIR = cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    for b in batch:
        answer = f"answer {b['question_id']}"
        gen_prompt = prompts_qa.build_answer_prompt(
            b["question"], (b.get("memories") or [])[:k], reference_date="2023",
            provenance=[], structured=False, text_block=None, chronological=False)
        gp = qa_eval._cache_path(ANSWERER, "", gen_prompt, False)
        gp.write_text(json.dumps({"content": answer}))

        judge_prompt = prompts_qa.build_judge_prompt(
            b["question"], b.get("gold", ""), answer, b.get("category", ""))
        jp = qa_eval._cache_path(JUDGE, prompts_qa.JUDGE_SYSTEM_PROMPT,
                                 judge_prompt, True)
        jp.write_text(json.dumps({"content": '{"label": "CORRECT"}'}))


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["qa_eval.py", *argv])
    return qa_eval.main()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", INVALID_KEY)
    monkeypatch.chdir(tmp_path)          # keep _load_api_key off the repo .env
    return tmp_path


def test_smoke_flow_marker_survives_gen_judge_tally(env, monkeypatch, capsys):
    tmp = env
    results = _results_json(tmp)
    cache = tmp / "cache"

    batch = qa_eval.load_batch(results, allow_unverified=True)
    sampled = qa_eval.apply_sample(list(batch), N_SAMPLE, 99)
    _seed_cache(cache, sampled, k=10)

    answers = tmp / "answers.jsonl"
    verdicts = tmp / "verdicts.jsonl"
    summary = tmp / "summary.json"

    rc = _run(monkeypatch, "answer-gen", str(results), "-o", str(answers),
              "--model", ANSWERER, "--sample-n", str(N_SAMPLE),
              "--sample-seed", "99", "--cache-dir", str(cache),
              "--allow-unverified-build")
    assert rc == 0, capsys.readouterr()

    gen_rows = [json.loads(l) for l in answers.read_text().splitlines() if l.strip()]
    assert len(gen_rows) == N_SAMPLE, f"expected {N_SAMPLE} answers, got {len(gen_rows)}"
    assert all(r[qa_eval.SAMPLE_KEY]["n"] == N_SAMPLE for r in gen_rows), \
        "STAGE 1->2 BREAK: answer-gen did not stamp the sample marker"

    rc = _run(monkeypatch, "answer-judge", str(answers), "-o", str(verdicts),
              "--model", JUDGE, "--cache-dir", str(cache))
    assert rc == 0, capsys.readouterr()

    v_rows = [json.loads(l) for l in verdicts.read_text().splitlines() if l.strip()]
    assert len(v_rows) == N_SAMPLE
    assert all(r[qa_eval.SAMPLE_KEY]["n"] == N_SAMPLE for r in v_rows), \
        "STAGE 2->3 BREAK: the judge dropped the sample marker"

    # The tally must REFUSE by default...
    rc = _run(monkeypatch, "answer-tally", str(results), "--verdicts", str(verdicts),
              "--summary", str(summary), "--allow-unverified-build")
    assert rc == 2, "a sampled tally must fail closed"
    assert "SAMPLED" in capsys.readouterr().err

    # ...and brand it when explicitly allowed.
    rc = _run(monkeypatch, "answer-tally", str(results), "--verdicts", str(verdicts),
              "--summary", str(summary), "--allow-sampled",
              "--allow-unverified-build")
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT QUOTABLE" in out and "[SAMPLE]" in out, out

    s = json.loads(summary.read_text())
    assert s["metric"] == "locomo_qa_gen_accuracy_SAMPLE"
    assert s["quotable"] is False
    assert s["overall"]["total"] == N_SAMPLE, \
        f"the sampled tally must score {N_SAMPLE}, not the full {N_TOTAL}"
    assert s["overall"]["correct"] == N_SAMPLE


def test_full_flow_is_quotable_and_unbranded(env, monkeypatch, capsys):
    """NON-VACUITY for the whole file: without --sample-n the same pipeline must
    produce a normal, quotable headline. An implementation that branded every
    run NOT QUOTABLE would pass every assertion above."""
    tmp = env
    results = _results_json(tmp)
    cache = tmp / "cache"
    batch = qa_eval.load_batch(results, allow_unverified=True)
    _seed_cache(cache, batch, k=10)

    answers, verdicts, summary = (tmp / "a.jsonl", tmp / "v.jsonl", tmp / "s.json")
    assert _run(monkeypatch, "answer-gen", str(results), "-o", str(answers),
                "--model", ANSWERER, "--cache-dir", str(cache),
                "--allow-unverified-build") == 0
    assert _run(monkeypatch, "answer-judge", str(answers), "-o", str(verdicts),
                "--model", JUDGE, "--cache-dir", str(cache)) == 0
    assert _run(monkeypatch, "answer-tally", str(results), "--verdicts", str(verdicts),
                "--summary", str(summary), "--allow-unverified-build") == 0

    out = capsys.readouterr().out
    assert "[HEADLINE]" in out and "NOT QUOTABLE" not in out, out
    s = json.loads(summary.read_text())
    assert s["quotable"] is True and s["overall"]["total"] == N_TOTAL


def test_total_outage_is_never_reported_as_success(env, monkeypatch, capsys):
    """RECALL-LEDGER §8 rule 13: "a total outage must never be reportable as a
    null result."

    With an empty cache every call 401s on the invalid key. Both stages must
    report NON-ZERO. `answer-judge` did NOT: it never counted failures and
    returned 0 unconditionally, so a judge pass in which every call failed wrote
    zero verdicts and reported success. Fixed 2026-08-13; this pins it.

    (This test was originally written expecting an *exception*. It failed, and
    the failure was the informative part — answer-gen returns a non-zero code,
    which is better, and the asymmetry with the judge is what exposed the bug.)
    """
    tmp = env
    results = _results_json(tmp)
    empty = tmp / "empty-cache"

    rc = _run(monkeypatch, "answer-gen", str(results), "-o", str(tmp / "x.jsonl"),
              "--model", ANSWERER, "--sample-n", "1",
              "--cache-dir", str(empty), "--allow-unverified-build")
    assert rc != 0, "answer-gen must not report success when every call failed"
    assert "FAIL" in capsys.readouterr().err

    # Judge the same way: a well-formed answers file, no usable cache entries.
    answers = tmp / "answers-for-judge.jsonl"
    answers.write_text(json.dumps({
        "key": "k1", "question_id": 1, "category": "single_hop",
        "question": "q?", "gold": "g", "generated_answer": "a"}) + "\n")
    rc = _run(monkeypatch, "answer-judge", str(answers), "-o", str(tmp / "v.jsonl"),
              "--model", JUDGE, "--cache-dir", str(empty))
    assert rc != 0, \
        "answer-judge must not report success when every call failed (rule 13)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

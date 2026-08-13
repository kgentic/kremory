"""G2 — the sweep-point provenance gate, wired onto the path that spends money.

`provenance.assert_provenance` was written for the recall-improvement e2e spec
(§S0-infra G1) to close "the 13.9%-class stale-file hole", is fully tested in
`test_provenance.py`, and until 2026-08-13 **had zero callers outside its own
test and its `__main__`.** It had never run on a path that spends money. This
wires it into `load_batch`, so every qa_eval subcommand inherits it, and these
tests pin the behaviour.

Design note tested below: the gate is INERT without `--expect`. An always-on
sweep-point assertion would refuse every historical results file — they were
measured at other sweep points, which is the entire purpose of a sweep — and a
guard that fires on ordinary work gets disabled
(`over-blocking-is-a-security-failure`). Stamp PRESENCE and the shipped-default
build check remain unconditional in `gate_shipped_defaults`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import qa_eval  # noqa: E402
from provenance import ProvenanceMismatch  # noqa: E402

# The real, retained artefact behind RECALL-LEDGER §4.20's pinned tripwire
# baseline. Used rather than a hand-built stamp because a fixture I author only
# proves the gate agrees with MY model of the stamp, not with the one the
# harness actually writes.
PINNED = Path(__file__).resolve().parents[2] / ".context" / "tripwire-conv0-2026-08-11.json"

pytestmark = pytest.mark.skipif(
    not PINNED.exists(),
    reason=f"pinned tripwire artefact not present at {PINNED}")


def _stamp() -> dict:
    return json.loads(PINNED.read_text())["provenance"]


def test_the_pinned_artefact_actually_carries_a_stamp():
    """NON-VACUITY. If the artefact had no `provenance` block, every match test
    below would be asserting against nothing."""
    s = _stamp()
    assert s, "pinned artefact carries no provenance stamp"
    assert "rrf_k" in s and "content_stream_weight" in s, s


def test_expect_matching_the_stamp_passes():
    s = _stamp()
    qa_eval.gate_recall_provenance(
        PINNED, {"rrf_k": s["rrf_k"], "content_stream_weight": s["content_stream_weight"]})


def test_expect_mismatching_the_stamp_raises():
    s = _stamp()
    wrong = float(s["content_stream_weight"]) + 1.0
    with pytest.raises(ProvenanceMismatch) as ei:
        qa_eval.gate_recall_provenance(PINNED, {"content_stream_weight": wrong})
    msg = str(ei.value)
    # The message must name BOTH values — "mismatch" alone forces the operator
    # to go and diff two files by hand.
    assert str(s["content_stream_weight"]) in msg and str(wrong) in msg, msg


def test_expect_unknown_key_raises():
    with pytest.raises(ProvenanceMismatch):
        qa_eval.gate_recall_provenance(PINNED, {"no_such_knob": 1})


def test_gate_is_inert_without_expect():
    """The over-blocking guard: no --expect, no opinion."""
    qa_eval.gate_recall_provenance(PINNED, {})


def test_jsonl_input_says_it_cannot_check_rather_than_passing_silently(tmp_path, capsys):
    """A prepared .jsonl batch carries no stamp. The gate must SAY it cannot
    check — silence here is indistinguishable from a pass."""
    p = tmp_path / "batch.jsonl"
    p.write_text('{"key":"k1"}\n')
    qa_eval.gate_recall_provenance(p, {"rrf_k": 60})
    assert "cannot be checked" in capsys.readouterr().err


# --------------------------------------------------------------------------
# parse_expect typing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pair,want", [
    ("rrf_k=60", {"rrf_k": 60}),
    ("content_stream_weight=1.5", {"content_stream_weight": 1.5}),
    ("features=content-search", {"features": "content-search"}),
    ("rerank_enabled=true", {"rerank_enabled": True}),
    ("rerank_enabled=false", {"rerank_enabled": False}),
])
def test_parse_expect_types(pair, want):
    assert qa_eval.parse_expect([pair]) == want


def test_parse_expect_rejects_a_bare_token():
    with pytest.raises(SystemExit):
        qa_eval.parse_expect(["rrf_k"])


def test_parse_expect_empty():
    assert qa_eval.parse_expect(None) == {}


# --------------------------------------------------------------------------
# end-to-end through main(), which is where the exit code is decided
# --------------------------------------------------------------------------

def test_main_returns_3_on_mismatch_rather_than_a_traceback(monkeypatch, tmp_path, capsys):
    s = _stamp()
    wrong = float(s["content_stream_weight"]) + 1.0
    monkeypatch.setattr(sys, "argv", [
        "qa_eval.py", "answer-tally", str(PINNED),
        "--verdicts", str(tmp_path / "none.jsonl"),
        "--expect", f"content_stream_weight={wrong}",
        "--allow-unverified-build",
    ])
    (tmp_path / "none.jsonl").write_text("")
    rc = qa_eval.main()
    assert rc == 3, "a provenance mismatch must be a clean refusal, not a crash"
    assert "REFUSING TO SCORE" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

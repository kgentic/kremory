#!/usr/bin/env python3
"""Round-trip tests for the S0-infra provenance stamp
(recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1] / R4).

Runnable two ways:
  - `python3 bench/locomo/test_provenance.py`  (plain, no pytest needed)
  - `pytest bench/locomo/test_provenance.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import (  # noqa: E402
    ProvenanceMismatch,
    assert_provenance,
    build_provenance,
)


def _write(tmp: Path, obj: dict) -> Path:
    p = tmp / "recall.json"
    with open(p, "w") as f:
        json.dump(obj, f)
    return p


def test_build_provenance_reflects_env():
    """build_provenance() sources the sweep knobs from KREMORY_* env."""
    for k in ("KREMORY_CONTENT_WEIGHT", "KREMORY_RRF_K", "KREMORY_RERANK",
              "KREMORY_RERANK_K", "KREMORY_FEATURES"):
        os.environ.pop(k, None)

    # Defaults when env absent.
    prov = build_provenance()
    assert prov["content_stream_weight"] == 1.0
    assert prov["rrf_k"] == 60
    assert prov["rerank_enabled"] is False
    assert prov["features"] == ""
    assert "git_sha" in prov and prov["git_sha"]  # present, never blank

    # Applied when env set.
    os.environ["KREMORY_CONTENT_WEIGHT"] = "2.0"
    os.environ["KREMORY_RRF_K"] = "1"
    os.environ["KREMORY_RERANK"] = "1"
    os.environ["KREMORY_FEATURES"] = "content-search,rerank,prometheus"
    prov = build_provenance()
    assert prov["content_stream_weight"] == 2.0
    assert prov["rrf_k"] == 1
    assert prov["rerank_enabled"] is True
    assert prov["features"] == "content-search,rerank,prometheus"

    for k in ("KREMORY_CONTENT_WEIGHT", "KREMORY_RRF_K", "KREMORY_RERANK",
              "KREMORY_FEATURES"):
        os.environ.pop(k, None)


def test_build_provenance_prefers_health_over_env():
    """TD-135: when a /health `scoring` block is present, build_provenance sources
    the scoring config + feature flags from IT — NOT env — even when env disagrees.
    The /health dict is injected directly so no live server is required."""
    # Env deliberately says something DIFFERENT from /health, to prove /health wins.
    os.environ["KREMORY_CONTENT_WEIGHT"] = "9.9"
    os.environ["KREMORY_RRF_K"] = "999"
    os.environ["KREMORY_FEATURES"] = "content-search"
    try:
        health = {
            "status": "ok",
            "content_search": True,
            "rerank": True,
            "prometheus": False,
            "scoring": {
                "content_stream_weight": 1.5,
                "rrf_k": 60,
                "graph_degree_weight": 0.05,
                "temporal_weight": 0.0,
            },
        }
        prov = build_provenance(health=health)
        # /health scoring wins over the (disagreeing) env.
        assert prov["content_stream_weight"] == 1.5
        assert prov["rrf_k"] == 60
        assert prov["graph_degree_weight"] == 0.05
        assert prov["temporal_weight"] == 0.0
        # Feature flags come from /health too (rerank bool + reconstructed features).
        assert prov["rerank_enabled"] is True
        assert prov["features"] == "content-search,rerank"
        assert prov["provenance_source"] == "health"
        assert "git_sha" in prov and prov["git_sha"]  # still from git
    finally:
        for k in ("KREMORY_CONTENT_WEIGHT", "KREMORY_RRF_K", "KREMORY_FEATURES"):
            os.environ.pop(k, None)


def test_build_provenance_falls_back_to_env_without_health():
    """No server + no /health → env fallback, flagged as such. Preserves the
    standalone (env-sourced) behaviour for older callers / offline use."""
    for k in ("KREMORY_CONTENT_WEIGHT", "KREMORY_RRF_K", "KREMORY_RERANK",
              "KREMORY_RERANK_K", "KREMORY_FEATURES"):
        os.environ.pop(k, None)
    os.environ["KREMORY_CONTENT_WEIGHT"] = "2.0"
    try:
        prov = build_provenance()  # neither server nor health
        assert prov["content_stream_weight"] == 2.0
        assert prov["rrf_k"] == 60
        assert prov["provenance_source"] == "env-fallback"
    finally:
        os.environ.pop("KREMORY_CONTENT_WEIGHT", None)


def test_build_provenance_health_without_scoring_falls_back_to_env():
    """An older server whose /health lacks a `scoring` block → env fallback
    (unfaithful, but recorded) rather than a crash."""
    for k in ("KREMORY_CONTENT_WEIGHT", "KREMORY_RRF_K", "KREMORY_FEATURES"):
        os.environ.pop(k, None)
    os.environ["KREMORY_CONTENT_WEIGHT"] = "3.0"
    try:
        health = {"status": "ok"}  # no scoring block (older server)
        prov = build_provenance(health=health)
        assert prov["content_stream_weight"] == 3.0
        assert prov["provenance_source"] == "env-fallback"
    finally:
        os.environ.pop("KREMORY_CONTENT_WEIGHT", None)


def test_assert_provenance_roundtrip_match():
    """A stamp written into the JSON header round-trips through assert_provenance."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stamp = build_provenance()
        path = _write(tmp, {"mode": "codemem", "provenance": stamp, "results": []})
        # Assert the exact stamp back — must not raise, returns the stamp.
        got = assert_provenance(path, {
            "content_stream_weight": stamp["content_stream_weight"],
            "rrf_k": stamp["rrf_k"],
            "git_sha": stamp["git_sha"],
        })
        assert got == stamp
        # Numeric-by-value: 60 == 60.0.
        assert_provenance(path, {"rrf_k": 60.0})


def test_assert_provenance_mismatch_raises():
    """A mismatched sweep knob raises (gate refuses to score)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _write(tmp, {"provenance": {"content_stream_weight": 1.0, "rrf_k": 60,
                                           "git_sha": "abc", "rerank_enabled": False,
                                           "features": ""}})
        raised = False
        try:
            assert_provenance(path, {"content_stream_weight": 2.0})  # wrong
        except ProvenanceMismatch:
            raised = True
        assert raised, "mismatched content_stream_weight must raise ProvenanceMismatch"


def test_assert_provenance_missing_stamp_raises():
    """A recall file with NO provenance stamp must be refused."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _write(tmp, {"mode": "codemem", "results": []})  # no provenance
        raised = False
        try:
            assert_provenance(path, {"rrf_k": 60})
        except ProvenanceMismatch:
            raised = True
        assert raised, "absent provenance stamp must raise ProvenanceMismatch"


def _run() -> int:
    tests = [
        test_build_provenance_reflects_env,
        test_build_provenance_prefers_health_over_env,
        test_build_provenance_falls_back_to_env_without_health,
        test_build_provenance_health_without_scoring_falls_back_to_env,
        test_assert_provenance_roundtrip_match,
        test_assert_provenance_mismatch_raises,
        test_assert_provenance_missing_stamp_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())

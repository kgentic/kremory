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
    ShippedDefaultsMismatch,
    assert_provenance,
    assert_shipped_defaults,
    build_provenance,
    shipped_default_features,
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



# ── W0.2 — shipped-default gate ──────────────────────────────────────────────


def _health_stamp(features: str) -> dict:
    """A stamp shaped like one build_provenance produces from GET /health."""
    return {
        "git_sha": "deadbeef",
        "features": features,
        "provenance_source": "health",
    }


def test_shipped_defaults_derived_from_cargo_toml():
    """The default set is PARSED, not restated. If this drifts, the gate is
    checking a fiction."""
    got = shipped_default_features()
    assert "content-search" in got, f"expected content-search in derived defaults, got {got}"
    assert "rerank" not in got, (
        f"rerank is deliberately opt-in (it pulls ort/ONNX) — derived set was {got}"
    )


def test_shipped_defaults_accepts_a_default_build():
    stamp = _health_stamp("content-search,prometheus")
    assert_shipped_defaults(stamp)


def test_shipped_defaults_rejects_missing_default_feature():
    """The literal ADR-078 case: the run lacked what consumers get."""
    stamp = _health_stamp("prometheus")
    try:
        assert_shipped_defaults(stamp)
    except ShippedDefaultsMismatch as e:
        assert "MISSING" in str(e), f"error must name the missing feature: {e}"
        return
    raise AssertionError("a run without content-search must NOT certify as default")


def test_shipped_defaults_rejects_extra_feature():
    """The inverse, and the one people forget: a run with MORE than consumers get
    is equally unquotable."""
    stamp = _health_stamp("content-search,rerank,prometheus")
    try:
        assert_shipped_defaults(stamp)
    except ShippedDefaultsMismatch as e:
        assert "EXTRA" in str(e), f"error must name the extra feature: {e}"
        return
    raise AssertionError("a run WITH rerank must NOT certify as the default build")


def test_shipped_defaults_rejects_env_fallback_stamp():
    """An env-fallback stamp is the harness describing itself. Gating on it is the
    self-attestation trap."""
    stamp = {"features": "content-search", "provenance_source": "env-fallback"}
    try:
        assert_shipped_defaults(stamp)
    except ShippedDefaultsMismatch as e:
        assert "env-fallback" in str(e), f"error must name the cause: {e}"
        return
    raise AssertionError("an env-fallback stamp must NOT certify a shipped-default run")


def test_shipped_defaults_prometheus_alone_does_not_fail():
    """False-positive guard. The harness REQUIRES prometheus to scrape /metrics;
    if its presence failed the gate, the gate would be switched off within a day
    (over-blocking is a control failure, not a strictness virtue)."""
    assert_shipped_defaults(_health_stamp("content-search,prometheus"))
    assert_shipped_defaults(_health_stamp("content-search"))



def _fake_repo(lib_default: list[str], mcp_default: list[str]) -> Path:
    """A minimal two-manifest repo root for the drift test."""
    root = Path(tempfile.mkdtemp())
    for crate, feats in (("kremory", lib_default), ("kremory-mcp", mcp_default)):
        d = root / "crates" / crate
        d.mkdir(parents=True)
        feats_str = ", ".join(f'"{f}"' for f in feats)
        (d / "Cargo.toml").write_text(f"[features]\ndefault = [{feats_str}]\n")
    return root


def test_shipped_defaults_fails_when_the_two_manifests_disagree():
    """REL-002. The library and the server binary are edited independently. If they
    diverge, 'the shipped build' is ambiguous and the gate must say so rather than
    silently certifying a server build against a library default."""
    root = _fake_repo(["content-search"], ["content-search", "rerank"])
    try:
        shipped_default_features(root)
    except ShippedDefaultsMismatch as e:
        assert "DIFFERENT default feature" in str(e), f"must name the cause: {e}"
        assert "kremory-mcp" in str(e), f"must name both manifests: {e}"
        return
    raise AssertionError("disagreeing manifests must not silently resolve to one of them")


def test_shipped_defaults_agreeing_manifests_resolve():
    root = _fake_repo(["content-search"], ["content-search"])
    assert shipped_default_features(root) == frozenset({"content-search"})


def _run() -> int:
    tests = [
        test_build_provenance_reflects_env,
        test_build_provenance_prefers_health_over_env,
        test_build_provenance_falls_back_to_env_without_health,
        test_build_provenance_health_without_scoring_falls_back_to_env,
        test_assert_provenance_roundtrip_match,
        test_assert_provenance_mismatch_raises,
        test_assert_provenance_missing_stamp_raises,
        test_shipped_defaults_derived_from_cargo_toml,
        test_shipped_defaults_accepts_a_default_build,
        test_shipped_defaults_rejects_missing_default_feature,
        test_shipped_defaults_rejects_extra_feature,
        test_shipped_defaults_rejects_env_fallback_stamp,
        test_shipped_defaults_prometheus_alone_does_not_fail,
        test_shipped_defaults_fails_when_the_two_manifests_disagree,
        test_shipped_defaults_agreeing_manifests_resolve,
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

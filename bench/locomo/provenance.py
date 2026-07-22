#!/usr/bin/env python3
"""Provenance stamp for LoCoMo recall-run JSON headers.

recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1]: closes the 13.9%-class
stale-file hole. Every recall-run JSON carries a `provenance` block recording the
exact sweep point the server ran at — the git SHA plus the KREMORY_* search-fusion
knobs the harness launched the server with. `assert_provenance` lets the
answer-tally / gate step REFUSE to score a recall file whose stamp does not match
the intended sweep point (fail-loud, never silently score the wrong file).

`build_provenance()` sources the scoring config + build-feature flags from the
SERVER's own `GET /health` (TD-135 — the single source of truth: the config the
search path ACTUALLY uses), so the stamp is faithful to what the server ran with
rather than what the HARNESS process's env happened to say (a divergence between
the two is exactly how a config-mismatch produced a bogus benchmark number). The
git SHA still comes from `git rev-parse HEAD`. If `/health` is unreachable or is
an older server without a `scoring` block, it falls back to the harness env and
logs a WARNING that the stamp MAY BE UNFAITHFUL.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

_LOG = logging.getLogger(__name__)

# Short — /health is a local liveness endpoint; the harness has already probed
# the server is up before this runs.
_HEALTH_TIMEOUT_S = 5.0


class ProvenanceMismatch(RuntimeError):
    """Raised by `assert_provenance` when a recall file's stamp does not match the
    intended sweep point. Bubbles up as a non-zero exit — the gate must NOT score
    a mismatched file (recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1])."""


def _git_sha() -> str:
    """`git rev-parse HEAD`, or an explicit sentinel on failure (never silently
    blank — the stamp MUST be present)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            # Resolve against this file's repo, not the CWD the harness ran from.
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        return f"unknown ({type(e).__name__})"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _fetch_health(server: str) -> dict[str, Any] | None:
    """`GET {server}/health`, parse JSON. Returns the parsed dict, or ``None`` on
    ANY failure (unreachable / non-200 / non-JSON) so the caller can fall back to
    env. Never raises."""
    url = server.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_S) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
        _LOG.warning(
            "provenance: GET %s failed (%s: %s) — will fall back to env",
            url, type(e).__name__, e,
        )
        return None


def _features_from_health(health: Mapping[str, Any]) -> str:
    """Reconstruct the cargo build-feature string from `/health`'s compile-time
    feature booleans — the ACTUAL compiled feature set, more faithful than the
    operator-recorded ``KREMORY_FEATURES`` env."""
    names = []
    if health.get("content_search"):
        names.append("content-search")
    if health.get("rerank"):
        names.append("rerank")
    if health.get("prometheus"):
        names.append("prometheus")
    return ",".join(names)


def build_provenance(
    server: str | None = None,
    *,
    health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance stamp — scoring config + feature flags from the
    server's `GET /health` (single source of truth, TD-135), git SHA from git.

    Args:
        server: base URL of the running `kremory-http` server (e.g.
            ``http://localhost:3179``). When given (and ``health`` is not), fetch
            ``{server}/health`` and source the scoring config + feature flags from
            it.
        health: a pre-fetched `/health` response dict, injected in place of a live
            fetch (used by unit tests so they never require a live server).

    Stamp fields:

    - ``git_sha``: `git rev-parse HEAD`.
    - ``content_stream_weight`` / ``rrf_k``: from ``/health``'s ``scoring`` block
      — the server's LIVE ``SearchConfig`` (reflecting any ``KREMORY_CONTENT_WEIGHT``
      / ``KREMORY_RRF_K`` boot overrides the server was launched with). Also
      stamps ``graph_degree_weight`` / ``temporal_weight`` when reported.
    - ``rerank_enabled``: from ``/health``'s compile-time ``rerank`` feature bool.
    - ``features``: reconstructed from ``/health``'s feature booleans (the actual
      compiled set).
    - ``provenance_source``: ``"health"`` when sourced from the server, or
      ``"env-fallback"`` when it fell back to env (see below).

    Fallback: if ``/health`` is unreachable or an older server lacks a ``scoring``
    block, source the scoring knobs from the harness ``KREMORY_*`` env instead and
    log a WARNING that the stamp MAY BE UNFAITHFUL to the server's active config.

    `assert_provenance`'s contract is unchanged — it still asserts only the keys
    the caller lists in ``expected``.
    """

    def _env_num(name: str, default: float, cast) -> Any:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return cast(raw.strip())
        except (ValueError, TypeError):
            return default

    # Prefer the server's ACTUAL active config (/health) — the single source of
    # truth. Only fetch when the caller didn't inject a `health` dict directly.
    if health is None and server is not None:
        health = _fetch_health(server)

    scoring = health.get("scoring") if isinstance(health, dict) else None

    if isinstance(scoring, dict):
        return {
            "git_sha": _git_sha(),
            "content_stream_weight": float(scoring.get("content_stream_weight", 1.0)),
            "rrf_k": int(scoring.get("rrf_k", 60)),
            "graph_degree_weight": scoring.get("graph_degree_weight"),
            "temporal_weight": scoring.get("temporal_weight"),
            "rerank_enabled": bool(health.get("rerank", False)),
            "features": _features_from_health(health),
            "provenance_source": "health",
        }

    # Env fallback. Warn LOUDLY that the stamp may be unfaithful — but only when a
    # server/health WAS expected (a bare `build_provenance()` is normal standalone
    # usage and must stay quiet).
    if server is not None or health is not None:
        _LOG.warning(
            "provenance: server /health did not provide a `scoring` block "
            "(unreachable or older server) — stamping from harness env, which MAY "
            "BE UNFAITHFUL to the server's active scoring config (TD-135)."
        )

    return {
        "git_sha": _git_sha(),
        "content_stream_weight": _env_num("KREMORY_CONTENT_WEIGHT", 1.0, float),
        "rrf_k": _env_num("KREMORY_RRF_K", 60, int),
        "rerank_enabled": _env_truthy("KREMORY_RERANK")
        or ("KREMORY_RERANK_K" in os.environ),
        "features": os.environ.get("KREMORY_FEATURES", ""),
        "provenance_source": "env-fallback",
    }


def assert_provenance(results_json_path: str | os.PathLike, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the recall-run JSON at ``results_json_path`` carries a `provenance`
    stamp matching every key in ``expected``. Returns the stamp on success.

    Raises `ProvenanceMismatch` (non-zero exit if unhandled) when the stamp is
    absent or any expected key differs — the gate MUST refuse to score a
    mismatched file (recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1]).

    ``expected`` need only list the keys that matter for the sweep point (e.g.
    ``{"content_stream_weight": 2.0, "rrf_k": 60}``); unlisted stamped keys are
    ignored. Numeric values compare by value (``60 == 60.0``).
    """
    path = Path(results_json_path)
    if not path.exists():
        raise ProvenanceMismatch(f"recall file does not exist: {path}")

    with open(path) as f:
        data = json.load(f)

    stamp = data.get("provenance")
    if stamp is None:
        raise ProvenanceMismatch(
            f"recall file {path} carries NO `provenance` stamp — refusing to score "
            f"(recall-improvement-e2e-spec-2026-07-22 §S0-infra G1)"
        )

    mismatches: list[str] = []
    for key, want in expected.items():
        if key not in stamp:
            mismatches.append(f"  {key}: MISSING from stamp (wanted {want!r})")
            continue
        got = stamp[key]
        # Compare numerics by value so 60 == 60.0; everything else by equality.
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            equal = float(want) == float(got)
        else:
            equal = want == got
        if not equal:
            mismatches.append(f"  {key}: stamp has {got!r}, wanted {want!r}")

    if mismatches:
        raise ProvenanceMismatch(
            f"provenance mismatch in {path} — refusing to score:\n"
            + "\n".join(mismatches)
            + f"\n  full stamp: {stamp!r}"
        )

    return stamp


def _main() -> int:
    """CLI: `provenance.py <recall.json> key=value [key=value ...]` — exits
    non-zero on mismatch so a gate/harness can shell out to it."""
    import sys

    if len(sys.argv) < 2:
        print("usage: provenance.py <recall.json> [key=value ...]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    expected: dict[str, Any] = {}
    for pair in sys.argv[2:]:
        if "=" not in pair:
            print(f"bad key=value arg: {pair!r}", file=sys.stderr)
            return 2
        k, v = pair.split("=", 1)
        # Best-effort typing: int, then float, then str.
        parsed: Any = v
        try:
            parsed = int(v)
        except ValueError:
            try:
                parsed = float(v)
            except ValueError:
                if v.lower() in {"true", "false"}:
                    parsed = v.lower() == "true"
        expected[k] = parsed
    try:
        stamp = assert_provenance(path, expected)
    except ProvenanceMismatch as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"provenance OK: {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

#!/usr/bin/env python3
"""Provenance stamp for LoCoMo recall-run JSON headers.

recall-improvement-e2e-spec-2026-07-22 §S0-infra [G1]: closes the 13.9%-class
stale-file hole. Every recall-run JSON carries a `provenance` block recording the
exact sweep point the server ran at — the git SHA plus the KREMORY_* search-fusion
knobs the harness launched the server with. `assert_provenance` lets the
answer-tally / gate step REFUSE to score a recall file whose stamp does not match
the intended sweep point (fail-loud, never silently score the wrong file).

`build_provenance()` sources every field from the harness's own environment (+ the
git call), so the stamp reflects what the server was actually told, not a guess.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


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


def build_provenance() -> dict[str, Any]:
    """Build the provenance stamp from the harness's own KREMORY_* env (+ git).

    Fields mirror the S0-infra sweep knobs so a tally can assert the recall file
    matches the intended sweep point:

    - ``git_sha``: `git rev-parse HEAD`
    - ``content_stream_weight``: from ``KREMORY_CONTENT_WEIGHT`` (default 1.0) —
      the library-side `search_env_overrides` knob.
    - ``rrf_k``: from ``KREMORY_RRF_K`` (default 60).
    - ``rerank_enabled``: from ``KREMORY_RERANK`` truthy, OR the presence of
      ``KREMORY_RERANK_K`` (reranker is a rebuild feature — the operator records
      it via env when the server was built ``--features ...,rerank``).
    - ``features``: from ``KREMORY_FEATURES`` — the operator-recorded cargo build
      feature set (e.g. ``content-search,rerank,prometheus``).

    Values are captured verbatim (numeric knobs parsed leniently, falling back to
    the default on a malformed value — the Rust boot path is the authoritative
    fail-loud validator; here we only need a faithful record of what was set).
    """

    def _num(name: str, default: float, cast) -> Any:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return cast(raw.strip())
        except (ValueError, TypeError):
            return default

    return {
        "git_sha": _git_sha(),
        "content_stream_weight": _num("KREMORY_CONTENT_WEIGHT", 1.0, float),
        "rrf_k": _num("KREMORY_RRF_K", 60, int),
        "rerank_enabled": _env_truthy("KREMORY_RERANK")
        or ("KREMORY_RERANK_K" in os.environ),
        "features": os.environ.get("KREMORY_FEATURES", ""),
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

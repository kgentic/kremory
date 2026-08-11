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


class ShippedDefaultsMismatch(ProvenanceMismatch):
    """Raised by `assert_shipped_defaults` when a run was measured on a build that
    is not the one a `cargo add kremory` user receives (ROADMAP W0.2)."""


# ── W0.2 — the shipped-default gate ──────────────────────────────────────────
#
# ADR-078 is the failure this exists to make structurally impossible: a measured
# +30.2pt sat above `content-search = []` for months, every published benchmark
# ran with the feature ON, and no consumer received it. Measured gap when finally
# checked: -32.2pt. "Remember to check the features" is not a control.
#
# The default set is DERIVED from Cargo.toml, never declared here — a
# hand-maintained copy would drift from the thing it claims to describe, and the
# drift would be silent (see `contract-first-before-new-public-surface`:
# derive > observe > declare).

# Features that /health can report back. `_features_from_health` reconstructs the
# stamp from exactly these three booleans, so a recall-affecting default OUTSIDE
# this set is unverifiable: its absence from the stamp would be indistinguishable
# from it being switched off. That case raises rather than passing quietly.
_HEALTH_REPORTABLE_FEATURES = frozenset({"content-search", "rerank", "prometheus"})

# Exempt from the comparison, with reasons. Keep this list SHORT and justified:
# every entry is a hole in the gate.
#
#   prometheus — compile-time gate on the `/metrics` endpoint only. The benchmark
#   REQUIRES it (the harness scrapes /metrics for the o11y block), so a strict
#   set-equality would fail every legitimate run and the gate would be disabled
#   within a day. It cannot change what recall returns: it installs a recorder
#   and adds a route, and touches no search path.
_EXEMPT_FROM_DEFAULTS_GATE = frozenset({"prometheus"})


# BOTH manifests are load-bearing and they are edited independently:
#   crates/kremory          — the library a `cargo add kremory` user gets
#   crates/kremory-mcp      — builds `kremory-http`, the binary the bench measures
#                             and the one GET /health reports features for
# Deriving from only ONE and comparing against the OTHER's stamp is a silent-drift
# hazard: they agree today (both `default = ["content-search"]`), and nothing
# would tell us if that stopped being true. (Quinn REL-002, 2026-08-03.)
_DEFAULT_MANIFESTS = (
    ("kremory", Path("crates") / "kremory" / "Cargo.toml"),
    ("kremory-mcp", Path("crates") / "kremory-mcp" / "Cargo.toml"),
)


def _default_features_of(manifest: Path) -> frozenset[str]:
    if not manifest.exists():
        raise ShippedDefaultsMismatch(
            f"cannot derive shipped defaults: {manifest} not found. The gate refuses "
            f"to assume a default set — an assumed default is the ADR-078 defect."
        )
    import tomllib

    with open(manifest, "rb") as f:
        data = tomllib.load(f)
    default = data.get("features", {}).get("default")
    if default is None:
        raise ShippedDefaultsMismatch(
            f"{manifest} has no [features] default key — cannot derive the shipped "
            f"build. Refusing rather than guessing."
        )
    return frozenset(default)


def shipped_default_features(repo_root: str | os.PathLike | None = None) -> frozenset[str]:
    """DERIVE the shipped default cargo feature set — from BOTH manifests, which
    must agree.

    Parsed, not restated: change a default in Cargo.toml and the gate follows on
    the next run. If the library and the server binary ever disagree about their
    default set, that is itself reported rather than silently resolved — the
    comparison would otherwise be checking a server build against a library
    default and calling the result 'shipped'.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    derived = {name: _default_features_of(root / rel) for name, rel in _DEFAULT_MANIFESTS}

    distinct = set(map(frozenset, derived.values()))
    if len(distinct) > 1:
        detail = "\n".join(f"  {name}: {sorted(feats)}" for name, feats in derived.items())
        raise ShippedDefaultsMismatch(
            "the library and the server binary declare DIFFERENT default feature "
            "sets, so 'the shipped build' is ambiguous and this gate cannot certify "
            "anything:\n" + detail + "\n\nReconcile the manifests, or pass "
            "--allow-unverified-build to score this as a diagnostic."
        )
    return next(iter(distinct))


def assert_shipped_defaults(
    stamp: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike | None = None,
) -> frozenset[str]:
    """Assert a run's provenance stamp describes the SHIPPED DEFAULT build.

    Returns the compared feature set on success; raises `ShippedDefaultsMismatch`
    otherwise. This is the W0.2 gate: a number measured on a build no consumer
    receives is not a number about the product.

    Three refusals, in order:

    1. **env-fallback stamps.** `build_provenance` falls back to reading the
       HARNESS's own env when `/health` is unreachable, and says in its own
       docstring that such a stamp "MAY BE UNFAITHFUL to the server's active
       config". Gating on it would be gating on a value the measured party
       supplies about itself — the same self-attestation trap the ADR-078 defect
       lived in. Only a `/health`-sourced stamp is admissible.
    2. **Unverifiable defaults.** A recall-affecting default that `/health`
       cannot report is a blind spot, not a pass.
    3. **The actual comparison**, over non-exempt features.
    """
    source = stamp.get("provenance_source")
    if source != "health":
        raise ShippedDefaultsMismatch(
            f"provenance_source is {source!r}, not 'health' — the stamp was built "
            f"from the HARNESS's env, which build_provenance itself documents as "
            f"possibly unfaithful to the server's real config. Refusing to certify "
            f"a shipped-default run on a self-reported stamp. Start the server so "
            f"GET /health answers, then re-run."
        )

    if "features" not in stamp:
        raise ShippedDefaultsMismatch(
            "stamp carries no `features` key — cannot verify the build. Refusing."
        )

    shipped = shipped_default_features(repo_root)

    unverifiable = (shipped - _EXEMPT_FROM_DEFAULTS_GATE) - _HEALTH_REPORTABLE_FEATURES
    if unverifiable:
        raise ShippedDefaultsMismatch(
            f"shipped defaults include {sorted(unverifiable)}, which GET /health does "
            f"not report — so this gate cannot see whether the run had them. Extend "
            f"`_features_from_health` (and the server's /health) before quoting a "
            f"number from this build. A blind spot must not read as a pass."
        )

    run = frozenset(f for f in str(stamp["features"]).split(",") if f)
    want = shipped - _EXEMPT_FROM_DEFAULTS_GATE
    got = run - _EXEMPT_FROM_DEFAULTS_GATE

    if got != want:
        missing = sorted(want - got)
        extra = sorted(got - want)
        detail = []
        if missing:
            detail.append(f"  MISSING (shipped by default, absent from the run): {missing}")
        if extra:
            detail.append(f"  EXTRA (run had it, consumers do NOT): {extra}")
        raise ShippedDefaultsMismatch(
            # Scoped claim, deliberately: what this gate can certify is that the
            # SERVER BUILD matched the declared default feature set. It does NOT
            # certify that the measured call path equals a library consumer's —
            # the bench drives the REST layer, which re-composes results (ROADMAP
            # W0.1). Overclaiming here would be its own false certification.
            "this run's server build does NOT match the declared default feature "
            "set — refusing to certify the number (ROADMAP W0.2):\n"
            + "\n".join(detail)
            + f"\n  shipped default (derived from crates/kremory/Cargo.toml): {sorted(want)}"
            + f"\n  this run: {sorted(got)}"
            + "\n  Exempt from comparison: "
            + f"{sorted(_EXEMPT_FROM_DEFAULTS_GATE)}"
            + "\n\nThis is the ADR-078 defect class: a +30.2pt win sat above "
            "`content-search = []` for months while every published benchmark ran "
            "with it ON and no consumer received it."
        )

    return want


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


def _binary_identity(health: Any) -> dict[str, Any]:
    """TD-202 — the identity of the BINARY that served this run, as the server
    itself reported it via ``/health.build``.

    ``git_sha`` records ``git rev-parse HEAD`` in the HARNESS's working tree,
    which says nothing about which binary answered the requests: the tree can
    have moved on, moved back, or be an entirely different checkout.
    ``.context/td186a-variance/`` already holds a run stamped with a commit
    authored TWELVE HOURS AFTER its binary was built.

    Normally that is recoverable by re-running. **The paid grader is not** — it
    runs ONCE, at the end, by design, so its provenance is the single stamp that
    can never be corrected afterwards. That is why this is worth a field.

    Returns an explicit ``{"available": False, ...}`` when the server did not
    report it (older binary, or unreachable) rather than omitting the key. A
    silently-absent key renders downstream as "nothing to see"; an explicit
    ``available: False`` renders as "we do not know", which is the true state.
    """
    build = health.get("build") if isinstance(health, dict) else None
    if not isinstance(build, dict):
        return {
            "available": False,
            "reason": "server /health reported no `build` block "
            "(pre-TD-202 binary, or unreachable)",
        }
    return {"available": True, **build}


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
            # TD-202: `git_sha` is the HARNESS's working tree, NOT the serving
            # binary. Retained under its original name so existing readers keep
            # working, and duplicated under an honestly-named key; `binary` below
            # is the authoritative record of what actually ran. Do not collapse
            # these into one field — a mismatch between them is exactly the
            # signal TD-202 existed because nobody could see.
            "git_sha": _git_sha(),
            "harness_git_sha": _git_sha(),
            "binary": _binary_identity(health),
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
        # TD-202 — see the health path above. On this fallback the server was not
        # reachable at all, so `binary` reports `available: False` honestly rather
        # than guessing from the harness's `target/` directory (which may not be
        # the binary that was launched).
        "git_sha": _git_sha(),
        "harness_git_sha": _git_sha(),
        "binary": _binary_identity(health),
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

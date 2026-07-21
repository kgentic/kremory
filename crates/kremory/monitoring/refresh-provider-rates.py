#!/usr/bin/env python3
"""Refresh `monitoring/provider-rates.toml` from the LiteLLM pricing SSOT.

## Why this exists (TD-133 C5 / R2)

kremory's cost-tracking (`core::rates::ProviderRates`, `TokenTrackingChatProvider`,
dream-pass budget accounting) reads `monitoring/provider-rates.toml`, which is
bundled into the crate at compile time via `include_str!` (see `core/rates.rs`).
Hand-typing per-provider $/token rates drifts silently from reality the moment a
provider changes pricing. LiteLLM (github.com/BerriAI/litellm, MIT license)
maintains a community-curated, daily-updated JSON of `input_cost_per_token` /
`output_cost_per_token` for ~3000 provider/model pairs — including every
provider kremory ships a rate for (openai, anthropic, voyage, groq). This
script makes `provider-rates.toml` SSOT-derived instead of hand-typed: it reads
the CURRENT toml to discover which (provider, model) pairs kremory actually
prices, looks each one up in the LiteLLM JSON, and rewrites the rate fields
in place.

## Usage

    python3 crates/kremory/monitoring/refresh-provider-rates.py
    python3 crates/kremory/monitoring/refresh-provider-rates.py --dry-run
    python3 crates/kremory/monitoring/refresh-provider-rates.py --source /path/to/local.json

Run this from anywhere — the toml path is resolved relative to this script's
own location (`monitoring/provider-rates.toml`, same directory).

After running, re-run the test suite (`cargo nextest run -p kremory`) — the
bundled rates are asserted against in `core/rates.rs` and
`core/ingest/mod.rs`, and a malformed regen fails those tests loudly rather
than silently shipping bad cost data (per the parse-loudly discipline this
crate follows generally).

## What this script does NOT do

- It is NOT part of `cargo build`, `cargo test`, or any CI/release pipeline.
  kremory is offline-first: the crate must build and run with zero network
  dependency, so pricing data is committed, not fetched at build/run time.
  This is a manual maintainer tool, run occasionally (e.g. when a provider
  announces new pricing, or a cost-tracking test starts looking stale).
- It does NOT add providers/models kremory doesn't already price. It refreshes
  the EXISTING (provider, model) pairs found in the current toml — adding a
  new provider/model to kremory's rate table is still a manual toml edit (then
  re-run this script on subsequent refreshes to keep it current).
- It does NOT drop entries absent from the LiteLLM SSOT. Self-hosted/local
  entries (`local`, `ollama`) have no vendor price to look up — they are
  preserved verbatim.

## Design notes for future maintainers

- Units: LiteLLM prices are USD **per token**. kremory's schema is USD **per
  1,000 tokens** (`cost_per_1k_*_tokens_usd`). This script multiplies by 1000
  (`USD_PER_TOKEN_TO_USD_PER_1K`) at the lookup boundary — nowhere else in the
  codebase should re-derive this conversion.
- Some LiteLLM keys are provider-prefixed (e.g. Groq's `groq/openai/gpt-oss-120b`
  — the `litellm_provider` is `"groq"` but the JSON key repeats it as a path
  segment). `build_litellm_index()` indexes both the raw key AND the
  prefix-stripped key so kremory's un-prefixed model id
  (`openai/gpt-oss-120b`) resolves.
- Per-direction fields (`cost_per_1k_input_tokens_usd` /
  `cost_per_1k_output_tokens_usd`) are refreshed ONLY for entries that already
  opt into them in the current toml (currently: the anthropic entries, per
  TD-133 C5a/C5b). This is discovered by parsing the current file, not
  hardcoded by provider name — an entry gains per-direction refresh the
  moment a maintainer hand-adds those two fields to it.
- Section banner comments (the `# ── Foo ──` dividers + explanatory prose) are
  static text owned by this script (`SECTIONS` below) rather than round-tripped
  from the file, because Python's stdlib has no comment-preserving TOML writer
  (`tomllib`, stdlib since 3.11, is read-only). The (provider, model,
  dimensions, per-direction-opt-in) DATA is always read fresh from the current
  file via `tomllib`, so adding/removing a rate row never requires a script
  change — only adding a brand-new SECTION (a new provider/dimension-shape
  combination) would.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

LITELLM_SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
LITELLM_LICENSE = "MIT (LiteLLM, github.com/BerriAI/litellm)"
TOML_PATH = Path(__file__).resolve().parent / "provider-rates.toml"
FETCH_TIMEOUT_S = 20

# LiteLLM prices are USD per SINGLE token; kremory's schema is USD per 1,000
# tokens. This is the ONE place that conversion happens.
USD_PER_TOKEN_TO_USD_PER_1K = 1000

MODULE_HEADER = """\
# kremory provider rates — canonical cost table (ADR D9 / v0.1.2; TD-133 C5a
# per-direction extension)
#
# Used by `TokenTrackingChatProvider` and `TokenTrackingEmbedder` (via
# `ProviderRates::lookup_rate`) and by dream-pass budget tracking (via
# `ProviderRates::cost_usd`) to compute cost estimates from
# `kremory_core_tokens_total` counter data.
#
# `cost_per_1k_tokens_usd`: USD cost per 1000 tokens — REQUIRED single-rate
#   fallback, used directly by `lookup_rate` (no direction concept) and by
#   `cost_usd` for entries that don't set the per-direction fields below. For
#   chat models this is conventionally the OUTPUT rate (more conservative /
#   expensive estimate).
# `cost_per_1k_input_tokens_usd` / `cost_per_1k_output_tokens_usd`: OPTIONAL.
#   When both are set, `ProviderRates::cost_usd()` charges input and output
#   tokens at their own rate instead of the blended single rate.
# For embedding models: single rate only (no input/output distinction).
#
# SSOT: this file is refreshed FROM the LiteLLM pricing dataset by
# `monitoring/refresh-provider-rates.py` — run that script to update rates
# rather than hand-editing the numbers below (hand-edits drift from the SSOT
# silently and desync `rates_as_of`). Hand-editing IS still correct for rows
# absent from the SSOT (local/self-hosted models) — the refresh script
# preserves those verbatim. See the provenance block below for details.
#
# ADR D7 cardinality discipline:
#   `provider` and `model` label values in metrics MUST match the keys here.
#   Unlisted provider/model combos will not produce cost estimates in dashboards.
"""


def provenance_block(rates_as_of: str) -> str:
    return f"""\
# ── Provenance ─────────────────────────────────────────────────────────────
# Source:  {LITELLM_SOURCE_URL}
# License: {LITELLM_LICENSE}
# Refreshed by: monitoring/refresh-provider-rates.py (rerun to update; prefer
#   this over hand-editing rates — hand-edits drift from the SSOT silently).
# rates_as_of below = the date this file was last regenerated from the SSOT.

rates_as_of = "{rates_as_of}"
"""


@dataclass(frozen=True)
class Section:
    """A group of rate entries sharing a banner comment in the rendered toml.

    `key` discriminates which parsed toml entries belong here:
    (provider, has_dimensions) — `has_dimensions` is True for embedding rows
    (they carry a `dimensions` field) and False for chat rows. This pair is
    enough to disambiguate every section in the current schema (e.g. "openai"
    appears in both an embeddings section and a chat section).
    """

    key: tuple[str, bool]
    banner: str  # static comment block, verbatim in output, no trailing blank line
    preserve: bool = False  # True = never refresh from LiteLLM (e.g. local/ollama)


SECTIONS: list[Section] = [
    Section(
        key=("openai", True),
        banner="# ── OpenAI embeddings ─────────────────────────────────────────────────────────",
    ),
    Section(
        key=("voyage", True),
        banner="# ── VoyageAI embeddings ───────────────────────────────────────────────────────",
    ),
    Section(
        key=("local", True),
        banner=(
            "# ── Local / self-hosted embeddings ────────────────────────────────────────────\n"
            "# Zero cost — CPU/GPU amortised separately via infra cost accounting. Not present\n"
            "# in the LiteLLM SSOT (self-hosted models have no vendor price) — preserved\n"
            "# verbatim by refresh-provider-rates.py rather than dropped."
        ),
        preserve=True,
    ),
    Section(
        key=("openai", False),
        banner=(
            "# ── OpenAI chat ───────────────────────────────────────────────────────────────\n"
            "# Using the output rate as the single blended rate (more conservative / expensive\n"
            "# estimate) — matches this file's single-rate convention."
        ),
    ),
    Section(
        key=("anthropic", False),
        banner=(
            "# ── Anthropic chat ────────────────────────────────────────────────────────────\n"
            "# Per-direction rates (TD-133 C5a/C5b) — migrated from the now-deleted\n"
            "# `anthropic_rate_per_token()` (formerly `core/ingest/mod.rs`), a duplicate\n"
            "# hand-rolled rate table used only by dream-pass cost accounting.\n"
            "# `cost_per_1k_tokens_usd` is retained as the single-rate fallback for callers\n"
            "# with no direction concept (e.g. lookup_rate) and is set to the OUTPUT rate."
        ),
    ),
    Section(
        key=("groq", False),
        banner=(
            "# ── Groq (OpenAI-compatible) chat ─────────────────────────────────────────────\n"
            "# `model` matches the id Groq exposes (the same string threaded as the metric\n"
            "# `model` label). NOTE: single-rate is a known imprecision (input is cheaper than\n"
            "# output); per-direction rates tracked as a follow-up."
        ),
    ),
    Section(
        key=("ollama", False),
        banner=(
            "# ── Ollama / local chat ───────────────────────────────────────────────────────\n"
            "# Self-hosted, no provider cost. Wildcard model matches any Ollama model. Not\n"
            "# present in the LiteLLM SSOT — preserved verbatim by refresh-provider-rates.py."
        ),
        preserve=True,
    ),
]


@dataclass
class CurrentEntry:
    provider: str
    model: str
    cost_per_1k_tokens_usd: float
    dimensions: int | None
    cost_per_1k_input_tokens_usd: float | None
    cost_per_1k_output_tokens_usd: float | None


def load_current_entries(toml_path: Path) -> tuple[str, list[CurrentEntry]]:
    """Parse the CURRENT toml to discover which (provider, model) pairs
    kremory prices, their existing per-direction opt-in, and dimensions.
    This is the "read the current file" step — it is what makes the refresh
    data-driven rather than a hardcoded provider/model list.
    """
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    entries = [
        CurrentEntry(
            provider=row["provider"],
            model=row["model"],
            cost_per_1k_tokens_usd=row["cost_per_1k_tokens_usd"],
            dimensions=row.get("dimensions"),
            cost_per_1k_input_tokens_usd=row.get("cost_per_1k_input_tokens_usd"),
            cost_per_1k_output_tokens_usd=row.get("cost_per_1k_output_tokens_usd"),
        )
        for row in data["providers"]
    ]
    return data["rates_as_of"], entries


def fetch_litellm_prices(source: str) -> dict:
    """Fetch the LiteLLM pricing JSON from `source` (an https URL or a local
    file path — the latter is supported so this function is testable
    offline). Raises on any fetch/parse failure; caller decides fallback.
    """
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=FETCH_TIMEOUT_S) as resp:  # noqa: S310
            raw = resp.read()
    else:
        raw = Path(source).read_bytes()
    return json.loads(raw)


def build_litellm_index(data: dict) -> dict[tuple[str, str], dict]:
    """Index LiteLLM entries by (litellm_provider, model_id).

    Some LiteLLM keys repeat the provider as a path prefix (e.g. Groq's
    `groq/openai/gpt-oss-120b`, Voyage's `voyage/voyage-3`) even though
    `litellm_provider` already names the provider. Index both the raw key and
    the prefix-stripped key so kremory's un-prefixed model ids resolve.
    """
    idx: dict[tuple[str, str], dict] = {}
    for key, entry in data.items():
        provider = entry.get("litellm_provider")
        if not provider or not isinstance(entry, dict):
            continue
        idx.setdefault((provider, key), entry)
        prefix = f"{provider}/"
        if key.startswith(prefix):
            idx.setdefault((provider, key[len(prefix) :]), entry)
    return idx


def fmt_rate(x: float) -> str:
    """Format a USD/1k rate for TOML output without float repr noise."""
    if x == 0:
        return "0.0"
    rounded = round(x, 10)
    s = f"{rounded:.10f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def fmt_per_million(x_per_token: float) -> str:
    """Format a USD-per-token rate as a human `$X.YZ/M` string for inline
    comments — always at least 2 decimal places, more if the source data
    carries extra precision (e.g. `$0.15/M`, `$1.00/M`, `$10.00/M`)."""
    per_m = x_per_token * 1_000_000
    integer, frac = f"{per_m:.4f}".split(".")
    frac = frac.rstrip("0").ljust(2, "0")
    return f"{integer}.{frac}"


@dataclass
class Refreshed:
    entry: CurrentEntry
    cost_per_1k_tokens_usd: float
    cost_per_1k_input_tokens_usd: float | None
    cost_per_1k_output_tokens_usd: float | None
    litellm_key: str | None  # None when preserved (no SSOT match)
    changed: bool


def refresh_entry(
    entry: CurrentEntry, section: Section, litellm_idx: dict[tuple[str, str], dict]
) -> Refreshed:
    is_embedding = entry.dimensions is not None
    match = None if section.preserve else litellm_idx.get((entry.provider, entry.model))

    if match is None:
        # No SSOT match (preserved provider, or a pair the SSOT doesn't carry)
        # — keep the current values byte-for-byte.
        return Refreshed(
            entry=entry,
            cost_per_1k_tokens_usd=entry.cost_per_1k_tokens_usd,
            cost_per_1k_input_tokens_usd=entry.cost_per_1k_input_tokens_usd,
            cost_per_1k_output_tokens_usd=entry.cost_per_1k_output_tokens_usd,
            litellm_key=None,
            changed=False,
        )

    in_rate_tok = match.get("input_cost_per_token") or 0.0
    out_rate_tok = match.get("output_cost_per_token") or 0.0
    in_1k = in_rate_tok * USD_PER_TOKEN_TO_USD_PER_1K
    out_1k = out_rate_tok * USD_PER_TOKEN_TO_USD_PER_1K

    if is_embedding:
        # Embedding models: single rate only, priced off the input side (no
        # output-token concept) — matches this file's existing convention.
        blended = in_1k
        want_per_direction = False
    else:
        # Chat models: blended = output rate (conservative), matching the
        # existing single-rate convention for entries that don't opt into
        # per-direction pricing.
        blended = out_1k
        want_per_direction = entry.cost_per_1k_input_tokens_usd is not None

    new_input = in_1k if want_per_direction else None
    new_output = out_1k if want_per_direction else None

    changed = (
        abs(blended - entry.cost_per_1k_tokens_usd) > 1e-12
        or abs((new_input or 0.0) - (entry.cost_per_1k_input_tokens_usd or 0.0)) > 1e-12
        or abs((new_output or 0.0) - (entry.cost_per_1k_output_tokens_usd or 0.0)) > 1e-12
    )

    return Refreshed(
        entry=entry,
        cost_per_1k_tokens_usd=blended,
        cost_per_1k_input_tokens_usd=new_input,
        cost_per_1k_output_tokens_usd=new_output,
        litellm_key=next(
            k for (prov, k), v in litellm_idx.items() if v is match and prov == entry.provider
        ),
        changed=changed,
    )


def render_entry(r: Refreshed, fetch_date: str) -> str:
    lines: list[str] = []
    if r.litellm_key is not None:
        if r.cost_per_1k_input_tokens_usd is not None and r.cost_per_1k_output_tokens_usd is not None:
            lines.append(
                f"# {r.entry.model}: ${fmt_per_million(r.cost_per_1k_input_tokens_usd / 1000)}/M in, "
                f"${fmt_per_million(r.cost_per_1k_output_tokens_usd / 1000)}/M out "
                f"(LiteLLM SSOT, verified {fetch_date})"
            )
        elif r.entry.dimensions is not None:
            lines.append(
                f"# {r.entry.model}: ${fmt_per_million(r.cost_per_1k_tokens_usd / 1000)}/M in "
                f"(LiteLLM SSOT, verified {fetch_date})"
            )
        else:
            lines.append(
                f"# {r.entry.model}: ${fmt_per_million(r.cost_per_1k_tokens_usd / 1000)}/M out "
                f"(LiteLLM SSOT, verified {fetch_date})"
            )
    lines.append("[[providers]]")
    lines.append(f'provider = "{r.entry.provider}"')
    lines.append(f'model    = "{r.entry.model}"')
    lines.append(f"cost_per_1k_tokens_usd = {fmt_rate(r.cost_per_1k_tokens_usd)}")
    if r.cost_per_1k_input_tokens_usd is not None:
        lines.append(f"cost_per_1k_input_tokens_usd = {fmt_rate(r.cost_per_1k_input_tokens_usd)}")
    if r.cost_per_1k_output_tokens_usd is not None:
        lines.append(f"cost_per_1k_output_tokens_usd = {fmt_rate(r.cost_per_1k_output_tokens_usd)}")
    if r.entry.dimensions is not None:
        lines.append(f"dimensions = {r.entry.dimensions}")
    return "\n".join(lines)


def render_toml(refreshed_by_section: list[tuple[Section, list[Refreshed]]], fetch_date: str) -> str:
    chunks = [MODULE_HEADER.rstrip("\n"), provenance_block(fetch_date).rstrip("\n")]
    for section, refreshed in refreshed_by_section:
        if not refreshed:
            continue
        entries_block = "\n\n".join(render_entry(r, fetch_date) for r in refreshed)
        chunks.append(f"{section.banner}\n\n{entries_block}")
    return "\n\n".join(chunks) + "\n"


def group_by_section(entries: list[CurrentEntry]) -> list[tuple[Section, list[CurrentEntry]]]:
    grouped: list[tuple[Section, list[CurrentEntry]]] = []
    for section in SECTIONS:
        provider, has_dims = section.key
        matches = [
            e for e in entries if e.provider == provider and (e.dimensions is not None) == has_dims
        ]
        grouped.append((section, matches))
    covered = {id(e) for _, es in grouped for e in es}
    leftover = [e for e in entries if id(e) not in covered]
    if leftover:
        names = ", ".join(f"{e.provider}/{e.model}" for e in leftover)
        raise SystemExit(
            f"refresh-provider-rates.py: no SECTIONS entry matches: {names} — "
            "add a new Section() for this (provider, has_dimensions) shape before refreshing."
        )
    return grouped


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: write to a temp file in the
    SAME directory, then `os.replace()` it onto `path` (atomic rename on
    POSIX and Windows). `provider-rates.toml` is bundled into the crate at
    COMPILE TIME via `include_str!` (`core/rates.rs`) — a process crash or
    interruption mid-`write_text()` would leave the file truncated, breaking
    `cargo build` for every consumer until the next successful refresh. The
    temp file is written+flushed to completion (or removed on any failure)
    before the rename, so `path` is never observed in a partial state.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh crates/kremory/monitoring/provider-rates.toml from the LiteLLM "
            "pricing SSOT (TD-133 C5 / R2). Manual maintainer tool — NOT run by "
            "cargo build/test/CI. See this script's module docstring for design notes."
        )
    )
    parser.add_argument(
        "--source",
        default=LITELLM_SOURCE_URL,
        help=(
            "LiteLLM pricing JSON source — an https URL (default) or a local file "
            "path (for offline testing)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the regenerated toml to stdout instead of writing it.",
    )
    args = parser.parse_args()

    _, current_entries = load_current_entries(TOML_PATH)

    try:
        litellm_data = fetch_litellm_prices(args.source)
    except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        print(
            f"refresh-provider-rates.py: FAILED to fetch LiteLLM SSOT from {args.source}: {exc}\n"
            "provider-rates.toml was NOT modified — it still reflects its last "
            "manually-or-script-verified state (see rates_as_of in the file). "
            "Retry when network is available; this is a manual refresh tool, not "
            "a build-time dependency.",
            file=sys.stderr,
        )
        return 1

    litellm_idx = build_litellm_index(litellm_data)
    fetch_date = datetime.now(timezone.utc).date().isoformat()

    grouped = group_by_section(current_entries)
    refreshed_by_section: list[tuple[Section, list[Refreshed]]] = []
    any_changed = False
    unmatched: list[str] = []
    for section, entries in grouped:
        refreshed = [refresh_entry(e, section, litellm_idx) for e in entries]
        refreshed_by_section.append((section, refreshed))
        for r in refreshed:
            if r.changed:
                any_changed = True
                print(
                    f"  CHANGED  {r.entry.provider}/{r.entry.model}: "
                    f"{r.entry.cost_per_1k_tokens_usd} -> {r.cost_per_1k_tokens_usd} "
                    f"(blended $/1k)",
                    file=sys.stderr,
                )
            elif r.litellm_key is None and not section.preserve:
                unmatched.append(f"{r.entry.provider}/{r.entry.model}")

    if unmatched:
        print(
            f"  NOTE: no LiteLLM SSOT match for: {', '.join(unmatched)} — kept "
            "existing values unchanged.",
            file=sys.stderr,
        )

    output = render_toml(refreshed_by_section, fetch_date)

    if args.dry_run:
        sys.stdout.write(output)
        return 0

    _atomic_write_text(TOML_PATH, output)
    print(
        f"refresh-provider-rates.py: wrote {TOML_PATH} (rates_as_of={fetch_date}, "
        f"{'changes applied' if any_changed else 'no numeric changes, date refreshed'})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# check-sdk-scenarios.sh — run each cross-language scenario pair and assert the
# Rust and Node examples agree on the facts they recall, not just that neither
# crashes.
#
# WHY THIS EXISTS
# ----------------
# `crates/kremory-napi/examples/offline-remember-recall.mjs`'s own header
# comment already promises this script exists: "The two are kept in lockstep
# by `scripts/check-sdk-scenarios.sh`, which runs BOTH and asserts they
# agree — so the docs-site language tabs cannot show a Node snippet that no
# longer works." It didn't exist. This closes that promise for real, rather
# than leaving a comment pointing at nothing.
#
# `check-examples.sh` (Rust) and `run-all.mjs` (Node) already prove each side
# doesn't crash and its OWN assertions hold. What neither proves is that the
# two SIDES agree with each other — that the Node "mirror" of a Rust example
# actually mirrors it, rather than having quietly diverged (different facts,
# different behaviour) while still individually passing. That's the gap this
# script closes: extract the deterministic fact lines each side prints and
# diff them directly.
#
# SCOPE
# -----
# Four pairs today. Each Node file carries a "Node mirror of ..." header
# naming the Rust example it mirrors. Add a pair here ONLY once both sides
# exist AND you've picked (or written) an extraction function below that
# proves they agree on SUBSTANCE — not on byte-identical text. Rust's and
# Node's default debug/inspect printers format the same data differently on
# purpose (an `enum` variant name vs. an explicit flattened `outcome` field,
# array quote style, `snake_case` vs `camelCase` labels, console.log's
# multi-line wrapping of wide objects) — a raw text diff would fail on a
# CORRECT binding, which is worse than no check at all. Each extractor below
# says, in its own comment, exactly which asymmetry it is normalizing past
# and why that asymmetry is not a bug.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# Defense-in-depth, not the primary fix: a mirrored example using a custom
# JS callback (embedder/extractor) can leave a napi-rs ThreadsafeFunction
# handle alive past process completion (found for real in this script's
# first run — see offline-remember-recall.mjs's own fix). The real fix is
# each example force-exiting once its work is done; this timeout just stops
# a future regression from hanging the whole check. `timeout`/`gtimeout`
# aren't on every machine by default, so degrade to no timeout rather than
# fail the script outright when neither is present.
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout || true)
run_bounded() {
  if [[ -n "$TIMEOUT_BIN" ]]; then
    "$TIMEOUT_BIN" 60 "$@"
  else
    "$@"
  fi
}

# One row per mirrored scenario: rust example name, node example path, and
# which extractor function (below) knows how to pull comparable substance
# out of that scenario's specific print format.
declare -A SCENARIOS=(
  [offline-remember-recall]="offline_remember_recall crates/kremory-napi/examples/offline-remember-recall.mjs extract_facts"
  [remembers-across-sessions]="remembers_across_sessions crates/kremory-napi/examples/remembers-across-sessions.mjs extract_bitemporal"
  [multi-tenant-isolation]="multi_tenant_isolation crates/kremory-napi/examples/multi-tenant-isolation.mjs extract_verbatim"
  [undoing-a-correction]="undoing_a_correction crates/kremory-napi/examples/undoing-a-correction.mjs extract_reversal"
)

# Facts are printed as bare `subject predicate object` lines before either
# side prints its "--- prompt-ready ---" marker (which embeds a live
# timestamp and would never diff clean run-to-run). Extracting only the fact
# lines keeps the comparison deterministic.
extract_facts() {
  sed -n '/--- prompt-ready ---/q; p' "$1" | grep -E '^[a-z0-9_]+ [a-z0-9_]+ .+$'
}

# multi-tenant-isolation: both examples were written to print the identical
# narrative text verbatim (no struct/enum auto-formatting involved — every
# line is a hand-written println/console.log of plain strings). A straight
# text compare is the right tool here; only strip trailing whitespace so an
# editor's whitespace-only pass can't cause a false failure.
extract_verbatim() {
  sed -e 's/[[:space:]]*$//' "$1"
}

# remembers-across-sessions: the only asymmetry is cosmetic — Rust's `{:?}`
# on a `Vec<String>` adds a space after each comma
# (`["London", "Berlin"]`) that Node's array literal does not
# (`["London","Berlin"]`), and the trailing timestamp label is
# `recorded_at=` (Rust) vs `recordedAt=` (Node). Stripping whitespace and
# underscores and lowercasing collapses both away without touching any
# actual data value.
extract_bitemporal() {
  tr -d ' \t_' <"$1" | tr 'A-Z' 'a-z'
}

# undoing-a-correction: NOT a cosmetic difference — `UnsupersedeOutcome` is
# a tagged Rust `enum` (`Cleared { .. }` / `NotSuperseded { .. }`, so Debug
# printing uses the variant name as the tag) but the napi binding
# deliberately flattens it to one JS struct with an explicit `outcome:
# "cleared" | "not_superseded"` string field instead (see
# `UnsupersedeOutcome` in index.d.ts) — plus Node's console.log wraps a
# 5-key object across multiple lines. Comparing raw text would fail on a
# CORRECT binding, so pull out only the two things that are actually
# supposed to agree: the recalled-fact lines (identical labels on both
# sides) and the ordered sequence of outcome tags, normalized to the same
# spelling. The Node example also logs an extra "supersede result" line
# with its OWN outcome ("bounded") that the Rust example never prints at
# all — excluded explicitly, not just skipped by accident.
extract_reversal() {
  {
    grep -E "^(recorded|after 'correction'|after undo)[[:space:]]*:" "$1" \
      | tr -d " \t\"'" | tr 'A-Z' 'a-z'
    grep -v 'supersede result' "$1" \
      | grep -oE "outcome: '[a-z_]+'|Cleared|NotSuperseded" \
      | sed -E "s/outcome: '([a-z_]+)'/\1/; s/^Cleared\$/cleared/; s/^NotSuperseded\$/not_superseded/"
  }
}

fails=0
for slug in "${!SCENARIOS[@]}"; do
  read -r rust_example node_path extractor <<<"${SCENARIOS[$slug]}"

  rust_log="/tmp/check-sdk-scenarios-${slug}-rust.log"
  node_log="/tmp/check-sdk-scenarios-${slug}-node.log"

  if ! run_bounded cargo run --quiet --example "$rust_example" >"$rust_log" 2>&1; then
    echo "FAIL  $slug: Rust example '$rust_example' did not run cleanly, or timed out (see $rust_log)" >&2
    tail -5 "$rust_log" | sed 's/^/        /' >&2
    fails=$((fails + 1))
    continue
  fi

  if ! run_bounded node "$node_path" >"$node_log" 2>&1; then
    echo "FAIL  $slug: Node example '$node_path' did not run cleanly, or timed out (see $node_log)" >&2
    tail -5 "$node_log" | sed 's/^/        /' >&2
    fails=$((fails + 1))
    continue
  fi

  rust_facts=$("$extractor" "$rust_log")
  node_facts=$("$extractor" "$node_log")

  if [[ -z "$rust_facts" ]]; then
    echo "FAIL  $slug: Rust example produced nothing $extractor recognised — extractor or example output has drifted" >&2
    fails=$((fails + 1))
    continue
  fi

  if [[ "$rust_facts" != "$node_facts" ]]; then
    echo "FAIL  $slug: Rust and Node disagree on substance — the docs-site tabs would show inconsistent examples" >&2
    diff <(echo "$rust_facts") <(echo "$node_facts") | sed 's/^/        /' >&2
    fails=$((fails + 1))
    continue
  fi

  echo "  ok    $slug ($(echo "$rust_facts" | wc -l | tr -d ' ') lines agree)"
done

echo
if (( fails )); then
  echo "FAIL: $fails of ${#SCENARIOS[@]} scenario pairs disagree or broke." >&2
  exit 1
fi
echo "All ${#SCENARIOS[@]} scenario pair(s) agree across Rust and Node."

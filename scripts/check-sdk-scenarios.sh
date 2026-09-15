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
# One pair today — offline_remember_recall / offline-remember-recall.mjs —
# because that is the only Node example carrying the "Node mirror of ..."
# header convention. Add a pair here ONLY once both sides exist AND print
# their recalled facts as bare `subject predicate object` lines (see the
# extraction regex below) — that convention is what makes the diff possible
# without parsing full program output.
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

# One row per mirrored scenario: rust example name, node example path.
declare -A SCENARIOS=(
  [offline-remember-recall]="offline_remember_recall crates/kremory-napi/examples/offline-remember-recall.mjs"
)

# Facts are printed as bare `subject predicate object` lines before either
# side prints its "--- prompt-ready ---" marker (which embeds a live
# timestamp and would never diff clean run-to-run). Extracting only the fact
# lines keeps the comparison deterministic.
extract_facts() {
  sed -n '/--- prompt-ready ---/q; p' "$1" | grep -E '^[a-z0-9_]+ [a-z0-9_]+ .+$'
}

fails=0
for slug in "${!SCENARIOS[@]}"; do
  read -r rust_example node_path <<<"${SCENARIOS[$slug]}"

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

  rust_facts=$(extract_facts "$rust_log")
  node_facts=$(extract_facts "$node_log")

  if [[ -z "$rust_facts" ]]; then
    echo "FAIL  $slug: Rust example printed no recognisable fact lines — extraction regex or example output has drifted" >&2
    fails=$((fails + 1))
    continue
  fi

  if [[ "$rust_facts" != "$node_facts" ]]; then
    echo "FAIL  $slug: Rust and Node recalled DIFFERENT facts — the docs-site tabs would show inconsistent examples" >&2
    diff <(echo "$rust_facts") <(echo "$node_facts") | sed 's/^/        /' >&2
    fails=$((fails + 1))
    continue
  fi

  echo "  ok    $slug ($(echo "$rust_facts" | wc -l | tr -d ' ') facts agree)"
done

echo
if (( fails )); then
  echo "FAIL: $fails of ${#SCENARIOS[@]} scenario pairs disagree or broke." >&2
  exit 1
fi
echo "All ${#SCENARIOS[@]} scenario pair(s) agree across Rust and Node."

#!/usr/bin/env bash
# check-all.sh — run every repo guard in one command, and report three states.
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
# This repo has six standalone guard scripts and NO aggregator, NO git hooks and
# NO CI (org Actions billing suspended). Each guard therefore runs only when
# someone remembers it exists. The o11y conformance audit says so in as many
# words: "With no CI (billing suspended), this gate runs only manually — no
# enforcement."
#
# A guard nobody invokes is the same defect as a floor nobody refreshes: the
# mechanism is present, reports nothing, and reads as safety. This gives every
# guard one entry point so "did the guards pass?" is one command with one answer.
#
# ── THREE STATES, DELIBERATELY ───────────────────────────────────────────────
# PASS / FAIL / SKIP are reported separately and SKIP is never folded into PASS.
# A guard that could not run has told you nothing, and the most common way a
# gate goes quietly dead is a skip that renders green.
#
#   exit 0  -> every guard that RAN passed (skips are listed, loudly)
#   exit 1  -> at least one guard failed
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#   bash scripts/check-all.sh
#   bash scripts/check-all.sh --results results/floor/<run>.json
#
# `--results` is what enables the eval floor. It is opt-in on purpose: silently
# scoring whatever results file happens to be newest would compare an unrelated
# historic run against the floor and call the answer meaningful.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

RESULTS=""
[[ "${1:-}" == "--results" ]] && { RESULTS="${2:-}"; shift 2 || true; }

PASS=(); FAIL=(); SKIP=()
run() { # name, reason-if-skipped(""), cmd...
  local name="$1" skip="$2"; shift 2
  if [[ -n "$skip" ]]; then SKIP+=("$name — $skip"); printf '  SKIP  %s\n' "$name"; return; fi
  if "$@" > "/tmp/check-all-$name.log" 2>&1; then
    PASS+=("$name"); printf '  PASS  %s\n' "$name"
  else
    FAIL+=("$name"); printf '  FAIL  %s  (see /tmp/check-all-%s.log)\n' "$name" "$name"
    sed 's/^/          /' "/tmp/check-all-$name.log" | head -8
  fi
}

echo "=== repo guards ==="
run dual-emit            "" bash scripts/check-dual-emit.sh
run sink-callsite        "" bash scripts/check-sink-callsite-coverage.sh
run migration-idempotency "" bash scripts/check-migration-idempotency.sh
run td-id-uniqueness     "" bash scripts/check-td-id-uniqueness.sh
run file-size-ratchet    "" bash scripts/check-file-size-ratchet.sh

# e2e-consumer lives OUTSIDE the workspace with its own Cargo.lock, because its
# job is to consume the crate as an external user would. That is correct and it
# is also why it rots: `cargo build --workspace` cannot see it. The workspace
# Cargo.toml already carries a comment about this exact failure class for a
# sibling crate ("10 compile errors accrued invisibly while excluded").
#
# It rotted on 2026-09-08: the 0.8.0 DreamRequest breaking change shipped a
# CHANGELOG migration note, and the repo's own consumer was never migrated, so
# it failed to COMPILE. The full behavioural gate (run-e2e-consumer.sh) catches
# it but costs ~4min/run and needs Ollama. The compile half is free and
# deterministic, so it belongs here — in seconds, not minutes.
run e2e-consumer-compiles "" bash -c 'cd e2e-consumer && cargo build --quiet'

# The 14 offline examples SHIP INSIDE the published crate, and each one asserts
# its own behaviour rather than merely compiling. Nothing ran them until now —
# the same gap that let e2e-consumer rot against 0.8.0 for a release. ~15s total.
run examples-run "" bash scripts/check-examples.sh

if [[ -n "$RESULTS" ]]; then
  run eval-floor "" bash scripts/check-eval-floor.sh "$RESULTS"
else
  run eval-floor "no --results given; producing one needs a benchmark run, which is a standing HITL gate" true
fi

echo
echo "=== summary: ${#PASS[@]} passed, ${#FAIL[@]} failed, ${#SKIP[@]} skipped ==="
if ((${#SKIP[@]})); then
  echo "SKIPPED (these told you NOTHING — do not read as pass):"
  printf '  - %s\n' "${SKIP[@]}"
fi
if ((${#FAIL[@]})); then
  echo "FAILED:"
  printf '  - %s\n' "${FAIL[@]}"
  exit 1
fi
echo "All guards that ran passed."

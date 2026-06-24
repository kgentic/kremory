#!/usr/bin/env bash
# test-observable.sh — observable workspace test run (2026-06-24)
#
# Runs `cargo test --workspace --no-fail-fast` with LIVE, timestamped per-line
# output teed to `target/test-logs/aggregate-<ts>.log`, then prints a summary.
#
# Why this exists: a plain `cargo test --workspace` goes silent for minutes on
# slow suites (e.g. a test that shells out to clippy, or real-LLM integration
# suites). With no timestamps you cannot tell "slow" from "hung" from "killed".
# Each line here is wall-clock stamped, so a suite that started 3 minutes ago
# and hasn't finished is visibly slow, not a mystery. The teed log survives the
# session, so a machine-sleep-killed run (see memory
# `feedback_background_agents_zombie_on_machine_sleep`) leaves a forensic trail.
#
# Exit code mirrors `cargo test` (preserved through the pipe via PIPESTATUS).
#
# Usage:
#   bash scripts/test-observable.sh                 # full workspace
#   bash scripts/test-observable.sh -p kremory      # one crate
#   bash scripts/test-observable.sh --ignored       # the on-demand slow tier
#   bash scripts/test-observable.sh -- --test-threads 1
#
# Any extra args are forwarded verbatim to `cargo test`.

set -uo pipefail

ts="$(date +%Y%m%dT%H%M%S)"
log_dir="target/test-logs"
mkdir -p "$log_dir"
log="${log_dir}/aggregate-${ts}.log"

start_epoch="$(date +%s)"
{
  echo "=== test-observable run ${ts} ==="
  echo "cmd: cargo test --workspace --no-fail-fast $*"
  echo "start: $(date -u +%FT%TZ)"
  echo
} | tee "$log"

# Sleep guard: a full workspace run can take many minutes; an idle laptop that
# dozes mid-run fragments the wall-clock and can have the background process
# killed on sleep (see memory `feedback_background_agents_zombie_on_machine_sleep`).
# On macOS, wrap in `caffeinate -d -i` to block display + idle sleep for the
# run's duration. No-op (and skipped) on platforms without caffeinate.
caffeine=()
if command -v caffeinate >/dev/null 2>&1; then
  caffeine=(caffeinate -d -i)
fi

# Live timestamped stream. `perl -pe` with `$|=1` is unbuffered so lines appear
# as cargo emits them; `strftime` prefixes wall-clock. perl ships on macOS +
# Linux by default (unlike gawk strftime or moreutils `ts`). NOTE: cargo
# block-buffers when its stdout is a pipe, so these prefixes mark line *arrival*,
# not cargo emit — trust the per-suite `finished in` (cargo's own timing) in the
# summary for authoritative durations; the live stamps are for liveness only.
"${caffeine[@]}" cargo test --workspace --no-fail-fast "$@" 2>&1 \
  | perl -pe 'BEGIN { $| = 1; use POSIX qw(strftime); } $_ = strftime("%H:%M:%S ", localtime) . $_' \
  | tee -a "$log"
test_rc="${PIPESTATUS[0]}"

end_epoch="$(date +%s)"
elapsed=$(( end_epoch - start_epoch ))

{
  echo
  echo "=== summary ==="
  echo "wall-clock: ${elapsed}s"

  # Total passed/failed: scan each `test result:` line for the number that
  # precedes the `passed`/`failed` token. Robust to the timestamp prefix shift.
  grep 'test result:' "$log" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^passed/) p += $(i-1);
        if ($i ~ /^failed/) f += $(i-1);
      }
    }
    END { printf "totals: %d passed, %d failed\n", p, f }
  '

  # Flag any suite that took >30s — the candidates for the slow tier.
  echo "slow suites (>30s):"
  grep 'finished in' "$log" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "in") {
          dur = $(i+1); sub(/s$/, "", dur);
          if (dur + 0 > 30) { print "  " dur "s  " $0; found = 1 }
        }
      }
    }
    END { if (!found) print "  (none)" }
  '

  echo "exit: ${test_rc}"
  echo "log:  ${log}"
} | tee -a "$log"

exit "${test_rc}"

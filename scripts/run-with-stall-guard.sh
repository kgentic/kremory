#!/bin/bash
# Run a long command under a STALL WATCHDOG that fails fast and LOUD.
#
# Why this exists (2026-08-04): a `cargo nextest` workspace run stalled three
# times in one session with ZERO output, and each time the only signal was
# "still running" — indistinguishable from healthy work. Roughly 40 minutes were
# spent re-diagnosing the same stall because nothing ever declared itself stuck.
#
# The rule this enforces: **silence is not progress.** If the command produces no
# new output for STALL_SECS, we assume it is wedged, kill it, and print WHY —
# including the diagnostics that actually identified the cause last time
# (syspolicyd saturation + children blocked in dyld before main).
#
#   usage: run-with-stall-guard.sh <logfile> <stall_secs> <max_secs> <cmd...>
#
# Exit codes:
#   0              command succeeded
#   <command's>    command failed on its own terms
#   86             STALLED  — no output for stall_secs (loud diagnostics printed)
#   87             TIMEOUT  — exceeded max_secs of total wall clock
set -uo pipefail

LOG="$1"; STALL_SECS="$2"; MAX_SECS="$3"; shift 3
: > "$LOG"

"$@" > "$LOG" 2>&1 &
CMD_PID=$!
START=$(date +%s)

# `stat -f %m` is macOS; GNU coreutils uses `-c %Y`.
mtime() { stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo 0; }

diagnose() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════════"
  echo "  command : $*"
  echo "  log     : $LOG ($(wc -l < "$LOG" 2>/dev/null || echo 0) lines)"
  echo "  elapsed : $(( $(date +%s) - START ))s"
  echo ""
  echo "── last 5 log lines ────────────────────────────────────────────"
  tail -5 "$LOG" 2>/dev/null || echo "  (log empty — never produced output)"
  echo ""
  # These two checks are here because they are what ACTUALLY found the cause.
  # macOS Gatekeeper validates every freshly-linked binary on first exec; a
  # parallel test runner launching many at once saturates it and every child
  # blocks in dyld BEFORE main, which looks identical to a hung test.
  echo "── syspolicyd (macOS Gatekeeper) ───────────────────────────────"
  ps -Ao pcpu,comm | grep '[s]yspolicyd' || echo "  not running"
  echo "  ^ sustained >100% means Gatekeeper is the bottleneck, NOT your code."
  echo ""
  echo "── stuck children ──────────────────────────────────────────────"
  pgrep -P "$CMD_PID" 2>/dev/null | head -8 | while read -r c; do
    printf '  %s %s\n' "$(ps -o stat=,time= -p "$c" 2>/dev/null)" \
      "$(ps -o args= -p "$c" 2>/dev/null | sed 's|.*/deps/||' | cut -c1-60)"
  done || echo "  none"
  echo "  ^ many children at ~0:00.0x CPU = blocked before main (dyld), not busy."
  echo "════════════════════════════════════════════════════════════════"
}

LAST_SIZE=-1
LAST_CHANGE=$(date +%s)

while kill -0 "$CMD_PID" 2>/dev/null; do
  sleep 10
  NOW=$(date +%s)
  SIZE=$(wc -c < "$LOG" 2>/dev/null || echo 0)

  if [ "$SIZE" != "$LAST_SIZE" ]; then
    LAST_SIZE=$SIZE
    LAST_CHANGE=$NOW
  fi

  if [ $(( NOW - LAST_CHANGE )) -ge "$STALL_SECS" ]; then
    diagnose "🛑 STALLED — no output for ${STALL_SECS}s. Killing."
    pkill -P "$CMD_PID" 2>/dev/null
    kill -9 "$CMD_PID" 2>/dev/null
    exit 86
  fi

  if [ $(( NOW - START )) -ge "$MAX_SECS" ]; then
    diagnose "🛑 TIMEOUT — exceeded ${MAX_SECS}s total. Killing."
    pkill -P "$CMD_PID" 2>/dev/null
    kill -9 "$CMD_PID" 2>/dev/null
    exit 87
  fi
done

wait "$CMD_PID"
RC=$?
echo "── finished: exit $RC after $(( $(date +%s) - START ))s ──"
tail -12 "$LOG"
exit $RC

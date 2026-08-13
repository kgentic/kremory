#!/usr/bin/env bash
# G3 — SPEND TRIPWIRE. Watch a running kremory-http's own cost meter and kill it
# if the run passes a ceiling the operator declared IN ADVANCE.
#
# WHY THIS EXISTS
# ---------------
# `kremory_core_cost_usd_total` has been emitted by `chat_tracking.rs:208` for
# months and **nothing has ever read it**. A paid ingest was therefore bounded
# only by an estimate in a document. This makes OVERSPEND structurally
# impossible rather than merely noticed afterwards on an invoice.
#
# It gates on an OBSERVED counter, not a declared budget — derive over declare.
#
# THREE DESIGN POINTS, EACH LOAD-BEARING
# --------------------------------------
# 1. SUM ACROSS LABEL SETS. The metric is labelled {operation,provider,model}.
#    Reading a single line under-reports: chat and embedding are separate
#    series, and a model swap mid-run opens another.
#
# 2. THE CEILING IS REQUIRED. Unset => refuse to start. A default ceiling is a
#    ceiling nobody chose, and it would be wrong for every run but one.
#
# 3. THE METER-LIVENESS CONTROL — the non-vacuity assertion, and the reason this
#    script is more than an awk one-liner. `chat_tracking.rs:206` emits ONLY
#    when `total_cost > 0.0`, and an unlisted (provider, model) has no rate, so
#    it emits NOTHING (asserted by `chat_tracking.rs:583`). A tripwire watching
#    an absent series reads $0.00 forever and never fires — a guard that is
#    green precisely when it is broken. So: past LIVENESS_AFTER_S the summed
#    cost MUST be > 0, or this aborts the run and says why.
#
# Independently, the server logs `no chat rate found in provider-rates.toml`
# (`chat_tracking.rs:198`) when it cannot price a call. If SPEND_GUARD_SERVER_LOG
# is given, a hit there aborts immediately — two independent signals for one
# failure, because this one is invisible in the metric by construction.
#
# NOTE ON DIRECTION OF ERROR: `provider-rates.toml:156` prices Groq
# `openai/gpt-oss-120b` at a single blended rate equal to its OUTPUT price, and
# the bench workload is ~64% input. The meter therefore OVER-reports, so this
# tripwire fires EARLY. That is the safe direction for a ceiling and is
# deliberate — do not "fix" it without re-reading RECALL-LEDGER §INGEST COST.
#
# usage:
#   KREMORY_BENCH_COST_CEILING_USD=5.00 \
#     ./scripts/bench-spend-guard.sh http://localhost:3179/metrics <server-pid>
#
# env:
#   KREMORY_BENCH_COST_CEILING_USD  REQUIRED. USD. No default, by design.
#   SPEND_GUARD_POLL_S              poll interval, default 30
#   SPEND_GUARD_LIVENESS_AFTER_S    integer seconds after which the cost series
#                                   MUST be present, default 300. 0 = check on
#                                   the very first poll.
#   SPEND_GUARD_NO_LIVENESS=1       disable the liveness check ENTIRELY. Separate
#                                   from the number on purpose: "0" reads like
#                                   "off" and used to mean it, which made the
#                                   check silently unreachable. Disabling is now
#                                   an explicit act that prints a warning.
#   SPEND_GUARD_SERVER_LOG          optional server log to grep for unpriced calls
#   SPEND_GUARD_ONESHOT=1           check once and exit (used by the RED-proofs)
#
# exit codes: 0 ok/under · 10 ceiling breached · 11 meter blind · 12 unpriced
#             call in log · 2 usage error
set -uo pipefail

METRICS_URL="${1:-}"
SERVER_PID="${2:-}"

if [ -z "$METRICS_URL" ]; then
  echo "usage: $0 <metrics-url> [server-pid]" >&2
  exit 2
fi

CEILING="${KREMORY_BENCH_COST_CEILING_USD:-}"
if [ -z "$CEILING" ]; then
  echo "FATAL: KREMORY_BENCH_COST_CEILING_USD is not set." >&2
  echo "       This is deliberate: a default ceiling is a ceiling nobody chose." >&2
  echo "       Set it to the number you are willing to spend on THIS run." >&2
  exit 2
fi
case "$CEILING" in
  ''|*[!0-9.]*) echo "FATAL: ceiling '$CEILING' is not a number" >&2; exit 2 ;;
esac

POLL_S="${SPEND_GUARD_POLL_S:-30}"
LIVENESS_AFTER_S="${SPEND_GUARD_LIVENESS_AFTER_S:-300}"
NO_LIVENESS="${SPEND_GUARD_NO_LIVENESS:-0}"
SERVER_LOG="${SPEND_GUARD_SERVER_LOG:-}"
ONESHOT="${SPEND_GUARD_ONESHOT:-0}"

# `[ "$ELAPSED" -ge "$LIVENESS_AFTER_S" ]` is INTEGER comparison: a value like
# "0.0001" makes `test` error, the branch is skipped, and the liveness check
# never runs — silently. That is precisely the failure this check exists to
# prevent, so validate rather than trust. Caught by the RED-proof, 2026-08-13.
case "$LIVENESS_AFTER_S" in
  ''|*[!0-9]*)
    echo "FATAL: SPEND_GUARD_LIVENESS_AFTER_S must be a non-negative INTEGER "\
"number of seconds, got '$LIVENESS_AFTER_S'. Use SPEND_GUARD_NO_LIVENESS=1 to disable." >&2
    exit 2 ;;
esac

# ── scrape + SUM across every label set of the cost series ───────────────────
# Anchored on `{` or whitespace so a future `kremory_core_cost_usd_total_foo`
# cannot be silently folded in, and `# TYPE`/`# HELP` comment lines cannot match.
sum_cost() {
  local body
  body="$(curl -sf --max-time 10 "$METRICS_URL" 2>/dev/null)" || return 1
  printf '%s\n' "$body" \
    | awk '/^kremory_core_cost_usd_total[{ ]/ { s += $NF } END { printf "%.6f", s+0 }'
}

# Was the series present AT ALL? Distinguishes "spent $0" from "meter blind",
# which are the same number and completely different facts.
series_present() {
  curl -sf --max-time 10 "$METRICS_URL" 2>/dev/null \
    | grep -qE '^kremory_core_cost_usd_total[{ ]'
}

gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }

kill_server() {
  [ -n "$SERVER_PID" ] || return 0
  echo "[spend-guard] killing server pid=$SERVER_PID" >&2
  kill "$SERVER_PID" 2>/dev/null || true
}

echo "[spend-guard] armed: ceiling=\$${CEILING} poll=${POLL_S}s liveness=${LIVENESS_AFTER_S}s url=${METRICS_URL}"
if [ "$NO_LIVENESS" = "1" ]; then
  echo "[spend-guard] ⚠️  LIVENESS CHECK DISABLED — if the model is unpriced this" >&2
  echo "                guard will read \$0.00 forever and never fire. You have" >&2
  echo "                turned off the only thing that can tell those apart." >&2
fi

START="$(date +%s)"
while :; do
  # (a) unpriced-call warning in the server's own log — the metric CANNOT show
  #     this, because an unpriced call emits nothing at all.
  if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ] \
     && grep -aq "no chat rate found in provider-rates.toml" "$SERVER_LOG"; then
    echo "[spend-guard] ABORT: server logged an UNPRICED chat call — the cost" >&2
    echo "              meter is blind for that (provider, model) and this" >&2
    echo "              tripwire cannot bound the run. Add the model to" >&2
    echo "              crates/kremory/monitoring/provider-rates.toml." >&2
    grep -a "no chat rate found" "$SERVER_LOG" | head -3 >&2
    kill_server
    exit 12
  fi

  COST="$(sum_cost)"
  SCRAPE_RC=$?
  ELAPSED=$(( $(date +%s) - START ))

  if [ $SCRAPE_RC -ne 0 ]; then
    # Server gone or /metrics unreachable. Not a spend event — the caller owns
    # server liveness. Say so and keep waiting rather than reporting $0.
    echo "[spend-guard] t=${ELAPSED}s scrape FAILED (server down or no --features prometheus?)"
  else
    echo "[spend-guard] t=${ELAPSED}s spend=\$${COST} / ceiling \$${CEILING}"

    if gt "$COST" "$CEILING"; then
      echo "[spend-guard] 🛑 CEILING BREACHED: \$${COST} > \$${CEILING}" >&2
      kill_server
      exit 10
    fi

    # (b) liveness: past the window, a meter reading exactly zero means the
    #     series is absent, not that the run is free.
    if [ "$NO_LIVENESS" != "1" ] && [ "$ELAPSED" -ge "$LIVENESS_AFTER_S" ]; then
      if ! series_present; then
        echo "[spend-guard] ABORT: after ${ELAPSED}s the cost series is ABSENT from" >&2
        echo "              ${METRICS_URL}." >&2
        echo "              This guard cannot bound a run it cannot measure. Likely:" >&2
        echo "                • the model is not in provider-rates.toml (unpriced" >&2
        echo "                  calls emit NO metric — chat_tracking.rs:206), or" >&2
        echo "                • the server was built without --features prometheus, or" >&2
        echo "                • no paid chat call has happened (is this the FREE path?)." >&2
        kill_server
        exit 11
      fi
    fi
  fi

  [ "$ONESHOT" = "1" ] && exit 0
  sleep "$POLL_S"
done

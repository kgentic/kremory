#!/usr/bin/env bash
# Run ONE benchmark arm against a freshly-booted kremory-http, then shut it down.
#
# Latency numbers are only trustworthy when exactly one server is running and the
# machine is otherwise idle (RECALL-LEDGER §8 method rule 6: "quality is
# contention-invariant; latency is not"). This script enforces the "one server at
# a time" half — the caller is responsible for the idle machine.
#
# usage:
#   KREMORY_RERANK_K=50 KREMORY_RERANK_MODEL=jina-v1-turbo-en \
#     ./run_arm.sh <label> <db-path> [extra harness args...]
#
# Reads o11y straight out of the harness result JSON, which snapshots the
# server's own /metrics — see harness.py's `o11y` block.
set -euo pipefail

LABEL="${1:?label required}"
DB="${2:?db path required}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/results/${LABEL}.json"
LOG="$HERE/results/${LABEL}.server.log"

# PORT lets several QUALITY arms run concurrently (quality is contention-invariant).
# ALLOW_CONCURRENT=1 waives the single-server guard — set it ONLY for quality runs;
# any run whose latency you intend to quote must keep the guard on.
PORT="${PORT:-3179}"
BASE="http://localhost:${PORT}"

if [ "${ALLOW_CONCURRENT:-0}" != "1" ] && curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1; then
  echo "FATAL: something is already listening on :$PORT — latency would be contended" >&2
  exit 1
fi

# ── Contention preflight (2026-08-07) ────────────────────────────────────────
# The single-server guard above enforces "one kremory-http". It CANNOT see the
# other things on this machine that contend for the SAME Ollama: the consumer
# E2E tier (`scripts/run-e2e-consumer.sh` — each run is a full real-Ollama
# ingest + dream), `--features llm-smoke` tests in record mode, and any manual
# probe. Those are invisible to a port check.
#
# Worked example this cost: a conv0 run on 2026-08-06 sat in dream for 13 HOURS
# against a measured ~57min baseline (TD-184), and the first two hypotheses were
# a code regression. It was neither — the operator (me) ran the E2E tier, two
# full test suites, clippy and Ollama probes concurrently. Load averaged 5.75-8.55
# with 4 Ollama processes. Per project memory
# `feedback_check_machine_load_before_chasing_timing_flakiness`: check load BEFORE
# chasing timing.
#
# WARN, never block: quality is contention-invariant, so a contended QUALITY arm
# is still valid — only its latency numbers are void. Blocking would fire on
# legitimate work (`over-blocking-is-a-security-failure`).
_load1="$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')"
_ollama_procs="$(pgrep -x ollama 2>/dev/null | wc -l | tr -d ' ')"
echo "=== preflight: load1=${_load1}  ollama_procs=${_ollama_procs}"
if awk "BEGIN{exit !(${_load1:-0} > 3.0)}" 2>/dev/null; then
  echo "    ⚠️  MACHINE IS NOT IDLE (load ${_load1}). Quality numbers remain valid;"
  echo "        LATENCY numbers from this run are VOID, and dream may take many times"
  echo "        its measured baseline. Stop the E2E tier / test suites first if you"
  echo "        intend to quote timings or expect baseline wall-clock."
fi

echo "=== arm: $LABEL  db=$DB"
# `|| true` is load-bearing under `set -euo pipefail` (line 15). `grep` exits 1
# when it matches nothing, and with `pipefail` that non-zero propagates out of
# the pipeline and `-e` kills the script — so running an arm with NO `KREMORY_*`
# overrides (i.e. the SHIPPED DEFAULT config, the most important arm to be able
# to measure) aborted immediately after printing the header, with no error.
# Observed 2026-08-07. Echoing the default case explicitly rather than printing
# nothing, so the run log always records which knobs were in force.
if ! env | grep -E '^KREMORY_' | sort | sed 's/^/    /'; then
  echo "    (no KREMORY_* overrides — shipped defaults)"
fi

KREMORY_MCP_DB_PATH="$DB" PORT="$PORT" "$ROOT/target/release/kremory-http" >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
if ! curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1; then
  echo "FATAL: server never became healthy — see $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

cd "$HERE"
python3 harness.py --skip-ingest --base-url "$BASE" --output "$OUT" "$@"

echo "=== arm $LABEL done -> $OUT"

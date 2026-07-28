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

echo "=== arm: $LABEL  db=$DB"
env | grep -E '^KREMORY_' | sort | sed 's/^/    /'

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

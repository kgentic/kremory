#!/usr/bin/env bash
# Re-QUERY an EXISTING graph with a different server-side retrieval mode.
#
# WHY THIS EXISTS (2026-08-20). The entity-arm question — do entity results earn
# their top-10 slots? — was first probed by ABLATING entity items from a finished
# result file. That leaves the answerer 25 episodes where the baseline had 50
# items, i.e. strictly less context, so the comparison is confounded.
#
# `mode=content` (BM25 over episode text only) fills the WHOLE budget with
# episodes, which is the honest counterfactual. And `--skip-ingest` means the
# SAME graph answers both modes, so this is genuinely PAIRED: no extraction
# noise, no dream non-determinism, no re-ingest. Minutes, not 90.
#
# Usage: run_recall_only.sh <label> <source-db> [extra harness args...]
# Cost: £0 — no extraction happens at all under --skip-ingest.
set -euo pipefail

LABEL="${1:?label required}"
SRC_DB="${2:?source db required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$ROOT/.context/td186a-variance"
DB="$OUT_DIR/${LABEL}.db"
LOG="$OUT_DIR/${LABEL}.server.log"
RESULT="$OUT_DIR/${LABEL}.json"
PORT="${PORT:-3221}"
BASE="http://localhost:${PORT}"

[ -f "$SRC_DB" ] || { echo "FATAL: source db not found: $SRC_DB" >&2; exit 1; }
# COPY, never operate on the original — the source run is provenance for a
# committed finding and must stay byte-identical.
cp "$SRC_DB" "$DB"
rm -f "$DB-wal" "$DB-shm"
echo "=== [$LABEL] copied $SRC_DB -> $DB ($(sqlite3 "$DB" 'SELECT recorded_at FROM entities;' | wc -l | tr -d ' ') entities)"

env -u KREMORY_MCP_CHAT_BASE_URL -u KREMORY_MCP_CHAT_API_KEY \
    KREMORY_MCP_DB_PATH="$DB" PORT="$PORT" \
    "$ROOT/target/release/kremory-http" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 || {
  echo "FATAL: server never became healthy — see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }

# Same fail-closed money guard as run_local_ingest.sh. Retained even though
# --skip-ingest performs no extraction: a guard that is skipped "because it
# can't fire this time" is a guard that will be missing when it can.
grep -aqi "cloud chat" "$LOG" && { echo "FATAL: PAID cloud-chat branch." >&2; exit 1; }
grep -aqi "kremory-http booting" "$LOG" || { echo "FATAL: no boot line in $LOG." >&2; exit 1; }
echo "=== [$LABEL] money guard PASSED:"; grep -ai "kremory-http booting" "$LOG" | sed 's/^/    /'

cd "$HERE"
python3 harness.py --mode codemem --scorer substring --skip-ingest \
  --conversations 0 --recall-limit 50 \
  --base-url "$BASE" --output "$RESULT" "${@:3}"

echo "=== [$LABEL] done -> $RESULT"

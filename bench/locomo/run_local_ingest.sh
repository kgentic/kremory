#!/usr/bin/env bash
# ONE full conv0 ingest against LOCAL Ollama, with a fail-closed money guard.
#
# WHY THE GUARD EXISTS
# --------------------
# `kremory-http.rs:1051-1078` selects the extraction provider like this:
#
#     match (KREMORY_MCP_CHAT_BASE_URL, KREMORY_MCP_CHAT_API_KEY) {
#         (Some(url), Some(key)) => cloud OpenAI-compatible chat,   // PAID
#         _                      => all-Ollama,                     // FREE
#     }
#
# and this repo's `.env` sets BOTH to Groq (`https://api.groq.com`, `gsk_…`).
# So the paid path is the DEFAULT for anyone who has sourced `.env`, and
# setting `KREMORY_MCP_OLLAMA_URL` does NOT override it — only ABSENCE of the
# chat vars does. An empty string does not help either: `.ok()` on `""` still
# yields `Some("")`, which takes the cloud branch.
#
# Hence: `env -u` both vars, then ASSERT on the server's own boot log that the
# cloud branch did not fire, and ABORT before ingesting if it did. The assertion
# is on an observation (what the server logged), not on our intent — a claim
# that we "set it up right" is exactly the kind that is wrong silently.
#
# Usage:  ./run_local_ingest.sh <label>
# Cost:   £0 (local Ollama + local embedder). Wall-clock ~1.5h incl. dream.

set -euo pipefail

LABEL="${1:?label required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$ROOT/.context/td186a-variance"
mkdir -p "$OUT_DIR"

DB="$OUT_DIR/${LABEL}.db"
LOG="$OUT_DIR/${LABEL}.server.log"
RESULT="$OUT_DIR/${LABEL}.json"
PORT="${PORT:-3220}"
BASE="http://localhost:${PORT}"

rm -f "$DB" "$DB-wal" "$DB-shm"

echo "=== [$LABEL] booting kremory-http with the chat vars UNSET (free path)"
env -u KREMORY_MCP_CHAT_BASE_URL -u KREMORY_MCP_CHAT_API_KEY \
    KREMORY_MCP_DB_PATH="$DB" PORT="$PORT" \
    "$ROOT/target/release/kremory-http" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1; then
  echo "FATAL: server never became healthy — see $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

# ── MONEY GUARD (fail closed) ────────────────────────────────────────────────
# The server logs "OpenAI-compatible cloud chat (extraction)" ONLY when it took
# the paid branch. Assert on that observation before spending an hour — and
# before spending money.
if grep -aqi "cloud chat" "$LOG"; then
  echo "FATAL: server took the PAID cloud-chat branch. Refusing to ingest." >&2
  grep -ai "cloud chat" "$LOG" >&2
  exit 1
fi
# Positive confirmation too — absence of one string is weak evidence on its own.
if ! grep -aqi "kremory-http booting" "$LOG"; then
  echo "FATAL: could not find the boot line in $LOG — cannot confirm provider." >&2
  exit 1
fi
echo "=== [$LABEL] money guard PASSED — local Ollama path confirmed:"
grep -ai "kremory-http booting" "$LOG" | sed 's/^/    /'

cd "$HERE"
python3 harness.py --mode codemem --server-mode recall --scorer substring \
  --conversations 0 --recall-limit 50 \
  --base-url "$BASE" --output "$RESULT"

echo "=== [$LABEL] ingest+score done"
echo "=== [$LABEL] extraction counts (COUNT(*) is unreliable on the vector-indexed"
echo "===          tables — count lines instead, per SYSTEM-PRIMER):"
# NB: count via `recorded_at`, NEVER `id`. On the vector-indexed tables a
# `SELECT id` (like `COUNT(*)`) silently returns ZERO rows even when the table
# is populated — measured on the COMPLETE run4 db, 2026-08-11:
#     facts:    SELECT id=0   SELECT recorded_at=443   COUNT(*)=0
#     episodes: SELECT id=0   SELECT recorded_at=111   COUNT(*)=0
#     entities: SELECT id=146 SELECT recorded_at=146   COUNT(*)=0
# Note `entities.id` reads correctly while `facts.id`/`episodes.id` do not —
# which is exactly why this must be a blanket rule and not a per-table
# judgement: the tables that look safe give no signal that the others aren't.
# This block previously used `SELECT id` for facts and episodes and therefore
# reported both as 0 on every successful run.
echo "    episodes: $(sqlite3 "$DB" 'SELECT recorded_at FROM episodes;' | wc -l | tr -d ' ')"
echo "    entities: $(sqlite3 "$DB" 'SELECT recorded_at FROM entities;' | wc -l | tr -d ' ')"
echo "    facts:    $(sqlite3 "$DB" 'SELECT recorded_at FROM facts;' | wc -l | tr -d ' ')"

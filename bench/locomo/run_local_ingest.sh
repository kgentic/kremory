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

# ── TRAJECTORY SAMPLER (TD-186a) ─────────────────────────────────────────────
# WHY: the end-of-run entity count is the product of TWO stochastic stages —
# extraction, THEN dream-phase merging — and only the compounded result was ever
# recorded. Measured on run5b (2026-08-11): entities peaked at 174 after ingest
# and finished at 120, i.e. the dream phase merged 31%. Meanwhile facts (which
# merging barely touches) span 398-443 across runs (~11%) while entities span
# 120-215 (~44%). That points the variance at the MERGE stage, not extraction —
# the opposite of where TD-186a has been looking.
#
# Sampling the counts over time separates the two: the PEAK is the extraction
# result, the FINAL is post-merge, and peak-minus-final is the merge delta. This
# is non-invasive — it only reads the DB, and touches neither kremory nor the
# harness — per instrument-real-data-flow-before-hypothesizing (measure at the
# stage boundary rather than inferring the layer from an end count).
TRAJ="$OUT_DIR/${LABEL}.trajectory.csv"
echo "ts,elapsed_s,episodes,entities,facts" > "$TRAJ"
(
  START=$(date +%s)
  while true; do
    # `recorded_at`, never `id` — see the note at the summary block below.
    ep=$(sqlite3 "$DB" 'SELECT count(recorded_at) FROM episodes;' 2>/dev/null || echo -1)
    en=$(sqlite3 "$DB" 'SELECT count(recorded_at) FROM entities;' 2>/dev/null || echo -1)
    fa=$(sqlite3 "$DB" 'SELECT count(recorded_at) FROM facts;'    2>/dev/null || echo -1)
    echo "$(date -u +%H:%M:%S),$(( $(date +%s) - START )),$ep,$en,$fa" >> "$TRAJ"
    sleep 60
  done
) &
SAMPLER_PID=$!
# Re-arm the trap to reap the sampler too — the original only knew about the server.
trap 'kill "$SAMPLER_PID" 2>/dev/null || true; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

cd "$HERE"
# `"${@:2}"` forwards any extra args after <label> straight to the harness — added
# 2026-08-19 for the dream keep-or-cut A/B, whose dream-off arm is exactly
# `--no-dream`. Nothing else about the invocation changes between the two arms,
# which is what makes the comparison paired.
#
# The inline scorer stays `substring` for continuity with every prior run. The
# PRIMARY metric for the keep-or-cut decision is qa-gen, scored OFFLINE from the
# persisted `recalled_memories` by `qa_eval.py`, so it needs no flag here.
python3 harness.py --mode codemem --server-mode recall --scorer substring \
  --conversations 0 --recall-limit 50 \
  --base-url "$BASE" --output "$RESULT" "${@:2}"

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

# ── CONFOUND CONTROLS (TD-200 / TD-186a) ─────────────────────────────────────
# Arm failures are NOT incidental. Each one means that episode's extraction
# degraded to a WEAKER ladder arm — a different extraction mechanism, not another
# sample from the same one. Measured: run4 had 5/130 requests (completed), run5
# had 3/57 (ABORTED — two landed on one request and crossed the harness's 90s
# client timeout). So the rate is not the story, the CLUSTERING is, and a run
# completes or dies partly on luck. Recording this per run turns a hypothesis
# into a column, so entity-count variance can be controlled for it instead of
# being attributed wholesale to "LLM noise".
echo "    arm_failures: $(grep -ac 'arm_failure' "$LOG" || echo 0)   <- TD-200 confound control"

# Extraction peak vs post-merge final — see the TRAJECTORY SAMPLER note above.
if [ -s "$TRAJ" ]; then
  PEAK=$(awk -F, 'NR>1 && $4>m {m=$4} END{print m+0}' "$TRAJ")
  FINAL=$(awk -F, 'END{print $4+0}' "$TRAJ")
  echo "    entities peak (post-ingest): $PEAK"
  echo "    entities final (post-dream): $FINAL"
  echo "    dream merge delta:           $((PEAK - FINAL))  <- TD-186a: which stage owns the variance"
  echo "    trajectory: $TRAJ"
fi

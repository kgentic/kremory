#!/usr/bin/env bash
# G4 — ONE full bench ingest against the PAID cloud extraction path, with every
# precondition asserted BEFORE a penny is spent.
#
# THE MIRROR IMAGE OF run_local_ingest.sh
# ---------------------------------------
# That script exists to prove the run is FREE: it unsets the cloud chat vars and
# then asserts, from the server's own boot log, that the cloud branch did NOT
# fire. This script asserts the exact opposite — and, unlike its sibling, a
# mistake here costs money rather than time.
#
# Both assert on an OBSERVATION (what the server logged) rather than on intent.
# A claim that "we set it up right" is exactly the kind that is wrong silently.
#
# FIVE PRECONDITIONS, EACH ABORTING BEFORE INGEST
# -----------------------------------------------
#   P1  the cloud chat vars are set                     — else this is the free path
#   P2  a spend ceiling is declared                     — no default, on purpose
#   P3  the server took the CLOUD branch (boot log)     — observed, not assumed
#   P4  the active model is PRICED in provider-rates.toml
#         → an unpriced model emits NO cost metric at all
#           (chat_tracking.rs:206), so the spend guard would read $0.00
#           forever and never fire. A blind ceiling is worse than none.
#   P5  the spend guard is running and armed
#
# G5 — CORPUS PROVENANCE, DERIVED NOT DECLARED
# --------------------------------------------
# `facts`/`episodes` carry no model or build column, `facts.properties` is
# uniformly empty, and `dream_pass_budget_usage` has 0 rows on the existing
# corpus. So today the only record of what built a corpus is prose in a doc that
# drifts from it — and `.context/full-corpus.db`'s mtime is NOT its ingest time
# (RECALL-LEDGER §4.20: a read is a write).
#
# This writes `<db>.provenance.json` from the server's own `GET /health` — the
# same TD-202 source the recall stamp already trusts — so the question "which
# binary and which model built the corpus I am about to grade?" is answerable
# from the artifact.
#
# usage:
#   KREMORY_BENCH_COST_CEILING_USD=25.00 ./run_paid_ingest.sh <label> [harness args...]
#
# 🛑 THIS SCRIPT SPENDS MONEY. It is never run autonomously.
set -euo pipefail

LABEL="${1:?label required}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="${PAID_INGEST_OUT_DIR:-$ROOT/.context/paid-ingest}"
mkdir -p "$OUT_DIR"

DB="$OUT_DIR/${LABEL}.db"
LOG="$OUT_DIR/${LABEL}.server.log"
RESULT="$OUT_DIR/${LABEL}.json"
PROV="$DB.provenance.json"
PORT="${PORT:-3221}"
BASE="http://localhost:${PORT}"
RATES="$ROOT/crates/kremory/monitoring/provider-rates.toml"

# ── P1: this must actually BE the paid path ─────────────────────────────────
# kremory-http takes the cloud branch only when BOTH vars are present
# (kremory-http.rs:1051-1078). Running this script on the free path would burn
# an hour and produce a corpus mislabelled as paid — worse than an error.
: "${KREMORY_MCP_CHAT_BASE_URL:?P1 FAILED: not set — this is the FREE path. Use run_local_ingest.sh, or source .env}"
: "${KREMORY_MCP_CHAT_API_KEY:?P1 FAILED: not set — this is the FREE path. Use run_local_ingest.sh, or source .env}"

# ── P2: the ceiling, declared in advance ────────────────────────────────────
CEILING="${KREMORY_BENCH_COST_CEILING_USD:?P2 FAILED: set KREMORY_BENCH_COST_CEILING_USD to the amount you are willing to spend on THIS run}"

echo "=== [$LABEL] PAID INGEST"
echo "    ceiling:  \$${CEILING}"
echo "    endpoint: ${KREMORY_MCP_CHAT_BASE_URL}"
echo "    db:       $DB"

rm -f "$DB" "$DB-wal" "$DB-shm"

# Overridable so the RED-proofs can stand a stub in for the real binary. Without
# this seam P3/P4/P5 could only be exercised by actually spending money, which
# means they would ship unproven — and an unproven guard is the thing this whole
# plan exists to stop shipping.
KREMORY_HTTP_BIN="${KREMORY_HTTP_BIN:-$ROOT/target/release/kremory-http}"
if [ ! -x "$KREMORY_HTTP_BIN" ]; then
  echo "FATAL: no executable server binary at $KREMORY_HTTP_BIN" >&2
  echo "       build it: cargo build --release -p kremory-mcp --bin kremory-http \\" >&2
  echo "                   --features prometheus,content-search" >&2
  echo "       (--features prometheus is REQUIRED: without /metrics the spend" >&2
  echo "        guard cannot see anything and P5 will abort.)" >&2
  exit 1
fi

KREMORY_MCP_DB_PATH="$DB" PORT="$PORT" "$KREMORY_HTTP_BIN" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "${GUARD_PID:-}" 2>/dev/null || true; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Bounded, and the bound is overridable. 180s is right for a cold release binary
# and absurd for a RED-proof asserting a refusal: a proof that takes three
# minutes to fail gets run less often, and a guard nobody runs is not a guard.
HEALTH_WAIT_S="${PAID_INGEST_HEALTH_WAIT_S:-180}"
for _ in $(seq 1 "$HEALTH_WAIT_S"); do
  curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && break
  # Fail FAST if the server died rather than burning the whole window waiting on
  # a process that is already gone — a dead server and a slow one look identical
  # from the outside, and only one of them is worth waiting for.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: the server process exited during startup — see $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 1
done
if ! curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1; then
  echo "FATAL: server never became healthy in ${HEALTH_WAIT_S}s — see $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

# ── P3: the server actually took the CLOUD branch (observed, not assumed) ────
if ! grep -aqi "cloud chat" "$LOG"; then
  echo "FATAL (P3): the server did NOT log the cloud-chat branch — it is running" >&2
  echo "            the FREE local path. Ingesting now would produce a corpus" >&2
  echo "            labelled paid that was built by Ollama." >&2
  grep -ai "kremory-http booting" "$LOG" >&2 || true
  exit 1
fi
echo "=== [$LABEL] P3 PASSED — cloud extraction branch confirmed:"
grep -ai "cloud chat" "$LOG" | head -2 | sed 's/^/    /'

# ── P4: the active model must be PRICED, or the spend guard is blind ─────────
# Ask the SERVER which model it is using rather than reading our own env: the
# model id defaults inside the binary and `KREMORY_MCP_MODEL_ID` is NOT in .env
# (SYSTEM-PRIMER §2), so the env is not authoritative here.
HEALTH="$(curl -sf --max-time 5 "$BASE/health" || echo '{}')"
MODEL="$(printf '%s' "$HEALTH" | python3 -c '
import json,sys
try:
    h = json.load(sys.stdin)
except Exception:
    h = {}
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("chat_model", "model", "model_id", "extraction_model") and isinstance(v, str):
                return v
            r = find(v)
            if r: return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r: return r
    return None
print(find(h) or "")' 2>/dev/null || echo "")"

if [ -z "$MODEL" ]; then
  echo "FATAL (P4): could not determine the active chat model from ${BASE}/health." >&2
  echo "            Refusing to spend: without the model id the price cannot be" >&2
  echo "            verified, and an unpriced model makes the spend guard blind." >&2
  exit 1
fi
if ! grep -aq "model *= *\"${MODEL}\"" "$RATES"; then
  echo "FATAL (P4): active model '${MODEL}' is NOT priced in" >&2
  echo "            ${RATES}." >&2
  echo "            An unpriced (provider, model) emits NO cost metric at all" >&2
  echo "            (chat_tracking.rs:206), so the spend guard would read \$0.00" >&2
  echo "            forever and never fire. Add the model, then re-run." >&2
  exit 1
fi
echo "=== [$LABEL] P4 PASSED — '${MODEL}' is priced; the cost meter can see this run"

# ── P5: arm the spend guard, and confirm it is alive ────────────────────────
KREMORY_BENCH_COST_CEILING_USD="$CEILING" \
SPEND_GUARD_SERVER_LOG="$LOG" \
  "$ROOT/scripts/bench-spend-guard.sh" "$BASE/metrics" "$SERVER_PID" \
  > "$OUT_DIR/${LABEL}.spend-guard.log" 2>&1 &
GUARD_PID=$!
sleep 2
if ! kill -0 "$GUARD_PID" 2>/dev/null; then
  echo "FATAL (P5): the spend guard exited immediately — see" >&2
  echo "            $OUT_DIR/${LABEL}.spend-guard.log" >&2
  cat "$OUT_DIR/${LABEL}.spend-guard.log" >&2
  exit 1
fi
echo "=== [$LABEL] P5 PASSED — spend guard armed (pid=$GUARD_PID, ceiling \$${CEILING})"

# ── G5: stamp the corpus BEFORE ingest, from the server's own /health ────────
# NB: `python3 - <<'PY'` already consumes stdin (that is where the SCRIPT comes
# from), so /health must arrive by another channel. The first cut of this used
# BOTH a heredoc and a herestring; only the last redirect survives, so the
# script body would have been replaced by the JSON.
KREMORY_HEALTH_JSON="$HEALTH" python3 - "$PROV" "$MODEL" "$DB" <<'PY'
import json, os, subprocess, sys, time
out, model, db = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    health = json.loads(os.environ.get("KREMORY_HEALTH_JSON") or "{}")
except Exception:
    health = {}
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip() or "unknown"
json.dump({
    "corpus_db": db,
    "extraction_model": model,
    "git_sha": sha,
    "ingested_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "health": health,
    "note": ("G5 corpus provenance. Written at INGEST time from the server's own "
             "GET /health (the TD-202 source the recall stamp also trusts). The DB's "
             "mtime is NOT its ingest time — a read is a write (RECALL-LEDGER §4.20)."),
}, open(out, "w"), indent=2)
print(f"[G5] corpus provenance -> {out}")
PY

echo "=== [$LABEL] ingesting (this is where the money goes) ==="
cd "$HERE"
python3 harness.py --mode codemem --server-mode recall --scorer substring \
  --base-url "$BASE" --output "$RESULT" "$@"

echo "=== [$LABEL] done"
echo "    spend: $(tail -2 "$OUT_DIR/${LABEL}.spend-guard.log" | head -1)"
# `recorded_at`, never `id` — SYSTEM-PRIMER §2; `SELECT id` reads 0 on the
# vector-indexed tables and WHICH tables those are varies per database.
echo "    episodes: $(sqlite3 "$DB" 'SELECT recorded_at FROM episodes;' | wc -l | tr -d ' ')"
echo "    entities: $(sqlite3 "$DB" 'SELECT recorded_at FROM entities;' | wc -l | tr -d ' ')"
echo "    facts:    $(sqlite3 "$DB" 'SELECT recorded_at FROM facts;'    | wc -l | tr -d ' ')"
echo "    provenance: $PROV"

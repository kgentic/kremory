#!/usr/bin/env bash
# T8 — prove ingest + dream produce a CORRECT GRAPH, end to end, at HEAD. £0.
#
# WHY THIS IS A SCRIPT NOW
# ------------------------
# T8 was run by hand on 2026-08-11 (artefacts survive at `.context/t8-fresh.db`,
# `.context/t8-server.log`) and there was no script — which is exactly why it was
# not re-run while 15 commits landed on `crates/`, several of them touching the
# write gate, embeddings and the merge gate. A tier that only exists as tribal
# knowledge is a tier that silently stops running.
#
# WHAT IT PROVES, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
# `graph_health` asserts pipeline MECHANISM — did every dream pass run, did
# aliases resolve, are there dangling references, are entities embedded, is
# namespace scoping intact. All model-INDEPENDENT, so it is valid against a
# local-Ollama corpus and costs nothing.
#
# It does NOT assert extraction QUALITY. Local `gemma4:e4b` produces ~32%
# self-loop facts against Groq's 0.8%; asserting quality here would fail for
# reasons that say nothing about correctness. Quality is REPORTED, never gated
# (graph_health.rs:16-27). That split is the entire point: verify mechanism
# free, BEFORE spending money on a quality number.
#
# THE FIXTURE IS NOT ARBITRARY. The six episodes below encode two alias pairs —
# `acme corporation`/`acme corp` and `Sarah Martinez`/`Dr. Sarah Martinez` — so
# the L4 alias-resolution path is actually EXERCISED. Without them dream's
# alias pass runs over an empty candidate set and reports success having done
# nothing, which is a green tier that proves nothing (the 2026-08-11 run checked
# for exactly this: `candidates_examined=1 merged_or_revoked=1`).
#
# usage: ./scripts/run-t8-graph-health.sh [label]
# exit:  0 = every HARD graph-health check passed
set -uo pipefail

LABEL="${1:-t8-verify}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/.context/t8"
mkdir -p "$OUT"
DB="$OUT/${LABEL}.db"
LOG="$OUT/${LABEL}.server.log"
PORT="${PORT:-3222}"
BASE="http://localhost:${PORT}"
NS="${LABEL}"

rm -f "$DB" "$DB-wal" "$DB-shm"

# ── MONEY GUARD (fail closed) — same shape as run_local_ingest.sh ────────────
# kremory-http takes the PAID cloud branch when BOTH chat vars are present, and
# this repo's .env sets them to Groq. `env -u` both, then ASSERT on the server's
# own boot log that the cloud branch did not fire. Assert on the OBSERVATION,
# never on the intent.
echo "=== [$LABEL] booting with chat vars UNSET (free local path)"
# ⚠️ RUST_LOG IS LOAD-BEARING, not decoration. kremory declares CUSTOM tracing
# targets (`kremory.l7`, `kremory.l5`, `kremory.graph.provenance`, …) and the
# alias-pass evidence this script asserts on is `tracing::info!(target:
# "kremory.l7", …)` at `core/disambiguation/mod.rs:819`. Boot without a filter
# that names that target and the line CANNOT be emitted — at which point the
# evidence check below is asserting on the absence of something that was never
# switched on, which is worse than not checking at all.
#
# Found the hard way 2026-08-13: the first run of this script omitted RUST_LOG,
# the check printed "SUSPICIOUS — no resolve_aliases.pass_complete line", and
# the cause was this omission, not the product. `run-e2e-consumer.sh` carries
# the same warning in its header, having cost three failed diagnoses there.
export RUST_LOG="${RUST_LOG:-warn,kremory=info,kremory.l7=info,kremory.l5=info,kremory.l4=info,kremory.graph.provenance=info}"
env -u KREMORY_MCP_CHAT_BASE_URL -u KREMORY_MCP_CHAT_API_KEY \
    KREMORY_MCP_DB_PATH="$DB" PORT="$PORT" RUST_LOG="$RUST_LOG" \
    "$ROOT/target/release/kremory-http" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server exited during startup — see $LOG" >&2; tail -20 "$LOG" >&2; exit 1
  fi
  sleep 1
done
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 || {
  echo "FATAL: server never became healthy — see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }

if grep -aqi "cloud chat" "$LOG"; then
  echo "FATAL: server took the PAID cloud-chat branch. Refusing to run." >&2
  grep -ai "cloud chat" "$LOG" >&2; exit 1
fi
grep -aqi "kremory-http booting" "$LOG" || {
  echo "FATAL: no boot line in $LOG — cannot confirm the provider." >&2; exit 1; }
echo "=== [$LABEL] money guard PASSED — local Ollama path confirmed"

# ── ingest ──────────────────────────────────────────────────────────────────
post() {
  curl -sf --max-time 300 -X POST "$BASE/memories" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c '
import json,sys
print(json.dumps({"namespace": sys.argv[1], "content": sys.argv[2]}))' "$NS" "$1")" \
    >/dev/null || { echo "FATAL: ingest failed for: $1" >&2; exit 1; }
  echo "    ingested: ${1:0:64}..."
}

echo "=== [$LABEL] ingesting 6 episodes (local extraction — this is the slow part)"
# ⚠️ DO NOT "improve" these sentences. They are the VERBATIM 2026-08-11 fixture,
# recovered from `.context/t8-fresh.db`, and they are the only input known to
# make the L7 alias path fire. Keeping them identical also makes this a
# CONTROLLED comparison: same input, newer code.
#
# What actually generates the candidate is NAME CONTAINMENT, not the honorific
# or corporate-suffix variation you might expect. The 08-11 run produced exactly
# one candidate — `boston office` -> potential_alias -> `boston` — from "Boston
# conference" / "Boston office" / "Boston team" recurring across episodes.
#
# A first attempt at this script used freshly-written sentences built around
# `Sarah Martinez`/`Dr. Sarah Martinez` and `Acme Corp`/`Acme Corporation`,
# on the assumption that those were the alias triggers. They are not: that
# fixture produced ZERO `potential_alias` facts, so the alias half of T8 was
# silently unexercised while the run still reported PASS.
post "Sarah Martinez joined Acme Corporation in March 2023 as lead engineer."
post "Acme Corp announced a new product on 12 April 2023. Sarah Martinez led the launch."
post "Dr. Sarah Martinez presented research at the Boston conference on 3 July 2023."
post "Acme Corporation opened a Boston office. Sarah is the site lead there."
post "The Boston office hired twelve engineers in May 2023 under Sarah Martinez."
post "Acme Corp reported record revenue for 2023, credited to the Boston team."

# ── dream ───────────────────────────────────────────────────────────────────
echo "=== [$LABEL] running dream (consolidation)"
curl -sf --max-time 900 -X POST "$BASE/consolidation/${LABEL}?namespace=${NS}" >/dev/null || {
  echo "FATAL: consolidation call failed" >&2; tail -20 "$LOG" >&2; exit 1; }

# Did the alias pass actually EXERCISE anything? A pass over an empty candidate
# set reports success having done nothing — a green tier proving nothing.
# ── ALIAS COVERAGE — reported precisely, NOT gated ──────────────────────────
# ⚠️ Read `core/disambiguation/mod.rs:806-814` before changing this. The
# `resolve_aliases.pass_complete` line is emitted for a NON-EMPTY candidate set
# ONLY; the empty case early-returns and deliberately logs nothing, because a
# "0 of 0" line on every dream is noise (Rule 41: a signal that fires on
# ordinary work gets ignored).
#
# So ABSENCE OF THE LINE IS LEGITIMATE, and an earlier version of this script
# treated it as a hard failure — an over-blocking guard on a normal outcome,
# which is the very thing Rule 41 forbids. That was written without reading the
# comment eight lines above the emission site.
#
# The real question is not "did the pass log" but "did this fixture GENERATE any
# alias candidates at all" — because if it did not, T8's alias coverage is ZERO
# and the verdict must say so rather than implying the path was exercised.
# Answer it from the DATABASE, which cannot be silenced by a log filter.
echo "=== [$LABEL] alias coverage:"
ALIAS_FACTS=$(sqlite3 "$DB" \
  "SELECT recorded_at FROM facts WHERE predicate = 'potential_alias';" 2>/dev/null | wc -l | tr -d ' ')
if grep -aq "resolve_aliases.pass_complete" "$LOG"; then
  echo "    EXERCISED — the L7 pass ran over a non-empty candidate set:"
  grep -a "resolve_aliases.pass_complete" "$LOG" | tail -2 | sed 's/^/      /'
elif [ "$ALIAS_FACTS" -gt 0 ]; then
  echo "    ⚠️  $ALIAS_FACTS potential_alias fact(s) exist but the L7 pass logged" >&2
  echo "        nothing — that combination should not happen. Investigate before" >&2
  echo "        trusting the alias half of the verdict." >&2
else
  echo "    NOT EXERCISED — extraction produced 0 \`potential_alias\` facts, so the"
  echo "    L7 pass had nothing to examine and correctly stayed silent."
  echo "    ⇒ This run validates every OTHER mechanism check, and says NOTHING"
  echo "      about alias resolution. Treat alias coverage as ABSENT, not green."
  echo "      (The 2026-08-11 hand-run did exercise it: candidates_examined=1.)"
fi

# `recorded_at`, never `id` — SYSTEM-PRIMER §2.
for t in episodes entities facts; do
  printf "    %-9s %s\n" "$t" "$(sqlite3 "$DB" "SELECT recorded_at FROM $t;" 2>/dev/null | wc -l | tr -d ' ')"
done

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# ── the actual gate ─────────────────────────────────────────────────────────
echo "=== [$LABEL] graph_health"
cd "$ROOT"
cargo run --release -p kremory --example graph_health --features content-search -- "$DB"
GH=$?
echo "=== [$LABEL] GRAPH_HEALTH_EXIT=$GH  (0 = every HARD check passed)"
exit $GH

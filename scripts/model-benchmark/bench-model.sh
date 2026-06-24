#!/usr/bin/env bash
# Benchmark ONE Ollama model through kremory's label_precision_benchmark harness.
# Captures classification correctness (precision/recall/F1/confusion/over-extraction),
# per-call latency vs the inline 30s budget, total ingest wall-clock, model size,
# and thinking-capability — all from a single inline ingest run (no extra LLM cost).
#
# Replicable by any dev who clones the repo. Prerequisites:
#   - Ollama running (default http://localhost:11434), `nomic-embed-text` pulled
#   - the model tag you pass (pulled on demand; 404s recorded, never recommended)
#   - Rust toolchain (`cargo test --features llm-integration` compiles the harness)
#
# Usage: scripts/model-benchmark/bench-model.sh <ollama-tag> [domain]
#   domain ∈ ground_truth.json keys (default: mock_interview)
# Env:
#   OLLAMA_HOST           default http://localhost:11434
#   KREMORY_BENCH_THINK   set "false" to disable reasoning on thinking models (.think(false))
#   BENCH_OUT             output dir (default: <repo>/.ai-docs/research/local-model-benchmark-2026-06-24)
set -uo pipefail

TAG="${1:?usage: bench-model.sh <tag> [domain]}"
DOMAIN="${2:-mock_interview}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$DIR" rev-parse --show-toplevel)"
OUT="${BENCH_OUT:-$REPO/.ai-docs/research/local-model-benchmark-2026-06-24}"
LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR" "$OUT/reports"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

SAN="$(echo "$TAG" | tr '/:' '__')"
THINKSFX=""; THINKOFF="no"; [ "${KREMORY_BENCH_THINK:-}" = "false" ] && { THINKSFX=".nothink"; THINKOFF="yes"; }
LOG="$LOGDIR/${SAN}.${DOMAIN}${THINKSFX}.log"
REPORT="$OUT/reports/${SAN}.${DOMAIN}${THINKSFX}.json"
TSV="$OUT/results.tsv"

[ -f "$TSV" ] || printf 'model\tdomain\tthink_off\tpullable\tthinking_cap\tsize\trecall\tprecision\tf1\tover_extr\tfacts\tmin_rel\tstage_ms_max\twallclock_s\tfits_30s\treport\n' > "$TSV"

echo "=== $TAG ($DOMAIN, think_off=$THINKOFF) ===" | tee "$LOG"

# 1. pull-verify (the 0.3.0 trap: never recommend an unpullable tag)
if ! ollama list | awk '{print $1}' | grep -qx "$TAG"; then
  echo "pulling $TAG ..." | tee -a "$LOG"
  if ! ollama pull "$TAG" >>"$LOG" 2>&1; then
    echo "PULL FAILED (404 / not on registry): $TAG" | tee -a "$LOG"
    printf '%s\t%s\t%s\tNO\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' "$TAG" "$DOMAIN" "$THINKOFF" >> "$TSV"
    exit 0
  fi
fi

# 2. capabilities + size (manifest read — no inference)
THINKING=$(ollama show "$TAG" 2>/dev/null | awk '/Capabilities/{f=1;next}/Parameters|Projector|System|License/{f=0}f' | grep -qi thinking && echo YES || echo NO)
SIZE=$(ollama list | awk -v m="$TAG" '$1==m{print $3$4}')

# 3. run the harness, timed
START=$(date +%s)
caffeinate -dims env \
  OLLAMA_HOST="$OLLAMA_HOST" \
  OLLAMA_CHAT_MODEL="$TAG" \
  OLLAMA_KEEP_ALIVE=1h \
  KREMORY_BENCH_DOMAIN="$DOMAIN" \
  KREMORY_BENCH_REPORT="$REPORT" \
  ${KREMORY_BENCH_THINK:+KREMORY_BENCH_THINK="$KREMORY_BENCH_THINK"} \
  cargo test -p kremory --features llm-integration \
    --test label_precision_benchmark --manifest-path "$REPO/Cargo.toml" \
    -- label_precision_gte_0_75_on_mock_interview --ignored --nocapture \
    >>"$LOG" 2>&1
RC=$?
END=$(date +%s); OUTER=$((END-START))

# 4. parse (newline-safe). Enriched metrics from the harness "ENRICHED METRICS"
#    block; latency from rql.extraction.stage_ms (slowest single call across
#    stages = 30s-budget determinant); wall-clock from cargo "finished in Ns".
# value AFTER last '=' (names may contain digits, e.g. "f1"); exclude the
# RESULT echo line + take the first (ENRICHED) match so a re-appended log can't poison it.
gnum(){ grep -v '^RESULT:' "$LOG" | grep -oE "$1" | head -1 | sed -E 's/.*=//; s/%//' | tr -d '\n'; }
RECALL=$(gnum 'recall=[0-9.]+%')
PREC=$(gnum 'precision=[0-9.]+%')
F1=$(gnum 'f1=[0-9.]+%')
OVER=$(gnum 'over_extraction=[0-9]+')
FACTS=$(gnum 'facts=[0-9]+')
MINREL=$(gnum 'min_relationships=[0-9]+')
MAX=$(grep -E 'rql\.extraction\.stage_ms' "$LOG" | grep -oE 'max=[0-9.]+' | grep -oE '[0-9.]+' | sort -g | tail -1 | tr -d '\n')
WALL=$(grep -oE 'finished in [0-9.]+s' "$LOG" | tail -1 | grep -oE '[0-9.]+' | tr -d '\n')
[ -z "${WALL:-}" ] && WALL="$OUTER"
[ -z "${PREC:-}" ] && PREC="FAIL(rc=$RC)"
FITS="?"; [ -n "${MAX:-}" ] && FITS=$(awk -v m="$MAX" 'BEGIN{print (m<30000)?"YES":"NO"}')

printf '%s\t%s\t%s\tYES\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$TAG" "$DOMAIN" "$THINKOFF" "$THINKING" "$SIZE" \
  "${RECALL:-?}" "${PREC:-?}" "${F1:-?}" "${OVER:-?}" "${FACTS:-?}" "${MINREL:-?}" \
  "${MAX:-?}" "$WALL" "$FITS" "$REPORT" >> "$TSV"

echo "RESULT: recall=${RECALL:-?}% precision=${PREC:-?}% f1=${F1:-?}% over=${OVER:-?} facts=${FACTS:-?} stage_ms_max=${MAX:-?} wall=${WALL}s thinking_cap=$THINKING size=$SIZE fits30s=$FITS" | tee -a "$LOG"

#!/usr/bin/env bash
# Full model-benchmark batch — sequential (avoids model-load contention; each
# run is individually caffeinate-guarded). Wrapper pulls-on-demand + records 404s.
#
# Non-thinking models: one run (interactive `with_ollama` default candidates).
# Thinking models: TWO runs — native (reasoning on) + .think(false) — to test
# whether disabling reasoning makes them fit the inline 30s budget.
#
# Usage: scripts/model-benchmark/run-batch.sh [domain]   (default mock_interview)
# Env: BENCH_OUT (output dir), OLLAMA_HOST. See bench-model.sh.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$DIR" rev-parse --show-toplevel)"
OUT="${BENCH_OUT:-$REPO/.ai-docs/research/local-model-benchmark-2026-06-24}"
DOMAIN="${1:-mock_interview}"
BENCH="$DIR/bench-model.sh"

# Non-thinking → interactive with_ollama default candidates (gguf, registry).
# Disk-safe core set (95%-full volume): already-pulled + 2 small critical pulls
# (qwen2.5:7b ~4.7G, gemma4:e2b ~3.1G). Breadth models below are a later add-on
# batch (need ~14G more disk); uncomment to include.
NONTHINK=(
  qwen2.5:1.5b
  qwen2.5:3b
  qwen2.5:7b
  qwen2.5:14b
  gemma4-e2b:latest
  gemma4:e2b
  llama3.2:3b
  # llama3.1:8b   # breadth — ~4.9G pull
  # gemma3:4b     # breadth — ~3.3G pull
  # mistral       # breadth — ~4.1G pull
  # phi4-mini     # breadth — ~2.5G pull
)
# Thinking → deferred/dream quality candidates (run native + think:false)
THINK=(
  gemma4:e4b
  qwen3.5:9b
)

rm -f "$OUT/results.tsv"; rm -rf "$OUT/reports"  # fresh start

echo "### BATCH START $(date) domain=$DOMAIN out=$OUT"
for m in "${NONTHINK[@]}"; do
  echo ">>> $m (native)"; "$BENCH" "$m" "$DOMAIN"
done
for m in "${THINK[@]}"; do
  echo ">>> $m (native / thinking on)"; "$BENCH" "$m" "$DOMAIN"
  echo ">>> $m (think:false)"; KREMORY_BENCH_THINK=false "$BENCH" "$m" "$DOMAIN"
done
echo "### BATCH DONE $(date)"
column -t -s$'\t' "$OUT/results.tsv"
echo "collate: python3 $DIR/collate.py"

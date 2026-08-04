#!/bin/bash
# RELEASE-GATE tier: drive the PUBLIC SDK end-to-end as a real consumer would.
#
# Why this exists (V1-CANONICAL §0b-sexies, P2)
# ---------------------------------------------
# Nothing in any gate ran `e2e-consumer/`, so E2E-2 (`recall().raw()` returning
# ContentPassages in a field named `entity_id`) broke the consumer journey
# SILENTLY for weeks after ADR-078 flipped `content-search` on by default. The
# harness's own lockfile was still pinned at kremory 0.4.0 — it had not been run
# since before the 0.5.0 bump.
#
# The 1,772-test deterministic suite was green throughout. It could not see any of
# it, because every one of those tests builds from the tree and calls internals;
# this drives the published surface against a real model.
#
# WHY N RUNS, NOT ONE
# -------------------
# This journey depends on LLM output, so it is NOT deterministic. Measured
# 2026-08-05: the acronym-merge defect (E2E-1) failed 1 run in 5. A single green
# run is not evidence — it is one sample from a distribution. The script reports a
# RATE and fails if ANY run fails.
#
# WHY THESE RUST_LOG TARGETS
# --------------------------
# kremory declares CUSTOM tracing targets (`kremory.l5`, `kremory.graph.provenance`,
# …). `RUST_LOG=kremory::core::canonicalization=debug` does NOT match them — a
# filter that looks right and silently captures nothing. During E2E-1 that cost
# three failed diagnoses and wrongly eliminated the correct hypothesis. Filter on
# the TARGET string, never the module path.
#
#   usage: scripts/run-e2e-consumer.sh [runs]     (default 3)
#
# Exit: 0 = every run passed · 1 = at least one failed · 2 = preflight failed
set -uo pipefail

RUNS="${1:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${TMPDIR:-/tmp}/kremory-e2e"
mkdir -p "$LOGDIR"

# Custom targets — see the note above. `kremory=info` keeps the journey readable;
# the rest surface merge/provenance events that are `debug` and would otherwise be
# invisible exactly when something goes wrong.
export RUST_LOG="${RUST_LOG:-warn,kremory=info,kremory.l5=debug,kremory.l4=debug,kremory.graph.provenance=debug,kremory.graph.merge_reembed=debug}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target-e2e}"

# ── Preflight ────────────────────────────────────────────────────────────────
# Fail LOUD and specific. A missing model must not surface as a mid-journey
# assertion failure that reads like a kremory bug.
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
if ! curl -s --max-time 5 "$OLLAMA/api/tags" >/dev/null 2>&1; then
  echo "PREFLIGHT FAIL: Ollama not reachable at $OLLAMA"
  echo "  This tier needs a real model. Start Ollama, or skip this tier explicitly —"
  echo "  do NOT treat a skipped run as a pass."
  exit 2
fi
MODELS="$(curl -s --max-time 5 "$OLLAMA/api/tags")"
for m in gemma4:e4b nomic-embed-text; do
  if ! grep -q "\"$m" <<<"$MODELS"; then
    echo "PREFLIGHT FAIL: model '$m' not present in Ollama."
    echo "  Run: ollama pull $m"
    exit 2
  fi
done
echo "preflight OK — Ollama up, gemma4:e4b + nomic-embed-text present"
echo "RUST_LOG=$RUST_LOG"
echo

pass=0; fail=0
for i in $(seq 1 "$RUNS"); do
  log="$LOGDIR/run-$i.log"
  # Exit code captured IMMEDIATELY after the binary. Reading it after a `tail` or
  # `grep` yields THAT command's status — which reported a panicked run as "exit 0"
  # four separate times on 2026-08-05.
  cargo run --release --quiet --manifest-path "$REPO/e2e-consumer/Cargo.toml" > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    pass=$((pass+1)); echo "RUN $i/$RUNS: PASS"
  else
    fail=$((fail+1))
    reason="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E 'panicked at|^Error:' | head -1 | cut -c1-140)"
    echo "RUN $i/$RUNS: FAIL (exit $rc) :: $reason"
    echo "            log: $log"
  fi
done

echo
echo "════════════════════════════════════════════════════"
echo "  E2E CONSUMER TIER: $pass passed, $fail failed of $RUNS"
echo "════════════════════════════════════════════════════"
if [ $fail -gt 0 ]; then
  echo "FAILED. Do not cut a release on this — the consumer journey is the only tier"
  echo "that exercises the PUBLISHED surface against a real model."
  exit 1
fi
exit 0

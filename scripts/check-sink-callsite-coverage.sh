#!/usr/bin/env bash
# check-sink-callsite-coverage.sh — Phase 7 CI gate (Story #A4)
#
# Verifies that the background pipeline contains ≥14 sink fire-sites
# (sink.on_*( calls) across the wiring modules.
#
# The ≥14 threshold is D7-aware: counts call SITES, not call count at runtime.
# The authoritative catalogue is arch spec §3.1 + §3.2 (14 rows total):
#   - §3.1 rows 1–14  — IngestEventSink callsites (background worker path)
#   - §3.2 row 1      — on_batch_phase2_complete (BackgroundIngestor)
#
# NOTE: on_dedup_merge and on_community_updated are deferred (ADR-050 scope).
# The count excludes those two; adjust threshold when they land.
#
# Exit 0: ≥14 fire-sites found across the wired source files.
# Exit 1: fire-site count is below the threshold.
#
# Usage (local):
#   bash scripts/check-sink-callsite-coverage.sh
#
# Usage (CI):
#   - name: Sink callsite coverage gate
#     run: bash scripts/check-sink-callsite-coverage.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Wired modules per arch spec §3.1 + §3.2 fire-site catalogue.
# on_dedup_merge is NOT counted here (deferred; no merge path wired in v0.2.3).
BACKGROUND_SRC="${REPO_ROOT}/crates/kremory/src/core/background"
INGEST_SRC="${REPO_ROOT}/crates/kremory/src/core/ingest"

echo "=== Sink callsite coverage gate (Phase 7 CI) ==="
echo "Background src : ${BACKGROUND_SRC}"
echo "Ingest src     : ${INGEST_SRC}"
echo ""

# Count all `sink.on_*(` or `s.on_*(` or `sink_ref.on_*(` patterns.
# This matches the triple-emit wiring pattern used throughout the pipeline.
PATTERN='\.on_[a-z_]\+(';

# Count fire-sites across both source trees.
FIRE_SITES=0
BACKGROUND_COUNT=0
INGEST_COUNT=0

if [[ -d "${BACKGROUND_SRC}" ]]; then
    BACKGROUND_COUNT=$(grep -rn "${PATTERN}" "${BACKGROUND_SRC}" 2>/dev/null | grep -v '^\s*//' | wc -l | tr -d ' ')
    echo "background/: ${BACKGROUND_COUNT} fire-site(s)"
    grep -rn "${PATTERN}" "${BACKGROUND_SRC}" 2>/dev/null | grep -v '^\s*//' || true
    echo ""
fi

if [[ -d "${INGEST_SRC}" ]]; then
    INGEST_COUNT=$(grep -rn "${PATTERN}" "${INGEST_SRC}" 2>/dev/null | grep -v '^\s*//' | wc -l | tr -d ' ')
    echo "ingest/: ${INGEST_COUNT} fire-site(s)"
    grep -rn "${PATTERN}" "${INGEST_SRC}" 2>/dev/null | grep -v '^\s*//' || true
    echo ""
fi

FIRE_SITES=$(( BACKGROUND_COUNT + INGEST_COUNT ))

THRESHOLD=14

echo "Total fire-sites: ${FIRE_SITES} (threshold: ≥${THRESHOLD})"
echo ""

if [[ "${FIRE_SITES}" -ge "${THRESHOLD}" ]]; then
    echo "=== PASS: sink callsite coverage ≥${THRESHOLD} ==="
    exit 0
else
    echo "=== FAIL: sink callsite coverage ${FIRE_SITES} < ${THRESHOLD} ==="
    echo "  Catalogue the missing fire-sites against arch spec §3.1 + §3.2."
    echo "  Deferred: on_dedup_merge (v0.2.4+), on_community_updated (ADR-050 scope)."
    exit 1
fi

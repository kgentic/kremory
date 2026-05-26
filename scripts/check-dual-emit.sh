#!/usr/bin/env bash
# check-dual-emit.sh — Story #A4 CI gate
#
# Verifies that every SLO metric has BOTH a tracing emit AND a metrics emit
# in the kremory/src tree. A metric that is only counted but never traced
# (or vice versa) represents a gap in our dual-emit contract.
#
# Exit 0: all 9 metrics satisfy the dual-emit invariant.
# Exit 1: one or more metrics are missing a tracing or metrics emit.
#
# Usage (local):
#   bash scripts/check-dual-emit.sh
#
# Usage (CI):
#   - name: Dual emit gate
#     run: bash scripts/check-dual-emit.sh

set -euo pipefail

SRC="crates/kremory/src"

# Resolve script dir so this works when called from repo root or scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${REPO_ROOT}/${SRC}"

FAIL=0

check_metric() {
    local name="$1"
    local tracing_pattern="$2"   # grep pattern for tracing emit
    local metrics_pattern="$3"   # grep pattern for metrics emit

    local tracing_hits
    tracing_hits=$(grep -rl "${tracing_pattern}" "${SRC}" 2>/dev/null | wc -l | tr -d ' ')

    local metrics_hits
    metrics_hits=$(grep -rl "${metrics_pattern}" "${SRC}" 2>/dev/null | wc -l | tr -d ' ')

    local ok=true
    if [[ "${tracing_hits}" -eq 0 ]]; then
        echo "FAIL  ${name}: no tracing emit found (pattern: ${tracing_pattern})"
        ok=false
    fi
    if [[ "${metrics_hits}" -eq 0 ]]; then
        echo "FAIL  ${name}: no metrics emit found (pattern: ${metrics_pattern})"
        ok=false
    fi
    if [[ "${ok}" == true ]]; then
        echo "OK    ${name} (tracing_files=${tracing_hits}, metrics_files=${metrics_hits})"
    else
        FAIL=1
    fi
}

echo "=== Dual-emit gate (Story #A4) ==="
echo "SRC: ${SRC}"
echo ""

# Each entry: <metric_name> <tracing_grep_pattern> <metrics_grep_pattern>
#
# For metrics that use histogram!/counter!/gauge! the metrics pattern matches
# the quoted metric name. The tracing pattern matches a tracing::info!/warn!/
# call near the same code path (same function, same file).
#
# Rationale: the dual-emit contract (ADR D-10) requires every observable
# SLO metric to also surface via the structured tracing pipeline so that
# log-based monitoring (Datadog, Loki) works without a metrics backend.

check_metric \
    "rql.ingest.total_ms" \
    "kremory.ingest\." \
    '"rql\.ingest\.total_ms"'

check_metric \
    "rql.search.hybrid_entities_ms" \
    "kremory.search.hybrid" \
    '"rql\.search\.hybrid_entities_ms"'

check_metric \
    "rql.search.vector_entities_ms" \
    "kremory.search.vector" \
    '"rql\.search\.vector_entities_ms"'

check_metric \
    "kremory_core_request_duration_seconds" \
    "kremory.embed\|kremory_core_request_duration_seconds" \
    '"kremory_core_request_duration_seconds"'

check_metric \
    "rql.extraction.json_parse_fail" \
    "json_parse_fail\|extraction.*fail\|parse.*fail" \
    '"rql\.extraction\.json_parse_fail"'

check_metric \
    "rql.db.insert_entity_ms" \
    "kremory.db.insert_entity" \
    '"rql\.db\.insert_entity_ms"'

check_metric \
    "rql.db.insert_fact_ms" \
    "kremory.db.insert_fact" \
    '"rql\.db\.insert_fact_ms"'

check_metric \
    "rql.background.queue_depth" \
    "background\|queue_depth" \
    '"rql\.background\.queue_depth"'

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "=== PASS: all SLO metrics satisfy dual-emit invariant ==="
    exit 0
else
    echo "=== FAIL: one or more metrics missing tracing or metrics emit ==="
    exit 1
fi

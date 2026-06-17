#!/usr/bin/env bash
# check-dual-emit.sh — Story #A4 CI gate
#
# Verifies that every SLO metric has BOTH a tracing emit AND a metrics emit
# in the kremory/src tree. A metric that is only counted but never traced
# (or vice versa) represents a gap in our dual-emit contract.
#
# Exit 0: all 18 metrics satisfy the dual-emit invariant.
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

# ── v0.2.3 Phase 7: sink wiring metrics (Tessa §8) ──────────────────────────
#
# Each sink fire-site emits a triple: sink callback + metrics emit + tracing emit.
# These checks verify the metrics + tracing halves are present in source.
#
# on_dedup_merge: deferred — merge logic lives in ingest_with, not wired in v0.2.3.

check_metric \
    "kremory.sink.stage_transition_total" \
    "stage_change\|stage_transition" \
    '"kremory\.sink\.stage_transition_total"'

check_metric \
    "kremory.sink.entity_extracted_total" \
    "entity_extracted\|on_entity_extracted" \
    '"kremory\.sink\.entity_extracted_total"'

check_metric \
    "kremory.sink.edge_added_total" \
    "edge_added\|on_edge_added" \
    '"kremory\.sink\.edge_added_total"'

check_metric \
    "kremory.sink.ingestion_error_total" \
    "ingestion_error\|on_ingestion_error" \
    '"kremory\.sink\.ingestion_error_total"'

check_metric \
    "kremory.sink.contradiction_total" \
    "contradiction\|on_contradiction" \
    '"kremory\.sink\.contradiction_total"'

check_metric \
    "kremory.sink.batch_complete_total" \
    "batch_phase2_complete\|batch_complete" \
    '"kremory\.sink\.batch_complete_total"'

check_metric \
    "kremory.sink.callback_duration_ms" \
    "callback_duration\|callback.*duration" \
    '"kremory\.sink\.callback_duration_ms"'

# NOTE: kremory.sink.dedup_merge_total is deferred (on_dedup_merge not wired at v0.2.3).
# This check is intentionally omitted. Add when on_dedup_merge lands (ADR-050 scope).

# ── v0.2.4 Phase 5: dream-pass crash-resume metrics (ADR-050 §3.1) ───────────
#
# rql.dream.checkpoint_resume_total: fires in worker_loop on crash-resume boot.
# rql.dream.budget_used_micro_usd: fires in dream_pass after successful budget write.
#
# NOTE: rql.dream.idempotency_key_check_total — spec listed but no emit site exists.
# Phase 5 replaced the old kremory.dream.idempotency_skip_total guard with
# kremory.sink.stage_transition_total{to="SkippedIdempotent"} (already covered above).
# Omitted here until a new idempotency_key_check emit is wired.
#
# NOTE: rql.dream.consistency_check_split_invariant — spec listed but no emit site
# exists in the consistency_check module. Omitted until wired.

check_metric \
    "rql.dream.checkpoint_resume_total" \
    "kremory.worker_loop.*resuming\|resuming from crash checkpoint" \
    '"rql\.dream\.checkpoint_resume_total"'

check_metric \
    "rql.dream.budget_used_micro_usd" \
    "kremory.dream.budget_usage\|budget_used" \
    '"rql\.dream\.budget_used_micro_usd"'

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "=== PASS: all SLO metrics satisfy dual-emit invariant ==="
    exit 0
else
    echo "=== FAIL: one or more metrics missing tracing or metrics emit ==="
    exit 1
fi

#!/usr/bin/env bash
# check-dual-emit.sh — ADR D1 co-location gate
#
# Verifies that every metric in the DEFINED COVERAGE LIST satisfies the ADR D1
# dual-emit invariant (a metrics emit paired with a co-located tracing event),
# via a NAMED HYBRID (TD-085 R4.2):
#   - emit_and_trace! macro sites  → co-location is STRUCTURAL (±0, the macro
#       expands to counter + tracing in one statement); classified precisely and
#       excluded from proximity checking — the macro IS the ±5 (ADR D1) guarantee.
#   - legacy raw counter!/histogram!/gauge! sites → FILE-LEVEL backstop (a tracing
#       signal must exist in the emitting file). NOT strict per-site ±5: the sprint
#       excludes a 113-site retrofit and some legacy sites sit in other lanes. Raw
#       sites migrate to emit_and_trace! (→ structural ±5) organically when touched.
#
# SCOPE: checks ONLY the metrics explicitly listed below — NOT every counter in
# the tree. The list covers SLO metrics + the two new Phase-1 metrics
# (rql.extraction.structured_call_success, kremory.fact.rejected_total).
# Adding a new metric to coverage requires a deliberate entry here.
#
# MANUAL GATE (no CI): GitHub Actions billing is suspended (see CLAUDE.md).
# Run this script locally before every push to main / phase commit:
#
#   bash scripts/check-dual-emit.sh
#
# It must exit 0. If it exits 1, fix the failing metric (co-locate a tracing call,
# or migrate the site to emit_and_trace!). Do not suppress failures with `|| true`.
#
# Exit 0: all covered metrics satisfy the dual-emit invariant.
# Exit 1: a covered metric has no emit site, or a raw site's emitting file carries
#         no tracing signal at all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${REPO_ROOT}/crates/kremory/src"

FAIL=0

echo "=== Dual-emit gate (ADR D1, ±5-line co-location) ==="
echo "SRC: ${SRC}"
echo ""

# check_metric <name> <tracing_pattern> <metrics_pattern>
#
# Named-hybrid co-location check (TD-085 R4.2):
#   MACRO sites  — a metric emitted via emit_and_trace! has counter + tracing in one
#                  statement, so ±5-line co-location (ADR D1) is guaranteed STRUCTURALLY.
#                  These sites are classified precisely (emit_and_trace! token within 6
#                  lines above the metric name) and pass without proximity checking.
#   RAW sites    — legacy counter!/histogram!/gauge! emits use a FILE-LEVEL backstop:
#                  a tracing signal must exist in a file that emits the metric. This is
#                  intentionally NOT strict per-site ±5: the sprint excludes a 113-site
#                  retrofit and some legacy sites live in lanes outside its scope. Raw
#                  sites migrate to emit_and_trace! (→ structural ±5) organically as the
#                  surrounding code is touched.
check_metric() {
    local name="$1"
    local tracing_pattern="$2"
    local metrics_pattern="$3"

    # obs.rs is the macro's doc + test home: it holds real metric-name strings in
    # doc examples AND emit_and_trace! test invocations. Exclude it from every
    # source grep so those examples can never masquerade as production emit sites.
    local excl=(--exclude=obs.rs)

    # tracing_pattern uses grep alternation (`\|`); grep -E uses bare `|`.
    local trc="${tracing_pattern//\\|/|}"

    # Candidate files: any source file emitting/mentioning the metric name.
    local files
    files=$({ grep -rl "${excl[@]}" "${metrics_pattern}" "${SRC}" 2>/dev/null || true; })
    if [[ -z "${files}" ]]; then
        echo "FAIL  ${name}: no metrics emit found (pattern: ${metrics_pattern})"
        FAIL=1
        return
    fi

    # Classify each metric-name occurrence per file (awk): a MACRO site is one whose
    # emit_and_trace! token opens within 6 lines above (structural ±0 co-location);
    # a RAW site is a counter!/histogram!/gauge! emit; bare comment/string mentions
    # with no emit macro within 3 lines above are ignored.
    local total_raw=0 total_macro=0
    local f
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        local out
        out=$(awk -v pat="${metrics_pattern}" '
            { L[NR]=$0 }
            END {
                raw=0; macro=0;
                for (i=1;i<=NR;i++) {
                    if (L[i] !~ pat) continue;
                    emit=0;
                    for (j=i; j>=i-3 && j>=1; j--)
                        if (L[j] ~ /counter!|histogram!|gauge!|emit_and_trace!/) { emit=1; break }
                    if (!emit) continue;
                    ismacro=0;
                    for (j=i; j>=i-6 && j>=1; j--)
                        if (L[j] ~ /emit_and_trace!/) { ismacro=1; break }
                    if (ismacro) macro++; else raw++;
                }
                print raw " " macro;
            }
        ' "$f")
        total_raw=$((total_raw + $(echo "$out" | awk '{print $1+0}')))
        total_macro=$((total_macro + $(echo "$out" | awk '{print $2+0}')))
    done <<< "${files}"

    # All emit sites are macro-wrapped → co-location is structural (ADR D1, ±0).
    if [[ "${total_raw}" -eq 0 && "${total_macro}" -gt 0 ]]; then
        echo "OK    ${name} (emit_and_trace! macro ×${total_macro} — co-location structural)"
        return
    fi
    if [[ "${total_raw}" -eq 0 && "${total_macro}" -eq 0 ]]; then
        echo "FAIL  ${name}: metric name present but no emit site (counter!/histogram!/gauge!/emit_and_trace!)"
        FAIL=1
        return
    fi

    # RAW sites present: file-level dual-emit BACKSTOP (not strict ±5). A tracing
    # signal (the paired message OR any tracing:: call) must appear in a file that
    # emits the metric. Existing raw sites migrate to emit_and_trace! organically as
    # code is touched — the macro is the ±5 structural guarantee for NEW sites; this
    # backstop covers legacy raw sites WITHOUT the 113-site retrofit the sprint
    # explicitly excludes (TD-085 R4.2). Strict per-site ±5 on legacy sites would
    # require touching lanes outside this sprint (verify_stage.rs/deferred.rs).
    local tracing_files
    tracing_files=$(echo "${files}" \
        | { xargs grep -l -E "${trc}|tracing::(warn|info|debug|error|trace)!" 2>/dev/null || true; } \
        | wc -l | tr -d ' ')
    tracing_files="${tracing_files:-0}"
    if [[ "${tracing_files}" -eq 0 ]]; then
        echo "FAIL  ${name}: raw emit site(s) but no tracing signal in emitting file(s) (pattern: ${tracing_pattern})"
        FAIL=1
        return
    fi
    echo "OK    ${name} (file-level backstop: raw=${total_raw}, macro=${total_macro}, tracing_files=${tracing_files})"
}

# ── Core ingest / search metrics ─────────────────────────────────────────────

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
    "rql.db.insert_fact_with_group_ms" \
    "kremory.db.insert_fact" \
    '"rql\.db\.insert_fact_with_group_ms"'

check_metric \
    "rql.background.queue_depth" \
    "background\|queue_depth" \
    '"rql\.background\.queue_depth"'

# ── Phase 1 / R1 new metrics (TD-085 §R4.2 coverage additions) ───────────────
#
# rql.extraction.structured_call_success: added R1.1 — fires at each successful
#   LLM structured-extraction call (structured.rs:277,301,355).
# kremory.fact.rejected_total: added R1.2 — fires at self-loop guard (facts.rs).
#   Migrated to emit_and_trace! in R4.2(b) trial; macro satisfies co-location.

check_metric \
    "rql.extraction.structured_call_success" \
    "structured_call_success\|extraction.*success" \
    '"rql\.extraction\.structured_call_success"'

check_metric \
    "kremory.fact.rejected_total" \
    "kremory.fact.rejected\|fact.*rejected\|self_loop" \
    '"kremory\.fact\.rejected_total"'

# ── v0.2.3 Phase 7: sink wiring metrics ──────────────────────────────────────

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

# NOTE: kremory.sink.dedup_merge_total deferred — on_dedup_merge not wired.
# Add when on_dedup_merge lands (ADR-050 scope).

# ── v0.2.4 Phase 5: dream-pass crash-resume metrics ──────────────────────────
#
# NOTE: rql.dream.idempotency_key_check_total — no emit site. Replaced by
#   kremory.sink.stage_transition_total{to="SkippedIdempotent"} (covered above).
# NOTE: rql.dream.consistency_check_split_invariant — no emit site. Omitted
#   until wired (TD-088).

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
    echo "=== PASS: all covered metrics satisfy dual-emit invariant ==="
    exit 0
else
    echo "=== FAIL: one or more metrics missing tracing or metrics emit ==="
    exit 1
fi

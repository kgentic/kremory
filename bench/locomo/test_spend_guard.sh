#!/usr/bin/env bash
# G3 RED-PROOFS for scripts/bench-spend-guard.sh — every exit path shown FIRING.
#
# Uses `file://` metrics fixtures instead of a live server, so this costs
# nothing and needs no build. What it proves is the guard's DECISION logic; the
# scrape itself is proven separately by pointing it at a real /metrics during
# Phase 2.
#
# The liveness proof (case 4) is the one that would otherwise never be tested,
# and it is the case where the guard is green precisely because it is broken.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="$(cd "$HERE/../.." && pwd)/scripts/bench-spend-guard.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
check() { # check <label> <expected-rc> <actual-rc>
  if [ "$2" = "$3" ]; then
    echo "  ✅ $1 (rc=$3)"; pass=$((pass+1))
  else
    echo "  ❌ $1 — expected rc=$2, got rc=$3"; fail=$((fail+1))
  fi
}

# ── fixtures ────────────────────────────────────────────────────────────────
cat > "$TMP/priced.txt" <<'EOF'
# HELP kremory_core_cost_usd_total cumulative provider cost
# TYPE kremory_core_cost_usd_total gauge
kremory_core_cost_usd_total{operation="chat",provider="groq",model="openai/gpt-oss-120b"} 3.25
kremory_core_cost_usd_total{operation="embed",provider="openai",model="text-embedding-3-small"} 0.40
kremory_core_chat_duration_seconds_count{provider="groq"} 47
EOF

# The blind case: server is up, other metrics flow, the COST series is absent.
cat > "$TMP/blind.txt" <<'EOF'
# TYPE kremory_core_chat_duration_seconds histogram
kremory_core_chat_duration_seconds_count{provider="groq"} 47
EOF

# A near-miss series name — must NOT be summed into the total.
cat > "$TMP/lookalike.txt" <<'EOF'
kremory_core_cost_usd_total_estimate{operation="chat"} 999.00
kremory_core_cost_usd_total{operation="chat",provider="groq",model="m"} 1.00
EOF

echo "=== G3 spend-guard RED-proofs ==="

# 1. ceiling unset ⇒ refuse to start (a default ceiling is a ceiling nobody chose)
env -u KREMORY_BENCH_COST_CEILING_USD SPEND_GUARD_ONESHOT=1 \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "ceiling unset refuses to start" 2 $?

# 2. non-numeric ceiling ⇒ usage error
KREMORY_BENCH_COST_CEILING_USD="lots" SPEND_GUARD_ONESHOT=1 \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "non-numeric ceiling rejected" 2 $?

# 3. UNDER the ceiling ⇒ passes. NON-VACUITY: without this, a guard that always
#    exited non-zero would satisfy every other case here.
KREMORY_BENCH_COST_CEILING_USD="10.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "under ceiling passes (non-vacuity)" 0 $?

# 4. OVER the ceiling ⇒ fires. 3.25 + 0.40 = 3.65 summed ACROSS label sets;
#    a guard reading only the first line would see 3.25 and NOT fire at 3.50.
KREMORY_BENCH_COST_CEILING_USD="3.50" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "ceiling breach fires (proves label-set SUMMING)" 10 $?

# 5. THE LIVENESS PROOF — series absent past the window ⇒ abort rather than
#    reporting a comfortable \$0.00 forever.
KREMORY_BENCH_COST_CEILING_USD="10.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_LIVENESS_AFTER_S=0 \
  "$GUARD" "file://$TMP/blind.txt" >/dev/null 2>&1
check "blind meter aborts (THE liveness proof)" 11 $?

# 5b. A FRACTIONAL liveness window must be REJECTED, not silently ignored.
#     This is the bug case 5 exposed on 2026-08-13: `[ 0 -ge 0.0001 ]` is an
#     integer comparison, `test` errors, the branch is skipped, and the liveness
#     check never runs — the guard reports success having checked nothing.
KREMORY_BENCH_COST_CEILING_USD="10.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_LIVENESS_AFTER_S=0.0001 \
  "$GUARD" "file://$TMP/blind.txt" >/dev/null 2>&1
check "fractional liveness window rejected (not silently skipped)" 2 $?

# 6. an unpriced-call warning in the server log ⇒ abort, independent of the metric
echo "WARN no chat rate found in provider-rates.toml provider=groq model=mystery" \
  > "$TMP/server.log"
KREMORY_BENCH_COST_CEILING_USD="10.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 SPEND_GUARD_SERVER_LOG="$TMP/server.log" \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "unpriced-call log line aborts" 12 $?

# 7. a clean server log must NOT abort (non-vacuity for case 6)
echo "kremory-http booting, all good" > "$TMP/clean.log"
KREMORY_BENCH_COST_CEILING_USD="10.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 SPEND_GUARD_SERVER_LOG="$TMP/clean.log" \
  "$GUARD" "file://$TMP/priced.txt" >/dev/null 2>&1
check "clean server log does not abort (non-vacuity)" 0 $?

# 8. a look-alike series name must not be summed: total is 1.00, not 1000.00,
#    so a 2.00 ceiling holds.
KREMORY_BENCH_COST_CEILING_USD="2.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 \
  "$GUARD" "file://$TMP/lookalike.txt" >/dev/null 2>&1
check "look-alike metric name not summed" 0 $?

# 9. unreachable metrics endpoint is NOT reported as \$0 spend
KREMORY_BENCH_COST_CEILING_USD="1.00" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 \
  "$GUARD" "file://$TMP/does-not-exist.txt" > "$TMP/out9.log" 2>&1
rc=$?
if [ "$rc" = "0" ] && grep -q "scrape FAILED" "$TMP/out9.log"; then
  echo "  ✅ unreachable endpoint reports scrape failure, not \$0 (rc=0)"; pass=$((pass+1))
else
  echo "  ❌ unreachable endpoint — rc=$rc, log:"; cat "$TMP/out9.log"; fail=$((fail+1))
fi

# 10. the KILL path actually kills. Proven on a real process, because "it calls
#     kill" is a claim about intent and this is a claim about outcome.
sleep 120 &
victim=$!
KREMORY_BENCH_COST_CEILING_USD="0.01" SPEND_GUARD_ONESHOT=1 \
  SPEND_GUARD_NO_LIVENESS=1 \
  "$GUARD" "file://$TMP/priced.txt" "$victim" >/dev/null 2>&1
rc=$?
sleep 0.3
if [ "$rc" = "10" ] && ! kill -0 "$victim" 2>/dev/null; then
  echo "  ✅ breach KILLS the server process (verified by postcondition)"; pass=$((pass+1))
else
  echo "  ❌ kill path — rc=$rc, victim alive=$(kill -0 "$victim" 2>/dev/null && echo yes || echo no)"
  fail=$((fail+1))
fi
kill "$victim" 2>/dev/null || true

echo "=== G3: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

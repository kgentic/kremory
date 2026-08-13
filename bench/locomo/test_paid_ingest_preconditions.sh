#!/usr/bin/env bash
# G4/G5 RED-PROOFS for run_paid_ingest.sh — each of the five preconditions shown
# ABORTING, plus the corpus-provenance sidecar shown being written.
#
# Zero spend. A stub stands in for `kremory-http` via KREMORY_HTTP_BIN: it serves
# a canned /health and /metrics and writes a canned boot log, so the paid-branch
# assertion (P3), the model-pricing assertion (P4) and the spend-guard arming
# (P5) can all be exercised without an API key or a real ingest.
#
# The script aborts BEFORE ingesting in every case here — which is the whole
# design: the expensive step is last, and every precondition gates it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCRIPT="$HERE/run_paid_ingest.sh"
TMP="$(mktemp -d)"
# A fresh port per run. A fixed one collides with a previous run's stub still
# in TIME_WAIT (or one a killed suite never reaped), and the symptom is a
# HANG in the health-wait loop, not an error — observed 2026-08-13 while the
# machine was also running nextest.
PORT=$(( 3300 + (RANDOM % 400) ))
# KEEP_TMP=1 preserves the working dir so a failure can be diagnosed from the
# stub's own log rather than guessed at.
trap 'pkill -f "stub-http.py $PORT" 2>/dev/null || true; [ "${KEEP_TMP:-0}" = "1" ] && echo "[kept] $TMP" || rm -rf "$TMP"' EXIT

pass=0; fail=0
check() {
  if [ "$2" = "$3" ]; then echo "  ✅ $1 (rc=$3)"; pass=$((pass+1))
  else echo "  ❌ $1 — expected rc=$2, got rc=$3"; fail=$((fail+1)); fi
}

# ── stub server ─────────────────────────────────────────────────────────────
cat > "$TMP/stub-http.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1])
MODEL = os.environ.get("STUB_MODEL", "openai/gpt-oss-120b")
BOOT = os.environ.get("STUB_BOOT_LINE", "kremory-http booting: OpenAI-compatible cloud chat (extraction)")
print(BOOT, flush=True)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/health"):
            body = json.dumps({"status": "ok", "chat": {"model": MODEL}}).encode()
            ct = "application/json"
        elif self.path.startswith("/metrics"):
            body = (b'kremory_core_cost_usd_total{operation="chat",provider="groq",'
                    b'model="' + MODEL.encode() + b'"} 0.01\n')
            ct = "text/plain"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

HTTPServer(("127.0.0.1", PORT), H).serve_forever()
PY

# The stub is launched BY the script under test (as KREMORY_HTTP_BIN), so it must
# behave like the real binary: honour $PORT, log its boot line to stdout.
cat > "$TMP/stub-bin.sh" <<EOF
#!/usr/bin/env bash
exec python3 "$TMP/stub-http.py" "\${PORT:-$PORT}"
EOF
chmod +x "$TMP/stub-bin.sh"

export PAID_INGEST_OUT_DIR="$TMP/out"
export KREMORY_HTTP_BIN="$TMP/stub-bin.sh"

# ONE PORT PER CASE. Sharing a port across sequential cases was the only
# cross-case coupling in this harness, and it made the suite FLAKY: a previous
# case's stub lingering (or its listening socket in TIME_WAIT) makes the next
# case's server never come up, which surfaces as "server never became healthy"
# — a timeout, not an error, so it reads as slowness rather than a collision.
# Observed 2026-08-13: the same suite passed, hung, failed 6/8, then passed
# again, on identical code. Fixed by removing the sharing rather than by
# lengthening the timeout, which would only have made the race rarer.
next_port() { PORT=$((PORT + 1)); export PORT; }
# Keep the RED-proofs fast: these assert REFUSALS, so a 180s health window is
# three minutes of waiting for a failure we want to observe immediately.
export PAID_INGEST_HEALTH_WAIT_S=60
# Harness args that would make a real run cheap — never reached in these tests.
ARGS=(--conversations 0)

echo "=== G4/G5 paid-ingest precondition RED-proofs ==="

# P1 — cloud vars absent ⇒ this is the free path, refuse
env -u KREMORY_MCP_CHAT_BASE_URL -u KREMORY_MCP_CHAT_API_KEY \
    KREMORY_BENCH_COST_CEILING_USD=5 \
    bash "$SCRIPT" p1 "${ARGS[@]}" >/dev/null 2>&1
check "P1 refuses when the cloud chat vars are unset" 1 $?

# P2 — no declared ceiling ⇒ refuse
env -u KREMORY_BENCH_COST_CEILING_USD \
    KREMORY_MCP_CHAT_BASE_URL=https://api.groq.com \
    KREMORY_MCP_CHAT_API_KEY=gsk_test \
    bash "$SCRIPT" p2 "${ARGS[@]}" >/dev/null 2>&1
check "P2 refuses without a declared spend ceiling" 1 $?

next_port
# P3 — server logs the FREE branch ⇒ refuse (observed, not assumed)
STUB_BOOT_LINE="kremory-http booting: all-Ollama (extraction)" \
KREMORY_MCP_CHAT_BASE_URL=https://api.groq.com \
KREMORY_MCP_CHAT_API_KEY=gsk_test \
KREMORY_BENCH_COST_CEILING_USD=5 \
    bash "$SCRIPT" p3 > "$TMP/p3.log" 2>&1
rc=$?
pkill -f "stub-http.py $PORT" 2>/dev/null || true; sleep 0.4
if [ "$rc" = "1" ] && grep -q "FATAL (P3)" "$TMP/p3.log"; then
  echo "  ✅ P3 refuses when the server took the FREE branch (rc=1)"; pass=$((pass+1))
else
  echo "  ❌ P3 — rc=$rc"; tail -5 "$TMP/p3.log"; fail=$((fail+1))
fi

next_port
# P4 — active model NOT priced ⇒ refuse, because the spend guard would be blind
STUB_MODEL="mystery/unpriced-model-9000" \
KREMORY_MCP_CHAT_BASE_URL=https://api.groq.com \
KREMORY_MCP_CHAT_API_KEY=gsk_test \
KREMORY_BENCH_COST_CEILING_USD=5 \
    bash "$SCRIPT" p4 > "$TMP/p4.log" 2>&1
rc=$?
pkill -f "stub-http.py $PORT" 2>/dev/null || true; sleep 0.4
if [ "$rc" = "1" ] && grep -q "FATAL (P4)" "$TMP/p4.log"; then
  echo "  ✅ P4 refuses an UNPRICED model (spend guard would be blind) (rc=1)"; pass=$((pass+1))
else
  echo "  ❌ P4 — rc=$rc"; tail -5 "$TMP/p4.log"; fail=$((fail+1))
fi

next_port
# NON-VACUITY — all five preconditions PASS on a well-formed setup, and the run
# then proceeds as far as the harness. Without this, a script that aborted
# unconditionally would satisfy every case above.
KREMORY_MCP_CHAT_BASE_URL=https://api.groq.com \
KREMORY_MCP_CHAT_API_KEY=gsk_test \
KREMORY_BENCH_COST_CEILING_USD=5 \
    bash "$SCRIPT" ok > "$TMP/ok.log" 2>&1
rc=$?
pkill -f "stub-http.py $PORT" 2>/dev/null || true; sleep 0.4
for p in P3 P4 P5; do
  if grep -q "$p PASSED" "$TMP/ok.log"; then
    echo "  ✅ $p PASSES on a well-formed setup (non-vacuity)"; pass=$((pass+1))
  else
    echo "  ❌ $p did not pass on a well-formed setup"; tail -20 "$TMP/ok.log"; fail=$((fail+1))
  fi
done

# G5 — the corpus provenance sidecar is written, and carries the facts that make
# a corpus attributable: which model, which build, when.
PROVFILE="$TMP/out/ok.db.provenance.json"
if [ -f "$PROVFILE" ]; then
  if python3 - "$PROVFILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
missing = [k for k in ("corpus_db", "extraction_model", "git_sha", "ingested_at_utc")
           if not d.get(k)]
assert not missing, f"sidecar missing {missing}"
assert d["extraction_model"] == "openai/gpt-oss-120b", d["extraction_model"]
assert d["git_sha"] != "unknown", "git sha not captured"
print("    model=%s sha=%s at=%s" % (d["extraction_model"], d["git_sha"][:8], d["ingested_at_utc"]))
PY
  then echo "  ✅ G5 corpus provenance sidecar written and complete"; pass=$((pass+1))
  else echo "  ❌ G5 sidecar present but incomplete"; fail=$((fail+1)); fi
else
  echo "  ❌ G5 sidecar not written at $PROVFILE"; tail -20 "$TMP/ok.log"; fail=$((fail+1))
fi

echo "=== G4/G5: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]

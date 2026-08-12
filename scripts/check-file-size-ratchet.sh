#!/usr/bin/env bash
# check-file-size-ratchet.sh — the guard TD-002 was CLOSED on but never had.
#
# TD-002 was closed citing a "module-size lint". `clippy::too_many_lines` is
# PER-FUNCTION by design, and no whole-file cap was configured anywhere. So the
# thing TD-002 promised never existed, nothing ever fired, and the two files it
# was meant to protect grew to 3,282 (`facade/mod.rs`) and 2,704
# (`ingest/pipeline/ingest_with.rs`) lines. Verified 2026-08-12 by the register
# audit; see TD-001, TD-002, TD-043, TD-045.
#
# ── ONE JOB ──────────────────────────────────────────────────────────────────
# This is a RATCHET, not a cap. It fails only when a file grows PAST the size it
# already had. It deliberately does NOT enforce an absolute limit, because a hard
# cap would fail on today's tree and a guard that fires on ordinary work gets
# disabled — which is how you end up with no guard at all, which is the situation
# it exists to end.
#
#   - file shrank            -> PASS (and you should re-baseline; it says so)
#   - file grew, <= baseline -> PASS
#   - file grew  > baseline  -> FAIL, naming the file and the delta
#   - brand-new file         -> PASS if under NEW_FILE_CEILING, else FAIL
#
# Re-baseline deliberately and on the record after a real split:
#   bash scripts/check-file-size-ratchet.sh --update
#
# No CI exists in this project; run it manually before a push, like
# `check-dual-emit.sh`.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
BASELINE="scripts/file-size-baseline.txt"
NEW_FILE_CEILING=800   # a NEW file may not be born a god-file

if [[ "${1:-}" == "--update" ]]; then
  git ls-files '*.rs' | while read -r f; do
    printf '%s %s\n' "$(wc -l < "$f" | tr -d ' ')" "$f"
  done | sort -k2 > "$BASELINE"
  echo "re-baselined $(wc -l < "$BASELINE" | tr -d ' ') files -> $BASELINE"
  echo "COMMIT THIS DELIBERATELY — a silent re-baseline turns the ratchet into decoration."
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "FAIL: $BASELINE missing. Create it with: bash $0 --update" >&2
  exit 1
fi

fail=0
grown=""
born=""

while read -r f; do
  cur=$(wc -l < "$f" | tr -d ' ')
  base=$(awk -v want="$f" '$2 == want {print $1; exit}' "$BASELINE")
  if [[ -z "$base" ]]; then
    if (( cur > NEW_FILE_CEILING )); then
      born+="  $f — born at $cur lines (ceiling $NEW_FILE_CEILING)"$'\n'
      fail=1
    fi
  elif (( cur > base )); then
    grown+="  $f — $base -> $cur (+$((cur - base)))"$'\n'
    fail=1
  fi
done < <(git ls-files '*.rs')

if (( fail )); then
  echo "FAIL: file-size ratchet (TD-002)" >&2
  [[ -n "$grown" ]] && { echo "" >&2; echo "GREW past its recorded size:" >&2; printf '%s' "$grown" >&2; }
  [[ -n "$born"  ]] && { echo "" >&2; echo "NEW file over the ceiling:" >&2; printf '%s' "$born" >&2; }
  echo "" >&2
  echo "Split the file, or re-baseline ON PURPOSE with: bash $0 --update" >&2
  exit 1
fi

echo "PASS: no .rs file grew past its recorded size (baseline: $(wc -l < "$BASELINE" | tr -d ' ') files)"

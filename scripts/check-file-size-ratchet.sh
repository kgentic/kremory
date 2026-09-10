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
#   - file shrank                    -> PASS (and you should re-baseline; it says so)
#   - file grew, <= baseline         -> PASS
#   - file grew  > baseline          -> see the TWO TIERS below
#   - brand-new file                 -> PASS if under NEW_FILE_CEILING, else FAIL
#
# ── THREE TIERS AND A GUARDED REFRESH (TD-243) ───────────────────────────────
# A single tier put the signal in the noise. On 2026-09-08 this script failed
# naming 100 files: the two god-files it exists for were entries 2 and 47, among
# 98 entries of +1 / +4 / +6 from ordinary work. Every entry was a TRUE growth —
# so this was not a false-positive problem, it was an ALARM FATIGUE problem, and
# a 100-line failure nobody reads is indistinguishable from a guard that is off.
# That is the exact state this script's own header says it exists to end.
#
#   PINNED file   (`pin=<TD>` in the baseline)  -> FAIL on ANY growth, and
#                                                  `--update` may NEVER raise it
#   WATCHED file  (baseline >= WATCH_LINES)     -> FAIL on ANY growth
#   ordinary file (baseline <  WATCH_LINES)     -> FAIL only past GROWTH_THRESHOLD,
#                                                  smaller growth reported as INFO
#
# WATCHED is DERIVED from the baseline, never hand-listed: a named list cannot
# notice the next file to cross the line, and keeps naming one that has since
# been split.
#
# PINNED cannot be derived — it is a statement that a file's size is SOMEONE'S
# TRACKED WORK, which no line count implies. It lives in the baseline file
# beside the number it protects, as a third field, so the policy and the data
# cannot drift apart.
#
# ── WHY `--update` IS NO LONGER FORBIDDEN ────────────────────────────────────
# TD-243 banned `--update` outright, to stop anyone clearing the red by
# re-baselining the two god-files. Correct intent, wrong mechanism: with the
# ONLY refresh path banned, the baseline froze on 2026-08-12 and four weeks of
# legitimate growth accumulated against a stale reference. A ratchet whose
# baseline can never be refreshed is not a ratchet; it is a flood that rises
# until someone deletes the guard.
#
# `--update` now REFUSES to raise a pinned entry — it can only lower one, which
# is what a real split produces. So refreshing ordinary drift is safe, the
# prohibition is enforced by the code rather than by a comment asking nicely,
# and the two files TD-043 / TD-045 track stay red until they are actually
# split. Enforce the invariant where the write happens, not in prose.
#
# INFO lines are printed, not suppressed. Drift you cannot see is drift you
# cannot ratchet later.
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
WATCH_LINES=1500       # at/above this, ANY growth fails — no threshold, no grace
GROWTH_THRESHOLD=25    # below WATCH_LINES, growth up to this is INFO, not FAIL

if [[ "${1:-}" == "--update" ]]; then
  if [[ ! -f "$BASELINE" ]]; then
    echo "FAIL: $BASELINE missing — refusing to create one from scratch here." >&2
    echo "A first baseline must be written deliberately, not as a side effect." >&2
    exit 1
  fi
  tmp=$(mktemp)
  held=0
  git ls-files '*.rs' | while read -r f; do
    cur=$(wc -l < "$f" | tr -d ' ')
    # Preserve the pin marker AND, for a pinned file, refuse to raise its
    # recorded size. Lowering is allowed — that is what a split produces, and
    # it is the only direction a pin is meant to move.
    read -r base pin < <(awk -v want="$f" '$2 == want {print $1, ($3 == "" ? "-" : $3); exit}' "$BASELINE")
    if [[ -n "${pin:-}" && "$pin" != "-" ]]; then
      if (( cur > base )); then
        printf '%s %s %s\n' "$base" "$f" "$pin" >> "$tmp"
        echo "HELD: $f pinned at $base (is $cur, +$((cur - base))) — $pin" >&2
      else
        printf '%s %s %s\n' "$cur" "$f" "$pin" >> "$tmp"
      fi
    else
      # A file with no baseline entry is being RECORDED for the first time. If
      # it is already over the ceiling that would have rejected it at birth, say
      # so — a refresh is where an oversized new file would otherwise slip in
      # silently, and this is the one moment anyone is looking.
      if [[ -z "${base:-}" ]] && (( cur > NEW_FILE_CEILING )); then
        echo "RECORDED OVER CEILING: $f at $cur lines (ceiling $NEW_FILE_CEILING) — now watched, any growth fails" >&2
      fi
      printf '%s %s\n' "$cur" "$f" >> "$tmp"
    fi
  done
  sort -k2 "$tmp" > "$BASELINE"
  rm -f "$tmp"
  held=$(awk 'NF == 3' "$BASELINE" | wc -l | tr -d ' ')
  echo "re-baselined $(wc -l < "$BASELINE" | tr -d ' ') files -> $BASELINE ($held pinned, never raised)"
  echo "COMMIT THIS DELIBERATELY — a silent re-baseline turns the ratchet into decoration."
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "FAIL: $BASELINE missing. Create it with: bash $0 --update" >&2
  exit 1
fi

fail=0
pinned=""
watched=""
grown=""
born=""
drift=""
drift_n=0
drift_lines=0

while read -r f; do
  cur=$(wc -l < "$f" | tr -d ' ')
  base=$(awk -v want="$f" '$2 == want {print $1; exit}' "$BASELINE")
  pin=$(awk -v want="$f" '$2 == want {print $3; exit}' "$BASELINE")
  if [[ -z "$base" ]]; then
    if (( cur > NEW_FILE_CEILING )); then
      born+="  $f — born at $cur lines (ceiling $NEW_FILE_CEILING)"$'\n'
      fail=1
    fi
  elif (( cur > base )); then
    delta=$(( cur - base ))
    if [[ -n "$pin" ]]; then
      # Pinned: its size is someone's tracked work. No grace, no threshold, and
      # `--update` above cannot raise it either.
      pinned+="  $f — $base -> $cur (+$delta) [pinned: $pin]"$'\n'
      fail=1
    elif (( base >= WATCH_LINES )); then
      # Already a god-file at baseline. No grace: the whole point of watching it
      # is that it must only ever get smaller.
      watched+="  $f — $base -> $cur (+$delta) [watched: baseline >= $WATCH_LINES]"$'\n'
      fail=1
    elif (( delta > GROWTH_THRESHOLD )); then
      grown+="  $f — $base -> $cur (+$delta, over +$GROWTH_THRESHOLD)"$'\n'
      fail=1
    else
      drift+="  $f — $base -> $cur (+$delta)"$'\n'
      drift_n=$(( drift_n + 1 ))
      drift_lines=$(( drift_lines + delta ))
    fi
  fi
done < <(git ls-files '*.rs')

# INFO first and on stdout: it is not a failure, and burying it under a FAIL on
# stderr is how the drift became invisible in the first place.
if (( drift_n )); then
  echo "INFO: $drift_n file(s) grew within +$GROWTH_THRESHOLD (+$drift_lines lines total) — not a failure:"
  printf '%s' "$drift"
  echo ""
fi

if (( fail )); then
  echo "FAIL: file-size ratchet (TD-002, tiers per TD-243)" >&2
  [[ -n "$pinned" ]] && { echo "" >&2; echo "PINNED file grew — this is the tracked signal, split it:" >&2; printf '%s' "$pinned" >&2; }
  [[ -n "$watched" ]] && { echo "" >&2; echo "WATCHED file grew (baseline >= $WATCH_LINES; any growth fails):" >&2; printf '%s' "$watched" >&2; }
  [[ -n "$grown" ]] && { echo "" >&2; echo "GREW past +$GROWTH_THRESHOLD:" >&2; printf '%s' "$grown" >&2; }
  [[ -n "$born"  ]] && { echo "" >&2; echo "NEW file over the ceiling:" >&2; printf '%s' "$born" >&2; }
  echo "" >&2
  echo "Split the file, or re-baseline ON PURPOSE with: bash $0 --update" >&2
  echo "-- which will NOT raise a pinned entry, so the lines above marked" >&2
  echo "   [pinned: ...] cannot be cleared that way. Splitting is the only exit." >&2
  exit 1
fi

echo "PASS: no .rs file grew past its recorded size (baseline: $(wc -l < "$BASELINE" | tr -d ' ') files)"

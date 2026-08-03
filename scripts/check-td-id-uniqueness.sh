#!/usr/bin/env bash
# check-td-id-uniqueness.sh — integrity guard over the tech-debt register.
#
# Runs TWO independent checks. Each has exactly one job, so neither over-blocks:
#
#   CHECK 1 — DUPLICATE DECLARATION
#     A TD id declared twice for two DIFFERENT items makes every cross-reference
#     to it unresolvable. On 2026-07-29, TD-079 named both "as_of point-in-time
#     recall" (CLOSED) and "Court entity type" (OPEN, HIGH) — and a session that
#     read the CLOSED one reported "TD-079 closed" while open HIGH-priority work
#     sat 68 lines below under the same number.
#
#   CHECK 2 — OPEN ITEM FILED UNDER `## Resolved`
#     An entry inside the Resolved section whose status is OPEN/PARTIAL is
#     invisible to anyone scanning for open work. Same class of defect: the
#     record says done, the work is not.
#
# WHY THE RESOLVED SECTION IS EXEMPT FROM CHECK 1: the register's own stated
# convention is "when an item is fixed, move it to the `## Resolved` section" —
# so a closure record legitimately repeats its id and title. A first draft of
# this guard flagged those and produced 5 false positives out of 9 findings. A
# guard that fires on correct work gets ignored, so Check 1 parses the section
# structure rather than matching heading text.
#
# Usage:  scripts/check-td-id-uniqueness.sh [path-to-register]
# Exit:   0 = clean · 1 = defects found

set -euo pipefail

REGISTER="${1:-.ai-docs/tech-debt/tech-debt-register.md}"

if [[ ! -f "$REGISTER" ]]; then
  echo "FAIL: register not found at '$REGISTER'" >&2
  exit 1
fi

# ── Locate the `## Resolved` section boundaries ──────────────────────────────
resolved_start="$(grep -nE '^## Resolved' "$REGISTER" | head -1 | cut -d: -f1 || true)"

if [[ -z "$resolved_start" ]]; then
  echo "FAIL: no '## Resolved' section found — the guard's assumptions about the" >&2
  echo "      register's structure have drifted. Fix the guard; do NOT assume clean." >&2
  exit 1
fi

resolved_end="$(
  awk -v s="$resolved_start" 'NR>s && /^## / {print NR; exit}' "$REGISTER"
)"
resolved_end="${resolved_end:-$(wc -l < "$REGISTER")}"

# Follow-up markers: recording progress in place is deliberate and good. What is
# banned is re-using a number for a DIFFERENT item. Extend deliberately — every
# addition widens what the guard stops seeing.
FOLLOWUP_MARKERS='RESOLVED|CORRECTION|update|UPDATE|A/B RESULT|Stage [A-Z]|CLOSED|✅|❌|NOT VIABLE|erratum|ERRATUM'

fail=0

# ── CHECK 1 — duplicate declarations outside the Resolved section ────────────
declarations="$(
  grep -nE '^### TD-[0-9]{3}' "$REGISTER" \
    | awk -F: -v s="$resolved_start" -v e="$resolved_end" '$1 < s || $1 > e' \
    | grep -vE "$FOLLOWUP_MARKERS" \
    || true
)"

# A guard that silently matches nothing reports success forever.
if [[ -z "$declarations" ]]; then
  echo "FAIL: no TD declarations matched — the guard's pattern has drifted from" >&2
  echo "      the register's format. Fix the guard; do NOT assume the file is clean." >&2
  exit 1
fi

total="$(echo "$declarations" | wc -l | tr -d ' ')"
dupes="$(echo "$declarations" | sed -E 's/^[0-9]+:### (TD-[0-9]{3}).*/\1/' | sort | uniq -d)"

if [[ -n "$dupes" ]]; then
  fail=1
  echo "FAIL [check 1] — TD id declared twice for different items; refs are ambiguous:"
  echo
  while read -r id; do
    [[ -z "$id" ]] && continue
    echo "  $id"
    echo "$declarations" | grep -E "^[0-9]+:### ${id}" | sed 's/^/      line /' | cut -c1-120
    echo
  done <<< "$dupes"
fi

# ── CHECK 2 — OPEN/PARTIAL entries filed under `## Resolved` ─────────────────
misfiled="$(
  awk -v s="$resolved_start" -v e="$resolved_end" '
    NR > s && NR < e {
      if ($0 ~ /^### TD-[0-9]{3}/) { hdr = $0; hdrline = NR; reported = 0 }
      if (!reported && hdr != "" && $0 ~ /^\*\*Status\*\*:.*(OPEN|STILL OPEN|PARTIAL)/) {
        printf "      line %s: %s\n", hdrline, substr(hdr, 1, 110)
        reported = 1
      }
    }
  ' "$REGISTER"
)"

if [[ -n "$misfiled" ]]; then
  fail=1
  echo "FAIL [check 2] — entries under '## Resolved' whose status is OPEN/PARTIAL."
  echo "                 Open work filed as done is invisible to anyone scanning."
  echo
  echo "$misfiled"
  echo
fi

if [[ $fail -eq 0 ]]; then
  echo "PASS: ${total} TD declarations outside '## Resolved', all ids unique;"
  echo "      no OPEN/PARTIAL entries misfiled under '## Resolved'."
  exit 0
fi

echo "Fix for check 1: give the NEWER item a fresh id; if renumbering would break"
echo "existing cross-references, record both in a collision index and never re-use"
echo "the number. Fix for check 2: move the entry back out of '## Resolved', or"
echo "correct its status if it is genuinely done."
exit 1

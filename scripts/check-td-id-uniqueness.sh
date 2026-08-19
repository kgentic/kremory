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

# ⚠️ The Resolved section's MEMBERS are themselves `## ` headings.
#
# This was `NR>s && /^## /` — "the section ends at the next level-2 heading". The
# very next level-2 heading is `## TD-216`, TWO LINES below `## Resolved`, so the
# exemption window was 2 lines wide and every closure record below it was reported
# as a duplicate declaration. That is how this guard came to fail on NINE ids, all
# of them legitimate: five closure records (TD-001/002/003/006/011) and four
# follow-up rows (TD-184/186/187/195). A permanently-red guard is a disabled one.
#
# The register nests resolved entries at the SAME heading level as the section
# that contains them, so containment is not encoded in the markup. The structural
# rule that does work: the section ends at the next level-2 heading that is not
# itself a TD entry (in practice, the next `## Session …`).
#
# `[0-9][0-9][0-9]` rather than `[0-9]{3}` — BSD awk does not enable ERE intervals
# by default, and a silently-non-matching pattern here would restore the exact
# 2-line window this comment exists to explain.
resolved_end="$(
  awk -v s="$resolved_start" '
    NR > s && /^## / && $0 !~ /^## (~~)?TD-[0-9][0-9][0-9]/ { print NR; exit }
  ' "$REGISTER"
)"
resolved_end="${resolved_end:-$(wc -l < "$REGISTER")}"

# Follow-up markers: recording progress in place is deliberate and good. What is
# banned is re-using a number for a DIFFERENT item. Extend deliberately — every
# addition widens what the guard stops seeing.
# `→` is the register's status-transition marker: a NEW declaration reads
# `### TD-037 — <description>` (em dash), a follow-up reads `### TD-166 → PARTIAL`.
# Matching the ARROW rather than the word "PARTIAL" is deliberately the narrower
# widening — "PARTIAL" could legitimately appear in a genuine new item's title,
# whereas the arrow only ever marks a transition on an EXISTING id.
#
# Added 2026-08-04 after implementing the collision-index exemption, which made
# the guard live again and immediately produced three false positives (TD-090,
# TD-132, TD-166) — all follow-ups. Per this guard's own design note: a guard that
# fires on correct work gets ignored.
# `[Cc]orrection` because the list was case-sensitive and the register writes
# "reference correction" in lower case.
#
# ⚠️ The arrow is matched POSITIONALLY — `### TD-nnn →`, immediately after the id —
# never as a bare `→`. A bare arrow was tried first and silently swallowed a REAL
# declaration whose TITLE contains one:
#     ### TD-080 — Swap `llm_json` → `jsonrepair`: drop transitive `clap` ...
# That turned a false-positive fix into a false NEGATIVE, which is strictly worse:
# the guard would have gone green while a genuine collision went unreported. Same
# defect class as matching signals anywhere in an input instead of within the same
# sub-unit. Scope every marker that could plausibly appear in prose.
FOLLOWUP_MARKERS='RESOLVED|[Cc]orrection|CORRECTION|update|UPDATE|A/B RESULT|Stage [A-Z]|CLOSED|✅|❌|NOT VIABLE|erratum|ERRATUM|### TD-[0-9]{3} →'

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

# ── The COLLISION INDEX exemption ────────────────────────────────────────────
#
# This guard's own failure message has always offered two remediations: renumber
# the newer item, OR — when renumbering would break live references — "record both
# in a collision index and never re-use the number." The register wrote that index.
# The guard could not READ it, so it stayed red forever.
#
# That is a guard whose policy is incomplete, and a permanently-red guard is a
# DISABLED guard: it trains everyone to ignore the signal, so a genuinely NEW
# collision would land in the noise. Implementing the exemption is what makes the
# guard live again — it goes green on the four known, documented collisions while
# still failing on a fifth.
#
# Checked in BOTH directions, so the index cannot rot:
#   forward  — a duplicate id is tolerated ONLY if the index declares it
#   backward — an index entry with no corresponding duplicate is itself a failure
# Without the backward check the index would silently accumulate stale entries and
# quietly widen what the guard stops seeing.
index_ids="$(
  awk '/^## .*ID COLLISION INDEX/{inidx=1; next}
       inidx && /^## /{exit}
       inidx' "$REGISTER" \
    | grep -oE '^\| \*\*TD-[0-9]{3}\*\*' \
    | grep -oE 'TD-[0-9]{3}' \
    | sort -u \
    || true
)"

unindexed_dupes=""
if [[ -n "$dupes" ]]; then
  while read -r id; do
    [[ -z "$id" ]] && continue
    if ! grep -qx "$id" <<< "$index_ids"; then
      unindexed_dupes+="${id}"$'\n'
    fi
  done <<< "$dupes"
fi

if [[ -n "${unindexed_dupes//[$'\n' ]/}" ]]; then
  fail=1
  echo "FAIL [check 1] — TD id declared twice for different items, and NOT recorded"
  echo "                 in the ID COLLISION INDEX; refs are ambiguous:"
  echo
  while read -r id; do
    [[ -z "$id" ]] && continue
    echo "  $id"
    echo "$declarations" | grep -E "^[0-9]+:### ${id}" | sed 's/^/      line /' | cut -c1-120
    echo
  done <<< "$unindexed_dupes"
fi

# ── CHECK 1b — stale index entries ───────────────────────────────────────────
# An id listed in the index that is no longer actually duplicated means the index
# is describing a condition that no longer exists. Left unchecked, the index grows
# into a blanket exemption nobody audits.
stale_index=""
if [[ -n "$index_ids" ]]; then
  while read -r id; do
    [[ -z "$id" ]] && continue
    if ! grep -qx "$id" <<< "$dupes"; then
      stale_index+="${id}"$'\n'
    fi
  done <<< "$index_ids"
fi

if [[ -n "${stale_index//[$'\n' ]/}" ]]; then
  fail=1
  echo "FAIL [check 1b] — ID COLLISION INDEX lists ids that are no longer duplicated."
  echo "                  Remove them; a stale exemption widens what the guard cannot see:"
  echo
  echo "$stale_index" | sed '/^$/d' | sed 's/^/      /'
  echo
fi

# ── CHECK 2 — OPEN/PARTIAL entries filed under `## Resolved` ─────────────────
misfiled="$(
  awk -v s="$resolved_start" -v e="$resolved_end" '
    NR > s && NR < e {
      # `#{2,3}` — resolved entries appear at BOTH heading levels. This matched
      # `###` only, so every `## TD-nnn` entry was invisible to check 2. Combined
      # with the 2-line window above, check 2 could not fire AT ALL: it was
      # reporting clean while `## TD-216` sat one line inside `## Resolved`
      # marked OPEN — the precise defect it was written to catch.
      if ($0 ~ /^#[#]#? (~~)?TD-[0-9][0-9][0-9]/) { hdr = $0; hdrline = NR; reported = 0 }
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
  indexed_count="$(echo "$index_ids" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo "PASS: ${total} TD declarations outside '## Resolved'."
  echo "      Every duplicated id is recorded in the ID COLLISION INDEX (${indexed_count})."
  echo "      No stale index entries; no OPEN/PARTIAL entries misfiled under '## Resolved'."
  if [[ "$indexed_count" -gt 0 ]]; then
    echo
    echo "NOTE: ${indexed_count} id(s) are DOCUMENTED collisions, not resolved ones."
    echo "      Cite them by item, never by bare id. No new TD may reuse these numbers."
  fi
  exit 0
fi

echo "Fix for check 1: give the NEWER item a fresh id; if renumbering would break"
echo "existing cross-references, record both in a collision index and never re-use"
echo "the number. Fix for check 2: move the entry back out of '## Resolved', or"
echo "correct its status if it is genuinely done."
exit 1

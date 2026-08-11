#!/usr/bin/env python3
"""Detector: a RESERVED_PREDICATE_* const that never reached RESERVED_PREDICATES.

WHY
---
TD-197 (fixed 2026-08-11): `potential_alias` is INTERNAL disambiguation
bookkeeping — a meta-edge predicate, not a domain fact. It was served to
consumers as if it were real knowledge in 142 of 199 benchmark questions on
the shipped-default `structured` path. Root cause: the spec'd
`is_reserved_predicate()` filter was never written, so nothing excluded it on
any read path.

The fix (`crates/kremory/src/core/disambiguation/mod.rs`) added
`is_reserved_predicate()`, backed by a `RESERVED_PREDICATES` slice. Its own
doc comment admits the residual hazard in so many words:

    "Rust has no reflection, so this list cannot be derived from the
    RESERVED_PREDICATE_* consts above automatically — when you add a new
    RESERVED_PREDICATE_* const, add it here too."

That is a human being asked to remember a two-place update. A module test
(`reserved_predicates_slice_contains_every_reserved_predicate_const`) already
mirrors the pairing — but the mirror is ALSO hand-typed
(`let all_reserved_predicate_consts = [RESERVED_PREDICATE_POTENTIAL_ALIAS];`).
Forget to update that literal array too, and the "safety net" passes anyway
while the real filter silently under-covers — TD-197's exact shape, one layer
deeper: a spec/doc said "the sync is enforced," and nothing outside a human's
memory actually enforced it.

WHAT THIS IS AND IS NOT
------------------------
Unlike `audit-spec-symbols.py` (a CANDIDATE GENERATOR over spec pseudo-code,
where a hit is a question for a human), this is a hard, mechanical ASSERTION.
`RESERVED_PREDICATE_*` is a formal declaration-site naming convention with
fixed Rust syntax — not semantic intent — so pattern-matching it is the
legitimate case per the project's own discriminator (a pattern matching a
FORMAL specification, not a semantic one). A const declared under that name
and absent from `RESERVED_PREDICATES` (or vice versa) is unambiguously a
defect, not a judgment call. So, unlike shape 1: **exit 1 on a real finding.**

This script does NOT need a human to decide whether a hit "counts" — that is
exactly the property that makes it safe to gate on.

RESIDUAL LIMIT (real, not fixable by this script): the derivation trusts the
`RESERVED_PREDICATE_` naming convention itself. A reserved value declared
under a different name (bypassing the convention) is invisible to this
detector, same as it is invisible to `is_reserved_predicate`'s own author-
maintained slice today. This is a naming-DISCIPLINE limit, not a sync-drift
bug — see `.ai-docs/tech-debt/tech-debt-register.md` TD-197 entry.

Reads the tree via a repomix pack rather than grepping directly: grep/rg/git
grep skip NUL-byte files SILENTLY with a success exit code, so a bare sweep
can report "no matches" when the truth is "did not look" (CLAUDE.md Rule 38).

Usage:
    npx repomix --quiet --include "crates/**/*.rs" --output /tmp/rpx-all.xml
    python3 scripts/audit-reserved-predicates.py /tmp/rpx-all.xml

Exit codes (matches the scripts/check-*.sh gate family, NOT audit-spec-
symbols.py's always-0 candidate-generator convention — this is an assert):
    0 = PASS — every RESERVED_PREDICATE_* const is in RESERVED_PREDICATES,
        and RESERVED_PREDICATES contains nothing else.
    1 = FAIL — sync mismatch found (a real TD-197-class defect), OR the pack
        didn't load / the expected symbols weren't found at all (refuse to
        report a vacuous pass).
    2 = usage error.
"""

from __future__ import annotations

import pathlib
import re
import sys

# A const declared under the reserved-predicate naming convention.
#   pub const RESERVED_PREDICATE_POTENTIAL_ALIAS: &str = "potential_alias";
DECL = re.compile(
    r'(?:pub(?:\([^)]*\))?\s+)?const\s+(RESERVED_PREDICATE_[A-Z0-9_]+)\s*:\s*&str\s*=\s*"([^"]*)"\s*;'
)

# The slice `is_reserved_predicate` actually consults.
#   pub const RESERVED_PREDICATES: &[&str] = &[RESERVED_PREDICATE_POTENTIAL_ALIAS];
ARRAY = re.compile(
    r"(?:pub(?:\([^)]*\))?\s+)?const\s+RESERVED_PREDICATES\s*:\s*&\[&str\]\s*=\s*&\[([^\]]*)\]\s*;"
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pack_path = pathlib.Path(sys.argv[1])
    if not pack_path.is_file():
        print(f"REFUSING: pack file not found at {pack_path}", file=sys.stderr)
        return 1
    pack = pack_path.read_text(errors="replace")

    # Non-vacuity: a pack that didn't load would make "no findings" look like
    # a clean pass instead of "did not look" (CLAUDE.md observability-first-
    # class + instrument-real-data-flow rules).
    if len(pack) < 20_000:
        print(
            f"REFUSING: pack is only {len(pack)} bytes — it likely did not load "
            f"the crate source. Re-run repomix.",
            file=sys.stderr,
        )
        return 1

    declared: dict[str, str] = {}
    for name, value in DECL.findall(pack):
        # A const CAN legitimately be declared once per module in normal Rust,
        # but re-exports (`pub use`) can make the same literal text appear
        # twice in one pack. Keep the first value seen; flag true conflicts.
        if name in declared and declared[name] != value:
            print(
                f"REFUSING: {name} declared twice with DIFFERENT values "
                f"({declared[name]!r} vs {value!r}) — ambiguous, fix source first.",
                file=sys.stderr,
            )
            return 1
        declared[name] = value

    if not declared:
        print(
            "REFUSING: found ZERO `RESERVED_PREDICATE_*` const declarations in the "
            "pack. Either the naming convention was abandoned (update this script) "
            "or the pack missed the source file — a silent pass here would be "
            "exactly the unvalidated-instrument trap this repo keeps hitting.",
            file=sys.stderr,
        )
        return 1

    array_match = ARRAY.search(pack)
    if array_match is None:
        print(
            "REFUSING: found RESERVED_PREDICATE_* const(s) but no `RESERVED_PREDICATES` "
            "slice declaration in the pack. If the slice was renamed, update this "
            "script's ARRAY pattern to match — do not treat this as a pass.",
            file=sys.stderr,
        )
        return 1

    array_members = [m.strip() for m in array_match.group(1).split(",") if m.strip()]

    declared_names = set(declared)
    array_names = set(array_members)

    missing_from_array = sorted(declared_names - array_names)
    dangling_in_array = sorted(array_names - declared_names)

    print(f"declared RESERVED_PREDICATE_* consts: {len(declared_names)}")
    for name in sorted(declared_names):
        print(f"  {name} = {declared[name]!r}")
    print(f"RESERVED_PREDICATES members: {len(array_members)}")
    for name in array_members:
        print(f"  {name}")
    print(f"\nderived reserved-value set (what is_reserved_predicate() actually "
          f"excludes today): {sorted(declared[n] for n in array_names if n in declared)}")

    fail = False
    if missing_from_array:
        fail = True
        print(
            f"\nFAIL: {len(missing_from_array)} const(s) declared under the "
            f"RESERVED_PREDICATE_* convention but MISSING from RESERVED_PREDICATES "
            f"— this is the TD-197 recurrence: a reserved value that "
            f"is_reserved_predicate() will silently NOT filter, so it reaches "
            f"recall() as if it were a domain fact:",
            file=sys.stderr,
        )
        for name in missing_from_array:
            print(f"      {name} = {declared[name]!r}", file=sys.stderr)

    if dangling_in_array:
        fail = True
        print(
            f"\nFAIL: {len(dangling_in_array)} identifier(s) in RESERVED_PREDICATES "
            f"do not correspond to any declared RESERVED_PREDICATE_* const — likely "
            f"a rename left the array pointing at a dead/renamed symbol (won't "
            f"compile) or the naming convention was bypassed:",
            file=sys.stderr,
        )
        for name in dangling_in_array:
            print(f"      {name}", file=sys.stderr)

    if fail:
        print(
            "\nFAIL: RESERVED_PREDICATE_* consts and RESERVED_PREDICATES are out of "
            "sync. Fix crates/kremory/src/core/disambiguation/mod.rs before merging.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: every RESERVED_PREDICATE_* const is exactly mirrored in "
          "RESERVED_PREDICATES — no TD-197-class drift detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

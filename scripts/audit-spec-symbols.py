#!/usr/bin/env python3
"""Detector: symbols a SPEC says exist, that do NOT exist in the tree.

WHY
---
Two of the defects found on 2026-08-10 have the identical shape — a spec
described a function, the function was never written, and nothing noticed:

  * TD-197  `is_reserved_predicate()` is specified in
            `specs/unified-extraction-implementation-spec-2026-06-03.md` §3 as the
            boundary keeping `potential_alias` meta-edges internal. It does not
            exist, so no read path filters them, and 142 of 199 benchmark
            questions served internal bookkeeping to the answerer as knowledge.
  * TD-155  DoD item (b) "carry `valid_at` on the wire" — specified 2026-07-28,
            never built, independently rediscovered as a "new" finding weeks later.

Both are SILENTLY INERT: nothing failed, nothing logged, the code did something
reasonable-looking and told nobody. That class does not surface through testing,
because there is no code to test. It surfaces through reconciliation.

WHAT THIS IS AND IS NOT
-----------------------
This is a CANDIDATE GENERATOR, not a verdict. Spec code blocks contain
illustrative pseudo-code, competitor code, and renamed-since symbols, so a hit
is a QUESTION ("was this ever built, and should it have been?"), not a defect.
Verify each before acting. Reporting it as anything stronger would be the
unvalidated-instrument trap this repo keeps hitting.

Reads the tree via a repomix pack rather than grepping directly: grep/rg/git grep
skip NUL-byte files SILENTLY with a success exit code, so a bare sweep can report
"no matches" when the truth is "did not look" (CLAUDE.md Rule 38).

Usage:
    npx repomix --quiet --include "crates/**/*.rs" --output /tmp/rpx-all.xml
    python3 scripts/audit-spec-symbols.py /tmp/rpx-all.xml [docs-root]
"""

from __future__ import annotations

import pathlib
import re
import sys
from collections import defaultdict

# Declarations inside a ```rust block that assert a symbol exists.
DECL = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
RUST_BLOCK = re.compile(r"```rust\n(.*?)```", re.S)

# Symbols too generic to be evidence of anything.
NOISE = {
    "main", "new", "default", "from", "into", "test", "run", "build", "get", "set",
    "handle", "next", "len", "is_empty", "clone", "fmt", "drop", "call", "poll",
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pack = pathlib.Path(sys.argv[1]).read_text(errors="replace")
    docs_root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".ai-docs")

    # Non-vacuity: a pack that didn't load would make everything look "missing".
    if len(pack) < 100_000:
        print(f"REFUSING: pack is only {len(pack)} bytes — it did not load. "
              f"Re-run repomix.", file=sys.stderr)
        return 1

    specs = sorted(docs_root.rglob("*.md"))
    if not specs:
        print(f"REFUSING: no markdown found under {docs_root}", file=sys.stderr)
        return 1

    missing: dict[str, list[str]] = defaultdict(list)
    seen = 0
    for spec in specs:
        try:
            text = spec.read_text(errors="replace")
        except OSError:
            continue
        for block in RUST_BLOCK.findall(text):
            for sym in DECL.findall(block):
                if sym in NOISE or len(sym) < 6:
                    continue
                seen += 1
                # A symbol "exists" if its name appears anywhere in the packed
                # source. Deliberately permissive — we want few false alarms.
                if sym not in pack:
                    rel = str(spec.relative_to(docs_root))
                    if rel not in missing[sym]:
                        missing[sym].append(rel)

    print(f"scanned {len(specs)} docs, {seen} symbol declarations in rust blocks")
    print(f"symbols NOT found anywhere in the packed source: {len(missing)}\n")
    for sym, where in sorted(missing.items(), key=lambda kv: -len(kv[1])):
        print(f"  {sym}")
        for w in where[:3]:
            print(f"      {w}")
    print("\nEach hit is a QUESTION, not a defect: spec blocks hold pseudo-code, "
          "competitor code, and since-renamed symbols. Verify before acting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
# Prune Rust build-cache bloat WITHOUT forcing a full rebuild.
#
# Why this exists (2026-08-04): `target/` reached **104 GB** on a 926 GB volume that
# was 91% full, and the bloat had a second, non-obvious cost — macOS Gatekeeper
# re-validates every freshly-linked binary on first exec, so thousands of stale
# duplicates turned each test run into a multi-minute stall (see
# `run-with-stall-guard.sh` and V1-CANONICAL §0b-bis).
#
# Measured composition of that 104 GB:
#   target/debug/incremental   42 GB   pure rebuild-speed cache, regenerates
#   target/debug/deps          61 GB   of which ~39 GB was stale duplicate hashes
#   target/debug/build          2 GB
#   target/release            1.5 GB
#
# WHY NOT `cargo clean`: it removes everything, including the ~15 GB of currently-valid
# artifacts, so the next build is a full cold rebuild (~6 min for this workspace) AND
# every binary must be re-validated by Gatekeeper. This script keeps what is live.
#
#   usage: prune-build-cache.sh [--apply] [--days N]
#          (default is a DRY RUN — it prints what it would delete and exits)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
APPLY=0
DAYS=2
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --days)  DAYS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[ -d target ] || { echo "no target/ here — nothing to do"; exit 0; }

echo "════════════════════════════════════════════════════════"
echo "  Rust build-cache prune  ($([ $APPLY -eq 1 ] && echo APPLY || echo 'DRY RUN'))"
echo "  keeping duplicates newer than ${DAYS}d"
echo "════════════════════════════════════════════════════════"
echo "before: $(du -sh target 2>/dev/null | cut -f1)"
echo ""

# ── 1. incremental cache ────────────────────────────────────────────────────
# Unambiguously safe: cargo regenerates it. The only cost is that the NEXT build
# is non-incremental. On a workspace that mostly runs whole test suites, that is a
# poor trade for tens of GB.
INC=target/debug/incremental
if [ -d "$INC" ]; then
  echo "1. incremental cache: $(du -sh $INC 2>/dev/null | cut -f1)"
  [ $APPLY -eq 1 ] && rm -rf "$INC" && echo "   removed"
else
  echo "1. incremental cache: absent"
fi
echo ""

# ── 2. stale duplicate artifacts in deps/ ───────────────────────────────────
# For each (stem, extension) group, keep the NEWEST hash and any duplicate touched
# within --days. Age-filtering matters: several hashes of one stem are often all
# VALID, corresponding to different feature sets (e.g. `--features llm-smoke` vs
# not). Deleting purely by "not newest" would evict a live feature set's artifacts
# and force a rebuild every time you switch. Age is the safer discriminator.
echo "2. stale duplicate artifacts in target/debug/deps:"
APPLY=$APPLY DAYS=$DAYS python3 - <<'PY'
import os, re, collections, time, sys
apply_ = os.environ.get('APPLY') == '1'
days   = float(os.environ.get('DAYS', '2'))
d = 'target/debug/deps'
pat = re.compile(r'^(.+)-([0-9a-f]{16})(\.[a-z0-9]+)?$')
groups = collections.defaultdict(list)
now = time.time()
if not os.path.isdir(d):
    print("   deps/ absent"); sys.exit(0)
for f in os.listdir(d):
    m = pat.match(f)
    if not m:
        continue
    p = os.path.join(d, f)
    try:
        st = os.stat(p)
    except OSError:
        continue
    groups[(m.group(1), m.group(3) or '')].append((st.st_mtime, st.st_size, f))

freed = files = 0
for _k, v in groups.items():
    if len(v) < 2:
        continue
    v.sort(reverse=True)          # newest first — always kept
    for mt, size, f in v[1:]:
        if now - mt <= days * 86400:
            continue              # recent duplicate: probably a live feature set
        freed += size; files += 1
        if apply_:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
print(f"   {files} files, {freed/1e9:.1f} GB{' — removed' if apply_ else ' (dry run)'}")
PY

echo ""
echo "after:  $(du -sh target 2>/dev/null | cut -f1)"
if [ $APPLY -eq 0 ]; then
  echo ""
  echo "DRY RUN — nothing deleted. Re-run with --apply to actually prune."
fi

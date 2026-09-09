#!/usr/bin/env bash
# check-examples.sh — run every offline example and fail if any breaks.
#
# WHY THIS EXISTS
# ---------------
# The examples ship INSIDE the published crate (`Cargo.toml` names each one in
# `include`), so a broken example is something users download. Nothing ran them.
#
# That is not hypothetical here. On 2026-09-08 `e2e-consumer/` was found not to
# COMPILE against 0.8.0 — the release made `DreamRequest` require `.execute()`,
# shipped the migration note, and never migrated the repo's own consumer. It had
# rotted for exactly one reason: no gate executed it.
#
# These examples are the same shape of artifact and were heading the same way.
#
# WHY IT RUNS THEM RATHER THAN JUST COMPILING
# -------------------------------------------
# Every offline example ASSERTS its own behaviour — that a tenant cannot see
# another's data, that erasure is surgical, that an undo restores. Compiling
# proves the API still exists; running proves the behaviour still holds. The
# second is the one that caught real bugs.
#
# They are deterministic and take about a second each, so there is no reason to
# settle for the weaker check.
#
# THE OLLAMA EXAMPLE IS DELIBERATELY EXCLUDED
# -------------------------------------------
# TWO are excluded, for different reasons:
#   agent_memory_with_ollama — needs a running daemon and a pulled model, and takes
# ~65s. A guard that fails on a laptop without Ollama would fire on ordinary work
# and get switched off, taking the rest with it.
#   hosted_providers — calls a PAID API. A guard that spends money every time it
#   runs is a guard people disable, and it would charge CI (if there were CI) on
#   every commit.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# Every example EXCEPT the Ollama one and the internal spikes/probes.
OFFLINE=(
  offline_remember_recall
  remembers_across_sessions
  correcting_the_record
  searching_documents
  a_long_document
  bulk_import
  ingest_without_blocking
  multi_tenant_isolation
  gdpr_erasure_by_source
  append_only_namespace
  deleting_and_restoring
  undoing_a_bad_change
  changing_embedding_model
  two_handles_one_database
  undoing_a_correction
  domain_entity_types
  dream_on_a_schedule
)

# Guard against the list drifting from what actually ships. A new example added
# to Cargo.toml but not here would silently never be checked — the exact failure
# this script exists to prevent, one level up.
declare -a SHIPPED
while IFS= read -r line; do SHIPPED+=("$line"); done < <(
  grep -oE '"examples/[a-z0-9_]+\.rs"' crates/kremory/Cargo.toml \
    | sed -E 's|"examples/([a-z0-9_]+)\.rs"|\1|' | sort
)
EXPECTED=$(printf '%s\n' "${OFFLINE[@]}" agent_memory_with_ollama hosted_providers | sort)
ACTUAL=$(printf '%s\n' "${SHIPPED[@]}")
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
  echo "FAIL: this script's list has drifted from Cargo.toml's published examples." >&2
  diff <(echo "$EXPECTED") <(echo "$ACTUAL") | sed 's/^/  /' >&2
  echo "  Add the new example to OFFLINE above (or to the exclusion) and re-run." >&2
  exit 1
fi

fails=0
for ex in "${OFFLINE[@]}"; do
  if cargo run --quiet --example "$ex" > "/tmp/example-$ex.log" 2>&1; then
    printf '  ok    %s\n' "$ex"
  else
    printf '  FAIL  %s  (see /tmp/example-%s.log)\n' "$ex" "$ex"
    tail -5 "/tmp/example-$ex.log" | sed 's/^/          /'
    fails=$((fails + 1))
  fi
done

echo
if (( fails )); then
  echo "FAIL: $fails of ${#OFFLINE[@]} examples broke. These ship in the published crate." >&2
  exit 1
fi
echo "All ${#OFFLINE[@]} offline examples ran and their assertions held."
echo "Two are excluded and must be run by hand:"
echo "  agent_memory_with_ollama — needs a running Ollama"
echo "  hosted_providers         — calls a PAID API"

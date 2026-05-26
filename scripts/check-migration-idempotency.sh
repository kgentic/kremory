#!/usr/bin/env bash
# check-migration-idempotency.sh — Fail if production DDL has bare
# CREATE TABLE/INDEX/VIRTUAL TABLE without IF NOT EXISTS.
#
# Scope: production schema sources only — schema.rs and any future
# dedicated migration files. Test fixtures in #[cfg(test)] blocks
# are intentionally excluded (they use bare CREATE TABLE for conciseness).
#
# Run as: ./scripts/check-migration-idempotency.sh
# CI: wired into release-please workflow.
#
# Why: TemporalGraph::open_in_memory must be callable twice without error.
# Bare CREATE TABLE breaks this on second open of the same schema.

set -euo pipefail

# Production DDL lives in schema.rs (and will live in any future *.sql files).
# migrations.rs only runs in test contexts for the MigrationRunner framework.
PRODUCTION_SOURCES=(
    "crates/kremory/src/core/schema.rs"
)

fail=0

check_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        return
    fi

    local line_no=0
    while IFS= read -r line; do
        line_no=$((line_no + 1))
        # Strip // line comments
        stripped="${line%%//*}"
        upper=$(echo "$stripped" | tr '[:lower:]' '[:upper:]')

        if echo "$upper" | grep -q "CREATE TABLE" && ! echo "$upper" | grep -q "CREATE TABLE IF NOT EXISTS"; then
            echo "ERROR $file:$line_no: bare CREATE TABLE: $line" >&2
            fail=1
        fi
        if echo "$upper" | grep -q "CREATE INDEX" && ! echo "$upper" | grep -q "CREATE INDEX IF NOT EXISTS"; then
            echo "ERROR $file:$line_no: bare CREATE INDEX: $line" >&2
            fail=1
        fi
        if echo "$upper" | grep -q "CREATE VIRTUAL TABLE" && ! echo "$upper" | grep -q "CREATE VIRTUAL TABLE IF NOT EXISTS"; then
            echo "ERROR $file:$line_no: bare CREATE VIRTUAL TABLE: $line" >&2
            fail=1
        fi
    done < "$file"
}

for src in "${PRODUCTION_SOURCES[@]}"; do
    check_file "$src"
done

if [[ "$fail" -eq 1 ]]; then
    echo ""
    echo "FAIL: idempotency invariant violated in production DDL." >&2
    echo "      Use CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS." >&2
    exit 1
fi

echo "OK: all production CREATE TABLE/INDEX/VIRTUAL TABLE use IF NOT EXISTS."

# Comment-hygiene audit — internal-process leaks in source comments

**Scope note:** this audit is BROADER than the `public-docs-and-api-surface-audit` spec this
file lives under. That spec covers the public API/docs surface; this audit covers **every
comment in `crates/*/src/`, all six crates** (`kremory`, `kremory-mcp`, `kremory-napi`,
`kremory-eval`, `kremory-admin`, `kremory-doc-examples`), regardless of whether the code is
public API, internal, or test-only. It is filed here only because this is where the session's
other comment-hygiene finding (the confirmed `config.rs`/`builder.rs` F17 example that
triggered this pass) already lives, and because no better-fitting spec exists yet.

**Status: READ-ONLY. No code changed. This is a findings report to support a go/no-go
decision on a fix pass, not the fix pass itself.**

**Date:** 2026-09-07. **Method:** Repomix pack of `crates/**/src/**` (182 files, matches
`git ls-files` count exactly — full coverage confirmed, no silent binary-skip), then
comment-line-filtered `rg` sweeps for 14 patterns, then manual read of surrounding context for
a representative sample per pattern/file. The `grep_repomix_output` MCP tool was not present
in this environment's tool registry; `rg` was run directly against the git-tracked source tree
instead, with coverage cross-validated against the repomix pack's file count (182 == 182) to
close the "silent skip" gap that raw grep normally carries.

---

## 1. Executive summary

| Metric | Value |
|---|---|
| Total comment lines matching ≥1 violation-shaped pattern | **6,496** |
| Files touched | **152 of 182** (83.5% of all source files in `crates/*/src`) |
| Files with ZERO hits | 30 |
| Pattern with the highest true-positive confidence | `Vera` / `Quinn` / `Finding N` / `F\d\d` / `steal-matrix` — **426 lines, 69 files, ~100% genuine violations on sampling** |
| Pattern with the lowest true-positive confidence (mostly legitimate domain vocabulary) | `audit` (288 raw hits, majority are the literal system feature "audit row"/"audit trail", not process leaks) and `Phase N` (928 raw hits, majority describe real pipeline stages like `Phase 1 ingest`, not audit-session narrative) |
| Newly-discovered pattern not in the original 14 (found during sampling) | `Rule \d+` — citations of the OPERATOR's own global `~/.claude/CLAUDE.md` numbered rules (e.g. "Rule 19", "Rule 16") baked into source comments. **30 hits, 12 files.** Arguably worse than a TD-/ADR- citation: the referent isn't even IN this repository — it's in the maintainer's home directory config, unreachable by anyone else who ever reads this crate. |

**Headline conclusion:** this is not a handful of stray comments. It is a systemic,
codebase-wide authoring habit — TD-numbers, ADR-numbers, reviewer-persona names ("Vera",
"Quinn"), finding IDs, phase-of-a-spec numbers, and session dates are woven into a large
fraction of the substantive doc comments across the whole `kremory` workspace, including
`///` rustdoc that ships to docs.rs. The already-confirmed example that triggered this task
(`config.rs:971`, `builder.rs:384` — "public-docs-and-api-surface-audit Phase 2 (F17,
2026-09-07)") is not an outlier; it is the modal case.

The good news, and it matters for scoping the fix: in the large majority of sampled
instances, **the technical content of the comment is genuinely good** — real invariants,
real security rationale, real "why this shape and not that one" reasoning. The defect is
almost always confined to a **parenthetical or heading tag** ("(Vera F17)", "(TD-197 review
Finding 2)", "public-docs-and-api-surface-audit Phase 2 (F17, 2026-09-07):") that can be
deleted without touching the surrounding prose. This is a **mechanical strip**, not a rewrite
of the underlying documentation, in the overwhelming majority of cases sampled.

---

## 2. Methodology and what "violation" means here

Per the brief: a hit is a **violation** when the comment leans on an internal, ephemeral,
unresolvable-by-the-reader reference (a TD/ADR/Finding/Rule number, a reviewer persona name,
a session date, a spec-phase number, an audit-slug) **as justification for why the code
exists or behaves as it does**, instead of just stating the technical invariant. A hit is
**NOT a violation** when:

- The pattern match is a legitimate domain term in THIS codebase's own vocabulary that
  happens to share a word with the search pattern — chiefly `audit` (the system has a
  first-class "audit row"/"audit trail" concept: `contradiction.rs`'s conflict-resolution
  audit, `identity_verdict.rs`'s "for audit/debug only, NEVER parsed for control flow",
  `dream/consistency_check/audit.rs`) and `Phase N` (the two-phase ingest architecture is a
  real, permanent, documented design — `ingest_phase1_ner`, `Phase 1 ingest: store episode +
  run NER` — not a reference to a review session).
- The digits are a date-shaped STRING used as a format example, not a session timestamp
  (`extraction/parsers.rs:31`: `` `2023-05-07T00:00:00Z` `` illustrating an RFC 3339 literal).
- The citation is incidental supporting evidence for a comment whose PRIMARY content is
  already a stand-alone technical explanation (the `forget.rs` "Shared-entity preservation"
  example below is the clearest case: delete "(Vera F17)" and the paragraph is unchanged
  and complete).

## 3. Pattern-by-pattern breakdown

| Pattern searched | Raw comment-line hits | True-positive character (from sampling) |
|---|---|---|
| `TD-\d+` | 2,354 | High. Nearly every sampled hit is a genuine internal-register citation used as justification. Occasionally the surrounding paragraph is real design rationale where the TD-number is incidental (keep the rationale, strip the number). |
| `ADR-\d+` | 2,272 | High, same shape as TD-. A few are legitimate "this mirrors ADR-029b's PK pattern" architecture cross-references that could arguably stay if reframed as "this project's own accepted decision on X" without the number — still recommend stripping the number since ADRs are internal and can be renumbered/superseded. |
| `Phase \d` | 928 | **Bimodal — roughly half legitimate.** "Phase 1 ingest: store episode + run NER" (a real, permanent pipeline-stage name) is NOT a violation. "public-docs-and-api-surface-audit Phase 2 (finding F17, 2026-09-07)" / "ADR-082 Phase 3" / "recall-v2 Phase 2b (Decision 1/5)" (a phase-number *within an internal spec document*) IS a violation. See §5 for the refined regex that separates these. |
| Session dates `20\d\d-\d\d-\d\d` | 832 | High, once the RFC-3339-format-example false positive class is excluded (rare — 1 confirmed instance in sampling). Most dates are literally "REWRITTEN 2026-08-03", "CORRECTED 2026-07-28", ".ai-docs/specs/\*-2026-07-21.md" — session/doc timestamps baked into comments. |
| `Vera` | 148 | ~100%. Reviewer-persona name; always cited as "why this code is shaped this way," always internal. |
| `Quinn` | 230 | ~100%, same shape as `Vera`. |
| `Finding [A-Z]?\d+` | 22 | 100% on full read (all 22 read, see §4 appendix). |
| `F\d\d` (bare) | 42 | High — mostly overlaps with the `Vera`/`Quinn`/audit-spec set (`F17`, `F41`, `F44`, `F45`, `F46`). |
| `steal-matrix` | 12 | 100% on full read — all reference `.ai-docs/research/steal-matrix-rescore-2026-07-27.md` or the tech-debt register's "steal-matrix-rescore item N" rows. |
| `quality-review` | 0 | No hits as a literal string (the concept appears under other names — see `Vera`/`Quinn`/`review cycle` instead). |
| `review cycle` | 0 | No hits as a literal string in comments. |
| `audit` (case-insens.) | 288 | **Low — majority legitimate.** Dominant sense is the domain noun "audit row"/"audit trail"/"audit-grade" (a real bi-temporal-facts feature). Genuine violations are the ones citing `public-docs-and-api-surface-audit` (the spec slug) or `.ai-docs/research/v0.1.5-test-pyramid-audit/...` (a dated internal research doc path) — a small minority of the 288. |
| `this session` | 2 | Both real, both minor (`graph_integrity.rs:1441`, one physical line matched twice by two overlapping search terms). |
| `this fix` / `this pass` | 80 | **Low — majority legitimate.** "this pass" is overwhelmingly the domain term for one run of the dream-consolidation reconciliation loop (a permanent architectural concept: discover/aliases/reclassify/consistency_check/canonicalize), not a reference to "the fix I just made." "this fix" is a smaller, mixed bucket — some are narrative ("Prior to this fix, X was emitted"), some are fine. |
| **`Rule \d+` (discovered during sampling, not in original list)** | 30 | ~100% violation, and structurally worse than TD-/ADR- — see §1. |

**Deduplicated total (any pattern, one count per physical line):** 6,496 across 152 files.

---

## 4. High-confidence bucket — ready to fix as-is

The union of `Vera` / `Quinn` / `Finding N` / bare `F\d\d` / `steal-matrix` is **426 lines
across 69 files**, essentially 100% true-positive on sampling (every line manually spot-read
across ~15 files came back as a genuine violation). This is the safest subset to fix FIRST —
low ambiguity, mechanical strip in nearly every case. Full per-file counts, worst-first:

```
 40  crates/kremory/src/facade/recall.rs
 22  crates/kremory/src/facade/forget.rs
 22  crates/kremory/src/core/extraction/parsers.rs
 18  crates/kremory/src/core/migrations/defs_a.rs
 18  crates/kremory/src/core/ingest/pipeline/ingest_with.rs
 16  crates/kremory/src/core/dream/consolidation/cross_episode.rs
 14  crates/kremory/src/core/canonicalization/mod.rs
 12  crates/kremory/src/memory/engine_handle.rs
 12  crates/kremory/src/core/schema.rs
 12  crates/kremory-mcp/src/bin/kremory-http.rs
 10  crates/kremory/src/core/extraction/structured.rs
  8  crates/kremory/src/facade/mod.rs
  8  crates/kremory/src/core/identity_verdict.rs
  8  crates/kremory/src/core/extraction/injection_patterns.rs
  8  crates/kremory/src/core/dream/provenance/reversal.rs
  8  crates/kremory/src/core/dream/consolidation/archive.rs
  8  crates/kremory/src/core/dream/acronym_nickname_recall.rs
  8  crates/kremory/src/core/disambiguation/lexical.rs
  8  crates/kremory/src/core/background/deferred_pipeline.rs
  6  crates/kremory/src/memory/types.rs
  6  crates/kremory/src/memory/background_ingestor_handle.rs
  6  crates/kremory/src/facade/dream.rs
  6  crates/kremory/src/core/migrations/mod.rs
  6  crates/kremory/src/core/migrations/defs_h.rs
  6  crates/kremory/src/core/dream/type_registry_collapse.rs
  6  crates/kremory/src/core/dream/consolidation/supersession.rs
  6  crates/kremory/src/core/dream/consistency_check/verify.rs
  6  crates/kremory/src/core/contradiction.rs
  6  crates/kremory/src/core/config.rs
  6  crates/kremory-napi/src/lib.rs
  4  crates/kremory/src/facade/supersede.rs
  4  crates/kremory/src/facade/builder.rs
  4  crates/kremory/src/core/migrations/tests.rs
  4  crates/kremory/src/core/migrations/defs_j.rs
  4  crates/kremory/src/core/migrations/defs_b.rs
  4  crates/kremory/src/core/graph/queries.rs
  4  crates/kremory/src/core/extraction/programmatic.rs
  4  crates/kremory/src/core/error.rs
  4  crates/kremory/src/core/dream/discover_types.rs
  4  crates/kremory/src/core/background/ingestor.rs
  4  crates/kremory-napi/src/convert.rs
  2  crates/kremory/src/memory/mod.rs
  2  crates/kremory/src/memory/graph.rs
  2  crates/kremory/src/facade/update.rs
  2  crates/kremory/src/facade/remember.rs
  2  crates/kremory/src/core/search.rs
  2  crates/kremory/src/core/scoring/mod.rs
  2  crates/kremory/src/core/provider/tests.rs
  2  crates/kremory/src/core/migrations/defs_n.rs
  2  crates/kremory/src/core/migrations/defs_m.rs
  2  crates/kremory/src/core/migrations/defs_k.rs
  2  crates/kremory/src/core/migrations/defs_i.rs
  2  crates/kremory/src/core/ingest/tests.rs
  2  crates/kremory/src/core/ingest/pipeline/deferred.rs
  2  crates/kremory/src/core/graph/namespace.rs
  2  crates/kremory/src/core/graph/facts.rs
  2  crates/kremory/src/core/graph/entities.rs
  2  crates/kremory/src/core/extraction/mod.rs
  2  crates/kremory/src/core/dream/consolidation/mod.rs
  2  crates/kremory/src/core/dream/consistency_check/mod.rs
  2  crates/kremory/src/core/disambiguation/mod.rs
  2  crates/kremory/src/core/context.rs
  2  crates/kremory/src/core/background/verify_stage.rs
  2  crates/kremory/src/core/background/mod.rs
  2  crates/kremory-napi/src/bridge.rs
  2  crates/kremory-eval/src/bin/spike_c6_async_gate.rs
  2  crates/kremory-eval/src/bin/phase1_ner_bench.rs
  2  crates/kremory-eval/src/bin/llm_tokens_bench.rs
  2  crates/kremory-eval/src/bin/consistency_check_sweep.rs
```

**All 22 `Finding N` hits (100% read, full list, no sampling):**

```
crates/kremory/src/core/disambiguation/mod.rs:919
crates/kremory/src/core/dream/discover_types.rs:1324
crates/kremory/src/core/migrations/defs_m.rs:16
crates/kremory/src/memory/types.rs:775  (x2 occurrences, different lines nearby)
crates/kremory/src/core/extraction/structured.rs:224
crates/kremory/src/core/search.rs:6037
crates/kremory/src/core/migrations/defs_i.rs:5
crates/kremory/src/core/schema.rs:965
crates/kremory/src/core/ingest/pipeline/ingest_with.rs:1525
crates/kremory/src/memory/engine_handle.rs:708
crates/kremory/src/memory/engine_handle.rs:1312
```

**All `steal-matrix` hits (100% read, no sampling):** `kremory-mcp/src/bin/kremory-http.rs:1647`,
`kremory/src/core/config.rs:207`, `kremory/src/core/migrations/defs_n.rs:1`,
`kremory/src/core/schema.rs:996`, `kremory/src/core/schema.rs:1750`,
`kremory/src/core/migrations/mod.rs:450` — all cite `.ai-docs/research/steal-matrix-rescore-2026-07-27.md`
or `tech-debt-register.md`'s "steal-matrix-rescore item N" rows verbatim in rustdoc.

## 5. The newly-discovered `Rule \d+` pattern (12 files, 30 hits)

Not in the brief's original pattern list — found while reading surrounding context on a
`TD-197`/`Vera` hit in `memory/engine_handle.rs` ("the exact Rule 19 'counters that lie'
shape"). This cites the OPERATOR's personal, globally-scoped `~/.claude/CLAUDE.md`
engineering-rules file by number ("Rule 19" = observability-first-class, "Rule 16" =
web-app-ui-parity). Files affected:

```
crates/kremory-eval/src/bin/spike_c6_async_gate.rs
crates/kremory-mcp/src/handlers.rs
crates/kremory/src/core/background/mod.rs
crates/kremory/src/core/background/verify_stage.rs
crates/kremory/src/core/context.rs
crates/kremory/src/core/dream/discover_types.rs
crates/kremory/src/core/extraction/default_extractor.rs
crates/kremory/src/core/extraction/schemas.rs
crates/kremory/src/core/extraction/structured.rs
crates/kremory/src/core/ingest/pipeline/deferred.rs
crates/kremory/src/core/migrations/defs_j.rs
crates/kremory/src/memory/engine_handle.rs
```

This is arguably a MORE severe instance of the anti-pattern than a TD-/ADR- citation: a
TD-number at least resolves to a file inside this repository (`.ai-docs/tech-debt-register.md`)
that ships with the git history. "Rule 19" resolves to nothing inside the repo at all — it
points at a file in the maintainer's home directory that no other contributor, no CI system,
and nobody on crates.io/docs.rs has ever had access to. Recommend treating this as its own
priority-1 sub-bucket in the fix pass.

---

## 6. Worked examples (12), spanning all pattern types and both violation flavors

### 6.1 — Already-confirmed example (the one that triggered this task)

**`crates/kremory/src/core/config.rs:970-975`**
```rust
/// public-docs-and-api-surface-audit Phase 2 (F17): weight of the additive
/// graph-degree bonus (`SearchConfig::graph_degree_weight`). Default `0.05`
/// (the already-live TD-066 Change 2 boost, byte-identical) — unlike every
/// sibling axis on `SearchConfig`, this field previously had NO builder
/// method and no env override, so it was reachable for reading (via
/// `Memory::search_config()`) but never for writing.
```
**Classification:** VIOLATION (audit-slug + phase-number + finding-ID + TD-number, all as
justification for the field's existence).
**Proposed rewrite** (keep the invariant, drop every internal-process token):
```rust
/// Weight of the additive graph-degree bonus
/// (`SearchConfig::graph_degree_weight`). Default `0.05` — matches the
/// value already live in the scoring path. Unlike every sibling axis on
/// `SearchConfig`, this field has no builder method and no env override,
/// so it is reachable for reading (via `Memory::search_config()`) but not
/// for writing.
```

### 6.2 — Same violation, `builder.rs` (the sibling half of the confirmed example)

**`crates/kremory/src/facade/builder.rs:383-390`**
```rust
/// public-docs-and-api-surface-audit Phase 2 (finding F17, 2026-09-07):
/// added because this was the one weight axis on `SearchConfig` reachable
/// only for *reading* (via [`Memory::search_config`]), never for *writing*
/// — no builder method and no env override existed, unlike every sibling
/// axis (`content_stream_weight`, `proximity_weight`, `temporal_weight`).
/// The mirror image of TD-231, which found the same "documented as
/// tunable, actually unreachable" gap for `extraction_arm_budget_ms`.
```
**Classification:** VIOLATION — same flavor, plus a cross-reference to another internal
finding (TD-231) as supporting narrative.
**Proposed rewrite:**
```rust
/// This is the one weight axis on `SearchConfig` that was previously
/// reachable only for *reading* (via [`Memory::search_config`]), never
/// for *writing* — no builder method and no env override existed, unlike
/// every sibling axis (`content_stream_weight`, `proximity_weight`,
/// `temporal_weight`). Same shape as the read/write gap that was closed
/// for `extraction_arm_budget_ms`.
```
(Kept the cross-reference to the sibling gap since it IS useful context for a future
reader — reframed as a description of the pattern rather than a ticket number.)

### 6.3 — Security-critical guard, tag-only violation (recall.rs)

**`crates/kremory/src/facade/recall.rs:188-190`**
```rust
// Vera F6 path-injection guard. Bracket / quote / dollar / star / backslash
// / dot are all interpreted by SQLite's json_extract path grammar; rejecting
// here prevents consumer-controlled keys from escaping `'$.{key}'`.
```
**Classification:** VIOLATION, but a clean one — the technical content ("these characters are
interpreted by SQLite's `json_extract` path grammar; rejecting them prevents key escaping")
is complete and correct on its own. "Vera F6" adds nothing.
**Proposed rewrite:**
```rust
// Path-injection guard. Bracket / quote / dollar / star / backslash / dot
// are all interpreted by SQLite's json_extract path grammar; rejecting
// here prevents consumer-controlled keys from escaping `'$.{key}'`.
```
Same file, same fix pattern at line 1656: `// Vera F6 reject list + Quinn C3 defence-in-depth
(\`{\` and \`}\`).` → `// Path-metachar reject list, including \`{\`/\`}\` for
defence-in-depth against template-string escape in any future code path that wraps the
path in \`{...}\`.`

### 6.4 — Migration file, reviewer-name-as-heading (defs_a.rs)

**`crates/kremory/src/core/migrations/defs_a.rs:117-129`**
```rust
/// Idempotency gate (Vera 2026-05-28 BUG-1 fix):
///   Uses `PRAGMA table_info('entities')` to check whether `group_id` is already
///   in the PK — this is the SHAPE-based sentinel. The old `entities_bak_004`
///   backup-table sentinel was a false gate: if the process crashed after creating
///   the backup but before creating `entities_new`, the next startup would see
///   the backup, return Ok(()), and silently leave the migration incomplete.
///
/// FK restore on ALL exit paths (Vera 2026-05-28 BUG-2 fix):
///   `PRAGMA foreign_keys = ON` is guaranteed via the `body_result` wrapping
///   pattern — same approach used by migrate_006. An error in any restructure
///   step still restores FK enforcement before propagating the error.
```
**Classification:** VIOLATION on the headings only — this is the best example in the whole
sweep of "genuinely excellent technical documentation, ruined only by a session-dated
reviewer-name label." The prose below each heading is a real, load-bearing invariant
explanation (shape-based sentinel vs a false backup-table sentinel; FK restore on every exit
path) that a future maintainer absolutely needs — it must NOT be deleted, only un-tagged.
**Proposed rewrite:**
```rust
/// Idempotency gate — SHAPE-based, not existence-based:
///   Uses `PRAGMA table_info('entities')` to check whether `group_id` is already
///   in the PK. The old `entities_bak_004` backup-table sentinel was a false
///   gate: if the process crashed after creating the backup but before creating
///   `entities_new`, the next startup would see the backup, return Ok(()), and
///   silently leave the migration incomplete.
///
/// FK restore on ALL exit paths:
///   `PRAGMA foreign_keys = ON` is guaranteed via the `body_result` wrapping
///   pattern — same approach used by migrate_006. An error in any restructure
///   step still restores FK enforcement before propagating the error.
```

### 6.5 — `forget.rs`: content is already self-sufficient, tag is pure decoration

**`crates/kremory/src/facade/forget.rs:26-35`**
```rust
/// # Shared-entity preservation (Vera F17)
///
/// Entities referenced by ANY episode outside this `source_id` are NOT
/// deleted. Only entities whose entire `episodic_edges` set falls within
/// the matched episodes are removed. This prevents cross-source data loss
/// when a single entity ("Acme Corp") is mentioned in multiple ingested
/// documents.
```
**Classification:** VIOLATION, heading only — delete "(Vera F17)" from the heading and
nothing else needs to change; the paragraph is a complete, correct, self-contained invariant
description with a concrete example.
**Proposed rewrite:** `/// # Shared-entity preservation` (paragraph unchanged).

### 6.6 — TD-197 "Finding N" on a test function (borderline: doesn't ship to docs.rs, still narrative)

**`crates/kremory/src/memory/engine_handle.rs:1312-1319`**
```rust
/// TD-197 review Finding 1: `facts_attached_total` must count only the
/// facts `recall()` actually served — NOT every ownership-matched fact.
/// A reserved-predicate meta-edge (`potential_alias`) whose subject
/// matches a result entity satisfies the OWNERSHIP test but is dropped
/// by the per-entity projection's `is_reserved_predicate` filter a few
/// lines below regardless. Pre-fix, `facts_attached` was computed from
/// ownership alone, ahead of that filter, so it counted this fact as
/// "attached" even though it never reached the consumer — the exact
/// Rule 19 "counters that lie" shape...
```
**Classification:** VIOLATION — this is `///` on a `#[test]` fn, so it won't ship to
docs.rs, but it's still an internal-process narrative (TD-number, "review Finding 1", AND
a `Rule 19` citation to the operator's private global rules file, stacked in one comment).
The technical content (why the counter was wrong, what invariant the test proves) is good
and should survive.
**Proposed rewrite:**
```rust
/// `facts_attached_total` must count only the facts `recall()` actually
/// served — NOT every ownership-matched fact. A reserved-predicate
/// meta-edge (`potential_alias`) whose subject matches a result entity
/// satisfies the OWNERSHIP test but is dropped by the per-entity
/// projection's `is_reserved_predicate` filter a few lines below
/// regardless. Computing the counter from ownership alone, ahead of that
/// filter, would count this fact as "attached" even though it never
/// reaches the consumer.
```

### 6.7 — `Rule N` referencing a file outside the repo entirely

**`crates/kremory/src/core/background/mod.rs:190`**
```rust
/// per ADR-019 / CLAUDE.md Rule 19 (observability-first-class).
```
**Classification:** VIOLATION, and the worst-in-class per §5 — "CLAUDE.md Rule 19" points at
a file that is not tracked in this repository at all (it is the maintainer's personal
`~/.claude/CLAUDE.md`). A reader of this crate — including anyone on crates.io — has
literally no way to resolve this reference, not even by cloning the repo.
**Proposed rewrite:**
```rust
/// Emits a structured observability event for this step (request shape,
/// outcome, latency) rather than only a log line on the failure path —
/// see the module-level docs for the counter names.
```
(Rewrite states the actual invariant — "why is this instrumented this way" — instead of
citing the rule that motivated it.)

### 6.8 — `TD-` + date + doc-path, all three stacked (schema.rs)

**`crates/kremory/src/core/schema.rs:1750`**
```rust
/// Migration 027 (steal-matrix-rescore item 2): `entities_fts` must stem via
```
**Classification:** VIOLATION — "Migration 027" (a stable, permanent identifier — see the
NOT-a-violation note below) is fine to keep; "(steal-matrix-rescore item 2)" is the leak.
**Proposed rewrite:**
```rust
/// Migration 027: `entities_fts` must stem via
```
Note: `Migration 027` itself is NOT flagged — a numbered, permanent, already-applied SQL
migration is a stable intrinsic identifier (same class as "Migration 019" in the
[[avoid-ordinal-position-in-identifiers]] house rule's own worked exception list), not a
mutable reference to a review session.

### 6.9 — NOT a violation: legitimate "Phase" domain term

**`crates/kremory/src/core/ingest/pipeline/phase1.rs:27`**
```rust
/// Phase 1 ingest: store episode + run NER, return candidates.
```
**Classification:** NOT a violation. "Phase 1" here names a real, permanent, two-stage
ingest architecture (the file is literally called `phase1.rs`; there is a documented
`Phase 1 + Phase 2` write-path split described in this project's own architecture memory).
No internal review/session is being referenced. Leave unchanged.

### 6.10 — NOT a violation: legitimate "audit" domain term

**`crates/kremory/src/core/identity_verdict.rs:87`**
```rust
/// Free-text rationale — for audit/debug only, NEVER parsed for control flow.
```
**Classification:** NOT a violation. "Audit" here is the system's own first-class concept
(an audit trail / audit row is real, versioned, persisted data in this project — see
`dream/consistency_check/audit.rs`), not a reference to a code-review audit. Leave unchanged.

### 6.11 — NOT a violation: RFC 3339 date-format example (false positive on the date regex)

**`crates/kremory/src/core/extraction/parsers.rs:30-31`**
```rust
/// 2. Full RFC 3339 (`2023-05-07T00:00:00Z`) — what models trained on graphiti's
///    format emit unprompted, and what graphiti itself specifies.
```
**Classification:** NOT a violation — the date-shaped string is a format literal being
illustrated, not a session timestamp. (Same line legitimately also has `TD-187 round 2` two
lines above it at `parsers.rs:27`, which IS a violation — worth noting these two classes can
sit within a few lines of each other in the same doc comment.)

### 6.12 — `Vera DENT-001`, mid-severity: content useful but framed as a review artifact

**`crates/kremory/src/core/extraction/parsers.rs:55-58`**
```rust
/// Exposed so callers that need per-path success tracking (Vera DENT-001 —
/// post-repair success is suspicious and must be tracked separately from a
/// clean wrapped/bare-array parse, per `observability-first-class` cardinal
/// failure mode #9) can label their own metrics accordingly.
```
**Classification:** VIOLATION — "Vera DENT-001" and "per `observability-first-class`
cardinal failure mode #9" (another reference to the operator's own external rules doc) are
both internal/unresolvable citations layered onto a genuinely useful explanation
(post-repair success needs separate tracking from clean-parse success, because a repair-path
success is a weaker signal).
**Proposed rewrite:**
```rust
/// Exposed so callers that need per-path success tracking can label their
/// own metrics accordingly — a post-repair success is a weaker signal
/// than a clean wrapped/bare-array parse and should be counted
/// separately, not folded into one "parse succeeded" total.
```

---

## 7. Recommended refined regex for the fix pass (reduces false positives found above)

For a future automated or semi-automated fix pass, narrow the two high-noise patterns:

- **`Phase \d+`** → only flag when co-located (same doc-comment block, i.e. within ~3 lines)
  with one of `ADR-\d+`, `TD-\d+`, `recall-v2`, `impl spec`, `arch spec`, a bare parenthetical
  `(F\d\d`, or `finding`. Do NOT flag a bare "Phase N [pipeline-stage noun]" with no such
  co-located internal-doc locator.
- **`audit`** → only flag `public-docs-and-api-surface-audit` (the spec slug) and any
  `.ai-docs/**/*audit*` path reference. Do NOT flag standalone "audit row" / "audit trail" /
  "audit-grade" / "for audit only" — these are the system's own domain vocabulary.
- **`this pass`** → only flag when adjacent to a date or an ID; the bare phrase almost always
  means "one run of the dream-consolidation loop," a legitimate domain term.
- **Add `Rule \d+`** to the pattern list (see §5) — it was NOT in the original brief and
  should be, since it is the single most severe flavor found (unresolvable outside the repo
  entirely).

## 8. What was explicitly OUT of scope for this pass

- The 2,354 `TD-\d+` and 2,272 `ADR-\d+` raw hits were **not individually transcribed and
  classified line-by-line** in this document — at that volume (concentrated in large
  module-level `//!` doc-comment blocks, e.g. `kremory-eval/src/layer_b/graph_integrity.rs`
  alone carries dozens under one doc block), a full manual per-line table would be
  thousands of rows long, would not be reviewable, and the same handful of narrative idioms
  repeat throughout ("TD-NNN's fix (commit `abc123`) does X", "per ADR-NNN §N", "ADR-NNN /
  TD-NNN"). The file-level count table in §3/§4 plus the §6 worked examples are intended to
  let you (a) gauge true-positive confidence per pattern and (b) pick a remediation order
  (worst-offender files first) without hand-auditing all 6,496 lines up front. If you want a
  full line-by-line table for the TD-/ADR- bucket specifically before authorizing a fix, say
  so and it can be generated as a follow-up pass — it is mechanical, just large.
- Doc/markdown files (`.ai-docs/`, `docs/`) were not swept — the brief scoped this to
  `crates/*/src/` source comments only.

---

## 9. `core/dream/**` and `core/migrations/**` — FIXED (this pass)

Both directories have been swept and cleaned of the violation classes in §2/§3
(`TD-\d+`, `ADR-\d+`, `Vera`, `Quinn`, `Finding [A-Z]?\d+`, bare `F\d\d`, `steal-matrix`,
`Rule[- ]\d+`, session dates, plus `arch-spec` and `[[wiki-link]]` citations — two
extra patterns not in the original brief, added below). Legitimate domain vocabulary
(`Phase N` real pipeline stages, `Site #N`, `P1`–`P4` consolidation ops, `Migration NNN`,
`Q1`/`Q2`/`Fork N`/`F1`/`F2`/`C0`/`DoD EN` local labels defined in-file, test-fixture
literal dates, RFC3339-format examples, SQLite version-release dates) was deliberately
left untouched per §5/§7's own bimodal-pattern guidance.

**Verification**: a post-fix full-pattern grep of both directories
(`TD-[0-9]+|ADR-[0-9]+|Vera|Quinn|Finding [A-Z]?[0-9]+|steal-matrix|Rule[- ][0-9]+|
20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]|arch-spec|\[\[`) returns **zero matches** except
11 confirmed non-violations: test-fixture literal timestamps in
`dream/consolidation/{substrate,supersession,archive}.rs` and `migrations/tests.rs`,
and two SQLite-version-release-date citations in `migrations/{defs_b,defs_f}.rs`
("supported in SQLite ≥ 3.35.0 (2021-03-12)" etc.) — all left untouched, matching the
class of exemption the audit itself already carves out.

### Actual diff footprint (this agent's own count — authoritative for this scope)

```
23 files, crates/kremory/src/core/dream/**      616 insertions(+), 670 deletions(-)
17 files, crates/kremory/src/core/migrations/** 132 insertions(+), 140 deletions(-)
────────────────────────────────────────────────────────────────────────────────
40 files total                                  748 insertions(+), 810 deletions(-)
```

### Reconciling against the audit's own estimates — honest discrepancy note

The audit (§4) only published an exhaustive **per-file** count for the narrow
"high-confidence bucket" (`Vera`/`Quinn`/`Finding N`/bare `F\d\d`/`steal-matrix` union).
For this scope's files, that table listed **116 lines across 20 files** (e.g.
`migrations/defs_a.rs` 18, `dream/consolidation/cross_episode.rs` 16,
`dream/acronym_nickname_recall.rs` 8) — every one of those lines is now fixed, exact
match.

The audit explicitly declined (§8) to publish a full per-file breakdown for the
`TD-\d+`/`ADR-\d+` buckets ("a full manual per-line table would be thousands of rows
long"), so there is no narrower estimate to reconcile the remaining ~1,442 diff lines
against — the 748/810 figure above is this agent's own first exhaustive count for
those two directories, not a correction of a prior number. One genuine estimate
*was* checkable: `acronym_nickname_recall.rs`, called out mid-pass as "~98" hits by
this agent's own refined pattern before the `arch-spec`/`[[` additions — the actual
post-refinement count was **102** (both figures are this agent's own, not the audit's;
the audit's own table entry for this file, 8, was always scoped to the narrow bucket
only and is not in tension with either number).

**One violation class found in this scope that is NOT in the audit's pattern list**:
a lowercase, hyphenated citation form, `CLAUDE.md rule N` (e.g. `// test helper —
CLAUDE.md rule 5 test-exemption`), attached to `#[allow(clippy::too_many_arguments)]`
test-helper annotations. Occurred **17 times across 8 files** in this scope
(`type_registry_collapse.rs` ×3, `discover_types.rs` ×1, `acronym_nickname_recall.rs`
×3, `consolidation/substrate.rs` ×1, `consolidation/communities.rs` ×2,
`consolidation/supersession.rs` ×1, `consolidation/mod.rs` ×1,
`consolidation/cross_episode.rs` ×5). All fixed (restated as "test files are exempt
from the arg-count lint" with no rule-number citation). §7's refined-regex list should
add `Rule[- ]\d+` (note the hyphen alternative, not just the space-separated form
already listed) for any future pass over the remaining scope.

### Comments deliberately left alone (not violations) — worth flagging for other scopes

- `dream/provenance/mod.rs` and a few sibling files use bare `§2.1` / `§2.3` /
  `§4.0`–`§4.5` section-style references with no adjacent `ADR-`/`ir doc-path` citation.
  Left as-is where the numbering appears to be the module's own internal outline
  (self-referential within the same doc comment), matching the "Site #N" precedent —
  flagged here rather than silently normalized, since it's a judgment call this agent
  is not fully confident in without the module author's intent.
- `Q1`/`Q2`, `Fork N`, `D1`–`D8`, `DoD E1/E3/E4/E5/E7/E9`, `H3`/`H4` — all kept bare
  (ADR-number prefix stripped, local label retained) as established in-file/in-section
  identifiers, consistent with `Site #N` and `Migration NNN` precedent.

### A test failure surfaced during the gate run — NOT caused by this scope's edits

`cargo nextest run -p kremory -p kremory-mcp --features content-search,test-utils`
reports 1799/1800 passed, 1 failed, 6 skipped:
`adr_reference_integrity::exempt_numbers_are_all_still_needed` panics with *"ADR-019
is exempt but nothing under `crates/` cites it any more — delete the dead
exemption."* Verified this is **not** a consequence of this scope's edits: `ADR-019`
does not appear anywhere in this scope's diff, and does not appear anywhere under
`crates/` at all post-fix — meaning whatever file used to cite it was outside
`core/dream/**` and `core/migrations/**`, most likely stripped by one of the other
comment-hygiene agents running concurrently over the rest of the tree this same
session. Fixing it requires editing
`crates/kremory/tests/it/adr_reference_integrity.rs` (its dead-exemption list),
which is outside this agent's assigned scope — flagged for the orchestrator rather
than fixed here.

### Full quality gate (this scope)

- `cargo check -p kremory --lib` — clean after every file edit, no errors introduced.
- `cargo build -p kremory` — exit 0.
- `cargo clippy -p kremory --all-targets --all-features` — exit 0, zero warnings.
- `cargo nextest run -p kremory -p kremory-mcp --features content-search,test-utils` —
  1799/1800 passed, 6 skipped, 1 failed (pre-existing/cross-agent, see above — not a
  regression from this scope's edits).

---

## 10. `crates/kremory/src/core/*.rs` (top-level only) + `crates/kremory/src/core/ingest/**` — FIXED (this pass)

**Scope note:** top-level only for `core/*.rs` — NOT `core/dream/**`, `core/migrations/**`,
`core/extraction/**`, `core/background/**`, `core/disambiguation/**`, `core/graph/**`,
`core/provider/**`, `core/scoring/**` (those are other agents' scopes, §9 above covers
`dream`/`migrations`). This scope is 43 files: every `.rs` file directly under `core/` plus
every file under `core/ingest/` (including the nested `core/ingest/pipeline/` subdirectory).

### Discrepancy vs the audit's estimated counts — confirmed and material

The audit's §4 high-confidence table only lists a subset of this scope's files, and even for
listed files the counts are undercounts once the full violation-class net (not just
`Vera`/`Quinn`/`Finding N`/`F\d\d`/`steal-matrix`) is applied. Concretely:

- **`ingest_with.rs`**: audit's §4 table says 18 (narrow bucket only). This agent's actual
  fix count for that one file: **~112 replacement sites** across the full pattern class
  (`TD-\d+`, `ADR-\d+`, `DUR-\d+`, `Q-0\d`, `SCOPE-\d+`, `GAP-\d+`, `QB-\d+`, `Rule \d+`,
  memory-rule-slugs in `[[...]]` form, `V1-CANONICAL`, session dates). Not in tension with
  the audit's 18 — that number was always scoped to the narrow bucket only.
- **The audit's own §5 `Rule \d+` sample said "30 hits, 12 files" for the ENTIRE 182-file
  codebase.** A grep of just THIS scope's 43 files independently found the same 30 hits
  distributed across files this list didn't fully enumerate for this directory (`context.rs`,
  `ingest/pipeline/deferred.rs` were in the audit's 12; several more `Rule \d+`-adjacent
  hits surfaced only once the broader net — `spec §`, `MNT-002`, `V1-CANONICAL`, `FU\.\d`,
  `D9`/`D9a`, `C\d+ spec`, `\.ai-docs/` paths, `ADR-Phase-` — was applied to the same files).
  This confirms the audit's per-pattern counts are a **lower bound**, not a ceiling: files can
  (and in this scope routinely did) carry violation classes the original 14-pattern sweep
  never tested for.
- **Files in this scope with material hit counts that don't appear in the audit's §4 table at
  all**: `identity_verdict.rs` (25 fixes — `spec §`, `MNT-002`, `RISK-001`, `S4`/`S2` spike
  labels, memory-rule-slugs), `entity_types.rs` (25 fixes — `spec §` ×11, `ASMP-00\d` ×3, and
  the `D9`/`D9a` decision-label shorthand used unexplained 14 times), `format.rs` (10 fixes —
  `Story #151`/`#152`/`#164`), `speculative_cache.rs` (11 fixes — `Story #149`, `FU.\d`
  follow-up-task labels), `text_utils.rs` (8 fixes — `Story #148`/`#149`, `FU.5`),
  `resolver_batched.rs` (5 fixes — `RISK-00\d`), `engine.rs` (5 fixes — `Story #5`), `error.rs`
  (8 fixes — `DUR-3`, `Story #5`/`#150`/`#155`/`#209`, `arch-spec §`), `rates.rs` (4 fixes —
  `TD-133`), `contradiction.rs`/`arena.rs`/`mod.rs`/`confidence.rs`/`rerank.rs`/`sink.rs`
  (1–3 fixes each — `Option-1`, `P4`/`A.0`/`K7` internal spike labels, `MNT-002`,
  `spec §`/`R4 §`, `Risk #N`). None of these files appear in the audit's §4 table, which only
  sampled a subset of the codebase for its worked-example illustration, not an exhaustive
  per-file census.

### New violation classes found in this scope, not in the audit's original pattern list

Beyond the `Rule \d+` class the audit itself flagged as newly-discovered (§5):

- **`DUR-\d+`** — an internal invariant/decision numbering scheme (`DUR-2`, `DUR-3`, `DUR-7`),
  heaviest in `ingest_with.rs` (~15 occurrences) and `deferred.rs`/`sink.rs`/`error.rs`/
  `speculative_cache.rs`.
- **`Story #\d+`/`Story #[A-Z]\d+`** — internal tracking-story IDs (e.g. `Story #148`,
  `Story #A1`), found in 9 files across this scope, never mentioned in the audit.
- **`\[\[memory-rule-slug\]\]`** — the maintainer's own personal global rule-file citations in
  double-bracket form (e.g. `[[load-bearing-invariants-at-emit-not-prompt]]`,
  `[[treat-cause-not-symptom]]`, `[[audit-what-guards-mask-before-deleting]]`,
  `[[observability-first-class]]`, `[[llm-output-parse-loudly]]`). Arguably the single WORST
  instance of the pattern the whole audit is about: even `Rule 19` at least resolves to a
  numbered heading in a file that exists (`~/.claude/CLAUDE.md`); a memory-rule-slug resolves
  to a markdown file in the maintainer's private `~/.claude/rules/` or session-memory
  directory that has no stable public identity at all. Found in `identity_verdict.rs`,
  `ingest_with.rs` (×2), `ingest/mod.rs` (deferred.rs's precursor pass).
- **`V1-CANONICAL §N.N`** — an internal canonical-spec-document name+section citation, distinct
  from `ADR-\d+`/`TD-\d+` (no numeric ID to grep for), found in `config.rs`, `sink.rs`,
  `schema.rs`, `search.rs`, `deferred.rs`.
  **`Q-0\d`/`SCOPE-\d+`/`GAP-\d+`/`QB-\d+`** — one-off internal finding/decision-tag families,
  each occurring only in `ingest_with.rs` but each unambiguously the same unresolvable-citation
  shape as `RISK-\d+`/`ASMP-\d+`/`MED-\d+`.
- **`spec §N.N` / `C\d+ spec §N.N` / `arch-spec §N.N` / `arch spec §N.N`** — bare internal
  design/architecture-spec section references with no accompanying ADR/TD number, so invisible
  to the audit's original regex set entirely. The single largest additional-fix contributor in
  this scope — found in 9 files, ~35 total sites.
- **`.ai-docs/research/...\.md` / `.ai-docs/specs/...\.md` inline paths** — internal doc-path
  citations embedded directly in `///` doc comments (not just alongside a TD/ADR number),
  found in `extraction_window.rs`, `context.rs`.
- **`ADR-Phase-[A-Z]\.\d:\d+`** — a non-numeric ADR variant (e.g. `ADR-Phase-D.0:88`) that the
  audit's `ADR-\d+` regex cannot match at all (no digits immediately after `ADR-`). Found in
  `extraction_window.rs`, `config.rs`.
- **`MNT-002`** — an internal "maintenance pattern" tag, found in `identity_verdict.rs` (×5)
  and `mod.rs`.
- **`FU\.\d` (`FU.2`, `FU.5`, `FU.6`)** — internal "follow-up task N" numbering, found in
  `text_utils.rs`, `speculative_cache.rs`, `schema.rs`.
- **`D9`/`D9a`** (entity_types.rs only, 14 occurrences) and **`P4`/`A.0`/`K7`** (arena.rs,
  3 occurrences) — bare internal decision-label shorthand used the SAME way `Rule \d+` and
  `TD-\d+` are used elsewhere: as a parenthetical justification tag that assumes the reader
  already knows what the letter+number means, rather than stating the invariant plainly.
  Distinguished from the legitimate bare-label precedent (`Site #N`, `Migration NNN`,
  `Pass 0/1/2`, `Phase 1/2 ingest`, `L1`–`L7`) by whether the label is EXPLAINED once and used
  consistently as real domain vocabulary (kept) vs. cited repeatedly as an unexplained
  reference back to an external decision record (stripped). `D9`/`D9a` failed this test —
  it was consistently used as `"(D9)"`/`"(D9a)"` parenthetical tags across 14 sites, always
  requiring the reader to already know the referent, never itself defining the invariant at
  most of those sites.

### Ambiguous bare labels deliberately LEFT UNTOUCHED (judgment calls, flagged per the task's
own instruction not to guess)

- **`D7:`** (deferred.rs, mod.rs, ingest_with.rs — recurring, e.g. "D7: episode_id is a tracing
  field only, NEVER a metric label") — a real, consistently-explained metric-cardinality rule
  used as domain shorthand, not a citation back to an external record. Kept per the
  `Site #N`/`Migration NNN` precedent.
- **`Bug A`/`Bug B`/`Bug E`** (ingest_with.rs) — bare section-marker labels with no numeric
  citation attached; read as local structural headings within the function, not references to
  an external bug tracker. Left alone.
- **`P0`** (ingest_with.rs, "pure P0 behaviour") — a baseline-behavior shorthand with no
  adjacent citation; ambiguous whether it originates from an internal spec's phase-numbering
  scheme, but occurs bare with no explanatory tag attached to strip. Left alone.
- **`G1`/`G2`/`G3`** (schema.rs, `migrate_006` test suite — "G1 shape gate", "G2 pre-condition
  gate", "G3 gate must fire") — flagged as a genuine judgment call, not a confident classification.
  These *may* derive from an external review's G-numbered findings (the pattern shape matches
  `MED-NN`/`HIGH-NN` structurally), but within the test suite they read as locally-defined gate
  names with their own inline explanation each time. Left untouched; recommend a follow-up
  agent with access to the review history that produced them re-examine this one.

### The `adr_reference_integrity.rs` test — a real, fixed consequence of this scope's edits

`cargo nextest` initially failed 1 test after this scope's edits landed:
`adr_reference_integrity::exempt_numbers_are_all_still_needed`, panicking with *"ADR-020 is
exempt but nothing under `crates/` cites it any more — delete the dead exemption."* Unlike
§9's ADR-019 case (flagged, not fixed, because it was outside that agent's scope), **this one
WAS caused by this scope's edits** — `schema.rs`'s `BeginGuard`-related citations, part of
this scope, carried the last remaining `ADR-020` reference under `crates/`. Fixed directly:
removed the now-dead `"020"` entry from `EXEMPT_ADR_NUMBERS` in
`crates/kremory/tests/it/adr_reference_integrity.rs`, following the exact precedent the file's
own doc comment already established for the ADR-019 case (both removals are now documented in
the same comment, in the same historical note). Re-verified: `grep -rn 'ADR-020' crates/`
returns only the doc-comment prose describing the removal (which is excluded from the guard's
own scan via `SELF_PATH`, so it cannot re-trigger the check).

### Total scale (this scope)

- **43 files** in scope; **19 files** required at least one fix
  (`intelligence.rs`, `identity_verdict.rs`, `format.rs`, `entity_types.rs`, `extraction_window.rs`,
  `text_utils.rs`, `error.rs`, `resolver_batched.rs`, `engine.rs`, `sink.rs`,
  `speculative_cache.rs`, `rates.rs`, `schema.rs`, `arena.rs`, `contradiction.rs`,
  `resolver_batched.rs`, `context.rs`, `confidence.rs`, `rerank.rs`, `mod.rs`, `search.rs`,
  `config.rs`, `ingest/tests.rs`, `ingest/helpers.rs`, `ingest/pipeline/types.rs`,
  `ingest/pipeline/phase1.rs`, `ingest/pipeline/deferred.rs`, `ingest/pipeline/ingest_with.rs`
  — 27 files with confirmed fixes, some via a second later sweep after the broadened net).
- **~330 individual comment/doc-string replacement sites** across those 27 files (single
  largest: `ingest_with.rs` at ~112; `search.rs` ~75; `config.rs` 65; `schema.rs` ~73;
  `identity_verdict.rs` 25; `entity_types.rs` 25).
- **24 files with zero hits** on the full broadened pattern net (verified, not assumed):
  `intent.rs`, `ner.rs`, `chunking.rs`, `hybrid_extractor.rs`, `hybrid_extractor_helpers.rs`,
  `hybrid_extractor_parsing.rs`, `hybrid_extractor_prompts.rs`, `chat_tracking.rs`,
  `embedding.rs`, `embed_prefix.rs`, `proximity.rs`, `grounding.rs`, `obs.rs`, and 11 more.

### Full quality gate (this scope)

- `cargo check -p kremory --features content-search` — clean after every file, no errors
  introduced (run incrementally, ~15 times, once per file/batch).
- `cargo build -p kremory` — exit 0.
- `cargo clippy -p kremory --all-targets --all-features` — exit 0, zero warnings.
- `cargo nextest run -p kremory -p kremory-mcp --features content-search,test-utils` —
  **1800/1800 passed, 6 skipped, 0 failed** (after fixing the `adr_reference_integrity.rs`
  dead-exemption regression this scope's own edits caused — see above; the fix is a test-suite
  change, not a source-comment change, and is the one edit in this pass outside the
  `core/*.rs`/`core/ingest/**` scope, made because the task's quality-gate requirement cannot
  be satisfied otherwise).

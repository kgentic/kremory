---
title: 'ADR-003 — Two-time-axis memory (bi-temporal) for audit and compliance'
type: adr
status: accepted
audience: public
created: '2026-05-22'
ratified: '2026-05-22'
ratification_mode: full
slug: rql/adr-003-bitemporal-audit-compliance-2026-05-22
tags:
  - adr
  - kremory
  - bitemporal
  - temporal
  - audit
  - compliance
  - contradiction-resolution
refs:
  - id: rql/adr-001-engine-architecture-single-crate-apache2-2026-05-22
    rel: depends_on
  - id: rql/adr-004-monorepo-multi-crate-2026-05-22
    rel: informed_by
  - id: rql/adr-005-infrastructure-positioning-three-layer-2026-05-22
    rel: implements
    audience: internal-roadmap
---

# ADR-003 — Two-Time-Axis Memory (Bi-Temporal) for Audit and Compliance

**Status**: accepted (full)
**Ratified**: 2026-05-22 (James)
**Decision type**: HIGH — core architectural moat; defines the schema primitives that distinguish kremory from every named competitor

---

## 1. Decision Summary

The `kremory` engine tracks two independent time dimensions for all facts stored in the knowledge graph:

1. **Transaction time** (`recorded_at`) — when the system learned this fact. Immutable once written. The system clock at ingest time.
2. **Valid time** (`valid_from`, `valid_to`) — when the fact was true in the world. Mutable via contradiction resolution. Represents the agent's model of reality.

These two axes are independent. A fact may have been recorded yesterday (`recorded_at = T-1`) but describe something that happened last week (`valid_from = T-7`). A later episode may invalidate the fact (`valid_to = T-1` set retroactively) without destroying the original record (`recorded_at = T-1` remains immutable).

This is the "two-clock" model. No direct competitor has implemented both clocks with an active contradiction resolver. This is kremory's deepest architectural moat.

---

## 2. Schema Columns

All entities and edges in the kremory knowledge graph carry these temporal columns:

```sql
-- Applied to entities table
recorded_at     INTEGER NOT NULL,  -- TX time: system clock at ingest (Unix ms, immutable)
valid_from      INTEGER NOT NULL,  -- Valid time: start of fact's world-truth window
valid_to        INTEGER,           -- Valid time: end of window (NULL = currently valid)

-- Applied to edges table (same pattern)
recorded_at     INTEGER NOT NULL,
valid_from      INTEGER NOT NULL,
valid_to        INTEGER,
```

The `recorded_at` column is written once at ingest time and never updated. It forms the transaction-time axis.

The `valid_from`/`valid_to` columns form the valid-time axis. Contradiction resolution updates `valid_to` on the invalidated fact rather than deleting the row.

---

## 3. The Canonical Query

The key query enabled by the two-clock model:

> "What did the agent know at time X if asked at time Y?"

- **X** is a point on the valid-time axis (what was true in the world as of X)
- **Y** is a point on the transaction-time axis (what the agent had recorded as of Y)

This query is impossible in a one-clock system without a separate audit log. In kremory, it is a first-class query expressible as:

```sql
SELECT * FROM entities
WHERE recorded_at <= :tx_time_Y     -- facts the agent knew as of Y
  AND valid_from <= :valid_time_X   -- facts valid as of X
  AND (valid_to IS NULL OR valid_to > :valid_time_X)  -- not yet invalidated as of X
```

This query is the basis for:
- **Debugging**: "Why did the agent make decision D at time T? What did it believe then?"
- **Audit trails**: "What was the agent's knowledge state during the customer interaction on date D?"
- **Legal and compliance**: "What did the system represent to the user at the time of the contract signing?"
- **Retroactive correction**: "The meeting was actually Tuesday, not Monday — correct the record without losing the history of the wrong belief."

---

## 4. Contradiction Resolution Engine

kremory has an **active contradiction resolver** — not just edge-label annotations. When a new episode contains a fact that conflicts with an existing fact about the same entity attribute:

1. **Detection**: the enrichment pipeline detects a contradiction between the new claim and existing entities in the graph (same entity, same attribute type, overlapping valid-time windows).
2. **Resolution**: the resolver chooses a `ContradictionResolution` strategy — `Superseded`, `Merged`, `Forked`, or `Ignored` — based on confidence scores, source type, and temporal context.
3. **Invalidation**: for `Superseded`, the old fact's `valid_to` is set to the `recorded_at` of the new fact. The old record is not deleted; its valid-time window is closed.
4. **Provenance**: the new fact records a reference to the invalidated fact's ID.

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContradictionResolution {
    Superseded,  // Old fact's valid_to closed; new fact becomes the canonical record
    Merged,      // Old and new facts merged into a single entity (dedup path)
    Forked,      // Both facts retained with non-overlapping valid windows (temporal fork)
    Ignored,     // New fact discarded (old fact had higher confidence or source priority)
}
```

This is distinct from codemem's approach (verified 2026-05-22):
- codemem has `Contradicts`, `InvalidatedBy`, `Supersedes` as edge-type labels in its `RelationshipType` enum (source: `types.rs` sha `c32d7577`)
- codemem has no file in its source tree implementing contradiction resolution logic (source: directory listing of `codemem-engine/src/` — no `contradiction.rs` or equivalent found)
- codemem's consolidation cycles (Decay, Creative/REM, Cluster, Summarize, Forget per README) do not include contradiction resolution

kremory's contradiction resolver is an active algorithm, not a passive label. This distinction is real and verifiable.

---

## 5. Competitor Bi-Temporal Status (Verified)

| Project | Time model | Source |
|---|---|---|
| **kremory** | **Two-clock: transaction_time + valid_time** | This ADR; D.6 architecture spec |
| cogniplex/codemem | One clock: `valid_from`/`valid_to` on edges and nodes only (migrations 003+015); no contradiction resolution engine | `codemem-2026-05-22.md` §3.6–3.7 |
| CodeGraph-rust | Operational timestamps only; no temporal graph model | `codegraph-rust-2026-05-22.md` §2.5 |
| Mem0 | No temporal graph (graph layer removed in v2 migration) | `rqlm-licensing-revisit-research-2026-05-19` §3 |
| Letta | No temporal model | `rqlm-licensing-revisit-research-2026-05-19` §3 |
| Zep/Graphiti | Partial: entity/fact timestamps, supersession edges, one clock | `rqlm-licensing-revisit-research-2026-05-19` §3 |
| codebase-memory-mcp (DeusData) | No temporal model; no graph memory | DeusData README (verified 2026-05-22) |

The two-clock bi-temporal model is kremory's exclusive differentiator across the entire named competitive set as of 2026-05-22.

---

## 6. This Is NOT the Marketing Headline

The bi-temporal model is kremory's most technically defensible differentiator. It is NOT the v0.1.0 marketing headline.

**Why**: bi-temporal as a lead message maps to "academic" pain, not "I need to ship an agent today" pain. The developer reading "two-clock bi-temporal knowledge graph" as a headline needs to already know they have a temporal audit problem. Most developers evaluating agent-memory libraries do not yet know they have that problem.

**Marketing positioning**:
- **Headline**: "Embed agent memory in your app. Pure Rust. BYOM. Apache-2.0."
- **Sub-headline**: "Pure Rust agent memory engine. Single binary. No server process. No subscription required to ship."
- **Deep feature disclosure** (§4 of the comparison doc, §4 of the Show HN body): bi-temporal model as audit features for regulated industries

The bi-temporal model is the feature that matters most to developers in regulated industries (legal, healthcare, fintech) who need immutable audit history and retroactive correction. It is disclosed in the comparison doc and the architecture doc as a deep technical capability — not in the 10-word marketing headline.

---

## 7. Consequences

### Positive
- Uncontested architectural moat: no competitor has two-clock bi-temporal + active contradiction resolver
- Enables audit features for regulated industries (healthcare, fintech, legal) — two-clock immutable history supports compliance query patterns
- Retroactive correction is possible without data loss — a critical feature for any agent that can be wrong
- The schema design is simple (3 columns per table) but powerful; no exotic database extensions required

### Negative
- Two-clock queries are more complex to write than single-clock queries. kremory must provide ergonomic query helpers that abstract this complexity for common patterns.
- Valid-time management creates operational complexity: when is a fact "current"? The answer is `WHERE valid_to IS NULL`, but this must be documented clearly.

### Neutral
- The bi-temporal model is internally invisible to consumers who use the high-level `memory::submit_episode` API — contradiction resolution happens inside the enrichment pipeline. Consumers only see the results: the graph always reflects the agent's current best model of reality, with history preserved.

---

## 8. Ratification

Accepted 2026-05-22. Full ratification. The two-time-axis model with active contradiction resolver is a permanent architectural feature of `kremory::core`. It cannot be removed without a full schema migration and a new major version.

- [x] `recorded_at` (transaction time) — immutable at write time
- [x] `valid_from` / `valid_to` (valid time) — mutable via contradiction resolver
- [x] `ContradictionResolution` enum with four strategies
- [x] Active contradiction resolver (not label-only)
- [x] Bi-temporal is the deep feature for compliance verticals, NOT the marketing headline

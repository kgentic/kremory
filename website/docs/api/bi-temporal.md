# Bi-temporal model


kremory's storage model uses two independent time axes per fact:

| Column | Axis | Mutability | Set by |
|---|---|---|---|
| `recorded_at` | Transaction time | Immutable | System clock at ingest |
| `valid_from` | Valid time | Mutable | `published_at` precedence chain |
| `valid_to` | Valid time | Mutable | Contradiction resolver (v0.1.1) |

### `published_at` precedence chain

For document-extracted facts, `valid_from` honours:

```
fact.valid_from  >  source_doc.published_at  >  ingest_time
```

Set `published_at` to anchor facts to the document's publication date, not the
moment of ingest:

```rust
mem.remember("Q4 2025 policy")
    .from_document("policy-q4-2025.pdf")
    .published_at(chrono::DateTime::parse_from_rfc3339("2025-10-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc))
    .await?;
// valid_from on extracted facts = 2025-10-01, regardless of when you call this
```

### Audit query

`.as_of(t)` is a **valid-time** query — *"what was TRUE in the world at t"* — and it is wired
end-to-end (ADR-068). This is the predicate the engine actually issues:

```sql
-- what kremory issues for as_of(t) — ONE bound parameter, valid-time only:
SELECT … FROM facts
WHERE (subject_id = ?1 OR object_id = ?1)
  AND expired_at IS NULL
  AND valid_from <= ?2
  AND (valid_to IS NULL OR valid_to > ?2)
```

Note what `as_of` does **not** do: it never gates on `recorded_at`. A fact kremory learned
*after* `t` still counts if its `valid_from` precedes `t` — learning something retroactively
does not change whether it was true then. `invalid_at` is likewise excluded from the
predicate: a fact contradicted in 2026 but valid across March 2024 still appears for
`as_of("2024-03-01")`, which is the point of a bi-temporal audit trail.

#### The second clock

**Every recalled fact carries both clocks.** `RetrievedFact` exposes all four temporal
fields — `valid_at`, `invalid_at` (world clock) and `recorded_at`, `expired_at` (system
clock) — so transaction time is available to you on every result:

```rust
// `.raw()` is required: bare `.recall(..)` resolves to a rendered String.
// `.raw()` yields Vec<RetrievedContext>, one per matched entity, each with `.facts`.
let contexts = mem.recall("alice's role").as_of(x).raw().await?;
let known_at_y: Vec<_> = contexts
    .into_iter()
    .flat_map(|c| c.facts)            // valid-time already filtered server-side by as_of(x)
    .filter(|f| f.recorded_at <= y)   // transaction time — filtered by you
    .collect();
```

**Limits, stated plainly.** `recorded_at` is *returned*, not *queryable*: kremory has no
server-side filter on it, so the snippet above post-filters whatever the top-k recall
returned rather than scanning history. That is enough to answer *"of the facts I retrieved,
which did the agent already know at Y"* — it is **not** a graph-wide audit scan.

`TemporalGraph::entity_history` does return one entity's full unfiltered history, but it is
**not reachable on the stable public API** (the only accessor, `temporal_graph_for_test`, is
`#[cfg(feature = "test-utils")]` and documented as not-public).

A server-side two-clock filter is a known, small, additive extension — one predicate and one
bound parameter on a query that already selects `recorded_at`. It is **not built**, because
nothing has asked for it and it cannot improve recall (a conjunctive filter only ever shrinks
a result set).

### Memory types (7-type taxonomy)

Stored facts are classified into one of 7 memory types, accessible via `kremory::MemoryType`:

| Variant | Meaning |
|---|---|
| `Episode` | Raw ingested episode (the original text) |
| `Entity` | Extracted named entity (person, org, concept) |
| `Fact` | Subject–predicate–object triple |
| `Community` | Graph community summary (from dream phase) |
| `Edge` | Relationship between entities |
| `Summary` | LLM-generated distillation |
| `Observation` | Temporal observation on an existing entity |

---

# kremory API reference

The complete reference for `kremory::Memory` — every builder knob, every request method, and
the reversibility, feature-flag and Node-binding surfaces around it. Reach for
[Getting Started](../getting-started.md) first if you have not run a working program yet; come
here once you are past that and need the full picture, or to look up a specific method.

The pages are ordered roughly the way a consumer adopts the surface, not by internal
implementation grouping. Pages with a "why would I want this" trade-off worth stating (Recall's
`.content()` versus `.raw()`, Reversibility's `merge_nogood` rationale) lead with the trade-off
before the code; pages that are simple mechanism just show the code.

> **v0.8** — The primary consumer surface is `kremory::Memory`. Substrate free-functions
> (`kremory::memory::submit_episode`, and friends) remain public for advanced users; most
> applications should use the facade described here. Since v0.1.3 the facade gained a
> fully-wired [dream consolidation phase](./dream.md), [reversible graph mutations with a
> see/undo surface](./reversibility.md), [BM25/FTS5 content recall](./recall.md)
> (`content-search` — a DEFAULT feature since ADR-078), and a
> [feature-flag matrix](./feature-flags.md). The [Node binding](./node-binding.md) mirrors the
> surface in camelCase.

---

## The pages

**Opening a `Memory`**

- **[Setup](./setup.md)** — the quickstart path, then the three tiers for wiring an LLM and an
  embedder: environment-driven shortcuts, named shortcuts, and the full builder with
  observability.
- **[Namespaces and multi-tenancy](./namespaces.md)** — default and per-call namespaces, the
  multi-tenant SaaS pattern, namespace policies (ADR-029a), and per-namespace entity-type
  vocabularies.

**Writing and reading**

- **[Ingest](./ingest.md)** — `remember()`, source metadata, supplying pre-extracted facts to
  skip the extraction LLM call, batch ingest, and chunking large documents.
- **[Recall](./recall.md)** — the recall terminals (prompt-ready string, `.raw()`, templates,
  `.content()` for BM25/FTS5 over raw episode text), plus session expansion and prior-turn
  replay for references that resolve across conversation turns.
- **[Bi-temporal model](./bi-temporal.md)** — the two clocks, `.as_of(t)` valid-time queries,
  the `published_at` precedence chain, and the audit query.

**Maintaining the graph**

- **[Dream phase and consolidation](./dream.md)** — what the background consolidation phase
  does, how to read `DreamSummary` honestly, tuning which operations run, namespace scoping,
  idempotent batch keys, and periodic scheduling.
- **[Reversibility and deletion](./reversibility.md)** — the mutation log, the unified
  `undo(mutation_id)` dispatcher and its per-kind siblings, the `merge_nogood` guarantee, the
  direct mutations, and `forget()` for GDPR erasure.

**Integrating**

- **[Async patterns and event sinks](./async-and-events.md)** — fire-and-forget ingest, batch
  status, the explicit await-enrichment form, the background ingestor, and memory-level or
  per-call event sinks.
- **[Advanced — substrate composition](./advanced.md)** — bringing a custom `GraphHandle`
  backend, reading back the active config, and the process-global engine singleton.
- **[Feature flags](./feature-flags.md)** — what every Cargo feature turns on, which are on by
  default, and the advanced `MemoryBuilder` tuning knobs.
- **[Node binding](./node-binding.md)** — how the JS surface mirrors this one.

Upgrading from an earlier version is covered separately in the
[upgrade guide](../releases/upgrade-guide.md).

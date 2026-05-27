# kremory API Reference

> **Scope**: This document covers the kremory v0.1.0 substrate-level public API
> (`kremory::memory::*` free functions and types). The consumer-facing facade
> (`kremory::Memory`) ships in a later revision; this placeholder exists to
> satisfy v0.1.0 Story #22 gate.

---

## Entry points — `kremory::memory`

All public async functions take a `&impl GraphHandle` (or `Arc<dyn GraphHandle>`)
as their first argument. They never own the handle; callers manage lifetime.

### `submit_episode`

```rust
pub async fn submit_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    scope: WorkspaceScope,
    batch_id: Option<String>,
    opts: SubmitOpts,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
) -> Result<EpisodeCommit>
```

Store one episode. Phase 1 (embed + graph write) completes synchronously.
Phase 2 (LLM enrichment) runs in-process if `opts.run_in_background = false`;
otherwise it is queued and this fn returns immediately.

### `submit_dream_phase`

```rust
pub async fn submit_dream_phase(
    graph: &dyn GraphHandle,
    scope: WorkspaceScope,
    provider: Arc<dyn ChatProvider>,
    batch_id: Option<String>,
    opts: DreamOpts,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
) -> Result<DreamHandle>
```

Submit a batch consolidation (dream phase) for the scope. Returns immediately
with a `DreamHandle`. Idempotent on `(scope, batch_id)`.

### `await_enrichment`

```rust
pub async fn await_enrichment(
    graph: &dyn GraphHandle,
    run_id: Uuid,
    timeout: Duration,
) -> Result<IngestStatus>
```

Poll until the Phase 2 run for `run_id` reaches a terminal state or `timeout` elapses.

### `await_dream`

```rust
pub async fn await_dream(
    graph: &dyn GraphHandle,
    run_id: Uuid,
    timeout: Duration,
) -> Result<DreamStatus>
```

Poll until the dream-phase run for `run_id` reaches a terminal state or `timeout` elapses.

### `await_batch_enrichment`

```rust
pub async fn await_batch_enrichment(
    graph: &dyn GraphHandle,
    batch_id: &str,
    timeout: Duration,
) -> Result<BatchStatus>
```

Wait until every episode in `batch_id` is terminal. Polls `graph_batch_status` internally.

### `ingest_episode` (backwards-compat)

```rust
pub async fn ingest_episode(
    graph: &dyn GraphHandle,
    content: &str,
    source_ref: SourceRef,
    structured_facts: Vec<StructuredFact>,
    provider: Arc<dyn ChatProvider>,
    scope: WorkspaceScope,
) -> Result<EpisodeCommit>
```

Deprecated wrapper around `submit_episode` with `SubmitOpts::default()` and no sink.

### `run_dream_phase` (backwards-compat)

```rust
pub async fn run_dream_phase(
    graph: &dyn GraphHandle,
    scope: WorkspaceScope,
    provider: Arc<dyn ChatProvider>,
) -> Result<DreamPhaseResult>
```

Deprecated synchronous dream phase. Prefer `submit_dream_phase` + `await_dream`.

### `search`

```rust
pub async fn search(
    graph: &dyn GraphHandle,
    scope: &WorkspaceScope,
    query: &str,
    opts: SearchOpts,
) -> Result<Vec<RetrievedContext>>
```

Hybrid retrieval over the scoped graph. Returns ranked `RetrievedContext` items.

### `context_block`

```rust
pub fn context_block(results: &[RetrievedContext], template: ContextTemplate) -> String
```

Render search results as a formatted context string for injection into an LLM prompt.
Synchronous; no I/O.

---

## Key types

### `WorkspaceScope`

Identifies a tenant + optional conversation thread. Entities are isolated per scope.

```rust
WorkspaceScope::new("workspace-id")
WorkspaceScope::with_thread("workspace-id", "thread-id")
```

### `SourceRef`

Metadata attached to one ingested episode.

```rust
SourceRef {
    kind: SourceKind,         // Meeting | Document | Chat | ...
    id: String,               // caller-assigned stable ID
    occurred_at: DateTime<Utc>,
    published_at: Option<DateTime<Utc>>,  // overrides valid_from when set
}
```

### `StructuredFact`

A pre-extracted (subject, predicate, object) triple, optionally with temporal bounds.

```rust
StructuredFact {
    subject: String,
    predicate: String,
    object: String,
    valid_from: Option<DateTime<Utc>>,
    valid_to: Option<DateTime<Utc>>,
    memory_type: Option<MemoryType>,
}
```

### `SearchOpts`

Controls retrieval behaviour.

```rust
SearchOpts {
    limit: Option<usize>,         // default 10
    filters: SearchFilters,
    as_of: Option<DateTime<Utc>>, // v0.1.1 — emits warn at v0.1.0
}
```

### `ContextTemplate`

```rust
pub enum ContextTemplate {
    Entities,    // name-summary pairs
    Temporal,    // fact triples with timestamps
    Full,        // entities + facts + edge summary
}
```

### `GraphHandle` trait

Storage-backend boundary. Implement this to plug kremory into a custom backend.
See `kremory::memory::graph::GraphHandle`.

For tests: `kremory::memory::StubGraphHandle` (enabled via `test-utils` feature).

---

## Error handling

All async functions return `kremory::memory::Result<T>`, which is
`Result<T, kremory::core::error::Error>`. Errors are non-exhaustive enums with
named struct variants; match on the variant you care about, use `_` for the rest.

---

## Feature flags

| Flag | Effect |
|---|---|
| `ner` | Enable NER-assisted extraction (requires model at runtime) |
| `llm` | Enable LLM enrichment phase (requires `ChatProvider` at runtime) |
| `otel` | Enable OpenTelemetry trace export (v0.1.1+) |
| `test-utils` | Export `StubGraphHandle` and test helpers |

---

*This document will be replaced by facade-first API docs when the `kremory::Memory`
consumer facade ships (Story A.9 / v0.1.x).*

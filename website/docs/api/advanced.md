# Advanced — substrate composition


For users who need full control: custom graph backends, multi-engine setups, direct
vector index manipulation, or cross-tenant orchestration above the facade.

Illustrative — `graph` and `provider` below stand for a `&dyn GraphHandle` and an
`Arc<dyn ChatProvider>` YOU supply (e.g. `kremory::memory::TemporalGraph`, or your own
`GraphHandle` impl); this snippet shows the call shape, not a standalone program:

```rust,ignore
use kremory::memory::{submit_episode, submit_dream_phase, await_enrichment, search,
                       context_block, GraphHandle, Namespace, SourceRef, SourceKind,
                       SubmitOpts, SearchOpts, ContextTemplate, SubmitEpisodeParams,
                       SearchParams};
use std::sync::Arc;

// You supply the graph handle (e.g. kremory::memory::TemporalGraph or your own impl)
// and manage LLM + embedder Arcs directly. Both fns take a single bundled params
// struct (args-as-object) rather than positional arguments.
let commit = submit_episode(SubmitEpisodeParams {
    graph: graph.as_ref(),
    content: "Alice prefers async Rust",
    source_ref: SourceRef {
        kind: SourceKind::Chat,
        id: "session-42".into(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    },
    structured_facts: vec![],   // empty = run Phase 2 LLM extraction
    provider: provider.clone(), // Arc<dyn ChatProvider>
    namespace: Namespace::new("user-alice"),
    batch_id: None,
    opts: SubmitOpts::default(),
    sink: None,
}).await?;

// Hybrid retrieval
let results = search(SearchParams {
    graph: graph.as_ref(),
    query: "rust preferences",
    namespace: Namespace::new("user-alice"),
    opts: SearchOpts { limit: Some(10), ..Default::default() },
}).await?;

// Render as prompt-ready string
let ctx = context_block(&results, ContextTemplate::TemporalFacts);
```

`GraphHandle::graph_run_consolidation` (the trait method backing dream consolidation) returns
`DreamPhaseResult` — the raw per-op counts that `impl From<DreamPhaseResult> for DreamSummary`
converts into the facade's `DreamSummary` (see [Dream phase and consolidation](./dream.md)). `IngestResult` is the return type of
`ingest_episode`, a `#[deprecated]` free function superseded by `submit_episode` — new code should
not reach for either directly.

**Diagnostic / lower-level dream internals** (Phase C developer surface, mainly for
debugging/tooling rather than mainline consumer code): `Memory::run_dream_pass_sync(opts:
DreamPassOpts) -> Result<DreamSummary>` runs a synchronous dream pass — note its own doc comment
flags that two of its sub-passes (type discovery, ghost-episode retry) are stubs returning
empty/zero counts, so treat it as diagnostic rather than a source of truth for those counts;
`Memory::ghost_episodes(group_id)` lists episode ids that never completed extraction (candidates
for re-ingest); `TypeProposal` is the Dream Pass 0 type-discovery output shape. `Memory::
assert_entity_type` (see [Namespaces and multi-tenancy](./namespaces.md)) also lives in this tier.

### Custom `GraphHandle` backend

Implement `GraphHandle` to plug kremory into a custom storage backend. `GraphHandle` is declared
with `#[async_trait]` (not native `async fn` in trait) — an implementer needs the SAME macro on
the `impl` block, exactly as below, for the desugared `async fn` signatures to match. `GraphHandle`
has 12 required methods with no default bodies (ADR section 4.9 — compiler-enforced shape stability), so
a real implementation is a substantial adapter; the sketch below is illustrative pseudo-code, not
a runnable snippet:

```rust,ignore
use kremory::GraphHandle;

struct MyGraphBackend { /* ... */ }

#[async_trait::async_trait]
impl GraphHandle for MyGraphBackend {
    // Implement all 12 required methods — ingest, dream, search, etc.
    // See `kremory::memory::graph::GraphHandle` for the full method list.
}

// Then pass it to Memory::open via a lower-level constructor (advanced)
```

### Introspection — reading back the ACTIVE config

`Memory::search_config() -> SearchConfig` and `Memory::contradiction_detection_enabled() -> bool`
are read-only accessors (both cheap — no I/O) that report the config the pipeline is *actually*
running: the compiled-in default, folded through any `KREMORY_*` env override at construction, and
any explicit builder `.with_*` call, in that precedence order. Reach for these instead of
re-reading env vars yourself — a transport or consumer that re-reads env independently can drift
from what the search path actually uses, silently reporting the wrong config to whoever is asking
(a health-check endpoint, a debug log, a benchmark harness). `search_config()` reads the single
source of truth the recall path itself resolves against, so it can't drift from it:

```rust
let cfg = mem.search_config();
println!("graph_degree_weight = {}", cfg.graph_degree_weight);
println!("contradiction detection on: {}", mem.contradiction_detection_enabled());
```

### Process-global engine singleton

`kremory::engine()` / `kremory::engine_init()` (re-exported from `core::engine`) are a
process-global handle to the same underlying engine the `Memory` facade wraps — an intentional,
consumer-facing escape hatch (not an internal accident) for advanced setups that need to reach the
engine directly rather than through a `Memory` instance. Most applications never need this; reach
for it only when composing multiple engines or orchestrating above the facade, per this section's
scope.

`kremory::CoreConfig` (a re-export of `core::config::Config`) is the substrate-level config type
these lower-level constructors and the process-global engine consume — see `core::config` for its
fields.

---

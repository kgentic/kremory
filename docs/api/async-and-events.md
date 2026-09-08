# Async patterns and event sinks

## Async patterns

### Fire-and-forget ingest (handle/polling)

By default, `remember` blocks until Phase 2 enrichment is complete. Use `.no_wait()`
when you want Phase 1 committed immediately and Phase 2 to run in the background:

```rust
use std::time::Duration;

// Phase 1 commits synchronously; Phase 2 queued in background
let commit: kremory::EpisodeCommit = mem.remember("Meeting notes...")
    .in_namespace(ns)
    .no_wait()    // returns after Phase 1 only
    .await?;

// Poll Phase 2 status
let status: kremory::IngestStatus = mem.status_of(&commit).await?;
// IngestStatus: Queued | Running | Succeeded | Failed | Cancelled

// Block on this specific handle with timeout
let final_status = mem.await_enrichment(&commit, Duration::from_secs(30)).await?;

// Cancel if not yet terminal
let outcome: kremory::CancelOutcome = mem.cancel(&commit).await?;
// CancelOutcome { phase: CancelledPhase::Phase2, ... }
```

### Batch status

```rust
// Block until all episodes in a batch reach terminal status
let batch_status: kremory::BatchStatus = mem.await_batch("import-2026-05-27", Duration::from_secs(60)).await?;
```

`await_batch` (and `await_dream`, `await_enrichment`) share one internal options type,
`AwaitOpts { timeout, poll_interval }`, built from the `Duration` argument you pass — not
something you construct yourself at the facade layer, but the name to grep for if you're reading
the substrate-level `await_*` free functions in `kremory::memory`.

### Explicit await-enrichment form

```rust
// Equivalent to the default blocking behaviour — useful when you want to be explicit
let commit = mem.remember("data")
    .await_enrichment()
    .await?;
```

### Background ingestor (OS-thread pipeline, ADR-051)

`Memory::send_batched(text, batch_id)` is a lighter-weight alternative to `remember(...)` for
high-throughput batch ingest. When a sink is configured on the builder via `.with_event_sink(...)`,
`send_batched` routes through a `BackgroundIngestor` — a dedicated OS thread (not a `tokio::spawn`
task) that drains a work queue and fires `on_batch_phase2_complete` on the configured sink when the
batch reaches terminal state. With no sink configured, it routes through the ordinary tokio-spawn
path instead, and the callback never fires (there is no listener to receive it).

```rust
mem.send_batched("Meeting notes...".to_string(), "batch-42".to_string()).await?;
```

Requires a `default_namespace` on the builder — `send_batched` has no per-call namespace override
(use `remember_batch().with_batch_id(...)` if you need per-entry namespace control). The
`BackgroundIngestor` internals (`IngestorConfig` — channel capacities, worker thread name;
`IngestGuard` — explicit shutdown handle; `IngestError` / `IngestErrorKind` — the error-feedback
channel's failure record and its coarse category; `IngestSendError` — enqueue-time failure) are
public but mainly relevant if you are tuning queue capacity or building your own supervision
around the worker thread — the defaults are sensible for most consumers.

---

## Event sinks

Implement `IngestEventSink` + `EnrichmentEventSink` to receive push notifications
during ingest and dream phases.

```rust
use kremory::{EnrichmentEventSink, IngestEventSink, ContradictionDetected, BatchPhase2Complete,
              IngestStatus, IngestionError, OnEdgeAddedParams};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct LiveDashboardSink {
    entity_count: Arc<AtomicUsize>,
    contradiction_count: Arc<AtomicUsize>,
}

impl IngestEventSink for LiveDashboardSink {
    fn on_entity_extracted(&self, _id: &str, _name: &str) {
        self.entity_count.fetch_add(1, Ordering::Relaxed);
    }
    fn on_edge_added(&self, _params: OnEdgeAddedParams<'_>) {}
    fn on_contradiction(&self, _e: ContradictionDetected) {
        self.contradiction_count.fetch_add(1, Ordering::Relaxed);
    }
    fn on_dedup_merge(&self, _surviving: &str, _absorbed: &str) {}
    fn on_stage_change(&self, _status: IngestStatus) {}
    fn on_ingestion_error(&self, _e: IngestionError) {}
}

impl EnrichmentEventSink for LiveDashboardSink {
    fn on_community_updated(&self, _id: &str, _count: usize) {}
    fn on_batch_phase2_complete(&self, _e: BatchPhase2Complete) {}
}
```

### Memory-level sink (applies to all operations)

```rust
let sink = Arc::new(LiveDashboardSink {
    entity_count: Arc::new(AtomicUsize::new(0)),
    contradiction_count: Arc::new(AtomicUsize::new(0)),
});

let mem = Memory::open("./agent.db")
    .with_llm(llm)
    .with_embedder(emb)
    .with_event_sink(sink.clone())   // default sink for all subsequent ops
    .await?;

// Both of these fire sink callbacks during enrichment
mem.remember("Event A").await?;
mem.remember("Event B").await?;
```

### Per-call override

`.with_event_sink(...)` on a single `remember(...)` call overrides the Memory-level default for
just that call — e.g. wire in your own audit-log sink (implementing `EnrichmentEventSink`, the
same trait `LiveDashboardSink` implements above) for one sensitive operation without touching the
default sink every other call uses:

```rust
// This call uses a one-off custom sink, overriding the Memory-level default —
// `MySink` here stands in for your own EnrichmentEventSink implementation
// (e.g. one that appends every event to an audit-log JSONL file).
mem.remember("sensitive operation")
    .with_event_sink(Arc::new(MySink))
    .await?;

// All other calls still use the Memory-level default sink
mem.remember("normal operation").await?;
```

---

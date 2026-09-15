# Dream phase + consolidation


The dream phase is the "settle the graph" pass. Call it periodically (e.g. a daily cron, or after
a bulk import) to keep recall quality high as the knowledge graph grows. A single `mem.dream()`
runs two sub-phases:

1. **Reconciliation** — per-entity cleanup: type discovery, alias resolution, reclassification,
   consistency-check, canonicalization (plus opt-in acronym/nickname recall + type-registry
   collapse).
2. **Consolidation** — graph-global cleanup: four ops — community detection, cross-episode entity
   merge, supersession sweep, fact archival.

### Why you need it — what extraction actually leaves behind

Language models do not produce canonical output. One sentence routinely yields the same
fact twice under different casing, plus the occasional nonsense triple. From a single
input, *"Ines runs the harbour pilot service at Lysfjord"*:

```text
Ines works_at lysfjord
ines works_at Lysfjord
```

⚠️ **This is not a small-model artefact.** The example above is from OpenAI's
`gpt-4o-mini`; a local `gemma4:e4b` does the same. Paying more per token buys better
triples, not canonical casing. Alias resolution and cross-episode merge are what
reconcile them, and that is `dream()`.

**It is deliberately conservative.** On a small graph it will report `merged: 0`, because
two mentions differing only in casing with one supporting sentence each is not enough
evidence to justify an irreversible merge. That is correct behaviour, not a failure — a
memory that merges eagerly on thin evidence corrupts itself quietly. Give it a real
corpus and the numbers stop being zero.

Runnable: `cargo run --example dream_on_a_schedule` for running it automatically, and
`cargo run --example agent_memory_with_ollama` to see the raw extraction output it exists
to clean up.

**All ops default ON.** This is safe because every destructive mutation is reversible — see
[Reversibility and deletion](./reversibility.md): you can always SEE what `dream()` changed and UNDO it. Cross-episode merge is the one
exception to "ON = commits" — it defaults to **Shadow** mode (computes + reports merge decisions
but fuses nothing) until you opt into Apply.

```rust
// Default: blocks until consolidation complete (~5–60s depending on corpus + models)
// .execute() is required — dream() is the single most consequential call in
// the API (it commits merges/archival/supersession by default), so it gets
// the same explicit destructive terminal as forget() / undo() / etc.
let summary = mem.dream().execute().await?;

// Reconciliation counts:
println!("types discovered:         {}", summary.types_discovered.len());
println!("entities reclassified:    {}", summary.entities_reclassified);
println!("aliases resolved:         {}", summary.aliases_resolved);
println!("canonicalization merges:  {}", summary.canonicalization_merges);
println!("consistency corrections:  {}", summary.consistency_check_corrected);

// Consolidation counts:
println!("communities updated:      {}", summary.communities_updated);
println!("cross-episode WOULD-merge:{}", summary.cross_episode_would_merge); // decisions
println!("cross-episode MERGED:     {}", summary.cross_episode_merged);      // actual fusions
println!("supersessions recorded:   {}", summary.supersessions_recorded);
println!("facts archived:           {}", summary.facts_archived);
println!("duration_ms:              {}", summary.duration_ms);
```

### Reading `DreamSummary` honestly

`DreamSummary` fields are designed so a zero is unambiguous:

- **`cross_episode_would_merge` vs `cross_episode_merged`** — `would_merge` counts every merge
  *decision* the op reached (both Shadow and Apply); `merged` counts fusions actually *committed*.
  In the default **Shadow** mode `would_merge` can be `> 0` while `merged == 0` ("it would have
  merged N pairs, but is in shadow — fused nothing").
- **`consolidation_ops_ran`** — a `ConsolidationOpsRan { community, cross_episode, archival,
  supersession_sweep }` struct of booleans. A consolidation count of `0` with the matching flag
  `true` reads as "the op ran and found nothing to change", NOT "the op was disabled".
- **`budget_exhausted`** — `true` when the consolidation budget ceiling (token/USD) tripped mid-run
  and at least one op was skipped.

These in-band signals are readable without any metrics recorder.

### Tuning which ops run — `DreamOpts` + `CrossEpisodeMode`

Toggle individual ops via `DreamOpts` (all fields default `true` except where noted):

```rust
use kremory::memory::types::{DreamOpts, CrossEpisodeMode};

let mut opts = DreamOpts::default();
opts.include_community_detection = false;   // skip P4 communities
opts.include_fact_archival = false;         // skip P2 archival
opts.include_supersession_sweep = true;     // keep the supersession window closeout
opts.include_consistency_check = false;     // skip the LLM type-verify pass
opts.max_episodes_per_run = Some(500);      // rate-limit LLM spend on large corpora (default: None)

let summary = mem.dream().with_opts(opts).execute().await?;
```

`DreamOpts` is `#[non_exhaustive]` — build it from `DreamOpts::default()` + field mutation, never a
struct literal.

For the cross-episode merge op, prefer the honest tri-state `CrossEpisodeMode` over toggling the
two coupled raw bools (`include_cross_episode_merges` + `cross_episode_dry_run`):

```rust
// Off     → op does not run
// Shadow  → compute + report merge decisions, fuse nothing (DEFAULT)
// Apply   → compute + commit merges (each fusion reversible via mem.unmerge — see reversibility.md)
let summary = mem.dream()
    .cross_episode(CrossEpisodeMode::Apply)
    .execute()
    .await?;
```

`.cross_episode(mode)` composes with `.with_opts(...)` — it overwrites only the two cross-episode
fields and preserves every other knob.

### Scoped to namespace

```rust
let summary = mem.dream()
    .in_namespace(Namespace::new("tenant-acme"))
    .execute()
    .await?;
```

> **Namespace policy:** `dream()` mutates the graph, so it is rejected on `AppendOnly` namespaces
> with `Err(Error::NamespacePolicyViolation { operation: "dream", .. })`.

### Idempotent batch key

```rust
// Safe to retry — same (namespace, batch_id) pair returns without re-running
let summary = mem.dream()
    .in_namespace(ns)
    .for_batch("daily-2026-05-27")
    .execute()
    .await?;
```

### Fire-and-forget (async handle)

```rust
// Returns DreamHandle immediately — use await_dream to poll
let handle: kremory::DreamHandle = mem.dream()
    .in_namespace(ns)
    .fire_and_forget()
    .await?;

// ... do other work ...

let summary = mem.await_dream(&handle, std::time::Duration::from_secs(120)).await?;
```

`DreamStatus` (`Pending | Processing | Complete | Failed(String)`) is `await_dream`'s result type.

Cancel a fire-and-forget run that hasn't finished yet with `Memory::cancel_dream(&handle)` — same
`CancelOutcome` shape as the ordinary ingest-side `mem.cancel(&commit)` (see [Async patterns and event sinks](./async-and-events.md)):

```rust
let outcome = mem.cancel_dream(&handle).await?;
```

### Periodic scheduling — `DreamSchedule`

Instead of calling `mem.dream()` yourself on a cron, configure a schedule once at construction:

```rust
use kremory::DreamSchedule;
use std::time::Duration;

let mem = Memory::open("./agent.db")
    .with_llm(llm)
    .with_embedder(emb)
    .with_dream_schedule(DreamSchedule::Interval(Duration::from_secs(300)))
    .await?;
```

`DreamSchedule::Off` (default) runs no automatic pass. `DreamSchedule::Interval(d)` re-triggers `d`
after each pass COMPLETES (not wall-clock periodic). `DreamSchedule::EveryNIngests(n)` triggers
after every `n`th successful ingest instead of a time interval. Stop a build-time schedule with
`mem.stop_dream_scheduler().await`, or start an independent one at runtime with
`mem.start_dream_scheduler(schedule) -> DreamSchedulerHandle` (`.with_dream_llm(...)` /
`.with_dream_model_id(...)` set a separate LLM slot for scheduled passes, distinct from the main
ingest/recall LLM).

> `DreamMode` (`Full` / `Light`) is a **reserved, not-yet-wired** enum for a future consolidation
> sub-mode — it does not gate any of the reconciliation passes documented above, which are
> controlled by `DreamOpts` instead. Ignore it until it ships.

---

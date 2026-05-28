# kremory API Reference

> **v0.1.0** — The primary consumer surface is `kremory::Memory`. Substrate free-functions
> (`kremory::memory::submit_episode`, etc.) remain public for advanced users; most applications
> should use the facade described in §1–§9.

---

## §1 — Quickstart

The fastest path to working agent memory. No provider configuration required when environment
variables are set.

```rust
use kremory::{Memory, Namespace};

// Auto-detect provider from environment:
//   $OLLAMA_HOST        → Ollama (llama3.1:8b + nomic-embed-text)
//   $OPENAI_API_KEY     → OpenAI (gpt-4o-mini + text-embedding-3-small)
//   $ANTHROPIC_API_KEY  → Anthropic LLM + deterministic embedder fallback (warns)
//   (none)              → Err(Error::NoProviderConfigured) — helpful message included
let mem = Memory::auto("./agent.db")
    .default_namespace(Namespace::new("user-jim"))
    .await?;

// Ingest — blocks until Phase 2 enrichment done (~500ms typical)
mem.remember("User prefers concise replies").await?;

// Recall — returns prompt-ready context string
let context: String = mem.recall("what does user prefer?").await?;

// Dream (consolidation) — blocks until done (~5–60s depending on corpus)
let summary = mem.dream().await?;
println!("communities updated: {}", summary.communities_updated);

// Forget (GDPR-style delete of everything in the default namespace)
let deleted_count = mem.forget().execute().await?;

// Explicit close (flushes WAL)
mem.close().await?;
```

`Memory` clones cheaply — it wraps an `Arc` internally:

```rust
let mem2 = mem.clone();   // cheap — Arc clone
tokio::spawn(async move { mem2.remember("Background task").await });
```

---

## §2 — Customizing the LLM/embedder

### Tier 1.5 — Named shortcuts

Skip environment detection; use a named provider directly.

```rust
// Ollama at localhost:11434 (default models: llama3.1:8b + nomic-embed-text)
let mem = Memory::with_ollama("./agent.db").await?;

// Ollama at a custom URL (useful for remote GPU machines)
let mem = Memory::with_ollama_at("http://192.168.1.5:11434", "./agent.db").await?;

// OpenAI — requires $OPENAI_API_KEY; uses gpt-4o-mini + text-embedding-3-small
let mem = Memory::with_openai("./agent.db").await?;

// Anthropic — requires $ANTHROPIC_API_KEY; LLM = claude-3-haiku, embedder falls back to
// deterministic sha256 (NOT semantic — warns at runtime; use with_openai for semantic recall)
let mem = Memory::with_anthropic("./agent.db").await?;
```

### Tier 2 — Builder (full control)

```rust
use kremory::{Memory, Namespace, DynEmbeddingProvider};
use std::sync::Arc;

let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))           // Arc<dyn ChatProvider> — required (untracked)
    .with_embedder(Arc::new(my_embedder)) // Arc<dyn DynEmbeddingProvider> — required
    .with_event_sink(Arc::new(MySink))    // Arc<dyn EnrichmentEventSink> — optional
    .default_namespace(Namespace::new("acme-corp"))  // optional
    .await?;
```

The builder is type-state guarded: `.await` on a `MemoryBuilder` without calling both
`.with_llm()` (or `.with_llm_tracked()`) and `.with_embedder()` is a **compile error**, not a runtime error.

```rust
// Compile error — missing .with_embedder()
let mem = Memory::open("./agent.db").with_llm(llm).await?; // ERROR
```

### Tier 2 — Builder with observability (v0.1.2+)

For automatic token + cost + duration metric emission, use `with_llm_tracked` instead of `with_llm`:

```rust
let mem = Memory::open("./agent.db")
    .with_llm_tracked("openai", "gpt-4o-mini", Arc::new(my_llm))  // explicit labels
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

The Tier 1 shortcuts (`with_ollama`, `with_openai`, `with_anthropic`) internally upgrade to `with_llm_tracked` — no opt-in required.

To override the bundled cost rates table:

```rust
let mem = Memory::open("./agent.db")
    .with_provider_rates_path("./my-rates.toml")  // override bundled rates
    .with_llm_tracked("openai", "gpt-4o-mini", Arc::new(my_llm))
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

Full observability surface — emitted metrics, label schema, OTel/OTLP export, cardinality discipline, dashboard examples — documented in [observability.md](observability.md).

---

## §3 — Namespaces + multi-tenancy

`Namespace { namespace: String, thread: Option<String> }` is the multi-tenant primitive.
Entities are **fully isolated per namespace** — no cross-namespace data leakage.

```rust
use kremory::Namespace;

// Simple namespace
let ns = Namespace::new("tenant-acme");

// Namespace + conversation thread
let ns_thread = Namespace::new("tenant-acme").with_thread("support-ticket-1042");
```

### Default namespace (set once at construction)

```rust
let mem = Memory::auto("./agent.db")
    .default_namespace(Namespace::new("user-jim"))
    .await?;

// All subsequent calls use "user-jim" unless overridden
mem.remember("User prefers dark mode").await?;
let ctx = mem.recall("UI preferences").await?;
```

### Per-call namespace override

```rust
// Override default for one specific call
mem.remember("Acme Corp ticket data")
    .in_namespace(Namespace::new("tenant-acme"))
    .await?;
```

### Required namespace (no default set)

If no `default_namespace` is set on `Memory` **and** `.in_namespace()` is not called,
the terminal `.await?` returns `Err(Error::MissingNamespace { request: "remember" })`.
This is a compile-time-visible design choice — the error type is named and matchable:

```rust
match mem.remember("data").await {
    Ok(commit) => { /* ... */ }
    Err(kremory::CoreError::MissingNamespace { request }) => {
        eprintln!("must call .in_namespace() or set a default_namespace for {request}");
    }
    Err(e) => return Err(e.into()),
}
```

### Multi-tenant SaaS pattern

```rust
// One Memory per process, many namespaces:
let mem = Memory::open("./shared.db")
    .with_llm(llm)
    .with_embedder(emb)
    .await?;

for tenant in &["acme", "globex", "initech"] {
    mem.remember(format!("Tenant {} onboarded", tenant))
        .in_namespace(Namespace::new(tenant))
        .await?;
}

// Each tenant's recall is namespace-scoped — no cross-tenant leakage
let ctx = mem.recall("onboarding status")
    .in_namespace(Namespace::new("acme"))
    .await?;
```

### Namespace policies (v0.1.4+, ADR-029a)

Per-namespace policy controls let callers DECLARE compliance intent (audit-grade,
non-forgettable, non-dream-eligible). v0.1.4 **PERSISTS** the declaration and
emits operational warnings, but does **NOT ENFORCE** the policy on
`dream()` / `forget()` / mutation operations — enforcement lands in v0.1.5+
per ADR-029b. Use the v0.1.4 window to capture intent + validate consumer
ergonomics ahead of enforcement.

```rust
use kremory::{ImmutabilityLevel, Memory, Namespace, NamespacePolicy};

let mem = Memory::open("./shared.db")
    .with_llm(llm)
    .with_embedder(emb)
    .await?;

// Canonical audit-grade preset: AppendOnly + non-forgettable + non-dream-eligible.
let audit = Namespace::new("compliance-log")
    .with_policy(NamespacePolicy::APPEND_ONLY)?;
mem.register_namespace(audit).await?;

// Or build a custom policy via fluent setters.
let custom = NamespacePolicy::new()
    .with_immutability(ImmutabilityLevel::Mutable)
    .with_forgettable(false)
    .with_dream_eligible(true);
let ns = Namespace::new("never-forget").with_policy(custom)?;
mem.register_namespace(ns).await?;
```

**Idempotency + immutability**: calling `register_namespace` with the SAME
policy is `Ok(())` (safe for startup-code re-execution). Calling with a
DIFFERENT policy on an existing namespace surfaces
`Err(MemoryError::Core(Error::NamespacePolicyImmutable { stored, attempted }))`.

**Lazy population**: namespaces observed via the first `remember()` / `recall()`
/ `forget()` / `dream()` call get a default-policy row written automatically.
Call `register_namespace` explicitly at startup for namespaces that need a
non-default policy — there is no retroactive upgrade at v0.1.4.

**Operational visibility**: every non-default policy registration emits
`tracing::warn!` on target `kremory.namespace` with the marker
`POLICY DECLARED BUT NOT ENFORCED`. Default-policy registrations are silent.

---

## §4 — Ingest

### Basic ingest

```rust
// Default: blocks until Phase 2 enrichment done (LLM entity/edge extraction)
let commit: kremory::EpisodeCommit = mem.remember("User scheduled meeting at 2pm").await?;
```

### Source metadata

```rust
use kremory::SourceKind;
use chrono::Utc;

mem.remember("Alice decided the team will use async channels")
    .from_chat("session-42")          // SourceKind::Chat
    .in_namespace(Namespace::new("team-alice"))
    .await?;

mem.remember("Q4 revenue target is $2M")
    .from_document("q4-plan-v2.pdf")  // SourceKind::Document
    .in_namespace(ns)
    .published_at(Utc::now())         // bi-temporal anchor: sets valid_from precedence
    .await?;

mem.remember("Support ticket #1042 opened")
    .from_note("ticket-1042")         // SourceKind::Note
    .in_namespace(ns)
    .await?;

// Full escape hatch
mem.remember("Custom source")
    .from_source("my-id", SourceKind::Document)
    .await?;
```

### Pre-extracted facts (skip Phase 2 LLM)

If you've already extracted structured facts, pass them directly — kremory skips Phase 2
LLM enrichment:

```rust
use kremory::StructuredFact;

mem.remember("Alice is the CEO of Acme Corp")
    .with_facts(vec![
        StructuredFact {
            subject: "Alice".into(),
            predicate: "is_ceo_of".into(),
            object: "Acme Corp".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        },
    ])
    .await?;
```

### Batch ingest

```rust
let commits: Vec<kremory::EpisodeCommit> = mem.remember_batch()
    .add("Meeting at 2pm")
        .from_chat("session-42")
        .in_namespace(Namespace::new("user-jim"))
    .add("Alice prefers async Rust")
        .from_note("note-7")
        .in_namespace(Namespace::new("user-jim"))
    .with_batch_id("import-2026-05-27")  // idempotent — safe to retry
    .await?;
```

---

## §5 — Recall

### Default (prompt-ready string)

```rust
let context: String = mem.recall("what does the user prefer?").await?;
// Ready to prepend to your LLM system prompt
```

### Namespace + top-k

```rust
let context = mem.recall("login issues this quarter")
    .in_namespace(Namespace::new("acme-corp").with_thread("support-q4"))
    .k(20)   // top-k clamp (default: 10)
    .await?;
```

### Recall templates

```rust
use kremory::RecallTemplate;

// Default — temporal facts with timestamps
let ctx = mem.recall("architecture decisions").await?;

// Entity-focused — name + description pairs
let ctx = mem.recall("team members")
    .as_template(RecallTemplate::Entities)
    .await?;

// Edge-focused — relationship graph summary
let ctx = mem.recall("org structure")
    .as_template(RecallTemplate::EdgeSummary)
    .await?;
```

### Raw results (for custom rendering)

```rust
use kremory::RetrievedContext;

let results: Vec<RetrievedContext> = mem.recall("preferences")
    .in_namespace(ns)
    .raw()
    .await?;

for r in &results {
    println!("{}: {:.3}", r.content, r.score);
}
```

### `as_of` (bi-temporal filtering)

```rust
use chrono::{Utc, Duration};

// v0.1.0: emits tracing::warn — filter wiring lands in v0.1.1
// Caller code is forward-compatible — same syntax works in v0.1.1
let ctx = mem.recall("what was the policy last week?")
    .as_of(Utc::now() - Duration::days(7))
    .await?;
```

---

## §6 — Dream phase + consolidation

The dream phase runs community detection + graph consolidation over all episodes
in a namespace. Call it periodically (e.g., daily cron, after a bulk import) to
keep recall quality high as the knowledge graph grows.

```rust
// Default: blocks until consolidation complete (~5–60s depending on corpus size)
let summary = mem.dream().await?;
println!("episodes processed: {}", summary.episodes_processed);
println!("communities updated: {}", summary.communities_updated);
println!("edges merged: {}", summary.edges_merged);
```

### Scoped to namespace

```rust
let summary = mem.dream()
    .in_namespace(Namespace::new("tenant-acme"))
    .await?;
```

### Idempotent batch key

```rust
// Safe to retry — same (namespace, batch_id) pair returns without re-running
let summary = mem.dream()
    .in_namespace(ns)
    .for_batch("daily-2026-05-27")
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

---

## §7 — Forget (GDPR)

`forget()` returns a builder; the destructive operation only fires on `.execute()`.
This explicit terminal makes the intent visible in code review.

```rust
// Forget everything in the default namespace
let deleted: u64 = mem.forget().execute().await?;
println!("{deleted} records deleted");

// Forget a specific namespace
let deleted = mem.forget()
    .in_namespace(Namespace::new("tenant-acme"))
    .execute()
    .await?;
```

---

## §8 — Async patterns

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
let batch_status = mem.await_batch("import-2026-05-27", Duration::from_secs(60)).await?;
```

### Explicit await-enrichment form

```rust
// Equivalent to the default blocking behaviour — useful when you want to be explicit
let commit = mem.remember("data")
    .await_enrichment()
    .await?;
```

---

## §9 — Event sinks

Implement `IngestEventSink` + `EnrichmentEventSink` to receive push notifications
during ingest and dream phases.

```rust
use kremory::{EnrichmentEventSink, IngestEventSink, ContradictionDetected, BatchPhase2Complete,
              IngestStatus, IngestionError};
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
    fn on_edge_added(&self, _from: &str, _to: &str, _predicate: &str) {}
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
let sink = Arc::new(LiveDashboardSink { /* ... */ });

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

```rust
// This call uses a one-off audit sink, overriding the Memory-level default
mem.remember("sensitive operation")
    .with_event_sink(Arc::new(AuditSink::new("audit-log.jsonl")))
    .await?;

// All other calls still use the Memory-level default sink
mem.remember("normal operation").await?;
```

---

## §10 — Advanced — substrate composition

For users who need full control: custom graph backends, multi-engine setups, direct
vector index manipulation, or cross-tenant orchestration above the facade.

```rust
use kremory::memory::{submit_episode, submit_dream_phase, await_enrichment, search,
                       context_block, GraphHandle, Namespace, SourceRef, SourceKind,
                       SubmitOpts, SearchOpts, ContextTemplate};
use std::sync::Arc;

// You supply the graph handle (e.g. kremory::memory::TemporalGraph or your own impl)
// and manage LLM + embedder Arcs directly.
let commit = submit_episode(
    graph.as_ref(),
    "Alice prefers async Rust",
    SourceRef {
        kind: SourceKind::Chat,
        id: "session-42".into(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    },
    vec![],             // structured_facts (empty = run Phase 2 LLM extraction)
    provider.clone(),   // Arc<dyn ChatProvider>
    Namespace::new("user-alice"),
    None,               // batch_id
    SubmitOpts::default(),
    None,               // event sink
).await?;

// Hybrid retrieval
let results = search(
    graph.as_ref(),
    &Namespace::new("user-alice"),
    "rust preferences",
    SearchOpts { limit: Some(10), ..Default::default() },
).await?;

// Render as prompt-ready string
let ctx = context_block(&results, ContextTemplate::TemporalFacts);
```

### Custom `GraphHandle` backend

Implement `GraphHandle` to plug kremory into a custom storage backend:

```rust
use kremory::GraphHandle;

struct MyGraphBackend { /* ... */ }

#[async_trait::async_trait]
impl GraphHandle for MyGraphBackend {
    // implement all required methods
}

// Then pass it to Memory::open via a lower-level constructor (advanced)
```

---

## §11 — Bi-temporal model

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

The two clocks enable: *"What did the agent know as of real-world date X, as recorded before system time Y?"*

```sql
-- kremory issues this internally for as_of queries (v0.1.1):
SELECT * FROM facts
WHERE recorded_at <= :system_time_Y
  AND valid_from  <= :real_world_time_X
  AND (valid_to IS NULL OR valid_to > :real_world_time_X)
```

Up through v0.1.3, `.as_of()` on `RecallRequest` is accepted and emits `tracing::warn` — the
filter wiring to SQL lands in a later release. Caller code is forward-compatible: same API,
no migration required.

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

## §12 — Migration guide

### v0.0.x → v0.1.0

No existing consumers to migrate — v0.1.0 is the first public release.

If you used an internal snapshot (pre-tag), the changes are:

| Before | After |
|---|---|
| `WorkspaceScope::new("id")` | `Namespace::new("id")` |
| `WorkspaceScope::with_thread("id", "t")` | `Namespace::new("id").with_thread("t")` |
| `MemoryHandle::open(path, emb)` | `Memory::auto(path).await?` or `Memory::open(path).with_llm(...).with_embedder(...).await?` |
| `handle.submit_episode(...)` (9-arg) | `mem.remember(content).from_chat(id).in_namespace(ns).await?` |
| `handle.context_block(&scope, query)` | `mem.recall(query).in_namespace(ns).await?` |

The substrate free-functions (`kremory::memory::submit_episode`, etc.) remain public and
unchanged — if you depend on them directly, no migration is needed.

### v0.1.0 → v0.1.1 (shipped 2026-05-27)

Recall pipeline redesign. No public API changes; substrate-level behavior fixes only.

| Area | Change |
|---|---|
| RRF (Reciprocal Rank Fusion) | Constant `k = 60` now used for hybrid score blending |
| Episodic edges | Now authoritative; deprecated fallback paths removed |
| LightRAG stub-entity | Forward references insert stub entities at first mention; promoted on re-ingestion |
| First-mention snippet | `source_ref` carries first-encountered snippet, not most recent |
| `SourceKind::Episode` | Replaces deprecated `SourceKind::Document` |

No migration required for facade-tier consumers.

### v0.1.1 → v0.1.2 (shipped 2026-05-28)

LLM observability parity. Additive only — no breaking changes.

| API | Change |
|---|---|
| `MemoryBuilder::with_llm_tracked(provider, model, llm)` | **NEW** — wraps in `TokenTrackingChatProvider` with explicit labels |
| `MemoryBuilder::with_provider_rates_path(PathBuf)` | **NEW** — override bundled cost-rates table |
| `MemoryBuilder::with_llm(llm)` | Unchanged — backward compatible (untracked path) |
| Tier 1 shortcuts (`with_ollama` etc.) | Internally upgraded to `with_llm_tracked` — auto-emit metrics |
| `kremory::observability` module | **NEW** — re-exports `TokenTrackingChatProvider`, `ProviderRates`, `TelemetryConfig`, `init_telemetry`, etc. |
| `init_telemetry(TelemetryConfig)` | Real implementation (was stub). Returns `TelemetryHandle`. Behind `otel` cargo feature. |
| `otel` cargo feature | Off by default. When enabled: installs `tracing-subscriber` + `tracing-opentelemetry` + OTLP gRPC exporter |
| `error.type` span attribute | **NEW** — bounded to `{server_error, client_error, parse_error}`; tracking span attribute (not histogram label) |
| `kremory_core_tokens_total` counter | Extended with `operation="chat"` emissions (embed emissions unchanged) |
| `kremory_core_cost_usd_total` gauge | **NEW** — cumulative USD cost, f64 |
| `kremory_core_chat_duration_seconds` histogram | **NEW** — per-call duration, `status` label bounded to `ok`/`error` |

See [observability.md](observability.md) for full surface.

### v0.1.2 → v0.1.3 (shipped 2026-05-28)

Hygiene only. No public API changes. `cargo clippy --all-targets --all-features` now passes clean.

### Future — v0.1.x → v0.2.0 (planned)

Will introduce cargo features for substrate-only consumers (per [ADR-028](../.ai-docs/adrs/rql/adr-028-defer-crate-split-cargo-features-2026-05-28.md)). Default-feature consumers continue working without changes. Substrate-only consumers will opt in via:

```toml
kremory = { version = "0.2", default-features = false, features = ["substrate"] }
```

Full migration guide will land alongside v0.2.0 ship.

---

*API reference current as of kremory v0.1.3 (2026-05-28). Facade design: ADR-027 (outside-in API design). Temporal model: ADR-003. BYOM contract: ADR-002. Crate topology: ADR-028 (supersedes ADR-007 + ADR-008).*

# kremory API Reference

> **v0.6.0** — The primary consumer surface is `kremory::Memory`. Substrate free-functions
> (`kremory::memory::submit_episode`, etc.) remain public for advanced users; most applications
> should use the facade described below. Since v0.1.3 the facade gained a fully-wired dream
> consolidation phase (§6), reversible graph mutations with a see/undo surface (§6a), opt-in
> BM25/FTS5 content recall (§5, `content-search` — a DEFAULT feature since ADR-078), and a feature-flag matrix
> (§13). The Node/napi binding mirrors the surface in camelCase (§14).

---

## §1 — Quickstart

The fastest path to working agent memory. No provider configuration required when environment
variables are set.

```rust
use kremory::{Memory, Namespace};

// Auto-detect provider from environment:
//   $OLLAMA_HOST        → Ollama (gemma4:e4b, reasoning disabled + nomic-embed-text)
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
// Ollama at localhost:11434 (default models: gemma4:e4b with reasoning disabled + nomic-embed-text)
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
`.with_llm()` (optionally followed by `.with_token_tracking()`) and `.with_embedder()` is a **compile error**, not a runtime error.

```rust
// Compile error — missing .with_embedder()
let mem = Memory::open("./agent.db").with_llm(llm).await?; // ERROR
```

### Tier 2 — Builder with observability (v0.1.2+)

For automatic token + cost + duration metric emission, add `.with_token_tracking(provider, model)` after `.with_llm(...)`:

```rust
let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))
    .with_token_tracking("openai", "gpt-4o-mini")  // explicit labels
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

The Tier 1 shortcuts (`with_ollama`, `with_openai`, `with_anthropic`) apply token tracking internally — no opt-in required.

To override the bundled cost rates table:

```rust
let mem = Memory::open("./agent.db")
    .with_provider_rates_path("./my-rates.toml")  // override bundled rates
    .with_llm(Arc::new(my_llm))
    .with_token_tracking("openai", "gpt-4o-mini")
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

### Content search — BM25/FTS5 over raw episode text (opt-in)

The default recall surface (`.await` / `.raw()` / `.as_template(...)`) searches the **knowledge
graph** (entities + facts, hybrid vector + keyword). A separate terminal, `.content()`, runs a
**BM25/FTS5 full-text search over the raw `episodes.content`** — the verbatim ingested text, not
the extracted graph. It is a sibling of `.raw()` and returns `Vec<ContentPassage>`.

`.content()` is gated behind the **`content-search`** cargo feature, which is **ON by default**
since ADR-078 (2026-07-28) — so the terminal and the `ContentPassage` type exist in a default build.
If you have disabled default features, re-enable it explicitly:

```toml
kremory = { version = "0.6", default-features = false, features = ["content-search"] }
```

⚠️ Disabling it does **not** just remove `.content()` — it also removes the BM25 content arm and the
dense episode arm from the ordinary `recall()` path, measured at **−32.2pt** ex-adversarial
substring recall (ADR-078).

```rust
use kremory::memory::types::ContentPassage;

let passages: Vec<ContentPassage> = mem.recall("async channels decision")
    .in_namespace(Namespace::new("team-alice"))
    .k(10)              // FTS row limit (default 10)
    .content()          // BM25-only terminal — requires the `content-search` feature
    .await?;

for p in &passages {
    // p.episode_id : i64        — the source episodes.id
    // p.snippet    : String     — FTS5 snippet() extract around the match (not the full body)
    // p.score      : f32        — BM25 rank; LOWER = more relevant (FTS convention)
    // p.source_ref : SourceRef  — kind == SourceKind::Episode; occurred_at = episode timestamp
    println!("[{:.3}] episode {}: {}", p.score, p.episode_id, p.snippet);
}
```

**When to reach for `.content()` instead of `.raw()`** — measured 2026-07-28 (TD-151), conv0, same
corpus, dense arm on, no reranker:

| terminal | what it queries | ex-adversarial substring recall | latency (mean) |
|---|---|---|---|
| `.content()` | BM25 + dense over raw episode text | **93.4%** | **12 ms** |
| `.raw()` | the above, RRF-fused with the entity/fact graph | 96.7% | 112 ms |

`.content()` gets **~97% of the accuracy for ~11% of the read cost**. If you are latency-bound and
your consumer only needs passages to read (not entities, facts, or bi-temporal answers), it is very
often the better trade — and a better one than any cross-encoder configuration, which buys accuracy
in the opposite direction (the cheapest reranker arm measured costs 0.44 s/query). Absolutes were
taken under load and are provisional; the ~9× ratio is not (ratios survive contention).

Notes (ADR-072 seq1):

- **BM25-only** — content passages are a distinct, un-fused stream. They are NOT blended into the
  entity/fact RRF ranking; `.content()` does not extend the graph-shaped `RetrievedContext`.
- **Single-namespace only** — use `.in_namespace(...)`. Multi-namespace fan-out (`.in_namespaces`)
  is not yet supported on this terminal and returns `Err`.
- Requires a `Memory` built via the builder/providers path (same as `.forget()` / the
  `filter_metadata` post-filter).

### §5.1 — Recipe: session expansion (right conversation, wrong turn)

A measured failure mode worth knowing about: on our LoCoMo miss-set, **50% of the evidence turns we
fail to retrieve sit in a session we ALREADY hit** — retrieval finds the right conversation and
returns the wrong turn. Widening `k` does not fix this reliably;
pulling the *rest of the hit's source* does.

kremory has no built-in "session" concept, and does not need one — the behaviour composes from
primitives that already ship. (The same `source_id` you set here also drives **write**-side
prior-turn replay — see §5.2. Threading a conversation buys both halves at once.)

```rust
// 1. At ingest, scope the source id to the unit you want to expand to.
//    Anything works: a chat session, a document, a ticket, a meeting.
//    `.from_chat(id)` / `.from_document(id)` / `.from_note(id)` are shortcuts
//    for `.from_source(id, kind)`.
mem.remember(turn_text)
    .from_chat("conv-26/session-3")
    .in_namespace(ns.clone())
    .await?;

// 2. At recall, read which source each hit came from.
let hits = mem.recall(question).in_namespace(ns.clone()).k(10).raw().await?;
let sources: HashSet<&str> = hits.iter()
    .flat_map(|h| h.source_refs.iter())
    .map(|s| s.id.as_str())
    .collect();

// 3. Pull the whole source for each hit and hand the union to your model.
for source_id in sources {
    let episodes = mem.recall_by_source_id(source_id, Some(ns.clone())).await?;
    // ... append to context, subject to your token budget
}
```

Cost: one extra indexed lookup per distinct source in the top-k — no embedding, no LLM, no second
ranking pass. The trade-off is **context size**, not latency: you are choosing to spend tokens
rather than retrieval quality, so cap the expansion (most-relevant source only, or a token budget).

Mirrored in the Node binding as `memory.recallBySourceId(sourceId, namespace)`.

---

### §5.2 — Prior-turn replay: references that resolve across turns

The same `source_id` you set for session expansion also does work at **write** time.

When you tag consecutive turns with one id, kremory shows the extractor the preceding turns of
that conversation, so anaphoric references resolve:

```rust
// Turn 1 — a concrete fact.
mem.remember("Bob: the blue jacket on the left is the one I want")
    .from_chat("conv-42")
    .in_namespace(ns.clone())
    .await?;

// Turn 2 — meaningless on its own. With the preceding turn in view, the
// extractor can resolve "that one" to the blue jacket.
mem.remember("Alice: I prefer that one too")
    .from_chat("conv-42")
    .in_namespace(ns.clone())
    .await?;
```

There is no flag to set. Replay fires when — and only when — you thread a conversation:

- **Tagged** (`.from_chat(id)` / `.from_document(id)` / `.from_source(id, kind)`) → the previous
  episodes sharing that id, within the same namespace, are replayed as context.
- **Untagged** → every `remember()` gets a fresh uuid, nothing matches, nothing changes.

Details that matter in practice:

| | |
|---|---|
| **Depth** | 10 preceding episodes. Both mem0 and Graphiti independently converged on 10. |
| **Order** | Oldest-first, by insertion order — not by `published_at`, which is a world clock you may legitimately set out of order. |
| **Scope** | Same `source_id` **and** same namespace. Never crosses either. |
| **Not extracted** | Replayed turns are context only. Facts are extracted from the current turn alone, so an early turn is not re-extracted on every later one. |
| **Bounded** | The replayed block has a character budget; if it is exceeded, the OLDEST turns are dropped first. |
| **Off switch** | `MemoryBuilder::prior_turn_replay_depth(0)`. |

```rust
// Disable, or change the depth:
let mem = Memory::builder()
    .prior_turn_replay_depth(0)   // 0 = off; default 10
    // ...
    .build().await?;
```

Threading a conversation therefore buys you both halves at once: **write**-side reference
resolution (this section) and **read**-side session expansion (§5.1).

See `.ai-docs/adrs/adr-080-prior-turn-replay-into-extraction-2026-09-06.md`.

---

## §6 — Dream phase + consolidation

The dream phase is the "settle the graph" pass. Call it periodically (e.g. a daily cron, or after
a bulk import) to keep recall quality high as the knowledge graph grows. A single `mem.dream()`
runs two sub-phases:

1. **Reconciliation** — per-entity cleanup: type discovery, alias resolution, reclassification,
   consistency-check, canonicalization (plus opt-in acronym/nickname recall + type-registry
   collapse).
2. **Consolidation** — graph-global cleanup: four ops — community detection, cross-episode entity
   merge, supersession sweep, fact archival.

**All ops default ON.** This is safe because every destructive mutation is reversible (ADR-073 —
see §6a): you can always SEE what `dream()` changed and UNDO it. Cross-episode merge is the one
exception to "ON = commits" — it defaults to **Shadow** mode (computes + reports merge decisions
but fuses nothing) until you opt into Apply.

```rust
// Default: blocks until consolidation complete (~5–60s depending on corpus + models)
let summary = mem.dream().await?;

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

let opts = DreamOpts {
    include_community_detection: false,   // skip P4 communities
    include_fact_archival: false,         // skip P2 archival
    include_supersession_sweep: true,     // keep the supersession window closeout
    include_consistency_check: false,     // skip the LLM type-verify pass
    max_episodes_per_run: Some(500),      // rate-limit LLM spend on large corpora (default: None)
    ..DreamOpts::default()
};

let summary = mem.dream().with_opts(opts).await?;
```

`DreamOpts` is `#[non_exhaustive]` — build it from `DreamOpts::default()` + field mutation, never a
struct literal.

For the cross-episode merge op, prefer the honest tri-state `CrossEpisodeMode` over toggling the
two coupled raw bools (`include_cross_episode_merges` + `cross_episode_dry_run`):

```rust
// Off     → op does not run
// Shadow  → compute + report merge decisions, fuse nothing (DEFAULT)
// Apply   → compute + commit merges (each fusion reversible via mem.unmerge — §6a)
let summary = mem.dream()
    .cross_episode(CrossEpisodeMode::Apply)
    .await?;
```

`.cross_episode(mode)` composes with `.with_opts(...)` — it overwrites only the two cross-episode
fields and preserves every other knob.

### Scoped to namespace

```rust
let summary = mem.dream()
    .in_namespace(Namespace::new("tenant-acme"))
    .await?;
```

> **Namespace policy:** `dream()` mutates the graph, so it is rejected on `AppendOnly` namespaces
> with `Err(Error::NamespacePolicyViolation { operation: "dream", .. })` (ADR-029b enforcement).

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

## §6a — Reversibility — see, trust, undo

Because `dream()` mutates the graph by default, kremory ships a **reversibility substrate**
(ADR-073): every logged destructive mutation can be inspected and undone. Undo is deterministic
(replayed from an in-transaction snapshot — no LLM), idempotent (a second undo is a zero-count
no-op, never a double-restore), and returns an **honest outcome** (the actual counts reversed,
never a bare "ok").

### SEE — inspect what changed

```rust
use kremory::{MutationRecord, MutationKind};

// Every logged mutation that touched one entity (newest-first, includes already-undone):
let history: Vec<MutationRecord> = mem.mutation_history("alice j")
    .in_namespace(Namespace::new("agent"))
    .await?;

for r in &history {
    // r.mutation_id: i64, r.kind: MutationKind, r.created_at: String (RFC3339),
    // r.undone: bool, r.group_id: String, r.affected_entities: Vec<String>, r.summary: String
    println!("#{} [{}] {}", r.mutation_id, if r.undone { "undone" } else { "live" }, r.summary);
}

// Or list a whole namespace's mutations, newest-first (LIVE / still-reversible by default):
let all: Vec<MutationRecord> = mem.list_mutations()
    .in_namespace(Namespace::new("agent"))
    .kind(MutationKind::EntityMerge)   // optional: filter to one kind
    .since(chrono::Utc::now() - chrono::Duration::days(1)) // optional: created_at lower bound
    .include_undone(true)              // optional: also show reversed mutations (default: false)
    .await?;
```

`list_mutations()` without `.in_namespace(...)` falls back to the `default_namespace`, else scans
**all** namespaces (a valid admin view). Both inspect requests are read-only — `.await` them.

### UNDO — the unified dispatcher + per-kind methods

The recommended entry point is `mem.undo(mutation_id)`: it reads the mutation's kind and dispatches
to the correct reversal, so you can iterate `list_mutations()` and undo uniformly without switching
on the kind by hand.

```rust
use kremory::Namespace;

if let Some(rec) = history.first() {
    // Optional .in_namespace(ns) GUARDS the undo to the mutation's original namespace.
    let outcome = mem.undo(rec.mutation_id).execute().await?;
    println!("reversed: {outcome:?}");   // UndoOutcome::{Unmerge|EditEntity|DeleteEntity|DeleteFact}(..)
}
```

`UndoOutcome` is a `#[non_exhaustive]` enum — one variant per log-dispatchable kind, each wrapping
that kind's honest per-op outcome. The per-kind methods remain available as the escape hatch when
you already know the kind:

| Method | Reverses | Argument | Honest outcome |
|---|---|---|---|
| `mem.undo(mutation_id)` | **any** logged mutation | `mutation_id` | `UndoOutcome` |
| `mem.unmerge(mutation_id)` | an `entity_merge` | `mutation_id` | `UnmergeOutcome` |
| `mem.undo_entity_edit(mutation_id)` | an `edit_entity` (rename/retype) | `mutation_id` | `EditEntityOutcome` |
| `mem.undo_delete_entity(mutation_id)` | a `delete_entity` | `mutation_id` | `DeleteEntityOutcome` |
| `mem.undo_delete_fact(mutation_id)` | a `delete_fact` | `mutation_id` | `DeleteFactOutcome` |
| `mem.unsupersede(fact_id)` | a supersession bound | **`fact_id`** | `UnsupersedeOutcome` |
| `mem.restore_archived_fact(archived_fact_id)` | a P2 fact archive | **`archived_fact_id`** | `RestoreArchivedOutcome` |

Each honest-outcome struct carries the actual counts reversed and an idempotency flag — e.g.
`UnmergeOutcome { restored_entity, keeper, facts_repointed, edges_restored, entities_reopened,
nogood_recorded, already_undone }`.

### The `merge_nogood` guarantee

`unmerge` (and `undo()` on an `entity_merge`) does more than split the pair back apart: it records
a **merge NOGOOD** for the split pair (`UnmergeOutcome.nogood_recorded == true`), so the **next
`dream()` will not re-merge them**. Undoing a merge is durable, not a one-cycle reprieve.

### The 4-of-8 tracked-kind boundary

There are eight `MutationKind` variants, but only **four are logged today and therefore reversible
via `undo()`**: `EntityMerge`, `EntityEdit`, `EntityDelete`, `FactDelete`. The other four
(`FactSupersede`, `FactArchive`, `CommunityAssign`, `CanonicalForm`) are **reserved** — not yet
produced into the mutation log — so:

- `list_mutations().kind(<a reserved kind>)` returns **empty by construction** (not "nothing
  changed").
- `undo(id)` on a would-be row of a reserved kind returns a loud `Error::UndoUnsupportedKind`.

`FactSupersede` / `FactArchive` are themselves reversible — but through their **domain-id** methods
`unsupersede(fact_id)` / `restore_archived_fact(archived_fact_id)`, not the `mutation_id`-based
`undo()` surface.

### Direct mutations (also reversible)

Beyond `dream()`, the same reversible substrate backs the direct consumer mutations:

```rust
// Rename a diarization placeholder — every fact / edge / membership re-points to the new id;
// undoable via the returned mutation_id.
let edit = mem.edit_entity("Speaker 1")
    .rename("alice")                      // XOR .retype(type_id)
    .in_namespace(Namespace::new("meeting"))
    .execute()
    .await?;
mem.undo_entity_edit(edit.mutation_id).execute().await?;

// Reversible delete — facts are ARCHIVED (recoverable), never hard-deleted.
let del = mem.delete_entity("bob").in_namespace(ns.clone()).execute().await?;
mem.undo_delete_entity(del.mutation_id).execute().await?;

let dfact = mem.delete_fact(fact_id).execute().await?;   // fact_id is global
mem.undo_delete_fact(dfact.mutation_id).execute().await?;

// Explicitly bound a fact's world-time validity window (consumer-driven supersession).
// .at(valid_to) is REQUIRED (no silent default). The next dream() supersession sweep closes it,
// or .close_now() retires already-past-dated bounds inline.
let outcome = mem.supersede(fact_id)
    .at(chrono::Utc::now())
    .close_now()
    .in_namespace(ns)
    .execute()
    .await?;   // SupersedeOutcome::{Bounded { retired } | RejectedTimeInversion | NotFound}
```

`edit_entity` / `delete_*` / `supersede` are all `#[must_use]` builders — nothing happens until you
call `.execute()`. Like `dream()`, `supersede()` is rejected on `AppendOnly` namespaces.

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
| `MemoryBuilder::with_token_tracking(provider, model)` | Composable knob — wraps the already-configured LLM in `TokenTrackingChatProvider` with explicit labels. Supersedes the deprecated `with_llm_tracked(WithLlmTrackedParams, llm)` |
| `MemoryBuilder::with_provider_rates_path(PathBuf)` | **NEW** — override bundled cost-rates table |
| `MemoryBuilder::with_llm(llm)` | Unchanged — backward compatible (untracked path) |
| Tier 1 shortcuts (`with_ollama` etc.) | Apply token tracking internally — auto-emit metrics |
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

### v0.1.3 → v0.3.2 — what changed (high level)

The changes since v0.1.3 are overwhelmingly **additive** — facade-tier consumer code from v0.1.3
continues to compile and run. The notable surface + behaviour changes, at the API-surface level:

| Area | Change |
|---|---|
| **Dream consolidation** (§6) | The dream phase is now fully wired — reconciliation passes (type discovery, aliases, reclassify, consistency-check, canonicalize) + four graph-global consolidation ops (community detection, cross-episode merge, supersession sweep, fact archival). All default ON; cross-episode merge defaults to **Shadow**. `DreamSummary` gained honest per-op fields (the old `episodes_processed` / `edges_merged` fields never existed on the shipped struct — use the fields in §6). |
| **Reversibility** (§6a) | **NEW** — `mem.mutation_history` / `list_mutations` (SEE), the unified `mem.undo(mutation_id)` dispatcher + per-kind `unmerge` / `undo_entity_edit` / `undo_delete_entity` / `undo_delete_fact` / `unsupersede` / `restore_archived_fact`, plus direct mutations `edit_entity` / `delete_entity` / `delete_fact` / `supersede`. Every destructive mutation is reversible (ADR-073). |
| **Content recall** (§5) | **NEW** — `mem.recall(q).content()` for BM25/FTS5 search over raw episode text, under the `content-search` feature (ADR-072) — **now ON by default** (ADR-078), which also RRF-fuses the BM25 + dense episode arms into the ordinary `recall()` path. |
| **Namespace policy enforcement** | v0.1.4 declared policies; **v0.1.5+ ENFORCES them** (ADR-029b): `dream()` / `forget()` / `supersede()` now return `Error::NamespacePolicyViolation` on `AppendOnly` namespaces. This is a behaviour change if you registered `AppendOnly` policies expecting the v0.1.4 declare-only semantics. |
| **Multi-namespace recall** | `recall(q).in_namespaces(&[...])` fans out across namespaces with cross-namespace RRF blending. |
| **Metadata filters** | `recall(q).filter_metadata(key, value)` / `.filter_metadata_in(key, &[..])` post-filter recall by episode metadata. |
| **Async extraction wait** | `mem.wait_for_processing(episode_id, timeout)` polls an episode to a terminal extraction state (ADR-051). |
| **Feature flags** (§13) | The crate has an explicit **empty** default feature set + opt-in features (`ner`, `embeddings`, `content-search`, `otel`, `trace`, …). There is no separate `substrate` feature — the facade + substrate free-functions ship in the one default surface. |
| **Node/napi binding** (§14) | The JS binding mirrors the surface in camelCase, including the undo + inspect surface. |

The substrate free-functions (`kremory::memory::submit_episode`, etc.) remain public and unchanged.

---

## §13 — Feature flags

kremory's `default` feature set is **`["content-search"]`** (ADR-078, 2026-07-28 — it was previously
empty). Opt into the surfaces below as needed:

| Feature | Default | Enables |
|---|---|---|
| *(default)* | — | Full `Memory` facade, bi-temporal graph, hybrid recall, dream phase, reversibility. BYOM LLM + embedder always available. |
| `content-search` | **ON** | Three things, not one: (a) the BM25/FTS5 **content arm that `recall()` RRF-fuses in automatically**, (b) the dense episode arm, and (c) Migrations 022 + 026 (`episodes_fts`, `episodes.embedding`). It also enables the explicit `mem.recall(q).content()` terminal + `ContentPassage` type (§5, ADR-072) — but that terminal is the *smallest* part of it. **Turning this off costs −32.2pt ex-adversarial substring recall** (measured, ADR-078); it is a default, not an extra. Zero additional dependencies. |
| `ner` | off | GLiNER hybrid extractor (`.with_gliner()` + `.with_llm(...)`); auto-downloads the ONNX GLiNER model on first use (`ort` / `ndarray` / `tokenizers` / `hf-hub`). |
| `embeddings` | off | Local ONNX embedding-provider support (`ort` / `ndarray` / `tokenizers` / `hf-hub`). BYOM embedders work without it. |
| `otel` | off | OTLP export — `tracing-subscriber` + `tracing-opentelemetry` + OTLP exporter; enables `init_telemetry(...)` (see [observability.md](observability.md)). |
| `trace` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "trace")]` sites exist in the crate (`Cargo.toml:59`). Was intended for hot-path span emission (ADR D3). |
| `unstable-graph` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "unstable-graph")]` sites exist (`Cargo.toml:80-87`). The planned v0.1.6 `Memory::get_related` traversal (G9) **never landed**; this row previously described it as if it had. |
| `unstable-tags` | off | **RESERVED — NOT YET WIRED. Enabling it does nothing.** Zero `#[cfg(feature = "unstable-tags")]` sites exist (`Cargo.toml:88-93`). The planned `episode_tags` junction + `with_tags` / `filter_tag_any` / `filter_tag_all` (G3.b) **never landed**; this row previously described it as if it had. |
| `test-utils` / `llm-smoke` / `llm-integration` | off | Test-harness gating only — not part of the stable consumer surface. |

```toml
# Example: content recall + OTLP export.
# `content-search` is listed explicitly for clarity, but it is ON by default
# since ADR-078 (2026-07-28) — you only need to name it if you have set
# `default-features = false`.
kremory = { version = "0.6", features = ["content-search", "otel"] }
```

---

## §14 — Node / napi binding

The `kremory-napi` crate exposes the `Memory` facade to Node via napi-rs, mirroring the Rust surface
in **camelCase**. The reversibility surface is fully mirrored:

- Inspect: `memory.mutationHistory(...)`, `memory.listMutations(...)` → `MutationRecord[]`.
- Undo: `memory.undo(mutationId)` (the unified dispatcher) plus the per-kind `memory.unmerge` /
  `memory.undoEntityEdit` / `memory.undoDeleteEntity` / `memory.undoDeleteFact`, and the direct
  mutations `memory.editEntity` / `memory.deleteEntity` / `memory.deleteFact`.
- `DreamSummary` mirrors as `JsDreamSummary` with the honest fields (`crossEpisodeWouldMerge` vs
  `crossEpisodeMerged`, `budgetExhausted`, …).

> **Updated 2026-07-28 (ADR-078).** This note previously said `content-search` was "a Rust-only
> opt-in ... not enabled in the current `kremory-napi` build". That understated it: the binding had
> **no passthrough for the feature at all**, so a Node consumer could not enable it under any
> circumstances and was locked to entity/fact-only recall — which measures **−32.2pt**.
> `kremory-napi` now declares `default = ["content-search"]` plus an opt-in `rerank` passthrough and
> a `RecallOptions.rerankK` knob, so JS `recall()` gets the same RRF-fused hybrid surface Rust does.
> What is still Rust-only is the **explicit `.content()` terminal + the `ContentPassage` type** — a
> new JS output type rather than a field mirror. The undo + inspect surface *is* available in JS.

---

*API reference current as of kremory v0.4.0 (2026-07-12). Facade design: ADR-027 (outside-in API design). Temporal model: ADR-003. BYOM contract: ADR-002. Dream reversibility: ADR-073. Content recall: ADR-072. Crate topology: ADR-028 (single-crate + cargo features, supersedes ADR-007 + ADR-008).*

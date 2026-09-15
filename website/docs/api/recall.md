# Recall


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
    println!("{}: {:.3}", r.summary, r.score);
}
```

`RetrievedContext::new(...)` / `RetrievedFact::new(...)` (bundled-params structs
`RetrievedContextNewParams` / `RetrievedFactNewParams`) are the constructors used when building
result sets by hand — mainly test fixtures and advanced substrate consumers, not the ordinary
`recall()` path above.

### `as_of` (bi-temporal filtering)

```rust
use chrono::{Utc, Duration};

// Valid-time filter: what was TRUE in the world at t.
// It never gates on recorded_at — see the bi-temporal notes below.
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
since content-search became a default feature (2026-07-28) — so the terminal and the `ContentPassage` type exist in a default build.
If you have disabled default features, re-enable it explicitly:

```toml
kremory = { version = "0.8", default-features = false, features = ["content-search"] }
```

⚠️ Disabling it does **not** just remove `.content()` — it also removes the BM25 content arm and the
dense episode arm from the ordinary `recall()` path, measured at **−32.2pt** ex-adversarial
substring recall.

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

**When to reach for `.content()` instead of `.raw()`** — measured 2026-07-28, conv0, same
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

Notes:

- **BM25-only** — content passages are a distinct, un-fused stream. They are NOT blended into the
  entity/fact RRF ranking; `.content()` does not extend the graph-shaped `RetrievedContext`.
- **Single-namespace only** — use `.in_namespace(...)`. Multi-namespace fan-out (`.in_namespaces`)
  is not yet supported on this terminal and returns `Err`.
- Requires a `Memory` built via the builder/providers path (same as `.forget()` / the
  `filter_metadata` post-filter).

### Recipe: session expansion (right conversation, wrong turn)

A measured failure mode worth knowing about: on our LoCoMo miss-set, **50% of the evidence turns we
fail to retrieve sit in a session we ALREADY hit** — retrieval finds the right conversation and
returns the wrong turn. Widening `k` does not fix this reliably;
pulling the *rest of the hit's source* does.

kremory has no built-in "session" concept, and does not need one — the behaviour composes from
primitives that already ship. (The same `source_id` you set here also drives **write**-side
prior-turn replay — see "Prior-turn replay" below. Threading a conversation buys both halves at once.)

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

### Prior-turn replay: references that resolve across turns

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
let mem = Memory::open("./agent.db")
    .with_llm(llm)
    .with_embedder(emb)
    .prior_turn_replay_depth(0)   // 0 = off; default 10
    .await?;
```

Threading a conversation therefore buys you both halves at once: **write**-side reference
resolution (this section) and **read**-side session expansion ("Recipe: session expansion", above).

See `.ai-docs/adrs/adr-080-prior-turn-replay-into-extraction-2026-09-06.md`.

---

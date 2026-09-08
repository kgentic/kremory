# Ingest


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
    .in_namespace(ns.clone())
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
    .entry("Meeting at 2pm")
        .from_chat("session-42")
        .in_namespace(Namespace::new("user-jim"))
        .done()
    .entry("Alice prefers async Rust")
        .from_document("note-7")   // EpisodeEntryBuilder has from_chat/from_document only
                                    // (no from_note/from_source — unlike RememberRequest, above)
        .in_namespace(Namespace::new("user-jim"))
        .done()
    .with_batch_id("import-2026-05-27")  // idempotent — safe to retry
    .await?;
```

### Chunking large documents before `remember()`

`kremory::split_for_embedding(text, max_chars)` splits text into chunks that fit your embedder's
context window (kremory never calls this automatically — you decide when a document is too large
and re-`remember()` each chunk yourself, per its own module doc comment):

```rust
use kremory::split_for_embedding;

let chunks = split_for_embedding("Some text\n\nfrom a\ndocument", 20);
assert!(chunks.iter().all(|c| c.chars().count() <= 20));
```

Returns the text unchanged (as a single-element `Vec`) when it already fits — always safe to call
unconditionally, including on short episodes. A reasonable `max_chars` for `nomic-embed-text`'s
~2048-token window is `6000`-`8000`; check your own embedder's real limit rather than assume.

---

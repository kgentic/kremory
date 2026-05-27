# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — Unreleased

First substantive release. Establishes the bi-temporal substrate, BYOM
architecture, and `kremory::memory` orchestration layer.

### Added

- **Bi-temporal storage** — every fact carries `recorded_at` (system time,
  immutable) and `valid_from`/`valid_to` (world time, mutable). Audit-grade
  history with no data loss. ADR-Phase-D.0.
- **BYOM (Bring Your Own Model)** — `Arc<dyn ChatProvider>` +
  `Arc<dyn EmbeddingProvider>` boundaries. No bundled model. ADR-D.0 §7.
- **`GraphHandle` trait** — pluggable storage-backend boundary behind
  `Arc<dyn GraphHandle>`. Enables backend swap (libSQL → kuzu → Neo4j) without
  breaking the public API. Story #211.
- **`StubGraphHandle`** — do-nothing test infra impl gated behind
  `#[cfg(any(test, feature = "test-utils"))]`. Story #A5.
- **4-byte magic prefix `KMRY`** — all binary snapshots begin with `b"KMRY"`.
  `validate_snapshot_header` rejects corrupt/truncated/wrong-version blobs.
  Stories #151, #152, #164.
- **`effective_k` clamp** — search LIMIT is clamped to `k.min(n_available).max(1)`.
  No panic on oversized k. Story #166.
- **`published_at` precedence** — when `SourceRef.published_at = Some(T)`,
  stored `Fact.valid_from = T` regardless of `StructuredFact.valid_from`. Story
  #318.
- **7-type `MemoryType` enum** — `Semantic | Episodic | Procedural | Emotional |
  Flashbulb | Prospective | Working`. Column added to `facts` table. Story #208.
- **SHA-256 content-hash dedup** — duplicate episode content returns
  `Err(Duplicate { existing_id })` via idempotent CAS. Story #209.
- **`BEGIN IMMEDIATE` writes** — concurrent writers serialise without deadlock.
  Story #210.
- **PRAGMA `user_version` migration tracking** — bumped after each migration run.
  Story #212.
- **`CREATE TABLE IF NOT EXISTS` guards** — all migrations idempotent on re-run.
  Story #213.
- **SQLite-first vector ordering** — `backfill_missing_embeddings` fn + test.
  Story #214.
- **`dirty` AtomicBool + `flush_if_dirty`** — double flush collapses to single
  I/O. Story #215.
- **Cascade delete — `forget_entity`** — removes entity + all dependent facts,
  episodic edges, and FTS rows in a single BEGIN IMMEDIATE transaction. Story
  #216.
- **Batch forget — `batch_forget`** — deletes up to 250 entities in 100-item
  transactional chunks. Story #217.
- **`access_count` field** — incremented on every search hit; queryable for LRU
  eviction. Story #247.
- **Background ingest queue** — fire-and-forget `BackgroundIngestor` with
  structured-tracing spans and `rql.background.*` metrics.
- **`OnceLock` lazy init** — deterministic cache initialisation; second call
  reuses first. Story #148.
- **Three-cache separation** — speculative, embedding, and community caches are
  independent tiers. Story #149.
- **Pre-mutation validation** — `IntraBatchDuplicate` error on intra-batch
  collisions. Story #150.
- **Named struct variants on `Error`** — all error variants carry named fields;
  no positional tuple variants. Story #155.
- **Zero `unwrap()` in non-test code** — enforced by
  `-D clippy::unwrap_used -D clippy::expect_used`. Story #9 / #156.
- **Dual-emit gate script** — `scripts/check-dual-emit.sh --ci` validates that
  every metric emits to both structured tracing and `metrics` crate. Story #A4.
- **SLO TOML** — `monitoring/kremory-memory-slos.toml` with 9 SLO definitions
  mapped to canonical metric names. Story #A5.
- **ADR index** — `.ai-docs/adrs/` with full decision chain from ADR-D.0 through
  the cycle-2 amendments. Story #1.
- **`docs/api.md`** — substrate-level API reference placeholder. Story #22.
- **`release-please` CI workflow** — automated changelog + version bump on merge
  to `main`. Story #6.
- **Lock ordering doc** — module-level comment in `kremory::core::engine` documents
  mutex acquisition order. Story #229.
- **`AtomicI64` CAS TTL sweep** — at most one sweep per 60-second window.
  Story #235.
- **`has_outer_transaction` flag** — nested transaction detection; inner commits
  demoted to savepoints. Story #246.
- **`recorded_at` rename** — legacy `created_at` column renamed to `recorded_at`
  across schema, DDL, and all struct fields. Story #A1.

### Architecture decisions (ADRs)

- **ADR-D.0** — single-crate, Apache-2.0, BYOM boundaries.
- **ADR-D.1 / D.2 / D.3** — substrate DDL, migration discipline, naming.
- **ADR-D.6** — async/event integration, `GraphHandle` trait shape, `StubGraphHandle`.
- **Cycle-2 amendments** — `as_of` no-op warn, `published_at` precedence, BYOM gate.

---

[Unreleased]: https://github.com/kgentic-dev/kremory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kgentic-dev/kremory/releases/tag/v0.1.0

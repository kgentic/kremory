# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — v0.1.4 in-progress

### Breaking

- **`RetrievedContext` is now `#[non_exhaustive]`** — construct via
  `RetrievedContext::new(entity_id, entity_name, summary, score, source_refs)`
  + `.with_*` fluent setters. Struct-literal construction from outside the
  crate is no longer source-compatible. Forward-compat preparation for
  ADR-029c multi-namespace recall (additive `namespace` field expected v0.1.5+).
  No downstream consumers known to be affected at v0.1.3.

### Added

- Internal `kremory-eval` crate (`publish = false`) — two-layer quality eval
  harness. Layer A: published-comparable benchmarks (LongMemEval). Layer B:
  diagnostic metrics (entity P/R/F1, RAGAS 6 metrics, graph integrity
  invariants, contradiction/temporal G-Eval). Judge: AA `LlamaCppProvider`
  with Gemma 4 E2B Q4_K_M. BYOM invariant intact — AA dependency stays in
  `kremory-eval` only.
- Inspect-AI-style scorer traits (`Score`, `Scorer`, `Dataset`, `Solver`)
  with `TieBreakPolicy::Pass` default and non-optional `reasoning: String`.

### Architecture decisions (ADRs)

- **ADR-029a** — `NamespacePolicy` struct + `register_namespace` (declare-
  but-don't-enforce). Independently ratifiable, no enforcement claims. See
  `.ai-docs/adrs/rql/adr-029-namespace-policy/adr-029a-namespace-policy-struct.md`.
- **ADR-029b** + **ADR-029c** — append-only enforcement + multi-namespace recall.
  Targeted v0.1.5+. See `.ai-docs/adrs/rql/adr-029-namespace-policy/`.

Doc-vs-code parity backfill — see also tier-D items in `.ai-docs/planning/roadmap-post-v013-2026-05-28.md`.

## [0.1.3] — 2026-05-28

Hygiene release. `cargo clippy --all-targets --all-features -- -D warnings` now passes
clean across the workspace.

### Changed

- **Test-scope `clippy::unwrap_used` + `clippy::expect_used` exemption** — added
  `#![allow(...)]` attributes to `#[cfg(test)] mod tests` blocks across 21 src/ files
  and 54 integration-test files in `tests/`. Production-code clippy strictness
  unchanged (still `-D warnings` with no allows in impl code).
- **Structural lint fixes (no `#[allow]`)** — `ner.rs` identity-op simplifications;
  `extraction.rs` + `speculative_cache.rs` `len() > 0` → `!is_empty()`;
  `contradiction.rs` targeted `#[allow(too_many_arguments)]` on test helper.

### Architecture decisions (ADRs)

- **ADR-028** — Defer kremory-core / kremory-ingest crate split — use cargo
  features instead (supersedes ADR-007 + ADR-008). See
  `.ai-docs/adrs/rql/adr-028-defer-crate-split-cargo-features-2026-05-28.md`.

## [0.1.2] — 2026-05-28

LLM observability parity. Brings chat-layer telemetry to the same standard the
embedder layer shipped with in v0.1.0. Closes Gaps A–E identified in the
v0.1.2 architecture spec.

### Added

- **`TokenTrackingChatProvider<L>` wrapper** (`kremory::core::chat_tracking`) —
  newtype that wraps any `ChatProvider` and emits token counters + cost gauge +
  duration histogram per `chat_with_tools` call. Spec Gap A.
- **`ProviderRates` loader** (`kremory::core::rates`) — bundled `include_str!`
  load of `crates/kremory/monitoring/provider-rates.toml`. 13 model entries:
  3 OpenAI embedders + 3 VoyageAI embedders + 2 local embedders + 2 OpenAI chat
  + 2 Anthropic chat + 1 Ollama wildcard. Runtime override via
  `MemoryBuilder::with_provider_rates_path(PathBuf)`. Spec Gap B.
- **`MemoryBuilder::with_llm_tracked<L>(provider, model, llm)`** — explicit
  builder method that wraps a custom `ChatProvider` in `TokenTrackingChatProvider`
  with provider+model labels. Tier 1 shortcuts (`with_ollama`, `with_openai`,
  `with_anthropic`) internally upgrade to call this. `with_llm` retained
  unchanged for backward compatibility. Spec Gap E.
- **`#[tracing::instrument]` on Engine spans** — `Engine::ingest_with` named
  `kremory.ingest`, `Engine::contextualize` named `kremory.contextualize`. GenAI
  OpenTelemetry SemConv parent spans. Spec Gap C.
- **Background-thread span context propagation** — `core/background.rs` captures
  `tracing::Span::current()` before `std::thread::spawn` and re-enters inside the
  spawned thread via `_enter` guard. Spec Gap C.
- **`otel` feature gate** — opt-in. When enabled, installs `tracing-subscriber`
  registry with `EnvFilter` + `fmt` + `tracing-opentelemetry` layer that exports
  to OTLP gRPC (configured via `OTEL_EXPORTER_OTLP_ENDPOINT`, default
  `http://localhost:4317`). When disabled, telemetry stays as standard `metrics`
  + `tracing` emission with no exporter. Spec Gap C.
- **`init_telemetry(TelemetryConfig)` real implementation** — replaces v0.1.0
  stub. Returns `TelemetryHandle` that owns the OTel `TracerProvider`. Caller
  retains handle for lifetime; `handle.shutdown()` flushes spans on graceful
  termination. Behind `otel` feature flag; remains a no-op stub when feature off.
- **`kremory::observability` module** — public re-export module that surfaces
  `TokenTrackingChatProvider`, `TokenTrackingEmbedder`, `ProviderRates`,
  `ProviderRateEntry`, `RatesError`, `PROVIDER_RATES`, `init_telemetry`,
  `TelemetryConfig`, `TelemetryHandle`, `TelemetryInitError`, `llm_error_type`.
- **`llm_error_type(LLMError) -> &'static str`** — maps all 11 verified
  `autoagents-llm 0.3.7` `LLMError` variants to 3 bounded values:
  `"server_error"` (Http / Provider / Generic / GuardrailBlocked /
  GuardrailExecutionFailed), `"client_error"` (Auth / InvalidRequest /
  NoToolSupport / ToolConfigError), `"parse_error"` (Json / ResponseFormatError).
  Emitted as `error.type` SPAN ATTRIBUTE (not histogram label) to keep
  cardinality discipline. Spec Gap D + ADR D7.
- **`kremory_core_tokens_total{operation, provider, model, direction}`
  counter** — chat operations now emit alongside the existing embed emissions.
  `operation = "chat" | "embed"`; `direction = "input" | "output"`.
- **`kremory_core_cost_usd_total{operation, provider, model}` gauge** — f64 USD
  cumulative cost. Emitted by both chat and embed paths. Gauge (not counter) per
  spec ADR D9 supersession on cost unit.
- **`kremory_core_chat_duration_seconds{provider, model, status}` histogram** —
  per-call wall-clock duration. `status = "ok" | "error"` (bounded 2 values).
- **11 new G_v012_* tests** — 6 unit tests in `core/chat_tracking.rs` +
  `core/rates.rs`, 3 facade integration tests in `tests/llm_integration.rs`,
  2 OTel integration tests in `tests/otel_integration.rs` (gated
  `#[cfg(feature = "otel")]`). No `#[ignore]` on any G_v012_*. Per spec §17.
- **`MockChatProviderTracking`** — test helpers crate `tests/helpers/mock_chat.rs`
  with `WithUsage(input, output)` / `AlwaysFail(error)` / `RespondThenFail`
  behavior variants. Backs G_v012_2/3/6/7/8/9 + future retry tests.

### Changed

- **`provider-rates.toml` relocated** — moved from `monitoring/provider-rates.toml`
  to `crates/kremory/monitoring/provider-rates.toml` so the bundled rates ship
  inside the published crate tarball. `include_str!` path adjusted accordingly
  (4 `..` → 2 `..`). Project-root `monitoring/` retains SLO + dual-emit-allowlist
  files; only rates moved.
- **`TokenTrackingEmbedder` cost emission** — now reads from `PROVIDER_RATES`
  global and emits `kremory_core_cost_usd_total{operation="embed"}` alongside
  the existing `kremory_core_tokens_total{operation="embed"}` counter.
- **Per-test clippy + recorder pattern** — G_v012_* unit tests use
  `metrics_util::debugging::DebuggingRecorder` + `metrics::with_local_recorder`
  for isolated metric capture, matching the existing `TokenTrackingEmbedder`
  test pattern.

### Known limitations (documented but unblocking)

- **`with_ollama` Tier 1 emits 0 token counters** — `autoagents-llm 0.3.7`
  Ollama backend's `ChatResponse::usage()` returns `None`. The wrapper still
  emits duration histogram + one-shot warning per `(provider, model)` pair.
  Token + cost flow lights up automatically when upstream ships
  `autoagents-llm 0.3.8+` with the Ollama backend's `usage()` override.
- **`rate_limited` and `timeout` error.type labels** — not producible from
  `autoagents-llm 0.3.7` (no structured variants for these conditions). Both
  route through `HttpError(_)` / `ProviderError(_)` → `"server_error"` per
  spec §17.6.
- **Prompt cache token tracking (Anthropic)** — `autoagents-llm`'s anthropic
  backend already parses `cache_creation_input_tokens` +
  `cache_read_input_tokens`; kremory wiring to emit these is deferred to v0.1.4
  (queued in `.ai-docs/planning/roadmap-post-v013-2026-05-28.md` item B.1).

### Architecture decisions (ADRs)

- **kremory-v012-architecture spec** — locked via Vera 3-cycle adversarial
  review (74/100 CONCERNS + 4 inline patches). Path:
  `.ai-docs/architecture/kremory-v012--llm-observability-parity-architecture.md`.
- **ADR D7 cardinality discipline** — `provider` + `model` labels bounded at
  builder construction; `error.type` bounded to 3 values via `llm_error_type`.
- **ADR D9 supersession on cost unit** — cost counter stores float USD directly
  (via `metrics::gauge!`), NOT integer micro-USD. ADR-028 documents the
  supersession.
- **Quinn adversarial review applied** — 10 of 11 spec-drift findings fixed
  in commit `af30e2d`. F-07 disputed (sync `_enter` guard sufficient for
  sync `worker_loop`). Full audit:
  `.ship/sessions/kremory-v012-impl-20260527-192208/swarm-memory/quinn-fixes-impl.md`.

## [0.1.1] — 2026-05-27

Recall pipeline redesign. Fixes 5 substrate bugs surfaced during integration
testing. No public API changes.

### Fixed

- **RRF (Reciprocal Rank Fusion) constant** — `k = 60` now used for hybrid
  retrieval score blending. Aligned with Zep/Graphiti production defaults.
- **`episodic_edges` authoritative** — episodic edges are the canonical record
  of entity-mention-in-episode; deprecated fallback paths removed.
- **LightRAG stub-entity insertion** — forward references in entity extraction
  insert stub entities at first mention; promoted to full entities when
  re-ingested with body.
- **First-mention snippet** — `source_ref` carries the first-encountered text
  snippet for the entity, not the most recent.
- **`SourceKind::Episode`** — replaces deprecated `SourceKind::Document` for
  ingestion sources; aligns with Graphiti semantic.

### Added

- **5 new G_v011_* integration tests** — `source_refs_carries_episode_kind`,
  `rrf_single_result_scores_one`, `standalone_entity_has_episodic_edge`,
  `stub_entity_inserted_on_forward_reference`, `stub_entity_promoted_on_reingestion`.

### Architecture decisions (ADRs)

- **kremory-v011-architecture spec** — recall pipeline redesign locked.
  Path: `.ai-docs/architecture/kremory-v011--recall-pipeline-redesign-architecture.md`.

## [0.1.0] — 2026-05-27

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

[Unreleased]: https://github.com/kgentic/kremory/compare/kremory-v0.1.3...HEAD
[0.1.3]: https://github.com/kgentic/kremory/compare/kremory-v0.1.2...kremory-v0.1.3
[0.1.2]: https://github.com/kgentic/kremory/compare/kremory-v0.1.1...kremory-v0.1.2
[0.1.1]: https://github.com/kgentic/kremory/compare/kremory-v0.1.0...kremory-v0.1.1
[0.1.0]: https://github.com/kgentic/kremory/releases/tag/kremory-v0.1.0

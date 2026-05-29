# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — targeting v0.1.5

### Breaking

- **`AppendOnly` enforcement is now ACTIVE** (ADR-029b §3.1). v0.1.4 shipped
  the `NamespacePolicy` declaration window with operational warnings but no
  enforcement; v0.1.5 closes the loop. `Memory::forget()` and `Memory::dream()`
  against a namespace with `immutability = AppendOnly` now return
  `Err(Error::NamespacePolicyViolation { namespace, operation, policy })`
  instead of succeeding with a `tracing::warn!`. Callers relying on the
  v0.1.4 warn-then-proceed behaviour must update to handle the new error
  variant. `AppendOnly` is now safe to use as a compliance contract.

- **Storage migration to composite primary keys.** `entities` table gains
  composite PK `(id, group_id)`; `facts` and `episodic_edges` gain composite
  foreign keys referencing it. Three sequential SQLite migrations
  (`migrate_004`, `migrate_005`, `migrate_006`) run on first open of a v0.1.4
  database; backup tables (`entities_bak_004`, `facts_bak_006`,
  `episodic_edges_bak_006`) are retained as rollback artifacts. Idempotent +
  partial-recovery aware. Direct-SQL consumers reading these tables must
  account for the new column shape and FK semantics. Operators are advised
  to run `kremory-admin backup` before first v0.1.5 open. See ADR-029b §5
  for the migration runbook and §6 for the rollback procedure.

### Added

- **Multi-namespace recall** (ADR-029c). `Memory::recall(query).in_namespaces(&[ns_a, ns_b, ns_c]).await`
  fans out across namespaces concurrently and returns a single result list
  with per-result attribution. New `RetrievedContext.namespace: Option<Namespace>`
  field (`#[serde(default)]` for v0.1.4 forward-compat) records the source
  namespace for each result. Mutual exclusion with `in_namespace(ns)` —
  setting both on the same request returns
  `Err(Error::ConflictingNamespaceSelectors)`. New `best_effort(bool)` builder
  setter for partial-failure-tolerant recall (default: fail-all). Auto-generated
  `recall_id: Uuid` propagated to every tracing span for cross-namespace query
  correlation; override via `with_recall_id(uuid)`.

- **`context_block` namespace attribution.** When recall results span multiple
  namespaces, `context_block` prepends `[ns:{group_id}]` to each rendered
  entry. Single-namespace results render without the prefix (no behavioural
  change for existing consumers).

- **`Memory::upgrade_namespace_policy(ns)`** — atomic policy transition
  (currently supports `Mutable → AppendOnly` upgrade direction). Wraps the
  policy update in a `BEGIN IMMEDIATE` transaction, drains the
  `NamespacePolicyCache` to prevent stale-read race, and stamps
  `namespaces.upgraded_at`. Grandfather policy for rows written under the
  prior policy is preserved. See ADR-029b §4 for upgrade semantics.

- **`kremory-admin` CLI** — new internal crate for operator workflows.
  Subcommands: `migrate` (run pending migrations on a database file),
  `upgrade-namespace` (offline policy transition), `backup` (file-level
  backup with metadata snapshot), `verify` (`PRAGMA foreign_key_check` +
  composite-PK integrity sweep). Distributed as a standalone binary.

- **`kremory-napi` crate** — Node.js / TypeScript binding via `napi-rs`
  (ADR-030 Decision 2 Form B). In-process async API surface:
  `JsMemory.open(path, opts?)`, `.ingest(text, opts?)`, `.recall(query, opts?)`,
  `.close()`. Plain-data option types match the Rust facade.
  Rust `Result<T, MemoryError>` maps to Promise rejection with the original
  error message preserved. Distributed as a prebuilt `.node` native module
  under `@kgentic/kremory-node`. Enables in-process kremory access for
  TypeScript hook scripts and aidocs-style tooling without HTTP/IPC overhead.

- **`Error::NamespacePolicyViolation { namespace, operation, policy }`** —
  typed error for `AppendOnly` mutation attempts. Pattern-matchable; carries
  the violating operation name (`"forget"`, `"dream"`, `"reassign"`) and the
  stored policy so callers can present meaningful diagnostics.

- **`Error::ConflictingNamespaceSelectors { request }`** — typed error for
  recall builder misuse (calling both `in_namespace(ns)` and
  `in_namespaces(&[..])` on the same request).

- **`kremory-napi` multi-namespace surface parity.** The Node.js binding
  gains the ADR-029c recall surface: `JsRecallOptions.inNamespaces`,
  `.bestEffort`, `.perNamespaceTopK`, and per-row `JsRetrievedContext.namespace`
  attribution. The previously silent-no-op `JsOpenOptions.defaultNamespace`
  is now plumbed — set at `JsMemory.open(path, opts)` and applied as a
  fallback to subsequent `ingest()` / `recall()` calls that omit per-call
  namespace. `embeddingDim` remains deferred per ADR-030 Form B and emits
  a `tracing::warn!` when set.

- **`kremory-napi` TypeScript smoke suite.** Nine-case `node:test`-driven
  suite (`__test__/smoke.test.mjs`) covering happy path, defaultNamespace
  fallback, namespace-required rejection, error mapping, multi-namespace
  recall, concurrent ingest, close idempotency, and conflicting-selector
  rejection. Companion `types.check.ts` typechecks the generated
  `index.d.ts` shape so a Rust-side rename breaks compile before runtime.

- **ADR-031 — kremory-napi surface parity policy.** Three-layer drift
  defense: (1) policy ADR encoding the contract, (2) pmcp PreToolUse
  advisory hook firing on edits to `crates/kremory/src/facade/**` and
  `memory/types.rs`, (3) mechanical `syn`-based parity test deferred to
  next session. Motivated by the ADR-029c surface drift caught only by
  manual cross-read while authoring the TS smoke suite.

### Changed

- **RRF fusion now keys on composite `(id, group_id)`.** Previously, recall
  with multiple namespaces could collapse same-name entities across
  namespaces into a single result. The fusion stage now treats
  `(id, namespace)` tuples as distinct keys, preserving namespace identity
  through the recall pipeline. Single-namespace recall unaffected.

- **`NamespacePolicyCache` capacity API.** Internal constructor now accepts
  `NonZeroUsize` directly rather than `usize` with a runtime fallback;
  guarantees a valid capacity at compile time. Default capacity unchanged
  (256 entries). No effect on public API.

### Fixed

- **`vector_distance_cos` NULL handling.** Vector cosine search previously
  panicked when SQLite returned `NULL` for zero-magnitude embedding vectors
  ("Null value" SqliteFailure). Recall now skips NULL-distance rows
  gracefully and continues. Surfaces silently as a one-result-fewer
  outcome rather than an error.

- **`NamespacePolicy` JSON backward compatibility.** Doc comment promised
  `#[serde(default)]` on `immutability`, `forgettable`, and `dream_eligible`
  fields, but the attribute was missing — meaning v0.1.3-era serialized
  policies (without those fields) would fail to deserialize. The attribute
  is now applied; old payloads load with default values.

- **`with_llm_tracked` doctest.** The hidden setup line used `impl Trait`
  in a let-binding position (Rust error E0562); replaced with an inline
  stub struct so the doctest compiles and runs.

### Deprecated

- None.

### Notes

- Test coverage expanded to the full kremory test pyramid: property tests
  (`proptest`) for RRF dedup + contradiction overflow + serde round-trip,
  concurrency tests (`tokio` multi-thread + `Barrier`) for recall fan-out +
  cache thundering herd + partial-migration recovery + policy upgrade
  atomicity, and `#[ignore]`-gated E2E tests against a real Ollama instance
  covering multi-namespace recall + `AppendOnly`/`Mutable` coexistence +
  `kremory-admin` CLI smoke. Canonical CI invocation:
  `cargo test --workspace -- --test-threads=1`.

- BYOM invariant intact. `autoagents-llm` remains a `kremory-eval`-only dev
  dependency; `kremory`, `kremory-napi`, `kremory-admin`, and `kremory-mcp`
  carry no LLM-provider dependency in their dependency graph.

---

## [0.1.4] — 2026-05-28

### Breaking

- **`RetrievedContext` is now `#[non_exhaustive]`** — construct via
  `RetrievedContext::new(entity_id, entity_name, summary, score, source_refs)`
  + `.with_*` fluent setters. Struct-literal construction from outside the
  crate is no longer source-compatible. Forward-compat preparation for
  ADR-029c multi-namespace recall (additive `namespace` field expected v0.1.5+).
  No downstream consumers known to be affected at v0.1.3.
- **`Namespace` gained a `policy: Option<NamespacePolicy>` field** (ADR-029a).
  Exhaustive struct expressions (`Namespace { namespace, thread }`) no longer
  compile from outside the crate. Migrate to
  `Namespace::new(namespace).with_thread(thread)` — already idiomatic, no
  caller using the constructor is affected. `#[serde(default)]` keeps v0.1.3-
  serialized JSON forward-compatible (the field deserializes to `None`).

> **⚠️ DECLARATION vs ENFORCEMENT** (ADR-029a): `NamespacePolicy` lets callers
> DECLARE per-namespace policy intent (e.g. `AppendOnly`, non-forgettable,
> non-dream-eligible). v0.1.4 **PERSISTS** the declaration and emits
> operational warnings, but does **NOT ENFORCE** the policy on `dream()` /
> `forget()` / mutation operations. Enforcement lands in v0.1.5+ per ADR-029b.
> **Do not rely on `AppendOnly` for compliance contracts until v0.1.5+.** Use
> the v0.1.4 declaration window to capture intent + validate napi-rs binding
> ergonomics with aidocs vNext.

### Added

- **`NamespacePolicy` struct + `Memory::register_namespace` facade method**
  (ADR-029a). Per-namespace policy primitive with three v0.1.4 fields:
  `immutability` (`Mutable` | `AppendOnly`), `forgettable`, `dream_eligible`.
  `#[non_exhaustive]` + fluent setters (`with_immutability`, `with_forgettable`,
  `with_dream_eligible`) + canonical `APPEND_ONLY` preset const + `validate()`
  cross-field coherence check. New `InvalidPolicyError` enum surfaces validation
  failures; new `Error::InvalidPolicy` + `Error::NamespacePolicyImmutable`
  variants on the core `Error` enum. `Memory::register_namespace(ns)` is
  idempotent on identical re-registration and atomic via `BEGIN IMMEDIATE`
  against concurrent `remember(...)` first-touch implicit creation. Every
  non-default policy declaration emits `tracing::warn!` on target
  `kremory.namespace` with the marker `POLICY DECLARED BUT NOT ENFORCED` —
  closes the declare-but-don't-enforce footgun (Vera cycle-1 HIGH-1
  mitigation #1). New `namespaces` SQLite table (unprefixed per ADR-029a
  Decision 8; the `rql_*` rename of existing tables lands in ADR-029b).
  Population is lazy: `remember`/`recall`/`forget`/`dream` write a
  default-policy row on first observation of a previously-unseen namespace.
  See ADR-029a §6 for the full backward-compat walkthrough and `docs/api.md`
  §3 for usage examples.

- Internal `kremory-eval` crate (`publish = false`) — two-layer quality eval
  harness. Layer A: published-comparable benchmarks (LongMemEval). Layer B:
  diagnostic metrics (entity P/R/F1, RAGAS 6 metrics, graph integrity
  invariants, contradiction/temporal G-Eval). Judge: AA `LlamaCppProvider`
  with Gemma 4 E2B Q4_K_M. BYOM invariant intact — AA dependency stays in
  `kremory-eval` only.
- Inspect-AI-style scorer traits (`Score`, `Scorer`, `Dataset`, `Solver`)
  with `TieBreakPolicy::Pass` default and non-optional `reasoning: String`.
- LongMemEval harness now wires `Memory` via the Tier-2 builder
  (`Memory::open(path).with_llm(...).with_embedder(...)`) with explicit
  Ollama providers. Models are env-configurable via standard `OLLAMA_HOST`,
  `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL` (defaults: `qwen2.5:14b`
  chat + `nomic-embed-text` 768-dim embed). Resolves O13 — `Memory::with_ollama`
  hardcoded `llama3.2` 3B which produced duplicate `VerbatimString` entity
  names on LongMemEval transcripts, tripping intra-batch dedup.
- `eval layer-a longmemeval --smoke` flag: runs against local synthetic
  fixtures with either judge, skipping the HuggingFace download. Smoke
  validates pipeline wiring before committing to a full Oracle baseline.
- `docs/eval.md` runbook + `docs/eval-fixtures.md` inventory.

### Changed

- `LongMemEvalScorer` now reads the structured `is_correct: bool` field from
  `JudgeVerdict` instead of substring-matching `"yes"` in `verdict.reasoning`.
  Upstream `evaluate_qa.py` substring-matches because its judge returns
  free-form text; `GemmaJudge` returns structured JSON, so the bool is more
  reliable across model families. Observed 2026-05-28 with Gemma 4 E2B-IT:
  positive reasoning ("matching the correct answer") without an explicit
  "yes" prefix caused false 0.0 scores. Existing MockJudge tests are
  unaffected — they already set both `is_correct` and `reasoning` consistently.
- **`SingleCallExtractor` default prompt switched from `V2SchemaLight` to
  `V3SchemaHybrid`** (`core::extraction::PromptVersion`). V2 omitted the
  "No duplicates." instruction to save ~100 prefill tokens — the savings
  was real but the cost was hidden: smaller models (`llama3.2:3b`,
  `gemma4-e2b`) emitted duplicate entity names that tripped kremory's
  intra-batch dedup invariant. V3 keeps V2's schema-first structure
  (GoLLIE +13 F1 signal) plus the explicit "No duplicates." line — adds
  ~20 prompt tokens, unlocks the smaller-model extraction matrix.
- **Intra-batch duplicate entity emission no longer FATAL** — kremory's
  ingest pipeline now emits a `tracing::warn!` and silently dedupes when
  the extractor produces duplicate normalized entity names, instead of
  returning `Err(IntraBatchDuplicate)`. The dedup logic at
  `core::ingest::ingest_with` runs immediately after the previously-fatal
  guard, so behaviour-equivalent rows land in the DB. Story #150's
  human-API-caller intent is preserved on the error enum (variant kept
  for forward-compat with a possible explicit strict-batch API) but the
  ingest hot path is now robust to noisy LLM extractors. Required because
  smaller local models emit duplicates by design — see the V3 prompt
  change above for the upstream root cause fix.
- **ADR-029a contract restoration in facade `forget()` + `dream()`** —
  parallel ADR-029b work-in-progress had wired AppendOnly enforcement
  directly into the facade, breaking the shipped `v0.1.4 declare-but-
  don't-enforce` promise. The enforcement is now a `tracing::warn!` with
  marker `POLICY DECLARED BUT NOT ENFORCED` matching the contract; the
  same operations remain functional. Enforcement lands in v0.1.5 per
  ADR-029b.

### Migration robustness

The v0.1.4 substrate migrations underwent a ship-architect + Vera
adversarial review (see `.ai-docs/architecture-review/v014-vera-adr-029b-
composite-fk-review-2026-05-28.md`). Three HIGH findings addressed:

- **Idempotency gates are SHAPE-based, not backup-table-presence based.**
  Previously a crash between `CREATE TABLE ..._bak_NNN` and the rename
  step would leave the backup-table sentinel present, causing the next
  startup to silently skip the migration and the DB to be permanently
  missing tables. Gates now inspect `PRAGMA foreign_key_list` (composite
  FK presence) and `PRAGMA table_info` (composite PK columns) — recovery-
  safe regardless of crash timing.
- **`PRAGMA foreign_keys = ON` is now guaranteed-restored on any error
  path** via an inner async block that lets the outer fn always issue the
  restore PRAGMA, regardless of whether the body succeeded or returned
  Err. Previously a `?`-early-exit between FK toggles left the connection
  with FK enforcement silently OFF for all subsequent application writes.
- **`facts_fts` virtual table is now DROP+CREATE+repopulated** after
  `facts` is restructured. Previously `DROP TABLE facts` did not cascade
  to the standalone fts5 table, leaving zombie rowids that returned
  phantom results on FTS queries.

Plus: per-step error context (each `?` site annotates which step failed),
`PRAGMA foreign_key_check` validation at the end of every restructure,
and convention-compliant `CREATE TABLE IF NOT EXISTS` on all migration-
local backup and `_new` tables (passes the `no_bare_create_table_in_schema_source`
hygiene gate).

### Test pyramid foundation

- 8 facade integration test files migrated from a shared `/tmp/test.db`
  path to per-test unique paths via a `unique_db_path` helper. Eliminates
  intermittent `SqliteFailure(5, "database is locked")` flakes under
  parallel `cargo test --workspace` execution.
- `LongMemEvalScorer` gained 2 regression tests that lock in the
  `is_correct` bool fix: positive reasoning without a `"yes"` prefix
  must score 1.0; reasoning containing `"yes"` with `is_correct=false`
  must score 0.0.
- `OllamaEmbedAdapter` (the eval-side BYOM bridge) gained 3 unit tests
  exercising the batch → single-text contract with fake `AutoEmbeddingProvider`
  impls. Includes an empty-batch error case + determinism check.
- `eval` bin CLI parser gained 9 unit tests covering all flag
  combinations (`--smoke`, `--judge`, `--sample`, unknown subcommands,
  invalid `--sample` values).
- One pre-existing doctest bug fixed: `MemoryBuilder::with_llm_tracked`
  rustdoc used `let x: impl Trait = todo!()` which is illegal Rust
  (E0562); rewritten as a generic on the enclosing example fn signature.

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

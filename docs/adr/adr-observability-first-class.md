---
title: 'kremory/kremory observability as first-class citizen — dual-emit, OTel SemConv, BYOM cost ledger, library-vs-app split'
type: adr
status: accepted
created: '2026-05-20'
updated: '2026-05-20'
ratified: '2026-05-20'
ratification_mode: 'full'
slug: kremory-memory-observability-first-class-2026-05-20
tags:
  - kremory
  - kremory
  - observability
  - opentelemetry
  - otlp
  - prometheus
  - tracing
  - cardinality
  - byom
  - cost-ledger
  - slo
  - dual-emit
  - genai-semconv
  - adr
refs:
  - id: kremory-memory-observability-first-class-research-2026-05-20
    rel: informed_by
  - id: rqlcm-mvp-v1-architecture-2026-05-20
    rel: extends
  - id: rqlcm-mvp-v1-test-strategy-2026-05-20
    rel: extends
  - id: plan-rqlcm-v0.1.0-ralph-loop-execution-runbook-2026-05-20
    rel: amends
  - id: kremory-async-event-handle-api-design-2026-05-19
    rel: extends
  - id: adr-observability-prometheus-facade
    rel: aligns_with
  - id: adr-observability-slo-toml
    rel: extends
  - id: adr-observability-dispatch-pattern
    rel: aligns_with
  - id: adr-observability-jsonl-streaming
    rel: aligns_with
---

# ADR — kremory/kremory Observability as First-Class Citizen

## Status

**Accepted** — 2026-05-20. Folds into v0.1.0 Phase B.1 (BYOM embedding subsystem absorbs token tracker + cost ledger) and Phase C.3 (CI dual-emit + cardinality gates). Adds ~3.75 AI-days to v0.1.0 envelope (10.25–14.75 → ~14–18.5). Per `feedback_no_shortcuts_zero_tech_debt`: observability that is documented in spec but unimplemented is tech debt — must ship with v0.1.0, not deferred.

## Context

### The gap

kremory/kremory v0.1.0 ships with **0/113 dual-emit compliance** — 113 production `metrics::counter!/histogram!/gauge!` sites paired with **zero** `tracing::*` calls. `crates/kremory/src/memory/src/graph.rs` (dream-phase orchestrator) is completely dark. No OTel integration. No BYOM cost observability. No SLO file. No consumer-facing observability API.

Full inventory + verification in companion research doc `kremory-memory-observability-first-class-research-2026-05-20.md`. Headline findings re-stated in §10 below.

### Constraints driving this ADR

1. **CLAUDE.md `.claude/rules/observability.md`** — dual-emit (`metrics::` + `tracing::`) is mandatory project-wide. kremory/kremory currently violates this in 100% of metric sites.
2. **`feedback_no_shortcuts_zero_tech_debt`** — observability documented in SCOPE-003 / SCOPE-303 / spec §841 but unimplemented = tech debt. Must fix in scope.
3. **kremory is OSS Apache-2.0** — published independently. External consumers (BYOM customers, future cloud) need a stable observability contract. Library cannot install global exporters.
4. **kremory = Zep-equivalent paid SDK** — multi-tenant; cost attribution per provider per tenant is a first-class product feature, not a debugging affordance.
5. **Cross-layer correlation** — kremory must be join-able to the host application's `CycleMetrics` JSONL for end-to-end traceability.
6. **Verified ecosystem pins** — `oss/AutoAgents/crates/autoagents-telemetry/` carries OTel 0.31.0 / tracing-opentelemetry 0.32.1 / opentelemetry-otlp 0.31.0. kremory/kremory aligns with these.
7. **Single verified prior art for LLM observability in Rust** — `rig` adopts OTel GenAI semantic conventions (`gen_ai.usage.input_tokens` etc.). Adopting these means OTLP-exportable with zero field renames later (Langfuse, Phoenix, OpenLLMetry, Honeycomb compatible).

## Decision

### D1 — Dual-emit is mandatory and CI-enforced

Every `metrics::counter!/histogram!/gauge!` site in kremory + kremory production code MUST have a paired `tracing::info!/warn!/error!` within 5 source lines. The pair is co-located, single-source-of-truth — no separate "logging strategy" that drifts from metrics.

**Enforcement**: `scripts/check-dual-emit.sh` runs in PR CI per test-strategy §11. Body of script per runbook §6.3 lines 885–901. Allowlist for explicit exceptions (e.g. test helpers) lives at `monitoring/dual-emit-allowlist.txt` with one path per line + justification comment.

**Exception process**: removal from gate requires a one-line allowlist entry with comment + reviewer ack in PR description. Permanent removal blocked.

### D2 — Library does NOT own exporters; consumer owns them

kremory emits only via `tracing` macros + `metrics::*!` macros. kremory MUST NEVER:
- Call `tracing::subscriber::set_global_default(...)`
- Call `metrics::set_global_recorder(...)`
- Call `opentelemetry::global::set_tracer_provider(...)`
- Initialize any HTTP listener, OTLP exporter, or stdout writer

kremory wraps kremory and exposes `init_telemetry(config: RqlmTelemetryConfig)` for cloud/BYOM consumers. the host application installs its own recorder + subscriber in `main.rs` per existing ADR-observability-prometheus-facade.

**Rationale**: confirmed via `metrics-rs` and `tracing` canonical docs (research doc §4.2). Universal pattern across tokio, axum, hyper, sea-orm, sqlx.

### D3 — OTel integration behind `otel` feature flag; default build OTel-free

```toml
[features]
default = []
trace   = []                                             # hot-path instrumentation gate
otel    = ["dep:tracing-opentelemetry", "dep:opentelemetry", "dep:opentelemetry_sdk"]
```

**Verified version pins** (from `autoagents-telemetry`):

```toml
tracing-opentelemetry = "0.32"
opentelemetry         = "0.31"
opentelemetry_sdk     = { version = "0.31", features = ["trace"] }
```

Default build (no `otel`, no `trace`) has zero OTel dependency cost — only `tracing = "0.1"` + `metrics = "0.24"` API crates compile in. Both are silent no-ops when no subscriber/recorder is installed (verified `metrics` 0.21+ behavior).

OTLP exporter (`opentelemetry-otlp = "0.31"`) lives in **kremory**, not kremory. kremory instruments; kremory + the host application + customer apps configure exporters.

### D4 — Hot-path instrumentation is compile-time gated (`trace` feature)

Per bevy / sqlx / tokio canonical pattern. Spans on the following surfaces MUST be wrapped in `#[cfg(feature = "trace")]`:

- `graph.rs` per-entity-update path (29 histogram sites today)
- `search.rs` per-query path (16 sites)
- `background.rs` ingestor worker loop body (per-event hot path)
- `extraction.rs` per-utterance path
- `ner.rs` per-utterance path

Spans on **outer-boundary** call surfaces are NOT gated — always instrumentable in default build:

- `add_episode` (kremory)
- `search` (kremory public API)
- `dream_phase_run` (kremory public API)
- `commit` (per ADR-kremory-async-event)

**Rationale**: runtime `EnvFilter` has non-zero perf impact per bevy `docs/profiling.md`. Hot loops (per-entity, per-utterance) are tick-loop equivalents — compile-time gate is mandatory.

### D5 — IngestEventSink propagates `opentelemetry::Context` via envelope field

`IngestEvent` carries:

```rust
pub struct IngestEvent {
    pub payload: IngestPayload,
    #[cfg(feature = "otel")]
    pub trace_cx: Option<opentelemetry::Context>,  // cloned from Span::current().context()
}
```

Producer-side: `event.trace_cx = Some(tracing::Span::current().context())` (via `OpenTelemetrySpanExt`).
Consumer-side worker: `let _guard = event.trace_cx.as_ref().map(|cx| cx.attach());` — spans here become children of the original caller's span.

When `otel` feature disabled, `trace_cx` field is cfg-omitted (zero overhead).

**Rationale**: `tracing`'s thread-local span context does NOT cross `tokio::sync::mpsc` channel sends. Pattern B from research doc §4.3 — verified canonical.

### D6 — Cross-crate `trace_id` field on `IngestResult`

```rust
pub struct IngestResult {
    pub episode_uuid: Uuid,
    pub trace_id: Option<Uuid>,  // caller-injected for cross-crate join with the host application CycleMetrics
    // ...
}
```

`add_episode(payload, trace_id: Option<Uuid>)` accepts caller's `trace_id`. Same UUID flows into kremory histogram exemplars and the host application's `CycleMetrics` JSONL → join key for correlation queries.

`trace_id` is NEVER a Prometheus label (cardinality discipline) — only an exemplar value on histograms and a `tracing::Span` field.

### D7 — Cardinality budget (Rust enums for all metric labels)

| Label | Bound | Enforcement mechanism |
|---|---|---|
| `provider` | ~5 (OpenAI / Anthropic / Voyage / Cohere / local) | Rust enum, `as_str()` for label conversion |
| `model` | ~10 per provider | Allow-listed from `monitoring/provider-rates.toml` at startup; panic on unknown |
| `operation` | ~8 (extract_nodes / dedup_nodes / dedup_edges / embed / rerank / dream_phase_synthesis / search / commit) | Rust enum, `&'static str` |
| `direction` | 2 (input / output) | Enum |
| `error_kind` | ~5 (rate_limit / auth / server_error / timeout / parse_error) | Enum |
| `group_id` | NOT a metric label by default | Application-level partition; appears as span field + JSONL field only |
| `provider_kind` (for AEC dedup-style labels) | bounded enum | Per existing `aec-health.jsonl` precedent |

**Forbidden as metric labels** (unbounded): `episode_uuid`, `entity_uuid`, `edge_uuid`, `user_id`, `request_id`, `trace_id`, `span_id`, raw entity names, raw prompt fragments.

**Forbidden labels surface in CI**: `scripts/check-cardinality.sh` greps for any `"<uuid|name|prompt|user>"` substring inside `metrics::*` macro calls in kremory + kremory source; fails CI on hit.

## D7 exceptions — approved runtime-string labels

`namespace` / `group_id` labels: used in `rql.ingest.*` and `kremory.ingest.*` metrics
(emit sites: `deferred.rs:369,518`, `verify_stage.rs:445`).

Rationale: `namespace` is admin-controlled, bounded to ≤~100 values per deployment
(not user-generated, not UUID-shaped). Typical deployments have 1–10 namespaces; even
multi-tenant SaaS with per-organisation namespaces stays well under 1000 series per
metric. This is an **approved exception** to the `&'static str` constraint in D7. Metric
consumers should apply a namespace-cardinality limit in their aggregation layer if the
deployment exceeds ~100 namespaces (e.g. a Prometheus `topk` or relabelling rule).

No other dynamic (runtime-string) labels are approved. Any new dynamic label requires a
new entry in this exceptions table with the same rationale format (deployment bound,
controlled-not-user-generated evidence, aggregation-layer mitigation).

### D8 — Adopt OTel GenAI semantic conventions for LLM-call span fields

All LLM-call sites (extraction, dedup, dream-phase synthesis) and embedding-call sites use OTel GenAI SemConv span fields:

```rust
tracing::info_span!(
    "kremory.embed",
    "gen_ai.system"                          = provider_label,
    "gen_ai.request.model"                   = model_label,
    "gen_ai.usage.input_tokens"              = tokens_input,
    "gen_ai.usage.output_tokens"             = tokens_output,
    "gen_ai.usage.cache_read.input_tokens"   = tokens_cached,
    operation                                = "embed",
)
```

Prometheus metric names (`kremory_core_tokens_total`) stay as Prometheus convention. Span fields use GenAI SemConv. Both surfaces cover the same data — Prometheus for pull-scrape aggregation, span fields for trace-driven debugging + Langfuse/Phoenix/OpenLLMetry integration.

**Rationale**: rig is the single verified prior art for LLM observability in Rust (DAG landscape research). Adopting its field namespace means OTLP-exportable with zero field renames when a collector is wired.

### D9 — BYOM cost ledger via versioned `provider-rates.toml`

File at `monitoring/provider-rates.toml`:

```toml
rates_as_of = "2026-05-20"

[openai.text-embedding-3-small]
input_usd_per_mtok = 0.020
batch_discount     = 0.50

[voyage.voyage-4-lite]
input_usd_per_mtok = 0.020
batch_discount     = 0.33

[local]
input_usd_per_mtok = 0.000
batch_discount     = 0.00
# ... etc
```

Read at startup. `kremory_core_cost_usd_total{provider, model, operation}` counter increments in integer micro-USD (×1e6 conversion) to avoid float drift. When rates change: update TOML, restart consumer — no code change.

**Quarterly review** of rate freshness — documented in `monitoring/provider-rates.toml` header comment. Stale rate (`rates_as_of` > 90 days old) emits `tracing::warn!` at startup.

### D10 — `TokenTrackingEmbedder` wrapper enforces emission (anti-langchain-rust pattern)

```rust
pub struct TokenTrackingEmbedder<E: EmbeddingProvider> { /* per research doc §5.6 */ }

impl<E: EmbeddingProvider> EmbeddingProvider for TokenTrackingEmbedder<E> {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // pre-count via tokenizer (approx if local)
        // record start, call inner, record elapsed
        // emit: kremory_core_tokens_total + kremory_core_cost_usd_total + kremory_core_request_duration_seconds
        // emit: tracing::info_span with gen_ai.* fields (D8)
    }
}
```

`EmbeddingProvider` trait gains:

```rust
trait EmbeddingProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn last_usage_tokens(&self) -> Option<u64> { None }  // default: caller falls back to pre-count
}
```

**Test discipline**: snapshot recorder test MUST observe `kremory_core_tokens_total` after any embed call. Failure mode prevented: langchain-rust pattern of capturing `TokenUsage` in result structs but never emitting. CI runs `tests/observability/embed_emits_token_counter.rs` with `DebuggingRecorder + with_local_recorder`.

### D11 — `monitoring/kremory-memory-slos.toml` extends ADR-D6 schema

Adds `labels` field for matching label-subset SLOs. Schema:

```toml
[[slo]]
metric      = "..."
labels      = {operation = "...", provider = "..."}  # optional, defaults to all labels
percentile  = "p95" | "rate_5m" | "rate_1h" | "max"
operator    = "lt" | "gt" | "le" | "ge"
threshold   = <number>
description = "..."
```

Full SLO list per research doc §5.3. `MetricsReport::assert_slos()` reader gets one new filter pass to match label subsets. Aligns with ADR-observability-slo-toml; not a parallel surface.

### D12 — Span name convention `{domain}.{operation}` lowercase dot-separated

| Domain | Operations |
|---|---|
| `kremory` | `add_episode`, `extract_nodes`, `extract_edges`, `dedup_nodes`, `dedup_edges`, `embed`, `graph_upsert`, `search`, `commit` |
| `kremory.dream_phase` | `community_recompute`, `supersession_sweep`, `cross_episode_distill`, `stale_archive` |

Matches Graphiti's verified convention (`search.embed_query_vector`, `execute_scopes`). Configurable prefix via `RqlcConfig::span_prefix: Option<String>` for multi-instance deployments.

### D13 — Metric naming convention: underscores throughout, `_seconds` unit suffix on histograms

Existing `rql.db.entity.insert_ms` → rename to `kremory_core_db_entity_insert_seconds` (Prometheus convention: dots cause issues in some PromQL contexts; `_seconds` is the canonical time unit suffix; histograms record in seconds-as-f64, not ms-as-u64).

Migration in Phase C.3 — one mechanical commit per crate. Update test-strategy §11 grep gate to match new names. Document old→new mapping in commit body for downstream consumer migration.

### D14 — Anti-pattern enforcement via grep gates

Three CI grep gates run in PR CI:

1. **Dual-emit gate** (`scripts/check-dual-emit.sh`): every `metrics::` paired with `tracing::` within 5 lines. Body per runbook §6.3.
2. **Cardinality gate** (`scripts/check-cardinality.sh`): no `"<uuid|name|prompt|user>"` substring in `metrics::*` macro calls.
3. **`.enter()`-across-`.await` gate** (`scripts/check-span-enter.sh`): no `.enter()` calls inside `async` functions in kremory/kremory source (use `.instrument()` instead).

Gates are blocking — failure = PR red.

### D15 — Consumer-facing observability API surface (the OSS contract)

```rust
// In kremory public API:
pub struct RqlcConfig {
    pub metrics_prefix: Option<String>,   // namespacing for multi-instance
    pub span_prefix:    Option<String>,   // for OTel span name discrimination
    // No recorder field, no tracer field, no subscriber field — those belong to consumer
}

// In kremory public API:
pub struct RqlmTelemetryConfig {
    pub tracer_provider: Option<Arc<opentelemetry_sdk::trace::SdkTracerProvider>>,
    pub meter_provider:  Option<Arc<opentelemetry_sdk::metrics::SdkMeterProvider>>,
    // Customer brings their own wired-up providers
}

pub fn init_telemetry(config: RqlmTelemetryConfig) -> Result<TelemetryHandle> {
    // Wires customer providers to global; documents consumer-owns-subscriber-init responsibility
    // Returns handle that holds providers for .shutdown() at process exit
}
```

`TelemetryHandle::shutdown()` calls `.shutdown()` on each provider (per OTel 0.28+ migration — `global::shutdown_tracer_provider()` was removed).

## Consequences

### Positive

- **First-class observability shipped with v0.1.0** — no tech debt deferral
- **OSS contract is clean** — kremory + kremory independently publishable; consumers own exporters
- **OTLP-ready** — `otel` feature flag + GenAI SemConv span fields = zero field renames when Langfuse/Phoenix/OpenLLMetry/Honeycomb collector wired
- **BYOM cost transparency** — token counts + USD ledger per provider/model/tenant; runaway cost guardrail via SLO
- **Cardinality safe** — every label bounded by enum or allow-list; Prometheus won't explode under multi-tenant load
- **Cross-crate correlation** — `trace_id` flows from the host application → kremory histogram exemplars → JSONL join key
- **Anti-patterns enforced** — CI grep gates catch dual-emit / cardinality / async-span bugs before merge
- **Hot-path perf preserved** — `trace` feature gates per-entity / per-utterance spans; default build has zero overhead

### Negative

- **+3.75 AI-days in v0.1.0** — pushes envelope from 10.25–14.75 to ~14–18.5 AI-days. Folded into B.1 + C.3, no new phase added.
- **OTel pre-1.0 churn risk** — opentelemetry-rust 0.31.0 may need pin update before v0.2.0; documented in maintenance runbook.
- **kremory carries 2 always-on dep crates** (`tracing`, `metrics`) — ~50KB binary size each. Acceptable; API crates are zero-cost when subscriber/recorder absent.
- **Span field naming dual-track** — Prometheus metric names use underscores; OTel span fields use GenAI SemConv dots (`gen_ai.usage.input_tokens`). Documented; not collision-prone (different surfaces).
- **`provider-rates.toml` requires quarterly maintenance** — rate drift is an ops chore. Mitigation: `tracing::warn!` on stale-rate detection (>90 days) + quarterly review meeting.

### Risks not yet addressed

- **`metrics-util::FanoutBuilder` exact API** — multi-recorder fan-out for simultaneous Prometheus + OTel meter export. Verify at implementation time (Phase C.3). Fallback: single-recorder selection (Prometheus OR OTel meter, not both) for v0.1.0.
- **Zep Cloud SLA reference baseline** — not surfaced by context7 research. May exist on status.getzep.com. Verify if kremory-cloud product launches.

## Alternatives Considered

### Alt 1 — Defer observability to v0.2.0

**Rejected.** Violates `feedback_no_shortcuts_zero_tech_debt`: SCOPE-003 + SCOPE-303 already documented in v0.1.0 spec. Deferring documented-but-unimplemented work = tech debt by definition. Per critical rule #8 (treat the cause): the cause of "100% dual-emit non-compliance" is "no enforcement gate"; deferring leaves the cause unfixed.

### Alt 2 — Prometheus-only, skip OTel entirely for v0.1.0

**Rejected.** kremory is the Zep-equivalent paid SDK layer — customers run it in their own cloud with existing OTel stacks (Honeycomb / Datadog / Grafana Tempo). Prometheus-only forces customers to add a parallel observability path for an embedded library, which is precisely the friction rig's GenAI SemConv design avoids. Zero-dep `otel` feature flag is the right cost/benefit.

### Alt 3 — Bundle OTLP exporter directly into kremory

**Rejected.** Violates library-vs-application split (verified canonical across tokio / axum / sea-orm / sqlx). Forces tonic + prost + hyper + tokio-rustls into every kremory consumer's dep graph, ~5–10MB binary inflation. Consumer-owned exporter is the universal Rust ecosystem pattern.

### Alt 4 — Custom `InternalEvent` typed-event trait (vector RFC 2064 pattern)

**Deferred to post-v0.1.0.** Justified at vector's hundred-contributor scale. Premature for v0.1.0 — adds indirection layer that obscures debug observability for marginal type-safety gain. Re-evaluate when kremory/kremory has > 10 active contributors.

### Alt 5 — Per-entity child spans (one span per entity per tick)

**Rejected.** Universal anti-pattern observed across bevy / tokio / ractor / sqlx. Bevy explicitly: one outer-boundary span, entity identity as field. Confirmed by DAG landscape research §4 (universal absence). Aligns with cardinality discipline — entity UUIDs are span fields, not span names.

## References

### Internal artefacts cited

This decision drew on internal design research, architecture specs, and test-strategy documents
that are part of kremory's private planning history rather than this public repository.

### External canonical references

- rig GenAI SemConv: https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-core/src/telemetry/mod.rs
- OTel Rust migration 0.28: https://github.com/open-telemetry/opentelemetry-rust/blob/main/docs/migration_0.28.md
- Prometheus instrumentation best practices: https://prometheus.io/docs/practices/instrumentation
- bevy compile-time gating: https://raw.githubusercontent.com/bevyengine/bevy/main/crates/bevy_log/Cargo.toml
- vector InternalEvent RFC 2064 (rejected alt): https://github.com/vectordotdev/vector/blob/master/rfcs/2020-03-17-2064-event-driven-observability.md
- Google SRE MWMBR alerts: https://sre.google/workbook/alerting-on-slos/

### Provider pricing (May 2026 verified)

- OpenAI: https://openai.com/api/pricing
- Voyage AI: https://docs.voyageai.com/docs/pricing
- Cohere: https://cohere.com/pricing
- Google Vertex: https://cloud.google.com/vertex-ai/generative-ai/pricing (TODO_VERIFY)

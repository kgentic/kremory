# kremory Observability Guide

Shipped in v0.1.2 (LLM observability parity). Embedder observability shipped earlier in v0.1.0. This doc is the canonical reference for both surfaces.

---

## Table of contents

1. [What kremory emits](#what-kremory-emits)
2. [Metric catalog](#metric-catalog)
3. [Tracing spans](#tracing-spans)
4. [Wiring observability into your app](#wiring-observability-into-your-app)
5. [Provider rates (cost emission)](#provider-rates-cost-emission)
6. [OTel / OTLP export](#otel--otlp-export)
7. [Cardinality discipline](#cardinality-discipline)
8. [Known limitations](#known-limitations)
9. [Dashboards + queries](#dashboards--queries)
10. [Cross-references](#cross-references)

---

## What kremory emits

For every LLM call (chat) and embedding call (embed), kremory emits:

- **Token counter** — cumulative input + output tokens per `(provider, model, direction)`
- **Cost gauge** — cumulative USD cost per `(provider, model)`, computed from a bundled rates table
- **Duration histogram** — wall-clock latency per call, labelled by outcome
- **Tracing span** — GenAI OpenTelemetry SemConv-aligned, with `error.type` attribute on failures

Emission is **automatic** when you use:
- Tier 1 shortcuts (`Memory::with_ollama`, `with_openai`, `with_anthropic`) — auto-wrap providers
- `MemoryBuilder::with_token_tracking(provider, model)` — explicit tracked wrap, applied after `.with_llm(...)`

Emission is **not** automatic for `MemoryBuilder::with_llm(llm)` — that path is intentionally untracked for users who don't want metric emission overhead.

---

## Metric catalog

All metrics use the [`metrics`](https://crates.io/crates/metrics) crate. Backend-agnostic — wire to Prometheus, StatsD, Datadog, OpenTelemetry, in-memory recorder for tests, etc.

### Counters

| Name | Labels | Increment unit | Emitted by |
|---|---|---|---|
| `kremory_core_tokens_total` | `operation`, `provider`, `model`, `direction` | u64 token count | `TokenTrackingChatProvider`, `TokenTrackingEmbedder` |

Label values:
- `operation`: `"chat"` or `"embed"`
- `provider`: bounded by `monitoring/provider-rates.toml` keys (`"openai"`, `"anthropic"`, `"ollama"`, `"voyage"`, `"local"`)
- `model`: model name string supplied at builder construction (BYOM = runtime value)
- `direction`: `"input"` or `"output"` (chat); embedder uses `"input"` only

### Gauges

| Name | Labels | Unit | Emitted by |
|---|---|---|---|
| `kremory_core_cost_usd_total` | `operation`, `provider`, `model` | f64 USD (cumulative) | `TokenTrackingChatProvider`, `TokenTrackingEmbedder` |

Notes:
- Cost is **cumulative** — increments on each call by `(input_tokens + output_tokens) × rate_per_1k / 1000.0`
- Rate looked up from bundled `provider-rates.toml` or runtime override
- If no rate is found for the `(provider, model)` pair: cost emission is **skipped** (no zero noise) and a one-shot `tracing::warn!` fires per unique pair

### Histograms

| Name | Labels | Unit | Emitted by |
|---|---|---|---|
| `kremory_core_chat_duration_seconds` | `provider`, `model`, `status` | seconds (f64) | `TokenTrackingChatProvider` |
| `kremory_core_embed_duration_seconds` | `provider`, `model`, `status` | seconds (f64) | `TokenTrackingEmbedder` |

Label values:
- `status`: `"ok"` or `"error"` (bounded 2 values per ADR D7)

`error.type` is **not** a histogram label — it's a tracing span attribute (see next section). This keeps histogram cardinality at 2 status values.

---

## Tracing spans

Every chat + embed call creates a span with [GenAI OpenTelemetry SemConv](https://opentelemetry.io/docs/specs/semconv/gen-ai/)-aligned attributes. Top-level kremory ops (`Engine::ingest_with`, `Engine::contextualize`) carry parent spans:

| Span name | Origin | Notes |
|---|---|---|
| `kremory.ingest` | `Engine::ingest_with` | Parent span for ingestion; child spans = chat + embed calls |
| `kremory.contextualize` | `Engine::contextualize` | Parent span for recall; child spans = chat + embed calls |
| `kremory.chat` | `TokenTrackingChatProvider::chat_with_tools` | Per-call chat span; carries `error.type` attribute on failure |
| `kremory.embed` | `TokenTrackingEmbedder::embed_batch` | Per-call embed span |

**`error.type` attribute** (set on error path):

Maps all 11 `autoagents-llm 0.3.7` `LLMError` variants to 3 bounded values:

| Bucket | `LLMError` variants |
|---|---|
| `"server_error"` | `HttpError`, `ProviderError`, `Generic`, `GuardrailBlocked`, `GuardrailExecutionFailed` |
| `"client_error"` | `AuthError`, `InvalidRequest`, `NoToolSupport`, `ToolConfigError` |
| `"parse_error"` | `JsonError`, `ResponseFormatError` |

For error-type breakdown in dashboards, query the span attribute (Langfuse / Phoenix / Tempo / Jaeger all expose this).

---

## ⚠️ Custom tracing targets — `RUST_LOG` by module path will MISS them

**Read this before writing any `RUST_LOG` filter against kremory.**

kremory declares **38 custom `target:` strings** on its `tracing` events. A `RUST_LOG`
directive matches the event's **target**, and when an event sets `target:` explicitly that
target *replaces* the module path. So:

```bash
# ❌ SILENTLY MATCHES NOTHING — merges are on target "kremory.l5", not this module path
RUST_LOG=kremory::core::canonicalization=debug

# ✅ filter on the TARGET string
RUST_LOG=warn,kremory=info,kremory.l5=debug,kremory.graph.provenance=debug
```

**Why this is called out so loudly.** The wrong-looking filter produces an *empty result*,
and an empty result is indistinguishable from "the thing did not happen". On 2026-08-05 that
cost three failed diagnoses of one bug (V1-CANONICAL section 0b-sexies, E2E-1) and — worse —
produced a *recorded* "hypothesis eliminated" note about what turned out to be the **correct**
root cause. A filter that captures nothing looks exactly like a system that did nothing.

**Compounding trap:** on the merge path, **success logs at `debug` while failure logs at
`warn`**. A run at `kremory=info` therefore shows failures and no successes — which reads as
"no merges occurred" when in fact none were visible.

### The targets that matter most when debugging

| target | what it carries |
|---|---|
| `kremory.l4` | ingest-time entity merges + lexical merge blocks |
| `kremory.l5` | dream-time canonicalization merges, blocks, per-pair failures |
| `kremory.l7` | alias resolution |
| `kremory.graph.provenance` | reversible-mutation rows — **the record that a merge committed** |
| `kremory.graph.merge_reembed` | post-merge keeper re-embedding |
| `kremory.dream.consolidation` / `.cross_episode` | consolidation decisions |
| `kremory.ingest.stub` | UNKNOWN stubs inserted for forward references |
| `kremory.ingest.namespace_mismatch` | composite-FK namespace drops (needs `KREMORY_DEBUG`) |
| `kremory.recall` / `.intent` / `.content_search` | recall path + arm selection |
| `kremory.extraction.parsers` | LLM output parse + repair arms |

Enumerate the current set at any time:

```bash
grep -rhoE 'target: "kremory[a-z0-9_.]*"' --include='*.rs' crates/kremory/src/ | sort -u
```

### Debugging recipe

```bash
# Everything on the merge/dream path — the filter to use when a dream pass misbehaves
RUST_LOG='warn,kremory=info,kremory.l4=debug,kremory.l5=debug,kremory.l7=debug,kremory.graph.provenance=debug'
```

`scripts/run-e2e-consumer.sh` sets exactly this and is the reference for a correct filter.

**Before concluding "X never happened" from a log**, prove your filter can produce a
positive: grep for something you are certain is in that run. A negative from an unvalidated
filter is not evidence.

---

## Wiring observability into your app

### Recommended pattern (Tier 1 shortcut — zero config)

```rust
use kremory::Memory;

let mem = Memory::with_ollama("./agent.db").await?;
// Token + cost + duration metrics auto-emit on every chat call
// Span tree auto-wires with kremory.ingest / kremory.contextualize parents
```

### Custom provider (Tier 2 — explicit labels)

```rust
use kremory::Memory;
use std::sync::Arc;

let my_llm = Arc::new(MyCustomChatProvider::new());
let my_embedder = Arc::new(MyCustomEmbeddingProvider::new());

let mem = Memory::open("./agent.db")
    .with_llm(my_llm)
    .with_token_tracking("my-provider", "my-model-v2")
    .with_embedder(my_embedder)
    .await?;
// Provider="my-provider", model="my-model-v2" on all emitted metrics
```

If `("my-provider", "my-model-v2")` is not in the rates table, cost emission silently skips + a one-shot warn fires. To suppress the warn for known-zero-cost providers (self-hosted models), add a row to your custom rates file with cost = 0.0.

### Opt out of tracking

```rust
// `with_llm` (plain) does NOT wrap — no token/cost/duration metrics
let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

Use this for benchmark setups or environments where metric emission overhead is unwanted.

### Reading metrics in your app

kremory does NOT install a metric recorder — that's your app's responsibility. Standard patterns:

```rust
// Prometheus exporter (via metrics-exporter-prometheus crate)
metrics_exporter_prometheus::PrometheusBuilder::new().install()?;

// In-memory recorder (for tests)
let recorder = metrics_util::debugging::DebuggingRecorder::new();
let snapshotter = recorder.snapshotter();
metrics::set_global_recorder(recorder)?;
// ... do work ...
let snapshot = snapshotter.snapshot();

// OpenTelemetry exporter — see "OTel / OTLP export" section
```

---

## Provider rates (cost emission)

Rates live in `crates/kremory/monitoring/provider-rates.toml` and are bundled into the published crate via `include_str!`. Schema:

```toml
rates_as_of = "2026-05-27"

[[providers]]
provider = "openai"
model    = "text-embedding-3-small"
cost_per_1k_tokens_usd = 0.00002
dimensions = 1536

[[providers]]
provider = "openai"
model    = "gpt-4o-mini"
cost_per_1k_tokens_usd = 0.0006   # output rate; v0.1.3 may add per-direction granularity
```

Fields:
- `provider`, `model` — label match keys
- `cost_per_1k_tokens_usd` — rate; cost = `tokens × rate / 1000.0`
- `dimensions` — optional, embedder only
- No `direction` field — single rate per chat model (output rate used as conservative estimate per v0.1.2 scope)

### Override at runtime

```rust
let mem = Memory::open("./agent.db")
    .with_provider_rates_path("./my-rates.toml")
    .with_llm(Arc::new(my_llm))
    .with_token_tracking("openai", "gpt-4o-mini")
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

Format identical to the bundled file. Useful for:
- Custom enterprise pricing (negotiated rates)
- Internal proxy / OpenRouter-style routing
- Testing without modifying the published crate

### Pre-loaded rates

```rust
use kremory::observability::ProviderRates;
use std::path::Path;

let rates = ProviderRates::from_path(Path::new("./my-rates.toml"))?;
let cost = rates.lookup_rate("openai", "gpt-4o-mini").unwrap_or(0.0);
```

`ProviderRates::from_path` takes a concrete `&Path` (not generic over `AsRef<Path>`) — a bare
`&str` literal does not coerce; wrap it in `Path::new(...)`. `from_path` returns
`Result<ProviderRates, RatesError>` — `RatesError::Io` if the file can't be read, `RatesError::Parse`
if the TOML is malformed. Each row deserialises into `kremory::observability::ProviderRateEntry`
(the schema from the "Provider rates (cost emission)" section above); `ProviderRates::cost_usd`
takes its arguments bundled as `CostUsdParams { provider, model, tokens_input, tokens_output }`
rather than four positional arguments.

---

## OTel / OTLP export

Enable the `otel` cargo feature:

```toml
[dependencies]
kremory = { version = "0.7", features = ["otel"] }
```

In your app:

```rust
use kremory::observability::{init_telemetry, TelemetryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install tracing-subscriber + tracing-opentelemetry + OTLP gRPC exporter
    let telemetry = init_telemetry(TelemetryConfig::default())?;

    // Configure OTLP endpoint via env var:
    //   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   (default)

    // ... your app ...
    let mem = kremory::Memory::with_openai("./agent.db").await?;
    mem.remember("Hello world").await?;

    // Flush spans on shutdown
    telemetry.shutdown();
    Ok(())
}
```

`init_telemetry` returns `Result<TelemetryHandle, TelemetryInitError>` — two variants:
`Exporter(String)` (OTLP exporter build failed) and `Subscriber(String)` (the `tracing-subscriber`
global default could not be installed — usually another subscriber was already registered).

What this installs:
- `tracing_subscriber::EnvFilter` (respects `RUST_LOG` env var; default `info,kremory=debug`)
- `tracing_subscriber::fmt` layer (stdout-formatted logs)
- `tracing_opentelemetry` layer (exports spans to OTLP)
- `opentelemetry_sdk::trace::TracerProvider` (batched span processor with Tokio runtime)

What it does NOT install:
- Metric exporter — wire that separately with your preferred backend (Prometheus, OTLP metrics, etc)

**Without the `otel` feature**: `init_telemetry` is a no-op stub that returns `Ok(...)`. Metrics + tracing still emit via the `metrics` + `tracing` crates; no OTLP exporter is configured. Use this for tests + non-OTel environments.

---

## Cardinality discipline

Per [the observability-first-class ADR](adr/adr-observability-first-class.md), label values are bounded:

| Label | Cardinality source | Bound |
|---|---|---|
| `operation` | code-enumerated | 2 values: `chat`, `embed` |
| `provider` | `provider-rates.toml` keys | ~5 today (openai, anthropic, ollama, voyage, local) |
| `model` | BYOM runtime value | bounded by user — one per `Memory` instance |
| `direction` | code-enumerated | 2 values: `input`, `output` |
| `status` | code-enumerated | 2 values: `ok`, `error` |
| `error.type` (span attribute) | code-enumerated via `llm_error_type()` | 3 values: `server_error`, `client_error`, `parse_error` |

**Do not add label values dynamically**. If you have a custom chat provider, choose a stable provider+model identifier at construction time. Per-request label values would explode cardinality and break metric backends.

---

## Known limitations

| Limitation | Cause | Workaround | Resolution path |
|---|---|---|---|
| `with_ollama` Tier 1 emits 0 token counts | `autoagents-llm 0.3.7` Ollama backend's `usage()` returns `None` | Use `.with_llm(custom_provider).with_token_tracking("ollama", model)` with a custom `ChatProvider` that overrides `usage()` by parsing Ollama's `prompt_eval_count` + `eval_count` fields | Awaiting `autoagents-llm 0.3.8` upstream PR (Ollama backend `usage()` override) |
| Google Gemini backend emits 0 token counts | Same — AA's `google.rs` doesn't override `usage()` | Same workaround | Awaiting `autoagents-llm 0.3.8`. Tracked roadmap item B.3 |
| Anthropic prompt-cache tokens not surfaced separately | v0.1.2 scope deferred; AA already parses the fields | Read `cache_creation_input_tokens` + `cache_read_input_tokens` from your own logs in the interim | v0.1.4 — kremory-side wiring; tracked roadmap item B.1 |
| `rate_limited` / `timeout` error.type labels not producible | `autoagents-llm 0.3.7` `LLMError` has no structured variants for these | Both route through `HttpError`/`ProviderError` → `"server_error"` bucket. Query span attributes for raw error message detail | Awaiting `autoagents-llm 0.3.8`+ |
| `with_provider_rates_path` runtime override is path-only, not in-memory | Simplicity tradeoff | Construct `ProviderRates::from_path(...)` directly + use lower-level APIs | Consider in-memory override option for v0.2.0 if requested |

---

## Dashboards + queries

### Prometheus / Grafana

Cost per provider per hour:
```promql
rate(kremory_core_cost_usd_total[1h])
```

Token throughput by direction:
```promql
sum by (direction) (rate(kremory_core_tokens_total{operation="chat"}[5m]))
```

Chat error rate:
```promql
sum(rate(kremory_core_chat_duration_seconds_count{status="error"}[5m]))
  /
sum(rate(kremory_core_chat_duration_seconds_count[5m]))
```

p99 chat latency:
```promql
histogram_quantile(0.99, rate(kremory_core_chat_duration_seconds_bucket[5m]))
```

### Langfuse / Phoenix (via OTel spans)

Error breakdown by `error.type`:
- Filter spans where `name = "kremory.chat"` and `status = error`
- Group by attribute `error.type`
- Buckets: `server_error`, `client_error`, `parse_error`

Slow ingest sessions:
- Filter spans where `name = "kremory.ingest"`
- Sort by `duration` descending
- Drill into child `kremory.chat` + `kremory.embed` spans to identify hot path

---

## Cross-references

- [Observability-first-class ADR](adr/adr-observability-first-class.md) — observability as first-class concern + dual-emit policy
- ADR D7 — cardinality discipline
- ADR D9 — cost emission ADR (superseded on cost-unit by v0.1.2 spec: gauge f64 USD, not micro-USD counter)
- `crates/kremory/monitoring/provider-rates.toml` — bundled rates source of truth

---

## Consumer notes

**[RISK-003] Tracing subscriber requirement.** kremory is a library and installs NO
tracing subscriber (ADR D2 — consumer owns the pipeline). To see operational signals
(warn/info/debug events from kremory internals), install a `tracing_subscriber` in your
binary or application at `WARN` level or below before calling any kremory API. With no
subscriber registered these events are silently dropped — no `eprintln!` fallback, no
stderr output. Minimal setup: `tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();`

**[ASMP-003] INFO volume.** kremory emits one unconditional `tracing::info!` event per
LLM extraction call (`rql.extraction.structured_call_success`). With `RUST_LOG=info`
that is one event per extraction round-trip. To suppress without losing WARN-level
signals: set `RUST_LOG=kremory=warn` at runtime, or compile kremory with the
`release_max_level_warn` feature for compile-time elimination of all INFO-and-below
tracing events (zero runtime overhead).

---
_Authored 2026-05-28 alongside v0.1.3 doc-vs-code parity backfill._

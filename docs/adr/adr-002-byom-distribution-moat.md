---
title: 'ADR-002 — BYOM (Bring Your Own Model) is the distribution moat'
type: adr
status: accepted
audience: public
created: '2026-05-22'
ratified: '2026-05-22'
ratification_mode: full
slug: rql/adr-002-byom-distribution-moat-2026-05-22
tags:
  - adr
  - kremory
  - byom
  - distribution
  - crates-io
  - embedding
  - moat
refs:
  - id: rql/adr-001-engine-architecture-single-crate-apache2-2026-05-22
    rel: depends_on
  - id: rql/adr-005-infrastructure-positioning-three-layer-2026-05-22
    rel: implements
    audience: internal-roadmap
  - id: rql/adr-rql-licensing-amendment-v3-kremory-single-crate-apache2-2026-05-21
    rel: informed_by
---

# ADR-002 — BYOM (Bring Your Own Model) is the Distribution Moat

**Status**: accepted (full)
**Ratified**: 2026-05-22 (James)
**Decision type**: HIGH — permanent architectural invariant; violation breaks crates.io distribution

---

## 1. Decision Summary

`kremory` does not bundle any embedding model. Consumers supply their own embedding backend via the `EmbeddingProvider` trait. This is a **permanent architectural invariant**, not a temporary shortcut.

This decision makes `kremory` publishable to crates.io without size restrictions and creates a structural distribution advantage over the closest Rust-language competitor.

---

## 2. The 440 MB Problem (Verified, Not Hypothetical)

The primary Rust competitor, cogniplex/codemem (v0.18.0, Apache-2.0, 11 GitHub stars as of 2026-05-22), bundles the Candle framework plus the BAAI/bge-base-en-v1.5 model:

```toml
# cogniplex/codemem Cargo.toml (verified 2026-05-22, sha 3f5015af)
candle-core = "0.10"
candle-nn = "0.10"
candle-transformers = "0.10"
hf-hub = "0.5"
tokenizers = "0.23"
```

The BAAI/bge-base-en-v1.5 model is approximately 440 MB (stated in codemem README, primary source: `gh api repos/cogniplex/codemem/contents/README.md`, decoded base64, 2026-05-22). The model auto-downloads on first use from Hugging Face Hub.

**crates.io size limits** (verified from crates.io documentation, 2026-05-22): crates.io imposes a 10 MB compressed crate size limit. A bundled 440 MB ONNX or Candle model binary cannot be included in a crate tarball. Codemem resolves this by downloading the model at runtime rather than bundling it in the crate — but this creates a hidden network dependency and prevents true offline operation.

**kremory's position**: by adopting BYOM, `kremory` ships to crates.io with no model weights, no Hugging Face dependency, no first-run download. The crate tarball is pure Rust code. This is verifiable by any developer who inspects kremory's dependency tree:

```bash
cargo tree -p kremory | grep -E 'candle|onnx|hf-hub|tokenizers'
# Expected: no output — BYOM invariant intact
```

---

## 3. The EmbeddingProvider Trait

kremory exposes an `EmbeddingProvider` trait that consumers implement to supply their preferred embedding backend:

```rust
/// Consumer-supplied embedding backend. kremory never implements this itself.
/// Consumers supply an Arc<dyn EmbeddingProvider + Send + Sync>.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Dimensionality of vectors produced by this provider.
    fn dimensions(&self) -> usize;

    /// Optional: token count from the last embed call (for cost tracking).
    fn last_usage_tokens(&self) -> Option<u64> { None }
}
```

Consumers may use any of these backends (none are bundled with kremory):

| Backend | How | Model examples |
|---|---|---|
| `autoagents-llm` OpenAI-compat | Pass `Arc<OpenAiCompatibleEmbeddingProvider>` | `text-embedding-3-small`, `nomic-embed-text` |
| Raw OpenAI API | Custom impl of `EmbeddingProvider` | `text-embedding-ada-002` |
| Ollama local | Custom impl pointing at `localhost:11434/api/embeddings` | `nomic-embed-text`, `mxbai-embed-large` |
| llama-cpp-2 local GGUF | Custom impl via llama-cpp-2 bindings | Any GGUF embedding model |
| candle (consumer-built) | Consumer bundles Candle + model themselves | Any HuggingFace model |

The consumer decides. kremory never decides for them.

---

## 4. Why This Is a Permanent Invariant, Not a Temporary Shortcut

Three reasons this must never be reversed:

### 4.1 crates.io publishability

The 10 MB compressed crate size limit is a hard platform constraint. If kremory bundled a model, it would be forced to either (a) make the model a runtime download (degrading offline capability) or (b) abandon crates.io as a distribution channel. Neither is acceptable. BYOM is what keeps kremory publishable on the canonical Rust package registry.

### 4.2 Cost control at the consumer layer

BYOM means kremory never purchases LLM or embedding compute on behalf of consumers. The BYOM invariant prevents any bundled model — even a free open-weights model — from creating an expectation of inference subsidy. Consumers control their own embedding costs; kremory has no compute cost floor.

### 4.3 Privacy and enterprise trust

Enterprise deployments (healthcare, legal, finance) cannot use a library that phones home to Hugging Face Hub or any external service during normal operation. BYOM means kremory has zero egress on its own initiative. All network traffic is the consumer's explicit choice (their embedding API key, their Ollama host). This makes kremory suitable for regulated-industry deployment from day one.

---

## 5. Corollary: BYOM is Also a Marketing Differentiator

Against every production agent-memory deployment that already has an embedding model configured (OpenAI, Ollama, Bedrock), bundling a 440 MB model is not a feature — it is dead weight. The correct positioning:

> "kremory doesn't ship a 440 MB model. It plugs into the embedding layer you already have."

This maps to the "Sub-headline: Pure Rust. BYOM. Single binary. Apache-2.0." positioning in the marketing strategy.

The BYOM choice directly enables the "Embeddable library" framing in ADR-005 section 4: kremory is infrastructure that embeds inside other people's products. Infrastructure that auto-downloads a 440 MB model on first use cannot credibly call itself embeddable.

---

## 6. CI Gate

The BYOM invariant is enforced in CI:

```bash
# .github/workflows/ci.yml grep-gates job
- name: BYOM invariant
  run: |
    cargo tree -p kremory > /tmp/tree.txt
    ! grep -qE 'candle|onnx|hf-hub|autoagents-llamacpp' /tmp/tree.txt \
      || (echo "BYOM violation — embedding model crept into kremory deps"; exit 1)
```

Any PR that adds a model-inference dependency to `kremory` directly will fail CI. Consumers who want Candle or llama-cpp can add it themselves.

---

## 7. Consequences

### Positive
- `cargo publish -p kremory` works on crates.io without size violations
- No first-run download surprise for developers evaluating kremory
- True offline operation — kremory works with no internet connection when the consumer uses a local embedding provider
- Zero LLM/embedding cost in kremory's own infrastructure P&L
- Enterprise-ready from day one (no egress, no Hugging Face dependency)

### Negative
- Consumers must wire an embedding provider before kremory can extract semantic features. The quickstart example must be clear about this requirement.
- BYOM creates a friction point for developers who just want "it to work" out of the box. This friction is intentional and acceptable; it is the exact friction that prevents us from publishing to crates.io otherwise.

### Neutral
- Codemem offers BYOM as an option (`CODEMEM_EMBED_PROVIDER` env var) but defaults to the bundled model. kremory treats BYOM as the only option. The two approaches target different audiences: codemem targets local-first zero-config devs; kremory targets production deployments with existing infra.

---

## 8. Ratification

Accepted 2026-05-22. Full ratification. BYOM is a permanent architectural invariant for the `kremory` engine crate. It can be relaxed for consumer crates (e.g., `kremory-cli` might offer a `--model` flag that downloads a model for convenience), but never for the engine crate itself.

- [x] `kremory` crate never imports candle, onnx, hf-hub, or any model-weight dependency
- [x] `EmbeddingProvider` trait is the only embedding interface
- [x] CI grep gate enforces the invariant on every PR
- [x] BYOM is documented as a permanent invariant in README and marketing copy

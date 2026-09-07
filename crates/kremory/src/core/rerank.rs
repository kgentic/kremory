//! Cross-encoder reranker.
//!
//! A post-fusion precision re-scoring pass over the top-`rerank_k` fused
//! candidates (the fusion stage's own output): a local BGE cross-encoder
//! that models true query-passage relevance directly, rather than the
//! BM25/vector-distance/RRF-rank PROXIES the fusion stage uses.
//!
//! Mechanism decision: local cross-encoder via `fastembed`, NOT an
//! LLM-rerank pass — deterministic (no sampling variance), $0 marginal cost,
//! keeps the recall (read) path LLM-free (kremory's own architecture already
//! treats read as LLM-free except for the separate write-path extraction
//! stage).

use std::sync::{Arc, OnceLock};

use crate::core::error::{Error, Result};

/// Process-wide reranker singleton. A single `FastEmbedReranker` per process
/// (not per `Memory` instance) is deliberate for v1: the model is stateless
/// and config-free (`BGERerankerBase`, no per-`Memory` customisation needed
/// yet), so sharing one instance across every `Memory` in the process avoids
/// redundant ONNX-session loads when a process opens multiple `Memory`
/// handles. `FastEmbedReranker::new()` itself is instant (the expensive
/// model load stays lazy, inside its own `OnceCell`, per `FastEmbedReranker`'s
/// doc comment) — a `.with_reranker(...)` builder hook for swapping in a
/// custom `Reranker` impl (the trait's dyn-compat exists precisely to enable
/// this) is a natural follow-up once a second implementation exists; not
/// required by this increment's DoD.
pub(crate) fn default_reranker() -> Arc<dyn Reranker> {
    static INSTANCE: OnceLock<Arc<FastEmbedReranker>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| Arc::new(FastEmbedReranker::new()))
        .clone()
}

/// BYOE (bring-your-own) reranker trait.
///
/// Dyn-compatible via `#[async_trait]` — native AFIT `dyn Trait` fails to
/// compile for a plain `async fn` method on current stable Rust (compile-
/// spiked against this exact shape before writing this trait, per
/// `mechanical-compile-spike-beats-paper-review`; `rustc --edition 2021`
/// on a bare `async fn rerank(&self, ...)` trait produces E0038 "method
/// `rerank` is `async`" — `#[async_trait]` is the documented fallback,
/// `trait-dyn-compat-when-extensible`). The rerank call does model
/// inference (I/O-latency-class work, tens of milliseconds per batch per
/// the `fastembed_rerank_spike` example's `warm_call_secs` measurement), so
/// `async_trait`'s `Box<dyn Future>` dispatch overhead is invisible.
#[async_trait::async_trait]
pub(crate) trait Reranker: Send + Sync {
    /// Re-score `candidates` — `(id, text)` pairs — against `query`.
    ///
    /// Returns `(id, score)` pairs. Implementations SHOULD return results
    /// sorted by score descending (matching `fastembed::TextRerank::rerank`'s
    /// own documented contract) but callers MUST NOT rely on it — the
    /// caller re-sorts explicitly.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>>;
}

/// Resolves which cross-encoder `FastEmbedReranker` loads, from the
/// `KREMORY_RERANK_MODEL` boot override. Default `BGERerankerBase` — byte-
/// identical to pre-override behaviour when unset.
///
/// Exists because an oracle decomposition showed the reranker
/// captured only ~43% of the reordering gain available over its own candidate
/// pool (nDCG@10 62.9 off → 75.6 on, versus a 91.9 perfect-reorder ceiling on
/// the SAME 50 items). The residual is therefore a property of the MODEL, not
/// of the call site — so model choice needs to be sweepable. Mirrors the
/// `KREMORY_RERANK_K` / `KREMORY_RRF_K` / `KREMORY_EPISODE_DENSE` pattern: an
/// A/B costs a server restart, not a rebuild.
///
/// Fail-loud on an unrecognised value (WARN + default) rather than a silent
/// substitution — a benchmark that quietly scored a different model than its
/// provenance stamp claims is the exact class of measurement corruption
/// this guards against. Pure fn (not inlined into the
/// `OnceCell` init) so it is unit-testable, per the `parse_rerank_k` precedent.
pub(crate) fn parse_reranker_model(raw: Option<&str>) -> fastembed::RerankerModel {
    let Some(raw) = raw else {
        return fastembed::RerankerModel::BGERerankerBase;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "bge-base" | "bgererankerbase" => fastembed::RerankerModel::BGERerankerBase,
        "bge-v2-m3" | "bgererankerv2m3" => fastembed::RerankerModel::BGERerankerV2M3,
        "jina-v1-turbo-en" | "jinarerankerv1turboen" => {
            fastembed::RerankerModel::JINARerankerV1TurboEn
        }
        "jina-v2-multilingual" | "jinarerankerv2basemultiligual" => {
            fastembed::RerankerModel::JINARerankerV2BaseMultiligual
        }
        other => {
            tracing::warn!(
                value = %other,
                "KREMORY_RERANK_MODEL is not a recognised reranker — falling back to \
                 bge-base. Valid: bge-base | bge-v2-m3 | jina-v1-turbo-en | \
                 jina-v2-multilingual"
            );
            fastembed::RerankerModel::BGERerankerBase
        }
    }
}

/// Reranker latency lever 2 (cross-encoder execution-provider spike): which
/// ONNX Runtime execution provider `FastEmbedReranker`
/// requests via `fastembed::RerankInitOptions::with_execution_providers`.
/// `Cpu` (the default) is a total no-op — an EMPTY `execution_providers`
/// Vec, byte-identical to pre-lever behaviour (`ort` defaults to CPU-only
/// when the Vec is empty).
///
/// ⚠️ `CoreMl` is a MEASURED NEGATIVE RESULT — kept as an explicit opt-in,
/// NOT recommended. It requests Apple's CoreML EP (`ort::ep::CoreML`).
/// Per-OP fallback to CPU is real (an unsupported op does not error the
/// session), but that is NOT the failure mode observed here: on this BGE
/// reranker session, CoreML registration succeeds and ONNX Runtime
/// partitions the graph into 30+ separate small CoreML sub-models (many
/// `ort::logging: Writing CoreML Model to ...mlmodel` lines per session
/// init) — classic excessive-partitioning pathology for BERT-class encoders
/// under ORT's default "arbitrary" CoreML registration. Empirically this
/// ballooned RSS from the CPU path's ~340MB to 7GB+ and a single rerank
/// call never completed within 60s (one run's server process was killed —
/// almost certainly OOM — after 89s with no response). Verified NOT a
/// system-memory-pressure artifact (18GB free at the time) and reproduced
/// twice. Do not default this on; do not assume "registers cleanly" implies
/// "runs fast" — the aggregate session-level behaviour can be catastrophic
/// even when no single op registration hard-errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RerankExecutionProvider {
    /// Default — no execution providers registered, `ort` runs CPU-only.
    Cpu,
    /// Apple CoreML EP (`ort::ep::CoreML`).
    CoreMl,
}

/// Resolves [`RerankExecutionProvider`] from the
/// `KREMORY_RERANK_EXECUTION_PROVIDER` boot override. Default `Cpu` —
/// byte-identical to pre-lever behaviour when unset. Fail-loud on an
/// unrecognised value (WARN + default), mirroring [`parse_reranker_model`]'s
/// own discipline — a benchmark that silently ran on a different EP than its
/// provenance stamp claims is exactly the measurement corruption this
/// guards against. Pure fn so it is unit-testable without a model
/// load, per the `parse_reranker_model` precedent.
pub(crate) fn parse_rerank_execution_provider(raw: Option<&str>) -> RerankExecutionProvider {
    let Some(raw) = raw else {
        return RerankExecutionProvider::Cpu;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "cpu" => RerankExecutionProvider::Cpu,
        "coreml" | "core-ml" | "core_ml" => RerankExecutionProvider::CoreMl,
        other => {
            tracing::warn!(
                value = %other,
                "KREMORY_RERANK_EXECUTION_PROVIDER is not a recognised execution provider — \
                 falling back to cpu. Valid: cpu | coreml"
            );
            RerankExecutionProvider::Cpu
        }
    }
}

/// Default `Reranker` impl wrapping `fastembed::TextRerank` (BGE reranker
/// base model by default; see [`parse_reranker_model`] for the
/// `KREMORY_RERANK_MODEL` sweep override).
///
/// Lazily initialises the ONNX session on first `rerank()` call (`OnceCell`),
/// not per-call — the `fastembed_rerank_spike` example
/// measured a ~2.5s warm-cache session load (and ~97s on a genuinely cold HF
/// Hub cache, first-ever run), which would be an unacceptable per-call tax.
/// Guarded by a `tokio::sync::Mutex` because `fastembed::TextRerank::rerank`
/// takes `&mut self` (its ONNX `Session` is not `Sync`-safe for concurrent
/// calls) — held only across the actual inference call, not the model load.
pub(crate) struct FastEmbedReranker {
    model: tokio::sync::OnceCell<Arc<tokio::sync::Mutex<fastembed::TextRerank>>>,
}

impl FastEmbedReranker {
    pub(crate) fn new() -> Self {
        Self {
            model: tokio::sync::OnceCell::new(),
        }
    }

    async fn model(&self) -> Result<Arc<tokio::sync::Mutex<fastembed::TextRerank>>> {
        let model = self
            .model
            .get_or_try_init(|| async {
                let start = std::time::Instant::now();
                // `TextRerank::try_new` is a blocking, CPU/IO-bound call (HF
                // Hub download on first-ever run, ONNX session build always) —
                // run it off the async runtime's worker threads so a cold
                // model load doesn't stall other in-flight recalls.
                // Read the overrides BEFORE `spawn_blocking` so the chosen
                // model + execution provider are observable in the log line
                // below even if the load fails.
                let chosen =
                    parse_reranker_model(std::env::var("KREMORY_RERANK_MODEL").ok().as_deref());
                let execution_provider = parse_rerank_execution_provider(
                    std::env::var("KREMORY_RERANK_EXECUTION_PROVIDER")
                        .ok()
                        .as_deref(),
                );
                tracing::info!(
                    reranker_model = ?chosen,
                    execution_provider = ?execution_provider,
                    "kremory.rerank.model_selected (KREMORY_RERANK_MODEL; default bge-base) / \
                     kremory.rerank.execution_provider_selected \
                     (KREMORY_RERANK_EXECUTION_PROVIDER; default cpu)"
                );
                let init_result = tokio::task::spawn_blocking(move || {
                    // Lever 2: `Cpu` passes an EMPTY Vec — `ort`'s
                    // own default when `RerankInitOptions::new` isn't given
                    // `.with_execution_providers(..)` — so this branch is
                    // byte-identical to pre-lever behaviour. `CoreMl` requests
                    // `ort::ep::CoreML` — MEASURED NEGATIVE (see
                    // `RerankExecutionProvider` doc comment above): this is a
                    // real opt-in knob, not a safe-by-construction one.
                    let init_options = match execution_provider {
                        RerankExecutionProvider::Cpu => fastembed::RerankInitOptions::new(chosen),
                        RerankExecutionProvider::CoreMl => {
                            fastembed::RerankInitOptions::new(chosen)
                                .with_execution_providers(vec![ort::ep::CoreML::default().build()])
                        }
                    };
                    fastembed::TextRerank::try_new(init_options)
                })
                .await;
                let elapsed_secs = start.elapsed().as_secs_f64();
                match init_result {
                    Ok(Ok(model)) => {
                        metrics::histogram!(
                            "kremory.rerank.duration_seconds",
                            "phase" => "cold_start"
                        )
                        .record(elapsed_secs);
                        metrics::counter!("kremory.rerank.model_load_total", "outcome" => "ok")
                            .increment(1);
                        Ok(Arc::new(tokio::sync::Mutex::new(model)))
                    }
                    Ok(Err(load_err)) => {
                        metrics::counter!("kremory.rerank.model_load_total", "outcome" => "fail")
                            .increment(1);
                        Err(Error::Search(format!(
                            "fastembed reranker model load failed: {load_err}"
                        )))
                    }
                    Err(join_err) => {
                        metrics::counter!("kremory.rerank.model_load_total", "outcome" => "fail")
                            .increment(1);
                        Err(Error::Search(format!(
                            "fastembed reranker model load task panicked: {join_err}"
                        )))
                    }
                }
            })
            .await?;
        Ok(Arc::clone(model))
    }
}

impl Default for FastEmbedReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Reranker for FastEmbedReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.model().await?;
        let ids: Vec<String> = candidates.iter().map(|(id, _)| id.clone()).collect();
        let texts: Vec<String> = candidates.iter().map(|(_, text)| text.clone()).collect();
        let query = query.to_string();

        // `TextRerank::rerank<S: AsRef<str>>` unifies `S` across BOTH the
        // query and the document slice — passing `query.as_str()` (`&str`)
        // pins `S = &str`, so `documents` must be `&[&str]`, not `Vec<String>`.
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let warm_start = std::time::Instant::now();
        let rerank_results = {
            let mut guard = model.lock().await;
            guard
                .rerank(query.as_str(), text_refs, false, None)
                .map_err(|e| Error::Search(format!("fastembed rerank call failed: {e}")))?
        };
        metrics::histogram!("kremory.rerank.duration_seconds", "phase" => "warm_call")
            .record(warm_start.elapsed().as_secs_f64());

        let mut out = Vec::with_capacity(rerank_results.len());
        for r in &rerank_results {
            let id = ids.get(r.index).cloned().ok_or_else(|| {
                Error::Search(format!(
                    "fastembed rerank returned out-of-range candidate index {}",
                    r.index
                ))
            })?;
            out.push((id, r.score));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `KREMORY_RERANK_MODEL` resolution — pure, so it is testable without a
    /// model load (the `parse_rerank_k` precedent in `bin/kremory-http.rs`).
    ///
    /// The unset/empty/unknown cases all resolving to `BGERerankerBase` is the
    /// load-bearing assertion: it pins that the override is byte-identical to
    /// pre-override behaviour unless deliberately set, so enabling the knob
    /// cannot silently change an existing benchmark's model.
    #[test]
    fn parse_reranker_model_defaults_and_resolves_each_alias() {
        use fastembed::RerankerModel as M;
        assert_eq!(parse_reranker_model(None), M::BGERerankerBase, "unset");
        assert_eq!(parse_reranker_model(Some("")), M::BGERerankerBase, "empty");
        assert_eq!(parse_reranker_model(Some("bge-base")), M::BGERerankerBase);
        assert_eq!(parse_reranker_model(Some("bge-v2-m3")), M::BGERerankerV2M3);
        assert_eq!(
            parse_reranker_model(Some("jina-v1-turbo-en")),
            M::JINARerankerV1TurboEn
        );
        assert_eq!(
            parse_reranker_model(Some("jina-v2-multilingual")),
            M::JINARerankerV2BaseMultiligual
        );
        // Case- and whitespace-insensitive, and accepts the raw enum spelling —
        // a sweep script should not fail on " BGERerankerV2M3 ".
        assert_eq!(
            parse_reranker_model(Some("  BGERerankerV2M3  ")),
            M::BGERerankerV2M3
        );
    }

    /// An unrecognised value falls back to the default LOUDLY (a WARN is
    /// emitted alongside) rather than erroring the whole recall — but it must
    /// never silently resolve to some *other* real model, which would make a
    /// benchmark score a different reranker than its provenance stamp claims.
    #[test]
    fn parse_reranker_model_unknown_value_falls_back_to_default() {
        assert_eq!(
            parse_reranker_model(Some("cohere-rerank-v3")),
            fastembed::RerankerModel::BGERerankerBase
        );
    }

    /// Reranker latency lever 2 — `KREMORY_RERANK_EXECUTION_PROVIDER`
    /// resolution. Unset/empty/`cpu` all resolving to `Cpu` is the load-
    /// bearing assertion (mirrors `parse_reranker_model_defaults_and_
    /// resolves_each_alias`): it pins that the override is byte-identical to
    /// pre-override behaviour unless deliberately set to `coreml`.
    #[test]
    fn parse_rerank_execution_provider_defaults_and_resolves_each_alias() {
        assert_eq!(
            parse_rerank_execution_provider(None),
            RerankExecutionProvider::Cpu,
            "unset"
        );
        assert_eq!(
            parse_rerank_execution_provider(Some("")),
            RerankExecutionProvider::Cpu,
            "empty"
        );
        assert_eq!(
            parse_rerank_execution_provider(Some("cpu")),
            RerankExecutionProvider::Cpu
        );
        assert_eq!(
            parse_rerank_execution_provider(Some("coreml")),
            RerankExecutionProvider::CoreMl
        );
        assert_eq!(
            parse_rerank_execution_provider(Some("core-ml")),
            RerankExecutionProvider::CoreMl
        );
        assert_eq!(
            parse_rerank_execution_provider(Some("core_ml")),
            RerankExecutionProvider::CoreMl
        );
        // Case- and whitespace-insensitive, matching parse_reranker_model.
        assert_eq!(
            parse_rerank_execution_provider(Some("  CoreML  ")),
            RerankExecutionProvider::CoreMl
        );
    }

    /// An unrecognised value falls back to `Cpu` LOUDLY (a WARN is emitted
    /// alongside) rather than erroring the whole recall — mirrors
    /// `parse_reranker_model_unknown_value_falls_back_to_default`.
    #[test]
    fn parse_rerank_execution_provider_unknown_value_falls_back_to_cpu() {
        assert_eq!(
            parse_rerank_execution_provider(Some("tensorrt")),
            RerankExecutionProvider::Cpu
        );
    }

    /// Fast tier (test pyramid): a deterministic mock
    /// `Reranker` proving the trait's dyn-dispatch shape works end to end —
    /// zero model load, zero I/O.
    struct MockReranker {
        // Maps candidate id -> the score this mock returns for it.
        scores: std::collections::HashMap<String, f32>,
    }

    #[async_trait::async_trait]
    impl Reranker for MockReranker {
        async fn rerank(
            &self,
            _query: &str,
            candidates: &[(String, String)],
        ) -> Result<Vec<(String, f32)>> {
            let mut out: Vec<(String, f32)> = candidates
                .iter()
                .map(|(id, _)| (id.clone(), *self.scores.get(id).unwrap_or(&0.0)))
                .collect();
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(out)
        }
    }

    #[tokio::test]
    async fn dyn_reranker_reorders_via_mock() {
        let mock = MockReranker {
            scores: [("low".to_string(), 0.1), ("high".to_string(), 0.9)]
                .into_iter()
                .collect(),
        };
        let reranker: Arc<dyn Reranker> = Arc::new(mock);
        let candidates = vec![
            ("low".to_string(), "low relevance text".to_string()),
            ("high".to_string(), "high relevance text".to_string()),
        ];
        let out = reranker.rerank("query", &candidates).await.unwrap();
        assert_eq!(
            out.first().map(|(id, _)| id.as_str()),
            Some("high"),
            "mock reranker must reorder 'high' above 'low' despite input order"
        );
    }

    #[tokio::test]
    async fn empty_candidates_returns_empty_without_model_load() {
        let reranker = FastEmbedReranker::new();
        let out = reranker.rerank("query", &[]).await.unwrap();
        assert!(
            out.is_empty(),
            "empty candidates must short-circuit before touching the model"
        );
    }

    /// Seam tier (test pyramid): loads the ACTUAL BGE
    /// model and asserts it orders a known relevant/irrelevant pair
    /// correctly. Gated `#[ignore]` — model download/session-load is
    /// network+disk-bound (see `fastembed_rerank_spike` example); run
    /// explicitly with `--features rerank -- --ignored`, not on every
    /// `cargo test`.
    #[cfg(feature = "rerank")]
    #[tokio::test]
    #[ignore = "loads the real BGE reranker model — network + disk on first run"]
    async fn fastembed_reranker_orders_relevant_pair_correctly() {
        let reranker = FastEmbedReranker::new();
        let candidates = vec![
            (
                "france".to_string(),
                "Paris is the capital of France.".to_string(),
            ),
            (
                "germany".to_string(),
                "Berlin is a city in Germany.".to_string(),
            ),
        ];
        let out = reranker
            .rerank("What is the capital of France?", &candidates)
            .await
            .unwrap();
        let france_score = out
            .iter()
            .find(|(id, _)| id == "france")
            .map(|(_, s)| *s)
            .expect("france candidate must be present");
        let germany_score = out
            .iter()
            .find(|(id, _)| id == "germany")
            .map(|(_, s)| *s)
            .expect("germany candidate must be present");
        assert!(
            france_score > germany_score,
            "BGE reranker must score the France passage above the Germany \
             passage for a France-capital query: france={france_score}, \
             germany={germany_score}"
        );
    }
}

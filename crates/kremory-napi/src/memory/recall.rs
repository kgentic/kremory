//! Read paths: lookup by source id, prompt-text templating, and the main
//! `recall` retrieval entry point.

use napi_derive::napi;
use super::JsMemory;
use crate::convert;
use crate::convert::*;
use kremory::Namespace;

#[napi]
impl JsMemory {
    /// Direct slug/source_id lookup — returns episodes matching `source_id`.
    /// v0.1.8 1:1 contract (renamed from `get_by_source_id`).
    ///
    /// Results are ordered newest-first. When `namespace` is omitted, the
    /// Memory handle's default namespace is used; when the handle has no default,
    /// results span all namespaces.
    ///
    /// `sourceUri` on each returned `JsEpisode` reflects the current DB value
    /// (round-trips correctly, incl. after `updateSourceUri`) — the substrate
    /// `recall_by_source_id` query projects `source_uri`.
    #[napi]
    pub async fn recall_by_source_id(
        &self,
        source_id: String,
        namespace: Option<String>,
    ) -> napi::Result<Vec<JsEpisode>> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let episodes = self
            .inner
            .recall_by_source_id(source_id.clone(), ns)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory recallBySourceId failed: {e}"))
            })?;

        Ok(episodes.into_iter().map(convert::episode_to_js).collect())
    }

    /// Search memory and return kremory's **prompt-ready rendering** of the
    /// results — the string form built for an LLM consumer, rather than the
    /// structured rows [`recall`] returns.
    ///
    /// Previously the binding hardcoded `.raw()`, so a Node
    /// consumer could not reach this rendering at all — even though the Rust
    /// facade has always offered both terminals and the MCP tool surface
    /// DEFAULTS to it. Measured worth on LoCoMo: **~+4pt answerability, +10.8pt
    /// on temporal questions**, on identical retrieval — the graph's output is
    /// far more usable when an agent can tell facts from source text.
    ///
    /// Mirrors the substrate's terminal shape deliberately (two methods, not a
    /// `format` enum): the CONSUMER chooses by picking a terminal, exactly as in
    /// Rust. No default is imposed on either side.
    ///
    /// `template` selects the rendering: `"temporal_facts"` (default — facts
    /// with their `valid_at` annotations), `"entities"`, or `"edge_summary"`.
    /// An unrecognised value is rejected rather than silently substituted.
    #[napi]
    pub async fn recall_as_prompt_text(
        &self,
        query: String,
        template: Option<String>,
        opts: Option<JsRecallOptions>,
    ) -> napi::Result<String> {
        let tmpl = match template.as_deref() {
            None | Some("temporal_facts") => kremory::RecallTemplate::TemporalFacts,
            Some("entities") => kremory::RecallTemplate::Entities,
            Some("edge_summary") => kremory::RecallTemplate::EdgeSummary,
            Some(other) => {
                return Err(napi::Error::from_reason(format!(
                    "unknown template {other:?} — expected one of: temporal_facts, \
                     entities, edge_summary"
                )))
            }
        };
        let namespace =
            convert::resolve_recall_namespace(&opts).or_else(|| self.default_namespace.clone());
        let k = opts
            .as_ref()
            .and_then(|o| o.k)
            .map(|k_val| usize::try_from(k_val).unwrap_or(10));

        let mut builder = self.inner.recall(query);
        if let Some(ns) = namespace {
            builder = builder.in_namespace(ns);
        }
        if let Some(k_val) = k {
            builder = builder.k(k_val);
        }
        if let Some(n) = opts
            .as_ref()
            .and_then(|o| o.rerank_k)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
        {
            builder = builder.rerank_k(n);
        }
        builder
            .as_template(tmpl)
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory recall failed: {e}")))
    }

    /// Search memory for context matching `query`.
    ///
    /// Returns up to `opts.k` (default 10) results ranked by relevance.
    #[napi]
    pub async fn recall(
        &self,
        query: String,
        opts: Option<JsRecallOptions>,
    ) -> napi::Result<Vec<JsRetrievedContext>> {
        let namespaces = convert::resolve_recall_namespaces(&opts);
        // Apply handle-level default ONLY when neither per-call selector is set.
        // If `in_namespaces` is set, we must NOT also inject a default — that
        // would trip `ConflictingNamespaceSelectors`.
        let namespace = convert::resolve_recall_namespace(&opts).or_else(|| {
            if namespaces.is_none() {
                self.default_namespace.clone()
            } else {
                None
            }
        });
        let best_effort = opts.as_ref().and_then(|o| o.best_effort);
        let per_namespace_top_k = opts
            .as_ref()
            .and_then(|o| o.per_namespace_top_k)
            .map(|n| usize::try_from(n).unwrap_or(10));
        let k = opts
            .as_ref()
            .and_then(|o| o.k)
            .map(|k_val| usize::try_from(k_val).unwrap_or(10));
        let as_of = opts
            .as_ref()
            .and_then(|o| o.as_of.as_deref())
            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok());
        // A negative or zero `rerankK` is dropped rather than coerced
        // to a default — `rerank_k` takes `usize`, and silently reranking at
        // some invented depth would be the "silent default" this project bans
        // on parsed input. Omit the knob and you get no rerank, which is the
        // same outcome as omitting the field.
        let rerank_k = opts
            .as_ref()
            .and_then(|o| o.rerank_k)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0);

        let results = {
            let mut builder = self.inner.recall(query);

            // Mutual exclusion is enforced at `.await` time by kremory's
            // `check_selectors`; surface both if caller sets both so the
            // ConflictingNamespaceSelectors error propagates naturally.
            if let Some(ns) = namespace {
                builder = builder.in_namespace(ns);
            }
            if let Some(ref nss) = namespaces {
                builder = builder.in_namespaces(nss);
            }
            if let Some(b) = best_effort {
                builder = builder.best_effort(b);
            }
            if let Some(n) = per_namespace_top_k {
                builder = builder.per_namespace_top_k(n);
            }
            if let Some(k_val) = k {
                builder = builder.k(k_val);
            }
            if let Some(as_of_ts) = as_of {
                builder = builder.as_of(as_of_ts);
            }
            if let Some(n) = rerank_k {
                builder = builder.rerank_k(n);
            }

            // B9: wire filterMetadata entries. Each entry maps to one
            // RecallRequest::filter_metadata(key, value) call. Validation
            // (key length, JSON-path metachars) is enforced by the substrate
            // at await time, surfacing as an Err from builder.raw().await.
            if let Some(filters) = opts.as_ref().and_then(|o| o.filter_metadata.as_ref()) {
                for f in filters {
                    builder = builder.filter_metadata(&f.key, f.value.clone());
                }
            }

            builder
                .raw()
                .await
                .map_err(|e| napi::Error::from_reason(format!("kremory recall failed: {e}")))?
        };

        Ok(results
            .into_iter()
            .map(convert::retrieved_context_to_js)
            .collect())
    }

}

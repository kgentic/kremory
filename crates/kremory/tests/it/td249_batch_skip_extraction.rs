#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-249 — `remember_batch()` must support `skip_extraction()`, so bulk import
//! works without an LLM.
//!
//! The batch path hard-coded `enrich_per_episode: true`, so importing existing
//! data required an LLM provider even when every triple was pinned via
//! `.with_facts(..)` — the exact combination the single-episode path supports,
//! and the one the error message itself recommends:
//!
//! > requires an LLM provider — ... or use .with_facts(…) to pin triples without LLM
//!
//! Bulk import is the first thing a real adopter does, so the gap sat on the
//! busiest on-ramp in the crate. Found by writing the bulk-import example.

use std::sync::Arc;

use kremory::{
    CoreResult, DynEmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace, StructuredFact,
};

/// Never invoked — every entry calls `.skip_extraction()`. Present only because
/// the builder requires an extractor to reach a buildable state without an LLM.
struct NoExtraction;

impl EntityExtractor for NoExtraction {
    fn name(&self) -> &'static str {
        "no-extraction"
    }

    async fn extract(
        &self,
        _text: &str,
        _ctx: &ExtractionContext<'_>,
    ) -> CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: Vec::new(),
            facts: Vec::new(),
        })
    }
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

#[tokio::test]
async fn batch_import_works_without_an_llm() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("td249");

    // NOTE: no `.with_llm(..)`. That is the point of the test.
    let mem = Memory::open(dir.path().join("t.db"))
        .default_namespace(ns.clone())
        .with_embedder(null_embedder())
        .with_extractor(Arc::new(NoExtraction))
        .await
        .expect("open without an LLM");

    let rows = [("acme", "invoicing"), ("globex", "sso"), ("initech", "export")];

    let mut batch = mem.remember_batch();
    for (customer, area) in rows {
        batch = batch
            .entry(format!("{customer} reported a problem with {area}."))
            .in_namespace(ns.clone())
            .with_facts(vec![StructuredFact {
                subject: customer.into(),
                predicate: "reported_issue_with".into(),
                object: area.into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .skip_extraction()
            .done();
    }

    let commits = batch
        .await
        .expect("TD-249: a pinned-fact batch must not require an LLM provider");

    assert_eq!(
        commits.len(),
        rows.len(),
        "every submitted entry must come back with a commit — a silent drop makes \
         an import look successful while losing data"
    );

    // The round trip matters more than the write returning Ok.
    let areas: Vec<String> = mem
        .recall("acme")
        .in_namespace(ns.clone())
        .raw()
        .await
        .expect("recall")
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.subject == "acme")
        .map(|f| f.object)
        .collect();

    assert!(
        areas.iter().any(|a| a == "invoicing"),
        "batch-imported facts must be recallable; got {areas:?}"
    );

    mem.close().await.expect("close");
}

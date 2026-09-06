//! TD-187 round 2 — does a per-fact `valid_at` actually reach `facts.valid_from`,
//! and does a fact WITHOUT one still fall back to the episode's time?
//!
//! The unit tests next to `parse_facts` prove the boundary parse. They prove
//! nothing about persistence: `ExtractedFact.valid_at` has to survive the ingest
//! pipeline and land in the column `as_of()` filters on (ADR-068). This test
//! drives the real `Memory::remember(...).published_at(ts)` path with a stub
//! extractor, then reads the dates back through the public `recall(...).raw()`
//! surface — no test-only shortcuts, no direct DB poke.
//!
//! # Why the assertion is a DIFFERENCE, not a value
//!
//! Before this change both fact-insert sites wrote `valid_from: ref_time`
//! unconditionally, so EVERY fact from one episode carried the SAME timestamp.
//! Asserting only "the dated fact has date X" would pass on a build where the
//! fallback silently overwrote it with an equal-looking value. Asserting the two
//! facts DIFFER is the property that was impossible before and is the whole point
//! of the field — it cannot pass vacuously against the old behaviour.
//!
//! # Scope note (verified, not assumed)
//!
//! This exercises the ENGINE-INLINE path, which is what a normal consumer drives
//! (see the reasoning in `td187_published_at_wiring.rs`). The separate OS-thread
//! `BackgroundIngestor` deferred path constructs its request with
//! `reference_time: None`, so it renders no date block and therefore yields no
//! `valid_at` — its fallback to `ref_time` is exercised by the undated fact here.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use kremory::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::{DynEmbeddingProvider, MockChatProvider};
use kremory::{Memory, Namespace};

/// Emits two facts about the same subject: one carrying an explicit `valid_at`
/// well before the document date, one carrying none.
struct TwoFactsOneDatedExtractor;

impl EntityExtractor for TwoFactsOneDatedExtractor {
    fn name(&self) -> &'static str {
        "td187-two-facts-one-dated"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Caroline".to_string(),
                    label: "Person".to_string(),
                    properties: serde_json::json!({"name": "Caroline"}),
                },
                ExtractedEntity {
                    name: "Pride fest".to_string(),
                    label: "Event".to_string(),
                    properties: serde_json::json!({"name": "Pride fest"}),
                },
            ],
            facts: vec![
                // DATED — the model resolved "last year" against the document date.
                ExtractedFact {
                    subject: "Caroline".to_string(),
                    predicate: "attended".to_string(),
                    object: "Pride fest".to_string(),
                    is_entity_ref: true,
                    confidence: 1.0,
                    valid_at: Some(Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap()),
                },
                // UNDATED — the text stated no resolvable time. Must fall back.
                ExtractedFact {
                    subject: "Caroline".to_string(),
                    predicate: "knows about".to_string(),
                    object: "Pride fest".to_string(),
                    is_entity_ref: true,
                    confidence: 1.0,
                    valid_at: None,
                },
            ],
        })
    }
}

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td187_valid_from_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dated_fact_keeps_its_own_valid_from_while_undated_falls_back() {
    let published = Utc.with_ymd_and_hms(2023, 8, 17, 0, 0, 0).unwrap();
    let fact_date = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();

    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 });
    let mem = Memory::open(unique_db("per_fact"))
        .with_llm(Arc::new(MockChatProvider::null()))
        .with_embedder(embedder)
        .with_extractor(Arc::new(TwoFactsOneDatedExtractor))
        .default_namespace(Namespace::new("td187valid"))
        .await
        .expect("Memory::open must succeed");

    mem.remember("Caroline talked about the Pride fest.")
        .published_at(published)
        .await
        .expect("remember must succeed");

    let ctx = mem
        .recall("Caroline")
        .raw()
        .await
        .expect("recall must succeed");

    let facts: Vec<_> = ctx.iter().flat_map(|c| c.facts.iter()).collect();

    // NON-VACUITY GUARD. Without this, every assertion below passes over an empty
    // vec and the test reports green against a pipeline that persisted nothing.
    assert!(
        facts.len() >= 2,
        "expected both facts back, got {} — the test cannot measure anything without them",
        facts.len()
    );

    let dated = facts
        .iter()
        .find(|f| f.predicate == "attended")
        .expect("the dated fact must come back");
    let undated = facts
        .iter()
        .find(|f| f.predicate == "knows about")
        .expect("the undated fact must come back");

    assert_eq!(
        dated.valid_at, fact_date,
        "a fact carrying its own valid_at must persist THAT date, not the document's"
    );
    assert_eq!(
        undated.valid_at, published,
        "a fact with no valid_at must fall back to the episode reference time — \
         this is what keeps the change strictly additive"
    );

    // The property that was IMPOSSIBLE before this change: two facts from one
    // episode holding two different world times. If this ever collapses, the
    // per-fact plumbing has regressed to document granularity.
    assert_ne!(
        dated.valid_at, undated.valid_at,
        "two facts from one episode must be able to hold DIFFERENT world times; \
         equal values mean valid_from has regressed to one-date-per-document"
    );
}

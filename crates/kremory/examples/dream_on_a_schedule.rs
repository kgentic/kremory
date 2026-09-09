//! **Runs offline.** Let consolidation run on its own while your application
//! does its actual job.
//!
//! ```text
//! cargo run --example dream_on_a_schedule
//! ```
//!
//! ## The problem this solves
//!
//! A graph built from a stream of text accumulates duplicates, aliases and
//! superseded facts. `dream()` cleans that up — but calling it by hand means
//! either a cron job outside your process, or remembering to call it, which is
//! the same as not doing it.
//!
//! `start_dream_scheduler` runs it inside your process on a policy you choose,
//! and hands back a handle so shutdown is orderly rather than a killed task.
//!
//! ## Three policies
//!
//! - `Off` — the default. Manual passes only.
//! - `Interval(d)` — after each `d` since the previous pass COMPLETED, so a slow
//!   pass cannot stack up behind itself.
//! - `EveryNIngests(n)` — after every `n` successful ingests. Work-proportional
//!   rather than clock-proportional, which is usually what you want for a bursty
//!   feed.
//!
//! ## What this asserts
//!
//! The lifecycle, not the consolidation outcome. Whether a given pass merges
//! anything depends entirely on the graph — see `agent_memory_with_ollama`,
//! where a deliberately small graph correctly merges nothing. What matters here
//! is that the scheduler starts, coexists with ongoing writes, and **stops when
//! told**, because a background task you cannot stop is a shutdown bug.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use kremory::{
    CoreResult, DreamSchedule, EmbeddingProvider, EntityExtractor, ExtractionContext,
    ExtractionResult, Memory, Namespace, StructuredFact,
};

const DEMO_DIM: usize = 16;

/// Deterministic stand-in embedder — see `offline_remember_recall.rs`.
struct DemoEmbedder {
    dim: usize,
}

impl EmbeddingProvider for DemoEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move {
            let mut v = vec![0f32; dim];
            for (i, b) in text.bytes().enumerate() {
                v[i % dim] += f32::from(b) / 255.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            Ok(v)
        }
    }
}

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("feed");

    let mem = Memory::open(dir.path().join("feed.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // A short interval so the example finishes quickly. In production this is
    // minutes or hours — consolidation is maintenance, not a hot path.
    let handle = mem.start_dream_scheduler(DreamSchedule::Interval(Duration::from_millis(200)));
    println!("scheduler started: a pass every 200ms");

    // ── Meanwhile, the application keeps working ────────────────────────────
    //
    // The point of a scheduler is that this loop does not have to know it
    // exists. No coordination, no pausing writes around a maintenance window.
    for i in 1..=6 {
        mem.remember(format!("Event {i} was recorded by the ingest worker."))
            .in_namespace(ns.clone())
            .with_facts(vec![StructuredFact {
                subject: format!("event-{i}"),
                predicate: "recorded_by".into(),
                object: "ingest worker".into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .skip_extraction()
            .await?;
        tokio::time::sleep(Duration::from_millis(80)).await;
    }
    println!("wrote 6 events while the scheduler ran alongside");

    // ── Orderly shutdown ────────────────────────────────────────────────────
    //
    // `stop()` consumes the handle and awaits the task, so when it returns the
    // scheduler is genuinely finished rather than merely signalled. That is what
    // makes it safe to close the database on the next line.
    handle.stop().await;
    println!("scheduler stopped cleanly");

    // The data written during the scheduler's lifetime is intact — a
    // consolidation pass running concurrently with writes must not eat them.
    let events: Vec<String> = mem
        .recall("event")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.predicate == "recorded_by")
        .map(|f| f.subject)
        .collect();

    println!("events still present: {}", events.len());
    assert!(
        !events.is_empty(),
        "writes made while the scheduler was running must survive it — a \
         concurrent consolidation pass must not eat live data"
    );

    // Closing after stop() is the correct order. Reverse it and you are closing
    // the database out from under a running background task.
    mem.close().await?;

    println!("\nConsolidation ran on its own, writes continued throughout, and shutdown");
    println!("was orderly. Use Interval for a steady trickle and EveryNIngests for a");
    println!("bursty feed, where work-proportional beats clock-proportional.");
    Ok(())
}

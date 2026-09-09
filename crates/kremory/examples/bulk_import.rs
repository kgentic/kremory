//! **Runs offline.** You already have data. Get it in.
//!
//! ```text
//! cargo run --example bulk_import
//! ```
//!
//! ## The problem this solves
//!
//! Nobody adopts a memory system on an empty database. The first real task is
//! always "here are 10,000 rows we already have" — support tickets, CRM notes, a
//! chat export.
//!
//! Looping `remember()` works but pays the per-call cost every time.
//! `remember_batch()` submits many episodes as one unit and hands back one
//! `EpisodeCommit` per entry, so you can map results back to your source rows.
//!
//! ## What to check when you import
//!
//! The assertion below is the one that matters and the one people skip: **every
//! row you submitted came back**. A batch that silently drops entries is the
//! worst failure mode here, because the import "succeeds" and you discover the
//! gap months later when something cannot be recalled.

use std::future::Future;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace, StructuredFact,
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

/// Rows from the system you are migrating off. `(ticket id, customer, product)`.
const TICKETS: &[(&str, &str, &str)] = &[
    ("T-1001", "acme", "invoicing"),
    ("T-1002", "globex", "sso"),
    ("T-1003", "initech", "invoicing"),
    ("T-1004", "acme", "export"),
    ("T-1005", "hooli", "sso"),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("tickets");

    let mem = Memory::open(dir.path().join("import.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // ── One batch, many entries ─────────────────────────────────────────────
    //
    // Each `.entry(..)` opens an episode; `.done()` closes it and returns you to
    // the batch. Awaiting the batch submits the lot.
    let mut batch = mem.remember_batch();
    for (id, customer, area) in TICKETS {
        batch = batch
            .entry(format!("Ticket {id}: {customer} reported a problem with {area}."))
            .in_namespace(ns.clone())
            .with_facts(vec![StructuredFact {
                subject: (*customer).into(),
                predicate: "reported_issue_with".into(),
                object: (*area).into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .skip_extraction()
            .done();
    }

    let commits = batch.await?;
    println!("submitted {} tickets, got {} commits", TICKETS.len(), commits.len());

    // THE assertion people skip. A silent drop here is discovered months later.
    assert_eq!(
        commits.len(),
        TICKETS.len(),
        "every submitted row must come back with a commit — a batch that quietly \
         drops entries makes the import look successful while losing data"
    );

    // ── And the imported data is actually usable ────────────────────────────
    //
    // Import is not done when the write returns; it is done when you can get the
    // data back out. Checking one customer proves the round trip.
    let acme_areas: Vec<String> = mem
        .recall("acme")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.subject == "acme" && f.predicate == "reported_issue_with")
        .map(|f| f.object)
        .collect();

    println!("acme reported issues with: {acme_areas:?}");

    let expected: Vec<&str> = TICKETS
        .iter()
        .filter(|(_, c, _)| *c == "acme")
        .map(|(_, _, a)| *a)
        .collect();

    for area in &expected {
        assert!(
            acme_areas.iter().any(|a| a == area),
            "imported ticket area {area:?} should be recallable for acme; got {acme_areas:?}"
        );
    }

    println!("\nAll {} rows landed and are queryable. Each entry got its own commit,", TICKETS.len());
    println!("so you can map results back to the source rows rather than guessing");
    println!("which of a thousand imports failed.");

    mem.close().await?;
    Ok(())
}

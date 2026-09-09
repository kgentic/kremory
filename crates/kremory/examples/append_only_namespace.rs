//! **Runs offline.** A namespace where the record cannot be rewritten, and the
//! system refuses rather than trusting you to remember.
//!
//! ```text
//! cargo run --example append_only_namespace
//! ```
//!
//! ## The problem this solves
//!
//! Some records must not be edited after the fact — a regulated audit trail, a
//! consent log, anything a court might read. "We agreed not to delete from that
//! namespace" is a convention, and conventions are what people forget at 2am
//! during an incident.
//!
//! `upgrade_namespace_policy` makes it structural. An append-only namespace
//! REFUSES mutating operations, and the refusal is a typed error you can match
//! on rather than a string you hope nobody ignores.
//!
//! ## The half that is usually skipped
//!
//! Proving a guard blocks the thing it should is only half a test. This also
//! proves it does NOT block ordinary work — writes still succeed, reads still
//! work — because a guard that fires on everything gets disabled, and then it
//! protects nothing at all.
//!
//! The upgrade is deliberately ONE-WAY: `AppendOnly → Mutable` is refused. A
//! protection you can quietly switch off is not a protection.

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let audit = Namespace::new("consent-log");
    let scratch = Namespace::new("working-notes");

    let mem = Memory::open(dir.path().join("regulated.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(scratch.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // Write the record BEFORE locking the namespace.
    mem.remember("Dana consented to marketing contact on 2026-03-01.")
        .in_namespace(audit.clone())
        .from_document("consent-2026-03-01")
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "consented_to".into(),
            object: "marketing contact".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await?;

    // ── Lock it ─────────────────────────────────────────────────────────────
    mem.upgrade_namespace_policy(audit.clone()).await?;
    println!("consent-log is now append-only");

    // ── It must REFUSE to erase ─────────────────────────────────────────────
    let refused = mem
        .forget()
        .in_namespace(audit.clone())
        .by_source_id("consent-2026-03-01")
        .execute()
        .await;

    match &refused {
        Err(e) => println!("  erase attempt   → refused: {e}"),
        Ok(_) => println!("  erase attempt   → SUCCEEDED (this is the bug)"),
    }
    assert!(
        refused.is_err(),
        "an append-only namespace must refuse forget() — if this succeeds the \
         guarantee is decorative"
    );

    // ── And the record is still there ───────────────────────────────────────
    //
    // A refusal that still half-applied would be worse than no guard.
    let survived: Vec<String> = mem
        .recall("dana")
        .in_namespace(audit.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| f.predicate)
        .collect();
    assert!(
        survived.iter().any(|p| p == "consented_to"),
        "the refused erase must leave the record intact; got {survived:?}"
    );
    println!("  record intact   → {survived:?}");

    // ── The lock is ONE-WAY ─────────────────────────────────────────────────
    //
    // Re-upgrading is a no-op (idempotent), but there is no downgrade at all —
    // the API simply does not offer one, which is stronger than offering one
    // that errors.
    mem.upgrade_namespace_policy(audit.clone()).await?;
    println!("  re-upgrade      → idempotent, no error");

    // ── And ordinary work is UNAFFECTED elsewhere ───────────────────────────
    //
    // This is the half people skip. A guard that blocks everything gets turned
    // off, and then it guards nothing.
    mem.remember("Rough notes about the Q2 campaign.")
        .in_namespace(scratch.clone())
        .with_facts(vec![StructuredFact {
            subject: "q2 campaign".into(),
            predicate: "status".into(),
            object: "draft".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await?;

    let erased_ok = mem
        .forget()
        .in_namespace(scratch.clone())
        .by_source_id("nonexistent-source")
        .execute()
        .await;
    assert!(
        erased_ok.is_ok(),
        "a MUTABLE namespace must still allow forget() — over-blocking is a \
         failure, not caution: got {erased_ok:?}"
    );
    println!("  mutable ns      → still fully writable and erasable");

    println!("\nThe guarantee is structural, not a convention someone has to remember.");
    println!("And it is scoped: the namespace next door is untouched.");

    mem.close().await?;
    Ok(())
}

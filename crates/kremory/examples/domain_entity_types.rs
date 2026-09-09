//! **Runs offline.** Your entities are not people and companies. Teach the
//! vocabulary.
//!
//! ```text
//! cargo run --example domain_entity_types
//! ```
//!
//! ## The problem this solves
//!
//! The default entity vocabulary is general-purpose — person, organisation,
//! place and so on. That is right for an assistant and wrong for a domain. A
//! legal system has courts and statutes; a warehouse has SKUs and pallets. Force
//! those through "organisation" and every downstream filter becomes a guess.
//!
//! `register_namespace_with_seed` lets a namespace declare its own vocabulary,
//! which the extractor then has available when it types what it finds.
//!
//! ## Three modes, and the ids matter
//!
//! - `Default` — the general vocabulary. Status quo.
//! - `Augment(specs)` — general vocabulary PLUS yours. **Use ids ≥ 10**; 0–9 are
//!   reserved for the defaults, and colliding would redefine them.
//! - `Replace(specs)` — yours only. **Greenfield only** — see below.
//!
//! ## The guard worth seeing fire
//!
//! `Replace` on a namespace that already has DIFFERENT types fails loudly with
//! no database write, rather than silently re-typing a populated graph. That is
//! the difference between a schema change and data corruption, and this example
//! triggers it deliberately.

use std::future::Future;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, EntityTypeSpec, ExtractionContext,
    ExtractionResult, Memory, Namespace, NamespaceSeed, SeedOutcome,
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

/// The vocabulary a legal-research consumer actually needs. Ids start at 10 —
/// 0–9 belong to the general defaults.
fn legal_vocabulary() -> Vec<EntityTypeSpec> {
    vec![
        EntityTypeSpec {
            id: 10,
            name: "Court".into(),
            description: "A court, tribunal, or judicial body.".into(),
        },
        EntityTypeSpec {
            id: 11,
            name: "Statute".into(),
            description: "An act, regulation, or piece of primary legislation.".into(),
        },
        EntityTypeSpec {
            id: 12,
            name: "CaseCitation".into(),
            description: "A reported decision, cited in the usual form.".into(),
        },
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let legal = Namespace::new("case-law");

    let mem = Memory::open(dir.path().join("legal.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(legal.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // ── Augment: keep the general types, add the domain ones ────────────────
    let outcome = mem
        .register_namespace_with_seed(legal.clone(), NamespaceSeed::Augment(legal_vocabulary()))
        .await?;
    println!("seeding 'case-law' : {outcome:?}");
    assert!(
        matches!(outcome, SeedOutcome::Seeded { .. }),
        "a fresh namespace should report Seeded; got {outcome:?}"
    );

    // ── Idempotent ──────────────────────────────────────────────────────────
    //
    // Re-registering is safe, which matters because this usually lives in
    // application startup and therefore runs on every boot.
    let again = mem
        .register_namespace_with_seed(legal.clone(), NamespaceSeed::Augment(legal_vocabulary()))
        .await?;
    println!("seeding again      : {again:?}");
    assert!(
        matches!(again, SeedOutcome::AlreadySeeded),
        "re-seeding a populated namespace must be a no-op, not a duplicate \
         write; got {again:?}"
    );

    // ── The guard: Replace on a populated namespace with DIFFERENT types ────
    //
    // This is the one worth watching. `Replace` would redefine the vocabulary a
    // populated graph is already typed against, so it refuses — and refuses
    // WITHOUT writing, rather than half-applying and leaving a graph typed
    // against two vocabularies at once.
    let refused = mem
        .register_namespace_with_seed(
            legal.clone(),
            NamespaceSeed::Replace(vec![EntityTypeSpec {
                id: 20,
                name: "Pallet".into(),
                description: "Completely unrelated vocabulary.".into(),
            }]),
        )
        .await;

    match &refused {
        Err(e) => println!("\nReplace on populated: refused — {e:?}"),
        Ok(o) => println!("\nReplace on populated: SUCCEEDED ({o:?}) — this is the bug"),
    }
    assert!(
        refused.is_err(),
        "Replace must refuse on a populated namespace whose types differ — \
         silently re-typing a live graph is data corruption, not a migration"
    );

    // ── And a DIFFERENT namespace is unaffected ─────────────────────────────
    //
    // The vocabulary is per-namespace, so the warehouse next door can have its
    // own without either side knowing.
    let warehouse = Namespace::new("logistics");
    let other = mem
        .register_namespace_with_seed(
            warehouse.clone(),
            NamespaceSeed::Augment(vec![EntityTypeSpec {
                id: 10,
                name: "Sku".into(),
                description: "A stock-keeping unit.".into(),
            }]),
        )
        .await?;
    println!("seeding 'logistics': {other:?}");
    assert!(
        matches!(other, SeedOutcome::Seeded { .. }),
        "a different namespace should seed independently; got {other:?}"
    );

    println!("\nId 10 means Court in 'case-law' and Sku in 'logistics'. Vocabulary is");
    println!("per-namespace, so one database serves domains that share no language —");
    println!("and Replace refuses to redefine a vocabulary a populated graph relies on.");

    mem.close().await?;
    Ok(())
}

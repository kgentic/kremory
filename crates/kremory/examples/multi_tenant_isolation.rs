//! **Runs offline.** One database, many customers, and no leakage between them.
//!
//! ```text
//! cargo run --example multi_tenant_isolation
//! ```
//!
//! ## The problem this solves
//!
//! You are building an assistant that serves several customers from one process.
//! Acme's facts must never surface in a Globex answer. Getting that wrong is not a
//! bug report, it is a disclosure incident.
//!
//! kremory's unit of isolation is the **namespace**. It is not a filter applied
//! after the fact — it scopes the query itself, so an omitted namespace cannot
//! silently widen a result set.
//!
//! ## What this example asserts, not just prints
//!
//! It writes facts for two tenants into ONE database and then proves, with hard
//! assertions rather than eyeballing, that:
//!
//!   - each tenant sees its own facts,
//!   - neither tenant sees the other's,
//!   - the isolation holds for a term that appears in BOTH tenants' data
//!     (the interesting case — a shared word is where a naive filter leaks).
//!
//! If a future change breaks scoping, this example stops exiting 0.

use std::future::Future;
use std::sync::Arc;

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace, StructuredFact};

const DEMO_DIM: usize = 16;

/// Deterministic stand-in embedder — see `offline_remember_recall.rs`.
/// Not for production; it hashes bytes rather than understanding meaning.
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

/// Never invoked — every write here calls `.skip_extraction()`. See
/// `offline_remember_recall.rs` for why the builder still requires one.
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

/// Everything one tenant said, and the answer that must never cross the boundary.
struct Tenant {
    ns: Namespace,
    subject: &'static str,
    /// Deliberately the same PREDICATE for both tenants — a shared term is exactly
    /// where a filter-after-the-fact implementation leaks.
    secret_tool: &'static str,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;

    let mem = Memory::open(dir.path().join("saas.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(Namespace::new("unused-default"))
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    let tenants = [
        Tenant {
            ns: Namespace::new("acme"),
            subject: "acme",
            secret_tool: "Postgres",
        },
        Tenant {
            ns: Namespace::new("globex"),
            subject: "globex",
            secret_tool: "ClickHouse",
        },
    ];

    // ── Write both tenants into the SAME database ───────────────────────────
    for t in &tenants {
        mem.remember(format!("Internal notes for {}.", t.subject))
            .in_namespace(t.ns.clone())
            .with_facts(vec![StructuredFact {
                subject: t.subject.into(),
                // Same predicate on both sides — the shared-term case.
                predicate: "runs_on".into(),
                object: t.secret_tool.into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .skip_extraction()
            .await?;
    }

    // ── Prove isolation, per tenant ─────────────────────────────────────────
    for t in &tenants {
        let objects: Vec<String> = mem
            .recall("what does the company run on")
            .in_namespace(t.ns.clone())
            .raw()
            .await?
            .into_iter()
            .flat_map(|c| c.facts)
            .map(|f| f.object)
            .collect();

        println!("{:>7} sees: {:?}", t.subject, objects);

        // Its own fact is present …
        assert!(
            objects.iter().any(|o| o == t.secret_tool),
            "{} should see its own fact {:?}, got {objects:?}",
            t.subject,
            t.secret_tool
        );

        // … and no other tenant's is.
        for other in &tenants {
            if other.subject == t.subject {
                continue;
            }
            assert!(
                !objects.iter().any(|o| o == other.secret_tool),
                "LEAK: {} saw {}'s fact {:?} — got {objects:?}",
                t.subject,
                other.subject,
                other.secret_tool
            );
        }
    }

    println!("\nBoth tenants live in one file and neither can see the other.");
    println!("The namespace scopes the QUERY — it is not a filter applied afterwards,");
    println!("so forgetting to pass one cannot silently widen a result set.");

    mem.close().await?;
    Ok(())
}

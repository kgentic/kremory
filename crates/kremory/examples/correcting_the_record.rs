//! **Runs offline.** A user tells you something has changed. You correct the
//! record without destroying what was true before.
//!
//! ```text
//! cargo run --example correcting_the_record
//! ```
//!
//! ## The problem this solves
//!
//! "Dana works at Acme" was true. Today she moved to Globex. A cache would
//! overwrite the old value and lose it. A log would append and leave you with two
//! contradictory answers and no way to tell which is current.
//!
//! kremory closes the old fact on the **world clock** instead: it stops being
//! true *as of a moment*, while remaining answerable for questions about the
//! past. Nothing is deleted.
//!
//! ## Two ways to correct a record — this shows the explicit one
//!
//! kremory can DETECT contradictions itself during ingest, but that requires a
//! language model (`TwoPoolDetector` takes a `ChatProvider`), so it belongs in
//! the Ollama example rather than here.
//!
//! This is the other path, and the one most applications actually want: the user
//! told you directly, so you correct it deliberately with `supersede()` rather
//! than hoping a model infers it. No model, no ambiguity, no cost.
//!
//! ## Note what makes this possible
//!
//! `supersede()` needs a `fact_id`, and until recently NOTHING on the public
//! surface returned one — the method was documented and uncallable (TD-244,
//! found by writing these examples). `recall().raw()` now carries `fact_id`,
//! which is what the code below reads.

use std::future::Future;
use std::sync::Arc;

use chrono::Utc;
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

/// Never invoked — every write here calls `.skip_extraction()`.
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

/// Every (predicate, object) currently true for Dana.
async fn current(mem: &Memory, ns: &Namespace) -> anyhow::Result<Vec<(String, String)>> {
    Ok(mem
        .recall("dana")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.predicate == "works_at")
        .map(|f| (f.predicate.clone(), f.object.clone()))
        .collect())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("crm");

    let mem = Memory::open(dir.path().join("crm.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // ── What we believed yesterday ──────────────────────────────────────────
    mem.remember("Dana joined Acme.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "works_at".into(),
            object: "Acme".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await?;

    println!("before the correction : {:?}", current(&mem, &ns).await?);

    // ── The correction ──────────────────────────────────────────────────────
    //
    // Find the fact to close. `fact_id` is the handle `supersede` takes; it is
    // `Option` because a content-search passage has no fact row behind it.
    let stale = mem
        .recall("dana")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == "works_at" && f.object == "Acme")
        .ok_or_else(|| anyhow::anyhow!("expected the Acme fact to be recallable"))?;

    let fact_id = stale
        .fact_id
        .ok_or_else(|| anyhow::anyhow!("recall must expose fact_id for supersede to be callable"))?;

    let moved_on = Utc::now();

    // `.at(..)` is REQUIRED — `execute()` refuses rather than defaulting to now,
    // because a missing bound is a caller bug and a silently mis-bounded fact is
    // worse than a loud error. (`.close_now()` is a different thing: it runs the
    // retirement sweep for already-past bounds.)
    // ⚠️ TWO STEPS, and missing the second is the trap.
    //
    // `.at(..)` closes the WORLD clock (`valid_to`) — "this stopped being true
    // then". `.close_now()` runs the RETIREMENT sweep, which sets the SYSTEM
    // clock (`expired_at`).
    //
    // A default recall filters on `expired_at` only — it answers "not retired",
    // NOT "true now". So `.at(..)` ALONE leaves the old fact still showing as
    // current, and you get two contradictory answers with no warning. Verified:
    // dropping `.close_now()` below makes this example print
    // `[("works_at", "Acme"), ("works_at", "Globex")]`.
    //
    // Use `.as_of(t)` when you want valid-time semantics; use `.close_now()`
    // when you are correcting the present.
    mem.supersede(fact_id)
        .in_namespace(ns.clone())
        .at(moved_on)
        .with_reason("Dana told us she moved to Globex")
        .close_now()
        .execute()
        .await?;

    // ── What is true now ────────────────────────────────────────────────────
    mem.remember("Dana joined Globex.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "works_at".into(),
            object: "Globex".into(),
            valid_from: Some(moved_on),
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await?;

    let now = current(&mem, &ns).await?;
    println!("after the correction  : {now:?}");

    // The point: exactly ONE current employer, and it is the new one.
    let employers: Vec<&String> = now.iter().map(|(_, o)| o).collect();
    assert_eq!(
        employers.len(),
        1,
        "exactly one employer should be current after the correction, got {employers:?} — \
         two would mean the old fact was never closed"
    );
    assert_eq!(
        employers[0], "Globex",
        "the current employer should be the corrected one, got {employers:?}"
    );

    println!("\nOne current answer, and the old one was CLOSED rather than deleted —");
    println!("'where did Dana work last year?' is still answerable via .as_of().");
    println!("A cache would have overwritten it; a log would have left you with both.");

    mem.close().await?;
    Ok(())
}

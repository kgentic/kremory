//! **Runs offline.** No Ollama, no API keys, no environment variables, no network.
//!
//! ```text
//! cargo run --example remembers_across_sessions
//! ```
//!
//! ## What this shows
//!
//! An agent learns something about a user, the world changes, and the agent has to
//! answer BOTH questions correctly:
//!
//!   - "Where does Alice live?"                  -> Berlin   (what is true now)
//!   - "Where did Alice live back in March?"     -> London   (what was true then)
//!
//! Most memory systems can only answer the first, because an update overwrites the
//! old value. kremory keeps both, because a fact carries a *validity window* — so the
//! old answer is still there, correctly bounded, rather than deleted.
//!
//! ## The two clocks (this is the part worth understanding)
//!
//! Every fact has two independent timelines:
//!
//!   - **valid time** — when the fact was true *in the world*. Mutable: London was
//!     true until March, then stopped being true. Queried with `.as_of(t)`.
//!   - **transaction time** (`recorded_at`) — when the database *learned* it.
//!     Immutable, and it never changes even when the fact is later corrected.
//!
//! They come apart constantly in real systems: you can learn in June that someone
//! moved in March. Valid time says March; transaction time says June. Collapsing them
//! into one "updated_at" column loses the distinction permanently.
//!
//! ## A naming trap worth knowing before you read the fields
//!
//! `recall(..).raw()` gives you `RetrievedFact`, which RENAMES the two window ends
//! relative to the stored row:
//!
//! | `RetrievedFact` | stored as | means |
//! |---|---|---|
//! | `valid_at`   | `Fact.valid_from` | when it became true |
//! | `invalid_at` | `Fact.valid_to`   | when it stopped being true |
//! | `recorded_at`| `Fact.recorded_at`| when we learned it |
//!
//! So on the consumer surface `invalid_at` is the END OF THE VALIDITY WINDOW. (On the
//! internal row type, a separate `invalid_at` column means something else again — the
//! contradiction resolver's mark. You will not see that one here.)

use std::future::Future;
use std::sync::Arc;

use chrono::{Duration, Utc};
use kremory::core::intelligence::{
    EntityExtractor, ExtractionContext, ExtractionResult,
};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace, StructuredFact};

/// Dimension of the stand-in embedder below. kremory defaults to 384, so any
/// custom embedder of a different size must declare it via `.embedding_dim(N)`.
const DEMO_DIM: usize = 16;

/// A deterministic, non-semantic stand-in so this example needs no embedding
/// service. **Do not copy this into production** — it hashes bytes, it does not
/// understand meaning. A real consumer calls their embedding backend (OpenAI,
/// Ollama `nomic-embed-text`, a local GGUF, sentence-transformers over HTTP)
/// inside `embed`. See `docs/api/setup.md` for the real thing.
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

/// kremory always wants to know HOW facts get out of prose, even when — as here —
/// no prose is ever parsed. We call `.skip_extraction()` on every write, so this is
/// never actually invoked; it exists to satisfy the builder, which requires either
/// `.with_llm(..)` (built-in extraction) or `.with_extractor(..)` (bring your own).
///
/// Wiring a real one is the BYOE path: return the entities and facts you found, and
/// kremory resolves and stores them.
struct NoExtraction;

impl EntityExtractor for NoExtraction {
    fn name(&self) -> &'static str {
        "no-extraction"
    }

    // `async fn` rather than `-> impl Future`: the trait is AFIT-shaped, so this is
    // the simpler spelling and the one clippy's `manual_async_fn` asks for.
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
    // A throwaway database. Nothing is left behind when the example exits.
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("agent-memory.db");

    let ns = Namespace::new("alice-assistant");

    // Open with an embedder and NO LLM. The type-state builder allows this: an LLM
    // is only needed when kremory has to EXTRACT facts from prose. Here we supply
    // the facts ourselves, so there is nothing to extract and nothing to call.
    let mem = Memory::open(&db)
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        // Required even though we never extract — see `NoExtraction` above.
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    let march = Utc::now() - Duration::days(120);
    let june = Utc::now() - Duration::days(30);

    // ── Record Alice's location history ─────────────────────────────────────
    //
    // `.with_facts(..)` + `.skip_extraction()` writes facts directly — no LLM in the
    // loop. This is also how you'd import an existing dataset.
    //
    // Each fact carries its own validity WINDOW. London is bounded (it stopped being
    // true in March); Berlin is open-ended (still true). Both rows live in the graph.
    mem.remember("Alice lived in London, then moved to Berlin in March.")
        .in_namespace(ns.clone())
        .with_facts(vec![
            StructuredFact {
                subject: "alice".into(),
                predicate: "lives_in".into(),
                object: "London".into(),
                valid_from: Some(march - Duration::days(365)),
                valid_to: Some(march), // the window CLOSES — not a delete
                memory_type: None,
            },
            StructuredFact {
                subject: "alice".into(),
                predicate: "lives_in".into(),
                object: "Berlin".into(),
                valid_from: Some(march),
                valid_to: None, // still true
                memory_type: None,
            },
        ])
        .skip_extraction()
        .await?;

    // ── 3. "Where does Alice live?" — as of NOW ─────────────────────────────
    let now_facts: Vec<_> = mem
        .recall("where does alice live")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .collect();

    let now_objects: Vec<String> = now_facts
        .iter()
        .map(|f| f.object.clone())
        .collect();

    println!("now      -> {now_objects:?}");
    assert!(
        now_objects.iter().any(|o| o.eq_ignore_ascii_case("Berlin")),
        "expected Berlin in a present-time recall, got {now_objects:?}"
    );

    // ── 4. "Where did Alice live in April?" — as of a PAST date ─────────────
    //
    // `.as_of(t)` is a VALID-TIME query: what was true in the world at t. April is
    // after the move, so Berlin. Ask about a date before the move and London is the
    // answer that comes back instead.
    let before_move = march - Duration::days(30);
    let then_facts: Vec<_> = mem
        .recall("where does alice live")
        .in_namespace(ns.clone())
        .as_of(before_move)
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .collect();

    let then_objects: Vec<String> = then_facts
        .iter()
        .map(|f| f.object.clone())
        .collect();

    println!("as_of(before move) -> {then_objects:?}");

    // ── 5. The old fact is BOUNDED, not deleted ─────────────────────────────
    //
    // This is the property that distinguishes a bi-temporal store from an
    // overwrite: the superseded value is still on disk with a closed window, and
    // its `recorded_at` still says when we first learned it.
    let closed_window_count = now_facts
        .iter()
        .chain(then_facts.iter())
        .filter(|f| f.invalid_at.is_some()) // = Fact.valid_to — the window closed
        .count();

    // `filter_map` rather than `filter` + `expect`: it carries the Some-ness in the
    // type instead of asserting it, so there is no unwrap to justify.
    // Only `now_facts` — the same closed fact also comes back from the as_of query,
    // and printing it twice reads as a bug rather than as two views of one row.
    for (f, closed_at) in now_facts
        .iter()
        .filter_map(|f| f.invalid_at.map(|t| (f, t)))
    {
        println!(
            "bounded  -> {} {} valid[{} .. {}]  recorded_at={}",
            f.subject,
            f.predicate,
            f.valid_at.date_naive(),
            closed_at.date_naive(),
            f.recorded_at.date_naive(),
        );
    }

    assert!(
        then_objects.iter().any(|o| o.eq_ignore_ascii_case("London")),
        "expected London when asking about a date before the move, got {then_objects:?}"
    );
    assert!(
        closed_window_count > 0,
        "expected at least one fact with a CLOSED validity window — \
         the whole point is that superseded facts are bounded, not deleted"
    );

    println!("\nBoth answers are correct at once, because a fact has a WINDOW, not");
    println!("a single timestamp. Nothing was overwritten and nothing was deleted.");
    let _ = june; // narrative only — recorded_at is engine-set, not caller-set

    mem.close().await?;
    Ok(())
}

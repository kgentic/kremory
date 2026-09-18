//! **Runs offline.** Two `Memory` handles over one database file — the shape
//! every real deployment ends up in.
//!
//! ```text
//! cargo run --example two_handles_one_database
//! ```
//!
//! ## The problem this solves
//!
//! An API server and a nightly job. Two request handlers on the same box. A
//! write path and a read path that were built separately. As soon as kremory is
//! real for you, more than one thing is holding it open, and the question stops
//! being academic: **does a write through one handle become visible to the
//! other?**
//!
//! kremory is embedded, so there is no server arbitrating this — the database
//! file is the shared state. That is a feature (no process to run, no network
//! hop) and it is also the thing to verify rather than assume.
//!
//! ## Scope: TWO HANDLES, ONE PROCESS
//!
//! Be precise about what this proves, because the two questions look alike and
//! are not. Everything below happens inside **one OS process**: two `Memory`
//! handles, one `tokio` runtime, one address space.
//!
//! And note what does *not* arbitrate them. kremory's write lock is an
//! in-process `AsyncMutex` (`core/schema.rs:415`), but it is **per handle** —
//! every `TemporalGraph::open_with_dim` mints a fresh one
//! (`core/schema.rs:460`), along with its own `libsql::Database` and its own
//! connection. So the mutex serialises TASKS INSIDE one handle and does
//! nothing between the two here; what keeps these two honest is SQLite/libsql
//! file locking plus `PRAGMA busy_timeout = 5000` (`core/schema.rs:456`).
//!
//! **Two separate PROCESSES are still a different question.** Same file
//! locking, but now across separate OS processes: separate WAL index mappings,
//! separate page caches, separate advisory locks, and no shared runtime to
//! fall back on. That is covered by
//! `crates/kremory/tests/it/cross_process_one_database.rs`, which spawns real
//! child processes, and summarised in `docs/deployment.md` §2. Do not read a
//! pass here as evidence about that.
//!
//! ## What this asserts
//!
//! A writer handle commits; a SEPARATE reader handle, opened independently on
//! the same path **in the same process**, sees it. And it checks the direction
//! people forget: the reader can write too, and the writer sees THAT.

use std::future::Future;
use std::path::Path;
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

/// Open an independent handle on the same file. Nothing is shared in process —
/// each gets its own embedder, its own `libsql::Database`, its own connection
/// and its own `write_lock`. The file is the only thing they have in common.
async fn open(path: &Path, ns: &Namespace) -> anyhow::Result<Memory> {
    Ok(Memory::open(path)
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?)
}

async fn objects_for(mem: &Memory, ns: &Namespace, predicate: &str) -> anyhow::Result<Vec<String>> {
    Ok(mem
        .recall("deployment")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.predicate == predicate)
        .map(|f| f.object)
        .collect())
}

fn fact(subject: &str, predicate: &str, object: &str) -> StructuredFact {
    StructuredFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        valid_from: None,
        valid_to: None,
        memory_type: None,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("shared.db");
    let ns = Namespace::new("ops");

    // Two independent handles — think "request path" and "nightly job", both
    // inside this one process. Two processes is `cross_process_one_database.rs`.
    let writer = open(&db, &ns).await?;
    let reader = open(&db, &ns).await?;
    println!("opened two independent handles on {}", db.display());

    // ── Writer commits ──────────────────────────────────────────────────────
    writer
        .remember("The deployment finished at 14:05.")
        .in_namespace(ns.clone())
        .with_facts(vec![fact("deployment", "finished_at", "14:05")])
        .skip_extraction()
        .await?;
    println!("  writer wrote    : deployment finished_at 14:05");

    // ── Reader sees it ──────────────────────────────────────────────────────
    let seen = objects_for(&reader, &ns, "finished_at").await?;
    println!("  reader sees     : {seen:?}");
    assert!(
        seen.iter().any(|o| o == "14:05"),
        "a committed write must be visible to an independently-opened handle in \
         the same process — if this fails, two components cannot share one \
         database and the embedded story does not hold; got {seen:?}"
    );

    // ── And the other direction ─────────────────────────────────────────────
    //
    // The half people skip: the "reader" is not special. It can write, and the
    // "writer" must see that too, or you have an accidental primary.
    reader
        .remember("Rollback was not required.")
        .in_namespace(ns.clone())
        .with_facts(vec![fact("deployment", "rollback", "not required")])
        .skip_extraction()
        .await?;

    let back = objects_for(&writer, &ns, "rollback").await?;
    println!("  writer sees     : {back:?}");
    assert!(
        back.iter().any(|o| o == "not required"),
        "visibility must hold in BOTH directions; got {back:?}"
    );

    // Closing one handle must not disturb the other.
    reader.close().await?;
    let after_close = objects_for(&writer, &ns, "finished_at").await?;
    assert!(
        after_close.iter().any(|o| o == "14:05"),
        "closing one handle must not affect another; got {after_close:?}"
    );
    println!("  after close     : surviving handle still works");

    println!("\nOne file, two handles, ONE process, writes visible both ways. There is");
    println!("no server mediating this — the database file is the shared state, which");
    println!("is why it is worth proving rather than assuming.");
    println!("Two separate PROCESSES are a different question: see");
    println!("crates/kremory/tests/it/cross_process_one_database.rs.");

    writer.close().await?;
    Ok(())
}

//! **Requires a running Ollama.** This is the one that shows what kremory is actually for.
//!
//! ```text
//! ollama serve                     # if it isn't already running
//! ollama pull gemma4:e4b           # the chat model used for extraction
//! ollama pull nomic-embed-text     # the embedding model
//! cargo run --example agent_memory_with_ollama
//! ```
//!
//! Nothing else is needed — no API keys, no cloud account. It checks for all of the
//! above on startup and tells you exactly what is missing rather than failing deep
//! inside an HTTP call.
//!
//! ## What this shows that the offline examples cannot
//!
//! `offline_remember_recall` and `remembers_across_sessions` hand kremory
//! ready-made facts. That keeps them instant and dependency-free, but it skips the
//! part that makes kremory a *memory* rather than a database:
//!
//! > **You give it prose. It works out the entities and relationships itself, and
//! > stores them as a graph you can query.**
//!
//! Here an assistant is told things in ordinary language across one conversation,
//! the process exits, and a *new* process — a new session, a new day — opens the
//! same database and still knows them.
//!
//! ## About extraction quality
//!
//! This runs `gemma4:e4b`, a small model that fits on a laptop. It will produce
//! duplicate facts under different casing and the occasional nonsense triple — you
//! will see both in the output. That is honest and expected; a larger extraction
//! model produces a cleaner graph. The example also runs `dream()`, the
//! consolidation pass — and honestly reports that on a graph this small it merges
//! nothing, because it requires real evidence before merging. That conservatism is
//! the point.
//!
//! ## Expect this to take a minute
//!
//! Extraction is a real language-model call per chunk, roughly 15-45s each on a
//! laptop. That is the honest cost of building a graph from prose, and it is why
//! `remember()` is something you do on ingest, not in a request handler.

use std::path::Path;

use kremory::{Memory, Namespace};

const OLLAMA: &str = "http://localhost:11434";
const CHAT_MODEL: &str = "gemma4:e4b";
const EMBED_MODEL: &str = "nomic-embed-text";

/// Check the daemon is listening before doing any work, using only `std`.
///
/// Deliberately NOT an HTTP probe: that would need a dependency an example should
/// not carry, and it would test a *model* of the system rather than the system. A
/// missing MODEL surfaces from the real call below, re-messaged with the fix.
fn preflight() -> anyhow::Result<()> {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;

    // Try EVERY resolved address, not just the first. On macOS `localhost` resolves
    // to ::1 before 127.0.0.1, and Ollama binds IPv4 only — so taking `.next()`
    // reports "connection refused" against a perfectly healthy daemon. A preflight
    // that fails on a working system is worse than no preflight at all.
    let addrs: Vec<_> = "localhost:11434".to_socket_addrs()?.collect();
    if addrs.is_empty() {
        anyhow::bail!("could not resolve localhost:11434");
    }

    let mut last_err = None;
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, Duration::from_secs(2)) {
            Ok(_) => return Ok(()),
            Err(e) => last_err = Some(format!("{addr}: {e}")),
        }
    }

    anyhow::bail!(
        "Nothing is listening on {OLLAMA} (tried {} address(es); last: {}).\n\
         Start it with `ollama serve`, then re-run this example.",
        addrs.len(),
        last_err.unwrap_or_default()
    )
}

/// Turn a deep provider failure into the sentence that fixes it.
fn explain(err: impl std::fmt::Display) -> anyhow::Error {
    let s = err.to_string();
    if s.contains("not found") || s.contains("model") {
        anyhow::anyhow!(
            "{s}\n\nThis usually means a model is not pulled. Run:\n  \
             ollama pull {CHAT_MODEL}\n  ollama pull {EMBED_MODEL}"
        )
    } else {
        anyhow::anyhow!(s)
    }
}

/// Everything the assistant was told, in the order a real conversation would.
const CONVERSATION: &[&str] = &[
    "Hi — I'm Alice Bramble. I lead the platform team at Northwind Logistics.",
    "We're migrating our fleet-tracking service from Postgres to ClickHouse this quarter. \
     Rui Ferreira is running that migration.",
    "One thing to remember about me: I want short answers. Bullet points, no preamble.",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    preflight()?;

    let dir = tempfile::tempdir()?;
    let db = dir.path().join("assistant.db");
    let ns = Namespace::new("alice");

    // ── Session one: the conversation happens ───────────────────────────────
    //
    // ONE call wires a local chat model AND a local embedder. Nothing else to
    // configure. (`None` takes the default chat model; pass `Some("qwen2.5:7b".into())`
    // for a lighter one.)
    println!("session 1 — opening with Ollama at {OLLAMA} ({CHAT_MODEL})");
    let mem = Memory::with_ollama_at_model(OLLAMA, None, &db)
        .await
        .map_err(explain)?;

    for (i, turn) in CONVERSATION.iter().enumerate() {
        println!("  ingesting turn {} of {} …", i + 1, CONVERSATION.len());
        // No `.with_facts()` here — kremory reads the sentence and decides for
        // itself what the entities and relationships are.
        mem.remember(*turn)
            .in_namespace(ns.clone())
            .await
            .map_err(explain)?;
    }

    // What did it actually understand? This is the graph it built from prose.
    let learned: Vec<_> = mem
        .recall("who is alice and what is she working on")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .collect();

    println!("\n  extracted from prose, with no schema supplied:");
    for f in &learned {
        println!("    {} — {} — {}", f.subject, f.predicate, f.object);
    }

    // ── Consolidation: clean up what a small model got messy ────────────────
    //
    // Look at the list above: the same fact appears twice under different casing,
    // plus a few triples that are frankly nonsense. That is NORMAL for a small
    // local model.
    //
    // `dream()` is the background pass that merges aliases, resolves contradictions
    // and archives noise. It is EVIDENCE-DRIVEN, not cosmetic — on a graph this
    // small it will usually report `merged: 0`, because two mentions with different
    // casing and one supporting sentence each is not enough evidence to justify an
    // irreversible merge. That is the correct conservative behaviour, not a failure:
    // a memory that merges eagerly on thin evidence corrupts itself quietly.
    //
    // Run it over a real corpus and the numbers below stop being zero.
    println!("\n  running dream() to consolidate …");
    let summary = mem.dream().in_namespace(ns.clone()).execute().await?;
    println!(
        "    entities merged: {} (0 is expected on a graph this small — see above)",
        summary.cross_episode_merged
    );
    println!("    passes run: {:?}", summary.consolidation_ops_ran);

    mem.close().await?;
    println!("\n  session 1 closed. Process state is gone; the database is not.");

    // ── Session two: a new day, a new process ───────────────────────────────
    //
    // Reopening the same path is the whole point — this is what "memory" means.
    // In a real assistant this is a separate run, hours or weeks later.
    println!("\nsession 2 — reopening the same database");
    let mem = Memory::with_ollama_at_model(OLLAMA, None, &db)
        .await
        .map_err(explain)?;

    let answer = mem
        .recall("what should I know about how Alice likes replies?")
        .in_namespace(ns.clone())
        .await?;

    println!("\n  prompt-ready context for the new session:\n");
    for line in answer.lines() {
        println!("    {line}");
    }

    assert!(
        !answer.trim().is_empty(),
        "expected the reopened session to recall something about Alice — \
         if this is empty, extraction produced no facts (check the ingest logs \
         with RUST_LOG=kremory=info)"
    );

    mem.close().await?;

    println!("\nNothing was passed between the two sessions in memory — only the file at");
    println!("{}. That is the difference between a chat history and a memory.",
        Path::new(&db).display());
    Ok(())
}

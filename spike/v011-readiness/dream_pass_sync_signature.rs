/// Spike 2: `Engine::run_dream_pass_sync(DreamOpts)` Send + 'static signature
///
/// Proves:
/// (a) The method can be invoked from a tokio::spawn task (requires Send + 'static)
/// (b) It can be called from a sync context via Handle::block_on
/// (c) Arc<Engine> can be shared across multiple concurrent spawn tasks
///
/// Mock Engine and DreamOpts — field shapes don't matter for compile spike.
/// Run via: cargo run --bin dream_pass_sync_signature

use std::sync::Arc;
use tokio::runtime::Handle;

// ---------------------------------------------------------------------------
// Mock types — minimal shapes sufficient to prove compile claims
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DreamOpts {
    /// How many relationship candidates to process per dream pass.
    pub max_candidates: usize,
    /// Whether to emit detailed trace logs.
    pub verbose: bool,
}

#[derive(Debug)]
struct DreamSummary {
    pub relationships_promoted: usize,
    pub entities_merged: usize,
}

type DreamResult = Result<DreamSummary, String>;

/// Mock Engine — contains an Arc<String> to simulate a shared handle inside.
/// The Arc makes the struct Clone + Send + Sync if its contents are Send + Sync.
struct Engine {
    db_handle: Arc<String>,
}

impl Engine {
    fn new() -> Self {
        Self {
            db_handle: Arc::new("mock-db-handle".to_string()),
        }
    }

    /// The method shape ADR-045 dream-impl wave requires:
    /// - async fn (returns a future)
    /// - &self (takes shared reference — must be Send + Sync for Arc<Engine> use)
    /// - DreamOpts value arg
    /// - Returns Result<DreamSummary>
    async fn run_dream_pass_sync(&self, opts: DreamOpts) -> DreamResult {
        // Simulate async work (would be DB + LLM calls in real impl)
        let _ = &self.db_handle;
        Ok(DreamSummary {
            relationships_promoted: opts.max_candidates,
            entities_merged: 0,
        })
    }
}

// Engine must be Send + Sync for Arc<Engine> to be Send + Sync.
// These are asserted at compile time by the spawn calls below.
// If Engine were not Send+Sync, the compiler would reject the spawn.

// ---------------------------------------------------------------------------
// Spike assertions
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let mut all_pass = true;

    // --- (a) tokio::spawn with owned Arc<Engine> ---
    {
        let engine = Arc::new(Engine::new());
        let opts = DreamOpts { max_candidates: 10, verbose: false };

        let handle = tokio::spawn(async move {
            engine.run_dream_pass_sync(opts).await
        });

        match handle.await {
            Ok(Ok(summary)) => {
                println!("  [PASS] (a) tokio::spawn: relationships_promoted={}", summary.relationships_promoted);
            }
            Ok(Err(e)) => {
                eprintln!("  [FAIL] (a) tokio::spawn: method error: {}", e);
                all_pass = false;
            }
            Err(e) => {
                eprintln!("  [FAIL] (a) tokio::spawn: join error: {}", e);
                all_pass = false;
            }
        }
    }

    // --- (b) Handle::block_on from a sync-style context ---
    // We're inside an async main, so we use spawn_blocking to get a true sync
    // thread, then call block_on via the handle obtained before entering it.
    {
        let engine = Arc::new(Engine::new());
        let opts = DreamOpts { max_candidates: 5, verbose: true };
        let handle = Handle::current();

        let engine_clone = Arc::clone(&engine);
        let result = tokio::task::spawn_blocking(move || {
            handle.block_on(engine_clone.run_dream_pass_sync(opts))
        })
        .await;

        match result {
            Ok(Ok(summary)) => {
                println!("  [PASS] (b) Handle::block_on from sync: relationships_promoted={}", summary.relationships_promoted);
            }
            Ok(Err(e)) => {
                eprintln!("  [FAIL] (b) Handle::block_on from sync: method error: {}", e);
                all_pass = false;
            }
            Err(e) => {
                eprintln!("  [FAIL] (b) Handle::block_on from sync: join error: {}", e);
                all_pass = false;
            }
        }
    }

    // --- (c) Arc<Engine> shared across multiple concurrent spawns ---
    {
        let engine = Arc::new(Engine::new());
        let mut handles = Vec::new();

        for i in 0..4usize {
            let eng = Arc::clone(&engine);
            let opts = DreamOpts { max_candidates: i + 1, verbose: false };
            handles.push(tokio::spawn(async move {
                eng.run_dream_pass_sync(opts).await
            }));
        }

        let mut all_ok = true;
        for (i, h) in handles.into_iter().enumerate() {
            match h.await {
                Ok(Ok(_)) => {}
                other => {
                    eprintln!("  [FAIL] (c) Arc<Engine> spawn {}: {:?}", i, other);
                    all_ok = false;
                    all_pass = false;
                }
            }
        }
        if all_ok {
            println!("  [PASS] (c) Arc<Engine> shared across 4 concurrent spawns");
        }
    }

    println!();
    if all_pass {
        println!("SPIKE 2: PASS");
    } else {
        println!("SPIKE 2: FAIL");
        std::process::exit(1);
    }
}

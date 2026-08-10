//! TD-187 round 2 — the REAL-LLM seam test for per-fact `valid_at`.
//!
//! # Why this test had to exist
//!
//! Everything else covering this feature is deterministic and hand-fed: the unit
//! tests hand-construct `TripletPromptParams`, the parser tests hand-write JSON,
//! and the persistence test uses a stub extractor that returns a `valid_at` I
//! chose myself. Not one of them puts a real model in the loop. The capability
//! probes that justified the design were worse in this specific respect — they
//! built prompts in Python and hit `/api/generate` directly, bypassing
//! `build_triplet_prompt`, `RawFact` and `parse_facts` entirely. **They validated
//! the idea and proved nothing about the shipped path.**
//!
//! Behaviour here genuinely depends on model output, which is exactly where the
//! test pyramid says a real-LLM + VCR test belongs. This is that test: one live
//! recording, deterministic replay forever after.
//!
//! # Mode selection (mirrors `golden_path_smoke.rs` §4.4)
//!
//!   * `KREMORY_VCR=record` → LIVE Ollama wrapped in `record(...)`; refreshes the
//!     committed cassette. Requires Ollama + the model pulled.
//!   * `KREMORY_VCR=replay` or unset → deterministic `replay(...)`, no Ollama.
//!     A missing cassette is a LOUD error.
//!
//! # What is asserted, and why it cannot pass vacuously
//!
//! The assertion is a STRICT INEQUALITY against the document date: at least one
//! persisted fact must carry a `valid_at` **strictly earlier** than the episode's
//! `published_at`.
//!
//! That is the one property the whole feature exists to deliver and the one thing
//! that was impossible before it. Pre-change, every fact from an episode was
//! written `valid_from: ref_time`, so EVERY fact equalled the document date and
//! this assertion could not have passed. It also cannot pass by accident: nothing
//! in the pipeline invents a backwards date, so a value earlier than the anchor
//! can only have come from the model resolving "last year" through the real
//! prompt, the real parser and the real persist path.
//!
//! No entity names and no counts are asserted — per the same nondeterminism
//! discipline as `golden_path_smoke.rs`, those would flake on every re-record.

#![cfg(feature = "llm-smoke")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use kremory::core::provider::{ChatProvider, DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::{Memory, Namespace};

enum Mode {
    Live,
    Replay,
}

fn resolve_mode() -> Mode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => Mode::Live,
        Ok("replay") | Err(_) => Mode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("td187_valid_at_llm_seam.json")
}

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_chat_model() -> String {
    std::env::var("KREMORY_TEST_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string())
}

/// per `feedback_td024` — `keep_alive` thrash hurts extraction precision.
fn real_ollama_chat() -> Arc<dyn ChatProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model(ollama_chat_model())
        .keep_alive("1h")
        .timeout_seconds(120)
        .build()
        .expect("real Ollama chat provider must build (KREMORY_VCR=record requires Ollama)");
    llm as Arc<dyn ChatProvider>
}

fn unique_db() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_td187_seam_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

/// Verbatim shape of the real corpus turn this feature was designed against
/// (LoCoMo conv0 `D12:15`, session dated 17 Aug 2023, gold answer "2022"). Kept
/// close to the original so the cassette exercises a realistic input rather than
/// a sentence written to be easy.
const TEXT: &str =
    "Caroline: I'm always here for you, Mel! We had a blast last year at the Pride fest. \
     Those supportive friends definitely make everything worth it!";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_llm_resolves_a_relative_date_through_the_shipped_path() {
    let published: DateTime<Utc> = Utc.with_ymd_and_hms(2023, 8, 17, 0, 0, 0).unwrap();
    let cassette = cassette_path();

    let (provider, embedder): (Arc<RecordReplayChatProvider>, Arc<dyn DynEmbeddingProvider>) =
        match resolve_mode() {
            Mode::Live => (
                Arc::new(RecordReplayChatProvider::record(
                    real_ollama_chat(),
                    cassette.clone(),
                    ollama_chat_model(),
                )),
                Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 }),
            ),
            Mode::Replay => (
                Arc::new(RecordReplayChatProvider::replay(cassette.clone()).expect(
                    "replay cassette must load — record it with \
                     KREMORY_VCR=record cargo nextest run -p kremory \
                     --features llm-smoke,content-search,test-utils \
                     -E 'test(/td187_valid_at_llm_seam/)'",
                )),
                Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 }),
            ),
        };

    let llm: Arc<dyn ChatProvider> = provider.clone();
    let mem = Memory::open(unique_db())
        .with_llm(llm)
        .with_embedder(embedder)
        .default_namespace(Namespace::new("td187seam"))
        .await
        .expect("Memory::open must succeed");

    // The real public API, with a DECLARED document anchor. Without
    // `.published_at(..)` no date block renders at all and the feature is inert —
    // so this call is load-bearing, not incidental setup.
    mem.remember(TEXT)
        .published_at(published)
        .await
        .expect("remember must succeed");

    // MANDATORY before reading the cassette back (NEW-202): the Phase-2 write
    // happens on the background worker thread and flush-on-Drop is not
    // deterministic.
    // NOT `let _ = provider.flush()`. A failed flush means the cassette was never
    // written, so a recording run would report success having produced nothing —
    // the exact silent no-op this project has a rule about. Fail loudly instead.
    provider
        .flush()
        .expect("cassette flush must succeed — a swallowed error here means the recording run \
                 produced no cassette while reporting success");

    // Recall the SUBJECT, not the object.
    //
    // Both extracted facts have `Caroline` as subject, so anchoring on her returns
    // both. The first cut queried "Pride fest" and read back the same `Friendship`
    // fact twice — because this test runs a `NullEmbeddingProvider`, so semantic
    // search is degenerate and the object-entity was never surfaced. That was a
    // defect in the READ instrument, not in the feature: the recorded cassette
    // showed the model had emitted `"valid_at": "2022-01-01"` correctly all along.
    //
    // The assertion below is unchanged and still strict — only the query that
    // feeds it was wrong.
    let ctx = mem
        .recall("Caroline")
        .raw()
        .await
        .expect("recall must succeed");
    let facts: Vec<_> = ctx.iter().flat_map(|c| c.facts.iter()).collect();

    // NON-VACUITY GUARD. Without this the assertion below iterates an empty vec
    // and reports green against a pipeline that extracted nothing at all — the
    // difference between "the model dated no fact" and "there were no facts".
    assert!(
        !facts.is_empty(),
        "no facts were extracted, so this test measured NOTHING about dating. \
         Re-record the cassette or check the model."
    );

    let earlier: Vec<_> = facts.iter().filter(|f| f.valid_at < published).collect();

    assert!(
        !earlier.is_empty(),
        "no fact carried a valid_at earlier than the document date ({published}). \
         The text says \"last year\", so a correctly-wired pipeline must resolve it \
         backwards. Every fact landing ON the anchor is precisely the pre-TD-187 \
         behaviour this feature exists to replace. Got: {:?}",
        facts.iter().map(|f| (&f.predicate, f.valid_at)).collect::<Vec<_>>()
    );
}

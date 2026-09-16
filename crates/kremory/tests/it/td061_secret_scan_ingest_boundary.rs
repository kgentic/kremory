//! TD-061 ingest-boundary integration test.
//!
//! Proves a secret-bearing document trips the scanner
//! (`core::secret_scan::scan_ingest_text`, unit-tested in-crate) at the
//! actual consumer surface: `Memory::remember(...)`, wired into
//! `Engine::ingest_with` BEFORE the episode insert. Three properties, one
//! test each:
//!
//! 1. Default (`SecretScanMode::FlagOnly`) — the hit is observable (a
//!    counter fires) but the episode is stored unmodified. Never silently
//!    dropped, never silently mutated.
//! 2. `SecretScanMode::Redact` — the hit is still observable, AND the
//!    persisted episode content no longer contains the raw secret.
//! 3. `with_secret_scan_enabled(false)` — a true no-op: no counter, no
//!    mutation. The off switch this repo's ingest-gating knobs all require
//!    (a lever with no control arm cannot be measured).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, SecretScanMode, SourceKind};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td061_secret_scan_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Sum a counter's recorded value from a local metrics snapshot. Mirrors the
/// helper already used by `dream_phase2_deterministic_passes.rs` for the
/// same `DebuggingRecorder` pattern.
fn counter_total(
    snapshot: &[(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
) -> u64 {
    snapshot
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == name)
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => Some(*c),
            _ => None,
        })
        .sum()
}

/// Shaped like a real AWS access key ID (`AKIA` + 16 base32 chars,
/// `[A-Z2-7]`) but deliberately NOT the classic AWS-docs `...EXAMPLE`
/// fixture — gitleaks' own `aws-access-token` rule allowlists any match
/// ending in `EXAMPLE`, which would make this test a guaranteed false
/// negative against the real ruleset. Not a live credential.
const AWS_SHAPED_SECRET: &str = "AKIAABCDEFGHIJKLMNOP";

#[tokio::test]
async fn secret_bearing_document_is_flagged_by_default() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let ns = Namespace::new("test-td061-flag");
    let mem = Memory::open(unique_db_path("flag_default"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open must succeed");

    let slug = "td061-doc-flag";
    let text = format!("Here is our deploy key: AWS_ACCESS_KEY_ID={AWS_SHAPED_SECRET}");

    mem.remember(text.clone())
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction() // test is about the scan, not extraction
        .await
        .expect("remember must succeed even when a secret is present — must NOT reject/drop it");

    // Flagged, not silently dropped: the counter fires.
    let snapshot = snapshotter.snapshot().into_vec();
    let hits = counter_total(&snapshot, "kremory.ingest.secret_scan.hits_total");
    assert!(hits >= 1, "secret_scan hit counter must fire; got {hits}");

    // Default mode is FlagOnly — the stored episode content is untouched.
    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");
    assert_eq!(episodes.len(), 1, "expected exactly one stored episode");
    assert_eq!(
        episodes[0].content, text,
        "FlagOnly (default) must not mutate the stored episode content"
    );
    assert!(
        episodes[0].content.contains(AWS_SHAPED_SECRET),
        "the raw secret must still be present under the default FlagOnly mode"
    );
}

#[tokio::test]
async fn secret_bearing_document_is_redacted_before_storage_in_redact_mode() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let ns = Namespace::new("test-td061-redact");
    let mem = Memory::open(unique_db_path("redact_mode"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_secret_scan_mode(SecretScanMode::Redact)
        .await
        .expect("Memory::open must succeed");

    let slug = "td061-doc-redact";
    let text = format!("Here is our deploy key: AWS_ACCESS_KEY_ID={AWS_SHAPED_SECRET}");

    mem.remember(text.clone())
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction()
        .await
        .expect("remember must succeed under Redact mode");

    // Still observable — a hit is a hit regardless of what happens to the text.
    let snapshot = snapshotter.snapshot().into_vec();
    let hits = counter_total(&snapshot, "kremory.ingest.secret_scan.hits_total");
    assert!(
        hits >= 1,
        "secret_scan hit counter must fire under Redact mode too; got {hits}"
    );

    // But the PERSISTED text must no longer carry the raw secret — this is
    // the actual storage boundary the scan exists to protect.
    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");
    assert_eq!(episodes.len(), 1, "expected exactly one stored episode");
    assert!(
        !episodes[0].content.contains(AWS_SHAPED_SECRET),
        "Redact mode must have removed the raw secret from stored content: {}",
        episodes[0].content
    );
    assert!(
        episodes[0].content.contains("[REDACTED_SECRET]"),
        "Redact mode's marker must be present in the stored content: {}",
        episodes[0].content
    );
}

#[tokio::test]
async fn disabling_the_scan_is_a_true_no_op() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let ns = Namespace::new("test-td061-disabled");
    let mem = Memory::open(unique_db_path("disabled"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_secret_scan_enabled(false)
        .await
        .expect("Memory::open must succeed");

    let slug = "td061-doc-disabled";
    let text = format!("Here is our deploy key: AWS_ACCESS_KEY_ID={AWS_SHAPED_SECRET}");

    mem.remember(text.clone())
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction()
        .await
        .expect("remember must succeed with the scan disabled");

    let snapshot = snapshotter.snapshot().into_vec();
    let hits = counter_total(&snapshot, "kremory.ingest.secret_scan.hits_total");
    assert_eq!(hits, 0, "disabled scan must never increment the hit counter");

    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");
    assert_eq!(episodes.len(), 1, "expected exactly one stored episode");
    assert_eq!(
        episodes[0].content, text,
        "disabled scan must leave the stored content byte-identical"
    );
}

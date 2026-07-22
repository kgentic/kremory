//! kremory::core — bi-temporal knowledge graph primitives.
//!
//! This module contains the per-episode graph primitives: entity/edge extraction,
//! dedup, fact invalidation, contradiction handling, hybrid retrieval, bi-temporal
//! queries, and community detection.
//!
//! The substrate core — schema, migrations, ingest, extraction, retrieval, dream.
//!
//! # BYOM invariant
//!
//! `core` consumes the `ChatProvider` trait from `autoagents-llm` ONLY — never
//! the concrete `autoagents-llamacpp` impl. Consumers wire their own backend.
//!
//! # Lock ordering
//!
//! All locks in this module must be acquired in the following fixed order to
//! prevent deadlocks. Never acquire a lock with a higher index while holding
//! one with a lower index.
//!
//! | Index | Lock | Location | Guards |
//! |-------|------|----------|--------|
//! | 1 | `write_lock: tokio::sync::Mutex<()>` | `TemporalGraph` (ADR-022) | Serialises concurrent write transactions (BEGIN IMMEDIATE). Acquired first — before any sub-lock — on every write path. |
//! | 2 | `session: std::sync::Mutex<ort::Session>` | `OrtEmbeddingProvider`, `NerModel` | Guards the ORT inference session. Short critical section; never held across await points. |
//! | 3 | `entries: std::sync::Mutex<HashMap>` | `SpeculativeCache` | Guards speculative-cache entries. Never held while acquiring index-2 locks. |
//! | 4 | `error_rx: std::sync::Mutex<Receiver<IngestError>>` | `BackgroundIngestor` | Guards the ingest-error channel receiver. Never held while acquiring index-1, -2, or -3 locks. |
//!
//! ## Rules
//!
//! 1. The ADR-022 `write_lock` (index 1) is **always acquired first** on any
//!    write path through `TemporalGraph`. It is held for the duration of the
//!    SQLite write transaction and released only after `COMMIT` or `ROLLBACK`.
//! 2. ORT session locks (index 2) are held only during model inference, never
//!    across an `.await`. Do not call any async function while holding one.
//! 3. `SpeculativeCache` (index 3) and the error-channel receiver (index 4)
//!    are structurally independent of each other and of ORT sessions. Acquire
//!    at most one of these at a time on any single code path.
//! 4. When adding new `Mutex` / `RwLock` fields to any type in this module,
//!    assign an index higher than any lock it may be nested inside, and update
//!    this table before merging.

// ── emit_and_trace! — dual-emit macro (ADR D1, §R4.2) ──────────────────────
//
// Expands to BOTH a metrics::counter!(...).increment(n) call AND a co-located
// tracing::<level>!(...) call in a single statement, satisfying the ADR D1
// ±5-source-line co-location requirement structurally.
//
// Defined here (in core/mod.rs) so the macro is available to ALL child modules
// without #[macro_export] (which would leak into the public kremory:: namespace).
// See crates/kremory/src/core/obs.rs for documentation and tests.
//
// Syntax:
//   emit_and_trace!(
//       counter: "<name>", "<key>" => "<val>"[, ...];
//       [n: <expr>;]
//       level: <level>;         // error|warn|info|debug|trace
//       [fields: <k = v>[, ...];]
//       msg: "<message>"
//   );
macro_rules! emit_and_trace {
    // ── with fields (any n) ──────────────────────────────────────────────────
    // Syntax:
    //   emit_and_trace!(
    //       counter: "name", "k" => "v";
    //       n: <expr>;          ← optional; defaults to 1
    //       level: warn;
    //       { field = %val, field2 = ?val2 }  ← tracing fields in braces
    //       msg: "message"
    //   );
    //
    // Using a braced group for fields avoids tt* ambiguity with the ; delimiter.

    // variant: explicit n, with fields in { }
    (
        counter: $cname:expr $(, $lk:expr => $lv:expr)*;
        n: $n:expr;
        level: $lvl:tt;
        { $($field:tt)* }
        msg: $msg:expr
    ) => {{
        metrics::counter!($cname $(, $lk => $lv)*).increment($n);
        tracing::$lvl!($($field)* $msg);
    }};

    // variant: default n=1, with fields in { }
    (
        counter: $cname:expr $(, $lk:expr => $lv:expr)*;
        level: $lvl:tt;
        { $($field:tt)* }
        msg: $msg:expr
    ) => {
        emit_and_trace!(
            counter: $cname $(, $lk => $lv)*;
            n: 1;
            level: $lvl;
            { $($field)* }
            msg: $msg
        )
    };

    // variant: explicit n, no fields
    (
        counter: $cname:expr $(, $lk:expr => $lv:expr)*;
        n: $n:expr;
        level: $lvl:tt;
        msg: $msg:expr
    ) => {{
        metrics::counter!($cname $(, $lk => $lv)*).increment($n);
        tracing::$lvl!($msg);
    }};

    // variant: default n=1, no fields
    (
        counter: $cname:expr $(, $lk:expr => $lv:expr)*;
        level: $lvl:tt;
        msg: $msg:expr
    ) => {
        emit_and_trace!(
            counter: $cname $(, $lk => $lv)*;
            n: 1;
            level: $lvl;
            msg: $msg
        )
    };
}

pub mod arena;
pub mod background;
pub mod canonicalization;
pub mod chat_tracking;
/// Confidence-aware merge helpers — noisy-OR combination (ADR-063 Site #6, the
/// deterministic half; the reject-floor gate is S4-blocked, see the module docs).
pub(crate) mod confidence;
pub mod config;
pub mod context;
pub mod contradiction;
pub mod disambiguation;
pub mod dream;
pub mod embedding;
pub mod engine;
pub mod entity_types;
pub mod error;
pub mod extraction;
pub mod extraction_window;
pub mod format;
pub mod graph;
pub mod grounding;
pub mod hybrid_extractor;
/// Shared identity-verdict schema + deterministic write-gate (ADR-063 spec §2).
/// Crate-internal — consumed by Site #5 (dream acronym/nickname recall), Site #3
/// (type-registry collapse), and composes with Site #6; not napi-exposed.
pub(crate) mod identity_verdict;
pub mod ingest;
pub mod intelligence;
/// Query-intent classification for recall scoring (TD-066 phase 1, recall-v2
/// spec Decision 1). Zero-LLM keyword heuristic; not yet wired into the live
/// recall path (phase 2) — see module docs for the HALF-FEATURE note.
pub(crate) mod intent;
pub mod migrations;
#[cfg(feature = "ner")]
pub mod ner;
pub mod obs;
pub mod provider;
pub mod rates;
pub mod reclassification;
/// TD-062 cross-encoder reranker (spec §3 Increment 3 / §4). Crate-internal —
/// the `Reranker` trait + `FastEmbedReranker` are consumed by the recall
/// pipeline (`facade::recall`); consumers configure it via
/// `SearchOpts.rerank_k`, not by naming the trait directly.
#[cfg(feature = "rerank")]
pub(crate) mod rerank;
pub mod resolver;
pub(crate) mod resolver_batched;
pub mod schema;
/// Post-RRF recall scoring axes (recall-v2 Phase 2b): ScoringWeights, per-intent
/// weight lookup, temporal-recency boost, read-side-pure (no `execute(`).
pub(crate) mod scoring;
pub mod search;
pub mod sink;
pub mod speculative_cache;
pub mod text_utils;

pub use background::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestGuard, IngestSendError, IngestorConfig,
};
pub use error::{ContradictionResolution, Error, IngestStatus, IngestionErrorKind, Result};
pub use sink::{
    ContradictionDetected, EntityId, EntityOrEdgeRef, IngestEventSink,
    IngestionError as SinkIngestionError, OnEdgeAddedParams, SinkFact,
};

// Test-utils re-export for the Site #2 metrics harness
// (`tests/dream_metrics_harness_site2.rs`, ADR-063 spec §4.3 sibling / §2.2).
// `identity_verdict` (the module itself) is `pub(crate) mod identity_verdict;`
// above — crate-internal, so even though its items are individually `pub` +
// `#[doc(hidden)]` (MNT-002 pattern), the module path is unreachable from an
// external integration-test binary without this re-export. Mirrors the
// `dream/mod.rs` test-utils re-export blocks exactly (same E0365-adjacent
// visibility requirement); not part of the stable public API contract.
#[cfg(any(test, feature = "test-utils"))]
pub use identity_verdict::{
    write_gate, DeterministicSignal, IdentityVerdictItem, WriteDecision, WriteGateInputs,
};

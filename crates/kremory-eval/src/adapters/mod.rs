//! Adapters — translation layers between kremory's public Memory API and
//! eval harness sample/output types.
//!
//! Each adapter maps a benchmark's session ingestion + query pattern onto
//! `kremory::Memory::remember()` / `Memory::recall()`.
//!
//! # API gaps surfaced (O8 / O9)
//!
//! The following gaps between kremory's v0.1.0 public API and LongMemEval's
//! requirements were discovered during adapter implementation:
//!
//! - **O8 — Per-session namespace isolation**: LongMemEval ingests multiple
//!   independent "haystack sessions" per question. The adapter uses one
//!   `Namespace` per question (derived from `question_id`) to prevent cross-
//!   contamination between questions. `Memory` supports this via
//!   `.in_namespace(ns)` today, but requires the caller to manage namespace
//!   lifecycle. A future "ephemeral namespace" or "scoped memory" API would
//!   simplify adapter code.
//!
//! - **O9 — No per-turn temporal anchoring**: `Memory::remember()` accepts
//!   `.published_at(ts)` for bi-temporal anchoring, but LongMemEval sessions
//!   carry per-session dates, not per-turn dates. The adapter sets
//!   `published_at` at the session level (one timestamp per haystack session).
//!   Fine-grained per-turn temporal anchoring is not expressible in v0.1.0.
//!
//! - **O10 — Token usage not surfaced from recall**: `Memory::recall()` returns
//!   a `String` context block. There is no mechanism to retrieve the number of
//!   input tokens used by the LLM during the recall operation. Token-efficiency
//!   scoring (O6 formula) is therefore skipped for cremory adapter outputs —
//!   `LongMemEvalOutput::input_tokens_used` will always be `None` until kremory
//!   exposes usage telemetry on its public API.

pub mod longmemeval_adapter;

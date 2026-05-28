//! LongMemEval → kremory adapter.
//!
//! Translates a [`LongMemEvalSample`] into kremory `Memory::remember()` +
//! `Memory::recall()` calls, producing a [`LongMemEvalOutput`] ready for the
//! [`LongMemEvalScorer`].
//!
//! # Session ingestion pattern
//!
//! For each LongMemEval sample:
//! 1. A unique `Namespace` is derived from `question_id` to isolate sessions
//!    across questions (prevents cross-contamination).
//! 2. Each haystack session is ingested as a single `remember()` call. Turns
//!    within a session are concatenated (`role: content\n…`) to form the
//!    episode text.
//! 3. `published_at` is set from the session's date in `haystack_dates`
//!    (format `YYYY/MM/DD`) when available. This enables bi-temporal anchoring
//!    at session granularity. Per-turn anchoring is not expressible in v0.1.0
//!    (see O9 in adapters/mod.rs).
//! 4. After ingestion, `recall(question)` is called in the same namespace.
//!    The returned context string becomes `LongMemEvalOutput::response`.
//!
//! # API gaps (O8, O9, O10)
//!
//! See `adapters/mod.rs` for a full list of kremory API gaps surfaced.
//!
//! # BYOM invariant
//!
//! This adapter does NOT depend on `autoagents-llamacpp` directly. The
//! `Memory` handle passed in already has a `ChatProvider` + `EmbeddingProvider`
//! wired by the caller.

use chrono::{NaiveDate, TimeZone, Utc};

use kremory::{Memory, Namespace};

use crate::{
    layer_a::longmemeval::{LongMemEvalOutput, LongMemEvalSample},
    types::{EvalErr, EvalError},
};

/// Run one LongMemEval sample through a kremory `Memory` handle.
///
/// Ingests all haystack sessions in temporal order, then calls `recall` with
/// the question. Returns the context string and token usage (always `None` in
/// v0.1.0 — see O10).
///
/// # Namespace isolation
///
/// Each question uses a fresh namespace derived from `question_id`. The caller
/// is responsible for passing a `Memory` handle that does NOT have a
/// `default_namespace` set (or that will not be reused across questions).
///
/// Because kremory does not expose a namespace-delete API in v0.1.0, adapters
/// that run many samples should pass a per-question temporary database (e.g.,
/// `:memory:` path) or accept the namespace leakage (acceptable for batch eval
/// where the DB is thrown away after the run).
pub async fn run_sample(
    memory: &Memory,
    sample: &LongMemEvalSample,
) -> EvalError<LongMemEvalOutput> {
    // Derive a per-question namespace from question_id.
    // Namespace::new panics on empty string — question_id is always non-empty.
    let namespace = Namespace::new(format!("lme-{}", sample.question_id));

    // Ingest each haystack session in order.
    for (session_idx, session_turns) in sample.haystack_sessions.iter().enumerate() {
        if session_turns.is_empty() {
            continue;
        }

        // Concatenate turns: "user: <content>\nassistant: <content>\n…"
        let episode_text: String = session_turns
            .iter()
            .map(|turn| format!("{}: {}", turn.role, turn.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Parse session date for bi-temporal anchoring (O9: session-level only).
        let published_at = sample
            .haystack_dates
            .get(session_idx)
            .and_then(|date_str| parse_haystack_date(date_str));

        // Session ID for source tagging (traceability).
        let session_id = sample
            .haystack_session_ids
            .get(session_idx)
            .cloned()
            .unwrap_or_else(|| format!("session-{}", session_idx));

        let mut req = memory
            .remember(episode_text)
            .in_namespace(namespace.clone())
            .from_chat(session_id);

        if let Some(ts) = published_at {
            req = req.published_at(ts);
        }

        req.await.map_err(|e| EvalErr::Other(format!(
            "remember failed for question_id={} session={}: {}",
            sample.question_id, session_idx, e
        )))?;
    }

    // Recall: query the memory with the question text.
    let context = memory
        .recall(sample.question.clone())
        .in_namespace(namespace)
        .await
        .map_err(|e| EvalErr::Other(format!(
            "recall failed for question_id={}: {}",
            sample.question_id, e
        )))?;

    // Token usage not available from kremory v0.1.0 public API (O10).
    Ok(LongMemEvalOutput {
        response: context,
        input_tokens_used: None,
    })
}

/// Parse a haystack date string in `YYYY/MM/DD` format into a UTC timestamp.
///
/// Returns `None` on parse failure (non-fatal — session is ingested without
/// temporal anchor).
fn parse_haystack_date(date_str: &str) -> Option<chrono::DateTime<Utc>> {
    NaiveDate::parse_from_str(date_str, "%Y/%m/%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|dt| Utc.from_utc_datetime(&dt))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::Datelike;

    use super::*;

    #[test]
    fn parse_date_valid() {
        let ts = parse_haystack_date("2024/03/15");
        assert!(ts.is_some());
        let dt = ts.unwrap();
        assert_eq!(dt.naive_utc().year(), 2024);
        assert_eq!(dt.naive_utc().month(), 3);
        assert_eq!(dt.naive_utc().day(), 15);
    }

    #[test]
    fn parse_date_invalid_returns_none() {
        assert!(parse_haystack_date("not-a-date").is_none());
        assert!(parse_haystack_date("").is_none());
        assert!(parse_haystack_date("2024-03-15").is_none()); // wrong separator
    }

    #[test]
    fn parse_date_boundary() {
        // Leap year day
        assert!(parse_haystack_date("2024/02/29").is_some());
        // Invalid leap year day
        assert!(parse_haystack_date("2023/02/29").is_none());
    }
}

//! Observability helpers — kremory internal.
//!
//! The `emit_and_trace!` macro is defined in `core/mod.rs` (not here) so it is
//! available to all child modules without `#[macro_export]`. This module holds
//! the compile-test that verifies all four macro variants expand correctly.
//!
//! ADR D1 (co-location requirement): every counter emit MUST be paired with a
//! co-located tracing event within ±5 source lines. `emit_and_trace!` enforces
//! this structurally — a single macro invocation expands to both calls, so the
//! two are syntactically inseparable.
//!
//! ADR D2 (library, no subscriber): kremory installs NO tracing subscriber.
//! With no subscriber registered, tracing events are silently dropped — see
//! `docs/observability.md` §Consumer notes for setup guidance.
//!
//! # Macro syntax
//!
//! ```rust,ignore
//! // With fields (braced group, all tracing field syntax supported):
//! emit_and_trace!(
//!     counter: "kremory.fact.rejected_total", "reason" => "self_loop";
//!     level: warn;
//!     { subject_id = %subject_id, predicate = ?predicate, }
//!     msg: "kremory.fact.rejected self_loop"
//! );
//!
//! // Without fields:
//! emit_and_trace!(
//!     counter: "rql.extraction.json_parse_fail";
//!     level: warn;
//!     msg: "json parse failed"
//! );
//!
//! // With explicit increment:
//! emit_and_trace!(
//!     counter: "rql.extraction.json_parse_fail", "arm" => "repair";
//!     n: 3;
//!     level: warn;
//!     msg: "bulk repair"
//! );
//! ```

#[cfg(test)]
mod tests {
    /// Verify `emit_and_trace!` compiles and expands without panicking in all
    /// four variants. metrics recorder and tracing subscriber are both optional
    /// at runtime (kremory is a library per ADR D2); no-op if absent.
    #[test]
    fn emit_and_trace_compiles_all_variants() {
        let val = "v";
        let num = 42u32;

        // variant 1: explicit n, with fields (braced)
        emit_and_trace!(
            counter: "test.metric.v1", "arm" => "a";
            n: 2;
            level: warn;
            { key = %val, }
            msg: "test v1"
        );

        // variant 2: explicit n, no fields
        emit_and_trace!(
            counter: "test.metric.v2", "arm" => "b";
            n: 1;
            level: debug;
            msg: "test v2"
        );

        // variant 3: default n, with fields (braced, ?-format)
        emit_and_trace!(
            counter: "test.metric.v3";
            level: info;
            { reason = ?num, }
            msg: "test v3"
        );

        // variant 4: default n, no fields
        emit_and_trace!(
            counter: "test.metric.v4", "reason" => "none";
            level: trace;
            msg: "test v4"
        );
    }
}

use crate::handlers::parse_rerank_k;

// ── parse_rerank_k (KREMORY_RERANK_K boot override) ───────────────────
//
// The boot
// override was previously inline in the `RERANK_K` `LazyLock`, which is
// read exactly once per process and therefore untestable by
// construction — a malformed `KREMORY_RERANK_K` would silently disable
// reranking and no test could ever have caught it before it reached
// `/health`'s `rerank_k` provenance stamp. `parse_rerank_k` is the
// extracted pure function; these tests exercise every branch directly.

/// Absent env var, and every value `usize::from_str` accepts, parse
/// exactly as documented on `parse_rerank_k` / the `RERANK_K` doc
/// comment. `"0"` is PINNED here, not rejected: `usize::from_str` has no
/// notion of "zero is invalid" and `parse_rerank_k` adds no additional
/// range check, so `KREMORY_RERANK_K=0` boots with reranking "requested"
/// at k=0 (a real, if unusual, config — `apply_rerank_with` then reranks
/// an empty head and is a behavioural no-op, see
/// `apply_rerank_real_path_tests` in `kremory::facade::recall`). If this
/// assertion ever changes, it means `parse_rerank_k` grew a deliberate
/// "reject zero" rule — update this pin, don't just delete it.
#[test]
fn parse_rerank_k_accepts_absent_and_valid_values() {
    assert_eq!(
        parse_rerank_k(None),
        None,
        "absent KREMORY_RERANK_K must disable reranking"
    );
    assert_eq!(
        parse_rerank_k(Some("50")),
        Some(50),
        "a plain valid usize must parse through"
    );
    assert_eq!(
        parse_rerank_k(Some(" 50 ")),
        Some(50),
        "surrounding whitespace must be trimmed before parsing (operators hand-editing \
         a .env file routinely leave stray whitespace)"
    );
    assert_eq!(
        parse_rerank_k(Some("0")),
        Some(0),
        "PINNED: \"0\" parses to Some(0), not None — parse_rerank_k performs no \
         zero-rejection range check, only usize parsing"
    );
}

/// Malformed, negative, and empty values must all disable reranking
/// (`None`) with a WARN, never silently accept garbage or panic —
/// config, like LLM output, must be parsed loudly, never silently
/// defaulted. A
/// misconfigured boot override must fail toward "reranking off"
/// by default, the same safe default as the var being
/// entirely absent.
#[test]
fn parse_rerank_k_rejects_malformed_negative_and_empty_values() {
    assert_eq!(
        parse_rerank_k(Some("abc")),
        None,
        "non-numeric value must disable reranking, not panic"
    );
    assert_eq!(
        parse_rerank_k(Some("-1")),
        None,
        "negative value must disable reranking — usize has no negative representation"
    );
    assert_eq!(
        parse_rerank_k(Some("")),
        None,
        "empty (but present) value must disable reranking, matching the malformed-value \
         path rather than being treated as a distinct absent-value case"
    );
}

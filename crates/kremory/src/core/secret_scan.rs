//! Ingest-boundary secret/token scanner (TD-061).
//!
//! `Engine::ingest_with` (`core::ingest::pipeline::ingest_with`) calls
//! [`scan_ingest_text`] on the caller-supplied episode text BEFORE anything
//! is persisted, extracted, or embedded — see that call site's own comment
//! for why "before the episode insert" is the actual boundary (the insert
//! itself IS storage of the raw text; embedding happens even later).
//!
//! # Why this wraps `secrets_scanner` instead of hand-rolling
//!
//! TD-061's original resolution path said "reuse the entropy fn
//! (`resolver.rs:37`) + add pattern rules ... kunickiaj `secret-scanner.ts`
//! reference [a regex-rule-based scanner, similar shape to gitleaks]". Per
//! this repo's "check a maintained 3p crate before hand-rolling a
//! security-adjacent detector" rule, crates.io was searched before writing
//! any of that by hand — see the `secrets_scanner` dependency comment in
//! `Cargo.toml` for the full candidate comparison (why `ripsecrets` and
//! `secretshape` were rejected). The short version: `secrets_scanner`
//! (`default-features = false`) already ships exactly the shape TD-061
//! asked for — Aho-Corasick keyword pre-filter + Shannon-entropy gate +
//! regex validation, driven by gitleaks' own 222-rule default TOML ruleset
//! (the same reference point TD-061 cited) — so re-deriving that ruleset by
//! hand here would be strictly worse security coverage for no benefit.
//!
//! # Never let a raw secret escape this module
//!
//! [`secrets_scanner::Finding::matched`] carries the **unredacted** matched
//! text whenever the underlying `Scanner`'s `ScanConfig::redact` is `false`
//! (the default, and the config this module's shared scanner uses). This
//! module MUST NOT put `Finding::matched` into a log line, a metric label,
//! or [`SecretScanOutcome::rule_ids`] — only `Finding::rule_id` (a fixed,
//! non-secret identifier like `"aws-access-token"`) ever leaves this module.

use std::sync::OnceLock;

use secrets_scanner::Scanner;

use crate::core::config::{SecretScanConfig, SecretScanMode};

/// Label handed to the underlying scanner as its `path` argument. Purely a
/// `Finding::file` tag for path-only rules (none of which apply to ingest
/// text) — kremory ingest has no real filesystem path here.
const SCAN_LABEL: &str = "<kremory-ingest-episode>";

/// Process-wide scanner built once from the bundled (compiled-in) gitleaks +
/// local ruleset. `Scanner::from_bundled()` touches no filesystem or env var
/// (unlike `Scanner::new()`'s three-tier loader), so this is deterministic
/// and safe to lazily share across every ingest call and every test in this
/// binary.
fn scanner() -> &'static Scanner {
    static SCANNER: OnceLock<Scanner> = OnceLock::new();
    SCANNER.get_or_init(|| {
        Scanner::from_bundled().unwrap_or_else(|e| {
            panic!(
                "secrets_scanner bundled ruleset failed to parse — this is a compiled-in \
                 asset shipped by the secrets_scanner crate itself, so a parse failure here \
                 is a dependency/build defect, not a runtime condition callers can react to: {e}"
            )
        })
    })
}

/// Force the scanner's ruleset compilation now, off the caller's timing
/// budget. `MemoryBuilder`'s `IntoFuture` impls call this (via
/// `spawn_blocking`, since ruleset compilation is CPU-bound) at construction
/// time — measured ~13.7s in an unoptimized debug build. Left to the
/// `scanner()` `OnceLock`'s natural lazy-on-first-use, that cost silently
/// landed on whichever `remember()` call happened to run first in a process,
/// which is what broke `td251_cancel_wedges_batch`'s tight per-trial timeout
/// (TD-061 follow-up fix, 2026-09-16). Idempotent — a second call after the
/// first is a cheap `OnceLock` read.
pub(crate) fn warm() {
    let _ = scanner();
}

/// Outcome of scanning one episode's text for secrets before ingest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SecretScanOutcome {
    /// Rule IDs matched (e.g. `"aws-access-token"`, `"jwt"`), one entry per
    /// finding, in scanner order. **Never** the matched secret text — see
    /// this module's doc comment for why that field is off-limits.
    pub rule_ids: Vec<String>,
    /// The text to actually carry forward into the episode
    /// insert/extraction/embedding. Byte-identical to the input text unless
    /// `mode == Redact` AND at least one finding fired, in which case every
    /// matched span is replaced with the scanner's fixed
    /// `[REDACTED_SECRET]` marker.
    pub text: String,
}

impl SecretScanOutcome {
    /// `true` when the scan found at least one match, regardless of mode.
    pub fn has_hits(&self) -> bool {
        !self.rule_ids.is_empty()
    }
}

/// Scan `text` per `cfg` and return the (possibly redacted) text plus the
/// rule IDs that matched. `cfg.enabled == false` is a no-op — returns the
/// input text unchanged and an empty `rule_ids`, byte-identical to
/// pre-TD-061 ingest.
pub(crate) fn scan_ingest_text(text: &str, cfg: &SecretScanConfig) -> SecretScanOutcome {
    if !cfg.enabled {
        return SecretScanOutcome {
            rule_ids: Vec::new(),
            text: text.to_string(),
        };
    }

    // `scan_and_redact_content` always computes both the finding list and a
    // redacted copy, regardless of `mode` — one scan pass serves both
    // branches below. `FlagOnly` pays for a redaction buffer it discards;
    // episode text is not a hot loop, so the one-code-path simplicity wins
    // over sparing that allocation.
    let output = scanner().scan_and_redact_content(SCAN_LABEL, text);
    let rule_ids: Vec<String> = output.findings.iter().map(|f| f.rule_id.clone()).collect();

    let out_text = match cfg.mode {
        SecretScanMode::FlagOnly => text.to_string(),
        SecretScanMode::Redact => output.redacted,
    };

    SecretScanOutcome {
        rule_ids,
        text: out_text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag_cfg() -> SecretScanConfig {
        SecretScanConfig {
            enabled: true,
            mode: SecretScanMode::FlagOnly,
        }
    }

    fn redact_cfg() -> SecretScanConfig {
        SecretScanConfig {
            enabled: true,
            mode: SecretScanMode::Redact,
        }
    }

    // ── one positive + one near-miss per pattern class named in TD-061 ──────

    #[test]
    fn aws_access_key_is_flagged() {
        // Shaped like a real `AKIA`-prefixed access key ID (base32 tail,
        // `[A-Z2-7]{16}`). Deliberately NOT the classic AWS-docs
        // `...EXAMPLE` fixture — gitleaks' own `aws-access-token` rule
        // allowlists any match ending in `EXAMPLE`, so that fixture is a
        // guaranteed miss against this exact ruleset.
        let text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids.iter().any(|r| r.contains("aws")),
            "expected an aws-* rule id, got {:?}",
            out.rule_ids
        );
        assert_eq!(out.text, text, "FlagOnly must never mutate the text");
    }

    #[test]
    fn aws_access_key_near_miss_does_not_fire() {
        // Right prefix, wrong shape (too short, lowercase) — must not match.
        let text = "AWS_ACCESS_KEY_ID=akia_not_a_real_key";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            !out.rule_ids.iter().any(|r| r.contains("aws")),
            "near-miss text incorrectly matched an aws-* rule: {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn github_token_is_flagged() {
        let text = "token: ghp_OhbVrpoiVgRV5IfLBcbfnoGMbJmTPSIAoCLr";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids.iter().any(|r| r.contains("github")),
            "expected a github-* rule id, got {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn github_token_near_miss_does_not_fire() {
        // Mentions the prefix in prose but is far too short to be a real token.
        let text = "the ghp_ prefix marks a GitHub personal access token";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            !out.rule_ids.iter().any(|r| r.contains("github")),
            "near-miss text incorrectly matched a github-* rule: {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn jwt_is_flagged() {
        // The canonical jwt.io example token.
        let text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.\
                     eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.\
                     SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids.iter().any(|r| r.contains("jwt")),
            "expected a jwt* rule id, got {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn jwt_near_miss_does_not_fire() {
        let text = "the JWT spec defines three dot-separated, base64url-encoded segments";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            !out.rule_ids.iter().any(|r| r.contains("jwt")),
            "near-miss text incorrectly matched a jwt* rule: {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn pem_private_key_block_is_flagged() {
        let text = "-----BEGIN RSA PRIVATE KEY-----\n\
                     MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEAy1Y+8RXJvvsX3jr9\n\
                     yTb2ZQmL8s5t8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s8t8s\n\
                     -----END RSA PRIVATE KEY-----";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids.iter().any(|r| r.contains("private-key")),
            "expected a private-key rule id, got {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn pem_mention_near_miss_does_not_fire() {
        let text = "PRIVATE KEY material must never be committed to a repo";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            !out.rule_ids.iter().any(|r| r.contains("private-key")),
            "near-miss text incorrectly matched the private-key rule: {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn generic_api_key_assignment_is_flagged() {
        let text = "api_key = \"sk_test_51H8x9KJ2eZvKYlo2C0FQwq4XyZ1a2B3c4D5e6F7g8H9\"";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids
                .iter()
                .any(|r| r.contains("api-key") || r.contains("stripe")),
            "expected a generic-api-key or stripe-* rule id, got {:?}",
            out.rule_ids
        );
    }

    #[test]
    fn generic_api_key_near_miss_empty_value_does_not_fire() {
        let text = "api_key = \"\" // left blank in this template";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(
            out.rule_ids.is_empty(),
            "empty-value assignment incorrectly matched a rule: {:?}",
            out.rule_ids
        );
    }

    // ── mode + toggle behaviour ─────────────────────────────────────────────

    #[test]
    fn disabled_is_a_no_op() {
        let text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP";
        let out = scan_ingest_text(
            text,
            &SecretScanConfig {
                enabled: false,
                mode: SecretScanMode::Redact,
            },
        );
        assert!(out.rule_ids.is_empty());
        assert_eq!(out.text, text);
    }

    #[test]
    fn redact_mode_removes_the_secret_from_the_returned_text() {
        let text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP";
        let out = scan_ingest_text(text, &redact_cfg());
        assert!(out.has_hits());
        assert!(
            !out.text.contains("AKIAABCDEFGHIJKLMNOP"),
            "raw secret leaked through Redact mode: {}",
            out.text
        );
        assert!(out.text.contains("[REDACTED_SECRET]"), "{}", out.text);
    }

    #[test]
    fn flag_only_mode_preserves_the_secret_in_the_returned_text() {
        // FlagOnly's whole point is "observe, don't mutate" — the episode
        // text stored/extracted/embedded downstream must be untouched.
        let text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(out.has_hits());
        assert_eq!(out.text, text);
    }

    #[test]
    fn clean_text_produces_no_hits() {
        let text = "Alice met Bob for coffee on Tuesday and discussed the roadmap.";
        let out = scan_ingest_text(text, &flag_cfg());
        assert!(out.rule_ids.is_empty(), "{:?}", out.rule_ids);
        assert_eq!(out.text, text);
        assert!(!out.has_hits());
    }
}

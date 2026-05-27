//! Provider cost rate table — bundled + runtime-override support.
//!
//! ## Purpose
//!
//! Provides `ProviderRates` (deserialized from `monitoring/provider-rates.toml`)
//! and the `PROVIDER_RATES` global registry. Cost-aware wrappers (e.g.
//! `TokenTrackingChatProvider`) call `PROVIDER_RATES.get()` to look up the
//! per-1k-token USD rate for a given `(provider, model)` pair and emit
//! `kremory_core_cost_usd_total` gauges.
//!
//! ## Initialization
//!
//! Call `init_bundled()` once at startup (idempotent — second call is a no-op)
//! to load the rates bundled at compile time. Use `init_from_path()` if the
//! caller wants to supply an updated rates file at runtime.
//!
//! ## TOML schema (monitoring/provider-rates.toml)
//!
//! ```toml
//! rates_as_of = "YYYY-MM-DD"
//!
//! [[providers]]
//! provider = "openai"
//! model    = "gpt-4o"
//! cost_per_1k_tokens_usd = 0.01
//! ```
//!
//! There is no `direction` field — the rate is a single value per model.
//! For chat models the output rate (more conservative / expensive) is used.

use std::path::Path;
use std::sync::OnceLock;

use serde::Deserialize;

/// Global provider rate table. Initialized once via `init_bundled()` or
/// `init_from_path()`. Reads are lock-free after initialization.
pub static PROVIDER_RATES: OnceLock<ProviderRates> = OnceLock::new();

/// Deserialized contents of `monitoring/provider-rates.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderRates {
    /// ISO-8601 date string indicating when these rates were last verified.
    pub rates_as_of: String,
    /// All rate entries across providers/models.
    pub providers: Vec<ProviderRateEntry>,
}

/// A single rate entry in the provider-rates table.
///
/// Re-exported via `kremory::observability` (spec line 457).
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderRateEntry {
    /// Provider identifier (e.g. `"openai"`, `"anthropic"`, `"voyage"`).
    pub provider: String,
    /// Model identifier (e.g. `"gpt-4o"`, `"claude-haiku-4-5"`).
    /// Use `"*"` as a wildcard for all models of a provider (e.g. Ollama).
    pub model: String,
    /// USD cost per 1,000 tokens.
    pub cost_per_1k_tokens_usd: f64,
    /// Embedding dimension count (only relevant for embedding models).
    #[serde(default)]
    pub dimensions: Option<u32>,
}

impl ProviderRates {
    /// Load bundled rates from `crates/kremory/monitoring/provider-rates.toml`
    /// (compile-time `include_str!`). The path is relative to this source file:
    /// `crates/kremory/src/core/rates.rs` → 2 levels up → `crates/kremory/` →
    /// `monitoring/provider-rates.toml`. The TOML lives INSIDE the published
    /// crate so the bundled rates ship with `cargo publish`.
    pub fn from_bundled() -> Result<Self, RatesError> {
        const BUNDLED: &str = include_str!("../../monitoring/provider-rates.toml");
        toml::from_str(BUNDLED).map_err(RatesError::Parse)
    }

    /// Load rates from a user-specified path at runtime.
    ///
    /// # Errors
    ///
    /// Returns `RatesError::Io` if the file cannot be read, or
    /// `RatesError::Parse` if the TOML is malformed.
    pub fn from_path(path: &Path) -> Result<Self, RatesError> {
        let content = std::fs::read_to_string(path).map_err(RatesError::Io)?;
        toml::from_str(&content).map_err(RatesError::Parse)
    }

    /// Look up the USD cost per 1k tokens for a `(provider, model)` pair.
    ///
    /// - Matches exact `provider` + `model` first.
    /// - Falls back to `provider` + `"*"` (wildcard model) if no exact match.
    ///
    /// Returns `None` if no matching entry exists in the table.
    pub fn lookup_rate(&self, provider: &str, model: &str) -> Option<f64> {
        // Exact match first.
        if let Some(r) = self
            .providers
            .iter()
            .find(|r| r.provider == provider && r.model == model)
        {
            return Some(r.cost_per_1k_tokens_usd);
        }
        // Wildcard model fallback (e.g. ollama/*).
        self.providers
            .iter()
            .find(|r| r.provider == provider && r.model == "*")
            .map(|r| r.cost_per_1k_tokens_usd)
    }
}

/// Initialize `PROVIDER_RATES` from the bundled TOML. Idempotent — a second
/// call is a no-op (the `OnceLock` value is already set).
///
/// # Errors
///
/// Returns `RatesError::Parse` if the bundled TOML fails to deserialize.
/// This indicates a compile-time asset issue and is unlikely in practice.
pub fn init_bundled() -> Result<(), RatesError> {
    let rates = ProviderRates::from_bundled()?;
    // OnceLock::set returns Err(value) if already set — we intentionally discard it.
    let _ = PROVIDER_RATES.set(rates);
    Ok(())
}

/// Initialize `PROVIDER_RATES` from a user-supplied file path at runtime.
/// Idempotent — a second call is a no-op.
///
/// # Errors
///
/// Returns `RatesError::Io` or `RatesError::Parse` if the file cannot be
/// read or parsed.
pub fn init_from_path(path: &Path) -> Result<(), RatesError> {
    let rates = ProviderRates::from_path(path)?;
    let _ = PROVIDER_RATES.set(rates);
    Ok(())
}

/// Errors that can occur when loading a rates table.
#[derive(Debug, thiserror::Error)]
pub enum RatesError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    // ── G_v012_4: provider_rates_bundled_defaults_load ───────────────────────

    /// G_v012_4 — Parse bundled rates TOML; assert `rates_as_of` matches the
    /// committed date and spot-check known rates for 3 embedder models and 5
    /// chat model entries.
    #[test]
    fn provider_rates_bundled_defaults_load() {
        let rates = ProviderRates::from_bundled().expect("bundled rates must parse");

        assert_eq!(
            rates.rates_as_of, "2026-05-27",
            "rates_as_of must match the committed date"
        );

        // ── Embedder spot-checks ─────────────────────────────────────────────
        let embedder_cases: &[(&str, &str)] = &[
            ("openai", "text-embedding-3-small"),
            ("voyage", "voyage-3"),
            ("local", "all-minilm-l6-v2"),
        ];
        for &(provider, model) in embedder_cases {
            assert!(
                rates.lookup_rate(provider, model).is_some(),
                "embedder rate for {provider}/{model} must exist in bundled TOML"
            );
        }

        // ── Chat spot-checks (2-arg lookup) ──────────────────────────────────
        let chat_cases: &[(&str, &str)] = &[
            ("openai", "gpt-4o-mini"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-haiku-4-5"),
            ("anthropic", "claude-sonnet-4-6"),
            ("ollama", "any-model"),
        ];
        for &(provider, model) in chat_cases {
            assert!(
                rates.lookup_rate(provider, model).is_some(),
                "chat rate for {provider}/{model} must exist in bundled TOML"
            );
        }
    }

    // ── G_v012_5: lookup_rate_unknown_provider_returns_none ──────────────────

    /// G_v012_5 — `lookup_rate` for a nonexistent provider/model pair must
    /// return `None` (no panics, no partial matches).
    #[test]
    fn lookup_rate_unknown_provider_returns_none() {
        let rates = ProviderRates::from_bundled().expect("bundled rates must parse");

        assert_eq!(
            rates.lookup_rate("nonexistent", "x"),
            None,
            "lookup_rate for unknown provider must return None"
        );
        assert_eq!(
            rates.lookup_rate("openai", "nonexistent-model"),
            None,
            "lookup_rate for known provider but unknown model must return None"
        );
    }
}

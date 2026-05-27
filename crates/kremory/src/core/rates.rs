//! Provider cost rate table — bundled + runtime-override support.
//!
//! ## Purpose
//!
//! Provides `ProviderRates` (deserialized from `monitoring/provider-rates.toml`)
//! and the `PROVIDER_RATES` global registry. Cost-aware wrappers (e.g.
//! `TokenTrackingChatProvider`) call `PROVIDER_RATES.get()` to look up the
//! per-1k-token USD rate for a given `(provider, model, direction)` triple and
//! emit `kremory_core_cost_usd_total` counters.
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
//! direction = "input"          # "input" | "output" — None for embed (single rate)
//! cost_per_1k_tokens_usd = 0.0025
//! ```

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
    /// All rate entries across providers/models/directions.
    pub providers: Vec<ProviderRate>,
}

/// A single rate entry in the provider-rates table.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderRate {
    /// Provider identifier (e.g. `"openai"`, `"anthropic"`, `"voyage"`).
    pub provider: String,
    /// Model identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    pub model: String,
    /// Token direction: `Some("input")` | `Some("output")` for chat/completion,
    /// `None` for embedding models (single rate).
    #[serde(default)]
    pub direction: Option<String>,
    /// USD cost per 1,000 tokens.
    pub cost_per_1k_tokens_usd: f64,
    /// Embedding dimension count (only relevant for embedding models).
    #[serde(default)]
    pub dimensions: Option<u32>,
}

impl ProviderRates {
    /// Load bundled rates from `monitoring/provider-rates.toml` (compile-time
    /// `include_str!`). The path is relative to this source file's location:
    /// `crates/kremory/src/core/rates.rs` → 4 levels up → project root →
    /// `monitoring/provider-rates.toml`.
    pub fn from_bundled() -> Result<Self, RatesError> {
        const BUNDLED: &str = include_str!("../../../../monitoring/provider-rates.toml");
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

    /// Look up the USD cost per 1k tokens for a `(provider, model, direction)` triple.
    ///
    /// - For chat/completion models: pass `direction = Some("input")` or
    ///   `Some("output")`.
    /// - For embedding models: pass `direction = None`.
    ///
    /// Returns `None` if no matching entry exists in the table.
    pub fn lookup_rate(&self, provider: &str, model: &str, direction: Option<&str>) -> Option<f64> {
        self.providers
            .iter()
            .find(|r| {
                r.provider == provider && r.model == model && r.direction.as_deref() == direction
            })
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

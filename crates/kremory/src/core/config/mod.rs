//! Pipeline configuration types.
//!
//! TD-243: split from a single 1631-line `config.rs` into per-concern
//! submodules (the file-size ratchet's WATCHED tier flags any file over
//! its 2026-08-12 baseline). This is a pure move — every type keeps its
//! original `core::config::<Name>` import path via the `pub use`
//! re-exports below, so no consumer (in-workspace or published-crate)
//! needs to change an import.
//!
//! - [`scan_configs`] — content-type + extraction-window + MinHash +
//!   entropy + secret-scan leaf config types.
//! - [`search`] — [`SearchConfig`], hybrid-search weighting.
//! - [`pipeline`] — [`PipelineConfig`] itself, plus the override-overlay
//!   and resolution-strategy/embedding-dim types it's built from.
//! - [`builder`] — [`PipelineConfigBuilder`], the fluent builder for
//!   [`PipelineConfig`] (and `PipelineConfig::builder()`, which
//!   constructs one — co-located here because it touches the builder's
//!   private `inner` field).
//! - [`telemetry`] — [`Config`], the core-layer telemetry prefix config
//!   (ADR D15). Re-exported at the crate root as `CoreConfig`.

mod builder;
mod pipeline;
mod scan_configs;
mod search;
mod telemetry;

#[cfg(test)]
mod tests;

pub use scan_configs::{
    ContentType, EntropyConfig, ExtractionWindowConfig, MinHashConfig, SecretScanConfig,
    SecretScanMode,
};
pub use search::SearchConfig;
pub use pipeline::{EmbeddingDim, PipelineConfig, ResolutionStrategy};
pub use builder::PipelineConfigBuilder;
pub use telemetry::Config;

// Crate-internal-only items. Not part of the public API (no `pub` in the
// original file either), but reachable at `core::config::X` from other
// crate-internal modules (facade/providers.rs, facade/builder.rs,
// memory/graph.rs) before this split — kept reachable at the same path.
pub(crate) use pipeline::{PipelineConfigOverrides, DEFAULT_CONTRADICTION_DETECTION_ENABLED};

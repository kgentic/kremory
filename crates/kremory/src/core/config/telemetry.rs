// ═══════════════════════════════════════════════════════════════════════════════
// Config — core-layer telemetry prefix config (ADR D15)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core-layer telemetry configuration.
///
/// Controls the `metrics_prefix` and `span_prefix` namespace so callers can
/// co-deploy multiple kremory instances without metric label collision (ADR D15).
///
/// Default prefixes match the canonical names in `monitoring/kremory-memory-slos.toml`.
/// Override only when running multiple kremory deployments in the same Prometheus
/// namespace (e.g. staging vs prod scraping into one cluster).
///
/// # Cardinality note (ADR D7)
///
/// Prefixes are `Option<String>` set once at startup — not per-request strings.
/// The prefix is prepended to the base metric name at registration time, not at
/// emit time, so there is no per-call allocation overhead.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Optional prefix prepended to all `metrics::counter!/histogram!/gauge!` names.
    ///
    /// Example: `Some("kremory_prod".to_string())` → `kremory_prod_core_tokens_total`.
    /// `None` (default) uses the canonical `kremory_core_*` namespace.
    pub metrics_prefix: Option<String>,

    /// Optional prefix prepended to all `tracing::info!/warn!/error!` span names.
    ///
    /// Example: `Some("prod".to_string())` → `prod.kremory.embed completed`.
    /// `None` (default) uses the canonical `kremory.*` span namespace.
    pub span_prefix: Option<String>,
}

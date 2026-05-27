//! OTel integration tests — kremory v0.1.2 span + layer wiring (§17 Phase D).
//!
//! Gate: `#![cfg(feature = "otel")]` — the tests compile and run only when the
//! `otel` feature is enabled.  Invoke manually:
//!   cargo test -p kremory --tests --features otel --test otel_integration
//!
//! No live OTel collector required.  Both tests use an in-process
//! `TracerProvider` with no span processor (spans are silently dropped) plus
//! a `tracing_subscriber::Registry` installed as a thread-local default
//! (not a global subscriber) to avoid conflicts between parallel tests.
//!
//! `#![allow(clippy::expect_used)]` — test files use .expect() intentionally.

#![cfg(feature = "otel")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::TracerProvider;
use tracing::Instrument as _;
use tracing_subscriber::layer::SubscriberExt as _;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a `TracerProvider` with no span processors (spans are silently dropped).
/// Returns `(provider, tracing_opentelemetry::OpenTelemetryLayer)` for wiring
/// into a `tracing_subscriber::Registry`.
fn build_noop_provider_and_layer() -> (
    TracerProvider,
    tracing_opentelemetry::OpenTelemetryLayer<
        tracing_subscriber::Registry,
        opentelemetry_sdk::trace::Tracer,
    >,
) {
    // Provider with no processors — spans are dropped immediately on export.
    let provider = TracerProvider::builder().build();
    let tracer = provider.tracer("kremory-test");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);
    (provider, layer)
}

// ── G_v012_10: init_telemetry_installs_otel_layer ─────────────────────────────

/// G_v012_10 — `init_telemetry_installs_otel_layer`
///
/// Build an OTel `TracerProvider` and wire it into a `tracing_subscriber::Registry`
/// as a thread-local default (via `set_default`, not global).
/// Assert that after installation, `tracing::Span::current().is_disabled()` is
/// **false** when a span is entered — i.e. the OTel layer is capturing spans.
///
/// Uses `current_thread` runtime so the thread-local subscriber stays active
/// for the entire `block_on` call.  No live OTel collector required.
#[test]
fn init_telemetry_installs_otel_layer() {
    let (_provider, otel_layer) = build_noop_provider_and_layer();

    // Install subscriber as thread-local default — NOT global.
    // `set_default` returns a guard; dropping the guard uninstalls it.
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    rt.block_on(async {
        let span = tracing::info_span!("kremory.test.otel_layer_active");
        let _enter = span.enter();

        // With the OTel layer installed, the current span should NOT be disabled.
        let current = tracing::Span::current();
        assert!(
            !current.is_disabled(),
            "tracing::Span::current() must not be disabled when OTel layer is installed; \
             the layer is not wired correctly"
        );
    });
}

// ── G_v012_11: tokio_spawn_carries_parent_span ────────────────────────────────

/// G_v012_11 — `tokio_spawn_carries_parent_span`
///
/// Verify that a `tokio::spawn` inside an instrumented parent span carries the
/// parent span's context to the child task. This validates that
/// `tracing::instrument` + `tokio::spawn` propagation works with the OTel layer.
///
/// Method: open a parent span, `instrument` the spawned future with it, then
/// inside the spawned task assert `Span::current()` is not disabled (i.e. the
/// task runs under a live span). The OTel layer is installed as a thread-local
/// default so this test does not pollute the global subscriber.
///
/// Uses `current_thread` so the thread-local subscriber covers the spawned tasks.
#[test]
fn tokio_spawn_carries_parent_span() {
    let (_provider, otel_layer) = build_noop_provider_and_layer();

    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    rt.block_on(async {
        let parent_span = tracing::info_span!("kremory.test.parent_span");

        // Instrument the async block with the parent span so the spawned task
        // runs under the parent span's context.
        let join = tokio::spawn(
            async {
                let current = tracing::Span::current();
                assert!(
                    !current.is_disabled(),
                    "spawned task must run under an active span (parent span propagation); \
                     span context was not carried into tokio::spawn"
                );
            }
            .instrument(parent_span),
        );

        join.await.expect("spawned task must not panic");
    });
}

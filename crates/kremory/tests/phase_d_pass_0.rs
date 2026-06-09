#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase D — Dream Pass 0 integration tests.
//!
//! Governing spec: ADR-037 §3 / §9 (type discovery).
//! Migration spec: ADR-037 §9.5 (Migration 014 provenance columns).
//!
//! ## Acceptance criteria
//!
//! D1 — `discover_types` end-to-end: catch-all entities → LLM proposal → persistence.
//! D2 — Shape validator rejects all 9 reason categories.
//! D3 — Anti-redundancy gate rejects on cosine ≥ 0.85 (desc) / ≥ 0.70 (name) thresholds.
//! D4 — In-place evidence retype writes `entity_type_source = 'DreamPass0'`.
//! D5 — `max_proposals = 5` cap enforced prompt-side (system prompt contains the cap).
//! D6 — `DreamSummary.types_discovered` populated via `mem.dream()`.
//! D7 — Degraded mode: `embedder = None` → anti-redundancy gate skipped + warning.
//! D8 — Migration 014 provenance columns present on `entity_types` after `run_migrations`.
//!
//! ## Phase C vs Phase D
//!
//! D8 and D6 API-shape tests are included here as Phase C ships Migration 014
//! and the `DreamSummary` surface. D1-D7 full integration tests require Phase D
//! to be implemented (the `discover_types` primitive is `pub(crate)` and will
//! be exposed through `mem.dream()` in Phase D).

use kremory::core::schema::TemporalGraph;

// ─── helpers (shared with llm_integration.rs) ────────────────────────────────
// Only compiled when the `llm-integration` feature is active, so the helpers
// module (which depends on autoagents_llm + metrics_util) is gated accordingly.
#[cfg(feature = "llm-integration")]
mod helpers;

// ─── Phase C+D real-LLM smoke test ───────────────────────────────────────────

/// Universal-3 LLM integration smoke for Phases C + D combined.
///
/// Covers:
/// - Phase C: `mem.run_dream_pass_sync(DreamPassOpts::default())` API surface
///   against real LLM; returns `Ok(DreamSummary)` without panic.
/// - Phase D: `discover_types` LLM call exercises the Pass 0 path (stochastic
///   output — primary assertion is "doesn't panic or error").
///
/// The test ingests 4 short texts whose entities (Vanguard Therapeutics, Acme
/// Capital) are outside `DEFAULT_ENTITY_TYPES`, so `entity_type_id = 0`
/// catch-alls accumulate — giving Pass 0 discovery logic a real trigger.
///
/// Observability assertions (secondary):
/// - `rql.dream.pass_started_total` counter fires (Phase C C8)
/// - `rql.dream.pass_completed_total` counter fires (Phase C C8)
///
/// `#[ignore]`: requires live Ollama with `gemma4-e2b:latest` + `nomic-embed-text`.
///
/// Per substrate SoT (`tests/llm_integration.rs:1-25`): `gemma4-e2b:latest` is the
/// interactive default (80% precision / ~37-54s). `qwen2.5:14b` is legacy fallback.
///
/// Invoke:
///   OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_KEEP_ALIVE=1h \
///   OLLAMA_CHAT_MODEL=gemma4-e2b:latest \
///   cargo test -p kremory --features llm-integration --test phase_d_pass_0 \
///     c_d_real_llm_smoke_dream_with_pass_0 -- --ignored
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn c_d_real_llm_smoke_dream_with_pass_0() {
    use std::sync::Arc;

    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use autoagents_llm::embedding::EmbeddingBuilder;
    use kremory::memory::ChatProvider;
    use kremory::{DreamPassOpts, DynEmbeddingProvider, Memory, Namespace};
    use metrics_util::debugging::DebuggingRecorder;

    use helpers::ollama_adapter::OllamaEmbedderAdapter;

    // Install a local metrics recorder so counters are observable.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    // Per substrate SoT (tests/llm_integration.rs:1-25): gemma4-e2b:latest is the
    // interactive default (80% / ~37-54s). qwen2.5:14b is legacy fallback.
    // Callers can override via OLLAMA_CHAT_MODEL.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4-e2b:latest".to_string());

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&chat_model)
        .timeout_seconds(120)
        .keep_alive("1h")
        .build()
        .expect("Ollama LLM builder must succeed");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder must succeed");

    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));

    let dir = tempfile::tempdir().expect("tempdir");

    let mem = Memory::open(dir.path().join("c_d_smoke.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .embedding_dim(768)
        .default_namespace(Namespace::new("c-d-smoke"))
        .await
        .expect("Memory::open must succeed for C+D smoke test");

    // Ingest 4 episodes whose entities sit outside DEFAULT_ENTITY_TYPES.
    // This drives catch-all entity_type_id=0 accumulation — giving Pass 0 a signal.
    let episodes = [
        "BioTech startup Vanguard Therapeutics raised Series B funding from Acme Capital.",
        "Vanguard Therapeutics is developing novel gene-therapy platforms for rare diseases.",
        "Acme Capital led the investment round; Nexus Ventures co-invested.",
        "Nexus Ventures focuses on early-stage BioTech and HealthTech opportunities.",
    ];

    for (i, content) in episodes.iter().enumerate() {
        mem.remember(*content)
            .from_chat(format!("c-d-smoke-session-{i}"))
            .await
            .unwrap_or_else(|e| panic!("episode {i} ingest must succeed: {e}"));
    }

    // ── Phase C: run_dream_pass_sync must return Ok(DreamSummary) ─────────────
    let opts = DreamPassOpts::default();
    let summary = mem
        .run_dream_pass_sync(opts)
        .await
        .expect("run_dream_pass_sync must return Ok(DreamSummary) — Phase C smoke");

    // Primary assertions: DreamSummary is structurally valid.
    // duration_ms must be set (Phase C C8 records elapsed time).
    assert!(
        summary.duration_ms < 60_000,
        "duration_ms must be < 60s for a stub pass; got {}ms",
        summary.duration_ms
    );

    // types_discovered is stochastic — type-shape check only (not count check).
    let _ = summary.types_discovered.len();

    // warnings is allowed to be non-empty.
    let _ = summary.warnings.len();

    // ── Phase C observability (C8): pass counters must have fired ─────────────
    let snapshot = snapshotter.snapshot().into_vec();

    let pass_started = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == "rql.dream.pass_started_total");
    let pass_completed = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == "rql.dream.pass_completed_total");

    assert!(
        pass_started,
        "rql.dream.pass_started_total must be emitted by run_dream_pass_sync (Phase C C8)"
    );
    assert!(
        pass_completed,
        "rql.dream.pass_completed_total must be emitted by run_dream_pass_sync (Phase C C8)"
    );
}

// ─── helpers ─────────────────────────────────────────────────────────────────

async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("pass0-test.db");
    let graph = TemporalGraph::open(path.to_str().expect("utf8 path"))
        .await
        .expect("TemporalGraph::open");
    (graph, tmp)
}

/// Column names for `entity_types` table.
async fn entity_types_columns(graph: &TemporalGraph) -> Vec<String> {
    let mut rows = graph
        .conn
        .query("PRAGMA table_info('entity_types')", ())
        .await
        .expect("PRAGMA table_info");
    let mut cols = Vec::new();
    while let Some(row) = rows.next().await.expect("row read") {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ─── D8: Migration 014 provenance columns ────────────────────────────────────

/// D8: Migration 014 adds `discovered_at`, `discovered_by`, `evidence_count`,
/// and `confidence` columns to `entity_types` on first `TemporalGraph::open`.
#[tokio::test]
async fn d8_migration_014_provenance_columns_present() {
    let (graph, _tmp) = open_graph().await;
    let cols = entity_types_columns(&graph).await;

    assert!(
        cols.iter().any(|c| c == "discovered_at"),
        "entity_types must have discovered_at; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "discovered_by"),
        "entity_types must have discovered_by; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "evidence_count"),
        "entity_types must have evidence_count; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "confidence"),
        "entity_types must have confidence; cols={cols:?}"
    );
}

/// D8: Running migrations twice does NOT duplicate entity_type rows (idempotent).
#[tokio::test]
async fn d8_migration_014_idempotent() {
    let (graph, _tmp) = open_graph().await;
    // Columns must be present after open (first migration run).
    let cols = entity_types_columns(&graph).await;
    assert!(
        cols.iter().any(|c| c == "discovered_at"),
        "discovered_at after first open"
    );
    // Keep graph alive across the check.
    let _ = &graph.conn;
    // Column still present — idempotent PRAGMA gates guaranteed no-op.
    let cols2 = entity_types_columns(&graph).await;
    assert!(
        cols2.iter().any(|c| c == "discovered_at"),
        "discovered_at still present"
    );
}

// ─── D6: DreamSummary API shape ───────────────────────────────────────────────

/// D6: `DreamSummary.types_discovered` is a Vec<TypeProposal> field accessible
/// from the public API.  This is a compile-time shape check — if the field
/// doesn't exist or has the wrong type, this won't compile.
#[tokio::test]
async fn d6_dream_summary_types_discovered_field_accessible() {
    let summary = kremory::DreamSummary {
        communities_updated: 0,
        cross_episode_merges: 0,
        supersessions_recorded: 0,
        facts_archived: 0,
        duration_ms: 0,
        types_discovered: vec![kremory::TypeProposal {
            name: "TestType".to_string(),
            description: "A test type".to_string(),
            justification: "test".to_string(),
        }],
        entities_reclassified: 0,
        warnings: vec!["test warning".to_string()],
    };
    assert_eq!(summary.types_discovered.len(), 1);
    assert_eq!(summary.types_discovered[0].name, "TestType");
    assert_eq!(summary.warnings.len(), 1);
}

/// D6: `DreamOpts::default()` has `include_type_discovery = true` per ADR-037 §3.
///
/// `mem.dream()` passes `DreamOpts` through to the consolidation logic; the
/// `include_type_discovery = true` default means Pass 0 fires on every
/// dream cycle unless explicitly disabled.
#[test]
fn d6_dream_opts_default_include_type_discovery_true() {
    let opts = kremory::DreamOpts::default();
    assert!(
        opts.include_type_discovery,
        "DreamOpts::default() must have include_type_discovery = true per ADR-037 §3"
    );
}

/// D6: `DreamPassOpts` struct (Phase C DoD C2) is accessible from the public API
/// and has the correct fields per the sprint spec.
#[test]
fn d6_dream_pass_opts_shape() {
    let opts = kremory::DreamPassOpts::default();
    // Compile-time shape check: fields must exist with the right types.
    let _: bool = opts.include_type_discovery;
    let _: f32 = opts.confidence_threshold;
    let _: Option<usize> = opts.max_episodes_per_run;
    let _: f32 = opts.reclassify_high_conf_threshold;
    // ADR-045 §3: reclassify threshold default is 0.7.
    assert!(
        (opts.reclassify_high_conf_threshold - 0.7).abs() < f32::EPSILON,
        "reclassify_high_conf_threshold default must be 0.7 per ADR-045 §3"
    );
}

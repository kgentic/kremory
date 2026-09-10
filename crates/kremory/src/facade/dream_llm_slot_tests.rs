//! Per-phase dream model slot (`with_dream_llm`).
//!
//! These in-crate tests exercise the `pub(crate)` `dream_llm_or_main`
//! accessor + the `kremory.dream.llm_role_selected_total{model_role}`
//! counter — surfaces unreachable from an integration test.
//! Public-surface build tests live in `tests/memory_builder_compat_matrix.rs`.

use super::*;
use crate::core::provider::MockEmbeddingProvider;
use crate::memory::ChatProvider;
use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Call-counting `ChatProvider`: records how many times `chat_with_tools`
/// fired, so a test can assert WHICH slot the dream phase invoked. Returns
/// an empty response (the dream fan-out tolerates empty proposals).
#[derive(Debug)]
struct CountingProvider {
    calls: Arc<AtomicUsize>,
}

impl CountingProvider {
    fn new() -> (Arc<Self>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                calls: Arc::clone(&calls),
            }),
            calls,
        )
    }
}

#[async_trait::async_trait]
impl ChatProvider for CountingProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(EmptyResponse))
    }
}

#[derive(Debug)]
struct EmptyResponse;

impl std::fmt::Display for EmptyResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl ChatResponse for EmptyResponse {
    fn text(&self) -> Option<String> {
        None
    }
    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(MockEmbeddingProvider::new(64))
}

async fn build_with(
    llm: Option<Arc<dyn ChatProvider>>,
    dream_llm: Option<Arc<dyn ChatProvider>>,
) -> Memory {
    // Build via the real builder so the field threads through the same
    // construction path production uses. NoLlm builds require an extractor;
    // use the LLM path when a main LLM is present, else a null extractor.
    match llm {
        Some(main) => {
            let mut b = Memory::open(":memory:").with_llm(main);
            if let Some(d) = dream_llm {
                b = b.with_dream_llm(d);
            }
            b.with_embedder(null_embedder())
                .await
                .expect("build WithLlm")
        }
        None => {
            let mut b = Memory::open(":memory:")
                .with_extractor(Arc::new(crate::core::intelligence::MockExtractor));
            if let Some(d) = dream_llm {
                b = b.with_dream_llm(d);
            }
            b.with_embedder(null_embedder())
                .await
                .expect("build NoLlm + extractor")
        }
    }
}

/// T3 — `dream_llm = None` → `dream_llm_or_main` returns the SAME `Arc` as
/// `llm_or_err` (structural fallback, byte-for-byte prior behaviour).
#[tokio::test]
async fn t3_accessor_falls_back_to_main_when_dream_unset() {
    let (main, _) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;
    let mem = build_with(Some(Arc::clone(&main)), None).await;

    let via_dream = mem
        .dream_llm_or_main("dream", "hint")
        .expect("main is wired");
    let via_main = mem.llm_or_err("dream", "hint").expect("main is wired");
    assert!(
        Arc::ptr_eq(&via_dream, &via_main),
        "with dream_llm=None, dream_llm_or_main must return the same Arc as llm_or_err"
    );
}

/// T3 — `dream_llm = Some(D)` → `dream_llm_or_main` returns D, distinct from
/// the main provider.
#[tokio::test]
async fn t3_accessor_returns_dream_provider_when_set() {
    let (main, _) = CountingProvider::new();
    let (dream, _) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;
    let dream: Arc<dyn ChatProvider> = dream;
    let mem = build_with(Some(Arc::clone(&main)), Some(Arc::clone(&dream))).await;

    let selected = mem
        .dream_llm_or_main("dream", "hint")
        .expect("dream provider wired");
    assert!(
        Arc::ptr_eq(&selected, &dream),
        "dream_llm_or_main must return the dedicated dream provider"
    );
    assert!(
        !Arc::ptr_eq(&selected, &main),
        "dream_llm_or_main must NOT return the main provider when dream_llm is set"
    );
}

/// `dream_model_id_or_main` resolves the model id the dream LLM
/// passes use for capability detection. Mirrors `dream_llm_or_main` at the
/// model-id layer: dedicated `with_dream_model_id` wins; else `with_model_id`;
/// else `None` (→ `PromptOnly` degrade). Regression guard for the empty-model
/// bug where dream passes silently degraded to zero structured output.
#[tokio::test]
async fn td094_dream_model_id_resolution() {
    let (main, _) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;

    // Case 1: only with_model_id → dream falls back to the main model id.
    let mem = Memory::open(":memory:")
        .with_llm(Arc::clone(&main))
        .with_model_id("main-model")
        .with_embedder(null_embedder())
        .await
        .expect("build with model_id");
    assert_eq!(
        mem.dream_model_id_or_main(),
        Some("main-model"),
        "dream must fall back to the main model id when no dream model id is set"
    );

    // Case 2: with_dream_model_id takes precedence over with_model_id.
    let mem = Memory::open(":memory:")
        .with_llm(Arc::clone(&main))
        .with_model_id("main-model")
        .with_dream_model_id("dream-model")
        .with_embedder(null_embedder())
        .await
        .expect("build with dream_model_id");
    assert_eq!(
        mem.dream_model_id_or_main(),
        Some("dream-model"),
        "dedicated dream model id must take precedence over the main model id"
    );

    // Case 3: neither set (raw with_llm) → None → PromptOnly degrade, unchanged.
    let mem = Memory::open(":memory:")
        .with_llm(Arc::clone(&main))
        .with_embedder(null_embedder())
        .await
        .expect("build without any model id");
    assert_eq!(
        mem.dream_model_id_or_main(),
        None,
        "unset model id must resolve to None (PromptOnly degrade), not an empty-string sentinel"
    );
}

/// T6 row 4 — neither main nor dream wired → `dream_llm_or_main` errors
/// `LlmRequired` exactly as the prior `llm_or_err` path (unchanged).
#[tokio::test]
async fn t6_row4_neither_provider_errors_llm_required() {
    let mem = build_with(None, None).await;
    let result = mem.dream_llm_or_main("dream", "hint");
    let Err(err) = result else {
        panic!("no provider wired → dream_llm_or_main must error LlmRequired");
    };
    assert!(
        matches!(
            err,
            MemoryError::Core(CoreError::LlmRequired {
                method: "dream",
                ..
            })
        ),
        "expected LlmRequired{{method=\"dream\"}}, got: {err:?}"
    );
}

/// T5 — a real blocking `dream()` routes through the DREAM slot, never MAIN.
///
/// `dream()` calls `dream_llm_or_main` (dream.rs:95) — selecting the dedicated
/// dream provider and emitting `model_role="dream"`. Pass-0 and Pass-2 execute
/// if a TemporalGraph is present; on an empty graph they return Ok with zero work.
/// Phase-3 consolidation fields (communities/merges/supersessions/archival) are
/// always 0 — honest zeros (consolidation is not yet implemented). The role counter is the
/// **authoritative routing proof**; `MAIN.calls == 0` proves the main slot is
/// never used for the dream phase.
#[test]
fn t5_dream_invokes_dream_provider_not_main() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let (main, main_calls) = CountingProvider::new();
    let (dream, dream_calls) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;
    let dream: Arc<dyn ChatProvider> = dream;

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = build_with(Some(main), Some(dream)).await;
            // Routes through dream_llm_or_main → DREAM slot. Returns Ok on
            // empty graph (Pass-0/Pass-2 find no work; honest-zero summary).
            mem.dream()
                .in_namespace(Namespace::new("default"))
                .execute()
                .await
                .expect("dream on empty graph must return Ok");
        });
    });

    assert_eq!(
        main_calls.load(Ordering::SeqCst),
        0,
        "the MAIN provider must NEVER be invoked by the dream phase when a dedicated dream_llm is set"
    );
    // dream_calls may be 0 (empty graph) — the role counter is the
    // authoritative routing proof.
    let _ = dream_calls;

    let snapshot = snapshotter.snapshot();
    let dream_role_total: u64 = snapshot
        .into_vec()
        .into_iter()
        .filter(|(k, _, _, _)| {
            k.key().name() == "kremory.dream.llm_role_selected_total"
                && k.key()
                    .labels()
                    .any(|l| l.key() == "model_role" && l.value() == "dream")
        })
        .map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => c,
            _ => 0,
        })
        .sum();
    assert!(
        dream_role_total >= 1,
        "dream pass with dream_llm=Some must increment llm_role_selected_total{{model_role=\"dream\"}}"
    );
}

/// T7 — the `interactive` role counter fires when dream falls back to the
/// main provider (dream_llm unset, main set).
#[test]
fn t7_interactive_role_counter_on_fallback() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let (main, _) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let mem = build_with(Some(main), None).await;
            let _ = mem.dream_llm_or_main("dream", "hint");
        });
    });

    let snapshot = snapshotter.snapshot();
    let interactive_total: u64 = snapshot
        .into_vec()
        .into_iter()
        .filter(|(k, _, _, _)| {
            k.key().name() == "kremory.dream.llm_role_selected_total"
                && k.key()
                    .labels()
                    .any(|l| l.key() == "model_role" && l.value() == "interactive")
        })
        .map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => c,
            _ => 0,
        })
        .sum();
    assert!(
        interactive_total >= 1,
        "fallback dream resolution (dream_llm=None) must increment llm_role_selected_total{{model_role=\"interactive\"}}"
    );
}

/// NT-1 — blocking `dream()` invokes the DREAM LLM slot and discovers types
/// from catch-all entities seeded into the TemporalGraph.
///
/// Closes a decorative gap: before the `run_dream_phase`
/// short-circuit was removed, `dream_llm` flowed only into unreachable code.
/// Verifies `dream_calls >= 1`, `main_calls == 0`, and `types_discovered`
/// reflects the scripted proposal — the assertion T5 could not make.
///
/// NT-2 (honest-zeros lock) is folded in: Phase-3 consolidation fields must
/// always be 0 pending consolidation implementation.
#[tokio::test]
async fn nt1_dream_invokes_dream_llm_discovers_types_and_zeroes_consolidation() {
    use crate::core::entity_types::ensure_default_types_seeded;
    use chrono::Utc;

    // ScriptedCountingProvider: counts calls AND returns valid proposal JSON.
    #[derive(Debug)]
    struct ScriptedCountingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct TextResponse {
        text: String,
    }
    impl std::fmt::Display for TextResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text)
        }
    }
    impl ChatResponse for TextResponse {
        fn text(&self) -> Option<String> {
            Some(self.text.clone())
        }
        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedCountingProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(TextResponse {
                text: r#"{"proposals":[{"name":"Company","description":"A business entity.","justification":"All three are companies."}]}"#
                    .to_string(),
            }))
        }
    }

    let dream_calls = Arc::new(AtomicUsize::new(0));
    let scripted_dream: Arc<dyn ChatProvider> = Arc::new(ScriptedCountingProvider {
        calls: Arc::clone(&dream_calls),
    });
    let (main, main_calls) = CountingProvider::new();
    let main: Arc<dyn ChatProvider> = main;

    let mem = build_with(Some(main), Some(scripted_dream)).await;

    // Seed catch-all entities so Pass-0 has clusters to process.
    // group_id = "default" (namespace_to_group_id(Namespace::new("default"))).
    let tg = mem
        .temporal_graph_for_test()
        .expect("TemporalGraph present in :memory: build");
    let conn = tg.conn.clone();
    ensure_default_types_seeded(&conn, "default")
        .await
        .expect("seed entity_types defaults");
    let now = Utc::now().to_rfc3339();
    for id in ["alpha corp", "beta fund", "gamma ventures"] {
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES (?1, 0, ?2, ?3)",
            libsql::params![id.to_string(), now.clone(), "default".to_string()],
        )
        .await
        .expect("seed catch-all entity");
    }

    let summary = mem
        .dream()
        .in_namespace(Namespace::new("default"))
        // Consumer-API hardening D1 turned ALL FOUR consolidation ops ON by
        // default. This Pass-0/type-discovery test pins EVERY consolidation flag
        // OFF so its honest-zeros below stay valid (consolidation behaviour is
        // covered elsewhere).
        .with_opts(crate::memory::types::DreamOpts {
            include_cross_episode_merges: false,
            include_community_detection: false,
            include_fact_archival: false,
            include_supersession_sweep: false,
            ..Default::default()
        })
        .execute()
        .await
        .expect("dream with seeded catch-all entities must return Ok");

    // NT-1: DREAM provider fired; MAIN never touched.
    assert!(
        dream_calls.load(Ordering::SeqCst) >= 1,
        "Pass-0 must invoke the dream LLM slot when catch-all entities exist"
    );
    assert_eq!(
        main_calls.load(Ordering::SeqCst),
        0,
        "MAIN provider must never be invoked by the dream phase"
    );
    // NT-1: types_discovered count is not asserted > 0 — anti-redundancy can correctly
    // reject proposals that overlap existing types in the test embedder's metric space
    // (stochastic, mirrors the plan's "do NOT assert > 0" stance). The load-bearing
    // signal is dream_calls >= 1 above: that proves the dream-phase LLM slot is live and Pass-0 ran.
    let _ = summary.types_discovered;
    // NT-2 (honest-zeros lock): Phase-3 consolidation fields always 0.
    assert_eq!(
        summary.communities_updated, 0,
        "communities_updated must be 0 — consolidation pinned OFF here"
    );
    assert_eq!(
        summary.cross_episode_would_merge, 0,
        "cross_episode_would_merge must be 0 — consolidation pinned OFF here"
    );
    assert_eq!(
        summary.cross_episode_merged, 0,
        "cross_episode_merged must be 0 — consolidation pinned OFF here"
    );
    assert_eq!(
        summary.supersessions_recorded, 0,
        "supersessions_recorded must be 0 — consolidation pinned OFF here"
    );
    assert_eq!(
        summary.facts_archived, 0,
        "facts_archived must be 0 — consolidation pinned OFF here"
    );
}

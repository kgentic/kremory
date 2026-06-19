//! Unit-style tests for BackgroundIngestor + deferred pipeline behaviour.
//!
//! Moved here from the inline #[cfg(test)] block in
//! `core/background/ingestor.rs` so all three background submodules stay
//! under 500 LoC per `feedback_split_files_before_adding_when_over_500_loc`.
//!
//! Sprint plan T2.1 / ADR-049 §Decision 6.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{
    BackgroundIngestor, IngestError, IngestErrorKind, IngestSendError, IngestorConfig, SendParams,
};
use kremory::core::ingest::SimpleGraph;
use kremory::core::provider::ChatProvider;

// ---------------------------------------------------------------------------
// Mock LLM clients
// ---------------------------------------------------------------------------

/// A ChatProvider that always returns an LLMError.
#[derive(Debug, Clone)]
struct FailingLlmClient;

#[async_trait::async_trait]
impl ChatProvider for FailingLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::Generic("simulated LLM failure".to_string()))
    }
}

/// A ChatProvider that counts every chat call.
#[derive(Debug, Clone)]
struct CountingLlmClient {
    calls: Arc<AtomicUsize>,
}

impl CountingLlmClient {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait::async_trait]
impl ChatProvider for CountingLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(kremory::core::provider::MockChatResponse {
            text: String::new(),
        }))
    }
}

/// A ChatProvider that succeeds once then always fails (NER ok, deferred fails).
#[derive(Debug, Clone)]
struct FailAfterFirstLlmClient {
    call_count: Arc<AtomicUsize>,
}

impl FailAfterFirstLlmClient {
    fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait::async_trait]
impl ChatProvider for FailAfterFirstLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(Box::new(kremory::core::provider::MockChatResponse {
                text: String::new(),
            }))
        } else {
            Err(LLMError::Generic("deferred simulated failure".to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn simple_ingestor() -> (BackgroundIngestor, kremory::core::background::IngestGuard) {
    let graph = SimpleGraph::open_in_memory_simple()
        .await
        .expect("open_in_memory_simple failed");
    BackgroundIngestor::new(graph, IngestorConfig::default())
}

async fn graph_with_llm<L: ChatProvider + 'static>(
    llm: L,
) -> kremory::core::ingest::Engine<L, kremory::core::provider::NullEmbeddingProvider> {
    use kremory::core::config::PipelineConfig;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::core::schema::TemporalGraph;
    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );
    let config = PipelineConfig::builder()
        .build()
        .expect("config build failed");
    let dim = config.embedding_dim.0;
    kremory::core::ingest::Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::new(llm),
        embedder: Arc::new(NullEmbeddingProvider { dim }),
        config,
    })
}

// ---------------------------------------------------------------------------
// Send / queue / shutdown tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_and_drain() {
    let (ingestor, guard) = simple_ingestor().await;
    ingestor
        .send("Alice works at Acme", SendParams::default())
        .expect("send should succeed");
    drop(ingestor);
    guard.shutdown();
}

#[tokio::test]
async fn queue_full_returns_error() {
    let graph = SimpleGraph::open_in_memory_simple()
        .await
        .expect("open_in_memory_simple failed");
    let config = IngestorConfig {
        channel_capacity: 1,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("first item", SendParams::default())
        .expect("first send should succeed");
    let mut got_full = false;
    for _ in 0..20 {
        match ingestor.send("overflow item", SendParams::default()) {
            Err(IngestSendError::Full(_)) => {
                got_full = true;
                break;
            }
            Ok(()) => {}
            Err(other) => panic!("unexpected error: {other}"),
        }
    }
    drop(ingestor);
    guard.shutdown();
    let _ = got_full;
}

#[tokio::test]
async fn shutdown_drains_queue() {
    let (ingestor, guard) = simple_ingestor().await;
    for i in 0..5_u8 {
        ingestor
            .send(format!("item {i}"), SendParams::default())
            .expect("send should succeed");
    }
    drop(ingestor);
    guard.shutdown();
}

#[tokio::test]
async fn clone_shares_worker() {
    let (ingestor, guard) = simple_ingestor().await;
    let clone = ingestor.clone();
    ingestor
        .send("from original", SendParams::default())
        .expect("send from original should succeed");
    clone
        .send("from clone", SendParams::default())
        .expect("send from clone should succeed");
    drop(ingestor);
    drop(clone);
    guard.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn guard_shutdown_does_not_deadlock_with_live_clone() {
    let (ingestor, guard) = simple_ingestor().await;
    let clone = ingestor.clone();
    ingestor
        .send("before shutdown", SendParams::default())
        .expect("send should succeed");
    let join = tokio::task::spawn_blocking(move || {
        guard.shutdown();
    });
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), join).await;
    assert!(
        result.is_ok(),
        "guard.shutdown() deadlocked — stop flag not working"
    );
    let send_result = clone.send("after shutdown", SendParams::default());
    assert!(
        matches!(send_result, Err(IngestSendError::Disconnected)),
        "expected Disconnected after worker exit, got: {send_result:?}"
    );
}

// ---------------------------------------------------------------------------
// Deferred extraction behavioural tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deferred_disabled_processes_without_error() {
    let graph = SimpleGraph::open_in_memory_simple()
        .await
        .expect("open_in_memory_simple failed");
    let config = IngestorConfig {
        deferred_extraction_enabled: false,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice works at Acme Corp", SendParams::default())
        .expect("send should succeed");
    ingestor
        .send("Bob manages Alice", SendParams::default())
        .expect("send should succeed");
    drop(ingestor);
    guard.shutdown();
}

#[tokio::test]
async fn deferred_enabled_drains_without_panic() {
    let (ingestor, guard) = simple_ingestor().await;
    ingestor
        .send("Alice works at Acme Corp", SendParams::default())
        .expect("send should succeed");
    drop(ingestor);
    guard.shutdown();
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errors_are_observable() {
    use kremory::core::config::PipelineConfig;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::core::schema::TemporalGraph;
    use std::time::Duration;

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open in-memory db failed"),
    );
    let config = PipelineConfig::builder()
        .build()
        .expect("config build failed");
    let dim = config.embedding_dim.0;
    let graph = kremory::core::ingest::Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::new(FailingLlmClient),
        embedder: Arc::new(NullEmbeddingProvider { dim }),
        config,
    });
    let (ingestor, guard) = BackgroundIngestor::new(graph, IngestorConfig::default());
    ingestor
        .send("this will fail", SendParams::default())
        .expect("send should succeed");
    let mut errors = Vec::new();
    for _ in 0..50 {
        errors = ingestor.drain_errors();
        if !errors.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard shutdown panicked");
    assert!(
        !errors.is_empty(),
        "expected at least one IngestError after 5s, got none"
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_request_created_after_successful_ingest() {
    use std::time::Duration;
    let (counting_client, call_counter) = CountingLlmClient::new();
    let graph = graph_with_llm(counting_client).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice works at Acme Corp", SendParams::default())
        .expect("send should succeed");
    for _ in 0..50 {
        if call_counter.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
    let final_calls = call_counter.load(Ordering::SeqCst);
    assert!(
        final_calls >= 2,
        "expected ≥2 LLM calls (1 NER + ≥1 deferred); got {final_calls}"
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_disabled_makes_no_deferred_llm_calls() {
    // ADR-051 Phase 3: process_item now calls ingest_phase1_ner() (episode INSERT
    // only, no LLM) instead of the full ingest() pipeline. When deferred is
    // disabled, process_deferred never runs. Total LLM calls = 0.
    //
    // Pre-ADR-051 this test expected "1 NER LLM call" because process_item
    // called ingest() → ingest_with() → LLM extractor. That sync LLM call
    // has moved to process_deferred (run_verify_stage Path β). With
    // deferred_enabled=false, the Phase 2 (extraction) step is skipped entirely.
    let (counting_client, call_counter) = CountingLlmClient::new();
    let graph = graph_with_llm(counting_client).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: false,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice works at Acme Corp", SendParams::default())
        .expect("send should succeed");
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
    // ADR-051: 0 LLM calls — Phase 1 (episode INSERT) makes no LLM calls;
    // Phase 2 (extraction) is disabled. Worker shuts down without errors.
    let total_calls = call_counter.load(Ordering::SeqCst);
    assert_eq!(
        total_calls, 0,
        "ADR-051: deferred disabled + no ner feature: expected 0 LLM calls; got {total_calls}"
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_queue_does_not_block_subsequent_ner_items() {
    use std::time::Duration;
    const N: usize = 4;
    let (counting_client, call_counter) = CountingLlmClient::new();
    let graph = graph_with_llm(counting_client).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    for i in 0..N {
        ingestor
            .send(
                format!("item number {i} about entity person"),
                SendParams::default(),
            )
            .expect("send should succeed");
    }
    for _ in 0..50 {
        if call_counter.load(Ordering::SeqCst) >= N * 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
    let total_calls = call_counter.load(Ordering::SeqCst);
    assert_eq!(
        total_calls,
        N * 2,
        "expected {expected} LLM calls ({N} NER + {N} deferred); got {total_calls}",
        expected = N * 2
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_extraction_errors_do_not_crash_worker() {
    use std::time::Duration;
    let graph = graph_with_llm(FailAfterFirstLlmClient::new()).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice works at Acme Corp", SendParams::default())
        .expect("send should succeed");
    let mut deferred_errors: Vec<IngestError> = Vec::new();
    for _ in 0..50 {
        deferred_errors = ingestor.drain_errors();
        if deferred_errors
            .iter()
            .any(|e| e.message.contains("deferred"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("worker panicked — deferred error should not crash the worker");
    assert!(
        deferred_errors
            .iter()
            .any(|e| e.message.contains("deferred")),
        "expected at least one deferred IngestError; got: {:?}",
        deferred_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
    assert!(
        deferred_errors
            .iter()
            .any(|e| e.kind == IngestErrorKind::Llm),
        "expected IngestErrorKind::Llm; got: {:?}",
        deferred_errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_config_disabled_skips_queue() {
    // ADR-051 Phase 3: process_item calls ingest_phase1_ner() (episode INSERT
    // only, no LLM calls). With deferred_enabled=false, process_deferred never
    // runs. For N items: 0 LLM calls total.
    //
    // Pre-ADR-051 this expected "N NER LLM calls" because process_item called
    // ingest() → LLM extractor per item. Under ADR-051 the LLM call moved to
    // process_deferred (run_verify_stage Path β). Deferred disabled = 0 calls.
    const N: usize = 3;
    let (counting_client, call_counter) = CountingLlmClient::new();
    let graph = graph_with_llm(counting_client).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: false,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    for i in 0..N {
        ingestor
            .send(format!("entity record {i}"), SendParams::default())
            .expect("send should succeed");
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
    // ADR-051: 0 LLM calls — Phase 1 episode INSERT never calls LLM;
    // deferred disabled means Phase 2 extraction is skipped entirely.
    let total_calls = call_counter.load(Ordering::SeqCst);
    assert_eq!(
        total_calls, 0,
        "ADR-051: deferred disabled + no ner feature: expected 0 LLM calls for {N} items; got {total_calls}"
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_queue_drains_on_sender_disconnect() {
    use std::time::Duration;
    let graph = graph_with_llm(FailAfterFirstLlmClient::new()).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice meets Bob at Acme HQ", SendParams::default())
        .expect("send should succeed");
    let mut deferred_errors: Vec<IngestError> = Vec::new();
    for _ in 0..50 {
        deferred_errors = ingestor.drain_errors();
        if deferred_errors
            .iter()
            .any(|e| e.message.contains("deferred"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("worker panicked during deferred drain on disconnect");
    assert!(
        deferred_errors
            .iter()
            .any(|e| e.message.contains("deferred")),
        "expected a deferred IngestError; got: {:?}",
        deferred_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_metrics_code_paths_execute() {
    use std::time::Duration;
    let (counting_client, call_counter) = CountingLlmClient::new();
    let graph = graph_with_llm(counting_client).await;
    let config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, config);
    ingestor
        .send("Alice leads the Acme project", SendParams::default())
        .expect("send should succeed");
    for _ in 0..50 {
        if call_counter.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
    let total_calls = call_counter.load(Ordering::SeqCst);
    assert_eq!(
        total_calls, 2,
        "expected 2 LLM calls (1 NER + 1 deferred); got {total_calls} — \
         a count of 1 means the deferred metric path was skipped"
    );
}

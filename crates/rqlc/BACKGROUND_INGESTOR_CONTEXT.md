# BackgroundIngestor Module Implementation Context

Generated from repomix analysis of rql-core codebase. All signatures and patterns extracted verbatim.

## Module Structure (lib.rs)

### Public Modules
```rust
pub mod chunker;
pub mod config;
pub mod error;
pub mod ner;
pub mod context;
pub mod contradiction;
pub mod extraction;
pub mod graph;
pub mod ingest;
pub mod intelligence;
pub mod provider;
pub mod resolver;
pub mod schema;
pub mod search;
pub mod speculative_cache;
pub mod text_utils;
```

### Public Exports
```rust
pub use error::{Result, RqlError};
pub use ner::inner::GlinerExtractor;  // feature-gated: #[cfg(feature = "ner")]
```

## Core Types

### RqlGraph (Main Intelligence Pipeline)
```rust
// ⚠️ SUPERSEDED: `RqlGraph<L: LlmClient, ...>` was removed in the AA migration.
// Current signature uses `ChatProvider` (from autoagents-llm) instead of `LlmClient`.
pub struct RqlGraph<L: ChatProvider, Emb: EmbeddingProvider> {
    pub(crate) graph: TemporalGraph,
    pub(crate) llm: Arc<L>,
    pub(crate) embedder: Arc<Emb>,
    pub(crate) config: PipelineConfig,
    /// Optional OOV auditor for language-agnostic entity safety net.
    pub(crate) oov_auditor: Option<text_utils::OovAuditor>,
}

impl<L: ChatProvider, Emb: EmbeddingProvider> RqlGraph<L, Emb> {
    pub fn new(
        graph: TemporalGraph,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
    ) -> Self { ... }

    pub fn graph(&self) -> &TemporalGraph { ... }

    pub async fn ingest(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        _group_id: Option<&str>,
        content_type: Option<ContentType>,
    ) -> Result<IngestionResult> {
        let extractor = NuExtractExtractor::new(Arc::clone(&self.llm));
        self.ingest_with(extractor, text, reference_time, _group_id, content_type)
            .await
    }

    pub async fn ingest_with<E: EntityExtractor>(
        &self,
        extractor: E,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        _group_id: Option<&str>,
        content_type: Option<ContentType>,
    ) -> Result<IngestionResult> {
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode
        let episode_id = self
            .graph
            .insert_episode(text, ref_time, Some("ingest"), None)
            .await?;

        // 2. Chunk
        let chunker = Chunker::new(self.config.chunk.clone());
        let chunks = chunker.split(text, &content_type);

        histogram!("rql.ingest.chunk_count").record(chunks.len() as f64);

        // 3. Extract from all chunks, merge results.
        // known_entities grows with each iteration so subsequent chunks receive
        // the entities already found in earlier chunks as context.
        let mut all_entities: Vec<ExtractedEntity> = Vec::new();
        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &all_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
            };
            let result = extractor.extract(chunk, &ctx).await?;
            all_entities.extend(result.entities);
            all_facts.extend(result.facts);
        // ... (continues with resolve, contradict, store phases)
    }
}
```

### IngestionResult
```rust
pub struct IngestionResult {
    /// The episode stored for this ingestion.
    pub episode_id: i64,
    /// Entity IDs that were created or merged.
    pub upserted_entities: Vec<String>,
    /// Fact IDs that were inserted.
    pub inserted_fact_ids: Vec<i64>,
    /// Fact IDs that were invalidated (contradicted/updated).
    pub invalidated_fact_ids: Vec<i64>,
    /// Entity pairs merged (canonical_id, alias_id).
    pub merged_entities: Vec<(String, String)>,
    /// Token usage across all LLM calls.
    pub token_usage: TokenUsage,
}
```

### Error Type
```rust
pub enum RqlError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("database error: {0}")]
    Database(#[from] libsql::Error),

    #[error("extraction failed: {0}")]
    Extraction(String),

    #[error("entity resolution failed: {0}")]
    Resolution(String),

    #[error("search error: {0}")]
    Search(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    // ... more variants
}

pub type Result<T> = std::result::Result<T, RqlError>;
```

### ContentType
```rust
pub enum ContentType {
    /// Plain unstructured text (e.g. transcripts, notes).
    Text,
    /// A discrete conversational message (e.g. chat turn, email).
    Message,
    /// Structured JSON payload; entity extraction is schema-aware.
    Json,
}
```

### PipelineConfig
```rust
pub struct PipelineConfig {
    /// Dimensionality of embedding vectors produced by the model.
    pub embedding_dim: EmbeddingDim,
    /// Chunking / splitting parameters.
    pub chunk: ChunkConfig,
    /// MinHash LSH parameters for near-duplicate detection.
    pub minhash: MinHashConfig,
    /// Entropy pre-filter parameters.
    pub entropy: EntropyConfig,
    /// Hybrid search fusion parameters.
    pub search: SearchConfig,
    /// Entity types the pipeline will extract and index (empty = all types).
    pub allowed_entity_types: Vec<String>,
    /// Relation / edge types the pipeline will resolve (empty = all types).
    pub allowed_edge_types: Vec<String>,
    /// Entity types that are explicitly excluded even if matched by extraction.
    pub excluded_entity_types: Vec<String>,
    pub cache_ttl: Duration,
    pub cache_max_entries: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            embedding_dim: EmbeddingDim::default(),  // 384
            chunk: ChunkConfig::default(),
            minhash: MinHashConfig::default(),
            entropy: EntropyConfig::default(),
            search: SearchConfig::default(),
            allowed_entity_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            excluded_entity_types: Vec::new(),
            cache_ttl: Duration::from_secs(300),
            cache_max_entries: 1000,
        }
    }
}
```

## Traits (intelligence.rs)

### EntityExtractor
```rust
pub trait EntityExtractor: Send + Sync {
    fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> impl std::future::Future<Output = Result<ExtractionResult>> + Send + 'a;
}
```

### ExtractionContext
```rust
pub struct ExtractionContext<'a> {
    /// If non-empty, only entities with these labels will be extracted.
    pub allowed_entity_types: &'a [String],
    /// If non-empty, only edges with these predicates will be extracted.
    pub allowed_edge_types: &'a [String],
    /// Entities already known to the graph — helps the extractor avoid duplicating context.
    pub known_entities: &'a [ExtractedEntity],
    /// Entity labels that must never be extracted regardless of `allowed_entity_types`.
    pub excluded_entity_types: &'a [String],
    /// The structural format of the input text, used to tailor the extraction prompt.
    pub content_type: ContentType,
}

impl<'a> Default for ExtractionContext<'a> {
    fn default() -> Self {
        Self {
            allowed_entity_types: &[],
            allowed_edge_types: &[],
            known_entities: &[],
            excluded_entity_types: &[],
            content_type: ContentType::Text,
        }
    }
}
```

### ExtractionResult
```rust
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub facts: Vec<ExtractedFact>,
}
```

## Provider Traits (provider.rs)

### ChatProvider
```rust
// ⚠️ SUPERSEDED: `LlmClient` / `LlmRequest` / `LlmResponse` were deleted in the AA migration.
// The BYOM seam is now `Arc<dyn ChatProvider>` from `autoagents-llm`.
// See rql-core/provider.rs for re-exports and stub equivalents.
//
// When `llm` feature is enabled:
//   pub use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatRole, ...};
// When `llm` feature is disabled:
//   provider::stubs::ChatProvider (compile-time stand-in)

// ChatProvider (autoagents_llm::chat::ChatProvider):
#[async_trait]
pub trait ChatProvider: Send + Sync {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;

    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;
}
```

### EmbeddingProvider
```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a;
}
```

## Test Implementations (for Testing)

### MockChatProvider
```rust
// ⚠️ SUPERSEDED: `NullLlmClient` and `MockLlmClient` were deleted in the AA migration.
// Use `MockChatProvider` (provider.rs) instead — it replaces both.
// `NullLlmClient` → `MockChatProvider::null()`
// `MockLlmClient` → `MockChatProvider::new(HashMap<String, String>)`

pub struct MockChatProvider {
    // Maps prompt substrings to canned response text.
    // Empty map → always returns "".
    responses: HashMap<String, String>,
}

impl MockChatProvider {
    /// Null variant — always returns empty text (replaces NullLlmClient).
    pub fn null() -> Self { ... }

    /// Fixture variant — returns canned text when any key appears in the prompt.
    pub fn new(responses: HashMap<String, String>) -> Self { ... }

    /// Convenience constructor — single key → value pair.
    pub fn with_response(key: impl Into<String>, value: impl Into<String>) -> Self { ... }
}

impl ChatProvider for MockChatProvider { ... }
```

### NullEmbeddingProvider
```rust
#[derive(Debug, Clone)]
pub struct NullEmbeddingProvider {
    pub dim: usize,
}

impl EmbeddingProvider for NullEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move { Ok(vec![0.0_f32; dim]) }
    }
}
```

### SimpleGraph Type Alias
```rust
/// Convenience type alias for tests and simple usage (no LLM/embedding).
// ⚠️ SUPERSEDED: was RqlGraph<NullLlmClient, NullEmbeddingProvider>.
// Now uses MockChatProvider (null variant) as the LLM stand-in.
pub type SimpleGraph = RqlGraph<MockChatProvider, NullEmbeddingProvider>;

impl SimpleGraph {
    /// Open an in-memory graph with null providers and default config.
    pub async fn open_in_memory_simple() -> Result<Self> {
        let graph = TemporalGraph::open_in_memory().await?;
        let config = PipelineConfig::builder().build()?;
        Ok(Self::new(
            graph,
            Arc::new(MockChatProvider::null()),
            Arc::new(NullEmbeddingProvider {
                dim: config.embedding_dim.0,
            }),
            config,
        ))
    }
}
```

## Dependencies (Cargo.toml)

**Async Runtime:**
```toml
tokio = { version = "1", features = ["full"] }
# Dev: tokio = { version = "1", features = ["full", "test-util"] }
```

**Test Patterns:**
All test modules follow this pattern:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    // ... test functions with #[tokio::test] or #[test]
}

#[tokio::test]
async fn test_something() { ... }
```

**Blocking Runtime Usage:**
```rust
fn block_on<F: Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}
```

## Key Implementation Patterns

1. **Full Pipeline Flow** (ingest_with):
   - Store episode (TemporalGraph::insert_episode)
   - Chunk text (Chunker::split)
   - Extract entities/facts (EntityExtractor::extract)
   - Resolve entities (CascadeResolver)
   - Detect contradictions (TwoPoolDetector)
   - Store in graph

2. **Metrics Integration:**
   - Uses `metrics` crate with histogram! and counter! macros
   - `rql.ingest.chunk_count`, `rql.ingest.extraction_time`, etc.

3. **Tokio Async:**
   - All LLM/embedding calls are async
   - Uses tokio::task::spawn_blocking for CPU-heavy work

4. **Feature Gating:**
   - `#[cfg(feature = "ner")]` for NER-specific code
   - `#[cfg(feature = "llm")]` for LLM features
   - `llm = ["dep:autoagents-llamacpp", "dep:autoagents-llm", "dep:futures"]` in Cargo.toml

5. **Test Utilities:**
   - `MockChatProvider` (replaces old `MockLlmClient` + `NullLlmClient`) for testing
   - `NullEmbeddingProvider` for lightweight embedding stand-in
   - `SimpleGraph` type alias (`RqlGraph<MockChatProvider, NullEmbeddingProvider>`) for in-memory testing

## Notes for BackgroundIngestor Implementation

1. **Async Trait Requirements:**
   - BackgroundIngestor trait should follow EntityExtractor pattern with impl Trait futures
   - Use `Send + 'async` bounds for executor compatibility

2. **DynEntityExtractor Shim:**
   - Create a wrapper type that adapts the BackgroundIngestor interface to EntityExtractor
   - Maintain ExtractionContext construction from existing graph state

3. **Work Queue:**
   - Use tokio::sync::mpsc or similar for background task queue
   - Respect Tokio graceful shutdown patterns

4. **Metrics:**
   - Follow existing patterns: `histogram!("rql.background_ingest....")`, `counter!(...)`
   - Track queue depth, processing time per item, error counts

5. **Result Propagation:**
   - Collect IngestionResult summaries from background worker
   - Provide access to cumulative results (via Result struct or Arc<Mutex<...>>)

6. **Configuration:**
   - Extend PipelineConfig or create BackgroundIngestionConfig sub-struct
   - Queue size, batch size, timeout behaviors, retry policies

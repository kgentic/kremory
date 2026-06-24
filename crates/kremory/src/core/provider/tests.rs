use super::*;

#[test]
fn null_embedding_returns_zero_vec() {
    let provider = NullEmbeddingProvider { dim: 384 };
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime build failed");
    let vec = rt.block_on(provider.embed("test")).expect("embed failed");
    assert_eq!(vec.len(), 384);
    assert!(vec.iter().all(|&v| v == 0.0_f32));
}

#[test]
fn null_embedding_dimension_matches() {
    let provider = NullEmbeddingProvider { dim: 128 };
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime build failed");
    let vec = rt
        .block_on(provider.embed("anything"))
        .expect("embed failed");
    assert_eq!(vec.len(), 128);
}

#[test]
fn mock_embedding_deterministic() {
    let provider = MockEmbeddingProvider::new(64);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime build failed");
    let a = rt
        .block_on(provider.embed("hello world"))
        .expect("embed failed");
    let b = rt
        .block_on(provider.embed("hello world"))
        .expect("embed failed");
    assert_eq!(a, b);
}

#[test]
fn mock_embedding_different_inputs_differ() {
    let provider = MockEmbeddingProvider::new(64);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime build failed");
    let a = rt
        .block_on(provider.embed("hello world"))
        .expect("embed failed");
    let b = rt
        .block_on(provider.embed("goodbye world"))
        .expect("embed failed");
    assert_ne!(a, b);
}

#[test]
fn token_usage_add() {
    let a = TokenUsage {
        prompt_tokens: 10,
        completion_tokens: 5,
    };
    let b = TokenUsage {
        prompt_tokens: 20,
        completion_tokens: 15,
    };
    let sum = a + b;
    assert_eq!(sum.prompt_tokens, 30);
    assert_eq!(sum.completion_tokens, 20);
    assert_eq!(sum.total(), 50);
}

#[test]
fn chat_msg_system_has_system_role() {
    let msg = chat_msg_system("you are helpful");
    assert_eq!(msg.role, ChatRole::System);
    assert_eq!(msg.content, "you are helpful");
    assert_eq!(msg.message_type, MessageType::Text);
}

#[test]
fn chat_msg_user_has_user_role() {
    let msg = chat_msg_user("hello");
    assert_eq!(msg.role, ChatRole::User);
    assert_eq!(msg.content, "hello");
    assert_eq!(msg.message_type, MessageType::Text);
}

#[tokio::test]
async fn mock_provider_null_returns_none() {
    let provider = MockChatProvider::null();
    let msgs = vec![chat_msg_user("extract entities")];
    let resp = provider
        .chat_with_tools(&msgs, None, None)
        .await
        .expect("mock chat_with_tools failed");
    // null provider returns "" which maps to None
    assert!(resp.text().is_none());
}

#[tokio::test]
async fn mock_provider_substring_match() {
    let mut map = std::collections::HashMap::new();
    map.insert("entities".to_string(), "[{\"name\":\"Alice\"}]".to_string());
    let provider = MockChatProvider::new(map);
    let msgs = vec![chat_msg_user("extract entities from this transcript")];
    let resp = provider
        .chat_with_tools(&msgs, None, None)
        .await
        .expect("mock chat_with_tools failed");
    assert_eq!(resp.text().as_deref(), Some("[{\"name\":\"Alice\"}]"));
}

#[tokio::test]
async fn mock_provider_no_match_returns_none() {
    let map = std::collections::HashMap::new();
    let provider = MockChatProvider::new(map);
    let msgs = vec![chat_msg_user("summarise the meeting")];
    let resp = provider
        .chat_with_tools(&msgs, None, None)
        .await
        .expect("mock chat_with_tools failed");
    assert!(resp.text().is_none());
}

// Real LlamaCppProvider smoke tests live in
// `rust-pipeline/tests/llamacpp_smoke.rs` per Vera D.1a cycle-1
// MEDIUM-1 + ADR-Phase-D.0 §7 — keeping `autoagents-llamacpp` out
// of rql-core's dev-dependencies is required for the strict BYOM
// invariant (`cargo tree -p rql-core | grep autoagents-llamacpp`
// must print empty).

// -----------------------------------------------------------------------
// ProviderCaps + capability_of — Phase 1 structured-output ladder (§5.2)
// -----------------------------------------------------------------------

#[test]
fn capability_of_anthropic_native() {
    assert_eq!(
        capability_of("claude-opus-4-7"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("claude-sonnet-4-6"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("claude-haiku-4-5-20251001"),
        ProviderCaps::NativeStructuredOutput
    );
}

#[test]
fn capability_of_openai_strict() {
    assert_eq!(
        capability_of("gpt-4o-2024-08-06"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("o1-preview"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("o3-mini"),
        ProviderCaps::NativeStructuredOutput
    );
}

#[test]
fn capability_of_openai_strict_later_dates() {
    // Dates strictly after 2024-08-06 must also be NativeStructuredOutput.
    assert_eq!(
        capability_of("gpt-4o-2024-12-17"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("gpt-4o-2025-01-01"),
        ProviderCaps::NativeStructuredOutput
    );
}

#[test]
fn capability_of_old_openai_not_strict() {
    // Pre-2024-08-06 OpenAI GPT models lack strict mode → PromptOnly.
    assert_eq!(capability_of("gpt-4-turbo"), ProviderCaps::PromptOnly);
    assert_eq!(capability_of("gpt-3.5-turbo"), ProviderCaps::PromptOnly);
    // gpt-4o with an older date → PromptOnly.
    assert_eq!(capability_of("gpt-4o-2024-05-13"), ProviderCaps::PromptOnly);
    // gpt-4o alone (no date) → PromptOnly (ambiguous, conservative).
    assert_eq!(capability_of("gpt-4o"), ProviderCaps::PromptOnly);
}

#[test]
fn capability_of_ollama_format_schema() {
    assert_eq!(capability_of("qwen2.5:14b"), ProviderCaps::FormatSchema);
    assert_eq!(
        capability_of("llama3.2:3b-instruct"),
        ProviderCaps::FormatSchema
    );
    // A hypothetical model name that looks like it could be OpenAI but uses Ollama tag syntax.
    assert_eq!(capability_of("gpt-oss:20b"), ProviderCaps::FormatSchema);
}

#[test]
fn capability_of_unknown_defaults_to_prompt_only() {
    // Bedrock ARNs / proxies / unrecognized strings.
    assert_eq!(
        capability_of("anthropic.claude-3-opus-bedrock"),
        ProviderCaps::PromptOnly
    );
    assert_eq!(capability_of("random-model-xyz"), ProviderCaps::PromptOnly);
    // Bedrock cross-region inference ARNs (us./eu./ap. prefixes) fall through
    // to PromptOnly via the default case — implicit conservative routing.
    assert_eq!(
        capability_of("us.anthropic.claude-opus-4-7-v1:0"),
        ProviderCaps::PromptOnly
    );
    assert_eq!(
        capability_of("eu.meta.llama3-70b-instruct-v1:0"),
        ProviderCaps::PromptOnly
    );
}

#[test]
fn capability_of_mistral_bare_ollama() {
    // Bare Mistral model names served via Ollama → FormatSchema.
    assert_eq!(capability_of("mistral-7b"), ProviderCaps::FormatSchema);
    assert_eq!(capability_of("mistral-nemo"), ProviderCaps::FormatSchema);
    // Bedrock Mistral ARNs (mistral. with dot) must remain PromptOnly.
    assert_eq!(
        capability_of("mistral.mistral-large-2407-v1:0"),
        ProviderCaps::PromptOnly
    );
}

#[test]
fn capability_of_gpt_4_dot_native() {
    // OpenAI gpt-4. numbered series → NativeStructuredOutput.
    assert_eq!(
        capability_of("gpt-4.1"),
        ProviderCaps::NativeStructuredOutput
    );
    assert_eq!(
        capability_of("gpt-4.5-preview"),
        ProviderCaps::NativeStructuredOutput
    );
    // gpt-4-turbo (hyphen, no dot) has no strict mode → PromptOnly.
    assert_eq!(capability_of("gpt-4-turbo"), ProviderCaps::PromptOnly);
}

#[test]
fn capability_of_claude_mythos_preview() {
    assert_eq!(
        capability_of("claude-mythos-preview"),
        ProviderCaps::NativeStructuredOutput
    );
}

// -----------------------------------------------------------------------
// Step 1 — AA adoption: mock_provider_basic
//
// gemma_provider_builds_v2 and gemma_chat_round_trip_v2 were Red-phase
// markers that duplicated llm_smoke::* tests (same assertions, different
// paths).  Removed in Green per spec: "Delete the v2 duplicates.
// Keep mock_provider_basic once the API supports it."
// -----------------------------------------------------------------------

#[cfg(test)]
mod aa_adoption_green {
    use super::*;

    /// Verifies `MockChatProvider::with_response` convenience constructor.
    ///
    /// Constructs a mock with a single key → value pair, sends a message
    /// containing the key, and asserts the response text exactly matches
    /// the fixture value.
    #[tokio::test]
    async fn mock_provider_basic() {
        let provider = MockChatProvider::with_response("arithmetic", "The answer is 4.");
        let msgs = vec![chat_msg_user("arithmetic: what is 2+2?")];
        let resp = provider
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("mock chat_with_tools should not fail");
        assert_eq!(
            resp.text().as_deref(),
            Some("The answer is 4."),
            "mock should return the exact fixture response for a matched key"
        );
    }
}

// -----------------------------------------------------------------------
// RecordReplayChatProvider — v0.2.4 Component 1 (spec §3.4 / §4 / §9 P2).
// -----------------------------------------------------------------------
#[cfg(test)]
mod record_replay {
    use super::*;

    /// A minimal stub `ChatProvider` whose `chat_with_tools` returns a fixed
    /// scripted response. No Ollama.
    #[derive(Debug)]
    struct StubProvider {
        response: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for StubProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> std::result::Result<
            Box<dyn autoagents_llm::chat::ChatResponse>,
            autoagents_llm::error::LLMError,
        > {
            Ok(Box::new(MockChatResponse {
                text: self.response.clone(),
            }))
        }
    }

    fn temp_cassette_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("kremory_vcr_{name}_{nanos}.json"));
        p
    }

    /// (a) Record captures a response; (b) Replay returns the recorded
    /// response for a matching fingerprint (round-trip). Spec §9 P2 DoD.
    #[tokio::test]
    async fn record_then_replay_round_trips() {
        let cassette = temp_cassette_path("round_trip");
        let stub = Arc::new(StubProvider {
            response: r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
        });

        // Record
        let recorder =
            RecordReplayChatProvider::record(stub.clone(), cassette.clone(), "gemma4-e2b:latest");
        let msgs = vec![chat_msg_user("Alice met Bob in Boston.")];
        let recorded = recorder
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("record chat should succeed");
        assert_eq!(
            recorded.text().as_deref(),
            Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#)
        );
        recorder.flush().expect("flush should write the cassette");

        // Replay — same request shape ⇒ same fingerprint ⇒ recorded response.
        let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
        let replayed = player
            .chat_with_tools(&msgs, None, None)
            .await
            .expect("replay should hit the recorded fingerprint");
        assert_eq!(
            replayed.text().as_deref(),
            Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#),
            "replay must return the exact recorded response text"
        );

        let _ = std::fs::remove_file(&cassette);
    }

    /// (b) Replay MISS returns a LOUD error naming the unmatched fingerprint.
    /// No silent default. Spec §4.3 / §9 P2 DoD.
    #[tokio::test]
    async fn replay_miss_is_loud_error() {
        let cassette = temp_cassette_path("miss");
        let stub = Arc::new(StubProvider {
            response: "recorded".to_string(),
        });
        let recorder =
            RecordReplayChatProvider::record(stub, cassette.clone(), "gemma4-e2b:latest");
        let recorded_msgs = vec![chat_msg_user("this exact request was recorded")];
        recorder
            .chat_with_tools(&recorded_msgs, None, None)
            .await
            .expect("record should succeed");
        recorder.flush().expect("flush");

        let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
        // A DIFFERENT request ⇒ different fingerprint ⇒ MISS.
        let other_msgs = vec![chat_msg_user("a totally different unrecorded request")];
        let err = player
            .chat_with_tools(&other_msgs, None, None)
            .await
            .expect_err("unrecorded fingerprint must produce a loud error, not a default");
        let msg = err.to_string();
        assert!(
            msg.contains("cassette MISS"),
            "error must name the miss: {msg}"
        );
        assert!(
            msg.contains("fingerprint="),
            "error must name the unmatched fingerprint: {msg}"
        );

        let _ = std::fs::remove_file(&cassette);
    }

    /// (c) `model_id()` returns the caller-supplied/cassette model, NOT empty.
    /// Spec §3.4 ASMP-002 / §9 P2 DoD.
    #[tokio::test]
    async fn model_delegation_not_default_empty() {
        let cassette = temp_cassette_path("model");
        let stub = Arc::new(StubProvider {
            response: "x".to_string(),
        });

        // Record + Passthrough return the caller-supplied model string.
        let recorder =
            RecordReplayChatProvider::record(stub.clone(), cassette.clone(), "gemma4-e2b:latest");
        assert_eq!(recorder.model_id(), "gemma4-e2b:latest");
        let pass = RecordReplayChatProvider::passthrough(stub.clone(), "gemma4-e2b:latest");
        assert_eq!(pass.model_id(), "gemma4-e2b:latest");

        // Replay returns the cassette header model.
        recorder.flush().expect("flush");
        let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
        assert_eq!(
            player.model_id(),
            "gemma4-e2b:latest",
            "replay model_id() must read the cassette header, not empty"
        );
        assert_ne!(
            player.model_id(),
            "",
            "model_id() must never collapse to empty"
        );

        let _ = std::fs::remove_file(&cassette);
    }

    /// (LOW-003) Record→replay round-trip through the production FormatSchema
    /// arm: a POPULATED `StructuredOutputFormat` participates in the fingerprint
    /// (§4.2). The other record_replay tests all pass `None` schema, so this is
    /// the only coverage that locks the `Some(schema)` serialization branch.
    /// Asserts (a) the populated-schema request round-trips, and (b) a DIFFERENT
    /// schema yields a cassette MISS — proving the schema is fingerprinted.
    #[tokio::test]
    async fn populated_schema_participates_in_fingerprint() {
        use autoagents_llm::chat::StructuredOutputFormat;

        let cassette = temp_cassette_path("populated_schema");
        let stub = Arc::new(StubProvider {
            response: r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
        });

        // Mirror how production builds the schema (structured.rs FormatSchema arm).
        let schema_a = StructuredOutputFormat {
            name: "entities".to_string(),
            description: None,
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": { "entities": { "type": "array" } },
                "required": ["entities"]
            })),
            strict: Some(false),
        };
        let msgs = vec![chat_msg_user("Alice met Bob in Boston.")];

        // Record with the populated schema.
        let recorder =
            RecordReplayChatProvider::record(stub.clone(), cassette.clone(), "gemma4-e2b:latest");
        recorder
            .chat_with_tools(&msgs, None, Some(schema_a.clone()))
            .await
            .expect("record chat with schema should succeed");
        recorder.flush().expect("flush should write the cassette");

        // Replay with the SAME schema ⇒ same fingerprint ⇒ recorded response.
        let player = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
        let replayed = player
            .chat_with_tools(&msgs, None, Some(schema_a.clone()))
            .await
            .expect("replay with the recorded schema should hit");
        assert_eq!(
            replayed.text().as_deref(),
            Some(r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#),
            "populated-schema replay must return the exact recorded response"
        );

        // Replay with a DIFFERENT schema (same messages/tools/model) ⇒ different
        // fingerprint ⇒ MISS. Proves the schema participates in the fingerprint.
        let schema_b = StructuredOutputFormat {
            name: "entities".to_string(),
            description: None,
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": { "facts": { "type": "array" } },
                "required": ["facts"]
            })),
            strict: Some(false),
        };
        let player2 = RecordReplayChatProvider::replay(cassette.clone()).expect("replay load");
        let err = player2
            .chat_with_tools(&msgs, None, Some(schema_b))
            .await
            .expect_err("a different schema must MISS, proving schema is fingerprinted");
        let msg = err.to_string();
        assert!(
            msg.contains("cassette MISS"),
            "different-schema request must miss: {msg}"
        );

        let _ = std::fs::remove_file(&cassette);
    }

    /// (NEW-201) Compile-level proof the decorator is `Send + Sync`: it must
    /// bind in the `Arc<dyn ChatProvider>` position. A `RefCell`-based impl
    /// would fail this bound.
    #[tokio::test]
    async fn decorator_is_send_sync_as_dyn() {
        let stub = Arc::new(StubProvider {
            response: "y".to_string(),
        });
        let provider: Arc<dyn ChatProvider> = Arc::new(RecordReplayChatProvider::passthrough(
            stub,
            "gemma4-e2b:latest",
        ));
        fn assert_send_sync<T: Send + Sync>(_t: &T) {}
        assert_send_sync(&provider);
    }
}

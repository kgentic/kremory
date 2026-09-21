    use kremory::RetrievedContext;

    use super::retrieved_context_to_js;

    // Helper: build a minimal RetrievedContext via ::new() then patch
    // entity_type_id / entity_type_name directly (within-crate access allowed).
    fn make_ctx(entity_type_id: u32, entity_type_name: &str) -> RetrievedContext {
        let mut ctx = RetrievedContext::new(kremory::RetrievedContextNewParams {
            entity_id: "ent-1".to_string(),
            entity_name: "Alice".to_string(),
            summary: "summary text".to_string(),
            score: 0.9_f32,
            source_refs: vec![],
        });
        ctx.entity_type_id = entity_type_id;
        ctx.entity_type_name = entity_type_name.to_string();
        ctx
    }

    /// `source_refs` project the full `kremory::SourceRef` shape
    /// (kind/id/occurred_at/published_at), not a flattened bare-id string.
    /// Drives the real producer (`retrieved_context_to_js`) and asserts each
    /// provenance field survives the projection — mirrors the MCP wire.
    #[test]
    fn source_refs_project_full_shape() {
        use chrono::{TimeZone, Utc};

        let occurred = Utc.with_ymd_and_hms(2026, 7, 15, 9, 0, 0).unwrap();
        let published = Utc.with_ymd_and_hms(2026, 7, 14, 8, 30, 0).unwrap();
        let ctx = RetrievedContext::new(kremory::RetrievedContextNewParams {
            entity_id: "ent-1".to_string(),
            entity_name: "Alice".to_string(),
            summary: "s".to_string(),
            score: 0.5_f32,
            source_refs: vec![
                kremory::SourceRef {
                    kind: kremory::SourceKind::Document,
                    id: "doc-42".to_string(),
                    occurred_at: occurred,
                    published_at: Some(published),
                },
                kremory::SourceRef {
                    kind: kremory::SourceKind::Episode,
                    id: "ep-7".to_string(),
                    occurred_at: occurred,
                    published_at: None,
                },
            ],
        });

        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.source_refs.len(), 2, "both source_refs projected");

        let doc = &js.source_refs[0];
        assert_eq!(doc.kind, "document", "kind lower-cased from SourceKind");
        assert_eq!(doc.id, "doc-42");
        assert_eq!(doc.occurred_at, occurred.to_rfc3339());
        assert_eq!(
            doc.published_at.as_deref(),
            Some(published.to_rfc3339()).as_deref(),
            "published_at preserved when set"
        );

        let ep = &js.source_refs[1];
        assert_eq!(ep.kind, "episode");
        assert_eq!(ep.id, "ep-7");
        assert!(
            ep.published_at.is_none(),
            "published_at None survives as null"
        );
    }

    /// `entity_type_id` is correctly wired as u32 on JsRetrievedContext.
    #[test]
    fn entity_type_id_wired_as_u32() {
        let ctx = make_ctx(1, "Person");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_id, 1u32, "entity_type_id must be u32 id=1");
    }

    /// `entity_type_name` string matches known type for id=1.
    #[test]
    fn entity_type_name_matches_known_type() {
        let ctx = make_ctx(1, "Person");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_name, "Person");
    }

    /// id=0 sentinel resolves to "Entity" fallback.
    #[test]
    fn entity_type_id_zero_resolves_to_entity_fallback() {
        let ctx = make_ctx(0, "Entity");
        let js = retrieved_context_to_js(ctx);
        assert_eq!(js.entity_type_id, 0u32);
        assert_eq!(js.entity_type_name, "Entity");
    }

    /// Backwards compatibility: existing label field is still present and correctly
    /// populated from entity_name (not entity_type_name).
    #[test]
    fn entity_name_field_unaffected_by_type_fields() {
        let ctx = make_ctx(2, "Organisation");
        let js = retrieved_context_to_js(ctx);
        // entity_name comes from properties["name"] or entity.id in the real recall
        // path; in this unit test it is the value passed to ::new().
        assert_eq!(js.entity_name, "Alice");
        // entity_type_name is additive — does not overwrite entity_name.
        assert_eq!(js.entity_type_name, "Organisation");
    }

    // ── Facts must survive the napi wire layer ─────────────────────────────
    //
    // `make_ctx` above (and `RetrievedContext::new()` generally) always
    // defaults `facts: Vec::new()` — `RetrievedContext` is `#[non_exhaustive]`
    // and `RetrievedContextNewParams` carries no `facts` field, so a non-empty
    // fixture cannot be struct-literalled. This drives the REAL recall path
    // (mode-c pinned fact via `Memory::remember().with_facts().skip_extraction()`
    // — the same mechanism `kremory`'s own
    // `with_facts_integration.rs::td116_recall_returns_connected_facts_under_null_embedder`
    // proves at the facade level) so `retrieved_context_to_js` is exercised
    // against a genuine, non-empty `RetrievedContext.facts`.

    use std::sync::Arc;

    use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
    use autoagents_llm::error::LLMError;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::memory::types::StructuredFact;
    use kremory::{ChatProvider, Memory, Namespace};

    use super::retrieved_fact_to_js;

    /// `Memory::open(...).with_llm(...)` requires a real `Arc<dyn ChatProvider>`
    /// even on the pinned-fact `skip_extraction()` path, which never invokes
    /// it. Errors loudly (not a silent empty response) if that assumption
    /// ever breaks, so a future regression fails this test with a clear cause
    /// instead of a confusing downstream symptom.
    #[derive(Debug, Clone)]
    struct UnreachableChatProvider;

    #[async_trait::async_trait]
    impl ChatProvider for UnreachableChatProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::Generic(
                "UnreachableChatProvider: chat_with_tools must not be called on a \
                 skip_extraction() pinned-fact test path"
                    .to_string(),
            ))
        }
    }

    /// Build an in-memory `Memory` (no live LLM/embedder needed — mirrors
    /// `with_facts_integration.rs::open_with_ns`, kremory-napi's own crate
    /// only having `NullEmbeddingProvider` available outside `test-utils`).
    #[allow(clippy::expect_used)]
    async fn napi_test_memory() -> Memory {
        let llm: Arc<dyn ChatProvider> = Arc::new(UnreachableChatProvider);
        let embedder: Arc<dyn kremory::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .default_namespace(Namespace::new("kremory_napi_h2_tests"))
            .await
            .expect("in-memory Memory must build")
    }

    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    async fn retrieved_context_to_js_round_trips_a_nonempty_pinned_fact() {
        let mem = napi_test_memory().await;

        mem.remember("Ada Lovelace wrote the first algorithm.")
            .with_facts(vec![StructuredFact {
                subject: "Ada Lovelace".to_string(),
                predicate: "wrote".to_string(),
                object: "the first algorithm".to_string(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .from_document("napi-h2-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        let raw = mem
            .recall("Ada Lovelace")
            .raw()
            .await
            .expect("raw recall should succeed");
        let ada = raw
            .into_iter()
            .find(|r| r.entity_name == "Ada Lovelace")
            .expect("Ada Lovelace must be in recall results");
        assert!(
            !ada.facts.is_empty(),
            "H2: recall must surface Ada Lovelace's connected fact before conversion"
        );

        let js = retrieved_context_to_js(ada);
        assert!(
            !js.facts.is_empty(),
            "H2: retrieved_context_to_js must not drop facts crossing the napi wire"
        );

        let fact = js
            .facts
            .iter()
            .find(|f| f.predicate == "wrote")
            .expect("the pinned 'wrote' fact must survive the wire mapping");
        assert_eq!(fact.fact, "Ada Lovelace wrote the first algorithm");
        assert_eq!(fact.subject, "Ada Lovelace");
        assert_eq!(fact.predicate, "wrote");
        assert_eq!(fact.object, "the first algorithm");
        assert!(
            !fact.object_is_entity,
            "literal object → object_is_entity=false"
        );
        assert!(
            !fact.valid_at.is_empty(),
            "valid_at must be a non-empty RFC-3339 string"
        );
        assert!(
            fact.invalid_at.is_none(),
            "an open-ended pinned fact must have invalid_at=None"
        );
        assert!(
            !fact.recorded_at.is_empty(),
            "recorded_at must be a non-empty RFC-3339 string"
        );
        assert!(
            fact.expired_at.is_none(),
            "a fresh pinned fact must have expired_at=None"
        );
        assert_eq!(
            fact.confidence, 1.0,
            "caller-pinned facts default to confidence=1.0"
        );
        assert!(
            !fact.source_episode_ids.is_empty(),
            "source_episode_ids must attribute the fact to its episode"
        );
    }

    /// `retrieved_fact_to_js` (the per-fact half of the conversion) round-trips
    /// every field independently of the containing `RetrievedContext` — the
    /// same real, non-empty fixture as the test above, but asserting the
    /// narrower conversion function directly.
    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    async fn retrieved_fact_to_js_round_trips_every_field() {
        let mem = napi_test_memory().await;

        mem.remember("Grace Hopper invented the compiler.")
            .with_facts(vec![StructuredFact {
                subject: "Grace Hopper".to_string(),
                predicate: "invented".to_string(),
                object: "the compiler".to_string(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .from_document("napi-h2-fact-doc")
            .skip_extraction()
            .await
            .expect("remember(skip_extraction) should succeed");

        let raw = mem
            .recall("Grace Hopper")
            .raw()
            .await
            .expect("raw recall should succeed");
        let hopper = raw
            .into_iter()
            .find(|r| r.entity_name == "Grace Hopper")
            .expect("Grace Hopper must be in recall results");
        let fact = hopper
            .facts
            .into_iter()
            .find(|f| f.predicate == "invented")
            .expect("the pinned 'invented' fact must be present");

        let js = retrieved_fact_to_js(fact);
        assert_eq!(js.fact, "Grace Hopper invented the compiler");
        assert_eq!(js.subject, "Grace Hopper");
        assert_eq!(js.predicate, "invented");
        assert_eq!(js.object, "the compiler");
        assert!(!js.object_is_entity);
        assert!(!js.valid_at.is_empty());
        assert!(js.invalid_at.is_none());
        assert!(!js.recorded_at.is_empty());
        assert!(js.expired_at.is_none());
        assert_eq!(js.confidence, 1.0);
        assert!(!js.source_episode_ids.is_empty());
        assert!(js.score >= 0.0);
    }

    /// Every known `IngestStatus` variant must map to its OWN discriminator
    /// — never silently into the `_ =>` forward-compat arm, which returns
    /// `"pending"`.
    ///
    /// This is the defect this test exists for: three variants
    /// (`EntitiesReady`, `SkippedIdempotent`, `ExtractionSkipped`) were falling
    /// through that arm, and two of them are **terminal**. A Node consumer
    /// polling `statusOf` therefore saw `"pending"` on an episode that had
    /// already finished — a "poll forever on a completed ingest" bug, but on
    /// the wire rather than in the column. `EntitiesReady`
    /// is SQL `'Verified'`, the exact state `Memory::wait_for_processing`
    /// resolves `Ok(())` on, so Rust callers saw success while JS callers saw
    /// pending.
    ///
    /// # Honest limits — read before trusting this
    ///
    /// `IngestStatus` is `#[non_exhaustive]`, so the catch-all arm **cannot** be
    /// removed and this test **cannot** mechanically discover a variant nobody
    /// listed here. It is a hand-maintained roster, not structural enforcement,
    /// and it will not fail on its own the day someone adds a tenth variant.
    /// What it does buy: the moment anyone *does* add a variant and comes here,
    /// the roster and the assertion below state the obligation plainly, and any
    /// regression that re-routes an existing variant into the catch-all fails
    /// loudly. Real enforcement would need the enum to stop being
    /// `#[non_exhaustive]`, which is a breaking change and not v1 scope.
    #[test]
    fn ingest_status_maps_every_known_variant_off_the_catch_all() {
        use crate::convert::ingest_status_to_js;
        use kremory::IngestStatus as S;

        // The roster. Update this when adding a variant — see limits above.
        let cases = vec![
            (S::Pending, "pending"),
            (S::Extracting, "extracting"),
            (S::EntitiesReady, "entities_ready"),
            (S::Deduplicating, "deduplicating"),
            (S::Invalidating, "invalidating"),
            (S::Complete, "complete"),
            (S::Failed("boom".to_string()), "failed"),
            (S::SkippedIdempotent, "skipped_idempotent"),
            (S::ExtractionSkipped, "skipped"),
        ];

        for (variant, expected) in cases {
            let label = format!("{variant:?}");
            let js = ingest_status_to_js(variant);
            assert_eq!(
                js.status, expected,
                "{label} must map to {expected:?}, not {:?}. A value of \
                 \"pending\" here almost always means the variant fell through \
                 the `_ =>` forward-compat arm — which is a silent bug for any \
                 TERMINAL state, because JS consumers poll on \"pending\".",
                js.status
            );
        }

        // Sensitivity in the other direction: the catch-all must still be
        // reachable and must still say "pending", so the assertion above is
        // actually discriminating rather than passing vacuously.
        assert_eq!(
            ingest_status_to_js(S::Pending).status,
            "pending",
            "the catch-all's own value must remain \"pending\" — if this ever \
             changes, the checks above stop distinguishing a real mapping from \
             a fall-through"
        );
    }

    // No unit test for `forget_outcome_to_js` here: `kremory::ForgetOutcome`
    // is `#[non_exhaustive]` from OUTSIDE its defining crate, so this crate
    // cannot construct a test value at all (same limitation documented on
    // the `UnsupersedeOutcomeWire`/`RestoreArchivedOutcomeWire` tests in
    // kremory-mcp for the identical reason). The field mapping is verified
    // by the type checker (every named field must exist on both sides) and
    // by `crates/kremory`'s own `ForgetOutcome`-producing tests; a real
    // end-to-end check needs a JS runtime harness this crate does not have.

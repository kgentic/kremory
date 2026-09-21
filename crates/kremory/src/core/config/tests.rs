// ─── Tests ───────────────────────────────────────────────────────────────────


use std::time::Duration;

    use super::*;

    #[test]
    fn test_default_config_builds() {
        let result = PipelineConfig::builder().build();
        assert!(result.is_ok(), "default config should build without error");
    }

    /// The sweep knobs' defaults must match whatever the CURRENT deliberate
    /// default is (not "byte-identical to the earlier default" — that framing
    /// died with the `rrf_k` flip; see the field's own doc comment for
    /// the measured justification). `rrf_k=1` (measured
    /// win over Cormack et al.'s general-purpose 60), `content_stream_weight
    /// =1.0` (equal-weight fusion, unchanged). Guards against a default drift
    /// silently changing the baseline sweep point.
    #[test]
    fn search_sweep_knob_defaults_unchanged() {
        let search = SearchConfig::default();
        assert_eq!(search.rrf_k, 1, "default RRF k must stay 1");
        assert_eq!(
            search.content_stream_weight, 1.0,
            "default content_stream_weight must stay 1.0 (equal-weight fusion)"
        );
        // The builder path (used by open_graph before env overrides) must agree.
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.rrf_k, 1);
        assert_eq!(built.content_stream_weight, 1.0);
        // The new builder setter threads the value.
        let tuned = PipelineConfig::builder()
            .content_stream_weight(2.5)
            .rrf_k(1)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.content_stream_weight, 2.5);
        assert_eq!(tuned.rrf_k, 1);
    }

    /// Reranker latency lever 1 — `0` (unlimited) is the default, byte-
    /// identical to pre-lever behaviour, and the builder setter reaches the
    /// live `SearchConfig`.
    #[test]
    fn rerank_candidate_max_chars_defaults_to_zero_unlimited() {
        assert_eq!(
            SearchConfig::default().rerank_candidate_max_chars,
            0,
            "default must stay 0 (unlimited, byte-identical pre-lever)"
        );
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.rerank_candidate_max_chars, 0);
        let tuned = PipelineConfig::builder()
            .rerank_candidate_max_chars(512)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.rerank_candidate_max_chars, 512);
    }

    /// Proximity ships OFF by default (byte-
    /// identical): `proximity_weight <= 0.0` skips the second graph query
    /// entirely (see `core::context::Engine::contextualize`).
    #[test]
    fn proximity_weight_defaults_to_zero_off() {
        assert_eq!(
            SearchConfig::default().proximity_weight,
            0.0,
            "default proximity_weight must stay 0.0 (axis OFF, byte-identical to earlier)"
        );
        assert_eq!(
            SearchConfig::default().proximity_hop_bound,
            2,
            "default proximity_hop_bound must stay the spec's starting value (2)"
        );
        assert_eq!(
            SearchConfig::default().proximity_fan_out_cap,
            8,
            "default proximity_fan_out_cap must match expansion_fan_out_cap's default (8)"
        );
        // The builder path (used by open_graph before env overrides) must agree.
        let built = PipelineConfig::builder().build().unwrap().search;
        assert_eq!(built.proximity_weight, 0.0);
        // The new builder setter threads the value.
        let tuned = PipelineConfig::builder()
            .proximity_weight(0.2)
            .build()
            .unwrap()
            .search;
        assert_eq!(tuned.proximity_weight, 0.2);
    }

    #[test]
    fn test_invalid_jaccard_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(2.0).build();
        assert!(result.is_err(), "jaccard_threshold > 1.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_jaccard_zero_rejected() {
        let result = PipelineConfig::builder().jaccard_threshold(0.0).build();
        assert!(result.is_err(), "jaccard_threshold = 0.0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("jaccard_threshold"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_weights_accept_any_positive_values() {
        let result = PipelineConfig::builder()
            .bm25_weight(0.7)
            .vector_weight(0.3)
            .build();
        assert!(
            result.is_ok(),
            "RRF weights are independent multipliers; sum-to-1.0 no longer required"
        );

        let result2 = PipelineConfig::builder()
            .bm25_weight(2.0)
            .vector_weight(1.0)
            .build();
        assert!(result2.is_ok(), "any positive values accepted");
    }

    #[test]
    fn test_invalid_embedding_dim_rejected() {
        let result = PipelineConfig::builder().embedding_dim(0).build();
        assert!(result.is_err(), "embedding_dim = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("embedding_dim"),
            "error should mention field name"
        );
    }

    #[test]
    fn test_invalid_min_tokens_rejected() {
        let result = PipelineConfig::builder().min_words(0).build();
        assert!(result.is_err(), "min_words = 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("min_words"), "error should mention field name");
    }

    #[test]
    fn test_max_less_than_min_rejected() {
        let result = PipelineConfig::builder()
            .min_words(800)
            .max_words(400)
            .build();
        assert!(result.is_err(), "max_words < min_words must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("max_words") || msg.contains("min_words"),
            "error should mention word-window fields"
        );
    }

    #[test]
    fn test_custom_config_builds() {
        let cfg = PipelineConfig::builder()
            .embedding_dim(768)
            .min_words(200)
            .max_words(600)
            .density_threshold(0.2)
            .num_permutations(64)
            .shingle_size(4)
            .band_size(8)
            .jaccard_threshold(0.85)
            .min_name_length(4)
            .min_token_count(1)
            .entropy_threshold(1.0)
            .bm25_weight(0.7)
            .vector_weight(0.3)
            .rrf_k(30)
            .top_k(5)
            .cache_ttl(Duration::from_secs(60))
            .cache_max_entries(500)
            .build()
            .expect("custom config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(768));
        assert_eq!(cfg.extraction_window.min_words, 200);
        assert_eq!(cfg.extraction_window.max_words, 600);
        assert!((cfg.extraction_window.density_threshold - 0.2).abs() < 1e-12);
        assert_eq!(cfg.minhash.num_permutations, 64);
        assert_eq!(cfg.minhash.shingle_size, 4);
        assert_eq!(cfg.minhash.band_size, 8);
        assert!((cfg.minhash.jaccard_threshold - 0.85).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 4);
        assert_eq!(cfg.entropy.min_token_count, 1);
        assert!((cfg.entropy.entropy_threshold - 1.0).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.7).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.3).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 30);
        assert_eq!(cfg.search.top_k, 5);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(60));
        assert_eq!(cfg.cache_max_entries, 500);
    }

    #[test]
    fn test_default_values_correct() {
        let cfg = PipelineConfig::builder()
            .build()
            .expect("default config should build");

        assert_eq!(cfg.embedding_dim, EmbeddingDim(384));
        assert_eq!(cfg.extraction_window.min_words, 100);
        assert!((cfg.extraction_window.density_threshold - 0.15).abs() < 1e-12);
        assert_eq!(cfg.extraction_window.max_words, 300);
        assert_eq!(cfg.minhash.num_permutations, 32);
        assert_eq!(cfg.minhash.shingle_size, 3);
        assert_eq!(cfg.minhash.band_size, 4);
        assert!((cfg.minhash.jaccard_threshold - 0.9).abs() < 1e-12);
        assert_eq!(cfg.entropy.min_name_length, 6);
        assert_eq!(cfg.entropy.min_token_count, 2);
        assert!((cfg.entropy.entropy_threshold - 1.5).abs() < 1e-12);
        assert!((cfg.search.bm25_weight - 0.5).abs() < 1e-12);
        assert!((cfg.search.vector_weight - 0.5).abs() < 1e-12);
        assert_eq!(cfg.search.rrf_k, 1);
        assert_eq!(cfg.search.top_k, 10);
        assert_eq!(cfg.cache_ttl, Duration::from_secs(300));
        assert_eq!(cfg.cache_max_entries, 1000);
    }

    #[test]
    fn test_ontology_config() {
        let allowed = vec!["Person".to_string(), "Organization".to_string()];
        let excluded = vec!["StopWord".to_string()];
        let edges = vec!["WORKS_AT".to_string()];

        let cfg = PipelineConfig::builder()
            .allowed_entity_types(allowed.clone())
            .excluded_entity_types(excluded.clone())
            .allowed_edge_types(edges.clone())
            .build()
            .expect("ontology config should build");

        assert_eq!(cfg.allowed_entity_types, allowed);
        assert_eq!(cfg.excluded_entity_types, excluded);
        assert_eq!(cfg.allowed_edge_types, edges);

        // Verify that allowed and excluded are independent — a type can
        // appear in excluded without being in allowed.
        assert!(!cfg.allowed_entity_types.contains(&"StopWord".to_string()));
        assert!(cfg.excluded_entity_types.contains(&"StopWord".to_string()));
    }

    #[test]
    fn test_extraction_arm_budget_ms_default() {
        let cfg = PipelineConfig::builder()
            .build()
            .expect("default config should build");
        assert_eq!(
            cfg.extraction_arm_budget_ms, 30_000,
            "default extraction_arm_budget_ms must be 30_000 (production fail-fast)"
        );
    }

    #[test]
    fn test_extraction_arm_budget_ms_custom() {
        let cfg = PipelineConfig::builder()
            .extraction_arm_budget_ms(180_000)
            .build()
            .expect("custom extraction_arm_budget_ms should build");
        assert_eq!(
            cfg.extraction_arm_budget_ms, 180_000,
            "extraction_arm_budget_ms builder method must propagate the custom value"
        );
    }

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::redundant_closure)]
/// Semantic retrieval quality benchmark for the rql-core crate.
///
/// Validates retrieval quality using REAL sentence embeddings from
/// all-MiniLM-L6-v2 (384-dim, L2-normalised) via ONNX Runtime.
///
/// ALL tests are `#[ignore]` — they require the model to be downloaded
/// from HuggingFace Hub on first run (~90 MB, cached in ~/.cache/huggingface).
///
/// Run with:
///   cargo test --test retrieval_semantic --features embeddings -- --nocapture --ignored
#[cfg(feature = "embeddings")]
mod semantic_tests {
    use std::sync::Arc;

    use chrono::Utc;
    use serde::Deserialize;

    use kremory::core::config::PipelineConfig;
    use kremory::core::context::{ContextResult, ContextualizeParams};
    use kremory::core::graph::{FactInsert, InsertEntityParams};
    use kremory::core::ingest::Engine;
    use kremory::core::provider::{EmbeddingProvider, MockChatProvider, OnnxEmbeddingProvider};
    use kremory::core::schema::{Entity, TemporalGraph};
    use kremory::core::search::{
        HybridSearchEntitiesParams, SearchFilters, SearchHit, VectorSearchEntitiesParams,
    };

    // ─── Ground truth types ───────────────────────────────────────────────────

    #[derive(Debug, Deserialize)]
    struct GroundTruthEntity {
        name: String,
        #[allow(dead_code)]
        label: String,
    }

    #[derive(Debug, Deserialize)]
    struct DomainGroundTruth {
        entities: Vec<GroundTruthEntity>,
        #[allow(dead_code)]
        min_relationships: usize,
    }

    fn load_ground_truth() -> std::collections::HashMap<String, DomainGroundTruth> {
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let raw = std::fs::read_to_string(gt_path)
            .unwrap_or_else(|e| panic!("failed to read ground_truth.json: {e}"));
        serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse ground_truth.json: {e}"))
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    /// Convert a display name like "Amazon Robotics" to a snake_case entity id.
    /// Lowercases, replaces non-alphanumeric runs with '_', trims leading/trailing underscores.
    fn entity_id(name: &str) -> String {
        let mut result = String::with_capacity(name.len());
        let mut prev_underscore = true;
        for ch in name.chars() {
            if ch.is_alphanumeric() {
                result.push(ch.to_lowercase().next().unwrap_or(ch));
                prev_underscore = false;
            } else if !prev_underscore {
                result.push('_');
                prev_underscore = true;
            }
        }
        result.trim_end_matches('_').to_owned()
    }

    /// Dot product of two L2-normalised vectors — equals cosine similarity.
    #[allow(dead_code)]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Return true when an entity with the expected id (or whose id/properties
    /// contain the expected name) appears anywhere in the hit list.
    fn entity_in_hits(hits: &[SearchHit<Entity>], expected_name: &str) -> bool {
        let expected_id = entity_id(expected_name);
        let needle = expected_name.to_lowercase();
        for hit in hits {
            let id_lc = hit.item.id.to_lowercase();
            let props = hit.item.properties.to_string().to_lowercase();
            if id_lc == expected_id
                || id_lc.contains(&needle)
                || needle.contains(&id_lc)
                || props.contains(&needle)
            {
                return true;
            }
        }
        false
    }

    // ─── Test 1: self-retrieval with real embeddings ──────────────────────────

    /// Every entity should be the top-1 result when queried with its own embedding.
    /// With a real sentence encoder this should be perfect (100%).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_semantic_self_retrieval() {
        let embedder = tokio::task::block_in_place(|| OnnxEmbeddingProvider::new())
            .expect("OnnxEmbeddingProvider::new() failed — check HuggingFace connectivity");

        let ground_truth = load_ground_truth();
        let domain = ground_truth
            .get("mock_interview")
            .expect("mock_interview not in ground truth");

        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("failed to open in-memory graph");

        // Insert entities with real embeddings
        for entity in &domain.entities {
            let id = entity_id(&entity.name);
            graph
                .insert_entity(InsertEntityParams {
                    id: &id,
                    entity_type_id: 0,
                    properties: serde_json::json!({"name": entity.name}),
                })
                .await
                .unwrap_or_else(|e| panic!("insert_entity({id}) failed: {e}"));

            let emb = embedder
                .embed(&entity.name)
                .await
                .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

            graph
                .set_entity_embedding(&id, &emb)
                .await
                .unwrap_or_else(|e| panic!("set_entity_embedding({id}) failed: {e}"));
        }

        eprintln!("\n{:═<65}", "");
        eprintln!("  SELF-RETRIEVAL (all-MiniLM-L6-v2, mock_interview)");
        eprintln!("{:═<65}", "");
        eprintln!("  {:<30} | {:>6} | {:>10}", "Entity", "Top-1?", "Cos-Sim");
        eprintln!("  {:-<30}-+-{:-<6}-+-{:-<10}", "", "", "");

        let mut all_self_retrieved = true;

        for entity in &domain.entities {
            let expected_id = entity_id(&entity.name);
            let query_emb = embedder
                .embed(&entity.name)
                .await
                .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

            let hits = graph
                .vector_search_entities(VectorSearchEntitiesParams {
                    query_embedding: &query_emb,
                    limit: 1,
                    filters: &SearchFilters::new(),
                })
                .await
                .unwrap_or_else(|e| panic!("vector_search_entities({}) failed: {e}", entity.name));

            let top1_id = hits.first().map(|h| h.item.id.as_str()).unwrap_or("<none>");
            let is_top1 = top1_id == expected_id;

            // Score = cosine similarity against the stored embedding.
            // Since the vector is re-computed from the same text, it should be ~1.0.
            // We use the search score (which is -cosine_distance, so 0 = identical).
            let score_str = hits
                .first()
                .map(|h| format!("{:.4}", 1.0 + h.score)) // distance→similarity: 1 - dist
                .unwrap_or_else(|| "N/A".to_owned());

            eprintln!(
                "  {:<30} | {:>6} | {:>10}",
                entity.name,
                if is_top1 { "YES" } else { "NO" },
                score_str,
            );

            if !is_top1 {
                all_self_retrieved = false;
            }
        }

        eprintln!("{:═<65}\n", "");

        assert!(
            all_self_retrieved,
            "self-retrieval should be perfect with real embeddings: every entity must be its own top-1"
        );
    }

    // ─── Test 2: semantic similarity — related-but-not-identical queries ──────

    /// Proves semantic search works: queries are paraphrases of entity names,
    /// not verbatim copies. The expected entity must appear in top-3 results.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_semantic_similarity_search() {
        let embedder = tokio::task::block_in_place(|| OnnxEmbeddingProvider::new())
            .expect("OnnxEmbeddingProvider::new() failed");

        let ground_truth = load_ground_truth();
        let domain = ground_truth
            .get("mock_interview")
            .expect("mock_interview not in ground truth");

        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("failed to open in-memory graph");

        for entity in &domain.entities {
            let id = entity_id(&entity.name);
            graph
                .insert_entity(InsertEntityParams {
                    id: &id,
                    entity_type_id: 0,
                    properties: serde_json::json!({"name": entity.name}),
                })
                .await
                .unwrap_or_else(|e| panic!("insert_entity({id}) failed: {e}"));

            let emb = embedder
                .embed(&entity.name)
                .await
                .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

            graph
                .set_entity_embedding(&id, &emb)
                .await
                .unwrap_or_else(|e| panic!("set_entity_embedding({id}) failed: {e}"));
        }

        // Semantic query → expected entity name
        let queries: &[(&str, &str)] = &[
            ("robotics company", "Amazon Robotics"),
            ("university education", "Northeastern University"),
            ("person named Ria", "Ria"),
            ("north african country", "Morocco"),
            ("east asian country", "South Korea"),
        ];

        eprintln!("\n{:═<70}", "");
        eprintln!("  SEMANTIC SIMILARITY SEARCH (all-MiniLM-L6-v2)");
        eprintln!("{:═<70}", "");
        eprintln!(
            "  {:<30} | {:<22} | {:>7} | {:>10}",
            "Query", "Expected", "Found?", "Top-3 IDs"
        );
        eprintln!("  {:-<30}-+-{:-<22}-+-{:-<7}-+-{:-<10}", "", "", "", "");

        let mut hits_count = 0usize;

        for (query, expected_name) in queries {
            let query_emb = embedder
                .embed(query)
                .await
                .unwrap_or_else(|e| panic!("embed({query}) failed: {e}"));

            let hits = graph
                .vector_search_entities(VectorSearchEntitiesParams {
                    query_embedding: &query_emb,
                    limit: 3,
                    filters: &SearchFilters::new(),
                })
                .await
                .unwrap_or_else(|e| panic!("vector_search_entities({query}) failed: {e}"));

            let found = entity_in_hits(&hits, expected_name);
            if found {
                hits_count += 1;
            }

            let top3: Vec<&str> = hits.iter().map(|h| h.item.id.as_str()).collect();
            eprintln!(
                "  {:<30} | {:<22} | {:>7} | {}",
                query,
                expected_name,
                if found { "YES" } else { "NO" },
                top3.join(", "),
            );
        }

        eprintln!("{:═<70}", "");
        eprintln!("  Found {}/{} semantic queries", hits_count, queries.len());
        eprintln!("{:═<70}\n", "");

        assert!(
            hits_count >= 3,
            "at least 3 out of 5 semantic queries must find their target in top-3 \
             (got {}/{})",
            hits_count,
            queries.len(),
        );
    }

    // ─── Test 3: hybrid vs pure vector ───────────────────────────────────────

    /// Demonstrates that hybrid search (vector + FTS RRF) adds value over pure
    /// vector search alone when tech term names match exactly in both modalities.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_hybrid_beats_pure_vector() {
        let embedder = tokio::task::block_in_place(|| OnnxEmbeddingProvider::new())
            .expect("OnnxEmbeddingProvider::new() failed");

        let ground_truth = load_ground_truth();
        let domain = ground_truth
            .get("tech_standup")
            .expect("tech_standup not in ground truth");

        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("failed to open in-memory graph");

        for entity in &domain.entities {
            let id = entity_id(&entity.name);
            graph
                .insert_entity(InsertEntityParams {
                    id: &id,
                    entity_type_id: 0,
                    properties: serde_json::json!({"name": entity.name}),
                })
                .await
                .unwrap_or_else(|e| panic!("insert_entity({id}) failed: {e}"));

            let emb = embedder
                .embed(&entity.name)
                .await
                .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

            graph
                .set_entity_embedding(&id, &emb)
                .await
                .unwrap_or_else(|e| panic!("set_entity_embedding({id}) failed: {e}"));
        }

        // Pairs of (semantic query, text for FTS, expected entity name)
        let cases: &[(&str, &str, &str)] = &[
            ("database system", "database", "PostgreSQL"),
            ("container orchestration", "container", "Kubernetes"),
        ];

        eprintln!("\n{:═<75}", "");
        eprintln!("  HYBRID vs PURE VECTOR (all-MiniLM-L6-v2, tech_standup)");
        eprintln!("{:═<75}", "");
        eprintln!(
            "  {:<26} | {:<12} | {:>12} | {:>14}",
            "Query (semantic)", "Expected", "Vec Found?", "Hybrid Found?"
        );
        eprintln!("  {:-<26}-+-{:-<12}-+-{:-<12}-+-{:-<14}", "", "", "", "");

        let mut vec_hits = 0usize;
        let mut hybrid_hits = 0usize;

        for (semantic_query, fts_text, expected_name) in cases {
            let query_emb = embedder
                .embed(semantic_query)
                .await
                .unwrap_or_else(|e| panic!("embed({semantic_query}) failed: {e}"));

            let pure_vec = graph
                .vector_search_entities(VectorSearchEntitiesParams {
                    query_embedding: &query_emb,
                    limit: 5,
                    filters: &SearchFilters::new(),
                })
                .await
                .unwrap_or_else(|e| panic!("vector_search_entities failed: {e}"));

            let hybrid = graph
                .hybrid_search_entities(HybridSearchEntitiesParams {
                    query_text: fts_text,
                    query_embedding: &query_emb,
                    limit: 5,
                    filters: &SearchFilters::new(),
                })
                .await
                .unwrap_or_else(|e| panic!("hybrid_search_entities failed: {e}"));

            let vec_found = entity_in_hits(&pure_vec, expected_name);
            let hyb_found = entity_in_hits(&hybrid, expected_name);

            if vec_found {
                vec_hits += 1;
            }
            if hyb_found {
                hybrid_hits += 1;
            }

            eprintln!(
                "  {:<26} | {:<12} | {:>12} | {:>14}",
                semantic_query,
                expected_name,
                if vec_found { "YES" } else { "NO" },
                if hyb_found { "YES" } else { "NO" },
            );
        }

        eprintln!("{:═<75}", "");
        eprintln!(
            "  Pure vector: {}/{} found   Hybrid: {}/{} found",
            vec_hits,
            cases.len(),
            hybrid_hits,
            cases.len(),
        );
        eprintln!("{:═<75}\n", "");

        // Hybrid must find at least as many as pure vector
        assert!(
            hybrid_hits >= vec_hits,
            "hybrid should find at least as many entities as pure vector \
             (hybrid={}, vector={})",
            hybrid_hits,
            vec_hits,
        );

        // At least one of the two test cases must succeed for hybrid
        assert!(
            hybrid_hits >= 1,
            "hybrid search should find at least 1 of the 2 test entities (found {})",
            hybrid_hits,
        );
    }

    // ─── Test 4: contextualize with real embedder ─────────────────────────────

    /// Verifies the full contextualize() path works with OnnxEmbeddingProvider.
    /// contextualize() uses FTS internally; this test confirms the pipeline
    /// handles a real embedding provider throughout ingest + retrieval.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_semantic_contextualize() {
        let embedder = Arc::new(
            tokio::task::block_in_place(|| OnnxEmbeddingProvider::new())
                .expect("OnnxEmbeddingProvider::new() failed"),
        );

        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open in-memory graph"),
        );

        let now = Utc::now();

        // Insert mock_interview entities with real embeddings
        let entities: &[(&str, &str, &str)] = &[
            ("ria", "Ria", "Person"),
            ("amazon_robotics", "Amazon Robotics", "Organisation"),
            (
                "northeastern_university",
                "Northeastern University",
                "Organisation",
            ),
            ("morocco", "Morocco", "Location"),
            ("south_korea", "South Korea", "Location"),
        ];

        for (id, name, _label) in entities {
            graph
                .insert_entity(InsertEntityParams {
                    id,
                    entity_type_id: 0,
                    properties: serde_json::json!({"name": name}),
                })
                .await
                .unwrap_or_else(|e| panic!("insert_entity({id}) failed: {e}"));

            let emb = embedder
                .embed(name)
                .await
                .unwrap_or_else(|e| panic!("embed({name}) failed: {e}"));

            graph
                .set_entity_embedding(id, &emb)
                .await
                .unwrap_or_else(|e| panic!("set_entity_embedding({id}) failed: {e}"));
        }

        // Insert facts connecting Ria to her employer and university
        graph
            .insert_fact(FactInsert::new("ria", "works_at", now).object_id("amazon_robotics"))
            .await
            .expect("insert works_at fact");

        graph
            .insert_fact(
                FactInsert::new("ria", "studied_at", now).object_id("northeastern_university"),
            )
            .await
            .expect("insert studied_at fact");

        let config = PipelineConfig::builder()
            .build()
            .expect("PipelineConfig::build");
        let rql: Engine<MockChatProvider, OnnxEmbeddingProvider> =
            Engine::new(kremory::core::ingest::EngineNewParams {
                graph,
                llm: Arc::new(MockChatProvider::null()),
                embedder,
                config,
                model: None,
            });

        // contextualize() uses FTS — "Ria" in the label should match.
        let ctx: ContextResult = rql
            .contextualize(ContextualizeParams {
                query: "Ria",
                group_id: None,
                limit: None,
                as_of: None,
            })
            .await
            .expect("contextualize failed");

        eprintln!("\n{:═<65}", "");
        eprintln!("  CONTEXTUALIZE (all-MiniLM-L6-v2)");
        eprintln!("{:═<65}", "");
        eprintln!(
            "  entities returned: {:?}",
            ctx.entities
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>()
        );
        eprintln!("  facts returned:    {}", ctx.facts.len());
        eprintln!("{:═<65}\n", "");

        let entity_ids: Vec<&str> = ctx.entities.iter().map(|e| e.id.as_str()).collect();

        assert!(
            entity_ids.contains(&"ria"),
            "contextualize('Ria') must include 'ria' in results; got: {:?}",
            entity_ids,
        );

        // 1-hop expansion from ria should include at least one neighbour
        assert!(
            entity_ids.contains(&"amazon_robotics")
                || entity_ids.contains(&"northeastern_university"),
            "1-hop expansion should include at least one neighbour of ria; got: {:?}",
            entity_ids,
        );

        assert!(
            !ctx.facts.is_empty(),
            "contextualize should return connecting facts; got 0 facts",
        );
    }

    // ─── Test 5: full benchmark across all 14 domains ─────────────────────────

    /// Loads all 14 domains, inserts entities with real embeddings, and measures
    /// vector recall@1 and hybrid recall@1 for self-retrieval.
    ///
    /// With a real 384-dim encoder, self-retrieval should be near-perfect.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_real_embedding_benchmark_summary() {
        // Load the embedding model ONCE — model loading takes ~2 s.
        let embedder = tokio::task::block_in_place(|| OnnxEmbeddingProvider::new())
            .expect("OnnxEmbeddingProvider::new() failed");

        let ground_truth = load_ground_truth();
        let mut domain_keys: Vec<String> = ground_truth.keys().cloned().collect();
        domain_keys.sort();

        struct DomainResult {
            key: String,
            entity_count: usize,
            vec_recall1: usize,
            hybrid_recall1: usize,
        }

        let mut results: Vec<DomainResult> = Vec::new();

        for domain_key in &domain_keys {
            let domain = ground_truth
                .get(domain_key)
                .expect("domain in ground truth");

            let graph = TemporalGraph::open_in_memory()
                .await
                .expect("failed to open in-memory graph");

            // Insert all entities with real embeddings
            for entity in &domain.entities {
                let id = entity_id(&entity.name);
                graph
                    .insert_entity(InsertEntityParams {
                        id: &id,
                        entity_type_id: 0,
                        properties: serde_json::json!({"name": entity.name}),
                    })
                    .await
                    .unwrap_or_else(|e| panic!("insert_entity({id}) failed: {e}"));

                let emb = embedder
                    .embed(&entity.name)
                    .await
                    .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

                graph
                    .set_entity_embedding(&id, &emb)
                    .await
                    .unwrap_or_else(|e| panic!("set_entity_embedding({id}) failed: {e}"));
            }

            let mut vec_recall1 = 0usize;
            let mut hybrid_recall1 = 0usize;

            for entity in &domain.entities {
                let expected_id = entity_id(&entity.name);

                let emb = embedder
                    .embed(&entity.name)
                    .await
                    .unwrap_or_else(|e| panic!("embed({}) failed: {e}", entity.name));

                // Vector recall@1
                let vec_hits = graph
                    .vector_search_entities(VectorSearchEntitiesParams {
                        query_embedding: &emb,
                        limit: 1,
                        filters: &SearchFilters::new(),
                    })
                    .await
                    .unwrap_or_else(|e| {
                        panic!("vector_search_entities({}) failed: {e}", entity.name)
                    });
                if vec_hits.first().map(|h| h.item.id.as_str()) == Some(&expected_id) {
                    vec_recall1 += 1;
                }

                // Hybrid recall@1 — use the entity name for both FTS and vector
                // Wrap in quotes for FTS5 safety with special characters
                let fts_query = format!("\"{}\"", entity.name.replace('"', "\"\""));
                let hybrid_hits = graph
                    .hybrid_search_entities(HybridSearchEntitiesParams {
                        query_text: &fts_query,
                        query_embedding: &emb,
                        limit: 1,
                        filters: &SearchFilters::new(),
                    })
                    .await
                    .unwrap_or_else(|e| {
                        panic!("hybrid_search_entities({}) failed: {e}", entity.name)
                    });
                if hybrid_hits.first().map(|h| h.item.id.as_str()) == Some(&expected_id) {
                    hybrid_recall1 += 1;
                }
            }

            results.push(DomainResult {
                key: domain_key.clone(),
                entity_count: domain.entities.len(),
                vec_recall1,
                hybrid_recall1,
            });
        }

        // ── Print formatted table ────────────────────────────────────────────

        eprintln!("\n{:═<65}", "");
        eprintln!("  SEMANTIC RETRIEVAL BENCHMARK (all-MiniLM-L6-v2, 384-dim)");
        eprintln!("{:═<65}", "");
        eprintln!(
            "  {:<24} | {:>8} | {:>15} | {:>15}",
            "Domain", "Entities", "Vector Recall@1", "Hybrid Recall@1"
        );
        eprintln!("  {:-<24}-+-{:-<8}-+-{:-<15}-+-{:-<15}", "", "", "", "");

        let mut total_entities = 0usize;
        let mut total_vec = 0usize;
        let mut total_hybrid = 0usize;

        for r in &results {
            let n = r.entity_count;
            let vec_pct = if n == 0 {
                100.0_f64
            } else {
                r.vec_recall1 as f64 / n as f64 * 100.0
            };
            let hyb_pct = if n == 0 {
                100.0_f64
            } else {
                r.hybrid_recall1 as f64 / n as f64 * 100.0
            };

            eprintln!(
                "  {:<24} | {:>8} | {:>14.1}% | {:>14.1}%",
                r.key, n, vec_pct, hyb_pct,
            );

            total_entities += n;
            total_vec += r.vec_recall1;
            total_hybrid += r.hybrid_recall1;
        }

        let overall_vec = if total_entities == 0 {
            1.0_f64
        } else {
            total_vec as f64 / total_entities as f64
        };
        let overall_hyb = if total_entities == 0 {
            1.0_f64
        } else {
            total_hybrid as f64 / total_entities as f64
        };

        eprintln!("  {:-<24}-+-{:-<8}-+-{:-<15}-+-{:-<15}", "", "", "", "");
        eprintln!(
            "  {:<24} | {:>8} | {:>14.1}% | {:>14.1}%",
            "OVERALL",
            total_entities,
            overall_vec * 100.0,
            overall_hyb * 100.0,
        );
        eprintln!("{:═<65}\n", "");

        // ── Assertions ────────────────────────────────────────────────────────

        assert!(
            overall_vec >= 0.85,
            "overall vector recall@1 {:.1}% is below 85% threshold ({}/{} found)",
            overall_vec * 100.0,
            total_vec,
            total_entities,
        );

        assert!(
            overall_hyb >= 0.90,
            "overall hybrid recall@1 {:.1}% is below 90% threshold ({}/{} found)",
            overall_hyb * 100.0,
            total_hybrid,
            total_entities,
        );
    }
}

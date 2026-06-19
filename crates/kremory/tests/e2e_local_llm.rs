#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// End-to-end integration test: real GGUF model → entity/fact extraction via ingest().
///
/// Requires the environment variable `RQL_MODEL_PATH` to point to a GGUF file.
/// If the variable is not set the test is skipped automatically.
///
/// Run with:
///   RQL_MODEL_PATH=../models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
///     cargo test --features llm --test e2e_local_llm -- --nocapture
mod common;

#[cfg(feature = "llm")]
mod llm_tests {
    use std::io::Write as _;
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use kremory::core::config::PipelineConfig;
    use kremory::core::extraction::NuExtractExtractor;
    use kremory::core::ingest::Engine;
    use kremory::core::provider::{ChatProvider as _, NullEmbeddingProvider};
    use kremory::core::schema::TemporalGraph;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    const MEETING_TRANSCRIPT: &str = "\
Alice: Good morning everyone. Let's get started. We need to review the \
Project Phoenix timeline. Sarah, where are we on the backend integration?

Sarah: Hi Alice. The backend API is about 80% complete. I expect to finish \
by March 15th. However, we need Bob's sign-off on the database schema changes \
before we can merge.

Bob: I reviewed the schema yesterday. There are two concerns. The budget for \
the cloud infrastructure has gone over by about $12,000 — we're now at $62,000 \
against a $50,000 allocation. We need approval from the CFO, Jennifer Lee, to \
proceed.

Alice: Understood. I'll reach out to Jennifer today. Bob, can you document the \
schema changes and send them to the engineering team by end of day?

Bob: Sure, I'll have that done by 5 PM. Also, the mobile team lead, David Chen, \
mentioned that the iOS release is blocked on the API changes Sarah is working on.

Sarah: Yes, David and I spoke earlier. Once the API is merged, his team needs \
about one week to integrate and test. So the mobile release would be around \
March 22nd at the earliest.

Alice: Let's set that as our target. I'll update the project board and notify \
the stakeholders at Acme Corporation. Final question — who owns the QA sign-off?

Bob: That would be Lisa Park from the QA team. She said she needs at least \
three business days after the API merge to run the full regression suite.";

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_e2e_ingest_with_local_llm() {
        let model_path = match std::env::var("RQL_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                println!("SKIP: RQL_MODEL_PATH not set — skipping e2e_local_llm test");
                return;
            }
        };

        // Set up metrics capture.  The local-recorder guard is thread-local so
        // we use current_thread runtime above to ensure all async poll frames
        // (including ingest) run on this thread and see the recorder.
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _recorder_guard = metrics::set_default_local_recorder(&recorder);

        let start = Instant::now();

        // Build LlamaCppProvider — async constructor, no spawn_blocking needed.
        let llm = build_llm(&model_path, 4096, 1024)
            .await
            .expect("failed to build LlamaCppProvider");

        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(2000)
            .build()
            .expect("PipelineConfig::build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };

        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open in-memory TemporalGraph"),
        );

        let rql = Engine::new(kremory::core::ingest::EngineNewParams {
            graph: graph,
            llm: Arc::new(llm),
            embedder: Arc::new(embedder),
            config: config,
        });

        let result = rql
            .ingest(kremory::core::ingest::IngestParams {
                text: MEETING_TRANSCRIPT,
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: kremory::core::ingest::SourceParams::default(),
            })
            .await
            .expect("ingest() returned an error");

        let elapsed = start.elapsed();

        // Take the snapshot now that ingest has completed.
        let snapshot = snapshotter.snapshot().into_vec();

        // ── Assertions ────────────────────────────────────────────────────────

        assert!(
            result.episode_id > 0,
            "episode_id should be positive, got {}",
            result.episode_id
        );

        assert!(
            result.upserted_entities.len() >= 5,
            "expected at least 5 entities from meeting transcript (got {}). \
             Phi-4-mini typically extracts 7+.",
            result.upserted_entities.len()
        );

        assert!(
            result.inserted_fact_ids.len() >= 3,
            "expected at least 3 facts from meeting transcript (got {}). \
             Phi-4-mini typically extracts 8+.",
            result.inserted_fact_ids.len()
        );

        // ── Metrics assertions ────────────────────────────────────────────────

        // Verify extraction stages were timed.
        let stage_timings: Vec<f64> = snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == "rql.extraction.stage_ms")
            .flat_map(|(.., v)| match v {
                DebugValue::Histogram(vals) => {
                    vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>()
                }
                _ => vec![],
            })
            .collect();
        // Engine::ingest() uses NuExtractExtractor which emits a single
        // "nuextract" stage timing.  DefaultExtractor (3-stage) emits 3; that
        // path is exercised by model_comparison and spike_hybrid_extractor.
        assert_eq!(
            stage_timings.len(),
            1,
            "ingest() uses NuExtractExtractor — should have exactly 1 stage timing (\"nuextract\"); got {}",
            stage_timings.len()
        );

        // Verify entity count was recorded.
        let entity_counts: Vec<f64> = snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == "rql.extraction.entity_count")
            .flat_map(|(.., v)| match v {
                DebugValue::Histogram(vals) => {
                    vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>()
                }
                _ => vec![],
            })
            .collect();
        assert!(
            !entity_counts.is_empty(),
            "entity_count metric should be emitted"
        );
        assert!(entity_counts[0] > 0.0, "should extract at least 1 entity");

        // Verify ingest total was recorded.
        let ingest_total: Vec<f64> = snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == "rql.ingest.total_ms")
            .flat_map(|(.., v)| match v {
                DebugValue::Histogram(vals) => {
                    vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>()
                }
                _ => vec![],
            })
            .collect();
        assert!(
            !ingest_total.is_empty(),
            "ingest total_ms should be recorded"
        );
        assert!(ingest_total[0] > 0.0, "ingest should take some time");

        // ── Print summary to stdout ───────────────────────────────────────────

        println!("\n=== E2E LLM Test Results ===");
        println!("Model: {model_path}");
        println!("Elapsed: {:.1}s", elapsed.as_secs_f64());
        println!("Episode ID: {}", result.episode_id);
        println!("Upserted entities ({}):", result.upserted_entities.len());
        for e in &result.upserted_entities {
            println!("  - {e}");
        }
        println!("Inserted fact IDs ({}):", result.inserted_fact_ids.len());
        for fid in &result.inserted_fact_ids {
            println!("  - {fid}");
        }
        println!("Merged entities ({}):", result.merged_entities.len());
        for (canonical, alias) in &result.merged_entities {
            println!("  - {canonical} ← {alias}");
        }
        println!("Invalidated fact IDs: {:?}", result.invalidated_fact_ids);

        println!("\n--- Metrics Snapshot ---");
        for (key, _unit, _desc, value) in &snapshot {
            let name = key.key().name();
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|l| format!("{}={}", l.key(), l.value()))
                .collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", labels.join(","))
            };
            match value {
                DebugValue::Counter(n) => println!("  {name}{label_str} = {n} (counter)"),
                DebugValue::Gauge(g) => println!("  {name}{label_str} = {g} (gauge)"),
                DebugValue::Histogram(vals) => {
                    let vals_f: Vec<f64> = vals.iter().map(|v| v.into_inner()).collect();
                    println!("  {name}{label_str} = {vals_f:?} (histogram)");
                }
            }
        }

        // ── Write log file ────────────────────────────────────────────────────

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let log_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
        std::fs::create_dir_all(&log_dir).expect("failed to create logs directory");

        let log_path = log_dir.join(format!("e2e-{timestamp}.log"));
        let mut log_file = std::fs::File::create(&log_path).expect("failed to create log file");

        writeln!(log_file, "=== E2E LLM Test Log ===").ok();
        writeln!(log_file, "Timestamp (unix): {timestamp}").ok();
        writeln!(log_file, "Model: {model_path}").ok();
        writeln!(log_file, "Elapsed: {:.1}s", elapsed.as_secs_f64()).ok();
        writeln!(log_file).ok();
        writeln!(log_file, "--- Input Text ---").ok();
        writeln!(log_file, "{MEETING_TRANSCRIPT}").ok();
        writeln!(log_file).ok();
        writeln!(
            log_file,
            "--- Extracted Entities ({}) ---",
            result.upserted_entities.len()
        )
        .ok();
        for e in &result.upserted_entities {
            writeln!(log_file, "  {e}").ok();
        }
        writeln!(log_file).ok();
        writeln!(
            log_file,
            "--- Inserted Fact IDs ({}) ---",
            result.inserted_fact_ids.len()
        )
        .ok();
        for fid in &result.inserted_fact_ids {
            writeln!(log_file, "  {fid}").ok();
        }
        writeln!(log_file).ok();
        writeln!(
            log_file,
            "--- Merged Entities ({}) ---",
            result.merged_entities.len()
        )
        .ok();
        for (canonical, alias) in &result.merged_entities {
            writeln!(log_file, "  {canonical} <- {alias}").ok();
        }
        writeln!(log_file).ok();
        writeln!(log_file, "--- Invalidated Fact IDs ---").ok();
        writeln!(log_file, "  {:?}", result.invalidated_fact_ids).ok();
        writeln!(log_file).ok();
        writeln!(log_file, "--- Metrics Snapshot ---").ok();
        for (key, _unit, _desc, value) in &snapshot {
            let name = key.key().name();
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|l| format!("{}={}", l.key(), l.value()))
                .collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", labels.join(","))
            };
            match value {
                DebugValue::Counter(n) => {
                    writeln!(log_file, "  {name}{label_str} = {n} (counter)").ok()
                }
                DebugValue::Gauge(g) => {
                    writeln!(log_file, "  {name}{label_str} = {g} (gauge)").ok()
                }
                DebugValue::Histogram(vals) => {
                    let vals_f: Vec<f64> = vals.iter().map(|v| v.into_inner()).collect();
                    writeln!(log_file, "  {name}{label_str} = {vals_f:?} (histogram)").ok()
                }
            };
        }

        println!("\nLog written to: {}", log_path.display());

        // Export machine-readable metrics JSON alongside the human-readable log.
        let exporter = super::common::MetricsExporter::new(log_dir);
        exporter
            .export(&snapshotter, "e2e-local-llm")
            .unwrap_or_else(|e| {
                eprintln!("Failed to export metrics: {e}");
                std::path::PathBuf::new()
            });
    }

    /// NuExtract E2E benchmark — single-pass template extraction via ingest_with().
    ///
    /// Requires RQL_NUEXTRACT_MODEL_PATH pointing to a NuExtract GGUF file.
    /// If not set, the test is skipped.
    ///
    /// Run with:
    ///   RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
    ///     cargo test --features llm --test e2e_local_llm test_e2e_nuextract -- --nocapture
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_e2e_nuextract_benchmark() {
        let model_path = match std::env::var("RQL_NUEXTRACT_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                println!("SKIP: RQL_NUEXTRACT_MODEL_PATH not set — skipping NuExtract benchmark");
                return;
            }
        };

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _recorder_guard = metrics::set_default_local_recorder(&recorder);

        let start = Instant::now();

        let llm = Arc::new(
            build_llm(&model_path, 4096, 1024)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        // Domain-specific entity types guide NuExtract's template extraction.
        // Without these, NuExtract only discovers obvious entities (people).
        // With them, it also finds organisations, projects, dates, etc.
        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(2000)
            .allowed_entity_types(vec![
                "Person".into(),
                "Organisation".into(),
                "Project".into(),
                "Location".into(),
                "Date".into(),
                "Money".into(),
            ])
            .build()
            .expect("PipelineConfig::build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };

        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open in-memory TemporalGraph"),
        );

        let rql = Engine::new(kremory::core::ingest::EngineNewParams {
            graph: graph,
            llm: llm.clone(),
            embedder: Arc::new(embedder),
            config: config,
        });

        let extractor = NuExtractExtractor::new(llm);
        let result = rql
            .ingest_with(
                &extractor,
                kremory::core::ingest::IngestWithParams {
                    text: MEETING_TRANSCRIPT,
                    reference_time: None,
                    group_id: None,
                    content_type: None,
                    source_params: kremory::core::ingest::SourceParams::default(),
                },
            )
            .await
            .expect("ingest_with(NuExtract) returned an error");

        let elapsed = start.elapsed();
        let snapshot = snapshotter.snapshot().into_vec();

        // ── Print summary ────────────────────────────────────────────────────

        println!("\n=== NuExtract E2E Benchmark ===");
        println!("Model: {model_path}");
        println!("Elapsed: {:.1}s", elapsed.as_secs_f64());
        println!("Episode ID: {}", result.episode_id);
        println!("Upserted entities ({}):", result.upserted_entities.len());
        for e in &result.upserted_entities {
            println!("  - {e}");
        }
        println!("Inserted fact IDs ({}):", result.inserted_fact_ids.len());
        for fid in &result.inserted_fact_ids {
            println!("  - {fid}");
        }
        println!("Merged entities ({}):", result.merged_entities.len());
        for (canonical, alias) in &result.merged_entities {
            println!("  - {canonical} ← {alias}");
        }
        println!("Invalidated fact IDs: {:?}", result.invalidated_fact_ids);

        println!("\n--- Metrics Snapshot ---");
        for (key, _unit, _desc, value) in &snapshot {
            let name = key.key().name();
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|l| format!("{}={}", l.key(), l.value()))
                .collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", labels.join(","))
            };
            match value {
                DebugValue::Counter(n) => println!("  {name}{label_str} = {n} (counter)"),
                DebugValue::Gauge(g) => println!("  {name}{label_str} = {g} (gauge)"),
                DebugValue::Histogram(vals) => {
                    let vals_f: Vec<f64> = vals.iter().map(|v| v.into_inner()).collect();
                    println!("  {name}{label_str} = {vals_f:?} (histogram)");
                }
            }
        }

        // ── Assertions — intentionally soft for spike benchmarking ────────────
        // NuExtract extraction quality is being evaluated, not asserted hard.

        assert!(result.episode_id > 0, "episode_id should be positive");
        println!(
            "\n--- Extraction Quality ---\nEntities: {}\nFacts inserted: {}\n(Compare with Phi-4-mini: 7 entities, 3 facts inserted)",
            result.upserted_entities.len(),
            result.inserted_fact_ids.len(),
        );

        // Export machine-readable metrics JSON for cross-run comparison.
        let log_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
        let exporter = super::common::MetricsExporter::new(log_dir);
        exporter
            .export(&snapshotter, "e2e-nuextract-benchmark")
            .unwrap_or_else(|e| {
                eprintln!("Failed to export metrics: {e}");
                std::path::PathBuf::new()
            });
    }

    /// Unconstrained entity extraction smoke test (formerly GBNF-constrained).
    ///
    /// GBNF grammar constraints are not used — they crash llama.cpp with SIGABRT
    /// (see feedback_gbnf_crashes_qwen) and are not supported by AutoAgents.
    /// AA's post-generation JSON extraction replaces grammar constraints.
    ///
    /// Kept as #[ignore] because it requires a model file and is not a CI gate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    #[ignore = "requires RQL_MODEL_PATH; not a CI gate — run manually"]
    async fn test_unconstrained_entity_extraction() {
        use kremory::core::provider::{chat_msg_system, chat_msg_user};

        let model_path = match std::env::var("RQL_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                println!("SKIP: RQL_MODEL_PATH not set");
                return;
            }
        };

        let llm = build_llm(&model_path, 4096, 512)
            .await
            .expect("failed to build LlamaCppProvider");

        let prompt = "Extract all named entities from this text.\n\nText: Alice works at Acme Corporation with Bob.\n\nOutput a JSON array of objects with \"name\" and \"label\" fields.";

        println!("=== Unconstrained Entity Extraction Test ===");
        println!("Model: {model_path}");

        let start = Instant::now();
        let msgs = vec![
            chat_msg_system("You are an entity extraction system. Output valid JSON only."),
            chat_msg_user(prompt),
        ];
        let result: Result<Box<dyn kremory::core::provider::ChatResponse>, _> =
            llm.chat_with_tools(&msgs, None, None).await;

        let elapsed = start.elapsed();

        match result {
            Ok(resp) => {
                let text = resp.text().unwrap_or_default();
                println!("Elapsed: {:.1}s", elapsed.as_secs_f64());
                println!("Output: {text}");

                let trimmed = text.trim();
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(trimmed);
                match parsed {
                    Ok(val) => {
                        println!("Valid JSON: {val}");
                    }
                    Err(e) => {
                        println!("JSON parse failed: {e}");
                        if let Some(start_idx) = trimmed.find('[') {
                            let json_part = &trimmed[start_idx..];
                            println!("Extracted JSON: {json_part}");
                        }
                    }
                }
            }
            Err(e) => {
                println!("LLM call failed after {:.1}s: {e}", elapsed.as_secs_f64());
                panic!("unconstrained entity extraction failed: {e}");
            }
        }
    }
}

// When the `llm` feature is not enabled, provide a no-op test so the file
// compiles without the feature and the test runner has something to report.
#[cfg(not(feature = "llm"))]
#[test]
fn test_e2e_local_llm_feature_not_enabled() {
    println!("SKIP: compile with --features llm to run the e2e_local_llm test");
}

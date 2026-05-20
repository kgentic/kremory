#![allow(dead_code)]
use metrics_util::debugging::{DebugValue, Snapshotter};
use serde::Serialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rql_core::intelligence::{ExtractedEntity, ExtractedFact, ExtractionResult};
use rql_core::text_utils::OovAuditor;

// PARKED 2026-05-18: D.1a cycle-2 BYOM strict gate removed
// `autoagents-llamacpp` from rqlc's dev-dependencies (ADR-Phase-D.0 §7
// — `cargo tree -p rql-core | grep autoagents-llamacpp` must print empty).
//
// The `LlamaCppProvider` / `LlamaCppReasoningFormat` re-exports + the
// async `build_llm` constructor that lived here are gated below until
// the rqlc-spikes carve-out (`.claude/PARKING_LOT.md` 2026-05-18 entry)
// moves these to a dedicated spike crate. Helpers UNUSED by the broken
// spike tests (fixtures / load_ground_truth / fuzzy_match / recall /
// OovAuditor / Snapshotter wrappers) remain available below.

#[cfg(any())]
pub use autoagents_llamacpp::{LlamaCppProvider, LlamaCppReasoningFormat};

/// Async convenience constructor for a real `LlamaCppProvider`. PARKED
/// until rqlc-spikes carve-out lands; see header note above.
#[cfg(any())]
pub async fn build_llm(
    _model_path: impl Into<PathBuf>,
    _n_ctx: u32,
    _max_tokens: u32,
) -> anyhow::Result<()> {
    unimplemented!("PARKED — see crates/rqlc/tests/common/mod.rs header")
}

#[derive(Debug, Clone, Copy)]
pub struct Fixture {
    pub name: &'static str,
    pub key: &'static str,
    pub path: &'static str,
}

pub fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "Mock Interview",
            key: "mock_interview",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
        },
        Fixture {
            name: "Medical Consult",
            key: "medical_consultation",
            path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/medical_consultation.txt"
            ),
        },
        Fixture {
            name: "Legal Deposition",
            key: "legal_deposition",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/legal_deposition.txt"),
        },
        Fixture {
            name: "Tech Standup",
            key: "tech_standup",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
        },
        Fixture {
            name: "Sales Call",
            key: "sales_call",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sales_call.txt"),
        },
        Fixture {
            name: "Podcast",
            key: "podcast_interview",
            path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/podcast_interview.txt"
            ),
        },
        Fixture {
            name: "Board Meeting",
            key: "board_meeting",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
        },
        Fixture {
            name: "News Article",
            key: "news_article",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
        },
        Fixture {
            name: "Academic Lecture",
            key: "academic_lecture",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/academic_lecture.txt"),
        },
        Fixture {
            name: "Customer Support",
            key: "customer_support",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/customer_support.txt"),
        },
        Fixture {
            name: "Slack Thread",
            key: "slack_thread",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/slack_thread.txt"),
        },
        Fixture {
            name: "Product Review",
            key: "product_review",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/product_review.txt"),
        },
        Fixture {
            name: "Short Snippet",
            key: "short_snippet",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/short_snippet.txt"),
        },
        Fixture {
            name: "Long Report",
            key: "long_report",
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/long_report.txt"),
        },
    ]
}

pub fn load_ground_truth() -> HashMap<String, Vec<String>> {
    let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
    let gt_raw = std::fs::read_to_string(gt_path).expect("ground_truth.json");
    let gt: serde_json::Value = serde_json::from_str(&gt_raw).expect("parse ground truth");
    gt.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            let entities: Vec<String> = v["entities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["name"].as_str().unwrap().to_lowercase())
                .collect();
            (k.clone(), entities)
        })
        .collect()
}

pub fn fuzzy_match(extracted: &str, expected: &str) -> bool {
    let e = extracted.to_lowercase();
    let x = expected.to_lowercase();
    e == x || e.contains(&x) || x.contains(&e)
}

pub fn recall(extracted: &[String], expected: &[String]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let found = expected
        .iter()
        .filter(|exp| extracted.iter().any(|ext| fuzzy_match(ext, exp)))
        .count();
    found as f64 / expected.len() as f64
}

pub fn load_auditor() -> OovAuditor {
    let aff = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/dictionaries/en_US.aff"
    ))
    .expect("en_US.aff");
    let dic = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/dictionaries/en_US.dic"
    ))
    .expect("en_US.dic");
    let dict = zspell::builder()
        .config_str(&aff)
        .dict_str(&dic)
        .build()
        .expect("build dictionary");
    let stops: HashSet<String> = stop_words::get(stop_words::LANGUAGE::English)
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    OovAuditor::new(dict, stops)
}

pub fn norm(s: &str) -> String {
    s.to_lowercase().trim().to_string()
}

pub fn dedup_entities(entities: &mut Vec<ExtractedEntity>) {
    entities.sort_by(|a, b| norm(&a.name).cmp(&norm(&b.name)));
    entities.dedup_by(|a, b| norm(&a.name) == norm(&b.name));
}

pub fn dedup_facts(facts: &mut Vec<ExtractedFact>) {
    facts.sort_by(|a, b| {
        (
            norm(&a.subject),
            norm(&a.predicate),
            norm(&a.object),
            a.is_entity_ref,
        )
            .cmp(&(
                norm(&b.subject),
                norm(&b.predicate),
                norm(&b.object),
                b.is_entity_ref,
            ))
    });
    facts.dedup_by(|a, b| {
        norm(&a.subject) == norm(&b.subject)
            && norm(&a.predicate) == norm(&b.predicate)
            && norm(&a.object) == norm(&b.object)
            && a.is_entity_ref == b.is_entity_ref
    });
}

pub fn merge_results(base: &ExtractionResult, additive: &ExtractionResult) -> ExtractionResult {
    let mut entities = base.entities.clone();
    entities.extend(additive.entities.clone());
    dedup_entities(&mut entities);

    let mut facts = base.facts.clone();
    facts.extend(additive.facts.clone());
    dedup_facts(&mut facts);

    ExtractionResult { entities, facts }
}

/// Reusable metrics exporter that writes DebuggingRecorder snapshots to timestamped JSON.
///
/// Usage in tests:
/// ```
/// let recorder = DebuggingRecorder::new();
/// let snapshotter = recorder.snapshotter();
/// let _guard = metrics::set_default_local_recorder(&recorder);
/// // ... run test ...
/// let exporter = MetricsExporter::new("rql-core/logs");
/// let path = exporter.export(&snapshotter, "my-benchmark").unwrap();
/// ```
pub struct MetricsExporter {
    output_dir: PathBuf,
}

impl MetricsExporter {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// Export metrics snapshot to a timestamped JSON file.
    /// Returns the path to the written file.
    /// Filename format: {unix_timestamp}-{label}-metrics.json
    pub fn export(&self, snapshotter: &Snapshotter, label: &str) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(&self.output_dir)?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let snapshot = snapshotter.snapshot().into_vec();

        let mut histograms = serde_json::Map::new();
        let mut counters = serde_json::Map::new();
        let mut gauges = serde_json::Map::new();

        for (key, _unit, _desc, value) in &snapshot {
            let name = key.key().name().to_string();
            let labels: Vec<String> = key
                .key()
                .labels()
                .map(|l| format!("{}={}", l.key(), l.value()))
                .collect();
            let full_name = if labels.is_empty() {
                name
            } else {
                format!("{}{{{}}}", name, labels.join(","))
            };

            match value {
                DebugValue::Counter(n) => {
                    counters.insert(full_name, json!(n));
                }
                DebugValue::Gauge(g) => {
                    gauges.insert(full_name, json!(g.into_inner()));
                }
                DebugValue::Histogram(vals) => {
                    let float_vals: Vec<f64> = vals.iter().map(|v| v.into_inner()).collect();
                    let count = float_vals.len();
                    let sum: f64 = float_vals.iter().sum();
                    let min = float_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = float_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mean = if count > 0 { sum / count as f64 } else { 0.0 };

                    histograms.insert(
                        full_name,
                        json!({
                            "count": count,
                            "sum": sum,
                            "min": min,
                            "max": max,
                            "mean": mean,
                            "values": float_vals
                        }),
                    );
                }
            }
        }

        let output = json!({
            "timestamp_unix": timestamp,
            "timestamp_iso": chrono::Utc::now().to_rfc3339(),
            "label": label,
            "histograms": histograms,
            "counters": counters,
            "gauges": gauges
        });

        let filename = format!("{timestamp}-{label}-metrics.json");
        let path = self.output_dir.join(&filename);
        let file = std::fs::File::create(&path)?;
        serde_json::to_writer_pretty(file, &output)?;

        println!("Metrics exported to: {}", path.display());
        Ok(path)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureBenchmarkRecord {
    pub architecture: String,
    pub model: String,
    pub fixture: String,
    pub fixture_key: String,
    pub entity_recall: f64,
    pub entity_count: usize,
    pub expected_entity_count: usize,
    pub relationship_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relationship_duplicates: Option<usize>,
    pub latency_ms: f64,
    pub document_level: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parser_ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_label: Option<String>,
    pub timestamp: String,
}

impl ArchitectureBenchmarkRecord {
    pub fn now_timestamp() -> String {
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
    }

    pub fn record_metrics(&self) {
        let mut labels = vec![
            ("architecture", self.architecture.clone()),
            ("model", self.model.clone()),
            ("fixture", self.fixture_key.clone()),
        ];

        if self.document_level {
            labels.push(("scope", "document".to_string()));
        } else {
            labels.push(("scope", "chunk".to_string()));
        }

        metrics::histogram!(
            "rql.architecture_bakeoff.entity_recall",
            "architecture" => labels[0].1.clone(),
            "model" => labels[1].1.clone(),
            "fixture" => labels[2].1.clone(),
            "scope" => labels[3].1.clone()
        )
        .record(self.entity_recall);

        metrics::histogram!(
            "rql.architecture_bakeoff.latency_ms",
            "architecture" => labels[0].1.clone(),
            "model" => labels[1].1.clone(),
            "fixture" => labels[2].1.clone(),
            "scope" => labels[3].1.clone()
        )
        .record(self.latency_ms);

        metrics::gauge!(
            "rql.architecture_bakeoff.entity_count",
            "architecture" => labels[0].1.clone(),
            "model" => labels[1].1.clone(),
            "fixture" => labels[2].1.clone(),
            "scope" => labels[3].1.clone()
        )
        .set(self.entity_count as f64);

        metrics::gauge!(
            "rql.architecture_bakeoff.expected_entity_count",
            "architecture" => labels[0].1.clone(),
            "model" => labels[1].1.clone(),
            "fixture" => labels[2].1.clone(),
            "scope" => labels[3].1.clone()
        )
        .set(self.expected_entity_count as f64);

        metrics::gauge!(
            "rql.architecture_bakeoff.relationship_count",
            "architecture" => labels[0].1.clone(),
            "model" => labels[1].1.clone(),
            "fixture" => labels[2].1.clone(),
            "scope" => labels[3].1.clone()
        )
        .set(self.relationship_count as f64);

        if let Some(dupes) = self.relationship_duplicates {
            metrics::gauge!(
                "rql.architecture_bakeoff.relationship_duplicates",
                "architecture" => labels[0].1.clone(),
                "model" => labels[1].1.clone(),
                "fixture" => labels[2].1.clone(),
                "scope" => labels[3].1.clone()
            )
            .set(dupes as f64);
        }

        if let Some(count) = self.candidate_count {
            metrics::gauge!(
                "rql.architecture_bakeoff.candidate_count",
                "architecture" => labels[0].1.clone(),
                "model" => labels[1].1.clone(),
                "fixture" => labels[2].1.clone(),
                "scope" => labels[3].1.clone()
            )
            .set(count as f64);
        }

        if let Some(ms) = self.pipeline_ms {
            metrics::histogram!(
                "rql.architecture_bakeoff.pipeline_ms",
                "architecture" => labels[0].1.clone(),
                "model" => labels[1].1.clone(),
                "fixture" => labels[2].1.clone(),
                "scope" => labels[3].1.clone()
            )
            .record(ms);
        }
    }
}

pub struct JsonlBenchmarkWriter {
    path: PathBuf,
}

impl JsonlBenchmarkWriter {
    pub fn new(output_dir: impl AsRef<Path>, label: &str) -> anyhow::Result<Self> {
        std::fs::create_dir_all(output_dir.as_ref())?;
        Ok(Self {
            path: output_dir.as_ref().join(format!("{label}.jsonl")),
        })
    }

    pub fn append(&self, record: &ArchitectureBenchmarkRecord) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{}", serde_json::to_string(record)?)?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

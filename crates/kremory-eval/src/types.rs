//! Shared type definitions for the kremory eval harness.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// EvalLayer
// ---------------------------------------------------------------------------

/// Identifies which layer of the eval hierarchy a report belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalLayer {
    /// Layer A: published-comparable benchmarks (LongMemEval, DMR, LoCoMo).
    A,
    /// Layer B: diagnostic metrics (entity P/R/F1, RAGAS, graph integrity).
    B,
}

// ---------------------------------------------------------------------------
// ScoreMetadata
// ---------------------------------------------------------------------------

/// Standardised metadata attached to a scorer result.
///
/// Scorers populate the fields relevant to their algorithm; others stay `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreMetadata {
    /// Entity-level precision (for F1 scorer).
    pub precision: Option<f64>,
    /// Entity-level recall (for F1 scorer).
    pub recall: Option<f64>,
    /// F1 harmonic mean (for F1 scorer).
    pub f1: Option<f64>,
    /// Number of true positives.
    pub true_positives: Option<usize>,
    /// Number of false positives.
    pub false_positives: Option<usize>,
    /// Number of false negatives.
    pub false_negatives: Option<usize>,
    /// Latency of judge inference in milliseconds (model-graded scorers).
    pub judge_latency_ms: Option<f64>,
    /// Model path used by judge (model-graded scorers).
    pub judge_model: Option<String>,
    /// Raw content returned by judge before parsing (debug aid).
    pub judge_raw_content: Option<String>,
}

impl From<ScoreMetadata> for serde_json::Value {
    fn from(m: ScoreMetadata) -> Self {
        serde_json::to_value(m).unwrap_or(serde_json::Value::Null)
    }
}

// ---------------------------------------------------------------------------
// EvalReport
// ---------------------------------------------------------------------------

/// Aggregate result for one eval run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Which eval layer produced this report.
    pub layer: EvalLayer,
    /// Benchmark or metric name, e.g. `"longmemeval"` or `"entity-f1"`.
    pub benchmark: String,
    /// Mean score across all samples, in `[0.0, 1.0]`.
    pub overall_score: f64,
    /// Per-category breakdown (category name → mean score).
    pub per_category: HashMap<String, f64>,
    /// Optional token-efficiency ratio (answered tokens / total tokens).
    pub token_efficiency: Option<f64>,
    /// UTC timestamp of when this report was produced.
    pub timestamp: DateTime<Utc>,
    /// kremory crate version that was evaluated.
    pub kremory_version: String,
}

// ---------------------------------------------------------------------------
// EvalError
// ---------------------------------------------------------------------------

/// Error type for eval operations.
///
/// The return type alias `EvalError<T>` is a `Result<T, EvalErr>`.
#[derive(Debug, Error)]
pub enum EvalErr {
    /// Judge inference failed.
    #[error("judge inference error: {0}")]
    JudgeInference(String),

    /// Judge returned output that could not be parsed as valid JSON verdict.
    #[error("judge parse error: {0}")]
    JudgeParse(String),

    /// A required field was missing in a sample or ground-truth record.
    #[error("missing field: {0}")]
    MissingField(String),

    /// I/O error reading fixtures or writing output.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialisation error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic wrapper for errors propagated from other crates.
    #[error("{0}")]
    Other(String),
}

/// Convenience result alias used throughout the eval crate.
pub type EvalError<T> = Result<T, EvalErr>;

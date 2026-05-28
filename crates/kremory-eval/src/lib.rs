//! kremory-eval — internal quality eval harness (not published)
//!
//! Layer A: published-comparable benchmarks (LongMemEval / DMR / LoCoMo)
//! Layer B: diagnostic metrics (entity P/R/F1 + RAGAS + graph integrity)
//!
//! See `.ai-docs/planning/quality-eval-strategy-2026-05-28.md` for design.
//!
//! # Trait overview (Inspect AI–inspired)
//!
//! ```text
//! Dataset  → Iterator<Item = Sample>
//! Solver   → async fn solve(&self, sample: &Sample) -> Output
//! Scorer   → async fn score(&self, sample: &Sample, output: &Output) -> Score
//! ```

pub mod judge;
pub mod layer_b;
pub mod scorers;
pub mod types;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core value types
// ---------------------------------------------------------------------------

/// Tie-break policy when a scorer is exactly on the boundary.
///
/// Phase 1 gate default: `Pass` (boundary values count as passing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TieBreakPolicy {
    /// A score exactly on the threshold boundary counts as passing.
    #[default]
    Pass,
    /// A score exactly on the threshold boundary counts as failing.
    Fail,
}

/// A single scoring result.
///
/// `value` is always in `[0.0, 1.0]`.
/// `reasoning` explains the verdict (mandatory for model-graded scorers,
/// optional for deterministic ones).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Numeric score in `[0.0, 1.0]`.
    pub value: f64,
    /// Human-readable explanation of this verdict.
    pub reasoning: String,
    /// Scorer-specific metadata (precision, recall, token counts, etc.).
    pub metadata: serde_json::Value,
}

impl Score {
    /// Convenience constructor for simple pass/fail with no metadata.
    pub fn binary(value: f64, reasoning: impl Into<String>) -> Self {
        Self {
            value,
            reasoning: reasoning.into(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Whether this score passes at `threshold`, honouring `policy`.
    pub fn passes(&self, threshold: f64, policy: TieBreakPolicy) -> bool {
        match policy {
            TieBreakPolicy::Pass => self.value >= threshold,
            TieBreakPolicy::Fail => self.value > threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A source of evaluation samples.
pub trait Dataset {
    /// The item type produced by this dataset.
    type Sample;
    /// Iterate over all samples.
    fn samples(&self) -> impl Iterator<Item = Self::Sample>;
}

/// Produces a model output for a given sample.
///
/// In practice this wraps kremory `Memory::open` + a query.
pub trait Solver<Sample, Output>: Send + Sync {
    /// Solve one sample, returning an output to be scored.
    fn solve(
        &self,
        sample: &Sample,
    ) -> impl std::future::Future<Output = Output> + Send;
}

/// Compares a `Solver` output against ground truth and returns a `Score`.
pub trait Scorer<Sample, Output>: Send + Sync {
    /// Score one (sample, output) pair.
    fn score(
        &self,
        sample: &Sample,
        output: &Output,
    ) -> impl std::future::Future<Output = crate::types::EvalError<Score>> + Send;
}

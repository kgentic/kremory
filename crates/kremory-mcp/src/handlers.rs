//! Transport-agnostic handler bodies shared by every kremory-mcp transport —
//! the MCP stdio tool surface (`lib.rs`) AND the REST bin
//! (`bin/kremory-http.rs`).
//!
//! Deliberately kept free of any transport-specific type (no `rmcp` types,
//! no `axum` types) — plain wire param structs in, plain wire output structs
//! (or `ToolError`) out, over a `&kremory::Memory` facade builder chain. Each
//! transport owns its own request/response marshalling (JSON-RPC `Parameters`
//! / `CallToolResult` for MCP; `Json`/`Query`/`Path` extractors + a
//! `ToolError -> (StatusCode, Json)` mapping for REST) and calls straight
//! into these functions.

use kremory::Memory;

use crate::conversions::ConversionError;
use crate::params::{
    DreamOutput, DreamParams, RecallFormat, RecallParams, RecallStructuredOutput, RecallTextOutput,
    RememberOutput, RememberParams,
};

// ────────────────────────────────────────────────────────────────────────
// Internal error unification
// ────────────────────────────────────────────────────────────────────────

/// Unifies [`ConversionError`] and `kremory::MemoryError` so callers can
/// record an observability outcome label BEFORE converting to whatever
/// transport-specific error shape they need (MCP `ErrorData` in `lib.rs`,
/// an HTTP status code in `bin/kremory-http.rs`).
#[derive(Debug)]
pub enum ToolError {
    InvalidParams(String),
    Internal(String),
}

impl ToolError {
    pub fn outcome_label(&self) -> &'static str {
        match self {
            ToolError::InvalidParams(_) => "invalid_params",
            ToolError::Internal(_) => "internal_error",
        }
    }
}

impl From<ConversionError> for ToolError {
    fn from(e: ConversionError) -> Self {
        ToolError::InvalidParams(e.to_string())
    }
}

impl From<kremory::MemoryError> for ToolError {
    fn from(e: kremory::MemoryError) -> Self {
        ToolError::Internal(e.to_string())
    }
}

// ────────────────────────────────────────────────────────────────────────
// Pure handler bodies — facade builder chains, shared across transports.
// ────────────────────────────────────────────────────────────────────────

pub async fn do_remember(
    mem: &Memory,
    params: RememberParams,
) -> Result<RememberOutput, ToolError> {
    let resolved = params.resolve()?;

    let mut req = mem
        .remember(resolved.content)
        .in_namespace(resolved.namespace);
    if let Some(ts) = resolved.published_at {
        req = req.published_at(ts);
    }
    match (resolved.source_kind, resolved.source_id) {
        (Some(kind), id) => {
            let id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            req = req.from_source(id, kind);
        }
        (None, Some(id)) => {
            req = req.from_source(id, kremory::SourceKind::Chat);
        }
        (None, None) => {}
    }
    if !resolved.facts.is_empty() {
        req = req.with_facts(resolved.facts);
    }
    if resolved.skip_extraction {
        req = req.skip_extraction();
    }

    let commit = req.await?;
    Ok(RememberOutput::from(commit))
}

pub async fn do_recall(mem: &Memory, params: RecallParams) -> Result<serde_json::Value, ToolError> {
    let resolved = params.resolve()?;

    let mut req = mem.recall(resolved.query).in_namespace(resolved.namespace);
    if let Some(k) = resolved.k {
        req = req.k(k);
    }
    if let Some(as_of) = resolved.as_of {
        req = req.as_of(as_of);
    }

    match resolved.format {
        RecallFormat::Text => {
            let block = req.as_template(resolved.template).await?;
            serde_json::to_value(RecallTextOutput { block }).map_err(|e| {
                ToolError::Internal(format!("failed to serialize recall text output: {e}"))
            })
        }
        RecallFormat::Structured => {
            let results = req.raw().await?;
            let count = results.len();
            let wire = RecallStructuredOutput {
                results: results.into_iter().map(Into::into).collect(),
                count,
            };
            serde_json::to_value(wire).map_err(|e| {
                ToolError::Internal(format!("failed to serialize recall structured output: {e}"))
            })
        }
    }
}

pub async fn do_dream(mem: &Memory, params: DreamParams) -> Result<DreamOutput, ToolError> {
    let resolved = params.resolve()?;

    let mut req = mem
        .dream()
        .in_namespace(resolved.namespace)
        .await_completion();
    if let Some(id) = resolved.batch_id {
        req = req.for_batch(id);
    }

    let summary = req.await?;
    Ok(DreamOutput::from(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_error_outcome_labels() {
        assert_eq!(
            ToolError::InvalidParams("x".into()).outcome_label(),
            "invalid_params"
        );
        assert_eq!(
            ToolError::Internal("x".into()).outcome_label(),
            "internal_error"
        );
    }
}

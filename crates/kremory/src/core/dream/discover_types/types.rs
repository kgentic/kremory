// ─── Public result types ──────────────────────────────────────────────────────

/// A single discovered or proposed entity type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeProposal {
    pub name: String,
    pub description: String,
    pub justification: String,
}

/// Result of a single `discover_types` invocation.
#[derive(Debug, Default)]
pub struct DiscoveryResult {
    /// All proposals emitted by the LLM (before gating).
    pub types_proposed: Vec<TypeProposal>,
    /// Proposals that passed both the shape validator and the anti-redundancy gate.
    pub types_accepted: Vec<TypeProposal>,
    /// Proposals that were rejected, with the reason.
    pub types_rejected: Vec<(TypeProposal, String)>,
    /// Warnings for operator attention (e.g. degraded-mode runs).
    pub warnings: Vec<String>,
    /// Number of evidence entities retyped in-place.
    pub entities_retyped: usize,
}

/// Maximum cluster candidates passed to the LLM in one call.
/// Enforced PROMPT-SIDE — embedded in the system prompt instruction.
pub(crate) const MAX_PROPOSALS: usize = 5;

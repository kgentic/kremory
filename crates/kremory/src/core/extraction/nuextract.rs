//! Removed in E-1 (Foundation Sprint 2026-06-09).
//!
//! `NuExtractExtractor` and `GroundedNuExtractExtractor` were deleted as part
//! of the BYOE redesign (ADR-039).  The production dispatch enum is now
//! `ExtractorKind` in `factory.rs`.  The `ner`-gated hybrid path is
//! `GlinerLlmExtractor` in `hybrid_typer.rs`.

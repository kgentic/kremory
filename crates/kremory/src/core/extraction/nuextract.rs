//! Removed as part of a BYOE (bring-your-own-extractor) redesign.
//!
//! `NuExtractExtractor` and `GroundedNuExtractExtractor` were deleted as part
//! of that redesign.  The production dispatch enum is now
//! `ExtractorKind` in `factory.rs`.  The `ner`-gated hybrid path is
//! `GlinerLlmExtractor` in `hybrid_typer.rs`.

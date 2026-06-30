use std::borrow::Cow;

use crate::core::config::EntityEmbeddingInput;

/// Compose the text to embed for an entity during indexing and disambiguation.
///
/// `NameContext` (default): returns `"{name}\n{context}"` — a richer embedding
/// that distinguishes homonyms and acronym expansions via surrounding context
/// (ADR-058 B1). Falls back to bare `name` when `context` is empty.
/// `Name`: returns bare `"{name}"` — original pre-B1 behaviour.
///
/// Returns [`Cow`] so the bare-`name` paths (mode `Name`, or empty context)
/// borrow without allocating; only the name+context join owns a new `String`.
/// This is the single composition point enforcing the index-time / probe-time
/// symmetry invariant (spec §H1) — both call sites MUST route through here.
pub(crate) fn compose_embed_text<'a>(
    name: &'a str,
    context: &str,
    mode: EntityEmbeddingInput,
) -> Cow<'a, str> {
    match mode {
        EntityEmbeddingInput::Name => Cow::Borrowed(name),
        EntityEmbeddingInput::NameContext if context.is_empty() => Cow::Borrowed(name),
        EntityEmbeddingInput::NameContext => Cow::Owned(format!("{name}\n{context}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_context_with_context_returns_combined() {
        let result = compose_embed_text(
            "Alice",
            "a senior engineer",
            EntityEmbeddingInput::NameContext,
        );
        assert_eq!(result, "Alice\na senior engineer");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn name_context_empty_context_returns_bare_name() {
        let result = compose_embed_text("Alice", "", EntityEmbeddingInput::NameContext);
        assert_eq!(result, "Alice");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn name_mode_ignores_context() {
        let result = compose_embed_text("Alice", "a senior engineer", EntityEmbeddingInput::Name);
        assert_eq!(result, "Alice");
        assert!(matches!(result, Cow::Borrowed(_)));
    }
}

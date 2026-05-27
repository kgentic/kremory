//! A.8a/A.8d — Tier 1 shortcut shape and env-gate tests.
//!
//! Tests `Memory::auto`, `Memory::with_ollama`, `Memory::with_openai`,
//! `Memory::with_anthropic`, and `Memory::with_ollama_at`.
//!
//! Tier 1 constructors wire real provider builders at v0.1.0 (no network calls at
//! construction time — network errors surface on first `.remember()`/`.recall()`).
//! `Memory::with_openai` / `Memory::with_anthropic` require env vars set.

use kremory::Memory;

/// `Memory::with_ollama` succeeds — constructs real Ollama provider + EngineGraphHandle.
#[tokio::test]
async fn with_ollama_succeeds_in_test_mode() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let _mem = Memory::with_ollama(tmp.path().join("kremory-tier1.db"))
        .await
        .expect("with_ollama should succeed");
}

/// `Memory::with_ollama_at` accepts a custom URL.
#[tokio::test]
async fn with_ollama_at_accepts_custom_url() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let _mem = Memory::with_ollama_at("http://my-ollama:11434", tmp.path().join("kremory-tier1.db"))
        .await
        .expect("with_ollama_at should succeed");
}

/// `Memory::with_openai` fails when `OPENAI_API_KEY` is not set.
#[tokio::test]
async fn with_openai_without_api_key_errors() {
    // Ensure the env var is NOT set for this test.
    // We cannot unset global env in a multi-threaded context safely, so
    // skip this assertion when the key IS set (CI may inject it).
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return; // Key is set → test inapplicable in this environment.
    }
    let result = Memory::with_openai("/tmp/test.db").await;
    assert!(result.is_err(), "should fail without OPENAI_API_KEY");
    let msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => unreachable!("already asserted is_err"),
    };
    assert!(
        msg.contains("OPENAI_API_KEY"),
        "error should mention OPENAI_API_KEY, got: {msg}"
    );
}

/// `Memory::with_anthropic` fails when `ANTHROPIC_API_KEY` is not set.
#[tokio::test]
async fn with_anthropic_without_api_key_errors() {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return;
    }
    let result = Memory::with_anthropic("/tmp/test.db").await;
    assert!(result.is_err(), "should fail without ANTHROPIC_API_KEY");
    let msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => unreachable!("already asserted is_err"),
    };
    assert!(
        msg.contains("ANTHROPIC_API_KEY"),
        "error should mention ANTHROPIC_API_KEY, got: {msg}"
    );
}

/// `Memory::auto` returns `NoProviderConfigured` when no env vars are set.
#[tokio::test]
async fn auto_without_any_provider_env_errors() {
    if std::env::var("OLLAMA_HOST").is_ok()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("ANTHROPIC_API_KEY").is_ok()
    {
        return; // Provider env is set → skip this assertion.
    }
    use kremory::MemoryError;
    let result = Memory::auto("/tmp/test.db").await;
    assert!(result.is_err(), "should fail when no provider env vars set");
    let is_correct_variant = match result {
        Err(MemoryError::NoProviderConfigured { .. }) => true,
        Err(e) => panic!("expected NoProviderConfigured, got: {e}"),
        Ok(_) => unreachable!("already asserted is_err"),
    };
    assert!(is_correct_variant);
}

/// `Memory` returned from Tier 1 shortcut is Clone.
#[tokio::test]
async fn tier1_memory_is_clone() {
    let mem = Memory::with_ollama("/tmp/test.db")
        .await
        .expect("should succeed");
    let _clone = mem.clone();
}

//! Shared test helpers for LLM integration tests.
//!
//! Gated: only compiled when `llm-integration` feature is active.
//! The submodules expose types used exclusively by `tests/llm_integration.rs`.

pub mod metrics_capture;
pub mod mock_chat;
pub mod ollama_adapter;

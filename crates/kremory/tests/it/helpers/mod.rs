#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Shared test helpers for LLM integration tests.
//!
//! Gated: only compiled when `llm-integration` feature is active.
//! The submodules expose types used exclusively by `tests/llm_integration.rs`.

pub mod metrics_capture;
pub mod mock_chat;
pub mod ollama_adapter;
pub mod recording_sink;
pub mod scripted_llm;

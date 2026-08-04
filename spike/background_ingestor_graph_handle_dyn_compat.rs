// Spike A: BackgroundIngestorGraphHandle dyn-compat compile spike
//
// This file documents the isolated architecture claim. The actual compile
// test is in crates/kremory/tests/spike_dyn_compat.rs (needs async-trait dep).
//
// Purpose: record the minimal shapes used in the compile spike for review.
// Run: see tests/spike_dyn_compat.rs
//
// Claim: BackgroundIngestorGraphHandle implements GraphHandle using
// #[async_trait] and coerces to Arc<dyn GraphHandle> without compile error,
// even with a Drop impl and Arc<Mutex<Option<IngestGuard>>> field.
//
// Three failure modes checked:
// 1. #[async_trait] macro generates code incompatible with dyn GraphHandle
// 2. Custom Drop impl on struct behind Arc<dyn> has weird semantics
// 3. Arc<BackgroundIngestorGraphHandle> as Arc<dyn GraphHandle> coerce fails
//
// VERDICT: See spike_dyn_compat test output.

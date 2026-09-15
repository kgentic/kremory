# Node / napi binding


The `kremory-napi` crate exposes the `Memory` facade to Node via napi-rs, mirroring the Rust surface
in **camelCase**. The reversibility surface is fully mirrored:

- Inspect: `memory.mutationHistory(...)`, `memory.listMutations(...)` → `MutationRecord[]`.
- Undo: `memory.undo(mutationId)` (the unified dispatcher) plus the per-kind `memory.unmerge` /
  `memory.undoEntityEdit` / `memory.undoDeleteEntity` / `memory.undoDeleteFact`, and the direct
  mutations `memory.editEntity` / `memory.deleteEntity` / `memory.deleteFact`.
- `DreamSummary` mirrors as `JsDreamSummary` with the honest fields (`crossEpisodeWouldMerge` vs
  `crossEpisodeMerged`, `budgetExhausted`, …).

> **Updated 2026-07-28.** This note previously said `content-search` was "a Rust-only
> opt-in ... not enabled in the current `kremory-napi` build". That understated it: the binding had
> **no passthrough for the feature at all**, so a Node consumer could not enable it under any
> circumstances and was locked to entity/fact-only recall — which measures **−32.2pt**.
> `kremory-napi` now declares `default = ["content-search"]` plus an opt-in `rerank` passthrough and
> a `RecallOptions.rerankK` knob, so JS `recall()` gets the same RRF-fused hybrid surface Rust does.
> What is still Rust-only is the **explicit `.content()` terminal + the `ContentPassage` type** — a
> new JS output type rather than a field mirror. The undo + inspect surface *is* available in JS.

---

*API reference current as of kremory v0.7 (2026-09-07). Facade design: outside-in API design. Temporal model: two independent clocks (transaction time + valid time). BYOM contract: bring-your-own-model, no bundled embedder. Dream reversibility: every committed mutation is reversible. Content recall: BM25/FTS5 search over raw episode text. Crate topology: single crate + cargo features.*

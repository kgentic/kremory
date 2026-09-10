# Examples — start here

Twenty-two runnable programs. Every one is real code that compiles against the
published crate and asserts its own behaviour, so if kremory changes and an
example stops being true, it stops passing.

**Twenty of the twenty-two need nothing but `cargo`.** One needs a model running
locally; one calls a paid API and is the only thing here that costs anything.

```sh
cargo run --example offline_remember_recall
```

## Which one do you want?

| If you are asking… | Read |
|---|---|
| What does this thing even do? | `offline_remember_recall` |
| How does it answer "where do they live *now*" and "…in March" at once? | `remembers_across_sessions` |
| Someone told me a fact changed. How do I correct it? | `correcting_the_record` |
| I have documents. Can I just search them? | `searching_documents` |
| My documents are long. Does the end of them survive? | `a_long_document` |
| I have ten thousand rows already. How do I get them in? | `bulk_import` |
| Ingest is slow. How do I not block my request handler? | `ingest_without_blocking` |
| I have many customers. How do I keep them apart? | `multi_tenant_isolation` |
| Someone invoked their right to erasure. | `gdpr_erasure_by_source` |
| This record must never be editable. | `append_only_namespace` |
| That one line should never have been recorded. | `deleting_and_restoring` |
| An automated job changed something wrongly. | `undoing_a_bad_change` |
| I switched embedding model. Now what? | `changing_embedding_model` |
| A web process AND a worker both need this. | `two_handles_one_database` |
| The correction I made was itself wrong. | `undoing_a_correction` |
| My entities are courts and statutes, not people. | `domain_entity_types` |
| Can consolidation just run by itself? | `dream_on_a_schedule` |
| Show me it building a graph from plain English. | `agent_memory_with_ollama` ⚠️ needs Ollama |
| Something is running and I need it to stop. | `cancelling_in_flight_work` |
| Prove to an auditor what happened to this record. | `who_touched_this_record` |
| Which of these changes can I take back? | `what_can_be_undone` |
| I want to use OpenAI or Claude instead. | `hosted_providers` ⚠️ costs money |

## The three that surprise people

Read these even if the table did not send you there — each documents a behaviour
that is correct and counter-intuitive, and getting it wrong is quiet rather than
loud.

- **`correcting_the_record`** — `supersede()` alone leaves the old fact reading
  as current until a retirement sweep runs. You want `.close_now()` too, and
  without it you get two contradictory answers with no warning.
- **`searching_documents`** — a returned "passage" is the whole EPISODE, not an
  extract. Ingest a 60-page PDF as one episode and a search result is 60 pages.
  Chunk at ingest.
- **`changing_embedding_model`** — after a model change, `backfill_*` reports
  success having done nothing, because no vector is *missing*; they are merely
  wrong. `reembed_all_*` is the one you want.
- **`what_can_be_undone`** — two reversal operations (`unmerge`,
  `restore_archived_fact`) are public but have **no reachable handle** from the
  API today (TD-250). The example asserts that limitation, so it will start
  failing when the gap closes.
- **`cancelling_in_flight_work`** — `await_batch()` after a cancel hangs until
  its timeout on some runs (TD-251). Poll `status_of` per episode instead.
- **`hosted_providers`** — `Memory::with_anthropic` gives you Claude and **no
  embedding model**, because Anthropic has no embedding API. Recall silently
  becomes structural rather than semantic. Pair Claude with a real embedder
  yourself.

## Running them all

```sh
bash scripts/check-examples.sh      # all 20 offline ones, ~25s
```

This runs in `scripts/check-all.sh` too. The examples ship inside the published
crate, so a broken one is something users download — and before this guard
existed, nothing executed them.

## What these are for

They are documentation that cannot lie, and they have earned their keep as tests:
writing them surfaced **five bugs** that code review and ~1,800 unit tests had
not, four of which are fixed. Three shared one shape — a capability present on
one path and missing from its sibling — which no test caught because each test
drives *a* path rather than *both*.

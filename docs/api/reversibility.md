# Reversibility and deletion

## Reversibility — see, trust, undo

Because `dream()` mutates the graph by default, kremory ships a **reversibility substrate**
(ADR-073): every logged destructive mutation can be inspected and undone. Undo is deterministic
(replayed from an in-transaction snapshot — no LLM), idempotent (a second undo is a zero-count
no-op, never a double-restore), and returns an **honest outcome** (the actual counts reversed,
never a bare "ok").

### SEE — inspect what changed

```rust
use kremory::{MutationRecord, MutationKind};

// Every logged mutation that touched one entity (newest-first, includes already-undone):
let history: Vec<MutationRecord> = mem.mutation_history("alice j")
    .in_namespace(Namespace::new("agent"))
    .await?;

for r in &history {
    // r.mutation_id: i64, r.kind: MutationKind, r.created_at: String (RFC3339),
    // r.undone: bool, r.group_id: String, r.affected_entities: Vec<String>, r.summary: String
    println!("#{} [{}] {}", r.mutation_id, if r.undone { "undone" } else { "live" }, r.summary);
}

// Or list a whole namespace's mutations, newest-first (LIVE / still-reversible by default):
let all: Vec<MutationRecord> = mem.list_mutations()
    .in_namespace(Namespace::new("agent"))
    .kind(MutationKind::EntityMerge)   // optional: filter to one kind
    .since(chrono::Utc::now() - chrono::Duration::days(1)) // optional: created_at lower bound
    .include_undone(true)              // optional: also show reversed mutations (default: false)
    .await?;
```

`list_mutations()` without `.in_namespace(...)` falls back to the `default_namespace`, else scans
**all** namespaces (a valid admin view). Both inspect requests are read-only — `.await` them.

### UNDO — the unified dispatcher + per-kind methods

The recommended entry point is `mem.undo(mutation_id)`: it reads the mutation's kind and dispatches
to the correct reversal, so you can iterate `list_mutations()` and undo uniformly without switching
on the kind by hand.

```rust
use kremory::Namespace;

if let Some(rec) = history.first() {
    // Optional .in_namespace(ns) GUARDS the undo to the mutation's original namespace.
    let outcome = mem.undo(rec.mutation_id).execute().await?;
    println!("reversed: {outcome:?}");   // UndoOutcome::{Unmerge|EditEntity|DeleteEntity|DeleteFact}(..)
}
```

`UndoOutcome` is a `#[non_exhaustive]` enum — one variant per log-dispatchable kind, each wrapping
that kind's honest per-op outcome. The per-kind methods remain available as the escape hatch when
you already know the kind:

| Method | Reverses | Argument | Honest outcome |
|---|---|---|---|
| `mem.undo(mutation_id)` | **any** logged mutation | `mutation_id` | `UndoOutcome` |
| `mem.unmerge(mutation_id)` | an `entity_merge` | `mutation_id` | `UnmergeOutcome` |
| `mem.undo_entity_edit(mutation_id)` | an `edit_entity` (rename/retype) | `mutation_id` | `EditEntityOutcome` |
| `mem.undo_delete_entity(mutation_id)` | a `delete_entity` | `mutation_id` | `DeleteEntityOutcome` |
| `mem.undo_delete_fact(mutation_id)` | a `delete_fact` | `mutation_id` | `DeleteFactOutcome` |
| `mem.unsupersede(fact_id)` | a supersession bound | **`fact_id`** | `UnsupersedeOutcome` |
| `mem.restore_archived_fact(archived_fact_id)` | a P2 fact archive | **`archived_fact_id`** | `RestoreArchivedOutcome` |

Each honest-outcome struct carries the actual counts reversed and an idempotency flag — e.g.
`UnmergeOutcome { restored_entity, keeper, facts_repointed, edges_restored, entities_reopened,
nogood_recorded, already_undone }`.

### The `merge_nogood` guarantee

`unmerge` (and `undo()` on an `entity_merge`) does more than split the pair back apart: it records
a **merge NOGOOD** for the split pair (`UnmergeOutcome.nogood_recorded == true`), so the **next
`dream()` will not re-merge them**. Undoing a merge is durable, not a one-cycle reprieve.

### The 4-of-8 tracked-kind boundary

There are eight `MutationKind` variants, but only **four are logged today and therefore reversible
via `undo()`**: `EntityMerge`, `EntityEdit`, `EntityDelete`, `FactDelete`. The other four
(`FactSupersede`, `FactArchive`, `CommunityAssign`, `CanonicalForm`) are **reserved** — not yet
produced into the mutation log — so:

- `list_mutations().kind(<a reserved kind>)` returns **empty by construction** (not "nothing
  changed").
- `undo(id)` on a would-be row of a reserved kind returns a loud `Error::UndoUnsupportedKind`.

`FactSupersede` / `FactArchive` are themselves reversible — but through their **domain-id** methods
`unsupersede(fact_id)` / `restore_archived_fact(archived_fact_id)`, not the `mutation_id`-based
`undo()` surface.

### Direct mutations (also reversible)

Beyond `dream()`, the same reversible substrate backs the direct consumer mutations:

```rust
// Rename a diarization placeholder — every fact / edge / membership re-points to the new id;
// undoable via the returned mutation_id.
let edit = mem.edit_entity("Speaker 1")
    .rename("alice")                      // XOR .retype(type_id)
    .in_namespace(Namespace::new("meeting"))
    .execute()
    .await?;
mem.undo_entity_edit(edit.mutation_id).execute().await?;

// Reversible delete — facts are ARCHIVED (recoverable), never hard-deleted.
let del = mem.delete_entity("bob").in_namespace(ns.clone()).execute().await?;
mem.undo_delete_entity(del.mutation_id).execute().await?;

let dfact = mem.delete_fact(fact_id).execute().await?;   // fact_id is global
mem.undo_delete_fact(dfact.mutation_id).execute().await?;

// Explicitly bound a fact's world-time validity window (consumer-driven supersession).
// .at(valid_to) is REQUIRED (no silent default).
//
// ⚠️ .at() ALONE IS USUALLY NOT WHAT YOU WANT — see the note below this block.
let outcome = mem.supersede(fact_id)
    .at(chrono::Utc::now())
    .close_now()
    .in_namespace(ns)
    .execute()
    .await?;   // SupersedeOutcome::{Bounded { retired } | RejectedTimeInversion | NotFound}
```

### ⚠️ `.at()` without `.close_now()` looks like it did nothing

Setting a bound records *when* the fact stopped being true. It does **not** retire the
fact — that happens on the next `dream()` supersession sweep, or immediately if you add
`.close_now()`.

So a correction applied with `.at(Utc::now())` alone leaves the old fact still showing up
in a default recall. Nothing errored, the call returned `Ok`, and the record still reads
as current. Anyone verifying their own correction will conclude supersession is broken.

```rust ignore
// Records the bound, but the fact still reads as current until dream() runs:
mem.supersede(fact_id).at(Utc::now()).execute().await?;

// Records the bound AND retires it now — what a correction usually means:
mem.supersede(fact_id).at(Utc::now()).close_now().execute().await?;
```

`.close_now()` only retires bounds already in the past. A **future-dated** bound cannot
be retired yet, so it returns `Bounded { retired: 0 }` and waits for a later sweep — the
count is in-band precisely so you can see the deferral rather than assume it failed.

Runnable: `cargo run --example correcting_the_record`, and
`cargo run --example undoing_a_correction` for when the correction was itself wrong.

`edit_entity` / `delete_*` / `supersede` are all `#[must_use]` builders — nothing happens until you
call `.execute()`. Like `dream()`, `supersede()` is rejected on `AppendOnly` namespaces.

---

## Forget (GDPR)

`forget()` returns a builder; the destructive operation only fires on `.execute()`.
This explicit terminal makes the intent visible in code review.

```rust
// Forget everything in the default namespace
let entities_removed: u64 = mem.forget().execute().await?;

// Forget a specific namespace
let entities_removed = mem.forget()
    .in_namespace(Namespace::new("tenant-acme"))
    .execute()
    .await?;

// Right-to-erasure: everything ONE source contributed, leaving other sources intact
let entities_removed = mem.forget()
    .in_namespace(ns)
    .by_source_id("support-chat-4417")
    .execute()
    .await?;
```

### ⚠️ The return value counts ENTITIES, so a successful erasure often returns `0`

Shared-entity preservation pins any subject that also appears in **another** source — the
normal case, since the same person appears in several documents. So a complete, correct
`by_source_id` erasure removes that source's facts and returns **0 entities**.

```rust ignore
// WRONG — a full erasure legitimately reports 0
if removed > 0 { println!("erased"); }
```

Treat a successful `Ok(_)` as the erasure having happened. The count tells you how many
entities became orphaned, not whether the operation worked. A richer return shape is
tracked but would be a breaking change.

Runnable: `cargo run --example gdpr_erasure_by_source` — it asserts both halves, that the
erased source is gone **and** that another source's facts about the same person survive.
Over-deletion is also a failure, and it is the half people forget to test.

### Which tool for which situation

Four operations, and picking the wrong one is the usual mistake:

| you want to say | use | reversible? |
|---|---|---|
| "erase everything this SOURCE contributed" (a data-subject request) | `forget().by_source_id(..)` | no |
| "this USED to be true" (a correction) | `supersede(fact_id).at(..).close_now()` | yes — `unsupersede` |
| "the correction was itself wrong" | `unsupersede(fact_id)` | yes — supersede again |
| "this should never have been recorded" | `delete_fact(fact_id)` | yes — `undo_delete_fact` |

`fact_id` comes from `RetrievedFact::fact_id` on the recall path; `mutation_id` for the
undos comes from `list_mutations()`.

---

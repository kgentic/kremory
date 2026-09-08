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
// .at(valid_to) is REQUIRED (no silent default). The next dream() supersession sweep closes it,
// or .close_now() retires already-past-dated bounds inline.
let outcome = mem.supersede(fact_id)
    .at(chrono::Utc::now())
    .close_now()
    .in_namespace(ns)
    .execute()
    .await?;   // SupersedeOutcome::{Bounded { retired } | RejectedTimeInversion | NotFound}
```

`edit_entity` / `delete_*` / `supersede` are all `#[must_use]` builders — nothing happens until you
call `.execute()`. Like `dream()`, `supersede()` is rejected on `AppendOnly` namespaces.

---

## Forget (GDPR)

`forget()` returns a builder; the destructive operation only fires on `.execute()`.
This explicit terminal makes the intent visible in code review.

```rust
// Forget everything in the default namespace
let deleted: u64 = mem.forget().execute().await?;
println!("{deleted} records deleted");

// Forget a specific namespace
let deleted = mem.forget()
    .in_namespace(Namespace::new("tenant-acme"))
    .execute()
    .await?;
```

---

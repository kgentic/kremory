//! **Runs offline.** Which changes can you take back, and how do you get hold of
//! the handle for each.
//!
//! ```text
//! cargo run --example what_can_be_undone
//! ```
//!
//! ## The problem this solves
//!
//! kremory has five reversal operations and they do not all work the same way.
//! Some take a `fact_id`, some a `mutation_id`; some hand you the id you need and
//! some make you go and look it up. Two of them you cannot currently reach from
//! the public API at all.
//!
//! Finding that out mid-implementation is expensive, so here is the whole map in
//! one runnable program.
//!
//! ## The map
//!
//! ```text
//!   FORWARD              REVERSE                      HANDLE COMES FROM
//!   edit_entity     →    undo_entity_edit(mut_id)     list_mutations()
//!   delete_fact     →    undo_delete_fact(mut_id)     list_mutations()
//!   supersede       →    unsupersede(fact_id)         recall() → fact_id
//!   dream() merge   →    unmerge(mut_id)              ⚠️ see below
//!   dream() archive →    restore_archived_fact(id)    ⚠️ see below
//! ```
//!
//! The first three are exercised below, end to end. The last two are **not
//! reachable from the public API today** and this example proves it rather than
//! asserting it:
//!
//! - **`unmerge`** — its id is obtainable once a merge exists, but nothing public
//!   CREATES a merge. Merges happen only inside `dream()`'s canonicalize pass,
//!   which requires real alias evidence and correctly declines on thin data. The
//!   assertion below demonstrates that on a deliberately merge-friendly corpus.
//! - **`restore_archived_fact`** — takes an `archived_fact_id` that no public
//!   method returns. `DreamSummary` reports `facts_archived` as a COUNT, not ids.
//!
//! Both are tracked as TD-250. **If a future change makes either reachable, the
//! assertions at the end of this file will start failing** — which is the point:
//! this example is also the tripwire that says "update the map".

use std::sync::Arc;

use chrono::Utc;
use kremory::{Memory, Namespace, StructuredFact};

/// A stub chat provider. `dream()` REFUSES to run without an LLM wired — even
/// for its deterministic passes — so an example that calls it cannot use the
/// extractor-only offline setup the other examples use:
///
/// ```text
/// operation `dream` requires an LLM provider — wire an LLM via
/// Memory::open(…).with_llm(…) to enable the dream consolidation phase
/// ```
///
/// Worth knowing before you plan a "consolidate on a schedule" job for a
/// deployment that has no model wired.
fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn kremory::DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

fn fact(subject: &str, predicate: &str, object: &str) -> StructuredFact {
    StructuredFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        valid_from: None,
        valid_to: None,
        memory_type: None,
    }
}

async fn predicates(mem: &Memory, ns: &Namespace) -> anyhow::Result<Vec<String>> {
    let mut v: Vec<String> = mem
        .recall("ingrid")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| f.predicate)
        .collect();
    v.sort();
    Ok(v)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("registry");

    let mem = Memory::open(dir.path().join("registry.db"))
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(null_embedder())
        .await?;

    mem.remember("Ingrid Halvorsen, harbour registry.")
        .in_namespace(ns.clone())
        .with_facts(vec![
            fact("ingrid", "role", "harbourmaster"),
            fact("ingrid", "berth", "D14"),
        ])
        .skip_extraction()
        .await?;
    println!("start: {:?}", predicates(&mem, &ns).await?);

    // ── 1. delete_fact → undo_delete_fact ───────────────────────────────────
    //
    // The handle for the FORWARD op comes from recall; the handle for the
    // REVERSE op comes from the mutation log. Two different lookups.
    let berth = mem
        .recall("ingrid")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == "berth")
        .and_then(|f| f.fact_id)
        .ok_or_else(|| anyhow::anyhow!("expected the berth fact"))?;

    mem.delete_fact(berth).execute().await?;
    let mid = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .await?
        .into_iter()
        .find(|m| !m.undone)
        .ok_or_else(|| anyhow::anyhow!("expected the delete in the log"))?
        .mutation_id;
    mem.undo_delete_fact(mid).execute().await?;
    let after = predicates(&mem, &ns).await?;
    println!("delete → undo      : {after:?}");
    assert!(
        after.iter().any(|p| p == "berth"),
        "undo_delete_fact should restore the fact; got {after:?}"
    );

    // ── 2. supersede → unsupersede ──────────────────────────────────────────
    //
    // The only pair where BOTH halves take the same handle — a fact_id, straight
    // off recall. No mutation-log lookup needed.
    mem.supersede(berth)
        .in_namespace(ns.clone())
        .at(Utc::now())
        .close_now()
        .execute()
        .await?;
    let closed = predicates(&mem, &ns).await?;
    assert!(
        !closed.iter().any(|p| p == "berth"),
        "supersede should close the fact; got {closed:?}"
    );

    mem.unsupersede(berth).execute().await?;
    let reopened = predicates(&mem, &ns).await?;
    println!("supersede → undo   : {reopened:?}");
    assert!(
        reopened.iter().any(|p| p == "berth"),
        "unsupersede should reopen it; got {reopened:?}"
    );

    // ── 3. The two you cannot reach ─────────────────────────────────────────
    //
    // Not narrated — demonstrated. A corpus built to be as merge-friendly as
    // possible: one subject in two casings, repeated supporting facts.
    for (s, p, o) in [
        ("Margarethe Solberg", "works_at", "Nordvik"),
        ("margarethe solberg", "works_at", "Nordvik"),
        ("Margarethe Solberg", "role", "auditor"),
        ("margarethe solberg", "role", "auditor"),
    ] {
        mem.remember(format!("{s} {p} {o}."))
            .in_namespace(ns.clone())
            .with_facts(vec![fact(s, p, o)])
            .skip_extraction()
            .await?;
    }

    let summary = mem.dream().in_namespace(ns.clone()).execute().await?;
    println!(
        "\ndream on merge-friendly data: merged={} archived={}",
        summary.cross_episode_merged, summary.facts_archived
    );

    // ⚠️ TRIPWIRE. These assert the CURRENT limitation (TD-250). If either
    // starts failing, the gap has been closed and the map above is out of date —
    // update this example rather than deleting the assertion.
    assert_eq!(
        summary.cross_episode_merged, 0,
        "TD-250 tripwire: a merge became reachable offline. That is good news — \
         `unmerge` can now be exercised end to end. Update the map in this file."
    );
    println!("  → no merge, so `unmerge` has nothing to reverse (TD-250).");
    println!("  → `facts_archived` is a COUNT; no public call returns archived ids,");
    println!("     so `restore_archived_fact` cannot be given one (TD-250).");

    println!("\nThree pairs work end to end. Two are public in one direction only.");
    println!("Assume every reversal has a reachable handle and you will find out");
    println!("otherwise halfway through building an undo feature.");

    mem.close().await?;
    Ok(())
}

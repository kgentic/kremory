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
//!   dream() archive →    undo(mut_id)                 list_mutations()
//!   dream() merge   →    unmerge(mut_id)              list_mutations()
//! ```
//!
//! All five are exercised below, end to end.
//!
//! **The merge one carries a trap worth reading before you use it.** This example
//! used to assert that merges were UNREACHABLE from the public API, on the evidence
//! that `dream()` reported `cross_episode_merged: 0` over a deliberately
//! merge-friendly corpus. That assertion was wrong, and wrong in an instructive way:
//! `dream()` defaults to `CrossEpisodeMode::Shadow`, where merges are DECIDED and
//! never committed, so `cross_episode_merged` is `0` by construction. The decision
//! was sitting in `cross_episode_would_merge` the whole time. A tripwire watching a
//! number the default configuration pins at zero can never fire (TD-250).
//!
//! `restore_archived_fact` used to be listed here too, for a sharper reason: it
//! takes an `archived_fact_id` and `DreamSummary` reports `facts_archived` as a
//! COUNT, so nothing named WHICH facts a dream retired. Fixed 2026-09-10 — the
//! archive op logs its mutation, so `list_mutations()` names them and `undo()`
//! reverses them, which is what section 4 below now demonstrates.

use std::sync::Arc;

use chrono::Utc;
use kremory::facade::MutationKind;
use kremory::memory::types::CrossEpisodeMode;
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
    predicates_of(mem, ns, "ingrid").await
}

async fn predicates_of(mem: &Memory, ns: &Namespace, subject: &str) -> anyhow::Result<Vec<String>> {
    let mut v: Vec<String> = mem
        .recall(subject)
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

/// The `fact_id` of one of a subject's facts, by predicate. `RetrievedFact` only
/// started carrying `fact_id` recently (TD-244) — before that, `supersede` had the
/// same unreachable-handle problem this file is about.
async fn fact_id_for(
    mem: &Memory,
    ns: &Namespace,
    (subject, predicate): (&str, &str),
) -> anyhow::Result<i64> {
    mem.recall(subject)
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == predicate)
        .and_then(|f| f.fact_id)
        .ok_or_else(|| anyhow::anyhow!("expected a `{predicate}` fact for {subject}"))
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

    // ── 3. dream() merge → unmerge ──────────────────────────────────────────
    //
    // Two spellings of one subject, four episodes, and — the part that actually
    // decides it — an IDENTICAL (predicate, object) fact shared between them.
    //
    // The merge gate is structural corroboration, not string similarity: two
    // entities must share a neighbour or an identical fact, otherwise they are
    // treated as HOMONYMS and left alone. Two people really can be called
    // Margarethe Solberg, and merging them would silently corrupt both.
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

    // Default dream: SHADOW. It decides, and commits nothing.
    let shadow = mem.dream().in_namespace(ns.clone()).execute().await?;
    println!(
        "\ndream (default = shadow): would_merge={} merged={}",
        shadow.cross_episode_would_merge, shadow.cross_episode_merged
    );
    assert_eq!(
        shadow.cross_episode_merged, 0,
        "shadow mode must not commit a merge"
    );
    println!("  → decided, not applied. Read would_merge here, NEVER merged.");

    // Apply: the same decision, committed and logged.
    let applied = mem
        .dream()
        .in_namespace(ns.clone())
        .cross_episode(CrossEpisodeMode::Apply)
        .execute()
        .await?;
    println!("dream (Apply)            : merged={}", applied.cross_episode_merged);
    assert_eq!(
        applied.cross_episode_merged, 1,
        "Apply must commit the merge shadow already approved"
    );

    let merges = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .kind(MutationKind::EntityMerge)
        .await?;
    println!("merge mutation           : {}", merges[0].summary);

    let unmerged = mem.unmerge(merges[0].mutation_id).execute().await?;
    println!(
        "merge → unmerge          : restored '{}' ({} facts re-pointed)",
        unmerged.restored_entity, unmerged.facts_repointed
    );
    assert_eq!(
        unmerged.restored_entity, "margarethe solberg",
        "the loser must come back"
    );

    // ── 4. dream() archive → undo ───────────────────────────────────────────
    //
    // A dream retires facts that have been closed longer than its grace window.
    // That is a destructive write, and until 2026-09-10 it reported only a COUNT
    // — so a consumer could see that three facts had gone and never learn which.
    // The archival is now logged, which makes it visible AND reversible through
    // the same list_mutations/undo pair as every other mutation.
    //
    // `archive_grace_days: 0` so a fact closed a moment ago is eligible; the
    // default is 90 days, which no example can wait for.
    let ns_arch = Namespace::new("archival");
    for (s, p, o) in [
        ("Nils Haugerud", "berth", "quay-9"),
        ("Nils Haugerud", "role", "harbourmaster"),
    ] {
        mem.remember(format!("{s} {p} {o}."))
            .in_namespace(ns_arch.clone())
            .with_facts(vec![fact(s, p, o)])
            .skip_extraction()
            .await?;
    }
    // Close ONE of them. The other stays live, so retiring the closed one does
    // not strand its subject — the archive op refuses to leave an entity with no
    // facts at all, which is why the second fact is here.
    let berth_id = fact_id_for(&mem, &ns_arch, ("Nils Haugerud", "berth")).await?;
    mem.supersede(berth_id)
        .in_namespace(ns_arch.clone())
        .at(Utc::now())
        .close_now()
        .execute()
        .await?;

    // `DreamOpts` is `#[non_exhaustive]`, so the struct-literal form
    // (`DreamOpts { archive_grace_days: Some(0), ..Default::default() }`) does NOT
    // compile outside the crate. Take the default and assign the field.
    let mut opts = kremory::memory::types::DreamOpts::default();
    opts.archive_grace_days = Some(0);
    let arch_summary = mem
        .dream()
        .in_namespace(ns_arch.clone())
        .with_opts(opts)
        .execute()
        .await?;
    println!("\ndream on a closed fact: archived={}", arch_summary.facts_archived);

    let archived = mem
        .list_mutations()
        .in_namespace(ns_arch.clone())
        .kind(kremory::MutationKind::FactArchive)
        .await?;
    println!("  archived, by name:");
    for record in &archived {
        println!("    #{} — {}", record.mutation_id, record.summary);
    }
    assert!(
        !archived.is_empty(),
        "a dream that archived {} fact(s) must name them — a COUNT is not a handle",
        arch_summary.facts_archived
    );

    mem.undo(archived[0].mutation_id).execute().await?;

    // Un-archiving restores the fact EXACTLY as it was archived — which means
    // still CLOSED, because being closed is what made it archival-eligible. So it
    // is back in the graph and still absent from a present-tense recall. That is
    // two separate reversals, not one: `undo` returns the row, `unsupersede`
    // re-opens it. An undo that quietly re-opened the fact as well would be
    // inventing a decision the caller never made.
    let still_closed = predicates_of(&mem, &ns_arch, "Nils Haugerud").await?;
    assert!(
        !still_closed.iter().any(|p| p == "berth"),
        "un-archiving must not silently re-open a closed fact; got {still_closed:?}"
    );
    mem.unsupersede(berth_id).execute().await?;
    let back = predicates_of(&mem, &ns_arch, "Nils Haugerud").await?;
    println!("archive → undo     : {back:?} (after re-opening it too)");
    assert!(
        back.iter().any(|p| p == "berth"),
        "undo + unsupersede should put the fact back in the present; got {back:?}"
    );

    println!("\nAll five pairs work end to end — but two of them needed something");
    println!("non-obvious first: a merge only commits under .cross_episode(Apply),");
    println!("and an archived fact is only nameable because the archival is logged.");
    println!("Check that a reversal has a reachable handle BEFORE you design around it.");

    mem.close().await?;
    Ok(())
}

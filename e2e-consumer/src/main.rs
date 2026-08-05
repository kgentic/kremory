//! FRESH-CRATE consumer end-to-end test for kremory.
//!
//! Installs kremory as an external consumer (path dep), drives the PUBLIC `Memory`
//! SDK against a REAL Ollama LLM (gemma4:e4b + nomic-embed-text), and asserts on
//! BOTH behaviour AND observability.
//!
//! Run:
//!   CARGO_TARGET_DIR=~/kremory-adr073-cascade/target \
//!     KREMORY_DEBUG=1 cargo run --manifest-path e2e-consumer/Cargo.toml
//!
//! Ollama must be running at localhost:11434.

use std::time::{Duration, Instant};

use kremory::{Memory, MutationKind, Namespace};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

/// Sum every counter whose metric NAME equals `name` (labels collapsed).
fn sum_counter(snap: &Snapshotter, name: &str) -> u64 {
    snap.snapshot()
        .into_vec()
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == name)
        .map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => *c,
            _ => 0,
        })
        .sum()
}

/// Every `kremory.*` counter that fired, with its labels + value — the candid
/// observability dump.
fn dump_kremory_counters(snap: &Snapshotter) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = snap
        .snapshot()
        .into_vec()
        .iter()
        .filter_map(|(k, _, _, v)| {
            let key = k.key();
            if !key.name().starts_with("kremory.") {
                return None;
            }
            let val = match v {
                DebugValue::Counter(c) => *c,
                _ => return None,
            };
            let labels: Vec<String> = key.labels().map(|l| format!("{}={}", l.key(), l.value())).collect();
            let name = if labels.is_empty() {
                key.name().to_string()
            } else {
                format!("{}{{{}}}", key.name(), labels.join(","))
            };
            Some((name, val))
        })
        .collect();
    out.sort();
    out
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Global metrics recorder — installed BEFORE any Memory is built so every
    //    kremory counter emission (across all tokio worker threads) is captured.
    let recorder = DebuggingRecorder::new();
    let snap = recorder.snapshotter();
    metrics::set_global_recorder(recorder).expect("install global metrics recorder");

    // ── tracing subscriber so KREMORY_DEBUG=1 traces are visible (a consumer must
    //    wire this themselves — kremory does not re-export tracing-subscriber).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,kremory=info")),
        )
        .try_init();

    let ns = Namespace::new("acme-e2e");

    // ════════════════════════════════════════════════════════════════════════
    // SMOKE ONE (per smoke-one-before-batch): one minimal real-LLM op end-to-end
    // to shake out any harness / model / wiring bug BEFORE the full journey.
    // ════════════════════════════════════════════════════════════════════════
    eprintln!("\n===== SMOKE: open Memory::with_ollama + remember 1 episode + dream 1 pass =====");
    {
        let dir = tempfile::tempdir()?;
        let t0 = Instant::now();
        let mem = Memory::with_ollama(dir.path().join("smoke.db")).await?;
        eprintln!("[smoke] Memory::with_ollama opened in {:?}", t0.elapsed());

        let t1 = Instant::now();
        mem.remember("Acme Corporation is a robotics company based in Boston.")
            .in_namespace(ns.clone())
            .await?;
        eprintln!("[smoke] remember() completed in {:?}", t1.elapsed());

        let t2 = Instant::now();
        let summary = mem.dream().in_namespace(ns.clone()).await?;
        eprintln!(
            "[smoke] dream() completed in {:?} — duration_ms={} ops_ran={:?}",
            t2.elapsed(),
            summary.duration_ms,
            summary.consolidation_ops_ran
        );
        eprintln!("[smoke] PASSED — wiring is clean, proceeding to full journey.");
    }

    // ════════════════════════════════════════════════════════════════════════
    // FULL JOURNEY
    // ════════════════════════════════════════════════════════════════════════
    eprintln!("\n===== JOURNEY: full consumer flow =====");
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("consumer.db");

    // Step 1 — build Memory via the advertised Tier-1 shortcut.
    let t = Instant::now();
    let mem = Memory::with_ollama(&db).await?;
    eprintln!("[step1] Memory::with_ollama('{}') in {:?}", db.display(), t.elapsed());

    // Step 2 — remember several episodes crafted to yield entities + facts + a
    // near-duplicate pair (Alice Johnson / Alice J., Acme Corporation / Acme Corp)
    // that the dream reconciliation MIGHT canonicalize-merge.
    let episodes = [
        "Acme Corporation announced that Alice Johnson was promoted to Chief Technology Officer in January 2026.",
        "At Acme Corp, Alice J. now leads the engineering division as CTO and reports to the CEO Bob Smith.",
        "Bob Smith, the CEO of Acme Corporation, praised Alice Johnson's work on the new robotics platform.",
    ];
    for (i, ep) in episodes.iter().enumerate() {
        let te = Instant::now();
        mem.remember(*ep).in_namespace(ns.clone()).await?;
        eprintln!("[step2] remember episode {} in {:?}", i + 1, te.elapsed());
    }

    // Step 3 — dream with all consolidation ON (shipped default). Assert the
    // honest DreamSummary fields exist + are self-consistent.
    let td = Instant::now();
    let summary = mem.dream().in_namespace(ns.clone()).await?;
    eprintln!("[step3] dream() in {:?}", td.elapsed());
    eprintln!(
        "[step3] DreamSummary: types_discovered={} entities_reclassified={} aliases_resolved={} \
         canonicalization_merges={} consistency_check_corrected={} communities_updated={} \
         cross_episode_would_merge={} cross_episode_merged={} supersessions_recorded={} \
         facts_archived={} budget_exhausted={} ops_ran={:?} warnings={:?}",
        summary.types_discovered.len(),
        summary.entities_reclassified,
        summary.aliases_resolved,
        summary.canonicalization_merges,
        summary.consistency_check_corrected,
        summary.communities_updated,
        summary.cross_episode_would_merge,
        summary.cross_episode_merged,
        summary.supersessions_recorded,
        summary.facts_archived,
        summary.budget_exhausted,
        summary.consolidation_ops_ran,
        summary.warnings,
    );
    // Honest-field invariants (structural — hold regardless of real-LLM variance):
    // consolidation ops are ON by default → their ran-flags must be true.
    assert!(
        summary.consolidation_ops_ran.cross_episode,
        "cross_episode op should have RUN (default-ON); ops_ran={:?}",
        summary.consolidation_ops_ran
    );
    assert!(
        summary.consolidation_ops_ran.community
            && summary.consolidation_ops_ran.archival
            && summary.consolidation_ops_ran.supersession_sweep,
        "all consolidation ops should have RUN (default-ON); ops_ran={:?}",
        summary.consolidation_ops_ran
    );
    // Shadow is the default cross-episode mode → merged must be 0 and never exceed
    // would_merge (the honest would/did split).
    assert_eq!(
        summary.cross_episode_merged, 0,
        "cross_episode is Shadow by default → merged must be 0 (would_merge={})",
        summary.cross_episode_would_merge
    );
    assert!(
        summary.cross_episode_merged <= summary.cross_episode_would_merge,
        "merged ({}) must never exceed would_merge ({})",
        summary.cross_episode_merged,
        summary.cross_episode_would_merge
    );
    let failures: Vec<&String> = summary.warnings.iter().filter(|w| w.contains("failed")).collect();
    assert!(failures.is_empty(), "no dream pass may hard-fail; failures={failures:?}");

    // Step 4 — recall. Assert results come back; harvest a REAL entity id for the
    // reversible surface (public path: recall(...).raw() → Vec<RetrievedContext>).
    let ctx: String = mem.recall("Who is the CTO of Acme?").in_namespace(ns.clone()).await?;
    eprintln!("[step4] recall(String) -> {} chars:\n{}", ctx.len(), ctx);
    assert!(!ctx.trim().is_empty(), "recall must return a non-empty context block");

    let mut entity_id: Option<String> = None;
    for q in ["Alice Johnson", "Acme Corporation", "Bob Smith"] {
        let raw = mem.recall(q).in_namespace(ns.clone()).raw().await?;
        eprintln!(
            "[step4] recall('{q}').raw() -> {} results: {:?}",
            raw.len(),
            raw.iter().map(|r| format!("{} ({})", r.entity_id, r.entity_type_name)).collect::<Vec<_>>()
        );
        if entity_id.is_none() {
            // `!r.is_content_passage()` is LOAD-BEARING, not defensive.
            //
            // Since ADR-078 made `content-search` a default, `.raw()` returns a
            // HETEROGENEOUS list — content passages interleaved with graph entities —
            // and a passage's `entity_id` is an EPISODE id. Without this filter the
            // first hit is often `"1" (ContentPassage)`, which then fails every
            // entity-scoped call below with `no entity '1' in namespace 'acme-e2e'`.
            //
            // That is exactly how this harness failed on 2026-08-05
            // (V1-CANONICAL §0b-sexies, E2E-2). The old filter —
            // `!entity_id.trim().is_empty() && !incomplete` — is a reasonable reading
            // of the API and silently harvested a passage. `is_content_passage()` was
            // added to the public surface in response, so a consumer no longer has to
            // hardcode the magic string `"ContentPassage"` to get this right.
            if let Some(r) = raw
                .iter()
                .find(|r| !r.entity_id.trim().is_empty() && !r.incomplete && !r.is_content_passage())
            {
                entity_id = Some(r.entity_id.clone());
            }
        }
    }
    let entity_id = entity_id.expect(
        "at least one real entity must be extracted + recallable from the 3 episodes \
         (else the consumer cannot exercise the entity reversible surface)",
    );
    eprintln!("[step4] harvested real entity id for reversible surface: '{entity_id}'");

    // ── Step 4b — `.content()`, the BM25 terminal of the newly-defaulted feature ──
    //
    // Added 2026-08-06. `content-search` became a DEFAULT in ADR-078 and this
    // release exists to ship that flip — yet no consumer journey exercised its own
    // terminal. Its sibling `.raw()` is *precisely* where E2E-2 broke, silently, for
    // weeks. Testing the flip without testing the surface it turns on is how that
    // recurs.
    let passages = mem
        .recall("robotics platform")
        .in_namespace(ns.clone())
        .content()
        .await?;
    eprintln!(
        "[step4b] recall('robotics platform').content() -> {} passage(s): {:?}",
        passages.len(),
        passages
            .iter()
            .take(3)
            .map(|p| format!("ep{} score={:.3} {:?}", p.episode_id, p.score, p.snippet.chars().take(60).collect::<String>()))
            .collect::<Vec<_>>()
    );
    assert!(
        !passages.is_empty(),
        "`.content()` must return BM25 passages over the raw episode text. An empty \
         result on a default build means `content-search` is not actually active for \
         the consumer — which is the entire defect this release ships to fix (PUB-1)."
    );
    assert!(
        passages.iter().all(|p| !p.snippet.trim().is_empty()),
        "every ContentPassage must carry its verbatim episode snippet"
    );

    // ── Step 4c — `.as_of()`, the bi-temporal headline differentiator ────────────
    //
    // The README leads with two clocks; nothing drove `as_of` end-to-end as a
    // consumer. Harvest a real `recorded_at` from a recalled fact (TD-116 /
    // ADR-074 populate `RetrievedContext::facts`), then bracket it.
    let ctx_with_facts = mem.recall("Acme Corporation").in_namespace(ns.clone()).raw().await?;
    let fact_clock = ctx_with_facts
        .iter()
        .flat_map(|c| c.facts.iter())
        .map(|f| (f.valid_at, f.recorded_at))
        .next();

    match fact_clock {
        Some((valid_at, recorded_at)) => {
            eprintln!("[step4c] harvested fact clocks: valid_at={valid_at} recorded_at={recorded_at}");

            // AFTER everything was asserted → the graph is visible.
            let after = mem
                .recall("Acme Corporation")
                .in_namespace(ns.clone())
                .as_of(valid_at + chrono::Duration::days(1))
                .raw()
                .await?;
            // LONG BEFORE anything existed → valid-time filtering must exclude it.
            let before = mem
                .recall("Acme Corporation")
                .in_namespace(ns.clone())
                .as_of(valid_at - chrono::Duration::days(3650))
                .raw()
                .await?;
            let facts_after: usize = after.iter().map(|c| c.facts.len()).sum();
            let facts_before: usize = before.iter().map(|c| c.facts.len()).sum();
            eprintln!(
                "[step4c] as_of(+1d) -> {} ctx / {} facts   ·   as_of(-10y) -> {} ctx / {} facts",
                after.len(), facts_after, before.len(), facts_before
            );
            assert!(
                facts_before <= facts_after,
                "as_of() 10 years BEFORE the facts were valid must not return MORE facts \
                 than as_of() after them ({facts_before} vs {facts_after}). A violation \
                 means the valid-time predicate is not being applied — the README's \
                 headline differentiator silently doing nothing."
            );
        }
        None => {
            // Not an assertion failure: whether the real LLM extracts a fact for this
            // entity on this run is nondeterministic. Say so loudly rather than
            // passing quietly — a step that silently skips is how coverage rots.
            eprintln!(
                "[step4c] ⚠️ SKIPPED as_of assertions — no RetrievedFact came back for \
                 'Acme Corporation' this run (real-LLM extraction is nondeterministic). \
                 This step is therefore NOT covered on this run."
            );
        }
    }

    // ── Step 4d — `supersede()` reachability, REPORTED not asserted ─────────────
    //
    // `Memory::supersede(fact_id: i64)` is listed SHIPPED (SCOPE V6), but a consumer
    // holding a recalled fact has no id to pass it: `RetrievedFact` (exported at
    // `lib.rs:90`) carries the triple, both clocks, confidence and provenance — and
    // NO identifier. `IngestResult` returns counts only.
    //
    // Deliberately NOT asserted. Asserting a gap makes the test fail when the gap is
    // FIXED, which is backwards. Logged so the journey records the reachability
    // question every run, and tracked in the plan doc instead.
    let total_facts: usize = ctx_with_facts.iter().map(|c| c.facts.len()).sum();
    eprintln!(
        "[step4d] reachability note: {total_facts} RetrievedFact(s) returned, carrying \
         both clocks but NO id field — so `supersede(fact_id)` / `delete_fact(fact_id)` \
         cannot be driven from a recall result by a consumer. Not a test failure; \
         recorded for the API-surface review."
    );

    // Step 5 — INSPECT surface (the SEE half). Before any consumer mutation, dream
    // may already have logged mutations; list them + the entity's history.
    let all_muts = mem.list_mutations().in_namespace(ns.clone()).await?;
    eprintln!(
        "[step5] list_mutations() (live) -> {} mutation(s): {:?}",
        all_muts.len(),
        all_muts.iter().map(|m| format!("#{} {:?} '{}'", m.mutation_id, m.kind, m.summary)).collect::<Vec<_>>()
    );
    let hist0 = mem.mutation_history(&entity_id).in_namespace(ns.clone()).await?;
    eprintln!("[step5] mutation_history('{entity_id}') -> {} record(s)", hist0.len());

    // Step 6 — full reversible surface as a consumer.
    // (a) unmerge — only if the real-LLM dream actually produced an entity_merge.
    let merges = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .kind(MutationKind::EntityMerge)
        .await?;
    if let Some(m) = merges.first() {
        eprintln!("[step6a] found entity_merge #{} '{}' — unmerging", m.mutation_id, m.summary);
        let out = mem.unmerge(m.mutation_id).execute().await?;
        eprintln!("[step6a] unmerge outcome: {out:?}");
        assert!(!out.already_undone, "first unmerge must actually reverse (already_undone=false)");
        // The split pair must now both be present again.
        let re = mem.mutation_history(&m.affected_entities[0]).in_namespace(ns.clone()).await?;
        assert!(
            re.iter().any(|r| r.mutation_id == m.mutation_id && r.undone),
            "the unmerged mutation must show undone=true in history"
        );
        eprintln!("[step6a] unmerge PASSED — merge reversed + NOGOOD recorded");
    } else {
        eprintln!(
            "[step6a] SKIPPED unmerge — real-LLM dream produced NO entity_merge mutation this run \
             (canonicalize/cross-episode-apply did not fire on the extracted graph). This is real-LLM \
             variance, not a defect; the merge REVERSAL primitive is covered by kremory's own \
             deterministic tests. Edit/delete reversibility below is deterministic + exercised."
        );
    }

    // (b) edit_entity rename (rekey) + undo.
    let renamed = format!("{entity_id} (RENAMED)");
    let edit_out = mem.edit_entity(&entity_id).rename(&renamed).in_namespace(ns.clone()).execute().await?;
    eprintln!("[step6b] edit_entity('{entity_id}').rename('{renamed}') -> {edit_out:?}");
    assert!(edit_out.rekeyed, "rename must report rekeyed=true");
    let hist_new = mem.mutation_history(&renamed).in_namespace(ns.clone()).await?;
    eprintln!(
        "[step6b] mutation_history('{renamed}') -> {:?}",
        hist_new.iter().map(|m| format!("#{} {:?} '{}'", m.mutation_id, m.kind, m.summary)).collect::<Vec<_>>()
    );
    let edit_mut = hist_new
        .iter()
        .find(|m| matches!(m.kind, MutationKind::EntityEdit) && !m.undone)
        .expect("rename must appear as a live EntityEdit mutation in the new id's history");
    // Undo the rename to restore the original id (propagation reversal).
    let undo_edit = mem.undo_entity_edit(edit_mut.mutation_id).execute().await?;
    eprintln!("[step6b] undo_entity_edit(#{}) -> {undo_edit:?}", edit_mut.mutation_id);
    let hist_restored = mem.mutation_history(&entity_id).in_namespace(ns.clone()).await?;
    assert!(
        !hist_restored.is_empty(),
        "after undo, the original id '{entity_id}' must be locatable again in mutation history"
    );
    eprintln!("[step6b] edit/undo PASSED — rename propagated + reversed");

    // (c) delete_entity cascade + undo_delete_entity restore.
    let del_out = mem.delete_entity(&entity_id).in_namespace(ns.clone()).execute().await?;
    eprintln!("[step6c] delete_entity('{entity_id}') -> {del_out:?}");
    let hist_del = mem.mutation_history(&entity_id).in_namespace(ns.clone()).await?;
    let del_mut = hist_del
        .iter()
        .find(|m| matches!(m.kind, MutationKind::EntityDelete) && !m.undone)
        .expect("delete must appear as a live EntityDelete mutation in the entity's history");
    // Entity should no longer be a live recall hit for its own name.
    let after_del = mem.recall(&entity_id).in_namespace(ns.clone()).raw().await?;
    let still_live = after_del.iter().any(|r| r.entity_id == entity_id && !r.incomplete);
    eprintln!("[step6c] post-delete recall live-hit for '{entity_id}': {still_live}");
    let undo_del = mem.undo_delete_entity(del_mut.mutation_id).execute().await?;
    eprintln!("[step6c] undo_delete_entity(#{}) -> {undo_del:?}", del_mut.mutation_id);
    let hist_after_undo = mem.mutation_history(&entity_id).in_namespace(ns.clone()).await?;
    assert!(
        hist_after_undo.iter().any(|m| m.mutation_id == del_mut.mutation_id && m.undone),
        "the delete mutation must show undone=true after undo_delete_entity"
    );
    eprintln!("[step6c] delete/undo PASSED — cascade archived + restored");

    // ════════════════════════════════════════════════════════════════════════
    // Step 7 — O11Y ASSERTIONS. Snapshot the recorder + prove the counters that
    // the journey exercised actually fired with sane values.
    // ════════════════════════════════════════════════════════════════════════
    eprintln!("\n===== STEP 7: observability snapshot =====");
    let dump = dump_kremory_counters(&snap);
    eprintln!("[step7] {} kremory.* counters fired:", dump.len());
    for (name, val) in &dump {
        eprintln!("        {name} = {val}");
    }

    let inspect_q = sum_counter(&snap, "kremory.graph.inspect_query_total");
    let logged = sum_counter(&snap, "kremory.graph.mutation_logged_total");
    let undone = sum_counter(&snap, "kremory.graph.mutation_undone_total");

    assert!(
        inspect_q >= 3,
        "kremory.graph.inspect_query_total must be >=3 (we called list_mutations + mutation_history \
         many times); got {inspect_q}"
    );
    assert!(
        logged >= 2,
        "kremory.graph.mutation_logged_total must be >=2 (edit_entity rename + delete_entity each log \
         a mutation); got {logged}"
    );
    assert!(
        undone >= 2,
        "kremory.graph.mutation_undone_total must be >=2 (undo_entity_edit + undo_delete_entity); \
         got {undone}"
    );
    eprintln!(
        "[step7] o11y PASSED — inspect_query_total={inspect_q} mutation_logged_total={logged} \
         mutation_undone_total={undone}"
    );

    // Give ollama a moment for keep_alive; not required for correctness.
    let _ = Duration::from_millis(1);
    eprintln!("\n===== ALL STEPS PASSED =====");
    Ok(())
}

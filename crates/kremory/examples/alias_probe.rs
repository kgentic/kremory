//! Diagnostic probe: run the REAL dream aliases pass against a REAL corpus DB.
//!
//! Written 2026-08-11 for `.ai-docs/plans/dream-ingest-correctness-investigation-2026-08-11.md` §1.
//!
//! ## Why this exists
//!
//! `.context/full-corpus.db` holds 42 `potential_alias` facts, ALL pending, ALL
//! same-namespace — including `3 july 2023 -> 3 july 2023` (identical strings).
//! The DB alone cannot distinguish "the aliases pass ran and retained all 42"
//! from "the pass never ran". The passing fixture test
//! (`it/dream_phase2_deterministic_passes.rs::dream_resolves_planted_potential_alias`)
//! plants a unit-vector embedding in a SINGLE namespace, so it cannot observe
//! either failure mode. This probe drives the production function
//! (`core::disambiguation::resolve_pending_aliases`) against real data.
//!
//! ## Pre-registered predictions (Rule 43 — written BEFORE the first run)
//!
//! * **P1** — the pass is reachable and effective; it simply never ran on this
//!   corpus. Running it now MUST resolve a non-zero count, and in particular the
//!   four `3 july 2023 -> 3 july 2023` rows MUST resolve (cosine(x,x) = 0
//!   distance ⇒ similarity 1.0 ⇒ >= `L4_MERGE_THRESHOLD` 0.95).
//! * **P2 (the reversal)** — if the pass resolves **0**, then "never ran" is
//!   REFUTED as the explanation. The pass would be reachable but ineffective,
//!   and the defect lives inside the function, not in the caller.
//! * **P3** — the re-similarity SQL (`disambiguation/mod.rs:669-671`) filters on
//!   `entities.id` alone with NO `group_id` predicate, while the `facts` FK
//!   declares `REFERENCES entities(id, group_id)`. For any alias endpoint whose
//!   name exists in >1 namespace (`session 4` exists in 10), the unscoped join
//!   returns multiple rows and `rows.next()` takes an arbitrary one. If P3 holds,
//!   the unscoped and namespace-scoped similarities will DIFFER for at least one
//!   pair.
//!
//! ## Usage (operates on a COPY — never point this at the live corpus)
//!
//! ```text
//! cp .context/full-corpus.db .context/alias-probe.db
//! cargo run --release -p kremory --example alias_probe --features content-search \
//!     -- .context/alias-probe.db [--apply]
//! ```
//!
//! Without `--apply` the probe is READ-ONLY: it reports similarities and stops.
//! With `--apply` it additionally calls the production pass and re-reads state.

use kremory::core::disambiguation::{
    resolve_pending_aliases, L4_MERGE_THRESHOLD, L4_REVOKE_THRESHOLD,
};
use kremory::core::schema::TemporalGraph;

/// nomic-embed-text dimensionality (the corpus embedder).
const DIM: usize = 768;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let db_path = argv
        .first()
        .cloned()
        .ok_or("usage: alias_probe <db-path> [--apply] [--delete-entity <id> <group_id>]")?;
    let apply = argv.iter().any(|a| a == "--apply");

    // `--delete-entity <id> <group_id>` — drop one entity row through libsql.
    //
    // Needed because the foreign `sqlite3` binary CANNOT delete from `entities`:
    // the table carries a `libsql_vector_idx` on `embedding` and the CLI reports
    // `unknown function: libsql_vector_idx()`. Used by the TD-203 §3 recall-value
    // measurement to complete an emulated L5 merge (repoint facts, then remove the
    // loser) so the loser does not linger as a 0-fact entity occupying a top-k slot.
    if let Some(i) = argv.iter().position(|a| a == "--delete-entity") {
        let id = argv
            .get(i + 1)
            .ok_or("--delete-entity needs <id> <group_id>")?;
        let group = argv
            .get(i + 2)
            .ok_or("--delete-entity needs <id> <group_id>")?;
        if db_path.contains("full-corpus") {
            return Err("refusing to mutate the canonical corpus — copy it first".into());
        }
        let graph = TemporalGraph::open_with_dim(&db_path, DIM).await?;
        let n = graph
            .conn
            .execute(
                "DELETE FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![id.clone(), group.clone()],
            )
            .await?;
        println!("deleted {n} entity row(s) for id={id} group_id={group}");
        return Ok(());
    }

    if db_path.contains("full-corpus") {
        return Err("refusing to run against the canonical corpus — copy it first".into());
    }

    let graph = TemporalGraph::open_with_dim(&db_path, DIM).await?;

    // ── Namespaces present, derived from the data (not hardcoded) ────────────
    let mut groups: Vec<String> = Vec::new();
    let mut rows = graph
        .conn
        .query(
            "SELECT DISTINCT subject_group_id FROM facts \
             WHERE predicate='potential_alias' ORDER BY 1",
            (),
        )
        .await?;
    while let Some(r) = rows.next().await? {
        groups.push(r.get::<String>(0)?);
    }
    println!("namespaces with pending aliases: {}", groups.len());

    // ── Phase A — BEFORE state ───────────────────────────────────────────────
    let before = pending_count(&graph).await?;
    println!("pending potential_alias BEFORE: {before}\n");

    // ── Phase B — similarity, computed BOTH ways (tests P3) ──────────────────
    println!(
        "{:<26} {:<26} {:<22} {:>10} {:>10} {:>7} {:>9}",
        "subject", "object", "namespace", "sim(prod)", "sim(ns)", "rows", "verdict"
    );
    let mut divergences = 0usize;
    let mut would_merge = 0usize;
    let mut would_revoke = 0usize;

    for g in &groups {
        for fact in graph.get_alias_facts_in_group(g).await? {
            let Some(object_id) = fact.object_id.clone() else {
                continue;
            };
            let group = fact.group_id.clone().unwrap_or_default();

            // EXACT production query — disambiguation/mod.rs:669-671, verbatim.
            let sim_prod = query_sim(
                &graph,
                "SELECT vector_distance_cos(a.embedding, b.embedding) \
                 FROM entities a, entities b \
                 WHERE a.id = ?1 AND b.id = ?2",
                &[fact.subject_id.clone(), object_id.clone()],
            )
            .await?;

            // How many rows does that unscoped join actually match?
            let matched = count_rows(
                &graph,
                "SELECT count(*) FROM entities a, entities b WHERE a.id = ?1 AND b.id = ?2",
                &[fact.subject_id.clone(), object_id.clone()],
            )
            .await?;

            // Namespace-scoped variant — what the FK says the key actually is.
            let sim_ns = query_sim(
                &graph,
                "SELECT vector_distance_cos(a.embedding, b.embedding) \
                 FROM entities a, entities b \
                 WHERE a.id = ?1 AND a.group_id = ?3 AND b.id = ?2 AND b.group_id = ?3",
                &[fact.subject_id.clone(), object_id.clone(), group.clone()],
            )
            .await?;

            let verdict = match sim_prod {
                Some(s) if s >= L4_MERGE_THRESHOLD => {
                    would_merge += 1;
                    "MERGE?"
                }
                Some(s) if s < L4_REVOKE_THRESHOLD => {
                    would_revoke += 1;
                    "REVOKE?"
                }
                Some(_) => "keep",
                None => "NULL-skip",
            };

            let diverged = match (sim_prod, sim_ns) {
                (Some(a), Some(b)) => (a - b).abs() > 1e-6,
                (a, b) => a.is_some() != b.is_some(),
            };
            if diverged {
                divergences += 1;
            }

            println!(
                "{:<26} {:<26} {:<22} {:>10} {:>10} {:>7} {:>9}{}",
                truncate(&fact.subject_id, 25),
                truncate(&object_id, 25),
                truncate(group.trim_start_matches("locomo-bench-"), 21),
                fmt(sim_prod),
                fmt(sim_ns),
                matched,
                verdict,
                if diverged { "  <-- DIVERGES" } else { "" },
            );
        }
    }

    println!(
        "\nP3 check — unscoped vs namespace-scoped similarity diverged on {divergences} pair(s)"
    );
    println!("threshold-implied outcomes (before lexical gate): merge {would_merge}, revoke {would_revoke}");

    if !apply {
        println!("\n(read-only; re-run with --apply to invoke resolve_pending_aliases)");
        return Ok(());
    }

    // ── Phase C — run the PRODUCTION pass ────────────────────────────────────
    let mut total = 0usize;
    for g in &groups {
        let n = resolve_pending_aliases(&graph, g).await?;
        println!("resolve_pending_aliases({g}) -> {n}");
        total += n;
    }

    // ── Phase D — AFTER state, read back from the DB ─────────────────────────
    let after = pending_count(&graph).await?;
    println!("\nresolved (returned by pass): {total}");
    println!("pending potential_alias AFTER: {after}  (was {before})");
    println!(
        "P1 {} — pass resolved {}",
        if total > 0 { "HOLDS" } else { "REFUTED" },
        total
    );

    Ok(())
}

async fn pending_count(graph: &TemporalGraph) -> Result<i64, Box<dyn std::error::Error>> {
    count_rows(
        graph,
        "SELECT count(*) FROM facts WHERE predicate='potential_alias' \
         AND valid_to IS NULL AND invalid_at IS NULL AND expired_at IS NULL",
        &[],
    )
    .await
}

async fn count_rows(
    graph: &TemporalGraph,
    sql: &str,
    params: &[String],
) -> Result<i64, Box<dyn std::error::Error>> {
    let mut rows = graph.conn.query(sql, params.to_vec()).await?;
    let row = rows.next().await?.ok_or("no row")?;
    Ok(row.get::<i64>(0)?)
}

async fn query_sim(
    graph: &TemporalGraph,
    sql: &str,
    params: &[String],
) -> Result<Option<f32>, Box<dyn std::error::Error>> {
    let mut rows = graph.conn.query(sql, params.to_vec()).await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let distance: Option<f64> = row.get(0)?;
    // Mirrors disambiguation/mod.rs:694 exactly.
    Ok(distance.map(|d| (1.0_f32 - d as f32).clamp(0.0, 1.0)))
}

fn fmt(v: Option<f32>) -> String {
    v.map_or_else(|| "NULL".to_owned(), |x| format!("{x:.4}"))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

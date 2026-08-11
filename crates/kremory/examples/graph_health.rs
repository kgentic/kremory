//! Graph-health property checker — asserts properties of the OUTPUT of a real
//! ingest + dream run, on any kremory database, for free.
//!
//! ## Why this exists
//!
//! 2026-08-11 established that the project had been measuring ingest **stability**
//! (TD-186a: ~6% run-to-run variance) and reporting it as ingest **correctness**.
//! A pipeline can be consistently wrong. Five minutes of looking at the real
//! database found four defects that a variance number could never surface.
//!
//! This is the missing instrument. Every check below asserts a property of the
//! GRAPH, not of the spread, and every one of them would have been RED on
//! `.context/full-corpus.db` before TD-203/TD-206.
//!
//! ## What it can and cannot tell you
//!
//! * **CAN** validate pipeline MECHANISM — did dream run every pass, did aliases
//!   resolve, are there dangling references, are entities embedded, is namespace
//!   scoping intact. All of this is model-independent, so it runs for £0 against
//!   a LOCAL Ollama-built corpus.
//! * **CANNOT** validate extraction QUALITY — local `gemma4:e4b` produces 32%
//!   self-loop facts and a 71% top-subject share against Groq's 0.8% / ~8.5%
//!   (SYSTEM-PRIMER §2). Quality is a property of the model and needs the paid
//!   path. The `--quality` section below is therefore REPORTED, never asserted.
//!
//! That split is the point: mechanism is verifiable free, so verify it free
//! BEFORE spending money on a quality number.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p kremory --example graph_health --features content-search -- <db-path>
//! ```
//!
//! Exit code 0 = every HARD check passed. Non-zero = at least one failed, and the
//! failing check names the defect class it belongs to.

use kremory::core::disambiguation::{L4_MERGE_THRESHOLD, L4_REVOKE_THRESHOLD};
use kremory::core::schema::TemporalGraph;

const DIM: usize = 768;

struct Report {
    failures: Vec<String>,
}

/// One hard check — args-as-object per TD-042 (workspace clippy
/// `too_many_arguments` threshold is 3 INCLUDING `&self`, and `#[allow]` is
/// banned outside test files).
struct Check<'a> {
    name: &'a str,
    ok: bool,
    detail: String,
}

impl Report {
    fn hard(&mut self, c: Check<'_>) {
        let Check { name, ok, detail } = c;
        if ok {
            println!("  PASS  {name:<42} {detail}");
        } else {
            println!("  FAIL  {name:<42} {detail}");
            self.failures.push(name.to_owned());
        }
    }
    fn info(name: &str, detail: String) {
        println!("  ----  {name:<42} {detail}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args()
        .nth(1)
        .ok_or("usage: graph_health <db-path>")?;
    let graph = TemporalGraph::open_with_dim(&db_path, DIM).await?;
    let mut r = Report {
        failures: Vec::new(),
    };

    println!("\ngraph health — {db_path}\n");

    // ── Did the pipeline produce anything at all? ────────────────────────────
    //
    // `COUNT(*)` LIES on the vector-indexed tables (SYSTEM-PRIMER §2 — it returns
    // 0 while thousands of rows exist), so every count here materialises rows via
    // a NON-indexed column. Getting this wrong produced a whole false "the graph
    // is empty" narrative on 2026-07-22.
    let entities = count_rows(&graph, "SELECT recorded_at FROM entities").await?;
    let facts = count_rows(&graph, "SELECT recorded_at FROM facts").await?;
    let episodes = count_rows(&graph, "SELECT id FROM episodes").await?;

    println!("PRODUCTION — did ingest write anything");
    r.hard(Check {
        name: "entities_written",
        ok: entities > 0,
        detail: format!("{entities} entities"),
    });
    r.hard(Check {
        name: "facts_written",
        ok: facts > 0,
        detail: format!("{facts} facts"),
    });
    r.hard(Check {
        name: "episodes_written",
        ok: episodes > 0,
        detail: format!("{episodes} episodes"),
    });

    // ── TD-203 D1 — self-loops ───────────────────────────────────────────────
    println!("\nMECHANISM — TD-203 D1: merge must not manufacture self-loops");
    let self_loops = count_rows(
        &graph,
        "SELECT recorded_at FROM facts \
         WHERE subject_id = object_id AND object_id IS NOT NULL AND expired_at IS NULL",
    )
    .await?;
    r.hard(Check {
        name: "no_live_self_loop_facts",
        ok: self_loops == 0,
        detail: format!("{self_loops} live `X pred X` facts (pre-fix corpus: 41)"),
    });

    let self_aliases = count_rows(
        &graph,
        "SELECT recorded_at FROM facts \
         WHERE predicate = 'potential_alias' AND subject_id = object_id AND expired_at IS NULL",
    )
    .await?;
    r.hard(Check {
        name: "no_self_referential_aliases",
        ok: self_aliases == 0,
        detail: format!("{self_aliases} live `X potential_alias X` (L4 cannot emit this shape; pre-fix: 18)"),
    });

    // ── TD-203 D2 — a completed dream leaves no RESOLVABLE alias ─────────────
    //
    // "Resolvable" is the load-bearing word: a mid-band candidate is a legitimate
    // KEEP. This recomputes the similarity exactly as `resolve_pending_aliases`
    // does — namespace-scoped, per TD-203 D4 — and counts only those that WOULD
    // have merged or revoked.
    println!("\nMECHANISM — TD-203 D2: dream must leave no RESOLVABLE pending alias");
    let mut resolvable = 0usize;
    let mut pending = 0usize;
    let mut rows = graph
        .conn
        .query(
            "SELECT subject_id, object_id, group_id FROM facts \
             WHERE predicate = 'potential_alias' AND expired_at IS NULL \
               AND object_id IS NOT NULL",
            (),
        )
        .await?;
    let mut pairs: Vec<(String, String, String)> = Vec::new();
    while let Some(row) = rows.next().await? {
        pairs.push((row.get(0)?, row.get(1)?, row.get(2)?));
    }
    for (subject, object, group) in &pairs {
        pending += 1;
        let mut d = graph
            .conn
            .query(
                "SELECT vector_distance_cos(a.embedding, b.embedding) \
                 FROM entities a, entities b \
                 WHERE a.id = ?1 AND a.group_id = ?3 AND b.id = ?2 AND b.group_id = ?3",
                libsql::params![subject.clone(), object.clone(), group.clone()],
            )
            .await?;
        if let Some(row) = d.next().await? {
            if let Some(dist) = row.get::<Option<f64>>(0)? {
                let sim = (1.0_f32 - dist as f32).clamp(0.0, 1.0);
                if !(L4_REVOKE_THRESHOLD..L4_MERGE_THRESHOLD).contains(&sim) {
                    resolvable += 1;
                }
            }
        }
    }
    r.hard(Check {
        name: "no_resolvable_pending_aliases",
        ok: resolvable == 0,
        detail: format!("{resolvable} resolvable of {pending} pending (pre-fix corpus: 28 of 42)"),
    });

    // ── Referential integrity — the silent branch TD-203 D3 made loud ────────
    println!("\nMECHANISM — referential integrity of fact endpoints");
    let dangling_subj = count_rows(
        &graph,
        "SELECT f.recorded_at FROM facts f WHERE f.expired_at IS NULL AND NOT EXISTS \
         (SELECT 1 FROM entities e WHERE e.id = f.subject_id AND e.group_id = f.subject_group_id)",
    )
    .await?;
    let dangling_obj = count_rows(
        &graph,
        "SELECT f.recorded_at FROM facts f WHERE f.expired_at IS NULL AND f.object_id IS NOT NULL \
         AND NOT EXISTS \
         (SELECT 1 FROM entities e WHERE e.id = f.object_id AND e.group_id = f.object_group_id)",
    )
    .await?;
    r.hard(Check {
        name: "no_dangling_fact_endpoints",
        ok: dangling_subj + dangling_obj == 0,
        detail: format!("{dangling_subj} subject + {dangling_obj} object reference a missing entity"),
    });

    // ── Entities are embedded — a NULL embedding is invisible to dense recall ─
    println!("\nMECHANISM — entities are retrievable");
    let unembedded = count_rows(
        &graph,
        "SELECT recorded_at FROM entities WHERE embedding IS NULL",
    )
    .await?;
    r.hard(Check {
        name: "all_entities_embedded",
        ok: unembedded == 0,
        detail: format!("{unembedded} entities with a NULL embedding"),
    });

    // ── TD-206 — namespace exposure (INFO: not a defect by itself) ───────────
    let shared = count_rows(
        &graph,
        "SELECT id FROM (SELECT id FROM entities GROUP BY id HAVING count(*) > 1)",
    )
    .await?;
    Report::info(
        "names_in_multiple_namespaces",
        format!("{shared} (TD-206 blast radius if any unscoped write returns)"),
    );

    // ── QUALITY — REPORTED, NEVER ASSERTED ───────────────────────────────────
    //
    // These are properties of the MODEL, not of the pipeline. Local gemma4:e4b
    // legitimately scores far worse than Groq here, so asserting on them would
    // make a local run fail for a reason that says nothing about correctness.
    // Reported so a local-vs-paid comparison is possible at a glance.
    println!("\nQUALITY — reported only (model-dependent; do NOT gate on these)");
    let catchall = count_rows(
        &graph,
        "SELECT recorded_at FROM entities WHERE entity_type_id = 0",
    )
    .await?;
    Report::info(
        "catch_all_entity_type_share",
        format!(
            "{catchall}/{entities} = {:.1}% (Groq ~3.8%, local gemma4:e4b ~32%)",
            pct(catchall, entities)
        ),
    );
    let mut top = graph
        .conn
        .query(
            "SELECT subject_id, count(*) c FROM facts WHERE expired_at IS NULL \
             GROUP BY subject_id ORDER BY c DESC LIMIT 1",
            (),
        )
        .await?;
    if let Some(row) = top.next().await? {
        let name: String = row.get(0)?;
        let c: i64 = row.get(1)?;
        Report::info(
            "top_subject_share",
            format!(
                "`{name}` holds {c} facts = {:.1}% (Groq ~8.5%, local ~71%)",
                pct(c as usize, facts)
            ),
        );
    }

    // ── Verdict ──────────────────────────────────────────────────────────────
    println!();
    if r.failures.is_empty() {
        println!("VERDICT: PASS — every hard mechanism check holds.");
        println!("NOTE: this validates MECHANISM only. Extraction QUALITY is model-dependent");
        println!("      and is not asserted here — see the reported section above.");
        Ok(())
    } else {
        println!("VERDICT: FAIL — {} check(s) failed:", r.failures.len());
        for f in &r.failures {
            println!("  - {f}");
        }
        Err("graph health check failed".into())
    }
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

/// Count by materialising a NON-indexed column — `COUNT(*)` returns 0 on the
/// `libsql_vector_idx`-carrying tables (SYSTEM-PRIMER §2).
async fn count_rows(
    graph: &TemporalGraph,
    sql: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut rows = graph.conn.query(sql, ()).await?;
    let mut n = 0usize;
    while rows.next().await?.is_some() {
        n += 1;
    }
    Ok(n)
}

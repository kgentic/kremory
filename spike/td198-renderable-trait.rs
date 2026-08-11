// TD-198 compile spike — verify the load-bearing claim: "the core renderers
// (render_entities / render_edge_summary / render_temporal_facts) can be made
// generic over a read-only accessor trait implemented by BOTH kremory's
// `RetrievedContext` (+ `RetrievedFact` + `SourceRef`) AND kremory-mcp's
// `RetrievedContextWire` (+ `RetrievedFactWire` + `SourceRefWire`)".
//
// Minimal standalone repro — no crate deps, `rustc --edition 2021` only.
// PASS/FAIL is reported by the orchestrator after running this file.

// ── companion traits (mirror the real design: Fact / SourceRef are
// independently-shaped between the two crates too — typed DateTime vs
// pre-formatted RFC-3339 String) ───────────────────────────────────────────

trait RenderableFact {
    fn fact_text(&self) -> &str;
    fn valid_at_rfc3339(&self) -> String;
    fn invalid_at_rfc3339(&self) -> Option<String>;
}

trait RenderableSourceRef {
    fn kind_label(&self) -> &str;
    fn ref_id(&self) -> &str;
    fn occurred_at_rfc3339(&self) -> String;
}

trait RenderableContext {
    type Fact: RenderableFact;
    type SourceRef: RenderableSourceRef;

    fn entity_name(&self) -> &str;
    fn summary(&self) -> &str;
    fn namespace_group_id(&self) -> Option<&str>;
    fn facts(&self) -> &[Self::Fact];
    fn source_refs(&self) -> &[Self::SourceRef];
}

// ── ONE generic renderer body (stand-in for render_entities /
// render_temporal_facts) — this is the property under test ─────────────────

fn render_entities<T: RenderableContext>(results: &[T]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        if let Some(gid) = r.namespace_group_id() {
            out.push_str("[ns:");
            out.push_str(gid);
            out.push_str("] ");
        }
        out.push_str("## ");
        out.push_str(r.entity_name());
        out.push('\n');
        out.push_str(r.summary());
        for f in r.facts() {
            out.push_str("\n- ");
            out.push_str(f.fact_text());
            out.push_str(" (valid_at=");
            out.push_str(&f.valid_at_rfc3339());
            if let Some(inv) = f.invalid_at_rfc3339() {
                out.push_str(", invalid_at=");
                out.push_str(&inv);
            }
            out.push(')');
        }
        for sr in r.source_refs() {
            out.push_str("\n  src: ");
            out.push_str(sr.kind_label());
            out.push(':');
            out.push_str(sr.ref_id());
            out.push_str(" @ ");
            out.push_str(&sr.occurred_at_rfc3339());
        }
    }
    out
}

// ── side A: mirrors kremory::RetrievedContext (typed chrono-shaped fields) ─

struct FactA {
    fact: String,
    valid_at_epoch_secs: i64, // stand-in for chrono::DateTime<Utc>
    invalid_at_epoch_secs: Option<i64>,
}

impl RenderableFact for FactA {
    fn fact_text(&self) -> &str {
        &self.fact
    }
    fn valid_at_rfc3339(&self) -> String {
        format!("ts={}", self.valid_at_epoch_secs) // stand-in for .to_rfc3339()
    }
    fn invalid_at_rfc3339(&self) -> Option<String> {
        self.invalid_at_epoch_secs.map(|s| format!("ts={s}"))
    }
}

struct SourceRefA {
    kind: &'static str, // stand-in for source_kind_label(self.kind) -> &'static str
    id: String,
    occurred_at_epoch_secs: i64,
}

impl RenderableSourceRef for SourceRefA {
    fn kind_label(&self) -> &str {
        self.kind
    }
    fn ref_id(&self) -> &str {
        &self.id
    }
    fn occurred_at_rfc3339(&self) -> String {
        format!("ts={}", self.occurred_at_epoch_secs)
    }
}

struct ContextA {
    entity_name: String,
    summary: String,
    namespace: Option<String>, // Option<Namespace> -> .namespace.as_ref().map(|ns| ns.namespace.as_str())
    facts: Vec<FactA>,
    source_refs: Vec<SourceRefA>,
}

impl RenderableContext for ContextA {
    type Fact = FactA;
    type SourceRef = SourceRefA;

    fn entity_name(&self) -> &str {
        &self.entity_name
    }
    fn summary(&self) -> &str {
        &self.summary
    }
    fn namespace_group_id(&self) -> Option<&str> {
        self.namespace.as_deref()
    }
    fn facts(&self) -> &[FactA] {
        &self.facts
    }
    fn source_refs(&self) -> &[SourceRefA] {
        &self.source_refs
    }
}

// ── side B: mirrors kremory-mcp::RetrievedContextWire (already-String
// pre-formatted temporal fields — the divergent shape that motivates the
// two companion traits rather than one shared struct) ──────────────────────

struct FactB {
    fact: String,
    valid_at: String, // already RFC-3339 String on the wire type
    invalid_at: Option<String>,
}

impl RenderableFact for FactB {
    fn fact_text(&self) -> &str {
        &self.fact
    }
    fn valid_at_rfc3339(&self) -> String {
        self.valid_at.clone()
    }
    fn invalid_at_rfc3339(&self) -> Option<String> {
        self.invalid_at.clone()
    }
}

struct SourceRefB {
    kind: String, // wire kind is already a String, not an enum
    id: String,
    occurred_at: String,
}

impl RenderableSourceRef for SourceRefB {
    fn kind_label(&self) -> &str {
        &self.kind
    }
    fn ref_id(&self) -> &str {
        &self.id
    }
    fn occurred_at_rfc3339(&self) -> String {
        self.occurred_at.clone()
    }
}

struct ContextB {
    entity_name: String,
    summary: String,
    namespace: Option<String>,
    facts: Vec<FactB>,
    source_refs: Vec<SourceRefB>,
}

impl RenderableContext for ContextB {
    type Fact = FactB;
    type SourceRef = SourceRefB;

    fn entity_name(&self) -> &str {
        &self.entity_name
    }
    fn summary(&self) -> &str {
        &self.summary
    }
    fn namespace_group_id(&self) -> Option<&str> {
        self.namespace.as_deref()
    }
    fn facts(&self) -> &[FactB] {
        &self.facts
    }
    fn source_refs(&self) -> &[SourceRefB] {
        &self.source_refs
    }
}

fn main() {
    let a = vec![ContextA {
        entity_name: "Alice".to_string(),
        summary: "A person".to_string(),
        namespace: Some("ns-a".to_string()),
        facts: vec![FactA {
            fact: "Alice likes tea".to_string(),
            valid_at_epoch_secs: 100,
            invalid_at_epoch_secs: None,
        }],
        source_refs: vec![SourceRefA {
            kind: "chat",
            id: "ep-1".to_string(),
            occurred_at_epoch_secs: 100,
        }],
    }];

    let b = vec![ContextB {
        entity_name: "Bob".to_string(),
        summary: "Another person".to_string(),
        namespace: None,
        facts: vec![],
        source_refs: vec![SourceRefB {
            kind: "episode".to_string(),
            id: "ep-2".to_string(),
            occurred_at: "2026-01-01T00:00:00Z".to_string(),
        }],
    }];

    // The property under test: ONE generic fn body, called from both sides.
    let out_a = render_entities(&a);
    let out_b = render_entities(&b);

    assert!(out_a.contains("Alice"));
    assert!(out_a.contains("Alice likes tea"));
    assert!(out_b.contains("Bob"));
    assert!(out_b.contains("ep-2"));

    println!("PASS");
    println!("{out_a}");
    println!("---");
    println!("{out_b}");
}

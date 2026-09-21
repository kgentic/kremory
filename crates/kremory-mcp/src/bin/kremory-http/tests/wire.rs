use kremory_mcp::params::{RecallTemplateWire, RetrievedContextWire, RetrievedFactWire, SourceRefWire};

use crate::handlers::{flatten_result_content, render_prompt_block};
use crate::wire::{SearchResultKindWire, SearchResultWire};

// ─── flatten_result_content (benchmark-load-bearing) ─────────────────

fn fact_wire(fact: &str) -> RetrievedFactWire {
    RetrievedFactWire {
        // These fixtures exercise the RENDERING of facts, not their identity,
        // so a `None` handle is the honest fixture value rather than an
        // invented id.
        fact_id: None,
        fact: fact.to_string(),
        subject: "s".into(),
        predicate: "p".into(),
        object: "o".into(),
        object_is_entity: false,
        valid_at: "2026-01-01T00:00:00+00:00".into(),
        invalid_at: None,
        recorded_at: "2026-01-01T00:00:00+00:00".into(),
        expired_at: None,
        confidence: 1.0,
        source_episode_ids: vec![1],
        score: 0.5,
    }
}

fn context_wire(
    name: &str,
    summary: &str,
    facts: Vec<RetrievedFactWire>,
) -> RetrievedContextWire {
    RetrievedContextWire {
        entity_id: name.to_lowercase(),
        entity_name: name.to_string(),
        summary: summary.to_string(),
        score: 0.9,
        incomplete: false,
        entity_type_id: 0,
        entity_type_name: "Entity".into(),
        namespace: None,
        source_refs: Vec::<SourceRefWire>::new(),
        facts,
    }
}

#[test]
fn flatten_result_content_includes_summary_and_every_fact() {
    let ctx = context_wire(
        "Ada Lovelace",
        "a mathematician",
        vec![
            fact_wire("Ada Lovelace wrote the first algorithm"),
            fact_wire("Ada Lovelace collaborated with Charles Babbage"),
        ],
    );
    let content = flatten_result_content(&ctx);
    // The entity name + summary line must be present.
    assert!(
        content.contains("Ada Lovelace") && content.contains("a mathematician"),
        "flattened content must carry entity name + summary: {content:?}"
    );
    // EVERY connected fact's natural-language string must survive — the
    // benchmark substring scorer relies on this.
    assert!(
        content.contains("Ada Lovelace wrote the first algorithm"),
        "fact 1 must appear in flattened content: {content:?}"
    );
    assert!(
        content.contains("Ada Lovelace collaborated with Charles Babbage"),
        "fact 2 must appear in flattened content: {content:?}"
    );
}

#[test]
fn flatten_result_content_with_no_facts_is_the_summary_line() {
    let ctx = context_wire("Grace Hopper", "a computer scientist", Vec::new());
    let content = flatten_result_content(&ctx);
    assert_eq!(content, "Grace Hopper: a computer scientist");
}

// ─── HTTP-path render_prompt_block vs library-path context_block
// must be byte-identical for the same logical item — this was previously
// unguardable by test, before the
// `RenderableContext` trait made both paths call the SAME generic fn. ──

/// The item is built ONCE as a real `kremory::RetrievedContext` (using
/// the documented cross-crate constructor, `RetrievedContext::new()` +
/// `with_namespace` + `with_facts` — the same path `conversions.rs:676`
/// already uses in production) and the wire twin is derived FROM it via
/// the existing `From<RetrievedContext> for RetrievedContextWire` impl —
/// never hand-built independently. This mirrors the real production shape
/// (kremory-mcp always converts FROM a facade `RetrievedContext`; it
/// never constructs a `RetrievedContextWire` from scratch). `SourceRef` is
/// NOT `#[non_exhaustive]` (`memory/types.rs:311`), so this test's item
/// genuinely exercises the source_refs render path on both sides.
///
/// SUPERSEDED: this doc comment previously said
/// `RetrievedFact` had "NO public constructor anywhere in the crate" and
/// so this fixture "exercises the facts=empty fallback branch, not the
/// facts-populated branch". That claim was itself the root cause: the
/// constructor gap was real, but the
/// fix was to ADD the constructor (`RetrievedFact::new` +
/// `RetrievedContext::with_facts`/`with_entity_type`), not to accept the
/// coverage gap as permanent. The fixture below now carries one fact and
/// drives `render_temporal_facts`'s facts-populated arm
/// (`memory/mod.rs`'s `valid_at`/`invalid_at` formatting branch) through
/// BOTH the library path and the HTTP path, side by side.
#[test]
fn td198_http_render_and_library_render_are_byte_identical() {
    use kremory::memory::ContextTemplate;
    use kremory::{
        RetrievedContext, RetrievedContextNewParams, RetrievedFact, RetrievedFactNewParams,
        SourceKind, SourceRef,
    };

    let occurred_at = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00+00:00")
        .expect("valid fixed RFC-3339 fixture timestamp")
        .with_timezone(&chrono::Utc);
    let valid_at = chrono::DateTime::parse_from_rfc3339("2025-06-15T00:00:00+00:00")
        .expect("valid fixed RFC-3339 fixture timestamp")
        .with_timezone(&chrono::Utc);
    let invalid_at = chrono::DateTime::parse_from_rfc3339("2026-02-01T00:00:00+00:00")
        .expect("valid fixed RFC-3339 fixture timestamp")
        .with_timezone(&chrono::Utc);

    // Built via the new public constructor + fluent setter — the
    // fixture this test needed all along, and previously impossible from
    // outside the `kremory` crate.
    let fact = RetrievedFact::new(RetrievedFactNewParams {
        fact: "Ada Lovelace invented the compiler".to_string(),
        subject: "Ada Lovelace".to_string(),
        predicate: "invented".to_string(),
        object: "the compiler".to_string(),
        object_is_entity: false,
        valid_at,
        recorded_at: valid_at,
        confidence: 0.95,
        source_episode_ids: vec![1],
        score: 0.9,
    })
    // Sets `invalid_at` so the fixture also exercises
    // `render_temporal_facts`'s `Some(inv) => ", invalid_at=..."` arm,
    // not just the `None` arm.
    .with_invalid_at(invalid_at);

    let core_ctx = RetrievedContext::new(RetrievedContextNewParams {
        entity_id: "e1".to_string(),
        entity_name: "Ada Lovelace".to_string(),
        summary: "a mathematician".to_string(),
        score: 0.9,
        source_refs: vec![SourceRef {
            kind: SourceKind::Chat,
            id: "ep-1".to_string(),
            occurred_at,
            published_at: None,
        }],
    })
    .with_namespace(kremory::Namespace::new("ns-a"))
    .with_facts(vec![fact]);

    // NON-VACUITY precondition — a test that could pass by rendering
    // nothing proves nothing. Measured baseline (recorded so a future
    // reader doesn't have to re-derive it): 1 item, 1 source_ref, 1 fact
    // (with `invalid_at` set — this closes the facts=empty gap in the
    // original fixture).
    assert_eq!(core_ctx.source_refs.len(), 1, "fixture must carry a source_ref");
    assert_eq!(core_ctx.facts.len(), 1, "fixture must carry a fact");

    let wire_ctx: RetrievedContextWire = core_ctx.clone().into();

    let cases = [
        (ContextTemplate::Entities, RecallTemplateWire::Entities),
        (ContextTemplate::EdgeSummary, RecallTemplateWire::EdgeSummary),
        (ContextTemplate::TemporalFacts, RecallTemplateWire::TemporalFacts),
    ];
    for (core_template, wire_template) in cases {
        let library_out =
            kremory::memory::context_block(std::slice::from_ref(&core_ctx), core_template);
        let http_out = render_prompt_block(std::slice::from_ref(&wire_ctx), wire_template);

        // NON-VACUITY postcondition on the render output itself, not just
        // the input — both paths must actually have rendered something.
        assert!(
            !library_out.is_empty(),
            "{core_template:?}: library path rendered nothing"
        );
        assert!(!http_out.is_empty(), "{core_template:?}: HTTP path rendered nothing");

        assert_eq!(
            library_out, http_out,
            "{core_template:?}: HTTP-path render_prompt_block and library-path \
             context_block diverged for the same logical item"
        );
    }
}

// ─── SearchResultWire::kind / source_episode_id
// — pure serialization, no live arm emits `Fact` yet ────

/// Entity and Episode items (the two kinds every live arm emits today)
/// serialize `source_episode_id` as `null` — proves the ADDITIVE fields
/// don't perturb the existing `{id, content, score}` shape consumers
/// (the LoCoMo harness) already read.
#[test]
fn search_result_wire_entity_and_episode_kinds_have_no_source_episode_id() {
    let entity = SearchResultWire {
        id: "e1".into(),
        content: "Ada Lovelace: a mathematician".into(),
        score: 0.9,
        kind: SearchResultKindWire::Entity,
        source_episode_id: None,
    };
    let episode = SearchResultWire {
        id: "42".into(),
        content: "Ada Lovelace wrote the first algorithm".into(),
        score: 0.7,
        kind: SearchResultKindWire::Episode,
        source_episode_id: None,
    };
    let ej = serde_json::to_value(&entity).expect("entity serializes");
    let pj = serde_json::to_value(&episode).expect("episode serializes");
    assert_eq!(ej["kind"], "entity", "entity kind: {ej}");
    assert!(ej["source_episode_id"].is_null(), "entity: {ej}");
    assert_eq!(pj["kind"], "episode", "episode kind: {pj}");
    assert!(pj["source_episode_id"].is_null(), "episode: {pj}");
}

/// A Fact-kind item — not yet emitted by any live path (a later,
/// separately-measured change wires the arm that would emit it) — carries its source
/// episode id on the wire. This is the shape `bench/locomo/
/// evidence_eval.py`'s fact-resolution path (Part B) depends on: proves
/// the wire CAN carry the provenance before the arm that would populate
/// it exists.
#[test]
fn search_result_wire_fact_kind_serializes_with_source_episode_id() {
    let fact = SearchResultWire {
        id: "fact-7".into(),
        content: "Caroline attended LGBTQ_support_group".into(),
        score: 0.8,
        kind: SearchResultKindWire::Fact,
        source_episode_id: Some(42),
    };
    let json = serde_json::to_value(&fact).expect("fact serializes");
    assert_eq!(json["kind"], "fact", "fact kind: {json}");
    assert_eq!(
        json["source_episode_id"], 42,
        "fact source_episode_id must round-trip: {json}"
    );
}

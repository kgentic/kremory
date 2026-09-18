"""Regression guards for the LoCoMo substring scorer.

These exist because of a defect class this repo has now hit FIVE times: an
ABSENCE (a missing field, a skipped file, an unmeasurable category) read as a
MEASUREMENT. The specific instance guarded here scored 446/446 adversarial
questions "correct" for months — category-5 gold lives under
`adversarial_answer`, `extract_questions` read `answer` with a `""` default,
and `"" in anything` is `True`.

Two of the four tests drive the REAL corpus rather than a hand-built fixture,
deliberately: a fixture encodes the author's mental model of the data, which is
the same model that produced the bug. Only the real file can disagree with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

# Both bench harnesses are named `harness`, so a plain `import harness` binds to
# whichever directory happens to be first on sys.path — the other suite then
# silently tests the WRONG module. Load this one by explicit path under a unique
# name so the suites can run together.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))
_spec = _ilu.spec_from_file_location(
    "_locomo_harness", _Path(__file__).resolve().parent / "harness.py"
)
harness = _ilu.module_from_spec(_spec)
_sys.modules["_locomo_harness"] = harness
_spec.loader.exec_module(harness)

DATA = Path(__file__).parent / "data" / "locomo10.json"


def _corpus() -> list[dict]:
    if not DATA.exists():
        pytest.skip(f"corpus not present: {DATA}")
    return json.loads(DATA.read_text())


def test_empty_gold_raises_instead_of_matching_everything() -> None:
    """`"" in anything` is True — an empty gold must never reach the scorer.

    This is the exact shape of the original bug: absence of a gold silently
    became a perfect score.
    """
    with pytest.raises(ValueError, match="empty gold"):
        harness.check_answer_in_memories("", ["any retrieved text at all"], "single-hop")
    with pytest.raises(ValueError, match="empty gold"):
        harness.check_answer_in_memories("   ", [], "temporal")


def test_abstention_category_is_refused_not_scored() -> None:
    """Abstention is an answerer property; a presence metric cannot see it.

    Refusal is structural (in the function) rather than a caller convention,
    so a future call site cannot reintroduce the number by forgetting a guard.
    """
    for category in sorted(harness.ABSTENTION_CATEGORIES):
        with pytest.raises(ValueError, match="abstention category"):
            harness.check_answer_in_memories("self-care is important", ["x"], category)


def test_every_corpus_question_yields_a_non_empty_gold() -> None:
    """Real corpus, not a fixture. 446 of 1986 questions carry their gold under
    a DIFFERENT key, which is what the original default silently swallowed."""
    seen = 0
    for sample in _corpus():
        for q in harness.extract_questions(sample):
            seen += 1
            assert str(q["answer"]).strip(), (
                f"empty gold survived extraction for {q['category']} "
                f"question {q['question_id']}"
            )
    assert seen == 1986, f"expected the full LoCoMo corpus, got {seen} questions"


def test_adversarial_gold_comes_from_the_adversarial_answer_field() -> None:
    """Real corpus. Asserts the distractor is carried through AND that scoring
    it is refused — carrying it is for offline abstention judging only."""
    advs = [
        q
        for sample in _corpus()
        for q in harness.extract_questions(sample)
        if q["category"] == "adversarial"
    ]
    assert len(advs) == 446, f"expected 446 adversarial questions, got {len(advs)}"

    raw = {
        (s.get("sample_id", ""), f"q_{i}"): q
        for s in _corpus()
        for i, q in enumerate(s.get("qa", []))
    }
    # Spot-check the join on the canonical example: the charity race was
    # MELANIE's, so "What did Caroline realize after her charity race?" is a
    # speaker-attribution false premise and "self-care is important" is the
    # distractor, not the gold.
    assert any(
        q["answer"] == "self-care is important" for q in advs
    ), "known distractor missing — extraction is not reading `adversarial_answer`"

    for q in advs:
        assert str(q["answer"]).strip()
        with pytest.raises(ValueError):
            harness.check_answer_in_memories(q["answer"], ["some text"], q["category"])


# ── TD-217: the headline must carry its own denominator ──────────────────────


def test_headline_names_the_unscored_categories_and_counts() -> None:
    """A run's quotable line must state what it excludes, in the line itself.

    The failure this guards is not a missing field — `unscored_stats` was
    always in the JSON. It is that the ONE string a human copies out ("98.0%")
    travelled without it, so it could sit in a comparison table beside a
    competitor number computed over ALL questions and look like a peer.
    """
    got = harness.build_headline(149, 152, {"adversarial": 47})
    assert "149/152" in got
    assert "98.0%" in got
    assert "47 adversarial" in got
    assert "NOT SCORED" in got
    # The total asked must appear, so the reader can see 152 != 199.
    assert "199" in got


def test_headline_is_plain_when_nothing_is_excluded() -> None:
    """No unscored categories → no caveat clause to mislead the reader either way."""
    got = harness.build_headline(10, 10, {})
    assert got == "10/10 = 100.0%"


def test_headline_never_emits_a_bare_percentage_when_questions_were_dropped() -> None:
    """Structural form of TD-217(b): for ANY non-empty exclusion the caveat is
    present. Asserting the property, not one sample — a future category added
    to ABSTENTION_CATEGORIES must not silently produce a bare number."""
    for cats in ({"adversarial": 1}, {"adversarial": 47, "made-up": 3}):
        got = harness.build_headline(1, 2, cats)
        assert "NOT SCORED" in got, got
        for cat, n in cats.items():
            assert f"{n} {cat}" in got, got


def test_shared_client_identity_is_not_shadowed() -> None:
    """Both harnesses must use the SAME classes from bench/common.

    Guards a bug introduced and caught during the 2026-07-28 de-fork: the
    harness kept a local `class KremoryStalled` while the shared client raised
    its own, producing two DISTINCT classes. `except KremoryStalled` then
    silently stops catching a stalled ingest — the fail-fast path goes dead
    while still looking present in the source. Identity, not just name.
    """
    import kremory_client

    assert harness.CodememClient is kremory_client.CodememClient
    assert harness.KremoryStalled is kremory_client.KremoryStalled
    assert harness.INGEST_BUDGET_S == kremory_client.INGEST_BUDGET_S


# ── TD-187: temporal grounding — the session anchor must reach kremory ────────


def test_parse_session_datetime_emits_utc_offset() -> None:
    """The anchor MUST carry a UTC offset or kremory rejects every store.

    kremory parses `published_at` with `DateTime::parse_from_rfc3339`
    (`crates/kremory-mcp/src/conversions.rs:150`), which REJECTS a naive
    timestamp; the accepted shape is pinned by `conversions.rs:551`. A first
    draft of `parse_session_datetime` returned `.isoformat()` on a naive
    datetime — every store would have 4xx'd and the whole bench run would have
    been lost. This test is the tripwire for that regression.
    """
    import re

    got = harness.parse_session_datetime("1:56 pm on 8 May, 2023")
    assert got is not None
    assert re.search(r"([+-]\d{2}:\d{2}|Z)$", got), f"no UTC offset: {got}"
    assert got.startswith("2023-05-08T13:56:00")


def test_parse_session_datetime_returns_none_not_now_on_failure() -> None:
    """Unparseable MUST stay None — never a wall-clock substitute.

    Load-bearing, not cosmetic: kremory renders the anchor into the extraction
    prompt ONLY when the caller declared one, because the VCR fingerprint
    (`core/provider/record_replay.rs:225-263`) hashes the message list. A
    wall-clock fallback here would change the prompt on every run and make all
    303 chat cassettes permanently un-replayable.
    """
    assert harness.parse_session_datetime("") is None
    assert harness.parse_session_datetime("   ") is None
    assert harness.parse_session_datetime("not a date at all") is None


def test_parse_session_datetime_covers_the_whole_real_corpus() -> None:
    """Every session header in the shipped corpus must parse — 272 of 272.

    Drives the REAL corpus, not a hand-made fixture: a hand-shaped string would
    only validate the author's mental model of the format
    (`smoke-one-before-batch-llm-validation`, subagent-fixture clause).
    """
    import json
    from pathlib import Path

    corpus = Path(__file__).parent / "data" / "locomo10.json"
    if not corpus.exists():
        pytest.skip("locomo10.json not present")

    data = json.loads(corpus.read_text())
    convs = data if isinstance(data, list) else [data]
    total = 0
    parsed = 0
    for conv in convs:
        for sess in harness.extract_sessions(conv):
            total += 1
            if harness.parse_session_datetime(sess["datetime"]):
                parsed += 1

    assert total > 0, "corpus yielded no sessions — extract_sessions broke"
    assert parsed == total, f"only {parsed} of {total} session headers parsed"


# ---------------------------------------------------------------------------
# Graph integrity — TD-223 / TD-224
# ---------------------------------------------------------------------------
#
# Same defect class as the rest of this file: an ABSENCE read as a MEASUREMENT.
# A benchmark scored 149/152 = 98.0% on a graph in which both speakers of the
# dialogue had been merged out of existence, identically to the graph where they
# survived. Nothing detected it. These guard the detector that now does — and,
# just as importantly, guard that it reports SKIPPED rather than a pass when it
# has nothing to look at.

import sqlite3 as _sqlite3


def _graph_db(tmp_path, merges, *, undone_index=None, bad_row=False):
    """Build a minimal graph_mutation_log. `merges` is a list of (keeper, loser).

    The `inputs` shape is copied from a real corrupted benchmark database rather
    than invented here — a fixture built from the author's mental model of the
    contract is the same model that would produce a wrong reader.
    """
    db = tmp_path / "graph.db"
    conn = _sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE graph_mutation_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "kind TEXT NOT NULL, group_id TEXT NOT NULL, created_at TEXT NOT NULL, "
        "undone_at TEXT, pre_state TEXT NOT NULL, inputs TEXT NOT NULL)"
    )
    for i, (keeper, loser) in enumerate(merges):
        lo, hi = sorted([keeper, loser])
        payload = {"pair_lo": lo, "pair_hi": hi, "keeper": keeper, "loser": loser,
                   "site": "site5_acronym_nickname", "cosine": None,
                   "structural_signal": True}
        if bad_row:
            payload = {"survivor": keeper, "absorbed": loser}
        conn.execute(
            "INSERT INTO graph_mutation_log (kind, group_id, created_at, undone_at, "
            "pre_state, inputs) VALUES ('entity_merge', 'default', "
            "'2026-08-17T09:45:39+00:00', ?, '{}', ?)",
            (("2026-08-17T10:00:00+00:00" if i == undone_index else None),
             json.dumps(payload)),
        )
    conn.commit()
    conn.close()
    return str(db)


def test_star_merges_are_not_a_chain(tmp_path):
    """THE FALSE-POSITIVE TEST. Two variants merged into one survivor is ordinary
    canonicalisation. Without this, a check that simply fired on "any merge
    happened" would look perfectly healthy against the chain test below."""
    db = _graph_db(tmp_path, [("pottery project", "pottery class"),
                              ("pottery project", "pottery")])
    r = harness.check_graph_integrity(db)
    assert r["status"] == "checked"
    assert r["clean"] is True, r
    assert r["chained_entities"] == 0
    assert r["live_merges"] == 2


def test_transitive_chain_is_detected_and_named(tmp_path):
    """The real corruption shape: a speaker absorbed into another entity, which
    is then itself absorbed."""
    db = _graph_db(tmp_path, [("caroline", "melanie"), ("loved ones", "caroline")])
    r = harness.check_graph_integrity(db)
    assert r["clean"] is False
    assert r["chained_entities"] == 1
    assert any("'caroline' absorbed [melanie] then was absorbed by 'loved ones'" in c
               for c in r["chains"]), r["chains"]


def test_undone_merge_is_not_live_damage(tmp_path):
    """A merge already reversed via unmerge no longer holds the graph in a
    chained state."""
    db = _graph_db(tmp_path, [("caroline", "melanie"), ("loved ones", "caroline")],
                   undone_index=1)
    r = harness.check_graph_integrity(db)
    assert r["clean"] is True, r
    assert r["live_merges"] == 1


@pytest.mark.parametrize("path", [None, "", "/tmp/definitely-not-a-graph.db"])
def test_absent_graph_reports_skipped_never_a_pass(path):
    """TD-224. The defect being guarded is a check that reports success while
    looking at nothing — so "no graph" must never surface as clean."""
    r = harness.check_graph_integrity(path)
    assert r["status"] == "skipped"
    assert "clean" not in r


def test_fanin_is_detected_where_no_chain_forms(tmp_path):
    """TD-256. THE BLIND SPOT. Nine dates absorbed into one keeper that is never
    itself absorbed — so `survivors & victims` is EMPTY and the chain check calls
    this clean. This is the shape that dominates the real corpus: 4 chains vs 20
    fan-ins on `.context/full-corpus.db`, 19 of them date collapses.
    """
    db = _graph_db(tmp_path, [("3 july 2023", "5 july 2023"),
                              ("3 july 2023", "6 july 2023"),
                              ("3 july 2023", "20 july 2023")])
    r = harness.check_graph_integrity(db)
    assert r["chained_entities"] == 0, "no chain exists — that is the whole point"
    assert r["fanin_entities"] == 1, r
    assert r["worst_fanin"] == 3, r
    assert any("'3 july 2023' absorbed 3:" in f for f in r["fanins"]), r["fanins"]


def test_fanin_does_not_flip_clean_for_ordinary_canonicalisation(tmp_path):
    """The R5 contract: fan-ins are REPORTED, never used to fail a run.

    At the only threshold that detects anything (2 losers) a legitimate variant
    merge is indistinguishable from damage without reading the names, so a
    two-variant canonicalisation must still report clean while being surfaced.
    Paired with test_star_merges_are_not_a_chain, which pins the same fixture.
    """
    db = _graph_db(tmp_path, [("pottery project", "pottery class"),
                              ("pottery project", "pottery")])
    r = harness.check_graph_integrity(db)
    assert r["clean"] is True, "fan-ins must not fail a run"
    assert r["fanin_entities"] == 1, "...but must still be reported"


def test_duplicate_loser_rows_do_not_inflate_a_fanin(tmp_path):
    """The log really does record the same loser twice ('1 february 2023' and
    '3 august 2023' on the full corpus). Counting rows instead of DISTINCT
    entities would promote a one-loser merge into a fan-in."""
    db = _graph_db(tmp_path, [("1 february 2023", "4 february 2023"),
                              ("1 february 2023", "4 february 2023")])
    r = harness.check_graph_integrity(db)
    assert r["fanin_entities"] == 0, "one distinct loser is not a fan-in"


def test_shared_fixture_matches_rust_implementation(tmp_path):
    """TD-256. The two implementations agree ON ONE FIXTURE, checked mechanically.

    `harness.check_graph_integrity` and invariant 6 in
    `crates/kremory-eval/src/layer_b/graph_integrity.rs` are mirrors of the same
    contract. Until this fixture existed the only thing keeping them aligned was a
    doc-comment asking a future editor to remember, and they had already drifted
    once elsewhere in this repo (the REST fusion copy, TD-173, four weeks).

    The Rust side asserts the SAME `expected` block from the SAME file
    (`shared_fixture_matches_python_implementation`), so changing one
    implementation alone turns its own suite red.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "graph_integrity_shared.json"
    with open(fixture_path) as f:
        fixture = json.load(f)

    db = _graph_db(tmp_path, [tuple(m) for m in fixture["merges"]])
    r = harness.check_graph_integrity(db)
    expected = fixture["expected"]

    assert r["status"] == "checked"
    for key, want in expected.items():
        assert r[key] == want, f"{key}: got {r[key]!r}, fixture says {want!r}"


def test_producer_shape_drift_errors_rather_than_reporting_zero(tmp_path):
    """A silently skipped merge row makes a corrupted graph look clean, which is
    exactly the failure this check exists to catch."""
    db = _graph_db(tmp_path, [("caroline", "melanie")], bad_row=True)
    r = harness.check_graph_integrity(db)
    assert r["status"] == "error"
    assert "keeper/loser" in r["reason"]

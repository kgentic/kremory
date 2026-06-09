#!/usr/bin/env python3
"""
LLM-as-Authority Spike — kremory v0.1.1 hybrid extractor PoC
=============================================================

Spike question: Can gemma4-e2b:latest handle a structured-output schema where
the LLM emits per-GLiNER-candidate decisions (Confirm/Reject/Modify) AND adds
missed entities, all integer-ID enum constrained per TD-013 discipline?

Throwaway code — relaxed quality rules apply (assert/unwrap/etc OK).

Model SoT: crates/kremory/tests/llm_integration.rs:1-25 — gemma4-e2b:latest
is the interactive default.
"""

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Model SoT: crates/kremory/tests/llm_integration.rs:1-25 (gemma4-e2b:latest = interactive default)
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma4-e2b:latest")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "1h")
FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../crates/kremory-eval/fixtures"
)

# ─── Entity type registry (mirrors DEFAULT_ENTITY_TYPES from entity_types.rs) ─
# IDs match crates/kremory/src/core/entity_types.rs DEFAULT_ENTITY_TYPES
ENTITY_TYPES = {
    0: "Entity",       # catch-all
    1: "Person",
    2: "Organisation",
    3: "Location",
    4: "Technology",
    5: "Product",
    6: "Event",
    7: "Date",
    8: "Time",
    9: "Concept",
    10: "Court",       # Added in Phase A fix b09e67f
}

# GLiNER label → entity_type_id mapping (GLiNER uses string labels, we map to IDs)
GLINER_LABEL_TO_ID = {
    "Person": 1,
    "Organisation": 2,
    "Location": 3,
    "Technology": 4,
    "Product": 5,
    "Event": 6,
    "Date": 7,
    "Time": 8,
    "Concept": 9,
    "Court": 10,
}

# Legal deposition ground truth (from fixtures/ground_truth.json)
GROUND_TRUTH = [
    {"name": "Gregory Tanaka", "label": "Person"},
    {"name": "CLIFFORD REEVES", "label": "Person"},
    {"name": "DIANA OHLSSON", "label": "Person"},
    {"name": "Sandra Obi", "label": "Person"},
    {"name": "Felix Drummond", "label": "Person"},
    {"name": "Margaret Solis", "label": "Person"},
    {"name": "Collette Beaumont", "label": "Person"},
    {"name": "Harlow Industries, Inc.", "label": "Organisation"},
    {"name": "Pinnacle Supply Co.", "label": "Organisation"},
    {"name": "Feldstein & Moorhouse LLP", "label": "Organisation"},
    {"name": "California Department of Industrial Relations", "label": "Organisation"},
    {"name": "Los Angeles Superior Court", "label": "Court"},
    {"name": "Los Angeles", "label": "Location"},
    {"name": "Fresno", "label": "Location"},
    {"name": "Alameda County", "label": "Location"},
    {"name": "February 14, 2024", "label": "Date"},
]

# ─── Simulated GLiNER output at threshold=0.2 ─────────────────────────────────
# Derived from Phase A lesson: at threshold=0.2, GLiNER catches most entities
# but has known misses. This simulates what GLiNER would emit.
# Source: .ai-docs/lessons/2026-06-09-v0-1-1-phase-a-llm-gate.md
#
# Confirmed catches at threshold=0.2 (from Phase A lesson):
#   - CLIFFORD REEVES → "Clifford Reeves" (after allcaps normalization) — BUT gemma4-e2b
#     misses this due to LLM typing issue, not GLiNER miss
# Known GLiNER miss regardless of threshold:
#   - "Feldstein & Moorhouse LLP" — & breaks GLiNER span detection
#
# For this spike we simulate the GLiNER output WITHOUT the known-miss entity
# to test whether the LLM's additional_entities mechanism can surface it.
SIMULATED_GLINER_CANDIDATES = [
    {"name": "Gregory Tanaka", "label": "Person", "confidence": 0.92},
    {"name": "Clifford Reeves", "label": "Person", "confidence": 0.45},   # allcaps normalized
    {"name": "Diana Ohlsson", "label": "Person", "confidence": 0.43},     # allcaps normalized
    {"name": "Sandra Obi", "label": "Person", "confidence": 0.88},
    {"name": "Felix Drummond", "label": "Person", "confidence": 0.85},
    {"name": "Margaret Solis", "label": "Person", "confidence": 0.87},
    {"name": "Collette Beaumont", "label": "Person", "confidence": 0.82},
    {"name": "Harlow Industries, Inc.", "label": "Organisation", "confidence": 0.79},
    {"name": "Pinnacle Supply Co.", "label": "Organisation", "confidence": 0.76},
    {"name": "California Department of Industrial Relations", "label": "Organisation", "confidence": 0.71},
    {"name": "Los Angeles Superior Court", "label": "Organisation", "confidence": 0.65},  # GLiNER misclassifies Court as Org
    {"name": "Los Angeles", "label": "Location", "confidence": 0.91},
    {"name": "Fresno", "label": "Location", "confidence": 0.88},
    {"name": "Alameda County", "label": "Location", "confidence": 0.85},
    {"name": "February 14, 2024", "label": "Date", "confidence": 0.93},
    # NOTE: "Feldstein & Moorhouse LLP" is intentionally ABSENT — GLiNER & limitation
]


# ─── Ollama chat call ─────────────────────────────────────────────────────────

def ollama_chat(messages: list, schema: Optional[dict] = None, temperature: float = 0.0) -> tuple[str, float]:
    """Call Ollama /api/chat. Returns (content, latency_ms)."""
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        }
    }
    if schema:
        payload["format"] = schema

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    latency_ms = (time.time() - t0) * 1000
    return result["message"]["content"], latency_ms


# ─── Schema definitions ───────────────────────────────────────────────────────

def build_current_schema(num_candidates: int) -> dict:
    """Current hybrid_typer.rs schema: {typings: [{idx, entity_type_id}]}"""
    idx_enum = list(range(num_candidates))
    type_id_enum = [id for id in ENTITY_TYPES.keys() if id != 0]
    return {
        "type": "object",
        "properties": {
            "typings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {
                            "type": "integer",
                            "enum": idx_enum,
                        },
                        "entity_type_id": {
                            "type": "integer",
                            "enum": type_id_enum,
                        }
                    },
                    "required": ["idx", "entity_type_id"]
                }
            }
        },
        "required": ["typings"]
    }


def build_authority_schema(num_candidates: int) -> dict:
    """
    New LLM-as-authority schema:
    {
      per_candidate: [{idx, decision: "Confirm"|"Reject"|"Modify", entity_type_id, name?}],
      additional_entities: [{name, entity_type_id}]
    }

    Integer-ID constrained per TD-013 discipline.
    """
    idx_enum = list(range(num_candidates))
    type_id_enum = [id for id in ENTITY_TYPES.keys() if id != 0]
    return {
        "type": "object",
        "properties": {
            "per_candidate": {
                "type": "array",
                "description": "One decision per GLiNER candidate, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {
                            "type": "integer",
                            "enum": idx_enum,
                            "description": "0-based index matching the candidate list"
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["Confirm", "Reject", "Modify"],
                            "description": "Confirm: keep as-is; Reject: not an entity; Modify: rename or reclassify"
                        },
                        "entity_type_id": {
                            "type": "integer",
                            "enum": type_id_enum,
                            "description": "Entity type id from the registry (required for Confirm/Modify)"
                        },
                        "name": {
                            "type": "string",
                            "description": "Modified name (only for Modify decision)"
                        }
                    },
                    "required": ["idx", "decision", "entity_type_id"]
                }
            },
            "additional_entities": {
                "type": "array",
                "description": "Entities the LLM spotted that GLiNER missed",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name exactly as it appears in the text"
                        },
                        "entity_type_id": {
                            "type": "integer",
                            "enum": type_id_enum,
                        }
                    },
                    "required": ["name", "entity_type_id"]
                }
            }
        },
        "required": ["per_candidate", "additional_entities"]
    }


# ─── Prompt builders ──────────────────────────────────────────────────────────

def build_registry_block() -> str:
    lines = ["Entity types — assign each entity's entity_type_id to ONE of these integer ids:"]
    for id, name in ENTITY_TYPES.items():
        if id == 0:
            lines.append(f"  {id} — {name} — catch-all (reserved; LLM must NOT emit 0)")
        else:
            lines.append(f"  {id} — {name}")
    return "\n".join(lines)


def build_current_prompt(text: str, candidates: list) -> str:
    """Current hybrid_typer.rs prompt shape."""
    candidates_block = "\n".join(f"  [{i}] {c['name']}" for i, c in enumerate(candidates))
    registry_block = build_registry_block()
    last_idx = len(candidates) - 1
    return f"""You are classifying entities found in a knowledge-graph ingest.

SOURCE TEXT:
{text}

CANDIDATE ENTITIES (already extracted by a fast NER pass — your job is to assign a type to each):
{candidates_block}

AVAILABLE ENTITY TYPES:
{registry_block}

For each candidate above, assign the most specific matching entity_type_id from the table.
id=0 (Entity) is FORBIDDEN — it is a reserved server-side catch-all. You MUST pick a specific id >= 1.
When in doubt, pick the closest semantic match (e.g. 'Superior Court' is a Court when Court is listed; otherwise an Organisation. A named medication is a Drug when Drug is listed; otherwise a Concept).

Output a JSON object with this exact shape, where each value is a real integer (NOT placeholder text):
{{
  "typings": [
    {{"idx": 0, "entity_type_id": 1}},
    {{"idx": 1, "entity_type_id": 2}}
  ]
}}

RULES:
- `idx` must be an integer from 0 to {last_idx} (inclusive). Use the integer prefix from `  [N] Name` above.
- `entity_type_id` must be an integer >= 1 chosen from the AVAILABLE ENTITY TYPES table above.
- Emit exactly one object per candidate. Do not skip any. Do not add extras.
- DO NOT include the candidate name, placeholder text like "<integer>", ellipsis, or any string values — only real integers."""


def build_authority_prompt(text: str, candidates: list) -> str:
    """New LLM-as-authority prompt: Confirm/Reject/Modify + add missed entities."""
    candidates_block = "\n".join(
        f"  [{i}] {c['name']} (hint: {c['label']}, conf={c['confidence']:.2f})"
        for i, c in enumerate(candidates)
    )
    registry_lines = []
    for id, name in ENTITY_TYPES.items():
        if id != 0:
            registry_lines.append(f"  id={id} → {name}")
    registry_str = "\n".join(registry_lines)
    last_idx = len(candidates) - 1
    type_ids = [str(id) for id in ENTITY_TYPES.keys() if id != 0]

    num_cands = len(candidates)
    # Build a concrete worked sub-example (3-candidate mini-demo) to force completion
    mini_demo = f"""{{"per_candidate":[{{"idx":0,"decision":"Confirm","entity_type_id":1}},{{"idx":1,"decision":"Modify","entity_type_id":2,"name":"Harlow Corp"}},{{"idx":2,"decision":"Reject","entity_type_id":1}}],"additional_entities":[{{"name":"Los Angeles","entity_type_id":3}}]}}"""

    return f"""You classify entities from a knowledge-graph ingest. Below are {num_cands} GLiNER candidates. For EACH candidate output exactly one per_candidate entry. Then add any entities GLiNER missed.

SOURCE TEXT:
{text}

CANDIDATES (idx 0 to {last_idx}):
{candidates_block}

ENTITY TYPE IDS:
{registry_str}

TASK: Produce a complete JSON response. For EVERY candidate from 0 to {last_idx}:
- "Confirm" = keep as valid entity, assign correct entity_type_id
- "Reject" = NOT a real named entity
- "Modify" = correct name/type, provide updated "name" field

EXAMPLE (3 candidates, 1 additional):
{mini_demo}

NOW OUTPUT THE REAL JSON for ALL {num_cands} candidates (idx 0 through {last_idx}).
entity_type_id must be one of: {', '.join(type_ids)}
NEVER output id=0. Output ONLY raw JSON (no markdown code blocks, no explanations):"""


# ─── Parsing ─────────────────────────────────────────────────────────────────

def parse_current_response(raw: str, candidates: list) -> list[dict]:
    """Parse current hybrid_typer.rs response format."""
    try:
        # Try direct JSON parse first
        data = json.loads(raw)
        typings_by_idx = {}
        for typing in data.get("typings", []):
            idx = typing.get("idx")
            type_id = typing.get("entity_type_id")
            if idx is not None and type_id is not None:
                typings_by_idx[int(idx)] = int(type_id)
    except json.JSONDecodeError:
        print(f"  [parse] JSON parse failed, attempting repair...")
        typings_by_idx = {}

    result = []
    for i, cand in enumerate(candidates):
        type_id = typings_by_idx.get(i)
        if type_id and type_id in ENTITY_TYPES and type_id != 0:
            label = ENTITY_TYPES[type_id]
        else:
            # Fall back to GLiNER label (current behavior in hybrid_typer.rs)
            label = cand["label"]
        result.append({"name": cand["name"], "label": label})
    return result


def parse_authority_response(raw: str, candidates: list) -> tuple[list[dict], list[dict], int]:
    """
    Parse LLM-as-authority response.
    Returns (final_entities, additional_entities_raw, reject_count)
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [parse] JSON parse failed")
        # Fall back to all GLiNER labels
        return [{"name": c["name"], "label": c["label"]} for c in candidates], [], 0

    per_candidate = {
        entry["idx"]: entry
        for entry in data.get("per_candidate", [])
        if isinstance(entry.get("idx"), int)
    }
    additional_raw = data.get("additional_entities", [])

    result = []
    reject_count = 0
    for i, cand in enumerate(candidates):
        entry = per_candidate.get(i)
        if not entry:
            # No decision for this candidate → fall back to GLiNER
            result.append({"name": cand["name"], "label": cand["label"]})
            continue

        decision = entry.get("decision", "Confirm")
        type_id = entry.get("entity_type_id", 0)

        if decision == "Reject":
            reject_count += 1
            continue  # Drop from output

        if decision == "Modify" and entry.get("name"):
            name = entry["name"]
        else:
            name = cand["name"]

        if type_id and type_id in ENTITY_TYPES and type_id != 0:
            label = ENTITY_TYPES[type_id]
        else:
            label = cand["label"]  # GLiNER fallback

        result.append({"name": name, "label": label})

    # Add additional entities
    additional_entities = []
    for ae in additional_raw:
        ae_name = ae.get("name", "").strip()
        ae_type_id = ae.get("entity_type_id", 0)
        if ae_name and ae_type_id and ae_type_id in ENTITY_TYPES and ae_type_id != 0:
            additional_entities.append({
                "name": ae_name,
                "label": ENTITY_TYPES[ae_type_id]
            })
            result.append({"name": ae_name, "label": ENTITY_TYPES[ae_type_id]})

    return result, additional_entities, reject_count


# ─── Scoring ──────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Simple case-insensitive + punctuation-stripped normalize."""
    return re.sub(r'[^a-z0-9\s]', '', name.lower()).strip()


def score_entities(extracted: list[dict], ground_truth: list[dict]) -> dict:
    """
    Compute precision/recall/F1 using fuzzy name match + label match.
    Mirrors kremory-eval's label_precision_benchmark logic.
    """
    gt_set = {(normalize_name(e["name"]), e["label"].lower()): e for e in ground_truth}
    tp_names = []
    fp_names = []

    for e in extracted:
        key = (normalize_name(e["name"]), e["label"].lower())
        if key in gt_set:
            tp_names.append(e["name"])
            del gt_set[key]  # matched
        else:
            # Also try label-only fallback (name match, any label)
            name_norm = normalize_name(e["name"])
            matched_by_name = [(k, v) for k, v in gt_set.items() if k[0] == name_norm]
            if matched_by_name:
                # Name matched but label wrong → partial match, count as FP
                fp_names.append(f"{e['name']} (wrong label: {e['label']})")
            else:
                fp_names.append(e["name"])

    fn_names = [v["name"] for v in gt_set.values()]

    tp = len(tp_names)
    fp = len(fp_names)
    fn = len(fn_names)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_names": tp_names,
        "fp_names": fp_names,
        "fn_names": fn_names,
    }


# ─── Main spike ───────────────────────────────────────────────────────────────

def run_spike():
    print("=" * 70)
    print("LLM-as-Authority Spike — kremory v0.1.1")
    print(f"Model (SoT: llm_integration.rs:1-25): {OLLAMA_CHAT_MODEL}")
    print(f"Fixture: legal_deposition (ground truth: {len(GROUND_TRUTH)} entities)")
    print(f"GLiNER candidates simulated: {len(SIMULATED_GLINER_CANDIDATES)}")
    print("=" * 70)

    # Read fixture
    fixture_path = os.path.join(FIXTURE_DIR, "legal_deposition.txt")
    with open(fixture_path) as f:
        text = f.read()

    candidates = SIMULATED_GLINER_CANDIDATES
    num_candidates = len(candidates)

    results = {}

    # ─── Test 1: Current schema (baseline) ────────────────────────────────────
    print("\n[Test 1] CURRENT schema (idx + entity_type_id) — baseline measurement")
    print(f"  Candidates: {num_candidates}")

    current_prompt = build_current_prompt(text, candidates)
    current_schema = build_current_schema(num_candidates)

    direct_parse_count = 0
    repair_path_count = 0

    run_start = time.time()
    # Run 3 times for consistency (legal_deposition has 1 "chunk" for simplicity)
    all_extracted_current = []
    raw_responses_current = []
    for run_i in range(3):
        raw, latency = ollama_chat(
            [{"role": "user", "content": current_prompt}],
            schema=None,  # Force LlmJsonRepair path (gemma4-e2b uses repair per Phase A)
        )
        raw_responses_current.append(raw)
        try:
            json.loads(raw)
            direct_parse_count += 1
        except json.JSONDecodeError:
            repair_path_count += 1
        entities = parse_current_response(raw, candidates)
        all_extracted_current.extend(entities)
        print(f"  Run {run_i+1}: latency={latency:.0f}ms, extracted={len(entities)}")
        if os.environ.get("KREMORY_DEBUG"):
            print(f"  Raw: {raw[:200]}...")

    # Deduplicate for scoring
    seen = set()
    deduped_current = []
    for e in all_extracted_current:
        key = (e["name"], e["label"])
        if key not in seen:
            seen.add(key)
            deduped_current.append(e)

    current_score = score_entities(deduped_current, GROUND_TRUTH)
    current_wall = time.time() - run_start

    print(f"\n  Q1 — Parse quality:")
    print(f"    Direct parse: {direct_parse_count}/3 ({direct_parse_count/3*100:.0f}%)")
    print(f"    Repair path:  {repair_path_count}/3")
    print(f"\n  Q2 — Precision vs baseline:")
    print(f"    Precision: {current_score['precision']*100:.1f}% ({current_score['tp']}/{current_score['tp']+current_score['fp']+current_score['fn']} entities)")
    print(f"    Recall:    {current_score['recall']*100:.1f}%")
    print(f"    F1:        {current_score['f1']*100:.1f}%")
    print(f"    TP: {current_score['tp_names']}")
    print(f"    FP: {current_score['fp_names'][:5]}{'...' if len(current_score['fp_names'])>5 else ''}")
    print(f"    FN: {current_score['fn_names']}")
    print(f"    Wall clock: {current_wall:.1f}s")

    results["current"] = {
        "direct_parse_rate": direct_parse_count / 3,
        "score": current_score,
        "wall_clock": current_wall,
    }

    # ─── Test 2: LLM-as-authority schema ──────────────────────────────────────
    print("\n[Test 2] LLM-AS-AUTHORITY schema (Confirm/Reject/Modify + additional)")
    print(f"  Candidates: {num_candidates}")

    authority_prompt = build_authority_prompt(text, candidates)
    authority_schema = build_authority_schema(num_candidates)

    direct_parse_count_auth = 0
    repair_path_count_auth = 0
    total_reject_count = 0
    all_additional = []

    run_start = time.time()
    all_extracted_auth = []
    raw_responses_auth = []
    for run_i in range(3):
        raw, latency = ollama_chat(
            [{"role": "user", "content": authority_prompt}],
            schema=None,  # Force LlmJsonRepair path
        )
        raw_responses_auth.append(raw)
        try:
            json.loads(raw)
            direct_parse_count_auth += 1
        except json.JSONDecodeError:
            repair_path_count_auth += 1

        entities, additional, rejects = parse_authority_response(raw, candidates)
        all_extracted_auth.extend(entities)
        all_additional.extend(additional)
        total_reject_count += rejects
        print(f"  Run {run_i+1}: latency={latency:.0f}ms, extracted={len(entities)}, "
              f"additional={len(additional)}, rejects={rejects}")
        if additional:
            print(f"    Additional: {[a['name'] for a in additional]}")
        if os.environ.get("KREMORY_DEBUG"):
            print(f"  Raw: {raw[:400]}...")

    # Deduplicate
    seen = set()
    deduped_auth = []
    for e in all_extracted_auth:
        key = (e["name"], e["label"])
        if key not in seen:
            seen.add(key)
            deduped_auth.append(e)

    auth_score = score_entities(deduped_auth, GROUND_TRUTH)
    auth_wall = time.time() - run_start

    avg_rejects = total_reject_count / 3
    reject_rate = avg_rejects / num_candidates

    print(f"\n  Q1 — Parse quality:")
    print(f"    Direct parse: {direct_parse_count_auth}/3 ({direct_parse_count_auth/3*100:.0f}%)")
    print(f"    Repair path:  {repair_path_count_auth}/3")
    print(f"\n  Q3 — Additional entities surfaced:")
    print(f"    Total additional across 3 runs: {len(all_additional)}")
    print(f"    All additional: {list(set(a['name'] for a in all_additional))}")
    known_misses = {"clifford reeves", "diana ohlsson", "feldstein  moorhouse llp", "feldstein moorhouse llp"}
    for a in all_additional:
        if normalize_name(a["name"]) in known_misses or "feldstein" in normalize_name(a["name"]):
            print(f"    *** KNOWN MISS RECOVERED: {a['name']} ({a['label']}) ***")

    print(f"\n  Rejection rate: {reject_rate*100:.1f}% avg ({avg_rejects:.1f}/{num_candidates} per run)")
    print(f"\n  Q2 — Precision vs baseline (81.2%):")
    print(f"    Precision: {auth_score['precision']*100:.1f}% ({auth_score['tp']}/{auth_score['tp']+auth_score['fp']+auth_score['fn']} entities)")
    print(f"    Recall:    {auth_score['recall']*100:.1f}%")
    print(f"    F1:        {auth_score['f1']*100:.1f}%")
    print(f"    TP: {auth_score['tp_names']}")
    print(f"    FP: {auth_score['fp_names'][:5]}{'...' if len(auth_score['fp_names'])>5 else ''}")
    print(f"    FN: {auth_score['fn_names']}")
    print(f"    Wall clock: {auth_wall:.1f}s")

    results["authority"] = {
        "direct_parse_rate": direct_parse_count_auth / 3,
        "score": auth_score,
        "wall_clock": auth_wall,
        "avg_reject_rate": reject_rate,
        "additional_surfaced": list(set(a["name"] for a in all_additional)),
    }

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPIKE SUMMARY")
    print("=" * 70)

    current_prec = current_score["precision"] * 100
    auth_prec = auth_score["precision"] * 100
    baseline = 81.2

    print(f"\nBaseline (from Phase A lesson 2026-06-09): {baseline}%")
    print(f"Current schema (re-measured):              {current_prec:.1f}%")
    print(f"LLM-as-authority schema:                   {auth_prec:.1f}%")

    # Q1
    q1_rate = (direct_parse_count_auth / 3) * 100
    q1_status = "VALIDATED" if q1_rate >= 70 else ("PARTIAL" if q1_rate >= 40 else "FAILED")
    print(f"\nQ1 (parse quality): direct-parse rate = {q1_rate:.0f}% → {q1_status}")

    # Q2
    q2_regression = auth_prec >= baseline
    q2_status = "VALIDATED" if q2_regression else "FAILED"
    print(f"Q2 (precision ≥ baseline): {auth_prec:.1f}% vs {baseline}% → {q2_status}")

    # Q3
    known_miss_names = ["Clifford Reeves", "CLIFFORD REEVES", "Diana Ohlsson", "DIANA OHLSSON",
                        "Feldstein", "Moorhouse"]
    q3_found = any(
        any(km.lower() in a.lower() for km in known_miss_names)
        for a in results["authority"]["additional_surfaced"]
    )
    q3_status = "VALIDATED" if q3_found else "FAILED"
    print(f"Q3 (LLM surfaces known miss): {q3_status}")
    print(f"  Additional surfaced: {results['authority']['additional_surfaced']}")

    # Verdict
    schema_complexity_ok = (q1_status != "FAILED")
    precision_ok = (q2_status == "VALIDATED")
    any_miss_caught = q3_found
    reject_rate_ok = results["authority"]["avg_reject_rate"] < 0.30

    print(f"\nRejection rate check: {results['authority']['avg_reject_rate']*100:.1f}% < 30% → {'OK' if reject_rate_ok else 'CONCERN'}")

    if schema_complexity_ok and precision_ok and any_miss_caught and reject_rate_ok:
        verdict = "FEASIBLE"
    elif schema_complexity_ok and precision_ok and reject_rate_ok:
        verdict = "FEASIBLE_WITH_CAVEATS"
    elif not schema_complexity_ok:
        verdict = "NOT_FEASIBLE"
    else:
        verdict = "NEEDS_DEEPER_SPIKE"

    print(f"\n{'=' * 70}")
    print(f"SPIKE VERDICT: {verdict}")
    print(f"{'=' * 70}")

    # Save raw responses for debugging
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    artifact_path = os.path.join(artifacts_dir, f"llm_authority_spike_{int(time.time())}.json")
    with open(artifact_path, "w") as f:
        json.dump({
            "model": OLLAMA_CHAT_MODEL,
            "fixture": "legal_deposition",
            "baseline": baseline,
            "results": results,
            "raw_responses_current": raw_responses_current,
            "raw_responses_authority": raw_responses_auth,
            "ground_truth": GROUND_TRUTH,
        }, f, indent=2)
    print(f"\nArtifacts saved: {artifact_path}")

    return verdict, results, q1_status, q2_status, q3_status, auth_prec


if __name__ == "__main__":
    verdict, results, q1, q2, q3, auth_prec = run_spike()
    baseline = 81.2

    print(f"""
---RESULT---
status: success
spike_verdict: {verdict}
q1_status: {q1}
q2_status: {q2} — precision {auth_prec:.1f}% vs {baseline}% baseline
q3_status: {q3} — {results['authority']['additional_surfaced']} surfaced
research_doc: .ai-docs/research/llm-as-authority-hybrid-extractor-spike-2026-06-09.md
spike_dir: spike/llm_as_authority_poc/
---END---
""")

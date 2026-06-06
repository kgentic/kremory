"""
TD-022 Q3 spike: GLiNER label-count degradation curve.
Measures ms/seq inference at 10 / 50 / 200 labels for gliner_small-v2.1.
"""

import time
import statistics

from gliner import GLiNER

TEXT = (
    "Ria moved from Morocco to Boston to attend Northeastern University in 2018. "
    "After graduation she worked at Amazon Robotics for two years before joining "
    "Boston Consulting Group in 2021. She specialises in supply chain analytics."
)

LABELS_10 = [
    "Person", "Organisation", "Location", "Date", "Time",
    "Money", "Quantity", "Event", "Concept", "Product",
]

LABELS_50 = LABELS_10 + [
    # Ground-truth fixture types
    "Court", "Drug", "Species", "Theory", "Publication", "Technology",
    "Tool", "Committee", "Feature", "Software", "Condition",
    # Filler types
    "Animal", "Building", "Movie", "Book", "Award", "Currency",
    "Disease", "Sport", "Language", "Country", "City", "River",
    "Planet", "Chemical", "Food", "Instrument", "Art", "Ideology",
    "Religion", "Ethnicity", "Occupation", "Vehicle", "Company",
    "University", "Hospital", "Airport", "Bridge", "Law", "Treaty",
]

# Pad to 200 with Wikidata-style types
LABELS_200 = LABELS_50 + [
    "Politician", "Athlete", "Lake", "Mountain", "Insect", "Mineral",
    "Volcano", "Galaxy", "Enzyme", "Protein", "Gene", "Virus",
    "Bacteria", "Algorithm", "Framework", "Library", "Protocol",
    "Standard", "Regulation", "Tax", "Tariff", "Bond", "Stock",
    "Index", "Commodity", "Derivative", "Mutual Fund", "Hedge Fund",
    "Startup", "Conglomerate", "Franchise", "Subsidiary",
    "Joint Venture", "Non-Profit", "Think Tank", "Lobby Group",
    "Trade Union", "Political Party", "Coalition", "Alliance",
    "International Organisation", "Government Agency", "Ministry",
    "Court Ruling", "Legislation", "Amendment", "Directive",
    "Regulation", "Decree", "Executive Order", "Referendum",
    "Election", "Debate", "Summit", "Conference", "Forum",
    "Exhibition", "Festival", "Competition", "Championship",
    "Tournament", "League", "Association", "Federation",
    "Consortium", "Cooperative", "Guild", "Brotherhood",
    "Order", "Sect", "Denomination", "Parish", "Diocese",
    "Monastery", "Temple", "Mosque", "Synagogue", "Cathedral",
    "Museum", "Gallery", "Library", "Archive", "Repository",
    "Database", "Registry", "Catalogue", "Index", "Directory",
    "Portal", "Platform", "Ecosystem", "Marketplace",
    "Exchange", "Auction", "Tender", "Procurement",
    "Contract", "Agreement", "Memorandum", "Charter",
    "Constitution", "Statute", "Ordinance", "By-Law",
    "Guideline", "Policy", "Strategy", "Framework",
    "Methodology", "Approach", "Model", "Paradigm",
    "Theory", "Hypothesis", "Theorem", "Lemma",
    "Corollary", "Axiom", "Postulate", "Principle",
    "Concept", "Notion", "Idea", "Theme", "Motif",
    "Pattern", "Trend", "Tendency", "Phenomenon",
    "Effect", "Impact", "Outcome", "Result",
    "Process", "Procedure", "Workflow", "Pipeline",
    "Architecture", "Infrastructure", "Network",
    "System", "Platform", "Environment",
    "Landscape", "Ecosystem", "Domain",
    "Sector", "Industry", "Market",
    "Segment", "Niche", "Category",
    "Subcategory", "Type", "Kind", "Variant",
    "Version", "Edition", "Release", "Update",
    "Patch", "Hotfix", "Feature", "Enhancement",
    "Bugfix", "Rollback",
]

# Trim to exactly 200
LABELS_200 = LABELS_200[:200]

assert len(LABELS_10) == 10, f"Expected 10 got {len(LABELS_10)}"
assert len(LABELS_50) == 50, f"Expected 50 got {len(LABELS_50)}"
assert len(LABELS_200) == 200, f"Expected 200 got {len(LABELS_200)}"

MODEL_NAME = "urchade/gliner_small-v2.1"
N_WARMUP = 1
N_MEASURE = 5


def run_curve(model_name: str, labels_sets: list[tuple[int, list[str]]]) -> None:
    print(f"\nLoading {model_name} ...")
    model = GLiNER.from_pretrained(model_name)
    print("Model loaded.\n")

    results = {}
    for n_labels, labels in labels_sets:
        # Warm-up
        for _ in range(N_WARMUP):
            model.predict_entities(TEXT, labels)

        # Measure
        times_ms = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            entities = model.predict_entities(TEXT, labels)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        mean_ms = statistics.mean(times_ms)
        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        results[n_labels] = (mean_ms, stdev_ms, entities)

        first5 = entities[:5]
        entity_strs = ", ".join(
            f"{e['text']} ({e['label']})" for e in first5
        )
        print(
            f"  {n_labels:3d} labels: mean={mean_ms:6.1f} ms  stdev={stdev_ms:5.1f} ms"
            f"  | found {len(entities)} entities — [{entity_strs}]"
        )

    # Summary
    mean_10 = results[10][0]
    mean_200 = results[200][0]
    slowdown = mean_200 / mean_10 if mean_10 > 0 else float("inf")

    print(f"\n{model_name} latency curve:")
    for n_labels, (mean_ms, stdev_ms, _) in results.items():
        budget_ok = "PASS" if mean_ms < 500 else "FAIL"
        print(f"  {n_labels:3d} labels:  mean={mean_ms:6.1f} ms (stdev={stdev_ms:.1f})  [{budget_ok} <500ms]")
    print(f"  → degradation 10→200: {slowdown:.1f}x slowdown")


if __name__ == "__main__":
    label_sets = [
        (10, LABELS_10),
        (50, LABELS_50),
        (200, LABELS_200),
    ]
    run_curve(MODEL_NAME, label_sets)

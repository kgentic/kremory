#!/usr/bin/env python3
"""RQL MetricsExporter JSON report tool.

Usage:
  python3 scripts/metrics_report.py --latest
  python3 scripts/metrics_report.py --compare logs/A.json logs/B.json
  python3 scripts/metrics_report.py --history retrieval-benchmark
"""

import argparse
import json
import sys
from pathlib import Path

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

LAYER_PREFIXES = ["rql.db.", "rql.search.", "rql.extraction.", "rql.llm.", "rql.ingest."]
LOGS_DIR = Path(__file__).parent.parent / "logs"
REGRESSION_WARN = 0.10
REGRESSION_FAIL = 0.20
SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sv):
        return sv[lo]
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def layer_for(name: str) -> str:
    for prefix in LAYER_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip(".")
    return "rql.other"


def fmt(v: float) -> str:
    return f"{v:.3f}"


def col(text: str, width: int) -> str:
    return text[:width].ljust(width)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_metrics_files(label: str | None = None) -> list[Path]:
    pattern = f"*-{label}-metrics.json" if label else "*-metrics.json"
    return sorted(LOGS_DIR.glob(pattern))


def find_latest() -> Path | None:
    files = find_metrics_files()
    return files[-1] if files else None


def grouped_histograms(histograms: dict) -> dict[str, list[tuple[str, dict]]]:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for name, h in sorted(histograms.items()):
        groups.setdefault(layer_for(name), []).append((name, h))
    return groups


def cmd_latest() -> int:
    path = find_latest()
    if path is None:
        print(f"{RED}No *-metrics.json files found in {LOGS_DIR}{RESET}", file=sys.stderr)
        return 1

    data = load_json(path)
    print(f"\n{BOLD}=== Latest run: {data.get('label', path.stem)} ==={RESET}")
    print(f"{DIM}File: {path.name}  |  Timestamp: {data.get('timestamp_iso', '?')}{RESET}\n")

    histograms = data.get("histograms", {})
    if histograms:
        cols = ["name", "count", "mean", "min", "p50", "p99", "max"]
        widths = [42, 7, 10, 10, 10, 10, 10]
        print(f"{BOLD}{'  '.join(col(c.upper(), w) for c, w in zip(cols, widths))}{RESET}")
        print("  ".join("-" * w for w in widths))
        for layer, entries in sorted(grouped_histograms(histograms).items()):
            print(f"\n{BOLD}{DIM}{layer}{RESET}")
            for name, h in entries:
                vals = h.get("values", [])
                row = [name, str(h.get("count", 0)), fmt(h.get("mean", 0.0)),
                       fmt(h.get("min", 0.0)), fmt(percentile(vals, 50)),
                       fmt(percentile(vals, 99)), fmt(h.get("max", 0.0))]
                print("  ".join(col(v, w) for v, w in zip(row, widths)))
    else:
        print(f"{DIM}(no histograms){RESET}")

    for section, items in [("COUNTERS", data.get("counters", {})),
                            ("GAUGES", data.get("gauges", {}))]:
        if items:
            print(f"\n{BOLD}{section}{RESET}")
            for k, v in sorted(items.items()):
                print(f"  {k}: {v}")

    print()
    return 0


def cmd_compare(file1: str, file2: str) -> int:
    p1, p2 = Path(file1), Path(file2)
    for p in (p1, p2):
        if not p.exists():
            print(f"{RED}File not found: {p}{RESET}", file=sys.stderr)
            return 1

    d1, d2 = load_json(p1), load_json(p2)
    h1, h2 = d1.get("histograms", {}), d2.get("histograms", {})

    print(f"\n{BOLD}=== Comparison ==={RESET}")
    print(f"  {DIM}Baseline  : {d1.get('label', p1.stem)} ({d1.get('timestamp_iso', '?')}){RESET}")
    print(f"  {DIM}Candidate : {d2.get('label', p2.stem)} ({d2.get('timestamp_iso', '?')}){RESET}\n")

    keys_both = sorted(set(h1) & set(h2))
    keys_added = sorted(set(h2) - set(h1))
    keys_removed = sorted(set(h1) - set(h2))
    any_fail = False

    if keys_both:
        cols = ["name", "baseline_mean", "candidate_mean", "delta_pct", "status"]
        widths = [42, 14, 14, 11, 10]
        print(f"{BOLD}{'  '.join(col(c.upper(), w) for c, w in zip(cols, widths))}{RESET}")
        print("  ".join("-" * w for w in widths))

        groups: dict[str, list[str]] = {}
        for name in keys_both:
            groups.setdefault(layer_for(name), []).append(name)

        for layer, names in sorted(groups.items()):
            print(f"\n{BOLD}{DIM}{layer}{RESET}")
            for name in names:
                m1 = h1[name].get("mean", 0.0)
                m2 = h2[name].get("mean", 0.0)
                chg = None if m1 == 0 else (m2 - m1) / m1

                if chg is None:
                    chg_str, status_str = "    n/a", f"{DIM}n/a{RESET}"
                elif chg > REGRESSION_FAIL:
                    any_fail = True
                    chg_str = f"{RED}+{chg*100:.1f}%{RESET}"
                    status_str = f"{RED}REGRESS{RESET}"
                elif chg > REGRESSION_WARN:
                    chg_str = f"{YELLOW}+{chg*100:.1f}%{RESET}"
                    status_str = f"{YELLOW}WARN{RESET}"
                elif chg < -REGRESSION_WARN:
                    chg_str = f"{GREEN}{chg*100:.1f}%{RESET}"
                    status_str = f"{GREEN}IMPROVE{RESET}"
                else:
                    chg_str = f"{chg*100:+.1f}%"
                    status_str = f"{DIM}ok{RESET}"

                parts = [col(name, widths[0]), col(fmt(m1), widths[1]),
                         col(fmt(m2), widths[2]), chg_str.ljust(widths[3]), status_str]
                print("  ".join(parts))

    if keys_added:
        print(f"\n{GREEN}New metrics in candidate:{RESET}")
        for k in keys_added:
            print(f"  + {k}")
    if keys_removed:
        print(f"\n{YELLOW}Metrics removed in candidate:{RESET}")
        for k in keys_removed:
            print(f"  - {k}")

    print()
    if any_fail:
        print(f"{RED}FAIL: one or more metrics regressed >{REGRESSION_FAIL*100:.0f}%{RESET}\n")
        return 1
    print(f"{GREEN}PASS: no regressions above threshold{RESET}\n")
    return 0


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    return "".join(
        SPARKLINE_CHARS[int((v - lo) / span * (len(SPARKLINE_CHARS) - 1)) if span else 4]
        for v in values
    )


def cmd_history(label: str) -> int:
    files = find_metrics_files(label)
    if not files:
        print(f"{RED}No metrics files matching label '{label}' in {LOGS_DIR}{RESET}", file=sys.stderr)
        return 1

    runs = sorted([load_json(p) for p in files], key=lambda d: d.get("timestamp_unix", 0))
    print(f"\n{BOLD}=== History: {label} ({len(runs)} run{'s' if len(runs) != 1 else ''}) ==={RESET}\n")

    key_suffixes = ("_ms", "_hits")
    all_names: set[str] = set()
    for run in runs:
        for name in run.get("histograms", {}):
            if any(name.endswith(s) for s in key_suffixes):
                all_names.add(name)
    if not all_names:
        for run in runs:
            all_names.update(run.get("histograms", {}).keys())

    for name in sorted(all_names):
        means, timestamps = [], []
        for run in runs:
            h = run.get("histograms", {}).get(name)
            if h:
                means.append(h.get("mean", 0.0))
                timestamps.append(run.get("timestamp_iso", "?")[:10])
        if not means:
            continue

        best, worst, latest = min(means), max(means), means[-1]
        print(f"  {BOLD}{name}{RESET}")
        print(f"    trend  : {sparkline(means)}")
        print(f"    best   : {GREEN}{fmt(best)}{RESET}  "
              f"worst: {RED}{fmt(worst)}{RESET}  latest: {BOLD}{fmt(latest)}{RESET}")
        if len(runs) <= 10:
            for ts, mean_val in zip(timestamps, means):
                bar = "#" * min(40, max(1, int(mean_val / worst * 20))) if worst > 0 else ""
                print(f"    {DIM}{ts}{RESET}  {fmt(mean_val):>10}  {bar}")
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RQL metrics report tool")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--latest", action="store_true", help="Show latest run summary (default)")
    group.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                       help="Compare two metric JSON files")
    group.add_argument("--history", metavar="LABEL", help="Show trend history for a label")
    args = parser.parse_args()

    if args.compare:
        return cmd_compare(args.compare[0], args.compare[1])
    if args.history:
        return cmd_history(args.history)
    return cmd_latest()


if __name__ == "__main__":
    sys.exit(main())

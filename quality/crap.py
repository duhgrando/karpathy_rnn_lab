"""Compute an approximate CRAP (Change Risk Anti-Patterns) score for every
function in domain/, application/, and tests/:

    CRAP(m) = complexity(m)^2 * (1 - coverage(m))^3 + complexity(m)

There's no maintained CRAP tool for Python (crap4j targets Java), so this
composes two well-known ones instead: `radon` for per-function cyclomatic
complexity, and `coverage.py` for line coverage, then applies the published
formula from the original CRAP paper (Alberto Savoia / Bob Evans, 2007).

Usage:
    coverage run --branch -m pytest
    coverage json -o coverage.json
    python quality/crap.py            # prints a table, exits 1 if any
                                       # function's CRAP score >= THRESHOLD

Only tested against radon's documented `cc -j` JSON shape (each function
entry has name / lineno / endline / complexity). If radon changes that
shape in a future release and this script errors on a missing key, the
fix is to inspect `radon cc -j domain application` directly and adjust
_run_radon()/_function_coverage() to match.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

THRESHOLD = 5.0
SOURCE_DIRS = ["domain", "application", "tests"]


def _run_radon() -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "-j", *SOURCE_DIRS],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return [
        {**entry, "file": file_path}
        for file_path, entries in data.items()
        for entry in entries
    ]


def _load_coverage() -> dict:
    coverage_json = Path("coverage.json")
    if not coverage_json.exists():
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
            check=True,
        )
    return json.loads(coverage_json.read_text())["files"]


def _function_coverage(func: dict, file_coverage: dict) -> float:
    start, end = func["lineno"], func["endline"]
    executed = set(file_coverage.get("executed_lines", []))
    missing = set(file_coverage.get("missing_lines", []))
    relevant = {line for line in executed | missing if start <= line <= end}
    if not relevant:
        return 1.0  # nothing measurable (e.g. a one-line signature) -> don't punish it
    return len(relevant & executed) / len(relevant)


def crap_score(complexity: int, coverage_fraction: float) -> float:
    return complexity ** 2 * (1 - coverage_fraction) ** 3 + complexity


def main() -> int:
    functions = _run_radon()
    coverage_by_file = _load_coverage()

    rows = []
    for func in functions:
        file_coverage = coverage_by_file.get(func["file"], {})
        coverage_fraction = _function_coverage(func, file_coverage)
        score = crap_score(func["complexity"], coverage_fraction)
        rows.append((func["file"], func["name"], func["complexity"], coverage_fraction, score))
    rows.sort(key=lambda row: row[-1], reverse=True)

    print(f"{'file':40} {'function':38} {'cc':>3} {'cov':>6} {'CRAP':>6}")
    for file_name, name, complexity, coverage_fraction, score in rows:
        flag = "  !!" if score >= THRESHOLD else ""
        print(f"{file_name:40} {name:38} {complexity:3d} {coverage_fraction:6.0%} {score:6.2f}{flag}")

    worst = max((row[-1] for row in rows), default=0.0)
    if worst >= THRESHOLD:
        print(f"\nFAIL: worst CRAP score {worst:.2f} >= {THRESHOLD}")
        return 1
    print(f"\nOK: worst CRAP score {worst:.2f} < {THRESHOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

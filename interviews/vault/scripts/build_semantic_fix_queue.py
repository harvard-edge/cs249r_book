#!/usr/bin/env python3
"""Build a prioritized fix queue from semantic audit JSONL results."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

VAULT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = VAULT_DIR / "audit" / "semantic-review-results"
DEFAULT_OUT = VAULT_DIR / "audit" / "semantic-review-results" / "fix_queue.md"
CHECKS = (
    "scenario_question_fit",
    "answer_correctness",
    "common_mistake_quality",
    "napkin_math_correctness",
    "physical_plausibility",
    "level_fit",
    "title_quality",
)
SEVERITY_WEIGHT = {"blocker": 300, "major": 200, "minor": 100, "none": 0}


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def score_row(row: dict[str, Any]) -> tuple[int, int, str]:
    failing_checks = sum(1 for key in CHECKS if row.get(key) == "fail")
    severity = SEVERITY_WEIGHT.get(str(row.get("severity", "none")), 0)
    return (severity + failing_checks, failing_checks, str(row.get("qid")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=200)
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        paths = sorted(
            path for path in RESULTS_DIR.glob("*_semantic_findings.jsonl") if not path.name.startswith("smoke_")
        )

    rows = read_rows(paths)
    needs_fix = [row for row in rows if row.get("verdict") == "needs_fix"]
    needs_fix.sort(key=score_row, reverse=True)

    by_track = collections.Counter(str(row["_path"].split("/")[4]) for row in needs_fix)
    by_severity = collections.Counter(str(row.get("severity", "none")) for row in needs_fix)
    by_check = collections.Counter()
    for row in needs_fix:
        for key in CHECKS:
            if row.get(key) == "fail":
                by_check[key] += 1

    lines = [
        "# Semantic Fix Queue",
        "",
        f"- Result files: {len(paths)}",
        f"- Needs fix: {len(needs_fix)}",
        "",
        "## Severity",
        "",
    ]
    for key, count in by_severity.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Tracks", ""])
    for key, count in by_track.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Failing Checks", ""])
    for key, count in by_check.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Top Items", ""])

    for row in needs_fix[: args.top]:
        qid = row.get("qid")
        path = row.get("_path")
        severity = row.get("severity")
        issues = row.get("issues", [])
        summary = " ".join(str(issue) for issue in issues[:2]) if issues else "No issues captured."
        lines.append(f"- `{qid}` [{path}]({Path(path).resolve()}) ({severity})")
        lines.append(f"  - {summary}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

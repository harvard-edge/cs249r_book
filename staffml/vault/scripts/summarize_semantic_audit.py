#!/usr/bin/env python3
"""Summarize semantic audit JSONL files into a release report."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

VAULT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = VAULT_DIR / "audit" / "semantic-review-results"
DEFAULT_OUT = RESULTS_DIR / "summary.md"
CHECK_KEYS = (
    "scenario_question_fit",
    "answer_correctness",
    "common_mistake_quality",
    "napkin_math_correctness",
    "physical_plausibility",
    "level_fit",
    "title_quality",
)


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = str(path)
                    rows.append(row)
    return rows


def markdown_counter(title: str, counter: collections.Counter[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counter:
        lines.append("- none")
    else:
        for key, count in counter.most_common():
            lines.append(f"- {key}: {count}")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        paths = sorted(
            path
            for path in RESULTS_DIR.glob("*.jsonl")
            if not path.name.startswith("smoke_")
        )

    rows = read_rows(paths)
    verdicts = collections.Counter(str(row.get("verdict", "missing")) for row in rows)
    severities = collections.Counter(str(row.get("severity", "missing")) for row in rows)
    statuses = collections.Counter(str(row.get("_audit_status", "ok")) for row in rows)
    failed_checks: collections.Counter[str] = collections.Counter()
    for row in rows:
        for key in CHECK_KEYS:
            if row.get(key) == "fail":
                failed_checks[key] += 1

    needs_fix = [
        row
        for row in rows
        if row.get("verdict") == "needs_fix" and row.get("_audit_status", "ok") == "ok"
    ]
    needs_fix.sort(key=lambda row: (str(row.get("severity")), str(row.get("qid"))))

    lines = [
        "# Semantic Audit Summary",
        "",
        f"- Result files: {len(paths)}",
        f"- Records read: {len(rows)}",
        "",
    ]
    lines.extend(markdown_counter("Audit Status", statuses))
    lines.extend(markdown_counter("Verdicts", verdicts))
    lines.extend(markdown_counter("Severities", severities))
    lines.extend(markdown_counter("Failed Checks", failed_checks))
    lines.extend(["## Needs Fix", ""])
    if not needs_fix:
        lines.append("- none")
    else:
        for row in needs_fix[: args.top]:
            issues = "; ".join(str(issue) for issue in row.get("issues", []))
            lines.append(
                f"- {row.get('qid')} ({row.get('severity')}, {row.get('_path')}): {issues}"
            )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

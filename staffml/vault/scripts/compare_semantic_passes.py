#!/usr/bin/env python3
"""Compare two semantic audit passes and summarize agreement."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

VAULT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = VAULT_DIR / "audit" / "semantic-review-results"
DEFAULT_OUT = RESULTS_DIR / "semantic_pass_compare.md"


def read_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["qid"])
                    rows[qid] = row
    return rows


def summarize_pair(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_verdict = left.get("verdict")
    right_verdict = right.get("verdict")
    left_sev = left.get("severity")
    right_sev = right.get("severity")
    if left_verdict == right_verdict and left_sev == right_sev:
        return "agree"
    if left_verdict != right_verdict and left_verdict == "needs_fix" and right_verdict == "needs_fix":
        return "disagree_severity"
    return "disagree"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", nargs="+", type=Path, required=True, help="First pass JSONL files")
    parser.add_argument("--right", nargs="+", type=Path, required=True, help="Second pass JSONL files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    left = read_rows(args.left)
    right = read_rows(args.right)
    qids = sorted(set(left) & set(right))

    verdict_counts = collections.Counter()
    severity_counts = collections.Counter()
    change_counts = collections.Counter()
    disagreements: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []

    for qid in qids:
        l = left[qid]
        r = right[qid]
        verdict_counts[(l.get("verdict"), r.get("verdict"))] += 1
        severity_counts[(l.get("severity"), r.get("severity"))] += 1
        label = summarize_pair(l, r)
        change_counts[label] += 1
        if label != "agree":
            left_fails = sum(1 for k in ("scenario_question_fit", "answer_correctness", "common_mistake_quality", "napkin_math_correctness", "physical_plausibility", "level_fit", "title_quality") if l.get(k) == "fail")
            right_fails = sum(1 for k in ("scenario_question_fit", "answer_correctness", "common_mistake_quality", "napkin_math_correctness", "physical_plausibility", "level_fit", "title_quality") if r.get(k) == "fail")
            score = left_fails + right_fails + (2 if l.get("severity") != r.get("severity") else 0)
            disagreements.append((score, qid, l, r))

    lines = [
        "# Semantic Pass Comparison",
        "",
        f"- Left files: {len(args.left)}",
        f"- Right files: {len(args.right)}",
        f"- Shared qids: {len(qids)}",
        "",
        "## Agreement",
        "",
    ]
    for key, count in change_counts.most_common():
        lines.append(f"- {key}: {count}")

    lines.extend(["", "## Verdict Pairs", ""])
    for (l_verdict, r_verdict), count in verdict_counts.most_common():
        lines.append(f"- {l_verdict} -> {r_verdict}: {count}")

    lines.extend(["", "## Severity Pairs", ""])
    for (l_sev, r_sev), count in severity_counts.most_common():
        lines.append(f"- {l_sev} -> {r_sev}: {count}")

    disagreements.sort(reverse=True)
    lines.extend(["", "## Top Disagreements", ""])
    for score, qid, l, r in disagreements[: args.top]:
        lines.append(
            f"- `{qid}` score={score} left={l.get('verdict')}/{l.get('severity')} right={r.get('verdict')}/{r.get('severity')}"
        )
        lines.append(f"  - left: {'; '.join(l.get('issues', [])[:2])}")
        lines.append(f"  - right: {'; '.join(r.get('issues', [])[:2])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

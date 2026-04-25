#!/usr/bin/env python3
"""Audit StaffML topic-track applicability sources.

Compares:
- schema enum topics,
- taxonomy_data.yaml topic-track applicability,
- paper app_matrix.tex topic labels,
- observed YAML topic-track pairs.

This report is advisory. It helps decide whether a sparse/invalid-looking cell
should be generated, retagged, or explicitly excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
TAXONOMY_DATA = VAULT_DIR / "schema" / "taxonomy_data.yaml"
PAPER_MATRIX = ROOT_DIR / "interviews" / "paper" / "tables" / "app_matrix.tex"
OUTPUT_DIR = VAULT_DIR / "_validation_results" / "gap_plan"

SCHEMA_DIR = VAULT_DIR / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from enums import VALID_TOPICS, VALID_TRACKS  # noqa: E402


def normalize_label(text: str) -> str:
    text = text.replace("\\&", "&")
    text = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def load_taxonomy() -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    data = yaml.safe_load(TAXONOMY_DATA.read_text(encoding="utf-8")) or {}
    topics = {topic["id"]: topic for topic in data.get("topics", [])}
    pairs = {
        (topic["id"], track)
        for topic in data.get("topics", [])
        for track in topic.get("tracks", [])
    }
    return topics, pairs


def load_observed_pairs() -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], list[str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in QUESTIONS_DIR.glob("*/*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pair = (data.get("topic", ""), data.get("track", ""))
        counts[pair] += 1
        if len(examples[pair]) < 5:
            examples[pair].append(str(path.relative_to(ROOT_DIR)))
    return counts, examples


def load_paper_topic_labels() -> set[str]:
    if not PAPER_MATRIX.exists():
        return set()
    labels: set[str] = set()
    for line in PAPER_MATRIX.read_text(encoding="utf-8").splitlines():
        if "&" not in line or "\\textbf" in line or "\\midrule" in line:
            continue
        cells = [cell.strip() for cell in line.split("&")]
        # Topic cells are columns 0, 5, and 10 in the matrix.
        for idx in (0, 5, 10):
            if idx < len(cells):
                label = normalize_label(cells[idx])
                if label:
                    labels.add(label)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_topics, taxonomy_pairs = load_taxonomy()
    observed_counts, observed_examples = load_observed_pairs()
    paper_labels = load_paper_topic_labels()

    schema_topics = set(VALID_TOPICS)
    taxonomy_topic_ids = set(taxonomy_topics)
    missing_from_taxonomy = sorted(schema_topics - taxonomy_topic_ids)
    extra_in_taxonomy = sorted(taxonomy_topic_ids - schema_topics)

    taxonomy_names_missing_from_paper = []
    for topic_id, topic in taxonomy_topics.items():
        if normalize_label(topic.get("name", topic_id)) not in paper_labels:
            taxonomy_names_missing_from_paper.append(
                {"topic": topic_id, "name": topic.get("name", topic_id)}
            )

    observed_not_applicable = []
    for pair, count in sorted(observed_counts.items(), key=lambda item: (-item[1], item[0])):
        topic, track = pair
        if track == "global":
            continue
        if topic in schema_topics and track in VALID_TRACKS and pair not in taxonomy_pairs:
            observed_not_applicable.append(
                {
                    "topic": topic,
                    "track": track,
                    "count": count,
                    "examples": observed_examples[pair],
                    "decision_needed": "mark_applicable_with_rationale_or_retag",
                }
            )

    applicable_with_no_observed = []
    for topic, track in sorted(taxonomy_pairs):
        if observed_counts.get((topic, track), 0) == 0:
            applicable_with_no_observed.append({"topic": topic, "track": track})

    report = {
        "schema_topic_count": len(schema_topics),
        "taxonomy_data_topic_count": len(taxonomy_topic_ids),
        "taxonomy_pair_count": len(taxonomy_pairs),
        "observed_pair_count": len(observed_counts),
        "paper_topic_label_count": len(paper_labels),
        "schema_topics_missing_from_taxonomy_data": missing_from_taxonomy,
        "taxonomy_topics_not_in_schema": extra_in_taxonomy,
        "taxonomy_topics_missing_from_paper_matrix_by_name": taxonomy_names_missing_from_paper,
        "observed_pairs_not_applicable": observed_not_applicable,
        "applicable_pairs_with_zero_observed_questions": applicable_with_no_observed,
    }
    (output_dir / "applicability_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# StaffML Applicability Matrix Audit",
        "",
        f"- Schema topics: {len(schema_topics)}",
        f"- taxonomy_data topics: {len(taxonomy_topic_ids)}",
        f"- taxonomy_data topic-track pairs: {len(taxonomy_pairs)}",
        f"- observed topic-track pairs: {len(observed_counts)}",
        f"- paper matrix topic labels parsed: {len(paper_labels)}",
        "",
        "## Schema Topics Missing From taxonomy_data",
        "",
    ]
    lines.extend(f"- `{topic}`" for topic in missing_from_taxonomy[:100])
    lines.extend(
        [
            "",
            "## Observed Pairs Not Marked Applicable",
            "",
            "| count | topic | track | examples |",
            "|---:|---|---|---|",
        ]
    )
    for item in observed_not_applicable[:120]:
        lines.append(
            f"| {item['count']} | `{item['topic']}` | `{item['track']}` | "
            + ", ".join(f"`{ex}`" for ex in item["examples"][:3])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Applicable Pairs With Zero Observed Questions",
            "",
            "| topic | track |",
            "|---|---|",
        ]
    )
    for item in applicable_with_no_observed[:120]:
        lines.append(f"| `{item['topic']}` | `{item['track']}` |")
    (output_dir / "applicability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote applicability audit to {output_dir}")
    print(f"Observed-not-applicable pairs: {len(observed_not_applicable)}")
    print(f"Applicable zero-observed pairs: {len(applicable_with_no_observed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

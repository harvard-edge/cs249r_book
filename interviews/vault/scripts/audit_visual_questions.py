#!/usr/bin/env python3
"""Audit and plan StaffML visual-question coverage."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
VISUALS_DIR = VAULT_DIR / "visuals"
OUTPUT_DIR = VAULT_DIR / "_validation_results" / "gap_plan"

VISUAL_ARCHETYPES: dict[str, dict[str, Any]] = {
    "collective-communication": {
        "visual": "ring/tree collective diagram",
        "why": "Communication algorithms are easier to reason about from topology and chunk-flow diagrams.",
    },
    "pipeline-parallelism": {
        "visual": "pipeline bubble timeline",
        "why": "Bubble fractions and stage imbalance are visual temporal phenomena.",
    },
    "kv-cache-management": {
        "visual": "KV-cache block/page layout",
        "why": "Paged vs contiguous allocation and fragmentation benefit from spatial diagrams.",
    },
    "queueing-theory": {
        "visual": "queueing hockey-stick curve",
        "why": "Tail latency growth near saturation is best conveyed with curves.",
    },
    "data-pipeline-engineering": {
        "visual": "data pipeline throughput diagram",
        "why": "Bottlenecks arise from stage rates and buffers.",
    },
    "memory-hierarchy-design": {
        "visual": "memory hierarchy data path",
        "why": "Latency/bandwidth tiers are hierarchical and spatial.",
    },
    "interconnect-topology": {
        "visual": "topology placement diagram",
        "why": "Bisection, locality, and placement require topology reasoning.",
    },
    "network-bandwidth-bottlenecks": {
        "visual": "fanout or bandwidth budget diagram",
        "why": "Traffic multiplication is clearer as a graph of edges and payloads.",
    },
    "duty-cycling": {
        "visual": "sleep/wake duty-cycle timeline",
        "why": "Average power is area under a time/power curve.",
    },
    "fault-tolerance-checkpointing": {
        "visual": "checkpoint/recovery timeline",
        "why": "RPO/RTO and lost work are timeline concepts.",
    },
}


def load_questions() -> list[dict[str, Any]]:
    questions = []
    for path in QUESTIONS_DIR.glob("*/*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["_path"] = str(path.relative_to(ROOT_DIR))
        questions.append(data)
    return questions


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    visual_questions = [q for q in questions if q.get("visual")]
    by_topic = Counter(q.get("topic") for q in questions)
    visual_by_topic = Counter(q.get("topic") for q in visual_questions)

    visual_items = []
    issues = []
    for q in visual_questions:
        visual = q.get("visual") or {}
        track = q.get("track", "")
        path = visual.get("path", "")
        asset = VISUALS_DIR / track / path
        item = {
            "id": q.get("id"),
            "status": q.get("status"),
            "track": track,
            "topic": q.get("topic"),
            "level": q.get("level"),
            "zone": q.get("zone"),
            "path": q.get("_path"),
            "visual_path": path,
            "asset_exists": asset.exists(),
            "has_alt": bool((visual.get("alt") or "").strip()),
            "has_caption": bool((visual.get("caption") or "").strip()),
            "question_references_visual": any(
                word in (q.get("question") or "").lower() for word in ("diagram", "figure", "visual")
            ),
        }
        visual_items.append(item)
        if not item["asset_exists"]:
            issues.append({"id": item["id"], "issue": "missing_asset", "path": str(asset.relative_to(VAULT_DIR))})
        if not item["has_alt"]:
            issues.append({"id": item["id"], "issue": "missing_alt"})
        if not item["question_references_visual"]:
            issues.append({"id": item["id"], "issue": "question_does_not_reference_visual"})

    candidates = []
    for topic, spec in VISUAL_ARCHETYPES.items():
        total = by_topic.get(topic, 0)
        visual_count = visual_by_topic.get(topic, 0)
        candidates.append(
            {
                "topic": topic,
                "total_questions": total,
                "visual_questions": visual_count,
                "recommended_visual": spec["visual"],
                "why": spec["why"],
                "priority": "high" if total >= 50 and visual_count == 0 else "medium",
            }
        )
    candidates.sort(key=lambda row: (row["visual_questions"], row["priority"] != "high", -row["total_questions"]))

    report = {
        "total_questions": len(questions),
        "visual_question_count": len(visual_questions),
        "visual_items": visual_items,
        "issues": issues,
        "candidate_topics": candidates,
    }
    (OUTPUT_DIR / "visual_question_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# StaffML Visual Question Audit",
        "",
        f"- Total YAML questions: {len(questions)}",
        f"- Visual questions: {len(visual_questions)}",
        f"- Issues: {len(issues)}",
        "",
        "## Existing Visual Questions",
        "",
        "| id | status | track | topic | asset | alt | caption | question references visual |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in visual_items:
        lines.append(
            f"| `{item['id']}` | {item['status']} | {item['track']} | `{item['topic']}` | "
            f"{item['asset_exists']} | {item['has_alt']} | {item['has_caption']} | "
            f"{item['question_references_visual']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Visual Topics",
            "",
            "| priority | topic | total questions | current visuals | recommended visual | why |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for item in candidates:
        lines.append(
            f"| {item['priority']} | `{item['topic']}` | {item['total_questions']} | "
            f"{item['visual_questions']} | {item['recommended_visual']} | {item['why']} |"
        )
    (OUTPUT_DIR / "visual_question_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote visual audit to {OUTPUT_DIR}")
    print(f"Visual questions: {len(visual_questions)}; candidate topics: {len(candidates)}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

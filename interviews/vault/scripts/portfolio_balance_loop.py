#!/usr/bin/env python3
"""Plan iterative StaffML portfolio-balancing passes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
OUTPUT_DIR = VAULT_DIR / "_validation_results" / "portfolio_loop"

SCHEMA_DIR = VAULT_DIR / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from enums import VALID_COMPETENCY_AREAS, VALID_LEVELS, VALID_TRACKS, VALID_ZONES  # noqa: E402

TRACK_WEIGHTS = {"global": 1.25, "tinyml": 1.15, "mobile": 1.10, "edge": 1.05, "cloud": 0.90}
ZONE_WEIGHTS = {
    "realization": 1.35,
    "specification": 1.25,
    "mastery": 1.20,
    "evaluation": 1.10,
    "diagnosis": 1.05,
    "implement": 0.85,
    "recall": 0.75,
}
LEVEL_WEIGHTS = {"L5": 1.20, "L6+": 1.20, "L4": 1.10, "L3": 1.00, "L2": 0.85, "L1": 0.75}


def load_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for path in QUESTIONS_DIR.rglob('*.yaml'):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["_path"] = str(path.relative_to(ROOT_DIR))
        questions.append(data)
    return questions


def gini(values: list[int]) -> float:
    values = sorted(v for v in values if v >= 0)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    n = len(values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def cv(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / mean


def suggested_zone(level: str) -> str:
    return {
        "L1": "recall",
        "L2": "fluency",
        "L3": "fluency",
        "L4": "diagnosis",
        "L5": "realization",
        "L6+": "mastery",
    }.get(level, "fluency")


def suggested_level(zone: str) -> str:
    if zone in {"implement", "fluency"}:
        return "L3"
    if zone in {"analyze", "diagnosis", "optimization"}:
        return "L4"
    if zone in {"design", "evaluation", "realization", "specification"}:
        return "L5"
    if zone == "mastery":
        return "L6+"
    return "L2"


def level_score(track: str, area: str, level: str, count: int) -> float:
    target = 8 if level in {"L1", "L2"} else 12
    area_weight = 1.15 if area in {"parallelism", "precision", "networking"} else 1.0
    return max(0, target - count) * TRACK_WEIGHTS.get(track, 1.0) * LEVEL_WEIGHTS.get(level, 1.0) * area_weight


def zone_score(track: str, area: str, zone: str, count: int) -> float:
    target = 8 if zone in {"recall", "implement"} else 10
    area_weight = 1.15 if area in {"parallelism", "precision", "networking"} else 1.0
    return max(0, target - count) * TRACK_WEIGHTS.get(track, 1.0) * ZONE_WEIGHTS.get(zone, 1.0) * area_weight


def build_plan(questions: list[dict[str, Any]], iterations: int, targets_per_iteration: int) -> dict[str, Any]:
    track_area_level = Counter((q.get("track", ""), q.get("competency_area", ""), q.get("level", "")) for q in questions)
    track_area_zone = Counter((q.get("track", ""), q.get("competency_area", ""), q.get("zone", "")) for q in questions)

    candidates: list[dict[str, Any]] = []
    for track in sorted(VALID_TRACKS):
        for area in sorted(VALID_COMPETENCY_AREAS):
            for level in sorted(VALID_LEVELS):
                count = track_area_level.get((track, area, level), 0)
                score = level_score(track, area, level, count)
                if score > 0:
                    candidates.append(
                        {
                            "kind": "track_area_level",
                            "track": track,
                            "competency_area": area,
                            "level": level,
                            "zone": suggested_zone(level),
                            "count": count,
                            "score": round(score, 3),
                        }
                    )
            for zone in sorted(VALID_ZONES):
                count = track_area_zone.get((track, area, zone), 0)
                score = zone_score(track, area, zone, count)
                if score > 0:
                    candidates.append(
                        {
                            "kind": "track_area_zone",
                            "track": track,
                            "competency_area": area,
                            "zone": zone,
                            "level": suggested_level(zone),
                            "count": count,
                            "score": round(score, 3),
                        }
                    )

    ranked = sorted(candidates, key=lambda item: (-item["score"], item["kind"], item["track"], item["competency_area"]))
    plan = []
    cursor = 0
    used: set[tuple[str, str, str, str, str]] = set()
    for iteration in range(1, iterations + 1):
        targets = []
        while cursor < len(ranked) and len(targets) < targets_per_iteration:
            target = ranked[cursor]
            cursor += 1
            key = (target["kind"], target["track"], target["competency_area"], target["level"], target["zone"])
            if key in used:
                continue
            used.add(key)
            targets.append(target)
        plan.append(
            {
                "iteration": iteration,
                "targets": targets,
                "stop_rule": "Generate only if a target still has fewer than the threshold count after validation.",
            }
        )

    level_values = [track_area_level.get((t, a, l), 0) for t in VALID_TRACKS for a in VALID_COMPETENCY_AREAS for l in VALID_LEVELS]
    zone_values = [track_area_zone.get((t, a, z), 0) for t in VALID_TRACKS for a in VALID_COMPETENCY_AREAS for z in VALID_ZONES]
    return {
        "baseline": {
            "questions": len(questions),
            "track_area_level": {
                "cells": len(level_values),
                "zero_cells": sum(1 for value in level_values if value == 0),
                "min": min(level_values),
                "max": max(level_values),
                "mean": round(sum(level_values) / len(level_values), 3),
                "cv": round(cv(level_values), 3),
                "gini": round(gini(level_values), 3),
            },
            "track_area_zone": {
                "cells": len(zone_values),
                "zero_cells": sum(1 for value in zone_values if value == 0),
                "min": min(zone_values),
                "max": max(zone_values),
                "mean": round(sum(zone_values) / len(zone_values), 3),
                "cv": round(cv(zone_values), 3),
                "gini": round(gini(zone_values), 3),
            },
        },
        "ranked_targets": ranked,
        "iterations": plan,
    }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(result["baseline"], indent=2) + "\n", encoding="utf-8")
    (out_dir / "ranked_targets.json").write_text(json.dumps(result["ranked_targets"], indent=2) + "\n", encoding="utf-8")
    (out_dir / "twenty_iteration_plan.json").write_text(json.dumps(result["iterations"], indent=2) + "\n", encoding="utf-8")

    lines = ["# StaffML Portfolio Balance Loop", ""]
    lines.append("## Baseline")
    for axis, stats in result["baseline"].items():
        if axis == "questions":
            lines.append(f"- Questions: {stats}")
        else:
            lines.append(
                f"- {axis}: zero_cells={stats['zero_cells']}, min={stats['min']}, "
                f"max={stats['max']}, cv={stats['cv']}, gini={stats['gini']}"
            )
    lines.extend(["", "## First 20 Iterations"])
    for item in result["iterations"]:
        lines.append("")
        lines.append(f"### Iteration {item['iteration']}")
        for target in item["targets"]:
            lines.append(
                f"- {target['kind']}: {target['track']} / {target['competency_area']} / "
                f"{target['level']} / {target['zone']} (count={target['count']}, score={target['score']})"
            )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--targets-per-iteration", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    result = build_plan(load_questions(), args.iterations, args.targets_per_iteration)
    write_outputs(result, args.output_dir)
    baseline = result["baseline"]
    print(f"Wrote portfolio loop artifacts to {args.output_dir}")
    print(f"Questions: {baseline['questions']}")
    print(f"track_area_level zero cells: {baseline['track_area_level']['zero_cells']}")
    print(f"track_area_zone zero cells: {baseline['track_area_zone']['zero_cells']}")


if __name__ == "__main__":
    main()

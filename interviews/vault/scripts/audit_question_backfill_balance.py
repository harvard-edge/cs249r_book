#!/usr/bin/env python3
"""Audit StaffML `question` field coverage and balance after backfill."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"

HARDWARE_NAMES = [
    "H100",
    "A100",
    "MI300X",
    "V100",
    "TPU",
    "Jetson",
    "Orin",
    "Hailo",
    "Hexagon",
    "Snapdragon",
    "Exynos",
    "Tensor G3",
    "A17",
    "Neural Engine",
    "Cortex-M4",
    "Cortex-M7",
    "Cortex-M33",
    "Ethos-U55",
    "ESP32",
    "nRF5340",
    "STM32",
    "Corstone",
]

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:"
    r"GB/s|TB/s|GB|MB|KB|B|TOPS|TFLOPS|FLOPS|ms|µs|us|s|mW|W|MHz|GHz|"
    r"mAh|mWh|tokens?|layers?|heads?|fps|FPS|%|x"
    r")?\b",
    re.I,
)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def scenario_anchors(scenario: str) -> set[str]:
    anchors: set[str] = set()
    for match in NUMBER_RE.findall(scenario):
        value = normalize(match)
        if value and any(ch.isdigit() for ch in value):
            anchors.add(value)
    scenario_lower = scenario.lower()
    for name in HARDWARE_NAMES:
        if name.lower() in scenario_lower:
            anchors.add(name.lower())
    return anchors


def question_mentions_anchor(question: str, anchors: set[str]) -> bool:
    q = normalize(question)
    compact = q.replace(" ", "")
    for anchor in anchors:
        a = normalize(anchor)
        if a in q or a.replace(" ", "") in compact:
            return True
    return False


def summarize(group: list[dict], corpus_median: float) -> dict:
    lengths = [len(q["question"]) for q in group if q["question"]]
    missing = sum(1 for q in group if not q["question"])
    eligible = 0
    anchor_hits = 0
    for q in group:
        anchors = q["anchors"]
        if not anchors or not q["question"]:
            continue
        eligible += 1
        if question_mentions_anchor(q["question"], anchors):
            anchor_hits += 1
    mean_len = statistics.mean(lengths) if lengths else 0.0
    median_len = statistics.median(lengths) if lengths else 0.0
    deviation = (mean_len - corpus_median) / corpus_median if corpus_median else 0.0
    return {
        "count": len(group),
        "populated": len(group) - missing,
        "missing": missing,
        "mean_length": round(mean_len, 1),
        "median_length": round(median_len, 1),
        "mean_length_deviation_from_corpus_median": round(deviation, 3),
        "length_flag": abs(deviation) > 0.30,
        "anchor_eligible": eligible,
        "anchor_hits": anchor_hits,
        "anchor_fraction": round(anchor_hits / eligible, 3) if eligible else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="_validation_results/question_backfill_balance_report.json",
        help="Path relative to interviews/vault for JSON report.",
    )
    args = parser.parse_args()

    records: list[dict] = []
    for path in sorted(QUESTIONS_DIR.glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        question = (data.get("question") or "").strip()
        scenario = data.get("scenario") or ""
        records.append(
            {
                "path": str(path.relative_to(VAULT_DIR)),
                "id": data.get("id", path.stem),
                "track": data.get("track", ""),
                "level": data.get("level", ""),
                "topic": data.get("topic", ""),
                "question": question,
                "anchors": scenario_anchors(str(scenario)),
            }
        )

    lengths = [len(r["question"]) for r in records if r["question"]]
    corpus_median = statistics.median(lengths) if lengths else 0.0

    by_track: dict[str, list[dict]] = defaultdict(list)
    by_level: dict[str, list[dict]] = defaultdict(list)
    by_track_level: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_track[record["track"]].append(record)
        by_level[record["level"]].append(record)
        by_track_level[f"{record['track']}:{record['level']}"].append(record)

    duplicate_exact = 0
    for record in records:
        if not record["question"]:
            continue
        path = VAULT_DIR / record["path"]
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        scenario = normalize(str(data.get("scenario") or ""))
        if normalize(record["question"]) in scenario:
            duplicate_exact += 1

    report = {
        "total": len(records),
        "populated": sum(1 for r in records if r["question"]),
        "missing": sum(1 for r in records if not r["question"]),
        "corpus_median_question_length": corpus_median,
        "duplicate_question_already_in_scenario": duplicate_exact,
        "by_track": {k: summarize(v, corpus_median) for k, v in sorted(by_track.items())},
        "by_level": {k: summarize(v, corpus_median) for k, v in sorted(by_level.items())},
        "by_track_level": {
            k: summarize(v, corpus_median) for k, v in sorted(by_track_level.items())
        },
        "question_count_by_topic": Counter(r["topic"] for r in records),
    }

    output = VAULT_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("total", "populated", "missing")}, indent=2))
    print("Corpus median question length:", corpus_median)
    print("Per-track summary:")
    for track, summary in report["by_track"].items():
        print(f"  {track}: {summary}")
    print("Report:", output.relative_to(VAULT_DIR))
    return 0 if report["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

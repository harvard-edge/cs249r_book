#!/usr/bin/env python3
"""Build a semantic review queue for published StaffML questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
DEFAULT_OUT = VAULT_DIR / "audit" / "semantic-review-queue"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def review_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    return {
        "qid": data.get("id"),
        "path": rel(path),
        "track": data.get("track"),
        "level": data.get("level"),
        "zone": data.get("zone"),
        "topic": data.get("topic"),
        "competency_area": data.get("competency_area"),
        "bloom_level": data.get("bloom_level"),
        "phase": data.get("phase"),
        "title": data.get("title"),
        "scenario": data.get("scenario"),
        "question": data.get("question"),
        "realistic_solution": details.get("realistic_solution"),
        "common_mistake": details.get("common_mistake"),
        "napkin_math": details.get("napkin_math"),
        "options": details.get("options"),
        "correct_index": details.get("correct_index"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--track", choices=["cloud", "edge", "mobile", "tinyml", "global"])
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.questions_dir.glob("*/*/*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            continue
        if data.get("status") != "published":
            continue
        if args.track and data.get("track") != args.track:
            continue
        rows.append(review_record(path, data))

    write_jsonl(args.out_dir / "published_semantic_queue.jsonl", rows)

    by_track: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_track.setdefault(str(row.get("track")), []).append(row)
    for track, track_rows in sorted(by_track.items()):
        write_jsonl(args.out_dir / f"{track}_published_semantic_queue.jsonl", track_rows)
        for idx in range(0, len(track_rows), args.batch_size):
            batch_no = idx // args.batch_size + 1
            batch_path = args.out_dir / "batches" / track / f"{track}_batch_{batch_no:03d}.jsonl"
            write_jsonl(batch_path, track_rows[idx : idx + args.batch_size])

    prompt = """You are reviewing StaffML interview-question YAML for release quality.

For each JSONL record, evaluate:
1. scenario_question_fit: Does the question follow from the scenario?
2. answer_correctness: Does realistic_solution answer the question directly?
3. common_mistake_quality: Is the pitfall plausible, specific, and useful?
4. napkin_math_correctness: Are formulas, units, and conclusions correct?
5. physical_plausibility: Are hardware/software numbers realistic?
6. level_fit: Does level/bloom/zone match the cognitive demand?
7. title_quality: Is the title concrete and searchable?

Return one JSON object per input record with:
qid, verdict ("pass"|"needs_fix"), severity ("blocker"|"major"|"minor"|"none"),
issues [short strings], suggested_fix_summary, and confidence.
Do not edit YAML directly during review. Produce findings only.
"""
    (args.out_dir / "semantic_review_prompt.md").write_text(prompt)

    print(f"Wrote {len(rows)} published-question records to {rel(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

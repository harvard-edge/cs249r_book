#!/usr/bin/env python3
"""Validate generated StaffML question candidates against release gates.

The script is intentionally local and deterministic. It checks the gates that
do not require remote model calls: schema, question shape, duplication signals,
topic-track applicability, zone-level affinity, chain references, topic
concentration, and provenance/human-review reporting.
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
CHAINS_PATH = VAULT_DIR / "chains.json"

if str(VAULT_DIR) not in sys.path:
    sys.path.insert(0, str(VAULT_DIR))
SCHEMA_DIR = VAULT_DIR / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from schema import Question  # noqa: E402
from enums import VALID_TOPICS, ZONE_LEVEL_AFFINITY  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.files_from:
        for line in Path(args.files_from).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                paths.append(ROOT_DIR / line if not line.startswith("/") else Path(line))
    paths.extend(Path(p) for p in args.paths)
    if not paths:
        paths = sorted(QUESTIONS_DIR.glob("*/*.yaml"))
    return sorted({p.resolve() for p in paths})


def load_applicability() -> set[tuple[str, str]]:
    data = yaml.safe_load(TAXONOMY_DATA.read_text(encoding="utf-8")) or {}
    pairs: set[tuple[str, str]] = set()
    for topic in data.get("topics", []):
        for track in topic.get("tracks", []):
            pairs.add((topic["id"], track))
    return pairs


def load_chain_ids() -> set[str]:
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    return {c.get("chain_id") for c in chains if c.get("chain_id")}


def gate_result(name: str, status: str, details: list[str] | None = None) -> dict[str, Any]:
    details = details or []
    return {
        "gate": name,
        "status": status,
        "details": details[:100],
        "total_issues": len(details),
    }


def validate(paths: list[Path]) -> dict[str, Any]:
    applicability = load_applicability()
    chain_ids = load_chain_ids()
    schema_errors: list[str] = []
    question_shape: list[str] = []
    duplicate_in_scenario: list[str] = []
    applicability_errors: list[str] = []
    affinity_warnings: list[str] = []
    chain_errors: list[str] = []
    exact_duplicate_questions: list[str] = []
    topic_errors: list[str] = []
    records: list[dict[str, Any]] = []

    question_texts: dict[str, list[str]] = defaultdict(list)
    provenance = Counter()
    human_review = Counter()

    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            schema_errors.append(f"{path}: YAML parse error: {exc}")
            continue
        try:
            Question(**data)
        except Exception as exc:  # noqa: BLE001 - report validation shape
            schema_errors.append(f"{path}: {exc}")

        qid = data.get("id", path.stem)
        question = (data.get("question") or "").strip()
        scenario = str(data.get("scenario") or "")
        topic = data.get("topic", "")
        track = data.get("track", "")
        zone = data.get("zone", "")
        level = data.get("level", "")
        provenance[data.get("provenance", "missing")] += 1
        hr = data.get("human_reviewed") or {}
        human_review[hr.get("status", "missing") if isinstance(hr, dict) else "missing"] += 1

        if not question:
            question_shape.append(f"{qid}: missing question")
        elif len(question) > 200:
            question_shape.append(f"{qid}: question length {len(question)} > 200")
        elif not question.endswith("?"):
            question_shape.append(f"{qid}: question does not end with ?")
        elif question.count("?") != 1:
            question_shape.append(f"{qid}: question has {question.count('?')} question marks")
        elif re.search(r"based on the above|according to the scenario", question, re.I):
            question_shape.append(f"{qid}: meta phrase in question")

        if question and normalize(question) in normalize(scenario):
            duplicate_in_scenario.append(qid)

        if topic not in VALID_TOPICS:
            topic_errors.append(f"{qid}: topic {topic!r} not in schema enum")
        if track != "global" and topic in VALID_TOPICS and (topic, track) not in applicability:
            applicability_errors.append(f"{qid}: {topic} not applicable to {track} in taxonomy_data")
        if zone in ZONE_LEVEL_AFFINITY and level not in ZONE_LEVEL_AFFINITY[zone]:
            affinity_warnings.append(f"{qid}: {zone}/{level} outside affinity")

        for chain in data.get("chains") or []:
            if isinstance(chain, dict) and chain.get("id") not in chain_ids:
                chain_errors.append(f"{qid}: unknown chain {chain.get('id')}")

        if question:
            question_texts[normalize(question)].append(qid)
        records.append(data)

    for question, ids in question_texts.items():
        if len(ids) > 1:
            exact_duplicate_questions.append(f"{ids[0]} and {len(ids)-1} others: {question[:120]}")

    topic_counts = Counter(q.get("topic", "") for q in records)
    threshold = max(1, int(len(records) * 0.15))
    topic_concentration = [
        f"{topic}: {count}" for topic, count in topic_counts.items() if count > threshold
    ]

    gates = [
        gate_result("schema", "PASS" if not schema_errors else "FAIL", schema_errors),
        gate_result("question-shape", "PASS" if not question_shape else "FAIL", question_shape),
        gate_result(
            "scenario-question-duplication",
            "WARN" if duplicate_in_scenario else "PASS",
            duplicate_in_scenario,
        ),
        gate_result("topic-schema", "PASS" if not topic_errors else "FAIL", topic_errors),
        gate_result(
            "topic-track-applicability",
            "PASS" if not applicability_errors else "FAIL",
            applicability_errors,
        ),
        gate_result(
            "zone-level-affinity",
            "WARN" if affinity_warnings else "PASS",
            affinity_warnings,
        ),
        gate_result("chain-integrity", "PASS" if not chain_errors else "FAIL", chain_errors),
        gate_result(
            "exact-question-dedup",
            "WARN" if exact_duplicate_questions else "PASS",
            exact_duplicate_questions,
        ),
        gate_result(
            "topic-concentration",
            "WARN" if topic_concentration else "PASS",
            topic_concentration,
        ),
    ]
    return {
        "total_files": len(paths),
        "gates": gates,
        "provenance_counts": dict(provenance),
        "human_review_counts": dict(human_review),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="YAML files to validate. Defaults to full corpus.")
    parser.add_argument("--files-from", help="Text file of paths relative to repo root.")
    parser.add_argument(
        "--output",
        default="_validation_results/gap_plan/validation_gate_results.json",
        help="Output path relative to interviews/vault.",
    )
    args = parser.parse_args()

    paths = load_paths(args)
    result = validate(paths)
    output = VAULT_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {result['total_files']} files")
    for gate in result["gates"]:
        print(f"{gate['status']:>4} {gate['gate']}: {gate['total_issues']}")
    print(f"Report: {output.relative_to(VAULT_DIR)}")
    return 1 if any(g["status"] == "FAIL" for g in result["gates"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

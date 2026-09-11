#!/usr/bin/env python3
"""Parallel semantic audit runner for published StaffML questions.

This consumes JSONL queues created by prepare_semantic_review_queue.py and
appends one structured finding per question. It is resumable: existing qids in
the output file are skipped on later runs. The runner batches a few questions
per model call so the model can use context more efficiently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

VAULT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VAULT_DIR.parents[1]
DEFAULT_QUEUE = VAULT_DIR / "audit" / "semantic-review-queue" / "published_semantic_queue.jsonl"
DEFAULT_OUT = VAULT_DIR / "audit" / "semantic-review-results" / "semantic_findings.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You are a strict StaffML release-quality reviewer.

Review each ML systems interview question for publishability. You are not
editing YAML; you are producing findings.

Evaluate:
- scenario_question_fit: the question follows naturally from the scenario.
- answer_correctness: realistic_solution directly and correctly answers it.
- common_mistake_quality: pitfall/rationale/consequence are plausible,
  specific, and pedagogically useful.
- napkin_math_correctness: formulas, units, conversions, assumptions, and
  conclusion are correct. Conceptual non-numeric math is allowed when the
  question is qualitative, but it must still be logically useful.
- physical_plausibility: hardware/software numbers and product names are real
  and plausible for the track.
- level_fit: level, bloom_level, and zone match the cognitive demand.
- title_quality: title is concrete, searchable, and not generic.

Be practical: only mark needs_fix when a release editor should change the YAML.
Do not nitpick phrasing that is already clear and correct.
"""

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "qid": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "needs_fix"]},
        "severity": {"type": "string", "enum": ["none", "minor", "major", "blocker"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "scenario_question_fit": {"type": "string", "enum": ["pass", "fail"]},
        "answer_correctness": {"type": "string", "enum": ["pass", "fail"]},
        "common_mistake_quality": {"type": "string", "enum": ["pass", "fail"]},
        "napkin_math_correctness": {"type": "string", "enum": ["pass", "fail"]},
        "physical_plausibility": {"type": "string", "enum": ["pass", "fail"]},
        "level_fit": {"type": "string", "enum": ["pass", "fail"]},
        "title_quality": {"type": "string", "enum": ["pass", "fail"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggested_fix_summary": {"type": "string"},
    },
    "required": [
        "qid",
        "verdict",
        "severity",
        "confidence",
        "scenario_question_fit",
        "answer_correctness",
        "common_mistake_quality",
        "napkin_math_correctness",
        "physical_plausibility",
        "level_fit",
        "title_quality",
        "issues",
        "suggested_fix_summary",
    ],
}

BATCH_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": RESULT_SCHEMA,
            "minItems": 1,
        }
    },
    "required": ["findings"],
}


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if isinstance(qid, str) and not row.get("_error") and row.get("_audit_status", "ok") == "ok":
                done.add(qid)
    return done


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "qid",
        "track",
        "level",
        "zone",
        "bloom_level",
        "topic",
        "competency_area",
        "title",
        "scenario",
        "question",
        "realistic_solution",
        "common_mistake",
        "napkin_math",
        "options",
        "correct_index",
    ]
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def audit_batch(
    client: OpenAI,
    model: str,
    records: list[dict[str, Any]],
    max_retries: int,
) -> list[dict[str, Any]]:
    payload = json.dumps([compact_record(record) for record in records], ensure_ascii=False, sort_keys=True)
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Return exactly one JSON finding per input record, in the same order. "
                            "Do not merge questions. Return an object with a `findings` array. "
                            "Input JSON array:\n" + payload
                        ),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "staffml_semantic_batch",
                        "strict": True,
                        "schema": BATCH_RESULT_SCHEMA,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            findings = parsed.get("findings")
            if not isinstance(findings, list) or len(findings) != len(records):
                raise ValueError(
                    f"Expected {len(records)} findings, got {len(findings) if isinstance(findings, list) else 'non-list'}"
                )
            out: list[dict[str, Any]] = []
            for record, finding in zip(records, findings):
                if not isinstance(finding, dict):
                    raise ValueError("Batch finding is not an object")
                qid = str(record["qid"])
                finding["qid"] = qid
                finding["_path"] = record.get("path")
                finding["_model"] = model
                finding["_audit_status"] = "ok"
                out.append(finding)
            return out
        except Exception as exc:  # noqa: BLE001 - durable audit record is better than crash
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))

    out: list[dict[str, Any]] = []
    for record in records:
        out.append(
            {
                "qid": str(record["qid"]),
                "verdict": "needs_fix",
                "severity": "major",
                "confidence": "low",
                "scenario_question_fit": "fail",
                "answer_correctness": "fail",
                "common_mistake_quality": "fail",
                "napkin_math_correctness": "fail",
                "physical_plausibility": "fail",
                "level_fit": "fail",
                "title_quality": "fail",
                "issues": [f"semantic audit API error: {last_error}"],
                "suggested_fix_summary": "Rerun semantic audit for this qid.",
                "_path": record.get("path"),
                "_model": model,
                "_audit_status": "api_error",
                "_error": last_error,
            }
        )
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("STAFFML_SEMANTIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--qid", action="append", help="Audit only this qid; may be repeated")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    rows = read_jsonl(args.queue)
    if args.qid:
        wanted = set(args.qid)
        rows = [row for row in rows if row.get("qid") in wanted]

    done = read_done_qids(args.out)
    todo = [row for row in rows if str(row.get("qid")) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"Queue: {len(rows)} records; already done: {len(done)}; this run: {len(todo)}",
        flush=True,
    )
    if not todo:
        return 0

    client = OpenAI(timeout=args.request_timeout)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        batches = [todo[idx : idx + max(1, args.batch_size)] for idx in range(0, len(todo), max(1, args.batch_size))]
        futures = {
            executor.submit(audit_batch, client, args.model, batch, args.max_retries): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            for result in results:
                append_jsonl(args.out, result)
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    print(f"progress: {completed}/{len(todo)} current={result.get('qid')}", flush=True)

    print(f"Wrote findings to {rel(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

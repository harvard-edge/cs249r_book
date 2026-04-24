#!/usr/bin/env python3
"""Chunked Gemini CLI math review for StaffML YAML questions.

This runner is deliberately review-first: it batches many questions into each
Gemini CLI call, asks for strict JSON, and writes reports/correction proposals.
It does not mutate YAML. Human or maintainer review should apply accepted fixes.

Example:

    python3 interviews/vault/scripts/gemini_cli_math_review.py \
      --model gemini-3.1-pro-preview --unverified-only --chunk-size 35 --max-calls 250
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OUTPUT_DIR = VAULT_DIR / "_validation_results" / "gemini_math_review"

HARDWARE_REFERENCE = """
Reference constants to use unless the question explicitly states otherwise:
- H100 SXM: 80 GB HBM3, 3.35 TB/s HBM bandwidth, 989 TFLOP/s FP16 tensor, 700 W.
- A100 80GB SXM: ~2.0 TB/s HBM2e bandwidth, 312 TFLOP/s FP16 tensor, 400 W.
- MI300X: 192 GB HBM3, 5.3 TB/s bandwidth, ~1307 TFLOP/s FP16 sparse peak.
- Jetson AGX Orin: up to 275 TOPS INT8, ~204.8 GB/s LPDDR5, 15-60 W modes.
- Hailo-8: 26 TOPS INT8, ~2.5 W accelerator power.
- Apple A17 Pro Neural Engine: roughly 35 TOPS.
- Snapdragon 8 Gen 3 Hexagon NPU: roughly 45 TOPS.
- Cortex-M4 examples commonly use 80-240 MHz, KB-scale SRAM.
- 1 byte = 8 bits. 1 GB/s = 1000 MB/s for napkin math unless question says GiB.
- FP16/BF16 weights: 2 bytes/parameter. INT8: 1 byte/parameter. INT4: 0.5 byte/parameter before metadata.
- KV cache: 2 x layers x KV heads x head dim x sequence length x batch x bytes.
- Ring AllReduce lower-bound byte term: 2(N-1)/N x payload / bandwidth.
- Power energy: Wh = W x hours; kWh cost = kWh x price.
""".strip()

PROMPT_TEMPLATE = """You are an expert ML systems math reviewer.

Review the StaffML questions below for arithmetic, unit conversions, hardware
specs, and whether the solution's conclusion follows from its math.

{hardware_reference}

Return STRICT JSON only with this schema:
[
  {{
    "id": "question-id",
    "status": "CORRECT" | "WARN" | "ERROR",
    "issues": ["short issue text"],
    "corrections": ["specific corrected value or wording"],
    "confidence": "low" | "medium" | "high"
  }}
]

Rules:
- Use ERROR for wrong arithmetic, wrong units, wrong hardware specs, or a conclusion contradicted by math.
- Use WARN for plausible but imprecise assumptions, missing caveats, or ambiguous unit conventions.
- Use CORRECT when math is sound enough for interview/practice use.
- Do not complain merely because napkin math is approximate.
- Check every question; do not skip.

Questions:
{questions_json}
"""


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths: list[Path]
    if args.files_from:
        paths = [
            ROOT_DIR / line.strip()
            for line in Path(args.files_from).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        paths = sorted(QUESTIONS_DIR.glob("*/*.yaml"))

    questions: list[dict[str, Any]] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        details = data.get("details") or {}
        if args.published_only and data.get("status") != "published":
            continue
        if args.draft_only and data.get("status") != "draft":
            continue
        if args.unverified_only and data.get("math_verified") is True:
            continue
        if args.require_napkin and not (details.get("napkin_math") or "").strip():
            continue
        questions.append(
            {
                "path": str(path.relative_to(ROOT_DIR)),
                "id": data.get("id", path.stem),
                "track": data.get("track"),
                "level": data.get("level"),
                "zone": data.get("zone"),
                "topic": data.get("topic"),
                "title": data.get("title"),
                "scenario": data.get("scenario"),
                "question": data.get("question"),
                "realistic_solution": details.get("realistic_solution"),
                "napkin_math": details.get("napkin_math"),
                "common_mistake": details.get("common_mistake"),
                "options": details.get("options"),
                "correct_index": details.get("correct_index"),
            }
        )
    return questions[: args.limit] if args.limit else questions


def chunk_questions(questions: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        grouped[(question.get("track") or "", question.get("topic") or "")].append(question)

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        for question in group:
            if len(current) >= chunk_size:
                chunks.append(current)
                current = []
            current.append(question)
    if current:
        chunks.append(current)
    return chunks


def slim(question: dict[str, Any]) -> dict[str, Any]:
    def cap(value: Any, n: int) -> Any:
        if not isinstance(value, str):
            return value
        return value if len(value) <= n else value[: n - 1] + "…"

    return {
        "id": question["id"],
        "track": question["track"],
        "level": question["level"],
        "topic": question["topic"],
        "title": cap(question["title"], 160),
        "scenario": cap(question["scenario"], 1200),
        "question": cap(question["question"], 300),
        "realistic_solution": cap(question["realistic_solution"], 1200),
        "napkin_math": cap(question["napkin_math"], 1000),
        "common_mistake": cap(question["common_mistake"], 500),
        "options": question.get("options"),
        "correct_index": question.get("correct_index"),
    }


def build_prompt(chunk: list[dict[str, Any]]) -> str:
    return PROMPT_TEMPLATE.format(
        hardware_reference=HARDWARE_REFERENCE,
        questions_json=json.dumps([slim(q) for q in chunk], ensure_ascii=False, indent=2),
    )


def parse_json_response(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def call_gemini(prompt: str, model: str, timeout: int) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        proc = subprocess.run(
            ["gemini", "-m", model, "-o", "text"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
    try:
        return parse_json_response(proc.stdout), None
    except Exception as exc:  # noqa: BLE001 - preserve raw response
        return None, f"json_parse_error: {exc}; raw={proc.stdout[:500]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-model-override",
        action="store_true",
        help=(
            "Allow a model other than gemini-3.1-pro-preview. By default this "
            "runner enforces the Pro preview model used for release math review."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=35)
    parser.add_argument("--max-calls", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--files-from")
    parser.add_argument("--published-only", action="store_true")
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--unverified-only", action="store_true")
    parser.add_argument("--require-napkin", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL and not args.allow_model_override:
        parser.error(
            f"--model must be {DEFAULT_MODEL!r} for release math review "
            "(pass --allow-model-override for experiments only)"
        )

    questions = load_questions(args)
    total_candidates = len(questions)
    chunks = chunk_questions(questions, args.chunk_size)
    chunks_before_cap = len(chunks)
    if len(chunks) > args.max_calls:
        chunks = chunks[: args.max_calls]
    reviewed_question_count = sum(len(chunk) for chunk in chunks)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "candidate_questions": total_candidates,
        "questions_planned_for_review": reviewed_question_count,
        "questions_deferred_by_call_cap": max(0, total_candidates - reviewed_question_count),
        "chunks_planned": len(chunks),
        "chunks_before_call_cap": chunks_before_cap,
        "chunk_size": args.chunk_size,
        "max_calls": args.max_calls,
        "dry_run": args.dry_run,
        "grouping": "track/topic",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        for idx, chunk in enumerate(chunks):
            prompt = build_prompt(chunk)
            (output_dir / f"prompt_{idx:03d}.txt").write_text(prompt, encoding="utf-8")
        print(f"Dry run prompts written to {output_dir}")
        return 0

    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        prompt = build_prompt(chunk)
        (output_dir / f"prompt_{idx:03d}.txt").write_text(prompt, encoding="utf-8")
        print(f"[{idx + 1}/{len(chunks)}] reviewing {len(chunk)} questions...")
        results, error = call_gemini(prompt, args.model, args.timeout)
        if error:
            failures.append({"chunk": idx, "error": error, "ids": [q["id"] for q in chunk]})
            (output_dir / f"error_{idx:03d}.txt").write_text(error, encoding="utf-8")
            print(f"  ERROR {error[:160]}")
        else:
            assert results is not None
            all_results.extend(results)
            (output_dir / f"result_{idx:03d}.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = defaultdict(int)
            for result in results:
                counts[result.get("status", "UNKNOWN")] += 1
            print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        time.sleep(args.sleep)

    (output_dir / "all_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = Counter(result.get("status", "UNKNOWN") for result in all_results)
    summary_payload = {
        "model": args.model,
        "reviewed": len(all_results),
        "status_counts": dict(summary),
        "failed_chunks": len(failures),
        "failed_question_ids": [qid for failure in failures for qid in failure.get("ids", [])],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Summary:", summary_payload, "output:", output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

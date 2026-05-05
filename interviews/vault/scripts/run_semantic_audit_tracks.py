#!/usr/bin/env python3
"""Launch semantic audit jobs per track with separate output files."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

VAULT_DIR = Path(__file__).resolve().parents[1]
QUEUE_DIR = VAULT_DIR / "audit" / "semantic-review-queue"
RESULTS_DIR = VAULT_DIR / "audit" / "semantic-review-results"
RUNNER = VAULT_DIR / "scripts" / "semantic_audit_questions.py"
TRACKS = ("cloud", "edge", "global", "mobile", "tinyml")
DEFAULT_MODEL = "gpt-5.4-mini"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    parser.add_argument("--model", default=os.environ.get("STAFFML_SEMANTIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers-per-track", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit-per-track", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    for track in args.tracks:
        queue = QUEUE_DIR / f"{track}_published_semantic_queue.jsonl"
        out = RESULTS_DIR / f"{track}_semantic_findings.jsonl"
        command = [
            sys.executable,
            str(RUNNER),
            "--queue",
            str(queue),
            "--out",
            str(out),
            "--model",
            args.model,
            "--workers",
            str(args.workers_per_track),
            "--batch-size",
            str(args.batch_size),
            "--max-retries",
            str(args.max_retries),
            "--request-timeout",
            str(args.request_timeout),
        ]
        if args.limit_per_track is not None:
            command.extend(["--limit", str(args.limit_per_track)])
        commands.append(command)

    for command in commands:
        print(shlex.join(command))

    if args.dry_run:
        return 0

    processes = [subprocess.Popen(command) for command in commands]
    return_codes = [process.wait() for process in processes]
    return 1 if any(return_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())

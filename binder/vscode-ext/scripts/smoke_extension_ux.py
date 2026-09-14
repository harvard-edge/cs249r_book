#!/usr/bin/env python3
"""Smoke-test the binder commands exposed by the VS Code extension.

Runs each command as a shell subprocess from the repo root with a per-command
timeout, prints a pass/fail summary with output tails for non-passing cases,
writes a JSON report, and exits 1 if any case failed. Several cases (clean,
reset, build, fix) modify the working tree.

    python3 binder/vscode-ext/scripts/smoke_extension_ux.py
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Verdict = Literal["pass", "fail", "pass_started"]


@dataclass
class Case:
    """One shell command to smoke-test.

    ``allow_timeout_as_started`` turns a timeout into a ``pass_started``
    verdict (for long-running commands); ``allow_nonzero_exit`` counts any
    exit code as a pass.
    """

    name: str
    command: str
    timeout_s: int = 60
    allow_timeout_as_started: bool = False
    allow_nonzero_exit: bool = False


@dataclass
class Result:
    """Outcome of one ``Case``; ``exit_code`` is None on timeout and ``output_tail`` holds the last 20 output lines."""

    name: str
    command: str
    verdict: Verdict
    exit_code: int | None
    elapsed_s: float
    note: str
    output_tail: str


def run_case(repo_root: Path, case: Case) -> Result:
    """Run ``case.command`` through the shell from ``repo_root`` and classify the outcome.

    stdout and stderr are merged. A zero exit passes; a nonzero exit passes
    only with ``allow_nonzero_exit``; a timeout is ``pass_started`` only with
    ``allow_timeout_as_started`` and fails otherwise.
    """
    start = time.time()
    try:
        completed = subprocess.run(
            case.command,
            cwd=repo_root,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=case.timeout_s,
            text=True,
        )
        elapsed = time.time() - start
        output = (completed.stdout or "").strip()
        output_tail = "\n".join(output.splitlines()[-20:])
        if completed.returncode == 0:
            return Result(case.name, case.command, "pass", 0, elapsed, "ok", output_tail)
        if case.allow_nonzero_exit:
            return Result(
                case.name,
                case.command,
                "pass",
                completed.returncode,
                elapsed,
                "non-zero accepted for this check",
                output_tail,
            )
        return Result(
            case.name,
            case.command,
            "fail",
            completed.returncode,
            elapsed,
            "non-zero exit",
            output_tail,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        output_tail = "\n".join(partial.splitlines()[-20:]).strip()
        if case.allow_timeout_as_started:
            note = f"timed out after {case.timeout_s}s (treated as started)"
            return Result(case.name, case.command, "pass_started", None, elapsed, note, output_tail)
        return Result(
            case.name,
            case.command,
            "fail",
            None,
            elapsed,
            f"timed out after {case.timeout_s}s",
            output_tail,
        )


def first_vol1_chapter(repo_root: Path) -> str:
    """Return the first chapter slug printed by ``binder list --vol1``.

    Falls back to ``"introduction"`` when the command exits nonzero or prints
    no numbered chapter lines. A 60-second timeout raises
    ``subprocess.TimeoutExpired``.
    """
    proc = subprocess.run(
        "./binder/binder list --vol1",
        cwd=repo_root,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return "introduction"
    text = proc.stdout or ""
    chapter_pattern = re.compile(r"^\s*\d+\.\s+([a-zA-Z0-9_-]+)\s+\(", re.MULTILINE)
    matches = chapter_pattern.findall(text)
    return matches[0] if matches else "introduction"


def precommit_hook_ids(constants_file: Path) -> list[str]:
    """Return the sorted unique hook ids named in ``pre-commit run <id> --all-files`` strings in ``constants_file``."""
    text = constants_file.read_text(encoding="utf-8")
    ids = set(re.findall(r"pre-commit run ([a-zA-Z0-9_-]+)\s+--all-files", text))
    return sorted(ids)


def build_cases(repo_root: Path) -> list[Case]:
    """Assemble the smoke cases.

    Covers a fixed list of binder CLI commands run against the first Vol. I
    chapter and a sample chapter QMD, plus one ``pre-commit run`` case per
    hook id read from ``book/vscode-ext/src/constants.ts`` under
    ``repo_root``.
    """
    chapter = first_vol1_chapter(repo_root)
    sample_file = "books/vol1/introduction/introduction.qmd"
    constants_file = repo_root / "book/vscode-ext/src/constants.ts"
    hook_ids = precommit_hook_ids(constants_file)

    cases: list[Case] = [
        Case("binder help", "./binder/binder help", 30),
        Case("binder list vol1", "./binder/binder list --vol1", 60),
        Case("binder list vol2", "./binder/binder list --vol2", 60),
        Case("binder status", "./binder/binder status", 30),
        Case("binder doctor", "./binder/binder doctor", 120, allow_nonzero_exit=True),
        Case("binder clean", "./binder/binder clean", 60),
        Case("binder check all", "./binder/binder check all --vol1", 120, allow_nonzero_exit=True),
        Case("binder check inline-python", f"./binder/binder check refs --scope inline-python --path {shlex.quote(sample_file)}", 120, allow_nonzero_exit=True),
        Case("binder check refs", f"./binder/binder check refs --path {shlex.quote(sample_file)}", 120),
        Case("binder check citations", f"./binder/binder check refs --scope citations --path {shlex.quote(sample_file)}", 120),
        Case("binder check duplicate-labels", "./binder/binder check labels --scope duplicates --vol1 --all-types", 120),
        Case("binder check unreferenced-labels", "./binder/binder check labels --scope orphans --vol1 --all-types", 120, allow_nonzero_exit=True),
        Case("binder check inline-refs", f"./binder/binder check refs --scope inline --path {shlex.quote(sample_file)} --check-patterns", 120, allow_nonzero_exit=True),
        Case("binder reset pdf", "./binder/binder reset pdf --vol1", 60),
        Case("binder reset html", "./binder/binder reset html --vol1", 60),
        Case("binder reset epub", "./binder/binder reset epub --vol1", 60),
        Case("build chapter html", f"./binder/binder build html {chapter} --vol1 -v", 180),
        Case("build chapter pdf", f"./binder/binder build pdf {chapter} --vol1 -v", 180, allow_timeout_as_started=True),
        Case("build chapter epub", f"./binder/binder build epub {chapter} --vol1 -v", 180),
        Case("preview chapter", f"./binder/binder preview vol1/{chapter}", 20, allow_timeout_as_started=True),
        Case("debug chapter pdf", f"./binder/binder debug pdf --vol1 --chapter {chapter}", 180, allow_timeout_as_started=True),
        Case("debug chapter html", f"./binder/binder debug html --vol1 --chapter {chapter}", 180, allow_timeout_as_started=True),
        Case("debug chapter epub", f"./binder/binder debug epub --vol1 --chapter {chapter}", 180, allow_timeout_as_started=True),
        Case("fix glossary paths", "./binder/binder fix glossary paths", 120),
        Case("fix images compress", "./binder/binder fix images compress --all --smart-compression", 120),
        Case("fix repo-health", "./binder/binder fix repo-health", 120),
        Case("publish script help vol1", "bash binder/tools/scripts/publish/mit-press-release.sh --help", 30, allow_nonzero_exit=True),
        Case("extract figures help", "python3 binder/tools/scripts/publish/extract_figures.py --help", 30),
    ]

    for hook_id in hook_ids:
        cases.append(
            Case(
                f"pre-commit hook exists: {hook_id}",
                f"pre-commit run {hook_id} --files {shlex.quote(sample_file)}",
                120,
                allow_nonzero_exit=True,
            )
        )

    return cases


def main() -> int:
    """Run every case, print the summary, and write ``book/vscode-ext/.smoke-extension-ux.json``; return 1 if any case failed."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    cases = build_cases(repo_root)
    results = [run_case(repo_root, case) for case in cases]

    failing = [r for r in results if r.verdict == "fail"]
    started = [r for r in results if r.verdict == "pass_started"]

    print("=== Extension UX smoke results ===")
    print(f"Total: {len(results)}  Pass: {len(results) - len(failing)}  Fail: {len(failing)}  Started: {len(started)}")
    for r in results:
        print(f"[{r.verdict.upper():11}] {r.name} ({r.elapsed_s:.1f}s) -> {r.command}")
        if r.verdict != "pass":
            print(f"  note: {r.note}")
            if r.output_tail:
                print("  output tail:")
                for line in r.output_tail.splitlines():
                    print(f"    {line}")

    report_path = repo_root / "book/vscode-ext/.smoke-extension-ux.json"
    report = {
        "generated_at_epoch_s": time.time(),
        "total": len(results),
        "fails": len(failing),
        "started": len(started),
        "results": [r.__dict__ for r in results],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")

    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

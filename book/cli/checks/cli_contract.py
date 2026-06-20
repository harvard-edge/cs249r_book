"""Public command contract checks for the Binder CLI.

This powers::

    ./book/binder check cli --scope contract

The check is intentionally small and example-heavy. It runs read-only commands
that define the public Binder surface and fails if help text, migration hints,
or exit codes drift.

Examples of what it catches:

* ``./book/binder build reset pdf --vol1`` accidentally working again.
  Canonical shape is ``./book/binder reset pdf --vol1``.
* ``./book/binder pdf reset --vol1`` accidentally working again.
  Top-level ``html`` / ``pdf`` / ``epub`` are removed; builds live under
  ``build`` and YAML resets live under ``reset``.
* ``./book/binder reset`` mutating state instead of showing help.
  Bare reset is informational; reset needs an explicit target.
* ``./book/binder check`` omitting a registered check group or scope from
  the live catalogue, which makes pre-commit failures harder to debug.

Every failure includes the command, expected condition, and a short output
excerpt so a human or LLM can fix the router/docs without reverse-engineering
the test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class ContractCase:
    name: str
    argv: tuple[str, ...]
    expected_exit: int
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    timeout_seconds: int = 20


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""
    suggestion: str = ""


CASES: tuple[ContractCase, ...] = (
    ContractCase(
        name="top-level help exposes canonical command families",
        argv=("help",),
        expected_exit=0,
        must_include=(
            "MLSysBook CLI",
            "build [fmt]",
            "check <group>",
            "pre-release",
            "reset <fmt|all>",
            "[--vol1|--vol2]",
        ),
    ),
    ContractCase(
        name="build help does not advertise reset",
        argv=("build", "--help"),
        expected_exit=0,
        must_include=(
            "Usage: ./binder build",
            "./binder build pdf intro,training --vol1",
        ),
        must_not_include=("build reset",),
    ),
    ContractCase(
        name="bare reset is help-only",
        argv=("reset",),
        expected_exit=0,
        must_include=(
            "binder reset <html|pdf|epub|all> [--vol1|--vol2]",
            "./binder reset pdf --vol1",
        ),
    ),
    ContractCase(
        name="reset help documents explicit target shape",
        argv=("reset", "help"),
        expected_exit=0,
        must_include=(
            "reset pdf --vol1",
            "reset epub --vol2",
            "reset all",
        ),
    ),
    ContractCase(
        name="reset rejects unknown target",
        argv=("reset", "nope"),
        expected_exit=1,
        must_include=("invalid choice", "html", "pdf", "epub", "all"),
    ),
    ContractCase(
        name="build reset is a hard migration error",
        argv=("build", "reset", "pdf", "--vol1"),
        expected_exit=1,
        must_include=(
            "`binder build reset` was removed.",
            "./binder reset <html|pdf|epub|all> [--vol1|--vol2]",
        ),
    ),
    ContractCase(
        name="top-level pdf command is removed",
        argv=("pdf", "reset", "--vol1"),
        expected_exit=1,
        must_include=(
            "Top-level 'pdf' commands were removed.",
            "Build with: ./binder build pdf",
            "Reset YAML with: ./binder reset pdf [--vol1|--vol2]",
        ),
    ),
    ContractCase(
        name="check catalogue includes CLI and math checks",
        argv=("check",),
        expected_exit=0,
        must_include=(
            "binder check <group>",
            "cli",
            "contract",
            "math",
            "multiplier-style",
        ),
    ),
    ContractCase(
        name="check cli help lists its scopes",
        argv=("check", "cli", "help"),
        expected_exit=0,
        # Assert on stable tokens only: the group header and the short scope
        # NAMES. Notes wrap and runner names get truncated with "…" once the
        # Rich table widens (e.g. when a group gains a scope), so asserting a
        # multi-word note or a full _run_* name is brittle. Scope names are
        # short and never wrap, so they are the durable contract.
        must_include=(
            "binder check cli",
            "contract",
            "binder-canonical",
        ),
    ),
    ContractCase(
        name="check math help shows multiplier prose scope",
        argv=("check", "math", "help"),
        expected_exit=0,
        must_include=(
            "binder check math",
            "prose-contract",
            "multiplier-style",
            "body-prose",
            "multiplier suffixes",
        ),
    ),
    ContractCase(
        name="format help remains routed through command namespace",
        argv=("format", "help"),
        expected_exit=0,
        must_include=("binder format", "python", "prettify"),
    ),
    ContractCase(
        name="bib help remains routed through command namespace",
        argv=("bib", "help"),
        expected_exit=0,
        must_include=("binder bib", "mechanical", "normalize", "sync"),
    ),
    ContractCase(
        name="bib mechanical accepts pre-commit option before filenames",
        argv=("bib", "mechanical", "--dry-run", "--pre-commit", "CITATION.bib"),
        expected_exit=0,
        must_include=("CITATION.bib", "Done:"),
    ),
    ContractCase(
        name="clean help documents artifact cleanup",
        argv=("clean", "help"),
        expected_exit=0,
        must_include=("binder clean", "artifacts"),
    ),
    ContractCase(
        name="pre-release help documents AI-native release gate",
        argv=("pre-release", "--help"),
        expected_exit=0,
        must_include=(
            "binder pre-release",
            "--dry-run",
            "--json",
            "--include-network",
            "--skip-build",
        ),
    ),
    ContractCase(
        name="pre-release dry-run emits machine-readable stage plan",
        argv=("pre-release", "--dry-run", "--json"),
        expected_exit=0,
        must_include=(
            '"schema_version": "binder-pre-release/v1"',
            '"status": "planned"',
            '"stages"',
            '"labels-vol1"',
            '"build-vol1-pdf"',
            '"pdf-vol2-verify"',
            '"final-check-all"',
        ),
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clean_output(text: str) -> str:
    return ANSI_ESCAPE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


def _excerpt(text: str, limit: int = 1400) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    compact = "\n".join(line for line in lines if line)
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "\n..."


def _command_label(argv: Iterable[str]) -> str:
    return "./book/binder " + " ".join(argv)


def run_contract(repo_root: Path | None = None) -> list[Violation]:
    root = repo_root or _repo_root()
    binder = root / "book" / "binder"
    if not binder.exists():
        return [
            Violation(
                file="book/binder",
                line=0,
                code="cli_contract_missing_entrypoint",
                message="Binder entry point does not exist.",
                context=str(binder),
                suggestion="Restore the public CLI executable at book/binder.",
            )
        ]

    env = os.environ.copy()
    env.update(
        {
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
            "TERM": "dumb",
        }
    )

    violations: list[Violation] = []
    for case in CASES:
        command = [str(binder), *case.argv]
        label = _command_label(case.argv)
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=case.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = _clean_output((exc.stdout or "") + (exc.stderr or ""))
            violations.append(
                Violation(
                    file="book/binder",
                    line=0,
                    code="cli_contract_timeout",
                    message=f"{label} timed out after {case.timeout_seconds}s.",
                    context=_excerpt(output),
                    suggestion=(
                        "CLI contract commands must be fast and read-only. "
                        "Keep build/render work out of help and migration paths."
                    ),
                )
            )
            continue

        output = _clean_output(completed.stdout + completed.stderr)
        if completed.returncode != case.expected_exit:
            violations.append(
                Violation(
                    file="book/cli/main.py",
                    line=0,
                    code="cli_contract_exit",
                    message=(
                        f"{label} returned {completed.returncode}; "
                        f"expected {case.expected_exit} ({case.name})."
                    ),
                    context=_excerpt(output),
                    suggestion=(
                        "Preserve documented command exit codes. Help commands "
                        "return 0; removed commands and parse errors return 1."
                    ),
                )
            )

        missing = [needle for needle in case.must_include if needle not in output]
        if missing:
            violations.append(
                Violation(
                    file="book/cli/main.py",
                    line=0,
                    code="cli_contract_missing_output",
                    message=f"{label} is missing expected output for: {', '.join(missing)}.",
                    context=_excerpt(output),
                    suggestion=(
                        "Update the command help/migration text or adjust this "
                        "contract if the public CLI shape intentionally changed."
                    ),
                )
            )

        unexpected = [needle for needle in case.must_not_include if needle in output]
        if unexpected:
            violations.append(
                Violation(
                    file="book/cli/main.py",
                    line=0,
                    code="cli_contract_unexpected_output",
                    message=f"{label} still prints retired text: {', '.join(unexpected)}.",
                    context=_excerpt(output),
                    suggestion="Remove stale help text so users follow the canonical command tree.",
                )
            )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    violations = run_contract()
    if args.json:
        print(json.dumps([v.__dict__ for v in violations], indent=2))
        return 1 if violations else 0

    for violation in violations:
        print(f"{violation.file}:{violation.line} [{violation.code}] {violation.message}")
        if violation.context:
            print(f"  source: {violation.context}")
        if violation.suggestion:
            print(f"  fix: {violation.suggestion}")
    if violations:
        print(f"Total CLI contract violations: {len(violations)}")
        return 1

    print(f"CLI contract passed ({len(CASES)} command cases).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

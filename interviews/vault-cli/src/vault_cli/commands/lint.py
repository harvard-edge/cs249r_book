"""``vault lint`` — author-facing linter for question YAMLs.

Usage:
    vault lint path/to/file.yaml
    vault lint interviews/vault/questions/cloud/
    vault lint --all

Emits three severities:
    ERROR    — schema violation; question cannot be loaded
    WARNING  — likely misclassification (zone-level affinity mismatch,
               human-review missing, etc.) per paper §3.3 Table 2
    INFO     — optional hygiene suggestions

Designed for interactive use by contributors. For CI-tier structural
validation use ``vault check``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.exit_codes import ExitCode
from vault_cli.loader import LoadedQuestion, iter_question_files, load_all
from vault_cli.models import Question
from vault_cli.yaml_io import load_file

# Locate enums for the affinity table.
_HERE = Path(__file__).resolve()
_ENUMS_CANDIDATES = [
    _HERE.parents[4] / "vault" / "schema",
    _HERE.parents[5] / "interviews" / "vault" / "schema",
]
for _c in _ENUMS_CANDIDATES:
    if (_c / "enums.py").exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break

from enums import ZONE_LEVEL_AFFINITY  # noqa: E402  type: ignore[import-not-found]


# ─── Severity ───────────────────────────────────────────────────────────────

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"


class Finding:
    __slots__ = ("severity", "path", "qid", "check", "message")

    def __init__(self, severity: str, path: Path, qid: str | None, check: str, message: str) -> None:
        self.severity = severity
        self.path = path
        self.qid = qid
        self.check = check
        self.message = message


# ─── Individual checks ──────────────────────────────────────────────────────


def _check_zone_level_affinity(q: Question) -> list[tuple[str, str]]:
    """Paper line 397: zones have natural level ranges; flag outliers."""
    affinity = ZONE_LEVEL_AFFINITY.get(q.zone)
    if affinity is None or q.level in affinity:
        return []
    return [(
        "zone-level-affinity",
        f"zone={q.zone!r} typically appears at levels {sorted(affinity)}, "
        f"but this question is {q.level!r}. Review classification.",
    )]


def _check_human_review(q: Question) -> list[tuple[str, str]]:
    """Suggest surfacing human review if the question is `published` but no
    human has verified it."""
    if q.status != "published":
        return []
    hr = q.human_reviewed
    if hr and hr.status == "verified":
        return []
    return [(
        "human-review-pending",
        "published question has no human-verified review yet "
        "(LLM stamps are informational; human review is authoritative).",
    )]


def _check_chains_well_formed(q: Question) -> list[tuple[str, str]]:
    """Multiple chains OK; duplicate positions within one chain is not."""
    if not q.chains:
        return []
    positions_by_chain: dict[str, list[int]] = {}
    for c in q.chains:
        positions_by_chain.setdefault(c.id, []).append(c.position)
    issues = []
    for cid, positions in positions_by_chain.items():
        if len(positions) != len(set(positions)):
            issues.append((
                "chain-duplicate-position",
                f"chain {cid!r} lists this question at duplicate positions {sorted(positions)}.",
            ))
    return issues


def _check_empty_strings(q: Question) -> list[tuple[str, str]]:
    """Flag fields that are empty strings rather than missing — usually a
    serialization bug from an earlier tool."""
    issues = []
    if q.title.strip() != q.title:
        issues.append(("title-whitespace", "title has leading/trailing whitespace."))
    return issues


ALL_CHECKS = [
    _check_zone_level_affinity,
    _check_human_review,
    _check_chains_well_formed,
    _check_empty_strings,
]


# ─── Linting drivers ────────────────────────────────────────────────────────


def _lint_question(lq: LoadedQuestion) -> list[Finding]:
    findings: list[Finding] = []
    q = lq.question
    for check in ALL_CHECKS:
        for check_name, msg in check(q):
            severity = WARN if check_name != "human-review-pending" else INFO
            findings.append(Finding(severity, lq.path, q.id, check_name, msg))
    return findings


def _lint_file(path: Path) -> list[Finding]:
    """Lint a single YAML file. Returns findings including load errors as ERRORs."""
    try:
        data = load_file(path)
    except Exception as exc:  # noqa: BLE001
        return [Finding(ERROR, path, None, "yaml-parse", f"YAML parse failed: {exc}")]
    try:
        q = Question.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        return [Finding(ERROR, path, data.get("id") if isinstance(data, dict) else None,
                        "schema", f"schema validation failed: {exc}")]
    lq = LoadedQuestion(question=q, path=path)
    return _lint_question(lq)


def _lint_dir(vault_dir: Path) -> list[Finding]:
    """Lint a whole vault. Uses load_all for efficiency."""
    loaded, load_errors = load_all(vault_dir)
    findings: list[Finding] = [
        Finding(ERROR, e.path, None, "load", e.message) for e in load_errors
    ]
    for lq in loaded:
        findings.extend(_lint_question(lq))
    return findings


def _print_findings(console: Console, findings: list[Finding], *, show_info: bool) -> None:
    if not findings:
        console.print("[green]No issues found.[/green]")
        return

    counts = {ERROR: 0, WARN: 0, INFO: 0}
    for f in findings:
        counts[f.severity] += 1

    table = Table(title="vault lint findings", show_lines=False)
    table.add_column("severity", no_wrap=True)
    table.add_column("id", no_wrap=True)
    table.add_column("check", no_wrap=True)
    table.add_column("message")

    severity_style = {ERROR: "bold red", WARN: "yellow", INFO: "dim"}
    for f in findings:
        if f.severity == INFO and not show_info:
            continue
        table.add_row(
            f"[{severity_style[f.severity]}]{f.severity}[/]",
            f.qid or "-",
            f.check,
            f.message,
        )
    console.print(table)

    summary = (
        f"[red]{counts[ERROR]} errors[/red] · "
        f"[yellow]{counts[WARN]} warnings[/yellow] · "
        f"[dim]{counts[INFO]} info[/dim]"
    )
    console.print(summary)


# ─── Typer registration ─────────────────────────────────────────────────────


def register(app: typer.Typer) -> None:
    @app.command("lint")
    def lint(
        target: Path = typer.Argument(
            ..., exists=True, help="YAML file or directory to lint."
        ),
        show_info: bool = typer.Option(False, "--info", help="Also show INFO findings."),
        vault_dir: Path = typer.Option(
            Path("interviews/vault"), "--vault-dir",
            help="Vault root (used when target is a directory).",
        ),
    ) -> None:
        """Lint one question or a whole directory."""
        console = Console()
        if target.is_file():
            findings = _lint_file(target)
        else:
            # Point the loader at whichever vault contains `target`.
            # If target IS the questions/ root or its parent, use that.
            vroot = target.parent if target.name == "questions" else target
            findings = [
                f for f in _lint_dir(vroot)
                if str(f.path).startswith(str(target))
            ]
        _print_findings(console, findings, show_info=show_info)

        if any(f.severity == ERROR for f in findings):
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)


__all__ = ["register", "ALL_CHECKS", "Finding"]

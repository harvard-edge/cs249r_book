"""``vault audit`` — Gemini-driven corpus audit + correction workflow.

Subcommands:

    vault audit run [--all|--tracks|--qids] [--propose-fixes] ...
        Wraps audit_corpus_batched.py. Audits the corpus (or a subset)
        against the four Gemini-judge gates (level_fit, coherence,
        math_correct, plus the regex format gate). With --propose-fixes,
        also requests suggested corrections per failed question.

    vault audit review --input PATH [--filter-gate G] [--auto-accept-format]
        Wraps apply_corrections.py. Walks a 01_audit.json from a
        --propose-fixes run; for each question with a proposed
        correction, prompts accept / reject / edit / skip.

    vault audit summarize --input PATH [--output PATH]
        Wraps summarize_audit.py. Generates AUDIT_FINDINGS markdown
        with per-gate counts, per-track failure rates, priority lists.

    vault audit merge --inputs PATH... --output PATH
        Wraps merge_audit_runs.py. Combines per-track parallel runs
        into one canonical 01_audit.json.

Per CORPUS_HARDENING_PLAN.md Phase 8: this is the user-facing CLI on
top of the script-level tooling. Cron workflow at
.github/workflows/staffml-audit-corpus-monthly.yml invokes the
underlying scripts directly; humans use this command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode

audit_app = typer.Typer(help="Gemini-driven corpus audit + correction workflow.")
console = Console()


def _scripts_dir() -> Path:
    """Resolve interviews/vault-cli/scripts/ regardless of install layout."""
    here = Path(__file__).resolve()
    # commands/audit.py → src/vault_cli/commands → src/vault_cli → src → vault-cli
    return here.parents[3] / "scripts"


def _exec(script_name: str, args: list[str]) -> int:
    """Run a script in vault-cli/scripts/ and propagate exit code."""
    script = _scripts_dir() / script_name
    if not script.exists():
        console.print(f"[red]error[/red]: {script} missing")
        return ExitCode.IO_ERROR
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ─── vault audit run ─────────────────────────────────────────────────────


@audit_app.command("run")
def run_cmd(
    all_corpus: bool = typer.Option(
        False, "--all",
        help="Audit the full published corpus (default if no other source given).",
    ),
    tracks: str | None = typer.Option(
        None, "--tracks",
        help="Comma-separated track filter (e.g. cloud,edge).",
    ),
    qids: str | None = typer.Option(
        None, "--qids",
        help="Comma-separated explicit qid list.",
    ),
    propose_fixes: bool = typer.Option(
        False, "--propose-fixes",
        help="Also ask Gemini to propose corrections for each failure.",
    ),
    workers: int = typer.Option(
        4, "--workers",
        help="Concurrent Gemini calls (default 4, max 8).",
    ),
    max_calls: int = typer.Option(
        250, "--max-calls",
        help="Cap Gemini calls this invocation; resume by re-running.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override batch size (default 30 audit-only, 20 with --propose-fixes).",
    ),
    output: Path | None = typer.Option(
        None, "--output",
        help="Output dir (default _pipeline/runs/<UTC-timestamp>/).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show plan without making Gemini calls.",
    ),
) -> None:
    """Run the batched corpus audit. Wraps audit_corpus_batched.py."""
    args: list[str] = []
    # Source mutex: --all OR --tracks OR --qids (script enforces).
    if qids:
        args += ["--qids", qids]
    elif tracks:
        args += ["--tracks", tracks]
    else:
        args += ["--all"]
    if propose_fixes:
        args += ["--propose-fixes"]
    args += ["--workers", str(workers), "--max-calls", str(max_calls)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if output:
        args += ["--output", str(output)]
    if dry_run:
        args += ["--dry-run"]
    raise typer.Exit(code=_exec("audit_corpus_batched.py", args))


# ─── vault audit review ──────────────────────────────────────────────────


@audit_app.command("review")
def review_cmd(
    input_path: Path = typer.Option(
        ..., "--input",
        help="Path to 01_audit.json from a --propose-fixes run.",
    ),
    dispositions_out: Path | None = typer.Option(
        None, "--dispositions-out",
        help="Sidecar JSON for resumable review (default <input-dir>/02_dispositions.json).",
    ),
    filter_track: str | None = typer.Option(
        None, "--filter-track",
        help="Only review qids in this track.",
    ),
    filter_gate: str | None = typer.Option(
        None, "--filter-gate",
        help="Only review rows where this gate failed.",
    ),
    auto_accept_format: bool = typer.Option(
        False, "--auto-accept-format",
        help="Auto-accept format-marker-only corrections (lower-risk).",
    ),
    limit: int | None = typer.Option(
        None, "--limit",
        help="Cap how many corrections to review this session.",
    ),
) -> None:
    """Walk proposed corrections interactively (a/r/e/s/q). Wraps apply_corrections.py."""
    args: list[str] = ["--input", str(input_path)]
    if dispositions_out:
        args += ["--dispositions-out", str(dispositions_out)]
    if filter_track:
        args += ["--filter-track", filter_track]
    if filter_gate:
        args += ["--filter-gate", filter_gate]
    if auto_accept_format:
        args += ["--auto-accept-format"]
    if limit is not None:
        args += ["--limit", str(limit)]
    raise typer.Exit(code=_exec("apply_corrections.py", args))


# ─── vault audit summarize ───────────────────────────────────────────────


@audit_app.command("summarize")
def summarize_cmd(
    input_path: Path = typer.Option(
        ..., "--input",
        help="Path to 01_audit.json.",
    ),
    output: Path | None = typer.Option(
        None, "--output",
        help="Output markdown (default docs/AUDIT_FINDINGS_<date>.md).",
    ),
    qid_limit: int = typer.Option(
        25, "--qid-limit",
        help="Max qids per priority list section.",
    ),
) -> None:
    """Generate AUDIT_FINDINGS markdown from a 01_audit.json."""
    args: list[str] = ["--input", str(input_path)]
    if output:
        args += ["--output", str(output)]
    if qid_limit != 25:
        args += ["--qid-limit", str(qid_limit)]
    raise typer.Exit(code=_exec("summarize_audit.py", args))


# ─── vault audit merge ───────────────────────────────────────────────────


@audit_app.command("merge")
def merge_cmd(
    inputs: list[Path] = typer.Option(
        ..., "--inputs",
        help="Run dirs to merge (multiple allowed).",
    ),
    output: Path = typer.Option(
        ..., "--output",
        help="Output dir (will create + write 01_audit.json).",
    ),
) -> None:
    """Merge multiple audit runs into one canonical 01_audit.json."""
    args: list[str] = ["--inputs", *(str(p) for p in inputs),
                        "--output", str(output)]
    raise typer.Exit(code=_exec("merge_audit_runs.py", args))


def register(app: typer.Typer) -> None:
    app.add_typer(audit_app, name="audit")


__all__ = ["register"]

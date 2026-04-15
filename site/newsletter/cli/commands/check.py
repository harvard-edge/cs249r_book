"""news check — run preflight validation on a draft."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand
from ..core.console import error
from ..core.theme import Theme
from ..core.validate import Check, Report, validate_draft


SEVERITY_STYLE = {
    "ok":    (Theme.SUCCESS,  "\u2713"),
    "warn":  (Theme.WARNING,  "\u26a0"),
    "error": (Theme.ERROR,    "\u2717"),
}


class CheckCommand(BaseCommand):
    category = "info"

    @property
    def name(self) -> str:
        return "check"

    @property
    def description(self) -> str:
        return "Preflight a draft (frontmatter, figures, deps, auth)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "path",
            type=Path,
            help="Path to the draft (or a slug resolved against drafts/)",
        )

    def run(self, args: Namespace) -> int:
        draft_path = self._resolve_draft_path(args.path)
        if draft_path is None:
            return 1

        report = validate_draft(draft_path, self.config.env_file)
        render_report(self.console, report)
        return 0 if report.ok else 1

    def _resolve_draft_path(self, path_arg: Path) -> Path | None:
        candidates = [path_arg]
        if not path_arg.suffix:
            candidates.append(self.config.drafts_dir / f"{path_arg.name}.md")
        if not path_arg.is_absolute() and "/" not in str(path_arg):
            candidates.append(self.config.drafts_dir / path_arg.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        error(f"Draft not found. Tried:\n  " + "\n  ".join(str(c) for c in candidates))
        return None


def render_report(console, report: Report) -> None:
    """Render a validation Report as a Rich table plus a summary panel."""
    console.print()
    console.print(Panel(
        f"[{Theme.EMPHASIS}]{report.draft.name}[/]",
        title="Preflight check",
        border_style=Theme.BORDER_DEFAULT,
        expand=False,
    ))

    table = Table(show_header=True, header_style=Theme.SECTION, box=None, padding=(0, 1))
    table.add_column(" ", width=2, no_wrap=True)
    table.add_column("Check", style=Theme.EMPHASIS)
    table.add_column("Detail", style=Theme.DIM)

    for check in report.checks:
        color, glyph = SEVERITY_STYLE[check.severity]
        table.add_row(f"[{color}]{glyph}[/]", check.name, check.message)
    console.print(table)
    console.print()

    _render_summary(console, report)


def _render_summary(console, report: Report) -> None:
    ok_count   = sum(1 for c in report.checks if c.severity == "ok")
    warn_count = sum(1 for c in report.checks if c.severity == "warn")
    err_count  = sum(1 for c in report.checks if c.severity == "error")

    if err_count:
        border = Theme.BORDER_ERROR
        status = f"[{Theme.ERROR}]Not ready to push.[/] {err_count} error(s) must be resolved."
    elif warn_count:
        border = Theme.BORDER_WARNING
        status = f"[{Theme.WARNING}]Ready with warnings.[/] {warn_count} warning(s); safe to push."
    else:
        border = Theme.BORDER_SUCCESS
        status = f"[{Theme.SUCCESS}]Ready to push.[/] All checks passed."

    summary = (
        f"{status}\n\n"
        f"[{Theme.DIM}]{ok_count} ok  ·  {warn_count} warn  ·  {err_count} error[/]"
    )
    console.print(Panel(summary, border_style=border, expand=False))

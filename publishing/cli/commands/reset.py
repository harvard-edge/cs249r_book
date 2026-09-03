"""Reset generated/local CLI state to a clean baseline."""

from __future__ import annotations

import argparse
from typing import List

from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.table import Table

console = Console()


class ResetCommand:
    """Reset build YAML manifests after scoped builds."""

    FORMATS = ("html", "pdf", "epub")

    def __init__(self, build_command):
        self.build_command = build_command

    def run(self, args: List[str]) -> bool:
        if not args or args[0] in ("help", "-h", "--help"):
            self._print_help()
            return True

        parser = argparse.ArgumentParser(
            prog="binder reset",
            description="Reset build YAML configs to their full-book state",
        )
        parser.add_argument(
            "target",
            choices=[*self.FORMATS, "all"],
            help="Build config family to reset",
        )
        volume = parser.add_mutually_exclusive_group()
        volume.add_argument("--vol1", action="store_true", help="Reset Volume I config(s)")
        volume.add_argument("--vol2", action="store_true", help="Reset Volume II config(s)")

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        selected_volume = "vol1" if ns.vol1 else "vol2" if ns.vol2 else None
        formats = self.FORMATS if ns.target == "all" else (ns.target,)

        ok = True
        for fmt in formats:
            ok = self.build_command.reset_build_config(fmt, selected_volume) and ok
        return ok

    def _print_help(self) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Command", style="cyan", width=34)
        table.add_column("Description", style="white", width=44)
        table.add_row("reset html", "Reset both HTML YAML manifests")
        table.add_row("reset pdf --vol1", "Reset Volume I PDF YAML manifest")
        table.add_row("reset epub --vol2", "Reset Volume II EPUB YAML manifest")
        table.add_row("reset all", "Reset HTML, PDF, and EPUB manifests")
        console.print(Panel(
            table,
            title=_rich_escape("binder reset <html|pdf|epub|all> [--vol1|--vol2]"),
            border_style="cyan",
        ))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder reset pdf --vol1[/cyan]")
        console.print("  [cyan]./binder reset html --vol2[/cyan]")
        console.print("  [cyan]./binder reset all[/cyan]")
        console.print()

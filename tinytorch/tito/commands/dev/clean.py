"""
Developer clean command for TinyTorch CLI.

Wraps clean targets so the VS Code extension and other tools
can call Tito instead of raw make commands.

Usage:
    tito dev clean          Clean all generated files (project root)
    tito dev clean site     Clean site build artifacts
"""

import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..base import BaseCommand


class DevCleanCommand(BaseCommand):
    """Developer clean command — removes build artifacts."""

    @property
    def name(self) -> str:
        return "clean"

    @property
    def description(self) -> str:
        return "Clean build artifacts"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            'target',
            nargs='?',
            default='all',
            choices=['all', 'guide', 'site', 'book', 'paper'],
            help='What to clean: all (default), guide (or site), book, paper'
        )

    def run(self, args: Namespace) -> int:
        target = args.target or 'all'
        console = self.console

        def _run_make_clean(dir_path: Path, label: str) -> int:
            if not dir_path.exists():
                console.print(f"[red]❌ Directory not found: {dir_path}[/red]")
                return 1
            console.print(f"[cyan]🧹 Cleaning {label} build artifacts...[/cyan]")
            try:
                res = subprocess.run(['make', 'clean'], cwd=str(dir_path))
                return res.returncode
            except FileNotFoundError:
                console.print("[red]❌ 'make' is not installed or not on your PATH[/red]")
                console.print("  This command needs GNU Make to run its clean targets.")
                console.print("  Windows: install via 'choco install make', WSL, or Git Bash's own")
                console.print("           MinGW package manager.")
                console.print("  macOS/Linux: usually preinstalled, or 'brew install make' / 'apt install make'.")
                return 1

        if target in ('guide', 'site'):
            return _run_make_clean(self.config.project_root / 'guide', 'guide')
        elif target == 'book':
            return _run_make_clean(self.config.project_root / 'book', 'book')
        elif target == 'paper':
            return _run_make_clean(self.config.project_root / 'paper', 'paper')
        else:
            rc = _run_make_clean(self.config.project_root, 'all generated files')
            guide_dir = self.config.project_root / 'guide'
            if guide_dir.exists() and (guide_dir / 'Makefile').exists():
                _run_make_clean(guide_dir, 'guide')
            return rc

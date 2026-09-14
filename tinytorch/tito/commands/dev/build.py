"""
Developer build command for TinyTorch CLI.

Wraps site/paper build targets so the VS Code extension and other tools
can call Tito instead of raw make commands.

Usage:
    tito dev build html     Build HTML site
    tito dev build serve    Build and serve locally
    tito dev build pdf      Build PDF course guide
    tito dev build paper    Build research paper
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..base import BaseCommand


BUILD_TARGETS = {
    'html': {
        'command': ['make', 'html'],
        'cwd': 'site',
        'label': 'Build HTML Site',
    },
    'serve': {
        'command': ['make', 'serve'],
        'cwd': 'site',
        'label': 'Build & Serve Site',
    },
    'pdf': {
        'command': ['make', 'pdf'],
        'cwd': 'site',
        'label': 'Build PDF Course Guide',
    },
    'paper': {
        'command': ['make', 'paper'],
        'cwd': 'site',
        'label': 'Build Research Paper',
    },
}


class DevBuildCommand(BaseCommand):
    """Developer build command — wraps make targets for site/paper builds."""

    @property
    def name(self) -> str:
        return "build"

    @property
    def description(self) -> str:
        return "Build site, PDF, or paper"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            'target',
            choices=list(BUILD_TARGETS.keys()),
            help='Build target: html, serve, pdf, paper'
        )

    def run(self, args: Namespace) -> int:
        target = args.target
        config = BUILD_TARGETS[target]
        console = self.console

        cwd = self.config.project_root / config['cwd']
        if not cwd.exists():
            console.print(f"[red]❌ Directory not found: {cwd}[/red]")
            return 1

        console.print(f"[cyan]🔨 {config['label']}...[/cyan]")

        # `make` isn't bundled with Git Bash on Windows (unlike git/python,
        # it has no equivalent auto-installed fallback), so this is a very
        # reachable crash for any Windows user without WSL or a separate
        # make install: subprocess.run raises FileNotFoundError with no
        # indication of *why*, rather than something recognizable as
        # "install this tool".
        try:
            result = subprocess.run(
                config['command'],
                cwd=str(cwd),
            )
        except FileNotFoundError:
            console.print("[red]❌ 'make' is not installed or not on your PATH[/red]")
            console.print("  This command needs GNU Make to run its build targets.")
            console.print("  Windows: install via 'choco install make', WSL, or Git Bash's own")
            console.print("           MinGW package manager.")
            console.print("  macOS/Linux: usually preinstalled, or 'brew install make' / 'apt install make'.")
            return 1

        return result.returncode

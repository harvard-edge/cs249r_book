"""
Health command for TinyTorch CLI: environment health check and validation.
"""

import sys
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from ..base import BaseCommand

class HealthCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "health"

    @property
    def description(self) -> str:
        return "Quick environment health check"

    def add_arguments(self, parser: ArgumentParser) -> None:
        # No arguments needed for quick health check
        pass

    def run(self, args: Namespace) -> int:
        console = self.console

        # Run quick health check
        console.print(Panel("💚 TinyTorch Environment Health Check",
                           title="System Health", border_style="bright_green"))
        console.print()

        # Environment checks table - STATUS ONLY (no version numbers)
        env_table = Table(title="Environment Check", show_header=True, header_style="bold blue")
        env_table.add_column("Component", style="cyan", width=30)
        env_table.add_column("Status", justify="center", width=15)

        # Python environment
        env_table.add_row("Python", "[green]✅ OK[/green]")

        # Virtual environment - check if it exists and if we're using it
        venv_exists = self.venv_path.exists()
        in_venv = (
            # Method 1: Check VIRTUAL_ENV environment variable (most reliable for activation)
            os.environ.get('VIRTUAL_ENV') is not None or
            # Method 2: Check sys.prefix vs sys.base_prefix (works for running Python in venv)
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            # Method 3: Check for sys.real_prefix (older Python versions)
            hasattr(sys, 'real_prefix')
        )

        if venv_exists and in_venv:
            venv_status = "[green]✅ OK[/green]"
        elif venv_exists:
            venv_status = "[yellow]⚠️  Not Activated[/yellow]"
        else:
            venv_status = "[red]❌ Missing[/red]"
        env_table.add_row("Virtual Environment", venv_status)

        # Required dependencies (from requirements.txt)
        required_deps = [
            ('NumPy', 'numpy'),
            ('Rich', 'rich'),
            ('PyYAML', 'yaml'),
            ('Pytest', 'pytest'),
            ('Jupytext', 'jupytext'),
        ]
        for display_name, import_name in required_deps:
            try:
                __import__(import_name)
                env_table.add_row(display_name, "[green]✅ OK[/green]")
            except ImportError:
                env_table.add_row(display_name, "[red]❌ Missing[/red]")

        # Optional dependencies (nice to have, not required for core workflow)
        optional_deps = [
            ('JupyterLab', 'jupyterlab'),
            ('Matplotlib', 'matplotlib'),
        ]
        for display_name, import_name in optional_deps:
            try:
                __import__(import_name)
                env_table.add_row(f"{display_name} (optional)", "[green]✅ Installed[/green]")
            except ImportError:
                env_table.add_row(f"{display_name} (optional)", "[dim]○ Not installed[/dim]")

        console.print(env_table)
        console.print()

        # Module structure table
        struct_table = Table(title="Module Structure", show_header=True, header_style="bold magenta")
        struct_table.add_column("Path", style="cyan", width=25)
        struct_table.add_column("Status", justify="left")
        struct_table.add_column("Type", style="dim", width=25)

        required_paths = [
            ('src/', 'Source modules directory (student workspace)'),
            ('tests/', 'Test suite directory'),
            ('tito/', 'CLI infrastructure'),
            ('requirements.txt', 'Dependencies file')
        ]

        for path, desc in required_paths:
            if Path(path).exists():
                struct_table.add_row(path, "[green]✅ Found[/green]", desc)
            else:
                struct_table.add_row(path, "[red]❌ Missing[/red]", desc)

        console.print(struct_table)
        console.print()

        # Module implementations
        console.print(Panel("📋 Implementation Status",
                           title="Module Status", border_style="bright_blue"))

        # Import and run the info command to show module status
        from .info import InfoCommand
        info_cmd = InfoCommand(self.config)
        info_args = ArgumentParser()
        info_cmd.add_arguments(info_args)
        info_args = info_args.parse_args([])  # Empty args for info
        return info_cmd.run(info_args)

"""
Info command for TinyTorch CLI: shows system and environment information.
"""

from argparse import ArgumentParser, Namespace
import sys
import os
import platform
import shutil
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from ..base import BaseCommand

class InfoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "info"

    @property
    def description(self) -> str:
        return "Show system and environment information"

    def add_arguments(self, parser: ArgumentParser) -> None:
        # No arguments needed
        pass

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print()
        console.print(Panel(
            "💻 TinyTorch System & Environment Information",
            title="System Info",
            border_style="bright_cyan"
        ))
        console.print()

        # System Information Table
        info_table = Table(title="System Details", show_header=True, header_style="bold cyan")
        info_table.add_column("Component", style="yellow", width=25)
        info_table.add_column("Value", style="white", width=50)

        # Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        info_table.add_row("Python Version", python_version)

        # Platform
        system_name = platform.system()
        system_release = platform.release()
        machine = platform.machine()
        info_table.add_row("Platform", f"{system_name} {system_release} ({machine})")

        # Working Directory
        working_dir = Path.cwd()
        info_table.add_row("Working Directory", str(working_dir))

        # Virtual Environment
        venv_exists = self.venv_path.exists()
        in_venv = (
            os.environ.get('VIRTUAL_ENV') is not None or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            hasattr(sys, 'real_prefix')
        )

        if venv_exists and in_venv:
            venv_status = "✅ OK"
            venv_path = os.environ.get('VIRTUAL_ENV', str(self.venv_path))
        elif venv_exists:
            venv_status = "⚠️  Not Activated"
            venv_path = str(self.venv_path)
        else:
            venv_status = "❌ Not Found"
            venv_path = "N/A"

        info_table.add_row("Virtual Environment", venv_status)
        if venv_path != "N/A":
            info_table.add_row("  └─ Path", venv_path)

        # TinyTorch Version
        try:
            import tinytorch
            tinytorch_version = getattr(tinytorch, '__version__', 'unknown')
            tinytorch_path = Path(tinytorch.__file__).parent
            info_table.add_row("TinyTorch Version", tinytorch_version)
            info_table.add_row("  └─ Location", str(tinytorch_path))
        except ImportError:
            info_table.add_row("TinyTorch Version", "❌ Not Installed")

        # Disk Space
        try:
            disk_usage = shutil.disk_usage(working_dir)
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            info_table.add_row("Disk Space", f"{free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_percent:.1f}% used)")
        except Exception:
            info_table.add_row("Disk Space", "Unable to determine")

        # Memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            free_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            used_percent = mem.percent
            info_table.add_row("Memory", f"{free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_percent:.1f}% used)")
        except ImportError:
            info_table.add_row("Memory", "Install psutil for memory info")
        except Exception:
            info_table.add_row("Memory", "Unable to determine")

        console.print(info_table)
        console.print()

        # Helpful tips pointing to other commands
        console.print(Panel(
            "[dim]💡 For more information:[/dim]\n"
            "• Run [cyan]tito system health[/cyan] for environment health check and validation",
            border_style="blue"
        ))

        return 0

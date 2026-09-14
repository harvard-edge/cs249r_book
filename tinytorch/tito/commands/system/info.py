"""
Info command for TinyTorch CLI: shows system and environment information.
"""

from argparse import ArgumentParser, Namespace
import json as json_module
import sys
import os
import platform
import shutil
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from ..base import BaseCommand


def _gather_system_info(venv_path: Path) -> dict:
    """Gather system information as a dictionary. Shared by both output modes."""
    # Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Platform
    system_name = platform.system()
    system_release = platform.release()
    machine = platform.machine()

    # Virtual Environment
    in_venv = (
        os.environ.get('VIRTUAL_ENV') is not None or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        hasattr(sys, 'real_prefix')
    )

    # TinyTorch Version
    try:
        import tinytorch
        tinytorch_version = getattr(tinytorch, '__version__', 'unknown')
    except ImportError:
        tinytorch_version = "not installed"

    # NumPy Version
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not installed"

    return {
        "python_version": python_version,
        "platform": f"{system_name} {system_release} ({machine})",
        "tinytorch_version": tinytorch_version,
        "numpy_version": numpy_version,
        "venv_active": in_venv,
    }


class InfoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "info"

    @property
    def description(self) -> str:
        return "Show system and environment information"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output as JSON (for IDE integrations)'
        )

    def run(self, args: Namespace) -> int:
        info = _gather_system_info(self.venv_path)

        if getattr(args, 'json', False):
            print(json_module.dumps(info))
            return 0

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

        info_table.add_row("Python Version", info["python_version"])
        info_table.add_row("Platform", info["platform"])

        # Working Directory
        working_dir = Path.cwd()
        info_table.add_row("Working Directory", str(working_dir))

        # Virtual Environment (detailed for Rich output)
        venv_exists = self.venv_path.exists()
        in_venv = info["venv_active"]

        if venv_exists and in_venv:
            venv_status = "✅ OK"
            venv_path_str = os.environ.get('VIRTUAL_ENV', str(self.venv_path))
        elif venv_exists:
            venv_status = "⚠️  Not Activated"
            venv_path_str = str(self.venv_path)
        else:
            venv_status = "❌ Not Found"
            venv_path_str = "N/A"

        info_table.add_row("Virtual Environment", venv_status)
        if venv_path_str != "N/A":
            info_table.add_row("  └─ Path", venv_path_str)

        # TinyTorch Version
        if info["tinytorch_version"] != "not installed":
            info_table.add_row("TinyTorch Version", info["tinytorch_version"])
            try:
                import tinytorch
                info_table.add_row("  └─ Location", str(Path(tinytorch.__file__).parent))
            except ImportError:
                pass
        else:
            info_table.add_row("TinyTorch Version", "❌ Not Installed")

        # NumPy Version
        if info["numpy_version"] != "not installed":
            info_table.add_row("NumPy Version", info["numpy_version"])
        else:
            info_table.add_row("NumPy Version", "❌ Not Installed")

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

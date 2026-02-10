"""
Health command for TinyTorch CLI: environment health check and validation.
"""

import sys
import os
import subprocess
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

        # Track issues for summary
        issues = []

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
            issues.append("Virtual environment exists but is not activated")
        else:
            venv_status = "[red]❌ Missing[/red]"
            issues.append("Virtual environment not found — run: tito setup")
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
                issues.append(f"{display_name} not installed")

        # Workflow-critical dependencies (needed for module complete/export)
        workflow_deps = [
            ('nbdev (export)', 'nbdev'),
            ('ipykernel (Jupyter)', 'ipykernel'),
        ]
        for display_name, import_name in workflow_deps:
            try:
                __import__(import_name)
                env_table.add_row(display_name, "[green]✅ OK[/green]")
            except ImportError:
                env_table.add_row(display_name, "[red]❌ Missing[/red]")
                issues.append(f"{display_name} not installed — run: pip install {import_name}")

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

        # ── Notebook Readiness checks ──
        # These diagnose the exact "ModuleNotFoundError" problem students hit
        nb_table = Table(title="Notebook Readiness", show_header=True, header_style="bold yellow")
        nb_table.add_column("Check", style="cyan", width=30)
        nb_table.add_column("Status", justify="center", width=15)
        nb_table.add_column("Detail", style="dim", width=35)

        # 1. Can we import the tinytorch package at all?
        try:
            import tinytorch
            nb_table.add_row(
                "TinyTorch package",
                "[green]✅ OK[/green]",
                f"v{getattr(tinytorch, '__version__', 'unknown')}"
            )
        except ImportError as e:
            nb_table.add_row(
                "TinyTorch package",
                "[red]❌ Not importable[/red]",
                "run: pip install -e ."
            )
            issues.append("tinytorch package not importable — run: pip install -e .")

        # 2. Does tinytorch/core/tensor.py exist? (the most common failure point)
        core_dir = self.config.project_root / "tinytorch" / "core"
        tensor_file = core_dir / "tensor.py"
        if tensor_file.exists():
            nb_table.add_row(
                "Core module files",
                "[green]✅ OK[/green]",
                f"{len(list(core_dir.glob('*.py')))} files in tinytorch/core/"
            )
        else:
            nb_table.add_row(
                "Core module files",
                "[red]❌ Missing[/red]",
                "tinytorch/core/tensor.py not found"
            )
            issues.append("tinytorch/core/tensor.py missing — package may be corrupted")

        # 3. Can the Tensor class actually be imported?
        try:
            from tinytorch.core.tensor import Tensor
            if Tensor is not None:
                nb_table.add_row(
                    "Tensor import",
                    "[green]✅ OK[/green]",
                    "from tinytorch.core.tensor import Tensor"
                )
            else:
                nb_table.add_row(
                    "Tensor import",
                    "[yellow]⚠️  None[/yellow]",
                    "Module 01 may not be exported yet"
                )
                issues.append("Tensor is None — complete Module 01: tito module complete 01")
        except ImportError as e:
            nb_table.add_row(
                "Tensor import",
                "[red]❌ Failed[/red]",
                str(e)[:35]
            )
            issues.append(f"Cannot import Tensor: {e}")

        # 4. Jupyter kernel check — does a kernel exist that points to this Python?
        kernel_status, kernel_detail = self._check_jupyter_kernel()
        nb_table.add_row("Jupyter kernel", kernel_status, kernel_detail)
        if "❌" in kernel_status or "⚠️" in kernel_status:
            issues.append(kernel_detail)

        # 5. Check that this Python == the Jupyter kernel's Python
        #    (catches the exact mismatch that causes ModuleNotFoundError in notebooks)
        kernel_python = self._get_kernel_python()
        if kernel_python:
            if os.path.realpath(kernel_python) == os.path.realpath(sys.executable):
                nb_table.add_row(
                    "Kernel ↔ tito Python",
                    "[green]✅ Match[/green]",
                    "Same interpreter"
                )
            else:
                nb_table.add_row(
                    "Kernel ↔ tito Python",
                    "[red]❌ Mismatch[/red]",
                    f"Kernel: {kernel_python}"
                )
                issues.append(
                    f"Jupyter kernel uses a different Python than tito — "
                    f"run: python -m ipykernel install --user --name tinytorch"
                )
        else:
            nb_table.add_row(
                "Kernel ↔ tito Python",
                "[dim]○ Skipped[/dim]",
                "No kernel to check"
            )

        console.print(nb_table)
        console.print()

        # ── Issues Summary ──
        if issues:
            console.print(Panel(
                "\n".join(f"  • {issue}" for issue in issues),
                title=f"⚠️  {len(issues)} issue{'s' if len(issues) > 1 else ''} found",
                border_style="yellow"
            ))
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

    def _check_jupyter_kernel(self):
        """Check if a TinyTorch Jupyter kernel is registered."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "jupyter", "kernelspec", "list"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "tinytorch" in result.stdout:
                return "[green]✅ Registered[/green]", "tinytorch kernel found"
            elif result.returncode == 0:
                # Jupyter works but no tinytorch kernel
                return (
                    "[yellow]⚠️  No tinytorch kernel[/yellow]",
                    "run: python -m ipykernel install --user --name tinytorch"
                )
            else:
                return "[yellow]⚠️  Cannot list[/yellow]", "jupyter kernelspec list failed"
        except FileNotFoundError:
            return "[dim]○ Skipped[/dim]", "jupyter not installed"
        except Exception:
            return "[dim]○ Skipped[/dim]", "could not check"

    def _get_kernel_python(self):
        """Get the Python executable path used by the default or tinytorch Jupyter kernel."""
        try:
            import json

            # Try tinytorch kernel first, then python3 default
            for kernel_name in ("tinytorch", "python3"):
                result = subprocess.run(
                    [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    return None

                data = json.loads(result.stdout)
                kernels = data.get("kernelspecs", {})
                if kernel_name in kernels:
                    kernel_dir = kernels[kernel_name].get("resource_dir", "")
                    kernel_json = Path(kernel_dir) / "kernel.json"
                    if kernel_json.exists():
                        spec = json.loads(kernel_json.read_text())
                        argv = spec.get("argv", [])
                        if argv:
                            return argv[0]  # First element is the Python path
        except Exception:
            pass
        return None

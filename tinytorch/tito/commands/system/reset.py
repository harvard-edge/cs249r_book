"""
System Reset Command for TinyTorch CLI.

Resets the TinyTorch development environment to a pristine state:
- Clears modules/ directory (student notebooks)
- Clears tinytorch/core/ (exported package code)
- Optionally resets progress tracking

This is useful for:
- Testing fresh install experience
- CI/CD pipeline resets
- Starting over with a clean slate
"""

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel

from ..base import BaseCommand
from ...core.modules import get_module_mapping


class SystemResetCommand(BaseCommand):
    """Command to reset TinyTorch to pristine state."""

    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset TinyTorch to pristine state"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add reset command arguments."""
        parser.add_argument(
            "--force", "-f",
            action="store_true",
            help="Skip confirmation prompt"
        )
        parser.add_argument(
            "--keep-progress",
            action="store_true",
            help="Keep progress tracking data (only reset code)"
        )
        parser.add_argument(
            "--ci",
            action="store_true",
            help="CI mode: no prompts, exit codes only"
        )

    def run(self, args: Namespace) -> int:
        """Execute the system reset."""
        console = self.console
        project_root = self.config.project_root

        # Show what will be reset
        if not args.ci:
            console.print()
            console.print(Panel(
                "[bold red]⚠️  SYSTEM RESET[/bold red]\n\n"
                "This will remove:\n"
                "  • [bold]modules/[/bold] - All student notebooks (01_tensor/, 02_activations/, etc.)\n"
                "  • [bold]tinytorch/core/[/bold] - All exported module code\n"
                + ("" if args.keep_progress else "  • [bold]Progress tracking[/bold] - Completion history\n") +
                "\n"
                "[dim]The src/ files remain untouched - you can rebuild everything with:[/dim]\n"
                "  [cyan]tito module complete --all[/cyan]",
                title="🔄 System Reset",
                border_style="red"
            ))
            console.print()

        # Confirm unless --force or --ci
        if not args.force and not args.ci:
            try:
                response = input("Are you sure you want to reset? (type 'yes' to confirm): ").strip().lower()
                if response != 'yes':
                    console.print("[yellow]Reset cancelled.[/yellow]")
                    return 0
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Reset cancelled.[/yellow]")
                return 0

        # Perform reset
        errors = []

        # 1. Clear modules/ directory
        modules_dir = project_root / "modules"
        modules_cleared = 0
        if modules_dir.exists():
            module_mapping = get_module_mapping()
            for item in modules_dir.iterdir():
                if item.is_dir() and item.name[0].isdigit():
                    try:
                        shutil.rmtree(item)
                        modules_cleared += 1
                    except Exception as e:
                        errors.append(f"modules/{item.name}: {e}")

        # 2. Clear tinytorch/core/ (keep __init__.py)
        core_dir = project_root / "tinytorch" / "core"
        core_cleared = 0
        if core_dir.exists():
            for py_file in core_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    try:
                        py_file.unlink()
                        core_cleared += 1
                    except Exception as e:
                        errors.append(f"tinytorch/core/{py_file.name}: {e}")

        # 3. Reset progress (unless --keep-progress)
        progress_reset = False
        if not args.keep_progress:
            progress_file = project_root / ".tinytorch" / "progress.json"
            if progress_file.exists():
                try:
                    progress_file.unlink()
                    progress_reset = True
                except Exception as e:
                    errors.append(f"progress.json: {e}")

        # Report results
        if args.ci:
            # JSON-style output for CI
            if errors:
                print(f"RESET FAILED: {len(errors)} errors")
                for err in errors:
                    print(f"  ERROR: {err}")
                return 1
            else:
                print(f"RESET OK: {modules_cleared} modules, {core_cleared} core files cleared")
                return 0
        else:
            if errors:
                console.print(Panel(
                    f"[red]Reset completed with errors:[/red]\n\n" +
                    "\n".join(f"  • {e}" for e in errors),
                    title="⚠️ Partial Reset",
                    border_style="yellow"
                ))
                return 1
            else:
                summary = []
                if modules_cleared > 0:
                    summary.append(f"{modules_cleared} module directories")
                if core_cleared > 0:
                    summary.append(f"{core_cleared} core files")
                if progress_reset:
                    summary.append("progress tracking")

                console.print(Panel(
                    f"[green]✅ Reset complete![/green]\n\n"
                    f"Cleared: {', '.join(summary) if summary else 'nothing to clear'}\n\n"
                    "[dim]To rebuild the package:[/dim]\n"
                    "  [cyan]tito module complete --all[/cyan]",
                    title="🔄 System Reset Complete",
                    border_style="green"
                ))
                return 0

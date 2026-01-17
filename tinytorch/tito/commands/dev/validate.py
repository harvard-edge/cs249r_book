"""
TinyTorch Release Validation Command

Simulates the complete student journey through the curriculum.
Uses the same commands students use to ensure the workflow works.

Flow:
1. Reset (clear modules/ and tinytorch/core/)
2. Phase 1 - Modules: For each module 01-20:
   a. tito dev export XX      (populate notebook - simulates student work)
   b. tito module complete XX (run tests + export to package)
3. Phase 2 - Milestones: Run all milestones
4. Summary

Usage:
    tito dev validate       # Full validation with rich output
    tito dev validate --ci  # JSON output for CI/CD
"""

import subprocess
import sys
import json
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from ..base import BaseCommand
from ..milestone import MILESTONE_SCRIPTS
from ...core.modules import get_module_mapping


@dataclass
class ModuleResult:
    """Result for a single module validation."""
    module_num: str
    module_name: str
    export_passed: bool = False
    export_time: float = 0.0
    complete_passed: bool = False
    complete_time: float = 0.0
    complete_message: str = ""

    @property
    def passed(self) -> bool:
        return self.export_passed and self.complete_passed


@dataclass
class MilestoneResult:
    """Result for a single milestone."""
    milestone_id: str
    name: str
    passed: bool = False
    duration: float = 0.0
    message: str = ""


@dataclass
class ValidationReport:
    """Complete validation report."""
    modules: List[ModuleResult] = field(default_factory=list)
    milestones: List[MilestoneResult] = field(default_factory=list)
    reset_passed: bool = False
    reset_time: float = 0.0
    total_duration: float = 0.0

    @property
    def all_passed(self) -> bool:
        if not self.reset_passed:
            return False
        if not all(m.passed for m in self.modules):
            return False
        if not all(m.passed for m in self.milestones):
            return False
        return True

    @property
    def modules_passed(self) -> int:
        return sum(1 for m in self.modules if m.passed)

    @property
    def milestones_passed(self) -> int:
        return sum(1 for m in self.milestones if m.passed)


class ValidateCommand(BaseCommand):
    """Simulate complete student journey for release validation."""

    @property
    def name(self) -> str:
        return "validate"

    @property
    def description(self) -> str:
        return "Simulate student journey for release validation"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--ci',
            action='store_true',
            help='CI mode: JSON output, strict exit codes'
        )
        parser.add_argument(
            '--skip-milestones',
            action='store_true',
            help='Skip milestone validation'
        )
        parser.add_argument(
            '--stop-on-fail',
            action='store_true',
            help='Stop at first failure'
        )
        parser.add_argument(
            '--module',
            type=str,
            help='Validate up to specific module (e.g., --module 08)'
        )

    def run(self, args: Namespace) -> int:
        """Run full curriculum validation simulating student workflow."""
        console = self.console
        project_root = self.config.project_root
        start_time = time.time()

        report = ValidationReport()

        # Get module list
        module_mapping = get_module_mapping()
        module_nums = sorted(module_mapping.keys(), key=lambda x: int(x))

        # Validate --module argument if provided
        if args.module:
            # Validate it's a valid integer
            try:
                target_int = int(args.module)
            except ValueError:
                console.print(f"[red]❌ Invalid module number: '{args.module}'[/red]")
                console.print("[dim]Module must be a number (e.g., --module 08)[/dim]")
                return 1

            target = args.module.zfill(2)

            # Check if module exists
            if target not in module_mapping:
                # Find max available module
                max_module = max(int(m) for m in module_nums)
                if target_int > max_module:
                    console.print(f"[yellow]⚠ Module {target} doesn't exist. Max is {str(max_module).zfill(2)}.[/yellow]")
                    console.print(f"[dim]Running all {len(module_nums)} available modules.[/dim]")
                    console.print()
                else:
                    console.print(f"[red]❌ Module {target} not found[/red]")
                    return 1
            else:
                module_nums = [m for m in module_nums if int(m) <= target_int]

        # Header
        if not args.ci:
            console.print()
            console.print(Panel(
                "[bold cyan]🔬 TinyTorch Release Validation[/bold cyan]\n\n"
                "[bold]Simulating student journey:[/bold]\n"
                f"  • {len(module_nums)} modules to validate\n"
                f"  • {len(MILESTONE_SCRIPTS)} milestones to run\n"
                "  • Using actual CLI commands\n\n"
                "[dim]This tests the real student workflow.[/dim]",
                border_style="cyan"
            ))
            console.print()

        # =====================================================================
        # Step 1: Reset (clear modules/ and tinytorch/core/)
        # =====================================================================
        if not args.ci:
            console.print("[bold]Step 1: Resetting to clean state...[/bold]")

        reset_start = time.time()
        report.reset_passed = self._reset_all(project_root)
        report.reset_time = time.time() - reset_start

        if not args.ci:
            if report.reset_passed:
                console.print(f"  [green]✓[/green] Cleared modules/ and tinytorch/core/ [dim]({report.reset_time:.1f}s)[/dim]")
            else:
                console.print(f"  [red]✗[/red] Reset failed")
                return self._finish(report, args, start_time)

        if not report.reset_passed:
            return self._finish(report, args, start_time)

        # =====================================================================
        # Step 2: Phase 1 - Validate all modules
        # =====================================================================
        if not args.ci:
            console.print()
            console.print(f"[bold]Step 2: Validating {len(module_nums)} modules...[/bold]")
            console.print()

        modules_failed = False
        for i, module_num in enumerate(module_nums, 1):
            module_name = module_mapping[module_num]
            result = ModuleResult(module_num=module_num, module_name=module_name)

            if not args.ci:
                console.print(f"[bold cyan]Module {module_num}[/bold cyan] ({module_name})")

            # Export from src (simulate student completing work)
            export_start = time.time()
            result.export_passed = self._export_module(project_root, module_num)
            result.export_time = time.time() - export_start

            if not args.ci:
                if result.export_passed:
                    console.print(f"  [green]✓[/green] Export (src → modules) [dim]({result.export_time:.1f}s)[/dim]")
                else:
                    console.print(f"  [red]✗[/red] Export failed")

            if not result.export_passed:
                report.modules.append(result)
                modules_failed = True
                if args.stop_on_fail:
                    break
                continue

            # Run module complete (tests + export to package)
            complete_start = time.time()
            result.complete_passed, result.complete_message = self._complete_module(
                project_root, module_num
            )
            result.complete_time = time.time() - complete_start

            if not args.ci:
                if result.complete_passed:
                    console.print(f"  [green]✓[/green] Complete (tests + package) [dim]({result.complete_time:.1f}s)[/dim]")
                    if result.complete_message:
                        console.print(f"    [dim]{result.complete_message}[/dim]")
                else:
                    console.print(f"  [red]✗[/red] Complete failed")
                    if result.complete_message:
                        console.print(f"    [red]{result.complete_message}[/red]")
                    modules_failed = True

            report.modules.append(result)

            if not result.complete_passed and args.stop_on_fail:
                break

            if not args.ci:
                console.print()

        # =====================================================================
        # Step 3: Phase 2 - Run all milestones
        # =====================================================================
        if not args.skip_milestones:
            if not args.ci:
                console.print()
                console.print(f"[bold]Step 3: Running {len(MILESTONE_SCRIPTS)} milestones...[/bold]")
                console.print()

            # Only run milestones if all modules passed (or if we want to try anyway)
            if modules_failed and args.stop_on_fail:
                if not args.ci:
                    console.print("  [yellow]Skipping milestones due to module failures[/yellow]")
            else:
                for milestone_id in sorted(MILESTONE_SCRIPTS.keys()):
                    milestone_info = MILESTONE_SCRIPTS[milestone_id]
                    milestone_name = milestone_info.get('name', f'Milestone {milestone_id}')

                    if not args.ci:
                        console.print(f"[bold cyan]Milestone {milestone_id}[/bold cyan]: {milestone_name}")

                    milestone_start = time.time()
                    passed, message = self._run_milestone(project_root, milestone_id)
                    duration = time.time() - milestone_start

                    result = MilestoneResult(
                        milestone_id=milestone_id,
                        name=milestone_name,
                        passed=passed,
                        duration=duration,
                        message=message
                    )
                    report.milestones.append(result)

                    if not args.ci:
                        if passed:
                            console.print(f"  [green]✓[/green] Passed [dim]({duration:.1f}s)[/dim]")
                        else:
                            console.print(f"  [red]✗[/red] Failed [dim]({duration:.1f}s)[/dim]")
                            if message:
                                console.print(f"    [red]{message}[/red]")
                        console.print()
        else:
            if not args.ci:
                console.print()
                console.print("[bold]Step 3: Skipping milestones (--skip-milestones)[/bold]")

        return self._finish(report, args, start_time)

    def _reset_all(self, project_root: Path) -> bool:
        """Reset modules/ and tinytorch/core/ to clean state."""
        import shutil

        try:
            # Clear modules/ (remove all module subdirectories)
            modules_dir = project_root / "modules"
            if modules_dir.exists():
                for item in modules_dir.iterdir():
                    if item.is_dir() and item.name[0].isdigit():
                        shutil.rmtree(item)

            # Clear tinytorch/core/ (remove all .py except __init__.py)
            core_dir = project_root / "tinytorch" / "core"
            if core_dir.exists():
                for py_file in core_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        py_file.unlink()

            return True
        except Exception:
            return False

    def _export_module(self, project_root: Path, module_num: str) -> bool:
        """Export module from src/ to modules/ using tito dev export."""
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / "bin" / "tito"), "dev", "export", module_num],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=120
            )
            return result.returncode == 0
        except Exception:
            return False

    def _complete_module(self, project_root: Path, module_num: str) -> tuple:
        """Run tito module complete (tests + export to package)."""
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / "bin" / "tito"), "module", "complete", module_num],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300  # 5 min per module
            )

            # Extract test summary from output if available
            message = ""
            for line in result.stdout.split('\n'):
                if 'passed' in line.lower() or 'failed' in line.lower():
                    message = line.strip()
                    break

            return result.returncode == 0, message
        except subprocess.TimeoutExpired:
            return False, "Timed out after 5 minutes"
        except Exception as e:
            return False, str(e)

    def _run_milestone(self, project_root: Path, milestone_id: str) -> tuple:
        """Run a milestone using tito milestone run. Returns (passed, message)."""
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / "bin" / "tito"), "milestone", "run", milestone_id],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=180  # 3 min per milestone
            )
            if result.returncode == 0:
                return True, ""
            else:
                # Try to extract error message
                error = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
                return False, error
        except subprocess.TimeoutExpired:
            return False, "Timed out after 3 minutes"
        except Exception as e:
            return False, str(e)

    def _finish(self, report: ValidationReport, args: Namespace, start_time: float) -> int:
        """Finish validation and output results."""
        report.total_duration = time.time() - start_time

        if args.ci:
            # JSON output
            output = {
                "success": report.all_passed,
                "duration_seconds": round(report.total_duration, 2),
                "summary": {
                    "modules_total": len(report.modules),
                    "modules_passed": report.modules_passed,
                    "milestones_total": len(report.milestones),
                    "milestones_passed": report.milestones_passed,
                },
                "modules": [
                    {
                        "num": m.module_num,
                        "name": m.module_name,
                        "export_passed": m.export_passed,
                        "complete_passed": m.complete_passed,
                    }
                    for m in report.modules
                ],
                "milestones": [
                    {
                        "id": m.milestone_id,
                        "name": m.name,
                        "passed": m.passed,
                    }
                    for m in report.milestones
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            # Rich summary
            console = self.console
            console.print()
            console.print("=" * 60)
            console.print("[bold]VALIDATION SUMMARY[/bold]")
            console.print("=" * 60)
            console.print()

            # Modules table
            console.print("[bold]Modules:[/bold]")
            table = Table(box=box.SIMPLE, show_header=False)
            table.add_column("Module", style="cyan", width=25)
            table.add_column("Export", justify="center", width=8)
            table.add_column("Complete", justify="center", width=10)

            for m in report.modules:
                export_status = "[green]✓[/green]" if m.export_passed else "[red]✗[/red]"
                complete_status = "[green]✓[/green]" if m.complete_passed else "[red]✗[/red]"
                table.add_row(f"{m.module_num} {m.module_name}", export_status, complete_status)

            console.print(table)

            # Milestones table
            if report.milestones:
                console.print()
                console.print("[bold]Milestones:[/bold]")
                for m in report.milestones:
                    status = "[green]✓[/green]" if m.passed else "[red]✗[/red]"
                    console.print(f"  {status} {m.milestone_id}: {m.name}")

            console.print()

            # Summary stats
            console.print(f"[bold]Modules:[/bold]    {report.modules_passed}/{len(report.modules)} passed")
            if report.milestones:
                console.print(f"[bold]Milestones:[/bold] {report.milestones_passed}/{len(report.milestones)} passed")
            console.print(f"[bold]Duration:[/bold]   {report.total_duration:.0f}s ({report.total_duration/60:.1f} min)")
            console.print()

            if report.all_passed:
                console.print(Panel(
                    "[bold green]✅ VALIDATION PASSED[/bold green]\n\n"
                    "All modules complete successfully.\n"
                    "All milestones run successfully.\n\n"
                    "[dim]Ready for release.[/dim]",
                    border_style="green"
                ))
            else:
                # Find failures
                failed_modules = [m for m in report.modules if not m.passed]
                failed_milestones = [m for m in report.milestones if not m.passed]

                failure_lines = []
                if failed_modules:
                    module_nums = ", ".join(m.module_num for m in failed_modules[:5])
                    if len(failed_modules) > 5:
                        module_nums += f" (+{len(failed_modules)-5} more)"
                    failure_lines.append(f"Failed modules: {module_nums}")
                if failed_milestones:
                    milestone_ids = ", ".join(m.milestone_id for m in failed_milestones)
                    failure_lines.append(f"Failed milestones: {milestone_ids}")

                console.print(Panel(
                    "[bold red]❌ VALIDATION FAILED[/bold red]\n\n"
                    + "\n".join(failure_lines) + "\n\n"
                    "[dim]Fix the issues above before releasing.[/dim]",
                    border_style="red"
                ))

        return 0 if report.all_passed else 1

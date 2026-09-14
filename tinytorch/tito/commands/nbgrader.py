"""
NBGrader integration commands for TinyTorch.

This command group is an instructor/developer convenience layer around
nbgrader's own workflow. TinyTorch owns module discovery and assignment
staging; nbgrader owns release, collection, autograding, feedback, and export.
"""

import json
import re
import subprocess
from argparse import ArgumentParser, Namespace, SUPPRESS
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand


MODULE_DIR_RE = re.compile(r"^(\d{2})_[A-Za-z0-9_]+$")
SOLUTION_BEGIN_MARKER = "### BEGIN SOLUTION"
SOLUTION_END_MARKER = "### END SOLUTION"
TEST_FUNCTION_RE = re.compile(r"^\s*def\s+test_", flags=re.MULTILINE)
SOLUTION_ROLE_RE = re.compile(r"\brole=[\"']?([A-Za-z0-9_-]+)")
RELEASE_TIERS = ("student", "challenge", "instructor")
VALID_SOLUTION_ROLES = {"core", "scaffold", "challenge", "instructor"}


class NBGraderCommand(BaseCommand):
    """NBGrader integration command group."""

    def __init__(self, config):
        super().__init__(config)
        self.project_root = config.project_root
        self.src_dir = self.project_root / "src"
        self.modules_dir = self.project_root / "modules"
        self.staging_dir = self.project_root / ".nbgrader_staging"
        self.assignments_dir = self.project_root / "assignments"
        self.source_dir = self.assignments_dir / "source"
        self.release_dir = self.assignments_dir / "release"
        self.submitted_dir = self.assignments_dir / "submitted"
        self.autograded_dir = self.assignments_dir / "autograded"
        self.feedback_dir = self.assignments_dir / "feedback"

    @property
    def name(self) -> str:
        return "nbgrader"

    @property
    def description(self) -> str:
        return "Assignment staging and grading workflow for nbgrader"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest="nbgrader_command",
            help="NBGrader subcommands",
            metavar="SUBCOMMAND",
        )

        subparsers.add_parser("init", help="Initialize nbgrader directories and config")

        generate_parser = subparsers.add_parser(
            "generate",
            help="Stage TinyTorch notebooks as nbgrader source assignments",
        )
        generate_parser.add_argument(
            "module",
            nargs="?",
            help="Module to stage (e.g., 01, 01_tensor, tensor)",
        )
        generate_parser.add_argument(
            "--all",
            action="store_true",
            help="Stage assignments for all modules",
        )
        generate_parser.add_argument(
            "--range",
            dest="module_range",
            help="Stage assignments for a module range (e.g., 01-04)",
        )
        generate_parser.add_argument(
            "--tier",
            choices=RELEASE_TIERS,
            default="student",
            help=SUPPRESS,
        )

        release_parser = subparsers.add_parser(
            "release",
            help="Create student release notebooks with nbgrader",
        )
        release_parser.add_argument("assignment", nargs="?", help="Assignment to release")
        release_parser.add_argument("--all", action="store_true", help="Release all staged assignments")

        collect_parser = subparsers.add_parser("collect", help="Collect student submissions")
        collect_parser.add_argument("assignment", nargs="?", help="Assignment to collect")
        collect_parser.add_argument("--all", action="store_true", help="Collect all released assignments")
        collect_parser.add_argument("--student", help="Limit collection to one student id")

        autograde_parser = subparsers.add_parser("autograde", help="Auto-grade submissions")
        autograde_parser.add_argument("assignment", nargs="?", help="Assignment to auto-grade")
        autograde_parser.add_argument("--all", action="store_true", help="Auto-grade all submitted assignments")
        autograde_parser.add_argument("--student", help="Limit autograding to one student id")
        autograde_parser.add_argument("--force", action="store_true", help="Overwrite existing autograded output")

        feedback_parser = subparsers.add_parser("feedback", help="Generate feedback for students")
        feedback_parser.add_argument("assignment", nargs="?", help="Assignment to generate feedback for")
        feedback_parser.add_argument("--all", action="store_true", help="Generate feedback for all graded assignments")
        feedback_parser.add_argument("--student", help="Limit feedback generation to one student id")

        subparsers.add_parser("status", help="Show local assignment status")

        analytics_parser = subparsers.add_parser("analytics", help="Show local assignment analytics")
        analytics_parser.add_argument("assignment", help="Assignment to analyze")

        report_parser = subparsers.add_parser("report", help="Export grades report with nbgrader")
        report_parser.add_argument(
            "--format",
            choices=["csv"],
            default="csv",
            help="Export format (default: csv)",
        )
        report_parser.add_argument(
            "--assignment",
            "--module",
            dest="assignment",
            help="Limit report to one assignment/module",
        )
        report_parser.add_argument("--student", help="Limit report to one student id")

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print(Panel(
            "[bold yellow]NBGrader Integration[/bold yellow]\n\n"
            "TinyTorch stages module notebooks for nbgrader, then delegates grading "
            "operations to the nbgrader CLI.\n\n"
            "[dim]This is instructor/developer tooling, not the main learner workflow.[/dim]",
            border_style="yellow",
        ))
        console.print()

        if not getattr(args, "nbgrader_command", None):
            console.print(Panel(
                "[bold cyan]NBGrader Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]init[/bold]       - Initialize nbgrader directories and config\n"
                "  • [bold]generate[/bold]   - Stage source assignments from TinyTorch notebooks\n"
                "  • [bold]release[/bold]    - Create student release notebooks\n"
                "  • [bold]collect[/bold]    - Collect student submissions\n"
                "  • [bold]autograde[/bold]  - Auto-grade submissions\n"
                "  • [bold]feedback[/bold]   - Generate feedback for students\n"
                "  • [bold]status[/bold]     - Show local assignment status\n"
                "  • [bold]analytics[/bold]  - Show local assignment analytics\n"
                "  • [bold]report[/bold]     - Export grades report\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito nbgrader init[/dim]\n"
                "[dim]  tito nbgrader generate 01[/dim]\n"
                "[dim]  tito nbgrader generate --all[/dim]\n"
                "[dim]  tito nbgrader generate --range 01-04[/dim]\n"
                "[dim]  tito nbgrader release 01_tensor[/dim]\n"
                "[dim]  tito nbgrader autograde 01_tensor --student student_id[/dim]",
                title="NBGrader Command Group",
                border_style="bright_cyan",
            ))
            return 0

        commands = {
            "init": self._init,
            "generate": self._generate,
            "release": self._release,
            "collect": self._collect,
            "autograde": self._autograde,
            "feedback": self._feedback,
            "status": self._status,
            "analytics": self._analytics,
            "report": self._report,
        }
        handler = commands.get(args.nbgrader_command)
        if handler is None:
            console.print(Panel(
                f"[red]Unknown nbgrader subcommand: {args.nbgrader_command}[/red]",
                title="Error",
                border_style="red",
            ))
            return 1
        return handler(args) if args.nbgrader_command not in {"init", "status"} else handler()

    def _discover_modules(self) -> Dict[str, str]:
        """Return module number to module directory mapping from src/."""
        modules = {}
        if not self.src_dir.exists():
            return modules
        for item in sorted(self.src_dir.iterdir()):
            if not item.is_dir():
                continue
            match = MODULE_DIR_RE.match(item.name)
            if match:
                modules[match.group(1)] = item.name
        return modules

    def _get_module_directories(self) -> List[str]:
        """Get all module directory names in curriculum order."""
        return list(self._discover_modules().values())

    def _resolve_module_name(self, module_input: str) -> Optional[str]:
        """Resolve module name from number, full directory name, or suffix."""
        modules = self._discover_modules()
        if not module_input:
            return None

        if module_input.isdigit():
            return modules.get(f"{int(module_input):02d}")

        if MODULE_DIR_RE.match(module_input):
            return module_input if module_input in modules.values() else None

        normalized = module_input.lower().replace("-", "_")
        for module_name in modules.values():
            suffix = module_name.split("_", 1)[1]
            if suffix == normalized:
                return module_name

        return None

    def _parse_module_range(self, range_str: str) -> List[str]:
        """Parse a numeric module range specification."""
        if "-" not in range_str:
            resolved = self._resolve_module_name(range_str)
            return [resolved] if resolved else []

        start, end = range_str.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return []

        start_num = int(start)
        end_num = int(end)
        if end_num < start_num:
            start_num, end_num = end_num, start_num

        modules = []
        discovered = self._discover_modules()
        for number in range(start_num, end_num + 1):
            module_name = discovered.get(f"{number:02d}")
            if module_name:
                modules.append(module_name)
        return modules

    def _short_name(self, module_name: str) -> str:
        return module_name.split("_", 1)[1] if "_" in module_name else module_name

    def _source_file_for(self, module_name: str) -> Path:
        return self.src_dir / module_name / f"{module_name}.py"

    def _notebook_for(self, module_name: str) -> Path:
        return self.modules_dir / module_name / f"{self._short_name(module_name)}.ipynb"

    def _assignment_file_for(self, module_name: str) -> Path:
        return self.source_dir / module_name / f"{self._short_name(module_name)}.ipynb"

    def _init(self) -> int:
        """Initialize nbgrader environment."""
        console = self.console
        console.print("Initializing NBGrader environment...")

        for directory in [
            self.assignments_dir,
            self.source_dir,
            self.release_dir,
            self.submitted_dir,
            self.autograded_dir,
            self.feedback_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"Created directory: {directory.relative_to(self.project_root)}")

        config_file = self.project_root / "nbgrader_config.py"
        if config_file.exists():
            console.print(f"Found nbgrader config: {config_file.relative_to(self.project_root)}")
        else:
            config_file.write_text(self._default_config_text(), encoding="utf-8")
            console.print(f"Created nbgrader config: {config_file.relative_to(self.project_root)}")

        version = self._run_external(["nbgrader", "--version"], capture_output=True)
        if version.returncode != 0:
            console.print("[red]NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1
        console.print(f"NBGrader version: {version.stdout.strip() or 'installed'}")

        result = self._run_external(["nbgrader", "db", "upgrade"], capture_output=True)
        if result.returncode != 0:
            console.print("[red]Failed to initialize nbgrader database[/red]")
            if result.stderr:
                console.print(f"[red]{result.stderr.strip()}[/red]")
            return 1

        console.print("[green]NBGrader environment initialized successfully.[/green]")
        return 0

    def _default_config_text(self) -> str:
        return (
            '"""TinyTorch nbgrader configuration."""\n\n'
            "c = get_config()  # noqa: F821\n\n"
            "# nbgrader works inside this course directory. It contains source/,\n"
            "# release/, submitted/, autograded/, feedback/, and gradebook.db.\n"
            'c.CourseDirectory.root = "assignments"\n'
            'c.CourseDirectory.course_id = "tinytorch"\n'
            "# TinyTorch validates marker metadata during staging. Allow written\n"
            "# reflection cells to be stripped without assigning manual-grade points.\n"
            "c.ClearSolutions.enforce_metadata = False\n"
            "c.ExecutePreprocessor.timeout = 120\n"
        )

    def _generate(self, args: Namespace) -> int:
        """Stage source assignments from TinyTorch notebooks."""
        console = self.console

        if args.all:
            modules_to_process = self._get_module_directories()
        elif getattr(args, "module_range", None):
            modules_to_process = self._parse_module_range(args.module_range)
            if not modules_to_process:
                console.print(f"[red]No modules found for range '{args.module_range}'[/red]")
                return 1
        elif args.module:
            resolved_module = self._resolve_module_name(args.module)
            if not resolved_module:
                console.print(f"[red]Module '{args.module}' not found[/red]")
                self._show_available_modules()
                return 1
            modules_to_process = [resolved_module]
        else:
            console.print("[red]Must specify either --all, --range, or a module name[/red]")
            return 1

        if not modules_to_process:
            console.print("[red]No TinyTorch modules found in src/[/red]")
            return 1

        release_tier = getattr(args, "tier", "student")
        console.print(f"Staging nbgrader source assignments: {', '.join(modules_to_process)}")

        for module_name in modules_to_process:
            if not self._generate_single_module(module_name, release_tier=release_tier):
                return 1

        console.print("[green]All requested source assignments staged successfully.[/green]")
        return 0

    def _generate_single_module(self, module_name: str, *, release_tier: str = "student") -> bool:
        """Stage one module notebook under assignments/source."""
        console = self.console
        notebook_file = self._ensure_module_notebook(module_name)
        if notebook_file is None:
            return False

        assignment_file = self._assignment_file_for(module_name)
        assignment_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            notebook = json.loads(notebook_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(f"[red]Invalid notebook JSON: {notebook_file.relative_to(self.project_root)} ({exc})[/red]")
            return False

        validation_errors = self._prepare_notebook_for_nbgrader(notebook, release_tier=release_tier)
        if validation_errors:
            console.print(f"[red]NBGrader metadata errors in {notebook_file.relative_to(self.project_root)}:[/red]")
            for error in validation_errors:
                console.print(f"[red]  • {error}[/red]")
            return False

        assignment_file.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        console.print(
            f"Staged {notebook_file.relative_to(self.project_root)} "
            f"→ {assignment_file.relative_to(self.project_root)}"
        )
        return True

    def _ensure_module_notebook(self, module_name: str) -> Optional[Path]:
        """Return a notebook reflecting the current src/ content.

        notebook_file (modules/<name>/<name>.ipynb) is the same path
        `tito module start`/`resume` open for students to edit. This method
        must never overwrite it: if it already exists, stage a fresh
        conversion into a private nbgrader-only cache instead, so a routine
        `git pull` that bumps a source file's mtime can't silently clobber a
        student's in-progress notebook when an instructor runs
        `tito nbgrader generate`.
        """
        notebook_file = self._notebook_for(module_name)
        source_file = self._source_file_for(module_name)

        if not source_file.exists():
            if notebook_file.exists():
                return notebook_file
            self.console.print(f"[red]Source file not found: {source_file.relative_to(self.project_root)}[/red]")
            return None

        notebook_up_to_date = (
            notebook_file.exists() and source_file.stat().st_mtime <= notebook_file.stat().st_mtime
        )
        if notebook_up_to_date:
            return notebook_file

        if notebook_file.exists():
            staged_file = self.staging_dir / module_name / f"{self._short_name(module_name)}.ipynb"
            reason = "Student notebook exists; staging nbgrader source separately"
            target = staged_file
        else:
            notebook_file.parent.mkdir(parents=True, exist_ok=True)
            reason = "Generated notebook missing"
            target = notebook_file

        target.parent.mkdir(parents=True, exist_ok=True)
        self.console.print(
            f"{reason}; converting {source_file.relative_to(self.project_root)} "
            f"→ {target.relative_to(self.project_root)}"
        )
        result = self._run_external([
            "jupytext",
            "--to",
            "ipynb",
            str(source_file),
            "--output",
            str(target),
        ], capture_output=True)
        if result.returncode != 0:
            self.console.print("[red]Failed to generate notebook with jupytext[/red]")
            if result.stderr:
                self.console.print(f"[red]{result.stderr.strip()}[/red]")
            return None
        return target

    def _prepare_notebook_for_nbgrader(self, notebook: Dict, *, release_tier: str = "student") -> List[str]:
        """Validate and normalize nbgrader cell metadata in a notebook."""
        errors = []
        if release_tier not in RELEASE_TIERS:
            return [f"Unknown TinyTorch release tier: {release_tier}"]

        cells = notebook.get("cells")
        if not isinstance(cells, list):
            return ["Notebook is missing a cells list"]

        grade_ids = set()
        nbgrader_cells = 0
        graded_cells = 0

        for index, cell in enumerate(cells, start=1):
            metadata = cell.setdefault("metadata", {})
            nbgrader = metadata.get("nbgrader")
            source = self._cell_source(cell)
            source, tier_errors = self._apply_release_tier(source, release_tier)
            if tier_errors:
                errors.extend(f"Cell {index}: {error}" for error in tier_errors)
            self._set_cell_source(cell, source)

            has_begin_marker = SOLUTION_BEGIN_MARKER in source
            has_end_marker = SOLUTION_END_MARKER in source
            has_solution_markers = has_begin_marker or has_end_marker
            has_complete_solution_markers = has_begin_marker and has_end_marker
            has_test_function = TEST_FUNCTION_RE.search(source) is not None
            is_markdown = cell.get("cell_type") == "markdown"

            if nbgrader is None:
                if has_solution_markers:
                    errors.append(f"Cell {index} contains a solution marker but no nbgrader metadata")
                if has_test_function:
                    errors.append(f"Cell {index} contains a test function but no nbgrader metadata")
                continue

            if not isinstance(nbgrader, dict):
                errors.append(f"Cell {index} has non-object nbgrader metadata")
                continue

            nbgrader_cells += 1
            nbgrader["schema_version"] = 3

            if has_begin_marker != has_end_marker:
                errors.append(f"Cell {index} has mismatched BEGIN/END SOLUTION markers")

            grade_id = nbgrader.get("grade_id")
            if not grade_id:
                errors.append(f"Cell {index} is missing grade_id")
            elif grade_id in grade_ids:
                errors.append(f"Duplicate grade_id '{grade_id}'")
            else:
                grade_ids.add(grade_id)

            if nbgrader.get("grade") is True:
                graded_cells += 1
                if has_complete_solution_markers and not has_test_function:
                    nbgrader["locked"] = False
                    nbgrader["solution"] = True
                else:
                    nbgrader["locked"] = True
                    nbgrader["solution"] = False
                if has_solution_markers and has_test_function:
                    errors.append(f"Graded test cell '{grade_id}' must not contain solution markers")
                if "points" not in nbgrader:
                    errors.append(f"Graded cell '{grade_id}' is missing points")
                continue

            nbgrader["grade"] = False
            if has_complete_solution_markers:
                nbgrader["solution"] = not is_markdown
                nbgrader["locked"] = False
            elif nbgrader.get("solution") is True:
                nbgrader["solution"] = False
                nbgrader["locked"] = True
            else:
                nbgrader["solution"] = False
                nbgrader.setdefault("locked", True)

        if nbgrader_cells == 0:
            errors.append("Notebook has no nbgrader metadata cells")
        if graded_cells == 0:
            errors.append("Notebook has no graded nbgrader cells")

        return errors

    @staticmethod
    def _cell_source(cell: Dict) -> str:
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        if isinstance(source, str):
            return source
        return ""

    @staticmethod
    def _set_cell_source(cell: Dict, source: str) -> None:
        cell["source"] = source

    def _apply_release_tier(self, source: str, release_tier: str) -> Tuple[str, List[str]]:
        """Apply TinyTorch release-role policy before nbgrader stripping."""
        if not source or (SOLUTION_BEGIN_MARKER not in source and SOLUTION_END_MARKER not in source):
            return source, []

        lines = source.splitlines()
        out = []
        errors = []
        in_solution = False
        action = "strip"

        for line in lines:
            if SOLUTION_BEGIN_MARKER in line:
                if in_solution:
                    errors.append("nested BEGIN SOLUTION marker")
                role = self._solution_role(line)
                if role not in VALID_SOLUTION_ROLES:
                    errors.append(f"unknown solution role '{role}'")
                    role = "core"
                action = self._solution_role_action(role, release_tier)
                in_solution = True
                if action == "strip":
                    out.append(line)
                continue

            if SOLUTION_END_MARKER in line:
                if not in_solution:
                    errors.append("END SOLUTION marker without matching BEGIN SOLUTION")
                    out.append(line)
                    continue
                if action == "strip":
                    out.append(line)
                in_solution = False
                action = "strip"
                continue

            if not in_solution or action in {"strip", "keep"}:
                out.append(line)

        if in_solution:
            errors.append("BEGIN SOLUTION marker without matching END SOLUTION")

        result = "\n".join(out)
        if source.endswith("\n"):
            result += "\n"
        return result, errors

    @staticmethod
    def _solution_role(marker_line: str) -> str:
        match = SOLUTION_ROLE_RE.search(marker_line)
        return match.group(1) if match else "core"

    @staticmethod
    def _solution_role_action(role: str, release_tier: str) -> str:
        if release_tier == "instructor":
            return "keep"
        if role == "instructor":
            return "remove"
        if release_tier == "challenge":
            return "strip" if role == "challenge" else "keep"
        return "strip" if role == "core" else "keep"

    def _release(self, args: Namespace) -> int:
        if args.all:
            return self._batch_operation("release", "generate_assignment", self.source_dir)
        if args.assignment:
            return self._single_operation("release", "generate_assignment", args.assignment)
        self.console.print("[red]Must specify either --all or an assignment name[/red]")
        return 1

    def _collect(self, args: Namespace) -> int:
        if args.all:
            return self._batch_operation("collect", "collect", self.release_dir, student=args.student)
        if args.assignment:
            return self._single_operation("collect", "collect", args.assignment, student=args.student)
        self.console.print("[red]Must specify either --all or an assignment name[/red]")
        return 1

    def _autograde(self, args: Namespace) -> int:
        extra_args = ["--force"] if args.force else None
        if args.all:
            return self._batch_operation(
                "autograde",
                "autograde",
                self.submitted_dir,
                student=args.student,
                extra_args=extra_args,
            )
        if args.assignment:
            return self._single_operation(
                "autograde",
                "autograde",
                args.assignment,
                student=args.student,
                extra_args=extra_args,
            )
        self.console.print("[red]Must specify either --all or an assignment name[/red]")
        return 1

    def _feedback(self, args: Namespace) -> int:
        if args.all:
            return self._batch_operation("feedback", "generate_feedback", self.autograded_dir, student=args.student)
        if args.assignment:
            return self._single_operation("feedback", "generate_feedback", args.assignment, student=args.student)
        self.console.print("[red]Must specify either --all or an assignment name[/red]")
        return 1

    def _single_operation(
        self,
        action: str,
        nbgrader_cmd: str,
        assignment: str,
        *,
        student: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        """Perform a single nbgrader operation."""
        console = self.console
        resolved_assignment = self._resolve_assignment_name(assignment) or assignment
        console.print(f"{action.title()} assignment: {resolved_assignment}")

        cmd = ["nbgrader", nbgrader_cmd, resolved_assignment, "--course-dir", str(self.assignments_dir)]
        if student:
            cmd.extend(["--student", student])
        if extra_args:
            cmd.extend(extra_args)

        result = self._run_external(cmd)
        if result.returncode != 0:
            console.print(f"[red]Failed to {action} assignment {resolved_assignment}[/red]")
            return result.returncode or 1

        console.print(f"[green]Assignment {resolved_assignment} {action} completed.[/green]")
        return 0

    def _batch_operation(
        self,
        action: str,
        nbgrader_cmd: str,
        source_dir: Path,
        *,
        student: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        """Perform a batch nbgrader operation."""
        if not source_dir.exists():
            self.console.print(f"[red]No {action} source directory found: {source_dir.relative_to(self.project_root)}[/red]")
            return 1

        assignments = sorted(d.name for d in source_dir.iterdir() if d.is_dir())
        if not assignments:
            self.console.print(f"[red]No assignments found for {action}[/red]")
            return 1

        self.console.print(f"Batch {action}: {len(assignments)} assignment(s)")
        for assignment in assignments:
            result = self._single_operation(
                action,
                nbgrader_cmd,
                assignment,
                student=student,
                extra_args=extra_args,
            )
            if result != 0:
                return result
        return 0

    def _resolve_assignment_name(self, assignment_input: str) -> Optional[str]:
        """Resolve numeric assignment aliases to staged/released module names."""
        resolved = self._resolve_module_name(assignment_input)
        return resolved

    def _status(self) -> int:
        """Show status of local assignment directories."""
        console = self.console
        console.print("Assignment Status:")

        for label, directory in [
            ("Source", self.source_dir),
            ("Release", self.release_dir),
            ("Submitted", self.submitted_dir),
            ("Autograded", self.autograded_dir),
            ("Feedback", self.feedback_dir),
        ]:
            assignments = sorted(d.name for d in directory.iterdir() if d.is_dir()) if directory.exists() else []
            console.print(f"{label}: {len(assignments)}")
            for assignment in assignments:
                console.print(f"  - {assignment}")

        return 0

    def _analytics(self, args: Namespace) -> int:
        """Show simple local analytics for an assignment."""
        assignment = self._resolve_assignment_name(args.assignment) or args.assignment
        submitted = self.submitted_dir / assignment
        autograded = self.autograded_dir / assignment

        self.console.print(f"Analytics for assignment: {assignment}")
        if not submitted.exists():
            self.console.print(f"[red]No submissions found for {assignment}[/red]")
            return 1

        submissions = sorted(d for d in submitted.iterdir() if d.is_dir())
        graded = sorted(d for d in autograded.iterdir() if d.is_dir()) if autograded.exists() else []
        self.console.print(f"Total submissions: {len(submissions)}")
        self.console.print(f"Graded submissions: {len(graded)}")
        self.console.print(f"Pending submissions: {max(0, len(submissions) - len(graded))}")
        return 0

    def _report(self, args: Namespace) -> int:
        """Export grades report."""
        # nbgrader's `export` app does not expose the `--course-dir` alias that
        # generate_assignment/collect/autograde/feedback accept; it only honours
        # the underlying CourseDirectory.root trait.
        cmd = ["nbgrader", "export", f"--CourseDirectory.root={self.assignments_dir}"]
        if args.assignment:
            assignment = self._resolve_assignment_name(args.assignment) or args.assignment
            cmd.extend(["--assignment", assignment])
        if args.student:
            cmd.extend(["--student", args.student])

        result = self._run_external(cmd)
        if result.returncode != 0:
            self.console.print("[red]Failed to generate grades report[/red]")
            return result.returncode or 1

        self.console.print("[green]Grades report exported by nbgrader.[/green]")
        return 0

    def _run_external(self, cmd: List[str], *, capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run an external command from the TinyTorch project root."""
        try:
            return subprocess.run(
                cmd,
                cwd=self.project_root,
                check=False,
                capture_output=capture_output,
                text=True, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))

    def _show_available_modules(self) -> None:
        modules = self._get_module_directories()
        if not modules:
            return
        text = Text()
        text.append("Available modules:\n", style="bold yellow")
        for module in modules:
            text.append(f"  • {module}\n", style="white")
        self.console.print(Panel(text, title="Available Modules", border_style="yellow"))

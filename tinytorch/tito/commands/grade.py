"""
Grade command for TinyTorch - wraps NBGrader functionality.

This command provides a simplified interface to NBGrader, allowing instructors
to manage assignments and grading without needing to know NBGrader details.
"""

import subprocess
from pathlib import Path
from argparse import Namespace
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from .base import BaseCommand


class GradeCommand(BaseCommand):
    """Handle grading operations through NBGrader."""

    @property
    def name(self) -> str:
        return "grade"

    @property
    def description(self) -> str:
        return "Simplified grading interface (instructor tool)"

    def add_arguments(self, parser):
        """Add arguments for the grade command."""
        # Subcommands for different grading operations
        grade_subparsers = parser.add_subparsers(dest='grade_action', help='Grade operations')

        # Release assignment to students
        release_parser = grade_subparsers.add_parser(
            'release',
            help='Release assignment to students (removes solutions)'
        )
        release_parser.add_argument(
            'module',
            help='Module name (e.g., 01_setup or setup)'
        )
        release_parser.add_argument(
            '--course-id',
            default='tinytorch',
            help='Course identifier (default: tinytorch)'
        )

        # Generate assignment (with solutions for instructor)
        generate_parser = grade_subparsers.add_parser(
            'generate',
            help='Generate assignment with solutions (instructor version)'
        )
        generate_parser.add_argument(
            'module',
            help='Module name to generate'
        )

        # Collect student submissions
        collect_parser = grade_subparsers.add_parser(
            'collect',
            help='Collect student submissions'
        )
        collect_parser.add_argument(
            'module',
            help='Module to collect'
        )
        collect_parser.add_argument(
            '--student',
            help='Specific student ID (collects all if not specified)'
        )

        # Autograde submissions
        autograde_parser = grade_subparsers.add_parser(
            'autograde',
            help='Automatically grade collected submissions'
        )
        autograde_parser.add_argument(
            'module',
            help='Module to autograde'
        )
        autograde_parser.add_argument(
            '--student',
            help='Specific student ID (grades all if not specified)'
        )

        # Manual grading interface
        manual_parser = grade_subparsers.add_parser(
            'manual',
            help='Open manual grading interface'
        )
        manual_parser.add_argument(
            'module',
            help='Module to grade manually'
        )

        # Generate feedback
        feedback_parser = grade_subparsers.add_parser(
            'feedback',
            help='Generate feedback for students'
        )
        feedback_parser.add_argument(
            'module',
            help='Module to generate feedback for'
        )

        # Export grades
        export_parser = grade_subparsers.add_parser(
            'export',
            help='Export grades to CSV'
        )
        export_parser.add_argument(
            '--module',
            help='Specific module (exports all if not specified)'
        )
        export_parser.add_argument(
            '--output',
            default='grades.csv',
            help='Output file name'
        )

        # Setup NBGrader course
        setup_parser = grade_subparsers.add_parser(
            'setup',
            help='Set up NBGrader course structure'
        )

    def run(self, args: Namespace) -> int:
        """Execute the grade command."""
        if not hasattr(args, 'grade_action') or not args.grade_action:
            self._show_help()
            return 0

        action = args.grade_action

        # Route to appropriate handler
        if action == 'release':
            return self._release_assignment(args)
        elif action == 'generate':
            return self._generate_assignment(args)
        elif action == 'collect':
            return self._collect_submissions(args)
        elif action == 'autograde':
            return self._autograde_submissions(args)
        elif action == 'manual':
            return self._manual_grade(args)
        elif action == 'feedback':
            return self._generate_feedback(args)
        elif action == 'export':
            return self._export_grades(args)
        elif action == 'setup':
            return self._setup_course(args)
        else:
            self._show_help()
            return 0

    def _show_help(self):
        """Show help information for grade command."""
        help_panel = Panel(
            "[bold cyan]TinyTorch Grade Command[/bold cyan]\n\n"
            "Simplified interface to NBGrader for managing assignments and grading.\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  tito grade setup         - Set up NBGrader course structure\n"
            "  tito grade generate MODULE - Generate instructor version with solutions\n"
            "  tito grade release MODULE  - Release student version (no solutions)\n"
            "  tito grade collect MODULE  - Collect student submissions\n"
            "  tito grade autograde MODULE - Auto-grade submissions\n"
            "  tito grade manual MODULE   - Manual grading interface\n"
            "  tito grade feedback MODULE - Generate student feedback\n"
            "  tito grade export         - Export grades to CSV\n\n"
            "[bold]Typical Workflow:[/bold]\n"
            "  1. tito grade setup        # One-time setup\n"
            "  2. tito grade generate 01_setup  # Create instructor version\n"
            "  3. tito grade release 01_setup   # Create student version\n"
            "  4. [Students complete work]\n"
            "  5. tito grade collect 01_setup   # Collect submissions\n"
            "  6. tito grade autograde 01_setup # Auto-grade\n"
            "  7. tito grade manual 01_setup    # Manual review\n"
            "  8. tito grade feedback 01_setup  # Generate feedback\n"
            "  9. tito grade export             # Export grades\n\n"
            "[dim]Note: NBGrader must be installed and configured[/dim]",
            title="Grade Help",
            border_style="bright_cyan"
        )
        self.console.print(help_panel)

    def _normalize_module_name(self, module: str) -> str:
        """Normalize module name to full format."""
        # If already in full format, return as is
        if module.startswith(tuple(f"{i:02d}_" for i in range(100))):
            return module

        # Try to find the module by short name
        source_dir = Path("modules")
        if source_dir.exists():
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name.endswith(f"_{module}"):
                    return module_dir.name

        return module

    def _release_assignment(self, args: Namespace) -> int:
        """Release assignment to students (removes solutions)."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Releasing Assignment: {module}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Creating student version...", total=None)

            try:
                # Step 1: Generate assignment first
                result = subprocess.run(
                    ["nbgrader", "generate_assignment", module,
                     "--source", f"modules/{module}",
                     "--force"],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.console.print(f"[red]❌ Failed to generate assignment: {result.stderr}[/red]")
                    return 1

                progress.update(task, description="Releasing to students...")

                # Step 2: Release assignment
                result = subprocess.run(
                    ["nbgrader", "release_assignment", module,
                     "--course", args.course_id],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.console.print(f"[red]❌ Failed to release: {result.stderr}[/red]")
                    return 1

            except FileNotFoundError:
                self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
                return 1

        self.console.print(f"[green]✅ Assignment {module} released to students![/green]")
        self.console.print(f"[dim]Student version available in: release/{args.course_id}/{module}/[/dim]")
        return 0

    def _generate_assignment(self, args: Namespace) -> int:
        """Generate assignment with solutions for instructor."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Generating Instructor Assignment: {module}[/bold]")

        try:
            result = subprocess.run(
                ["nbgrader", "generate_assignment", module,
                 "--source", f"modules/{module}",
                 "--force"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.console.print(f"[red]❌ Failed to generate: {result.stderr}[/red]")
                return 1

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        self.console.print(f"[green]✅ Instructor version generated![/green]")
        self.console.print(f"[dim]Available in: source/{module}/[/dim]")
        return 0

    def _collect_submissions(self, args: Namespace) -> int:
        """Collect student submissions."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Collecting Submissions: {module}[/bold]")

        cmd = ["nbgrader", "collect", module]
        if args.student:
            cmd.extend(["--student", args.student])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.console.print(f"[red]❌ Collection failed: {result.stderr}[/red]")
                return 1

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        self.console.print(f"[green]✅ Submissions collected![/green]")
        return 0

    def _autograde_submissions(self, args: Namespace) -> int:
        """Auto-grade collected submissions."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Auto-Grading: {module}[/bold]")

        cmd = ["nbgrader", "autograde", module]
        if args.student:
            cmd.extend(["--student", args.student])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.console.print(f"[red]❌ Auto-grading failed: {result.stderr}[/red]")
                return 1

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        self.console.print(f"[green]✅ Auto-grading complete![/green]")
        self.console.print("[dim]Use 'tito grade manual' for manual review[/dim]")
        return 0

    def _manual_grade(self, args: Namespace) -> int:
        """Open manual grading interface."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Opening Manual Grading Interface[/bold]")
        self.console.print("[dim]This will open in your browser...[/dim]")

        try:
            # Launch formgrader interface
            subprocess.Popen(["nbgrader", "formgrader"])
            self.console.print("[green]✅ Grading interface launched![/green]")
            self.console.print("[dim]Access at: http://localhost:5000[/dim]")

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        return 0

    def _generate_feedback(self, args: Namespace) -> int:
        """Generate feedback for students."""
        module = self._normalize_module_name(args.module)

        self.console.print(f"\n[bold]Generating Feedback: {module}[/bold]")

        try:
            result = subprocess.run(
                ["nbgrader", "generate_feedback", module],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.console.print(f"[red]❌ Feedback generation failed: {result.stderr}[/red]")
                return 1

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        self.console.print(f"[green]✅ Feedback generated![/green]")
        return 0

    def _export_grades(self, args: Namespace) -> int:
        """Export grades to CSV."""
        self.console.print(f"\n[bold]Exporting Grades[/bold]")

        try:
            cmd = ["nbgrader", "export"]
            if args.module:
                module = self._normalize_module_name(args.module)
                cmd.extend(["--assignment", module])
            cmd.extend(["--to", args.output])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.console.print(f"[red]❌ Export failed: {result.stderr}[/red]")
                return 1

        except FileNotFoundError:
            self.console.print("[red]❌ NBGrader not found. Install with: pip install nbgrader[/red]")
            return 1

        self.console.print(f"[green]✅ Grades exported to {args.output}![/green]")
        return 0

    def _setup_course(self, args: Namespace) -> int:
        """Set up NBGrader course structure."""
        self.console.print("\n[bold]Setting Up NBGrader Course[/bold]")

        # Create necessary directories
        dirs_to_create = [
            "source",
            "release",
            "submitted",
            "autograded",
            "feedback"
        ]

        for dir_name in dirs_to_create:
            Path(dir_name).mkdir(exist_ok=True)
            self.console.print(f"  ✅ Created {dir_name}/")

        # Create nbgrader_config.py if it doesn't exist
        config_file = Path("nbgrader_config.py")
        if not config_file.exists():
            config_content = '''"""NBGrader configuration for TinyTorch."""

c = get_config()

c.CourseDirectory.course_id = "tinytorch"
c.CourseDirectory.source_directory = "modules"
c.CourseDirectory.release_directory = "release"
c.CourseDirectory.submitted_directory = "submitted"
c.CourseDirectory.autograded_directory = "autograded"
c.CourseDirectory.feedback_directory = "feedback"

# Exchange settings
c.Exchange.root = "/tmp/exchange"
c.Exchange.course_id = "tinytorch"

# Grading settings
c.ExecuteOptions.timeout = 300  # 5 minutes per cell
c.ExecuteOptions.allow_errors = True
c.ExecuteOptions.interrupt_on_timeout = True
'''
            config_file.write_text(config_content)
            self.console.print("  ✅ Created nbgrader_config.py")

        self.console.print("[green]✅ NBGrader course setup complete![/green]")
        self.console.print("\n[bold]Next Steps:[/bold]")
        self.console.print("  1. tito grade generate 01_setup  # Create instructor version")
        self.console.print("  2. tito grade release 01_setup   # Create student version")

        return 0

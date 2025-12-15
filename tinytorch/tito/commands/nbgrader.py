"""
NBGrader integration commands for TinyTorch.

This module provides commands for managing nbgrader assignments,
auto-grading, and feedback generation with proper hierarchical module support.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class NBGraderCommand(BaseCommand):
    """NBGrader integration command group."""

    def __init__(self, config):
        super().__init__(config)
        self.assignments_dir = Path("assignments")
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
        return "Assignment management and auto-grading commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='nbgrader_command',
            help='NBGrader subcommands',
            metavar='SUBCOMMAND'
        )

        # Init subcommand
        init_parser = subparsers.add_parser(
            'init',
            help='Initialize nbgrader environment'
        )

        # Generate subcommand
        generate_parser = subparsers.add_parser(
            'generate',
            help='Generate assignments from TinyTorch modules'
        )
        generate_parser.add_argument(
            'module',
            nargs='?',
            help='Module to generate assignment for (e.g., setup, tensor, 01_setup)'
        )
        generate_parser.add_argument(
            '--all',
            action='store_true',
            help='Generate assignments for all modules'
        )
        generate_parser.add_argument(
            '--range',
            help='Generate assignments for module range (e.g., 01-04, setup-tensor)'
        )

        # Release subcommand
        release_parser = subparsers.add_parser(
            'release',
            help='Release assignments to students'
        )
        release_parser.add_argument(
            'assignment',
            nargs='?',
            help='Assignment to release'
        )
        release_parser.add_argument(
            '--all',
            action='store_true',
            help='Release all assignments'
        )

        # Collect subcommand
        collect_parser = subparsers.add_parser(
            'collect',
            help='Collect student submissions'
        )
        collect_parser.add_argument(
            'assignment',
            nargs='?',
            help='Assignment to collect'
        )
        collect_parser.add_argument(
            '--all',
            action='store_true',
            help='Collect all submissions'
        )

        # Autograde subcommand
        autograde_parser = subparsers.add_parser(
            'autograde',
            help='Auto-grade submissions'
        )
        autograde_parser.add_argument(
            'assignment',
            nargs='?',
            help='Assignment to auto-grade'
        )
        autograde_parser.add_argument(
            '--all',
            action='store_true',
            help='Auto-grade all submissions'
        )

        # Feedback subcommand
        feedback_parser = subparsers.add_parser(
            'feedback',
            help='Generate feedback for students'
        )
        feedback_parser.add_argument(
            'assignment',
            nargs='?',
            help='Assignment to generate feedback for'
        )
        feedback_parser.add_argument(
            '--all',
            action='store_true',
            help='Generate feedback for all assignments'
        )

        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='Show assignment status'
        )

        # Analytics subcommand
        analytics_parser = subparsers.add_parser(
            'analytics',
            help='Show assignment analytics'
        )
        analytics_parser.add_argument(
            'assignment',
            help='Assignment to analyze'
        )

        # Report subcommand
        report_parser = subparsers.add_parser(
            'report',
            help='Export grades report'
        )
        report_parser.add_argument(
            '--format',
            choices=['csv', 'json'],
            default='csv',
            help='Export format (default: csv)'
        )

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'nbgrader_command') or not args.nbgrader_command:
            console.print(Panel(
                "[bold cyan]NBGrader Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]init[/bold]       - Initialize nbgrader environment\n"
                "  • [bold]generate[/bold]   - Generate assignments from modules\n"
                "  • [bold]release[/bold]    - Release assignments to students\n"
                "  • [bold]collect[/bold]    - Collect student submissions\n"
                "  • [bold]autograde[/bold]  - Auto-grade submissions\n"
                "  • [bold]feedback[/bold]   - Generate feedback for students\n"
                "  • [bold]status[/bold]     - Show assignment status\n"
                "  • [bold]analytics[/bold]  - Show assignment analytics\n"
                "  • [bold]report[/bold]     - Export grades report\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito nbgrader init[/dim]\n"
                "[dim]  tito nbgrader generate setup[/dim]\n"
                "[dim]  tito nbgrader generate --all[/dim]\n"
                "[dim]  tito nbgrader generate --range 01-04[/dim]\n"
                "[dim]  tito nbgrader release setup[/dim]\n"
                "[dim]  tito nbgrader autograde --all[/dim]",
                title="NBGrader Command Group",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.nbgrader_command == 'init':
            return self._init()
        elif args.nbgrader_command == 'generate':
            return self._generate(args)
        elif args.nbgrader_command == 'release':
            return self._release(args)
        elif args.nbgrader_command == 'collect':
            return self._collect(args)
        elif args.nbgrader_command == 'autograde':
            return self._autograde(args)
        elif args.nbgrader_command == 'feedback':
            return self._feedback(args)
        elif args.nbgrader_command == 'status':
            return self._status()
        elif args.nbgrader_command == 'analytics':
            return self._analytics(args)
        elif args.nbgrader_command == 'report':
            return self._report(args)
        else:
            console.print(Panel(
                f"[red]Unknown nbgrader subcommand: {args.nbgrader_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1

    def _get_module_directories(self) -> List[Path]:
        """Get all module directories with proper hierarchy support."""
        source_dir = Path("modules")
        if not source_dir.exists():
            return []

        # Get all numbered module directories
        module_dirs = []
        for item in source_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                module_dirs.append(item)

        # Sort by number prefix if present
        def sort_key(path: Path) -> tuple:
            name = path.name
            if name[:2].isdigit():
                return (int(name[:2]), name)
            return (999, name)  # Non-numbered modules go last

        return sorted(module_dirs, key=sort_key)

    def _resolve_module_name(self, module_input: str) -> Optional[str]:
        """Resolve module name from various input formats."""
        # If it's already a directory name, use it
        if Path(f"modules/{module_input}").exists():
            return module_input

        # Try to find by number prefix
        if module_input.isdigit():
            prefix = module_input.zfill(2)
            source_dir = Path("modules")
            for item in source_dir.iterdir():
                if item.is_dir() and item.name.startswith(prefix):
                    return item.name

        # Try to find by name suffix
        source_dir = Path("modules")
        for item in source_dir.iterdir():
            if item.is_dir() and item.name.endswith(f"_{module_input}"):
                return item.name

        return None

    def _parse_module_range(self, range_str: str) -> List[str]:
        """Parse module range specification."""
        if "-" not in range_str:
            return [range_str]

        start, end = range_str.split("-", 1)
        start_num = int(start) if start.isdigit() else 0
        end_num = int(end) if end.isdigit() else 99

        modules = []
        for module_dir in self._get_module_directories():
            name = module_dir.name
            if name[:2].isdigit():
                num = int(name[:2])
                if start_num <= num <= end_num:
                    modules.append(name)

        return modules

    def _init(self) -> int:
        """Initialize nbgrader environment."""
        console = self.console
        console.print("🔧 Initializing NBGrader environment...")

        # Check if nbgrader is installed
        try:
            result = subprocess.run(
                ["nbgrader", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"✅ NBGrader version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("❌ NBGrader not found. Please install with: pip install nbgrader")
            return 1

        # Create directory structure
        directories = [
            self.assignments_dir,
            self.source_dir,
            self.release_dir,
            self.submitted_dir,
            self.autograded_dir,
            self.feedback_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"📁 Created directory: {directory}")

        # Check if nbgrader_config.py exists
        config_file = Path("nbgrader_config.py")
        if config_file.exists():
            console.print(f"✅ Found nbgrader config: {config_file}")
        else:
            console.print("⚠️  NBGrader config not found. Please create nbgrader_config.py")
            return 1

        # Initialize nbgrader database
        try:
            subprocess.run(
                ["nbgrader", "db", "upgrade"],
                check=True,
                capture_output=True
            )
            console.print("✅ NBGrader database initialized")
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to initialize database: {e}")
            return 1

        console.print("🎉 NBGrader environment initialized successfully!")
        return 0

    def _generate(self, args: Namespace) -> int:
        """Generate assignments from TinyTorch modules."""
        console = self.console

        modules_to_process = []

        if args.all:
            # Generate all modules
            module_dirs = self._get_module_directories()
            modules_to_process = [d.name for d in module_dirs]
        elif hasattr(args, 'range') and args.range:
            # Generate range of modules
            modules_to_process = self._parse_module_range(args.range)
        elif args.module:
            # Generate specific module
            resolved_module = self._resolve_module_name(args.module)
            if resolved_module:
                modules_to_process = [resolved_module]
            else:
                console.print(f"❌ Module '{args.module}' not found")
                return 1
        else:
            console.print("❌ Must specify either --all, --range, or a module name")
            return 1

        console.print(f"🔄 Generating assignments for modules: {modules_to_process}")

        for module_name in modules_to_process:
            success = self._generate_single_module(module_name)
            if not success:
                console.print(f"❌ Failed to generate assignment for {module_name}")
                return 1

        console.print("✅ All assignments generated successfully!")
        return 0

    def _generate_single_module(self, module_name: str) -> bool:
        """Generate assignment from a single module."""
        console = self.console
        console.print(f"📝 Generating assignment for module: {module_name}")

        # Find the module development file in TinyTorch modules directory
        module_dir = Path("modules") / module_name

        # Extract the short name from the module directory name
        # e.g., "00_setup" -> "setup", "01_tensor" -> "tensor"
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name

        # Look for module Python file
        dev_file = module_dir / f"{short_name}.py"

        if not dev_file.exists():
            console.print(f"❌ Module file not found in: {module_dir}")
            console.print(f"   Looking for: {short_name}.py")
            return False

        # Convert to notebook and generate assignment
        try:
            # First convert .py to .ipynb using jupytext
            console.print(f"🔄 Converting {dev_file} to notebook...")
            # Use the same filename as the source file, just change extension
            notebook_file = dev_file.with_suffix('.ipynb')

            result = subprocess.run([
                "jupytext", "--to", "ipynb", str(dev_file)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"❌ Failed to convert notebook: {result.stderr}")
                return False

            # Generate nbgrader assignment using the enhanced generator
            sys.path.insert(0, str(Path.cwd()))
            from bin.generate_student_notebooks import NotebookGenerator

            generator = NotebookGenerator(use_nbgrader=True)
            notebook = generator.process_notebook(notebook_file)

            # Save to assignments/source
            assignment_dir = self.source_dir / module_name
            assignment_dir.mkdir(parents=True, exist_ok=True)

            # Use short name for notebook (e.g., "tensor.ipynb" not "01_tensor.ipynb")
            short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name
            assignment_file = assignment_dir / f"{short_name}.ipynb"
            generator.save_student_notebook(notebook, assignment_file)

            console.print(f"✅ Assignment created: {assignment_file}")
            return True

        except Exception as e:
            console.print(f"❌ Error generating assignment: {e}")
            return False

    def _release(self, args: Namespace) -> int:
        """Release assignments to students."""
        console = self.console

        if args.all:
            return self._batch_operation("release", "generate_assignment")
        elif args.assignment:
            return self._single_operation("release", "generate_assignment", args.assignment)
        else:
            console.print("❌ Must specify either --all or an assignment name")
            return 1

    def _collect(self, args: Namespace) -> int:
        """Collect student submissions."""
        console = self.console

        if args.all:
            return self._batch_operation("collect", "collect")
        elif args.assignment:
            return self._single_operation("collect", "collect", args.assignment)
        else:
            console.print("❌ Must specify either --all or an assignment name")
            return 1

    def _autograde(self, args: Namespace) -> int:
        """Auto-grade submissions."""
        console = self.console

        if args.all:
            return self._batch_operation("autograde", "autograde")
        elif args.assignment:
            return self._single_operation("autograde", "autograde", args.assignment)
        else:
            console.print("❌ Must specify either --all or an assignment name")
            return 1

    def _feedback(self, args: Namespace) -> int:
        """Generate feedback for students."""
        console = self.console

        if args.all:
            return self._batch_operation("feedback", "generate_feedback")
        elif args.assignment:
            return self._single_operation("feedback", "generate_feedback", args.assignment)
        else:
            console.print("❌ Must specify either --all or an assignment name")
            return 1

    def _single_operation(self, action: str, nbgrader_cmd: str, assignment: str) -> int:
        """Perform a single nbgrader operation."""
        console = self.console

        action_icons = {
            "release": "🚀",
            "collect": "📥",
            "autograde": "🎯",
            "feedback": "📋"
        }

        console.print(f"{action_icons.get(action, '🔄')} {action.title()}ing assignment: {assignment}")

        try:
            subprocess.run([
                "nbgrader", nbgrader_cmd, assignment
            ], check=True)
            console.print(f"✅ Assignment {assignment} {action}d successfully")
            return 0
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to {action} assignment {assignment}: {e}")
            return 1

    def _batch_operation(self, action: str, nbgrader_cmd: str) -> int:
        """Perform a batch nbgrader operation."""
        console = self.console

        # Determine which directory to look in based on action
        source_dirs = {
            "release": self.source_dir,
            "collect": self.release_dir,
            "autograde": self.submitted_dir,
            "feedback": self.autograded_dir
        }

        source_dir = source_dirs.get(action)
        if not source_dir or not source_dir.exists():
            console.print(f"❌ No {action} source directory found")
            return 1

        assignments = [d.name for d in source_dir.iterdir() if d.is_dir()]

        if not assignments:
            console.print(f"❌ No assignments found for {action}")
            return 1

        console.print(f"🔄 Batch {action}ing {len(assignments)} assignments...")

        for assignment in assignments:
            result = self._single_operation(action, nbgrader_cmd, assignment)
            if result != 0:
                return result

        console.print(f"✅ All assignments {action}d successfully!")
        return 0

    def _status(self) -> int:
        """Show status of all assignments."""
        console = self.console
        console.print("📊 Assignment Status:")

        # List source assignments
        if self.source_dir.exists():
            source_assignments = [d.name for d in self.source_dir.iterdir() if d.is_dir()]
            console.print(f"📚 Source assignments: {len(source_assignments)}")
            for assignment in source_assignments:
                console.print(f"  - {assignment}")

        # List released assignments
        if self.release_dir.exists():
            released_assignments = [d.name for d in self.release_dir.iterdir() if d.is_dir()]
            console.print(f"🚀 Released assignments: {len(released_assignments)}")
            for assignment in released_assignments:
                console.print(f"  - {assignment}")

        # List submitted assignments
        if self.submitted_dir.exists():
            submitted_assignments = [d.name for d in self.submitted_dir.iterdir() if d.is_dir()]
            console.print(f"📥 Submitted assignments: {len(submitted_assignments)}")
            for assignment in submitted_assignments:
                console.print(f"  - {assignment}")

        # List graded assignments
        if self.autograded_dir.exists():
            graded_assignments = [d.name for d in self.autograded_dir.iterdir() if d.is_dir()]
            console.print(f"🎯 Graded assignments: {len(graded_assignments)}")
            for assignment in graded_assignments:
                console.print(f"  - {assignment}")

        return 0

    def _analytics(self, args: Namespace) -> int:
        """Show analytics for an assignment."""
        console = self.console
        assignment = args.assignment

        console.print(f"📈 Analytics for assignment: {assignment}")

        # Check submissions
        assignment_dir = self.submitted_dir / assignment
        if not assignment_dir.exists():
            console.print(f"❌ No submissions found for {assignment}")
            return 1

        submissions = [d for d in assignment_dir.iterdir() if d.is_dir()]
        console.print(f"📊 Total submissions: {len(submissions)}")

        # Show grading status
        graded_dir = self.autograded_dir / assignment
        if graded_dir.exists():
            graded_submissions = [d for d in graded_dir.iterdir() if d.is_dir()]
            console.print(f"✅ Graded submissions: {len(graded_submissions)}")
            console.print(f"⏳ Pending submissions: {len(submissions) - len(graded_submissions)}")

        return 0

    def _report(self, args: Namespace) -> int:
        """Export grades report."""
        console = self.console
        format_type = args.format

        console.print(f"📊 Generating grades report in {format_type} format...")

        try:
            if format_type == "csv":
                subprocess.run([
                    "nbgrader", "export"
                ], check=True)
                console.print("✅ Grades report exported to grades.csv")
            else:
                console.print(f"❌ Unsupported format: {format_type}")
                return 1

            return 0
        except subprocess.CalledProcessError:
            console.print("❌ Failed to generate grades report")
            return 1

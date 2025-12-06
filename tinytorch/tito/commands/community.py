"""
Tiny🔥Torch Community Commands

Login, logout, and connect with the TinyTorch community.
"""

import json
import webbrowser
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand
from .login import LoginCommand, LogoutCommand


class CommunityCommand(BaseCommand):
    """Community commands - login, logout, leaderboard, and benchmarks."""

    @property
    def name(self) -> str:
        return "community"

    @property
    def description(self) -> str:
        return "Join the global community - connect with builders worldwide"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add community subcommands."""
        subparsers = parser.add_subparsers(
            dest='community_command',
            help='Community operations',
            metavar='COMMAND'
        )

        # Login command (delegates to LoginCommand)
        login_parser = subparsers.add_parser(
            'login',
            help='Log in to TinyTorch via web browser'
        )
        LoginCommand(self.config).add_arguments(login_parser)

        # Logout command (delegates to LogoutCommand)
        logout_parser = subparsers.add_parser(
            'logout',
            help='Log out of TinyTorch'
        )
        LogoutCommand(self.config).add_arguments(logout_parser)

        # Leaderboard command (opens browser)
        leaderboard_parser = subparsers.add_parser(
            'leaderboard',
            help='View global leaderboard (opens in browser)'
        )

        # Compete command (opens browser)
        compete_parser = subparsers.add_parser(
            'compete',
            help='Join competitions and challenges (opens in browser)'
        )

        # Submit command
        submit_parser = subparsers.add_parser(
            'submit',
            help='Submit your benchmark results to the leaderboard'
        )
        submit_parser.add_argument(
            'submission_file',
            help='Path to submission JSON file (e.g., submission.json)'
        )

    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: login, logout, leaderboard, compete, submit[/yellow]")
            return 1

        if args.community_command == 'login':
            return LoginCommand(self.config).run(args)
        elif args.community_command == 'logout':
            return LogoutCommand(self.config).run(args)
        elif args.community_command == 'leaderboard':
            return self._open_leaderboard(args)
        elif args.community_command == 'compete':
            return self._open_compete(args)
        elif args.community_command == 'submit':
            return self._submit_benchmark(args)
        else:
            self.console.print(f"[red]❌ Unknown community command: {args.community_command}[/red]")
            return 1

    def _open_leaderboard(self, args: Namespace) -> int:
        """Open community leaderboard in browser."""
        import webbrowser

        leaderboard_url = "https://tinytorch.ai/leaderboard"

        self.console.print(f"[cyan]🏆 Opening leaderboard...[/cyan]")
        try:
            webbrowser.open(leaderboard_url)
            self.console.print(f"[green]✅[/green] Browser opened: [cyan]{leaderboard_url}[/cyan]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not open browser automatically[/yellow]")
            self.console.print(f"[dim]Please visit: {leaderboard_url}[/dim]")

        return 0

    def _open_compete(self, args: Namespace) -> int:
        """Open competitions page in browser."""
        import webbrowser

        compete_url = "https://tinytorch.ai/compete"

        self.console.print(f"[cyan]🎯 Opening competitions...[/cyan]")
        try:
            webbrowser.open(compete_url)
            self.console.print(f"[green]✅[/green] Browser opened: [cyan]{compete_url}[/cyan]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not open browser automatically[/yellow]")
            self.console.print(f"[dim]Please visit: {compete_url}[/dim]")

        return 0

    def _submit_benchmark(self, args: Namespace) -> int:
        """Submit benchmark results to the leaderboard."""
        console = self.console
        submission_file = Path(args.submission_file)

        # Check if file exists
        if not submission_file.exists():
            console.print(Panel(
                f"[red]❌ Submission file not found[/red]\n\n"
                f"[dim]Path: {submission_file}[/dim]\n\n"
                f"Make sure you've generated a submission file first:\n"
                f"  • Run Module 20 (Capstone)\n"
                f"  • Generate submission: [cyan]save_submission(submission, 'submission.json')[/cyan]",
                title="File Not Found",
                border_style="red"
            ))
            return 1

        # Load and validate submission
        try:
            with open(submission_file, 'r') as f:
                submission = json.load(f)
        except json.JSONDecodeError as e:
            console.print(Panel(
                f"[red]❌ Invalid JSON file[/red]\n\n"
                f"[dim]Error: {e}[/dim]\n\n"
                f"Make sure your file contains valid JSON.",
                title="Invalid JSON",
                border_style="red"
            ))
            return 1

        # Validate submission schema
        console.print(Panel(
            "[cyan]🔍 Validating submission...[/cyan]",
            title="Validation",
            border_style="cyan"
        ))

        try:
            # Import validation function from Module 20
            import sys
            sys.path.insert(0, str(self.config.project_root / "src" / "20_capstone"))
            from importlib import import_module
            capstone = import_module("20_capstone")

            # Validate
            capstone.validate_submission_schema(submission)

            # Show validation success
            console.print()
            console.print("[green]✅ Submission validated successfully![/green]")
            console.print()

            # Display submission summary
            table = Table(title="Submission Summary", show_header=True, box=None)
            table.add_column("Field", style="cyan", width=25)
            table.add_column("Value", style="green")

            table.add_row("TinyTorch Version", submission.get('tinytorch_version', 'N/A'))
            table.add_row("Submission Type", submission.get('submission_type', 'N/A'))
            table.add_row("Timestamp", submission.get('timestamp', 'N/A'))

            baseline = submission.get('baseline', {})
            metrics = baseline.get('metrics', {})
            table.add_row("", "")
            table.add_row("[bold]Baseline Model[/bold]", "")
            table.add_row("  Model Name", baseline.get('model_name', 'N/A'))
            table.add_row("  Parameters", f"{metrics.get('parameter_count', 0):,}")
            table.add_row("  Size", f"{metrics.get('model_size_mb', 0):.2f} MB")
            table.add_row("  Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            table.add_row("  Latency", f"{metrics.get('latency_ms_mean', 0):.2f} ms")

            if 'optimized' in submission:
                optimized = submission['optimized']
                opt_metrics = optimized.get('metrics', {})
                improvements = submission.get('improvements', {})

                table.add_row("", "")
                table.add_row("[bold]Optimized Model[/bold]", "")
                table.add_row("  Model Name", optimized.get('model_name', 'N/A'))
                table.add_row("  Parameters", f"{opt_metrics.get('parameter_count', 0):,}")
                table.add_row("  Size", f"{opt_metrics.get('model_size_mb', 0):.2f} MB")
                table.add_row("  Accuracy", f"{opt_metrics.get('accuracy', 0)*100:.1f}%")
                table.add_row("  Latency", f"{opt_metrics.get('latency_ms_mean', 0):.2f} ms")

                table.add_row("", "")
                table.add_row("[bold]Improvements[/bold]", "")
                table.add_row("  Speedup", f"{improvements.get('speedup', 0):.2f}x")
                table.add_row("  Compression", f"{improvements.get('compression_ratio', 0):.2f}x")
                table.add_row("  Accuracy Delta", f"{improvements.get('accuracy_delta', 0)*100:+.1f}%")

            console.print()
            console.print(table)
            console.print()

        except AssertionError as e:
            console.print(Panel(
                f"[red]❌ Validation failed[/red]\n\n"
                f"[dim]Error: {e}[/dim]\n\n"
                f"Check your submission matches the required schema:\n"
                f"  • Required fields: tinytorch_version, submission_type, timestamp, system_info, baseline\n"
                f"  • Baseline metrics: parameter_count, model_size_mb, accuracy, latency_ms_mean\n"
                f"  • Accuracy must be in [0, 1]\n"
                f"  • All counts/sizes/latencies must be positive",
                title="Validation Failed",
                border_style="red"
            ))
            return 1
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Validation error[/red]\n\n"
                f"[dim]Error: {e}[/dim]",
                title="Error",
                border_style="red"
            ))
            return 1

        # Show "coming soon" message for actual submission
        console.print(Panel(
            "[bold yellow]🚧 Submission to Leaderboard - Coming Soon![/bold yellow]\n\n"
            "[dim]Your submission has been validated successfully![/dim]\n\n"
            "The TinyTorch community leaderboard is currently under development.\n"
            "Soon you'll be able to:\n"
            "  • Submit your optimized models to the global leaderboard\n"
            "  • Compare your results with other learners worldwide\n"
            "  • Participate in TinyML optimization challenges\n"
            "  • Earn badges and achievements\n\n"
            f"[green]✓[/green] Your submission file is ready: [cyan]{submission_file}[/cyan]\n"
            "[dim]Keep this file - you'll be able to submit it once the leaderboard launches![/dim]\n\n"
            "In the meantime:\n"
            "  • View community: [cyan]tito community leaderboard[/cyan]\n"
            "  • Join challenges: [cyan]tito community compete[/cyan]",
            title="🎯 Leaderboard Coming Soon",
            border_style="yellow"
        ))

        return 0

"""
Maintenance commands for MLSysBook CLI.

Handles setup, switch, hello, about, and other maintenance operations.
"""

import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class MaintenanceCommand:
    """Handles maintenance operations for the MLSysBook."""

    def __init__(self, config_manager, chapter_discovery):
        """Initialize maintenance command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def switch_format(self, format_type: str) -> bool:
        """Switch active configuration format.

        Args:
            format_type: Format to switch to ('html', 'pdf', 'epub')

        Returns:
            True if switch succeeded, False otherwise
        """
        if format_type not in ["html", "pdf", "epub"]:
            console.print("[red]❌ Format must be 'html', 'pdf', or 'epub'[/red]")
            console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
            return False

        console.print(f"[blue]🔄 Switching to {format_type.upper()} configuration...[/blue]")

        try:
            # Set up the symlink
            config_name = self.config_manager.setup_symlink(format_type)
            console.print(f"[green]✅ Switched to {format_type.upper()} configuration[/green]")
            console.print(f"[dim]🔗 Active config: {config_name}[/dim]")

            # Show current status
            self.config_manager.show_symlink_status()

            return True

        except Exception as e:
            console.print(f"[red]❌ Error switching format: {e}[/red]")
            return False

    def show_hello(self) -> bool:
        """Show welcome message and quick start guide."""
        # Banner
        banner = Panel(
            "[bold blue]📚 Welcome to MLSysBook CLI v2.0![/bold blue]\n"
            "[dim]⚡ Modular, maintainable, and fast[/dim]\n\n"
            "[green]🎯 Ready to build amazing ML systems content![/green]",
            title="👋 Hello!",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(banner)

        # Quick start table
        quick_table = Table(show_header=True, header_style="bold green", box=None)
        quick_table.add_column("Action", style="green", width=25)
        quick_table.add_column("Command", style="cyan", width=30)
        quick_table.add_column("Description", style="dim", width=35)

        quick_table.add_row("🚀 Get started", "./binder help", "Show all available commands")
        quick_table.add_row("📋 List chapters", "./binder list", "See all available chapters")
        quick_table.add_row("🏗️ Build a chapter", "./binder build intro", "Build introduction chapter")
        quick_table.add_row("🌐 Preview live", "./binder preview intro", "Start live development server")
        quick_table.add_row("🏥 Health check", "./binder doctor", "Run comprehensive diagnostics")

        console.print(Panel(quick_table, title="🚀 Quick Start", border_style="green"))

        # Tips
        tips = Panel(
            "[bold magenta]💡 Pro Tips:[/bold magenta]\n"
            "• Use [cyan]./binder build intro,ml_systems[/cyan] to build multiple chapters\n"
            "• Use [cyan]./binder preview[/cyan] for live development with hot reload\n"
            "• Use [cyan]./binder doctor[/cyan] to check system health\n"
            "• Use [cyan]./binder clean[/cyan] to clean up build artifacts",
            title="💡 Tips",
            border_style="magenta"
        )
        console.print(tips)

        return True

    def show_about(self) -> bool:
        """Show information about the MLSysBook project."""
        # Project info
        about_panel = Panel(
            "[bold blue]📚 Machine Learning Systems Textbook[/bold blue]\n\n"
            "[white]A comprehensive textbook on engineering machine learning systems,[/white]\n"
            "[white]covering principles and practices for building AI solutions in real-world environments.[/white]\n\n"
            "[green]🎯 Author:[/green] Prof. Vijay Janapa Reddi (Harvard University)\n"
            "[green]🌐 Website:[/green] https://mlsysbook.ai\n"
            "[green]📖 Repository:[/green] https://github.com/harvard-edge/cs249r_book\n"
            "[green]⚡ CLI Version:[/green] v2.0 (Modular Architecture)",
            title="ℹ️ About MLSysBook",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(about_panel)

        # Statistics
        chapters = self.chapter_discovery.get_all_chapters()
        stats_table = Table(show_header=True, header_style="bold blue", box=None)
        stats_table.add_column("Metric", style="blue", width=20)
        stats_table.add_column("Value", style="green", width=15)
        stats_table.add_column("Description", style="dim", width=35)

        stats_table.add_row("📄 Chapters", str(len(chapters)), "Total number of chapters")
        stats_table.add_row("🏗️ Formats", "3", "HTML, PDF, EPUB supported")
        stats_table.add_row("🔧 Commands", "10+", "Build, preview, maintenance")
        stats_table.add_row("🏥 Health Checks", "18", "Comprehensive diagnostics")

        console.print(Panel(stats_table, title="📊 Project Statistics", border_style="cyan"))

        # Architecture info
        arch_panel = Panel(
            "[bold magenta]🏗️ Modular CLI Architecture:[/bold magenta]\n\n"
            "[cyan]• ConfigManager:[/cyan] Handles Quarto configurations and format switching\n"
            "[cyan]• ChapterDiscovery:[/cyan] Finds and validates chapter files\n"
            "[cyan]• BuildCommand:[/cyan] Manages build operations for all formats\n"
            "[cyan]• PreviewCommand:[/cyan] Handles live development servers\n"
            "[cyan]• DoctorCommand:[/cyan] Performs comprehensive health checks\n"
            "[cyan]• CleanCommand:[/cyan] Cleans artifacts and restores configs\n"
            "[cyan]• MaintenanceCommand:[/cyan] Handles setup and maintenance tasks",
            title="🔧 Architecture",
            border_style="magenta"
        )
        console.print(arch_panel)

        return True

    def setup_environment(self) -> bool:
        """Setup development environment (simplified version)."""
        console.print("[bold blue]🔧 MLSysBook Environment Setup[/bold blue]")
        console.print("[dim]Setting up your development environment...[/dim]\n")

        # Run doctor command for comprehensive check
        console.print("[blue]🏥 Running health check first...[/blue]")

        # Import and run doctor (avoiding circular imports)
        from .doctor import DoctorCommand
        doctor = DoctorCommand(self.config_manager, self.chapter_discovery)
        health_ok = doctor.run_health_check()

        if health_ok:
            console.print("\n[green]✅ Environment setup complete![/green]")
            console.print("[dim]💡 Your system is healthy and ready for development[/dim]")
        else:
            console.print("\n[yellow]⚠️ Environment setup completed with issues[/yellow]")
            console.print("[dim]💡 Please review the health check results above[/dim]")

        # Show next steps
        next_steps = Panel(
            "[bold green]🚀 Next Steps:[/bold green]\n\n"
            "1. [cyan]./binder list[/cyan] - See all available chapters\n"
            "2. [cyan]./binder build intro[/cyan] - Build your first chapter\n"
            "3. [cyan]./binder preview intro[/cyan] - Start live development\n"
            "4. [cyan]./binder help[/cyan] - Explore all commands",
            title="🎯 Getting Started",
            border_style="green"
        )
        console.print(next_steps)

        return health_ok

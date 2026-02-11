#!/usr/bin/env python3
"""
MLSysBook CLI - Modular Entry Point

A refactored, modular command-line interface for building, previewing,
and managing the Machine Learning Systems textbook.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import our modular components
try:
    # When run as installed package
    from cli.core.config import ConfigManager
    from cli.core.discovery import ChapterDiscovery, AmbiguousChapterError
    from cli.commands.build import BuildCommand
    from cli.commands.preview import PreviewCommand
    from cli.commands.doctor import DoctorCommand
    from cli.commands.clean import CleanCommand
    from cli.commands.maintenance import MaintenanceCommand
    from cli.commands.debug import DebugCommand
    from cli.commands.validate import ValidateCommand
except ImportError:
    # When run as local script
    from core.config import ConfigManager
    from core.discovery import ChapterDiscovery, AmbiguousChapterError
    from commands.build import BuildCommand
    from commands.preview import PreviewCommand
    from commands.doctor import DoctorCommand
    from commands.clean import CleanCommand
    from commands.maintenance import MaintenanceCommand
    from commands.debug import DebugCommand
    from commands.validate import ValidateCommand

console = Console()


class MLSysBookCLI:
    """Main CLI application class."""

    def __init__(self, verbose: bool = False, open_after: bool = False):
        """Initialize the CLI with all components.

        Args:
            verbose: If True, stream build output in real-time
            open_after: If True, open build output after successful build
        """
        self.root_dir = Path.cwd()
        self.verbose = verbose
        self.open_after = open_after

        # Initialize core components
        self.config_manager = ConfigManager(self.root_dir)
        self.chapter_discovery = ChapterDiscovery(self.config_manager.book_dir)

        # Initialize command handlers
        self.build_command = BuildCommand(self.config_manager, self.chapter_discovery, verbose=verbose, open_after=open_after)
        self.preview_command = PreviewCommand(self.config_manager, self.chapter_discovery)
        self.doctor_command = DoctorCommand(self.config_manager, self.chapter_discovery)
        self.clean_command = CleanCommand(self.config_manager, self.chapter_discovery)
        self.maintenance_command = MaintenanceCommand(self.config_manager, self.chapter_discovery)
        self.debug_command = DebugCommand(self.config_manager, self.chapter_discovery, verbose=verbose)
        self.validate_command = ValidateCommand(self.config_manager, self.chapter_discovery)

    def show_banner(self):
        """Display the CLI banner."""
        banner = Panel(
            "[bold blue]📚 MLSysBook CLI v2.0[/bold blue]\n"
            "[dim]⚡ Modular, maintainable, and fast[/dim]",
            border_style="cyan"
        )
        console.print(banner)

    def show_help(self):
        """Display help information."""
        self.show_banner()

        # Fast Chapter Commands
        fast_table = Table(show_header=True, header_style="bold green", box=None)
        fast_table.add_column("Command", style="green", width=35)
        fast_table.add_column("Description", style="white", width=30)
        fast_table.add_column("Example", style="dim", width=30)

        fast_table.add_row("build [fmt] [chapter[,ch2,...]]", "Build HTML/PDF/EPUB by format", "./binder build pdf intro")
        fast_table.add_row("preview [chapter[,ch2,...]]", "Start live dev server with hot reload", "./binder preview intro")

        # Volume Commands
        vol_table = Table(show_header=True, header_style="bold magenta", box=None)
        vol_table.add_column("Command", style="magenta", width=35)
        vol_table.add_column("Description", style="white", width=30)
        vol_table.add_column("Example", style="dim", width=30)

        vol_table.add_row("build html --vol1", "Build Volume I website", "./binder build html --vol1")
        vol_table.add_row("build html --vol2", "Build Volume II website", "./binder build html --vol2")
        vol_table.add_row("build pdf --vol1", "Build Volume I as PDF", "./binder build pdf --vol1")
        vol_table.add_row("build pdf --vol2", "Build Volume II as PDF", "./binder build pdf --vol2")
        vol_table.add_row("build epub --vol1", "Build Volume I as EPUB", "./binder build epub --vol1")
        vol_table.add_row("build epub --vol2", "Build Volume II as EPUB", "./binder build epub --vol2")
        vol_table.add_row("list --vol1", "List Volume I chapters", "./binder list --vol1")
        vol_table.add_row("list --vol2", "List Volume II chapters", "./binder list --vol2")

        # Full Book Commands
        full_table = Table(show_header=True, header_style="bold blue", box=None)
        full_table.add_column("Command", style="blue", width=35)
        full_table.add_column("Description", style="white", width=30)
        full_table.add_column("Example", style="dim", width=30)

        full_table.add_row("build", "Build entire book as static HTML", "./binder build")
        full_table.add_row("build html --all", "Build ALL chapters using HTML config", "./binder build html --all")
        full_table.add_row("preview", "Start live dev server for entire book", "./binder preview")
        full_table.add_row("build pdf --all", "Build full book (both volumes)", "./binder build pdf --all")
        full_table.add_row("build epub --all", "Build full book (both volumes)", "./binder build epub --all")

        # Management Commands
        mgmt_table = Table(show_header=True, header_style="bold blue", box=None)
        mgmt_table.add_column("Command", style="green", width=35)
        mgmt_table.add_column("Description", style="white", width=30)
        mgmt_table.add_column("Example", style="dim", width=30)

        mgmt_table.add_row("debug <fmt> --vol1|--vol2", "Find failing chapter + section", "./binder debug pdf --vol1")
        mgmt_table.add_row("debug <fmt> --chapter <ch>", "Section-level debug (skip scan)", "./binder debug pdf --vol1 --chapter intro")
        mgmt_table.add_row("maintain <topic> ...", "Run maintenance namespace commands", "./binder maintain repo-health")
        mgmt_table.add_row("validate <subcommand>", "Run native validation checks", "./binder validate inline-refs")
        mgmt_table.add_row("clean", "Clean build artifacts", "./binder clean")
        mgmt_table.add_row("switch <format>", "Switch active config", "./binder switch pdf")
        mgmt_table.add_row("list", "List available chapters", "./binder list")
        mgmt_table.add_row("status", "Show current config status", "./binder status")
        mgmt_table.add_row("doctor", "Run comprehensive health check", "./binder doctor")
        mgmt_table.add_row("setup", "Setup development environment", "./binder setup")
        mgmt_table.add_row("hello", "Show welcome message", "./binder hello")
        mgmt_table.add_row("about", "Show project information", "./binder about")
        mgmt_table.add_row("help", "Show this help", "./binder help")

        # Display tables
        console.print(Panel(fast_table, title="⚡ Fast Chapter Commands", border_style="green"))
        console.print(Panel(vol_table, title="📖 Volume Commands", border_style="magenta"))
        console.print(Panel(full_table, title="📚 Full Book Commands", border_style="blue"))
        console.print(Panel(mgmt_table, title="🔧 Management", border_style="yellow"))

        # Pro Tips
        examples = Text()
        examples.append("🎯 Modular CLI Examples:\n", style="bold magenta")
        examples.append("  ./binder build pdf --vol1 ", style="cyan")
        examples.append("# Build Volume I as PDF\n", style="dim")
        examples.append("  ./binder build pdf --vol2 ", style="cyan")
        examples.append("# Build Volume II as PDF\n", style="dim")
        examples.append("  ./binder build pdf vol1/intro ", style="cyan")
        examples.append("# Build specific chapter (disambiguate with vol prefix)\n", style="dim")
        examples.append("  ./binder build pdf --all ", style="cyan")
        examples.append("# Build entire book as PDF (both volumes)\n", style="dim")
        examples.append("  ./binder list --vol1 ", style="cyan")
        examples.append("# List only Volume I chapters\n", style="dim")

        console.print(Panel(examples, title="💡 Pro Tips", border_style="magenta"))

        # Command Aliases
        aliases_text = Text()
        aliases_text.append("🔗 Command Aliases:\n", style="bold cyan")
        aliases_text.append("  b → build    ", style="green")
        aliases_text.append("  p → preview\n", style="green")
        aliases_text.append("  l → list     ", style="green")
        aliases_text.append("  s → status     ", style="green")
        aliases_text.append("  d → doctor     ", style="green")
        aliases_text.append("  h → help\n", style="green")

        console.print(Panel(aliases_text, title="⚡ Shortcuts", border_style="cyan"))

        # Global Options
        options_text = Text()
        options_text.append("🔧 Global Options:\n", style="bold yellow")
        options_text.append("  -v, --verbose  ", style="yellow")
        options_text.append("Stream build output in real-time\n", style="dim")
        options_text.append("  -o, --open     ", style="yellow")
        options_text.append("Open output after successful build\n", style="dim")
        options_text.append("  Example: ", style="dim")
        options_text.append("./binder build pdf --vol1 -v -o", style="cyan")

        console.print(Panel(options_text, title="⚙️ Options", border_style="yellow"))

    def _parse_build_args(self, args):
        """Parse `binder build` arguments into format, scope, and targets."""
        format_type = None
        volume = None
        build_all = False
        remaining = []

        for arg in args:
            lower = arg.lower()
            if lower == "--vol1":
                volume = "vol1"
            elif lower == "--vol2":
                volume = "vol2"
            elif lower == "--all":
                build_all = True
            elif format_type is None and lower in ("html", "pdf", "epub"):
                format_type = lower
            else:
                remaining.append(arg)

        if format_type is None:
            format_type = "html"

        chapters_arg = remaining[0] if remaining else None
        return format_type, volume, build_all, chapters_arg

    def handle_build_command(self, args):
        """Handle unified build command.

        Usage:
            ./binder build
            ./binder build pdf
            ./binder build epub --vol2
            ./binder build html intro,frameworks
        """
        if "-h" in args or "--help" in args:
            console.print("Usage: ./binder build [html|pdf|epub] [chapters] [--vol1|--vol2|--all]", markup=False)
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder build[/dim]")
            console.print("[dim]  ./binder build pdf[/dim]")
            console.print("[dim]  ./binder build pdf intro,training --vol1[/dim]")
            console.print("[dim]  ./binder build html --all[/dim]")
            return True

        self.config_manager.show_symlink_status()
        format_type, volume, build_all, chapters_arg = self._parse_build_args(args)

        if build_all and chapters_arg:
            console.print("[red]❌ Cannot combine explicit chapters with --all[/red]")
            return False

        if build_all:
            if format_type == "html":
                console.print("[green]🌐 Building HTML with ALL chapters...[/green]")
                return self.build_command.build_html_only()
            console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
            return self.build_command.build_full(format_type)

        if volume and not chapters_arg:
            volume_name = "Volume I" if volume == "vol1" else "Volume II"
            console.print(f"[magenta]🏗️ Building {volume_name} ({format_type.upper()})...[/magenta]")
            return self.build_command.build_volume(volume, format_type)

        if volume and chapters_arg:
            chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
            console.print(f"[green]🏗️ Building {format_type.upper()} chapters in {volume}: {chapters_arg}[/green]")
            return self.build_command.build_chapters_with_volume(chapter_list, format_type, volume)

        if chapters_arg:
            chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
            console.print(f"[green]🏗️ Building {format_type.upper()} chapter(s): {chapters_arg}[/green]")
            if format_type == "html":
                return self.build_command.build_html_only(chapter_list)
            return self.build_command.build_chapters(chapter_list, format_type)

        console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
        if format_type == "html":
            return self.build_command.build_full("html")
        return self.build_command.build_full(format_type)

    def handle_preview_command(self, args):
        """Handle preview command."""
        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - preview entire book
            console.print("[blue]🌐 Starting preview for entire book...[/blue]")
            return self.preview_command.preview_full("html")
        else:
            # Chapter specified
            chapter = args[0]
            if ',' in chapter:
                console.print("[yellow]⚠️ Preview only supports single chapters, not multiple[/yellow]")
                console.print("[dim]💡 Use the first chapter from your list[/dim]")
                chapter = chapter.split(',')[0].strip()

            console.print(f"[blue]🌐 Starting preview for chapter: {chapter}[/blue]")
            return self.preview_command.preview_chapter(chapter)

    def handle_doctor_command(self, args):
        """Handle doctor/health check command."""
        return self.doctor_command.run_health_check()

    def handle_clean_command(self, args):
        """Handle clean command."""
        if len(args) > 0:
            # Clean specific format
            format_type = args[0].lower()
            if format_type in ["html", "pdf", "epub"]:
                return self.clean_command.clean_format(format_type)
            else:
                console.print(f"[red]❌ Unknown format: {format_type}[/red]")
                console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
                return False
        else:
            # Clean all
            return self.clean_command.clean_all()

    def handle_switch_command(self, args):
        """Handle switch command."""
        if len(args) < 1:
            console.print("[red]❌ Usage: ./binder switch <format>[/red]")
            console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
            return False

        format_type = args[0].lower()
        return self.maintenance_command.switch_format(format_type)

    def handle_setup_command(self, args):
        """Handle setup command."""
        return self.maintenance_command.setup_environment()

    def handle_hello_command(self, args):
        """Handle hello command."""
        return self.maintenance_command.show_hello()

    def handle_about_command(self, args):
        """Handle about command."""
        return self.maintenance_command.show_about()

    def handle_maintain_command(self, args):
        """Handle maintenance namespace command."""
        return self.maintenance_command.run_namespace(args)


    def handle_debug_command(self, args):
        """Handle debug command.

        Usage:
            ./binder debug pdf --vol1
            ./binder debug html --vol2 --chapter training
        """
        # Parse args: first positional is format, then flags
        format_type = None
        volume = None
        chapter = None

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--vol1":
                volume = "vol1"
            elif arg == "--vol2":
                volume = "vol2"
            elif arg == "--chapter" and i + 1 < len(args):
                i += 1
                chapter = args[i]
            elif arg in ("pdf", "html", "epub") and format_type is None:
                format_type = arg
            else:
                # Try as format or chapter
                if format_type is None and arg in ("pdf", "html", "epub"):
                    format_type = arg
                else:
                    console.print(f"[red]Unknown argument: {arg}[/red]")
                    return False
            i += 1

        if not format_type:
            format_type = "pdf"  # Default to PDF
        if not volume:
            console.print("[red]Please specify --vol1 or --vol2[/red]")
            console.print("[yellow]Usage: ./binder debug <pdf|html|epub> --vol1|--vol2 [--chapter <name>][/yellow]")
            return False

        return self.debug_command.debug_build(format_type, volume, chapter)

    def handle_list_command(self, args):
        """Handle list chapters command."""
        volume = None
        if len(args) > 0:
            if args[0] == "--vol1":
                volume = "vol1"
            elif args[0] == "--vol2":
                volume = "vol2"

        self.chapter_discovery.show_chapters(volume=volume)
        return True

    def handle_validate_command(self, args):
        """Handle validate command group."""
        return self.validate_command.run(args)

    def handle_status_command(self, args):
        """Handle status command."""
        console.print("[bold blue]📊 MLSysBook CLI Status[/bold blue]")
        console.print(f"[dim]Root directory: {self.root_dir}[/dim]")
        console.print(f"[dim]Book directory: {self.config_manager.book_dir}[/dim]")

        # Show config status
        self.config_manager.show_symlink_status()

        # Show chapter count
        chapters = self.chapter_discovery.get_all_chapters()
        console.print(f"[dim]Available chapters: {len(chapters)}[/dim]")

        return True

    def run(self, args):
        """Run the CLI with given arguments."""
        if len(args) < 1:
            self.show_help()
            return True

        command = args[0].lower()
        command_args = args[1:]

        # Command mapping
        commands = {
            "build": self.handle_build_command,
            "preview": self.handle_preview_command,
            "clean": self.handle_clean_command,
            "debug": self.handle_debug_command,
            "switch": self.handle_switch_command,
            "list": self.handle_list_command,
            "status": self.handle_status_command,
            "doctor": self.handle_doctor_command,
            "maintain": self.handle_maintain_command,
            "validate": self.handle_validate_command,
            "setup": self.handle_setup_command,
            "hello": self.handle_hello_command,
            "about": self.handle_about_command,

            "help": lambda args: self.show_help() or True,
        }

        # Command aliases
        aliases = {
            "b": "build",
            "p": "preview",
            "l": "list",
            "s": "status",
            "d": "doctor",
            "h": "help",
        }

        # Resolve aliases
        if command in aliases:
            command = aliases[command]

        if command in ("html", "pdf", "epub"):
            console.print(f"[red]❌ Top-level '{command}' command was removed.[/red]")
            console.print(f"[yellow]💡 Use: ./binder build {command} ...[/yellow]")
            return False

        if command in commands:
            try:
                return commands[command](command_args)
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                return False
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                return False
        else:
            console.print(f"[red]❌ Unknown command: {command}[/red]")
            console.print("[yellow]💡 Use './binder help' to see available commands[/yellow]")
            return False


def main():
    """Main entry point."""
    # Check for global flags
    args = sys.argv[1:]
    verbose = False
    open_after = False

    if "-v" in args:
        verbose = True
        args.remove("-v")
    elif "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if "-o" in args:
        open_after = True
        args.remove("-o")
    elif "--open" in args:
        open_after = True
        args.remove("--open")

    cli = MLSysBookCLI(verbose=verbose, open_after=open_after)
    success = cli.run(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

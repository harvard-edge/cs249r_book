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
    from cli.core.discovery import ChapterDiscovery
    from cli.commands.build import BuildCommand
    from cli.commands.preview import PreviewCommand
    from cli.commands.doctor import DoctorCommand
    from cli.commands.clean import CleanCommand
    from cli.commands.maintenance import MaintenanceCommand
except ImportError:
    # When run as local script
    from core.config import ConfigManager
    from core.discovery import ChapterDiscovery
    from commands.build import BuildCommand
    from commands.preview import PreviewCommand
    from commands.doctor import DoctorCommand
    from commands.clean import CleanCommand
    from commands.maintenance import MaintenanceCommand

console = Console()


class MLSysBookCLI:
    """Main CLI application class."""

    def __init__(self):
        """Initialize the CLI with all components."""
        self.root_dir = Path.cwd()

        # Initialize core components
        self.config_manager = ConfigManager(self.root_dir)
        self.chapter_discovery = ChapterDiscovery(self.config_manager.book_dir)

        # Initialize command handlers
        self.build_command = BuildCommand(self.config_manager, self.chapter_discovery)
        self.preview_command = PreviewCommand(self.config_manager, self.chapter_discovery)
        self.doctor_command = DoctorCommand(self.config_manager, self.chapter_discovery)
        self.clean_command = CleanCommand(self.config_manager, self.chapter_discovery)
        self.maintenance_command = MaintenanceCommand(self.config_manager, self.chapter_discovery)

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

        fast_table.add_row("build [chapter[,ch2,...]]", "Build static files to disk (HTML)", "./binder build intro,ops")
        fast_table.add_row("html [chapter[,ch2,...]]", "Build HTML using quarto-html.yml", "./binder html intro")
        fast_table.add_row("preview [chapter[,ch2,...]]", "Start live dev server with hot reload", "./binder preview intro")
        fast_table.add_row("pdf [chapter[,ch2,...]]", "Build PDF (specified chapters)", "./binder pdf intro")
        fast_table.add_row("epub [chapter[,ch2,...]]", "Build EPUB (specified chapters)", "./binder epub intro")

        # Full Book Commands
        full_table = Table(show_header=True, header_style="bold blue", box=None)
        full_table.add_column("Command", style="blue", width=35)
        full_table.add_column("Description", style="white", width=30)
        full_table.add_column("Example", style="dim", width=30)

        full_table.add_row("build", "Build entire book as static HTML", "./binder build")
        full_table.add_row("html --all", "Build ALL chapters using quarto-html.yml", "./binder html --all")
        full_table.add_row("preview", "Start live dev server for entire book", "./binder preview")
        full_table.add_row("pdf --all", "Build full book (auto-uncomments all)", "./binder pdf --all")
        full_table.add_row("epub --all", "Build full book (auto-uncomments all)", "./binder epub --all")

        # Management Commands
        mgmt_table = Table(show_header=True, header_style="bold blue", box=None)
        mgmt_table.add_column("Command", style="green", width=35)
        mgmt_table.add_column("Description", style="white", width=30)
        mgmt_table.add_column("Example", style="dim", width=30)

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
        console.print(Panel(full_table, title="📚 Full Book Commands", border_style="blue"))
        console.print(Panel(mgmt_table, title="🔧 Management", border_style="yellow"))

        # Pro Tips
        examples = Text()
        examples.append("🎯 Modular CLI Examples:\n", style="bold magenta")
        examples.append("  ./binder build intro,ml_systems ", style="cyan")
        examples.append("# Build multiple chapters (HTML)\n", style="dim")
        examples.append("  ./binder html intro ", style="cyan")
        examples.append("# Build HTML with index.qmd + intro chapter only\n", style="dim")
        examples.append("  ./binder html --all ", style="cyan")
        examples.append("# Build HTML with ALL chapters\n", style="dim")
        examples.append("  ./binder pdf intro ", style="cyan")
        examples.append("# Build single chapter as PDF\n", style="dim")
        examples.append("  ./binder pdf --all ", style="cyan")
        examples.append("# Build entire book as PDF (uncomments all)\n", style="dim")

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

    def handle_build_command(self, args):
        """Handle build command (HTML format)."""
        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - build entire book
            console.print("[green]🏗️ Building entire book (HTML)...[/green]")
            return self.build_command.build_full("html")
        else:
            # Chapters specified
            chapters = args[0]
            console.print(f"[green]🏗️ Building chapter(s): {chapters}[/green]")
            chapter_list = [ch.strip() for ch in chapters.split(',')]
            return self.build_command.build_chapters(chapter_list, "html")

    def handle_html_command(self, args):
        """Handle HTML build command."""
        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - show error
            console.print("[red]❌ Error: Please specify chapters or use --all flag[/red]")
            console.print("[yellow]💡 Usage: ./binder html <chapter> or ./binder html --all[/yellow]")
            return False
        elif args[0] == "--all":
            # Build all chapters using HTML config
            console.print("[green]🌐 Building HTML with ALL chapters...[/green]")
            return self.build_command.build_html_only()
        else:
            # Chapters specified
            chapters = args[0]
            console.print(f"[green]🌐 Building HTML with chapters: {chapters}[/green]")
            chapter_list = [ch.strip() for ch in chapters.split(',')]
            return self.build_command.build_html_only(chapter_list)

    def handle_pdf_command(self, args):
        """Handle PDF build command."""
        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - show error
            console.print("[red]❌ Error: Please specify chapters or use --all flag[/red]")
            console.print("[yellow]💡 Usage: ./binder pdf <chapter> or ./binder pdf --all[/yellow]")
            return False
        elif args[0] == "--all":
            # Build entire book
            console.print("[red]📄 Building entire book (PDF)...[/red]")
            return self.build_command.build_full("pdf")
        else:
            # Chapters specified
            chapters = args[0]
            console.print(f"[red]📄 Building chapter(s) as PDF: {chapters}[/red]")
            chapter_list = [ch.strip() for ch in chapters.split(',')]
            return self.build_command.build_chapters(chapter_list, "pdf")

    def handle_epub_command(self, args):
        """Handle EPUB build command."""
        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - show error
            console.print("[red]❌ Error: Please specify chapters or use --all flag[/red]")
            console.print("[yellow]💡 Usage: ./binder epub <chapter> or ./binder epub --all[/yellow]")
            return False
        elif args[0] == "--all":
            # Build entire book
            console.print("[purple]📚 Building entire book (EPUB)...[/purple]")
            return self.build_command.build_full("epub")
        else:
            # Chapters specified
            chapters = args[0]
            console.print(f"[purple]📚 Building chapter(s) as EPUB: {chapters}[/purple]")
            chapter_list = [ch.strip() for ch in chapters.split(',')]
            return self.build_command.build_chapters(chapter_list, "epub")

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


    def handle_list_command(self, args):
        """Handle list chapters command."""
        self.chapter_discovery.show_chapters()
        return True

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
            "html": self.handle_html_command,
            "preview": self.handle_preview_command,
            "pdf": self.handle_pdf_command,
            "epub": self.handle_epub_command,
            "clean": self.handle_clean_command,
            "switch": self.handle_switch_command,
            "list": self.handle_list_command,
            "status": self.handle_status_command,
            "doctor": self.handle_doctor_command,
            "setup": self.handle_setup_command,
            "hello": self.handle_hello_command,
            "about": self.handle_about_command,

            "help": lambda args: self.show_help() or True,
        }

        # Command aliases
        aliases = {
            "b": "build",
            "p": "preview",
            "pdf": "pdf",  # Keep pdf as explicit command
            "epub": "epub",  # Keep epub as explicit command
            "l": "list",
            "s": "status",
            "d": "doctor",
            "h": "help",
        }

        # Resolve aliases
        if command in aliases:
            command = aliases[command]

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
            console.print("[yellow]💡 Use './binder2 help' to see available commands[/yellow]")
            return False


def main():
    """Main entry point."""
    cli = MLSysBookCLI()
    success = cli.run(sys.argv[1:])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

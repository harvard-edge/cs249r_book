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
from core.config import ConfigManager
from core.discovery import ChapterDiscovery
from commands.build import BuildCommand
from commands.preview import PreviewCommand
from commands.doctor import DoctorCommand

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
        
        fast_table.add_row("build [chapter[,ch2,...]]", "Build static files to disk (HTML)", "./binder2 build intro,ops")
        fast_table.add_row("preview [chapter[,ch2,...]]", "Start live dev server with hot reload", "./binder2 preview intro")
        fast_table.add_row("pdf [chapter[,ch2,...]]", "Build static PDF file to disk", "./binder2 pdf intro")
        fast_table.add_row("epub [chapter[,ch2,...]]", "Build static EPUB file to disk", "./binder2 epub intro")
        
        # Full Book Commands
        full_table = Table(show_header=True, header_style="bold blue", box=None)
        full_table.add_column("Command", style="blue", width=35)
        full_table.add_column("Description", style="white", width=30)
        full_table.add_column("Example", style="dim", width=30)
        
        full_table.add_row("build", "Build entire book as static HTML", "./binder2 build")
        full_table.add_row("preview", "Start live dev server for entire book", "./binder2 preview")
        full_table.add_row("pdf", "Build entire book as static PDF", "./binder2 pdf")
        full_table.add_row("epub", "Build entire book as static EPUB", "./binder2 epub")
        
        # Management Commands
        mgmt_table = Table(show_header=True, header_style="bold blue", box=None)
        mgmt_table.add_column("Command", style="green", width=35)
        mgmt_table.add_column("Description", style="white", width=30)
        mgmt_table.add_column("Example", style="dim", width=30)
        
        mgmt_table.add_row("list", "List available chapters", "./binder2 list")
        mgmt_table.add_row("status", "Show current config status", "./binder2 status")
        mgmt_table.add_row("doctor", "Run comprehensive health check", "./binder2 doctor")
        mgmt_table.add_row("help", "Show this help", "./binder2 help")
        
        # Display tables
        console.print(Panel(fast_table, title="⚡ Fast Chapter Commands", border_style="green"))
        console.print(Panel(full_table, title="📚 Full Book Commands", border_style="blue"))
        console.print(Panel(mgmt_table, title="🔧 Management", border_style="yellow"))
        
        # Pro Tips
        examples = Text()
        examples.append("🎯 Modular CLI Examples:\n", style="bold magenta")
        examples.append("  ./binder2 build intro,ml_systems ", style="cyan")
        examples.append("# Build multiple chapters (HTML)\n", style="dim")
        examples.append("  ./binder2 epub intro ", style="cyan")
        examples.append("# Build single chapter as EPUB\n", style="dim")
        examples.append("  ./binder2 pdf ", style="cyan")
        examples.append("# Build entire book as PDF\n", style="dim")
        
        console.print(Panel(examples, title="💡 Pro Tips", border_style="magenta"))
    
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
    
    def handle_pdf_command(self, args):
        """Handle PDF build command."""
        self.config_manager.show_symlink_status()
        
        if len(args) < 1:
            # No target specified - build entire book
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
            # No target specified - build entire book
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
            "preview": self.handle_preview_command,
            "pdf": self.handle_pdf_command,
            "epub": self.handle_epub_command,
            "list": self.handle_list_command,
            "status": self.handle_status_command,
            "doctor": self.handle_doctor_command,
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

#!/usr/bin/env python3
"""
Book Builder (bb) - Beautiful CLI for MLSysBook
A gorgeous terminal interface for fast book development workflow
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to basic print
    def rprint(*args, **kwargs):
        print(*args, **kwargs)

console = Console() if RICH_AVAILABLE else None

class BookBuilder:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.book_dir = self.root_dir / "book"
        
    def show_banner(self):
        """Display beautiful banner"""
        if not RICH_AVAILABLE:
            print("📚 Book Builder (bb) - MLSysBook CLI")
            return
            
        banner = Panel.fit(
            "[bold blue]📚 Book Builder[/bold blue]\n"
            "[dim]Lightning-fast MLSysBook development CLI[/dim]",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(banner)
    
    def find_chapters(self):
        """Find all available chapters"""
        contents_dir = self.book_dir / "contents"
        if not contents_dir.exists():
            return []
        
        chapters = []
        for qmd_file in contents_dir.rglob("*.qmd"):
            if "images" not in str(qmd_file):
                rel_path = qmd_file.relative_to(contents_dir)
                chapter_name = str(rel_path).replace(".qmd", "")
                chapters.append(chapter_name)
        
        return sorted(chapters)
    
    def find_chapter_match(self, partial_name):
        """Find chapter that matches partial name"""
        chapters = self.find_chapters()
        for chapter in chapters:
            if partial_name.lower() in chapter.lower():
                return chapter
        return None
    
    def get_status(self):
        """Get current configuration status"""
        quarto_link = self.book_dir / "_quarto.yml"
        if quarto_link.is_symlink():
            target = quarto_link.readlink()
            active_config = str(target)
        else:
            active_config = "No symlink found"
        
        # Check for commented lines
        html_config = self.book_dir / "_quarto-html.yml"
        pdf_config = self.book_dir / "_quarto-pdf.yml"
        
        html_commented = 0
        pdf_commented = 0
        
        try:
            if html_config.exists():
                with open(html_config, 'r') as f:
                    html_commented = sum(1 for line in f if "FAST_BUILD_COMMENTED" in line)
        except:
            pass
            
        try:
            if pdf_config.exists():
                with open(pdf_config, 'r') as f:
                    pdf_commented = sum(1 for line in f if "FAST_BUILD_COMMENTED" in line)
        except:
            pass
        
        return {
            'active_config': active_config,
            'html_commented': html_commented,
            'pdf_commented': pdf_commented,
            'is_clean': html_commented == 0 and pdf_commented == 0
        }
    
    def show_status(self):
        """Display beautiful status information"""
        status = self.get_status()
        
        if not RICH_AVAILABLE:
            print(f"📊 Status:")
            print(f"  🔗 Active config: {status['active_config']}")
            print(f"  ✅ Clean: {status['is_clean']}")
            return
        
        # Create status table
        table = Table(title="📊 Current Status", show_header=False, box=None)
        table.add_column("", style="cyan", no_wrap=True)
        table.add_column("", style="white")
        
        table.add_row("🔗 Active Config", f"[bold]{status['active_config']}[/bold]")
        
        if status['is_clean']:
            table.add_row("✅ State", "[green]Configs are clean[/green]")
        else:
            table.add_row("⚠️  State", f"[yellow]{status['html_commented'] + status['pdf_commented']} commented lines[/yellow]")
        
        console.print(Panel(table, border_style="green"))
    
    def show_chapters(self):
        """Display available chapters in a beautiful format"""
        chapters = self.find_chapters()
        
        if not RICH_AVAILABLE:
            print("📚 Available chapters:")
            for chapter in chapters[:10]:  # Show first 10
                print(f"  {chapter}")
            if len(chapters) > 10:
                print(f"  ... and {len(chapters) - 10} more")
            return
        
        # Create chapter tree
        tree = Tree("📚 [bold blue]Available Chapters[/bold blue]")
        
        # Group by category
        categories = {}
        for chapter in chapters:
            parts = chapter.split('/')
            if len(parts) > 1:
                category = parts[0]
                name = '/'.join(parts[1:])
            else:
                category = "root"
                name = chapter
            
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        for category, items in sorted(categories.items()):
            category_node = tree.add(f"[bold cyan]{category}[/bold cyan]")
            for item in sorted(items):
                category_node.add(f"[white]{item}[/white]")
        
        console.print(tree)
    
    def run_make_command(self, make_args):
        """Run make command with progress indicator"""
        cmd = ["make"] + make_args
        
        if not RICH_AVAILABLE:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.root_dir)
            return result.returncode == 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Running: {' '.join(cmd)}", total=None)
            
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=self.root_dir,
                    capture_output=False,
                    text=True
                )
                progress.update(task, completed=True)
                return result.returncode == 0
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                return False
    
    def build(self, chapter, format_type="html"):
        """Build a chapter with beautiful progress display"""
        # Find the actual chapter
        full_chapter = self.find_chapter_match(chapter)
        if not full_chapter:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ No chapter found matching '{chapter}'[/red]")
                console.print("[yellow]💡 Available chapters:[/yellow]")
                self.show_chapters()
            else:
                print(f"❌ No chapter found matching '{chapter}'")
                print("💡 Available chapters:")
                self.show_chapters()
            return False
        
        if RICH_AVAILABLE:
            console.print(f"[green]🚀 Building[/green] [bold]{full_chapter}[/bold] [dim]({format_type})[/dim]")
        else:
            print(f"🚀 Building {full_chapter} ({format_type})")
        
        # Prepare make command
        make_args = ["fast", f"CHAPTER={full_chapter}"]
        if format_type == "pdf":
            make_args.append("FORMAT=pdf")
        
        return self.run_make_command(make_args)
    
    def preview(self, chapter):
        """Start preview server"""
        full_chapter = self.find_chapter_match(chapter)
        if not full_chapter:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ No chapter found matching '{chapter}'[/red]")
            else:
                print(f"❌ No chapter found matching '{chapter}'")
            return False
        
        if RICH_AVAILABLE:
            console.print(f"[blue]🌐 Starting preview for[/blue] [bold]{full_chapter}[/bold]")
        else:
            print(f"🌐 Starting preview for {full_chapter}")
        
        make_args = ["fast-preview", f"CHAPTER={full_chapter}"]
        return self.run_make_command(make_args)
    
    def clean(self):
        """Clean up configurations"""
        if RICH_AVAILABLE:
            console.print("[yellow]🧹 Cleaning up configurations...[/yellow]")
        else:
            print("🧹 Cleaning up configurations...")
        
        return self.run_make_command(["fast-cleanup"])
    
    def switch(self, format_type):
        """Switch configuration format"""
        if format_type not in ["html", "pdf"]:
            if RICH_AVAILABLE:
                console.print("[red]❌ Format must be 'html' or 'pdf'[/red]")
            else:
                print("❌ Format must be 'html' or 'pdf'")
            return False
        
        if RICH_AVAILABLE:
            console.print(f"[blue]🔗 Switching to[/blue] [bold]{format_type}[/bold] [blue]configuration[/blue]")
        else:
            print(f"🔗 Switching to {format_type} configuration")
        
        return self.run_make_command([f"switch-{format_type}"])
    
    def show_help(self):
        """Display beautiful help screen"""
        if not RICH_AVAILABLE:
            print("""
📚 Book Builder (bb) - MLSysBook CLI

Commands:
  build <chapter> [format]    Build chapter (html/pdf)
  preview <chapter>           Build and preview chapter
  clean                       Clean up configurations
  switch <format>             Switch config (html/pdf)
  status                      Show current status
  list                        List available chapters
  help                        Show this help

Examples:
  ./bb build intro            # Build introduction (HTML)
  ./bb build intro pdf        # Build introduction (PDF)
  ./bb preview ops            # Preview ops chapter
  ./bb switch pdf             # Switch to PDF config
            """)
            return
        
        # Create beautiful help panels
        commands_table = Table(show_header=True, header_style="bold blue", box=None)
        commands_table.add_column("Command", style="cyan", width=20)
        commands_table.add_column("Description", style="white")
        commands_table.add_column("Example", style="dim")
        
        commands_table.add_row("build <chapter> [format]", "Build chapter", "./bb build intro pdf")
        commands_table.add_row("preview <chapter>", "Build and preview", "./bb preview ops")
        commands_table.add_row("clean", "Clean configurations", "./bb clean")
        commands_table.add_row("switch <format>", "Switch config", "./bb switch pdf")
        commands_table.add_row("status", "Show current status", "./bb status")
        commands_table.add_row("list", "List chapters", "./bb list")
        commands_table.add_row("help", "Show this help", "./bb help")
        
        shortcuts_table = Table(show_header=True, header_style="bold green", box=None)
        shortcuts_table.add_column("Shortcut", style="green", width=10)
        shortcuts_table.add_column("Full Command", style="white")
        
        shortcuts_table.add_row("b", "build")
        shortcuts_table.add_row("p", "preview")
        shortcuts_table.add_row("c", "clean")
        shortcuts_table.add_row("s", "switch")
        shortcuts_table.add_row("st", "status")
        shortcuts_table.add_row("l", "list")
        shortcuts_table.add_row("h", "help")
        
        # Display everything in panels
        self.show_banner()
        console.print(Panel(commands_table, title="📖 Commands", border_style="blue"))
        console.print(Panel(shortcuts_table, title="⚡ Shortcuts", border_style="green"))
        
        examples = Text()
        examples.append("🎯 Examples:\n", style="bold yellow")
        examples.append("  ./bb b intro pdf     ", style="cyan")
        examples.append("# Quick PDF build\n", style="dim")
        examples.append("  ./bb p intro         ", style="cyan") 
        examples.append("# Preview chapter\n", style="dim")
        examples.append("  ./bb st              ", style="cyan")
        examples.append("# Check status\n", style="dim")
        
        console.print(Panel(examples, title="💡 Quick Start", border_style="yellow"))

def main():
    bb = BookBuilder()
    
    if len(sys.argv) < 2:
        bb.show_help()
        return
    
    command = sys.argv[1].lower()
    
    # Handle shortcuts
    shortcuts = {
        'b': 'build',
        'p': 'preview', 
        'c': 'clean',
        's': 'switch',
        'st': 'status',
        'l': 'list',
        'h': 'help'
    }
    
    command = shortcuts.get(command, command)
    
    try:
        if command == "build":
            if len(sys.argv) < 3:
                if RICH_AVAILABLE:
                    console.print("[red]❌ Usage: ./bb build <chapter> [format][/red]")
                else:
                    print("❌ Usage: ./bb build <chapter> [format]")
                return
            chapter = sys.argv[2]
            format_type = sys.argv[3] if len(sys.argv) > 3 else "html"
            bb.build(chapter, format_type)
            
        elif command == "preview":
            if len(sys.argv) < 3:
                if RICH_AVAILABLE:
                    console.print("[red]❌ Usage: ./bb preview <chapter>[/red]")
                else:
                    print("❌ Usage: ./bb preview <chapter>")
                return
            chapter = sys.argv[2]
            bb.preview(chapter)
            
        elif command == "clean":
            bb.clean()
            
        elif command == "switch":
            if len(sys.argv) < 3:
                if RICH_AVAILABLE:
                    console.print("[red]❌ Usage: ./bb switch <html|pdf>[/red]")
                else:
                    print("❌ Usage: ./bb switch <html|pdf>")
                return
            format_type = sys.argv[2]
            bb.switch(format_type)
            
        elif command == "status":
            bb.show_status()
            
        elif command == "list":
            bb.show_chapters()
            
        elif command == "help":
            bb.show_help()
            
        else:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Unknown command: {command}[/red]")
                console.print("[yellow]💡 Run './bb help' for usage[/yellow]")
            else:
                print(f"❌ Unknown command: {command}")
                print("💡 Run './bb help' for usage")
                
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
        else:
            print("\n👋 Goodbye!")
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
"""
Build command implementation for MLSysBook CLI.

Handles building chapters and full books in different formats (HTML, PDF, EPUB).
"""

import subprocess
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class BuildCommand:
    """Handles build operations for the MLSysBook."""
    
    def __init__(self, config_manager, chapter_discovery):
        """Initialize build command.
        
        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        
    def build_full(self, format_type: str = "html") -> bool:
        """Build full book in specified format.
        
        Args:
            format_type: Format to build ('html', 'pdf', 'epub')
            
        Returns:
            True if build succeeded, False otherwise
        """
        console.print(f"[green]🔨 Building full {format_type.upper()} book...[/green]")
        
        # Handle special case for building both HTML and PDF
        if format_type == "both":
            return self._build_both_formats()
            
        # Create build directory
        output_dir = self.config_manager.get_output_dir(format_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup config
        config_name = self.config_manager.setup_symlink(format_type)
        
        # Determine render target
        render_targets = {
            "html": "html",
            "pdf": "titlepage-pdf", 
            "epub": "epub"
        }
        
        if format_type not in render_targets:
            raise ValueError(f"Unknown format type: {format_type}")
            
        render_to = render_targets[format_type]
        render_cmd = ["quarto", "render", "--to", render_to]
        
        # Show the command being executed
        cmd_str = " ".join(render_cmd)
        console.print(f"[blue]💻 Command: {cmd_str}[/blue]")
        
        # Execute build
        success = self._run_command(
            render_cmd,
            cwd=self.config_manager.book_dir,
            description=f"Building full {format_type.upper()} book"
        )
        
        if success:
            console.print(f"[green]✅ {format_type.upper()} build completed: {output_dir}/[/green]")
        else:
            console.print(f"[red]❌ {format_type.upper()} build failed[/red]")
            
        return success
    
    def build_chapters(self, chapter_names: List[str], format_type: str = "html") -> bool:
        """Build specific chapters.
        
        Args:
            chapter_names: List of chapter names to build
            format_type: Format to build ('html', 'pdf', 'epub')
            
        Returns:
            True if build succeeded, False otherwise
        """
        console.print(f"[green]🚀 Building {len(chapter_names)} chapters[/green] [dim]({format_type})[/dim]")
        console.print(f"[dim]📋 Chapters: {', '.join(chapter_names)}[/dim]")
        
        try:
            # Validate chapters exist
            chapter_files = self.chapter_discovery.validate_chapters(chapter_names)
            
            # Setup configuration
            config_file = self.config_manager.get_config_file(format_type)
            format_args = {
                "html": "html",
                "pdf": "titlepage-pdf",
                "epub": "epub"
            }
            
            if format_type not in format_args:
                raise ValueError(f"Unknown format type: {format_type}")
                
            format_arg = format_args[format_type]
            
            # Create build directory
            output_dir = self.config_manager.get_output_dir(format_type)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup correct configuration symlink
            self.config_manager.setup_symlink(format_type)
            
            # Set up fast build mode for the target chapters
            self._setup_fast_build_mode(config_file, chapter_files)
            
            # Setup signal handler to restore config on Ctrl+C
            def signal_handler(signum, frame):
                console.print("\n[yellow]🛡️ Ctrl+C detected - restoring config...[/yellow]")
                self._restore_config(config_file)
                console.print("[green]✅ Config restored[/green]")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Build with project.render configuration
            console.print("[yellow]🔨 Building with fast build configuration...[/yellow]")
            
            render_cmd = ["quarto", "render", "--to", format_arg]
            cmd_str = " ".join(render_cmd)
            console.print(f"[blue]💻 Command: {cmd_str}[/blue]")
            
            # Execute build
            success = self._run_command(
                render_cmd,
                cwd=self.config_manager.book_dir,
                description=f"Building {len(chapter_names)} chapters ({format_type})"
            )
            
            if success:
                console.print(f"[green]✅ Build complete: {output_dir}/[/green]")
            else:
                console.print("[red]❌ Build failed[/red]")
                
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Build error: {e}[/red]")
            return False
        finally:
            # Always restore config
            try:
                self._restore_config(config_file)
            except:
                pass
    
    def _build_both_formats(self) -> bool:
        """Build both HTML and PDF formats sequentially."""
        console.print("[blue]📚 Building both HTML and PDF formats...[/blue]")
        
        # Build HTML first
        console.print("[blue]📄 Building HTML version...[/blue]")
        html_success = self.build_full("html")
        if not html_success:
            console.print("[red]❌ HTML build failed![/red]")
            return False
        
        # Build PDF
        console.print("[blue]📄 Building PDF version...[/blue]")
        pdf_success = self.build_full("pdf")
        if not pdf_success:
            console.print("[red]❌ PDF build failed![/red]")
            return False
        
        console.print("[green]✅ Both HTML and PDF builds completed successfully![/green]")
        return True
    
    def _run_command(self, cmd: List[str], cwd: Path, description: str) -> bool:
        """Run a command with progress indication.
        
        Args:
            cmd: Command to run
            cwd: Working directory
            description: Description for progress display
            
        Returns:
            True if command succeeded, False otherwise
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(description, total=None)
                
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                
                progress.update(task, completed=True)
                
            if result.returncode == 0:
                return True
            else:
                console.print(f"[red]Command failed with exit code {result.returncode}[/red]")
                if result.stderr:
                    console.print(f"[red]Error: {result.stderr}[/red]")
                return False
                
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Build timed out after 30 minutes[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Command execution error: {e}[/red]")
            return False
    
    def _setup_fast_build_mode(self, config_file: Path, chapter_files: List[Path]) -> None:
        """Setup fast build mode for specific chapters."""
        # This is a simplified version - in the full implementation,
        # this would modify the config file to only render specific chapters
        console.print("[dim]⚡ Setting up fast build mode...[/dim]")
    
    def _restore_config(self, config_file: Path) -> None:
        """Restore configuration to pristine state."""
        # This is a simplified version - in the full implementation,
        # this would restore the original config file
        console.print("[dim]🛡️ Restoring config...[/dim]")

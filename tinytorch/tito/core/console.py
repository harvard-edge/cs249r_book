"""
Console management for consistent CLI output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.align import Align
from typing import Optional
import sys

# Global console instance
_console: Optional[Console] = None

def get_console() -> Console:
    """Get the global console instance."""
    global _console
    if _console is None:
        _console = Console(stderr=False)
    return _console

def print_banner(compact: bool = False):
    """Print the TinyTorch banner using Rich with clean block text style."""
    console = get_console()
    if compact:
        print_compact_banner()
    else:
        # Create banner text that matches the clean block text theme
        banner_text = Text()
        banner_text.append("Tiny", style="dim cyan")
        banner_text.append("🔥", style="red")
        banner_text.append("TORCH", style="bold orange1")
        banner_text.append(": Don't import it. Build it.", style="dim")
        console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

def print_compact_banner():
    """Print a compact TinyTorch banner with 'Tiny' above TORCH."""
    console = get_console()
    # Create compact banner text
    banner_text = Text()
    banner_text.append("Tiny", style="dim cyan")
    banner_text.append("\n🔥", style="red")
    banner_text.append("TORCH", style="bold orange1")
    banner_text.append(": Don't import it. Build it.", style="dim")
    console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

def print_ascii_logo(compact: bool = False):
    """Print the clean, minimal ASCII art TinyTorch logo."""
    console = get_console()
    
    if compact:
        print_compact_ascii_logo()
        return
    
    # Create styled logo text with proper Rich formatting
    logo_text = Text()
    
    # ============================================
    # TINYTORCH LOGO - EDIT HERE!
    # ============================================
    # To edit: Change the ASCII characters in logo_lines
    # Add/remove spaces at the beginning of each line to adjust positioning
    
    logo_lines = [
        # Flames positioned closer to T and H for visual cohesion
        "      🔥                                         🔥",              # Flames above T and H
        "      ████████╗  ██████╗  ██████╗   ██████╗ ██╗  ██╗",           # TORCH line 1
        "      ╚T═██╔══╝ ██╔═══██╗ ██╔══██╗ ██╔════╝ ██║  ██║",           # TORCH line 2
        "       I ██║    ██║   ██║ ██████╔╝ ██║      ███████║",           # TORCH line 3
        "       N ██║    ██║   ██║ ██╔══██╗ ██║      ██╔══██║",           # TORCH line 4
        "       Y ██║    ╚██████╔╝ ██║  ██║ ╚██████╗ ██║  ██║",           # TORCH line 5
        "         ╚═╝     ╚═════╝  ╚═╝  ╚═╝  ╚═════╝ ╚═╝  ╚═╝"            # TORCH line 6
    ]
    
    # ============================================
    # COLOR CONFIGURATION - EDIT COLORS HERE!
    # ============================================
    # Available colors: black, red, green, yellow, blue, magenta, cyan, white
    # Prefix with 'bright_' for brighter versions (e.g., 'bright_red')
    # Add 'bold' for bold text (e.g., 'bold red' or 'bold bright_black')
    
    FLAME_COLOR = "yellow"              # Color for 🔥 emoji
    TINY_COLOR = "bold orange1"         # Color for "tiny" text (flame effect!)
    TORCH_COLOR = "bold white"          # Color for "TORCH" text (better contrast)
    TAGLINE_COLOR = "bold orange1"      # Color for tagline (matches flame glow)
    
    # Process and apply colors to each line
    for i, line in enumerate(logo_lines):
        if i == 0:  # Flame line
            logo_text.append(line, style=FLAME_COLOR)
        elif i >= 1 and i <= 5:  # Lines with tiny letters (t,i,n,y) + TORCH
            # Color individual tiny letters within the line
            for char in line:
                if char in 'TINY':
                    logo_text.append(char, style=TINY_COLOR)
                else:
                    logo_text.append(char, style=TORCH_COLOR)
        else:  # Pure TORCH lines
            logo_text.append(line, style=TORCH_COLOR)
        logo_text.append("\n")
    
    # Add tagline (aligned under TORCH, matches flame glow)
    logo_text.append("\n                🔥 Don't import it. Build it.", style=TAGLINE_COLOR)
    logo_text.append("\n")
        
    # Combine logo and tagline
    full_content = Text()
    full_content.append(logo_text)
    
    # Display centered with rich styling
    console.print()
    console.print(Panel(
        Align.center(full_content),
        border_style="bright_blue",
        padding=(1, 2)
    ))
    console.print()

def print_compact_ascii_logo():
    """Print the compact ASCII art TinyTorch logo - same as main logo now."""
    # Just use the main logo since it's already compact and clean
    print_ascii_logo(compact=False)

def print_error(message: str, title: str = "Error"):
    """Print an error message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[red]❌ {message}[/red]", title=title, border_style="red"))

def print_success(message: str, title: str = "Success"):
    """Print a success message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[green]✅ {message}[/green]", title=title, border_style="green"))

def print_warning(message: str, title: str = "Warning"):
    """Print a warning message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[yellow]⚠️ {message}[/yellow]", title=title, border_style="yellow"))

def print_info(message: str, title: str = "Info"):
    """Print an info message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[cyan]ℹ️ {message}[/cyan]", title=title, border_style="cyan")) 
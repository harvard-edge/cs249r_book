"""
Cross-platform browser opening utility for TinyTorch CLI.
Handles WSL, macOS, Linux, and Windows environments gracefully.
"""
import webbrowser
import subprocess
import platform
from typing import Optional
from rich.console import Console
from rich.panel import Panel


def is_wsl() -> bool:
    """Check if running in WSL (Windows Subsystem for Linux) environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def open_url(url: str, console: Optional[Console] = None, show_manual_fallback: bool = True) -> bool:
    """
    Open URL in browser with cross-platform support.

    Args:
        url: The URL to open
        console: Optional Rich console for output
        show_manual_fallback: Whether to show manual instructions if browser fails

    Returns:
        True if browser was opened successfully, False otherwise
    """
    if console is None:
        console = Console()

    browser_opened = False
    system = platform.system()

    # Try WSL-specific approach first
    if is_wsl():
        console.print("[cyan]Detected WSL environment - opening Windows browser...[/cyan]")
        browser_opened = _open_url_wsl(url)

    # Try macOS-specific approach
    elif system == "Darwin":
        browser_opened = _open_url_macos(url)

    # Try Windows-specific approach
    elif system == "Windows":
        browser_opened = _open_url_windows(url)

    # Try standard webbrowser module
    if not browser_opened:
        try:
            browser_opened = webbrowser.open(url)
        except Exception:
            pass

    # Handle success/failure
    if browser_opened:
        console.print(f"[green]✓[/green] Browser opened to: [cyan]{url}[/cyan]")
    else:
        if show_manual_fallback:
            console.print()
            console.print(Panel(
                f"[yellow]⚠️  Could not open browser automatically[/yellow]\n\n"
                f"Please manually open this URL in your browser:\n\n"
                f"[cyan]{url}[/cyan]\n\n"
                f"Copy and paste this link into your browser to continue.",
                title="Manual Browser Access Required",
                border_style="yellow"
            ))
            console.print()
        else:
            console.print(f"[yellow]⚠️  Could not open browser. Please manually visit:[/yellow] [cyan]{url}[/cyan]")

    return browser_opened


def _open_url_wsl(url: str) -> bool:
    """Try to open URL in Windows browser from WSL."""
    try:
        # Method 1: Use cmd.exe to start default browser
        result = subprocess.run(
            ['cmd.exe', '/c', 'start', url],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True

        # Method 2: Try powershell.exe
        result = subprocess.run(
            ['powershell.exe', '-Command', f'Start-Process "{url}"'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _open_url_macos(url: str) -> bool:
    """Try to open URL in macOS default browser."""
    try:
        result = subprocess.run(
            ['open', url],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _open_url_windows(url: str) -> bool:
    """Try to open URL in Windows default browser (native Windows only, not WSL)."""
    try:
        # Method 1: Use os.startfile (Windows-specific, most reliable)
        import os
        if hasattr(os, 'startfile'):
            os.startfile(url)
            return True

        # Method 2: Use start command via cmd
        result = subprocess.run(
            ['cmd', '/c', 'start', '', url],  # Empty string after 'start' handles URLs with special chars
            capture_output=True,
            timeout=10,  # Longer timeout for slower Windows systems
            shell=False
        )
        return result.returncode == 0
    except Exception:
        return False

"""
Setup command for Tiny🔥Torch CLI: First-time environment setup and configuration.

This replaces the old 01_setup module with a proper CLI command that handles:
- Package installation and virtual environment setup
- Environment validation and compatibility checking
- User profile creation for development tracking
- Workspace initialization for Tiny🔥Torch development
"""

import subprocess
import sys
import os
import platform
import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from .base import BaseCommand
from .login import LoginCommand
from ..core.console import get_console
from ..core.auth import is_logged_in
from ..core.browser import open_url

def _print_file_update(console, file_path: Path) -> None:
    """Print a notification when a file is created or updated."""
    try:
        if file_path.is_relative_to(Path.home()):
            relative_path = file_path.relative_to(Path.home())
            console.print(f"[dim]📝 Updated: ~/{relative_path}[/dim]")
        else:
            console.print(f"[dim]📝 Updated: {file_path}[/dim]")
    except (ValueError, AttributeError):
        console.print(f"[dim]📝 Updated: {file_path}[/dim]")

class SetupCommand(BaseCommand):
    """First-time setup command for Tiny🔥Torch development environment."""

    @property
    def name(self) -> str:
        return "setup"

    @property
    def description(self) -> str:
        return "Set up your development environment (idempotent)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add setup command arguments."""
        parser.description = (
            "Set up your Tiny🔥Torch development environment.\n\n"
            "This command is idempotent - safe to run multiple times. "
            "It will skip steps that are already complete and only set up what's missing.\n\n"
            "Steps performed:\n"
            "  1. Create virtual environment (.venv)\n"
            "  2. Install required packages (numpy, jupyter, etc.)\n"
            "  3. Create user profile (~/.tinytorch/profile.json)\n"
            "  4. Validate environment"
        )
        parser.add_argument(
            '--skip-venv',
            action='store_true',
            help='Skip virtual environment creation'
        )
        parser.add_argument(
            '--skip-packages',
            action='store_true',
            help='Skip package installation'
        )
        parser.add_argument(
            '--skip-profile',
            action='store_true',
            help='Skip user profile creation'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Prompt to recreate existing components (venv, profile)'
        )

    def get_existing_venv_path(self) -> Optional[Path]:
        """Return the path to an existing venv, or None if not found."""
        venv_paths = [
            self.config.project_root / ".venv",
            self.config.project_root / "venv",
            self.config.project_root / "tinytorch-env",
        ]
        for venv_path in venv_paths:
            if venv_path.exists():
                return venv_path
        return None

    def get_profile_path(self) -> Path:
        """Return the path to the profile file."""
        return Path.home() / ".tinytorch" / "profile.json"

    def check_existing_setup(self) -> Dict[str, Any]:
        """Check what parts of setup already exist.

        Returns a dict with status of each component.
        """
        profile_path = self.get_profile_path()
        venv_path = self.get_existing_venv_path()

        return {
            "has_profile": profile_path.exists(),
            "profile_path": profile_path,
            "has_venv": venv_path is not None,
            "venv_path": venv_path,
        }

    def _check_package_installed(self, package_name: str) -> bool:
        """Check if a package is already installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def install_packages(self) -> bool:
        """Install required packages for Tiny🔥Torch development."""
        # Essential packages for TinyTorch
        packages = [
            ("numpy", "numpy>=1.21.0"),
            ("jupyter", "jupyter>=1.0.0"),
            ("jupyterlab", "jupyterlab>=3.0.0"),
            ("jupytext", "jupytext>=1.13.0"),
            ("rich", "rich>=12.0.0"),
            ("pyyaml", "pyyaml>=6.0"),
            ("psutil", "psutil>=5.8.0"),
        ]

        # First, check what's already installed
        to_install = []
        already_installed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Checking installed packages...", total=None)
            for pkg_name, pkg_spec in packages:
                if self._check_package_installed(pkg_name):
                    already_installed.append(pkg_name)
                else:
                    to_install.append((pkg_name, pkg_spec))

        if already_installed:
            self.console.print(f"[green]✅ Already installed:[/green] [dim]{', '.join(already_installed)}[/dim]")

        if not to_install:
            self.console.print("[green]✅ All dependencies already installed[/green]")
        else:
            self.console.print(f"[cyan]📦 Installing:[/cyan] {', '.join(p[0] for p in to_install)}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                for pkg_name, pkg_spec in to_install:
                    task = progress.add_task(f"Installing {pkg_name}...", total=None)

                    try:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", "-q", pkg_spec
                        ], capture_output=True, text=True, timeout=120)

                        if result.returncode == 0:
                            progress.update(task, description=f"[green]✅ {pkg_name}[/green]")
                        else:
                            progress.update(task, description=f"[red]❌ {pkg_name} failed[/red]")
                            self.console.print(f"[red]Error installing {pkg_spec}: {result.stderr}[/red]")
                            return False

                    except subprocess.TimeoutExpired:
                        progress.update(task, description=f"[yellow]⏰ {pkg_name} timed out[/yellow]")
                        self.console.print(f"[yellow]Warning: {pkg_spec} installation timed out[/yellow]")
                    except Exception as e:
                        progress.update(task, description=f"[red]❌ {pkg_name} error[/red]")
                        self.console.print(f"[red]Error installing {pkg_spec}: {e}[/red]")
                        return False

        # Install Tiny🔥Torch in development mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Installing Tiny🔥Torch in development mode...", total=None)

            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q", "-e", "."
                ], cwd=self.config.project_root, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    progress.update(task, description="[green]✅ Tiny🔥Torch installed[/green]")
                    return True
                else:
                    progress.update(task, description="[red]❌ Tiny🔥Torch install failed[/red]")
                    self.console.print(f"[red]Failed to install Tiny🔥Torch: {result.stderr}[/red]")
                    return False

            except Exception as e:
                progress.update(task, description="[red]❌ Tiny🔥Torch error[/red]")
                self.console.print(f"[red]Error installing Tiny🔥Torch: {e}[/red]")
                return False

    def create_virtual_environment(self, force: bool = False) -> bool:
        """Create a virtual environment for Tiny🔥Torch development.

        Args:
            force: If True, recreate even if venv exists (after user confirmation).
        """
        venv_path = self.config.project_root / ".venv"

        if venv_path.exists():
            if not force:
                # Silently use existing - this is idempotent behavior
                self.console.print(f"[green]✅ Using existing virtual environment[/green] [dim]({venv_path})[/dim]")
                return True

            # Force mode - ask before destroying
            if not Confirm.ask(f"[yellow]Recreate virtual environment at {venv_path}?[/yellow] This will delete the existing one"):
                self.console.print("[green]✅ Keeping existing virtual environment[/green]")
                return True

            self.console.print("🐍 Recreating virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            self.console.print("🐍 Creating virtual environment...")

        try:
            # Detect Apple Silicon and force arm64 if needed
            arch = platform.machine()
            python_exe = sys.executable

            if platform.system() == "Darwin" and arch == "x86_64":
                # Check if we're on Apple Silicon but running Rosetta
                import subprocess as sp
                try:
                    # Check actual hardware
                    hw_check = sp.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True
                    )
                    if "Apple" in hw_check.stdout:
                        self.console.print("[yellow]⚠️  Detected Apple Silicon but Python is running in Rosetta (x86_64)[/yellow]")
                        self.console.print("[cyan]🔧 Creating arm64 native environment for better performance...[/cyan]")
                        # Force arm64 Python
                        python_exe = f"arch -arm64 {python_exe}"
                except:
                    pass

            # Create virtual environment (potentially with arch prefix)
            if "arch -arm64" in python_exe:
                result = subprocess.run(
                    f'{python_exe} -m venv {venv_path}',
                    shell=True,
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run([
                    python_exe, "-m", "venv", str(venv_path)
                ], capture_output=True, text=True)

            if result.returncode != 0:
                self.console.print(f"[red]Failed to create virtual environment: {result.stderr}[/red]")
                return False

            self.console.print(f"✅ Virtual environment created at {venv_path}")

            # Verify architecture
            venv_python = venv_path / "bin" / "python3"
            if venv_python.exists():
                arch_check = subprocess.run(
                    [str(venv_python), "-c", "import platform; print(platform.machine())"],
                    capture_output=True, text=True
                )
                if arch_check.returncode == 0:
                    venv_arch = arch_check.stdout.strip()
                    self.console.print(f"📐 Virtual environment architecture: {venv_arch}")

            return True

        except Exception as e:
            self.console.print(f"[red]Error creating virtual environment: {e}[/red]")
            return False


    def create_user_profile(self, force: bool = False) -> Dict[str, Any]:
        """Create user profile for development tracking.

        Args:
            force: If True, prompt to update existing profile.
        """
        # Use .tinytorch directory (flat structure, not nested under community/)
        tinytorch_dir = Path.home() / ".tinytorch"
        tinytorch_dir.mkdir(parents=True, exist_ok=True)
        profile_path = tinytorch_dir / "profile.json"

        if profile_path.exists():
            import json
            with open(profile_path, 'r') as f:
                existing_profile = json.load(f)

            if not force:
                # Silently use existing profile
                self.console.print(f"[green]✅ Using existing profile[/green] [dim]({existing_profile.get('name', 'Unknown')})[/dim]")
                return existing_profile

            # Force mode - ask before overwriting
            if not Confirm.ask("[yellow]Update your existing profile?[/yellow]"):
                self.console.print("[green]✅ Keeping existing profile[/green]")
                return existing_profile

        self.console.print("👋 Creating your Tiny🔥Torch development profile...")

        # Collect user information
        name = Prompt.ask("Your name", default="Tiny🔥Torch Developer")
        email = Prompt.ask("Your email (optional)", default="dev@tinytorch.local")
        affiliation = Prompt.ask("Your affiliation (university, company, etc.)", default="Independent")

        # Create profile
        profile = {
            "name": name,
            "email": email,
            "affiliation": affiliation,
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created": datetime.datetime.now().isoformat(),
            "setup_version": "2.0",
            "modules_completed": [],
            "last_active": datetime.datetime.now().isoformat()
        }

        # Save profile
        import json
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

        _print_file_update(self.console, profile_path)
        self.console.print(f"✅ Profile created for {profile['name']}")
        return profile

    def validate_environment(self) -> bool:
        """Validate the development environment setup."""
        checks = [
            ("Python version (≥3.8)", self.check_python_version),
            ("NumPy", self.check_numpy),
            ("Jupyter", self.check_jupyter),
            ("TinyTorch CLI", self.check_tinytorch_package)
        ]

        all_passed = True
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Validating environment...", total=None)
            for check_name, check_func in checks:
                try:
                    passed = check_func()
                    results.append((check_name, passed, None))
                    if not passed:
                        all_passed = False
                except Exception as e:
                    results.append((check_name, False, str(e)))
                    all_passed = False

        # Print results
        for check_name, passed, error in results:
            if passed:
                self.console.print(f"  [green]✅ {check_name}[/green]")
            elif error:
                self.console.print(f"  [red]❌ {check_name}: {error}[/red]")
            else:
                self.console.print(f"  [red]❌ {check_name}[/red]")

        return all_passed

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        return sys.version_info >= (3, 8)

    def check_numpy(self) -> bool:
        """Check if NumPy is installed and working."""
        try:
            import numpy as np
            # Test basic operation
            arr = np.array([1, 2, 3])
            return len(arr) == 3
        except ImportError:
            return False

    def check_jupyter(self) -> bool:
        """Check if Jupyter is installed."""
        try:
            import jupyter
            import jupyterlab
            return True
        except ImportError:
            return False

    def check_tinytorch_package(self) -> bool:
        """Check if Tiny🔥Torch package is installed (tito CLI)."""
        try:
            import tito
            return True
        except ImportError:
            return False

    def print_success_message(self, profile: Dict[str, Any]) -> None:
        """Print success message with next steps."""
        success_text = Text()
        success_text.append("🎉 Tiny🔥Torch setup completed successfully!\n\n", style="bold green")
        success_text.append(f"👋 Welcome, {profile['name']}!\n", style="bold")
        success_text.append(f"📧 Email: {profile['email']}\n", style="dim")
        success_text.append(f"🏢 Affiliation: {profile['affiliation']}\n", style="dim")
        success_text.append(f"💻 Platform: {profile['platform']}\n", style="dim")
        success_text.append(f"🐍 Python: {profile['python_version']}\n\n", style="dim")

        success_text.append("🔥 Activate your environment:\n\n", style="bold yellow")
        success_text.append("  source .venv/bin/activate", style="bold cyan")
        success_text.append("  # On Windows: .venv\\Scripts\\activate\n\n", style="dim")

        success_text.append("🚀 Start building ML systems:\n\n", style="bold green")
        success_text.append("  tito module start 01", style="bold green")
        success_text.append("  # Begin with tensor foundations\n\n", style="dim")

        success_text.append("💡 Essential commands:\n", style="bold")
        success_text.append("  • ", style="dim")
        success_text.append("tito system health", style="green")
        success_text.append(" - Check environment\n", style="dim")
        success_text.append("  • ", style="dim")
        success_text.append("tito module status", style="green")
        success_text.append(" - Track progress\n", style="dim")

        self.console.print(Panel(
            success_text,
            title="🔥 Tiny🔥Torch Setup Complete!",
            border_style="green"
        ))

    def prompt_community_registration(self) -> None:
        """Prompt user to join the TinyTorch community."""
        # Check if already logged in
        if is_logged_in():
             self.console.print("\n[green]✅ You are already connected to the TinyTorch community.[/green]")
             return

        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]🌍 Join the TinyTorch Community[/bold cyan]\n\n"
            "Connect at [link=https://mlsysbook.ai/tinytorch/community/?action=join]mlsysbook.ai/tinytorch/community/?action=join[/link]\n\n"
            "[dim]• See learners worldwide\n"
            "• Leaderboard submissions\n"
            "• Progress syncing[/dim]",
            border_style="cyan",
            box=box.ROUNDED
        ))

        join = Confirm.ask("\n[bold]Join the community?[/bold]", default=True)

        if join:
            self.console.print("\n[cyan]Starting community login process...[/cyan]")
            login_cmd = LoginCommand(self.config)

            # Create a dummy Namespace for login command arguments
            login_args = Namespace(force=False)

            try:
                login_result = login_cmd.run(login_args)

                if login_result == 0:
                    self.console.print("[green]✅ Successfully connected to the TinyTorch community![/green]")

                    # Post-login profile update prompt
                    self.console.print()
                    self.console.print(Panel(
                        "[bold magenta]✨ Update Community Profile ✨[/bold magenta]\n\n"
                        "Your CLI is now connected. Would you like to update your profile on the TinyTorch community website?",
                        title="Community Profile Update",
                        border_style="magenta",
                        box=box.ROUNDED
                    ))
                    if Confirm.ask("[bold]Update your community profile?[/bold]", default=True):
                        self.console.print("[dim]Opening profile editor...[/dim]")
                        open_url("https://mlsysbook.ai/tinytorch/community/?action=profile&community=true", self.console, show_manual_fallback=True)
                else:
                    self.console.print("[yellow]⚠️  Community connection failed or was cancelled. You can try again later with 'tito login'.[/yellow]")
            except Exception as e:
                 self.console.print(f"[yellow]⚠️  Error during login: {e}[/yellow]")
        else:
            self.console.print("[dim]No problem! You can join anytime at mlsysbook.ai/tinytorch/community/[/dim]")

    def prompt_community_login(self) -> None:
        """Prompt user to log in to the TinyTorch community via CLI."""
        self.console.print()
        if Confirm.ask("[bold]Would you like to connect your TinyTorch CLI to the community now (for leaderboard submissions, progress syncing, etc.)?[/bold]", default=True):
            self.console.print("\n[cyan]Starting community login process...[/cyan]")
            login_cmd = LoginCommand(self.config)

            # Create a dummy Namespace for login command arguments
            login_args = Namespace(force=False)

            try:
                login_result = login_cmd.run(login_args)

                if login_result == 0:
                    self.console.print("[green]✅ Successfully connected to the TinyTorch community![/green]")
                else:
                    self.console.print("[yellow]⚠️  Community connection failed or was cancelled. You can try again later with 'tito community login'.[/yellow]")
            except Exception as e:
                 self.console.print(f"[yellow]⚠️  Error during login: {e}[/yellow]")
        else:
            self.console.print("[dim]You can connect to the community anytime with 'tito community login'.[/dim]")

    def run(self, args: Namespace) -> int:
        """Execute the setup command."""
        self.console.print(Panel(
            "🔥 Tiny🔥Torch First-Time Setup\n\n"
            "This will configure your development environment for building ML systems from scratch.",
            title="Welcome to Tiny🔥Torch!",
            border_style="bright_green"
        ))

        # Check existing setup status
        status = self.check_existing_setup()
        is_fresh_install = not status["has_venv"] and not status["has_profile"]

        if args.force:
            self.console.print("[yellow]⚠️  Force mode: will prompt to recreate existing components[/yellow]\n")
        elif not is_fresh_install:
            self.console.print("[dim]Checking existing setup...[/dim]\n")

        try:
            # Step 1: Virtual environment
            self.console.print("[bold]Step 1/4:[/bold] Virtual Environment")
            if not args.skip_venv:
                if not self.create_virtual_environment(force=args.force):
                    self.console.print("[yellow]⚠️  Virtual environment setup failed, but continuing...[/yellow]")
            else:
                self.console.print("[dim]  ⏭️  Skipped (--skip-venv)[/dim]")
            self.console.print()

            # Step 2: Install packages
            self.console.print("[bold]Step 2/4:[/bold] Package Installation")
            if not args.skip_packages:
                if not self.install_packages():
                    self.console.print("[red]❌ Package installation failed[/red]")
                    return 1
            else:
                self.console.print("[dim]  ⏭️  Skipped (--skip-packages)[/dim]")
            self.console.print()

            # Step 3: Create user profile
            self.console.print("[bold]Step 3/4:[/bold] User Profile")
            profile = {}
            if not args.skip_profile:
                profile = self.create_user_profile(force=args.force)
            else:
                self.console.print("[dim]  ⏭️  Skipped (--skip-profile)[/dim]")
            self.console.print()

            # Step 4: Validate environment
            self.console.print("[bold]Step 4/4:[/bold] Environment Validation")
            if not self.validate_environment():
                self.console.print("[yellow]⚠️  Some validation checks failed, but setup completed[/yellow]")
            self.console.print()

            # Success!
            if profile:
                self.print_success_message(profile)
            else:
                self.console.print("[green]✅ Setup completed successfully![/green]")
                self.console.print("💡 Try: [bold]tito module start 01[/bold]")

            # Prompt to join community
            self.prompt_community_registration()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return 130
        except Exception as e:
            self.console.print(f"[red]Setup failed: {e}[/red]")
            return 1

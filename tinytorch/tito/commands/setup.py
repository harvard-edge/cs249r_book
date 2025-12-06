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
import webbrowser
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
        return "First-time setup: install packages, create profile, initialize workspace"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add setup command arguments."""
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
            help='Force setup even if already configured'
        )

    def check_existing_setup(self) -> bool:
        """Check if Tiny🔥Torch is already set up."""
        # Check for profile file in .tinytorch (flat structure)
        profile_path = Path.home() / ".tinytorch" / "profile.json"

        # Check for virtual environment
        venv_paths = [
            self.config.project_root / ".venv",
            self.config.project_root / "venv",
            self.config.project_root / "tinytorch-env",
        ]

        has_profile = profile_path.exists()
        has_venv = any(venv_path.exists() for venv_path in venv_paths)

        return has_profile and has_venv

    def install_packages(self) -> bool:
        """Install required packages for Tiny🔥Torch development."""
        self.console.print("📦 Installing Tiny🔥Torch dependencies...")

        # Essential packages for TinyTorch
        packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "jupytext>=1.13.0",
            "rich>=12.0.0",
            "pyyaml>=6.0",
            "psutil>=5.8.0"
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            for package in packages:
                task = progress.add_task(f"Installing {package.split('>=')[0]}...", total=None)

                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True, timeout=120)

                    if result.returncode == 0:
                        progress.update(task, description=f"✅ {package.split('>=')[0]} installed")
                    else:
                        progress.update(task, description=f"❌ {package.split('>=')[0]} failed")
                        self.console.print(f"[red]Error installing {package}: {result.stderr}[/red]")
                        return False

                except subprocess.TimeoutExpired:
                    progress.update(task, description=f"⏰ {package.split('>=')[0]} timed out")
                    self.console.print(f"[yellow]Warning: {package} installation timed out[/yellow]")
                except Exception as e:
                    progress.update(task, description=f"❌ {package.split('>=')[0]} error")
                    self.console.print(f"[red]Error installing {package}: {e}[/red]")
                    return False

        # Install Tiny🔥Torch in development mode
        try:
            self.console.print("🔧 Installing Tiny🔥Torch in development mode...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], cwd=self.config.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                self.console.print("✅ Tiny🔥Torch installed in development mode")
                return True
            else:
                self.console.print(f"[red]Failed to install Tiny🔥Torch: {result.stderr}[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]Error installing Tiny🔥Torch: {e}[/red]")
            return False

    def create_virtual_environment(self) -> bool:
        """Create a virtual environment for Tiny🔥Torch development."""
        venv_path = self.config.project_root / ".venv"

        if venv_path.exists():
            if not Confirm.ask(f"Virtual environment already exists at {venv_path}. Recreate?"):
                self.console.print("[green]✅ Using existing virtual environment[/green]")
                return True

            self.console.print("🐍 Recreating virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            self.console.print("🐍 Setting up virtual environment...")

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


    def create_user_profile(self) -> Dict[str, Any]:
        """Create user profile for development tracking."""
        self.console.print("👋 Creating your Tiny🔥Torch development profile...")

        # Use .tinytorch directory (flat structure, not nested under community/)
        tinytorch_dir = Path.home() / ".tinytorch"
        tinytorch_dir.mkdir(parents=True, exist_ok=True)
        profile_path = tinytorch_dir / "profile.json"

        if profile_path.exists():
            if not Confirm.ask("Profile already exists. Update it?"):
                import json
                with open(profile_path, 'r') as f:
                    return json.load(f)

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
        self.console.print("🔍 Validating environment...")

        checks = [
            ("Python version", self.check_python_version),
            ("NumPy installation", self.check_numpy),
            ("Jupyter installation", self.check_jupyter),
            ("TinyTorch package", self.check_tinytorch_package)
        ]

        all_passed = True

        for check_name, check_func in checks:
            try:
                if check_func():
                    self.console.print(f"  ✅ {check_name}")
                else:
                    self.console.print(f"  ❌ {check_name}")
                    all_passed = False
            except Exception as e:
                self.console.print(f"  ❌ {check_name}: {e}")
                all_passed = False

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
        """Check if Tiny🔥Torch package is installed."""
        try:
            import tinytorch
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
            "Connect at [link=https://tinytorch.ai/community/?action=join]tinytorch.ai/community/?action=join[/link]\n\n"
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
                else:
                    self.console.print("[yellow]⚠️  Community connection failed or was cancelled. You can try again later with 'tito login'.[/yellow]")
            except Exception as e:
                 self.console.print(f"[yellow]⚠️  Error during login: {e}[/yellow]")
        else:
            self.console.print("[dim]No problem! You can join anytime at tinytorch.ai/community[/dim]")

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

        # Check if already set up
        if not args.force and self.check_existing_setup():
            if not Confirm.ask("Tiny🔥Torch appears to be already set up. Continue anyway?"):
                self.console.print("✅ Setup cancelled. You're ready to go!")
                self.console.print("💡 Try: tito module start 01")
                return 0

        try:
            # Step 1: Virtual environment (optional)
            if not args.skip_venv:
                if not self.create_virtual_environment():
                    self.console.print("[yellow]⚠️  Virtual environment setup failed, but continuing...[/yellow]")

            # Step 2: Install packages
            if not args.skip_packages:
                if not self.install_packages():
                    self.console.print("[red]❌ Package installation failed[/red]")
                    return 1

            # Step 3: Create user profile
            profile = {}
            if not args.skip_profile:
                profile = self.create_user_profile()

            # Step 4: Validate environment
            if not self.validate_environment():
                self.console.print("[yellow]⚠️  Some validation checks failed, but setup completed[/yellow]")

            # Success!
            if profile:  # Only print if profile was created
                self.print_success_message(profile)
            else:
                self.console.print("✅ Setup completed successfully!")
                self.console.print("💡 Try: tito module start 01")

            # Prompt to join community
            self.prompt_community_registration()

            # Prompt to login to CLI
            # self.prompt_community_login()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return 130
        except Exception as e:
            self.console.print(f"[red]Setup failed: {e}[/red]")
            return 1

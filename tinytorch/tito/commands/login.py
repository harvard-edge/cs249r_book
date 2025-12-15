# tito/commands/login.py
import time
from argparse import ArgumentParser, Namespace
from rich.prompt import Confirm
from tito.commands.base import BaseCommand
from tito.core.auth import AuthReceiver, save_credentials, delete_credentials, ENDPOINTS, is_logged_in
from tito.core.browser import open_url

class LoginCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "login"

    @property
    def description(self) -> str:
        return "Log in to TinyTorch via web browser"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--force", action="store_true", help="Force re-login")

    def run(self, args: Namespace) -> int:
        # Adapted logic from api.py
        if args.force:
            delete_credentials()
            self.console.print("Cleared existing credentials.")

        # Check if already logged in (unless force was used)
        if is_logged_in():
            self.console.print("[green]You are already logged in.[/green]")
            if Confirm.ask("[bold yellow]Do you want to force re-login?[/bold yellow]", default=False):
                delete_credentials()
                self.console.print("Cleared existing credentials. Proceeding with new login...")
            else:
                self.console.print("Login cancelled.")
                return 0

        receiver = AuthReceiver()
        try:
            port = receiver.start()
            
            # Build the target URL with both old (redirect_port) and new (redirect_url) parameters
            # for backward compatibility. The website will use redirect_url if available,
            # otherwise fall back to constructing from redirect_port
            import urllib.parse
            callback_url = receiver.get_redirect_url()
            target_url = f"{ENDPOINTS['cli_login']}?redirect_port={port}&redirect_url={urllib.parse.quote(callback_url)}"
            
            open_url(target_url, self.console, show_manual_fallback=True)
            
            self.console.print()
            from rich.progress import Progress, SpinnerColumn, TextColumn
            
            # Wait for tokens with spinner, but stop the server AFTER exiting progress context
            tokens = None
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Waiting for browser authentication...[/cyan]", total=None)
                
                # Wait for tokens without stopping the server yet
                import time
                start_time = time.time()
                timeout = 120
                while getattr(receiver.server, "auth_data", None) is None:
                    if time.time() - start_time > timeout:
                        break
                    time.sleep(0.25)
                
                tokens = getattr(receiver.server, "auth_data", None)
            
            # Now stop the server AFTER the progress spinner is done
            receiver.stop()
            
            if tokens:
                save_credentials(tokens)
                self.console.print(f"[green]Success! Logged in as {tokens['user_email']}[/green]")
                return 0
            else:
                self.console.print("[red]Login timed out.[/red]")
                return 1
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return 1


class LogoutCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "logout"

    @property
    def description(self) -> str:
        return "Log out of TinyTorch by clearing stored credentials"

    def add_arguments(self, parser: ArgumentParser) -> None:
        pass  # No arguments needed

    def run(self, args: Namespace) -> int:
        try:
            receiver = AuthReceiver()
            port = receiver.start()

            # Use the WSL-aware callback_host from the receiver
            logout_url = f"http://{receiver.callback_host}:{port}/logout"
            
            self.console.print("Opening browser to complete logout...")
            self.console.print(f"[dim]Contacting local auth endpoint: {logout_url}[/dim]")
            open_url(logout_url, self.console, show_manual_fallback=True)
            
            # Wait for logout with spinner, but stop the server AFTER exiting progress context
            from rich.progress import Progress, SpinnerColumn, TextColumn
            logout_confirmed = False
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Waiting for browser confirmation...[/cyan]", total=None)
                
                # Wait for logout signal without stopping the server yet
                import time
                start_time = time.time()
                timeout = 30
                while not getattr(receiver.server, "logout_requested", False):
                    if time.time() - start_time > timeout:
                        break
                    time.sleep(0.25)
                
                logout_confirmed = getattr(receiver.server, "logout_requested", False)
            
            # Now stop the server AFTER the progress spinner is done
            receiver.stop()

            if not logout_confirmed:
                self.console.print("[yellow]Logout confirmation not received (timed out). Please ensure the browser tab opened.[/yellow]")
                self.console.print("[dim]If issues persist, you can manually delete credentials at ~/.tinytorch/credentials.json[/dim]")
                return 1

            delete_credentials()
            self.console.print("[green]✅ Successfully logged out of TinyTorch![/green]")
            return 0
        except Exception as e:
            self.console.print(f"[red]Error during logout: {e}[/red]")
            return 1

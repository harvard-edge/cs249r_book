# tito/commands/login.py
import webbrowser
import time
from argparse import ArgumentParser, Namespace
from rich.prompt import Confirm
from tito.commands.base import BaseCommand
from tito.core.auth import AuthReceiver, save_credentials, delete_credentials, ENDPOINTS, is_logged_in

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
            target_url = f"{ENDPOINTS['cli_login']}?redirect_port={port}"
            self.console.print(f"Opening browser to: [cyan]{target_url}[/cyan]")
            self.console.print("Waiting for authentication...")
            webbrowser.open(target_url)
            tokens = receiver.wait_for_tokens()
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
            # Start local server for logout redirect
            receiver = AuthReceiver()
            port = receiver.start()

            # Open browser to local logout endpoint
            logout_url = f"http://127.0.0.1:{port}/logout"
            self.console.print(f"Opening browser to complete logout...")
            webbrowser.open(logout_url)

            # Give browser time to redirect and close
            time.sleep(2.0)

            # Clean up server
            receiver.stop()

            # Delete local credentials
            delete_credentials()
            self.console.print("[green]Successfully logged out of TinyTorch![/green]")
            return 0
        except Exception as e:
            self.console.print(f"[red]Error during logout: {e}[/red]")
            return 1

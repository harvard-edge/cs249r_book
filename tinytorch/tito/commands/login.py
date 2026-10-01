# tito/commands/login.py
import time
from argparse import ArgumentParser, Namespace
from rich.panel import Panel
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
            self.console.print()
            self.console.print(Panel(
                "[bold yellow]⚠️  This will clear your existing credentials[/bold yellow]",
                title="Warning",
                border_style="yellow"
            ))
            self.console.print()
            if Confirm.ask("[yellow]Force re-login?[/yellow]", default=False):
                delete_credentials()
                self.console.print("Cleared existing credentials. Proceeding with new login...")
            else:
                self.console.print("Login cancelled.")
                return 0

        receiver = AuthReceiver()
        try:
            port = receiver.start()
            
            # The login URL carries redirect_port / redirect_url plus a random
            # `state` nonce that the callback must echo (CSRF protection; see
            # REQUIRE_CALLBACK_STATE in tito/core/auth.py).
            target_url = receiver.get_login_url()
            
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
                timeout = 300
                while getattr(receiver.server, "auth_data", None) is None:
                    if getattr(receiver.server, "save_error", None):
                        break
                    if time.time() - start_time > timeout:
                        break
                    time.sleep(0.25)
                
                tokens = getattr(receiver.server, "auth_data", None)
                save_error = getattr(receiver.server, "save_error", None)
            
            # Now stop the server AFTER the progress spinner is done
            receiver.stop()

            if save_error:
                self.console.print(f"[red]Login succeeded in the browser, but {save_error}[/red]")
                self.console.print("[yellow]Check that the directory is writable (or set TINYTORCH_CREDENTIALS_DIR) and run 'tito community login' again.[/yellow]")
                return 1
            
            if tokens:
                try:
                    save_credentials(tokens)
                except OSError as e:
                    self.console.print(f"[red]Could not save credentials: {e}[/red]")
                    return 1
                self.console.print(f"[green]Success! Logged in as {tokens['user_email']}[/green]")
                self._record_sync_consent()
                self._offer_post_login_sync()
                return 0
            else:
                self.console.print("[red]Login timed out.[/red]")
                self.console.print("\n[yellow]💡 If the browser didn't open or authentication failed, you can try again manually with:[/yellow]")
                self.console.print("   [bold green]tito community login[/bold green]\n")
                return 1
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return 1

    def _record_sync_consent(self) -> None:
        """Show exactly what sync uploads and record whether auto-sync is allowed.

        Logging in is the moment a student opts into the community features,
        so this is where the disclosure is shown. On an interactive terminal
        the student chooses; on a non-interactive one (Git Bash / MinTTY,
        where we cannot prompt) logging in after seeing the disclosure counts
        as consent, which keeps the #1849 fix working for those users. Either
        way it can be changed later (TITO_NO_SYNC=1 or
        'tito community sync --disable-auto').
        """
        from tito.core import runtime
        from tito.core.submission import (
            show_sync_disclosure, set_auto_sync_consent, sync_disabled_by_env, NO_SYNC_ENV,
        )

        self.console.print()
        show_sync_disclosure(self.console)
        if sync_disabled_by_env():
            self.console.print(f"[dim]{NO_SYNC_ENV} is set: automatic sync stays off.[/dim]")
            return
        if runtime.is_interactive():
            allowed = Confirm.ask(
                "[bold yellow]Sync automatically after you complete modules and milestones?[/bold yellow]",
                default=True,
            )
        else:
            allowed = True
            self.console.print("[dim]Automatic sync enabled. Turn it off with 'tito community sync --disable-auto'.[/dim]")
        try:
            set_auto_sync_consent(allowed)
        except OSError as e:
            self.console.print(f"[yellow]Could not save your sync preference: {e}[/yellow]")

    def _offer_post_login_sync(self) -> None:
        """Offer to upload progress completed *before* logging in.

        Closes the complete-then-login gap: without this, modules finished while
        logged out never reached the dashboard, because automatic sync only ran
        during 'tito module complete' (#1849). Only fires when local progress
        actually exists, so a fresh login stays quiet.
        """
        import json
        from tito.core.submission import auto_sync_after_completion
        from tito.core.modules import get_module_mapping

        progress_file = self.config.project_root / ".tito" / "progress.json"
        try:
            data = json.loads(progress_file.read_text(encoding='utf-8')) if progress_file.exists() else {}
            completed = data.get("completed_modules", [])
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            completed = []

        if not completed:
            return  # nothing completed yet; sync will happen as modules complete

        self.console.print()
        auto_sync_after_completion(
            self.config,
            self.console,
            total_modules=len(get_module_mapping()),
            prompt=f"You have {len(completed)} completed module(s) saved locally. Upload them now?",
        )


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
            receiver = AuthReceiver(mode="logout")
            receiver.start()

            # WSL-aware host, plus the per-run nonce: /logout ignores any
            # request that does not carry it.
            logout_url = receiver.get_logout_url()
            
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
                timeout = 60
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

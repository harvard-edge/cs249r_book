"""
Handles data aggregation and submission to the Supabase Edge Function.

This version is refactored into a class-based handler that integrates
with the TinyTorch CLI's config and console objects, using only standard libraries.
"""
import json
import os
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import certifi

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich import box

# Local import for auth handler
from . import auth
from . import runtime
from .config import CLIConfig

# Where users should report sync problems (kept in one place).
ISSUE_URL = "https://github.com/harvard-edge/cs249r_book/issues/1849"

# ---------------------------------------------------------------------------
# Sync consent and opt-out
# ---------------------------------------------------------------------------
# Automatic sync (after `tito module complete`, a milestone, or login) uploads
# progress to the TinyTorch website. It is governed by:
#   * TITO_NO_SYNC=1           -- never sync automatically (explicit
#                                 `tito community sync` still works).
#   * ~/.tinytorch/sync.json   -- {"auto_sync": true|false}, the persisted
#                                 consent, stored next to credentials.json.
# On a non-interactive terminal (no way to ask), automatic sync runs only when
# the stored consent is true. Consent is recorded at `tito community login`
# after the disclosure below is shown, and can be changed with
# `tito community sync --enable-auto / --disable-auto`.
NO_SYNC_ENV = "TITO_NO_SYNC"
SYNC_SETTINGS_FILE_NAME = "sync.json"

SYNC_DISCLOSURE = (
    "When you sync, tito uploads to the TinyTorch website "
    "(tinytorch.netlify.app / Supabase):\n"
    "  • your account email (as user_id)\n"
    "  • which modules you completed, and when\n"
    "  • which milestones you unlocked/completed, and when\n"
    "  • your completion percentage and current streak\n"
    "No code, notebooks, or test output is uploaded.\n"
    f"Opt out of automatic sync: set {NO_SYNC_ENV}=1, or run "
    "'tito community sync --disable-auto'."
)


def sync_disabled_by_env() -> bool:
    """True when the user has opted out of automatic sync via TITO_NO_SYNC."""
    return os.environ.get(NO_SYNC_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _sync_settings_path() -> Path:
    return auth._credentials_dir() / SYNC_SETTINGS_FILE_NAME


def load_sync_settings() -> Dict[str, Any]:
    p = _sync_settings_path()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _save_sync_settings(data: Dict[str, Any]) -> None:
    auth._ensure_dir()
    p = _sync_settings_path()
    tmp = p.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(p))


def get_auto_sync_consent() -> Optional[bool]:
    """Stored consent for automatic sync: True, False, or None (never asked)."""
    value = load_sync_settings().get("auto_sync")
    return value if isinstance(value, bool) else None


def set_auto_sync_consent(allowed: bool) -> None:
    data = load_sync_settings()
    data["auto_sync"] = bool(allowed)
    _save_sync_settings(data)


def show_sync_disclosure(console: Console) -> None:
    """Print exactly what a sync sends, and how to opt out."""
    from rich.panel import Panel
    console.print(Panel(SYNC_DISCLOSURE, title="What 'sync' shares", border_style="cyan"))


def _milestone_registry() -> Dict[str, Dict[str, Any]]:
    """The milestone registry the CLI runs from (single source of truth)."""
    from ..commands.milestone import MILESTONE_SCRIPTS  # lazy: avoid import cycle
    return MILESTONE_SCRIPTS


class SubmissionError(Exception):
    """Custom exception for submission-related errors."""
    pass


@dataclass
class SyncResult:
    """Outcome of a progress-sync attempt.

    ``ok``        -- the request succeeded AND the server confirmed our progress
                     was persisted (this is what callers should gate success on).
    ``accepted``  -- the server returned 2xx (it received the request), even if
                     it did not confirm persistence.
    ``synced_modules`` -- the count the server reported back (``None`` if absent).
    ``sent_modules``   -- how many completed modules we uploaded.
    ``error``     -- human-readable failure reason, if any.

    The distinction between ``ok`` and ``accepted`` exists so a server that
    replies 2xx but reports zero/None synced modules is surfaced as a WARNING
    rather than masked as success. Hiding that was the bug that made dashboard
    desync invisible (issue #1849).
    """
    ok: bool
    accepted: bool = False
    synced_modules: Optional[int] = None
    sent_modules: int = 0
    error: Optional[str] = None

    def __bool__(self) -> bool:  # lets legacy ``if result:`` callers keep working
        return self.ok

class SubmissionHandler:
    """
    Handles assembling progress data and submitting it to a remote server.
    """

    def __init__(self, config: CLIConfig, console: Console):
        """
        Initialize the handler with CLI config and console.

        Args:
            config: The CLI configuration object.
            console: The rich console for output.
        """
        self.config = config
        self.console = console
        self.auth_handler = auth  # Using the auth module directly for now

        # TODO: In the future, the API endpoint could be made configurable via CLIConfig
        self.edge_function_url = "https://zrvmjrxhokwwmjacyhpq.supabase.co/functions/v1/upload-progress"

        # Derive paths from the project root in config
        self.tito_dir = self.config.project_root / ".tito"
        self.progress_file = self.tito_dir / "progress.json"
        self.milestones_file = self.tito_dir / "milestones.json"
        self.config_file = self.tito_dir / "config.json" # Though config is passed via CLIConfig

    def _read_json_safe(self, path: Path) -> Dict[str, Any]:
        """Helper to read JSON files safely."""
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.console.print(f"[yellow]Warning: Could not read {path}: {e}[/yellow]")
            return {}

    def _format_milestones(self, local_data: Dict) -> list:
        """Transforms local milestone storage format to the API array format."""
        unlocked = local_data.get("unlocked_milestones", [])
        completed = local_data.get("completed_milestones", [])
        unlock_dates = local_data.get("unlock_dates", {})
        completion_dates = local_data.get("completion_dates", {})

        # Names come from the milestone registry so new milestones (e.g. 07)
        # are never reported as "Milestone NN".
        milestone_names = {
            m_id: info.get("name", f"Milestone {m_id}")
            for m_id, info in _milestone_registry().items()
        }

        formatted = []
        for m_id in unlocked:
            formatted.append({
                "id": m_id,
                "name": milestone_names.get(m_id, f"Milestone {m_id}"),
                "unlocked_at": unlock_dates.get(m_id),
                "completed": m_id in completed,
                "completed_at": completion_dates.get(m_id)
            })
        return formatted

    def assemble_payload(self, total_modules: int = 20) -> Dict[str, Any]:
        """
        Reads distinct local files and assembles the Unified Payload.
        """
        progress_data = self._read_json_safe(self.progress_file)
        # Normalize through the milestone ledger so only milestones complete
        # under the per-part rule are reported (legacy files included).
        from .milestone_tracker import normalize
        milestone_data = normalize(self._read_json_safe(self.milestones_file))

        completed_modules = progress_data.get("completed_modules", [])

        payload = {
            # user_id will be derived from the auth token on the backend,
            # but we can send a placeholder if needed for schema validation.
            "user_id": self.auth_handler.get_user_email() or "anonymous", # Using get_user_email from auth module
            "timestamp": progress_data.get("last_updated", ""),
            "version": "1.0",
            "module_progress": {
                "total_modules": total_modules,
                "completed_count": len(completed_modules),
                "completed_modules": completed_modules,
                "completion_dates": progress_data.get("completion_dates", {}),
                "completion_percentage": (len(completed_modules) / total_modules) * 100 if total_modules > 0 else 0,
            },
            "milestone_progress": {
                "total_milestones": len(_milestone_registry()),
                "unlocked_count": milestone_data.get("total_unlocked", 0),
                "unlocked_milestones": self._format_milestones(milestone_data)
            },
            "statistics": {
                "current_streak_days": progress_data.get("streak", 0)
            }
        }
        return payload

    def sync_progress(self, total_modules: int = 20, is_retry: bool = False) -> SyncResult:
        """Assemble local progress and upload it to the TinyTorch backend.

        Returns a :class:`SyncResult`. Callers should check ``result.ok``;
        ``SyncResult.__bool__`` returns ``result.ok`` so ``if result:`` still
        works for legacy call sites.
        """
        token = self.auth_handler.get_token()
        if not token:
            self.console.print("❌ [bold red]You are not logged in.[/bold red] Please run 'tito community login' first.")
            return SyncResult(ok=False, error="not logged in")

        if not is_retry:
            self.console.print("📦 Assembling local progress...")

        try:
            payload = self.assemble_payload(total_modules=total_modules)
            if not is_retry:
                self.console.print("Submitting payload:")
                table = Table(show_header=False, box=box.MINIMAL, padding=(0, 1))
                table.add_column("Field", style="dim")
                table.add_column("Value")

                table.add_row("User ID", payload['user_id'])
                table.add_row("Timestamp", payload['timestamp'])
                table.add_row("Version", payload['version'])

                table.add_row("")
                table.add_row("[bold]Module Progress[/bold]")
                table.add_row("  Total Modules", str(payload['module_progress']['total_modules']))
                table.add_row("  Completed", str(payload['module_progress']['completed_count']))
                table.add_row("  Completed Modules", ", ".join(payload['module_progress']['completed_modules']))
                table.add_row("  Completion %", f"{payload['module_progress']['completion_percentage']:.2f}%")

                table.add_row("")
                table.add_row("[bold]Milestone Progress[/bold]")
                table.add_row("  Total Milestones", str(payload['milestone_progress']['total_milestones']))
                table.add_row("  Unlocked", str(payload['milestone_progress']['unlocked_count']))

                table.add_row("")
                table.add_row("[bold]Statistics[/bold]")
                table.add_row("  Current Streak", str(payload['statistics']['current_streak_days']))

                self.console.print(table)
        except Exception as e:
            self.console.print(f"❌ [red]Error assembling payload: {e}[/red]")
            return SyncResult(ok=False, error=f"payload assembly failed: {e}")

        if not is_retry:
            self.console.print("🚀 Syncing with TinyTorch Cloud...")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            self.edge_function_url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method="POST"
        )

        # Create SSL context with certifi certificates for macOS compatibility
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        try:
            with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
                sent = payload['module_progress']['completed_count']
                if 200 <= response.status < 300:
                    resp_body = json.loads(response.read().decode('utf-8'))
                    synced = resp_body.get('synced_modules')
                    return self._interpret_2xx(synced, sent)
                else:
                    self.console.print(f"⚠️ Server returned status: {response.status}")
                    # Try to read error message from response body
                    try:
                        error_resp = json.loads(response.read().decode('utf-8'))
                        detail = error_resp.get('error', 'No message provided.')
                    except json.JSONDecodeError:
                        detail = response.read().decode('utf-8')[:200] + "..."
                    self.console.print(f"   [dim red]Error details: {detail}[/dim red]")
                    return SyncResult(ok=False, accepted=False, sent_modules=sent,
                                      error=f"HTTP {response.status}: {detail}")

        except urllib.error.HTTPError as e:
            if e.code == 401 and not is_retry:
                self.console.print("🔑 Token expired. Attempting to refresh...")
                new_token = self.auth_handler.refresh_token(self.console)
                if new_token:
                    self.console.print("✅ Token refreshed successfully. Retrying submission...")
                    return self.sync_progress(total_modules=total_modules, is_retry=True)
                else:
                    self.console.print("❌ [bold red]Token refresh failed.[/bold red]")
                    self.console.print("   Run 'tito community login --force' to refresh.")
                    return SyncResult(ok=False, error="token refresh failed")
            elif e.code == 401 and is_retry:
                self.console.print("❌ [bold red]Unauthorized.[/bold red] Your session may have expired.")
                self.console.print("   Run 'tito community login --force' to refresh.")
                return SyncResult(ok=False, error="unauthorized after refresh")
            else:
                self.console.print(f"❌ [red]Upload failed (HTTP {e.code}): {e.reason}[/red]")
                error_body = ""
                try: # Attempt to read error body if available
                    error_body = e.read().decode('utf-8')
                    error_json = json.loads(error_body)
                    self.console.print(f"   [dim red]Error details: {error_json.get('error', 'No message provided.')}[/dim red]")
                except (json.JSONDecodeError, Exception):
                    if error_body:
                        self.console.print(f"   [dim red]Error details: {error_body[:200]}...[/dim red]")
            return SyncResult(ok=False, error=f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            self.console.print(f"❌ [red]Network error:[/red] Could not connect to the server.")
            self.console.print(f"   [dim]{e.reason}[/dim]")
            return SyncResult(ok=False, error=f"network error: {e.reason}")
        except TimeoutError:
            self.console.print("❌ [red]Network error:[/red] Connection timed out.")
            return SyncResult(ok=False, error="connection timed out")

    def _interpret_2xx(self, synced: Optional[int], sent: int) -> SyncResult:
        """Turn a 2xx response into an honest result.

        The server returns ``synced_modules``. A 2xx with ``synced_modules``
        null/0 while we sent completed modules means the request was *accepted*
        but the backend did not confirm anything was persisted -- the exact
        failure that silently desynced dashboards (#1849). Surface it as a
        warning, not a green success.
        """
        if sent == 0:
            self.console.print("✅ [green]Already up to date[/green] — no completed modules to sync yet.")
            return SyncResult(ok=True, accepted=True, synced_modules=0, sent_modules=0)

        confirmed = isinstance(synced, int) and synced > 0
        if confirmed:
            self.console.print("✅ [bold green]Sync successful![/bold green]")
            self.console.print(f"   Modules synced: {synced}")
            return SyncResult(ok=True, accepted=True, synced_modules=synced, sent_modules=sent)

        # Accepted (2xx) but the server did not confirm persistence.
        self.console.print("⚠️  [yellow]Sync request accepted, but the server did not confirm your progress was saved.[/yellow]")
        self.console.print(f"   [dim]Sent {sent} completed module(s); server reported synced_modules={synced!r}.[/dim]")
        self.console.print("   [dim]Your dashboard may still show old progress. Re-run [bold]tito community sync[/bold];[/dim]")
        self.console.print(f"   [dim]if it persists, please report it at {ISSUE_URL}[/dim]")
        return SyncResult(ok=False, accepted=True, synced_modules=synced, sent_modules=sent,
                          error="server accepted but did not confirm persistence")


def auto_sync_after_completion(config: CLIConfig, console: Console, *,
                               total_modules: int = 20,
                               prompt: str = "Sync your progress to the TinyTorch website?") -> Optional[SyncResult]:
    """Shared auto-sync used after a module or milestone completes (and on login).

    This is the single decision point for *whether* an automatic sync runs, so
    the rule lives in one place instead of being duplicated (and drifting)
    across the module, milestone, and login commands:

    - In CI / automation: do nothing (never sync automatically there).
    - TITO_NO_SYNC=1: do nothing (user opt-out).
    - Not logged in: print a hint, do nothing.
    - Interactive terminal: ask first (default yes), then sync. The first time,
      show exactly what is uploaded.
    - Non-interactive but a real user (Git Bash / MinTTY / IDE terminal where
      ``stdin.isatty()`` is False): sync without prompting ONLY if the user
      previously consented (recorded at `tito community login`). Skipping
      silently here was the #1849 bug, so when we skip we say how to sync.

    Returns the :class:`SyncResult` when a sync was attempted, else ``None``.
    """
    if runtime.is_ci():
        return None

    if sync_disabled_by_env():
        console.print(f"[dim]Automatic sync is off ({NO_SYNC_ENV} is set). Run 'tito community sync' to upload manually.[/dim]")
        return None

    if not auth.is_logged_in():
        console.print("[dim]💡 Run 'tito community login' to sync your progress to the TinyTorch website.[/dim]")
        return None

    consent = get_auto_sync_consent()

    if consent is False:
        # An explicit 'tito community sync --disable-auto' means no prompt after
        # every completion either; manual sync still works.
        console.print("[dim]Automatic sync is off. Run 'tito community sync' to upload manually.[/dim]")
        return None

    if runtime.is_interactive() and consent is not True:
        if consent is None:
            # Never recorded a choice on this machine (e.g. logged in with an
            # older tito): show what is sent before asking. This path does not
            # write anything; consent is recorded at login or via
            # 'tito community sync --enable-auto/--disable-auto'.
            show_sync_disclosure(console)
        if not Confirm.ask(f"[bold yellow]{prompt}[/bold yellow]", default=True):
            console.print("[dim]Skipped. Run 'tito community sync' anytime to upload your progress.[/dim]")
            return None
    else:
        if consent is not True:
            console.print(
                "[dim]Not syncing automatically (no saved consent on this machine). "
                "Run 'tito community sync' to upload, or 'tito community sync --enable-auto' "
                "to sync automatically from now on.[/dim]"
            )
            return None
        # No usable TTY to prompt on, but the user opted in -> sync quietly.
        console.print("[dim]Syncing your progress to the TinyTorch website…[/dim]")

    return SubmissionHandler(config, console).sync_progress(total_modules=total_modules)

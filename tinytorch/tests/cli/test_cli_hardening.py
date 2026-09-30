"""
Regression tests for the tito CLI release hardening pass.

Unit-level only: no real sockets, no network, no login, no student-module
changes. Each test names the defect it guards.
"""

import os
import ssl
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console

from tito.core import auth
from tito.core import submission
from tito.core.config import CLIConfig
from tito.core.submission import SubmissionHandler, SyncResult, auto_sync_after_completion


def _quiet_console():
    return Console(file=open(os.devnull, "w"), force_terminal=False)


@pytest.fixture
def creds_dir(tmp_path, monkeypatch):
    """Point credentials + sync settings at a temp dir (never the real ~)."""
    d = tmp_path / "tinytorch-home"
    monkeypatch.setattr(auth, "CREDENTIALS_DIR", str(d))
    return d


# ---------------------------------------------------------------------------
# 1. Local auth callback server: loopback bind + state nonce
# ---------------------------------------------------------------------------

class _FakeServer:
    def __init__(self, mode="login", expected_state="s3cret"):
        self.mode = mode
        self.expected_state = expected_state
        self.auth_data = None
        self.logout_requested = False
        self.save_error = None


def _call_handler(path, server):
    """Drive CallbackHandler.do_GET without any socket."""
    h = auth.CallbackHandler.__new__(auth.CallbackHandler)
    h.path = path
    h.server = server
    h.sent = {"status": None, "headers": {}}

    def send_error(code, message=None):
        h.sent["status"] = code

    def send_response(code, message=None):
        h.sent["status"] = code

    def send_header(k, v):
        h.sent["headers"][k] = v

    h.send_error = send_error
    h.send_response = send_response
    h.send_header = send_header
    h.end_headers = lambda: None
    h.do_GET()
    return h.sent


TOKENS = "access_token=AT&refresh_token=RT&email=a%40b.edu"


def test_callback_with_wrong_state_is_rejected(creds_dir):
    server = _FakeServer()
    sent = _call_handler(f"/callback?{TOKENS}&state=attacker", server)
    assert sent["status"] == 403
    assert server.auth_data is None
    assert not (creds_dir / "credentials.json").exists()


def test_callback_with_matching_state_is_accepted(creds_dir):
    server = _FakeServer()
    sent = _call_handler(f"/callback?{TOKENS}&state=s3cret", server)
    assert sent["status"] == 302
    assert server.auth_data["access_token"] == "AT"
    assert (creds_dir / "credentials.json").exists()


def test_callback_without_state_rejected_when_required(creds_dir, monkeypatch):
    monkeypatch.setenv("TITO_AUTH_REQUIRE_STATE", "1")
    server = _FakeServer()
    sent = _call_handler(f"/callback?{TOKENS}", server)
    assert sent["status"] == 403
    assert server.auth_data is None


def test_callback_accepted_only_once(creds_dir):
    server = _FakeServer()
    _call_handler(f"/callback?{TOKENS}&state=s3cret", server)
    sent = _call_handler("/callback?access_token=EVIL&refresh_token=EVIL&state=s3cret", server)
    assert sent["status"] == 409
    assert server.auth_data["access_token"] == "AT"


def test_callback_save_failure_is_surfaced(creds_dir, monkeypatch):
    def boom(data):
        raise PermissionError("read-only")
    monkeypatch.setattr(auth, "save_credentials", boom)
    server = _FakeServer()
    sent = _call_handler(f"/callback?{TOKENS}&state=s3cret", server)
    assert sent["status"] == 500
    assert server.auth_data is None
    assert "read-only" in server.save_error


def test_logout_requires_logout_mode_and_state():
    login_server = _FakeServer(mode="login")
    assert _call_handler("/logout?state=s3cret", login_server)["status"] == 403
    assert login_server.logout_requested is False

    logout_server = _FakeServer(mode="logout")
    assert _call_handler("/logout", logout_server)["status"] == 403
    assert _call_handler("/logout?state=nope", logout_server)["status"] == 403
    assert logout_server.logout_requested is False

    assert _call_handler("/logout?state=s3cret", logout_server)["status"] == 302
    assert logout_server.logout_requested is True


def test_callback_not_served_in_logout_mode(creds_dir):
    server = _FakeServer(mode="logout")
    assert _call_handler(f"/callback?{TOKENS}&state=s3cret", server)["status"] == 404
    assert server.auth_data is None


def test_bind_host_is_loopback_except_wsl(monkeypatch):
    monkeypatch.setattr(auth, "_is_wsl", lambda: False)
    assert auth._server_bind_host() == "127.0.0.1"
    monkeypatch.setattr(auth, "_is_wsl", lambda: True)
    assert auth._server_bind_host() == "0.0.0.0"


def test_login_url_carries_fresh_state():
    r1, r2 = auth.AuthReceiver(), auth.AuthReceiver()
    r1.port = r2.port = 54321
    q = parse_qs(urlparse(r1.get_login_url()).query)
    assert q["redirect_port"] == ["54321"]
    assert q["state"] == [r1.state]
    assert len(r1.state) >= 32 and r1.state != r2.state

    r1_logout = auth.AuthReceiver(mode="logout")
    r1_logout.port = 1
    assert parse_qs(urlparse(r1_logout.get_logout_url()).query)["state"] == [r1_logout.state]


# ---------------------------------------------------------------------------
# 2. Credential file is owner-only from creation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_credentials_file_mode_is_0600(creds_dir, monkeypatch):
    opened = []
    real_open = os.open

    def spy_open(path, flags, mode=0o777, *a, **k):
        opened.append((str(path), mode))
        return real_open(path, flags, mode, *a, **k)

    monkeypatch.setattr(auth.os, "open", spy_open)
    old = os.umask(0)
    try:
        auth.save_credentials({"access_token": "x", "refresh_token": "y"})
    finally:
        os.umask(old)

    p = creds_dir / "credentials.json"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert opened and all(mode == 0o600 for _, mode in opened)
    assert not list(creds_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 3. Update check never disables TLS verification
# ---------------------------------------------------------------------------

def _update_cmd():
    from tito.commands.system.update import UpdateCommand
    cmd = UpdateCommand(CLIConfig.from_project_root(Path.cwd()))
    cmd.console = _quiet_console()
    return cmd


def test_update_check_verifies_tls(monkeypatch):
    import urllib.request
    seen = {}

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"[]"

    def fake_urlopen(req, timeout=None, context=None):
        seen["ctx"] = context
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _update_cmd()._get_latest_version_urllib()
    ctx = seen["ctx"]
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_update_check_failure_reports_and_returns_none(monkeypatch):
    import urllib.request

    def fail(*a, **k):
        raise ssl.SSLCertVerificationError("bad cert")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    cmd = _update_cmd()
    cmd.console = Console(record=True, file=open(os.devnull, "w"))
    assert cmd._get_latest_version_urllib() == (None, None)
    assert "Couldn't check for updates" in cmd.console.export_text()


def test_update_module_has_no_cert_none():
    src = (Path(__file__).parents[2] / "tito" / "commands" / "system" / "update.py").read_text(encoding="utf-8")
    assert "CERT_NONE" not in src
    assert "check_hostname = False" not in src


# ---------------------------------------------------------------------------
# 4. Sync: opt-out, consent on non-TTY, milestone registry
# ---------------------------------------------------------------------------

def _sync_spy(monkeypatch):
    ran = {}

    def fake_sync(self, total_modules=20, is_retry=False):
        ran["ran"] = True
        return SyncResult(ok=True, accepted=True, synced_modules=1, sent_modules=1)

    monkeypatch.setattr(SubmissionHandler, "sync_progress", fake_sync)
    return ran


@pytest.fixture
def logged_in_user(monkeypatch, creds_dir):
    monkeypatch.delenv(submission.NO_SYNC_ENV, raising=False)
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: False)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: True)


def test_sync_respects_tito_no_sync(monkeypatch, logged_in_user):
    monkeypatch.setenv("TITO_NO_SYNC", "1")
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: False)
    submission.set_auto_sync_consent(True)
    ran = _sync_spy(monkeypatch)
    assert auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), _quiet_console()) is None
    assert "ran" not in ran


def test_non_tty_without_consent_does_not_upload(monkeypatch, logged_in_user):
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: False)
    ran = _sync_spy(monkeypatch)
    assert submission.get_auto_sync_consent() is None
    assert auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), _quiet_console()) is None
    assert "ran" not in ran

    submission.set_auto_sync_consent(False)
    assert auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), _quiet_console()) is None
    assert "ran" not in ran


def test_non_tty_with_consent_uploads(monkeypatch, logged_in_user, creds_dir):
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: False)
    submission.set_auto_sync_consent(True)
    assert (creds_dir / "sync.json").exists()
    ran = _sync_spy(monkeypatch)
    result = auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), _quiet_console())
    assert ran.get("ran") is True and result.ok


def test_interactive_prompt_does_not_write_consent(monkeypatch, logged_in_user, creds_dir):
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: True)
    monkeypatch.setattr(submission.Confirm, "ask", lambda *a, **k: True)
    ran = _sync_spy(monkeypatch)
    auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), _quiet_console())
    assert ran.get("ran") is True
    assert not (creds_dir / "sync.json").exists()


def test_payload_milestones_follow_registry(tmp_path, monkeypatch):
    from tito.commands.milestone import MILESTONE_SCRIPTS
    tito_dir = tmp_path / ".tito"
    tito_dir.mkdir()
    (tito_dir / "milestones.json").write_text(
        '{"unlocked_milestones": ["07"], "completed_milestones": ["07"], "total_unlocked": 1}',
        encoding="utf-8",
    )
    monkeypatch.setattr(submission.auth, "get_user_email", lambda: "a@b.edu")
    handler = SubmissionHandler(CLIConfig.from_project_root(tmp_path), _quiet_console())
    payload = handler.assemble_payload(total_modules=20)
    mp = payload["milestone_progress"]
    assert mp["total_milestones"] == len(MILESTONE_SCRIPTS) == 7
    assert mp["unlocked_milestones"][0]["name"] == MILESTONE_SCRIPTS["07"]["name"]


# ---------------------------------------------------------------------------
# 5. `tito system health` exit code reflects reported issues
# ---------------------------------------------------------------------------

def test_health_returns_nonzero_when_issues(monkeypatch):
    from tito.commands.system.health import HealthCommand
    cmd = HealthCommand(CLIConfig.from_project_root(Path.cwd()))
    cmd.console = _quiet_console()
    monkeypatch.setattr(cmd, "_check_jupyter_kernel",
                        lambda: ("[red]❌ Missing[/red]", "no tinytorch kernel"))
    monkeypatch.setattr(cmd, "_get_kernel_python", lambda: None)
    assert cmd.run(SimpleNamespace()) == 1


# ---------------------------------------------------------------------------
# 6. Version falls back to installed metadata when pyproject.toml is absent
# ---------------------------------------------------------------------------

def test_tinytorch_version_falls_back_to_metadata(monkeypatch):
    import importlib.metadata
    import tinytorch
    monkeypatch.setattr(tinytorch, "_version_from_pyproject", lambda p: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.8.7")
    assert tinytorch._get_version() == "9.8.7"


def test_tito_version_falls_back_to_metadata(monkeypatch):
    import importlib.metadata
    from tito import main as tito_main
    monkeypatch.setattr(tito_main, "_version_from_pyproject", lambda p: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.8.7")
    assert tito_main._get_version() == "9.8.7"


def test_pyproject_parser_ignores_other_projects(tmp_path):
    from tito import main as tito_main
    other = tmp_path / "pyproject.toml"
    other.write_text('[tool.x]\nversion = "1"\n[project]\nname = "other"\nversion = "2"\n', encoding="utf-8")
    assert tito_main._version_from_pyproject(other) is None
    other.write_text('[tool.x]\nversion = "1"\n[project]\nname="tinytorch"\nversion = "0.3.0"\n', encoding="utf-8")
    assert tito_main._version_from_pyproject(other) == "0.3.0"


def test_tito_and_package_versions_agree():
    import tinytorch
    from tito import main as tito_main
    assert tito_main._get_version() == tinytorch.__version__


# ---------------------------------------------------------------------------
# 7. `tito module status` with zero modules (installed wheel, no src/)
# ---------------------------------------------------------------------------

def test_module_status_with_zero_modules(monkeypatch, tmp_path):
    from tito.commands.module import workflow
    monkeypatch.setattr(workflow, "get_module_mapping", lambda: {})
    cmd = workflow.ModuleWorkflowCommand(CLIConfig.from_project_root(tmp_path))
    cmd.console = Console(record=True, file=open(os.devnull, "w"))
    assert cmd.show_status() == 1
    assert "No TinyTorch modules found" in cmd.console.export_text()


# ---------------------------------------------------------------------------
# 8. venv creation never goes through a shell
# ---------------------------------------------------------------------------

def test_setup_venv_uses_argv_under_rosetta(monkeypatch, tmp_path):
    from tito.commands import setup as setup_mod
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append((cmd, k))
        if cmd[:2] == ["sysctl", "-n"]:
            return SimpleNamespace(returncode=0, stdout="Apple M2", stderr="")
        return SimpleNamespace(returncode=0, stdout="arm64", stderr="")

    monkeypatch.setattr(setup_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup_mod.subprocess, "run", fake_run)

    root = tmp_path / "dir with space; echo pwned"
    root.mkdir()
    cmd = setup_mod.SetupCommand(CLIConfig.from_project_root(root))
    cmd.console = _quiet_console()
    assert cmd.create_virtual_environment() is True
    venv_calls = [c for c, k in calls if isinstance(c, list) and "venv" in c]
    assert venv_calls, calls
    assert venv_calls[0][:2] == ["arch", "-arm64"]
    assert venv_calls[0][-1] == str(root / ".venv")
    assert all(not k.get("shell") for _, k in calls)

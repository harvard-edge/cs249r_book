"""
Regression tests for community progress sync (issue #1849).

These guard the bug where the TinyTorch dashboard did not reflect CLI progress:

1. Automatic sync was silently skipped on any non-TTY shell (Git Bash / MinTTY
   on Windows, IDE terminals), because the trigger gated on
   ``sys.stdin.isatty()``. The fix decouples "should we sync" from "should we
   prompt": a logged-in, non-CI user always syncs; the prompt is only skipped
   when there is no usable TTY.
2. A 2xx response with ``synced_modules`` null/0 was reported as success using
   the local count, hiding backend failures. The fix reports an honest
   accepted-but-unconfirmed warning instead.
3. There was no standalone resync command and login did not sync, so progress
   completed before logging in never reached the dashboard.
"""

import sys
from pathlib import Path

import pytest

# Add tito to path (CI exports the package first; these tests only need tito).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console

from tito.core import runtime
from tito.core import submission
from tito.core.submission import (
    SyncResult,
    SubmissionHandler,
    auto_sync_after_completion,
)
from tito.core.config import CLIConfig


@pytest.fixture
def handler():
    config = CLIConfig.from_project_root(Path.cwd())
    return SubmissionHandler(config, Console())


# ---------------------------------------------------------------------------
# runtime: CI vs interactivity are independent
# ---------------------------------------------------------------------------

def test_is_ci_detects_env(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert runtime.is_ci() is False
    monkeypatch.setenv("CI", "true")
    assert runtime.is_ci() is True


def test_is_interactive_false_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert runtime.is_interactive() is False


def test_is_interactive_false_without_tty(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    class _NoTTY:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.setattr(sys, "stdout", _NoTTY())
    assert runtime.is_interactive() is False


# ---------------------------------------------------------------------------
# auto_sync_after_completion: the trigger decision
# ---------------------------------------------------------------------------

def test_auto_sync_runs_when_non_interactive_but_logged_in(monkeypatch):
    """THE core regression: a logged-in user on a non-TTY shell must still sync.

    Previously this path returned early on ``not sys.stdin.isatty()`` and never
    uploaded -- the silent skip behind the dashboard desync.
    """
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: False)
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: False)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: True)

    calls = {}

    def fake_sync(self, total_modules=20, is_retry=False):
        calls["ran"] = True
        return SyncResult(ok=True, accepted=True, synced_modules=2, sent_modules=2)

    monkeypatch.setattr(SubmissionHandler, "sync_progress", fake_sync)

    result = auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), Console())
    assert calls.get("ran") is True
    assert result is not None and result.ok


def test_auto_sync_prompts_when_interactive(monkeypatch):
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: False)
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: True)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: True)

    asked = {}
    monkeypatch.setattr(submission.Confirm, "ask", lambda *a, **k: asked.setdefault("asked", True) or True)

    ran = {}
    monkeypatch.setattr(
        SubmissionHandler, "sync_progress",
        lambda self, total_modules=20, is_retry=False: ran.setdefault("ran", True) or SyncResult(ok=True),
    )

    auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), Console())
    assert asked.get("asked") is True
    assert ran.get("ran") is True


def test_auto_sync_interactive_decline_does_not_sync(monkeypatch):
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: False)
    monkeypatch.setattr(submission.runtime, "is_interactive", lambda: True)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: True)
    monkeypatch.setattr(submission.Confirm, "ask", lambda *a, **k: False)

    ran = {}
    monkeypatch.setattr(
        SubmissionHandler, "sync_progress",
        lambda self, total_modules=20, is_retry=False: ran.setdefault("ran", True),
    )

    result = auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), Console())
    assert "ran" not in ran
    assert result is None


def test_auto_sync_skipped_in_ci(monkeypatch):
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: True)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: True)

    ran = {}
    monkeypatch.setattr(
        SubmissionHandler, "sync_progress",
        lambda self, total_modules=20, is_retry=False: ran.setdefault("ran", True),
    )

    result = auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), Console())
    assert "ran" not in ran
    assert result is None


def test_auto_sync_hint_when_not_logged_in(monkeypatch):
    monkeypatch.setattr(submission.runtime, "is_ci", lambda: False)
    monkeypatch.setattr(submission.auth, "is_logged_in", lambda: False)

    ran = {}
    monkeypatch.setattr(
        SubmissionHandler, "sync_progress",
        lambda self, total_modules=20, is_retry=False: ran.setdefault("ran", True),
    )

    result = auto_sync_after_completion(CLIConfig.from_project_root(Path.cwd()), Console())
    assert "ran" not in ran
    assert result is None


# ---------------------------------------------------------------------------
# honest 2xx interpretation
# ---------------------------------------------------------------------------

def test_2xx_confirmed_is_ok(handler):
    result = handler._interpret_2xx(synced=2, sent=2)
    assert result.ok is True
    assert result.accepted is True
    assert result.synced_modules == 2


def test_2xx_null_synced_is_unconfirmed_warning(handler):
    """2xx but server reports nothing persisted while we sent modules -> NOT ok."""
    result = handler._interpret_2xx(synced=None, sent=2)
    assert result.ok is False
    assert result.accepted is True
    assert result.sent_modules == 2


def test_2xx_zero_synced_is_unconfirmed_warning(handler):
    result = handler._interpret_2xx(synced=0, sent=3)
    assert result.ok is False
    assert result.accepted is True


def test_2xx_nothing_to_sync_is_ok(handler):
    result = handler._interpret_2xx(synced=None, sent=0)
    assert result.ok is True
    assert result.accepted is True


def test_sync_result_truthiness():
    assert bool(SyncResult(ok=True)) is True
    assert bool(SyncResult(ok=False, accepted=True)) is False


# ---------------------------------------------------------------------------
# the recovery command exists and is wired
# ---------------------------------------------------------------------------

def test_community_sync_subcommand_registered():
    import argparse
    from tito.commands.community import CommunityCommand

    cmd = CommunityCommand(CLIConfig.from_project_root(Path.cwd()))
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    args = parser.parse_args(["sync"])
    assert args.community_command == "sync"


def test_community_sync_requires_login(monkeypatch):
    from tito.commands.community import CommunityCommand
    import argparse

    monkeypatch.setattr("tito.commands.community.auth.is_logged_in", lambda: False)
    cmd = CommunityCommand(CLIConfig.from_project_root(Path.cwd()))
    ns = argparse.Namespace(community_command="sync")
    assert cmd.run(ns) == 1  # not logged in -> nonzero exit, no upload attempted

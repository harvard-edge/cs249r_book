"""Phase 0 smoke tests.

These assert the package imports, exposes a version, and the Typer app's
``--version`` flag returns cleanly. They are intentionally minimal —
per-command contract tests arrive in Phase 1.
"""

from __future__ import annotations

from typer.testing import CliRunner

from vault_cli import __version__
from vault_cli.exit_codes import ExitCode
from vault_cli.main import app


def test_package_exposes_version() -> None:
    """__version__ matches semver-ish shape."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_version_flag() -> None:
    """``vault --version`` prints the version and exits 0."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "vault" in result.stdout
    assert __version__ in result.stdout


def test_cli_no_args_shows_help() -> None:
    """Bare ``vault`` invocation shows help and exits non-zero (Typer default)."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    # Typer exits 2 when no_args_is_help is set and no args were passed.
    assert result.exit_code != ExitCode.SUCCESS
    assert "Usage" in result.stdout or "Usage" in result.output


def test_exit_code_taxonomy_is_stable() -> None:
    """Regression guard: renumbering exit codes breaks scripts pinned to them."""
    assert ExitCode.SUCCESS == 0
    assert ExitCode.VALIDATION_FAILURE == 1
    assert ExitCode.USAGE_ERROR == 2
    assert ExitCode.IO_ERROR == 3
    assert ExitCode.NETWORK_ERROR == 4
    assert ExitCode.USER_ABORTED == 5

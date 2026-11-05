"""The duplicate-command usage path reports a CLI error without a Rich exception."""
import typer
from typer.testing import CliRunner

from vault_cli.commands import dup
from vault_cli.exit_codes import ExitCode


def test_dup_without_action_reports_usage_instead_of_crashing():
    app = typer.Typer()
    dup.register(app)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == ExitCode.USAGE_ERROR, result.output
    assert "usage: vault dup" in result.output

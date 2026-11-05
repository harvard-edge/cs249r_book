"""Authoring commands use the current model fields without bypassing safety gates."""
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from vault_cli.commands import authoring
from vault_cli.exit_codes import ExitCode
from vault_cli.loader import LoadedQuestion
from vault_cli.models import Question


def fixture_question(tmp_path: Path, monkeypatch, **overrides):
    payload = dict(id="cloud-9999", track="cloud", level="L4", zone="recall",
                   topic="memory-hierarchy-design", competency_area="memory", title="Test question",
                   scenario="An authored test scenario.", details={"realistic_solution": "An answer."},
                   provenance="human")
    payload.update(overrides)
    question = Question.model_validate(payload)
    vault = tmp_path / "vault"
    path = vault / "questions/cloud/cloud-9999.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("test fixture\n")
    monkeypatch.setattr(authoring, "load_all", lambda _: ([LoadedQuestion(question, path)], []))
    app = typer.Typer()
    authoring.register(app)
    return app, vault, path


@pytest.mark.parametrize("command", ["rm", "move"])
def test_plural_chain_membership_refuses_destructive_authoring(tmp_path, monkeypatch, command):
    app, vault, path = fixture_question(tmp_path, monkeypatch,
        chains=[{"id": "first-chain", "position": 0}, {"id": "second-chain", "position": 1}])
    args = [command, "cloud-9999", "--vault-dir", str(vault)]
    args += ["--hard"] if command == "rm" else ["--to", "cloud/L4/recall", "--allow-dirty", "--dry-run"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == ExitCode.USER_ABORTED, result.output
    assert "first-chain" in result.output and "second-chain" in result.output
    assert path.read_text() == "test fixture\n"


def test_unchained_move_dry_run_uses_current_model(tmp_path, monkeypatch):
    app, vault, path = fixture_question(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["move", "cloud-9999", "--vault-dir", str(vault),
                                    "--to", "cloud/L4/recall", "--allow-dirty", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would git mv" in result.output and path.exists()


@pytest.mark.parametrize("provenance,review,eligible", [
    ("human", None, True),
    ("llm-then-human-edited", None, False),
    ("llm-then-human-edited", {"status": "not-reviewed"}, False),
    ("llm-then-human-edited", {"status": "verified", "by": "Test reviewer", "date": "2026-11-05"}, True),
    ("llm-draft", {"status": "verified", "by": "Test reviewer", "date": "2026-11-05"}, False),
])
def test_exemplar_eligibility_uses_current_review_metadata(tmp_path, monkeypatch, provenance, review, eligible):
    app, vault, path = fixture_question(tmp_path, monkeypatch, provenance=provenance, human_reviewed=review)
    result = CliRunner().invoke(app, ["mark-exemplar", "cloud-9999", "--vault-dir", str(vault)])
    assert result.exit_code == (0 if eligible else ExitCode.VALIDATION_FAILURE), result.output
    assert path.exists() != eligible
    assert (vault / "exemplars/cloud/cloud-9999.yaml").exists() == eligible

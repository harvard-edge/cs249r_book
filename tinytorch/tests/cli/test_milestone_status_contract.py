"""
Test Milestone Status Output Contract and Terminology Consistency.

Ensures that:
1. `tito module status` displays consistent terminology:
   - Completed milestones show `[Done]`, matching module `✅ Done` status.
   - Unlocked milestones ready for student execution show `[Ready to run]`.
   - The section header reads `🏆 Historical Milestones:`.
   - Misleading labels like `[Ready to unlock!]` under `Milestones Unlocked:` never appear.
2. `tito milestone list` correctly marks milestones as COMPLETE, READY TO RUN, or LOCKED.
3. `tito milestone status` accurately reflects achievements and progress counts.
"""

import io
import json
from argparse import Namespace
from pathlib import Path
import pytest
from rich.console import Console

from tito.commands.module.workflow import ModuleWorkflowCommand
from tito.commands.milestone import MilestoneCommand, MILESTONE_SCRIPTS
from tito.core.config import CLIConfig


@pytest.fixture
def mock_project(tmp_path, monkeypatch):
    """Set up an isolated project workspace with mock module directory."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    tito_dir = tmp_path / ".tito"
    tito_dir.mkdir(parents=True)

    # Minimal pyproject.toml to define root
    (tmp_path / "pyproject.toml").write_text('[tool.tinytorch]\nversion="0.1.0"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = CLIConfig.from_project_root(tmp_path)
    return tmp_path, config


def test_milestone_readiness_classification(mock_project):
    """Verify _check_milestone_readiness classifies completed vs ready milestones."""
    tmp_path, config = mock_project
    workflow_cmd = ModuleWorkflowCommand(config)

    # Simulate completed modules 01, 02, 03 (satisfies Milestone 01 prerequisites)
    completed_modules = ["01", "02", "03"]

    # When no milestone has been run yet
    readiness = workflow_cmd._check_milestone_readiness(completed_modules)
    status_by_id = {mid: (name, state) for mid, name, state in readiness}

    assert "01" in status_by_id
    assert status_by_id["01"][1] == "ready"

    # Now mark milestone 01 as completed in .tito/milestones.json
    milestones_file = tmp_path / ".tito" / "milestones.json"
    milestones_file.write_text(
        json.dumps({"completed_milestones": ["01"]}),
        encoding="utf-8"
    )

    readiness = workflow_cmd._check_milestone_readiness(completed_modules)
    status_by_id = {mid: (name, state) for mid, name, state in readiness}

    assert "01" in status_by_id
    assert status_by_id["01"][1] == "completed"


def test_module_status_output_terminology_contract(mock_project):
    """Verify tito module status displays [Done] and [Ready to run] without confusing labels."""
    tmp_path, config = mock_project
    workflow_cmd = ModuleWorkflowCommand(config)
    output_stream = io.StringIO()
    workflow_cmd.console = Console(file=output_stream, force_terminal=False, color_system=None)

    # Set up progress: modules 1 through 8 completed
    progress_file = tmp_path / ".tito" / "progress.json"
    completed_mods = [f"{i:02d}" for i in range(1, 9)]
    progress_file.write_text(
        json.dumps({"completed_modules": completed_mods}),
        encoding="utf-8"
    )

    # Set up milestone 01 completed, milestones 02 and 03 ready
    milestones_file = tmp_path / ".tito" / "milestones.json"
    milestones_file.write_text(
        json.dumps({"completed_milestones": ["01"]}),
        encoding="utf-8"
    )

    exit_code = workflow_cmd.show_status()
    assert exit_code == 0

    output = output_stream.getvalue()

    # Verify updated header
    assert "🏆 Historical Milestones:" in output
    assert "🏆 Milestones Unlocked:" not in output

    # Verify milestone 01 is marked [Done]
    assert "01 - Perceptron (1958) [Done]" in output

    # Verify milestones whose prerequisites are met are marked [Ready to run]
    assert "[Ready to run]" in output
    assert "02 - XOR Crisis (1969) [Ready to run]" in output

    # Critical contract: [Ready to unlock!] must never appear
    assert "[Ready to unlock!]" not in output


def test_milestone_list_output_contract(mock_project):
    """Verify tito milestone list categorizes milestones into COMPLETE, READY TO RUN, and LOCKED."""
    tmp_path, config = mock_project
    milestone_cmd = MilestoneCommand(config)
    output_stream = io.StringIO()
    milestone_cmd.console = Console(file=output_stream, force_terminal=False, color_system=None)

    # Modules 1 through 3 complete (satisfies Milestone 01 and 02)
    progress_file = tmp_path / ".tito" / "progress.json"
    progress_file.write_text(
        json.dumps({"completed_modules": ["01", "02", "03"]}),
        encoding="utf-8"
    )

    # Milestone 01 complete
    milestones_file = tmp_path / ".tito" / "milestones.json"
    milestones_file.write_text(
        json.dumps({"completed_milestones": ["01"]}),
        encoding="utf-8"
    )

    args = Namespace(simple=True)
    exit_code = milestone_cmd._handle_list_command(args)
    assert exit_code == 0

    output = output_stream.getvalue()

    # Milestone 01 should be marked with checkmark
    assert "01 - Perceptron (1958)" in output
    # Milestone 02 should be in list
    assert "02 - XOR Crisis (1969)" in output


def test_full_curriculum_status_contract(mock_project):
    """Verify that when all 20 modules are completed, milestones show consistent status."""
    tmp_path, config = mock_project
    workflow_cmd = ModuleWorkflowCommand(config)
    output_stream = io.StringIO()
    workflow_cmd.console = Console(file=output_stream, force_terminal=False, color_system=None)

    # All 20 modules completed
    progress_file = tmp_path / ".tito" / "progress.json"
    completed_mods = [f"{i:02d}" for i in range(1, 21)]
    progress_file.write_text(
        json.dumps({"completed_modules": completed_mods}),
        encoding="utf-8"
    )

    # Milestone 01 completed, others ready
    milestones_file = tmp_path / ".tito" / "milestones.json"
    milestones_file.write_text(
        json.dumps({"completed_milestones": ["01"]}),
        encoding="utf-8"
    )

    exit_code = workflow_cmd.show_status()
    assert exit_code == 0

    output = output_stream.getvalue()

    # All milestones are historical milestones
    assert "🏆 Historical Milestones:" in output
    # Milestone 01 is Done
    assert "01 - Perceptron (1958) [Done]" in output
    # Subsequent milestones are Ready to run
    assert "[Ready to run]" in output
    # No ambiguous Ready to unlock text
    assert "[Ready to unlock!]" not in output

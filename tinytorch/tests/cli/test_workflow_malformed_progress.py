"""
Malformed .tito/progress.json and .tito/milestones.json tests for
ModuleWorkflowCommand (the `tito module start/resume/complete` workflow).

Same bug class already found and fixed twice in tito/commands/milestone.py:
json.load() succeeds on syntactically valid JSON of the wrong shape (a
bare list instead of the expected object), and code that assumes a dict
crashes with a raw AttributeError/TypeError deep inside whichever
command happened to run next.

get_progress_data() is the single source of progress data for 14+ call
sites across the module workflow (is_module_started, mark_module_started,
update_progress, ...), so an unvalidated wrong-type progress.json broke
essentially the entire `tito module` command surface, not just one
command.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.module.workflow import ModuleWorkflowCommand
from tito.core.config import CLIConfig


def _make_command(tmp_path):
    config = CLIConfig(
        project_root=tmp_path,
        assignments_dir=tmp_path,
        tinytorch_dir=tmp_path,
        bin_dir=tmp_path,
        modules_dir=tmp_path,
    )
    return ModuleWorkflowCommand(config)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestGetProgressDataWrongType:
    """progress.json is syntactically valid JSON, but not an object."""

    def test_list_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps(["01_tensor", "02_activations"]))
        cmd = _make_command(tmp_path)

        result = cmd.get_progress_data()

        assert isinstance(result, dict)
        assert result["completed_modules"] == []

    def test_string_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps("not an object"))
        cmd = _make_command(tmp_path)

        result = cmd.get_progress_data()

        assert isinstance(result, dict)

    def test_is_module_started_with_wrong_type_progress_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps(["01_tensor"]))
        cmd = _make_command(tmp_path)

        result = cmd.is_module_started("01")

        assert result is False

    def test_mark_module_started_with_wrong_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        """mark_module_started mutates and re-saves progress; a wrong-type
        file must not crash even on the write path."""
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps(42))
        cmd = _make_command(tmp_path)

        cmd.mark_module_started("01")

        saved = json.loads((tmp_path / ".tito" / "progress.json").read_text(encoding="utf-8"))
        assert saved["started_modules"] == ["01"]

    def test_valid_dict_progress_json_still_works(self, tmp_path, monkeypatch):
        """Regression guard: the type check must not break the normal case."""
        monkeypatch.chdir(tmp_path)
        _write(
            tmp_path / ".tito" / "progress.json",
            json.dumps({"completed_modules": ["01_tensor", "02_activations"]}),
        )
        cmd = _make_command(tmp_path)

        result = cmd.get_progress_data()

        assert result["completed_modules"] == ["01_tensor", "02_activations"]


class TestCheckMilestoneUnlocksWrongType:
    """milestones.json is syntactically valid JSON, but not an object."""

    def test_list_type_milestones_json_does_not_crash(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps({"completed_modules": ["01_tensor"]}))
        _write(tmp_path / ".tito" / "milestones.json", json.dumps([1, 2, 3]))
        cmd = _make_command(tmp_path)

        # Must not raise.
        cmd._check_milestone_unlocks("01_tensor")

    def test_list_type_milestones_json_still_completes_the_unlock_check(self, tmp_path, monkeypatch, capsys):
        """The outer function-level exception handler already prevents a
        crash from ever reaching the caller (it's designed not to fail
        the workflow), so a 'does not raise' test alone can't tell a real
        fix from silently swallowing the error and giving up early. This
        confirms the function actually completes its normal logic
        (treats the malformed file as empty and rewrites it as a valid
        object) rather than bailing into the broad except."""
        monkeypatch.chdir(tmp_path)
        # 03_layers requires modules [1, 2, 3]; completing 03_layers should
        # unlock milestone 01 (requires_modules [1, 2, 3]).
        _write(
            tmp_path / ".tito" / "progress.json",
            json.dumps({"completed_modules": ["01_tensor", "02_activations", "03_layers"]}),
        )
        _write(tmp_path / ".tito" / "milestones.json", json.dumps([1, 2, 3]))
        cmd = _make_command(tmp_path)

        cmd._check_milestone_unlocks("03_layers")

        capsys.readouterr()  # drain output, not asserted on here
        milestones_path = tmp_path / ".tito" / "milestones.json"
        saved = json.loads(milestones_path.read_text(encoding="utf-8"))
        assert isinstance(saved, dict)
        assert "01" in saved.get("unlocked_milestones", []), (
            "Malformed milestones.json should be treated as empty and "
            "still let a genuinely-unlocked milestone be recorded, not "
            "silently give up"
        )

    def test_valid_dict_milestones_json_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / ".tito" / "progress.json", json.dumps({"completed_modules": ["01_tensor", "02_activations", "03_layers"]}))
        _write(
            tmp_path / ".tito" / "milestones.json",
            json.dumps({"unlocked_milestones": [], "completed_milestones": []}),
        )
        cmd = _make_command(tmp_path)

        # Must not raise, and should actually process the valid data.
        cmd._check_milestone_unlocks("03_layers")

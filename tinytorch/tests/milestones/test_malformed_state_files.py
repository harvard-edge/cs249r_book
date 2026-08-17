"""
Malformed .tito state file tests.

TinyTorch's progress/milestone tracking already had a defensive pattern
in tito/commands/milestone.py: catch (json.JSONDecodeError, IOError) around
every state-file read and fall back to sensible empty defaults rather than
crashing. That pattern correctly handles a file that is not valid JSON at
all (truncated, empty, binary garbage).

It did not handle a file that IS valid JSON but the wrong shape (a bare
list or string instead of the expected object), a real corruption mode:
a bad merge conflict resolution, a manual edit gone wrong, or an
interrupted write from something else can all produce syntactically valid
JSON of the wrong type. json.load() succeeds, .get() on a list or string
then raises a raw AttributeError with no useful message.

These tests reproduce that class of corruption directly (crafting the
state file's content in a temp .tito directory) rather than going through
a full tito CLI invocation, so they stay fast and independent of any real
project checkout.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.milestone import _load_completed_module_numbers, MilestoneSystem


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestProgressJsonWrongType:
    """progress.json is syntactically valid JSON, but not an object."""

    def test_list_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), json.dumps(["01_tensor", "02_activations"]))

        result = _load_completed_module_numbers()

        assert result == set()

    def test_string_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), json.dumps("not an object"))

        result = _load_completed_module_numbers()

        assert result == set()

    def test_int_type_progress_json_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), json.dumps(42))

        result = _load_completed_module_numbers()

        assert result == set()

    def test_is_module_completed_with_wrong_type_progress_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), json.dumps(["01_tensor"]))
        ms = MilestoneSystem(config=None)

        result = ms._is_module_completed("01_tensor")

        assert result is False

    def test_valid_dict_progress_json_still_works(self, tmp_path, monkeypatch):
        """Regression guard: the type check must not break the normal case."""
        monkeypatch.chdir(tmp_path)
        _write(
            Path(".tito/progress.json"),
            json.dumps({"completed_modules": ["01_tensor", "02_activations"]}),
        )

        result = _load_completed_module_numbers()

        assert result == {1, 2}


class TestMilestonesJsonWrongType:
    """milestones.json is syntactically valid JSON, but not an object."""

    def test_list_type_milestones_json_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/milestones.json"), json.dumps([1, 2, 3]))
        ms = MilestoneSystem(config=None)

        result = ms._get_milestone_progress_data()

        assert isinstance(result, dict)
        assert result["unlocked_milestones"] == []
        assert result["completed_milestones"] == []

    def test_string_type_milestones_json_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/milestones.json"), json.dumps("corrupted"))
        ms = MilestoneSystem(config=None)

        result = ms._get_milestone_progress_data()

        assert isinstance(result, dict)
        assert result["total_unlocked"] == 0

    def test_valid_dict_milestones_json_still_works(self, tmp_path, monkeypatch):
        """Regression guard: the type check must not break the normal case."""
        monkeypatch.chdir(tmp_path)
        stored = {
            "completed_milestones": ["01"],
            "completion_dates": {"01": "2026-01-01T00:00:00"},
            "unlocked_milestones": ["01"],
            "unlock_dates": {"01": "2026-01-01T00:00:00"},
            "total_unlocked": 1,
            "achievements": [],
        }
        _write(Path(".tito/milestones.json"), json.dumps(stored))
        ms = MilestoneSystem(config=None)

        result = ms._get_milestone_progress_data()

        assert result == stored


class TestTrulyMalformedJsonStillFallsBackCleanly:
    """
    Sanity check that the pre-existing (json.JSONDecodeError, IOError)
    handling for genuinely invalid JSON, truncated, empty, binary garbage,
    still works after adding the isinstance(dict) check alongside it.
    """

    def test_truncated_progress_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), '{"completed_modules": ["01_tensor"')

        result = _load_completed_module_numbers()

        assert result == set()

    def test_empty_progress_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(Path(".tito/progress.json"), "")

        result = _load_completed_module_numbers()

        assert result == set()

    def test_binary_garbage_milestones_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / ".tito" / "milestones.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x01\x02not json at all\xff\xfe")
        ms = MilestoneSystem(config=None)

        result = ms._get_milestone_progress_data()

        assert isinstance(result, dict)
        assert result["unlocked_milestones"] == []

"""
Malformed progress.json/milestones.json tests for
SubmissionHandler._read_json_safe() / assemble_payload().

Same bug class already found and fixed in tito/commands/milestone.py and
tito/commands/module/workflow.py: json.load() succeeds on syntactically
valid JSON of the wrong shape, and code that assumes a dict crashes.
_read_json_safe()'s own docstring and type signature promise a dict but
never validated it, so assemble_payload() crashed with a raw
AttributeError on a wrong-type progress.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.core.submission import SubmissionHandler
from tito.core.config import CLIConfig
from tito.core.console import get_console


def _make_handler(tmp_path):
    config = CLIConfig(
        project_root=tmp_path,
        assignments_dir=tmp_path,
        tinytorch_dir=tmp_path,
        bin_dir=tmp_path,
        modules_dir=tmp_path,
    )
    return SubmissionHandler(config, get_console())


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestReadJsonSafeWrongType:
    def test_list_type_returns_empty_dict(self, tmp_path):
        _write(tmp_path / "data.json", json.dumps([1, 2, 3]))
        handler = _make_handler(tmp_path)

        result = handler._read_json_safe(tmp_path / "data.json")

        assert result == {}

    def test_string_type_returns_empty_dict(self, tmp_path):
        _write(tmp_path / "data.json", json.dumps("not an object"))
        handler = _make_handler(tmp_path)

        result = handler._read_json_safe(tmp_path / "data.json")

        assert result == {}

    def test_valid_dict_still_returned(self, tmp_path):
        _write(tmp_path / "data.json", json.dumps({"key": "value"}))
        handler = _make_handler(tmp_path)

        result = handler._read_json_safe(tmp_path / "data.json")

        assert result == {"key": "value"}


class TestAssemblePayloadWrongTypeFiles:
    def test_wrong_type_progress_json_does_not_crash(self, tmp_path):
        _write(tmp_path / ".tito" / "progress.json", json.dumps(["01_tensor"]))
        _write(tmp_path / ".tito" / "milestones.json", json.dumps({"completed_milestones": []}))
        handler = _make_handler(tmp_path)

        payload = handler.assemble_payload()

        assert payload["module_progress"]["completed_count"] == 0

    def test_wrong_type_milestones_json_does_not_crash(self, tmp_path):
        _write(tmp_path / ".tito" / "progress.json", json.dumps({"completed_modules": ["01_tensor"]}))
        _write(tmp_path / ".tito" / "milestones.json", json.dumps("corrupted"))
        handler = _make_handler(tmp_path)

        payload = handler.assemble_payload()

        assert payload["module_progress"]["completed_count"] == 1

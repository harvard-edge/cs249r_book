"""
Malformed .tito/progress.json test for
LoginCommand._offer_post_login_sync().

Same bug class already found and fixed in tito/commands/milestone.py,
tito/commands/module/workflow.py, tito/core/submission.py, and
tito/core/auth.py: json.loads() succeeds on syntactically valid JSON of
the wrong shape, and .get() on the wrong type raises AttributeError,
which wasn't one of the exceptions this call site caught.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.login import LoginCommand
from tito.core.config import CLIConfig


def _make_command(tmp_path):
    config = CLIConfig(
        project_root=tmp_path,
        assignments_dir=tmp_path,
        tinytorch_dir=tmp_path,
        bin_dir=tmp_path,
        modules_dir=tmp_path,
    )
    return LoginCommand(config)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestOfferPostLoginSyncMalformedProgress:
    def test_list_type_progress_json_does_not_crash(self, tmp_path):
        _write(tmp_path / ".tito" / "progress.json", json.dumps(["01_tensor"]))
        cmd = _make_command(tmp_path)

        # Must not raise.
        cmd._offer_post_login_sync()

    def test_string_type_progress_json_does_not_crash(self, tmp_path):
        _write(tmp_path / ".tito" / "progress.json", json.dumps("corrupted"))
        cmd = _make_command(tmp_path)

        cmd._offer_post_login_sync()

    def test_missing_progress_json_does_not_crash(self, tmp_path):
        cmd = _make_command(tmp_path)

        cmd._offer_post_login_sync()

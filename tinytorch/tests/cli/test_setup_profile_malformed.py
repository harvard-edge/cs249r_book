"""
Malformed/corrupted ~/.tinytorch/profile.json tests for
SetupCommand.create_user_profile().

create_user_profile() previously read profile.json with zero exception
handling at all, not even a narrow catch: a corrupted file (bad merge,
interrupted write, manual edit gone wrong) crashed `tito setup`
completely, with no way to recover short of manually deleting the file.
The write side wasn't atomic either, so an interrupted write could be
the very thing that corrupts the file in the first place.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.setup import SetupCommand
from tito.core.config import CLIConfig


def _make_command(tmp_path):
    config = CLIConfig(
        project_root=tmp_path,
        assignments_dir=tmp_path,
        tinytorch_dir=tmp_path,
        bin_dir=tmp_path,
        modules_dir=tmp_path,
    )
    return SetupCommand(config)


class TestCreateUserProfileMalformedFile:
    def test_corrupted_json_does_not_crash(self, tmp_path):
        home = tmp_path / "home"
        (home / ".tinytorch").mkdir(parents=True)
        (home / ".tinytorch" / "profile.json").write_text("not valid json {{{", encoding="utf-8")

        cmd = _make_command(tmp_path)
        with patch("pathlib.Path.home", return_value=home), \
             patch("tito.commands.setup.Prompt.ask", side_effect=["Test User", "test@example.com", "Test Org"]):
            profile = cmd.create_user_profile()

        assert profile["name"] == "Test User"

    def test_wrong_type_profile_json_does_not_crash(self, tmp_path):
        """Valid JSON, but not an object (e.g. a bare list)."""
        home = tmp_path / "home"
        (home / ".tinytorch").mkdir(parents=True)
        (home / ".tinytorch" / "profile.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        cmd = _make_command(tmp_path)
        with patch("pathlib.Path.home", return_value=home), \
             patch("tito.commands.setup.Prompt.ask", side_effect=["Test User", "test@example.com", "Test Org"]):
            profile = cmd.create_user_profile()

        assert profile["name"] == "Test User"

    def test_valid_profile_json_still_reused(self, tmp_path):
        """Regression guard: a valid existing profile must still be
        returned as-is, not overwritten."""
        home = tmp_path / "home"
        (home / ".tinytorch").mkdir(parents=True)
        (home / ".tinytorch" / "profile.json").write_text(
            json.dumps({"name": "Existing User", "email": "e@example.com"}), encoding="utf-8"
        )

        cmd = _make_command(tmp_path)
        with patch("pathlib.Path.home", return_value=home):
            profile = cmd.create_user_profile()

        assert profile["name"] == "Existing User"

    def test_new_profile_write_leaves_no_temp_file_behind(self, tmp_path):
        home = tmp_path / "home"
        (home / ".tinytorch").mkdir(parents=True)

        cmd = _make_command(tmp_path)
        with patch("pathlib.Path.home", return_value=home), \
             patch("tito.commands.setup.Prompt.ask", side_effect=["Test User", "test@example.com", "Test Org"]):
            cmd.create_user_profile()

        profile_path = home / ".tinytorch" / "profile.json"
        assert profile_path.exists()
        assert not (home / ".tinytorch" / "profile.json.tmp").exists()

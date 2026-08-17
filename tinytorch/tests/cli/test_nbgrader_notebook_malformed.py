"""
Malformed notebook JSON tests for
NBGraderCommand._prepare_notebook_for_nbgrader().

Same bug class already found and fixed across tito/: json.loads()
succeeds on syntactically valid JSON of the wrong shape (a bad merge,
interrupted write, manually-edited .ipynb gone wrong), and code that
assumes a dict crashes. This function already validated that
notebook["cells"] is a list, but not that notebook itself is a dict,
so notebook.get("cells") crashed before that check ever ran.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.nbgrader import NBGraderCommand
from tito.core.config import CLIConfig


def _make_command(tmp_path):
    config = CLIConfig(
        project_root=tmp_path,
        assignments_dir=tmp_path,
        tinytorch_dir=tmp_path,
        bin_dir=tmp_path,
        modules_dir=tmp_path,
    )
    return NBGraderCommand(config)


class TestPrepareNotebookWrongType:
    def test_list_type_notebook_does_not_crash(self, tmp_path):
        cmd = _make_command(tmp_path)

        errors = cmd._prepare_notebook_for_nbgrader(["not", "a", "dict"], release_tier="student")

        assert errors == ["Notebook file does not contain a JSON object"]

    def test_string_type_notebook_does_not_crash(self, tmp_path):
        cmd = _make_command(tmp_path)

        errors = cmd._prepare_notebook_for_nbgrader("corrupted", release_tier="student")

        assert errors == ["Notebook file does not contain a JSON object"]

    def test_valid_dict_without_cells_still_reports_missing_cells(self, tmp_path):
        """Regression guard: the new dict-shape check must not shadow the
        existing, more specific 'missing cells list' error."""
        cmd = _make_command(tmp_path)

        errors = cmd._prepare_notebook_for_nbgrader({"metadata": {}}, release_tier="student")

        assert errors == ["Notebook is missing a cells list"]

    def test_valid_notebook_still_processes_normally(self, tmp_path):
        """Regression guard: a well-formed (if minimal) notebook must
        still reach the normal cell-processing logic, not get rejected
        by the new dict-shape check."""
        cmd = _make_command(tmp_path)
        notebook = {"cells": [], "metadata": {}}

        errors = cmd._prepare_notebook_for_nbgrader(notebook, release_tier="student")

        assert "Notebook file does not contain a JSON object" not in errors
        assert "Notebook is missing a cells list" not in errors

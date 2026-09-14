"""Tests for TinyTorch release tiers and scaffolding policies.

Verifies:
1. Scaffolding tier resolution (student, challenge, instructor)
2. Marker parsing for roles: core, scaffold, challenge, instructor
3. Preservation of scaffold code in student tier (One-Archetype pedagogical model)
4. CLI execution of `tito nbgrader generate` with `--tier`
"""

import json
import os
import pytest
import subprocess
import sys
from pathlib import Path

from tito.commands.nbgrader import (
    NBGraderCommand,
    RELEASE_TIERS,
    VALID_SOLUTION_ROLES,
)


class DummyConfig:
    def __init__(self, root: Path):
        self.project_root = root


@pytest.fixture
def nbgrader_cmd(tmp_path):
    return NBGraderCommand(DummyConfig(tmp_path))


class TestReleaseTierScaffolding:
    """Test unit logic for solution roles and release tiers."""

    def test_tier_constants(self):
        assert set(RELEASE_TIERS) == {"student", "challenge", "instructor"}
        assert set(VALID_SOLUTION_ROLES) == {"core", "scaffold", "challenge", "instructor"}

    def test_solution_role_parsing(self):
        assert NBGraderCommand._solution_role("### BEGIN SOLUTION") == "core"
        assert NBGraderCommand._solution_role("### BEGIN SOLUTION role=\"core\"") == "core"
        assert NBGraderCommand._solution_role("### BEGIN SOLUTION role=\"scaffold\"") == "scaffold"
        assert NBGraderCommand._solution_role("### BEGIN SOLUTION role='challenge'") == "challenge"
        assert NBGraderCommand._solution_role("### BEGIN SOLUTION role=instructor") == "instructor"

    def test_solution_role_actions(self):
        # Student tier: core is stripped (for student to do), scaffold is kept (pre-solved)
        assert NBGraderCommand._solution_role_action("core", "student") == "strip"
        assert NBGraderCommand._solution_role_action("scaffold", "student") == "keep"
        assert NBGraderCommand._solution_role_action("instructor", "student") == "remove"

        # Challenge tier: challenge is stripped
        assert NBGraderCommand._solution_role_action("challenge", "challenge") == "strip"
        assert NBGraderCommand._solution_role_action("scaffold", "challenge") == "keep"

        # Instructor tier: all solutions kept, no stripping
        assert NBGraderCommand._solution_role_action("core", "instructor") == "keep"
        assert NBGraderCommand._solution_role_action("scaffold", "instructor") == "keep"

    def test_apply_release_tier_student(self, nbgrader_cmd):
        source = (
            "def demo():\n"
            "    ### BEGIN SOLUTION role=\"core\"\n"
            "    return 42\n"
            "    ### END SOLUTION\n"
            "    ### BEGIN SOLUTION role=\"scaffold\"\n"
            "    scaffold_var = 10\n"
            "    ### END SOLUTION\n"
        )
        transformed, errors = nbgrader_cmd._apply_release_tier(source, "student")
        assert not errors
        # Core solution block should retain BEGIN/END SOLUTION markers for nbgrader to strip
        assert "### BEGIN SOLUTION" in transformed
        assert "### END SOLUTION" in transformed
        assert "return 42" in transformed
        # Scaffold block should NOT have markers, but its code should be preserved
        assert "role=\"scaffold\"" not in transformed
        assert "scaffold_var = 10" in transformed

    def test_apply_release_tier_instructor(self, nbgrader_cmd):
        source = (
            "def demo():\n"
            "    ### BEGIN SOLUTION role=\"core\"\n"
            "    return 42\n"
            "    ### END SOLUTION\n"
        )
        transformed, errors = nbgrader_cmd._apply_release_tier(source, "instructor")
        assert not errors
        # In instructor tier, solution markers are stripped so code remains intact
        assert "### BEGIN SOLUTION" not in transformed
        assert "### END SOLUTION" not in transformed
        assert "return 42" in transformed


class TestReleaseTierCLI:
    """Test CLI commands for nbgrader generation across tiers."""

    @pytest.mark.parametrize("tier", ["student", "challenge", "instructor"])
    def test_nbgrader_generate_tiers(self, tier):
        repo_root = Path(__file__).parent.parent.parent
        tito_bin = repo_root / "bin" / "tito"

        cmd = [
            sys.executable,
            str(tito_bin),
            "nbgrader", "generate", "01",
            "--tier", tier,
        ]
        env = dict(os.environ)
        env["TITO_ALLOW_SYSTEM"] = "1"
        res = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert res.returncode == 0, f"Generate failed for tier {tier}:\n{res.stderr}\n{res.stdout}"
        assert "Staged" in res.stdout

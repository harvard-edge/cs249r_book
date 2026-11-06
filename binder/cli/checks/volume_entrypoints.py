"""Exercise generated volume entry points through their read-only regression suite."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .cli_contract import Violation

TEST_FILES = ("binder/tests/test_volume_index.py", "binder/tests/test_selective_volume_build.py")


def run_contract(root: Path) -> list[Violation]:
    """Fail on source corruption, invalid includes, or selective-build state drift."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *TEST_FILES, "-q"], cwd=root,
            capture_output=True, text=True, check=False, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [Violation(TEST_FILES[0], 0, "BINDER-INDEX-001", str(exc))]
    if result.returncode == 0:
        return []
    return [Violation(
        TEST_FILES[0], 0, "BINDER-INDEX-001", "Volume entry-point regression checks failed",
        context=(result.stdout + result.stderr)[-8000:],
        suggestion="Rerun ./binder/binder check cli --scope volume-entrypoints for diagnostics.",
    )]

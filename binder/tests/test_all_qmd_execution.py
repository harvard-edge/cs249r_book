"""Test that all Python blocks across Volumes I, II, III, and IV execute cleanly."""

from pathlib import Path
import pytest
from binder.cli.checks.code_exec import check_code_exec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BOOKS_DIR = REPO_ROOT / "books"
VOL_DIRS = [BOOKS_DIR / f"vol{i}" for i in range(1, 5)]


def get_qmd_files():
    files = []
    for vol_dir in VOL_DIRS:
        if vol_dir.exists():
            files.extend(sorted(vol_dir.glob("**/*.qmd")))
    return files


@pytest.mark.parametrize("qmd_path", get_qmd_files(), ids=lambda p: p.relative_to(BOOKS_DIR).as_posix())
def test_qmd_blocks_execute(qmd_path):
    """Verify that every python block in a book QMD file executes without error."""
    issues = check_code_exec(qmd_path)
    if issues:
        pytest.fail(f"{issues[0].file.relative_to(BOOKS_DIR)} L{issues[0].line}: {issues[0].message}")

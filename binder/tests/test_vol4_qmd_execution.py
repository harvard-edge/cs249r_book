"""Test that all Python blocks across Volume IV chapters execute cleanly."""

import os
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BOOKS_DIR = REPO_ROOT / "books"
VOL4_DIR = BOOKS_DIR / "vol4"


def get_vol4_qmd_files():
    if not VOL4_DIR.exists():
        return []
    return sorted(VOL4_DIR.glob("*/*.qmd"))


@pytest.mark.parametrize("qmd_path", get_vol4_qmd_files(), ids=lambda p: p.relative_to(VOL4_DIR).as_posix())
def test_vol4_qmd_blocks_execute(qmd_path):
    """Verify that every python block in a Volume IV QMD file executes without error."""
    content = qmd_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```\{python\}(.*?)```", content, re.DOTALL)
    if not blocks:
        pytest.skip(f"No python blocks in {qmd_path.name}")

    old_cwd = os.getcwd()
    try:
        os.chdir(BOOKS_DIR)
        scope = {}
        for idx, block in enumerate(blocks):
            clean_lines = [line for line in block.splitlines() if not line.strip().startswith("#|")]
            code = "\n".join(clean_lines)
            try:
                exec(code, scope)
            except Exception as exc:
                pytest.fail(f"{qmd_path.name} block {idx} execution failed: {exc}")
    finally:
        os.chdir(old_cwd)

"""Run the PDF compressor's preservation regression contract."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .cli_contract import Violation

TEST_FILES = ("binder/tests/test_pdf_compression.py", "binder/tests/test_pdf_layout_checks.py",
              "binder/tests/test_save_latex_log.py")


def run_contract(root: Path) -> list[Violation]:
    """Exercise lossless preservation and intentional corruption cases."""
    test_file = TEST_FILES[0]
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", *TEST_FILES, "-q"],
                                cwd=root, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [Violation(test_file, 0, "BINDER-PDF-001", str(exc))]
    if result.returncode == 0:
        return []
    return [Violation(test_file, 0, "BINDER-PDF-001", "PDF compression preservation contract failed",
                      context=(result.stdout + result.stderr)[-8000:])]

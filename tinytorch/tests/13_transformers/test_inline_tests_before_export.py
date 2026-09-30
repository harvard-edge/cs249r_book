"""Module 13's inline tests must pass before Module 13 is exported.

`tito module complete 13` runs the notebook's inline tests (Step 1) BEFORE it
exports the notebook (Step 2). A student completing Module 13 for the first time
therefore has no tinytorch.core.transformers yet. The notebook's GPT tests
passed only when a previous export existed, because models/transformer.py
pulled TransformerBlock and LayerNorm from that export. They now pass the
notebook's own classes to GPT; this test holds that line by running the module
with tinytorch.core.transformers blocked.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "13_transformers" / "13_transformers.py"

RUNNER = """
import runpy, sys
sys.modules["tinytorch.core.transformers"] = None   # Module 13 not exported yet
runpy.run_path(sys.argv[1], run_name="__main__")
import importlib
try:
    importlib.import_module("tinytorch.core.transformers")
except ImportError:
    print("EXPORT-STILL-BLOCKED")
"""


def test_module_13_inline_tests_pass_without_its_own_export():
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(SOURCE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-3000:]
    assert "ALL TESTS PASSED" in result.stdout, output[-3000:]
    assert "Full transformer pipeline works" in result.stdout, output[-3000:]
    assert "EXPORT-STILL-BLOCKED" in result.stdout, output[-3000:]

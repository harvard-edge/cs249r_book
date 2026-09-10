#!/usr/bin/env python3
"""Check source solutions in a disposable package, without overwriting notebooks.

Added 2026-09-11 after reference solutions passed release checks despite broken
training and numerical edge cases. Shared by pre-commit and CI.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
REGRESSIONS = (
    "06_autograd/test_graph_lifetime.py",
    "08_training/test_accumulation_windows.py",
    "15_quantization/test_constant_roundtrip.py",
    "16_compression/test_distillation_training.py",
)


def main() -> int:
    import jupytext
    from nbdev.export import nb_export

    sources = sorted((ROOT / "src").glob("[0-9][0-9]_*/*.py"))
    if len(sources) != 20:
        print(f"Expected 20 source modules, found {len(sources)}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="tinytorch-reference-") as tmp:
        sandbox = Path(tmp)
        package = sandbox / "tinytorch"
        for subdir in ("", "core", "perf"):
            target = package / subdir
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "tinytorch" / subdir / "__init__.py", target / "__init__.py")
        for source in sources:
            notebook = sandbox / f"{source.stem}.ipynb"
            jupytext.write(jupytext.read(source), notebook)
            nb_export(str(notebook), lib_path=str(package))

        tests = sandbox / "tests"
        tests.mkdir()
        for relative in REGRESSIONS:
            source = ROOT / "tests" / relative
            shutil.copyfile(source, tests / source.name)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(sandbox), TINYTORCH_QUIET="1")
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short",
             "-c", str(ROOT / "pyproject.toml"), "--confcutdir", str(sandbox), str(tests)],
            cwd=sandbox, env=env,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

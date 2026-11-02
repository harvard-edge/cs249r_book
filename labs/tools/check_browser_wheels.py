#!/usr/bin/env python3
"""Reject browser wheels whose runtime code/data differs from the release source.

2026-10-20: native labs loaded current sources while checked-in WASM wheels
contained older code and registry data. Check content, not wheel timestamps.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt"}


def check_wheels(root: Path = ROOT) -> list[str]:
    failures = []
    for project, package in ((root / "mlsysim", "mlsysim"), (root / "labs", "mlsysbook_labs")):
        project_toml = (project / "pyproject.toml").read_text()
        project_section = project_toml.split("[project]", 1)[1].split("\n[", 1)[0]
        version = re.search(r'^version\s*=\s*"([^"]+)"', project_section, re.M).group(1)
        wheel = root / "labs" / "wheels" / f"{package}-{version}-py3-none-any.whl"
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                for source in sorted((project / package).rglob("*")):
                    if not source.is_file() or source.suffix not in RUNTIME_SUFFIXES:
                        continue
                    relative = source.relative_to(project).as_posix()
                    if relative not in names:
                        failures.append(f"{wheel.name}: missing {relative}")
                    elif archive.read(relative) != source.read_bytes():
                        failures.append(f"{wheel.name}: stale {relative}")
        except (OSError, zipfile.BadZipFile) as error:
            failures.append(f"{wheel.name}: {error}")
    return failures


if __name__ == "__main__":
    problems = check_wheels()
    if problems:
        print("Browser runtime differs from source:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Rebuild both wheels with python3 -m build --wheel <project> --outdir labs/wheels", file=sys.stderr)
        raise SystemExit(1)
    print("Browser wheels match runtime source and registry data.")

#!/usr/bin/env python3
"""Run registry-migration CI gates (book sources, appendix cells, provenance)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MLSYSIM = REPO / "mlsysim"


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd or REPO)


def main() -> int:
    steps: list[tuple[str, list[str], Path | None]] = [
        ("book registry sources", [sys.executable, "book/tools/audit/book_check_registry_sources.py"], REPO),
        ("appendix LEGO cells", [sys.executable, "book/tools/audit/generate_appendix_constants.py", "--verify"], REPO),
        (
            "mlsysim provenance",
            [sys.executable, "-m", "mlsysim.tools.audit_provenance", "--scope", "all", "--strict"],
            MLSYSIM,
        ),
        (
            "pytest migration gates",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_provenance.py",
                "tests/test_provenance_audit.py",
                "tests/test_constants_allowlist.py",
                "../book/tests/test_appendix_constants.py",
                "../book/tests/test_appendix_lineage.py",
                "../book/tests/test_no_legacy_constant_refs.py",
                "../book/tests/test_mlsysim_registry_coverage.py",
                "../book/tests/test_mlsysim_registry_parity.py",
                "--no-cov",
                "-q",
            ],
            MLSYSIM,
        ),
    ]
    failed = 0
    for label, cmd, cwd in steps:
        print(f"\n=== {label} ===")
        if _run(cmd, cwd=cwd) != 0:
            failed += 1
    if failed:
        print(f"\n{failed} gate(s) failed")
        return 1
    print("\nAll registry migration gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run end-to-end gates for the constants → registry migration build.

Canonical entry point: ``./binder check registry``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    binder = REPO_ROOT / "book" / "binder"
    proc = subprocess.run([str(binder), "check", "registry"], cwd=REPO_ROOT)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

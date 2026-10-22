#!/usr/bin/env python3
"""Quarto hook: compile chapter-owned concepts into the series explorer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "binder"))
from cli.commands.concepts import run

if __name__ == "__main__":
    raise SystemExit(0 if run(ROOT, ["build"]) else 1)

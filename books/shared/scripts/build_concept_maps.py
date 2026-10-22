#!/usr/bin/env python3
"""Quarto hook for chapter concept-map viewers; no model calls at render time."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "binder"))
from cli.commands.concepts import run

if __name__ == "__main__":
    raise SystemExit(0 if run(ROOT, ["build", *sys.argv[1:]]) else 1)

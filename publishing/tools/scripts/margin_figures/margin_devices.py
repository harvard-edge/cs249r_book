#!/usr/bin/env python3
"""Compatibility wrapper for the Book Tools margin-device package.

Stable margin drawing devices live in ``book.tools.figures.margin.devices``.
This module remains so older scripts or ad-hoc commands that import
``margin_devices`` from this directory keep working.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from book.tools.figures.margin.devices import *  # noqa: F401,F403,E402

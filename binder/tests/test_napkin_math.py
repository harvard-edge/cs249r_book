"""Napkin Math sheets build from the current mlsysim registry.

Each volume module computes every number on its sheet from mlsysim and, for
Volumes III and IV, asserts that its printed formulas agree with the book's
scenario solvers. Building every sheet here catches registry or scenario
changes that would silently change or break a sheet. Layout is checked by
`python3 -m napkin_math pdf`, which needs a browser and is not run here.
"""
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "books" / "shared"
VOLUMES = ["vol1", "vol2", "vol3", "vol4"]


@pytest.fixture(scope="module")
def napkin():
    sys.path.insert(0, str(SHARED))
    try:
        yield importlib.import_module("napkin_math.core")
    finally:
        sys.path.remove(str(SHARED))


@pytest.mark.parametrize("vol", VOLUMES)
def test_sheet_builds_and_renders(napkin, vol):
    sheet = importlib.import_module(f"napkin_math.{vol}").build()
    assert sheet.volume == VOLUMES.index(vol) + 1
    assert len(sheet.pages) >= 2  # pages 1-2 are the core handout; more pages may follow
    html = napkin.render(sheet)
    text = re.sub(r"<[^>]+>", " ", html)
    for bad in (r"\bnan\b", r"\binf\b", r"\bNone\b", r"\{\{", r"\}\}"):
        assert not re.search(bad, text), f"{vol}: rendered text matches {bad!r}"
    for page in sheet.pages:
        assert len(page.cols) == 3

"""
Labs Test Configuration
=======================

Shared fixtures for testing Marimo lab notebooks.
Three test levels:

  Level 1 (Static):  AST parse, structure checks, import validation
  Level 2 (Engine):  Run cells headlessly via marimo.App.run(), check computations
  Level 3 (Widget):  Simulate widget interactions, verify prediction/reveal flow

Usage:
  python3 -m pytest labs/tests/ -v                    # All levels
  python3 -m pytest labs/tests/ -v -k "static"        # Level 1 only (fast, CI)
  python3 -m pytest labs/tests/ -v -k "engine"        # Level 2 (medium, CI)
  python3 -m pytest labs/tests/ -v -k "widget"        # Level 3 (slow, optional)
"""

import ast
import glob
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Discovery: collect all lab notebook paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"

VOL1_LABS = sorted(glob.glob(str(LABS_ROOT / "vol1" / "lab_*.py")))
VOL2_LABS = sorted(glob.glob(str(LABS_ROOT / "vol2" / "lab_*.py")))
ALL_LABS = VOL1_LABS + VOL2_LABS


def lab_id(path: str) -> str:
    """Extract a short ID like 'vol1/lab_01' from a full path."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=ALL_LABS, ids=[lab_id(p) for p in ALL_LABS])
def lab_path(request):
    """Parametrized fixture yielding each lab file path."""
    return request.param


@pytest.fixture(params=VOL1_LABS, ids=[lab_id(p) for p in VOL1_LABS])
def vol1_lab_path(request):
    return request.param


@pytest.fixture(params=VOL2_LABS, ids=[lab_id(p) for p in VOL2_LABS])
def vol2_lab_path(request):
    return request.param


@pytest.fixture(scope="session")
def mlsysim():
    """Import mlsysim once for the session."""
    sys.path.insert(0, str(REPO_ROOT))
    import mlsysim
    return mlsysim

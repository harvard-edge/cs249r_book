"""
TinyTorch CLI Package

A professional command-line interface for the TinyTorch ML system.
Organized with clean separation of concerns and proper error handling.
"""

from pathlib import Path as _Path

def _get_version() -> str:
    """Read version from pyproject.toml (single source of truth)."""
    try:
        pyproject_path = _Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            for line in content.splitlines():
                if line.strip().startswith("version"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "0.0.0-dev"

__version__ = _get_version()
__author__ = "TinyTorch Team"

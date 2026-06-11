"""Shared paths for release-audit helper scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


REPO_ROOT = _env_path(
    "MLSYSBOOK_REPO_ROOT",
    Path(__file__).resolve().parents[4],
)
DATA_ROOT = _env_path(
    "MLSYSBOOK_RELEASE_AUDIT_DATA",
    REPO_ROOT / "book/tools/audit/release/data",
)
OUT_ROOT = _env_path(
    "MLSYSBOOK_RELEASE_AUDIT_OUT",
    REPO_ROOT / "book/quarto/_build/release_audit",
)

QUARTO_ROOT = REPO_ROOT / "book/quarto/contents"
LEDGER_DIR = OUT_ROOT / "ledgers"
SCRIPT_DIR = OUT_ROOT / "scripts"
LOG_DIR = OUT_ROOT / "logs"

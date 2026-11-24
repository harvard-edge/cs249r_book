"""Regression tests for mlsysim/docs curriculum cross-chapter links."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_checker():
    checker_path = (
        Path(__file__).resolve().parents[1] / "docs" / "scripts" / "check_curriculum_links.py"
    )
    spec = importlib.util.spec_from_file_location("check_curriculum_links", checker_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_curriculum_links_manifest_and_docs_valid() -> None:
    checker = _load_checker()
    errors = checker.check_all()
    assert not errors, "Curriculum link errors:\n" + "\n".join(errors)

"""Book appendix references must resolve to sourced MLSysIM registry entries."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from book.tools.audit.appendix_lineage import (
    audit_appendix_defaults,
    audit_appendix_literature,
    audit_appendix_pricing,
    audit_appendix_reliability,
)


def test_appendix_has_no_defaults_refs() -> None:
    issues = audit_appendix_defaults()
    assert issues == [], "\n".join(issues)


def test_appendix_pricing_lineage() -> None:
    issues = audit_appendix_pricing()
    assert issues == [], "\n".join(issues)


def test_appendix_reliability_lineage() -> None:
    issues = audit_appendix_reliability()
    assert issues == [], "\n".join(issues)


def test_appendix_literature_lineage() -> None:
    issues = audit_appendix_literature()
    assert issues == [], "\n".join(issues)

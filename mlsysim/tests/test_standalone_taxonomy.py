from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "mlsysim"


def test_package_provenance_ids_do_not_use_book_namespace():
    """MLSysIM package data should name provenance by domain, not by consumer."""
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md"}:
            continue

        text = path.read_text(encoding="utf-8")
        if "BOOK_" in text or "prov:book-" in text:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))

    assert not offenders

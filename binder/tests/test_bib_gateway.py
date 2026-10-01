import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BINDER_ROOT = Path(__file__).resolve().parents[1]
if str(BINDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BINDER_ROOT))

from cli.core import bib_gateway


def test_compute_entry_hash_normalized():
    raw_1 = "@article{key,\n  title = {A Title},\n  year = {2020},\n}"
    raw_2 = "@article{key, title = {A Title}, year = {2020}, }"
    assert bib_gateway.compute_entry_hash(raw_1) == bib_gateway.compute_entry_hash(raw_2)


def test_derive_citekey():
    assert bib_gateway.derive_citekey("Vaswani, Ashish and Shazeer, Noam", 2017, "Attention Is All You Need") == "vaswani2017attention"
    assert bib_gateway.derive_citekey("Kaiming He and Xiangyu Zhang", 2016, "Deep Residual Learning") == "he2016deep"
    assert bib_gateway.derive_citekey("", 2024, "Autonomous Vehicle Safety") == "ref2024autonomous"


def test_format_bibtex_entry_pages_dash():
    fields = {
        "title": "A Sample Paper",
        "author": "Smith, Alice",
        "year": "2024",
        "journal": "Nature",
        "pages": "10-25",
        "doi": "10.1038/example",
    }
    formatted = bib_gateway.format_bibtex_entry("article", "smith2024sample", fields)
    assert "@article{smith2024sample," in formatted
    assert "  pages = {10--25}," in formatted
    assert "  doi = {10.1038/example}," in formatted


def test_add_entry_success(tmp_path: Path):
    bib_file = tmp_path / "books" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{existing,\n  title = {Existing},\n  year = {2020},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "binder" / "tools" / "bib_verification_manifest.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}", encoding="utf-8")

    entry_data = {
        "key": "new2024paper",
        "entry_type": "article",
        "fields": {
            "title": "New Verified Publication",
            "author": "Doe, Jane",
            "year": "2024",
            "journal": "Science",
            "doi": "10.1126/science.123456",
        },
        "evidence": {
            "source": "crossref",
            "doi": "10.1126/science.123456",
        },
    }

    ok, msg, rec = bib_gateway.add_entry(tmp_path, entry_data, check_duplicates=True)
    assert ok
    assert rec is not None
    assert rec["status"] == "verified"
    assert rec["entry_hash"] is not None

    # Check manifest file updated
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "new2024paper" in manifest
    assert manifest["new2024paper"]["entry_hash"] == rec["entry_hash"]

    # Check references.bib contains entry
    content = bib_file.read_text(encoding="utf-8")
    assert "@article{new2024paper," in content


def test_add_entry_rejects_duplicate_key(tmp_path: Path):
    bib_file = tmp_path / "books" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{duplicate2024,\n  title = {First},\n  year = {2024},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "binder" / "tools" / "bib_verification_manifest.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}", encoding="utf-8")

    entry_data = {
        "key": "duplicate2024",
        "entry_type": "article",
        "fields": {"title": "Second", "year": "2024"},
        "evidence": {"source": "crossref"},
    }

    ok, msg, rec = bib_gateway.add_entry(tmp_path, entry_data, check_duplicates=True)
    assert not ok
    assert "already exists" in msg


def test_add_entry_rejects_duplicate_doi(tmp_path: Path):
    bib_file = tmp_path / "books" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{orig2024,\n  title = {Orig},\n  year = {2024},\n  doi = {10.1145/123456},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "binder" / "tools" / "bib_verification_manifest.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}", encoding="utf-8")

    entry_data = {
        "key": "differentkey2024",
        "entry_type": "article",
        "fields": {"title": "Same DOI Paper", "year": "2024", "doi": "10.1145/123456"},
        "evidence": {"source": "crossref"},
    }

    ok, msg, rec = bib_gateway.add_entry(tmp_path, entry_data, check_duplicates=True)
    assert not ok
    assert "already cited under key" in msg


def test_ingest_inbox(tmp_path: Path):
    inbox_file = tmp_path / "books" / "bib-inbox.bib"
    inbox_file.parent.mkdir(parents=True)
    inbox_content = """@misc{fakeentry,
  title = {Totally Fabricated Nonexistent Paper},
  author = {Ghost, Bob},
  year = {2024},
}"""
    inbox_file.write_text(inbox_content, encoding="utf-8")

    manifest_file = tmp_path / "binder" / "tools" / "bib_verification_manifest.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}", encoding="utf-8")

    bib_file = tmp_path / "books" / "references.bib"
    bib_file.write_text("", encoding="utf-8")

    added, rejected, msgs = bib_gateway.ingest_inbox(tmp_path, inbox_file)
    assert added == 0
    assert rejected == 1
    assert "Rejected unverified" in msgs[0]

    # Fabricated entry remains in inbox, references.bib remains untouched
    assert "@misc{fakeentry" in inbox_file.read_text(encoding="utf-8")
    assert bib_file.read_text(encoding="utf-8") == ""

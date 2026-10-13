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
    bib_file = tmp_path / "bibs" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{existing,\n  title = {Existing},\n  year = {2020},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "bibs" / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
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
    assert manifest["new2024paper"]["file"] == "bibs/references.bib"

    # Check references.bib contains entry
    content = bib_file.read_text(encoding="utf-8")
    assert "@article{new2024paper," in content


def test_add_entry_legacy_fallback(tmp_path: Path):
    """Fallback to books/references.bib and binder/tools/bib_verification_manifest.json if bibs/ not present."""
    bib_file = tmp_path / "books" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{existing,\n  title = {Existing},\n  year = {2020},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "binder" / "tools" / "bib_verification_manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text("{}", encoding="utf-8")

    entry_data = {
        "key": "legacy2024",
        "entry_type": "article",
        "fields": {"title": "Legacy Test", "year": "2024"},
        "evidence": {"source": "crossref"},
    }

    ok, msg, rec = bib_gateway.add_entry(tmp_path, entry_data, check_duplicates=True)
    assert ok
    assert rec is not None
    assert "legacy2024" in json.loads(manifest_file.read_text(encoding="utf-8"))


def test_add_entry_rejects_duplicate_key(tmp_path: Path):
    bib_file = tmp_path / "bibs" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{duplicate2024,\n  title = {First},\n  year = {2024},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "bibs" / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
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
    bib_file = tmp_path / "bibs" / "references.bib"
    bib_file.parent.mkdir(parents=True)
    bib_file.write_text("@article{orig2024,\n  title = {Orig},\n  year = {2024},\n  doi = {10.1145/123456},\n}\n", encoding="utf-8")

    manifest_file = tmp_path / "bibs" / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
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
    inbox_file = tmp_path / "bibs" / "inbox.bib"
    inbox_file.parent.mkdir(parents=True)
    inbox_content = """@misc{fakeentry,
  title = {Totally Fabricated Nonexistent Paper},
  author = {Ghost, Bob},
  year = {2024},
}"""
    inbox_file.write_text(inbox_content, encoding="utf-8")

    manifest_file = tmp_path / "bibs" / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text("{}", encoding="utf-8")

    bib_file = tmp_path / "bibs" / "references.bib"
    bib_file.write_text("", encoding="utf-8")

    added, rejected, msgs = bib_gateway.ingest_inbox(tmp_path, inbox_file)
    assert added == 0
    assert rejected == 1
    assert "Rejected unverified" in msgs[0]

    # Fabricated entry remains in inbox, references.bib remains untouched
    assert "@misc{fakeentry" in inbox_file.read_text(encoding="utf-8")
    assert bib_file.read_text(encoding="utf-8") == ""


# --- bib reseal (2026-10-13) -------------------------------------------------

_RESEAL_BIB = """@inproceedings{doe2024paper,
  title = {Position: A study of {LLM} things},
  author = {Doe, Jane and M\\k{a}dry, Aleksander and Alves da Costa, Theo},
  year = {2024},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  pages = {10--20},
}
"""


def _reseal_repo(tmp_path: Path) -> Path:
    (tmp_path / "bibs").mkdir()
    (tmp_path / "bibs" / "references.bib").write_text(_RESEAL_BIB, encoding="utf-8")
    old = {"doe2024paper": {"key": "doe2024paper", "entry_hash": "stale", "status": "verified",
                            "details": {"source": "verified_web", "supersedes": "arXiv:0000.00000"}}}
    (tmp_path / "bibs" / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    return tmp_path


def _resolved(**overrides):
    evidence = {
        "source": "verified_web",
        "url": "https://proceedings.mlr.press/v235/doe24a.html",
        "canonical_title": "Position: A study of LLM things",
        "canonical_authors": ["Doe, Jane", "Madry, Aleksander", "Costa, Theo Alves da"],
        "canonical_year": 2024,
        "canonical_pages": "10--20",
    }
    evidence.update(overrides)
    return {"key": "doe2024paper", "entry_type": "misc", "fields": {}, "evidence": evidence}


def test_reseal_rerecords_hash_when_source_agrees(tmp_path: Path):
    repo = _reseal_repo(tmp_path)
    ok, msg, record = bib_gateway.reseal_entry(repo, "doe2024paper", _resolved(), "house style")
    assert ok, msg
    manifest = json.loads((repo / "bibs" / "manifest.json").read_text(encoding="utf-8"))
    rec = manifest["doe2024paper"]
    entry = bib_gateway.bib_lint.parse_bib(_RESEAL_BIB)[0][0]
    assert rec["entry_hash"] == bib_gateway.compute_entry_hash(entry.raw)
    assert rec["entry_type"] == "inproceedings"
    assert rec["year"] == "2024"
    assert rec["details"]["checked_fields"] == ["title", "author", "year", "pages"]
    assert rec["details"]["supersedes"] == "arXiv:0000.00000"
    assert rec["details"]["style_edit"] == "house style"


def test_reseal_refuses_year_author_and_page_mismatch(tmp_path: Path):
    repo = _reseal_repo(tmp_path)
    before = (repo / "bibs" / "manifest.json").read_text(encoding="utf-8")
    for bad in (
        {"canonical_year": 2022},
        {"canonical_authors": ["Doe, Jane", "Smith, Ann", "Costa, Theo"]},
        {"canonical_pages": "11--20"},
        {"canonical_title": "An entirely different paper about databases"},
    ):
        ok, msg, _ = bib_gateway.reseal_entry(repo, "doe2024paper", _resolved(**bad), "x")
        assert not ok, bad
    assert (repo / "bibs" / "manifest.json").read_text(encoding="utf-8") == before


def test_reseal_dry_run_writes_nothing(tmp_path: Path):
    repo = _reseal_repo(tmp_path)
    before = (repo / "bibs" / "manifest.json").read_text(encoding="utf-8")
    ok, _, record = bib_gateway.reseal_entry(repo, "doe2024paper", _resolved(), "x", dry_run=True)
    assert ok and record["status"] == "verified"
    assert (repo / "bibs" / "manifest.json").read_text(encoding="utf-8") == before


def test_reseal_rejects_unknown_key(tmp_path: Path):
    repo = _reseal_repo(tmp_path)
    ok, msg, _ = bib_gateway.reseal_entry(repo, "missing2020key", _resolved(), "x")
    assert not ok and "found 0" in msg


def test_parse_citation_meta_highwire_tags():
    page = (
        '<meta name="citation_title" content="Position: Will we run out of data?">'
        '<meta name="citation_author" content="Pablo Villalobos">'
        '<meta content="Anson Ho" name="citation_author">'
        '<meta name="citation_publication_date" content="2024/07/08">'
        '<meta name="citation_firstpage" content="49523">'
        '<meta name="citation_lastpage" content="49544">'
        '<meta name="citation_conference_title" content="International Conference on Machine Learning">'
    )
    meta = bib_gateway.parse_citation_meta(page)
    assert meta["title"] == "Position: Will we run out of data?"
    assert meta["authors"] == ["Villalobos, Pablo", "Ho, Anson"]
    assert meta["year"] == 2024
    assert meta["pages"] == "49523--49544"
    assert meta["venue"] == "International Conference on Machine Learning"
    assert bib_gateway.parse_citation_meta("<html><title>x</title></html>") == {}


def test_parse_bibtex_export_publisher_record():
    export = """@techreport{bibcite_36455,
  author = {Arman Shehabi and Sarah Josephine Smith and Md AbuBakar Siddik},
  title = {2024 United States Data Center Energy Usage Report},
  year = {2024},
  institution = {Lawrence Berkeley National Laboratory},
  doi = {10.71468/P1WC7Q},
}"""
    meta = bib_gateway.parse_bibtex_export(export)
    assert meta["title"] == "2024 United States Data Center Energy Usage Report"
    assert meta["authors"] == ["Shehabi, Arman", "Smith, Sarah Josephine", "Siddik, Md AbuBakar"]
    assert meta["year"] == 2024
    assert meta["doi"] == "10.71468/P1WC7Q"
    assert meta["venue"] == "Lawrence Berkeley National Laboratory"
    assert bib_gateway.parse_bibtex_export("not bibtex") == {}

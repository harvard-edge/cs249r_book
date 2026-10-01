"""Tests for the bib-verify check scope in ValidateCommand."""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from binder.cli.commands.validate import ValidateCommand


def _compute_hash(raw: str) -> str:
    normalized = re.sub(r"\s+", " ", raw.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_bib_verify_clean_match(tmp_path: Path):
    """Verified entry with matching hash passes with 0 issues."""
    bib = tmp_path / "refs.bib"
    raw_entry = """@article{good2020,
  title = {A Genuine Publication},
  author = {Good, Alice},
  year = {2020},
  journal = {Nature Machine Intelligence},
  doi = {10.1038/s42256-020-0001},
}"""
    bib.write_text(raw_entry, encoding="utf-8")

    manifest = {
        "good2020": {
            "key": "good2020",
            "file": "refs.bib",
            "entry_hash": _compute_hash(raw_entry),
            "status": "verified",
        }
    }
    manifest_dir = tmp_path / "review" / "bib-verification"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert result.passed
    assert len(result.issues) == 0


def test_bib_verify_flags_invalid_entry(tmp_path: Path):
    """Entry flagged as invalid in manifest produces an error."""
    bib = tmp_path / "refs.bib"
    raw_entry = """@article{fake2020,
  title = {A Hallucinated Paper},
  author = {Ghost, Bob},
  year = {2020},
  journal = {Imaginary Journal},
}"""
    bib.write_text(raw_entry, encoding="utf-8")

    manifest = {
        "fake2020": {
            "key": "fake2020",
            "file": "refs.bib",
            "entry_hash": _compute_hash(raw_entry),
            "status": "invalid",
        }
    }
    manifest_dir = tmp_path / "review" / "bib-verification"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert not result.passed
    assert any(i.code == "bib_invalid_entry" and i.severity == "error" for i in result.issues)


def test_bib_verify_flags_hash_drift(tmp_path: Path):
    """Entry modified after verification flags hash drift warning."""
    bib = tmp_path / "refs.bib"
    original_entry = """@article{smith2020,
  title = {Initial Paper Title},
  author = {Smith, John},
  year = {2020},
  journal = {IEEE Transactions on Computers},
}"""
    modified_entry = """@article{smith2020,
  title = {Tampered Paper Title},
  author = {Smith, John},
  year = {2020},
  journal = {IEEE Transactions on Computers},
}"""
    bib.write_text(modified_entry, encoding="utf-8")

    manifest = {
        "smith2020": {
            "key": "smith2020",
            "file": "refs.bib",
            "entry_hash": _compute_hash(original_entry),
            "status": "verified",
        }
    }
    manifest_dir = tmp_path / "review" / "bib-verification"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert not result.passed
    assert any(i.code == "bib_verification_drift" and i.severity == "warning" for i in result.issues)


def test_bib_verify_flags_missing_sha(tmp_path: Path):
    """Entry in manifest without entry_hash produces an error."""
    bib = tmp_path / "refs.bib"
    raw_entry = """@article{nosha2020,
  title = {Un-hashed Publication},
  author = {Doe, Jane},
  year = {2020},
  journal = {Journal of Systems},
}"""
    bib.write_text(raw_entry, encoding="utf-8")

    manifest = {
        "nosha2020": {
            "key": "nosha2020",
            "file": "refs.bib",
            "entry_hash": None,
            "status": "verified",
        }
    }
    manifest_dir = tmp_path / "review" / "bib-verification"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert not result.passed
    assert any(i.code == "bib_missing_sha" and i.severity == "error" for i in result.issues)


def test_bib_verify_flags_unverified_entry_without_doi(tmp_path: Path):
    """Entry not in manifest without DOI flags bib_unverified_entry warning."""
    bib = tmp_path / "refs.bib"
    raw_entry = """@article{unknown2024,
  title = {Unpublished Draft},
  author = {Anonymous, Author},
  year = {2024},
  journal = {Internal Notes},
}"""
    bib.write_text(raw_entry, encoding="utf-8")

    manifest = {}
    manifest_dir = tmp_path / "review" / "bib-verification"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert not result.passed
    assert any(i.code == "bib_unverified_entry" and i.severity == "warning" for i in result.issues)


def test_bib_verify_handles_missing_manifest(tmp_path: Path):
    """Missing manifest produces a clear warning issue."""
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{k, title={T}, year={2020}}", encoding="utf-8")

    cmd = ValidateCommand(
        config_manager=SimpleNamespace(root_dir=tmp_path, book_dir=tmp_path),
        chapter_discovery=None,
    )
    result = cmd._run_bib_verify(bib)
    assert not result.passed
    assert any(i.code == "bib_verify_missing_manifest" for i in result.issues)


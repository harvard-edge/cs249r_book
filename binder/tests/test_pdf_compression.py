"""Exercise actual PDF preservation and reject damaged compression candidates."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from binder.cli.checks.margin_geometry import scan_page
from binder.cli.commands import _pdf_checks

SCRIPT = Path(__file__).resolve().parents[2] / "books/shared/publish/compress_pdf.py"
spec = importlib.util.spec_from_file_location("book_pdf_compression", SCRIPT)
compression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compression)


@pytest.fixture
def manuscript(tmp_path):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        for title in ("Chapter one", "Chapter two"):
            page = doc.new_page()
            page.insert_text((72, 72), title)
            page.draw_rect(pymupdf.Rect(72, 100, 180, 170), color=(0, 0, 1), fill=(0, 1, 0))
            bitmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 20, 20), False)
            bitmap.clear_with(180)
            page.insert_image(pymupdf.Rect(200, 100, 260, 160), stream=bitmap.tobytes("png"))
        doc[0].insert_link({"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(72, 72, 180, 90),
                            "page": 1, "to": pymupdf.Point(72, 72)})
        doc[1].insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 72, 180, 90),
                            "uri": "https://mlsysbook.ai/"})
        doc.set_toc([[1, "One", 1], [1, "Two", 2]])
        doc.set_metadata({"title": "Compression regression fixture", "author": "MLSysBook"})
        doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 1}])
        doc.embfile_add("receipt.txt", b"Constructed regression fixture attachment")
        doc.save(source)
    return source


def test_lossless_rewrite_preserves_actual_pdf(manuscript, tmp_path):
    output = tmp_path / "compressed.pdf"
    original = manuscript.read_bytes()
    stats = compression.compress_lossless(manuscript, output)
    with pymupdf.open(manuscript) as source, pymupdf.open(output) as candidate:
        assert stats["verified_pages"] == len(source)
        assert compression.verify_lossless_pdf(source, candidate) == len(source)
    assert stats["original_size"] == len(original)
    assert stats["final_size"] == output.stat().st_size
    assert manuscript.read_bytes() == original


@pytest.mark.parametrize("damage", ["page", "text", "drawing", "link", "outline", "metadata", "geometry", "labels", "attachment"])
def test_verifier_rejects_changed_pdf(manuscript, damage):
    with pymupdf.open(manuscript) as source, pymupdf.open(manuscript) as candidate:
        if damage == "page":
            candidate.delete_page(1)
        elif damage == "text":
            candidate[0].insert_text((72, 200), "Unexpected extra text")
        elif damage == "drawing":
            candidate[0].draw_rect(pymupdf.Rect(80, 110, 100, 130), fill=(1, 0, 0))
        elif damage == "link":
            candidate[0].delete_link(candidate[0].get_links()[0])
        elif damage == "outline":
            candidate.set_toc([])
        elif damage == "metadata":
            candidate.set_metadata({"title": "Wrong title"})
        elif damage == "geometry":
            candidate[0].set_cropbox(pymupdf.Rect(0, 0, 500, 700))
        elif damage == "labels":
            candidate.set_page_labels([])
        elif damage == "attachment":
            candidate.embfile_del("receipt.txt")
        with pytest.raises(RuntimeError, match="Lossless compression changed"):
            compression.verify_lossless_pdf(source, candidate)


def test_failed_verification_keeps_existing_destination(manuscript, tmp_path, monkeypatch):
    output = tmp_path / "published.pdf"
    output.write_bytes(b"Existing destination must survive")
    def reject(*args):
        raise RuntimeError("Verification rejected candidate")
    monkeypatch.setattr(compression, "verify_lossless_pdf", reject)
    with pytest.raises(RuntimeError, match="Verification rejected"):
        compression.compress_lossless(manuscript, output)
    assert output.read_bytes() == b"Existing destination must survive"
    assert not list(tmp_path.glob(".pdf-lossless-*"))


def test_refuses_to_overwrite_input(manuscript):
    original = manuscript.read_bytes()
    with pytest.raises(ValueError, match="different"):
        compression.compress_lossless(manuscript, manuscript)
    assert manuscript.read_bytes() == original


def test_trim_check_finds_text_omitted_by_normal_extraction():
    with pymupdf.open() as document:
        page = document.new_page(width=576, height=720)
        page.insert_text((100, 745), "Paragraph hidden below the page")
        assert "Paragraph hidden" not in page.get_text()
        findings = scan_page(page, 10)
        assert any(f.issue == "trim-overflow-bottom" and f.bbox[3] > page.rect.height
                   for f in findings)


@pytest.mark.parametrize("printer_marks", [False, True])
def test_trim_check_uses_actual_page_geometry(printer_marks):
    with pymupdf.open() as document:
        page = document.new_page(width=612 if printer_marks else 576,
                                 height=756 if printer_marks else 720)
        if printer_marks:
            page.set_trimbox(pymupdf.Rect(18, 18, 594, 738))
        page.insert_text((100, 730 if printer_marks else 700), "Text inside the trim")
        assert not any(f.issue.startswith("trim-overflow-") for f in scan_page(page, 10))


def test_existing_cover_printer_label_exemption_survives_unclipped_scan():
    with pymupdf.open() as document:
        page = document.new_page(width=612, height=756)
        page.set_trimbox(pymupdf.Rect(18, 18, 594, 738))
        page.insert_text((590, 752), "3")
        assert not any(f.issue.startswith("trim-overflow-") for f in scan_page(page, 1))


def test_geometry_scan_error_is_not_silently_skipped(monkeypatch, tmp_path):
    def fail_scan(*args, **kwargs):
        raise ValueError("Unusable geometry result")
    failed_scanner = SimpleNamespace(scan_pdf=fail_scan)
    monkeypatch.setitem(sys.modules, "cli.checks.margin_geometry", failed_scanner)
    monkeypatch.setitem(sys.modules, "binder.cli.checks.margin_geometry", failed_scanner)
    with pytest.raises(RuntimeError, match="geometry scan failed"):
        _pdf_checks.scan_margin_geometry_summary(tmp_path / "book.pdf")

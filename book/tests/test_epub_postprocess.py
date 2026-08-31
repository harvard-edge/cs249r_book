from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "quarto" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from epub_postprocess import (  # noqa: E402
    declare_nav_mathml_property,
    sanitize_xml_for_epubcheck,
)


def _write_epub_manifest(tmp_path, properties, nav_body):
    epub_dir = tmp_path / "EPUB"
    epub_dir.mkdir()
    (epub_dir / "nav.xhtml").write_text(nav_body, encoding="utf-8")
    (epub_dir / "content.opf").write_text(
        f'<package><manifest><item id="nav" href="nav.xhtml" '
        f'media-type="application/xhtml+xml" properties="{properties}"/>'
        f'</manifest></package>',
        encoding="utf-8",
    )


def _opf(tmp_path):
    return (tmp_path / "EPUB" / "content.opf").read_text(encoding="utf-8")


def test_declare_nav_mathml_property_adds_when_nav_contains_mathml(tmp_path):
    _write_epub_manifest(
        tmp_path,
        "nav",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><math></math></body></html>',
    )

    assert declare_nav_mathml_property(tmp_path) == 1
    assert 'properties="nav mathml"' in _opf(tmp_path)


def test_declare_nav_mathml_property_removes_stale_mathml(tmp_path):
    _write_epub_manifest(
        tmp_path,
        "nav mathml",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>No math here</body></html>',
    )

    assert declare_nav_mathml_property(tmp_path) == 1
    assert 'properties="nav"' in _opf(tmp_path)
    assert "mathml" not in _opf(tmp_path)


def test_declare_nav_mathml_property_keeps_correct_mathml(tmp_path):
    _write_epub_manifest(
        tmp_path,
        "nav mathml",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><math></math></body></html>',
    )

    assert declare_nav_mathml_property(tmp_path) == 0
    assert 'properties="nav mathml"' in _opf(tmp_path)


def test_sanitize_xml_removes_textless_nav_link_only(tmp_path):
    epub_dir = tmp_path / "EPUB"
    epub_dir.mkdir()
    nav_path = epub_dir / "nav.xhtml"
    nav_path.write_text(
        '<nav><ol><li id="empty"><a href="text/ch001.xhtml#section" /></li>'
        '<li id="named"><a href="text/ch002.xhtml#chapter">Chapter</a></li>'
        '</ol></nav>',
        encoding="utf-8",
    )

    counts = sanitize_xml_for_epubcheck(tmp_path)
    nav = nav_path.read_text(encoding="utf-8")

    assert counts["empty_nav_links"] == 1
    assert 'id="empty"' not in nav
    assert '<a href="text/ch002.xhtml#chapter">Chapter</a>' in nav

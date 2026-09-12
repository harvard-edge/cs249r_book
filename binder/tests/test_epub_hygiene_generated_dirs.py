"""EPUB hygiene scans authored sources only, never render output or Quarto caches (2026-09-12).

Stale SVG copies under ``books/.quarto/_freeze/**/mediabag/`` failed every local
EPUB build's preflight while CI, which starts from a clean checkout, passed.
"""

from pathlib import Path

from binder.cli.commands._epub_checks import find_hygiene_issues

BAD_SVG = '<svg xmlns="http://www.w3.org/2000/svg" aria-label="bad\x03label"></svg>\n'


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_hygiene_ignores_generated_svgs_but_flags_authored_ones(tmp_path):
    books = tmp_path / "books"
    write(books / ".quarto/_freeze/vol1/ch/mediabag/stale.svg", BAD_SVG)
    write(books / "_build/html-vol1/figures/copy.svg", BAD_SVG)
    write(books / "vol1/01_intro/01_intro_files/figure-html/render.svg", BAD_SVG)

    issues, files_checked = find_hygiene_issues(tmp_path)
    assert files_checked == 0
    assert issues == []

    write(books / "vol1/01_intro/images/svg/authored.svg", BAD_SVG)
    issues, files_checked = find_hygiene_issues(tmp_path)
    assert files_checked == 1
    assert issues and all("authored.svg" in issue.file for issue in issues)

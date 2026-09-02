"""Regression tests for format-specific sidenote offset metadata."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / "book" / "quarto" / "filters" / "sidenote.lua"


def _render(markdown: str, target: str) -> str:
    result = subprocess.run(
        [
            "quarto",
            "pandoc",
            "--from",
            "markdown",
            "--to",
            target,
            "--lua-filter",
            str(FILTER),
        ],
        input=markdown,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_html_strips_print_only_offset_but_keeps_note_text():
    html = _render(
        "A statement.[^note]\n\n[^note]: [offset=-18mm] **Term**: Explanation.\n",
        "html",
    )

    assert "offset=-18mm" not in html
    assert "Term" in html
    assert "Explanation" in html


def test_latex_preserves_offset_as_sidenote_placement():
    latex = _render(
        "A statement.[^note]\n\n[^note]: [offset=-18mm] **Term**: Explanation.\n",
        "latex",
    )

    assert r"\styledsidenote[][-18mm]{" in latex
    assert "[offset=-18mm]" not in latex
    assert "Explanation" in latex

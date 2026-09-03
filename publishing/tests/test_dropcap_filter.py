"""Regression tests for chapter-level drop-cap placement."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / "book" / "quarto" / "filters" / "dropcap.lua"


def _render(markdown):
    result = subprocess.run(
        [
            "quarto",
            "pandoc",
            "--from",
            "markdown",
            "--to",
            "latex",
            "--lua-filter",
            str(FILTER),
        ],
        input=markdown,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_chapter_attribute_places_dropcap_after_h1():
    latex = _render(
        """# Appendix {dropcap="chapter-opening"}

Opening prose belongs at the front.

## Summary {.unnumbered}

Summary prose comes later.
"""
    )

    assert r"\lettrine{O}{pening}" in latex
    assert r"\lettrine{S}{ummary}" not in latex


def test_document_metadata_supports_single_document_render():
    latex = _render(
        """---
dropcap: chapter-opening
---
# Appendix

Opening prose belongs at the front.
"""
    )

    assert r"\lettrine{O}{pening}" in latex


def test_rejected_candidate_does_not_leak_into_later_section():
    latex = _render(
        """# Chapter

## First Section

[@source] supports this citation-led paragraph.

## Summary {.unnumbered}

Summary prose comes later.
"""
    )

    assert r"\lettrine" not in latex


def test_default_mode_still_targets_first_numbered_section():
    latex = _render(
        """# Chapter

Front matter remains unchanged.

## Orientation {.unnumbered}

Unnumbered prose remains unchanged.

## First Section

Opening prose receives the drop cap.
"""
    )

    assert r"\lettrine{O}{pening}" in latex
    assert r"\lettrine{F}{ront}" not in latex
    assert r"\lettrine{U}{nnumbered}" not in latex

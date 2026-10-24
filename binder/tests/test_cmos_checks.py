"""Tests for CMOS and MIT Press style checks in binder.

Covers:
- Footnote placement: snug against punctuation, after quotes, no space before/after (CMOS §14.19 / §14.26)
- Quotation mark punctuation: periods and commas inside quotes (CMOS §6.9 / MIT Press §10)
- Comma after e.g. and i.e. (CMOS §6.51 / MIT Press §10.10)
- Period in vs. (CMOS §10.42 / MIT Press §10.5)
- En dash in number ranges (CMOS §6.78 / MIT Press §8)
"""

from types import SimpleNamespace
from pathlib import Path

from binder.cli.commands.validate import ValidateCommand


def _command(tmp_path: Path) -> ValidateCommand:
    return ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )


def test_footnote_placement_flags_before_punctuation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with footnote before period[^fn-sample-term].

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    codes = [issue.code for issue in result.issues]
    assert "footnote_inside_punctuation" in codes


def test_footnote_placement_flags_spaced_before_punctuation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with space before period [^fn-sample-term] .

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    codes = [issue.code for issue in result.issues]
    assert "footnote_inside_punctuation" in codes


def test_footnote_placement_flags_before_closing_quote(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with footnote before quote "example"[^fn-sample-term]".

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    codes = [issue.code for issue in result.issues]
    assert "footnote_inside_quote" in codes


def test_footnote_placement_flags_spaced_from_punctuation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with loose space after period. [^fn-sample-term]

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    codes = [issue.code for issue in result.issues]
    assert "footnote_spaced_from_punctuation" in codes


def test_footnote_placement_valid_snug_after_punctuation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with valid snug footnote placement.[^fn-sample-term]

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    assert result.issues == []


def test_footnote_placement_valid_after_closing_quote(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Prose with valid footnote after quote: "quoted phrase."[^{fn-sample-term}]

[^fn-sample-term]: **Sample Term**: Definition here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_placement(tmp_path)
    assert result.issues == []


def test_quote_punctuation_flags_period_and_comma_outside(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Here is a "phrase", and another "phrase".
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_quote_punctuation(tmp_path)
    assert len(result.issues) == 2
    assert all(i.code == "quote_punctuation_placement" for i in result.issues)


def test_quote_punctuation_accepts_period_and_comma_inside(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Here is a "phrase," and another "phrase."
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_quote_punctuation(tmp_path)
    assert result.issues == []


def test_quote_punctuation_ignores_code_and_math(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Inline code `"key",` and math $x".$ are ignored.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_quote_punctuation(tmp_path)
    assert result.issues == []


def test_eg_ie_comma_validation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """For example, e.g. without comma is flagged, but e.g., with comma is fine.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_eg_ie_comma(tmp_path)
    assert len(result.issues) == 1
    assert result.issues[0].code == "mitpress_eg_ie_comma"


def test_vs_period_validation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Compare training vs inference vs. testing.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_vs_period(tmp_path)
    assert len(result.issues) == 1
    assert result.issues[0].code == "mitpress_vs_period"


def test_hyphen_range_validation(tmp_path: Path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Number range 100-200 needs en dash 100–200.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_mitpress_hyphen_range(tmp_path)
    assert len(result.issues) == 1
    assert result.issues[0].code == "mitpress_hyphen_range"

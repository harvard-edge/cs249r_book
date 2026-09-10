from pathlib import Path

import pytest

from bindery.cli.commands.build import BuildCommand
from bindery.cli.main import MLSysBookCLI


def test_build_parser_accepts_pdf_presentation_flags():
    cli = object.__new__(MLSysBookCLI)

    parsed = cli._parse_build_args(
        ["pdf", "--vol1", "--no-cover", "--print-marks"]
    )

    assert parsed == (
        "pdf",
        "vol1",
        False,
        None,
        False,
        False,
        False,
        True,
        True,
    )


def test_set_pdf_cover_changes_true_switch_to_false(tmp_path: Path):
    config = tmp_path / "_quarto-pdf-vol1.yml"
    config.write_text(
        "format:\n  titlepage-pdf:\n    coverpage: true\n",
        encoding="utf-8",
    )

    changed = BuildCommand._set_pdf_cover(config, enabled=False)

    assert changed is True
    assert "    coverpage: false\n" in config.read_text(encoding="utf-8")


def test_set_pdf_cover_leaves_requested_switch_unchanged(tmp_path: Path):
    config = tmp_path / "_quarto-pdf-vol2.yml"
    original = "format:\n  titlepage-pdf:\n    coverpage: true\n"
    config.write_text(original, encoding="utf-8")

    changed = BuildCommand._set_pdf_cover(config, enabled=True)

    assert changed is False
    assert config.read_text(encoding="utf-8") == original


def test_set_pdf_cover_rejects_ambiguous_manifest(tmp_path: Path):
    config = tmp_path / "_quarto-pdf-vol1.yml"
    config.write_text(
        "format:\n  titlepage-pdf:\n    coverpage: false\n"
        "other:\n    coverpage: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        BuildCommand._set_pdf_cover(config, enabled=False)


def test_set_pdf_print_marks_changes_false_switch_to_true(tmp_path: Path):
    header = tmp_path / "header-includes.tex"
    header.write_text(
        "\\newif\\ifCropMarks\n\\CropMarksfalse\n",
        encoding="utf-8",
    )

    changed = BuildCommand._set_pdf_print_marks(header, enabled=True)

    assert changed is True
    assert "\\CropMarkstrue\n" in header.read_text(encoding="utf-8")


def test_all_volume_pdf_configs_include_cover_by_default():
    config_dir = Path(__file__).resolve().parents[2] / "books" / "config"

    for volume in range(1, 5):
        config = config_dir / f"_quarto-pdf-vol{volume}.yml"
        assert "    coverpage: true\n" in config.read_text(encoding="utf-8")


def test_shared_header_omits_print_marks_by_default():
    header = (
        Path(__file__).resolve().parents[2]
        / "books"
        / "shared"
        / "tex"
        / "header-includes.tex"
    )

    assert "\\CropMarksfalse\n" in header.read_text(encoding="utf-8")

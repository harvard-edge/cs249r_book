"""Regression tests for binder check reporting: no silent passes, usable locations."""

from pathlib import Path
from types import SimpleNamespace

from binder.cli.commands import _index_checks
from binder.cli.commands.validate import ValidateCommand, ValidationRunResult


def _command(book_dir: Path) -> ValidateCommand:
    return ValidateCommand(
        config_manager=SimpleNamespace(book_dir=book_dir, root_dir=book_dir.parent),
        chapter_discovery=None,
    )


def test_audit_check_scans_a_single_file_path(tmp_path):
    # --path <file> used to reach rglob() on a file, scan nothing, and pass.
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text("Models are pre-trained on large corpora.\n", encoding="utf-8")

    result = _command(tmp_path)._run_mitpress_compound_prefix(qmd)

    assert result.files_checked == 1
    assert [issue.line for issue in result.issues] == [1]
    assert result.issues[0].file == "chapter.qmd"


def test_python_syntax_reports_book_relative_path_for_single_file(tmp_path):
    chapter = tmp_path / "vol1" / "01_intro"
    chapter.mkdir(parents=True)
    qmd = chapter / "01_intro.qmd"
    qmd.write_text("```{python}\nx = (\n```\n", encoding="utf-8")

    result = _command(tmp_path)._run_python_syntax(qmd)

    assert result.issues and result.issues[0].file == "vol1/01_intro/01_intro.qmd"
    assert result.issues[0].suggestion


def test_table_caption_orphan_keeps_scanning_after_tikz_block(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "```{.tikz}\n\\draw (0,0) -- (1,1);\n```\n\n"
        "::: {.content-visible when-format=\"html\"}\n"
        "::: {tbl-colwidths=\"[50,50]\"}\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n:::\n\n"
        ": **Caption**: text. {#tbl-x}\n\n:::\n",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_table_caption_orphan(qmd)

    assert [issue.code for issue in result.issues] == ["table_caption_orphan"]


def test_citations_checks_every_volume_against_shared_bibliography(tmp_path):
    (tmp_path / "references.bib").write_text("@book{known2020,\n  title={T},\n}\n", encoding="utf-8")
    chapter = tmp_path / "vol4" / "01_intro"
    chapter.mkdir(parents=True)
    qmd = chapter / "01_intro.qmd"
    qmd.write_text("As shown [@known2020] and [@unknown2021].\n", encoding="utf-8")

    result = _command(tmp_path)._run_citations(qmd)

    assert [(i.code, i.context) for i in result.issues] == [("missing_citation", "@unknown2021")]


def test_citations_reports_missing_bibliography(tmp_path):
    chapter = tmp_path / "vol1" / "01_intro"
    chapter.mkdir(parents=True)
    qmd = chapter / "01_intro.qmd"
    qmd.write_text("As shown [@known2020].\n", encoding="utf-8")

    result = _command(tmp_path)._run_citations(qmd)

    assert [i.code for i in result.issues] == ["bibliography_missing"]


def test_checker_error_names_the_scope_and_a_rerun_command():
    issue = ValidateCommand._checker_error("vol1/x.qmd", 0, "content", ValueError("boom"))

    assert issue.severity == "error"
    assert "ValueError: boom" in issue.message
    assert "--scope content" in issue.suggestion


def test_missing_pdf_is_reported_not_passed(tmp_path, monkeypatch):
    command = _command(tmp_path)
    monkeypatch.setattr(
        command, "_pdf_targets", lambda root, flags: (tmp_path, tmp_path / "books", ["vol1"])
    )

    result = command._run_pdf_purpose_overflow(tmp_path, vol1=True)

    assert not result.passed
    assert result.issues[0].code == "missing-pdf"
    assert "build pdf --vol1" in result.issues[0].suggestion


def test_index_anti_pattern_points_at_first_occurrence(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text("Intro.\n\nThe term\\index{The Thing} appears.\n", encoding="utf-8")

    issues = _index_checks.check_anti_patterns(tmp_path)

    leading = [i for i in issues if i.code == "article_leading"]
    assert leading and (leading[0].file, leading[0].line) == ("chapter.qmd", 3)
    assert "chapter.qmd:3" in leading[0].message


def test_catalog_fills_missing_suggestions():
    from binder.cli.commands.validate import ValidationIssue

    run = ValidationRunResult(
        name="x", description="x", files_checked=1, elapsed_ms=0,
        issues=[ValidationIssue(file="f", line=1, code="duplicate_word", message="m")],
    )

    ValidateCommand._fill_catalog_suggestions([run])

    assert run.issues[0].suggestion
    assert "group" not in run.to_dict() and "scope" not in run.to_dict()
